//! K-cache mean-centering bias — fork-compatible sidecar support.
//!
//! The PrismML fork's `load_kv_mean_center` subtracts a calibrated
//! per-(layer, kv_head, channel) mean `k̄` from every K vector at
//! cache-write time. This is **exactly softmax-invariant**: a query's
//! score row becomes `q·(k_t − k̄) = q·k_t − q·k̄`, and the `q·k̄`
//! term is the same constant for every cached position `t`, which
//! softmax cannot see. No read-side compensation exists or is needed —
//! the only effect is that the *cached representation* is centered,
//! which measurably reduces uniform-quantizer (Q4_0) error when the
//! per-channel K distribution has a large mean.
//!
//! ## Sidecar format (byte-compatible with the fork)
//!
//! A GGUF file containing, per centered layer `il` (model layer
//! index; layers may be omitted → left uncentered):
//!
//! ```text
//!   tensor "kv_bar.blk.{il}.k"  F32  [head_dim, n_kv_heads]
//! ```
//!
//! plus the metadata key `kv_mean_center.k_rot: bool` recording
//! whether calibration ran in the whitened (Hadamard-rotated) basis.
//! A basis mismatch *degrades* quality instead of improving it, so —
//! exactly like the fork — a recorded mismatch refuses to load and a
//! missing key only warns.
//!
//! The flat layout of each tensor (`bias[h * head_dim + d]`) matches
//! the forward's `k_buf` layout, so the subtraction is a single
//! element-wise loop over `d_kv` floats.

use std::path::Path;

use rustllama_gguf::{Gguf, GgmlType, MetadataValue};

/// Loaded, validated bias data. Attached to the KV cache by the
/// engine; consumed by the Q4_0 KV write arms.
#[derive(Debug)]
pub struct KvBiasData {
    /// One entry per model layer; `None` = uncentered layer. The
    /// vector inside is `n_kv_heads * head_dim` long, laid out
    /// `[kv_head][channel]` — identical to the forward's `k_buf`.
    pub per_layer: Vec<Option<Vec<f32>>>,
    /// The calibration basis recorded in the file (`None` if the
    /// file predates basis recording).
    pub k_rot: Option<bool>,
}

impl KvBiasData {
    /// Number of layers that actually carry a bias.
    pub fn n_centered(&self) -> usize {
        self.per_layer.iter().filter(|l| l.is_some()).count()
    }

    /// The bias slice for one model layer, if centered.
    #[inline]
    pub fn layer(&self, li: usize) -> Option<&[f32]> {
        self.per_layer.get(li).and_then(|l| l.as_deref())
    }

    /// Load and validate a sidecar. `n_layers`/`head_dim`/`n_kv_heads`
    /// come from the model config; every present tensor must be F32
    /// with exactly `head_dim * n_kv_heads` elements (mirrors the
    /// fork's checks). Basis validation against the *active* whitening
    /// state is the caller's job (it knows the KV dtype).
    pub fn load(
        path: &Path,
        n_layers: usize,
        head_dim: usize,
        n_kv_heads: usize,
    ) -> Result<Self, String> {
        let gguf = Gguf::open(path)
            .map_err(|e| format!("kv-bias sidecar {}: {e}", path.display()))?;

        let k_rot = match gguf.metadata_get("kv_mean_center.k_rot") {
            Some(MetadataValue::Bool(b)) => Some(*b),
            Some(other) => {
                return Err(format!(
                    "kv-bias sidecar {}: kv_mean_center.k_rot must be bool, got {other:?}",
                    path.display()
                ));
            }
            None => None,
        };

        let expect = head_dim * n_kv_heads;
        let mut per_layer: Vec<Option<Vec<f32>>> = vec![None; n_layers];
        let mut n_centered = 0usize;
        for li in 0..n_layers {
            let name = format!("kv_bar.blk.{li}.k");
            let Some(info) = gguf.tensor(&name) else { continue };
            if info.dtype != GgmlType::F32 {
                return Err(format!(
                    "kv-bias tensor {name} must be F32 (got {})",
                    info.dtype.as_str()
                ));
            }
            let n_elems = info.element_count() as usize;
            if n_elems != expect {
                return Err(format!(
                    "kv-bias tensor {name} has {n_elems} elements, expected \
                     {expect} (head_dim {head_dim} × n_kv_heads {n_kv_heads})"
                ));
            }
            let bytes = gguf
                .tensor_bytes(&name)
                .ok_or_else(|| format!("kv-bias tensor {name}: data out of bounds"))?;
            let mut vals = vec![0f32; n_elems];
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                vals[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            per_layer[li] = Some(vals);
            n_centered += 1;
        }
        if n_centered == 0 {
            return Err(format!(
                "kv-bias sidecar {} contains no kv_bar.blk.*.k tensors",
                path.display()
            ));
        }
        Ok(Self { per_layer, k_rot })
    }

    /// Write a sidecar in the fork's format, declaring each tensor
    /// with the fork's exact 2-D `[head_dim, n_kv_heads]` shape.
    /// Behind `kv-bias-write` (gguf encoder tables); reading needs no
    /// feature. Unit tests get it via the synth dev-dependency, whose
    /// `synth` feature implies `encoder`.
    #[cfg(any(test, feature = "kv-bias-write"))]
    pub fn save_with_geometry(
        &self,
        path: &Path,
        head_dim: usize,
        n_kv_heads: usize,
    ) -> Result<(), String> {
        use rustllama_gguf::write::{GgufWriter, WriteError};
        let stringify = |e: WriteError| format!("kv-bias write {}: {e}", path.display());
        let mut w = GgufWriter::create(path).map_err(stringify)?;
        w.add_metadata(
            "kv_mean_center.k_rot",
            MetadataValue::Bool(self.k_rot.unwrap_or(false)),
        )
        .map_err(stringify)?;
        let mut names = Vec::new();
        for (li, layer) in self.per_layer.iter().enumerate() {
            if let Some(vals) = layer {
                assert_eq!(vals.len(), head_dim * n_kv_heads);
                let name = format!("kv_bar.blk.{li}.k");
                w.declare_tensor(
                    name.clone(),
                    vec![head_dim as u64, n_kv_heads as u64],
                    GgmlType::F32,
                )
                .map_err(stringify)?;
                names.push((name, li));
            }
        }
        w.finish_header().map_err(stringify)?;
        for (name, li) in &names {
            let vals = self.per_layer[*li].as_ref().unwrap();
            let mut bytes = Vec::with_capacity(vals.len() * 4);
            for v in vals {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            w.write_tensor_data(name, &bytes).map_err(stringify)?;
        }
        w.finish().map_err(stringify)?;
        Ok(())
    }
}

/// Calibration accumulator: per-(layer, kv_head, channel) running sum
/// and count of K rows observed at KV write time (post-RoPE, post-
/// whitening when active — i.e. in exactly the basis the bias will be
/// applied in). Enabled only while a calibration run holds the global
/// handle; the fast path in the forward is a single relaxed-atomic
/// check.
#[derive(Debug, Default)]
pub struct KvCalibAccum {
    /// `sums[li]` is `n_kv_heads * head_dim` long (lazily sized on
    /// first observation of that layer).
    pub sums: Vec<Vec<f64>>,
    pub counts: Vec<u64>,
}

impl KvCalibAccum {
    pub fn new(n_layers: usize) -> Self {
        Self {
            sums: vec![Vec::new(); n_layers],
            counts: vec![0; n_layers],
        }
    }

    /// Fold the accumulated means into a [`KvBiasData`].
    pub fn into_bias(self, k_rot: bool) -> KvBiasData {
        let per_layer = self
            .sums
            .into_iter()
            .zip(self.counts.iter())
            .map(|(sum, &cnt)| {
                if sum.is_empty() || cnt == 0 {
                    None
                } else {
                    Some(sum.iter().map(|s| (*s / cnt as f64) as f32).collect())
                }
            })
            .collect();
        KvBiasData {
            per_layer,
            k_rot: Some(k_rot),
        }
    }
}

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

static CALIB_ON: AtomicBool = AtomicBool::new(false);
static CALIB: Mutex<Option<KvCalibAccum>> = Mutex::new(None);

/// Begin a calibration run: installs a fresh accumulator. Only one
/// run at a time; the forward starts feeding K rows on the next token.
pub fn calib_begin(n_layers: usize) {
    let mut g = CALIB.lock().expect("kv-calib lock");
    *g = Some(KvCalibAccum::new(n_layers));
    CALIB_ON.store(true, Ordering::Release);
}

/// End the run and take the accumulator.
pub fn calib_take() -> Option<KvCalibAccum> {
    CALIB_ON.store(false, Ordering::Release);
    CALIB.lock().expect("kv-calib lock").take()
}

/// Hot-path gate — a single relaxed atomic load when calibration is
/// off (the overwhelmingly common case).
#[inline]
pub fn calib_active() -> bool {
    CALIB_ON.load(Ordering::Relaxed)
}

/// Record one layer's K rows (`k_buf` layout, `n_kv_heads * head_dim`)
/// for the current position. No-op if no run is active.
pub fn calib_observe(li: usize, k_rows: &[f32]) {
    if !calib_active() {
        return;
    }
    let mut g = CALIB.lock().expect("kv-calib lock");
    if let Some(acc) = g.as_mut() {
        if li >= acc.sums.len() {
            return;
        }
        let sums = &mut acc.sums[li];
        if sums.is_empty() {
            sums.resize(k_rows.len(), 0.0);
        }
        if sums.len() == k_rows.len() {
            for (s, &v) in sums.iter_mut().zip(k_rows) {
                *s += v as f64;
            }
            acc.counts[li] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_round_trip() {
        let mut per_layer = vec![None; 4];
        per_layer[1] = Some((0..8).map(|i| i as f32 * 0.25 - 1.0).collect());
        per_layer[3] = Some((0..8).map(|i| -(i as f32) * 0.5).collect());
        let bias = KvBiasData {
            per_layer,
            k_rot: Some(true),
        };
        let tmp = std::env::temp_dir().join("rustllama-kvbias-roundtrip.gguf");
        bias.save_with_geometry(&tmp, 4, 2).expect("save");
        let loaded = KvBiasData::load(&tmp, 4, 4, 2).expect("load");
        assert_eq!(loaded.k_rot, Some(true));
        assert_eq!(loaded.n_centered(), 2);
        assert!(loaded.layer(0).is_none());
        assert_eq!(loaded.layer(1).unwrap(), bias.layer(1).unwrap());
        assert_eq!(loaded.layer(3).unwrap(), bias.layer(3).unwrap());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn load_rejects_wrong_element_count() {
        let mut per_layer = vec![None; 2];
        per_layer[0] = Some(vec![0.5f32; 8]);
        let bias = KvBiasData {
            per_layer,
            k_rot: Some(false),
        };
        let tmp = std::env::temp_dir().join("rustllama-kvbias-badshape.gguf");
        bias.save_with_geometry(&tmp, 4, 2).expect("save");
        // Expect head_dim*n_kv_heads = 16, file has 8.
        let err = KvBiasData::load(&tmp, 2, 8, 2).unwrap_err();
        assert!(err.contains("expected 16"), "unexpected error: {err}");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn calib_accumulator_means() {
        calib_begin(2);
        assert!(calib_active());
        calib_observe(0, &[1.0, 2.0]);
        calib_observe(0, &[3.0, 6.0]);
        calib_observe(1, &[10.0, -10.0]);
        let acc = calib_take().expect("accum");
        assert!(!calib_active());
        let bias = acc.into_bias(true);
        assert_eq!(bias.layer(0).unwrap(), &[2.0, 4.0]);
        assert_eq!(bias.layer(1).unwrap(), &[10.0, -10.0]);
        assert_eq!(bias.k_rot, Some(true));
    }
}
