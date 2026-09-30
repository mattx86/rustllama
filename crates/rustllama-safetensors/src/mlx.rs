//! Apple **MLX** quantized-model safetensors loader (affine mode).
//!
//! `mlx-lm` publishes LLM checkpoints as a HuggingFace-style directory:
//! a `config.json` carrying the architecture hyperparameters **plus** a
//! `"quantization"` block, one or more `*.safetensors` shards, and a
//! `tokenizer.json` (standard HF tokenizer — handled by
//! `rustllama-tokenizer`, not here). This module reads the quantization
//! config, enumerates the safetensors tensors, and pulls each quantized
//! linear/embedding's `{weight, scales, biases}` triple into a
//! [`MlxAffineQuant`]; full-precision tensors (norms, un-quantized
//! embeddings / `lm_head`, biases) carry through as raw bytes.
//!
//! # Detection
//!
//! An MLX-quantized checkpoint is identified by **both**:
//! 1. a `"quantization"` object in `config.json` (with `group_size` /
//!    `bits`, optionally `mode` + per-layer overrides), and
//! 2. `.scales` / `.biases` sibling tensors next to packed `.weight`s.
//!
//! Keying off the sibling tensors (rather than a fixed list of module
//! names) means we don't have to know *which* layers mlx-lm chose to
//! quantize — `nn.Linear` and often the token embedding / `lm_head`
//! (`QuantizedEmbedding`) carry the siblings; norms don't.
//!
//! # Format
//!
//! See [`rustllama_tensor::mlx_affine`] for the bit-packing + dequant
//! derivation with MLX source citations. In short: `.weight` is packed
//! `bits`-bit codes as a `uint32` little-endian bitstream; `.scales` /
//! `.biases` are per-`group_size` f16/bf16; dequant is the plain affine
//! `w = scale*q + bias`.
//!
//! # Out of scope (this phase)
//!
//! - **Non-affine modes** (`mxfp4` / `nvfp4` / `mxfp8`): the mode is
//!   parsed + surfaced, but loading a non-affine weight triple is
//!   rejected with [`MlxError::UnsupportedMode`] — those layouts have no
//!   `biases` and use e8m0/e4m3 scales (a later slice; rustllama already
//!   has MXFP/NVFP4 decoders in `rustllama-kernels-cpu` to build on).
//! - **Model wiring**: turning an [`MlxModel`] into a runnable
//!   `LlamaModel` (weight-name mapping, arch dispatch, matvec routing) is
//!   the next, build-env-gated phase in `rustllama-models`. This module
//!   deliberately depends only on `safetensors` + `rustllama-tensor` so
//!   it stays plain-cargo testable.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use safetensors::tensor::TensorView;
use safetensors::{Dtype as StDtype, SafeTensors};

use rustllama_tensor::{MlxAffineError, MlxAffineQuant};

use crate::SafetensorsError;

/// The three tensor-name suffixes that make up one MLX affine weight.
const SUFFIX_WEIGHT: &str = ".weight";
const SUFFIX_SCALES: &str = ".scales";
const SUFFIX_BIASES: &str = ".biases";

/// MLX quantization mode recorded in `config.json`'s `quantization.mode`.
/// Only [`MlxQuantMode::Affine`] (the default + overwhelmingly common
/// mode) is loadable in this phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlxQuantMode {
    /// Affine (asymmetric) — packed uint32 codes + per-group scale AND
    /// bias. The default when `mode` is absent.
    Affine,
    /// OCP MXFP4 microscaling (e8m0 block scale, no bias). Not loaded yet.
    Mxfp4,
    /// NVIDIA NVFP4 (e4m3 block scale, no bias). Not loaded yet.
    Nvfp4,
    /// OCP MXFP8 microscaling. Not loaded yet.
    Mxfp8,
    /// Any other / future mode string, preserved verbatim.
    Other(String),
}

impl MlxQuantMode {
    fn from_str(s: &str) -> Self {
        match s {
            "affine" => Self::Affine,
            "mxfp4" => Self::Mxfp4,
            "nvfp4" => Self::Nvfp4,
            "mxfp8" => Self::Mxfp8,
            other => Self::Other(other.to_string()),
        }
    }
}

/// Resolved quantization parameters for one layer (global defaults or a
/// per-layer override).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxLayerQuant {
    pub group_size: usize,
    pub bits: u32,
    pub mode: MlxQuantMode,
}

/// The `config.json` `"quantization"` block. `group_size` / `bits` /
/// `mode` are the model-wide defaults; `overrides` maps an MLX module
/// path (e.g. `model.layers.0.mlp.gate_proj`) to either a custom
/// [`MlxLayerQuant`] (`Some`) or an explicit skip (`None`, from a
/// `false`/`null` value). mlx-lm writes these when a model is partially
/// or heterogeneously quantized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxQuantConfig {
    pub group_size: usize,
    pub bits: u32,
    pub mode: MlxQuantMode,
    pub overrides: BTreeMap<String, Option<MlxLayerQuant>>,
}

impl MlxQuantConfig {
    /// Resolve the quant params for the module at `module_path`.
    /// Returns `None` when an override explicitly skips the layer.
    /// Layers with no override fall back to the global defaults.
    pub fn for_layer(&self, module_path: &str) -> Option<MlxLayerQuant> {
        match self.overrides.get(module_path) {
            Some(Some(over)) => Some(over.clone()),
            Some(None) => None, // explicit skip (false / null)
            None => Some(MlxLayerQuant {
                group_size: self.group_size,
                bits: self.bits,
                mode: self.mode.clone(),
            }),
        }
    }

    /// Parse the `"quantization"` block out of a raw `config.json`.
    /// Returns `Ok(None)` when there is no such block (i.e. not an
    /// MLX-quantized checkpoint). The JSON shape mixes scalar globals
    /// (`group_size`/`bits`/`mode`) with arbitrary per-layer keys whose
    /// values are either an object `{group_size, bits, mode?}` or a bare
    /// `false`/`null`, so we walk it as a generic `serde_json::Value`.
    pub fn parse_from_config_json(json: &str) -> Result<Option<Self>, MlxError> {
        let root: serde_json::Value = serde_json::from_str(json)?;
        // mlx-lm has historically used both `quantization` and
        // `quantization_config`; accept either (prefer `quantization`).
        let q = root
            .get("quantization")
            .or_else(|| root.get("quantization_config"));
        let Some(q) = q else {
            return Ok(None);
        };
        let obj = q.as_object().ok_or(MlxError::MalformedQuantConfig)?;

        let group_size = obj
            .get("group_size")
            .and_then(|v| v.as_u64())
            .ok_or(MlxError::MalformedQuantConfig)? as usize;
        let bits = obj
            .get("bits")
            .and_then(|v| v.as_u64())
            .ok_or(MlxError::MalformedQuantConfig)? as u32;
        let mode = obj
            .get("mode")
            .and_then(|v| v.as_str())
            .map(MlxQuantMode::from_str)
            .unwrap_or(MlxQuantMode::Affine);

        // Any remaining object/false/null value keyed by a non-scalar
        // field is a per-layer override.
        let mut overrides = BTreeMap::new();
        for (k, v) in obj {
            if matches!(k.as_str(), "group_size" | "bits" | "mode") {
                continue;
            }
            match v {
                serde_json::Value::Bool(false) | serde_json::Value::Null => {
                    overrides.insert(k.clone(), None);
                }
                serde_json::Value::Bool(true) => {
                    // `true` = quantize with the global defaults.
                    overrides.insert(
                        k.clone(),
                        Some(MlxLayerQuant {
                            group_size,
                            bits,
                            mode: mode.clone(),
                        }),
                    );
                }
                serde_json::Value::Object(o) => {
                    let gs = o
                        .get("group_size")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize)
                        .unwrap_or(group_size);
                    let b = o
                        .get("bits")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32)
                        .unwrap_or(bits);
                    let m = o
                        .get("mode")
                        .and_then(|v| v.as_str())
                        .map(MlxQuantMode::from_str)
                        .unwrap_or_else(|| mode.clone());
                    overrides.insert(
                        k.clone(),
                        Some(MlxLayerQuant {
                            group_size: gs,
                            bits: b,
                            mode: m,
                        }),
                    );
                }
                // Anything else (a stray scalar) is ignored rather than
                // aborting the whole parse.
                _ => {}
            }
        }

        Ok(Some(Self {
            group_size,
            bits,
            mode,
            overrides,
        }))
    }
}

/// Dtype of a full-precision (non-quantized) MLX tensor's raw bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxFullDtype {
    F32,
    F16,
    Bf16,
}

/// A full-precision tensor carried through verbatim (norms, un-quantized
/// embeddings / `lm_head`, linear biases). Raw little-endian bytes are
/// kept as-is so the later model-wiring phase decides the in-memory
/// dtype; no premature f32 blow-up of a large embedding table.
#[derive(Debug, Clone)]
pub struct MlxFullTensor {
    pub name: String,
    pub dtype: MlxFullDtype,
    pub shape: Vec<u64>,
    pub bytes: Vec<u8>,
}

/// The tensors of an MLX checkpoint, split into quantized weights +
/// full-precision tensors, plus the parsed quant config.
///
/// `quant` is keyed by **module path** (the shared prefix, e.g.
/// `model.layers.0.self_attn.q_proj`), i.e. the `.weight` name with the
/// `.weight` suffix stripped. `full` is keyed by the tensor's own name.
///
// TODO(mlx phase B — models wiring): consume an `MlxModel` in
// `rustllama-models` to build a runnable model — map the HF-style module
// paths to rustllama's weight slots (reuse/extend `name_map`), dispatch on
// the `architectures` field (via `rustllama_models::llama_config`), and
// route the quant weights through `kernels-cpu::mlx_affine`
// (`matvec_mlx_affine_w_f32_a`) on CPU / the Metal path on Apple Silicon.
// That phase is build-env-gated, so it lives in the models crate, not here.
#[derive(Debug, Clone)]
pub struct MlxModel {
    pub config: MlxQuantConfig,
    pub quant: BTreeMap<String, MlxAffineQuant>,
    pub full: BTreeMap<String, MlxFullTensor>,
}

#[derive(Debug, thiserror::Error)]
pub enum MlxError {
    #[error("config.json parse: {0}")]
    Json(#[from] serde_json::Error),
    #[error("safetensors deserialize: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("safetensors io: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors: {0}")]
    Container(#[from] SafetensorsError),
    #[error(
        "config.json `quantization` block is malformed (need integer \
         `group_size` + `bits`)"
    )]
    MalformedQuantConfig,
    #[error("not an MLX-quantized checkpoint: no `quantization` block in config.json")]
    NotMlx,
    #[error(
        "mlx layer `{module}` uses mode {mode:?}; this phase loads only \
         affine-quantized weights (mxfp4/nvfp4/mxfp8 are a later slice)"
    )]
    UnsupportedMode { module: String, mode: MlxQuantMode },
    #[error(
        "mlx layer `{module}` is explicitly skipped by a per-layer override \
         in config.json, yet ships packed `.scales`/`.biases` tensors"
    )]
    SkippedButPacked { module: String },
    #[error(
        "mlx weight `{name}` has dtype {dtype:?}; MLX packs affine codes \
         into uint32"
    )]
    WeightNotU32 { name: String, dtype: StDtype },
    #[error(
        "mlx weight `{name}` packed shape {shape:?} is not 2-D \
         [out_features, in_features*bits/32]"
    )]
    WeightNotMatrix { name: String, shape: Vec<u64> },
    #[error(
        "mlx weight `{name}`: packed row width {row_words} words × 32 bits is \
         not divisible by bits {bits} — cannot recover in_features"
    )]
    RowWidthIndivisible {
        name: String,
        row_words: u64,
        bits: u32,
    },
    #[error(
        "mlx tensor `{name}` has unsupported float dtype {dtype:?}; \
         scales/biases + full tensors must be F16, BF16, or F32"
    )]
    UnsupportedFloatDtype { name: String, dtype: StDtype },
    #[error("mlx affine geometry: {0}")]
    Geometry(#[from] MlxAffineError),
}

/// Is this a MLX-quantized checkpoint? True when `config.json` has a
/// `quantization` block **and** the safetensors carries `.scales` +
/// `.biases` sibling tensors. Cheap detection primitive the engine's
/// format sniffer can call before committing to the MLX load path.
pub fn is_mlx_model(config_json: &str, safetensors_bytes: &[u8]) -> bool {
    let has_q = matches!(
        MlxQuantConfig::parse_from_config_json(config_json),
        Ok(Some(_))
    );
    if !has_q {
        return false;
    }
    let Ok(st) = SafeTensors::deserialize(safetensors_bytes) else {
        return false;
    };
    let mut has_scales = false;
    let mut has_biases = false;
    for name in st.names() {
        if name.ends_with(SUFFIX_SCALES) {
            has_scales = true;
        }
        if name.ends_with(SUFFIX_BIASES) {
            has_biases = true;
        }
        if has_scales && has_biases {
            return true;
        }
    }
    false
}

/// Decode a F16 / BF16 / F32 tensor payload to `Vec<f32>`.
///
/// safetensors byte slices come straight from an mmap with no alignment
/// guarantee, so we decode element-by-element from little-endian bytes
/// (the same hazard [`crate::convert`] handles with `as_f16_slice`).
fn floats_to_f32(name: &str, view: &TensorView<'_>) -> Result<Vec<f32>, MlxError> {
    let raw = view.data();
    Ok(match view.dtype() {
        StDtype::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        StDtype::F16 => raw
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        StDtype::BF16 => raw
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => {
            return Err(MlxError::UnsupportedFloatDtype {
                name: name.to_string(),
                dtype: other,
            })
        }
    })
}

fn full_dtype(name: &str, dtype: StDtype) -> Result<MlxFullDtype, MlxError> {
    Ok(match dtype {
        StDtype::F32 => MlxFullDtype::F32,
        StDtype::F16 => MlxFullDtype::F16,
        StDtype::BF16 => MlxFullDtype::Bf16,
        other => {
            return Err(MlxError::UnsupportedFloatDtype {
                name: name.to_string(),
                dtype: other,
            })
        }
    })
}

/// Load one safetensors shard's tensors into the `quant` + `full` maps,
/// using `config` for per-layer `group_size`/`bits`. Shared by the
/// in-memory and directory entry points; call it once per shard and the
/// maps accumulate across shards.
fn load_shard_into(
    config: &MlxQuantConfig,
    bytes: &[u8],
    quant: &mut BTreeMap<String, MlxAffineQuant>,
    full: &mut BTreeMap<String, MlxFullTensor>,
) -> Result<(), MlxError> {
    let st = SafeTensors::deserialize(bytes)?;

    // Pass 1: find every quantized module prefix `P` such that
    // `P.weight` + `P.scales` + `P.biases` all exist in this shard, and
    // record the three tensor names they consume.
    let names: BTreeSet<String> = st.names().into_iter().cloned().collect();
    let mut prefixes: Vec<String> = Vec::new();
    let mut consumed: BTreeSet<String> = BTreeSet::new();
    for name in &names {
        let Some(prefix) = name.strip_suffix(SUFFIX_SCALES) else {
            continue;
        };
        let weight = format!("{prefix}{SUFFIX_WEIGHT}");
        let biases = format!("{prefix}{SUFFIX_BIASES}");
        if names.contains(&weight) && names.contains(&biases) {
            prefixes.push(prefix.to_string());
            consumed.insert(weight);
            consumed.insert(name.clone());
            consumed.insert(biases);
        }
    }

    // Pass 2: build a quant weight per prefix.
    for prefix in prefixes {
        let weight_name = format!("{prefix}{SUFFIX_WEIGHT}");
        let scales_name = format!("{prefix}{SUFFIX_SCALES}");
        let biases_name = format!("{prefix}{SUFFIX_BIASES}");
        let weight = st.tensor(&weight_name)?;
        let scales = st.tensor(&scales_name)?;
        let biases = st.tensor(&biases_name)?;

        let layer = config.for_layer(&prefix).ok_or(MlxError::SkippedButPacked {
            module: prefix.clone(),
        })?;
        if layer.mode != MlxQuantMode::Affine {
            return Err(MlxError::UnsupportedMode {
                module: prefix.clone(),
                mode: layer.mode,
            });
        }

        // Packed weight must be uint32, 2-D [out_features, in_words].
        if weight.dtype() != StDtype::U32 {
            return Err(MlxError::WeightNotU32 {
                name: weight_name.clone(),
                dtype: weight.dtype(),
            });
        }
        let pshape: Vec<u64> = weight.shape().iter().map(|&d| d as u64).collect();
        if pshape.len() != 2 {
            return Err(MlxError::WeightNotMatrix {
                name: weight_name.clone(),
                shape: pshape,
            });
        }
        let out_features = pshape[0];
        let row_words = pshape[1];
        // in_features = (row_words * 32) / bits. MLX packs `bits*in`
        // bits per row into uint32 words; recover in from the word count.
        let row_bits = row_words * 32;
        if row_bits % layer.bits as u64 != 0 {
            return Err(MlxError::RowWidthIndivisible {
                name: weight_name.clone(),
                row_words,
                bits: layer.bits,
            });
        }
        let in_features = row_bits / layer.bits as u64;

        let scales_f32 = floats_to_f32(&scales_name, &scales)?;
        let biases_f32 = floats_to_f32(&biases_name, &biases)?;

        let q = MlxAffineQuant {
            // The uint32 tensor's raw LE bytes *are* the bitstream.
            packed: Arc::from(weight.data().to_vec()),
            scales: scales_f32,
            biases: biases_f32,
            group_size: layer.group_size,
            bits: layer.bits,
            shape: vec![out_features, in_features],
            name: weight_name.clone(),
        };
        // Cross-check packed length + scale/bias counts vs geometry.
        q.validate()?;
        quant.insert(prefix, q);
    }

    // Pass 3: everything not part of a quant triple is a full tensor.
    for (name, view) in st.tensors() {
        if consumed.contains(&name) {
            continue;
        }
        let dtype = full_dtype(&name, view.dtype())?;
        let shape: Vec<u64> = view.shape().iter().map(|&d| d as u64).collect();
        full.insert(
            name.clone(),
            MlxFullTensor {
                name,
                dtype,
                shape,
                bytes: view.data().to_vec(),
            },
        );
    }

    Ok(())
}

/// Load an MLX checkpoint from an already-parsed `config.json` string +
/// a single in-memory safetensors blob. The unit-testable core of the
/// loader (the directory entry point layers file IO + multi-shard merge
/// on top).
pub fn load_mlx_from_bytes(
    config_json: &str,
    safetensors_bytes: &[u8],
) -> Result<MlxModel, MlxError> {
    let config = MlxQuantConfig::parse_from_config_json(config_json)?
        .ok_or(MlxError::NotMlx)?;
    let mut quant = BTreeMap::new();
    let mut full = BTreeMap::new();
    load_shard_into(&config, safetensors_bytes, &mut quant, &mut full)?;
    Ok(MlxModel {
        config,
        quant,
        full,
    })
}

/// Load an MLX checkpoint from a model directory: read `config.json`,
/// enumerate every `*.safetensors` shard, and merge their tensors. The
/// tokenizer (`tokenizer.json`) is intentionally left to the caller /
/// `rustllama-tokenizer`.
///
/// Shards are discovered by extension rather than by parsing
/// `model.safetensors.index.json` — loading every shard is simpler and
/// robust to a missing / out-of-date index.
pub fn load_mlx_dir(dir: &Path) -> Result<MlxModel, MlxError> {
    let config_json = std::fs::read_to_string(dir.join("config.json"))?;
    let config = MlxQuantConfig::parse_from_config_json(&config_json)?
        .ok_or(MlxError::NotMlx)?;

    // Collect shard paths, sorted so the merge order is deterministic.
    let mut shards: Vec<std::path::PathBuf> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            shards.push(path);
        }
    }
    shards.sort();

    let mut quant = BTreeMap::new();
    let mut full = BTreeMap::new();
    for shard in shards {
        // mmap each shard (validated + parsed by safetensors on deserialize).
        let mmap = crate::open_safetensors(&shard)?;
        load_shard_into(&config, &mmap[..], &mut quant, &mut full)?;
    }
    Ok(MlxModel {
        config,
        quant,
        full,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::Dtype as StDtype;
    use std::collections::BTreeMap as Map;

    // --- config parsing ---------------------------------------------------

    #[test]
    fn parses_global_affine_config() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64,
            "quantization": {"group_size": 64, "bits": 4}
        }"#;
        let cfg = MlxQuantConfig::parse_from_config_json(json).unwrap().unwrap();
        assert_eq!(cfg.group_size, 64);
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.mode, MlxQuantMode::Affine); // default when absent
        // A layer with no override resolves to the globals.
        let l = cfg.for_layer("model.layers.0.self_attn.q_proj").unwrap();
        assert_eq!((l.group_size, l.bits), (64, 4));
    }

    #[test]
    fn parses_per_layer_overrides_and_skips() {
        let json = r#"{
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "model.layers.0.mlp.gate_proj": {"group_size": 32, "bits": 8},
                "lm_head": false,
                "model.embed_tokens": null
            }
        }"#;
        let cfg = MlxQuantConfig::parse_from_config_json(json).unwrap().unwrap();
        let over = cfg.for_layer("model.layers.0.mlp.gate_proj").unwrap();
        assert_eq!((over.group_size, over.bits), (32, 8));
        // `false` and `null` both mean "skip".
        assert!(cfg.for_layer("lm_head").is_none());
        assert!(cfg.for_layer("model.embed_tokens").is_none());
    }

    #[test]
    fn no_quant_block_is_not_mlx() {
        let json = r#"{"architectures": ["LlamaForCausalLM"], "hidden_size": 8}"#;
        assert!(MlxQuantConfig::parse_from_config_json(json).unwrap().is_none());
    }

    // --- tensor building --------------------------------------------------

    /// Pack `qs` (values pre-masked to `bits`) into an MLX little-endian
    /// bitstream, returned as uint32 LE bytes (what safetensors stores).
    fn pack_u32_le(qs: &[u32], bits: u32) -> Vec<u8> {
        let total_bits = qs.len() * bits as usize;
        assert_eq!(total_bits % 32, 0, "row must be uint32-aligned");
        let mut bytes = vec![0u8; total_bits / 8];
        let mut bit_pos = 0usize;
        for &q in qs {
            let mut got = 0u32;
            while got < bits {
                let abs = bit_pos + got as usize;
                let byte_idx = abs / 8;
                let bit_in_byte = (abs % 8) as u32;
                let avail = 8 - bit_in_byte;
                let take = avail.min(bits - got);
                let mask = (1u32 << take) - 1;
                bytes[byte_idx] |= (((q >> got) & mask) as u8) << bit_in_byte;
                got += take;
            }
            bit_pos += bits as usize;
        }
        bytes
    }

    fn f16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
            .collect()
    }

    /// Build a one-linear MLX safetensors blob:
    ///   `<p>.weight` (u32 packed), `<p>.scales`, `<p>.biases` (f16)
    /// plus a bare `model.norm.weight` full tensor. `group_size` divides
    /// `in_f`; `out_f` rows.
    fn make_mlx_blob(
        p: &str,
        out_f: usize,
        in_f: usize,
        group_size: usize,
        bits: u32,
    ) -> (Vec<u8>, Vec<u32>, Vec<f32>, Vec<f32>) {
        let n = out_f * in_f;
        let maxv = 1u32 << bits;
        let qs: Vec<u32> = (0..n).map(|i| (i as u32).wrapping_mul(2246822519) % maxv).collect();
        let row_words = in_f * bits as usize / 32;
        let packed = pack_u32_le(&qs, bits);

        // Use scale/bias values that are *exactly* representable in f16
        // (multiples of 2^-4 / 2^-5) so the f16 round-trip through the
        // loader is lossless and the test can assert exact equality.
        let n_groups = out_f * (in_f / group_size);
        let scales: Vec<f32> = (0..n_groups).map(|g| 0.5 + g as f32 * 0.0625).collect();
        let biases: Vec<f32> = (0..n_groups).map(|g| -0.25 + g as f32 * 0.03125).collect();

        let norm: Vec<f32> = vec![1.0; out_f];

        let scales_b = f16_bytes(&scales);
        let biases_b = f16_bytes(&biases);
        let norm_b = f16_bytes(&norm);

        // Hold owned byte buffers alive for the TensorView borrows.
        let mut map: Map<String, TensorView<'_>> = Map::new();
        let wv = TensorView::new(StDtype::U32, vec![out_f, row_words], &packed).unwrap();
        let sv = TensorView::new(StDtype::F16, vec![out_f, in_f / group_size], &scales_b).unwrap();
        let bv = TensorView::new(StDtype::F16, vec![out_f, in_f / group_size], &biases_b).unwrap();
        let nv = TensorView::new(StDtype::F16, vec![out_f], &norm_b).unwrap();
        map.insert(format!("{p}.weight"), wv);
        map.insert(format!("{p}.scales"), sv);
        map.insert(format!("{p}.biases"), bv);
        map.insert("model.norm.weight".to_string(), nv);
        let blob = safetensors::serialize(&map, &None).unwrap();
        (blob, qs, scales, biases)
    }

    #[test]
    fn loads_affine_triple_and_full_tensor() {
        let p = "model.layers.0.self_attn.q_proj";
        let (blob, qs, scales, biases) = make_mlx_blob(p, 4, 128, 64, 4);
        let cfg_json = r#"{"quantization": {"group_size": 64, "bits": 4}}"#;

        assert!(is_mlx_model(cfg_json, &blob));

        let model = load_mlx_from_bytes(cfg_json, &blob).unwrap();
        // One quant weight (keyed by module path) + one full tensor.
        assert_eq!(model.quant.len(), 1);
        assert_eq!(model.full.len(), 1);
        let q = model.quant.get(p).expect("quant present");
        assert_eq!(q.shape, vec![4, 128]);
        assert_eq!(q.group_size, 64);
        assert_eq!(q.bits, 4);
        assert_eq!(q.scales, scales);
        assert_eq!(q.biases, biases);
        q.validate().unwrap();

        // Decode via the kernels-cpu reference and check a couple cells
        // against the affine formula on the known q/scale/bias.
        let mut out = vec![0f32; q.n_elements() as usize];
        rustllama_kernels_cpu::mlx_affine::dequantize_mlx_affine(
            &q.packed, &q.scales, &q.biases, q.group_size, q.bits, &mut out,
        );
        for i in [0usize, 63, 64, 200, 511] {
            let g = i / q.group_size;
            let want = scales[g] * qs[i] as f32 + biases[g];
            assert!((out[i] - want).abs() < 1e-3, "cell {i}: {} vs {want}", out[i]);
        }

        let norm = model.full.get("model.norm.weight").expect("norm present");
        assert_eq!(norm.dtype, MlxFullDtype::F16);
        assert_eq!(norm.shape, vec![4]);
    }

    #[test]
    fn rejects_non_affine_mode() {
        let p = "model.layers.0.mlp.gate_proj";
        let (blob, _, _, _) = make_mlx_blob(p, 4, 64, 32, 4);
        let cfg_json = r#"{"quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4"}}"#;
        match load_mlx_from_bytes(cfg_json, &blob) {
            Err(MlxError::UnsupportedMode { mode: MlxQuantMode::Mxfp4, .. }) => {}
            other => panic!("expected UnsupportedMode, got {other:?}"),
        }
    }

    #[test]
    fn non_u32_weight_rejected() {
        // A `.weight`/`.scales`/`.biases` trio whose weight is F16, not
        // the required packed uint32.
        let n = 64usize;
        let w = f16_bytes(&vec![0.0; n]);
        let s = f16_bytes(&vec![1.0; 1]);
        let b = f16_bytes(&vec![0.0; 1]);
        let mut map: Map<String, TensorView<'_>> = Map::new();
        let p = "model.layers.0.self_attn.k_proj";
        map.insert(format!("{p}.weight"), TensorView::new(StDtype::F16, vec![1, n], &w).unwrap());
        map.insert(format!("{p}.scales"), TensorView::new(StDtype::F16, vec![1, 1], &s).unwrap());
        map.insert(format!("{p}.biases"), TensorView::new(StDtype::F16, vec![1, 1], &b).unwrap());
        let blob = safetensors::serialize(&map, &None).unwrap();
        let cfg_json = r#"{"quantization": {"group_size": 64, "bits": 4}}"#;
        match load_mlx_from_bytes(cfg_json, &blob) {
            Err(MlxError::WeightNotU32 { dtype: StDtype::F16, .. }) => {}
            other => panic!("expected WeightNotU32, got {other:?}"),
        }
    }

    #[test]
    fn detection_requires_scales_and_biases_siblings() {
        // Quant block present but no .scales/.biases tensors → not MLX.
        let norm = f16_bytes(&vec![1.0; 4]);
        let mut map: Map<String, TensorView<'_>> = Map::new();
        map.insert("model.norm.weight".into(), TensorView::new(StDtype::F16, vec![4], &norm).unwrap());
        let blob = safetensors::serialize(&map, &None).unwrap();
        let cfg_json = r#"{"quantization": {"group_size": 64, "bits": 4}}"#;
        assert!(!is_mlx_model(cfg_json, &blob));
    }
}
