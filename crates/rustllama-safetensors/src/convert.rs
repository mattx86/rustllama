//! Group HuggingFace safetensors tensors per linear and produce
//! GGUF-layout dense weights. The A-2b slice of the AWQ/GPTQ →
//! `LlamaModel` arc.
//!
//! For each unique target GGUF name produced by [`map_hf_to_gguf`],
//! this module:
//!
//! 1. Collects the source HF tensors (1 for Plain, 3 for AWQ
//!    `qweight + scales + qzeros`, 4 for GPTQ `+ g_idx`).
//! 2. Dispatches to the right dequant kernel.
//! 3. Transposes the `[in_features, out_features]` row-major f16
//!    output into `[out_features, in_features]` — the GGUF /
//!    `nn.Linear.weight` convention rustllama-models consumes.
//! 4. Returns the dense f16 (or pass-through) buffer plus shape.
//!
//! Bias tensors and per-tensor norms route through the `Plain` path
//! with a dtype conversion to f16 if the source isn't already f16
//! or f32.

use std::borrow::Cow;
use std::collections::BTreeMap;

use bytemuck::try_cast_slice;
use half::f16;
use safetensors::tensor::TensorView;
use safetensors::{Dtype as StDtype, SafeTensors};

use crate::dequant::{
    dequant_awq_int4_to_f16, dequant_gptq_int4_to_f16, DequantError,
};
use crate::name_map::{map_hf_to_gguf, HfTensorKind};

/// A converted tensor ready to hand to the model loader: the GGUF
/// name + shape + raw f16 (or passthrough) bytes. Shape is in
/// GGUF / PyTorch convention `[out_features, in_features]` for
/// linear weights; norm/embedding shapes pass through unchanged.
#[derive(Debug, Clone)]
pub struct ConvertedTensor {
    pub gguf_name: String,
    pub shape: Vec<u64>,
    /// Dtype of the bytes payload. v1 always emits F16 for quant
    /// trios (dequantized at load time, no MMQ path); Plain
    /// tensors keep their source dtype if it's already F32 or F16,
    /// otherwise convert to F16.
    pub dtype: ConvertedDtype,
    pub bytes: Vec<u8>,
}

/// Dtype of a [`ConvertedTensor`]'s bytes. Matches the rustllama-tensor
/// dtypes the model loader expects; A-2c plugs these into
/// `Tensor::from_storage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvertedDtype {
    F32,
    F16,
}

#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("safetensors deserialize: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("dequant: {0}")]
    Dequant(#[from] DequantError),
    #[error(
        "group `{gguf_name}` mixes incompatible roles: \
         qweight={qweight}, scales={scales}, qzeros={qzeros}, \
         g_idx={g_idx}, plain={plain} (a quant trio needs all of \
         qweight+scales+qzeros and no plain weight; plain tensors \
         must be alone in their group)"
    )]
    InconsistentGroup {
        gguf_name: String,
        qweight: usize,
        scales: usize,
        qzeros: usize,
        g_idx: usize,
        plain: usize,
    },
    #[error(
        "group `{gguf_name}` qweight has unsupported dtype {dtype:?}; \
         AWQ/GPTQ pack into int32"
    )]
    WrongQweightDtype { gguf_name: String, dtype: StDtype },
    #[error(
        "group `{gguf_name}` scales has unsupported dtype {dtype:?}; \
         AWQ/GPTQ store scales as fp16"
    )]
    WrongScalesDtype { gguf_name: String, dtype: StDtype },
    #[error(
        "group `{gguf_name}` qzeros has unsupported dtype {dtype:?}; \
         AWQ/GPTQ pack qzeros into int32"
    )]
    WrongQzerosDtype { gguf_name: String, dtype: StDtype },
    #[error(
        "group `{gguf_name}` g_idx has unsupported dtype {dtype:?}; \
         GPTQ stores g_idx as int32"
    )]
    WrongGIdxDtype { gguf_name: String, dtype: StDtype },
    #[error(
        "group `{gguf_name}` qweight is {dims} dimensions; expected 2"
    )]
    BadQweightRank { gguf_name: String, dims: usize },
    #[error(
        "group `{gguf_name}` scales is {dims} dimensions; expected 2"
    )]
    BadScalesRank { gguf_name: String, dims: usize },
    #[error(
        "plain tensor `{gguf_name}` has unsupported dtype {dtype:?}; \
         supported: F32, F16, BF16"
    )]
    UnsupportedPlainDtype {
        gguf_name: String,
        dtype: StDtype,
    },
}

/// Convert a safetensors file's contents to a `Vec<ConvertedTensor>`
/// in GGUF naming + layout. Each unique target GGUF name appears
/// exactly once in the output.
///
/// Tensors whose HF name doesn't map to any known GGUF slot are
/// skipped silently (rotary inv_freq, future / unknown keys) —
/// matches the A-2a contract.
pub fn convert_safetensors_to_gguf_tensors(
    bytes: &[u8],
) -> Result<Vec<ConvertedTensor>, ConvertError> {
    let st = SafeTensors::deserialize(bytes)?;

    // Group every recognized HF tensor under its target GGUF name.
    // Use `BTreeMap` so the output order is deterministic across
    // runs (the model loader doesn't care, but tests do).
    let mut groups: BTreeMap<String, Group<'_>> = BTreeMap::new();
    for (hf_name, view) in st.tensors() {
        let Some(mapped) = map_hf_to_gguf(&hf_name) else {
            continue;
        };
        let entry = groups.entry(mapped.gguf_name.clone()).or_insert_with(Group::new);
        entry.add(mapped.kind, view);
    }

    let mut out = Vec::with_capacity(groups.len());
    for (gguf_name, group) in groups {
        let converted = convert_group(&gguf_name, group)?;
        out.push(converted);
    }
    Ok(out)
}

/// One group of HF tensors that share a target GGUF name. The
/// gguf_name itself lives in the outer `BTreeMap`'s key; the group
/// only holds the typed slots.
struct Group<'a> {
    qweight: Option<TensorView<'a>>,
    scales: Option<TensorView<'a>>,
    qzeros: Option<TensorView<'a>>,
    g_idx: Option<TensorView<'a>>,
    plain: Option<TensorView<'a>>,
}

impl<'a> Group<'a> {
    fn new() -> Self {
        Self {
            qweight: None,
            scales: None,
            qzeros: None,
            g_idx: None,
            plain: None,
        }
    }

    fn add(&mut self, kind: HfTensorKind, view: TensorView<'a>) {
        match kind {
            HfTensorKind::Qweight => self.qweight = Some(view),
            HfTensorKind::Scales => self.scales = Some(view),
            HfTensorKind::Qzeros => self.qzeros = Some(view),
            HfTensorKind::GIdx => self.g_idx = Some(view),
            HfTensorKind::Plain => self.plain = Some(view),
        }
    }

    /// Total tensor count, used for the InconsistentGroup error.
    fn counts(&self) -> [usize; 5] {
        [
            self.qweight.is_some() as usize,
            self.scales.is_some() as usize,
            self.qzeros.is_some() as usize,
            self.g_idx.is_some() as usize,
            self.plain.is_some() as usize,
        ]
    }
}

fn convert_group(
    gguf_name: &str,
    group: Group<'_>,
) -> Result<ConvertedTensor, ConvertError> {
    let [q, s, z, g, p] = group.counts();
    // Quant trio: all three required core tensors present, no plain.
    let is_quant = q == 1 && s == 1 && z == 1 && p == 0;
    let is_plain = p == 1 && q == 0 && s == 0 && z == 0 && g == 0;
    if !is_quant && !is_plain {
        return Err(ConvertError::InconsistentGroup {
            gguf_name: gguf_name.into(),
            qweight: q,
            scales: s,
            qzeros: z,
            g_idx: g,
            plain: p,
        });
    }

    if is_plain {
        return convert_plain(gguf_name, group.plain.unwrap());
    }
    // Quant path. g_idx presence drives AWQ vs GPTQ dispatch.
    let qweight = group.qweight.unwrap();
    let scales = group.scales.unwrap();
    let qzeros = group.qzeros.unwrap();
    let g_idx_view = group.g_idx;
    convert_quant_trio(gguf_name, qweight, scales, qzeros, g_idx_view)
}

fn convert_plain(
    gguf_name: &str,
    view: TensorView<'_>,
) -> Result<ConvertedTensor, ConvertError> {
    let shape: Vec<u64> = view.shape().iter().map(|&d| d as u64).collect();
    match view.dtype() {
        StDtype::F32 => Ok(ConvertedTensor {
            gguf_name: gguf_name.into(),
            shape,
            dtype: ConvertedDtype::F32,
            bytes: view.data().to_vec(),
        }),
        StDtype::F16 => Ok(ConvertedTensor {
            gguf_name: gguf_name.into(),
            shape,
            dtype: ConvertedDtype::F16,
            bytes: view.data().to_vec(),
        }),
        StDtype::BF16 => {
            // BF16 → F16 conversion at load. Quantized HF models
            // often store norm weights as BF16 even when the quant
            // bits are int4. rustllama's loader handles BF16 raw,
            // but A-2c gets a cleaner job if everything is f16 here.
            let raw = view.data();
            let mut out = Vec::with_capacity(raw.len());
            // BF16 is the top 16 bits of f32. Cast each pair to f32
            // (zero the bottom 16 bits) then round to f16.
            for chunk in raw.chunks_exact(2) {
                let bf_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let f32_bits = (bf_bits as u32) << 16;
                let f = f32::from_bits(f32_bits);
                let h = f16::from_f32(f);
                out.extend_from_slice(&h.to_le_bytes());
            }
            Ok(ConvertedTensor {
                gguf_name: gguf_name.into(),
                shape,
                dtype: ConvertedDtype::F16,
                bytes: out,
            })
        }
        other => Err(ConvertError::UnsupportedPlainDtype {
            gguf_name: gguf_name.into(),
            dtype: other,
        }),
    }
}

/// Reinterpret a safetensors `i32` payload as `&[i32]` without copying
/// when the mmap slice is already 4-byte aligned. safetensors tensors
/// are borrowed straight from an mmap whose per-tensor offset carries
/// no alignment guarantee, so a direct `bytemuck::cast_slice::<u8, i32>`
/// would panic (`TargetAlignmentGreaterAndInputNotAligned`) on an
/// unaligned tensor. Fall back to an owned, correctly-aligned
/// `Vec<i32>` decoded from little-endian bytes in that case.
fn as_i32_slice(bytes: &[u8]) -> Cow<'_, [i32]> {
    match try_cast_slice::<u8, i32>(bytes) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => {
            let mut v = Vec::with_capacity(bytes.len() / 4);
            for c in bytes.chunks_exact(4) {
                v.push(i32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Cow::Owned(v)
        }
    }
}

/// f16 analogue of [`as_i32_slice`]. AWQ/GPTQ scales ship as fp16 and
/// share the same unaligned-mmap hazard (2-byte alignment for `f16`).
fn as_f16_slice(bytes: &[u8]) -> Cow<'_, [f16]> {
    match try_cast_slice::<u8, f16>(bytes) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => {
            let mut v = Vec::with_capacity(bytes.len() / 2);
            for c in bytes.chunks_exact(2) {
                v.push(f16::from_bits(u16::from_le_bytes([c[0], c[1]])));
            }
            Cow::Owned(v)
        }
    }
}

fn convert_quant_trio(
    gguf_name: &str,
    qweight: TensorView<'_>,
    scales: TensorView<'_>,
    qzeros: TensorView<'_>,
    g_idx: Option<TensorView<'_>>,
) -> Result<ConvertedTensor, ConvertError> {
    // Dtype checks: qweight + qzeros must be I32, scales must be F16.
    if qweight.dtype() != StDtype::I32 {
        return Err(ConvertError::WrongQweightDtype {
            gguf_name: gguf_name.into(),
            dtype: qweight.dtype(),
        });
    }
    if scales.dtype() != StDtype::F16 {
        return Err(ConvertError::WrongScalesDtype {
            gguf_name: gguf_name.into(),
            dtype: scales.dtype(),
        });
    }
    if qzeros.dtype() != StDtype::I32 {
        return Err(ConvertError::WrongQzerosDtype {
            gguf_name: gguf_name.into(),
            dtype: qzeros.dtype(),
        });
    }
    if qweight.shape().len() != 2 {
        return Err(ConvertError::BadQweightRank {
            gguf_name: gguf_name.into(),
            dims: qweight.shape().len(),
        });
    }
    if scales.shape().len() != 2 {
        return Err(ConvertError::BadScalesRank {
            gguf_name: gguf_name.into(),
            dims: scales.shape().len(),
        });
    }
    // safetensors payloads are borrowed straight from the mmap; a
    // tensor whose byte offset isn't 4-/2-aligned would make a direct
    // `cast_slice` panic. `as_*_slice` borrows when aligned and copies
    // into an aligned buffer otherwise.
    let qweight_i32 = as_i32_slice(qweight.data());
    let scales_f16 = as_f16_slice(scales.data());
    let qzeros_i32 = as_i32_slice(qzeros.data());

    // Dispatch: AWQ if no g_idx; GPTQ if g_idx is present.
    let (in_features, out_features, dequant) = match g_idx {
        None => {
            // AWQ layout: qweight is [in_features, out_features / 8].
            let in_f = qweight.shape()[0];
            let out_packs = qweight.shape()[1];
            let out_f = out_packs * 8;
            // Group size derives from in_features / n_groups.
            // Scales shape is [n_groups, out_features].
            let n_groups = scales.shape()[0];
            if n_groups == 0 || in_f % n_groups != 0 {
                return Err(ConvertError::Dequant(
                    DequantError::GroupSizeMisaligned {
                        group_size: 0,
                        in_features: in_f,
                    },
                ));
            }
            let group_size = in_f / n_groups;
            let buf = dequant_awq_int4_to_f16(
                &qweight_i32,
                &scales_f16,
                &qzeros_i32,
                in_f,
                out_f,
                group_size,
            )?;
            (in_f, out_f, buf)
        }
        Some(g_idx_view) => {
            if g_idx_view.dtype() != StDtype::I32 {
                return Err(ConvertError::WrongGIdxDtype {
                    gguf_name: gguf_name.into(),
                    dtype: g_idx_view.dtype(),
                });
            }
            // GPTQ layout: qweight is [in_features / 8, out_features].
            let in_packs = qweight.shape()[0];
            let in_f = in_packs * 8;
            let out_f = qweight.shape()[1];
            let g_idx_i32 = as_i32_slice(g_idx_view.data());
            let buf = dequant_gptq_int4_to_f16(
                &qweight_i32,
                &scales_f16,
                &qzeros_i32,
                &g_idx_i32,
                in_f,
                out_f,
            )?;
            (in_f, out_f, buf)
        }
    };

    // Transpose [in_features, out_features] → [out_features, in_features]
    // so the output matches PyTorch / GGUF nn.Linear.weight convention.
    let mut transposed = vec![f16::ZERO; in_features * out_features];
    for i in 0..in_features {
        for j in 0..out_features {
            transposed[j * in_features + i] = dequant[i * out_features + j];
        }
    }
    let bytes: Vec<u8> = transposed
        .iter()
        .flat_map(|h| h.to_le_bytes())
        .collect();
    Ok(ConvertedTensor {
        gguf_name: gguf_name.into(),
        shape: vec![out_features as u64, in_features as u64],
        dtype: ConvertedDtype::F16,
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant::AWQ_PACK_ORDER;
    use bytemuck::cast_slice;
    use std::collections::BTreeMap;

    fn pack8(lanes: [u8; 8]) -> i32 {
        let mut acc: u32 = 0;
        for (k, v) in lanes.iter().enumerate() {
            acc |= ((*v as u32) & 0xF) << (k * 4);
        }
        acc as i32
    }

    fn i32_bytes(v: &[i32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn f16_bytes(v: &[f16]) -> Vec<u8> {
        v.iter().flat_map(|h| h.to_le_bytes()).collect()
    }

    /// Build an AWQ-shape safetensors blob for one Q linear layer
    /// (`model.layers.0.self_attn.q_proj`) + one bare norm tensor.
    fn make_awq_blob(in_f: usize, out_f: usize, group_size: usize) -> Vec<u8> {
        let out_packs = out_f / 8;
        let n_groups = in_f / group_size;
        // qweight: [in_f, out_f/8] all lanes = 5
        let qw: Vec<i32> = (0..in_f * out_packs).map(|_| pack8([5; 8])).collect();
        let qw_bytes = i32_bytes(&qw);
        // scales: [n_groups, out_f] all = 0.5
        let sc: Vec<f16> = vec![f16::from_f32(0.5); n_groups * out_f];
        let sc_bytes = f16_bytes(&sc);
        // qzeros: [n_groups, out_f/8] all lanes = 2
        let qz: Vec<i32> = (0..n_groups * out_packs).map(|_| pack8([2; 8])).collect();
        let qz_bytes = i32_bytes(&qz);
        // norm: [d_model] f16 (use out_f as the dim — arbitrary)
        let norm: Vec<f16> = vec![f16::from_f32(1.0); out_f];
        let norm_bytes = f16_bytes(&norm);

        let mut map: BTreeMap<String, TensorView<'_>> = BTreeMap::new();
        let qw_view = TensorView::new(StDtype::I32, vec![in_f, out_packs], &qw_bytes)
            .expect("qweight view");
        let sc_view = TensorView::new(StDtype::F16, vec![n_groups, out_f], &sc_bytes)
            .expect("scales view");
        let qz_view = TensorView::new(StDtype::I32, vec![n_groups, out_packs], &qz_bytes)
            .expect("qzeros view");
        let norm_view = TensorView::new(StDtype::F16, vec![out_f], &norm_bytes)
            .expect("norm view");
        map.insert(
            "model.layers.0.self_attn.q_proj.qweight".into(),
            qw_view,
        );
        map.insert(
            "model.layers.0.self_attn.q_proj.scales".into(),
            sc_view,
        );
        map.insert(
            "model.layers.0.self_attn.q_proj.qzeros".into(),
            qz_view,
        );
        map.insert("model.layers.0.input_layernorm.weight".into(), norm_view);
        safetensors::serialize(&map, &None).expect("serialize")
    }

    #[test]
    fn convert_awq_blob_produces_dequanted_q_proj_and_passthrough_norm() {
        // (w - z) * s = (5 - 2) * 0.5 = 1.5 for every cell.
        let in_f = 8; // 1 group at group_size=8
        let out_f = 16;
        let blob = make_awq_blob(in_f, out_f, 8);
        let tensors = convert_safetensors_to_gguf_tensors(&blob).unwrap();
        // Expect two output tensors, in BTreeMap order:
        //   "blk.0.attn_norm.weight" (Plain, f16)
        //   "blk.0.attn_q.weight"    (dequanted, f16, [out_f, in_f])
        let by_name: BTreeMap<String, ConvertedTensor> =
            tensors.iter().map(|t| (t.gguf_name.clone(), t.clone())).collect();
        let q = by_name.get("blk.0.attn_q.weight").expect("q dequanted");
        assert_eq!(q.dtype, ConvertedDtype::F16);
        assert_eq!(q.shape, vec![out_f as u64, in_f as u64]);
        // Every cell should be (5-2)*0.5 = 1.5
        let q_vals: &[f16] = cast_slice(&q.bytes);
        for (i, v) in q_vals.iter().enumerate() {
            assert!(
                (v.to_f32() - 1.5).abs() < 1e-3,
                "cell {i}: got {}",
                v.to_f32(),
            );
        }
        let n = by_name.get("blk.0.attn_norm.weight").expect("norm plain");
        assert_eq!(n.dtype, ConvertedDtype::F16);
        let n_vals: &[f16] = cast_slice(&n.bytes);
        assert!(n_vals.iter().all(|h| (h.to_f32() - 1.0).abs() < 1e-3));
    }

    #[test]
    fn convert_awq_blob_transposes_to_out_in_convention() {
        // Use distinguishable per-column scales so a missing transpose
        // would surface as a cell-mismatch.
        let in_f = 8;
        let out_f = 16;
        let out_packs = out_f / 8;
        // group_size is implicit: in_f == 8 == in_f / n_groups means
        // one group covers the whole input dimension. Pinned via the
        // shape declaration of `scales` below ([n_groups=1, out_f]).
        let n_groups = 1;
        // qweight: output column j holds int4 = (i+j) mod 16. autoawq
        // interleaves the 8 columns of a pack, so nibble position k
        // stores the value destined for output column
        // jp*8 + AWQ_PACK_ORDER[k]. Packing this way keeps the
        // post-dequant assertion below in clean output-column order —
        // and it fails loudly if the dequant ignores the interleave.
        let mut qw: Vec<i32> = Vec::with_capacity(in_f * out_packs);
        for i in 0..in_f {
            for jp in 0..out_packs {
                let mut lanes = [0u8; 8];
                for k in 0..8 {
                    let j = jp * 8 + AWQ_PACK_ORDER[k];
                    lanes[k] = ((i + j) % 16) as u8;
                }
                qw.push(pack8(lanes));
            }
        }
        let qw_bytes = i32_bytes(&qw);
        // scales[g=0, j] = j+1 (distinct per column).
        let sc: Vec<f16> = (0..out_f).map(|j| f16::from_f32((j + 1) as f32)).collect();
        let sc_bytes = f16_bytes(&sc);
        // zeros all 0 so dequant = w * s.
        let qz: Vec<i32> = vec![pack8([0; 8]); out_packs];
        let qz_bytes = i32_bytes(&qz);

        let mut map: BTreeMap<String, TensorView<'_>> = BTreeMap::new();
        let qw_view = TensorView::new(StDtype::I32, vec![in_f, out_packs], &qw_bytes)
            .expect("qw view");
        let sc_view = TensorView::new(StDtype::F16, vec![n_groups, out_f], &sc_bytes)
            .expect("sc view");
        let qz_view = TensorView::new(StDtype::I32, vec![n_groups, out_packs], &qz_bytes)
            .expect("qz view");
        map.insert(
            "model.layers.0.self_attn.q_proj.qweight".into(),
            qw_view,
        );
        map.insert(
            "model.layers.0.self_attn.q_proj.scales".into(),
            sc_view,
        );
        map.insert(
            "model.layers.0.self_attn.q_proj.qzeros".into(),
            qz_view,
        );
        let blob = safetensors::serialize(&map, &None).unwrap();
        let tensors = convert_safetensors_to_gguf_tensors(&blob).unwrap();
        let q = tensors
            .iter()
            .find(|t| t.gguf_name == "blk.0.attn_q.weight")
            .expect("q dequanted");
        assert_eq!(q.shape, vec![out_f as u64, in_f as u64]);
        let vals: &[f16] = cast_slice(&q.bytes);
        // Output layout: row-major [out_f, in_f]. Cell (j, i) should
        // equal ((i+j)%16) * (j+1).
        for j in 0..out_f {
            for i in 0..in_f {
                let cell = vals[j * in_f + i].to_f32();
                let exp = (((i + j) % 16) as f32) * ((j + 1) as f32);
                assert!(
                    (cell - exp).abs() < 1e-2,
                    "cell (j={j}, i={i}) got {cell} exp {exp}",
                );
            }
        }
    }

    #[test]
    fn convert_gptq_blob_dispatches_via_g_idx() {
        // 1 group, in_f=8, out_f=8. g_idx all zero. Reuse the AWQ
        // single-block formula but add a g_idx tensor — that's the
        // discriminator for GPTQ dispatch.
        let in_f = 8;
        let out_f = 8;
        let in_packs = in_f / 8;
        let out_packs = out_f / 8;
        // qweight: [in_f/8, out_f]. One row, 8 columns. Col j packs
        // lanes [j, j+1, ..., j+7].
        let mut qw: Vec<i32> = Vec::new();
        for j in 0..out_f {
            let mut lanes = [0u8; 8];
            for k in 0..8 {
                lanes[k] = ((j + k) % 16) as u8;
            }
            qw.push(pack8(lanes));
        }
        let qw_bytes = i32_bytes(&qw);
        // scales: [1 group, out_f] all = 0.5
        let sc: Vec<f16> = vec![f16::from_f32(0.5); out_f];
        let sc_bytes = f16_bytes(&sc);
        // qzeros: [1 group, out_f/8] lanes all 2 → actual zero after +1 = 3
        let qz: Vec<i32> = vec![pack8([2; 8])];
        let qz_bytes = i32_bytes(&qz);
        // g_idx: [in_f] all 0
        let gi: Vec<i32> = vec![0i32; in_f];
        let gi_bytes = i32_bytes(&gi);

        let mut map: BTreeMap<String, TensorView<'_>> = BTreeMap::new();
        let qw_view = TensorView::new(StDtype::I32, vec![in_packs, out_f], &qw_bytes)
            .expect("qw view");
        let sc_view = TensorView::new(StDtype::F16, vec![1, out_f], &sc_bytes)
            .expect("sc view");
        let qz_view = TensorView::new(StDtype::I32, vec![1, out_packs], &qz_bytes)
            .expect("qz view");
        let gi_view = TensorView::new(StDtype::I32, vec![in_f], &gi_bytes)
            .expect("g_idx view");
        map.insert(
            "model.layers.0.self_attn.k_proj.qweight".into(),
            qw_view,
        );
        map.insert(
            "model.layers.0.self_attn.k_proj.scales".into(),
            sc_view,
        );
        map.insert(
            "model.layers.0.self_attn.k_proj.qzeros".into(),
            qz_view,
        );
        map.insert("model.layers.0.self_attn.k_proj.g_idx".into(), gi_view);
        let blob = safetensors::serialize(&map, &None).unwrap();
        let tensors = convert_safetensors_to_gguf_tensors(&blob).unwrap();
        let k = tensors
            .iter()
            .find(|t| t.gguf_name == "blk.0.attn_k.weight")
            .expect("k dequanted");
        // [out_f, in_f] in GPTQ convention with the +1 zero-point.
        assert_eq!(k.shape, vec![out_f as u64, in_f as u64]);
        let vals: &[f16] = cast_slice(&k.bytes);
        // After transpose: cell (j, i) = (((j+i)%16) - 3) * 0.5
        for j in 0..out_f {
            for i in 0..in_f {
                let cell = vals[j * in_f + i].to_f32();
                let w = ((j + i) % 16) as i32;
                let exp = (w - 3) as f32 * 0.5;
                assert!(
                    (cell - exp).abs() < 1e-2,
                    "cell (j={j}, i={i}) got {cell} exp {exp}",
                );
            }
        }
    }

    #[test]
    fn inconsistent_group_with_plain_plus_quant_errors() {
        // Pathological: qweight + a `.weight` plain tensor land in the
        // same GGUF group. Must reject — we can't tell which one wins.
        let in_f = 8;
        let out_f = 16;
        let blob = make_awq_blob(in_f, out_f, 8);
        // Slip a `model.layers.0.self_attn.q_proj.weight` into the
        // mix to force a Plain into the q_proj group.
        let plain: Vec<f16> = vec![f16::from_f32(0.0); in_f * out_f];
        let plain_bytes = f16_bytes(&plain);
        let mut map: BTreeMap<String, TensorView<'_>> = BTreeMap::new();
        // Re-parse the blob, copy its tensors into the new map.
        let st = SafeTensors::deserialize(&blob).unwrap();
        for (name, view) in st.tensors() {
            map.insert(name, view);
        }
        let plain_view =
            TensorView::new(StDtype::F16, vec![in_f, out_f], &plain_bytes)
                .expect("plain view");
        map.insert(
            "model.layers.0.self_attn.q_proj.weight".into(),
            plain_view,
        );
        let merged = safetensors::serialize(&map, &None).unwrap();
        let err = convert_safetensors_to_gguf_tensors(&merged).unwrap_err();
        match err {
            ConvertError::InconsistentGroup { ref gguf_name, .. }
                if gguf_name == "blk.0.attn_q.weight" => {}
            other => panic!("expected InconsistentGroup for q_proj, got {other:?}"),
        }
    }

    #[test]
    fn missing_qzeros_in_quant_trio_errors() {
        // qweight + scales but no qzeros — InconsistentGroup.
        let in_f = 8;
        let out_f = 16;
        let out_packs = out_f / 8;
        let group_size = 8;
        let n_groups = in_f / group_size;
        let qw: Vec<i32> = (0..in_f * out_packs).map(|_| pack8([0; 8])).collect();
        let qw_bytes = i32_bytes(&qw);
        let sc: Vec<f16> = vec![f16::from_f32(1.0); n_groups * out_f];
        let sc_bytes = f16_bytes(&sc);
        let mut map: BTreeMap<String, TensorView<'_>> = BTreeMap::new();
        let qw_view = TensorView::new(StDtype::I32, vec![in_f, out_packs], &qw_bytes)
            .unwrap();
        let sc_view = TensorView::new(StDtype::F16, vec![n_groups, out_f], &sc_bytes)
            .unwrap();
        map.insert(
            "model.layers.0.self_attn.q_proj.qweight".into(),
            qw_view,
        );
        map.insert(
            "model.layers.0.self_attn.q_proj.scales".into(),
            sc_view,
        );
        let blob = safetensors::serialize(&map, &None).unwrap();
        let err = convert_safetensors_to_gguf_tensors(&blob).unwrap_err();
        match err {
            ConvertError::InconsistentGroup {
                qweight: 1,
                scales: 1,
                qzeros: 0,
                ..
            } => {}
            other => panic!("expected InconsistentGroup, got {other:?}"),
        }
    }

    #[test]
    fn wrong_qweight_dtype_errors() {
        // qweight encoded as F16 instead of I32 — reject loudly so
        // the user knows the file isn't real AWQ/GPTQ.
        let in_f = 8;
        let out_f = 16;
        let out_packs = out_f / 8;
        let group_size = 8;
        let n_groups = in_f / group_size;
        let qw_f16: Vec<f16> = vec![f16::ZERO; in_f * out_packs * 2];
        let qw_bytes = f16_bytes(&qw_f16);
        let sc: Vec<f16> = vec![f16::ZERO; n_groups * out_f];
        let sc_bytes = f16_bytes(&sc);
        let qz: Vec<i32> = vec![0i32; n_groups * out_packs];
        let qz_bytes = i32_bytes(&qz);
        let mut map: BTreeMap<String, TensorView<'_>> = BTreeMap::new();
        // Use F16 dtype despite name being qweight — should be I32.
        let qw_view = TensorView::new(StDtype::F16, vec![in_f, out_packs * 2], &qw_bytes)
            .unwrap();
        let sc_view = TensorView::new(StDtype::F16, vec![n_groups, out_f], &sc_bytes)
            .unwrap();
        let qz_view = TensorView::new(StDtype::I32, vec![n_groups, out_packs], &qz_bytes)
            .unwrap();
        map.insert(
            "model.layers.0.self_attn.q_proj.qweight".into(),
            qw_view,
        );
        map.insert(
            "model.layers.0.self_attn.q_proj.scales".into(),
            sc_view,
        );
        map.insert(
            "model.layers.0.self_attn.q_proj.qzeros".into(),
            qz_view,
        );
        let blob = safetensors::serialize(&map, &None).unwrap();
        let err = convert_safetensors_to_gguf_tensors(&blob).unwrap_err();
        match err {
            ConvertError::WrongQweightDtype { dtype: StDtype::F16, .. } => {}
            other => panic!("expected WrongQweightDtype, got {other:?}"),
        }
    }

    #[test]
    fn unrecognized_hf_tensor_is_skipped_silently() {
        // A tensor whose HF name doesn't map (e.g. `vision_tower.*`)
        // should be omitted from the output rather than aborting.
        let dummy = vec![0u8, 0u8];
        let mut map: BTreeMap<String, TensorView<'_>> = BTreeMap::new();
        let view = TensorView::new(StDtype::F16, vec![1usize], &dummy).unwrap();
        map.insert("vision_tower.layers.0.weight".into(), view);
        // Plus one recognized tensor so we don't get an empty output.
        let norm: Vec<f16> = vec![f16::from_f32(1.0); 4];
        let norm_bytes = f16_bytes(&norm);
        let norm_view = TensorView::new(StDtype::F16, vec![4usize], &norm_bytes)
            .unwrap();
        map.insert("model.norm.weight".into(), norm_view);
        let blob = safetensors::serialize(&map, &None).unwrap();
        let tensors = convert_safetensors_to_gguf_tensors(&blob).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].gguf_name, "output_norm.weight");
    }

    #[test]
    fn bf16_plain_tensor_converts_to_f16() {
        // Qwen2.5-Coder AWQ ships norm weights in BF16. Verify the
        // load path converts cleanly.
        let n = 8usize;
        // bf16 of 1.0 = 0x3F80 (sign 0, exp 127, mantissa 0)
        let bf_bytes: Vec<u8> =
            std::iter::repeat([0x80u8, 0x3Fu8]).take(n).flatten().collect();
        let mut map: BTreeMap<String, TensorView<'_>> = BTreeMap::new();
        let view = TensorView::new(StDtype::BF16, vec![n], &bf_bytes).unwrap();
        map.insert("model.norm.weight".into(), view);
        let blob = safetensors::serialize(&map, &None).unwrap();
        let tensors = convert_safetensors_to_gguf_tensors(&blob).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].dtype, ConvertedDtype::F16);
        let vals: &[f16] = cast_slice(&tensors[0].bytes);
        assert_eq!(vals.len(), n);
        for v in vals {
            assert!(
                (v.to_f32() - 1.0).abs() < 1e-3,
                "bf16(1.0) should round to f16(1.0); got {}",
                v.to_f32(),
            );
        }
    }
}
