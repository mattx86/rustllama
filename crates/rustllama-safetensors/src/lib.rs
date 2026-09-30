//! HuggingFace safetensors loader for **AWQ** and **GPTQ** quantized
//! models.
//!
//! # Background
//!
//! Most quantized coding-LLM checkpoints released on HuggingFace ship
//! in one of two related 4-bit-per-group formats:
//!
//! - **AWQ** (Activation-aware Weight Quantization, Lin et al. 2023)
//! - **GPTQ** (Frantar et al. 2022)
//!
//! Both store packed 4-bit weights + per-group fp16 scales + per-group
//! zero points, layered on top of the standard `.safetensors`
//! container format. They differ on:
//!
//! - The bit-packing layout of `qweight` (column-major 8x4-bit per
//!   int32 vs row-major 4-bit pairs).
//! - The presence of a per-channel re-ordering index (`g_idx`) — GPTQ
//!   uses it for actorder grouping; AWQ does not emit one.
//! - The zero-point semantics (asymmetric vs symmetric).
//!
//! # A-0 scope
//!
//! This module ships the *detection* primitive: open a `.safetensors`
//! file, inspect its header, classify the quant scheme by looking at
//! tensor-name suffixes (`qweight` / `scales` / `qzeros` / `g_idx`),
//! and return a [`SafetensorsFormat`] enum the engine can dispatch on.
//!
//! Subsequent slices land:
//! - A-1: 4-bit dequant kernels for AWQ + GPTQ → fp16 row-major
//! - A-2: full safetensors → `LlamaModel` weight conversion path
//! - A-3: auto-detect at engine load (`.gguf` vs `.safetensors`)

use std::path::Path;

use safetensors::SafeTensors;

pub mod build;
pub mod convert;
pub mod dequant;
pub mod hf_config;
pub mod mlx;
pub mod name_map;
pub use build::{build_llama_model_from_safetensors, BuildError};
pub use convert::{
    convert_safetensors_to_gguf_tensors, ConvertError, ConvertedDtype,
    ConvertedTensor,
};
pub use dequant::{
    dequant_awq_int4_to_f16, dequant_gptq_int4_to_f16, DequantError,
};
pub use hf_config::{parse_hf_config, HfConfigError};
pub use mlx::{
    is_mlx_model, load_mlx_dir, load_mlx_from_bytes, MlxError, MlxFullDtype,
    MlxFullTensor, MlxLayerQuant, MlxModel, MlxQuantConfig, MlxQuantMode,
};
pub use name_map::{map_hf_to_gguf, HfTensor, HfTensorKind};

/// Classification of a safetensors file's quant scheme. Detected by
/// scanning tensor names for the suffixes each format emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetensorsFormat {
    /// AWQ-style packing: `qweight` + `scales` + `qzeros`, **no**
    /// `g_idx`. Each block of 8 4-bit weights is packed column-major
    /// into one int32.
    AwqInt4,
    /// GPTQ-style packing: `qweight` + `scales` + `qzeros` + `g_idx`.
    /// The `g_idx` per-channel index distinguishes GPTQ from AWQ —
    /// it carries the "actorder" group remapping the GPTQ algorithm
    /// produces during quantization.
    GptqInt4,
    /// Plain `.safetensors` with no quant suffixes — most likely an
    /// unquantized fp16/bf16/fp32 HF checkpoint. The loader can fall
    /// through to a dequantization-free path for these.
    Unquantized,
    /// Some quant tensors found (e.g. `qweight`) but the layout
    /// doesn't match either AWQ or GPTQ. Reject loudly rather than
    /// guess — likely a newer / experimental format we don't yet
    /// support.
    Unknown,
}

#[derive(Debug, thiserror::Error)]
pub enum SafetensorsError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors deserialize: {0}")]
    Deserialize(#[from] safetensors::SafeTensorError),
    #[error("file is empty: {path}")]
    Empty { path: String },
    #[error("file does not exist: {path}")]
    NotFound { path: String },
}

/// Map a `.safetensors` file into memory and return an iteration-ready
/// handle. Wraps `safetensors::SafeTensors::deserialize` over a
/// `memmap2::Mmap` so the per-tensor byte slices stay live for the
/// caller's borrow lifetime without copying.
///
/// Returned tuple: `(mmap, header_string)`. The `header` is the raw
/// JSON metadata string used by [`detect_format`]; callers that only
/// need the format classification can drop the mmap immediately.
pub fn open_safetensors(path: &Path) -> Result<memmap2::Mmap, SafetensorsError> {
    let path_str = path.display().to_string();
    if !path.exists() {
        return Err(SafetensorsError::NotFound { path: path_str });
    }
    let file = std::fs::File::open(path)?;
    let meta = file.metadata()?;
    if meta.len() == 0 {
        return Err(SafetensorsError::Empty { path: path_str });
    }
    // SAFETY: read-only mmap of a file we opened; safetensors crate
    // re-validates the header on parse, so a bad file surfaces as
    // Deserialize, not a memory-safety failure.
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    Ok(mmap)
}

/// Inspect a safetensors file and classify its quant scheme.
///
/// Detection rule (matches the HuggingFace `transformers` /
/// `autoawq` / `auto-gptq` conventions used in the wild):
///
/// - Any tensor name ending in `g_idx` => `GptqInt4`.
///   (`g_idx` is GPTQ's actorder remap and is never present in AWQ.)
/// - Otherwise, any tensor name ending in `qweight` + `scales` +
///   `qzeros` (all three present somewhere in the file) => `AwqInt4`.
/// - Otherwise, no `qweight` anywhere => `Unquantized`.
/// - Otherwise => `Unknown` (some packing-suffix found but layout
///   doesn't match either format).
pub fn detect_format(path: &Path) -> Result<SafetensorsFormat, SafetensorsError> {
    let mmap = open_safetensors(path)?;
    detect_format_from_bytes(&mmap[..])
}

/// Slice-level variant of [`detect_format`] for tests that mint a
/// safetensors header in memory without writing it to disk.
pub fn detect_format_from_bytes(
    bytes: &[u8],
) -> Result<SafetensorsFormat, SafetensorsError> {
    let st = SafeTensors::deserialize(bytes)?;
    let names: Vec<&str> = st.names().into_iter().map(|s| s.as_str()).collect();

    let has_g_idx = names.iter().any(|n| ends_with_segment(n, "g_idx"));
    if has_g_idx {
        return Ok(SafetensorsFormat::GptqInt4);
    }
    let has_qweight = names.iter().any(|n| ends_with_segment(n, "qweight"));
    let has_scales = names.iter().any(|n| ends_with_segment(n, "scales"));
    let has_qzeros = names.iter().any(|n| ends_with_segment(n, "qzeros"));
    if has_qweight && has_scales && has_qzeros {
        return Ok(SafetensorsFormat::AwqInt4);
    }
    if !has_qweight && !has_qzeros {
        return Ok(SafetensorsFormat::Unquantized);
    }
    Ok(SafetensorsFormat::Unknown)
}

/// Does `name` end with `.suffix` (or equal `suffix` exactly)?
/// Used to match `model.layers.0.self_attn.q_proj.qweight` against
/// the literal `"qweight"` without a regex.
fn ends_with_segment(name: &str, suffix: &str) -> bool {
    if name == suffix {
        return true;
    }
    if let Some(rest) = name.strip_suffix(suffix) {
        return rest.ends_with('.');
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;
    use std::collections::BTreeMap;

    /// Build a minimal in-memory safetensors blob containing tensors
    /// with the provided names + a few bytes of fp16 zero payload.
    /// Header is what the safetensors crate produces from
    /// `serialize`, so the layout is byte-identical to a real
    /// HuggingFace file at the size scale tests care about.
    fn make_safetensors(names: &[&str]) -> Vec<u8> {
        // Minimal payload: each tensor is a [1] fp16 zero. Dtype +
        // shape + the actual two bytes are all the safetensors header
        // validates; the detection logic only inspects names.
        let one_fp16 = vec![0u8, 0u8];
        let mut tensors: BTreeMap<String, TensorView<'_>> = BTreeMap::new();
        for name in names {
            let view = TensorView::new(Dtype::F16, vec![1usize], &one_fp16)
                .expect("tensor view");
            tensors.insert((*name).to_string(), view);
        }
        safetensors::serialize(&tensors, &None).expect("serialize")
    }

    #[test]
    fn awq_layout_detected() {
        // Real-world AWQ HF model exposes (per layer, per linear):
        //   <prefix>.qweight  <prefix>.scales  <prefix>.qzeros
        // and NO g_idx anywhere in the file.
        let blob = make_safetensors(&[
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.qzeros",
            "model.norm.weight",
        ]);
        assert_eq!(
            detect_format_from_bytes(&blob).unwrap(),
            SafetensorsFormat::AwqInt4
        );
    }

    #[test]
    fn gptq_layout_detected_via_g_idx() {
        // GPTQ has the same qweight/scales/qzeros trio as AWQ but
        // ALSO ships a g_idx per linear — that's the discriminator.
        let blob = make_safetensors(&[
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.qzeros",
            "model.layers.0.self_attn.q_proj.g_idx",
        ]);
        assert_eq!(
            detect_format_from_bytes(&blob).unwrap(),
            SafetensorsFormat::GptqInt4
        );
    }

    #[test]
    fn g_idx_anywhere_classifies_as_gptq_even_with_full_awq_trio() {
        // The g_idx check fires first by design: any single g_idx in
        // the file means GPTQ. Pin that ordering — AWQ checkpoints
        // never emit g_idx, so a file with any of them must be GPTQ
        // (or a future variant we'd reject as Unknown via a different
        // signal).
        let blob = make_safetensors(&[
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.qzeros",
            "model.layers.7.mlp.gate_proj.g_idx",
        ]);
        assert_eq!(
            detect_format_from_bytes(&blob).unwrap(),
            SafetensorsFormat::GptqInt4
        );
    }

    #[test]
    fn unquantized_fp16_checkpoint_detected() {
        // Bare HF fp16 checkpoint: no qweight/qzeros/g_idx anywhere.
        let blob = make_safetensors(&[
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]);
        assert_eq!(
            detect_format_from_bytes(&blob).unwrap(),
            SafetensorsFormat::Unquantized
        );
    }

    #[test]
    fn partial_quant_layout_classifies_as_unknown() {
        // qweight present but no qzeros + no g_idx → not AWQ, not
        // GPTQ, but clearly quantized. Reject loudly so we don't
        // hand a half-known layout to the load path.
        let blob = make_safetensors(&[
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.scales",
            "model.norm.weight",
        ]);
        assert_eq!(
            detect_format_from_bytes(&blob).unwrap(),
            SafetensorsFormat::Unknown
        );
    }

    #[test]
    fn name_that_contains_qweight_in_the_middle_does_not_count() {
        // The suffix-only match rule means a hypothetical tensor
        // named `model.qweight_metadata` would NOT count as a packed
        // quant tensor. Pin the boundary so a future tensor-name
        // collision can't accidentally upgrade an unquantized
        // checkpoint to AwqInt4.
        let blob = make_safetensors(&[
            "model.embed_tokens.weight",
            "model.qweight_metadata", // contains substring, not suffix
        ]);
        assert_eq!(
            detect_format_from_bytes(&blob).unwrap(),
            SafetensorsFormat::Unquantized
        );
    }

    #[test]
    fn ends_with_segment_matches_exact_and_dotted_suffix() {
        // Standalone unit on the helper. Equal → match. `.suffix`
        // boundary → match. No boundary → no match.
        assert!(ends_with_segment("qweight", "qweight"));
        assert!(ends_with_segment("a.b.qweight", "qweight"));
        assert!(!ends_with_segment("aqweight", "qweight"));
        assert!(!ends_with_segment("qweighta", "qweight"));
        assert!(!ends_with_segment("qweight.x", "qweight"));
    }

    #[test]
    fn detect_format_on_real_file_round_trip() {
        // Write a real file to a temp path, run the public
        // disk-based entry point, verify it agrees with the slice
        // version. Pins that mmap + parse don't drop on the floor.
        let blob = make_safetensors(&[
            "model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.qzeros",
        ]);
        let path = std::env::temp_dir().join("rustllama-st-detect-awq.safetensors");
        std::fs::write(&path, &blob).unwrap();
        assert_eq!(detect_format(&path).unwrap(), SafetensorsFormat::AwqInt4);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn detect_format_errors_on_missing_file() {
        let bogus = std::env::temp_dir().join("rustllama-st-does-not-exist.safetensors");
        let _ = std::fs::remove_file(&bogus);
        match detect_format(&bogus) {
            Err(SafetensorsError::NotFound { .. }) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn detect_format_errors_on_empty_file() {
        let p = std::env::temp_dir().join("rustllama-st-empty.safetensors");
        std::fs::write(&p, []).unwrap();
        match detect_format(&p) {
            Err(SafetensorsError::Empty { .. }) => {}
            other => panic!("expected Empty, got {other:?}"),
        }
        let _ = std::fs::remove_file(&p);
    }
}
