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
//! # Non-affine microscaling modes (`mxfp4` / `mxfp8` / `nvfp4`)
//!
//! Real `mlx-lm` also ships **non-affine** microscaling checkpoints. These
//! drop the `.biases` sidecar entirely: `mx.quantize` returns only
//! `(weight, scales)` for them (three tensors for affine, two for these).
//! Confirmed on-disk layout (MLX `mlx.core.quantize` docs + a real
//! `mlx-community` `mxfp4` `config.json` + the `ml-explore/mlx-lm` FP8
//! issues):
//!
//! | mode    | group | bits | element | block scale (uint8) | biases |
//! |---------|-------|------|---------|---------------------|--------|
//! | `mxfp4` | 32    | 4    | E2M1    | E8M0                | none   |
//! | `mxfp8` | 32    | 8    | E4M3    | E8M0                | none   |
//! | `nvfp4` | 16    | 4    | E2M1    | E4M3                | none   |
//!
//! The packed `.weight` is still a uint32 bitstream (elements packed
//! low→high bits within each 32-bit word — byte-identical ordering to our
//! own MXFP/NVFP block codes); the `.scales` tensor is `uint8`, one raw
//! E8M0/E4M3 byte per group. These are loaded into [`MlxMicroQuant`] and —
//! because our GGUF-style block layout for these formats is just
//! `[group codes][1 scale byte]` per block — repacked to that layout and
//! handed to rustllama's **existing, parity-checked Wave-2
//! `dequant_mxfp4/mxfp8/nvfp4`** (see [`MlxMicroQuant::to_gguf_blocks`]).
//! The engine then dequant→transcodes them to a standard GGUF block-quant
//! exactly like the affine path, so they run on every backend. A
//! genuinely unknown `mode` string is still rejected with
//! [`MlxError::UnsupportedMode`].
//!
//! # Out of scope (this phase)
//!
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

/// A single **non-affine microscaling** MLX weight (`mxfp4` / `mxfp8` /
/// `nvfp4`). Unlike [`MlxAffineQuant`] there is no per-group bias — the
/// value is `codebook[code] * block_scale`, with the block scale a raw
/// E8M0 (mxfp*) or E4M3 (nvfp4) byte.
///
/// `shape` is the logical, dequantized `[out_features, in_features]`
/// (row-major, groups along the input dim), matching the affine struct.
#[derive(Debug, Clone)]
pub struct MlxMicroQuant {
    /// Packed `bits`-bit element codes — the raw little-endian bytes of the
    /// uint32 `.weight` tensor. MLX packs elements low→high within each
    /// 32-bit word, which (for 4-bit) puts element `2j` in byte `j`'s low
    /// nibble and `2j+1` in its high nibble, and (for 8-bit) element `j` in
    /// byte `j` — **byte-identical** to the code ordering our
    /// `dequant_mxfp4/mxfp8/nvfp4` references expect, so no bit-shuffle is
    /// needed on repack. `Arc` to share a tied embedding/`lm_head` cheaply.
    pub packed: Arc<[u8]>,
    /// Per-group block scales as the **raw uint8 bytes** from the `.scales`
    /// tensor (E8M0 for mxfp4/mxfp8, E4M3 for nvfp4 — decoded by the Wave-2
    /// dequant, not here). Length `out_features * (in_features/group_size)`,
    /// row-major to match `shape`.
    pub scales: Vec<u8>,
    /// The microscaling mode — [`MlxQuantMode::Mxfp4`] / `Mxfp8` / `Nvfp4`.
    pub mode: MlxQuantMode,
    /// Elements per block (32 for mxfp4/mxfp8, 16 for nvfp4).
    pub group_size: usize,
    /// Bits per element (4 for mxfp4/nvfp4, 8 for mxfp8).
    pub bits: u32,
    /// Logical dequantized shape `[out_features, in_features]`.
    pub shape: Vec<u64>,
    /// The `.weight` tensor name, for diagnostics.
    pub name: String,
}

impl MlxMicroQuant {
    /// `out_features` (rows) — first logical dim.
    pub fn out_features(&self) -> u64 {
        self.shape[0]
    }
    /// `in_features` (cols, the quantized/contraction dim) — second dim.
    pub fn in_features(&self) -> u64 {
        self.shape[1]
    }
    /// Total logical elements (`out_features * in_features`).
    pub fn n_elements(&self) -> u64 {
        self.shape[0] * self.shape[1]
    }

    /// The `(bits, group_size)` a microscaling mode mandates on disk (fixed
    /// by the OCP/NVIDIA specs and MLX's implementation). `None` for any
    /// non-micro mode.
    pub(crate) fn mode_geometry(mode: &MlxQuantMode) -> Option<(u32, usize)> {
        match mode {
            MlxQuantMode::Mxfp4 => Some((4, 32)),
            MlxQuantMode::Mxfp8 => Some((8, 32)),
            MlxQuantMode::Nvfp4 => Some((4, 16)),
            _ => None,
        }
    }

    /// Cross-check the packed/scale buffer lengths against the geometry so a
    /// malformed file fails loudly instead of decoding garbage. Also
    /// enforces that `group_size`/`bits` match what the mode requires (so a
    /// `config.json` claiming e.g. `mxfp4` at group 64 — which our fixed-32
    /// decoder can't honor — is rejected, not silently mis-decoded).
    pub fn validate(&self) -> Result<(), MlxError> {
        if self.shape.len() != 2 {
            return Err(MlxError::MicroNotMatrix {
                name: self.name.clone(),
                shape: self.shape.clone(),
            });
        }
        let (exp_bits, exp_group) = Self::mode_geometry(&self.mode).ok_or_else(|| {
            MlxError::UnsupportedMode {
                module: self.name.clone(),
                mode: self.mode.clone(),
            }
        })?;
        if self.bits != exp_bits || self.group_size != exp_group {
            return Err(MlxError::MicroGeometry {
                name: self.name.clone(),
                mode: self.mode.clone(),
                group_size: self.group_size,
                bits: self.bits,
                expected_group: exp_group,
                expected_bits: exp_bits,
            });
        }
        let in_f = self.in_features();
        if in_f % self.group_size as u64 != 0 {
            return Err(MlxError::MicroGroupMisaligned {
                name: self.name.clone(),
                in_features: in_f,
                group_size: self.group_size,
            });
        }
        let n = self.n_elements();
        let expected_packed = (n * self.bits as u64 / 8) as usize;
        if self.packed.len() != expected_packed {
            return Err(MlxError::MicroPackedLen {
                name: self.name.clone(),
                got: self.packed.len(),
                expected: expected_packed,
                n_elements: n,
                bits: self.bits,
            });
        }
        let n_groups = (n / self.group_size as u64) as usize;
        if self.scales.len() != n_groups {
            return Err(MlxError::MicroScaleCount {
                name: self.name.clone(),
                got: self.scales.len(),
                expected: n_groups,
            });
        }
        Ok(())
    }

    /// Repack MLX's `(packed uint32 codes, uint8 block scales)` into
    /// rustllama's **GGUF-style block layout** so the existing Wave-2
    /// `dequant_mxfp4/mxfp8/nvfp4` (and the whole quant-kernel stack) decode
    /// it unchanged.
    ///
    /// Our block layout is `[group_code_bytes][1 scale byte]` per group, and
    /// MLX's packed stream already stores each group's codes contiguously in
    /// exactly our byte/nibble order (see [`MlxMicroQuant::packed`]). So the
    /// repack is a pure per-group splice — copy the group's code bytes, then
    /// append its one scale byte. No bit shuffling, no transpose.
    ///
    /// - mxfp4: 16 code bytes + 1 E8M0 byte → 17-byte block (32 elems)
    /// - mxfp8: 32 code bytes + 1 E8M0 byte → 33-byte block (32 elems)
    /// - nvfp4:  8 code bytes + 1 E4M3 byte →  9-byte block (16 elems)
    ///
    /// `validate()` must have passed (lengths consistent); callers in this
    /// crate run it at load time.
    pub fn to_gguf_blocks(&self) -> Vec<u8> {
        let group_code_bytes = self.group_size * self.bits as usize / 8;
        let n_blocks = self.scales.len();
        let mut out = Vec::with_capacity(n_blocks * (group_code_bytes + 1));
        for bi in 0..n_blocks {
            let start = bi * group_code_bytes;
            out.extend_from_slice(&self.packed[start..start + group_code_bytes]);
            out.push(self.scales[bi]);
        }
        out
    }
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
    /// Affine-quantized weights (packed codes + per-group scale AND bias),
    /// keyed by module path.
    pub quant: BTreeMap<String, MlxAffineQuant>,
    /// Non-affine microscaling weights (`mxfp4`/`mxfp8`/`nvfp4`: packed
    /// codes + per-group uint8 scale, no bias), keyed by module path.
    pub micro: BTreeMap<String, MlxMicroQuant>,
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
        "mlx layer `{module}` uses unrecognized quantization mode {mode:?}; \
         supported: affine, mxfp4, mxfp8, nvfp4"
    )]
    UnsupportedMode { module: String, mode: MlxQuantMode },
    #[error(
        "mlx layer `{module}` is config-moded `affine` but ships no `.biases` \
         tensor (affine quant needs a per-group scale AND bias)"
    )]
    AffineMissingBiases { module: String },
    #[error(
        "mlx micro `{name}`: mode {mode:?} mandates group_size {expected_group} \
         / bits {expected_bits}, but config/file give group_size {group_size} \
         / bits {bits} (the microscaling decoders are fixed to the spec geometry)"
    )]
    MicroGeometry {
        name: String,
        mode: MlxQuantMode,
        group_size: usize,
        bits: u32,
        expected_group: usize,
        expected_bits: u32,
    },
    #[error(
        "mlx micro `{name}`: shape {shape:?} is not 2-D [out_features, in_features]"
    )]
    MicroNotMatrix { name: String, shape: Vec<u64> },
    #[error(
        "mlx micro `{name}`: in_features {in_features} not a multiple of \
         group_size {group_size}"
    )]
    MicroGroupMisaligned {
        name: String,
        in_features: u64,
        group_size: usize,
    },
    #[error(
        "mlx micro `{name}`: packed bytes {got} != expected {expected} \
         ({n_elements} elems × {bits} bits / 8)"
    )]
    MicroPackedLen {
        name: String,
        got: usize,
        expected: usize,
        n_elements: u64,
        bits: u32,
    },
    #[error(
        "mlx micro `{name}`: scales count {got} != expected {expected} groups"
    )]
    MicroScaleCount {
        name: String,
        got: usize,
        expected: usize,
    },
    #[error(
        "mlx micro weight `{name}` has scales dtype {dtype:?}; non-affine MLX \
         stores E8M0/E4M3 block scales as uint8"
    )]
    MicroScalesNotU8 { name: String, dtype: StDtype },
    #[error("mlx: model directory has no `*.safetensors` shards to load")]
    NoShards,
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
/// `quantization` block **and** the safetensors carries a packed uint32
/// `.weight` next to a per-group `.scales` sidecar. Cheap detection
/// primitive the engine's format sniffer can call before committing to the
/// MLX load path.
///
/// Detection keys on the **uint32 `.weight` + `.scales` pair**, NOT on
/// `.biases`: affine checkpoints add a `.biases`, but the non-affine
/// microscaling modes (`mxfp4`/`mxfp8`/`nvfp4`) have none — requiring
/// biases would reject every non-affine model. The uint32 `.weight` is
/// also what separates MLX from AWQ/GPTQ (whose packed tensor is
/// `.qweight`, with a plain fp16 `.weight` at most), so this stays a clean
/// MLX-vs-AWQ discriminator. Only the safetensors HEADER is parsed.
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
    let names: BTreeSet<String> = st.names().into_iter().cloned().collect();
    for name in &names {
        let Some(prefix) = name.strip_suffix(SUFFIX_SCALES) else {
            continue;
        };
        let weight = format!("{prefix}{SUFFIX_WEIGHT}");
        if !names.contains(&weight) {
            continue;
        }
        // The packed weight must be uint32 (MLX's code bitstream); a plain
        // fp16 `.weight` next to `.scales` is not an MLX quant module.
        if let Ok(w) = st.tensor(&weight) {
            if w.dtype() == StDtype::U32 {
                return true;
            }
        }
    }
    false
}

/// Directory-level, **shard-aware** MLX detection. True when `dir` holds a
/// `config.json` with a `quantization` block AND — across ALL of its
/// `*.safetensors` shards merged — a packed uint32 `.weight` sits next to a
/// `.scales` sibling.
///
/// This exists because [`is_mlx_model`] inspects a single blob, which is
/// wrong for a **sharded** checkpoint: HuggingFace shards split by byte
/// size, so a module's uint32 `.weight` and its `.scales` can land in
/// DIFFERENT shards. Probing one shard would miss the pair and mis-route
/// the model away from the MLX loader. Here we merge every shard's tensor
/// headers (header-only, via mmap — no weight data is read) before looking
/// for the pair, mirroring [`load_mlx_dir`]'s merge-then-match. Any IO /
/// parse error is swallowed as `false`.
pub fn is_mlx_dir(dir: &Path) -> bool {
    let Ok(config_json) = std::fs::read_to_string(dir.join("config.json")) else {
        return false;
    };
    if !matches!(
        MlxQuantConfig::parse_from_config_json(&config_json),
        Ok(Some(_))
    ) {
        return false;
    }
    let Ok(shard_paths) = resolve_shard_paths(dir) else {
        return false;
    };
    // mmap every shard + parse its header; hold them alive for the views.
    let mut mmaps = Vec::with_capacity(shard_paths.len());
    for p in &shard_paths {
        let Ok(m) = crate::open_safetensors(p) else {
            return false;
        };
        mmaps.push(m);
    }
    // Merge name → dtype across all shards (header metadata only).
    let mut name_dtype: BTreeMap<String, StDtype> = BTreeMap::new();
    for m in &mmaps {
        let Ok(st) = SafeTensors::deserialize(&m[..]) else {
            return false;
        };
        for (name, view) in st.tensors() {
            name_dtype.insert(name, view.dtype());
        }
    }
    for name in name_dtype.keys() {
        if let Some(prefix) = name.strip_suffix(SUFFIX_SCALES) {
            let weight = format!("{prefix}{SUFFIX_WEIGHT}");
            if name_dtype.get(&weight) == Some(&StDtype::U32) {
                return true;
            }
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

/// Split a merged `name → TensorView` map (one shard's worth, or every
/// shard merged together) into affine-quantized weights, non-affine
/// microscaling weights, and full-precision tensors.
///
/// **Affine vs micro routing is by the ground-truth presence of a
/// `.biases` sibling**, not by the config `mode` string: MLX emits three
/// tensors (`weight`+`scales`+`biases`) for affine and two
/// (`weight`+`scales`) for the microscaling modes, so biases-present ⇔
/// affine. A model may MIX both (e.g. attention/embeddings affine, MLP
/// mxfp4) — per-layer routing handles that. For a micro layer the specific
/// format (mxfp4/mxfp8/nvfp4) comes from the resolved config mode; a layer
/// the config marks `affine` but that ships no `.biases` is a malformed
/// file and errors.
fn collect_mlx_tensors(
    config: &MlxQuantConfig,
    tensors: &BTreeMap<String, TensorView<'_>>,
    quant: &mut BTreeMap<String, MlxAffineQuant>,
    micro: &mut BTreeMap<String, MlxMicroQuant>,
    full: &mut BTreeMap<String, MlxFullTensor>,
) -> Result<(), MlxError> {
    // Pass 1: find every quant module prefix `P` with both `P.weight` and
    // `P.scales` present (a `.biases` is optional — affine has it, micro
    // doesn't). Collect prefixes first so the mutable build loop isn't
    // tangled with the immutable key scan.
    let mut prefixes: Vec<String> = Vec::new();
    let mut consumed: BTreeSet<String> = BTreeSet::new();
    for name in tensors.keys() {
        let Some(prefix) = name.strip_suffix(SUFFIX_SCALES) else {
            continue;
        };
        let weight = format!("{prefix}{SUFFIX_WEIGHT}");
        if tensors.contains_key(&weight) {
            prefixes.push(prefix.to_string());
        }
    }

    // Pass 2: build an affine or micro weight per prefix.
    for prefix in prefixes {
        let weight_name = format!("{prefix}{SUFFIX_WEIGHT}");
        let scales_name = format!("{prefix}{SUFFIX_SCALES}");
        let biases_name = format!("{prefix}{SUFFIX_BIASES}");
        let weight = tensors.get(&weight_name).expect("checked in pass 1");
        let scales = tensors.get(&scales_name).expect("checked in pass 1");
        let has_biases = tensors.contains_key(&biases_name);

        consumed.insert(weight_name.clone());
        consumed.insert(scales_name.clone());
        if has_biases {
            consumed.insert(biases_name.clone());
        }

        let layer = config.for_layer(&prefix).ok_or(MlxError::SkippedButPacked {
            module: prefix.clone(),
        })?;

        // Packed weight must be uint32, 2-D [out_features, in_words] — same
        // for affine and micro (both pack codes into a uint32 bitstream).
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

        if has_biases {
            // ---- Affine: packed codes + per-group f16 scale AND bias. ----
            let biases = tensors.get(&biases_name).expect("has_biases checked");
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

            let scales_f32 = floats_to_f32(&scales_name, scales)?;
            let biases_f32 = floats_to_f32(&biases_name, biases)?;

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
            q.validate()?;
            quant.insert(prefix, q);
        } else {
            // ---- Non-affine: route by the resolved config mode. ----
            match layer.mode {
                MlxQuantMode::Affine => {
                    return Err(MlxError::AffineMissingBiases {
                        module: prefix.clone(),
                    });
                }
                MlxQuantMode::Mxfp4 | MlxQuantMode::Mxfp8 | MlxQuantMode::Nvfp4 => {
                    // Bits are fixed by the mode (not by `layer.bits`, which
                    // for microscaling is the nominal 4/8 but we re-derive to
                    // be safe); validate() cross-checks group_size.
                    let (exp_bits, _exp_group) = MlxMicroQuant::mode_geometry(&layer.mode)
                        .expect("micro mode has geometry");
                    let row_bits = row_words * 32;
                    if row_bits % exp_bits as u64 != 0 {
                        return Err(MlxError::RowWidthIndivisible {
                            name: weight_name.clone(),
                            row_words,
                            bits: exp_bits,
                        });
                    }
                    let in_features = row_bits / exp_bits as u64;

                    // Block scales are raw uint8 (E8M0 / E4M3) — not floats.
                    if scales.dtype() != StDtype::U8 {
                        return Err(MlxError::MicroScalesNotU8 {
                            name: scales_name.clone(),
                            dtype: scales.dtype(),
                        });
                    }

                    let m = MlxMicroQuant {
                        packed: Arc::from(weight.data().to_vec()),
                        scales: scales.data().to_vec(),
                        mode: layer.mode.clone(),
                        group_size: layer.group_size,
                        bits: exp_bits,
                        shape: vec![out_features, in_features],
                        name: weight_name.clone(),
                    };
                    m.validate()?;
                    micro.insert(prefix, m);
                }
                MlxQuantMode::Other(_) => {
                    return Err(MlxError::UnsupportedMode {
                        module: prefix.clone(),
                        mode: layer.mode.clone(),
                    });
                }
            }
        }
    }

    // Pass 3: everything not part of a quant/micro group is a full tensor.
    for (name, view) in tensors {
        if consumed.contains(name) {
            continue;
        }
        let dtype = full_dtype(name, view.dtype())?;
        let shape: Vec<u64> = view.shape().iter().map(|&d| d as u64).collect();
        full.insert(
            name.clone(),
            MlxFullTensor {
                name: name.clone(),
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
    let st = SafeTensors::deserialize(safetensors_bytes)?;
    let tensors: BTreeMap<String, TensorView<'_>> = st.tensors().into_iter().collect();
    let mut quant = BTreeMap::new();
    let mut micro = BTreeMap::new();
    let mut full = BTreeMap::new();
    collect_mlx_tensors(&config, &tensors, &mut quant, &mut micro, &mut full)?;
    Ok(MlxModel {
        config,
        quant,
        micro,
        full,
    })
}

/// Resolve the ordered list of `*.safetensors` shard files to load from a
/// model directory.
///
/// Prefers `model.safetensors.index.json` (the canonical shard manifest):
/// its `weight_map` names the shard file each tensor lives in, so we load
/// exactly the referenced shards (robust to stray `.safetensors` in the
/// dir). Falls back to globbing `*.safetensors` when there's no index or
/// its `weight_map` is unusable — a single-file `model.safetensors` model
/// has no index and is found by the glob.
fn resolve_shard_paths(dir: &Path) -> Result<Vec<std::path::PathBuf>, MlxError> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.is_file() {
        let s = std::fs::read_to_string(&index_path)?;
        let v: serde_json::Value = serde_json::from_str(&s)?;
        if let Some(wm) = v.get("weight_map").and_then(|m| m.as_object()) {
            // weight_map: tensor_name → shard_filename. Dedup to the set of
            // referenced shards; BTreeSet gives a sorted, deterministic order.
            let mut files: BTreeSet<String> = BTreeSet::new();
            for val in wm.values() {
                if let Some(f) = val.as_str() {
                    files.insert(f.to_string());
                }
            }
            if !files.is_empty() {
                return Ok(files.into_iter().map(|f| dir.join(f)).collect());
            }
        }
        // Index present but no usable weight_map → fall through to globbing.
    }

    // No (usable) index: load every `*.safetensors` shard, sorted so the
    // merge order is deterministic.
    let mut shards: Vec<std::path::PathBuf> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            shards.push(path);
        }
    }
    shards.sort();
    if shards.is_empty() {
        return Err(MlxError::NoShards);
    }
    Ok(shards)
}

/// Load an MLX checkpoint from a model directory: read `config.json`,
/// resolve its `*.safetensors` shards (via `model.safetensors.index.json`
/// when present, else by globbing), and **merge every shard's tensors into
/// one map before matching quant groups**. The tokenizer (`tokenizer.json`)
/// is intentionally left to the caller / `rustllama-tokenizer`.
///
/// Merging *before* matching is what makes sharding correct: HuggingFace
/// shards split purely by byte size, so a module's `.weight`, `.scales` and
/// `.biases` can land in **different** shards. Matching per-shard would miss
/// any group straddling a shard boundary; merging first sees the whole
/// tensor set at once.
pub fn load_mlx_dir(dir: &Path) -> Result<MlxModel, MlxError> {
    let config_json = std::fs::read_to_string(dir.join("config.json"))?;
    let config = MlxQuantConfig::parse_from_config_json(&config_json)?
        .ok_or(MlxError::NotMlx)?;

    let shard_paths = resolve_shard_paths(dir)?;

    // mmap + deserialize every shard, and hold the mmaps AND the parsed
    // `SafeTensors` alive for the whole merge — the `TensorView`s borrow
    // from them.
    let mut mmaps = Vec::with_capacity(shard_paths.len());
    for p in &shard_paths {
        mmaps.push(crate::open_safetensors(p)?);
    }
    let mut sts = Vec::with_capacity(mmaps.len());
    for m in &mmaps {
        sts.push(SafeTensors::deserialize(&m[..])?);
    }

    // Merge all shards' tensors into one `name → view` map, then match.
    let mut tensors: BTreeMap<String, TensorView<'_>> = BTreeMap::new();
    for st in &sts {
        for (name, view) in st.tensors() {
            tensors.insert(name, view);
        }
    }

    let mut quant = BTreeMap::new();
    let mut micro = BTreeMap::new();
    let mut full = BTreeMap::new();
    collect_mlx_tensors(&config, &tensors, &mut quant, &mut micro, &mut full)?;
    Ok(MlxModel {
        config,
        quant,
        micro,
        full,
    })
}

// =====================================================================
// Writer (produce path) — the inverse of the loader above.
//
// Serializes a set of MLX affine-quantized + full-precision tensors to
// an `mlx-lm`-shaped model directory: `model.safetensors` (packed
// weight as uint32, scales/biases as f16), a `config.json` carrying the
// `quantization` block, and (optionally) a copied `tokenizer.json`.
// Round-trips back through [`load_mlx_dir`] / [`is_mlx_model`].
// =====================================================================

/// One tensor to write into an MLX model directory. A [`Quant`] weight
/// emits the three sibling tensors (`<name>.weight` uint32,
/// `<name>.scales`/`<name>.biases` f16); a [`Full`] tensor is written
/// verbatim at its dtype (norms, biases, un-quantized embeddings).
///
/// [`Quant`]: MlxWriteTensor::Quant
/// [`Full`]: MlxWriteTensor::Full
#[derive(Debug, Clone)]
pub enum MlxWriteTensor {
    /// Affine-quantized 2-D linear/embedding weight. `packed` is the
    /// LSB-first bitstream from
    /// `rustllama_kernels_cpu::mlx_affine::quantize_mlx_affine`
    /// (already padded to a whole number of uint32 words);
    /// `scales`/`biases` are per-group f32 (rounded to f16 on disk, the
    /// mlx-lm default — see the writer note). `name` is the module path
    /// (the `.weight`-stripped prefix the loader keys on); `shape` is
    /// the logical `[out_features, in_features]`.
    Quant {
        name: String,
        packed: Vec<u8>,
        scales: Vec<f32>,
        biases: Vec<f32>,
        group_size: usize,
        bits: u32,
        shape: Vec<u64>,
    },
    /// A full-precision tensor carried through verbatim at `dtype`.
    /// `bytes` are the raw little-endian payload matching `dtype`.
    Full {
        name: String,
        dtype: MlxFullDtype,
        shape: Vec<u64>,
        bytes: Vec<u8>,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum MlxWriteError {
    #[error("mlx write io: {0}")]
    Io(#[from] std::io::Error),
    #[error("mlx write serialize: {0}")]
    Serialize(#[from] safetensors::SafeTensorError),
    #[error("mlx write config.json: {0}")]
    Json(#[from] serde_json::Error),
    #[error(
        "mlx write `{name}`: quant shape {shape:?} is not 2-D \
         [out_features, in_features]"
    )]
    QuantNotMatrix { name: String, shape: Vec<u64> },
    #[error(
        "mlx write `{name}`: in_features {in_features} not a multiple of \
         group_size {group_size}"
    )]
    GroupMisaligned {
        name: String,
        in_features: u64,
        group_size: usize,
    },
    #[error(
        "mlx write `{name}`: in_features {in_features} × bits {bits} = \
         {row_bits} is not a multiple of 32 — cannot pack into uint32 words"
    )]
    RowNotWordAligned {
        name: String,
        in_features: u64,
        bits: u32,
        row_bits: u64,
    },
    #[error(
        "mlx write `{name}`: packed {got} bytes != expected {expected} \
         for [{out_features}, {in_features}] at {bits} bits"
    )]
    PackedLen {
        name: String,
        got: usize,
        expected: usize,
        out_features: u64,
        in_features: u64,
        bits: u32,
    },
    #[error(
        "mlx write `{name}`: expected {expected} scale/bias groups, got \
         scales {scales} / biases {biases}"
    )]
    GroupCount {
        name: String,
        expected: usize,
        scales: usize,
        biases: usize,
    },
}

/// MLX's `mode="affine"` value string for the `quantization.mode` key.
const MODE_AFFINE: &str = "affine";

/// f32 → f16 little-endian bytes. mlx-lm stores `scales`/`biases` in the
/// model's compute dtype (overwhelmingly f16/bf16); we pick **f16** to
/// match that default and to halve the sidecar size vs f32. The loader
/// decodes f16 → f32 on read, so the only cost is the f16 rounding of
/// the per-group scale/bias (well inside the affine quant error).
fn scales_to_f16_le(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
        .collect()
}

fn full_st_dtype(d: MlxFullDtype) -> StDtype {
    match d {
        MlxFullDtype::F32 => StDtype::F32,
        MlxFullDtype::F16 => StDtype::F16,
        MlxFullDtype::Bf16 => StDtype::BF16,
    }
}

/// An owned tensor payload the serializer borrows from. We stage every
/// tensor's bytes in a `Vec` first (held for the whole serialize call),
/// then build the borrowing `TensorView`s in a second pass — the same
/// lifetime dance `crate::convert` uses.
struct OwnedTensor {
    name: String,
    dtype: StDtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

/// Serialize `tensors` into an MLX model directory at `dir`.
///
/// Writes `model.safetensors` (single shard — sharding is out of scope,
/// see the module TODO), a `config.json` whose `quantization` block
/// carries `default_group_size`/`default_bits` plus a per-layer override
/// for any [`MlxWriteTensor::Quant`] whose geometry differs from those
/// defaults, and — when `tokenizer_src` names an existing file — a copy
/// of it as `tokenizer.json`. `extra_config` supplies the non-quant
/// `config.json` fields (architecture, dims, …); its keys are written
/// verbatim, and a caller-supplied `quantization` key is overwritten.
///
/// The result round-trips: [`is_mlx_model`] returns true for the written
/// directory and [`load_mlx_dir`] reconstructs each weight.
pub fn write_mlx_dir(
    dir: &Path,
    tensors: &[MlxWriteTensor],
    default_group_size: usize,
    default_bits: u32,
    extra_config: serde_json::Map<String, serde_json::Value>,
    tokenizer_src: Option<&Path>,
) -> Result<(), MlxWriteError> {
    std::fs::create_dir_all(dir)?;

    let mut owned: Vec<OwnedTensor> = Vec::with_capacity(tensors.len());
    // Per-layer quant overrides for config.json — only emitted for a
    // Quant tensor whose (group_size, bits) differs from the defaults.
    let mut overrides: serde_json::Map<String, serde_json::Value> =
        serde_json::Map::new();

    for t in tensors {
        match t {
            MlxWriteTensor::Full {
                name,
                dtype,
                shape,
                bytes,
            } => {
                owned.push(OwnedTensor {
                    name: name.clone(),
                    dtype: full_st_dtype(*dtype),
                    shape: shape.iter().map(|&d| d as usize).collect(),
                    bytes: bytes.clone(),
                });
            }
            MlxWriteTensor::Quant {
                name,
                packed,
                scales,
                biases,
                group_size,
                bits,
                shape,
            } => {
                if shape.len() != 2 {
                    return Err(MlxWriteError::QuantNotMatrix {
                        name: name.clone(),
                        shape: shape.clone(),
                    });
                }
                let out_features = shape[0];
                let in_features = shape[1];
                if in_features % *group_size as u64 != 0 {
                    return Err(MlxWriteError::GroupMisaligned {
                        name: name.clone(),
                        in_features,
                        group_size: *group_size,
                    });
                }
                // MLX packs each row into whole uint32 words; that needs
                // `in_features * bits` to be a multiple of 32 (always
                // true when group_size — a multiple of 32 — divides
                // in_features, but we check rather than assume).
                let row_bits = in_features * *bits as u64;
                if row_bits % 32 != 0 {
                    return Err(MlxWriteError::RowNotWordAligned {
                        name: name.clone(),
                        in_features,
                        bits: *bits,
                        row_bits,
                    });
                }
                let row_words = (row_bits / 32) as usize;
                let expected_bytes =
                    (out_features as usize) * row_words * 4;
                if packed.len() != expected_bytes {
                    return Err(MlxWriteError::PackedLen {
                        name: name.clone(),
                        got: packed.len(),
                        expected: expected_bytes,
                        out_features,
                        in_features,
                        bits: *bits,
                    });
                }
                let n_groups =
                    (out_features * (in_features / *group_size as u64)) as usize;
                if scales.len() != n_groups || biases.len() != n_groups {
                    return Err(MlxWriteError::GroupCount {
                        name: name.clone(),
                        expected: n_groups,
                        scales: scales.len(),
                        biases: biases.len(),
                    });
                }

                let groups_per_row = (in_features / *group_size as u64) as usize;
                // `<name>.weight` — packed uint32; the byte view already
                // IS the uint32-LE array (see the format docs).
                owned.push(OwnedTensor {
                    name: format!("{name}{SUFFIX_WEIGHT}"),
                    dtype: StDtype::U32,
                    shape: vec![out_features as usize, row_words],
                    bytes: packed.clone(),
                });
                // `<name>.scales` / `<name>.biases` — per-group f16,
                // row-major `[out_features, in_features/group_size]`.
                owned.push(OwnedTensor {
                    name: format!("{name}{SUFFIX_SCALES}"),
                    dtype: StDtype::F16,
                    shape: vec![out_features as usize, groups_per_row],
                    bytes: scales_to_f16_le(scales),
                });
                owned.push(OwnedTensor {
                    name: format!("{name}{SUFFIX_BIASES}"),
                    dtype: StDtype::F16,
                    shape: vec![out_features as usize, groups_per_row],
                    bytes: scales_to_f16_le(biases),
                });

                // Record a per-layer override when this weight's quant
                // geometry diverges from the model-wide defaults.
                if *group_size != default_group_size || *bits != default_bits {
                    overrides.insert(
                        name.clone(),
                        serde_json::json!({
                            "group_size": *group_size,
                            "bits": *bits,
                            "mode": MODE_AFFINE,
                        }),
                    );
                }
            }
        }
    }

    // Build the borrowing views in a second pass (owned outlives `map`).
    let mut map: BTreeMap<String, safetensors::tensor::TensorView<'_>> =
        BTreeMap::new();
    for o in &owned {
        let view = safetensors::tensor::TensorView::new(
            o.dtype,
            o.shape.clone(),
            &o.bytes,
        )?;
        map.insert(o.name.clone(), view);
    }
    // Tag the file as MLX (mlx-lm writes a `format` metadata key). Not
    // load-bearing for our loader, but keeps the artifact honest.
    let metadata: std::collections::HashMap<String, String> =
        std::collections::HashMap::from([("format".to_string(), "mlx".to_string())]);
    let blob = safetensors::serialize(&map, &Some(metadata))?;
    std::fs::write(dir.join("model.safetensors"), &blob)?;

    // config.json: caller fields + the quantization block.
    let mut config = extra_config;
    let mut quant = serde_json::Map::new();
    quant.insert("group_size".into(), serde_json::json!(default_group_size));
    quant.insert("bits".into(), serde_json::json!(default_bits));
    quant.insert("mode".into(), serde_json::json!(MODE_AFFINE));
    for (k, v) in overrides {
        quant.insert(k, v);
    }
    config.insert("quantization".into(), serde_json::Value::Object(quant));
    let config_str = serde_json::to_string_pretty(&serde_json::Value::Object(config))?;
    std::fs::write(dir.join("config.json"), config_str)?;

    // Copy the tokenizer when the source has one (standard HF
    // `tokenizer.json`; the loader leaves tokenizer handling to
    // rustllama-tokenizer).
    if let Some(src) = tokenizer_src {
        if src.exists() {
            std::fs::copy(src, dir.join("tokenizer.json"))?;
        }
    }

    Ok(())
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
    fn rejects_unknown_mode() {
        // A non-affine pair (weight u32 + scales u8, no biases) whose
        // config mode is an unrecognized string → UnsupportedMode.
        let p = "model.layers.0.mlp.gate_proj";
        let blob = make_micro_blob(p, 4, 64, 32, 4, &vec![127u8; 4 * 2]).0;
        let cfg_json =
            r#"{"quantization": {"group_size": 32, "bits": 4, "mode": "frobnicate"}}"#;
        match load_mlx_from_bytes(cfg_json, &blob) {
            Err(MlxError::UnsupportedMode {
                mode: MlxQuantMode::Other(m),
                ..
            }) => assert_eq!(m, "frobnicate"),
            other => panic!("expected UnsupportedMode, got {other:?}"),
        }
    }

    #[test]
    fn affine_mode_without_biases_is_rejected() {
        // config says affine but the file ships no `.biases` → malformed.
        let p = "model.layers.0.mlp.up_proj";
        let blob = make_micro_blob(p, 4, 64, 32, 4, &vec![127u8; 4 * 2]).0;
        let cfg_json = r#"{"quantization": {"group_size": 32, "bits": 4, "mode": "affine"}}"#;
        match load_mlx_from_bytes(cfg_json, &blob) {
            Err(MlxError::AffineMissingBiases { .. }) => {}
            other => panic!("expected AffineMissingBiases, got {other:?}"),
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

    // --- non-affine microscaling (mxfp4 / mxfp8 / nvfp4) -----------------

    /// Build a one-linear **non-affine** MLX safetensors blob:
    ///   `<p>.weight` (u32 packed codes), `<p>.scales` (u8 block scales)
    /// plus a bare `model.norm.weight` full tensor. NO `.biases` — that's
    /// what marks it non-affine. `scale_bytes` is the raw E8M0/E4M3 byte
    /// per group (len = out_f * in_f/group). Returns (blob, codes).
    fn make_micro_blob(
        p: &str,
        out_f: usize,
        in_f: usize,
        group: usize,
        bits: u32,
        scale_bytes: &[u8],
    ) -> (Vec<u8>, Vec<u32>) {
        let n = out_f * in_f;
        let maxv = 1u32 << bits;
        let codes: Vec<u32> =
            (0..n).map(|i| (i as u32).wrapping_mul(2246822519) % maxv).collect();
        let row_words = in_f * bits as usize / 32;
        let packed = pack_u32_le(&codes, bits);
        assert_eq!(scale_bytes.len(), out_f * (in_f / group));

        let norm = f16_bytes(&vec![1.0; out_f]);

        let mut map: Map<String, TensorView<'_>> = Map::new();
        let wv = TensorView::new(StDtype::U32, vec![out_f, row_words], &packed).unwrap();
        let sv = TensorView::new(StDtype::U8, vec![out_f, in_f / group], scale_bytes).unwrap();
        let nv = TensorView::new(StDtype::F16, vec![out_f], &norm).unwrap();
        map.insert(format!("{p}.weight"), wv);
        map.insert(format!("{p}.scales"), sv);
        map.insert("model.norm.weight".to_string(), nv);
        let blob = safetensors::serialize(&map, &None).unwrap();
        (blob, codes)
    }

    // Local decoders mirroring the Wave-2 references, so the value checks
    // below don't depend on `rustllama-gguf`'s crate-private helpers.
    const E2M1: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    fn e8m0(b: u8) -> f32 {
        if b == 0xFF {
            f32::NAN
        } else {
            (2.0f32).powi(b as i32 - 127)
        }
    }
    fn e4m3(b: u8) -> f32 {
        let sign = (b & 0x80) != 0;
        let exp = (b >> 3) & 0x0F;
        let mant = b & 0x07;
        if exp == 0x0F && mant == 0x07 {
            return f32::NAN;
        }
        let v = if exp == 0 {
            (mant as f32) * (1.0 / 512.0)
        } else {
            (1.0 + (mant as f32) / 8.0) * (2.0f32).powi(exp as i32 - 7)
        };
        if sign {
            -v
        } else {
            v
        }
    }

    #[test]
    fn parses_non_affine_modes() {
        for (s, want) in [
            ("mxfp4", MlxQuantMode::Mxfp4),
            ("mxfp8", MlxQuantMode::Mxfp8),
            ("nvfp4", MlxQuantMode::Nvfp4),
        ] {
            let json = format!(
                r#"{{"quantization": {{"group_size": 32, "bits": 4, "mode": "{s}"}}}}"#
            );
            let cfg = MlxQuantConfig::parse_from_config_json(&json).unwrap().unwrap();
            assert_eq!(cfg.mode, want);
        }
    }

    #[test]
    fn loads_mxfp4_and_dequants_through_wave2() {
        // in_f=128, group 32, 4-bit E2M1, E8M0 scales. Use a spread of
        // scale exponents so a wrong element↔scale pairing would show.
        let p = "model.layers.0.self_attn.q_proj";
        let (out_f, in_f, group) = (4usize, 128usize, 32usize);
        let n_groups = out_f * (in_f / group);
        let scales: Vec<u8> = (0..n_groups).map(|g| (126 + g % 4) as u8).collect();
        let (blob, codes) = make_micro_blob(p, out_f, in_f, group, 4, &scales);
        let cfg_json = r#"{"quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4"}}"#;

        assert!(is_mlx_model(cfg_json, &blob), "mxfp4 (no biases) must detect as MLX");

        let model = load_mlx_from_bytes(cfg_json, &blob).unwrap();
        assert_eq!(model.quant.len(), 0, "no affine weights");
        assert_eq!(model.micro.len(), 1, "one micro weight");
        assert_eq!(model.full.len(), 1, "the norm");
        let m = model.micro.get(p).expect("micro present");
        assert_eq!(m.shape, vec![out_f as u64, in_f as u64]);
        assert_eq!(m.mode, MlxQuantMode::Mxfp4);
        assert_eq!((m.group_size, m.bits), (32, 4));
        m.validate().unwrap();

        // Repack → Wave-2 dequant, then check each cell is codebook*scale
        // with the CORRECT per-group scale (the repack's whole job).
        let blocks = m.to_gguf_blocks();
        let n = (out_f * in_f) as usize;
        let mut out = vec![0f32; n];
        rustllama_gguf::dequant::dequant_mxfp4(&blocks, &mut out);
        for i in 0..n {
            let g = i / group;
            let want = E2M1[codes[i] as usize] * e8m0(scales[g]);
            assert!((out[i] - want).abs() < 1e-6, "cell {i}: {} vs {want}", out[i]);
        }
    }

    #[test]
    fn loads_mxfp8_and_dequants_through_wave2() {
        // 8-bit E4M3 elements, E8M0 scales, group 32.
        let p = "model.layers.0.mlp.down_proj";
        let (out_f, in_f, group) = (2usize, 64usize, 32usize);
        let n_groups = out_f * (in_f / group);
        let scales: Vec<u8> = (0..n_groups).map(|g| (127 + g) as u8).collect();
        let (blob, codes) = make_micro_blob(p, out_f, in_f, group, 8, &scales);
        let cfg_json = r#"{"quantization": {"group_size": 32, "bits": 8, "mode": "mxfp8"}}"#;

        let model = load_mlx_from_bytes(cfg_json, &blob).unwrap();
        let m = model.micro.get(p).expect("micro present");
        assert_eq!((m.group_size, m.bits), (32, 8));
        assert_eq!(m.mode, MlxQuantMode::Mxfp8);

        let blocks = m.to_gguf_blocks();
        let n = (out_f * in_f) as usize;
        let mut out = vec![0f32; n];
        rustllama_gguf::dequant::dequant_mxfp8(&blocks, &mut out);
        for i in 0..n {
            let g = i / group;
            // 8-bit code byte = the element's E4M3 byte directly.
            let want = e4m3(codes[i] as u8) * e8m0(scales[g]);
            if want.is_nan() {
                assert!(out[i].is_nan(), "cell {i} expected NaN");
            } else {
                assert!((out[i] - want).abs() < 1e-5, "cell {i}: {} vs {want}", out[i]);
            }
        }
    }

    #[test]
    fn loads_nvfp4_and_dequants_through_wave2() {
        // 4-bit E2M1 elements, group 16, E4M3 block scales.
        let p = "model.layers.0.self_attn.v_proj";
        let (out_f, in_f, group) = (3usize, 32usize, 16usize);
        let n_groups = out_f * (in_f / group);
        // 0x38 = E4M3 1.0; 0x40 = 2.0 — a two-value scale spread.
        let scales: Vec<u8> = (0..n_groups)
            .map(|g| if g % 2 == 0 { 0x38 } else { 0x40 })
            .collect();
        let (blob, codes) = make_micro_blob(p, out_f, in_f, group, 4, &scales);
        let cfg_json = r#"{"quantization": {"group_size": 16, "bits": 4, "mode": "nvfp4"}}"#;

        let model = load_mlx_from_bytes(cfg_json, &blob).unwrap();
        let m = model.micro.get(p).expect("micro present");
        assert_eq!((m.group_size, m.bits), (16, 4));
        assert_eq!(m.mode, MlxQuantMode::Nvfp4);

        let blocks = m.to_gguf_blocks();
        let n = (out_f * in_f) as usize;
        let mut out = vec![0f32; n];
        rustllama_gguf::dequant::dequant_nvfp4(&blocks, &mut out);
        for i in 0..n {
            let g = i / group;
            let want = E2M1[codes[i] as usize] * e4m3(scales[g]);
            assert!((out[i] - want).abs() < 1e-6, "cell {i}: {} vs {want}", out[i]);
        }
    }

    #[test]
    fn micro_wrong_group_size_rejected() {
        // config claims mxfp4 at group 64, but mxfp4 is fixed to group 32.
        let p = "model.layers.0.mlp.gate_proj";
        let scales = vec![127u8; 2 * (128 / 64)];
        let (blob, _) = make_micro_blob(p, 2, 128, 64, 4, &scales);
        let cfg_json = r#"{"quantization": {"group_size": 64, "bits": 4, "mode": "mxfp4"}}"#;
        match load_mlx_from_bytes(cfg_json, &blob) {
            Err(MlxError::MicroGeometry { expected_group: 32, group_size: 64, .. }) => {}
            other => panic!("expected MicroGeometry, got {other:?}"),
        }
    }

    #[test]
    fn mixed_affine_and_micro_in_one_model() {
        // One affine layer (global affine) + one mxfp4 layer (per-layer
        // override). Proves per-layer routing by biases-presence + mode.
        let affine_p = "model.layers.0.self_attn.q_proj";
        let micro_p = "model.layers.0.mlp.gate_proj";

        // Affine triple via the affine helper (weight+scales+biases).
        let (aff_blob, _, _, _) = make_mlx_blob(affine_p, 4, 128, 32, 4);
        // Micro pair.
        let micro_scales = vec![127u8; 4 * (128 / 32)];
        let (mic_blob, _) = make_micro_blob(micro_p, 4, 128, 32, 4, &micro_scales);

        // Merge the two blobs' tensors into one safetensors file.
        let aff = SafeTensors::deserialize(&aff_blob).unwrap();
        let mic = SafeTensors::deserialize(&mic_blob).unwrap();
        let mut map: Map<String, TensorView<'_>> = Map::new();
        for (n, v) in aff.tensors() {
            map.insert(n, v);
        }
        for (n, v) in mic.tensors() {
            // skip the duplicate norm from the second blob
            if !map.contains_key(&n) {
                map.insert(n, v);
            }
        }
        let blob = safetensors::serialize(&map, &None).unwrap();

        let cfg_json = format!(
            r#"{{"quantization": {{"group_size": 32, "bits": 4, "mode": "affine",
                 "{micro_p}": {{"group_size": 32, "bits": 4, "mode": "mxfp4"}}}}}}"#
        );
        let model = load_mlx_from_bytes(&cfg_json, &blob).unwrap();
        assert_eq!(model.quant.len(), 1, "one affine");
        assert_eq!(model.micro.len(), 1, "one micro");
        assert!(model.quant.contains_key(affine_p));
        assert!(model.micro.contains_key(micro_p));
    }

    // --- multi-shard loading (index.json weight_map) ---------------------

    /// Write a 2-shard MLX model dir whose quant triple is deliberately
    /// SPLIT across shards (weight in shard 1; scales + biases in shard 2),
    /// plus a `model.safetensors.index.json` weight_map. Returns the dir.
    fn write_split_shard_dir(tag: &str, with_index: bool) -> std::path::PathBuf {
        let p = "model.layers.0.self_attn.q_proj";
        let (out_f, in_f, group, bits) = (4usize, 128usize, 32usize, 4u32);
        let n = out_f * in_f;
        let codes: Vec<u32> = (0..n).map(|i| (i as u32).wrapping_mul(2246822519) % 16).collect();
        let packed = pack_u32_le(&codes, bits);
        let row_words = in_f * bits as usize / 32;
        let n_groups = out_f * (in_f / group);
        let scales: Vec<f32> = (0..n_groups).map(|g| 0.5 + g as f32 * 0.0625).collect();
        let biases: Vec<f32> = (0..n_groups).map(|g| -0.25 + g as f32 * 0.03125).collect();
        let norm = f16_bytes(&vec![1.0; out_f]);
        let scales_b = f16_bytes(&scales);
        let biases_b = f16_bytes(&biases);

        let dir =
            std::env::temp_dir().join(format!("rustllama-mlx-shard-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Shard 1: the packed weight + the full norm.
        let mut m1: Map<String, TensorView<'_>> = Map::new();
        m1.insert(
            format!("{p}.weight"),
            TensorView::new(StDtype::U32, vec![out_f, row_words], &packed).unwrap(),
        );
        m1.insert(
            "model.norm.weight".into(),
            TensorView::new(StDtype::F16, vec![out_f], &norm).unwrap(),
        );
        let b1 = safetensors::serialize(&m1, &None).unwrap();
        std::fs::write(dir.join("model-00001-of-00002.safetensors"), &b1).unwrap();

        // Shard 2: the scales + biases (same module → cross-shard triple).
        let mut m2: Map<String, TensorView<'_>> = Map::new();
        m2.insert(
            format!("{p}.scales"),
            TensorView::new(StDtype::F16, vec![out_f, in_f / group], &scales_b).unwrap(),
        );
        m2.insert(
            format!("{p}.biases"),
            TensorView::new(StDtype::F16, vec![out_f, in_f / group], &biases_b).unwrap(),
        );
        let b2 = safetensors::serialize(&m2, &None).unwrap();
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), &b2).unwrap();

        if with_index {
            let index = serde_json::json!({
                "metadata": {"total_size": (b1.len() + b2.len())},
                "weight_map": {
                    format!("{p}.weight"): "model-00001-of-00002.safetensors",
                    "model.norm.weight": "model-00001-of-00002.safetensors",
                    format!("{p}.scales"): "model-00002-of-00002.safetensors",
                    format!("{p}.biases"): "model-00002-of-00002.safetensors",
                }
            });
            std::fs::write(
                dir.join("model.safetensors.index.json"),
                serde_json::to_string_pretty(&index).unwrap(),
            )
            .unwrap();
        }

        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["Qwen2ForCausalLM"],"quantization":{"group_size":32,"bits":4}}"#,
        )
        .unwrap();
        dir
    }

    #[test]
    fn multi_shard_merges_cross_shard_triple_via_index() {
        let dir = write_split_shard_dir("idx", true);
        let model = load_mlx_dir(&dir).unwrap();
        // The affine triple, split across two shards, must still match.
        assert_eq!(model.quant.len(), 1, "cross-shard triple merged");
        assert_eq!(model.full.len(), 1, "the norm");
        let q = model
            .quant
            .get("model.layers.0.self_attn.q_proj")
            .expect("quant weight merged across shards");
        assert_eq!(q.shape, vec![4, 128]);
        q.validate().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn multi_shard_merges_without_index_via_glob() {
        // Same split, but NO index.json → glob-every-shard fallback also
        // merges-before-matching, so the cross-shard triple still loads.
        let dir = write_split_shard_dir("glob", false);
        let model = load_mlx_dir(&dir).unwrap();
        assert_eq!(model.quant.len(), 1, "cross-shard triple merged (glob path)");
        assert!(model.quant.contains_key("model.layers.0.self_attn.q_proj"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn is_mlx_dir_detects_sharded_across_shards() {
        // The regression this guards: detection must look ACROSS shards.
        // Here the uint32 `.weight` is in shard 1 and `.scales` in shard 2,
        // so a single-shard probe (old behavior) would miss the pair and
        // mis-route the model away from the MLX loader.
        let dir = write_split_shard_dir("detect-idx", true);
        assert!(is_mlx_dir(&dir), "sharded MLX dir (index) must detect as MLX");
        let _ = std::fs::remove_dir_all(&dir);

        let dir2 = write_split_shard_dir("detect-glob", false);
        assert!(is_mlx_dir(&dir2), "sharded MLX dir (glob) must detect as MLX");
        let _ = std::fs::remove_dir_all(&dir2);
    }

    #[test]
    fn is_mlx_dir_false_without_quant_block() {
        // A dir with a safetensors shard but no `quantization` block in
        // config.json is not MLX.
        let dir = std::env::temp_dir()
            .join(format!("rustllama-mlx-notmlx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let norm = f16_bytes(&vec![1.0; 4]);
        let mut map: Map<String, TensorView<'_>> = Map::new();
        map.insert(
            "model.norm.weight".into(),
            TensorView::new(StDtype::F16, vec![4], &norm).unwrap(),
        );
        let blob = safetensors::serialize(&map, &None).unwrap();
        std::fs::write(dir.join("model.safetensors"), &blob).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["LlamaForCausalLM"],"hidden_size":4}"#,
        )
        .unwrap();
        assert!(!is_mlx_dir(&dir), "no quant block → not MLX");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- writer (produce path) -------------------------------------------

    /// Deterministic pseudo-random f32 in `[-2, 2)`, matching the
    /// kernels-cpu encode tests' generator so the round-trip covers a
    /// realistic weight spread without an `rand` dep.
    fn pseudo_f32(i: usize) -> f32 {
        let h = (i as u32).wrapping_mul(2654435761) ^ 0x9E37_79B9;
        (h % 10_007) as f32 / 10_007.0 * 4.0 - 2.0
    }

    #[test]
    fn write_mlx_dir_roundtrips_through_loader() {
        use rustllama_kernels_cpu::mlx_affine::{
            dequantize_mlx_affine, quantize_mlx_affine,
        };

        let out_f = 4usize;
        let in_f = 128usize;
        let group_size = 64usize;
        let bits = 4u32;

        // Synthetic [out_f, in_f] weight, row-major (in_f contiguous) —
        // exactly the MLX/GGUF nn.Linear layout the loader reconstructs.
        let weights: Vec<f32> = (0..out_f * in_f).map(pseudo_f32).collect();
        let (packed, scales, biases) =
            quantize_mlx_affine(&weights, group_size, bits);

        // A 1-D norm written as a full (un-quantized) f16 passthrough.
        let norm: Vec<f32> = (0..out_f).map(|i| 1.0 + i as f32 * 0.1).collect();
        let norm_bytes = f16_bytes(&norm);

        let tensors = vec![
            MlxWriteTensor::Quant {
                name: "model.layers.0.self_attn.q_proj".into(),
                packed,
                scales: scales.clone(),
                biases: biases.clone(),
                group_size,
                bits,
                shape: vec![out_f as u64, in_f as u64],
            },
            MlxWriteTensor::Full {
                name: "model.norm.weight".into(),
                dtype: MlxFullDtype::F16,
                shape: vec![out_f as u64],
                bytes: norm_bytes,
            },
        ];

        // Unique temp dir per process so parallel test runs don't clash.
        let dir = std::env::temp_dir()
            .join(format!("rustllama-mlx-write-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let mut extra = serde_json::Map::new();
        extra.insert("architectures".into(), serde_json::json!(["LlamaForCausalLM"]));
        extra.insert("hidden_size".into(), serde_json::json!(out_f));
        write_mlx_dir(&dir, &tensors, group_size, bits, extra, None).unwrap();

        // The written dir must classify as MLX.
        let cfg_json = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let st_bytes = std::fs::read(dir.join("model.safetensors")).unwrap();
        assert!(is_mlx_model(&cfg_json, &st_bytes), "written dir not detected as MLX");

        // Load it back and check geometry + reconstruction.
        let model = load_mlx_dir(&dir).unwrap();
        assert_eq!(model.config.group_size, group_size);
        assert_eq!(model.config.bits, bits);
        assert_eq!(model.quant.len(), 1);
        assert_eq!(model.full.len(), 1);

        let q = model
            .quant
            .get("model.layers.0.self_attn.q_proj")
            .expect("quant weight present");
        assert_eq!(q.shape, vec![out_f as u64, in_f as u64]);
        assert_eq!(q.group_size, group_size);
        assert_eq!(q.bits, bits);
        q.validate().unwrap();

        let mut recon = vec![0f32; q.n_elements() as usize];
        dequantize_mlx_affine(
            &q.packed, &q.scales, &q.biases, q.group_size, q.bits, &mut recon,
        );
        // Tolerance: affine quant error (scale/2) + slack for the f16
        // rounding of the per-group scale/bias.
        for i in 0..weights.len() {
            let g = i / group_size;
            let tol = scales[g] * 0.5 + 0.03;
            assert!(
                (recon[i] - weights[i]).abs() <= tol,
                "cell {i}: recon {} vs orig {} (tol {tol})",
                recon[i],
                weights[i],
            );
        }

        // The full norm survived as an f16 passthrough of the right shape.
        let norm_t = model.full.get("model.norm.weight").expect("norm present");
        assert_eq!(norm_t.dtype, MlxFullDtype::F16);
        assert_eq!(norm_t.shape, vec![out_f as u64]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_mlx_dir_emits_hf_module_names_via_gguf_map() {
        // The `quantize --to-mlx` path (rustllama-cli) translates GGUF
        // tensor names to HF names via `map_gguf_to_hf` before handing them
        // to `write_mlx_dir`. Mirror that here: start from GGUF names, map
        // them, write, and confirm the written dir (a) classifies as MLX
        // and (b) load_mlx_dir keys the quant weights by the HF module
        // path real mlx-lm models use — proving the output is upstream-
        // loadable, not GGUF-named.
        use rustllama_kernels_cpu::mlx_affine::{dequantize_mlx_affine, quantize_mlx_affine};

        let out_f = 4usize;
        let in_f = 128usize;
        let group_size = 64usize;
        let bits = 4u32;

        // GGUF names as `quantize --to-mlx` sees them in the source model.
        let gguf_q = "blk.0.attn_q.weight";
        let gguf_norm = "output_norm.weight";
        let hf_q = crate::map_gguf_to_hf(gguf_q).expect("attn_q maps");
        let hf_norm = crate::map_gguf_to_hf(gguf_norm).expect("output_norm maps");
        assert_eq!(hf_q, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(hf_norm, "model.norm.weight");

        let weights: Vec<f32> = (0..out_f * in_f).map(pseudo_f32).collect();
        let (packed, scales, biases) = quantize_mlx_affine(&weights, group_size, bits);
        let norm: Vec<f32> = (0..out_f).map(|i| 1.0 + i as f32 * 0.1).collect();

        let tensors = vec![
            MlxWriteTensor::Quant {
                // Writer wants the module path (name minus `.weight`),
                // exactly as the CLI derives it from the mapped HF name.
                name: hf_q.strip_suffix(".weight").unwrap().to_string(),
                packed,
                scales: scales.clone(),
                biases: biases.clone(),
                group_size,
                bits,
                shape: vec![out_f as u64, in_f as u64],
            },
            MlxWriteTensor::Full {
                name: hf_norm.clone(),
                dtype: MlxFullDtype::F16,
                shape: vec![out_f as u64],
                bytes: f16_bytes(&norm),
            },
        ];

        let dir = std::env::temp_dir()
            .join(format!("rustllama-mlx-hf-names-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let mut extra = serde_json::Map::new();
        extra.insert("architectures".into(), serde_json::json!(["Qwen2ForCausalLM"]));
        write_mlx_dir(&dir, &tensors, group_size, bits, extra, None).unwrap();

        let cfg_json = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let st_bytes = std::fs::read(dir.join("model.safetensors")).unwrap();
        assert!(is_mlx_model(&cfg_json, &st_bytes), "written dir not detected as MLX");

        let model = load_mlx_dir(&dir).unwrap();
        // The quant weight is keyed by the HF module path (not the GGUF
        // `blk.0.attn_q`), so upstream mlx-lm — and our own load path which
        // HF->GGUF-maps on wiring — finds it.
        let q = model
            .quant
            .get("model.layers.0.self_attn.q_proj")
            .expect("quant keyed by HF module path");
        assert!(model.quant.get("blk.0.attn_q").is_none(), "must not emit GGUF name");
        assert_eq!(q.shape, vec![out_f as u64, in_f as u64]);
        q.validate().unwrap();

        // And it still dequants to the original weights within quant error.
        let mut recon = vec![0f32; q.n_elements() as usize];
        dequantize_mlx_affine(&q.packed, &q.scales, &q.biases, q.group_size, q.bits, &mut recon);
        for i in 0..weights.len() {
            let g = i / group_size;
            let tol = scales[g] * 0.5 + 0.03;
            assert!((recon[i] - weights[i]).abs() <= tol, "cell {i}");
        }

        // The norm carried through under its HF name.
        assert!(model.full.contains_key("model.norm.weight"), "norm keyed by HF name");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
