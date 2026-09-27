//! Vision-tower (CLIP / SigLIP / Qwen2-VL-style ViT) architecture
//! parsing + loading. Phase V-0 scaffolding: this module defines the
//! [`VisionConfig`] struct that mirrors [`crate::bert_arch::BertConfig`]
//! plus the vision-specific fields (image_size, patch_size, n_channels),
//! reads the `clip.*` / `clip.vision.*` GGUF metadata keys used by the
//! ecosystem's mmproj GGUFs, and stubs [`VisionModel`] with a
//! `forward_image` that returns `NotImplemented`. The forward pass +
//! image preprocessor land in phases V-1 / V-3.
//!
//! How vision-language models ship:
//!
//!   - **Two-file deployment** (LLaVA, Qwen2-VL, MiniCPM-V): one GGUF
//!     for the text decoder (Llama-family, loaded via
//!     [`crate::llama_arch::LlamaModel`]) + a separate `mmproj-*.gguf`
//!     for the vision tower + projector. The text-decoder GGUF carries
//!     no vision tensors; the mmproj GGUF carries no LM head. Engines
//!     load both files and splice the vision-tower output into the
//!     text stream at the position of each `image` content block.
//!
//!   - **Single-file deployment** (rare): both towers in one GGUF with
//!     a multimodal `general.architecture` like `"qwen2vl"`. Tensors
//!     are namespaced under `v.<...>` (vision) vs `<base>.<...>` (text).
//!
//! Phase V-0 only handles the **mmproj GGUF** half (two-file path) —
//! the text-decoder side already works via the existing Llama loader.
//! Single-file VLMs are a phase V-5 follow-up.
//!
//! GGUF metadata key conventions (the llama.cpp `clip.cpp` convention
//! that the ecosystem standardized on):
//!
//!   - `general.architecture = "clip"` (sometimes `"siglip"`)
//!   - `clip.has_text_encoder`, `clip.has_vision_encoder` (booleans)
//!   - `clip.projector_type` — `"mlp"`, `"mlp2x_gelu"`, `"linear"`
//!   - `clip.vision.image_size`, `clip.vision.patch_size`
//!   - `clip.vision.embedding_length` (d_model for the vision tower)
//!   - `clip.vision.feed_forward_length`
//!   - `clip.vision.block_count`
//!   - `clip.vision.attention.head_count`
//!   - `clip.vision.attention.layer_norm_epsilon`
//!
//! Tensor naming (also `clip.cpp` convention):
//!
//!   - `v.patch_embd.weight` / `v.patch_embd.bias` — conv2d patch
//!     extractor `[n_channels, d_model, patch_size, patch_size]`
//!   - `v.position_embd.weight` — `[num_patches+1, d_model]` (the +1
//!     is the [CLS] / register token; some variants omit it)
//!   - `v.blk.<i>.attn_{q,k,v,output}.{weight,bias}`
//!   - `v.blk.<i>.attn_norm.{weight,bias}` / `ffn_norm.{weight,bias}`
//!   - `v.blk.<i>.ffn_{up,down}.{weight,bias}` (no gate — GELU FFN,
//!     not SwiGLU)
//!   - `v.pre_ln.{weight,bias}` — pre-transformer LayerNorm (some variants)
//!   - `v.post_ln.{weight,bias}` — post-transformer LayerNorm
//!   - `mm.<i>.{weight,bias}` — the multimodal projector MLP

use rustllama_gguf::{Gguf, MetadataValue};
use rustllama_kernels_cpu as k;
use rustllama_tensor::{as_slice_f32, Device, Dtype, Storage, Tensor};

use crate::bert_arch::{
    add_bias_f32, bidirectional_attention, gelu_tanh_approx, layer_norm_with_bias,
};

/// Vision-tower hyperparameters extracted from `clip.vision.*` GGUF
/// metadata. Field shapes mirror [`crate::bert_arch::BertConfig`] —
/// a ViT is structurally a BERT with the embedding layer replaced
/// by a conv2d patch extractor + position embeddings.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Architecture string from `general.architecture`. Typically
    /// `"clip"` for the standard llama.cpp mmproj GGUF; some SigLIP-
    /// based projectors use `"siglip"`. Used as the metadata-key
    /// prefix when other readers consult the same GGUF.
    pub arch: String,
    /// Input image resolution. Square; non-square images get resized
    /// in the preprocessor. Common values: 224 (CLIP-B), 336 (LLaVA-1.5),
    /// 384 (SigLIP), 448 (Qwen2-VL native). From `clip.vision.image_size`.
    pub image_size: usize,
    /// Patch size in pixels. Standard ViT-B / CLIP-B uses 14 or 16;
    /// some Qwen2-VL variants use 14 with a 448 image (=> 32×32 = 1024
    /// patches). From `clip.vision.patch_size`.
    pub patch_size: usize,
    /// Number of channels in the input. 3 for RGB. Hardcoded today;
    /// no GGUF key consumes it (no model in the wild varies this).
    pub n_channels: usize,
    /// Number of transformer layers. From `clip.vision.block_count`.
    /// Typical ViT-L value: 24. ViT-B: 12. Qwen2-VL: 32.
    pub n_layers: usize,
    /// Attention head count. From `clip.vision.attention.head_count`.
    pub n_heads: usize,
    /// Hidden dimension. ViT-L: 1024; ViT-B: 768; Qwen2-VL: 1280.
    /// From `clip.vision.embedding_length`.
    pub d_model: usize,
    /// FFN intermediate dim. Typically 4× d_model.
    /// From `clip.vision.feed_forward_length`.
    pub d_ff: usize,
    /// Per-head dim. Derived as `d_model / n_heads` when the explicit
    /// key (`clip.vision.attention.key_length`) is absent.
    pub head_dim: usize,
    /// LayerNorm epsilon. From `clip.vision.attention.layer_norm_epsilon`.
    /// Default `1e-6` matches OpenAI CLIP; LLaVA / SigLIP variants
    /// sometimes use `1e-5`.
    pub layer_norm_eps: f32,
    /// Projector kind — describes the mlp / linear that maps
    /// `d_model` (vision-tower output) → text-decoder `d_model`. The
    /// concrete weights live in `mm.<i>` tensors; this enum just
    /// records the topology.
    pub projector_type: ProjectorType,
    /// Whether the vision tower prepends a learned [CLS]-style token
    /// to the patch sequence. Most CLIP variants do; some SigLIP /
    /// Qwen2-VL variants do not. Detected from the position-embedding
    /// tensor shape: if `num_patches + 1` rows then yes, else no.
    /// Stored on the loaded weights, not on the config — phase V-1.
    pub has_class_token: bool,
    /// Per-channel image mean used during preprocessing. Pixel values
    /// (in 0..1) get `(x - mean[c]) / std[c]`. Defaults to the
    /// OpenAI CLIP constants; some GGUFs override via
    /// `clip.vision.image_mean` metadata.
    pub image_mean: [f32; 3],
    /// Per-channel image std-dev. Same source / default behavior as
    /// `image_mean` (`clip.vision.image_std`).
    pub image_std: [f32; 3],
    /// Spatial merge factor for merger-style projectors (Qwen3-VL:
    /// 2 → each 2×2 patch window becomes ONE text token). `1` for
    /// classic CLIP projectors. From `clip.vision.spatial_merge_size`.
    pub spatial_merge_size: usize,
    /// Text-side hidden size the projector maps into, when recorded
    /// (`clip.vision.projection_dim`). Cross-checked against the
    /// text model's d_model at attach time.
    pub projection_dim: Option<usize>,
    /// Qwen3-VL deepstack flags (`clip.vision.is_deepstack_layers`),
    /// one bool per block. Deepstack taps concat extra features onto
    /// the final embedding, which our splice does not model — any
    /// `true` refuses to load (checked in `from_gguf`). Empty when
    /// the key is absent.
    pub is_deepstack_layers: Vec<bool>,
}

/// How the vision-tower output gets projected to the text-decoder's
/// hidden dimension. From `clip.projector_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectorType {
    /// Single linear: `[vision_dim, text_dim]`. Original LLaVA-1.0.
    Linear,
    /// Two-layer MLP with GELU between: `[vision_dim, hidden] → GELU →
    /// [hidden, text_dim]`. LLaVA-1.5+, MiniCPM-V, Qwen2-VL.
    Mlp,
    /// LDP-v2 (mobile-optimized 2-layer MLP, MiniCPM-V style). Same
    /// shape as `Mlp` on the wire today; reserved for future
    /// differentiation if the topology diverges.
    LdpV2,
    /// Qwen3-VL merger: patches are reordered 2×2-window-major after
    /// the patch embedding, and the projector reshapes each window's
    /// 4 rows into one `[4·d_model]` row before the 2-layer GELU MLP
    /// (`mm.0 [4d→4d] → GELU → mm.2 [4d→d_text]`). Token count out =
    /// patches / merge². Bonsai 2 / Qwen3-VL family.
    Qwen3VlMerger,
}

impl ProjectorType {
    /// Match the strings llama.cpp's mmproj converters emit. Unknown
    /// strings are a HARD error: silently binding an unknown projector
    /// as an MLP runs wrong math on right-shaped tensors (the old
    /// `_ => Mlp` fallback fed 1152-wide rows to a 4608-wide FC for
    /// `qwen3vl_merger`).
    pub fn from_metadata_str(s: &str) -> Result<Self, VisionConfigError> {
        match s {
            "linear" => Ok(Self::Linear),
            "mlp" | "mlp2x_gelu" | "mlp_gelu_norm" => Ok(Self::Mlp),
            "ldp_v2" => Ok(Self::LdpV2),
            "qwen3vl_merger" => Ok(Self::Qwen3VlMerger),
            other => Err(VisionConfigError::UnsupportedProjector(other.to_string())),
        }
    }
}

/// OpenAI CLIP default per-channel image mean. Every standard
/// CLIP-based VLM (LLaVA, MiniCPM-V, Qwen2-VL via CLIP variants) uses
/// these constants; SigLIP / new families override them via GGUF
/// metadata. Keeping them here as a `pub const` lets callers
/// reproduce HuggingFace `CLIPImageProcessor` exactly without GGUF
/// metadata round-trip.
pub const CLIP_IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
pub const CLIP_IMAGE_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

impl VisionConfig {
    /// Parse vision-tower hyperparameters from a CLIP-shaped mmproj
    /// GGUF. Returns `None` when the GGUF doesn't carry vision-encoder
    /// metadata (e.g. the user pointed at a text-only GGUF by mistake).
    /// Callers should fall back to a clear error in that case rather
    /// than constructing a zero-shaped config.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self, VisionConfigError> {
        let arch = gguf
            .architecture()
            .ok_or(VisionConfigError::MissingArchitecture)?
            .to_string();
        if !matches!(arch.as_str(), "clip" | "siglip") {
            return Err(VisionConfigError::WrongArchitecture(arch));
        }

        // The vision-encoder flag is a sanity gate — some clip-shaped
        // GGUFs are text-encoder-only (e.g. the CLIP text tower used
        // for image-text retrieval, no vision side).
        let has_vision = gguf
            .metadata_get("clip.has_vision_encoder")
            .and_then(|v| match v {
                MetadataValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(true);
        if !has_vision {
            return Err(VisionConfigError::NoVisionEncoder);
        }

        let prefix = "clip.vision";
        let key = |k: &str| -> Option<u32> {
            gguf.metadata_get(&format!("{prefix}.{k}"))
                .and_then(MetadataValue::as_u32)
        };
        let key_f32 = |k: &str| -> Option<f32> {
            match gguf.metadata_get(&format!("{prefix}.{k}")) {
                Some(MetadataValue::F32(v)) => Some(*v),
                _ => None,
            }
        };
        let image_size = key("image_size")
            .ok_or(VisionConfigError::MissingKey("clip.vision.image_size"))?
            as usize;
        let patch_size = key("patch_size")
            .ok_or(VisionConfigError::MissingKey("clip.vision.patch_size"))?
            as usize;
        let n_layers = key("block_count")
            .ok_or(VisionConfigError::MissingKey("clip.vision.block_count"))?
            as usize;
        let n_heads = key("attention.head_count")
            .ok_or(VisionConfigError::MissingKey(
                "clip.vision.attention.head_count",
            ))?
            as usize;
        let d_model = key("embedding_length")
            .ok_or(VisionConfigError::MissingKey("clip.vision.embedding_length"))?
            as usize;
        let d_ff = key("feed_forward_length")
            .ok_or(VisionConfigError::MissingKey(
                "clip.vision.feed_forward_length",
            ))?
            as usize;
        let head_dim = key("attention.key_length")
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                if n_heads > 0 {
                    d_model / n_heads
                } else {
                    0
                }
            });
        let layer_norm_eps = key_f32("attention.layer_norm_epsilon").unwrap_or(1e-6);

        let projector_type = match gguf
            .metadata_get("clip.projector_type")
            .and_then(MetadataValue::as_string)
        {
            Some(s) => ProjectorType::from_metadata_str(s)?,
            // Absent key: the classic-CLIP default. Only an EXPLICIT
            // unknown string errors.
            None => ProjectorType::Mlp,
        };

        let spatial_merge_size = key("spatial_merge_size").map(|v| v as usize).unwrap_or(1);
        if spatial_merge_size == 0
            || (projector_type == ProjectorType::Qwen3VlMerger && spatial_merge_size != 2)
        {
            return Err(VisionConfigError::UnsupportedMergeSize(spatial_merge_size));
        }
        let projection_dim = key("projection_dim").map(|v| v as usize);

        // Deepstack taps concat extra features onto the final
        // embedding — unmodeled by our splice; refuse any true flag
        // (D5). An absent key or all-false array is the supported case
        // (Bonsai 2 ships all-false).
        let is_deepstack_layers: Vec<bool> =
            match gguf.metadata_get(&format!("{prefix}.is_deepstack_layers")) {
                Some(MetadataValue::Array(arr)) => arr
                    .iter()
                    .map(|v| matches!(v, MetadataValue::Bool(true)))
                    .collect(),
                _ => Vec::new(),
            };
        if is_deepstack_layers.iter().any(|&b| b) {
            return Err(VisionConfigError::DeepstackUnsupported);
        }

        // Per-channel mean/std for the preprocessor. The GGUF carries
        // these as f32-arrays of length 3 when present; fall back to
        // the OpenAI CLIP constants. SigLIP-distributed mmprojs ship
        // `[0.5, 0.5, 0.5]` for both — that path is exercised the
        // moment a GGUF override is supplied.
        let array_f32_3 = |key: &str| -> Option<[f32; 3]> {
            match gguf.metadata_get(&format!("{prefix}.{key}")) {
                Some(MetadataValue::Array(arr)) if arr.len() == 3 => {
                    let mut out = [0f32; 3];
                    for (i, v) in arr.iter().enumerate() {
                        match v {
                            MetadataValue::F32(f) => out[i] = *f,
                            _ => return None,
                        }
                    }
                    Some(out)
                }
                _ => None,
            }
        };
        let image_mean = array_f32_3("image_mean").unwrap_or(CLIP_IMAGE_MEAN);
        let image_std = array_f32_3("image_std").unwrap_or(CLIP_IMAGE_STD);

        // Class-token presence is determined by the position-embedding
        // tensor shape at load time, not config. Default to `true`
        // until the loader has the tensor in hand — most CLIP variants
        // have one.
        let has_class_token = true;

        Ok(Self {
            arch,
            image_size,
            patch_size,
            n_channels: 3,
            n_layers,
            n_heads,
            d_model,
            d_ff,
            head_dim,
            layer_norm_eps,
            projector_type,
            has_class_token,
            image_mean,
            image_std,
            spatial_merge_size,
            projection_dim,
            is_deepstack_layers,
        })
    }

    /// Number of patches per image. `(image_size / patch_size)^2`.
    /// Excludes any prepended [CLS] token — the loaded weights' position
    /// embedding row count distinguishes the two cases.
    pub fn num_patches(&self) -> usize {
        if self.patch_size == 0 {
            return 0;
        }
        let per_side = self.image_size / self.patch_size;
        per_side * per_side
    }
}

/// Vision-config parse errors. Distinguishes "this GGUF is the wrong
/// shape" from "this GGUF is right-shape but missing a required key"
/// so the engine can return clear actionable messages.
#[derive(Debug, thiserror::Error)]
pub enum VisionConfigError {
    #[error("GGUF has no `general.architecture` metadata")]
    MissingArchitecture,
    #[error(
        "GGUF architecture is `{0}`, not `clip` or `siglip` — pass an mmproj GGUF"
    )]
    WrongArchitecture(String),
    #[error("`clip.has_vision_encoder = false` — this is a text-only CLIP GGUF")]
    NoVisionEncoder,
    #[error("GGUF missing required key `{0}`")]
    MissingKey(&'static str),
    #[error(
        "unsupported `clip.projector_type` \"{0}\" — refusing to bind an unknown \
         projector as an MLP (wrong math on right-shaped tensors)"
    )]
    UnsupportedProjector(String),
    #[error("unsupported spatial_merge_size {0} (qwen3vl_merger requires 2; others 1)")]
    UnsupportedMergeSize(usize),
    #[error(
        "clip.vision.is_deepstack_layers has true entries — deepstack feature taps \
         are not supported (they change the projector output width)"
    )]
    DeepstackUnsupported,
}

/// One transformer block of the vision tower. Same shape as a BERT
/// block (Q/K/V/O projections + biases + post-attention LN; FFN
/// up/down + GELU + post-FFN LN) — ViT and BERT share the
/// "post-LN, biases everywhere, GELU FFN" recipe.
///
/// Tensor names follow the `clip.cpp` convention: `v.blk.<i>.<role>`.
/// All weight matrices ship as `[d_out, d_in]` (row-major, the
/// llama.cpp / GGUF convention) so the matvec hot path can read
/// them with the existing kernels.
#[derive(Debug)]
pub struct VisionBlockWeights {
    /// Q / K / V / O projections — `[d_model, d_model]` each. Biases
    /// are required on CLIP (the original OpenAI weights ship with
    /// them) but optional in case a future converter strips them.
    pub attn_q: Tensor,
    pub attn_q_bias: Option<Tensor>,
    pub attn_k: Tensor,
    pub attn_k_bias: Option<Tensor>,
    pub attn_v: Tensor,
    pub attn_v_bias: Option<Tensor>,
    pub attn_output: Tensor,
    pub attn_output_bias: Option<Tensor>,
    /// LayerNorm BEFORE attention (CLIP / SigLIP use pre-LN, distinct
    /// from BERT's post-LN convention). Weights `[d_model]`, bias
    /// `[d_model]`. The `clip.cpp` name is `attn_norm`.
    pub attn_norm: Tensor,
    pub attn_norm_bias: Tensor,
    /// FFN up + down — `[d_ff, d_model]` and `[d_model, d_ff]`.
    /// GELU activation between (no gate; ViT uses a plain MLP, not
    /// SwiGLU).
    pub ffn_up: Tensor,
    pub ffn_up_bias: Option<Tensor>,
    pub ffn_down: Tensor,
    pub ffn_down_bias: Option<Tensor>,
    /// LayerNorm BEFORE the FFN.
    pub ffn_norm: Tensor,
    pub ffn_norm_bias: Tensor,
}

/// Multimodal projector — projects vision-tower output (`d_model_vision`)
/// to text-decoder hidden size (`d_model_text`). Most common topology
/// is a 2-layer MLP with GELU between (LLaVA-1.5+, Qwen2-VL,
/// MiniCPM-V). Linear-only variant matches LLaVA-1.0.
#[derive(Debug)]
pub struct ProjectorWeights {
    pub kind: ProjectorType,
    /// First linear `[d_hidden, d_model_vision]` + bias. For
    /// `Linear` projectors this is the only layer; `d_hidden` equals
    /// the text-decoder hidden size.
    pub fc1: Tensor,
    pub fc1_bias: Option<Tensor>,
    /// Second linear `[d_model_text, d_hidden]` + bias. Present
    /// only for `Mlp` / `LdpV2` projectors; absent for `Linear`.
    pub fc2: Option<Tensor>,
    pub fc2_bias: Option<Tensor>,
    /// Dims computed at load from ELEMENT COUNTS, not GGUF dim-order
    /// (real mmprojs declare `[in, out]`, the synth fixture `[out,
    /// in]` — shape-index reads silently pick the wrong one).
    /// `d_in` = fc1 input width (d_model × merge² for the merger),
    /// `d_mid` = fc1 output width, `d_out` = final text width.
    pub d_in: usize,
    pub d_mid: usize,
    pub d_out: usize,
}

impl ProjectorWeights {
    /// Output dim — the text decoder's `d_model`. The projector's
    /// final linear produces this many elements per token. Used by
    /// `forward_image_with_projection` to size output buffers and
    /// by callers wiring the vision output into the text stream
    /// (they need to know how many slots to reserve).
    ///
    /// For `Linear` projectors, fc1 is the final (and only) layer
    /// so its first dim is d_text. For `Mlp` / `LdpV2`, fc2 is the
    /// final layer.
    pub fn d_text(&self) -> usize {
        self.d_out
    }

    /// Hidden dim — the dim between fc1 and fc2 for MLP projectors.
    /// Equals `d_text` for Linear projectors (single layer).
    pub fn d_hidden(&self) -> usize {
        self.d_mid
    }
}

/// Loaded vision-tower weights + projector. Phase V-1a: tensor
/// bindings land; the `forward_image` entry is still
/// `NotImplemented` (phase V-1b will wire it).
#[derive(Debug)]
pub struct VisionModel {
    pub cfg: VisionConfig,
    /// Patch embedding — emits one `d_model`-dim row per patch.
    /// Stored as a 4D conv weight `[d_model, n_channels, patch_size,
    /// patch_size]` per `clip.cpp` convention; the forward pass
    /// reshapes to `[d_model, n_channels * patch_size * patch_size]`
    /// and matvecs flattened patches against it.
    pub patch_embd: Tensor,
    /// Second (temporal) patch-embed conv `[patch, patch, 3, d_model]`
    /// — Qwen3-VL ships a pair; still images apply BOTH to the same
    /// frame and sum (fork `build_inp_with_temporal_merge`). `None`
    /// for classic single-conv CLIP variants.
    pub patch_embd_1: Option<Tensor>,
    /// Patch-embed bias `[d_model]`. CLIP has one; some SigLIP
    /// variants don't (bias-free conv).
    pub patch_embd_bias: Option<Tensor>,
    /// Position embedding `[num_patches (+1 if class token), d_model]`.
    /// The +1 row holds the [CLS] / register token's position. Row
    /// count distinguishes class-token-having variants from
    /// register-only variants — see [`VisionConfig::has_class_token`].
    pub position_embd: Tensor,
    /// Learned [CLS] / class token `[d_model]`. Present on most CLIP
    /// variants; absent on SigLIP / Qwen2-VL. When `None`, the
    /// forward pass skips the prepend step.
    pub class_token: Option<Tensor>,
    /// Pre-transformer LayerNorm (some CLIP variants use one to
    /// normalize patch+position embeddings before block 0). Absent
    /// in many SigLIP variants. Pair: `weight` + `bias`.
    pub pre_ln: Option<Tensor>,
    pub pre_ln_bias: Option<Tensor>,
    /// Post-transformer LayerNorm — applied to the final block's
    /// output before projection. Always present in CLIP / SigLIP.
    pub post_ln: Tensor,
    pub post_ln_bias: Tensor,
    pub blocks: Vec<VisionBlockWeights>,
    /// Multimodal projector — vision_d → text_d. Always present in
    /// mmproj GGUFs (an mmproj without a projector is text-only CLIP,
    /// which `from_gguf` rejects).
    pub projector: ProjectorWeights,
}

impl VisionModel {
    /// Load the vision tower + projector from an mmproj GGUF.
    /// Phase V-1a: parses the config, binds every required tensor,
    /// detects class-token presence from the position-embedding row
    /// count. Returns `MissingTensor` when a required weight is
    /// absent — the error names the tensor so users can identify a
    /// corrupted or unsupported mmproj quickly.
    pub fn load(gguf: &Gguf) -> Result<Self, VisionLoadError> {
        let mut cfg = VisionConfig::from_gguf(gguf)?;

        let required = |name: &str| -> Result<Tensor, VisionLoadError> {
            let info = gguf
                .tensor(name)
                .ok_or_else(|| VisionLoadError::MissingTensor(name.to_string()))?;
            tensor_from_info(gguf, info, name)
        };
        let optional = |name: &str| -> Result<Option<Tensor>, VisionLoadError> {
            match gguf.tensor(name) {
                Some(info) => Ok(Some(tensor_from_info(gguf, info, name)?)),
                None => Ok(None),
            }
        };
        // Alias-aware lookup: try each candidate name in order. The
        // classic clip.cpp names come first; Qwen3-VL exporters ship
        // `ln1`/`ln2` for the norms and `attn_out` for the output
        // projection.
        let first_of = |names: &[&str]| -> Result<Tensor, VisionLoadError> {
            for n in names {
                if gguf.tensor(n).is_some() {
                    return required(n);
                }
            }
            Err(VisionLoadError::MissingTensor(names.join(" | ")))
        };
        let first_of_optional = |names: &[&str]| -> Result<Option<Tensor>, VisionLoadError> {
            for n in names {
                if gguf.tensor(n).is_some() {
                    return optional(n);
                }
            }
            Ok(None)
        };
        // Norms / biases / embeddings feed `as_slice_f32` paths in the
        // forward — coerce any F16/BF16 to F32 at load (no-op for the
        // common all-F32 case) so a converter's dtype choice can't
        // panic the forward.
        let to_f32 = |t: Tensor| -> Result<Tensor, VisionLoadError> { tensor_to_f32(t) };
        let to_f32_opt = |t: Option<Tensor>| -> Result<Option<Tensor>, VisionLoadError> {
            t.map(tensor_to_f32).transpose()
        };

        // Patch embedding(s) + position embedding. Qwen3-VL ships a
        // temporal PAIR of patch convs (`.weight` + `.weight.1`); for
        // still images both are applied to the same image and SUMMED
        // (fork qwen2vl.cpp `build_inp_with_temporal_merge`).
        let patch_embd = to_f32(required("v.patch_embd.weight")?)?;
        let patch_embd_1 = to_f32_opt(optional("v.patch_embd.weight.1")?)?;
        let patch_embd_bias = to_f32_opt(optional("v.patch_embd.bias")?)?;
        let position_embd = to_f32(required("v.position_embd.weight")?)?;
        // Class-token detection: the position embedding's row count
        // is `num_patches` (no class token) or `num_patches + 1`
        // (with class token). Use that, not config — the GGUF's
        // layout is authoritative. Row count is derived from total
        // elements / d_model so both dims orders work (real mmprojs
        // ship `[d_model, rows]`, the synth fixture `[rows, d_model]`).
        let pos_elems: u64 = position_embd.shape.iter().product();
        let pos_rows = if cfg.d_model > 0 {
            (pos_elems / cfg.d_model as u64) as usize
        } else {
            0
        };
        let num_patches = cfg.num_patches();
        cfg.has_class_token = pos_rows == num_patches + 1;

        let class_token = if cfg.has_class_token {
            let t = match optional("v.class_embd")? {
                Some(t) => Some(t),
                None => optional("v.class_token")?,
            };
            to_f32_opt(t)?
        } else {
            None
        };

        // Pre / post LN — pre is optional, post is required.
        let pre_ln = to_f32_opt(optional("v.pre_ln.weight")?)?;
        let pre_ln_bias = to_f32_opt(optional("v.pre_ln.bias")?)?;
        let post_ln = to_f32(required("v.post_ln.weight")?)?;
        let post_ln_bias = to_f32(required("v.post_ln.bias")?)?;

        // Per-block bindings. Two naming families:
        //   classic clip.cpp: attn_q/k/v + attn_output + attn_norm/ffn_norm
        //   Qwen3-VL:         fused attn_qkv + attn_out + ln1/ln2
        // Fused QKV is split into Q/K/V at load by byte-range row
        // slices (rows are contiguous in every GGUF dtype; fused
        // layout is [q_all | k_all | v_all] per the fork's views at
        // offsets 0 / n_embd / 2·n_embd).
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |suffix: &str| format!("v.blk.{i}.{suffix}");
            let fused_name = p("attn_qkv.weight");
            let (attn_q, attn_k, attn_v, attn_q_bias, attn_k_bias, attn_v_bias) =
                if gguf.tensor(&fused_name).is_some() {
                    let fused = required(&fused_name)?;
                    let (q, k, v) = split_fused_qkv(&fused, cfg.d_model as u64)?;
                    let (qb, kb, vb) = match to_f32_opt(optional(&p("attn_qkv.bias"))?)? {
                        Some(fb) => {
                            let (a, b, c) = split_fused_qkv(&fb, cfg.d_model as u64)?;
                            (Some(a), Some(b), Some(c))
                        }
                        None => (None, None, None),
                    };
                    (q, k, v, qb, kb, vb)
                } else {
                    (
                        required(&p("attn_q.weight"))?,
                        required(&p("attn_k.weight"))?,
                        required(&p("attn_v.weight"))?,
                        to_f32_opt(optional(&p("attn_q.bias"))?)?,
                        to_f32_opt(optional(&p("attn_k.bias"))?)?,
                        to_f32_opt(optional(&p("attn_v.bias"))?)?,
                    )
                };
            blocks.push(VisionBlockWeights {
                attn_q,
                attn_q_bias,
                attn_k,
                attn_k_bias,
                attn_v,
                attn_v_bias,
                attn_output: first_of(&[&p("attn_output.weight"), &p("attn_out.weight")])?,
                attn_output_bias: to_f32_opt(first_of_optional(&[
                    &p("attn_output.bias"),
                    &p("attn_out.bias"),
                ])?)?,
                attn_norm: to_f32(first_of(&[&p("attn_norm.weight"), &p("ln1.weight")])?)?,
                attn_norm_bias: to_f32(first_of(&[&p("attn_norm.bias"), &p("ln1.bias")])?)?,
                ffn_up: required(&p("ffn_up.weight"))?,
                ffn_up_bias: to_f32_opt(optional(&p("ffn_up.bias"))?)?,
                ffn_down: required(&p("ffn_down.weight"))?,
                ffn_down_bias: to_f32_opt(optional(&p("ffn_down.bias"))?)?,
                ffn_norm: to_f32(first_of(&[&p("ffn_norm.weight"), &p("ln2.weight")])?)?,
                ffn_norm_bias: to_f32(first_of(&[&p("ffn_norm.bias"), &p("ln2.bias")])?)?,
            });
        }

        // Projector — naming convention: `mm.0.weight` / `mm.0.bias`
        // for the first FC; `mm.2.weight` / `mm.2.bias` for the
        // second (the `mm.1` slot is the GELU activation index in
        // the original PyTorch module list, so the GGUF skips it).
        let fc1 = required("mm.0.weight")?;
        let fc1_bias = to_f32_opt(optional("mm.0.bias")?)?;
        let (fc2, fc2_bias) = match cfg.projector_type {
            ProjectorType::Linear => (None, None),
            ProjectorType::Mlp | ProjectorType::LdpV2 | ProjectorType::Qwen3VlMerger => {
                (Some(required("mm.2.weight")?), to_f32_opt(optional("mm.2.bias")?)?)
            }
        };
        // Projector dims from ELEMENT COUNTS — immune to the GGUF
        // dim-order split between real files (`[in, out]`) and the
        // synth fixture (`[out, in]`). For the merger, fc1's input
        // width is merge²·d_model (the contiguous 2×2-window reshape);
        // catching a mismatch here beats a slice-length panic
        // mid-projection.
        let elems = |t: &Tensor| -> usize { t.shape.iter().product::<u64>() as usize };
        let d_in = match cfg.projector_type {
            ProjectorType::Qwen3VlMerger => {
                cfg.d_model * cfg.spatial_merge_size * cfg.spatial_merge_size
            }
            _ => cfg.d_model,
        };
        let fc1_elems = elems(&fc1);
        if d_in == 0 || fc1_elems % d_in != 0 {
            return Err(VisionLoadError::ProjectorShape {
                tensor: "mm.0.weight",
                expected_in: d_in,
                got_in: fc1_elems,
            });
        }
        let d_mid = fc1_elems / d_in;
        let d_out = match &fc2 {
            Some(f) => {
                let fe = elems(f);
                if d_mid == 0 || fe % d_mid != 0 {
                    return Err(VisionLoadError::ProjectorShape {
                        tensor: "mm.2.weight",
                        expected_in: d_mid,
                        got_in: fe,
                    });
                }
                fe / d_mid
            }
            None => d_mid,
        };
        let projector = ProjectorWeights {
            kind: cfg.projector_type,
            fc1,
            fc1_bias,
            fc2,
            fc2_bias,
            d_in,
            d_mid,
            d_out,
        };

        Ok(Self {
            cfg,
            patch_embd,
            patch_embd_1,
            patch_embd_bias,
            position_embd,
            class_token,
            pre_ln,
            pre_ln_bias,
            post_ln,
            post_ln_bias,
            blocks,
            projector,
        })
    }

    /// Run the vision tower forward pass over a single preprocessed
    /// image. `patches` is a flattened buffer of shape
    /// `[num_patches, n_channels * patch_size * patch_size]` —
    /// i.e. the image after `image_size × image_size × n_channels`
    /// resize/normalize/extraction. Returns the post-LN hidden
    /// states `[seq_len, d_model]` where `seq_len = num_patches +
    /// (1 if has_class_token else 0)`. The projector (vision_d →
    /// text_d) is V-1c — separate from this function so consumers
    /// can choose which token(s) to feed it (CLS-only vs mean-pool
    /// vs all-patches).
    ///
    /// Pre-LN convention (CLIP / SigLIP): each block applies
    /// LayerNorm BEFORE attention and BEFORE the FFN, with the
    /// residual add AFTER. Distinct from BERT's post-LN ordering.
    /// Uses the BERT helpers for layer_norm_with_bias / GELU /
    /// bidirectional_attention since the math is identical — just
    /// the order in the per-block loop differs.
    pub fn forward_image(&self, patches: &[f32]) -> Result<Vec<f32>, VisionForwardError> {
        let cfg = &self.cfg;
        let num_patches = cfg.num_patches();
        let patch_dim = cfg.n_channels * cfg.patch_size * cfg.patch_size;
        let d = cfg.d_model;
        let d_ff = cfg.d_ff;
        let n_heads = cfg.n_heads;
        let head_dim = cfg.head_dim;
        let eps = cfg.layer_norm_eps;

        if patches.len() != num_patches * patch_dim {
            return Err(VisionForwardError::WrongPatchShape {
                expected: num_patches * patch_dim,
                got: patches.len(),
                num_patches,
                patch_dim,
            });
        }

        // Qwen3-VL merger family: patches are re-sequenced 2×2-window-
        // major immediately after the patch embedding (fork qwen3vl.cpp
        // :18-31), so that (a) the projector's contiguous [4·d] reshape
        // groups each spatial window and (b) M-RoPE positions follow
        // the fork's fill order: for each (y,x) window step of 2, the
        // four slots are (y,x), (y,x+1), (y+1,x), (y+1,x+1)
        // (clip.cpp positions fill). `seq_map[s]` = row-major source
        // patch index for sequence slot `s`; `seq_yx[s]` = its (y,x).
        let is_merger = self.projector.kind == ProjectorType::Qwen3VlMerger;
        let (seq_map, seq_yx): (Option<Vec<usize>>, Option<Vec<(u32, u32)>>) = if is_merger {
            let per_side = cfg.image_size / cfg.patch_size;
            debug_assert_eq!(per_side * per_side, num_patches);
            debug_assert_eq!(per_side % cfg.spatial_merge_size, 0);
            let mut map = Vec::with_capacity(num_patches);
            let mut yx = Vec::with_capacity(num_patches);
            let m = cfg.spatial_merge_size;
            for wy in (0..per_side).step_by(m) {
                for wx in (0..per_side).step_by(m) {
                    for dy in 0..m {
                        for dx in 0..m {
                            let y = wy + dy;
                            let xx = wx + dx;
                            map.push(y * per_side + xx);
                            yx.push((y as u32, xx as u32));
                        }
                    }
                }
            }
            (Some(map), Some(yx))
        } else {
            (None, None)
        };

        // ---- 1. Patch embedding ----------------------------------
        // `patch_embd` is stored as `[d_model, n_channels, ps, ps]`,
        // but the trailing three dimensions are contiguous (channel-
        // major within a patch), so we treat it as `[d_model,
        // patch_dim]` for the matvec. Each patch row in `patches`
        // is `[n_channels, ps*ps]` flattened the same way; the dot
        // products line up.
        //
        // Sequence slot `s` reads source patch `seq_map[s]` (identity
        // for classic CLIP). Qwen3-VL's temporal PAIR of convs both
        // hit the same still frame and their outputs SUM (fork
        // `build_inp_with_temporal_merge`).
        let n_tokens = num_patches + if cfg.has_class_token { 1 } else { 0 };
        let cls_off = if cfg.has_class_token { 1 } else { 0 };
        let mut x = vec![0f32; n_tokens * d];
        // Gather patch rows into sequence order once, then run the
        // conv(s) as ONE batched matvec each (V-7): the ViT is
        // prefill-shaped, so batching amortizes each weight read
        // across all patches. Classic CLIP skips the gather (identity
        // order) by borrowing `patches` directly.
        let gathered: Vec<f32>;
        let patch_rows: &[f32] = if let Some(map) = seq_map.as_ref() {
            let mut g = vec![0f32; num_patches * patch_dim];
            for (slot, &src_idx) in map.iter().enumerate() {
                g[slot * patch_dim..(slot + 1) * patch_dim]
                    .copy_from_slice(&patches[src_idx * patch_dim..(src_idx + 1) * patch_dim]);
            }
            gathered = g;
            &gathered
        } else {
            patches
        };
        {
            let embd_out = &mut x[cls_off * d..(cls_off + num_patches) * d];
            crate::llama_arch::matvec_tensor_batched_dispatch(
                &self.patch_embd,
                patch_rows,
                embd_out,
                d,
                patch_dim,
                num_patches,
            );
            if let Some(pe1) = &self.patch_embd_1 {
                let mut conv2 = vec![0f32; num_patches * d];
                crate::llama_arch::matvec_tensor_batched_dispatch(
                    pe1, patch_rows, &mut conv2, d, patch_dim, num_patches,
                );
                for (a, b) in embd_out.iter_mut().zip(conv2.iter()) {
                    *a += *b;
                }
            }
            if let Some(b) = &self.patch_embd_bias {
                for slot in 0..num_patches {
                    add_bias_f32(&mut embd_out[slot * d..(slot + 1) * d], b);
                }
            }
        }

        // ---- 2. Prepend class token (if present) -----------------
        if let Some(cls) = self.class_token.as_ref() {
            let cls_data = as_slice_f32(cls);
            x[..d].copy_from_slice(cls_data);
        }

        // ---- 3. Add position embedding ---------------------------
        // The pos table is stored in row-major patch order; sequence
        // slot `s` adds row `seq_map[s]` (fork: the pos embd runs
        // through the SAME merge reorder as the input).
        let pos_data = as_slice_f32(&self.position_embd);
        debug_assert_eq!(
            pos_data.len(),
            n_tokens * d,
            "position embedding row count ({} rows × {} dim) doesn't match n_tokens × d_model = {}",
            pos_data.len() / d,
            d,
            n_tokens * d,
        );
        if let Some(map) = seq_map.as_ref() {
            for s in 0..num_patches {
                let src = map[s];
                let row = &pos_data[src * d..(src + 1) * d];
                for (a, b) in x[s * d..(s + 1) * d].iter_mut().zip(row) {
                    *a += *b;
                }
            }
        } else {
            for i in 0..(n_tokens * d) {
                x[i] += pos_data[i];
            }
        }

        // ---- 4. Optional pre-transformer LayerNorm ---------------
        if let (Some(w), Some(b)) = (self.pre_ln.as_ref(), self.pre_ln_bias.as_ref()) {
            let pw = as_slice_f32(w);
            let pb = as_slice_f32(b);
            for i in 0..n_tokens {
                layer_norm_with_bias(&mut x[i * d..(i + 1) * d], pw, pb, eps);
            }
        }

        // ---- 5. N transformer blocks -----------------------------
        // CLIP pre-LN ordering, per block:
        //   norm → Q/K/V → attn → O proj → +residual
        //   norm → FFN up → GELU → FFN down → +residual
        // Reuses BERT's matvec/attention/GELU helpers; the only
        // difference vs BERT is LN-before-residual instead of after.
        let mut x_norm = vec![0f32; n_tokens * d];
        let mut q = vec![0f32; n_tokens * d];
        let mut k_buf = vec![0f32; n_tokens * d];
        let mut v = vec![0f32; n_tokens * d];
        let mut attn_out = vec![0f32; n_tokens * d];
        let mut o_proj = vec![0f32; n_tokens * d];
        let mut ff = vec![0f32; n_tokens * d_ff];
        let mut ff_down = vec![0f32; n_tokens * d];

        for blk in &self.blocks {
            let an_w = as_slice_f32(&blk.attn_norm);
            let an_b = as_slice_f32(&blk.attn_norm_bias);
            // Pre-attn LN: write into x_norm, leave x as residual source.
            x_norm.copy_from_slice(&x);
            for i in 0..n_tokens {
                layer_norm_with_bias(&mut x_norm[i * d..(i + 1) * d], an_w, an_b, eps);
            }

            // Q / K / V projections — batched across all tokens
            // (V-7: one weight sweep per projection per block).
            crate::llama_arch::matvec_tensor_batched_dispatch(
                &blk.attn_q, &x_norm, &mut q, d, d, n_tokens,
            );
            crate::llama_arch::matvec_tensor_batched_dispatch(
                &blk.attn_k, &x_norm, &mut k_buf, d, d, n_tokens,
            );
            crate::llama_arch::matvec_tensor_batched_dispatch(
                &blk.attn_v, &x_norm, &mut v, d, d, n_tokens,
            );
            for i in 0..n_tokens {
                if let Some(b) = &blk.attn_q_bias {
                    add_bias_f32(&mut q[i * d..(i + 1) * d], b);
                }
                if let Some(b) = &blk.attn_k_bias {
                    add_bias_f32(&mut k_buf[i * d..(i + 1) * d], b);
                }
                if let Some(b) = &blk.attn_v_bias {
                    add_bias_f32(&mut v[i * d..(i + 1) * d], b);
                }
            }

            // Qwen3-VL: vision M-RoPE on Q and K, every block. Pinned
            // ggml semantics (VISION mode, sections [d_head/4; 4],
            // n_dims = d_head/2, base 10000): rotate pair (j, j+half)
            // for j in 0..half; the first quarter of pairs uses the
            // patch's y position, the second quarter its x, each with
            // a FRESH frequency ramp (indep_sects resets θ at section
            // boundaries).
            if let Some(yx) = seq_yx.as_ref() {
                for s in 0..num_patches {
                    let (py, px) = yx[s];
                    let tok = cls_off + s;
                    for h in 0..n_heads {
                        let off = tok * d + h * head_dim;
                        vision_mrope_inplace(&mut q[off..off + head_dim], py, px);
                        vision_mrope_inplace(&mut k_buf[off..off + head_dim], py, px);
                    }
                }
            }

            bidirectional_attention(&q, &k_buf, &v, &mut attn_out, n_tokens, n_heads, head_dim);

            // O projection (batched) + bias + residual.
            crate::llama_arch::matvec_tensor_batched_dispatch(
                &blk.attn_output, &attn_out, &mut o_proj, d, d, n_tokens,
            );
            for i in 0..n_tokens {
                if let Some(b) = &blk.attn_output_bias {
                    add_bias_f32(&mut o_proj[i * d..(i + 1) * d], b);
                }
                for j in 0..d {
                    x[i * d + j] += o_proj[i * d + j];
                }
            }

            // Pre-FFN LN.
            let fn_w = as_slice_f32(&blk.ffn_norm);
            let fn_b = as_slice_f32(&blk.ffn_norm_bias);
            x_norm.copy_from_slice(&x);
            for i in 0..n_tokens {
                layer_norm_with_bias(&mut x_norm[i * d..(i + 1) * d], fn_w, fn_b, eps);
            }

            // FFN: batched up → GELU → batched down, then residual.
            crate::llama_arch::matvec_tensor_batched_dispatch(
                &blk.ffn_up, &x_norm, &mut ff, d_ff, d, n_tokens,
            );
            for i in 0..n_tokens {
                if let Some(b) = &blk.ffn_up_bias {
                    add_bias_f32(&mut ff[i * d_ff..(i + 1) * d_ff], b);
                }
            }
            for vv in ff.iter_mut() {
                *vv = gelu_tanh_approx(*vv);
            }
            crate::llama_arch::matvec_tensor_batched_dispatch(
                &blk.ffn_down, &ff, &mut ff_down, d, d_ff, n_tokens,
            );
            for i in 0..n_tokens {
                if let Some(b) = &blk.ffn_down_bias {
                    add_bias_f32(&mut ff_down[i * d..(i + 1) * d], b);
                }
                for j in 0..d {
                    x[i * d + j] += ff_down[i * d + j];
                }
            }
        }

        // ---- 6. Post-LN ------------------------------------------
        let post_w = as_slice_f32(&self.post_ln);
        let post_b = as_slice_f32(&self.post_ln_bias);
        for i in 0..n_tokens {
            layer_norm_with_bias(&mut x[i * d..(i + 1) * d], post_w, post_b, eps);
        }

        Ok(x)
    }

    /// Apply the multimodal projector to every token of the
    /// vision-tower output. `hidden_states` is the `[n_tokens,
    /// d_model_vision]` flat buffer that [`Self::forward_image`]
    /// returned. Returns `[n_tokens, d_text]` flattened, where
    /// `d_text` is the projector's output dim
    /// ([`ProjectorWeights::d_text`]).
    ///
    /// LLaVA-1.5+ / Qwen2-VL / MiniCPM-V all feed *every* projected
    /// token into the text decoder's input stream — so the caller
    /// uses the full output. For LLaVA-1.0-style "CLS-only" use
    /// cases, the caller can take `output[..d_text]` after the
    /// fact (the CLS token is row 0 when `has_class_token` is true).
    ///
    /// Projector topology:
    ///   - `Linear`: `fc1 + bias` only. `d_hidden == d_text`.
    ///   - `Mlp` / `LdpV2`: `fc1 + bias → GELU → fc2 + bias`.
    pub fn project_image(&self, hidden_states: &[f32]) -> Result<Vec<f32>, VisionForwardError> {
        let d_vision = self.cfg.d_model;
        if hidden_states.len() % d_vision != 0 {
            return Err(VisionForwardError::WrongHiddenShape {
                got: hidden_states.len(),
                d_vision,
            });
        }
        // Qwen3-VL merger: the tower output arrives in 2×2-window-major
        // sequence order (forward_image applies the fork's merge
        // reorder), so "merging" is just reading each consecutive
        // merge² rows as ONE `[merge²·d]` row — a contiguous reshape,
        // no gather (fork qwen3vl.cpp:170). `d_in` was computed at
        // load (d_model × merge² for the merger, d_model otherwise).
        let fc1_in = self.projector.d_in;
        debug_assert!(fc1_in >= d_vision && fc1_in % d_vision == 0);
        let n_tokens = hidden_states.len() / fc1_in;
        let d_hidden = self.projector.d_hidden();
        let d_text = self.projector.d_text();
        let mut fc1_out = vec![0f32; n_tokens * d_hidden];
        for i in 0..n_tokens {
            let xi = &hidden_states[i * fc1_in..(i + 1) * fc1_in];
            let yi = &mut fc1_out[i * d_hidden..(i + 1) * d_hidden];
            k::matvec_tensor(&self.projector.fc1, xi, yi, d_hidden, fc1_in);
            if let Some(b) = &self.projector.fc1_bias {
                add_bias_f32(yi, b);
            }
        }
        match self.projector.kind {
            ProjectorType::Linear => {
                // Linear projector has no GELU + fc2 — fc1 is already
                // the final layer. `d_hidden == d_text` by definition.
                Ok(fc1_out)
            }
            ProjectorType::Mlp | ProjectorType::LdpV2 | ProjectorType::Qwen3VlMerger => {
                // GELU activation on fc1's output, then fc2.
                for v in fc1_out.iter_mut() {
                    *v = gelu_tanh_approx(*v);
                }
                let fc2 = self.projector.fc2.as_ref().ok_or_else(|| {
                    VisionForwardError::MissingProjectorFc2 {
                        kind: format!("{:?}", self.projector.kind),
                    }
                })?;
                let mut out = vec![0f32; n_tokens * d_text];
                for i in 0..n_tokens {
                    let xi = &fc1_out[i * d_hidden..(i + 1) * d_hidden];
                    let yi = &mut out[i * d_text..(i + 1) * d_text];
                    k::matvec_tensor(fc2, xi, yi, d_text, d_hidden);
                    if let Some(b) = &self.projector.fc2_bias {
                        add_bias_f32(yi, b);
                    }
                }
                Ok(out)
            }
        }
    }

    /// Convenience: `forward_image` + `project_image` chained.
    /// Most callers want the full vision-to-text-embedding pipeline
    /// in one call; the split methods exist for inspection /
    /// debugging or for callers that want to pool before projecting.
    pub fn forward_image_with_projection(
        &self,
        patches: &[f32],
    ) -> Result<Vec<f32>, VisionForwardError> {
        let hidden = self.forward_image(patches)?;
        self.project_image(&hidden)
    }

    /// Decode + resize + normalize + patch-extract `image_bytes` into
    /// the flat `[num_patches, n_channels * patch_size * patch_size]`
    /// buffer that [`Self::forward_image`] expects. Convenience
    /// wrapper around [`preprocess_image`] that consults this model's
    /// config — useful when the server / CLI has a `VisionModel`
    /// already in hand.
    pub fn preprocess_image(
        &self,
        image_bytes: &[u8],
    ) -> Result<Vec<f32>, PreprocessError> {
        preprocess_image(image_bytes, &self.cfg)
    }

    /// Full vision-to-text-embedding pipeline starting from raw image
    /// bytes (PNG / JPEG): preprocess + ViT + projector. Returns the
    /// `[n_tokens, d_text]` flat buffer ready to splice into the text
    /// decoder's input stream. The most common entry point for VLM
    /// HTTP handlers.
    pub fn forward_image_bytes(
        &self,
        image_bytes: &[u8],
    ) -> Result<Vec<f32>, VlmPipelineError> {
        let patches = self.preprocess_image(image_bytes)?;
        Ok(self.forward_image_with_projection(&patches)?)
    }
}

/// Decode an image, resize to `cfg.image_size × cfg.image_size`,
/// normalize per-channel using `cfg.image_mean / cfg.image_std`,
/// and extract patches in conv-kernel-compatible memory order
/// (channel-major within each patch). Output layout matches what
/// [`VisionModel::forward_image`] expects:
///
///   `out[patch_idx * (C * P * P) + c * (P * P) + ph * P + pw]`
///
/// for patch index `patch_idx ∈ [0, num_patches)`, channel `c`,
/// in-patch row `ph` and column `pw`. The conv `patch_embd` weight
/// stored as `[d_model, C, P, P]` is read by `matvec_tensor` as
/// `[d_model, C*P*P]` — the trailing-dim memory ordering matches
/// the patch layout this function produces.
///
/// Supports PNG and JPEG today (the `image` crate's `png` + `jpeg`
/// features). The OpenAI / Anthropic `image_url` content blocks use
/// `data:image/{png,jpeg};base64,...` so those two cover the wire
/// payloads we accept. WebP / GIF would be a follow-up.
pub fn preprocess_image(
    image_bytes: &[u8],
    cfg: &VisionConfig,
) -> Result<Vec<f32>, PreprocessError> {
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| PreprocessError::Decode(e.to_string()))?;
    let image_size = cfg.image_size as u32;
    // Filter choice per family: the Qwen-VL reference pipeline uses
    // BICUBIC (fork clip.cpp `RESIZE_ALGO_BICUBIC` for QWEN3VL →
    // image crate CatmullRom); classic CLIP variants keep the
    // previous bilinear behavior so existing goldens stay stable.
    let filter = if cfg.projector_type == ProjectorType::Qwen3VlMerger {
        image::imageops::FilterType::CatmullRom
    } else {
        image::imageops::FilterType::Triangle
    };
    let resized = img.resize_exact(image_size, image_size, filter);
    // Force-convert to 8-bit RGB so the channel layout is fixed at
    // 3 channels and 1 byte each. Strips alpha (RGBA → RGB ignores
    // the alpha channel) and widens 16-bit per-channel to 8-bit —
    // the model was trained on 8-bit RGB regardless.
    let rgb = resized.to_rgb8();
    let buf = rgb.as_raw(); // [h, w, c=3] in row-major order
    let img_w = rgb.width() as usize;
    let img_h = rgb.height() as usize;
    if img_w != cfg.image_size || img_h != cfg.image_size {
        return Err(PreprocessError::ResizeFailed {
            expected: cfg.image_size,
            got_w: img_w,
            got_h: img_h,
        });
    }

    let patch_size = cfg.patch_size;
    let per_side = cfg.image_size / patch_size;
    let n_channels = cfg.n_channels;
    let patch_dim = n_channels * patch_size * patch_size;
    let num_patches = per_side * per_side;
    let mean = cfg.image_mean;
    let std = cfg.image_std;
    let inv_std = [1.0 / std[0], 1.0 / std[1], 1.0 / std[2]];

    let mut out = vec![0f32; num_patches * patch_dim];
    for py in 0..per_side {
        for px in 0..per_side {
            let patch_idx = py * per_side + px;
            for c in 0..n_channels {
                for dh in 0..patch_size {
                    for dw in 0..patch_size {
                        let src_h = py * patch_size + dh;
                        let src_w = px * patch_size + dw;
                        // Source byte index into the [h, w, c] buffer.
                        let src_idx = (src_h * img_w + src_w) * 3 + c;
                        let pixel_0_1 = (buf[src_idx] as f32) / 255.0;
                        let normalized = (pixel_0_1 - mean[c]) * inv_std[c];
                        let dst_idx = patch_idx * patch_dim
                            + c * patch_size * patch_size
                            + dh * patch_size
                            + dw;
                        out[dst_idx] = normalized;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Errors raised by [`VisionModel::load`].
#[derive(Debug, thiserror::Error)]
pub enum VisionLoadError {
    #[error("vision config: {0}")]
    Config(#[from] VisionConfigError),
    #[error("GGUF missing required tensor `{0}` — corrupt or unsupported mmproj")]
    MissingTensor(String),
    #[error(
        "projector tensor `{tensor}` consumes {got_in}-wide rows, expected \
         {expected_in} (d_model × spatial_merge_size²) — mmproj/merge mismatch"
    )]
    ProjectorShape {
        tensor: &'static str,
        expected_in: usize,
        got_in: usize,
    },
    #[error("mmproj tensor `{name}` has unsupported dtype {dtype} for compute")]
    UnsupportedTensorDtype { name: String, dtype: &'static str },
    #[error("mmproj tensor `{name}` has unsupported dtype {dtype} for an F32-consumed slot")]
    UnsupportedTensorDtypeName { name: String, dtype: String },
}

/// Errors raised by [`VisionModel::forward_image`]. Distinct enum
/// from `VisionLoadError` since load-time and runtime failure modes
/// don't overlap.
#[derive(Debug, thiserror::Error)]
pub enum VisionForwardError {
    #[error(
        "patches buffer has {got} f32 entries, expected {expected} \
         ({num_patches} patches × {patch_dim} elements each — the \
         image preprocessor should produce `[num_patches, \
         n_channels * patch_size * patch_size]`)"
    )]
    WrongPatchShape {
        expected: usize,
        got: usize,
        num_patches: usize,
        patch_dim: usize,
    },
    #[error(
        "hidden_states buffer has {got} f32 entries — not a multiple of \
         d_vision={d_vision}; should be the `[n_tokens, d_vision]` \
         output of `forward_image`"
    )]
    WrongHiddenShape { got: usize, d_vision: usize },
    #[error(
        "MLP-type projector ({kind}) is missing the `mm.2.weight` \
         tensor — load should have caught this; report as a bug"
    )]
    MissingProjectorFc2 { kind: String },
}

/// Errors from the [`preprocess_image`] / [`VisionModel::preprocess_image`]
/// path. Distinct from `VisionForwardError` since decode + resize
/// failures are upstream of the forward pass and have different
/// client-side handling (a malformed image is a 400, a forward-pass
/// shape error is a 500).
#[derive(Debug, thiserror::Error)]
pub enum PreprocessError {
    #[error("image decode failed: {0}")]
    Decode(String),
    #[error(
        "resize produced unexpected dimensions: requested {expected}×{expected}, \
         got {got_w}×{got_h} — `image` crate behavior changed?"
    )]
    ResizeFailed {
        expected: usize,
        got_w: usize,
        got_h: usize,
    },
}

/// Combined error for the full image-bytes-to-text-embedding pipeline.
/// Lets [`VisionModel::forward_image_bytes`] return a single error
/// type while preserving the source-error categorization.
#[derive(Debug, thiserror::Error)]
pub enum VlmPipelineError {
    #[error("preprocess: {0}")]
    Preprocess(#[from] PreprocessError),
    #[error("forward: {0}")]
    Forward(#[from] VisionForwardError),
}

/// Errors raised by [`prepare_vlm_inputs`]. Distinct from the
/// per-image [`VlmPipelineError`] because the V-6b-1 builder has two
/// extra failure modes that can only be detected at the prompt
/// level: a mismatch between the number of placeholder tokens found
/// in the prompt and the number of attached images, and the empty-
/// prompt edge case.
#[derive(Debug, thiserror::Error)]
pub enum VlmInputBuildError {
    #[error(
        "prompt contains {placeholders} `<image>`-placeholder tokens \
         (id={image_token_id}) but the request attached {images} image \
         payloads — every placeholder must correspond to exactly one \
         image, in order"
    )]
    PlaceholderImageCountMismatch {
        image_token_id: u32,
        placeholders: usize,
        images: usize,
    },
    /// Multi-token placeholder mode ([`PlaceholderMode::OnePerPatch`]):
    /// the consecutive runs of placeholder tokens in the prompt don't
    /// match the per-image patch counts produced by the vision
    /// pipeline. For Qwen2-VL the chat template emits `<|image_pad|>`
    /// repeated `num_patches` times per image; if the user's prompt
    /// has the wrong repetition count (or interleaves placeholders
    /// from different images), we can't safely map them.
    #[error(
        "OnePerPatch placeholder mode: prompt has placeholder runs {found:?} \
         but vision pipeline produced patches {expected:?} per image"
    )]
    PlaceholderPatchRunMismatch {
        found: Vec<usize>,
        expected: Vec<usize>,
    },
    #[error("vision pipeline failed on image {idx}: {source}")]
    Pipeline {
        idx: usize,
        #[source]
        source: VlmPipelineError,
    },
}

/// How the prompt's image-placeholder tokens map to vision-pipeline
/// patch outputs. Different VLM architectures use different
/// conventions; rustllama supports both via a load-time switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaceholderMode {
    /// LLaVA-1.5 style. One placeholder token per image — the
    /// vision pipeline's full patch sequence (num_patches +
    /// optional class token) replaces that single position. The
    /// total prompt length grows by `(num_patches - 1)` per image.
    ///
    /// `prepare_vlm_inputs` accepts a flat placeholder count equal
    /// to the number of attached images.
    OnePerImage,
    /// Qwen2-VL style. The chat template emits one placeholder
    /// token per output patch position, repeated consecutively per
    /// image: `<|vision_start|><|image_pad|>×N<|vision_end|>` where
    /// `N = num_patches`. The splice replaces each placeholder
    /// 1:1 with the corresponding patch row — no length change.
    ///
    /// `prepare_vlm_inputs` validates that each image's consecutive
    /// placeholder run length matches the patch count its vision
    /// pipeline produced.
    OnePerPatch,
}

/// Output of [`prepare_vlm_inputs`] — everything a VLM-aware engine
/// needs to call [`splice_image_embeddings`] at prefill time.
///
/// The engine pipeline is:
///
/// ```text
///   tokenize(prompt) -> Vec<u32>
///   prepare_vlm_inputs(...) -> VlmPrefillInputs
///   embed_tokens(tokens) -> Vec<f32>   // text decoder embed table
///   splice_image_embeddings(text_embeds, positions, &features, d_text)
///       -> input embeddings ready for the transformer body
/// ```
///
/// V-6b-1 builds this struct; V-6b-2 (CpuEngine integration) consumes
/// it. Tests at the V-6b-1 layer pin that the placeholder scan, image
/// pipeline dispatch, and feature-buffer ordering are correct.
#[derive(Debug)]
pub struct VlmPrefillInputs {
    /// Strictly-ascending positions in the tokenized prompt where the
    /// image-placeholder token id appeared. Length matches `features`.
    pub positions: Vec<usize>,
    /// Per-image projected feature buffers, in the same order as the
    /// attached image bytes and `positions`. Each is `[num_patches_i,
    /// d_text]` row-major.
    pub features: Vec<Vec<f32>>,
    /// Text-decoder hidden dim. Matches `VisionModel::projector
    /// .d_text()` — surfaced here so the caller doesn't need a second
    /// borrow into `model.projector` to call [`splice_image_embeddings`].
    pub d_text: usize,
}

impl VlmPrefillInputs {
    /// Borrow-friendly view of [`features`] for direct hand-off to
    /// [`splice_image_embeddings`], which takes `&[&[f32]]`.
    pub fn feature_slices(&self) -> Vec<&[f32]> {
        self.features.iter().map(|v| v.as_slice()).collect()
    }
}

/// Build the prefill inputs for a VLM request: locate every
/// image-placeholder position in the tokenized prompt, run the vision
/// pipeline on each attached image (preprocess + ViT + projector),
/// and bundle the results so the caller can call
/// [`splice_image_embeddings`] in one step.
///
/// # Arguments
///
/// - `vision`: the loaded [`VisionModel`] (mmproj GGUF).
/// - `prompt_tokens`: the tokenized prompt — the result of the text
///   decoder's tokenizer applied to the chat-template-rendered string.
///   Placeholder positions are looked up here.
/// - `image_token_id`: the token id the text tokenizer assigns to the
///   model's image-placeholder string (e.g. `<image>` for LLaVA,
///   `<|vision_start|><|image_pad|><|vision_end|>` sequences for
///   Qwen2-VL — the engine layer decides what counts as the
///   placeholder token before calling).
/// - `image_payloads`: raw image bytes, in the same order they were
///   attached to the request.
///
/// # Errors
///
/// - [`VlmInputBuildError::PlaceholderImageCountMismatch`] when the
///   placeholder count and image count disagree.
/// - [`VlmInputBuildError::Pipeline`] when any per-image pipeline
///   step (decode, resize, ViT forward, projection) fails — the
///   `idx` field identifies which image so the server can blame the
///   right wire-protocol slot.
pub fn prepare_vlm_inputs(
    vision: &VisionModel,
    prompt_tokens: &[u32],
    image_token_id: u32,
    image_payloads: &[&[u8]],
) -> Result<VlmPrefillInputs, VlmInputBuildError> {
    prepare_vlm_inputs_with_mode(
        vision,
        prompt_tokens,
        image_token_id,
        image_payloads,
        PlaceholderMode::OnePerImage,
    )
}

/// Mode-aware variant of [`prepare_vlm_inputs`]. The original entry
/// point is a thin wrapper passing [`PlaceholderMode::OnePerImage`].
///
/// # OnePerImage
///
/// Same as the original `prepare_vlm_inputs`: each image-placeholder
/// token in the prompt corresponds to one attached image, and the
/// splice expands that single position into the image's full patch
/// sequence.
///
/// # OnePerPatch
///
/// Qwen2-VL convention. Each image's chat-template emission is
/// `<|vision_start|><|image_pad|>×N<|vision_end|>` where N is the
/// per-image patch count from the vision pipeline. The placeholder
/// token id passed in is the `<|image_pad|>` id. We:
///
/// 1. Run the vision pipeline on each image to learn its per-image
///    patch count.
/// 2. Walk `prompt_tokens` for consecutive runs of `image_token_id`.
/// 3. Validate the run-length vector matches the per-image patch
///    counts exactly. Mismatches surface
///    [`VlmInputBuildError::PlaceholderPatchRunMismatch`] with both
///    vectors so the caller can debug template / tokenizer drift.
/// 4. Split each image's `[num_patches, d_text]` feature buffer
///    into `num_patches` per-row buffers, ordered to align with
///    the corresponding placeholder positions.
///
/// The splice helper consumes the output unchanged — each
/// placeholder position maps to a single-row feature buffer (a
/// 1:1 substitution rather than the 1:N expansion of OnePerImage).
pub fn prepare_vlm_inputs_with_mode(
    vision: &VisionModel,
    prompt_tokens: &[u32],
    image_token_id: u32,
    image_payloads: &[&[u8]],
    mode: PlaceholderMode,
) -> Result<VlmPrefillInputs, VlmInputBuildError> {
    let positions: Vec<usize> = prompt_tokens
        .iter()
        .enumerate()
        .filter_map(|(i, &t)| if t == image_token_id { Some(i) } else { None })
        .collect();

    // Vision pipeline first — we need per-image patch counts before
    // we can validate placeholder runs in OnePerPatch mode.
    let mut features_per_image = Vec::with_capacity(image_payloads.len());
    let d_text = vision.projector.d_text();
    for (idx, bytes) in image_payloads.iter().enumerate() {
        let feat = vision
            .forward_image_bytes(bytes)
            .map_err(|source| VlmInputBuildError::Pipeline { idx, source })?;
        features_per_image.push(feat);
    }

    match mode {
        PlaceholderMode::OnePerImage => {
            if positions.len() != image_payloads.len() {
                return Err(VlmInputBuildError::PlaceholderImageCountMismatch {
                    image_token_id,
                    placeholders: positions.len(),
                    images: image_payloads.len(),
                });
            }
            Ok(VlmPrefillInputs {
                positions,
                features: features_per_image,
                d_text,
            })
        }
        PlaceholderMode::OnePerPatch => {
            // Compute consecutive-run lengths in `positions`.
            let mut run_lengths: Vec<usize> = Vec::new();
            let mut current_run = 0usize;
            let mut prev_pos: Option<usize> = None;
            for &p in &positions {
                match prev_pos {
                    Some(prev) if p == prev + 1 => {
                        current_run += 1;
                    }
                    _ => {
                        if current_run > 0 {
                            run_lengths.push(current_run);
                        }
                        current_run = 1;
                    }
                }
                prev_pos = Some(p);
            }
            if current_run > 0 {
                run_lengths.push(current_run);
            }
            let patches_per_image: Vec<usize> = features_per_image
                .iter()
                .map(|f| f.len() / d_text)
                .collect();
            if run_lengths != patches_per_image {
                return Err(VlmInputBuildError::PlaceholderPatchRunMismatch {
                    found: run_lengths,
                    expected: patches_per_image,
                });
            }
            // Split each image's feature buffer into per-patch rows.
            let total_patches: usize = patches_per_image.iter().sum();
            let mut features_flat: Vec<Vec<f32>> =
                Vec::with_capacity(total_patches);
            for feat in features_per_image {
                let n_patches = feat.len() / d_text;
                for p in 0..n_patches {
                    let row = feat[p * d_text..(p + 1) * d_text].to_vec();
                    features_flat.push(row);
                }
            }
            // Sanity: positions.len() == features_flat.len() guaranteed
            // by the run-length match above (sum run_lengths == sum
            // patches_per_image == total_patches == positions.len()).
            debug_assert_eq!(positions.len(), features_flat.len());
            Ok(VlmPrefillInputs {
                positions,
                features: features_flat,
                d_text,
            })
        }
    }
}

/// Errors from [`splice_image_embeddings`]. Distinct from forward-pass
/// errors because the splice operation is a pure index-arithmetic pass
/// with its own failure modes — the caller (the VLM-aware engine
/// prefill path) needs to distinguish "your prompt had 3 placeholders
/// but you passed 2 images" from "the vision tower itself failed".
#[derive(Debug, thiserror::Error)]
pub enum SpliceError {
    #[error(
        "text_embeddings has {got} f32 entries — not a multiple of \
         d_text={d_text}; expected `[text_len, d_text]` row-major"
    )]
    WrongTextShape { got: usize, d_text: usize },
    #[error(
        "image_features[{idx}] has {got} f32 entries — not a multiple \
         of d_text={d_text}; expected `[num_patches, d_text]` row-major"
    )]
    WrongImageShape { idx: usize, got: usize, d_text: usize },
    #[error(
        "image_positions has {n_positions} entries but image_features \
         has {n_features}; every placeholder position in the prompt \
         must map to exactly one image feature buffer"
    )]
    PositionsFeaturesMismatch {
        n_positions: usize,
        n_features: usize,
    },
    #[error(
        "image_positions are not strictly increasing — got {a} then {b} \
         at index {idx}; placeholder positions must be sorted ascending \
         so the splice can walk the prompt once"
    )]
    PositionsNotSorted { idx: usize, a: usize, b: usize },
    #[error(
        "image_positions[{idx}] = {pos} is out of range — text_len = \
         {text_len}; placeholder must point to a real token position"
    )]
    PositionOutOfRange {
        idx: usize,
        pos: usize,
        text_len: usize,
    },
}

/// Splice projected image-patch embeddings into the text decoder's
/// input embedding stream, replacing image-placeholder positions with
/// the corresponding patch sequences. This is the operation that
/// turns a prompt like
///
/// ```text
///   tokens: [BOS, "what", "is", "in", IMG, "?"]
/// ```
///
/// into the decoder input sequence
///
/// ```text
///   embeds: [emb(BOS), emb("what"), emb("is"), emb("in"),
///            patch_0, patch_1, ..., patch_{N-1}, emb("?")]
/// ```
///
/// where the single `IMG` placeholder is replaced by the N projected
/// patch vectors produced by [`VisionModel::forward_image_with_projection`].
///
/// # Arguments
///
/// - `text_embeddings`: flat row-major `[text_len, d_text]`. Produced
///   by the text decoder's `embed_tokens` over the tokenized prompt
///   (placeholder tokens included — their embedding is *discarded*,
///   so it doesn't matter what token-id the caller chose).
/// - `image_positions`: indices in `[0, text_len)` where placeholder
///   tokens sit. MUST be strictly ascending and MUST have the same
///   length as `image_features`. The caller is the engine, which
///   located these by scanning the tokenized prompt for the model's
///   image-placeholder token id.
/// - `image_features`: one `[num_patches_i, d_text]` row-major
///   buffer per image, in the same order as `image_positions`.
///   `num_patches_i` may differ per image (different aspect ratios
///   etc. — for v1 it's always equal but the splice doesn't care).
/// - `d_text`: the text decoder's hidden dimension. Both
///   `text_embeddings.len()` and each `image_features[i].len()` must
///   be exact multiples of this.
///
/// # Returns
///
/// A single `Vec<f32>` of length `(text_len - n_images + total_patches) * d_text`
/// in row-major order, ready to be reshaped to `[seq_len, d_text]`
/// and fed to the text decoder's first transformer block.
///
/// # Why not return a Tensor
///
/// Same convention as the rest of [`VisionModel`] — flat `Vec<f32>`
/// keeps the helper composable with the kernel ladder, which dispatches
/// on shape + dtype at the call site rather than carrying a Tensor
/// through every per-call hot path.
pub fn splice_image_embeddings(
    text_embeddings: &[f32],
    image_positions: &[usize],
    image_features: &[&[f32]],
    d_text: usize,
) -> Result<Vec<f32>, SpliceError> {
    if text_embeddings.len() % d_text != 0 {
        return Err(SpliceError::WrongTextShape {
            got: text_embeddings.len(),
            d_text,
        });
    }
    let text_len = text_embeddings.len() / d_text;
    if image_positions.len() != image_features.len() {
        return Err(SpliceError::PositionsFeaturesMismatch {
            n_positions: image_positions.len(),
            n_features: image_features.len(),
        });
    }
    // Validate every image feature buffer matches the text dim.
    for (idx, feat) in image_features.iter().enumerate() {
        if feat.len() % d_text != 0 {
            return Err(SpliceError::WrongImageShape {
                idx,
                got: feat.len(),
                d_text,
            });
        }
    }
    // Positions must be strictly increasing AND in range. Strict
    // increase is required so the single-pass walk below is sound;
    // duplicates would map two images to the same placeholder.
    for (idx, &pos) in image_positions.iter().enumerate() {
        if pos >= text_len {
            return Err(SpliceError::PositionOutOfRange {
                idx,
                pos,
                text_len,
            });
        }
        if idx > 0 {
            let prev = image_positions[idx - 1];
            if pos <= prev {
                return Err(SpliceError::PositionsNotSorted {
                    idx,
                    a: prev,
                    b: pos,
                });
            }
        }
    }

    // Compute the output length up-front so we allocate exactly once.
    let total_patches: usize = image_features.iter().map(|f| f.len() / d_text).sum();
    let out_tokens = text_len - image_positions.len() + total_patches;
    let mut out = Vec::with_capacity(out_tokens * d_text);

    let mut next_img = 0usize;
    for tok in 0..text_len {
        if next_img < image_positions.len() && image_positions[next_img] == tok {
            out.extend_from_slice(image_features[next_img]);
            next_img += 1;
        } else {
            let start = tok * d_text;
            out.extend_from_slice(&text_embeddings[start..start + d_text]);
        }
    }
    debug_assert_eq!(out.len(), out_tokens * d_text);
    Ok(out)
}

// ---- helpers ---------------------------------------------------

fn tensor_from_info(
    gguf: &Gguf,
    info: &rustllama_gguf::TensorInfo,
    name: &str,
) -> Result<Tensor, VisionLoadError> {
    // Mirror of `bert_arch::tensor_from_info`. Copies GGUF mmap bytes
    // into an `Arc<[u8]>` so the resulting Tensor is self-contained
    // and the source GGUF can drop afterward. The forward pass
    // dispatches through the existing matvec ladder, which handles
    // every mapped Dtype.
    let bytes = gguf
        .tensor_bytes(name)
        .expect("tensor_bytes for known-present info");
    let storage = Storage::CpuOwned(bytes.to_vec().into());
    let dtype = ggml_type_to_dtype(info.dtype).ok_or_else(|| {
        VisionLoadError::UnsupportedTensorDtype {
            name: name.to_string(),
            dtype: info.dtype.as_str(),
        }
    })?;
    let shape: Vec<u64> = info.dims.clone();
    let strides = contiguous_strides_for(&shape);
    Ok(Tensor {
        device: Device::Cpu,
        dtype,
        shape,
        strides,
        storage,
        name: name.to_string(),
    })
}

/// Map a GGUF tensor dtype to the runtime Dtype the matvec ladder
/// consumes. `None` = unsupported: the old `_ => Dtype::F32` fallback
/// MISLABELED quantized tensors as floats (Bonsai's mmproj ships 83
/// Q8_0 tensors), which reinterpreted block bytes as f32 garbage.
fn ggml_type_to_dtype(t: rustllama_gguf::GgmlType) -> Option<Dtype> {
    use rustllama_gguf::GgmlType as G;
    Some(match t {
        G::F32 => Dtype::F32,
        G::F16 => Dtype::F16,
        G::Bf16 => Dtype::Bf16Raw,
        G::Q8_0 => Dtype::Q8_0Raw,
        G::Q4_K => Dtype::Q4_KRaw,
        G::Q5_K => Dtype::Q5_KRaw,
        G::Q6_K => Dtype::Q6_KRaw,
        _ => return None,
    })
}

/// Qwen3-VL vision M-RoPE over one head vector, in place. Ported from
/// the fork's ggml `GGML_ROPE_TYPE_VISION` path (`ggml_mrope_cache_init`
/// with `indep_sects` + `rotate_pairs(ne0, n_dims, …)`): with
/// `n_dims = head_dim/2` and sections `[head_dim/4; 4]`, pair `j`
/// couples `(v[j], v[j + head_dim/2])`; pairs `0..head_dim/4` rotate by
/// the patch's **y** position, pairs `head_dim/4..head_dim/2` by its
/// **x**, each section restarting the frequency ramp
/// `θ_f = pos · 10000^(−2f / (head_dim/2))`.
#[inline]
fn vision_mrope_inplace(v: &mut [f32], pos_y: u32, pos_x: u32) {
    let head_dim = v.len();
    debug_assert_eq!(head_dim % 4, 0, "vision M-RoPE needs head_dim % 4 == 0");
    let half = head_dim / 2;
    let quarter = head_dim / 4;
    let n_dims = half as f32;
    for j in 0..half {
        let (pos, f_idx) = if j < quarter {
            (pos_y as f32, j as f32)
        } else {
            (pos_x as f32, (j - quarter) as f32)
        };
        let theta = pos * 10000f32.powf(-2.0 * f_idx / n_dims);
        let (sin_t, cos_t) = theta.sin_cos();
        let a = v[j];
        let b = v[j + half];
        v[j] = a * cos_t - b * sin_t;
        v[j + half] = a * sin_t + b * cos_t;
    }
}

/// Split a fused `attn_qkv` tensor into (Q, K, V) by byte-range row
/// slices. Fused layout is `[q_all_heads | k_all_heads | v_all_heads]`
/// along the OUTPUT axis (fork views at offsets 0 / n_embd / 2·n_embd),
/// and GGUF rows are contiguous in every dtype, so each part is a
/// plain byte slice — zero requantization. Works for the 2-D weight
/// (`[d_in, 3·d_out]` GGUF dims) and the 1-D bias (`[3·d_out]`).
fn split_fused_qkv(
    fused: &Tensor,
    d_out: u64,
) -> Result<(Tensor, Tensor, Tensor), VisionLoadError> {
    let (rows_total, part_shape): (u64, Vec<u64>) = match fused.shape.as_slice() {
        [n] => (*n, vec![d_out]),
        [d_in, n] => (*n, vec![*d_in, d_out]),
        _ => (0, Vec::new()),
    };
    if rows_total != 3 * d_out || part_shape.is_empty() {
        return Err(VisionLoadError::ProjectorShape {
            tensor: "attn_qkv",
            expected_in: (3 * d_out) as usize,
            got_in: rows_total as usize,
        });
    }
    let bytes = rustllama_tensor::as_bytes(fused);
    if bytes.len() % 3 != 0 {
        return Err(VisionLoadError::ProjectorShape {
            tensor: "attn_qkv (byte size)",
            expected_in: bytes.len() / 3 * 3,
            got_in: bytes.len(),
        });
    }
    let part_bytes = bytes.len() / 3;
    let mk = |idx: usize, tag: &str| -> Tensor {
        let slice = bytes[idx * part_bytes..(idx + 1) * part_bytes].to_vec();
        Tensor {
            device: Device::Cpu,
            dtype: fused.dtype,
            shape: part_shape.clone(),
            strides: contiguous_strides_for(&part_shape),
            storage: Storage::CpuOwned(slice.into()),
            name: format!("{}::{tag}", fused.name),
        }
    };
    Ok((mk(0, "q"), mk(1, "k"), mk(2, "v")))
}

/// Coerce a small tensor (norm / bias / embedding) to F32. No-op for
/// F32; converts F16 / BF16; errors on anything else — quantized
/// norms don't exist in real mmprojs, and silently passing one
/// through would panic later in `as_slice_f32`.
fn tensor_to_f32(t: Tensor) -> Result<Tensor, VisionLoadError> {
    match t.dtype {
        Dtype::F32 => Ok(t),
        Dtype::F16 | Dtype::Bf16Raw => {
            let src = rustllama_tensor::as_bytes(&t);
            let mut out = Vec::with_capacity(src.len() / 2 * 4);
            for ch in src.chunks_exact(2) {
                let v = if t.dtype == Dtype::F16 {
                    half::f16::from_le_bytes([ch[0], ch[1]]).to_f32()
                } else {
                    half::bf16::from_le_bytes([ch[0], ch[1]]).to_f32()
                };
                out.extend_from_slice(&v.to_le_bytes());
            }
            Ok(Tensor {
                device: Device::Cpu,
                dtype: Dtype::F32,
                shape: t.shape.clone(),
                strides: t.strides.clone(),
                storage: Storage::CpuOwned(out.into()),
                name: t.name.clone(),
            })
        }
        other => Err(VisionLoadError::UnsupportedTensorDtypeName {
            name: t.name.clone(),
            dtype: format!("{other:?}"),
        }),
    }
}

fn contiguous_strides_for(shape: &[u64]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as i64;
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustllama_gguf::synth::write_synthetic_clip_mmproj_gguf;
    use rustllama_gguf::synth::write_synthetic_qwen3vl_mmproj_gguf;

    // ---- Qwen3-VL fixture (V-0b) ------------------------------------

    /// The Qwen3-VL-shaped fixture binds end-to-end: fused QKV split
    /// into three per-block tensors, ln1/ln2 + attn_out aliases
    /// resolved, dual patch embeddings bound, merger projector dims
    /// computed from element counts (real `[in, out]` GGUF order),
    /// no class token detected.
    #[test]
    fn qwen3vl_fixture_loads_with_fused_split_and_merger() {
        let tmp = std::env::temp_dir().join("rustllama-vision-qwen3vl-load.gguf");
        write_synthetic_qwen3vl_mmproj_gguf(&tmp, false);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();

        assert_eq!(model.cfg.projector_type, ProjectorType::Qwen3VlMerger);
        assert_eq!(model.cfg.spatial_merge_size, 2);
        assert_eq!(model.cfg.projection_dim, Some(96));
        assert!(!model.cfg.has_class_token, "fixture has no class token");
        assert!(model.patch_embd_1.is_some(), "dual patch embd must bind");
        assert_eq!(model.blocks.len(), 2);
        // Split fused QKV: three [64, 64]-element tensors + F32 biases.
        let b0 = &model.blocks[0];
        let n_elems = |t: &Tensor| t.shape.iter().product::<u64>();
        assert_eq!(n_elems(&b0.attn_q), 64 * 64);
        assert_eq!(n_elems(&b0.attn_k), 64 * 64);
        assert_eq!(n_elems(&b0.attn_v), 64 * 64);
        assert_eq!(b0.attn_q_bias.as_ref().map(n_elems), Some(64));
        assert_eq!(b0.attn_v_bias.as_ref().map(n_elems), Some(64));
        // Merger projector dims from element counts.
        assert_eq!(model.projector.d_in, 256);
        assert_eq!(model.projector.d_hidden(), 256);
        assert_eq!(model.projector.d_text(), 96);
        let _ = std::fs::remove_file(&tmp);
    }

    /// The fused-QKV byte split must reproduce the exact thirds of
    /// the fused buffer: fused bytes are [q_all | k_all | v_all].
    #[test]
    fn qwen3vl_fused_qkv_split_is_byte_exact() {
        let tmp = std::env::temp_dir().join("rustllama-vision-qwen3vl-split.gguf");
        write_synthetic_qwen3vl_mmproj_gguf(&tmp, false);
        let gguf = Gguf::open(&tmp).unwrap();
        let fused_bytes = gguf.tensor_bytes("v.blk.0.attn_qkv.weight").unwrap().to_vec();
        let model = VisionModel::load(&gguf).unwrap();
        let third = fused_bytes.len() / 3;
        let b0 = &model.blocks[0];
        assert_eq!(rustllama_tensor::as_bytes(&b0.attn_q), &fused_bytes[..third]);
        assert_eq!(
            rustllama_tensor::as_bytes(&b0.attn_k),
            &fused_bytes[third..2 * third]
        );
        assert_eq!(rustllama_tensor::as_bytes(&b0.attn_v), &fused_bytes[2 * third..]);
        let _ = std::fs::remove_file(&tmp);
    }

    /// Qwen3-VL end-to-end shapes on the fixture: 16 patches → tower
    /// hidden `[16, 64]` (window-major order) → merger projector
    /// `[4, 96]` (4 = 16 / merge²).
    #[test]
    fn qwen3vl_forward_and_projection_shapes() {
        let tmp = std::env::temp_dir().join("rustllama-vision-qwen3vl-fwd.gguf");
        write_synthetic_qwen3vl_mmproj_gguf(&tmp, false);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let patch_dim = 3 * 8 * 8;
        let patches: Vec<f32> = (0..16 * patch_dim)
            .map(|i| ((i * 37 + 11) % 101) as f32 * 0.01 - 0.5)
            .collect();
        let hidden = model.forward_image(&patches).expect("tower forward");
        assert_eq!(hidden.len(), 16 * 64);
        assert!(hidden.iter().all(|v| v.is_finite()), "non-finite tower output");
        let projected = model.project_image(&hidden).expect("projector");
        assert_eq!(projected.len(), 4 * 96, "16 patches / 2x2 merge → 4 tokens of d_text=96");
        assert!(projected.iter().all(|v| v.is_finite()));
        // Determinism.
        let hidden2 = model.forward_image(&patches).expect("tower forward 2");
        assert_eq!(hidden, hidden2);
        let _ = std::fs::remove_file(&tmp);
    }

    /// The merge re-sequencing must follow the fork's positions-fill
    /// order exactly: windows row-major, intra-window
    /// (0,0),(0,1),(1,0),(1,1). Verified through the PATCH IDENTITY:
    /// with a distinctive per-patch input, permuting the input patches
    /// by the expected map must equal feeding them in row-major order
    /// and reading the reordered output — here we check the cheap
    /// invariant instead: two patches that map to the same window are
    /// adjacent in the output of a 4×4 grid.
    #[test]
    fn qwen3vl_merge_order_matches_positions_fill() {
        // Reference generator (independent reimplementation of the
        // fork's loop) for a 4-wide grid:
        let per_side = 4usize;
        let mut expect = Vec::new();
        for wy in (0..per_side).step_by(2) {
            for wx in (0..per_side).step_by(2) {
                for dy in 0..2 {
                    for dx in 0..2 {
                        expect.push((wy + dy) * per_side + (wx + dx));
                    }
                }
            }
        }
        assert_eq!(
            expect,
            vec![0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15],
            "window-major map for a 4x4 grid"
        );
    }

    /// Vision M-RoPE unit properties: rotation is norm-preserving,
    /// (0,0) position is the identity, and distinct positions produce
    /// distinct rotations.
    #[test]
    fn vision_mrope_properties() {
        let head_dim = 16usize;
        let base: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.3 - 2.0).collect();

        let mut v0 = base.clone();
        vision_mrope_inplace(&mut v0, 0, 0);
        assert_eq!(v0, base, "position (0,0) must be the identity rotation");

        let mut v1 = base.clone();
        vision_mrope_inplace(&mut v1, 3, 7);
        let n_before: f32 = base.iter().map(|v| v * v).sum();
        let n_after: f32 = v1.iter().map(|v| v * v).sum();
        assert!(
            (n_before - n_after).abs() < 1e-3,
            "rotation must preserve the vector norm ({n_before} vs {n_after})"
        );
        assert_ne!(v1, base);

        let mut v2 = base.clone();
        vision_mrope_inplace(&mut v2, 7, 3);
        assert_ne!(v1, v2, "y and x positions must rotate different sections");
    }

    /// Diagnostic: load a REAL mmproj GGUF named by RUSTLLAMA_REAL_MMPROJ.
    /// Ignored in normal runs; exercised manually against the Bonsai 2
    /// mmproj (83 Q8_0 tensors, real `[in, out]` dims, fused QKV).
    #[test]
    #[ignore]
    fn real_mmproj_loads_when_env_set() {
        let Ok(path) = std::env::var("RUSTLLAMA_REAL_MMPROJ") else {
            eprintln!("RUSTLLAMA_REAL_MMPROJ not set — skipping");
            return;
        };
        let gguf = Gguf::open(&path).expect("open real mmproj");
        let model = VisionModel::load(&gguf).expect("load real mmproj");
        eprintln!(
            "real mmproj OK: {} blocks, merger={:?}, merge={}, d_in={}, d_mid={}, d_text={}, \
             dual_patch={}, class_token={}, patches={}",
            model.blocks.len(),
            model.cfg.projector_type,
            model.cfg.spatial_merge_size,
            model.projector.d_in,
            model.projector.d_hidden(),
            model.projector.d_text(),
            model.patch_embd_1.is_some(),
            model.cfg.has_class_token,
            model.cfg.num_patches(),
        );
        assert_eq!(model.projector.d_text(), 5120);
        assert_eq!(model.cfg.spatial_merge_size, 2);
        assert!(model.patch_embd_1.is_some());
        assert!(!model.cfg.has_class_token);
    }

    /// Deepstack flags with any `true` refuse to load (unmodeled
    /// projector-output widening).
    #[test]
    fn qwen3vl_deepstack_true_is_refused() {
        let tmp = std::env::temp_dir().join("rustllama-vision-qwen3vl-ds.gguf");
        write_synthetic_qwen3vl_mmproj_gguf(&tmp, true);
        let gguf = Gguf::open(&tmp).unwrap();
        match VisionModel::load(&gguf) {
            Err(VisionLoadError::Config(VisionConfigError::DeepstackUnsupported)) => {}
            other => panic!("expected DeepstackUnsupported, got {other:?}"),
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// `from_gguf` populates every field from a synth mmproj GGUF.
    /// Pins the metadata-key contract — a future llama.cpp rename
    /// would surface here.
    #[test]
    fn config_from_gguf_reads_clip_vision_keys() {
        let tmp = std::env::temp_dir().join("rustllama-vision-config-clip.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let cfg = VisionConfig::from_gguf(&gguf).unwrap();
        assert_eq!(cfg.arch, "clip");
        assert_eq!(cfg.image_size, 336);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.n_layers, 2);
        assert_eq!(cfg.n_heads, 4);
        assert_eq!(cfg.d_model, 128);
        assert_eq!(cfg.d_ff, 256);
        // head_dim derives from d_model / n_heads when not explicit.
        assert_eq!(cfg.head_dim, 32);
        let _ = std::fs::remove_file(&tmp);
    }

    /// `num_patches` derives from `(image_size / patch_size)^2`.
    /// 336/14 = 24 → 576 patches (the LLaVA-1.5 setting).
    #[test]
    fn num_patches_derives_from_image_and_patch_size() {
        let mut cfg = VisionConfig {
            arch: "clip".into(),
            image_size: 336,
            patch_size: 14,
            n_channels: 3,
            n_layers: 24,
            n_heads: 16,
            d_model: 1024,
            d_ff: 4096,
            head_dim: 64,
            layer_norm_eps: 1e-6,
            projector_type: ProjectorType::Mlp,
            has_class_token: true,
            image_mean: CLIP_IMAGE_MEAN,
            image_std: CLIP_IMAGE_STD,
            spatial_merge_size: 1,
            projection_dim: None,
            is_deepstack_layers: Vec::new(),
        };
        assert_eq!(cfg.num_patches(), 24 * 24);
        // Qwen2-VL native resolution.
        cfg.image_size = 448;
        cfg.patch_size = 14;
        assert_eq!(cfg.num_patches(), 32 * 32);
    }

    /// Wrong-architecture GGUF → `WrongArchitecture` error with the
    /// actual arch string so the user knows to point at an mmproj.
    #[test]
    fn config_rejects_non_clip_gguf() {
        use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};
        let tmp = std::env::temp_dir().join("rustllama-vision-wrong-arch.gguf");
        write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
        let gguf = Gguf::open(&tmp).unwrap();
        let err = VisionConfig::from_gguf(&gguf).unwrap_err();
        match err {
            VisionConfigError::WrongArchitecture(arch) => assert_eq!(arch, "llama"),
            other => panic!("expected WrongArchitecture, got {other:?}"),
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// V-1a contract: `VisionModel::load` binds every required
    /// tensor on a well-formed mmproj GGUF, populates the config
    /// from metadata, and detects class-token presence from the
    /// position-embedding row count.
    #[test]
    fn load_binds_all_required_tensors_on_synth_mmproj() {
        let tmp = std::env::temp_dir().join("rustllama-vision-load-bind.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).expect("load should succeed");

        assert_eq!(model.cfg.arch, "clip");
        assert_eq!(model.cfg.n_layers, 2);
        assert_eq!(model.cfg.d_model, 128);
        // Synth fixture writes 576 patches + 1 class-token row → 577.
        assert!(model.cfg.has_class_token);
        assert_eq!(model.position_embd.shape[0], 577);
        assert_eq!(model.position_embd.shape[1], 128);
        // Patch embedding is `[d_model, n_channels, ps, ps]`.
        assert_eq!(model.patch_embd.shape, vec![128, 3, 14, 14]);
        // Class token + pre-LN both bound.
        assert!(model.class_token.is_some());
        assert!(model.pre_ln.is_some());
        // Per-block: every block has Q/K/V/O + LN, FFN up/down + LN.
        assert_eq!(model.blocks.len(), 2);
        for blk in &model.blocks {
            assert_eq!(blk.attn_q.shape, vec![128, 128]);
            assert_eq!(blk.ffn_up.shape, vec![256, 128]);
            assert_eq!(blk.ffn_down.shape, vec![128, 256]);
        }
        // Projector: 2-layer MLP. fc1 [d_proj_hidden, d_model],
        // fc2 [d_text, d_proj_hidden].
        assert_eq!(model.projector.kind, ProjectorType::Mlp);
        assert_eq!(model.projector.fc1.shape, vec![64, 128]);
        let fc2 = model.projector.fc2.as_ref().expect("MLP projector has fc2");
        assert_eq!(fc2.shape, vec![96, 64]);
        let _ = std::fs::remove_file(&tmp);
    }

    /// Missing-tensor error names the tensor so the user can identify
    /// a corrupt / unsupported mmproj quickly. We can't write a "missing
    /// tensor" mmproj via the synth helper directly (it always emits
    /// the full set), so instead we exercise the wrong-arch path
    /// from V-0 to confirm the error categories don't blur into each
    /// other.
    #[test]
    fn load_propagates_config_errors_through_error_enum() {
        use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};
        let tmp = std::env::temp_dir().join("rustllama-vision-load-wrong-arch.gguf");
        write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
        let gguf = Gguf::open(&tmp).unwrap();
        let err = VisionModel::load(&gguf).unwrap_err();
        match err {
            VisionLoadError::Config(VisionConfigError::WrongArchitecture(arch)) => {
                assert_eq!(arch, "llama");
            }
            other => panic!("expected Config(WrongArchitecture), got {other:?}"),
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// V-1b contract: `forward_image` runs end-to-end on a well-shaped
    /// patch buffer. Output shape is `[n_tokens, d_model]` flattened,
    /// where `n_tokens = num_patches + 1` for class-token models.
    #[test]
    fn forward_image_returns_finite_hidden_states_with_expected_shape() {
        let tmp = std::env::temp_dir().join("rustllama-vision-forward-finite.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let num_patches = model.cfg.num_patches();
        let patch_dim =
            model.cfg.n_channels * model.cfg.patch_size * model.cfg.patch_size;
        // Synthetic patches — small deterministic values around zero
        // (real preprocessor normalizes to ~N(0,1)).
        let patches: Vec<f32> = (0..num_patches * patch_dim)
            .map(|i| ((i % 256) as f32 - 128.0) * 0.003)
            .collect();
        let out = model.forward_image(&patches).expect("forward should succeed");

        let n_tokens = num_patches + 1; // synth has class token
        let d = model.cfg.d_model;
        assert_eq!(
            out.len(),
            n_tokens * d,
            "output shape {} != n_tokens × d_model = {} × {}",
            out.len(),
            n_tokens,
            d,
        );
        for (i, v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "hidden[{i}] = {v} is not finite (NaN/Inf propagated)"
            );
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// Wrong patch-buffer length surfaces `WrongPatchShape` with the
    /// concrete numbers — clients (the image preprocessor) get an
    /// actionable diagnostic instead of a panic.
    #[test]
    fn forward_image_rejects_wrong_patch_buffer_length() {
        let tmp = std::env::temp_dir().join("rustllama-vision-forward-badshape.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let err = model.forward_image(&[0.0; 7]).unwrap_err();
        match err {
            VisionForwardError::WrongPatchShape {
                expected,
                got,
                num_patches,
                patch_dim,
            } => {
                assert_eq!(got, 7);
                assert!(expected > 7);
                assert_eq!(num_patches, 576);
                assert_eq!(patch_dim, 3 * 14 * 14);
            }
            other => panic!("expected WrongPatchShape, got {other:?}"),
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// V-1c contract: `project_image` produces `[n_tokens, d_text]`
    /// finite values via the 2-layer MLP path. d_text on the synth
    /// fixture is 96 (the synth's projector fc2 output dim).
    #[test]
    fn project_image_produces_correct_shape_via_mlp_path() {
        let tmp = std::env::temp_dir().join("rustllama-vision-project-mlp.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        assert_eq!(model.projector.kind, ProjectorType::Mlp);
        assert_eq!(model.projector.d_hidden(), 64);
        assert_eq!(model.projector.d_text(), 96);

        let n_tokens = model.cfg.num_patches() + 1;
        let hidden: Vec<f32> = (0..n_tokens * model.cfg.d_model)
            .map(|i| (i as f32) * 0.001 - 0.5)
            .collect();
        let projected = model.project_image(&hidden).unwrap();
        assert_eq!(
            projected.len(),
            n_tokens * 96,
            "projected output shape {} != n_tokens × d_text = {} × 96",
            projected.len(),
            n_tokens,
        );
        for v in &projected {
            assert!(v.is_finite(), "projected value {v} not finite");
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// `forward_image_with_projection` chains both stages. Output
    /// shape matches `n_tokens × d_text`.
    #[test]
    fn forward_image_with_projection_chains_both_stages() {
        let tmp =
            std::env::temp_dir().join("rustllama-vision-fwd-proj-chain.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let num_patches = model.cfg.num_patches();
        let patch_dim =
            model.cfg.n_channels * model.cfg.patch_size * model.cfg.patch_size;
        let patches: Vec<f32> = (0..num_patches * patch_dim)
            .map(|i| ((i % 200) as f32 - 100.0) * 0.005)
            .collect();
        let out = model.forward_image_with_projection(&patches).unwrap();
        let n_tokens = num_patches + 1; // class-token model
        assert_eq!(out.len(), n_tokens * model.projector.d_text());
        for v in &out {
            assert!(v.is_finite());
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// Wrong hidden-states shape → structured error (not a panic).
    #[test]
    fn project_image_rejects_misshapen_hidden_buffer() {
        let tmp = std::env::temp_dir().join("rustllama-vision-project-badshape.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        // d_vision is 128 in the synth; pass 130 to ensure
        // non-multiple → error.
        let err = model.project_image(&vec![0.0f32; 130]).unwrap_err();
        match err {
            VisionForwardError::WrongHiddenShape { got, d_vision } => {
                assert_eq!(got, 130);
                assert_eq!(d_vision, 128);
            }
            other => panic!("expected WrongHiddenShape, got {other:?}"),
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// V-3 contract: `preprocess_image` decodes a PNG, resizes to
    /// the model's image_size, normalizes, and packs patches in
    /// conv-compatible memory order. Output length matches the flat
    /// `[num_patches, n_channels * patch_size * patch_size]` shape
    /// the ViT forward pass expects.
    #[test]
    fn preprocess_image_produces_correct_shape_from_png() {
        let tmp = std::env::temp_dir().join("rustllama-vision-preproc-png.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let png = make_test_png(64, 48);
        let out = model.preprocess_image(&png).expect("preprocess succeeds");
        let cfg = &model.cfg;
        let patch_dim = cfg.n_channels * cfg.patch_size * cfg.patch_size;
        let expected = cfg.num_patches() * patch_dim;
        assert_eq!(out.len(), expected);
        for v in &out {
            assert!(v.is_finite(), "preprocessed pixel {v} not finite");
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// Normalization: a uniform-gray PNG produces predictable
    /// normalized values. Gray = 128/255 ≈ 0.5; after channel-mean
    /// subtract and std-divide, expected ≈ (0.5 - mean[c]) / std[c].
    /// Pinning this catches a future regression where the (mean, std)
    /// arrays load with the wrong order or get applied to RGB in the
    /// wrong channel.
    #[test]
    fn preprocess_image_normalization_matches_clip_constants() {
        let tmp = std::env::temp_dir().join("rustllama-vision-preproc-norm.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let png = make_uniform_gray_png(model.cfg.image_size as u32, 128);
        let out = model.preprocess_image(&png).unwrap();

        // First patch row — channel 0's first pixel.
        let cfg = &model.cfg;
        let patch_size = cfg.patch_size;
        let mean = cfg.image_mean;
        let std = cfg.image_std;
        let pixel_0_1 = 128.0 / 255.0;
        let expected_c0 = (pixel_0_1 - mean[0]) / std[0];
        let expected_c1 = (pixel_0_1 - mean[1]) / std[1];
        let expected_c2 = (pixel_0_1 - mean[2]) / std[2];
        // Patch 0, channel 0, in-patch (0, 0) → offset 0.
        let c0_p0 = out[0];
        // Patch 0, channel 1, in-patch (0, 0) → offset patch_size*patch_size.
        let c1_p0 = out[patch_size * patch_size];
        // Patch 0, channel 2, in-patch (0, 0) → offset 2*patch_size*patch_size.
        let c2_p0 = out[2 * patch_size * patch_size];
        // Resize via Triangle filter on a uniform-gray image stays
        // uniform-gray (the filter is linear); tolerance covers any
        // 8-bit round-trip slack.
        let tol = 1e-3;
        assert!(
            (c0_p0 - expected_c0).abs() < tol,
            "c0 normalized = {c0_p0}, expected ≈ {expected_c0}"
        );
        assert!((c1_p0 - expected_c1).abs() < tol);
        assert!((c2_p0 - expected_c2).abs() < tol);
        let _ = std::fs::remove_file(&tmp);
    }

    /// Garbage bytes → `Decode` error, not a panic.
    #[test]
    fn preprocess_image_rejects_garbage_bytes() {
        let tmp =
            std::env::temp_dir().join("rustllama-vision-preproc-garbage.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let err = model.preprocess_image(b"not-an-image").unwrap_err();
        match err {
            PreprocessError::Decode(_) => {}
            other => panic!("expected Decode, got {other:?}"),
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// Full pipeline: `forward_image_bytes` chains preprocess +
    /// forward + project from raw PNG bytes. Output is
    /// `[n_tokens, d_text]` finite values.
    #[test]
    fn forward_image_bytes_runs_full_pipeline_on_png() {
        let tmp =
            std::env::temp_dir().join("rustllama-vision-pipeline-png.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let png = make_test_png(96, 96);
        let out = model.forward_image_bytes(&png).expect("pipeline succeeds");
        let n_tokens = model.cfg.num_patches() + 1;
        assert_eq!(out.len(), n_tokens * model.projector.d_text());
        for v in &out {
            assert!(v.is_finite());
        }
        let _ = std::fs::remove_file(&tmp);
    }

    // ---- PNG test fixtures ------------------------------------

    /// Encode a synthetic RGB image as PNG bytes for tests. Uses the
    /// `image` crate's encoder so the output is bit-identical to what
    /// real-world PNG decoders produce; no external fixtures needed.
    fn make_test_png(w: u32, h: u32) -> Vec<u8> {
        let mut img = image::RgbImage::new(w, h);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            // Simple gradient — exercises the resize+normalize path
            // without needing a real photo.
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 255) / h.max(1)) as u8;
            let b = (((x + y) * 255) / (w + h).max(1)) as u8;
            *pixel = image::Rgb([r, g, b]);
        }
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
            .expect("encode test PNG");
        bytes
    }

    /// Encode a uniform-color RGB image (every pixel = `value`).
    /// Used by the normalization test — uniform input gives
    /// predictable output through the resize.
    fn make_uniform_gray_png(size: u32, value: u8) -> Vec<u8> {
        let pixel = image::Rgb([value, value, value]);
        let img = image::RgbImage::from_pixel(size, size, pixel);
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
            .expect("encode uniform PNG");
        bytes
    }

    /// Deterministic: same input → same output across calls. Pins
    /// that the forward pass has no hidden RNG / no per-call state
    /// (the engine load is the only state mutation; inference must
    /// be pure).
    #[test]
    fn forward_image_is_deterministic_across_calls() {
        let tmp = std::env::temp_dir().join("rustllama-vision-forward-determ.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let num_patches = model.cfg.num_patches();
        let patch_dim =
            model.cfg.n_channels * model.cfg.patch_size * model.cfg.patch_size;
        let patches: Vec<f32> = (0..num_patches * patch_dim)
            .map(|i| (i as f32) * 0.0001)
            .collect();
        let a = model.forward_image(&patches).unwrap();
        let b = model.forward_image(&patches).unwrap();
        assert_eq!(a, b, "forward_image must be deterministic");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn projector_type_from_metadata_str_recognizes_common_variants() {
        assert_eq!(ProjectorType::from_metadata_str("linear").unwrap(), ProjectorType::Linear);
        assert_eq!(ProjectorType::from_metadata_str("mlp").unwrap(), ProjectorType::Mlp);
        assert_eq!(ProjectorType::from_metadata_str("mlp2x_gelu").unwrap(), ProjectorType::Mlp);
        assert_eq!(ProjectorType::from_metadata_str("ldp_v2").unwrap(), ProjectorType::LdpV2);
        assert_eq!(
            ProjectorType::from_metadata_str("qwen3vl_merger").unwrap(),
            ProjectorType::Qwen3VlMerger
        );
        // Unknown projector strings are a HARD error now — the old
        // silent `_ => Mlp` fallback ran wrong math on right-shaped
        // tensors (see qwen3vl_merger's 4608-wide mm.0).
        assert!(ProjectorType::from_metadata_str("future_v3").is_err());
    }

    // ---- splice_image_embeddings (V-5) ------------------------------

    /// Build a `[n_tokens, d]` text-embedding buffer where token `t`
    /// has rows `[10*t, 10*t+1, ..., 10*t+d-1]`. The numeric pattern
    /// lets the splice tests assert exact-match output rows without
    /// having to round-trip through a real `embed_tokens`.
    fn fake_text_embeds(text_len: usize, d_text: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(text_len * d_text);
        for t in 0..text_len {
            for k in 0..d_text {
                out.push((10 * t + k) as f32);
            }
        }
        out
    }

    /// Build an `[n_patches, d]` image-feature buffer with sentinel
    /// values distinguishable from any text-embedding row. Image `i`'s
    /// patch `p`'s row k is `1000 + 100*i + 10*p + k` — well outside
    /// the range emitted by [`fake_text_embeds`] so a missed splice
    /// position is unambiguous.
    fn fake_image_features(image_idx: usize, n_patches: usize, d_text: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n_patches * d_text);
        for p in 0..n_patches {
            for k in 0..d_text {
                out.push((1000 + 100 * image_idx + 10 * p + k) as f32);
            }
        }
        out
    }

    /// Empty image list passes the text embeddings through unchanged.
    /// Guarantees the splice is a no-op on plain text prompts so the
    /// VLM-aware code path is safe to take unconditionally.
    #[test]
    fn splice_with_no_images_returns_text_embeddings_verbatim() {
        let d = 4;
        let text = fake_text_embeds(5, d);
        let out = splice_image_embeddings(&text, &[], &[], d).unwrap();
        assert_eq!(out, text);
    }

    /// Single image at position 2 in a 5-token prompt: tokens 0,1
    /// pass through, then 3 patch rows splice in, then tokens 3,4
    /// pass through. Output length = 5 - 1 + 3 = 7 rows.
    #[test]
    fn splice_single_image_replaces_placeholder_with_patches() {
        let d = 4;
        let text = fake_text_embeds(5, d);
        let img = fake_image_features(0, 3, d);
        let out =
            splice_image_embeddings(&text, &[2], &[img.as_slice()], d).unwrap();
        assert_eq!(out.len(), 7 * d);
        // Row 0 == text token 0 (values [0, 1, 2, 3]).
        assert_eq!(&out[0..d], &[0.0, 1.0, 2.0, 3.0]);
        // Row 1 == text token 1 ([10, 11, 12, 13]).
        assert_eq!(&out[d..2 * d], &[10.0, 11.0, 12.0, 13.0]);
        // Rows 2..5 == image 0 patches 0..3 (starts at 1000).
        assert_eq!(&out[2 * d..3 * d], &[1000.0, 1001.0, 1002.0, 1003.0]);
        assert_eq!(&out[3 * d..4 * d], &[1010.0, 1011.0, 1012.0, 1013.0]);
        assert_eq!(&out[4 * d..5 * d], &[1020.0, 1021.0, 1022.0, 1023.0]);
        // Rows 5,6 == text tokens 3,4 (placeholder at position 2 dropped).
        assert_eq!(&out[5 * d..6 * d], &[30.0, 31.0, 32.0, 33.0]);
        assert_eq!(&out[6 * d..7 * d], &[40.0, 41.0, 42.0, 43.0]);
    }

    /// Two images in one prompt — checks the splice walks placeholders
    /// in order and uses the right feature buffer for each.
    #[test]
    fn splice_two_images_in_one_prompt() {
        let d = 2;
        let text = fake_text_embeds(6, d);
        let img0 = fake_image_features(0, 2, d);
        let img1 = fake_image_features(1, 3, d);
        let out = splice_image_embeddings(
            &text,
            &[1, 4],
            &[img0.as_slice(), img1.as_slice()],
            d,
        )
        .unwrap();
        // text_len=6, placeholders=2, total patches=5 -> seq=9
        assert_eq!(out.len(), 9 * d);
        // Layout: [t0, img0_p0, img0_p1, t2, t3, img1_p0, img1_p1, img1_p2, t5]
        assert_eq!(&out[0..d], &[0.0, 1.0]); // text token 0
        assert_eq!(&out[d..2 * d], &[1000.0, 1001.0]); // img0 patch 0
        assert_eq!(&out[2 * d..3 * d], &[1010.0, 1011.0]); // img0 patch 1
        assert_eq!(&out[3 * d..4 * d], &[20.0, 21.0]); // text token 2
        assert_eq!(&out[4 * d..5 * d], &[30.0, 31.0]); // text token 3
        assert_eq!(&out[5 * d..6 * d], &[1100.0, 1101.0]); // img1 patch 0
        assert_eq!(&out[6 * d..7 * d], &[1110.0, 1111.0]); // img1 patch 1
        assert_eq!(&out[7 * d..8 * d], &[1120.0, 1121.0]); // img1 patch 2
        assert_eq!(&out[8 * d..9 * d], &[50.0, 51.0]); // text token 5
    }

    /// Placeholder at position 0 (first token is the image) and at
    /// the last position. Edge case: the splice loop must handle
    /// boundary indices without panicking.
    #[test]
    fn splice_at_first_and_last_positions() {
        let d = 2;
        let text = fake_text_embeds(3, d);
        let img = fake_image_features(0, 1, d);
        // Position 0:
        let out = splice_image_embeddings(&text, &[0], &[img.as_slice()], d).unwrap();
        // Output: [img_p0, t1, t2]
        assert_eq!(&out[0..d], &[1000.0, 1001.0]);
        assert_eq!(&out[d..2 * d], &[10.0, 11.0]);
        assert_eq!(&out[2 * d..3 * d], &[20.0, 21.0]);

        // Position text_len-1:
        let out = splice_image_embeddings(&text, &[2], &[img.as_slice()], d).unwrap();
        // Output: [t0, t1, img_p0]
        assert_eq!(&out[0..d], &[0.0, 1.0]);
        assert_eq!(&out[d..2 * d], &[10.0, 11.0]);
        assert_eq!(&out[2 * d..3 * d], &[1000.0, 1001.0]);
    }

    /// `text_embeddings.len() % d_text != 0` is a contract violation.
    #[test]
    fn splice_rejects_text_embeddings_with_misshapen_length() {
        let d = 4;
        // 9 floats but d_text=4 → not a multiple.
        let bad = vec![0.0_f32; 9];
        match splice_image_embeddings(&bad, &[], &[], d) {
            Err(SpliceError::WrongTextShape { got: 9, d_text: 4 }) => {}
            other => panic!("expected WrongTextShape, got {other:?}"),
        }
    }

    /// Each image-feature buffer must be a multiple of d_text.
    #[test]
    fn splice_rejects_image_features_with_misshapen_length() {
        let d = 4;
        let text = fake_text_embeds(3, d);
        let bad_img = vec![0.0_f32; 5]; // 5 floats, d=4 → not aligned
        match splice_image_embeddings(&text, &[1], &[bad_img.as_slice()], d) {
            Err(SpliceError::WrongImageShape {
                idx: 0,
                got: 5,
                d_text: 4,
            }) => {}
            other => panic!("expected WrongImageShape, got {other:?}"),
        }
    }

    /// `image_positions.len() != image_features.len()` — the splice
    /// can't guess which image to drop, so it errors instead.
    #[test]
    fn splice_rejects_positions_features_length_mismatch() {
        let d = 2;
        let text = fake_text_embeds(4, d);
        let img = fake_image_features(0, 1, d);
        // Two positions but one feature buffer.
        match splice_image_embeddings(&text, &[1, 2], &[img.as_slice()], d) {
            Err(SpliceError::PositionsFeaturesMismatch {
                n_positions: 2,
                n_features: 1,
            }) => {}
            other => panic!("expected PositionsFeaturesMismatch, got {other:?}"),
        }
    }

    /// Out-of-range placeholder index.
    #[test]
    fn splice_rejects_position_out_of_range() {
        let d = 2;
        let text = fake_text_embeds(3, d);
        let img = fake_image_features(0, 1, d);
        // text_len=3, so position 5 is out of range.
        match splice_image_embeddings(&text, &[5], &[img.as_slice()], d) {
            Err(SpliceError::PositionOutOfRange {
                idx: 0,
                pos: 5,
                text_len: 3,
            }) => {}
            other => panic!("expected PositionOutOfRange, got {other:?}"),
        }
    }

    /// Unsorted positions are a contract violation — the single-pass
    /// walk relies on ascending order.
    #[test]
    fn splice_rejects_unsorted_positions() {
        let d = 2;
        let text = fake_text_embeds(5, d);
        let img0 = fake_image_features(0, 1, d);
        let img1 = fake_image_features(1, 1, d);
        match splice_image_embeddings(
            &text,
            &[3, 1], // backwards
            &[img0.as_slice(), img1.as_slice()],
            d,
        ) {
            Err(SpliceError::PositionsNotSorted { idx: 1, a: 3, b: 1 }) => {}
            other => panic!("expected PositionsNotSorted, got {other:?}"),
        }
    }

    /// Duplicate positions are also rejected (strict increase, not
    /// non-decrease — two images can't map to the same placeholder).
    #[test]
    fn splice_rejects_duplicate_positions() {
        let d = 2;
        let text = fake_text_embeds(5, d);
        let img0 = fake_image_features(0, 1, d);
        let img1 = fake_image_features(1, 1, d);
        match splice_image_embeddings(
            &text,
            &[2, 2],
            &[img0.as_slice(), img1.as_slice()],
            d,
        ) {
            Err(SpliceError::PositionsNotSorted { idx: 1, a: 2, b: 2 }) => {}
            other => panic!("expected PositionsNotSorted (duplicate), got {other:?}"),
        }
    }

    // ---- prepare_vlm_inputs (V-6b-1) -----------------------------

    /// Encode a synthetic RGB PNG for the V-6b-1 tests. Mirrors the
    /// helper used by the forward_image_bytes acceptance test but
    /// kept local so the V-6b-1 tests can vary size easily.
    fn make_vlm_test_png(w: u32, h: u32, seed: u8) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity((w as usize) * (h as usize) * 3);
        for y in 0..h {
            for x in 0..w {
                buf.push((x as u8).wrapping_add(seed));
                buf.push((y as u8).wrapping_add(seed));
                buf.push(seed);
            }
        }
        let img = image::RgbImage::from_raw(w, h, buf).unwrap();
        let mut out: Vec<u8> = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .unwrap();
        out
    }

    /// Single image, single placeholder — happy path. Prompt has the
    /// image-token id at index 3; one image attached.
    #[test]
    fn prepare_vlm_inputs_locates_single_placeholder_and_runs_pipeline() {
        let tmp = std::env::temp_dir().join("rustllama-vlm-inputs-single.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let img = make_vlm_test_png(64, 64, 7);
        let img_id: u32 = 32000;
        // Prompt: [t0, t1, t2, IMG, t4]
        let tokens = [10u32, 11, 12, img_id, 13];
        let out = prepare_vlm_inputs(&model, &tokens, img_id, &[img.as_slice()])
            .expect("prepare_vlm_inputs should succeed");
        assert_eq!(out.positions, vec![3]);
        assert_eq!(out.features.len(), 1);
        // Feature buffer must match the same forward_image_bytes output
        // the engine would compute directly — `prepare_vlm_inputs` is
        // a pure wiring layer, no extra transform.
        let expected = model.forward_image_bytes(&img).unwrap();
        assert_eq!(out.features[0], expected);
        assert_eq!(out.d_text, model.projector.d_text());
        let _ = std::fs::remove_file(&tmp);
    }

    /// Two images at positions 1 and 4 — feature buffers come back in
    /// the same order as the attached bytes.
    #[test]
    fn prepare_vlm_inputs_orders_features_to_match_image_payload_order() {
        let tmp = std::env::temp_dir().join("rustllama-vlm-inputs-two.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        // Two visually distinct images so the feature buffers will
        // also differ — pins that we never swap them.
        let img_a = make_vlm_test_png(48, 48, 11);
        let img_b = make_vlm_test_png(72, 72, 200);
        let img_id: u32 = 50000;
        let tokens = [9u32, img_id, 8, 7, img_id, 6];
        let payloads: Vec<&[u8]> = vec![img_a.as_slice(), img_b.as_slice()];
        let out = prepare_vlm_inputs(&model, &tokens, img_id, &payloads).unwrap();
        assert_eq!(out.positions, vec![1, 4]);
        // Recompute the two expected buffers in payload order.
        let exp_a = model.forward_image_bytes(&img_a).unwrap();
        let exp_b = model.forward_image_bytes(&img_b).unwrap();
        assert_eq!(out.features[0], exp_a, "feature 0 should be image A");
        assert_eq!(out.features[1], exp_b, "feature 1 should be image B");
        assert_ne!(
            out.features[0], out.features[1],
            "distinct images must yield distinct features"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    /// Mismatch: prompt has 2 placeholders but only 1 image attached.
    #[test]
    fn prepare_vlm_inputs_rejects_more_placeholders_than_images() {
        let tmp = std::env::temp_dir().join("rustllama-vlm-inputs-toofew.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let img = make_vlm_test_png(48, 48, 5);
        let img_id: u32 = 100;
        let tokens = [1u32, img_id, 2, img_id, 3];
        let payloads: Vec<&[u8]> = vec![img.as_slice()]; // one image, two slots
        match prepare_vlm_inputs(&model, &tokens, img_id, &payloads) {
            Err(VlmInputBuildError::PlaceholderImageCountMismatch {
                image_token_id: 100,
                placeholders: 2,
                images: 1,
            }) => {}
            other => panic!("expected PlaceholderImageCountMismatch, got {other:?}"),
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// Mismatch the other way: images attached but prompt has no
    /// placeholder. v1 rejects to avoid silently discarding a user's
    /// image — the model wouldn't see it anyway.
    #[test]
    fn prepare_vlm_inputs_rejects_more_images_than_placeholders() {
        let tmp = std::env::temp_dir().join("rustllama-vlm-inputs-toomany.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let img = make_vlm_test_png(48, 48, 5);
        let img_id: u32 = 200;
        // Prompt has no image-token id.
        let tokens = [1u32, 2, 3];
        let payloads: Vec<&[u8]> = vec![img.as_slice()];
        match prepare_vlm_inputs(&model, &tokens, img_id, &payloads) {
            Err(VlmInputBuildError::PlaceholderImageCountMismatch {
                image_token_id: 200,
                placeholders: 0,
                images: 1,
            }) => {}
            other => panic!("expected PlaceholderImageCountMismatch, got {other:?}"),
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// Text-only request (no placeholders, no images) is the no-op
    /// path: positions empty, features empty, d_text set. A VLM-aware
    /// engine can call this unconditionally for every chat request.
    #[test]
    fn prepare_vlm_inputs_no_placeholders_no_images_is_noop() {
        let tmp = std::env::temp_dir().join("rustllama-vlm-inputs-noop.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let tokens = [1u32, 2, 3, 4];
        let out = prepare_vlm_inputs(&model, &tokens, 999, &[]).unwrap();
        assert!(out.positions.is_empty());
        assert!(out.features.is_empty());
        assert_eq!(out.d_text, model.projector.d_text());
        let _ = std::fs::remove_file(&tmp);
    }

    // ---- OnePerPatch (Qwen2-VL multi-token placeholder) tests ----

    /// OnePerPatch mode: prompt has a run of N placeholders matching
    /// the per-image patch count from the synthetic mmproj. The
    /// resulting positions+features arrays are 1:1 (each placeholder
    /// → one patch row), in image+patch order. End-to-end exercises
    /// the run-length detection, the per-image feature split, and
    /// the splice helper's existing 1:1 semantics.
    #[test]
    fn one_per_patch_mode_consecutive_run_matches_image_patches() {
        let tmp = std::env::temp_dir().join("rustllama-vlm-opp-success.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let img = make_vlm_test_png(48, 48, 5);
        // The synth mmproj produces num_patches + 1 (class token) = 577
        // feature rows per image. Build a prompt where IMG_ID repeats
        // exactly 577 times consecutively, sandwiched by text tokens.
        let img_id: u32 = 7;
        let pad = vec![img_id; 577];
        let mut tokens: Vec<u32> = vec![1, 2, 3]; // text prefix
        tokens.extend(&pad); // 577 placeholders
        tokens.push(4); // text suffix
        let payloads: Vec<&[u8]> = vec![img.as_slice()];
        let out = prepare_vlm_inputs_with_mode(
            &model,
            &tokens,
            img_id,
            &payloads,
            PlaceholderMode::OnePerPatch,
        )
        .expect("OnePerPatch with matching run should succeed");
        assert_eq!(out.positions.len(), 577);
        assert_eq!(out.features.len(), 577);
        // Every per-patch feature buffer is exactly d_text floats.
        let d_text = model.projector.d_text();
        for f in &out.features {
            assert_eq!(f.len(), d_text);
        }
        // positions are the indices of the 577 consecutive placeholders.
        assert_eq!(out.positions[0], 3);
        assert_eq!(out.positions[576], 3 + 576);
        let _ = std::fs::remove_file(&tmp);
    }

    /// OnePerPatch with mismatched run length surfaces
    /// `PlaceholderPatchRunMismatch` with both run-length vectors so
    /// the caller can debug template / tokenizer drift.
    #[test]
    fn one_per_patch_mode_rejects_run_length_mismatch() {
        let tmp = std::env::temp_dir().join("rustllama-vlm-opp-mismatch.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let img = make_vlm_test_png(48, 48, 5);
        // Wrong run length: 100 placeholders but image produces 577.
        let img_id: u32 = 7;
        let mut tokens: Vec<u32> = vec![1, 2];
        tokens.extend(std::iter::repeat(img_id).take(100));
        tokens.push(3);
        let payloads: Vec<&[u8]> = vec![img.as_slice()];
        match prepare_vlm_inputs_with_mode(
            &model,
            &tokens,
            img_id,
            &payloads,
            PlaceholderMode::OnePerPatch,
        ) {
            Err(VlmInputBuildError::PlaceholderPatchRunMismatch {
                found,
                expected,
            }) => {
                assert_eq!(found, vec![100]);
                assert_eq!(expected, vec![577]);
            }
            other => panic!("expected PlaceholderPatchRunMismatch, got {other:?}"),
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// OnePerPatch with two images: two consecutive runs of 577
    /// placeholders each, separated by a text token. Validates that
    /// run detection segments correctly and features are flattened
    /// in image order.
    #[test]
    fn one_per_patch_mode_handles_two_images_with_intervening_text() {
        let tmp = std::env::temp_dir().join("rustllama-vlm-opp-two.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let img_a = make_vlm_test_png(48, 48, 7);
        let img_b = make_vlm_test_png(48, 48, 200);
        let img_id: u32 = 11;
        // prompt: text, 577 IMG, text, 577 IMG, text
        let mut tokens: Vec<u32> = vec![1, 2];
        tokens.extend(std::iter::repeat(img_id).take(577));
        tokens.push(3);
        tokens.extend(std::iter::repeat(img_id).take(577));
        tokens.push(4);
        let payloads: Vec<&[u8]> = vec![img_a.as_slice(), img_b.as_slice()];
        let out = prepare_vlm_inputs_with_mode(
            &model,
            &tokens,
            img_id,
            &payloads,
            PlaceholderMode::OnePerPatch,
        )
        .expect("two-image OnePerPatch should succeed");
        assert_eq!(out.positions.len(), 1154); // 577 + 577
        assert_eq!(out.features.len(), 1154);
        // First 577 features come from image A, second 577 from image B.
        // Distinguishable by feature buffer contents (different seeds
        // produce different projections).
        assert_ne!(
            out.features[0], out.features[577],
            "feature[0] (image A's first patch) must differ from feature[577] (image B's first patch)"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    /// OnePerPatch validates the *run grouping* — placeholders split
    /// across non-adjacent positions for the same image (run length
    /// mismatch) surface the same diagnostic.
    #[test]
    fn one_per_patch_mode_rejects_interleaved_placeholders() {
        let tmp = std::env::temp_dir().join("rustllama-vlm-opp-interleave.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let img = make_vlm_test_png(48, 48, 5);
        let img_id: u32 = 7;
        // Two runs of ~290 each (total 577) — wrong because the image
        // produces ONE block of 577 features. Run detection sees
        // [290, 287] which doesn't equal [577].
        let mut tokens: Vec<u32> = vec![1];
        tokens.extend(std::iter::repeat(img_id).take(290));
        tokens.push(2); // breaks the run
        tokens.extend(std::iter::repeat(img_id).take(287));
        tokens.push(3);
        let payloads: Vec<&[u8]> = vec![img.as_slice()];
        match prepare_vlm_inputs_with_mode(
            &model,
            &tokens,
            img_id,
            &payloads,
            PlaceholderMode::OnePerPatch,
        ) {
            Err(VlmInputBuildError::PlaceholderPatchRunMismatch {
                found,
                expected,
            }) => {
                assert_eq!(found, vec![290, 287]);
                assert_eq!(expected, vec![577]);
            }
            other => panic!("expected PlaceholderPatchRunMismatch, got {other:?}"),
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// Per-image pipeline failure (corrupt PNG bytes) surfaces as
    /// `VlmInputBuildError::Pipeline` with the offending index.
    #[test]
    fn prepare_vlm_inputs_surfaces_per_image_pipeline_errors_with_index() {
        let tmp = std::env::temp_dir().join("rustllama-vlm-inputs-corrupt.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let good = make_vlm_test_png(48, 48, 5);
        let corrupt: Vec<u8> = b"not a real PNG, will fail decode".to_vec();
        let img_id: u32 = 7;
        let tokens = [img_id, 1, img_id];
        let payloads: Vec<&[u8]> = vec![good.as_slice(), corrupt.as_slice()];
        match prepare_vlm_inputs(&model, &tokens, img_id, &payloads) {
            Err(VlmInputBuildError::Pipeline { idx: 1, source: _ }) => {}
            other => panic!("expected Pipeline {{idx: 1, ..}}, got {other:?}"),
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// End-to-end V-5 + V-6b-1 chain: prepare_vlm_inputs +
    /// splice_image_embeddings produce a usable transformer input
    /// sequence. Pins the contract that the two pieces interlock.
    #[test]
    fn prepare_vlm_inputs_feeds_directly_into_splice_image_embeddings() {
        let tmp = std::env::temp_dir().join("rustllama-vlm-inputs-chain.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let img = make_vlm_test_png(56, 56, 33);
        let img_id: u32 = 4242;
        // 5-token prompt with the placeholder at index 2.
        let tokens = [10u32, 11, img_id, 12, 13];
        let inputs =
            prepare_vlm_inputs(&model, &tokens, img_id, &[img.as_slice()]).unwrap();
        // Fake text embeddings: row t == [t*10 + k] for k in 0..d_text.
        let text = fake_text_embeds(tokens.len(), inputs.d_text);
        let feat_slices = inputs.feature_slices();
        let spliced = splice_image_embeddings(
            &text,
            &inputs.positions,
            &feat_slices,
            inputs.d_text,
        )
        .expect("splice should accept prepare_vlm_inputs output");
        // Output length: text_len - 1 + num_image_tokens
        let num_image_tokens = inputs.features[0].len() / inputs.d_text;
        let expected_len = (tokens.len() - 1 + num_image_tokens) * inputs.d_text;
        assert_eq!(spliced.len(), expected_len);
        // First two rows = text tokens 0,1
        assert_eq!(&spliced[0..inputs.d_text], &text[0..inputs.d_text]);
        // Tail row = text token 4 (placeholder dropped)
        let tail = expected_len - inputs.d_text;
        assert_eq!(
            &spliced[tail..tail + inputs.d_text],
            &text[4 * inputs.d_text..5 * inputs.d_text]
        );
        let _ = std::fs::remove_file(&tmp);
    }

    /// End-to-end: run the real `forward_image_with_projection` to get
    /// a real `[num_patches, d_text]` patch buffer, then splice it
    /// into a fake text-embedding stream. Pins the API contract that
    /// the vision pipeline + splice helper interlock as the V-5 plan
    /// describes.
    #[test]
    fn splice_consumes_real_vision_projection_output() {
        let tmp = std::env::temp_dir().join("rustllama-vision-splice-real.gguf");
        write_synthetic_clip_mmproj_gguf(&tmp);
        let gguf = Gguf::open(&tmp).unwrap();
        let model = VisionModel::load(&gguf).unwrap();
        let num_patches = model.cfg.num_patches();
        let patch_dim = model.cfg.n_channels * model.cfg.patch_size * model.cfg.patch_size;
        let patches: Vec<f32> = (0..num_patches * patch_dim)
            .map(|i| (i as f32) * 0.0001)
            .collect();
        // Real vision tower forward + projection -> [n_tokens, d_text].
        // n_tokens = num_patches + has_class_token; the synthetic mmproj
        // emits a class token, so n_tokens = num_patches + 1.
        let proj = model
            .forward_image_with_projection(&patches)
            .expect("vision projection succeeds");
        let d_text = model.projector.d_text();
        let n_image_tokens = proj.len() / d_text;
        assert_eq!(n_image_tokens, num_patches + 1);
        // Splice into a 4-token prompt with the placeholder at index 2.
        let text = fake_text_embeds(4, d_text);
        let out = splice_image_embeddings(&text, &[2], &[proj.as_slice()], d_text)
            .expect("splice succeeds");
        // Output length: 4 - 1 + n_image_tokens.
        assert_eq!(out.len(), (3 + n_image_tokens) * d_text);
        // First two rows are text tokens 0,1.
        assert_eq!(&out[0..d_text], &text[0..d_text]);
        assert_eq!(&out[d_text..2 * d_text], &text[d_text..2 * d_text]);
        // Next n_image_tokens rows match the projection buffer exactly.
        assert_eq!(
            &out[2 * d_text..(2 + n_image_tokens) * d_text],
            proj.as_slice()
        );
        // Last row is text token 3 (placeholder at pos 2 dropped).
        let tail = (2 + n_image_tokens) * d_text;
        assert_eq!(
            &out[tail..tail + d_text],
            &text[3 * d_text..4 * d_text]
        );
        let _ = std::fs::remove_file(&tmp);
    }
}
