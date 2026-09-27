//! Parse Llama-family hyperparameters from a GGUF file's metadata.
//!
//! All models in the "Llama family" (Llama / Mistral / Qwen2 / Qwen2.5 /
//! DeepSeek / Phi3 / Yi) share the same metadata key shape; only the prefix
//! differs (it equals the architecture name). This module reads keys with
//! the appropriate prefix.

use rustllama_gguf::{Gguf, MetadataValue};

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub arch: String,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub vocab_size: usize,
    pub ctx_train: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub tie_word_embeddings: bool,
    /// Number of Multi-Token Prediction (MTP) heads on this model.
    /// DeepSeek-V3 ships with extra transformer-block-shaped modules
    /// at the tail of the network that predict the +1, +2, …, +N
    /// tokens from the final hidden state. When > 0, the loader binds
    /// the `mtp.{i}.*` tensors into `LlamaWeights::mtp_heads`. Default
    /// 0 (no MTP), in which case the model loads and runs through the
    /// unchanged single-head path.
    pub n_mtp_heads: u32,
    /// MoE (mixture-of-experts) configuration. `Some` when the GGUF
    /// declares `{arch}.expert_count > 0` (Qwen3-MoE, Mixtral,
    /// DeepSeek-V3 family); `None` for dense models. The loader uses
    /// the presence of this field to decide between binding dense
    /// FFN tensors (`ffn_gate.weight`/`ffn_up.weight`/`ffn_down.weight`)
    /// and MoE expert tensors (`ffn_gate_inp.weight` router +
    /// `ffn_gate_exps.weight`/`ffn_up_exps.weight`/`ffn_down_exps.weight`).
    /// MoE forward pass is a follow-up turn — load succeeds for
    /// inspection (`/v1/models` / GGUF inspector) but the engine
    /// fails fast with `LlamaLoadError::UnsupportedMoe` at first
    /// inference attempt so users get a clear "MoE not yet
    /// implemented" message rather than a cryptic shape mismatch.
    pub moe: Option<MoeConfig>,
    /// Hybrid transformer+SSM (Mamba-style) configuration.
    /// `Some` when the GGUF declares `{arch}.full_attention_interval`
    /// AND the `{arch}.ssm.*` family of metadata keys. Models that
    /// match: Qwen3.5-MoE-Hybrid (`qwen35moe`) and any future
    /// llama.cpp ports of Jamba / Zamba / Hymba.
    ///
    /// On a hybrid model, layer `i` is a full-attention layer when
    /// `(i + 1) % full_attention_interval == 0`; otherwise it's an
    /// SSM block with a gated attention side-channel. The Phase-2
    /// loader binds the right tensor set per layer; the Phase-3
    /// forward dispatches to the right kernel per layer.
    pub hybrid: Option<HybridConfig>,
    /// PrismML Hadamard rotation metadata (`prism.hadamard.*`).
    /// `Some` on Bonsai-family ternary GGUFs; the forward pass MUST
    /// apply the activation transform when present (enforced at
    /// weight-load time — a model with this metadata refuses to run
    /// on a forward path that hasn't wired the transform).
    pub hadamard: Option<HadamardConfig>,
}

#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Layer-stride at which a full-attention layer appears.
    /// Layer `i` is full-attention iff `(i + 1) % full_attention_interval == 0`.
    /// For `qwen35moe`'s 41 layers with interval 4 that's layers
    /// 3, 7, 11, …, 39 — 10 full-attention layers (the model dump
    /// shows 11 because layer 40, the final layer, also lands in
    /// the full-attention slot under the same modulo rule).
    pub full_attention_interval: u32,
    /// SSM hidden state size (`d_state` in the Mamba paper).
    /// Sourced from `{arch}.ssm.state_size`. `qwen35moe`: 128.
    pub ssm_state_size: u32,
    /// SSM 1D-conv kernel width. Sourced from `{arch}.ssm.conv_kernel`.
    /// `qwen35moe`: 4.
    pub ssm_conv_kernel: u32,
    /// SSM group count (groupwise norm). Sourced from
    /// `{arch}.ssm.group_count`. `qwen35moe`: 16.
    pub ssm_group_count: u32,
    /// Selective-scan dt projection rank. Sourced from
    /// `{arch}.ssm.time_step_rank`. `qwen35moe`: 32.
    pub ssm_time_step_rank: u32,
    /// Inner expanded dim of the SSM block (input projection
    /// width). Sourced from `{arch}.ssm.inner_size`.
    /// `qwen35moe`: 4096 (= 2 × d_model).
    pub ssm_inner_size: u32,
    /// Per-expert FFN width for the shared expert. Pure-MoE
    /// models with a router-gated shared expert (Qwen3.5-MoE)
    /// declare this separately from the main expert d_ff
    /// because the shared expert's FFN can be a different size.
    /// `None` when the model doesn't ship a shared expert.
    pub shared_expert_feed_forward_length: Option<u32>,
    /// Number of MTP heads (typically 1 on `qwen35moe`).
    /// Sourced from `{arch}.nextn_predict_layers`. This is the
    /// *hybrid-arch* MTP key; pure DeepSeek-V3 uses
    /// `{arch}.mtp.head_count`. The loader checks both.
    pub nextn_predict_layers: u32,
}

/// PrismML Hadamard rotation metadata (`prism.hadamard.*`) — present
/// on Bonsai-family ternary GGUFs whose weights are stored in a
/// rotated basis. The runtime MUST apply the matching activation
/// transform (see `rustllama_kernels_cpu::hadamard`) or the model
/// produces garbage; every validation here is a hard error for the
/// same reason the reference implementation refuses to run: wrong
/// rotation math silently destroys the model.
#[derive(Clone)]
pub struct HadamardConfig {
    /// Rotation block size (1024 on Bonsai 2). Power of two.
    pub block_size: usize,
    /// Tensor names whose INPUT activations get the forward
    /// rotation before their matmul (weights are folded `W·Rᵀ`).
    pub weight_names: Vec<String>,
    /// Tensor names whose OUTPUT gets the inverse rotation — only
    /// `token_embd.weight` is legal (rotated embedding rows).
    pub inverse_weight_names: Vec<String>,
    /// Per-full-input-width ±1.0 sign vectors, keyed by width.
    /// Empty map ⇔ `sign_mode = "identity"`. A weight of input
    /// width `w` uses `signs_by_width[&w]` across its whole width
    /// (each block consumes its own slice).
    pub signs_by_width: std::collections::HashMap<usize, std::sync::Arc<Vec<f32>>>,
    /// When true, `ssm_out`'s input activation needs the tiled→
    /// grouped V-head permutation before rotation (the fold was
    /// done in HF's grouped head order while the GDN runtime
    /// produces tiled order). Mirrors the fork's `gdn_v_grouped`.
    pub gdn_v_grouped: bool,
}

impl std::fmt::Debug for HadamardConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Sign vectors run to hundreds of KB — summarize.
        f.debug_struct("HadamardConfig")
            .field("block_size", &self.block_size)
            .field("weight_names", &self.weight_names.len())
            .field("inverse_weight_names", &self.inverse_weight_names)
            .field(
                "sign_widths",
                &self.signs_by_width.keys().collect::<Vec<_>>(),
            )
            .field("gdn_v_grouped", &self.gdn_v_grouped)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Number of experts per MoE layer. Sourced from
    /// `{arch}.expert_count`. Mixtral-8x7B: 8; Qwen3-MoE-A30B: 64;
    /// DeepSeek-V3: 256 (with additional `expert_shared_count`
    /// shared experts).
    pub n_experts: u32,
    /// Number of experts routed to per token (top-K). Sourced from
    /// `{arch}.expert_used_count`. Mixtral: 2; Qwen3-MoE: 8;
    /// DeepSeek-V3: 8.
    pub n_experts_used: u32,
    /// Number of "shared" experts that are always active (in
    /// addition to the top-K routed experts). DeepSeek-V3 uses
    /// shared experts to capture common patterns; Mixtral / Qwen3-MoE
    /// have 0. Sourced from `{arch}.expert_shared_count` (defaults
    /// to 0 when the key is absent).
    pub n_experts_shared: u32,
}

impl LlamaConfig {
    pub fn from_gguf(gguf: &Gguf) -> Result<Self, ConfigError> {
        let arch = gguf
            .architecture()
            .ok_or(ConfigError::Missing("general.architecture"))?
            .to_string();

        let key = |name: &str| format!("{arch}.{name}");

        let n_layers = u32_required(gguf, &key("block_count"))? as usize;
        let d_model = u32_required(gguf, &key("embedding_length"))? as usize;
        // MoE-aware `d_ff` resolution. The dense `feed_forward_length`
        // key exists on some MoE GGUFs as a legacy/info field even
        // though the model has no dense FFN path; preferring it over
        // `expert_feed_forward_length` reads the WRONG width and
        // makes the per-expert tensor slicing offset garbage. So
        // prefer the MoE-specific key when ANY expert metadata is
        // present, and fall through to dense otherwise.
        let has_moe_meta = u32_optional(gguf, &key("expert_count")).unwrap_or(0) > 0;
        let d_ff = if has_moe_meta {
            u32_optional(gguf, &key("expert_feed_forward_length"))
                .or_else(|| u32_optional(gguf, &key("feed_forward_length")))
                .ok_or(ConfigError::Missing("expert_feed_forward_length"))? as usize
        } else {
            u32_optional(gguf, &key("feed_forward_length"))
                .or_else(|| u32_optional(gguf, &key("expert_feed_forward_length")))
                .ok_or(ConfigError::Missing("feed_forward_length"))? as usize
        };
        let n_heads = u32_required(gguf, &key("attention.head_count"))? as usize;
        let n_kv_heads =
            u32_optional(gguf, &key("attention.head_count_kv")).unwrap_or(n_heads as u32) as usize;
        let rms_eps = f32_optional(gguf, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-5);
        let head_dim = u32_optional(gguf, &key("attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or_else(|| d_model / n_heads);
        let rope_dim = u32_optional(gguf, &key("rope.dimension_count"))
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        let rope_theta = f32_optional(gguf, &key("rope.freq_base")).unwrap_or(10000.0);
        let ctx_train = u32_optional(gguf, &key("context_length")).unwrap_or(2048) as usize;

        let vocab_size = match gguf.metadata_get("tokenizer.ggml.tokens") {
            Some(MetadataValue::Array(v)) => v.len(),
            _ => u32_optional(gguf, &key("vocab_size")).unwrap_or(0) as usize,
        };
        if vocab_size == 0 {
            return Err(ConfigError::Missing(
                "tokenizer.ggml.tokens or *.vocab_size",
            ));
        }

        let bos_token_id = u32_optional(gguf, "tokenizer.ggml.bos_token_id");
        let eos_token_id = u32_optional(gguf, "tokenizer.ggml.eos_token_id");

        // Many Llama-family models tie input and output embeddings. The GGUF
        // convention is to *omit* `output.weight` when this is the case.
        let tie_word_embeddings = gguf.tensor("output.weight").is_none();

        // MoE detection. Real Qwen3-MoE / Mixtral / DeepSeek-V3 GGUFs
        // declare `{arch}.expert_count > 0`. Dense models either omit
        // the key entirely or set it to 0. When MoE is detected,
        // `expert_used_count` is the top-K and `expert_shared_count`
        // defaults to 0 (DeepSeek's shared-expert design is the
        // exception that does set it).
        // MTP detection. DeepSeek-V3-MTP GGUFs carry
        // `{arch}.mtp.head_count` and store the head as
        // separate `mtp.{i}.*` transformer blocks. Hybrid Qwen-MoE
        // models declare `{arch}.nextn_predict_layers` instead and
        // store the head as `blk.{N}.nextn.*` tensors — that path
        // is routed via [`HybridConfig::nextn_predict_layers`] +
        // [`LlamaWeights::nextn_head`], NOT `n_mtp_heads`. We
        // therefore only honor the DeepSeek key here; the hybrid
        // key is intentionally ignored to keep the two MTP
        // conventions on disjoint code paths.
        let n_mtp_heads = u32_optional(gguf, &key("mtp.head_count")).unwrap_or(0);

        let moe = match u32_optional(gguf, &key("expert_count")) {
            Some(n) if n > 0 => Some(MoeConfig {
                n_experts: n,
                n_experts_used: u32_optional(gguf, &key("expert_used_count")).unwrap_or(1),
                n_experts_shared: u32_optional(gguf, &key("expert_shared_count")).unwrap_or(0),
            }),
            _ => None,
        };

        // Hybrid transformer+SSM detection. Triggered by the presence
        // of `{arch}.full_attention_interval` AND
        // `{arch}.ssm.state_size`. We require both so a pure-transformer
        // model that happens to declare `full_attention_interval` for
        // some other reason can't accidentally route to the SSM path.
        let hybrid = match (
            u32_optional(gguf, &key("full_attention_interval")),
            u32_optional(gguf, &key("ssm.state_size")),
        ) {
            (Some(interval), Some(state_size)) if interval > 0 => Some(HybridConfig {
                full_attention_interval: interval,
                ssm_state_size: state_size,
                ssm_conv_kernel: u32_optional(gguf, &key("ssm.conv_kernel")).unwrap_or(4),
                ssm_group_count: u32_optional(gguf, &key("ssm.group_count")).unwrap_or(1),
                ssm_time_step_rank: u32_optional(gguf, &key("ssm.time_step_rank"))
                    .unwrap_or(state_size / 4),
                ssm_inner_size: u32_optional(gguf, &key("ssm.inner_size"))
                    .unwrap_or((d_model * 2) as u32),
                shared_expert_feed_forward_length: u32_optional(
                    gguf,
                    &key("expert_shared_feed_forward_length"),
                ),
                nextn_predict_layers: u32_optional(gguf, &key("nextn_predict_layers"))
                    .unwrap_or(0),
            }),
            _ => None,
        };

        let hadamard = parse_hadamard(gguf)?;

        let cfg = Self {
            arch,
            n_layers,
            n_heads,
            n_kv_heads,
            d_model,
            d_ff,
            head_dim,
            rope_dim,
            rope_theta,
            rms_eps,
            vocab_size,
            ctx_train,
            bos_token_id,
            eos_token_id,
            tie_word_embeddings,
            n_mtp_heads,
            moe,
            hybrid,
            hadamard,
        };
        // Diagnostic: print key config values at load time so hybrid
        // model bugs can be quickly localized to a wrong-loaded constant.
        eprintln!(
            "[CFG DUMP] arch={} n_layers={} d_model={} n_heads={} n_kv_heads={} \
             head_dim={} rope_dim={} rope_theta={} d_ff={} vocab={} moe={:?} hybrid={:?}",
            cfg.arch, cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
            cfg.head_dim, cfg.rope_dim, cfg.rope_theta, cfg.d_ff, cfg.vocab_size,
            cfg.moe, cfg.hybrid,
        );
        Ok(cfg)
    }

    pub fn n_gqa(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("missing required GGUF metadata key: {0}")]
    Missing(&'static str),
    #[error("metadata key {0} has unexpected type")]
    BadType(String),
    #[error("invalid GGUF metadata: {0}")]
    Invalid(String),
}

/// Tensor-name suffixes whose input activations may legally be
/// Hadamard-folded (mirror of the reference implementation's
/// whitelist). A `prism.hadamard.weight_names` entry outside this
/// set means a rotation convention we haven't wired — hard error.
const HADAMARD_FOLDABLE_SUFFIXES: &[&str] = &[
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_qkv.weight",
    "attn_gate.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
    "ssm_out.weight",
];

/// Parse + hard-validate `prism.hadamard.*`. Returns `Ok(None)` when
/// the version key is absent (non-Prism model). Every violation is a
/// hard error: the reference implementation refuses to run wrong
/// rotation math and so do we — a lenient fallback here would mean
/// silently garbled output.
fn parse_hadamard(gguf: &Gguf) -> Result<Option<HadamardConfig>, ConfigError> {
    let version = match u32_optional(gguf, "prism.hadamard.version") {
        None => return Ok(None),
        Some(v) => v,
    };
    let bad = |msg: String| ConfigError::Invalid(msg);
    if version != 1 {
        return Err(bad(format!("unsupported prism.hadamard.version: {version}")));
    }
    let block_size = u32_required(gguf, "prism.hadamard.block_size")? as usize;
    if block_size < 2 || !block_size.is_power_of_two() {
        return Err(bad(format!("invalid prism.hadamard.block_size: {block_size}")));
    }
    let transform = str_required(gguf, "prism.hadamard.transform")?;
    if transform != "normalized-sylvester-walsh-hadamard" {
        return Err(bad(format!("unsupported prism.hadamard.transform: {transform}")));
    }
    let axis = str_required(gguf, "prism.hadamard.axis")?;
    if axis != "input-last-dimension" {
        return Err(bad(format!("unsupported prism.hadamard.axis: {axis}")));
    }
    let sign_mode = str_required(gguf, "prism.hadamard.sign_mode")?;
    if sign_mode != "identity" && sign_mode != "explicit" {
        return Err(bad(format!("unsupported prism.hadamard.sign_mode: {sign_mode}")));
    }

    let weight_names = str_array(gguf, "prism.hadamard.weight_names")?;
    if weight_names.is_empty() {
        return Err(bad("prism.hadamard.weight_names is empty".into()));
    }
    let mut seen = std::collections::HashSet::new();
    for name in &weight_names {
        let ok = name == "output.weight"
            || HADAMARD_FOLDABLE_SUFFIXES
                .iter()
                .any(|suf| name.starts_with("blk.") && name.ends_with(suf));
        if !ok {
            return Err(bad(format!(
                "prism.hadamard.weight_names entry not on the foldable whitelist: {name}"
            )));
        }
        if !seen.insert(name.clone()) {
            return Err(bad(format!("duplicate prism.hadamard.weight_names entry: {name}")));
        }
    }
    let inverse_weight_names =
        str_array(gguf, "prism.hadamard.inverse_weight_names").unwrap_or_default();
    for name in &inverse_weight_names {
        if name != "token_embd.weight" {
            return Err(bad(format!(
                "prism.hadamard.inverse_weight_names supports only token_embd.weight, got {name}"
            )));
        }
    }

    let mut signs_by_width = std::collections::HashMap::new();
    if sign_mode == "explicit" {
        let widths = i32_array(gguf, "prism.hadamard.sign_widths")?;
        let values = i32_array(gguf, "prism.hadamard.sign_values")?;
        if widths.is_empty() {
            return Err(bad(
                "prism.hadamard.sign_mode is explicit but sign_widths is empty".into(),
            ));
        }
        let mut off = 0usize;
        for &w in &widths {
            if w <= 0 || (w as usize) % block_size != 0 {
                return Err(bad(format!("invalid prism.hadamard sign width: {w}")));
            }
            let w = w as usize;
            if off + w > values.len() {
                return Err(bad(format!(
                    "prism.hadamard.sign_values too short: need {} have {}",
                    off + w,
                    values.len()
                )));
            }
            let mut signs = Vec::with_capacity(w);
            for &v in &values[off..off + w] {
                match v {
                    1 => signs.push(1.0f32),
                    -1 => signs.push(-1.0f32),
                    other => {
                        return Err(bad(format!(
                            "prism.hadamard.sign_values entry must be ±1, got {other}"
                        )))
                    }
                }
            }
            if signs_by_width.insert(w, std::sync::Arc::new(signs)).is_some() {
                return Err(bad(format!("duplicate prism.hadamard sign width: {w}")));
            }
            off += w;
        }
        if off != values.len() {
            return Err(bad(format!(
                "prism.hadamard.sign_values length {} does not match sum of widths {}",
                values.len(),
                off
            )));
        }
    }

    let gdn_v_grouped = bool_optional(gguf, "prism.hadamard.gdn_v_grouped").unwrap_or(false);

    Ok(Some(HadamardConfig {
        block_size,
        weight_names,
        inverse_weight_names,
        signs_by_width,
        gdn_v_grouped,
    }))
}

fn str_required(gguf: &Gguf, key: &str) -> Result<String, ConfigError> {
    match gguf.metadata_get(key) {
        Some(MetadataValue::String(s)) => Ok(s.clone()),
        Some(_) => Err(ConfigError::BadType(key.to_string())),
        None => Err(ConfigError::Missing(Box::leak(
            key.to_string().into_boxed_str(),
        ))),
    }
}

fn str_array(gguf: &Gguf, key: &str) -> Result<Vec<String>, ConfigError> {
    match gguf.metadata_get(key) {
        Some(MetadataValue::Array(items)) => items
            .iter()
            .map(|v| match v {
                MetadataValue::String(s) => Ok(s.clone()),
                _ => Err(ConfigError::BadType(key.to_string())),
            })
            .collect(),
        Some(_) => Err(ConfigError::BadType(key.to_string())),
        None => Err(ConfigError::Missing(Box::leak(
            key.to_string().into_boxed_str(),
        ))),
    }
}

fn i32_array(gguf: &Gguf, key: &str) -> Result<Vec<i32>, ConfigError> {
    match gguf.metadata_get(key) {
        Some(MetadataValue::Array(items)) => items
            .iter()
            .map(|v| match v {
                MetadataValue::I32(x) => Ok(*x),
                MetadataValue::I64(x) if *x >= i32::MIN as i64 && *x <= i32::MAX as i64 => {
                    Ok(*x as i32)
                }
                MetadataValue::U32(x) if *x <= i32::MAX as u32 => Ok(*x as i32),
                _ => Err(ConfigError::BadType(key.to_string())),
            })
            .collect(),
        Some(_) => Err(ConfigError::BadType(key.to_string())),
        None => Err(ConfigError::Missing(Box::leak(
            key.to_string().into_boxed_str(),
        ))),
    }
}

fn bool_optional(gguf: &Gguf, key: &str) -> Option<bool> {
    match gguf.metadata_get(key)? {
        MetadataValue::Bool(b) => Some(*b),
        _ => None,
    }
}

fn u32_required(gguf: &Gguf, key: &str) -> Result<u32, ConfigError> {
    match gguf.metadata_get(key) {
        Some(MetadataValue::U32(v)) => Ok(*v),
        Some(MetadataValue::U64(v)) if *v <= u32::MAX as u64 => Ok(*v as u32),
        Some(MetadataValue::I32(v)) if *v >= 0 => Ok(*v as u32),
        Some(MetadataValue::I64(v)) if *v >= 0 && *v <= u32::MAX as i64 => Ok(*v as u32),
        Some(_) => Err(ConfigError::BadType(key.to_string())),
        None => Err(ConfigError::Missing(Box::leak(
            key.to_string().into_boxed_str(),
        ))),
    }
}

fn u32_optional(gguf: &Gguf, key: &str) -> Option<u32> {
    u32_required(gguf, key).ok()
}

fn f32_optional(gguf: &Gguf, key: &str) -> Option<f32> {
    match gguf.metadata_get(key)? {
        MetadataValue::F32(v) => Some(*v),
        MetadataValue::F64(v) => Some(*v as f32),
        _ => None,
    }
}
