//! Parse HuggingFace `config.json` into a rustllama
//! [`LlamaConfig`](rustllama_models::llama_config::LlamaConfig).
//!
//! AWQ / GPTQ safetensors models ship with a sibling `config.json`
//! that carries the architecture hyperparameters (n_layers, n_heads,
//! hidden_size, …). Unlike GGUF, safetensors itself doesn't store any
//! of that — the `.safetensors` file is purely a typed-tensor
//! container — so the loader has to read `config.json` to know what
//! shape the tensors *should* be and how many layers to bind.
//!
//! Source: the HF `transformers` `PretrainedConfig` schema. Field
//! names mirror that schema verbatim; some keys are optional and
//! default to derived / well-known values when absent.

use rustllama_models::llama_config::{LlamaConfig, MoeConfig};
use serde::Deserialize;

/// HuggingFace `config.json` shape — just the fields rustllama needs.
/// Extra keys in the source are ignored by serde so the parser
/// doesn't fail on future / model-specific additions.
#[derive(Debug, Deserialize)]
struct HfConfigJson {
    /// e.g. `["Qwen2ForCausalLM"]`, `["LlamaForCausalLM"]`. The
    /// loader normalizes the first entry into a lowercase short
    /// name (`qwen2`, `llama`) that matches the GGUF architecture
    /// string convention.
    #[serde(default)]
    architectures: Vec<String>,

    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    /// GQA: number of K/V heads. Defaults to `num_attention_heads`
    /// (no grouping) when absent — matches Llama 1 / Mistral.
    #[serde(default)]
    num_key_value_heads: Option<u32>,
    /// RoPE base frequency. Default 10_000 (Llama 1 convention).
    /// Qwen2.5-Coder uses 1_000_000.
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    /// RMSNorm epsilon. Default 1e-5; some checkpoints use 1e-6.
    #[serde(default = "default_rms_eps")]
    rms_norm_eps: f32,
    /// Total context length the model was trained for.
    #[serde(default = "default_ctx_train")]
    max_position_embeddings: u32,
    vocab_size: u32,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    bos_token_id: Option<u32>,
    /// Some HF configs declare `eos_token_id` as an array (multiple
    /// EOS variants — Qwen2.5-Coder lists `[151645, 151643]`); take
    /// the first as the canonical id.
    #[serde(default)]
    eos_token_id: Option<EosTokenId>,
    /// Per-head dim. Optional — defaults to `hidden_size /
    /// num_attention_heads` when absent (the universal Llama-family
    /// formula).
    #[serde(default)]
    head_dim: Option<u32>,

    // ---- MoE (mixture-of-experts) fields --------------------------
    // Present on MoE checkpoints (Qwen2-MoE / Qwen3-MoE / OLMoE /
    // Mixtral). All optional so dense configs parse unchanged. MoE
    // detection keys on a non-zero expert count.
    /// Qwen2/Qwen3-MoE / OLMoE expert count (`num_experts`).
    #[serde(default)]
    num_experts: Option<u32>,
    /// Mixtral's name for the same field (`num_local_experts`).
    #[serde(default)]
    num_local_experts: Option<u32>,
    /// Top-K routed experts per token (`num_experts_per_tok`).
    #[serde(default)]
    num_experts_per_tok: Option<u32>,
    /// Per-expert FFN width. On MoE models this — NOT
    /// `intermediate_size` — is the expert feed-forward length the
    /// per-expert tensors slice to (mirrors the GGUF loader's
    /// preference for `expert_feed_forward_length`). Absent on
    /// Mixtral (whose experts use `intermediate_size`).
    #[serde(default)]
    moe_intermediate_size: Option<u32>,
    /// Qwen2-MoE always-on shared-expert FFN width. Its presence
    /// marks a (sigmoid-gated) shared expert; the width differs from
    /// `moe_intermediate_size` so the loader derives it from the
    /// tensor itself at build time (this field only flags presence).
    #[serde(default)]
    shared_expert_intermediate_size: Option<u32>,
}

/// HF allows `eos_token_id: 151645` or `eos_token_id: [151645, 151643]`.
/// Untagged so serde tries both forms.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosTokenId {
    Single(u32),
    Multi(Vec<u32>),
}

impl EosTokenId {
    fn first(&self) -> Option<u32> {
        match self {
            Self::Single(v) => Some(*v),
            Self::Multi(v) => v.first().copied(),
        }
    }
}

fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_rms_eps() -> f32 {
    1e-5
}
fn default_ctx_train() -> u32 {
    2048
}

#[derive(Debug, thiserror::Error)]
pub enum HfConfigError {
    #[error("config.json parse: {0}")]
    Json(#[from] serde_json::Error),
    #[error("config.json missing `architectures` (or it's empty)")]
    MissingArch,
    #[error(
        "architecture {0} is not a Llama-family model — rustllama v1 \
         supports Llama, Qwen2/Qwen2.5, Mistral, DeepSeek-Coder, Phi3, \
         and Yi (all of which ship `*ForCausalLM` and follow the \
         standard Llama tensor layout)"
    )]
    UnsupportedArch(String),
}

/// Parse a HuggingFace `config.json` (as a `&str`) and return the
/// corresponding [`LlamaConfig`] rustllama uses.
///
/// Architecture string is normalized: `"Qwen2ForCausalLM"` →
/// `"qwen2"`, `"LlamaForCausalLM"` → `"llama"`, etc. The conversion
/// strips the `ForCausalLM` / `ForSequenceClassification` / similar
/// suffix and lowercases the remainder.
pub fn parse_hf_config(json: &str) -> Result<LlamaConfig, HfConfigError> {
    let raw: HfConfigJson = serde_json::from_str(json)?;
    let arch_raw = raw
        .architectures
        .first()
        .ok_or(HfConfigError::MissingArch)?;
    let arch = normalize_arch(arch_raw)
        .ok_or_else(|| HfConfigError::UnsupportedArch(arch_raw.clone()))?;

    let n_heads = raw.num_attention_heads as usize;
    let n_kv_heads =
        raw.num_key_value_heads.unwrap_or(raw.num_attention_heads) as usize;
    let d_model = raw.hidden_size as usize;
    let head_dim = raw
        .head_dim
        .map(|v| v as usize)
        .unwrap_or_else(|| d_model / n_heads.max(1));
    let eos_id = raw.eos_token_id.as_ref().and_then(EosTokenId::first);

    // MoE detection from the HF config. A non-zero `num_experts`
    // (`num_local_experts` on Mixtral) marks a mixture-of-experts
    // checkpoint. `n_experts_shared` flags Qwen2-MoE's always-on
    // shared expert (presence of `shared_expert_intermediate_size`).
    let n_experts = raw.num_experts.or(raw.num_local_experts).unwrap_or(0);
    let moe = if n_experts > 0 {
        Some(MoeConfig {
            n_experts,
            n_experts_used: raw.num_experts_per_tok.unwrap_or(1),
            n_experts_shared: if raw.shared_expert_intermediate_size.is_some() {
                1
            } else {
                0
            },
        })
    } else {
        None
    };

    // `d_ff` is the FFN width the per-layer tensors slice to. On MoE
    // models that's the PER-EXPERT width (`moe_intermediate_size`),
    // not the dense `intermediate_size` (which, where present, sizes
    // only the Qwen2-MoE shared expert). Mirrors the GGUF loader
    // preferring `expert_feed_forward_length` under expert metadata.
    // Mixtral carries no `moe_intermediate_size`; its experts use
    // `intermediate_size`, so fall back to it.
    let d_ff = if moe.is_some() {
        raw.moe_intermediate_size
            .unwrap_or(raw.intermediate_size) as usize
    } else {
        raw.intermediate_size as usize
    };

    Ok(LlamaConfig {
        arch,
        n_layers: raw.num_hidden_layers as usize,
        n_heads,
        n_kv_heads,
        d_model,
        d_ff,
        head_dim,
        // HF doesn't separate rope dim from head dim — they're equal
        // for every Llama-family model.
        rope_dim: head_dim,
        rope_theta: raw.rope_theta,
        rms_eps: raw.rms_norm_eps,
        vocab_size: raw.vocab_size as usize,
        ctx_train: raw.max_position_embeddings as usize,
        bos_token_id: raw.bos_token_id,
        eos_token_id: eos_id,
        tie_word_embeddings: raw.tie_word_embeddings,
        n_mtp_heads: 0,
        moe, // MoE now parsed from HF config (MLX MoE load path)
        hybrid: None, // HF configs don't yet carry SSM keys for our hybrid path
        hadamard: None, // Prism rotation metadata is GGUF-only
    })
}

/// Normalize a HF `architectures` entry into the lowercase short
/// name rustllama uses (`qwen2`, `llama`, etc.). Returns `None` when
/// the entry doesn't end in a recognized causal-LM suffix.
fn normalize_arch(raw: &str) -> Option<String> {
    // Suffixes ordered by length (longest first) so the strip never
    // leaves residue. e.g. `Qwen2VLForConditionalGeneration` → unsupported
    // (returns None), `Qwen2ForCausalLM` → "qwen2".
    const SUFFIXES: &[&str] = &[
        "ForCausalLM",
        "ForConditionalGeneration",
        "LMHeadModel",
    ];
    for suffix in SUFFIXES {
        if let Some(stem) = raw.strip_suffix(suffix) {
            if stem.is_empty() {
                return None;
            }
            // For v1 we only support the Causal-LM family of dense
            // Llama-shape decoders. `ForConditionalGeneration` and
            // `LMHeadModel` are recognized to produce a useful
            // error message but rejected at the next gate.
            if *suffix == "ForCausalLM" {
                return Some(stem.to_lowercase());
            } else {
                return None;
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_qwen2_coder_style_config() {
        // Mirrors the real Qwen2.5-Coder-7B-Instruct config.json
        // shape (truncated to the fields the parser reads).
        let json = r#"{
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "vocab_size": 152064,
            "tie_word_embeddings": false,
            "bos_token_id": 151643,
            "eos_token_id": [151645, 151643],
            "head_dim": 128
        }"#;
        let cfg = parse_hf_config(json).unwrap();
        assert_eq!(cfg.arch, "qwen2");
        assert_eq!(cfg.n_layers, 28);
        assert_eq!(cfg.n_heads, 28);
        assert_eq!(cfg.n_kv_heads, 4);
        assert_eq!(cfg.d_model, 3584);
        assert_eq!(cfg.d_ff, 18944);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 152064);
        assert_eq!(cfg.ctx_train, 32768);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1e-3);
        assert!((cfg.rms_eps - 1e-6).abs() < 1e-9);
        assert_eq!(cfg.bos_token_id, Some(151643));
        // First entry of the eos array wins.
        assert_eq!(cfg.eos_token_id, Some(151645));
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn parses_llama_style_config_with_single_eos() {
        // Llama-2 / Llama-3 ship `eos_token_id` as a single integer.
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "max_position_embeddings": 4096,
            "vocab_size": 32000,
            "bos_token_id": 1,
            "eos_token_id": 2
        }"#;
        let cfg = parse_hf_config(json).unwrap();
        assert_eq!(cfg.arch, "llama");
        // num_key_value_heads absent → defaults to n_heads (no GQA).
        assert_eq!(cfg.n_kv_heads, 32);
        assert_eq!(cfg.eos_token_id, Some(2));
        // rope_theta absent → default 10_000.
        assert!((cfg.rope_theta - 10_000.0).abs() < 1e-3);
        // rms_norm_eps absent → default 1e-5.
        assert!((cfg.rms_eps - 1e-5).abs() < 1e-9);
        // head_dim derives from d_model / n_heads.
        assert_eq!(cfg.head_dim, 128);
    }

    #[test]
    fn parses_mistral_style_config() {
        let json = r#"{
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 32768,
            "vocab_size": 32000,
            "rope_theta": 10000.0
        }"#;
        let cfg = parse_hf_config(json).unwrap();
        assert_eq!(cfg.arch, "mistral");
        assert_eq!(cfg.n_kv_heads, 8);
    }

    #[test]
    fn rejects_vlm_style_arch_for_now() {
        // Qwen2-VL is a `ForConditionalGeneration` — the safetensors
        // loader v1 doesn't handle the vision-tower layout. Reject
        // with a clear message rather than silently load only the
        // text decoder.
        let json = r#"{
            "architectures": ["Qwen2VLForConditionalGeneration"],
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "vocab_size": 151936
        }"#;
        match parse_hf_config(json) {
            Err(HfConfigError::UnsupportedArch(ref s))
                if s == "Qwen2VLForConditionalGeneration" => {}
            other => panic!("expected UnsupportedArch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_config_with_no_architectures_field() {
        // All other required fields present so the JSON parse succeeds
        // and the `architectures` check is what surfaces.
        let json = r#"{
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000
        }"#;
        match parse_hf_config(json) {
            Err(HfConfigError::MissingArch) => {}
            other => panic!("expected MissingArch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_completely_unknown_architecture() {
        let json = r#"{
            "architectures": ["FrobnicatorClassifier"],
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "vocab_size": 32
        }"#;
        match parse_hf_config(json) {
            Err(HfConfigError::UnsupportedArch(ref s))
                if s == "FrobnicatorClassifier" => {}
            other => panic!("expected UnsupportedArch, got {other:?}"),
        }
    }

    #[test]
    fn extra_unknown_keys_are_ignored() {
        // serde's default `deny_unknown_fields = false` is the v1
        // posture — future HF schema additions don't break the
        // loader.
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 256,
            "future_field": "this should not break parsing",
            "another_one": {"nested": "object"}
        }"#;
        let cfg = parse_hf_config(json).unwrap();
        assert_eq!(cfg.arch, "llama");
    }

    #[test]
    fn malformed_json_returns_json_error() {
        let json = "{ not even close to valid json";
        match parse_hf_config(json) {
            Err(HfConfigError::Json(_)) => {}
            other => panic!("expected Json error, got {other:?}"),
        }
    }

    #[test]
    fn parses_qwen2_moe_config_with_shared_expert() {
        // Mirrors mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit (Qwen2-MoE):
        // 60 experts top-4, per-expert FFN 1408, shared expert 5632.
        let json = r#"{
            "architectures": ["Qwen2MoeForCausalLM"],
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "moe_intermediate_size": 1408,
            "shared_expert_intermediate_size": 5632,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "num_experts": 60,
            "num_experts_per_tok": 4,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0
        }"#;
        let cfg = parse_hf_config(json).unwrap();
        assert_eq!(cfg.arch, "qwen2moe");
        // d_ff is the PER-EXPERT width, not intermediate_size.
        assert_eq!(cfg.d_ff, 1408);
        let moe = cfg.moe.expect("MoE detected");
        assert_eq!(moe.n_experts, 60);
        assert_eq!(moe.n_experts_used, 4);
        // shared_expert_intermediate_size present → 1 shared expert.
        assert_eq!(moe.n_experts_shared, 1);
    }

    #[test]
    fn parses_mixtral_num_local_experts_without_moe_intermediate() {
        // Mixtral names the count `num_local_experts` and has no
        // `moe_intermediate_size` — experts use `intermediate_size`.
        let json = r#"{
            "architectures": ["MixtralForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "vocab_size": 32000
        }"#;
        let cfg = parse_hf_config(json).unwrap();
        let moe = cfg.moe.expect("MoE detected via num_local_experts");
        assert_eq!(moe.n_experts, 8);
        assert_eq!(moe.n_experts_used, 2);
        assert_eq!(moe.n_experts_shared, 0);
        // No moe_intermediate_size → fall back to intermediate_size.
        assert_eq!(cfg.d_ff, 14336);
    }

    #[test]
    fn dense_config_still_has_no_moe() {
        let json = r#"{
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "vocab_size": 152064
        }"#;
        let cfg = parse_hf_config(json).unwrap();
        assert!(cfg.moe.is_none());
        assert_eq!(cfg.d_ff, 18944);
    }

    #[test]
    fn head_dim_falls_back_to_d_model_over_n_heads() {
        // Llama 1: head_dim not present, derive from hidden_size / num_heads.
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000
        }"#;
        let cfg = parse_hf_config(json).unwrap();
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.rope_dim, 128);
    }
}
