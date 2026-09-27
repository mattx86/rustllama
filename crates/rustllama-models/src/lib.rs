//! Model trait and per-architecture implementations.
//!
//! Phase 1 ships `llama_arch::LlamaModel`, a pure-Rust forward pass that
//! covers Llama / Mistral / Mistral 3 / Qwen2 / DeepSeek / Phi-3 / Yi.

use rustllama_tensor::{Device, Tensor};

pub mod accel;
pub mod bert_arch;
pub mod imatrix_collect;
pub mod kv_bias;
pub mod llama_arch;
pub mod llama_config;
pub mod mamba_arch;
pub mod moe;
pub mod page_table;
pub mod paged_kv_cache;
pub mod paged_kv_store;
pub mod shared_paged_kv;
pub mod vision_arch;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelArch {
    /// Covers Llama, Llama-2, Llama-3, Mistral, Mistral 3.x, Qwen2,
    /// Qwen2.5-Coder, DeepSeek-V2/Coder-V2, Phi-3, Yi.
    ///
    /// All of these share the same forward-pass shape from our
    /// engine's perspective: pre-norm RMSNorm + GQA attention with
    /// RoPE + SwiGLU FFN + tied or untied LM head. They differ in
    /// hyperparameters (`d_model`, `n_heads`, `rope.freq_base`,
    /// `ctx_train`) which are read straight from the GGUF metadata —
    /// no per-arch code path is needed for compute.
    ///
    /// Mistral 3.x note: the upstream model uses interleaved
    /// sliding-window attention (4 K window per layer). Our engine
    /// runs full attention instead — correct but ~2-4× slower at
    /// long contexts. The fused SWA forward pass is a v1.x follow-up.
    Llama,
    /// MoE variants — placeholder for v2.
    LlamaMoe,
    Unknown(String),
}

impl ModelArch {
    pub fn from_gguf(arch: Option<&str>) -> Self {
        match arch {
            Some(
                "llama" | "mistral" | "mistral3" | "qwen2" | "qwen2.5" | "qwen2_5" | "qwen3"
                | "qwen35" | "qwen3_5" | "deepseek" | "deepseek2" | "phi3" | "yi",
            ) => Self::Llama,
            // Transformer-MoE archs that share the rustllama attention +
            // per-layer MoE forward path. NOTE: this enum is vestigial —
            // the real loader (`LlamaConfig::from_gguf`) is arch-string-
            // agnostic and metadata-driven, so these GGUFs load and
            // inference regardless of this classification. Qwen3 / Qwen3.5
            // MoE (incl. the hybrid DeltaNet-SSM + MTP variants that back
            // Ornith 1.5) run through the regular engine boundary today;
            // the earlier "needs SSM/MTP wiring first" caveat is resolved.
            Some(
                "qwen2_moe" | "qwen3moe" | "qwen35moe" | "qwen3_5_moe" | "mixtral"
                | "deepseek_v3",
            ) => Self::LlamaMoe,
            Some(other) => Self::Unknown(other.to_string()),
            None => Self::Unknown(String::new()),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("unsupported model architecture: {0:?}")]
    UnsupportedArch(ModelArch),
    #[error("required tensor missing from GGUF: {0}")]
    MissingTensor(String),
    #[error("llama load: {0}")]
    LlamaLoad(#[from] llama_arch::LlamaLoadError),
}

pub type Result<T> = std::result::Result<T, ModelError>;

#[derive(Debug, Clone)]
pub struct Batch {
    pub tokens: Vec<i32>,
    pub positions: Vec<i32>,
}

#[derive(Debug, Default)]
pub struct KvCache {
    pub seq_len: u32,
}

pub trait Model: Send + Sync {
    fn n_layers(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn device(&self) -> &Device;
    fn forward(&self, batch: &Batch, kv: &mut KvCache, out: &mut Tensor) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_gguf_recognizes_llama_family_archs() {
        // Llama-family (single-engine forward pass): all map to
        // `Llama`. Pinning each entry so a regression in the match
        // arm gets caught without having to grep the source.
        for arch in [
            "llama", "mistral", "mistral3", "qwen2", "qwen2.5", "qwen2_5", "qwen3", "qwen35",
            "qwen3_5", "deepseek", "deepseek2", "phi3", "yi",
        ] {
            assert_eq!(
                ModelArch::from_gguf(Some(arch)),
                ModelArch::Llama,
                "{arch} should map to Llama"
            );
        }
    }

    #[test]
    fn from_gguf_recognizes_moe_archs() {
        for arch in ["qwen2_moe", "qwen3moe", "qwen35moe", "qwen3_5_moe", "mixtral", "deepseek_v3"] {
            assert_eq!(
                ModelArch::from_gguf(Some(arch)),
                ModelArch::LlamaMoe,
                "{arch} should map to LlamaMoe"
            );
        }
    }

    #[test]
    fn from_gguf_unknown_passes_through_with_name() {
        // Unknown architectures retain the raw string so the engine
        // can surface a clear "unsupported arch X" error to the user
        // instead of silently falling back to Llama.
        assert_eq!(
            ModelArch::from_gguf(Some("gemma3")),
            ModelArch::Unknown("gemma3".to_string()),
        );
        assert_eq!(
            ModelArch::from_gguf(None),
            ModelArch::Unknown(String::new()),
        );
    }
}
