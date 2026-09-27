//! BERT-family architecture parsing + loading. v1.1 foundation:
//! this module reads BERT GGUF metadata + extracts tensor references
//! into a typed [`BertWeights`]. The forward pass + pooling are a
//! follow-up turn; [`BertModel::forward_embed`] today returns
//! `NotImplemented` so the loader contract is testable end-to-end.
//!
//! BERT differs from the Llama family in ways that matter for the
//! architecture parsing:
//!
//!   - **Bidirectional attention** (no causal mask)
//!   - **LayerNorm** with learned bias, not RMSNorm (no rotation)
//!   - **No RoPE** — uses absolute position embeddings + a token-type
//!     ("segment") embedding that's added at the embedding layer
//!   - **GELU FFN** (not SwiGLU) — single up + down projection,
//!     activation in between
//!   - **Bias terms everywhere** — Q/K/V/O projections and FFN
//!     linears all carry bias tensors that Llama omits
//!   - **Pooling step** at the output: typically mean over tokens
//!     (BGE/E5) or CLS-token-only (some bert-base variants)
//!
//! GGUF metadata key conventions follow the `bert.*` prefix
//! (sometimes `nomic_bert.*` for Nomic-distributed models — both
//! handled here via the generic `<arch>.<key>` reader the same way
//! `LlamaConfig` does for the Llama family).

use rustllama_gguf::{Gguf, MetadataValue};
use rustllama_kernels_cpu as k;
use rustllama_tensor::{as_slice_f32, Device, Dtype, Tensor};

/// BERT-family hyperparameters extracted from GGUF metadata. Field
/// shapes mirror [`crate::llama_config::LlamaConfig`] where the
/// concept applies; BERT-only fields (`layer_norm_eps`,
/// `pooling_type`, `n_token_types`) carry the differences.
#[derive(Debug, Clone)]
pub struct BertConfig {
    /// Architecture string from `general.architecture` — typically
    /// `"bert"` for standard BERT or `"nomic_bert"` for the Nomic
    /// embedding variant. Used as the metadata-key prefix when
    /// other readers consult the same GGUF.
    pub arch: String,
    /// Number of transformer layers. From `<arch>.block_count`.
    pub n_layers: usize,
    /// Attention head count. BERT doesn't use GQA, so we don't
    /// carry a separate `n_kv_heads` field — Q/K/V all use the
    /// same head count.
    pub n_heads: usize,
    /// Hidden dimension (embedding size). 384 for small, 768 for
    /// base, 1024 for large. From `<arch>.embedding_length`.
    pub d_model: usize,
    /// Feed-forward intermediate dim. Typically 4× `d_model`.
    /// From `<arch>.feed_forward_length`.
    pub d_ff: usize,
    /// Per-head dim. Derived as `d_model / n_heads` when the
    /// explicit `<arch>.attention.key_length` key is absent.
    pub head_dim: usize,
    /// LayerNorm epsilon. From `<arch>.attention.layer_norm_epsilon`
    /// (NOT `layer_norm_rms_epsilon` — BERT uses standard LN,
    /// not RMSNorm). Default `1e-12` matches HuggingFace.
    pub layer_norm_eps: f32,
    /// Vocab size from `tokenizer.ggml.tokens`.
    pub vocab_size: usize,
    /// Max position embeddings supported. BERT typically caps at
    /// 512; some BGE/E5 variants extend to 2048 / 8192.
    pub ctx_train: usize,
    /// Number of token-type / segment IDs. Standard BERT uses 2
    /// (sentence-A / sentence-B); single-sentence-task models
    /// often use 1. From `<arch>.token_types_count` (a rustllama
    /// extension; default 2).
    pub n_token_types: usize,
    /// How to collapse per-token hidden states into a single
    /// embedding vector. The GGUF carries this in
    /// `<arch>.pooling_type` (rustllama convention: `"mean"` for
    /// average-pool, `"cls"` for first-token-only). Defaults to
    /// `"mean"` since BGE/E5 (the most common embedding GGUFs)
    /// use mean pooling.
    pub pooling_type: PoolingType,
}

/// How to derive a single embedding vector from per-token hidden
/// states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingType {
    /// Average over the sequence dimension. BGE / E5 / GTE.
    Mean,
    /// Take the first token's hidden state ([CLS] for BERT, [s]
    /// for some variants). Original BERT classification head.
    Cls,
}

impl PoolingType {
    fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "cls" | "cls_token" | "first" => Self::Cls,
            // Anything else → mean. BGE/E5 don't always set the
            // metadata key explicitly, and mean is the right
            // default for the embedding-model-in-GGUF case.
            _ => Self::Mean,
        }
    }
}

impl BertConfig {
    /// Parse BERT hyperparameters from a GGUF file's metadata.
    /// Errors on missing required fields (`block_count`,
    /// `embedding_length`, etc.) and on a vocab table that's
    /// neither in `tokenizer.ggml.tokens` nor `<arch>.vocab_size`.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self, BertConfigError> {
        let arch = gguf
            .architecture()
            .ok_or(BertConfigError::Missing("general.architecture"))?
            .to_string();
        // We accept both `bert` and `nomic_bert` — they share the
        // GGUF key shape. Future BERT-derived archs slot in the
        // same way. Reject everything else to avoid silently
        // mis-parsing a Llama as BERT.
        match arch.as_str() {
            "bert" | "nomic_bert" => {}
            other => {
                return Err(BertConfigError::UnsupportedArch(other.to_string()));
            }
        }

        let key = |name: &str| format!("{arch}.{name}");
        let n_layers = u32_required(gguf, &key("block_count"))? as usize;
        let d_model = u32_required(gguf, &key("embedding_length"))? as usize;
        let d_ff = u32_required(gguf, &key("feed_forward_length"))? as usize;
        let n_heads = u32_required(gguf, &key("attention.head_count"))? as usize;
        let head_dim = u32_optional(gguf, &key("attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or_else(|| d_model / n_heads.max(1));
        let layer_norm_eps =
            f32_optional(gguf, &key("attention.layer_norm_epsilon")).unwrap_or(1e-12);
        let ctx_train = u32_optional(gguf, &key("context_length")).unwrap_or(512) as usize;
        let n_token_types =
            u32_optional(gguf, &key("token_types_count")).unwrap_or(2) as usize;

        let vocab_size = match gguf.metadata_get("tokenizer.ggml.tokens") {
            Some(MetadataValue::Array(v)) => v.len(),
            _ => u32_optional(gguf, &key("vocab_size")).unwrap_or(0) as usize,
        };
        if vocab_size == 0 {
            return Err(BertConfigError::Missing(
                "tokenizer.ggml.tokens or <arch>.vocab_size",
            ));
        }

        let pooling_type = gguf
            .metadata_get(&key("pooling_type"))
            .and_then(|v| match v {
                MetadataValue::String(s) => Some(PoolingType::from_str(s)),
                _ => None,
            })
            .unwrap_or(PoolingType::Mean);

        Ok(Self {
            arch,
            n_layers,
            n_heads,
            d_model,
            d_ff,
            head_dim,
            layer_norm_eps,
            vocab_size,
            ctx_train,
            n_token_types,
            pooling_type,
        })
    }
}

/// Per-layer BERT weight tensor references. Held as borrowed
/// [`Tensor`] handles into the GGUF mmap — no copying. The forward
/// pass (next turn) consumes these via the existing matvec / norm
/// dispatch helpers.
#[derive(Debug)]
pub struct BertLayerWeights {
    pub attn_q: Tensor,
    pub attn_q_bias: Option<Tensor>,
    pub attn_k: Tensor,
    pub attn_k_bias: Option<Tensor>,
    pub attn_v: Tensor,
    pub attn_v_bias: Option<Tensor>,
    pub attn_output: Tensor,
    pub attn_output_bias: Option<Tensor>,
    /// LayerNorm AFTER the attention residual (post-LN convention,
    /// which BERT uses — distinct from the pre-LN convention some
    /// later transformers adopt).
    pub attn_output_norm: Tensor,
    pub attn_output_norm_bias: Tensor,
    pub ffn_up: Tensor,
    pub ffn_up_bias: Option<Tensor>,
    pub ffn_down: Tensor,
    pub ffn_down_bias: Option<Tensor>,
    /// LayerNorm AFTER the FFN residual (the second post-LN in
    /// each BERT block).
    pub layer_output_norm: Tensor,
    pub layer_output_norm_bias: Tensor,
}

/// Classification head bound onto a BERT body for reranker /
/// classifier tasks. Present in BGE-reranker GGUFs (shipped under
/// the `cls.weight` / `cls.bias` tensor names by convention) and
/// in any sequence-classification-task BERT derivative. Absent
/// for pure embedding models — `forward_classify` errors with
/// `NoClassifierHead` when called on a model without one.
///
/// Shape: `weight` is `[d_model, n_labels]` (BGE-reranker uses
/// `n_labels = 1` and reads a single relevance scalar; multi-
/// class rerankers use a wider head).
#[derive(Debug)]
pub struct ClassifierHead {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub n_labels: usize,
}

/// Loaded BERT model. The embedding-layer weights + N per-layer
/// weight bundles live here; pooling + the forward pass come
/// next turn.
#[derive(Debug)]
pub struct BertModel {
    pub cfg: BertConfig,
    pub token_embd: Tensor,
    /// Token-type / segment embedding. Standard BERT has shape
    /// `[2, d_model]`; single-sentence-task models often have
    /// shape `[1, d_model]`. Optional because some embedding-
    /// focused BERT variants drop it entirely.
    pub token_types: Option<Tensor>,
    /// Absolute position embedding `[ctx_train, d_model]`. Always
    /// present in standard BERT (RoPE-based BERT derivatives like
    /// some recent extensions would set this to None — out of
    /// scope for v1).
    pub position_embd: Tensor,
    pub token_embd_norm: Tensor,
    pub token_embd_norm_bias: Tensor,
    pub layers: Vec<BertLayerWeights>,
    /// Classification head — present on reranker / sequence-
    /// classification GGUFs, absent on pure embedding models.
    /// Loaded from `cls.weight` / `cls.bias` (preferred — the
    /// llama.cpp convention) with a fallback to
    /// `classifier.weight` / `classifier.bias` for converters
    /// that preserve the HF naming.
    pub classifier_head: Option<ClassifierHead>,
}

impl BertModel {
    /// Load a BERT model from a GGUF file. Reads the config + binds
    /// all weight tensors. Returns an error on missing required
    /// tensors or arch mismatch; never validates dtype (the forward
    /// pass — when implemented — will handle dtype routing the same
    /// way the Llama path does via `matvec_tensor_dispatch`).
    pub fn load(gguf: &Gguf) -> Result<Self, BertLoadError> {
        let cfg = BertConfig::from_gguf(gguf).map_err(BertLoadError::Config)?;
        let required = |name: &str| -> Result<Tensor, BertLoadError> {
            let info = gguf
                .tensor(name)
                .ok_or_else(|| BertLoadError::MissingTensor(name.to_string()))?;
            Ok(tensor_from_info(gguf, info, name))
        };
        let optional = |name: &str| -> Option<Tensor> {
            gguf.tensor(name).map(|info| tensor_from_info(gguf, info, name))
        };

        let token_embd = required("token_embd.weight")?;
        let token_types = optional("token_types.weight");
        let position_embd = required("position_embd.weight")?;
        let token_embd_norm = required("token_embd_norm.weight")?;
        let token_embd_norm_bias = required("token_embd_norm.bias")?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |suffix: &str| format!("blk.{i}.{suffix}");
            layers.push(BertLayerWeights {
                attn_q: required(&p("attn_q.weight"))?,
                attn_q_bias: optional(&p("attn_q.bias")),
                attn_k: required(&p("attn_k.weight"))?,
                attn_k_bias: optional(&p("attn_k.bias")),
                attn_v: required(&p("attn_v.weight"))?,
                attn_v_bias: optional(&p("attn_v.bias")),
                attn_output: required(&p("attn_output.weight"))?,
                attn_output_bias: optional(&p("attn_output.bias")),
                attn_output_norm: required(&p("attn_output_norm.weight"))?,
                attn_output_norm_bias: required(&p("attn_output_norm.bias"))?,
                ffn_up: required(&p("ffn_up.weight"))?,
                ffn_up_bias: optional(&p("ffn_up.bias")),
                ffn_down: required(&p("ffn_down.weight"))?,
                ffn_down_bias: optional(&p("ffn_down.bias")),
                layer_output_norm: required(&p("layer_output_norm.weight"))?,
                layer_output_norm_bias: required(&p("layer_output_norm.bias"))?,
            });
        }

        // Classification head — optional. Try the llama.cpp
        // convention first (`cls.weight` / `cls.bias`), then the
        // HF-preserving convention (`classifier.weight` /
        // `classifier.bias`). When neither is present this is a
        // pure embedding model and `forward_classify` will return
        // NoClassifierHead.
        let classifier_head = optional("cls.weight")
            .map(|w| (w, optional("cls.bias")))
            .or_else(|| {
                optional("classifier.weight").map(|w| (w, optional("classifier.bias")))
            })
            .map(|(weight, bias)| {
                // Weight shape is `[d_model, n_labels]` — derive
                // n_labels from the second dim (or fall back to 1
                // for a degenerate single-scalar head).
                let n_labels = match weight.shape.get(1).copied() {
                    Some(n) if n > 0 => n as usize,
                    _ => 1,
                };
                ClassifierHead {
                    weight,
                    bias,
                    n_labels,
                }
            });

        Ok(Self {
            cfg,
            token_embd,
            token_types,
            position_embd,
            token_embd_norm,
            token_embd_norm_bias,
            layers,
            classifier_head,
        })
    }

    /// Run the embedding forward pass over `tokens` and return a
    /// pooled embedding vector of length `cfg.d_model`. Per
    /// `cfg.pooling_type`: mean over tokens (BGE/E5) or CLS-token-
    /// only (vanilla BERT).
    ///
    /// Shape:
    ///   - input: `[N]` token ids (all positive, < vocab_size)
    ///   - output: `[d_model]` f32 (the pooled embedding)
    ///
    /// **Performance**: v1.1 foundation runs entirely on CPU with
    /// per-token matvecs. GPU dispatch via `try_matvec_*_usm_f32`
    /// is automatic for quantized embedding-model weights (Q4/Q8)
    /// since `matvec_tensor` already routes through that path —
    /// but the bidirectional attention + LayerNorm + GELU are
    /// pure CPU for now. A follow-up turn can wire the per-block
    /// hot paths to USM kernels the same way the Llama forward
    /// does, but for BGE-small / E5-base-class models the CPU
    /// pass is already fast enough for interactive RAG.
    pub fn forward_embed(&self, tokens: &[i32]) -> Result<Vec<f32>, BertForwardError> {
        let x = self.forward_layers(tokens)?;
        let n = tokens.len();
        let d = self.cfg.d_model;

        let mut pooled = vec![0f32; d];
        match self.cfg.pooling_type {
            PoolingType::Mean => {
                for i in 0..n {
                    for j in 0..d {
                        pooled[j] += x[i * d + j];
                    }
                }
                let inv_n = 1.0 / n as f32;
                for v in &mut pooled {
                    *v *= inv_n;
                }
            }
            PoolingType::Cls => {
                pooled.copy_from_slice(&x[0..d]);
            }
        }
        Ok(pooled)
    }

    /// Reranker / sequence-classification forward pass. Runs the
    /// BERT body, takes the CLS position output, and projects
    /// through the loaded classifier head.
    ///
    /// Returns a length-`n_labels` score vector. For BGE-reranker
    /// (`n_labels = 1`) this is a single relevance scalar — the
    /// caller can apply a sigmoid to map it to a probability if
    /// desired, but for ranking purposes the raw logit is enough.
    ///
    /// Errors with `NoClassifierHead` if the model wasn't loaded
    /// from a reranker GGUF (i.e. neither `cls.weight` nor
    /// `classifier.weight` was present).
    pub fn forward_classify(&self, tokens: &[i32]) -> Result<Vec<f32>, BertForwardError> {
        let head = self
            .classifier_head
            .as_ref()
            .ok_or(BertForwardError::NoClassifierHead)?;
        let x = self.forward_layers(tokens)?;
        let d = self.cfg.d_model;
        // Always CLS for classification — `pooling_type` only
        // governs the embedding-pooling path.
        let cls = &x[0..d];
        let mut out = vec![0f32; head.n_labels];
        k::matvec_tensor(&head.weight, cls, &mut out, head.n_labels, d);
        if let Some(b) = &head.bias {
            add_bias_f32(&mut out, b);
        }
        Ok(out)
    }

    /// Run the BERT body (embedding + pre-LN + N transformer blocks)
    /// and return the final hidden states as a flat
    /// `[n * d_model]` f32 buffer. Shared by `forward_embed` and
    /// `forward_classify`; pooling / classification heads consume
    /// it differently.
    fn forward_layers(&self, tokens: &[i32]) -> Result<Vec<f32>, BertForwardError> {
        let n = tokens.len();
        if n == 0 {
            return Err(BertForwardError::EmptyInput);
        }
        if n > self.cfg.ctx_train {
            return Err(BertForwardError::SequenceTooLong {
                n,
                max: self.cfg.ctx_train,
            });
        }
        for &id in tokens {
            if id < 0 || (id as usize) >= self.cfg.vocab_size {
                return Err(BertForwardError::TokenOutOfVocab {
                    id,
                    vocab: self.cfg.vocab_size,
                });
            }
        }

        let d = self.cfg.d_model;
        let d_ff = self.cfg.d_ff;
        let n_heads = self.cfg.n_heads;
        let head_dim = self.cfg.head_dim;
        let eps = self.cfg.layer_norm_eps;

        // ---- 1. Embedding layer ---------------------------------
        // x[i, :] = token_embd[tokens[i]] + token_types[0] + position_embd[i]
        // Token-type 0 for all tokens since this is single-sentence
        // embedding (no sentence-A/B distinction in RAG flows).
        let mut x = vec![0f32; n * d];
        for i in 0..n {
            k::embed_lookup_tensor(
                &self.token_embd,
                &tokens[i..i + 1],
                &mut x[i * d..(i + 1) * d],
                d,
            );
            if let Some(tt) = &self.token_types {
                add_embedding_row(tt, 0, &mut x[i * d..(i + 1) * d]);
            }
            add_embedding_row(&self.position_embd, i as i32, &mut x[i * d..(i + 1) * d]);
        }

        // ---- 2. Pre-block LayerNorm (BERT applies LN to the
        //         summed embeddings before any transformer block).
        let pre_w = as_slice_f32(&self.token_embd_norm);
        let pre_b = as_slice_f32(&self.token_embd_norm_bias);
        for i in 0..n {
            layer_norm_with_bias(&mut x[i * d..(i + 1) * d], pre_w, pre_b, eps);
        }

        // Reusable per-block scratch.
        let mut q = vec![0f32; n * d];
        let mut k_buf = vec![0f32; n * d];
        let mut v = vec![0f32; n * d];
        let mut attn_out = vec![0f32; n * d];
        let mut o_proj = vec![0f32; n * d];
        let mut ff = vec![0f32; n * d_ff];
        let mut ff_down = vec![0f32; n * d];
        let mut tmp_row = vec![0f32; d];
        let _ = &mut tmp_row;

        // ---- 3. N transformer blocks ----------------------------
        for layer in &self.layers {
            // Q/K/V projections, per token (loop matvec — correct +
            // dtype-agnostic via `matvec_tensor`'s built-in dispatch).
            for i in 0..n {
                let xi = &x[i * d..(i + 1) * d];
                k::matvec_tensor(&layer.attn_q, xi, &mut q[i * d..(i + 1) * d], d, d);
                if let Some(b) = &layer.attn_q_bias {
                    add_bias_f32(&mut q[i * d..(i + 1) * d], b);
                }
                k::matvec_tensor(&layer.attn_k, xi, &mut k_buf[i * d..(i + 1) * d], d, d);
                if let Some(b) = &layer.attn_k_bias {
                    add_bias_f32(&mut k_buf[i * d..(i + 1) * d], b);
                }
                k::matvec_tensor(&layer.attn_v, xi, &mut v[i * d..(i + 1) * d], d, d);
                if let Some(b) = &layer.attn_v_bias {
                    add_bias_f32(&mut v[i * d..(i + 1) * d], b);
                }
            }

            // Bidirectional attention (no causal mask — the whole
            // point of BERT). Per-head, per-query-position softmax
            // over all key positions; accumulates V to attn_out.
            bidirectional_attention(
                &q,
                &k_buf,
                &v,
                &mut attn_out,
                n,
                n_heads,
                head_dim,
            );

            // Output projection + bias.
            for i in 0..n {
                let ai = &attn_out[i * d..(i + 1) * d];
                k::matvec_tensor(
                    &layer.attn_output,
                    ai,
                    &mut o_proj[i * d..(i + 1) * d],
                    d,
                    d,
                );
                if let Some(b) = &layer.attn_output_bias {
                    add_bias_f32(&mut o_proj[i * d..(i + 1) * d], b);
                }
            }

            // Residual + post-attn LayerNorm. BERT uses POST-LN
            // (norm AFTER the residual), distinct from the pre-LN
            // convention some later transformers adopt.
            let pa_w = as_slice_f32(&layer.attn_output_norm);
            let pa_b = as_slice_f32(&layer.attn_output_norm_bias);
            for i in 0..n {
                for j in 0..d {
                    x[i * d + j] += o_proj[i * d + j];
                }
                layer_norm_with_bias(&mut x[i * d..(i + 1) * d], pa_w, pa_b, eps);
            }

            // FFN: up → GELU → down, per token.
            for i in 0..n {
                let xi = &x[i * d..(i + 1) * d];
                k::matvec_tensor(
                    &layer.ffn_up,
                    xi,
                    &mut ff[i * d_ff..(i + 1) * d_ff],
                    d_ff,
                    d,
                );
                if let Some(b) = &layer.ffn_up_bias {
                    add_bias_f32(&mut ff[i * d_ff..(i + 1) * d_ff], b);
                }
                for v in &mut ff[i * d_ff..(i + 1) * d_ff] {
                    *v = gelu_tanh_approx(*v);
                }
                k::matvec_tensor(
                    &layer.ffn_down,
                    &ff[i * d_ff..(i + 1) * d_ff],
                    &mut ff_down[i * d..(i + 1) * d],
                    d,
                    d_ff,
                );
                if let Some(b) = &layer.ffn_down_bias {
                    add_bias_f32(&mut ff_down[i * d..(i + 1) * d], b);
                }
            }

            // Residual + post-FFN LayerNorm.
            let lo_w = as_slice_f32(&layer.layer_output_norm);
            let lo_b = as_slice_f32(&layer.layer_output_norm_bias);
            for i in 0..n {
                for j in 0..d {
                    x[i * d + j] += ff_down[i * d + j];
                }
                layer_norm_with_bias(&mut x[i * d..(i + 1) * d], lo_w, lo_b, eps);
            }
        }

        Ok(x)
    }
}

// ----- Forward-pass helpers ----------------------------------------

/// Add one row of an embedding table to `dst` (no overwrite — `dst`
/// already holds prior summed terms). Uses [`k::embed_lookup_tensor`]
/// for the per-dtype gather, then accumulates into `dst`. Avoids
/// allocating a fresh scratch per call by reusing a stack-sized
/// local buffer — `dst.len()` here is `d_model` which is bounded
/// by the BERT-family sizes (typically ≤ 1024).
fn add_embedding_row(table: &Tensor, id: i32, dst: &mut [f32]) {
    let d = dst.len();
    // Per-call alloc. For embedding-model sizes (d ≤ 1024) the
    // allocator hit is negligible vs the matvec costs.
    let mut tmp = vec![0f32; d];
    k::embed_lookup_tensor(table, &[id], &mut tmp, d);
    for (dst_i, &t) in dst.iter_mut().zip(tmp.iter()) {
        *dst_i += t;
    }
}

/// In-place LayerNorm with learned scale + bias (BERT convention,
/// distinct from RMSNorm). `weight` and `bias` must each have
/// length equal to `x.len()`.
pub(crate) fn layer_norm_with_bias(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    let n = x.len();
    debug_assert_eq!(weight.len(), n, "LN weight length mismatch");
    debug_assert_eq!(bias.len(), n, "LN bias length mismatch");
    let mean = x.iter().copied().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let inv_std = (var + eps).sqrt().recip();
    for (i, v) in x.iter_mut().enumerate() {
        *v = (*v - mean) * inv_std * weight[i] + bias[i];
    }
}

/// Add a bias vector element-wise. Skipped via `Option` at the
/// call site for layers that omit it (rare in BERT but possible).
pub(crate) fn add_bias_f32(x: &mut [f32], bias: &Tensor) {
    let b = as_slice_f32(bias);
    debug_assert_eq!(b.len(), x.len(), "bias length mismatch");
    for (xi, &bi) in x.iter_mut().zip(b.iter()) {
        *xi += bi;
    }
}

/// GELU activation, "tanh approximation" form used by BERT + BGE +
/// E5 + GTE. Matches the reference within ~1e-7. The "exact" GELU
/// (using erf) produces near-identical embeddings, but BERT
/// implementations standardized on this form so weights are
/// trained against it.
pub(crate) fn gelu_tanh_approx(x: f32) -> f32 {
    const C: f32 = 0.7978845608; // sqrt(2/π)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

/// Bidirectional multi-head self-attention. Per head, for each query
/// position, computes `softmax(Q @ K^T / sqrt(head_dim)) @ V` over
/// ALL key positions (no causal mask — the headline difference from
/// the Llama forward path).
///
/// Buffers `q`, `k`, `v`, `out` are all shape `[n, n_heads*head_dim]`
/// row-major. Heads are interleaved within each row.
pub(crate) fn bidirectional_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n: usize,
    n_heads: usize,
    head_dim: usize,
) {
    let d = n_heads * head_dim;
    let scale = (head_dim as f32).sqrt().recip();
    // Per-query scratch reused across the inner loops.
    let mut scores = vec![0f32; n];
    for h in 0..n_heads {
        let head_off = h * head_dim;
        for qi in 0..n {
            // 1. Dot products Q[qi, head_off..head_off+head_dim] · K[ki, …]
            for (ki, score) in scores.iter_mut().enumerate() {
                let mut dot = 0f32;
                for d_idx in 0..head_dim {
                    dot += q[qi * d + head_off + d_idx]
                        * k[ki * d + head_off + d_idx];
                }
                *score = dot * scale;
            }
            // 2. Softmax over the n scores.
            k_softmax::softmax_f32_inplace(&mut scores);
            // 3. Weighted sum of V rows for this head.
            for d_idx in 0..head_dim {
                let mut sum = 0f32;
                for (ki, &s) in scores.iter().enumerate() {
                    sum += s * v[ki * d + head_off + d_idx];
                }
                out[qi * d + head_off + d_idx] = sum;
            }
        }
    }
}

// `softmax_f32_inplace` is the reusable CPU kernel. Aliased
// here for clarity at the bidirectional_attention call site.
mod k_softmax {
    pub use rustllama_kernels_cpu::softmax_f32_inplace;
}

fn tensor_from_info(
    gguf: &Gguf,
    info: &rustllama_gguf::TensorInfo,
    name: &str,
) -> Tensor {
    // Copy GGUF mmap bytes into an `Arc<[u8]>` for `Storage::CpuOwned`
    // (the only storage variant rustllama-tensor exposes today).
    // Llama loader does the same — the mmap → owned copy doubles
    // the working set on load but keeps the Tensor self-contained
    // so the GGUF mmap can drop afterward.
    use rustllama_tensor::Storage;
    let bytes = gguf
        .tensor_bytes(name)
        .expect("tensor_bytes for known-present info");
    let storage = Storage::CpuOwned(bytes.to_vec().into());
    let dtype = ggml_type_to_dtype(info.dtype);
    let shape: Vec<u64> = info.dims.clone();
    let strides = contiguous_strides_for(&shape);
    Tensor {
        device: Device::Cpu,
        dtype,
        shape,
        strides,
        storage,
        name: name.to_string(),
    }
}

fn ggml_type_to_dtype(t: rustllama_gguf::GgmlType) -> Dtype {
    use rustllama_gguf::GgmlType as G;
    match t {
        G::F32 => Dtype::F32,
        G::F16 => Dtype::F16,
        G::Bf16 => Dtype::Bf16Raw,
        G::Q4_0 => Dtype::Q4_0Raw,
        G::Q4_1 => Dtype::Q4_1Raw,
        G::Q5_0 => Dtype::Q5_0Raw,
        G::Q5_1 => Dtype::Q5_1Raw,
        G::Q8_0 => Dtype::Q8_0Raw,
        G::Q2_K => Dtype::Q2_KRaw,
        G::Q3_K => Dtype::Q3_KRaw,
        G::Q4_K => Dtype::Q4_KRaw,
        G::Q5_K => Dtype::Q5_KRaw,
        G::Q6_K => Dtype::Q6_KRaw,
        G::Q8_K => Dtype::Q8_KRaw,
        // The IQ-family + sub-4-bit quants exist on the dtype side
        // but we don't expect embedding models to ship in those
        // formats (they're chat-LLM optimizations). Pass through
        // as F32 — if a model surprises us the forward pass (when
        // implemented) will route through the matvec_tensor
        // dispatch which handles every dtype.
        _ => Dtype::F32,
    }
}

fn contiguous_strides_for(shape: &[u64]) -> Vec<i64> {
    // rustllama-tensor uses `pub type Strides = Vec<i64>` to
    // permit negative strides (reversed views) in the future. We
    // emit the standard row-major positive strides.
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as i64;
    }
    strides
}

#[derive(Debug, thiserror::Error)]
pub enum BertConfigError {
    #[error("missing required GGUF metadata key: {0}")]
    Missing(&'static str),
    #[error("GGUF architecture {0:?} is not a BERT-family arch (expected `bert` or `nomic_bert`)")]
    UnsupportedArch(String),
    #[error("metadata key {0} has unexpected type")]
    BadType(String),
}

#[derive(Debug, thiserror::Error)]
pub enum BertLoadError {
    #[error("config parse: {0}")]
    Config(#[from] BertConfigError),
    #[error("missing required tensor: {0}")]
    MissingTensor(String),
}

#[derive(Debug, thiserror::Error)]
pub enum BertForwardError {
    #[error("empty input — embedding requires at least one token")]
    EmptyInput,
    #[error("sequence too long: {n} tokens, model trained for {max}")]
    SequenceTooLong { n: usize, max: usize },
    #[error("token id {id} out of vocab (model has {vocab} tokens)")]
    TokenOutOfVocab { id: i32, vocab: usize },
    #[error(
        "model has no classifier head — `forward_classify` requires a reranker-style \
         GGUF with `cls.weight` (or `classifier.weight`)"
    )]
    NoClassifierHead,
}

// Local copies of the GGUF metadata readers. Mirrors what
// `LlamaConfig` does so this module stays self-contained — both
// files can evolve independently.

fn u32_required(gguf: &Gguf, key: &str) -> Result<u32, BertConfigError> {
    match gguf.metadata_get(key) {
        Some(MetadataValue::U32(v)) => Ok(*v),
        Some(MetadataValue::U64(v)) if *v <= u32::MAX as u64 => Ok(*v as u32),
        Some(MetadataValue::I32(v)) if *v >= 0 => Ok(*v as u32),
        Some(MetadataValue::I64(v)) if *v >= 0 && *v <= u32::MAX as i64 => Ok(*v as u32),
        Some(_) => Err(BertConfigError::BadType(key.to_string())),
        None => Err(BertConfigError::Missing(Box::leak(
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
