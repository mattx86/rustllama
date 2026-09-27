//! Mamba / State-Space Model (SSM) architecture — CPU forward pass.
//!
//! Mamba replaces the standard transformer self-attention with a
//! **selective scan** over a structured state-space model. Each
//! decoder layer runs:
//!
//!   1. RMSNorm over the residual stream.
//!   2. `in_proj`: project to `(x, z)` where each is `[d_inner]`.
//!   3. **Conv1d** of width `d_conv` over the per-channel `x`
//!      stream — local mixing across nearby tokens via a rolling
//!      per-channel state buffer.
//!   4. `x_proj` of the post-Conv1d activation produces
//!      `(dt_pre, B, C)`:
//!        - `dt_pre` ∈ ℝ^dt_rank, projected through `dt_proj` and
//!          softplus'd to per-channel time-steps `dt` ∈ ℝ^d_inner.
//!        - `B, C` ∈ ℝ^d_state — per-token modulation matrices.
//!   5. **Selective scan**: per channel, the recurrence
//!        `s ← exp(dt · A) · s + dt · B · x_post_conv`
//!        `y ← Cᵀ · s + D · x_post_conv`
//!      runs over the SSM state vector `s ∈ ℝ^d_state` carried in
//!      [`MambaCache`].
//!   6. Gate by `SiLU(z)`.
//!   7. `out_proj`: back to `[d_model]`; add to residual.
//!
//! Final `output_norm` + LM head produce logits.
//!
//! ## V1 status
//!
//! - **CPU forward**: shipped. Serial selective scan (no parallel-
//!   scan kernel); F32 activation throughout with F16/F32 weight
//!   dequant on-the-fly. Tests cover Conv1d state lifecycle +
//!   selective-scan equivalence to the reference recurrence.
//! - **GPU forward**: deferred. The selective scan needs a SYCL
//!   parallel-scan kernel (Hillel-prefix), which is a focused
//!   follow-up turn.
//! - **Prefill**: implemented as a serial loop over tokens calling
//!   `forward_one` repeatedly. A true parallel prefill needs the
//!   same parallel-scan kernel as the GPU path.

use rustllama_gguf::{GgmlType, Gguf, MetadataValue};

/// Read a u32-like metadata field, accepting any of u32/u64/i32.
fn meta_u32(gguf: &Gguf, key: &str) -> Option<u32> {
    match gguf.metadata_get(key)? {
        MetadataValue::U32(v) => Some(*v),
        MetadataValue::U64(v) => Some(*v as u32),
        MetadataValue::I32(v) => Some(*v as u32),
        _ => None,
    }
}

fn meta_f32(gguf: &Gguf, key: &str) -> Option<f32> {
    match gguf.metadata_get(key)? {
        MetadataValue::F32(v) => Some(*v),
        _ => None,
    }
}

/// Mamba / SSM hyperparameters parsed from GGUF metadata.
#[derive(Debug, Clone)]
pub struct MambaConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    /// SSM state dimension per channel. Typically 16 in Mamba-1,
    /// 128 in Mamba-2.
    pub d_state: usize,
    /// Conv1d kernel size. Typically 4.
    pub d_conv: usize,
    /// Inner expansion dimension (2 × d_model in canonical Mamba).
    pub d_inner: usize,
    /// `dt_proj` low-rank projection rank. Typically `ceil(d_model/16)`.
    pub dt_rank: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum MambaError {
    #[error("GGUF metadata missing required Mamba field: {0}")]
    MissingMetadata(&'static str),
    #[error("Mamba weight tensor missing: {0}")]
    MissingTensor(String),
    #[error("Mamba weight dtype unsupported: {got:?} for tensor {tensor}")]
    UnsupportedDtype { tensor: String, got: GgmlType },
    #[error("Mamba weight shape mismatch on tensor {tensor}: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        tensor: String,
        expected: Vec<u64>,
        got: Vec<u64>,
    },
    #[error("Mamba GGUF detected (arch = {0}) but no forward path enabled for this build")]
    LoadRefused(String),
}

impl MambaConfig {
    /// Detect a Mamba checkpoint from GGUF metadata. Returns the
    /// architecture name when recognized; `None` otherwise.
    pub fn detect(gguf: &Gguf) -> Option<String> {
        let arch = gguf.architecture()?.to_lowercase();
        if !matches!(arch.as_str(), "mamba" | "mamba2" | "jamba" | "falcon-mamba") {
            return None;
        }
        let has_state = meta_u32(gguf, &format!("{arch}.ssm.state_size")).is_some();
        let has_conv = meta_u32(gguf, &format!("{arch}.ssm.conv_kernel")).is_some();
        if has_state || has_conv {
            Some(arch)
        } else {
            None
        }
    }

    /// Parse the SSM hyperparameter set from GGUF metadata.
    pub fn from_gguf(gguf: &Gguf, arch: &str) -> Result<Self, MambaError> {
        let get_u32 = |key: &str| -> Option<u32> { meta_u32(gguf, key) };
        let get_f32 = |key: &str| -> Option<f32> { meta_f32(gguf, key) };
        let d_model = get_u32(&format!("{arch}.embedding_length"))
            .ok_or(MambaError::MissingMetadata("embedding_length"))?
            as usize;
        let n_layers = get_u32(&format!("{arch}.block_count"))
            .ok_or(MambaError::MissingMetadata("block_count"))? as usize;
        let vocab_size = get_u32(&format!("{arch}.vocab_size"))
            .or_else(|| get_u32("tokenizer.ggml.vocab_size"))
            .ok_or(MambaError::MissingMetadata("vocab_size"))? as usize;
        let d_state = get_u32(&format!("{arch}.ssm.state_size"))
            .ok_or(MambaError::MissingMetadata("ssm.state_size"))? as usize;
        let d_conv = get_u32(&format!("{arch}.ssm.conv_kernel"))
            .ok_or(MambaError::MissingMetadata("ssm.conv_kernel"))? as usize;
        let d_inner = get_u32(&format!("{arch}.ssm.inner_size"))
            .map(|v| v as usize)
            .unwrap_or(2 * d_model);
        let dt_rank = get_u32(&format!("{arch}.ssm.time_step_rank"))
            .map(|v| v as usize)
            .unwrap_or((d_model + 15) / 16);
        let eps = get_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .or_else(|| get_f32(&format!("{arch}.ssm.layer_norm_rms_epsilon")))
            .unwrap_or(1e-5);
        Ok(Self {
            d_model,
            n_layers,
            vocab_size,
            d_state,
            d_conv,
            d_inner,
            dt_rank,
            eps,
        })
    }
}

/// Per-layer weight set. All tensors are stored as `Vec<f32>` after
/// dequant — Mamba's weights are typically small enough (≤300M
/// params for a 1B-class model) that the F32-resident cost is fine
/// for v1. A follow-up turn keeps weights packed in their GGUF
/// dtype and dequants on-demand per matvec.
#[derive(Debug)]
pub struct MambaLayerWeights {
    /// RMSNorm scale. Shape `[d_model]`.
    pub attn_norm: Vec<f32>,
    /// in_proj. Shape `[2 * d_inner, d_model]` row-major:
    /// `out = w · x`. The first `d_inner` rows produce x; the
    /// next `d_inner` rows produce z.
    pub in_proj: Vec<f32>,
    /// Conv1d weights. Shape `[d_inner, d_conv]` row-major:
    /// each channel has its own `d_conv`-tap filter.
    pub conv1d_weight: Vec<f32>,
    /// Conv1d bias. Shape `[d_inner]`.
    pub conv1d_bias: Vec<f32>,
    /// x_proj. Shape `[dt_rank + 2 * d_state, d_inner]` row-major.
    /// Produces (dt_pre, B, C) per channel.
    pub x_proj: Vec<f32>,
    /// dt_proj. Shape `[d_inner, dt_rank]` row-major: low-rank
    /// projection of dt_pre.
    pub dt_proj: Vec<f32>,
    /// dt_proj bias. Shape `[d_inner]`. The softplus operates on
    /// `dt_proj · dt_pre + dt_bias`.
    pub dt_bias: Vec<f32>,
    /// A_log. Shape `[d_state, d_inner]` row-major. We exponentiate
    /// at forward time: `A = -exp(A_log)`, so `A_s,c < 0` always.
    pub a_log: Vec<f32>,
    /// D. Shape `[d_inner]`. Per-channel skip connection.
    pub d: Vec<f32>,
    /// out_proj. Shape `[d_model, d_inner]` row-major.
    pub out_proj: Vec<f32>,
}

/// Full model weights: per-layer SSM blocks plus the
/// embedding table, final norm, and LM head.
#[derive(Debug)]
pub struct MambaWeights {
    /// `[vocab, d_model]` row-major.
    pub token_embd: Vec<f32>,
    pub blocks: Vec<MambaLayerWeights>,
    /// `[d_model]`.
    pub output_norm: Vec<f32>,
    /// `[vocab, d_model]` row-major. `None` when the model ties
    /// `token_embd` (saves vocab × d_model floats).
    pub output: Option<Vec<f32>>,
}

/// Loaded Mamba model: config + weights. Construct via
/// [`MambaModel::load`].
#[derive(Debug)]
pub struct MambaModel {
    pub cfg: MambaConfig,
    pub weights: MambaWeights,
}

/// Per-layer state carried across decode tokens. Mamba's "KV cache"
/// equivalent — far smaller than transformer KV (independent of
/// sequence length, only `d_inner × (d_conv - 1) + d_state ×
/// d_inner` floats per layer).
#[derive(Debug)]
pub struct MambaLayerCache {
    /// Rolling Conv1d state. Shape `[d_inner, d_conv - 1]` row-major.
    /// Column `j` is the activation from `j+1` tokens ago; column 0
    /// is the oldest, column `d_conv-2` is the most recent.
    pub conv_state: Vec<f32>,
    /// SSM state. Shape `[d_state, d_inner]` row-major.
    pub ssm_state: Vec<f32>,
}

impl MambaLayerCache {
    pub fn new(cfg: &MambaConfig) -> Self {
        Self {
            conv_state: vec![0.0; cfg.d_inner * (cfg.d_conv.saturating_sub(1).max(1))],
            ssm_state: vec![0.0; cfg.d_state * cfg.d_inner],
        }
    }
}

/// Full-model cache: one [`MambaLayerCache`] per layer.
#[derive(Debug)]
pub struct MambaCache {
    pub layers: Vec<MambaLayerCache>,
}

impl MambaCache {
    pub fn new(cfg: &MambaConfig) -> Self {
        Self {
            layers: (0..cfg.n_layers).map(|_| MambaLayerCache::new(cfg)).collect(),
        }
    }

    /// Reset every layer's state to zero. Equivalent to starting a
    /// fresh sequence.
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            for v in &mut l.conv_state {
                *v = 0.0;
            }
            for v in &mut l.ssm_state {
                *v = 0.0;
            }
        }
    }
}

// =============================================================
// Math helpers
// =============================================================

/// RMSNorm over a single row. `out[i] = x[i] / sqrt(mean(x²) + eps) * w[i]`.
fn rms_norm(x: &[f32], w: &[f32], out: &mut [f32], eps: f32) {
    debug_assert_eq!(x.len(), w.len());
    debug_assert_eq!(x.len(), out.len());
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let scale = 1.0 / ((sum_sq / n as f32) + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * w[i];
    }
}

/// SiLU activation: `silu(x) = x · sigmoid(x)`. Numerically stable form.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softplus: `softplus(x) = log(1 + exp(x))`. We use the
/// numerically-stable identity `softplus(x) = max(x, 0) + log(1 +
/// exp(-|x|))` so large positive `x` doesn't blow up `exp`.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// `out[i] = sum_j w[i, j] * x[j]`. Row-major. Naive serial path.
fn matvec(w: &[f32], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(w.len(), m * k);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);
    for i in 0..m {
        let row = &w[i * k..(i + 1) * k];
        let mut acc = 0.0f32;
        for j in 0..k {
            acc += row[j] * x[j];
        }
        out[i] = acc;
    }
}

/// Step the Conv1d for one token. `x_new[c]` is the channel's
/// incoming value; the function:
///   1. Shifts the rolling state left by 1 column (drops oldest).
///   2. Appends `x_new[c]` as the rightmost column.
///   3. Computes `out[c] = bias[c] + sum_k weight[c, k] · history[c, k]`
///      where `history` is `[state cols] + [x_new]` (d_conv columns).
///
/// Returns the post-Conv1d activation (NOT silu'd yet; caller
/// applies SiLU per the Mamba paper).
fn conv1d_step(
    x_new: &[f32],
    weight: &[f32],
    bias: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    d_inner: usize,
    d_conv: usize,
) {
    debug_assert_eq!(x_new.len(), d_inner);
    debug_assert_eq!(weight.len(), d_inner * d_conv);
    debug_assert_eq!(bias.len(), d_inner);
    debug_assert_eq!(out.len(), d_inner);
    let state_cols = d_conv.saturating_sub(1).max(1);
    debug_assert_eq!(state.len(), d_inner * state_cols);
    for c in 0..d_inner {
        let weight_row = &weight[c * d_conv..(c + 1) * d_conv];
        let state_row = &mut state[c * state_cols..(c + 1) * state_cols];
        // Convolution: oldest state col first, newest = x_new[c].
        let mut acc = bias[c];
        if d_conv >= 2 {
            for k in 0..state_cols {
                acc += weight_row[k] * state_row[k];
            }
            acc += weight_row[d_conv - 1] * x_new[c];
            // Shift left: state[k] ← state[k+1], state[last] ← x_new[c]
            for k in 0..state_cols - 1 {
                state_row[k] = state_row[k + 1];
            }
            state_row[state_cols - 1] = x_new[c];
        } else {
            // d_conv == 1 → no history; just bias + weight * x.
            acc += weight_row[0] * x_new[c];
        }
        out[c] = acc;
    }
}

/// Per-channel selective-scan step. Updates `ssm_state` in place
/// and writes `y` (the per-channel SSM output before gating).
///
/// `A` has shape `[d_state, d_inner]` row-major (the log values
/// from `a_log` already exponentiated + negated by the caller).
/// `B`, `C` have shape `[d_state]`. `dt` has shape `[d_inner]`.
/// `x_post_conv` has shape `[d_inner]`.
fn selective_scan_step(
    x_post_conv: &[f32],
    dt: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    ssm_state: &mut [f32],
    y: &mut [f32],
    d_inner: usize,
    d_state: usize,
) {
    debug_assert_eq!(x_post_conv.len(), d_inner);
    debug_assert_eq!(dt.len(), d_inner);
    debug_assert_eq!(a.len(), d_state * d_inner);
    debug_assert_eq!(b.len(), d_state);
    debug_assert_eq!(c.len(), d_state);
    debug_assert_eq!(d.len(), d_inner);
    debug_assert_eq!(ssm_state.len(), d_state * d_inner);
    debug_assert_eq!(y.len(), d_inner);
    for chan in 0..d_inner {
        let dt_c = dt[chan];
        let x_c = x_post_conv[chan];
        // Update SSM state column for this channel.
        let mut y_c = 0.0f32;
        for s in 0..d_state {
            let idx = s * d_inner + chan;
            let a_sc = a[idx];
            // Discretize A: dA = exp(dt · A). A is already negative
            // (caller did `A = -exp(a_log)`); the dt·A product is
            // negative so exp(dt·A) ∈ (0, 1].
            let da = (dt_c * a_sc).exp();
            // Discretize B: dB · x = dt · B[s] · x_c. We fuse the
            // multiplication into the state update.
            let db_x = dt_c * b[s] * x_c;
            ssm_state[idx] = da * ssm_state[idx] + db_x;
            // Output contribution: C[s] · state.
            y_c += c[s] * ssm_state[idx];
        }
        // Skip connection: D · x.
        y_c += d[chan] * x_c;
        y[chan] = y_c;
    }
}

// =============================================================
// Weight loader
// =============================================================

fn load_tensor_f32(gguf: &Gguf, name: &str) -> Result<Vec<f32>, MambaError> {
    let info = gguf
        .tensor(name)
        .ok_or_else(|| MambaError::MissingTensor(name.to_string()))?;
    let bytes = gguf
        .tensor_bytes(name)
        .ok_or_else(|| MambaError::MissingTensor(name.to_string()))?;
    let n = info.element_count() as usize;
    let mut out = vec![0.0f32; n];
    match info.dtype {
        GgmlType::F32 => {
            // Hand-unpack F32 bytes — Mamba's loader can avoid the
            // bytemuck dep this way, matching the existing
            // rustllama-models conventions.
            for i in 0..n {
                let b = &bytes[i * 4..(i + 1) * 4];
                out[i] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        GgmlType::F16 => {
            rustllama_gguf::dequant::dequant_f16(&bytes[..n * 2], &mut out);
        }
        other => {
            return Err(MambaError::UnsupportedDtype {
                tensor: name.to_string(),
                got: other,
            });
        }
    }
    Ok(out)
}

fn check_shape(
    name: &str,
    got: &[u64],
    expected: &[u64],
) -> Result<(), MambaError> {
    if got == expected {
        Ok(())
    } else {
        Err(MambaError::ShapeMismatch {
            tensor: name.to_string(),
            expected: expected.to_vec(),
            got: got.to_vec(),
        })
    }
}

impl MambaModel {
    /// Load a Mamba model from a GGUF file. Detects the
    /// architecture via [`MambaConfig::detect`] and falls through
    /// to the weight loader. Returns
    /// [`MambaError::MissingMetadata`] when the GGUF isn't Mamba-
    /// shaped.
    pub fn load(gguf: &Gguf) -> Result<Self, MambaError> {
        let arch = MambaConfig::detect(gguf)
            .ok_or(MambaError::MissingMetadata("general.architecture (not Mamba)"))?;
        let cfg = MambaConfig::from_gguf(gguf, &arch)?;

        // GGUF tensor name prefix differs per arch family; the
        // standard llama.cpp converter emits `token_embd.weight`
        // and `blk.<n>.<field>.weight` for both Mamba-1 and
        // Mamba-2.
        let d_model = cfg.d_model;
        let d_inner = cfg.d_inner;
        let d_state = cfg.d_state;
        let d_conv = cfg.d_conv;
        let dt_rank = cfg.dt_rank;
        let vocab = cfg.vocab_size;

        let token_embd = load_tensor_f32(gguf, "token_embd.weight")?;
        check_shape(
            "token_embd.weight",
            &gguf.tensor("token_embd.weight").unwrap().dims,
            &[d_model as u64, vocab as u64],
        )?;

        // Final RMSNorm. Sometimes called `output_norm.weight`.
        let output_norm = load_tensor_f32(gguf, "output_norm.weight")?;

        // LM head — tied to token_embd when missing.
        let output = match gguf.tensor("output.weight") {
            Some(_) => Some(load_tensor_f32(gguf, "output.weight")?),
            None => None,
        };

        // Per-layer block loader. llama.cpp's mamba converter uses
        // `blk.<n>.ssm_*` for SSM tensors and `blk.<n>.attn_norm`
        // for the pre-block RMSNorm (kept named `attn_norm` for
        // consistency with the transformer naming, even though
        // there's no attention here).
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for n in 0..cfg.n_layers {
            let p = |suffix: &str| format!("blk.{n}.{suffix}");
            let attn_norm = load_tensor_f32(gguf, &p("attn_norm.weight"))?;
            let in_proj = load_tensor_f32(gguf, &p("ssm_in.weight"))?;
            let conv1d_weight = load_tensor_f32(gguf, &p("ssm_conv1d.weight"))?;
            let conv1d_bias = load_tensor_f32(gguf, &p("ssm_conv1d.bias"))?;
            let x_proj = load_tensor_f32(gguf, &p("ssm_x.weight"))?;
            let dt_proj = load_tensor_f32(gguf, &p("ssm_dt.weight"))?;
            let dt_bias = load_tensor_f32(gguf, &p("ssm_dt.bias"))?;
            let a_log = load_tensor_f32(gguf, &p("ssm_a"))?;
            let d = load_tensor_f32(gguf, &p("ssm_d"))?;
            let out_proj = load_tensor_f32(gguf, &p("ssm_out.weight"))?;
            // Sanity-check the canonical shapes — fast-fail rather
            // than silently produce garbage from a misshaped tensor.
            if attn_norm.len() != d_model
                || in_proj.len() != 2 * d_inner * d_model
                || conv1d_weight.len() != d_inner * d_conv
                || conv1d_bias.len() != d_inner
                || x_proj.len() != (dt_rank + 2 * d_state) * d_inner
                || dt_proj.len() != d_inner * dt_rank
                || dt_bias.len() != d_inner
                || a_log.len() != d_state * d_inner
                || d.len() != d_inner
                || out_proj.len() != d_model * d_inner
            {
                return Err(MambaError::ShapeMismatch {
                    tensor: p("(layer)"),
                    expected: vec![],
                    got: vec![],
                });
            }
            blocks.push(MambaLayerWeights {
                attn_norm,
                in_proj,
                conv1d_weight,
                conv1d_bias,
                x_proj,
                dt_proj,
                dt_bias,
                a_log,
                d,
                out_proj,
            });
        }
        Ok(Self {
            cfg,
            weights: MambaWeights {
                token_embd,
                blocks,
                output_norm,
                output,
            },
        })
    }

    /// Embed a sequence of token ids into `[ids.len(), d_model]`
    /// row-major. Used by external callers building the prefill
    /// loop.
    pub fn embed_tokens(&self, ids: &[i32]) -> Vec<f32> {
        let d = self.cfg.d_model;
        let mut out = vec![0.0f32; ids.len() * d];
        for (slot, &id) in ids.iter().enumerate() {
            let id = id.max(0) as usize;
            let src = &self.weights.token_embd[id * d..(id + 1) * d];
            out[slot * d..(slot + 1) * d].copy_from_slice(src);
        }
        out
    }

    /// Forward pass on **one** token. Updates `cache` in place and
    /// writes the logit vector into `logits_out`. Mirrors
    /// [`crate::llama_arch::LlamaModel::forward_one`]'s contract so
    /// the engine's generation loop can dispatch on architecture
    /// without re-doing the sampling glue.
    pub fn forward_one(
        &self,
        token_id: i32,
        cache: &mut MambaCache,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        let d_model = cfg.d_model;
        let d_inner = cfg.d_inner;
        let d_state = cfg.d_state;
        let d_conv = cfg.d_conv;
        let dt_rank = cfg.dt_rank;
        let eps = cfg.eps;
        assert_eq!(logits_out.len(), cfg.vocab_size);
        assert_eq!(cache.layers.len(), cfg.n_layers);

        // 1. Embed.
        let id = token_id.max(0) as usize;
        let mut x = self.weights.token_embd[id * d_model..(id + 1) * d_model].to_vec();

        // Reusable scratch.
        let mut norm = vec![0.0f32; d_model];
        let mut xz = vec![0.0f32; 2 * d_inner];
        let mut x_branch_post_conv = vec![0.0f32; d_inner];
        let mut dt_bc = vec![0.0f32; dt_rank + 2 * d_state];
        let mut dt_pre_proj = vec![0.0f32; d_inner];
        let mut dt_after_softplus = vec![0.0f32; d_inner];
        let mut a_neg = vec![0.0f32; d_state * d_inner];
        let mut y = vec![0.0f32; d_inner];
        let mut out_resid = vec![0.0f32; d_model];

        for (l, block) in self.weights.blocks.iter().enumerate() {
            // 2. RMSNorm.
            rms_norm(&x, &block.attn_norm, &mut norm, eps);
            // 3. in_proj: [d_model] → [2*d_inner].
            matvec(&block.in_proj, &norm, &mut xz, 2 * d_inner, d_model);
            let (x_branch_in, z) = xz.split_at(d_inner);
            // 4. Conv1d: rolling state + bias. Then SiLU.
            conv1d_step(
                x_branch_in,
                &block.conv1d_weight,
                &block.conv1d_bias,
                &mut cache.layers[l].conv_state,
                &mut x_branch_post_conv,
                d_inner,
                d_conv,
            );
            for v in &mut x_branch_post_conv {
                *v = silu(*v);
            }
            // 5. x_proj: [d_inner] → [dt_rank + 2*d_state]. Split
            //    into (dt_pre, B, C).
            matvec(
                &block.x_proj,
                &x_branch_post_conv,
                &mut dt_bc,
                dt_rank + 2 * d_state,
                d_inner,
            );
            let (dt_pre, bc) = dt_bc.split_at(dt_rank);
            let (b_vec, c_vec) = bc.split_at(d_state);

            // 6. dt = softplus(dt_proj · dt_pre + dt_bias).
            matvec(&block.dt_proj, dt_pre, &mut dt_pre_proj, d_inner, dt_rank);
            for i in 0..d_inner {
                dt_after_softplus[i] = softplus(dt_pre_proj[i] + block.dt_bias[i]);
            }
            // 7. A = -exp(a_log).
            for (a_neg_v, a_log_v) in a_neg.iter_mut().zip(block.a_log.iter()) {
                *a_neg_v = -(a_log_v.exp());
            }
            // 8. Selective scan: updates ssm_state in place, writes y.
            selective_scan_step(
                &x_branch_post_conv,
                &dt_after_softplus,
                &a_neg,
                b_vec,
                c_vec,
                &block.d,
                &mut cache.layers[l].ssm_state,
                &mut y,
                d_inner,
                d_state,
            );
            // 9. Gate by SiLU(z).
            for i in 0..d_inner {
                y[i] *= silu(z[i]);
            }
            // 10. out_proj: [d_inner] → [d_model].
            matvec(&block.out_proj, &y, &mut out_resid, d_model, d_inner);
            // 11. Residual add.
            for i in 0..d_model {
                x[i] += out_resid[i];
            }
        }

        // 12. Final RMSNorm.
        rms_norm(&x.clone(), &self.weights.output_norm, &mut x, eps);
        // 13. LM head. Tied embeddings: token_embd has shape
        //     [vocab, d_model], same as the output projection's
        //     transposed shape, so the matmul math is identical
        //     either way.
        let head: &[f32] = self
            .weights
            .output
            .as_deref()
            .unwrap_or(&self.weights.token_embd);
        matvec(head, &x, logits_out, cfg.vocab_size, d_model);
    }

    /// Drive a sequence of tokens through the model serially, ending
    /// with the next-token logits in `logits_out`. The cache carries
    /// per-layer state across tokens. For prefill of a long prompt
    /// followed by generation, call `forward_sequence(prompt, ...)`
    /// once, then loop on `forward_one(sampled_token, ...)`.
    pub fn forward_sequence(
        &self,
        token_ids: &[i32],
        cache: &mut MambaCache,
        logits_out: &mut [f32],
    ) {
        if token_ids.is_empty() {
            return;
        }
        // For all but the last, we just update the state. For the
        // last, we also compute logits.
        let (init, last) = token_ids.split_at(token_ids.len() - 1);
        // Scratch logits buffer for the non-final steps. The
        // selective scan side-effects through `cache`; logits are
        // discarded.
        let mut scratch = vec![0.0f32; self.cfg.vocab_size];
        for &id in init {
            self.forward_one(id, cache, &mut scratch);
        }
        self.forward_one(last[0], cache, logits_out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_cfg() -> MambaConfig {
        // Tiny shape just for verifying the math composes.
        MambaConfig {
            d_model: 16,
            n_layers: 1,
            vocab_size: 32,
            d_state: 8,
            d_conv: 4,
            d_inner: 32,
            dt_rank: 2,
            eps: 1e-5,
        }
    }

    #[test]
    fn rms_norm_zeros_pass_through_scale() {
        let x = vec![0.0; 8];
        let w = vec![2.0; 8];
        let mut out = vec![0.0; 8];
        rms_norm(&x, &w, &mut out, 1e-5);
        // x=0 → rms=0 → output is 0 (regardless of scale).
        for v in out {
            assert!(v.abs() < 1e-4);
        }
    }

    #[test]
    fn silu_and_softplus_basic() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!(silu(10.0) > 9.9 && silu(10.0) < 10.001);
        assert!(softplus(0.0).abs() - (1.0_f32.ln() + 0.69314) < 0.01);
        assert!((softplus(100.0) - 100.0).abs() < 1e-3);
        assert!(softplus(-100.0) > 0.0 && softplus(-100.0) < 1e-30);
    }

    #[test]
    fn conv1d_step_rolls_state() {
        // d_inner=1, d_conv=3 → state cols = 2.
        let weight = vec![1.0, 2.0, 3.0]; // [in_ch=1, k=3]
        let bias = vec![0.0];
        let mut state = vec![0.0, 0.0];
        let mut out = vec![0.0];

        // Token 1: x_new=10. State was [0,0]. Conv: 1*0 + 2*0 + 3*10 = 30.
        conv1d_step(&[10.0], &weight, &bias, &mut state, &mut out, 1, 3);
        assert!((out[0] - 30.0).abs() < 1e-4);
        // State after: shift left, append 10 → [0, 10].
        assert!((state[0] - 0.0).abs() < 1e-4);
        assert!((state[1] - 10.0).abs() < 1e-4);

        // Token 2: x_new=20. State [0,10]. Conv: 1*0 + 2*10 + 3*20 = 80.
        conv1d_step(&[20.0], &weight, &bias, &mut state, &mut out, 1, 3);
        assert!((out[0] - 80.0).abs() < 1e-4);
        // State: [10, 20].
        assert!((state[0] - 10.0).abs() < 1e-4);
        assert!((state[1] - 20.0).abs() < 1e-4);

        // Token 3: x_new=30. State [10,20]. Conv: 1*10 + 2*20 + 3*30 = 140.
        conv1d_step(&[30.0], &weight, &bias, &mut state, &mut out, 1, 3);
        assert!((out[0] - 140.0).abs() < 1e-4);
    }

    #[test]
    fn selective_scan_step_zero_input_keeps_state() {
        // With x=0 and dt·A·exp converging to 1 (A close to 0):
        // state ← exp(dt·A)·state + dt·B·x = exp(0)·state + 0 = state.
        let d_inner = 2;
        let d_state = 3;
        let x = vec![0.0; d_inner];
        let dt = vec![0.1; d_inner];
        // A=0 in log-space means -exp(0) = -1. exp(0.1 · -1) = exp(-0.1) ≈ 0.905.
        // So state decays each step. Let's verify that.
        let a = vec![-1.0; d_state * d_inner];
        let b = vec![0.5; d_state];
        let c = vec![1.0; d_state];
        let d_skip = vec![0.0; d_inner];
        let mut state = vec![1.0; d_state * d_inner];
        let mut y = vec![0.0; d_inner];
        selective_scan_step(
            &x, &dt, &a, &b, &c, &d_skip, &mut state, &mut y, d_inner, d_state,
        );
        // Each state cell should now be exp(0.1 * -1) ≈ 0.9048.
        for &v in &state {
            assert!((v - 0.9048).abs() < 0.001, "decay state, got {v}");
        }
        // y = C · state = 1 * 0.9048 * d_state = 2.7145
        for &v in &y {
            assert!((v - 0.9048 * 3.0).abs() < 0.01, "y = sum(C·state), got {v}");
        }
    }

    #[test]
    fn matvec_basic() {
        // 3x2: out[i] = w[i][0]*x[0] + w[i][1]*x[1]
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![10.0, 20.0];
        let mut out = vec![0.0; 3];
        matvec(&w, &x, &mut out, 3, 2);
        assert_eq!(out, vec![50.0, 110.0, 170.0]);
    }

    #[test]
    fn config_detect_typecheck() {
        // Just pin the signature.
        fn _ck(g: &Gguf) -> Option<String> {
            MambaConfig::detect(g)
        }
    }

    #[test]
    fn cache_reset_zeros() {
        let cfg = synth_cfg();
        let mut cache = MambaCache::new(&cfg);
        for l in &mut cache.layers {
            l.conv_state[0] = 42.0;
            l.ssm_state[0] = 13.0;
        }
        cache.reset();
        for l in &cache.layers {
            assert_eq!(l.conv_state[0], 0.0);
            assert_eq!(l.ssm_state[0], 0.0);
        }
    }

    #[test]
    fn cache_dims_match_config() {
        let cfg = synth_cfg();
        let cache = MambaCache::new(&cfg);
        assert_eq!(cache.layers.len(), cfg.n_layers);
        for l in &cache.layers {
            assert_eq!(l.conv_state.len(), cfg.d_inner * (cfg.d_conv - 1));
            assert_eq!(l.ssm_state.len(), cfg.d_state * cfg.d_inner);
        }
    }
}
