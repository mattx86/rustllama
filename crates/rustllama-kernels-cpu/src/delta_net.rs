//! Gated DeltaNet forward kernels for the `qwen35moe` hybrid arch
//! (and other future models using the same convention).
//!
//! Reference: `fla/layers/gated_deltanet.py` and
//! `fla/ops/gated_delta_rule/naive.py` in
//! [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention).
//! Paper: "Gated Delta Networks: Improving Mamba2 with Delta Rule"
//! (arxiv 2412.06464).
//!
//! ## Per-token recurrence (one V-head)
//! ```text
//! S_t = exp(g_t) * S_{t-1}                  // decay; g from a_proj(hidden)
//! v_tilde = (v_t - S_{t-1} @ k_t) * beta_t  // delta correction
//! S_t = S_t + outer(k_t, v_tilde)           // rank-1 update
//! o_t = q_t @ S_t                           // output
//! ```
//! `beta = sigmoid(b_proj(hidden))`. State `S` per V-head has shape
//! `[head_qk_dim, head_v_dim]` — carried across decoded tokens via
//! the engine's KV-cache extension (added in Phase 3.7).
//!
//! ## What this module provides
//! - [`conv1d_depthwise_step_f32`] — substep 3.1, depthwise conv1d with
//!   circular state for one token
//! - [`silu_f32_inplace`] — substep 3.3, SiLU activation
//! - [`delta_rule_step_f32`] — substep 3.4, the single-token recurrent
//!   update for one V-head
//! - [`gated_rmsnorm_f32_inplace`] — substep 3.5, RMSNorm gated by a
//!   sigmoid-of-projection signal
//!
//! Substeps 3.2 (the q/k/v/alpha/beta/gate projections) and 3.6 (the
//! output projection) are plain matvecs handled at the call site via
//! the existing `matvec_*` kernels — no new kernel needed here.

use crate::rmsnorm_f32;
use std::sync::OnceLock;

/// Substep 3.1: depthwise 1D convolution forward for one token.
///
/// Each of `channels` channels is convolved independently with a kernel
/// of width `kernel`. `conv_state` is a per-sequence circular buffer
/// of shape `[kernel - 1, channels]` holding the prior `kernel-1`
/// input values. The new input `x_in[c]` is appended at the "current
/// step" position; the output is the dot of the kernel against the
/// `kernel`-token window ending at the current step.
///
/// State update: after computing the output, the oldest column of
/// `conv_state` is dropped and `x_in` becomes the newest column. We
/// implement this as a simple right-shift to keep the buffer layout
/// trivial (the per-decode cost is `(kernel - 1) * channels` copies
/// — negligible at decode speeds).
///
/// `weight` is laid out as `[channels, kernel]` row-major matching
/// the GGUF tensor (`ssm_conv1d.weight` whose GGUF metadata reports
/// `ne = [kernel, channels]` — i.e., kernel is the fastest/innermost
/// dim, so flat layout is `weight[c * kernel + k]`).
pub fn conv1d_depthwise_step_f32(
    x_in: &[f32],
    weight: &[f32],
    conv_state: &mut [f32],
    out: &mut [f32],
    channels: usize,
    kernel: usize,
) {
    assert_eq!(x_in.len(), channels);
    assert_eq!(out.len(), channels);
    assert_eq!(weight.len(), kernel * channels);
    assert_eq!(
        conv_state.len(),
        (kernel - 1) * channels,
        "conv_state must be sized [(kernel-1) × channels]"
    );

    // Compute output: out[c] = sum over k of weight[c, k] * window[k, c]
    // where window[0..kernel-1, c] = conv_state and window[kernel-1, c] = x_in.
    // conv_state is kept in [kernel-1, channels] row-major (kernel outer,
    // channels inner) so its indexing stays `[k * channels + c]` below.
    for c in 0..channels {
        let mut acc = 0.0f32;
        let w_base = c * kernel;
        for k in 0..kernel - 1 {
            acc += weight[w_base + k] * conv_state[k * channels + c];
        }
        acc += weight[w_base + (kernel - 1)] * x_in[c];
        out[c] = acc;
    }

    // Shift state: drop column 0 (oldest), shift columns 1..kernel-1
    // left by one, then place x_in at column kernel-2 (the new
    // "most-recent prior" slot for the next call).
    for k in 0..kernel - 2 {
        for c in 0..channels {
            conv_state[k * channels + c] = conv_state[(k + 1) * channels + c];
        }
    }
    if kernel >= 2 {
        let last_col = kernel - 2;
        for c in 0..channels {
            conv_state[last_col * channels + c] = x_in[c];
        }
    }
}

/// Substep 3.3: SiLU activation in place. `x[i] := x[i] / (1 + exp(-x[i]))`.
///
/// The hybrid model applies SiLU after the conv1d on q/k/v (matches
/// the `act_fn = nn.SiLU()` in the fla reference).
pub fn silu_f32_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { silu_f32_inplace_avx2(x) };
            return;
        }
    }
    silu_f32_inplace_scalar(x);
}

/// Scalar fallback / parity reference for [`silu_f32_inplace`].
pub(crate) fn silu_f32_inplace_scalar(x: &mut [f32]) {
    for v in x.iter_mut() {
        // Numerically stable SiLU = x * sigmoid(x).
        let s = if *v >= 0.0 {
            1.0 / (1.0 + (-*v).exp())
        } else {
            let e = v.exp();
            e / (1.0 + e)
        };
        *v *= s;
    }
}

/// AVX2 SiLU in place, `x := x / (1 + exp(-x))`, using the crate's
/// shared [`crate::expf_approx_avx2`] (same polynomial the SwiGLU FFN
/// path already relies on, ~2e-6 relative error → ~1e-3 abs vs the
/// libm scalar reference; see `silu_mul_f32`). Tail lanes fall back
/// to the exact scalar formula.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn silu_f32_inplace_avx2(x: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let n8 = n & !7;
    let xp = x.as_mut_ptr();
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let mut p = 0;
    while p < n8 {
        let xv = _mm256_loadu_ps(xp.add(p));
        // silu(x) = x / (1 + exp(-x))
        let e = crate::expf_approx_avx2(_mm256_sub_ps(zero, xv));
        let silu = _mm256_div_ps(xv, _mm256_add_ps(one, e));
        _mm256_storeu_ps(xp.add(p), silu);
        p += 8;
    }
    while p < n {
        let v = *xp.add(p);
        let s = if v >= 0.0 {
            1.0 / (1.0 + (-v).exp())
        } else {
            let e = v.exp();
            e / (1.0 + e)
        };
        *xp.add(p) = v * s;
        p += 1;
    }
}

/// Substep 3.4 — the single-token Gated Delta Rule recurrent update for
/// **one V-head**. Caller iterates this over all `n_v_heads` heads,
/// passing the per-head slices and state.
///
/// Shapes:
/// - `q`, `k`: `[head_qk_dim]` — query/key for this V-head's
///   associated QK-head (with GQA-style grouping when n_v_heads >
///   n_qk_heads; the caller handles head→head mapping)
/// - `v`: `[head_v_dim]` — value for this V-head
/// - `g`: scalar — log-decay for this V-head (per `exp(g) * S_{t-1}`)
/// - `beta`: scalar — sigmoid'd learning-rate for the delta correction
/// - `state`: `[head_qk_dim, head_v_dim]` row-major recurrent state,
///   mutated in place. **The state layout has the K-dim as the outer
///   (row) dim** so the rank-1 update `outer(k, v_tilde)` is a series
///   of row-scaled writes.
/// - `out`: `[head_v_dim]` — output, written (not accumulated)
///
/// Reference: `fla/ops/gated_delta_rule/naive.py`. The exact line that
/// pins the math is the inner loop:
/// ```python
/// h = h.clone() * g[..., None, None].exp()
/// b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)  # v - S^T @ k
/// b_v = b_v * b_beta[..., None]
/// h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)  # S += outer(k, v_tilde)
/// o[..., i] = torch.einsum('bhd,bhdm->bhm', b_q, h)     # q^T @ S
/// ```
/// Per-head L2 normalization, in-place: `x := x / sqrt(sum(x*x) + eps)`.
///
/// HF `Qwen3NextGatedDeltaNet` calls `recurrent_gated_delta_rule(..,
/// use_qk_l2norm_in_kernel=True)`, which L2-normalizes Q and K per
/// head before the recurrence. Without this the magnitude of the
/// recurrent state grows unboundedly with sequence length because
/// `outer(k, v_tilde)` is not bounded — every step corrupts the
/// residual stream proportional to `|k|·|v|·|q|`.
///
/// Match HF's `qwen3_next.l2norm` (`F.normalize`-equivalent) using
/// the same `eps=1e-6` they pass in the kernel call.
pub fn l2norm_f32_inplace(x: &mut [f32], eps: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            unsafe { l2norm_f32_inplace_avx512f(x, eps) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { l2norm_f32_inplace_avx2(x, eps) };
            return;
        }
    }
    l2norm_f32_inplace_scalar(x, eps);
}

/// Scalar fallback / parity reference for [`l2norm_f32_inplace`].
pub(crate) fn l2norm_f32_inplace_scalar(x: &mut [f32], eps: f32) {
    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }
    let inv = (sum_sq + eps).sqrt().recip();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// AVX2 L2 normalize in place. Vectorized sum-of-squares (8-wide FMA,
/// same horizontal-reduce order as `rmsnorm_f32_row_avx2`) then a
/// broadcast scale. Tail lanes handled scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn l2norm_f32_inplace_avx2(x: &mut [f32], eps: f32) {
    use std::arch::x86_64::*;
    let n = x.len();
    let n8 = n & !7;
    let xp = x.as_mut_ptr();

    let mut acc = _mm256_setzero_ps();
    let mut p = 0;
    while p < n8 {
        let v = _mm256_loadu_ps(xp.add(p));
        acc = _mm256_fmadd_ps(v, v, acc);
        p += 8;
    }
    // Horizontal reduce of 8 lanes (matches rmsnorm_f32_row_avx2).
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps::<1>(acc);
    let s128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf = _mm_movehl_ps(shuf, sums);
    let sums = _mm_add_ss(sums, shuf);
    let mut sum_sq = _mm_cvtss_f32(sums);
    while p < n {
        let v = *xp.add(p);
        sum_sq += v * v;
        p += 1;
    }

    let inv = (sum_sq + eps).sqrt().recip();
    let inv_b = _mm256_set1_ps(inv);
    let mut p = 0;
    while p < n8 {
        let v = _mm256_loadu_ps(xp.add(p));
        _mm256_storeu_ps(xp.add(p), _mm256_mul_ps(v, inv_b));
        p += 8;
    }
    while p < n {
        *xp.add(p) *= inv;
        p += 1;
    }
}

/// AVX-512 L2 normalize in place. 16-wide sum-of-squares + broadcast
/// scale; scalar tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn l2norm_f32_inplace_avx512f(x: &mut [f32], eps: f32) {
    use std::arch::x86_64::*;
    let n = x.len();
    let n16 = n & !15;
    let xp = x.as_mut_ptr();

    let mut acc = _mm512_setzero_ps();
    let mut p = 0;
    while p < n16 {
        let v = _mm512_loadu_ps(xp.add(p));
        acc = _mm512_fmadd_ps(v, v, acc);
        p += 16;
    }
    let mut sum_sq = _mm512_reduce_add_ps(acc);
    while p < n {
        let v = *xp.add(p);
        sum_sq += v * v;
        p += 1;
    }

    let inv = (sum_sq + eps).sqrt().recip();
    let inv_b = _mm512_set1_ps(inv);
    let mut p = 0;
    while p < n16 {
        let v = _mm512_loadu_ps(xp.add(p));
        _mm512_storeu_ps(xp.add(p), _mm512_mul_ps(v, inv_b));
        p += 16;
    }
    while p < n {
        *xp.add(p) *= inv;
        p += 1;
    }
}

/// Alloc-per-call wrapper around
/// [`delta_rule_step_f32_with_scratch`] — kept for tests and one-shot
/// callers. Hot paths (the hybrid decode/prefill loops, which run
/// this 32 heads × 30 layers per token) pass a reused scratch buffer
/// instead: the per-call `Vec` here was ~960 allocations per decoded
/// token on qwen35moe.
pub fn delta_rule_step_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: f32,
    beta: f32,
    state: &mut [f32],
    out: &mut [f32],
    head_qk_dim: usize,
    head_v_dim: usize,
) {
    let mut v_tilde = vec![0.0f32; head_v_dim];
    delta_rule_step_f32_with_scratch(
        q, k, v, g, beta, state, out, head_qk_dim, head_v_dim, &mut v_tilde,
    );
}

/// One Delta-Rule recurrence step. `v_tilde` is caller scratch of
/// length `head_v_dim`; its contents are overwritten. Numerically
/// identical to [`delta_rule_step_f32`].
#[allow(clippy::too_many_arguments)]
pub fn delta_rule_step_f32_with_scratch(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: f32,
    beta: f32,
    state: &mut [f32],
    out: &mut [f32],
    head_qk_dim: usize,
    head_v_dim: usize,
    v_tilde: &mut [f32],
) {
    assert_eq!(q.len(), head_qk_dim);
    assert_eq!(k.len(), head_qk_dim);
    assert_eq!(v.len(), head_v_dim);
    assert_eq!(out.len(), head_v_dim);
    assert_eq!(state.len(), head_qk_dim * head_v_dim);
    assert_eq!(v_tilde.len(), head_v_dim);

    // Dominant per-token DeltaNet cost (n_v_heads × n_ssm_layers calls
    // per decoded token). The recurrence is sequential across tokens,
    // but every inner loop here runs over `head_v_dim` (the contiguous
    // state-row / output dim, typically 128) and vectorizes cleanly.
    // The outer `i` reduction order is preserved bit-for-bit against
    // the scalar reference; only the mul+add → FMA fusion differs
    // (≈1 ulp), so SIMD tracks scalar to well within 1e-4.
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above; lengths asserted.
            unsafe {
                delta_rule_step_f32_with_scratch_avx512(
                    q, k, v, g, beta, state, out, head_qk_dim, head_v_dim, v_tilde,
                )
            };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above; lengths asserted.
            unsafe {
                delta_rule_step_f32_with_scratch_avx2(
                    q, k, v, g, beta, state, out, head_qk_dim, head_v_dim, v_tilde,
                )
            };
            return;
        }
    }
    delta_rule_step_f32_with_scratch_scalar(
        q, k, v, g, beta, state, out, head_qk_dim, head_v_dim, v_tilde,
    );
}

/// Scalar fallback / parity reference for
/// [`delta_rule_step_f32_with_scratch`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn delta_rule_step_f32_with_scratch_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: f32,
    beta: f32,
    state: &mut [f32],
    out: &mut [f32],
    head_qk_dim: usize,
    head_v_dim: usize,
    v_tilde: &mut [f32],
) {
    // 1. State decay: S *= exp(g).
    let decay = g.exp();
    for s in state.iter_mut() {
        *s *= decay;
    }

    // 2. Compute v_tilde = (v - S^T @ k) * beta.
    //    S has shape [head_qk_dim, head_v_dim] row-major, so
    //    (S^T @ k)[j] = sum_i S[i, j] * k[i] for j in 0..head_v_dim.
    v_tilde.fill(0.0);
    for i in 0..head_qk_dim {
        let k_i = k[i];
        if k_i == 0.0 {
            continue;
        }
        let row = &state[i * head_v_dim..(i + 1) * head_v_dim];
        for j in 0..head_v_dim {
            v_tilde[j] += row[j] * k_i;
        }
    }
    // Now v_tilde = S^T @ k. Apply v_tilde = (v - v_tilde) * beta.
    for j in 0..head_v_dim {
        v_tilde[j] = (v[j] - v_tilde[j]) * beta;
    }

    // 3. Rank-1 state update: S += outer(k, v_tilde).
    //    S[i, j] += k[i] * v_tilde[j].
    for i in 0..head_qk_dim {
        let k_i = k[i];
        if k_i == 0.0 {
            continue;
        }
        let row = &mut state[i * head_v_dim..(i + 1) * head_v_dim];
        for j in 0..head_v_dim {
            row[j] += k_i * v_tilde[j];
        }
    }

    // 4. Output: out = q^T @ S.
    //    out[j] = sum_i q[i] * S[i, j].
    out.fill(0.0);
    for i in 0..head_qk_dim {
        let q_i = q[i];
        if q_i == 0.0 {
            continue;
        }
        let row = &state[i * head_v_dim..(i + 1) * head_v_dim];
        for j in 0..head_v_dim {
            out[j] += q_i * row[j];
        }
    }
}

/// AVX2 (f32×8 + FMA) DeltaNet recurrence step. Numerically matches
/// [`delta_rule_step_f32_with_scratch_scalar`] to ≈1e-4 (identical
/// outer reduction order; only the fused multiply-add differs). All
/// inner loops run over `head_v_dim` with an 8-wide body and a scalar
/// tail for the remainder lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn delta_rule_step_f32_with_scratch_avx2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: f32,
    beta: f32,
    state: &mut [f32],
    out: &mut [f32],
    head_qk_dim: usize,
    head_v_dim: usize,
    v_tilde: &mut [f32],
) {
    use std::arch::x86_64::*;
    let vd = head_v_dim;
    let vd8 = vd & !7;
    let decay = g.exp();
    let sp = state.as_mut_ptr();
    let vtp = v_tilde.as_mut_ptr();

    // 1. State decay over the flat [head_qk_dim * head_v_dim] buffer.
    {
        let total = head_qk_dim * vd;
        let t8 = total & !7;
        let decay_b = _mm256_set1_ps(decay);
        let mut p = 0;
        while p < t8 {
            let s = _mm256_loadu_ps(sp.add(p));
            _mm256_storeu_ps(sp.add(p), _mm256_mul_ps(s, decay_b));
            p += 8;
        }
        while p < total {
            *sp.add(p) *= decay;
            p += 1;
        }
    }

    // 2. v_tilde = S^T @ k (accumulate row-scaled), i outer to preserve
    //    the scalar reduction order.
    {
        let mut j = 0;
        while j < vd8 {
            _mm256_storeu_ps(vtp.add(j), _mm256_setzero_ps());
            j += 8;
        }
        while j < vd {
            *vtp.add(j) = 0.0;
            j += 1;
        }
    }
    for i in 0..head_qk_dim {
        let k_i = *k.get_unchecked(i);
        if k_i == 0.0 {
            continue;
        }
        let k_b = _mm256_set1_ps(k_i);
        let row = sp.add(i * vd);
        let mut j = 0;
        while j < vd8 {
            let acc = _mm256_loadu_ps(vtp.add(j));
            let rv = _mm256_loadu_ps(row.add(j));
            _mm256_storeu_ps(vtp.add(j), _mm256_fmadd_ps(rv, k_b, acc));
            j += 8;
        }
        while j < vd {
            *vtp.add(j) += *row.add(j) * k_i;
            j += 1;
        }
    }
    // v_tilde = (v - v_tilde) * beta.
    {
        let vp = v.as_ptr();
        let beta_b = _mm256_set1_ps(beta);
        let mut j = 0;
        while j < vd8 {
            let vv = _mm256_loadu_ps(vp.add(j));
            let tv = _mm256_loadu_ps(vtp.add(j));
            _mm256_storeu_ps(vtp.add(j), _mm256_mul_ps(_mm256_sub_ps(vv, tv), beta_b));
            j += 8;
        }
        while j < vd {
            *vtp.add(j) = (*vp.add(j) - *vtp.add(j)) * beta;
            j += 1;
        }
    }

    // 3. Rank-1 update S[i, j] += k[i] * v_tilde[j].
    for i in 0..head_qk_dim {
        let k_i = *k.get_unchecked(i);
        if k_i == 0.0 {
            continue;
        }
        let k_b = _mm256_set1_ps(k_i);
        let row = sp.add(i * vd);
        let mut j = 0;
        while j < vd8 {
            let sv = _mm256_loadu_ps(row.add(j));
            let tv = _mm256_loadu_ps(vtp.add(j));
            _mm256_storeu_ps(row.add(j), _mm256_fmadd_ps(k_b, tv, sv));
            j += 8;
        }
        while j < vd {
            *row.add(j) += k_i * *vtp.add(j);
            j += 1;
        }
    }

    // 4. Output out[j] = sum_i q[i] * S[i, j], i outer (reduction order
    //    preserved).
    let op = out.as_mut_ptr();
    {
        let mut j = 0;
        while j < vd8 {
            _mm256_storeu_ps(op.add(j), _mm256_setzero_ps());
            j += 8;
        }
        while j < vd {
            *op.add(j) = 0.0;
            j += 1;
        }
    }
    for i in 0..head_qk_dim {
        let q_i = *q.get_unchecked(i);
        if q_i == 0.0 {
            continue;
        }
        let q_b = _mm256_set1_ps(q_i);
        let row = sp.add(i * vd);
        let mut j = 0;
        while j < vd8 {
            let acc = _mm256_loadu_ps(op.add(j));
            let rv = _mm256_loadu_ps(row.add(j));
            _mm256_storeu_ps(op.add(j), _mm256_fmadd_ps(rv, q_b, acc));
            j += 8;
        }
        while j < vd {
            *op.add(j) += q_i * *row.add(j);
            j += 1;
        }
    }
}

/// AVX-512 (f32×16 + FMA) DeltaNet recurrence step. Same algorithm and
/// reduction order as the AVX2 variant, 16-wide inner bodies.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn delta_rule_step_f32_with_scratch_avx512(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: f32,
    beta: f32,
    state: &mut [f32],
    out: &mut [f32],
    head_qk_dim: usize,
    head_v_dim: usize,
    v_tilde: &mut [f32],
) {
    use std::arch::x86_64::*;
    let vd = head_v_dim;
    let vd16 = vd & !15;
    let decay = g.exp();
    let sp = state.as_mut_ptr();
    let vtp = v_tilde.as_mut_ptr();

    // 1. State decay.
    {
        let total = head_qk_dim * vd;
        let t16 = total & !15;
        let decay_b = _mm512_set1_ps(decay);
        let mut p = 0;
        while p < t16 {
            let s = _mm512_loadu_ps(sp.add(p));
            _mm512_storeu_ps(sp.add(p), _mm512_mul_ps(s, decay_b));
            p += 16;
        }
        while p < total {
            *sp.add(p) *= decay;
            p += 1;
        }
    }

    // 2. v_tilde = S^T @ k.
    {
        let mut j = 0;
        while j < vd16 {
            _mm512_storeu_ps(vtp.add(j), _mm512_setzero_ps());
            j += 16;
        }
        while j < vd {
            *vtp.add(j) = 0.0;
            j += 1;
        }
    }
    for i in 0..head_qk_dim {
        let k_i = *k.get_unchecked(i);
        if k_i == 0.0 {
            continue;
        }
        let k_b = _mm512_set1_ps(k_i);
        let row = sp.add(i * vd);
        let mut j = 0;
        while j < vd16 {
            let acc = _mm512_loadu_ps(vtp.add(j));
            let rv = _mm512_loadu_ps(row.add(j));
            _mm512_storeu_ps(vtp.add(j), _mm512_fmadd_ps(rv, k_b, acc));
            j += 16;
        }
        while j < vd {
            *vtp.add(j) += *row.add(j) * k_i;
            j += 1;
        }
    }
    // v_tilde = (v - v_tilde) * beta.
    {
        let vp = v.as_ptr();
        let beta_b = _mm512_set1_ps(beta);
        let mut j = 0;
        while j < vd16 {
            let vv = _mm512_loadu_ps(vp.add(j));
            let tv = _mm512_loadu_ps(vtp.add(j));
            _mm512_storeu_ps(vtp.add(j), _mm512_mul_ps(_mm512_sub_ps(vv, tv), beta_b));
            j += 16;
        }
        while j < vd {
            *vtp.add(j) = (*vp.add(j) - *vtp.add(j)) * beta;
            j += 1;
        }
    }

    // 3. Rank-1 update.
    for i in 0..head_qk_dim {
        let k_i = *k.get_unchecked(i);
        if k_i == 0.0 {
            continue;
        }
        let k_b = _mm512_set1_ps(k_i);
        let row = sp.add(i * vd);
        let mut j = 0;
        while j < vd16 {
            let sv = _mm512_loadu_ps(row.add(j));
            let tv = _mm512_loadu_ps(vtp.add(j));
            _mm512_storeu_ps(row.add(j), _mm512_fmadd_ps(k_b, tv, sv));
            j += 16;
        }
        while j < vd {
            *row.add(j) += k_i * *vtp.add(j);
            j += 1;
        }
    }

    // 4. Output.
    let op = out.as_mut_ptr();
    {
        let mut j = 0;
        while j < vd16 {
            _mm512_storeu_ps(op.add(j), _mm512_setzero_ps());
            j += 16;
        }
        while j < vd {
            *op.add(j) = 0.0;
            j += 1;
        }
    }
    for i in 0..head_qk_dim {
        let q_i = *q.get_unchecked(i);
        if q_i == 0.0 {
            continue;
        }
        let q_b = _mm512_set1_ps(q_i);
        let row = sp.add(i * vd);
        let mut j = 0;
        while j < vd16 {
            let acc = _mm512_loadu_ps(op.add(j));
            let rv = _mm512_loadu_ps(row.add(j));
            _mm512_storeu_ps(op.add(j), _mm512_fmadd_ps(rv, q_b, acc));
            j += 16;
        }
        while j < vd {
            *op.add(j) += q_i * *row.add(j);
            j += 1;
        }
    }
}

/// Substep 3.5: gated RMSNorm, applied per-head after the recurrence.
///
/// Computes `x := rmsnorm(x) * silu(gate)` (the fla reference uses
/// `FusedRMSNormGated` which fuses the silu-and-multiply with the norm
/// for speed — we keep them as two steps here since the kernels
/// already exist). Operates on a single head's vector.
///
/// `norm_weight` is the learned per-channel scale (length `head_v_dim`).
pub fn gated_rmsnorm_f32_inplace(
    x: &mut [f32],
    gate: &[f32],
    norm_weight: &[f32],
    eps: f32,
) {
    let mut scratch = vec![0.0f32; x.len()];
    gated_rmsnorm_f32_scratch(x, gate, norm_weight, eps, &mut scratch);
}

/// Allocation-free variant of [`gated_rmsnorm_f32_inplace`] for hot
/// loops (the hybrid decode path calls this per head per SSM layer —
/// the allocating version cost ~2 heap allocations × n_heads ×
/// n_layers per decoded token). `scratch.len() >= x.len()`.
///
/// Bitwise-identical to the allocating version: rmsnorm first (same
/// row kernel), then `x[i] = norm[i] * silu(gate[i])` with silu
/// computed element-wise exactly as [`silu_f32_inplace`] does.
pub fn gated_rmsnorm_f32_scratch(
    x: &mut [f32],
    gate: &[f32],
    norm_weight: &[f32],
    eps: f32,
    scratch: &mut [f32],
) {
    assert_eq!(x.len(), gate.len());
    assert_eq!(x.len(), norm_weight.len());
    assert!(scratch.len() >= x.len());
    let tmp = &mut scratch[..x.len()];
    // `rmsnorm_f32` already dispatches to AVX-512/AVX2/scalar internally.
    rmsnorm_f32(x, norm_weight, tmp, eps);
    // Fused `x = tmp * silu(gate)` — the only remaining scalar loop.
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above; lengths asserted.
            unsafe { gated_silu_mul_avx2(x, tmp, gate) };
            return;
        }
    }
    gated_silu_mul_scalar(x, tmp, gate);
}

/// Scalar `dst = normed * silu(gate)` — the trailing step of
/// [`gated_rmsnorm_f32_scratch`], kept as the parity reference and
/// non-x86 fallback (numerically-stable SiLU, matching
/// [`silu_f32_inplace_scalar`]).
pub(crate) fn gated_silu_mul_scalar(dst: &mut [f32], normed: &[f32], gate: &[f32]) {
    for ((d, &n), &g) in dst.iter_mut().zip(normed.iter()).zip(gate.iter()) {
        let s = if g >= 0.0 {
            1.0 / (1.0 + (-g).exp())
        } else {
            let e = g.exp();
            e / (1.0 + e)
        };
        *d = n * (g * s);
    }
}

/// AVX2 `dst = normed * silu(gate)` using the shared
/// [`crate::expf_approx_avx2`]. Tracks the scalar reference to ≈1e-3
/// (the expf polynomial's budget, same as the SwiGLU `silu_mul_f32`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn gated_silu_mul_avx2(dst: &mut [f32], normed: &[f32], gate: &[f32]) {
    use std::arch::x86_64::*;
    let n = dst.len();
    let n8 = n & !7;
    let dp = dst.as_mut_ptr();
    let np = normed.as_ptr();
    let gp = gate.as_ptr();
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let mut p = 0;
    while p < n8 {
        let g = _mm256_loadu_ps(gp.add(p));
        let nv = _mm256_loadu_ps(np.add(p));
        // silu(g) = g / (1 + exp(-g))
        let e = crate::expf_approx_avx2(_mm256_sub_ps(zero, g));
        let silu = _mm256_div_ps(g, _mm256_add_ps(one, e));
        _mm256_storeu_ps(dp.add(p), _mm256_mul_ps(nv, silu));
        p += 8;
    }
    while p < n {
        let g = *gp.add(p);
        let s = if g >= 0.0 {
            1.0 / (1.0 + (-g).exp())
        } else {
            let e = g.exp();
            e / (1.0 + e)
        };
        *dp.add(p) = *np.add(p) * (g * s);
        p += 1;
    }
}

/// Numerically stable softplus: `ln(1 + exp(x))`.
fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Numerically stable sigmoid.
fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Tiny scalar matvec: `out[i] = sum_j weight[i, j] * x[j]`.
/// Row-major weight `[m, n]`. Used internally by the layer forward;
/// matches the convention of the bigger matvec kernels elsewhere
/// in this crate without depending on their dispatch surface.
fn matvec_f32_row_major(weight: &[f32], x: &[f32], out: &mut [f32], m: usize, n: usize) {
    assert_eq!(weight.len(), m * n);
    assert_eq!(x.len(), n);
    assert_eq!(out.len(), m);
    for i in 0..m {
        let row = &weight[i * n..(i + 1) * n];
        let mut acc = 0.0f32;
        for j in 0..n {
            acc += row[j] * x[j];
        }
        out[i] = acc;
    }
}

/// Single-token forward for one **Gated DeltaNet** layer (the
/// non-attention layer kind in `qwen35moe`-family models).
///
/// **State**: `conv_state` and `recurrent_state` are mutated in
/// place. The caller (engine) carries them across decode steps.
///
/// ## Shapes
///
/// | Argument | Shape | Notes |
/// |---|---|---|
/// | `hidden_in` | `[d_model]` | post-attn-norm input |
/// | `w_attn_qkv` | `[2 * ssm_inner, d_model]` | fused `[q, k, v]` projection (q+k = d_model, v = ssm_inner; total 2 × ssm_inner) |
/// | `w_attn_gate` | `[ssm_inner, d_model]` | gate path |
/// | `w_ssm_conv1d` | `[kernel, 2 * ssm_inner]` | depthwise conv1d, kernel typically 4 |
/// | `w_ssm_alpha` | `[n_v_heads, d_model]` | `a_proj` (decay-control) |
/// | `w_ssm_beta` | `[n_v_heads, d_model]` | `b_proj` (learning-rate, sigmoid'd post-proj) |
/// | `ssm_a_log` | `[n_v_heads]` | per-head learned decay constant |
/// | `ssm_dt_bias` | `[n_v_heads]` | bias on `a_proj` pre-softplus |
/// | `ssm_norm_weight` | `[head_v_dim]` | per-head RMSNorm scale |
/// | `w_ssm_out` | `[d_model, ssm_inner]` | output projection |
/// | `conv_state` | `[(kernel - 1) * 2 * ssm_inner]` | conv carry buffer |
/// | `recurrent_state` | `[n_v_heads * head_qk_dim * head_v_dim]` | per-head Delta-Rule state |
/// | `out` | `[d_model]` | residual-add output (caller adds to residual) |
///
/// ## Per-token math
/// 1. Project: `qkv = w_attn_qkv @ hidden`, `gate = w_attn_gate @ hidden`,
///    `alpha = w_ssm_alpha @ hidden`, `beta = sigmoid(w_ssm_beta @ hidden)`.
/// 2. Conv1d depthwise on `qkv` (8192 channels), then SiLU activation.
/// 3. Split conv'd qkv into `q[0..d_model]`, `k[d_model..2*d_model]`,
///    `v[2*d_model..2*ssm_inner]` (wait — this only works when
///    `q_dim + k_dim + v_dim = 2 * ssm_inner`, which on `qwen35moe`
///    is `2048 + 2048 + 4096 = 8192 = 2 * 4096`). The layout
///    parameter `head_qk_dim * n_qk_heads` pins the split.
/// 4. For each V-head h:
///    - `decay_h = -exp(ssm_a_log[h]) * softplus(alpha[h] + ssm_dt_bias[h])`
///    - K-head pick: `kh = h * n_qk_heads / n_v_heads` (GQA grouping)
///    - `delta_rule_step` with that head's q/k/v slices
/// 5. Apply gated RMSNorm per-head: `out_h = rmsnorm(out_h) * silu(gate_h)`.
/// 6. Project back: `out_final = w_ssm_out @ out_concat`.
///
/// ## Caveats
/// - **Correctness vs reference is NOT YET validated** — Phase 8 of the
///   roadmap will diff this against a Python reference forward. Until
///   then this function "executes" but the output is suspect.
/// - The conv1d state shift treats `kernel = 4` as the typical case
///   (the buffer holds 3 prior tokens). For decode-only Phase 3
///   that's the entire signature; for prefill (Phase 8) a separate
///   chunked variant will land.
#[allow(clippy::too_many_arguments)]
pub fn delta_net_layer_forward_f32(
    hidden_in: &[f32],
    w_attn_qkv: &[f32],
    w_attn_gate: &[f32],
    w_ssm_conv1d: &[f32],
    w_ssm_alpha: &[f32],
    w_ssm_beta: &[f32],
    ssm_a_log: &[f32],
    ssm_dt_bias: &[f32],
    ssm_norm_weight: &[f32],
    w_ssm_out: &[f32],
    conv_state: &mut [f32],
    recurrent_state: &mut [f32],
    out: &mut [f32],
    d_model: usize,
    ssm_inner: usize,
    n_qk_heads: usize,
    n_v_heads: usize,
    head_qk_dim: usize,
    head_v_dim: usize,
    conv_kernel: usize,
    rms_eps: f32,
) {
    assert_eq!(hidden_in.len(), d_model);
    assert_eq!(out.len(), d_model);
    assert_eq!(n_qk_heads * head_qk_dim, d_model, "QK head dims must sum to d_model");
    assert_eq!(n_v_heads * head_v_dim, ssm_inner, "V head dims must sum to ssm_inner");
    assert_eq!(ssm_a_log.len(), n_v_heads);
    assert_eq!(ssm_dt_bias.len(), n_v_heads);
    assert_eq!(ssm_norm_weight.len(), head_v_dim);
    assert_eq!(n_v_heads % n_qk_heads, 0, "V heads must be a multiple of QK heads (GQA grouping)");
    let v_per_qk = n_v_heads / n_qk_heads;

    let qkv_dim = 2 * ssm_inner; // q (d_model) + k (d_model) + v (ssm_inner)
    assert_eq!(d_model + d_model + ssm_inner, qkv_dim, "qkv dim layout mismatch");
    assert_eq!(w_attn_qkv.len(), qkv_dim * d_model);
    assert_eq!(w_attn_gate.len(), ssm_inner * d_model);
    assert_eq!(w_ssm_conv1d.len(), conv_kernel * qkv_dim);
    assert_eq!(w_ssm_alpha.len(), n_v_heads * d_model);
    assert_eq!(w_ssm_beta.len(), n_v_heads * d_model);
    assert_eq!(w_ssm_out.len(), d_model * ssm_inner);
    assert_eq!(conv_state.len(), (conv_kernel - 1) * qkv_dim);
    assert_eq!(recurrent_state.len(), n_v_heads * head_qk_dim * head_v_dim);

    // --- 1. Projections ---
    let mut qkv_pre = vec![0.0f32; qkv_dim];
    matvec_f32_row_major(w_attn_qkv, hidden_in, &mut qkv_pre, qkv_dim, d_model);

    let mut gate = vec![0.0f32; ssm_inner];
    matvec_f32_row_major(w_attn_gate, hidden_in, &mut gate, ssm_inner, d_model);

    let mut alpha = vec![0.0f32; n_v_heads];
    matvec_f32_row_major(w_ssm_alpha, hidden_in, &mut alpha, n_v_heads, d_model);

    let mut beta = vec![0.0f32; n_v_heads];
    matvec_f32_row_major(w_ssm_beta, hidden_in, &mut beta, n_v_heads, d_model);
    for b in beta.iter_mut() {
        *b = sigmoid_f32(*b);
    }

    // --- 2. Conv1d depthwise on qkv, then SiLU ---
    let mut qkv_conv = vec![0.0f32; qkv_dim];
    conv1d_depthwise_step_f32(
        &qkv_pre,
        w_ssm_conv1d,
        conv_state,
        &mut qkv_conv,
        qkv_dim,
        conv_kernel,
    );
    silu_f32_inplace(&mut qkv_conv);

    // --- 3. Per-K-head interleaved qkv layout (HF Qwen3-Next
    //         `fix_query_key_value_ordering`). Every K-head's
    //         Q/K/V slots sit contiguously in one block of
    //         `2*head_qk_dim + v_per_qk*head_v_dim` channels.
    //         NOT a flat [all_q | all_k | all_v] layout — the old
    //         flat split was correct only for kh=0 by coincidence.
    let qkv_stride_per_kh = 2 * head_qk_dim + v_per_qk * head_v_dim;
    debug_assert_eq!(qkv_stride_per_kh * n_qk_heads, qkv_dim);

    // --- 4. Per-head Delta Rule ---
    let mut v_out = vec![0.0f32; ssm_inner]; // concatenated per-head outputs
    for h in 0..n_v_heads {
        // Effective decay for this head (the kernel-internal preprocessing
        // confirmed by the fla fused_recurrent reference).
        let decay = -(ssm_a_log[h]).exp() * softplus_f32(alpha[h] + ssm_dt_bias[h]);

        // K-head index this V-head shares with (GQA: 2 V-heads per K-head
        // on qwen35moe with n_v=32, n_qk=16 → v_per_qk=2).
        let kh = h / v_per_qk;
        let v_within_kh = h % v_per_qk;
        let kh_base = kh * qkv_stride_per_kh;
        let q_off = kh_base;
        let k_off = kh_base + head_qk_dim;
        let v_off = kh_base + 2 * head_qk_dim + v_within_kh * head_v_dim;

        // HF `use_qk_l2norm_in_kernel=True` — L2-normalize Q/K per
        // head before the recurrence.
        let mut q_head = qkv_conv[q_off..q_off + head_qk_dim].to_vec();
        let mut k_head = qkv_conv[k_off..k_off + head_qk_dim].to_vec();
        l2norm_f32_inplace(&mut q_head, 1e-6);
        l2norm_f32_inplace(&mut k_head, 1e-6);
        let v_head = &qkv_conv[v_off..v_off + head_v_dim];

        let state_off = h * head_qk_dim * head_v_dim;
        let state_head = &mut recurrent_state[state_off..state_off + head_qk_dim * head_v_dim];

        let mut out_head = vec![0.0f32; head_v_dim];
        delta_rule_step_f32(
            &q_head,
            &k_head,
            v_head,
            decay,
            beta[h],
            state_head,
            &mut out_head,
            head_qk_dim,
            head_v_dim,
        );

        // --- 5. Gated RMSNorm per head ---
        let gate_head = &gate[h * head_v_dim..(h + 1) * head_v_dim];
        gated_rmsnorm_f32_inplace(&mut out_head, gate_head, ssm_norm_weight, rms_eps);

        // Write into concatenated buffer.
        v_out[h * head_v_dim..(h + 1) * head_v_dim].copy_from_slice(&out_head);
    }

    // --- 6. Output projection back to d_model ---
    matvec_f32_row_major(w_ssm_out, &v_out, out, d_model, ssm_inner);
}

// ============================================================================
// Chunked-parallel prefill scan for the Gated Delta Rule
// ============================================================================
//
// The per-token [`delta_rule_step_f32`] recurrence is strictly sequential
// across tokens (each token's `v_tilde` reads the running state). During
// prefill we already have the whole prompt in hand, so we can trade the
// long serial dependency chain (T steps) for a *chunked* scan: split the T
// tokens into fixed-length chunks, compute every chunk's intra-chunk
// operators independently (embarrassingly parallel — rayon), then thread the
// recurrent state through only the (few) chunk boundaries. This is the
// standard chunked / "UT-transform" delta-rule formulation and is
// numerically equivalent (to within f32 rounding) to running the per-token
// step T times.
//
// ## Derivation (one V-head, matching the scalar reference exactly)
//
// Reference per-token math (see `delta_rule_step_f32_with_scratch_scalar`),
// with `S_in` = state entering the chunk, local token index `t`:
//   γ_t = exp(g_t)                                   (per-token decay)
//   D_t = γ_t · S_{t-1}                              (decayed prior state)
//   u_t = β_t·(v_t − Dᵀ_t k_t)                       (= v_tilde)
//   S_t = D_t + k_t u_tᵀ                             (rank-1 update)
//   o_t = S_tᵀ q_t                                   (output, uses updated S_t)
//
// Let `cg_t = Σ_{s=0..t} g_s` be the cumulative log-decay inside the chunk
// (so γ ratios are `exp(cg_t − cg_r)` and never exceed 1 for the decays this
// arch produces, which are always ≤ 0). Unrolling gives, for r ≤ t:
//   S_t = Γ_t·S_in + Σ_{r≤t} exp(cg_t−cg_r)·k_r u_rᵀ,     Γ_t = exp(cg_t)
// and the (t) delta corrections satisfy the unit-lower-triangular system
//   u_t + Σ_{r<t} A[t,r]·u_r = β_t v_t − β_t Γ_t (S_inᵀ k_t)
//   A[t,r] = β_t · exp(cg_t−cg_r) · (k_r·k_t)         (r < t)
// Split the right-hand side into a chunk-local part (`β_t v_t`, independent
// of `S_in`) and a cross part (`−β_t Γ_t S_inᵀ k_t = S_inᵀ p_t`,
// `p_t = −β_t Γ_t k_t`). Because forward substitution is linear:
//   U        = U_local + W·S_in
//   U_local  solves (I+T)·U_local = diag(β)·V           (chunk-local)
//   W        solves (I+T)·W        = P                   (chunk-local, [L×qk])
// The outputs and the boundary state then decompose into a chunk-local piece
// plus a cheap `·S_in` matmul:
//   o_t       = O_intra_local[t] + (B·W·S_in)[t] + (Q̂·S_in)[t]
//               B[t,r] = exp(cg_t−cg_r)(k_r·q_t) for r≤t, Q̂_t = Γ_t q_t
//   S_out     = Γ_{L-1}·S_in + K̂ᵀ·U,  K̂_r = exp(cg_{L-1}−cg_r) k_r
// Everything named `_local`, plus `W`, `B·W` (=`bw`), `Q̂` (=`qg`) and `K̂`
// (=`kd`) depends only on this chunk's q/k/v/g/β, so it is built in parallel;
// the serial boundary pass is just a handful of `[L×qk]·[qk×v]` matmuls per
// chunk carrying `S_in`.
//
// ## Equivalence caveat
// This is exact in real arithmetic. In f32 the reordered accumulation (a
// cumulative `exp` of summed log-decays vs. repeated multiply, and a
// triangular solve vs. the direct recurrence) drifts by a few ulp per step;
// with the decays this arch produces (g ≤ 0 ⇒ every `exp(cg_t−cg_r) ≤ 1`,
// so nothing amplifies) and L2-normed q/k it tracks the sequential path to
// well under 1e-4 for realistic chunk sizes. There is no gating term that
// forces a fallback: the decay `g` is the only gate and it is handled
// exactly above. A chunk length of 1 is routed to the literal per-token
// path for bit-for-bit identity.

/// Default chunk length used by [`delta_rule_prefill_chunked`] when
/// `RUSTLLAMA_SSM_CHUNK` is unset or `0`. 64 keeps each chunk's O(L²)
/// intra-chunk work small while still giving rayon enough chunks to spread.
pub const SSM_CHUNK_DEFAULT: usize = 64;

/// Chunk length for the prefill scan, read once from `RUSTLLAMA_SSM_CHUNK`.
/// `0`/unset → [`SSM_CHUNK_DEFAULT`]; `1` forces the exact per-token
/// sequential path. Cached in a `OnceLock` so we don't touch the
/// environment per SSM layer per prefill. (Plain env→value init — not a
/// self-referential `OnceLock`.)
fn ssm_chunk_len() -> usize {
    static CHUNK: OnceLock<usize> = OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("RUSTLLAMA_SSM_CHUNK")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&c| c > 0)
            .unwrap_or(SSM_CHUNK_DEFAULT)
    })
}

/// Per-chunk operators, all independent of the incoming recurrent state so
/// they can be built in parallel. Row-major throughout; `qk`/`vd` are
/// `head_qk_dim`/`head_v_dim`. See the module-level derivation for the math.
struct ChunkPlan {
    start: usize,          // first (global) token index of this chunk
    len: usize,            // L, tokens in this chunk
    u_local: Vec<f32>,     // [L*vd]  (I+T)^{-1} diag(β) V
    w: Vec<f32>,           // [L*qk]  (I+T)^{-1} P,  P_t = -β_t Γ_t k_t
    o_intra_local: Vec<f32>, // [L*vd]  B · U_local
    bw: Vec<f32>,          // [L*qk]  B · W
    qg: Vec<f32>,          // [L*qk]  Q̂_t = Γ_t q_t   (for the inter-chunk output)
    kd: Vec<f32>,          // [L*qk]  K̂_r = exp(cg_{L-1}-cg_r) k_r (state carry)
    last_decay: f32,       // Γ_{L-1} = exp(cg_{L-1})
}

/// Build one chunk's [`ChunkPlan`] (the parallel, `S_in`-independent work).
#[allow(clippy::too_many_arguments)]
fn build_chunk_plan(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    start: usize,
    len: usize,
    qk: usize,
    vd: usize,
) -> ChunkPlan {
    // Cumulative log-decay within the chunk: cg[t] = Σ_{s=0..t} g[start+s].
    let mut cg = vec![0.0f32; len];
    {
        let mut acc = 0.0f32;
        for t in 0..len {
            acc += g[start + t];
            cg[t] = acc;
        }
    }

    // Forward-substitution solve of the unit-lower-triangular system for both
    // right-hand sides at once (they share the same T[t,r] coefficients):
    //   u_local: RHS_t = β_t v_t
    //   w      : RHS_t = P_t = -β_t Γ_t k_t
    let mut u_local = vec![0.0f32; len * vd];
    let mut w = vec![0.0f32; len * qk];
    for t in 0..len {
        let gt = start + t;
        let bt = beta[gt];
        // Seed both RHS.
        for j in 0..vd {
            u_local[t * vd + j] = bt * v[gt * vd + j];
        }
        let neg_bg = -bt * cg[t].exp();
        for i in 0..qk {
            w[t * qk + i] = neg_bg * k[gt * qk + i];
        }
        // Subtract T[t,r]·X[r] for r<t, where T[t,r]=β_t·exp(cg_t-cg_r)·(k_r·k_t).
        for r in 0..t {
            let gr = start + r;
            let mut kk = 0.0f32;
            for i in 0..qk {
                kk += k[gt * qk + i] * k[gr * qk + i];
            }
            let coeff = bt * (cg[t] - cg[r]).exp() * kk;
            if coeff != 0.0 {
                let (u_t, u_r) = (t * vd, r * vd);
                for j in 0..vd {
                    u_local[u_t + j] -= coeff * u_local[u_r + j];
                }
                let (w_t, w_r) = (t * qk, r * qk);
                for i in 0..qk {
                    w[w_t + i] -= coeff * w[w_r + i];
                }
            }
        }
    }

    // Intra-chunk output operator B[t,r]=exp(cg_t-cg_r)(k_r·q_t), r≤t, applied
    // to the chunk-local U and W: O_intra_local = B·U_local, bw = B·W.
    let mut o_intra_local = vec![0.0f32; len * vd];
    let mut bw = vec![0.0f32; len * qk];
    for t in 0..len {
        let gt = start + t;
        for r in 0..=t {
            let gr = start + r;
            let mut kq = 0.0f32; // q_t · k_r
            for i in 0..qk {
                kq += q[gt * qk + i] * k[gr * qk + i];
            }
            let b = (cg[t] - cg[r]).exp() * kq;
            if b != 0.0 {
                let (o_t, u_r) = (t * vd, r * vd);
                for j in 0..vd {
                    o_intra_local[o_t + j] += b * u_local[u_r + j];
                }
                let (bw_t, w_r) = (t * qk, r * qk);
                for i in 0..qk {
                    bw[bw_t + i] += b * w[w_r + i];
                }
            }
        }
    }

    // Inter-chunk output keys Q̂_t = Γ_t q_t and state-carry keys
    // K̂_r = exp(cg_{L-1}-cg_r) k_r.
    let mut qg = vec![0.0f32; len * qk];
    let mut kd = vec![0.0f32; len * qk];
    let cg_last = cg[len - 1];
    for t in 0..len {
        let gt = start + t;
        let eg = cg[t].exp(); // Γ_t
        let ekd = (cg_last - cg[t]).exp(); // Γ_{L-1}/Γ_t
        for i in 0..qk {
            qg[t * qk + i] = eg * q[gt * qk + i];
            kd[t * qk + i] = ekd * k[gt * qk + i];
        }
    }

    ChunkPlan {
        start,
        len,
        u_local,
        w,
        o_intra_local,
        bw,
        qg,
        kd,
        last_decay: cg_last.exp(),
    }
}

/// Chunked-parallel prefill scan of the Gated Delta Rule for **one V-head**.
///
/// Numerically equivalent (within f32 tolerance) to calling
/// [`delta_rule_step_f32`] `seq_len` times with the same per-token inputs,
/// carrying `state` across the calls — but the O(L²) intra-chunk work for
/// every chunk is computed in parallel and only the recurrent state is
/// threaded serially across the chunk boundaries.
///
/// ## Parameters (whole-sequence, token-major; `qk`=`head_qk_dim`,
///    `vd`=`head_v_dim`)
/// Each maps directly to the per-token [`delta_rule_step_f32`] argument the
/// caller would otherwise pass token by token:
/// - `q`, `k`: `[seq_len * qk]` — per-token query/key rows, already
///   L2-normalized exactly as the single-token path expects (the caller
///   applies `l2norm_f32_inplace` per token before packing).
/// - `v`: `[seq_len * vd]` — per-token value rows.
/// - `g`: `[seq_len]` — per-token log-decay scalars (the `g`/`decay`
///   argument of the step fn; this arch's values are ≤ 0).
/// - `beta`: `[seq_len]` — per-token learning-rate scalars, already
///   sigmoid'd (the `beta` argument of the step fn).
/// - `state`: `[qk * vd]` recurrent state, row-major with the K-dim outer —
///   the *same* layout as the step fn. Read as the incoming state and
///   **overwritten in place** with the final post-sequence state, so the
///   caller can keep decoding from it.
/// - `out`: `[seq_len * vd]` — per-token outputs, written (token-major).
///
/// Chunk length comes from `RUSTLLAMA_SSM_CHUNK` (see [`ssm_chunk_len`]).
/// Decode (single-token) behavior is unaffected: `seq_len <= 1` and a chunk
/// length of 1 both route to the exact per-token path.
#[allow(clippy::too_many_arguments)]
pub fn delta_rule_prefill_chunked(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    seq_len: usize,
    head_qk_dim: usize,
    head_v_dim: usize,
) {
    delta_rule_prefill_chunked_with_chunk(
        q,
        k,
        v,
        g,
        beta,
        state,
        out,
        seq_len,
        head_qk_dim,
        head_v_dim,
        ssm_chunk_len(),
    );
}

/// [`delta_rule_prefill_chunked`] with an explicit chunk length (mostly for
/// tests / autotuning). `chunk_len == 0` → [`SSM_CHUNK_DEFAULT`];
/// `chunk_len == 1` → the exact per-token sequential path.
#[allow(clippy::too_many_arguments)]
pub fn delta_rule_prefill_chunked_with_chunk(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    seq_len: usize,
    head_qk_dim: usize,
    head_v_dim: usize,
    chunk_len: usize,
) {
    let qk = head_qk_dim;
    let vd = head_v_dim;
    assert_eq!(q.len(), seq_len * qk);
    assert_eq!(k.len(), seq_len * qk);
    assert_eq!(v.len(), seq_len * vd);
    assert_eq!(g.len(), seq_len);
    assert_eq!(beta.len(), seq_len);
    assert_eq!(state.len(), qk * vd);
    assert_eq!(out.len(), seq_len * vd);

    if seq_len == 0 {
        return;
    }

    let l0 = if chunk_len == 0 { SSM_CHUNK_DEFAULT } else { chunk_len };

    // Exact per-token fallback: chunk length 1, or a trivially short
    // sequence. Guarantees decode and `RUSTLLAMA_SSM_CHUNK=1` are bitwise
    // the sequential path.
    if l0 == 1 || seq_len == 1 {
        let mut v_tilde = vec![0.0f32; vd];
        for t in 0..seq_len {
            delta_rule_step_f32_with_scratch(
                &q[t * qk..(t + 1) * qk],
                &k[t * qk..(t + 1) * qk],
                &v[t * vd..(t + 1) * vd],
                g[t],
                beta[t],
                state,
                &mut out[t * vd..(t + 1) * vd],
                qk,
                vd,
                &mut v_tilde,
            );
        }
        return;
    }

    let n_chunks = seq_len.div_ceil(l0);

    // --- Parallel phase: every chunk's S_in-independent operators. ---
    let plans: Vec<ChunkPlan> = {
        use rayon::prelude::*;
        (0..n_chunks)
            .into_par_iter()
            .map(|c| {
                let start = c * l0;
                let len = (start + l0).min(seq_len) - start;
                build_chunk_plan(q, k, v, g, beta, start, len, qk, vd)
            })
            .collect()
    };

    // --- Serial phase: thread the recurrent state through the boundaries. ---
    // Per token we fuse the three `[L×qk]·[qk×vd]` products that fold S_in in:
    //   ws = W·S_in            → U     = U_local + ws
    //   oc = (B·W)·S_in        → O_intra = O_intra_local + oc
    //   oi = Q̂·S_in            → o_t   = O_intra + oi  (inter-chunk output)
    // then update the state:  S ← Γ_{L-1}·S + K̂ᵀ·U.
    let mut u = vec![0.0f32; l0 * vd]; // reused; sized to the max chunk
    let mut ws = vec![0.0f32; vd];
    let mut oi = vec![0.0f32; vd];
    let mut oc = vec![0.0f32; vd];
    for plan in &plans {
        let len = plan.len;
        let start = plan.start;
        for t in 0..len {
            ws.iter_mut().for_each(|x| *x = 0.0);
            oi.iter_mut().for_each(|x| *x = 0.0);
            oc.iter_mut().for_each(|x| *x = 0.0);
            let w_t = &plan.w[t * qk..(t + 1) * qk];
            let qg_t = &plan.qg[t * qk..(t + 1) * qk];
            let bw_t = &plan.bw[t * qk..(t + 1) * qk];
            for i in 0..qk {
                let wi = w_t[i];
                let qi = qg_t[i];
                let bi = bw_t[i];
                let srow = &state[i * vd..(i + 1) * vd];
                for j in 0..vd {
                    let s = srow[j];
                    ws[j] += wi * s;
                    oi[j] += qi * s;
                    oc[j] += bi * s;
                }
            }
            let u_row = &mut u[t * vd..(t + 1) * vd];
            let out_row = &mut out[(start + t) * vd..(start + t + 1) * vd];
            let ul = &plan.u_local[t * vd..(t + 1) * vd];
            let ol = &plan.o_intra_local[t * vd..(t + 1) * vd];
            for j in 0..vd {
                let uj = ul[j] + ws[j];
                u_row[j] = uj;
                out_row[j] = ol[j] + oc[j] + oi[j];
            }
        }

        // State carry: S ← Γ_{L-1}·S first (does not depend on U), then add
        // K̂ᵀ·U (uses the U just computed, not S).
        let decay = plan.last_decay;
        for s in state.iter_mut() {
            *s *= decay;
        }
        for t in 0..len {
            let kd_t = &plan.kd[t * qk..(t + 1) * qk];
            let u_row = &u[t * vd..(t + 1) * vd];
            for i in 0..qk {
                let ki = kd_t[i];
                if ki != 0.0 {
                    let srow = &mut state[i * vd..(i + 1) * vd];
                    for j in 0..vd {
                        srow[j] += ki * u_row[j];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify conv1d state-shifting is a true depthwise convolution.
    /// Setup: 2 channels, kernel 3. Feed 4 tokens through; check the
    /// last output matches a hand-rolled conv.
    #[test]
    fn conv1d_step_matches_hand_rolled_window() {
        let channels = 2;
        let kernel = 3;
        // weight[c, k] row-major (channel outer, kernel inner) —
        // matches the GGUF `ssm_conv1d.weight` layout (ne=[kernel,channels]).
        // c=0: k=0,1,2 → 0.5, 0.25, 0.1
        // c=1: k=0,1,2 → 1.0, 2.0, 3.0
        let weight: Vec<f32> = vec![
            0.5, 0.25, 0.1,
            1.0, 2.0, 3.0,
        ];
        let inputs = [
            [1.0_f32, 1.0_f32],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ];
        let mut state = vec![0.0f32; (kernel - 1) * channels];
        let mut last_out = vec![0.0f32; channels];
        for x in &inputs {
            conv1d_depthwise_step_f32(x, &weight, &mut state, &mut last_out, channels, kernel);
        }
        // After 4 tokens with state init 0, the window for token 3 is
        // [token 1, token 2, token 3] = [[2,2], [3,3], [4,4]].
        // out[0] = 0.5*2 + 0.25*3 + 0.1*4 = 1.0 + 0.75 + 0.4 = 2.15
        // out[1] = 1.0*2 + 2.0*3 + 3.0*4 = 2 + 6 + 12 = 20.0
        assert!((last_out[0] - 2.15).abs() < 1e-5, "got {}", last_out[0]);
        assert!((last_out[1] - 20.0).abs() < 1e-5, "got {}", last_out[1]);
    }

    #[test]
    fn silu_is_x_times_sigmoid_x() {
        let mut x = vec![-2.0f32, 0.0, 1.0, 5.0];
        silu_f32_inplace(&mut x);
        // SiLU(-2) = -2 * sigmoid(-2) = -2 * 0.119... ≈ -0.2384
        // SiLU(0) = 0
        // SiLU(1) = 1 * sigmoid(1) = 1 * 0.731... ≈ 0.7311
        // SiLU(5) = 5 * sigmoid(5) ≈ 5 * 0.9933 ≈ 4.9665
        let expect = [-0.2384, 0.0, 0.7311, 4.9665];
        for (got, exp) in x.iter().zip(expect.iter()) {
            assert!((got - exp).abs() < 1e-3, "got {got} expected {exp}");
        }
    }

    /// Test delta rule with identity-like inputs. With state=0,
    /// g=0 (decay=1), beta=1: first call sets S = outer(k, v),
    /// output = q^T @ outer(k, v) = (q·k) * v.
    #[test]
    fn delta_rule_first_token_matches_qk_v() {
        let head_qk = 4;
        let head_v = 3;
        let q = vec![1.0_f32, 0.0, 0.0, 0.0];
        let k = vec![0.5_f32, 0.5, 0.0, 0.0];
        let v = vec![1.0_f32, 2.0, 3.0];
        let mut state = vec![0.0f32; head_qk * head_v];
        let mut out = vec![0.0f32; head_v];
        delta_rule_step_f32(&q, &k, &v, 0.0, 1.0, &mut state, &mut out, head_qk, head_v);
        // After first call: S = outer(k, v_tilde) where v_tilde = v.
        // out = q^T @ S = q[0] * S[0, :] = 1.0 * (k[0] * v) = 0.5 * v.
        let expect = [0.5, 1.0, 1.5];
        for (got, exp) in out.iter().zip(expect.iter()) {
            assert!((got - exp).abs() < 1e-5, "got {got} expected {exp}");
        }
    }

    /// Test the decay term: with state pre-populated, g=ln(0.5) → decay
    /// factor 0.5, second-call output before the rank-1 update should
    /// reflect the decayed prior state.
    #[test]
    fn delta_rule_decay_halves_prior_state() {
        let head_qk = 2;
        let head_v = 2;
        // Set initial state S = [[1, 0], [0, 1]] (identity).
        let mut state = vec![1.0_f32, 0.0, 0.0, 1.0];
        // Apply: g = ln(0.5), q = [1, 1], k = v = 0. With k=0 there's
        // no rank-1 update, and v - S^T @ k = 0 - 0 = 0 so v_tilde = 0.
        // After decay: S = 0.5 * I. Output = q^T @ S = [0.5, 0.5].
        let q = vec![1.0_f32, 1.0];
        let k = vec![0.0_f32, 0.0];
        let v = vec![0.0_f32, 0.0];
        let mut out = vec![0.0f32; head_v];
        let g = 0.5_f32.ln();
        delta_rule_step_f32(&q, &k, &v, g, 1.0, &mut state, &mut out, head_qk, head_v);
        assert!((out[0] - 0.5).abs() < 1e-5, "got {}", out[0]);
        assert!((out[1] - 0.5).abs() < 1e-5, "got {}", out[1]);
        // State should be the decayed identity.
        assert!((state[0] - 0.5).abs() < 1e-5);
        assert!((state[3] - 0.5).abs() < 1e-5);
    }

    /// Full DeltaNet layer end-to-end. Tiny synthetic dimensions
    /// (d_model=8, ssm_inner=16, 2 QK heads × 4, 4 V heads × 4)
    /// matching the layout that the real qwen35moe model uses
    /// at scale. The test asserts:
    /// 1. The forward executes without panicking
    /// 2. State buffers are mutated (not left at their init values)
    /// 3. Output has the right shape and is finite
    /// 4. Output changes between two distinct inputs (the layer
    ///    is not a no-op)
    #[test]
    fn delta_net_layer_forward_executes_end_to_end() {
        let d_model = 8;
        let ssm_inner = 16;
        let n_qk_heads = 2;
        let n_v_heads = 4; // 2 V-heads per K-head
        let head_qk_dim = 4; // 2 * 4 = 8 = d_model
        let head_v_dim = 4; // 4 * 4 = 16 = ssm_inner
        let conv_kernel = 4;
        let qkv_dim = 2 * ssm_inner; // = 32 = 8 (q) + 8 (k) + 16 (v)
        let eps = 1e-5;

        // Deterministic small weights via a tiny LCG.
        let mut s: u32 = 0x12345678;
        let mut rng = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let mk = |n: usize, rng: &mut dyn FnMut() -> f32| -> Vec<f32> {
            (0..n).map(|_| rng() * 0.1).collect()
        };

        let w_attn_qkv = mk(qkv_dim * d_model, &mut rng);
        let w_attn_gate = mk(ssm_inner * d_model, &mut rng);
        let w_ssm_conv1d = mk(conv_kernel * qkv_dim, &mut rng);
        let w_ssm_alpha = mk(n_v_heads * d_model, &mut rng);
        let w_ssm_beta = mk(n_v_heads * d_model, &mut rng);
        // A_log = ln(0.5) → exp(A_log) = 0.5 → moderate decay
        let ssm_a_log: Vec<f32> = vec![0.5_f32.ln(); n_v_heads];
        let ssm_dt_bias: Vec<f32> = vec![0.0; n_v_heads];
        let ssm_norm_weight: Vec<f32> = vec![1.0; head_v_dim];
        let w_ssm_out = mk(d_model * ssm_inner, &mut rng);

        let hidden_a: Vec<f32> = (0..d_model).map(|_| rng() * 0.5).collect();
        let hidden_b: Vec<f32> = (0..d_model).map(|_| rng() * 0.5).collect();

        let mut conv_state = vec![0.0f32; (conv_kernel - 1) * qkv_dim];
        let mut rec_state = vec![0.0f32; n_v_heads * head_qk_dim * head_v_dim];
        let mut out_a = vec![0.0f32; d_model];
        let mut out_b = vec![0.0f32; d_model];

        delta_net_layer_forward_f32(
            &hidden_a,
            &w_attn_qkv,
            &w_attn_gate,
            &w_ssm_conv1d,
            &w_ssm_alpha,
            &w_ssm_beta,
            &ssm_a_log,
            &ssm_dt_bias,
            &ssm_norm_weight,
            &w_ssm_out,
            &mut conv_state,
            &mut rec_state,
            &mut out_a,
            d_model, ssm_inner, n_qk_heads, n_v_heads,
            head_qk_dim, head_v_dim, conv_kernel, eps,
        );

        // State should now reflect the first token's contribution.
        assert!(
            rec_state.iter().any(|&v| v != 0.0),
            "recurrent state must be mutated by first call"
        );
        assert!(
            conv_state.iter().any(|&v| v != 0.0),
            "conv state must be mutated by first call"
        );
        // Output should be finite.
        for &v in &out_a {
            assert!(v.is_finite(), "output must be finite, got {v}");
        }

        // Second call with a different input — output should differ.
        delta_net_layer_forward_f32(
            &hidden_b,
            &w_attn_qkv,
            &w_attn_gate,
            &w_ssm_conv1d,
            &w_ssm_alpha,
            &w_ssm_beta,
            &ssm_a_log,
            &ssm_dt_bias,
            &ssm_norm_weight,
            &w_ssm_out,
            &mut conv_state,
            &mut rec_state,
            &mut out_b,
            d_model, ssm_inner, n_qk_heads, n_v_heads,
            head_qk_dim, head_v_dim, conv_kernel, eps,
        );
        let diff: f32 = out_a.iter().zip(out_b.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0, "output should change between distinct inputs");
    }

    /// Test the delta correction: feeding the same (k, v) twice with
    /// beta=1 should produce a v_tilde that subtracts out the prior
    /// contribution. The recurrent state behaves like a "running
    /// estimate" of v given k.
    #[test]
    fn delta_rule_repeated_kv_converges() {
        let head_qk = 1;
        let head_v = 1;
        let k = vec![1.0_f32];
        let v = vec![1.0_f32];
        let mut state = vec![0.0f32; 1];
        let mut out = vec![0.0f32; 1];
        // Pass 1: v_tilde = (1 - 0*1)*1 = 1. S = 0 + 1*1 = 1. Out = 1*1 = 1.
        delta_rule_step_f32(&[1.0], &k, &v, 0.0, 1.0, &mut state, &mut out, head_qk, head_v);
        assert!((out[0] - 1.0).abs() < 1e-5);
        // Pass 2: prior S=1. v_tilde = (1 - 1*1)*1 = 0. S = 1 + 0 = 1.
        // Out = 1 * 1 = 1 (unchanged — the rule has "learned" v|k).
        delta_rule_step_f32(&[1.0], &k, &v, 0.0, 1.0, &mut state, &mut out, head_qk, head_v);
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((state[0] - 1.0).abs() < 1e-5);
    }

    // ---- SIMD-vs-scalar parity ----

    /// Small deterministic LCG → f32 in [-0.5, 0.5).
    fn rvec(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    /// The dispatched `delta_rule_step_f32_with_scratch` must match the
    /// scalar reference across a sequence of tokens (recurrent state
    /// carried) for dims that both hit and miss the SIMD lane width.
    #[test]
    fn delta_rule_simd_matches_scalar() {
        for &(qd, vd) in &[(1usize, 1usize), (3, 5), (8, 8), (7, 9), (16, 16), (128, 128), (17, 131)] {
            let mut st_ref = vec![0.0f32; qd * vd];
            let mut st_simd = vec![0.0f32; qd * vd];
            let mut vt_ref = vec![0.0f32; vd];
            let mut vt_simd = vec![0.0f32; vd];
            let mut out_ref = vec![0.0f32; vd];
            let mut out_simd = vec![0.0f32; vd];
            for t in 0..6u32 {
                let base = (qd * 131 + vd * 17) as u32 + t * 7919;
                let q = rvec(qd, base + 1);
                let k = rvec(qd, base + 2);
                let v = rvec(vd, base + 3);
                let g = rvec(1, base + 4)[0]; // log-decay in [-0.5, 0)
                let beta = 0.5 + 0.5 * rvec(1, base + 5)[0].abs();
                delta_rule_step_f32_with_scratch_scalar(
                    &q, &k, &v, g, beta, &mut st_ref, &mut out_ref, qd, vd, &mut vt_ref,
                );
                delta_rule_step_f32_with_scratch(
                    &q, &k, &v, g, beta, &mut st_simd, &mut out_simd, qd, vd, &mut vt_simd,
                );
                for j in 0..vd {
                    assert!(
                        (out_ref[j] - out_simd[j]).abs() < 1e-4,
                        "out mismatch qd={qd} vd={vd} t={t} j={j}: ref={} simd={}",
                        out_ref[j], out_simd[j]
                    );
                }
                for (a, b) in st_ref.iter().zip(st_simd.iter()) {
                    assert!((a - b).abs() < 1e-4, "state mismatch qd={qd} vd={vd} t={t}");
                }
            }
        }
    }

    /// Direct AVX2 vs scalar (guarantees the AVX2 body is exercised
    /// even on AVX-512 hosts, where the dispatcher would pick 512).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn delta_rule_avx2_matches_scalar() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for &(qd, vd) in &[(3usize, 5usize), (8, 8), (17, 131), (128, 128)] {
            let mut st_ref = vec![0.0f32; qd * vd];
            let mut st_s = vec![0.0f32; qd * vd];
            let mut vt = vec![0.0f32; vd];
            let mut o_ref = vec![0.0f32; vd];
            let mut o_s = vec![0.0f32; vd];
            for t in 0..4u32 {
                let base = (qd + vd) as u32 * 101 + t * 613;
                let q = rvec(qd, base + 1);
                let k = rvec(qd, base + 2);
                let v = rvec(vd, base + 3);
                let g = rvec(1, base + 4)[0];
                let beta = 0.7;
                delta_rule_step_f32_with_scratch_scalar(
                    &q, &k, &v, g, beta, &mut st_ref, &mut o_ref, qd, vd, &mut vt,
                );
                unsafe {
                    delta_rule_step_f32_with_scratch_avx2(
                        &q, &k, &v, g, beta, &mut st_s, &mut o_s, qd, vd, &mut vt,
                    );
                }
                for j in 0..vd {
                    assert!((o_ref[j] - o_s[j]).abs() < 1e-4, "avx2 out qd={qd} vd={vd} j={j}");
                }
            }
        }
    }

    /// Direct AVX-512 vs scalar (only when the host supports it).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn delta_rule_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        for &(qd, vd) in &[(3usize, 5usize), (16, 16), (17, 131), (128, 128)] {
            let mut st_ref = vec![0.0f32; qd * vd];
            let mut st_s = vec![0.0f32; qd * vd];
            let mut vt = vec![0.0f32; vd];
            let mut o_ref = vec![0.0f32; vd];
            let mut o_s = vec![0.0f32; vd];
            for t in 0..4u32 {
                let base = (qd + vd) as u32 * 223 + t * 877;
                let q = rvec(qd, base + 1);
                let k = rvec(qd, base + 2);
                let v = rvec(vd, base + 3);
                let g = rvec(1, base + 4)[0];
                let beta = 0.4;
                delta_rule_step_f32_with_scratch_scalar(
                    &q, &k, &v, g, beta, &mut st_ref, &mut o_ref, qd, vd, &mut vt,
                );
                unsafe {
                    delta_rule_step_f32_with_scratch_avx512(
                        &q, &k, &v, g, beta, &mut st_s, &mut o_s, qd, vd, &mut vt,
                    );
                }
                for j in 0..vd {
                    assert!((o_ref[j] - o_s[j]).abs() < 1e-4, "avx512 out qd={qd} vd={vd} j={j}");
                }
                for (a, b) in st_ref.iter().zip(st_s.iter()) {
                    assert!((a - b).abs() < 1e-4, "avx512 state qd={qd} vd={vd}");
                }
            }
        }
    }

    #[test]
    fn l2norm_simd_matches_scalar() {
        for &d in &[1usize, 3, 7, 8, 9, 16, 17, 128, 129] {
            let x = rvec(d, d as u32 * 31 + 5);
            let mut a = x.clone();
            let mut b = x.clone();
            l2norm_f32_inplace(&mut a, 1e-6);
            l2norm_f32_inplace_scalar(&mut b, 1e-6);
            for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                assert!((av - bv).abs() < 1e-5, "l2norm d={d} i={i}: simd={av} scalar={bv}");
            }
        }
    }

    #[test]
    fn silu_simd_matches_scalar() {
        for &d in &[1usize, 3, 7, 8, 9, 16, 17, 128, 4096] {
            // Scale into a realistic activation range (a few units).
            let x: Vec<f32> = rvec(d, d as u32 * 13 + 2).iter().map(|v| v * 8.0).collect();
            let mut a = x.clone();
            let mut b = x.clone();
            silu_f32_inplace(&mut a);
            silu_f32_inplace_scalar(&mut b);
            for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                // expf-approx budget (matches the crate's silu_mul test).
                assert!((av - bv).abs() < 1e-3, "silu d={d} i={i}: simd={av} scalar={bv}");
            }
        }
    }

    #[test]
    fn gated_rmsnorm_simd_matches_scalar() {
        for &d in &[1usize, 3, 7, 8, 9, 16, 17, 128] {
            let x = rvec(d, d as u32 * 19 + 3);
            let gate: Vec<f32> = rvec(d, d as u32 * 23 + 4).iter().map(|v| v * 6.0).collect();
            let nw = rvec(d, d as u32 * 29 + 5);
            let mut a = x.clone();
            let mut scratch_a = vec![0.0f32; d];
            gated_rmsnorm_f32_scratch(&mut a, &gate, &nw, 1e-5, &mut scratch_a);
            // Reference: identical rmsnorm (already dispatched) + scalar silu.
            let mut tmp = vec![0.0f32; d];
            crate::rmsnorm_f32(&x, &nw, &mut tmp, 1e-5);
            let mut b = vec![0.0f32; d];
            gated_silu_mul_scalar(&mut b, &tmp, &gate);
            for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                assert!((av - bv).abs() < 1e-3, "gated_rmsnorm d={d} i={i}: simd={av} scalar={bv}");
            }
        }
    }

    // ---- Chunked prefill scan vs. sequential per-token reference ----

    /// The chunked-parallel prefill scan must reproduce, for every token,
    /// the exact sequence of outputs (and the final recurrent state) that a
    /// token-by-token `delta_rule_step` recurrence produces. We sweep head
    /// dims that hit and miss SIMD lane widths, a range of sequence lengths,
    /// and several chunk lengths (including 0→default, 1→sequential, and a
    /// chunk larger than the sequence). Inputs mimic the real call site:
    /// per-token Q/K are L2-normed and the log-decays are ≤ 0.
    #[test]
    fn prefill_chunked_matches_sequential_step() {
        let cases = [
            (1usize, 1usize, 5usize),
            (4, 4, 7),
            (8, 8, 33),
            (16, 16, 64),
            (32, 48, 130),
            (48, 32, 200),
            (13, 29, 97),
        ];
        // Tolerance the reordered-accumulation math supports for these
        // (mild, non-amplifying) decays and L2-normed keys.
        let tol = 1e-4f32;
        let mut worst = 0.0f32;

        for &(qk, vd, t_len) in &cases {
            // Build whole-sequence inputs (token-major).
            let mut q = rvec(t_len * qk, (qk * 7 + vd * 13 + t_len * 3) as u32 + 1);
            let mut k = rvec(t_len * qk, (qk * 11 + vd * 5 + t_len * 17) as u32 + 2);
            let v = rvec(t_len * vd, (qk * 3 + vd * 19 + t_len * 23) as u32 + 3);
            // Per-token L2-norm on Q/K, exactly as the layer forward does.
            for t in 0..t_len {
                l2norm_f32_inplace_scalar(&mut q[t * qk..(t + 1) * qk], 1e-6);
                l2norm_f32_inplace_scalar(&mut k[t * qk..(t + 1) * qk], 1e-6);
            }
            // g ≤ 0 (this arch's decays are always negative), β ∈ (0.1, 0.9).
            let gr = rvec(t_len, (t_len * 31 + qk) as u32 + 4);
            let br = rvec(t_len, (t_len * 37 + vd) as u32 + 5);
            let g: Vec<f32> = gr.iter().map(|x| -0.05 - 0.25 * x.abs()).collect();
            let beta: Vec<f32> = br.iter().map(|x| 0.1 + 0.8 * x.abs()).collect();

            // Reference: sequential per-token scalar recurrence.
            let mut st_ref = vec![0.0f32; qk * vd];
            let mut out_ref = vec![0.0f32; t_len * vd];
            let mut vt = vec![0.0f32; vd];
            for t in 0..t_len {
                delta_rule_step_f32_with_scratch_scalar(
                    &q[t * qk..(t + 1) * qk],
                    &k[t * qk..(t + 1) * qk],
                    &v[t * vd..(t + 1) * vd],
                    g[t],
                    beta[t],
                    &mut st_ref,
                    &mut out_ref[t * vd..(t + 1) * vd],
                    qk,
                    vd,
                    &mut vt,
                );
            }

            // Chunked scan for a spread of chunk lengths.
            for &cl in &[0usize, 1, 2, 3, 8, 16, 64, 256] {
                let mut st = vec![0.0f32; qk * vd];
                let mut out_c = vec![0.0f32; t_len * vd];
                delta_rule_prefill_chunked_with_chunk(
                    &q, &k, &v, &g, &beta, &mut st, &mut out_c, t_len, qk, vd, cl,
                );
                for (idx, (a, b)) in out_ref.iter().zip(out_c.iter()).enumerate() {
                    let d = (a - b).abs();
                    worst = worst.max(d);
                    assert!(
                        d < tol,
                        "output mismatch qk={qk} vd={vd} T={t_len} chunk={cl} idx={idx}: \
                         ref={a} chunked={b} (diff={d})"
                    );
                }
                for (a, b) in st_ref.iter().zip(st.iter()) {
                    let d = (a - b).abs();
                    worst = worst.max(d);
                    assert!(
                        d < tol,
                        "final-state mismatch qk={qk} vd={vd} T={t_len} chunk={cl}: \
                         ref={a} chunked={b} (diff={d})"
                    );
                }
            }
        }
        // Informational: the achieved max abs diff across all cases.
        assert!(worst < tol, "worst abs diff {worst} exceeded tol {tol}");
    }

    /// A chunk length of 1 must be *bit-for-bit* the per-token path (it is
    /// routed straight to `delta_rule_step_f32_with_scratch`), so it matches
    /// the dispatched (SIMD) step exactly, not merely within tolerance.
    #[test]
    fn prefill_chunked_len1_is_exact_sequential() {
        let (qk, vd, t_len) = (17usize, 31usize, 40usize);
        let mut q = rvec(t_len * qk, 12345);
        let mut k = rvec(t_len * qk, 22345);
        let v = rvec(t_len * vd, 32345);
        for t in 0..t_len {
            l2norm_f32_inplace(&mut q[t * qk..(t + 1) * qk], 1e-6);
            l2norm_f32_inplace(&mut k[t * qk..(t + 1) * qk], 1e-6);
        }
        let g: Vec<f32> = rvec(t_len, 42345).iter().map(|x| -0.1 - 0.2 * x.abs()).collect();
        let beta: Vec<f32> = rvec(t_len, 52345).iter().map(|x| 0.2 + 0.6 * x.abs()).collect();

        // Dispatched per-token reference.
        let mut st_ref = vec![0.0f32; qk * vd];
        let mut out_ref = vec![0.0f32; t_len * vd];
        let mut vt = vec![0.0f32; vd];
        for t in 0..t_len {
            delta_rule_step_f32_with_scratch(
                &q[t * qk..(t + 1) * qk],
                &k[t * qk..(t + 1) * qk],
                &v[t * vd..(t + 1) * vd],
                g[t],
                beta[t],
                &mut st_ref,
                &mut out_ref[t * vd..(t + 1) * vd],
                qk,
                vd,
                &mut vt,
            );
        }

        let mut st = vec![0.0f32; qk * vd];
        let mut out_c = vec![0.0f32; t_len * vd];
        delta_rule_prefill_chunked_with_chunk(
            &q, &k, &v, &g, &beta, &mut st, &mut out_c, t_len, qk, vd, 1,
        );
        assert_eq!(out_ref, out_c, "chunk=1 outputs must be bit-identical");
        assert_eq!(st_ref, st, "chunk=1 final state must be bit-identical");
    }
}
