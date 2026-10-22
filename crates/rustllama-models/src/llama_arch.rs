//! Forward pass for the Llama family (Llama / Mistral / Qwen2 / DeepSeek /
//! Phi-3 / Yi). All these models share the same block structure:
//!   pre-attn RMSNorm → Q/K/V proj → RoPE → GQA attention → out proj → residual
//!   pre-ffn  RMSNorm → SwiGLU (gate * up → down) → residual
//! ending in a final RMSNorm + LM head.
//!
//! Phase 1 keeps all weights as F16 and runs activations in F32 on the CPU.
//! The same code path will accept SYCL-allocated weights via the `Engine`
//! trait once phase 3 lands.

use rustllama_gguf::{GgmlType, Gguf};
use rustllama_kernels_cpu as k;
use rustllama_tensor::{Dtype, Tensor, TensorError};

use crate::llama_config::{ConfigError, LlamaConfig};

/// Weight matvec dispatch with USM short-circuit. Tries the
/// USM-resident kernel via [`crate::accel::try_matvec_tensor_usm_f32`]
/// first (only Q8_0Raw weights are wired today; the hook returns
/// `false` for every other dtype, so this is a free no-op on
/// non-Q8_0 paths). On USM miss, falls back to the CPU dispatch in
/// `rustllama_kernels_cpu::matvec_tensor`.
#[inline]
pub(crate) fn matvec_tensor_dispatch(
    w: &Tensor,
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k_dim: usize,
) {
    // C1a: the weight must hold exactly `m * k_dim` elements. Every kernel
    // below (CPU + USM) indexes `w` as an `m × k` matrix; an oversized or
    // undersized weight silently reads the wrong rows / stride (this is how
    // the shared-expert-width bug mis-strided in release — debug_assert in
    // the kernels caught it only in tests). Order-agnostic (one u64 compare),
    // negligible next to the matvec. If a legitimate caller intentionally
    // views a sub-range, relax that call site, not this check.
    assert_eq!(
        w.element_count() as usize,
        m * k_dim,
        "matvec_tensor_dispatch: weight `{}` has {} elements, expected m*k = {}*{} = {}",
        w.name,
        w.element_count(),
        m,
        k_dim,
        m * k_dim,
    );
    // C1b (tests only): on a 2-D weight, the DimOrder-tagged orientation
    // must resolve to the SAME (out, in) the caller passes — this catches
    // a TRANSPOSED weight (identical element count, swapped dims) that
    // C1a's order-agnostic u64 compare can't. Guarded to 2-D (out_dim /
    // in_dim's domain) and debug-only: during the DimOrder migration an
    // un-migrated constructor still defaults to GGUF order, so a stale tag
    // must never panic a release build.
    debug_assert!(
        w.shape.len() != 2
            || w.dim_order == rustllama_tensor::DimOrder::Unspecified
            || (w.out_dim(), w.in_dim()) == (m, k_dim),
        "matvec_tensor_dispatch: weight `{}` dim_order {:?} shape {:?} \
         resolves to (out={}, in={}) but caller passed (m={}, k={})",
        w.name,
        w.dim_order,
        w.shape,
        w.out_dim(),
        w.in_dim(),
        m,
        k_dim,
    );
    // Three-tier dispatch:
    //   1. Quantized USM matvec (Q8_0 / Q4_K / Q5_K / Q6_K) —
    //      packed-format-specific kernels with hand-tuned LWS.
    //   2. F16 USM matvec via the generic gemm_f16 kernel — picks
    //      up unquantized models without forcing CPU.
    //   3. CPU fallback for everything else (Bf16, F32, sub-4-bit
    //      K-quants, IQ-family, NVFP4, …).
    //
    // Each tier returns `false` cleanly when it doesn't apply, so
    // the chain reads as "try GPU paths in priority order, then
    // CPU". Adding new GPU dtypes is a new try_* helper + a new
    // arm here.
    // imatrix calibration hook: record this tensor's per-input-column
    // activation sample. Near-free when not collecting (one relaxed
    // atomic load). `x` has length `k_dim` = the tensor's input-column
    // count, exactly the importance vector's width.
    if crate::imatrix_collect::is_collecting() {
        crate::imatrix_collect::record(&w.name, x);
    }

    // H8: F16-input mixed-precision matvec (per-dtype `matvec_f16in:<dtype>`
    // verdict-gated, fail-closed). Tried before the F32 USM path; on miss
    // (dtype off / no verdict) falls straight through with no cost.
    let base_done = crate::accel::try_matvec_tensor_f16in_usm_f32(w, x, out, m, k_dim)
        || crate::accel::try_matvec_tensor_usm_f32(w, x, out, m, k_dim)
        || crate::accel::try_matvec_f16_usm_f32(w, x, out, m, k_dim);
    if !base_done {
        k::matvec_tensor(w, x, out, m, k_dim);
    }
    // P11 (LoRA, Phase 2): additive low-rank delta `y += s·B·(A·x)` for the
    // thread-active adapter, applied once around whichever base path ran. A
    // no-op (one thread_local borrow + Option check, no write to `out`) unless
    // an adapter is active AND adapts `w.name`, so the no-LoRA path stays
    // byte-identical. The batched / gate-up-fused fast paths funnel an
    // *adapted* tensor through this single-row dispatch (see their gates) so
    // the delta is applied exactly once per row.
    crate::lora::apply_delta(&w.name, x, out);
}

/// Numerically-stable logistic sigmoid `1 / (1 + e^-x)`, evaluated as
/// `e^x / (1 + e^x)` for `x < 0` so the `exp` never overflows on large
/// negative inputs. Factored out of the ~8 identical per-head Q-gate /
/// DeltaNet-beta loops in this module; the two-branch form is preserved
/// bit-for-bit (same operations, same order), so gate outputs are
/// unchanged.
#[inline]
fn sigmoid_stable(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Batched analogue of [`matvec_tensor_dispatch`] — computes
/// `out[n, m] = sum_k W[m, k] * x[n, k]` for all `n in 0..n_rows`,
/// `m in 0..m_dim` in one shot. Tries the batched USM kernel first;
/// on miss, falls back to a per-row loop of the single-row dispatch
/// (which itself short-circuits to USM if possible, else CPU). The
/// per-row fallback keeps behavior identical to the pre-batched
/// code path when the GPU isn't available.
#[inline]
pub(crate) fn matvec_tensor_batched_dispatch(
    w: &Tensor,
    x: &[f32],
    out: &mut [f32],
    m_dim: usize,
    k_dim: usize,
    n_rows: usize,
) {
    // C1a: the shared weight is an `m_dim × k_dim` matrix (reused across all
    // `n_rows`); see `matvec_tensor_dispatch`.
    assert_eq!(
        w.element_count() as usize,
        m_dim * k_dim,
        "matvec_tensor_batched_dispatch: weight `{}` has {} elements, expected m*k = {}*{} = {}",
        w.name,
        w.element_count(),
        m_dim,
        k_dim,
        m_dim * k_dim,
    );
    // C1b (tests only): see `matvec_tensor_dispatch`.
    debug_assert!(
        w.shape.len() != 2
            || w.dim_order == rustllama_tensor::DimOrder::Unspecified
            || (w.out_dim(), w.in_dim()) == (m_dim, k_dim),
        "matvec_tensor_batched_dispatch: weight `{}` dim_order {:?} shape {:?} \
         resolves to (out={}, in={}) but caller passed (m={}, k={})",
        w.name,
        w.dim_order,
        w.shape,
        w.out_dim(),
        w.in_dim(),
        m_dim,
        k_dim,
    );
    // P11 (LoRA): when an adapter adapts THIS tensor — EITHER the single
    // thread-active adapter (contiguous path) OR some row's adapter in the
    // active per-row table (batched / ragged path) — bypass the batched USM +
    // PTQ1_0 fast paths (which don't run the per-row LoRA hook) so every row
    // funnels through the single-row `matvec_tensor_dispatch`. The delta then
    // lands exactly once per row, from `ACTIVE_LORA` or `ROW_ADAPTERS[row]` as
    // selected in `apply_delta`. Non-adapted tensors (and the no-LoRA case)
    // keep the fast paths, so default behavior is byte-identical.
    let lora_adapted = crate::lora::lora_adapts(&w.name) || crate::lora::row_lora_adapts(&w.name);
    if !lora_adapted
        && crate::accel::try_matvec_tensor_batched_usm_f32(w, x, out, m_dim, k_dim, n_rows)
    {
        return;
    }
    // PTQ1_0 (the 27B's every projection/FFN/LM-head dtype): the CPU
    // batched kernel decodes each weight row's trits once and replays
    // the dot for all n rows — bitwise-equal to the per-row loop
    // below, ~n× less extraction work and weight DRAM traffic. This
    // was the structural cause of the per-token prefill floor: the
    // "batched" fallback re-streamed all 5.65 GB of weights once per
    // token. GPU runs never reach this CPU fallback for PTQ1_0: the
    // batched USM gate above now maps PTQ1_0 (matching the single-row
    // gate), so `try_matvec_tensor_batched_usm_f32` handles it on-device
    // first — this fallback is the CPU-placed / no-GPU path only.
    if !lora_adapted && w.dtype == rustllama_tensor::Dtype::PTQ1_0Raw && n_rows > 1 {
        k::matvec_tensor_batched(w, x, out, m_dim, k_dim, n_rows);
        return;
    }
    // Fallback: per-row dispatch via the single-row hook, which
    // tries single-row USM then CPU. Identical to the pre-batched
    // behavior. P11: when a LoRA selection is active, set the current row
    // index before each single-row dispatch so `apply_delta` picks this row's
    // adapter from the per-row table (a no-op `Cell` store for the single-
    // active path, where the row index is never read). Gated on `lora_adapted`
    // so the no-LoRA CPU-fallback path pays nothing.
    for i in 0..n_rows {
        if lora_adapted {
            crate::lora::set_current_lora_row(i);
        }
        let x_row = &x[i * k_dim..(i + 1) * k_dim];
        let out_row = &mut out[i * m_dim..(i + 1) * m_dim];
        matvec_tensor_dispatch(w, x_row, out_row, m_dim, k_dim);
    }
}

/// H4: Fused gate + up matvec dispatch. Tries the fused USM kernel
/// (currently Q4_K / Q8_0); on miss, falls back to two independent
/// `matvec_tensor_dispatch` calls. Behavior is identical to the
/// dual-matvec path apart from the dispatch count.
pub(crate) fn matvec_tensor_gate_up_fused_dispatch(
    w_gate: &Tensor,
    w_up: &Tensor,
    x: &[f32],
    gate_out: &mut [f32],
    up_out: &mut [f32],
    m: usize,
    k_dim: usize,
) {
    // C1a: both projections are `m × k_dim` matrices; see
    // `matvec_tensor_dispatch`. Guards the gate/up shared-expert path too.
    assert_eq!(
        w_gate.element_count() as usize,
        m * k_dim,
        "gate_up_fused: gate `{}` has {} elements, expected {}*{} = {}",
        w_gate.name,
        w_gate.element_count(),
        m,
        k_dim,
        m * k_dim,
    );
    assert_eq!(
        w_up.element_count() as usize,
        m * k_dim,
        "gate_up_fused: up `{}` has {} elements, expected {}*{} = {}",
        w_up.name,
        w_up.element_count(),
        m,
        k_dim,
        m * k_dim,
    );
    // C1b (tests only): both projections' tagged orientation must match
    // the caller's (out, in); see `matvec_tensor_dispatch`.
    debug_assert!(
        w_gate.shape.len() != 2
            || w_gate.dim_order == rustllama_tensor::DimOrder::Unspecified
            || (w_gate.out_dim(), w_gate.in_dim()) == (m, k_dim),
        "gate_up_fused: gate `{}` dim_order {:?} shape {:?} resolves to \
         (out={}, in={}) but caller passed (m={}, k={})",
        w_gate.name,
        w_gate.dim_order,
        w_gate.shape,
        w_gate.out_dim(),
        w_gate.in_dim(),
        m,
        k_dim,
    );
    debug_assert!(
        w_up.shape.len() != 2
            || w_up.dim_order == rustllama_tensor::DimOrder::Unspecified
            || (w_up.out_dim(), w_up.in_dim()) == (m, k_dim),
        "gate_up_fused: up `{}` dim_order {:?} shape {:?} resolves to \
         (out={}, in={}) but caller passed (m={}, k={})",
        w_up.name,
        w_up.dim_order,
        w_up.shape,
        w_up.out_dim(),
        w_up.in_dim(),
        m,
        k_dim,
    );
    // During imatrix calibration, force the two-matvec base path so
    // each tensor is recorded exactly once via `matvec_tensor_dispatch`
    // (the fused USM kernel bypasses that hook). No double-count: only
    // the base dispatch records.
    // Backend order mirrors the single-matvec dispatch: native CUDA, then MLX
    // (Apple), then the SYCL USM path. Each is inert off its backend
    // (`cuda_active`/`mlx_active`/USM-ctx checks) and returns false on a miss,
    // falling through to the two-matvec base path below.
    // P11 (LoRA): if either projection is adapted — by the single thread-active
    // adapter OR some row's adapter in the active per-row table — force the
    // two-matvec base path so each `matvec_tensor_dispatch` applies its own
    // delta exactly once (the fused kernels bypass that hook).
    let lora_adapted = crate::lora::lora_adapts(&w_gate.name)
        || crate::lora::lora_adapts(&w_up.name)
        || crate::lora::row_lora_adapts(&w_gate.name)
        || crate::lora::row_lora_adapts(&w_up.name);
    if !lora_adapted
        && !crate::imatrix_collect::is_collecting()
        && (crate::accel::try_matvec_tensor_gate_up_fused_cuda_f32(
            w_gate, w_up, x, gate_out, up_out, m, k_dim,
        ) || crate::accel::try_matvec_tensor_gate_up_fused_mlx_f32(
            w_gate, w_up, x, gate_out, up_out, m, k_dim,
        ) || crate::accel::try_matvec_tensor_gate_up_fused_usm_f32(
            w_gate, w_up, x, gate_out, up_out, m, k_dim,
        ))
    {
        return;
    }
    matvec_tensor_dispatch(w_gate, x, gate_out, m, k_dim);
    matvec_tensor_dispatch(w_up, x, up_out, m, k_dim);
}

/// Rotate a single activation vector when `w` is Hadamard-folded.
/// Returns the slice to feed the matmul: `x` untouched for unfolded
/// weights (or no Hadamard state, or the `RUSTLLAMA_NO_HADAMARD`
/// lever), else the rotated copy in `scratch`. The scratch is
/// reusable immediately after the (synchronous) matmul consuming it.
fn hadamard_pre<'a>(
    had: Option<&HadamardState>,
    w: &Tensor,
    x: &'a [f32],
    scratch: &'a mut Vec<f32>,
) -> &'a [f32] {
    let Some(h) = had else { return x };
    if no_hadamard_enabled() || !h.is_folded(&w.name) {
        return x;
    }
    if scratch.len() < x.len() {
        scratch.resize(x.len(), 0.0);
    }
    k::hadamard::hadamard_forward(
        x,
        h.signs_for(x.len()),
        h.block_size,
        &mut scratch[..x.len()],
    );
    &scratch[..x.len()]
}

/// Rotate `buf` IN PLACE iff `w` is Hadamard-folded (and Hadamard is enabled);
/// returns whether it rotated. Bit-identical to a per-projection [`hadamard_pre`]
/// (same signs/block/order — the rotation depends only on `buf` + its width, not
/// on `w` beyond its fold flag), but lets several folded projections that share
/// one input (q/k/v off a single attn-norm) rotate the input ONCE instead of
/// re-running the WHT per projection into a scratch buffer. The caller must
/// ensure those projections agree on foldedness (they do on Bonsai — asserted).
fn hadamard_rotate_inplace(had: Option<&HadamardState>, w: &Tensor, buf: &mut [f32]) -> bool {
    let Some(h) = had else { return false };
    if no_hadamard_enabled() || !h.is_folded(&w.name) {
        return false;
    }
    k::hadamard::hadamard_forward_inplace(buf, h.signs_for(buf.len()), h.block_size);
    true
}

/// `ssm_out` variant of [`hadamard_pre`]: when the model declares
/// `gdn_v_grouped`, the fold was done in HF's GROUPED V-head order
/// (head `h' = k·rep + r`) while the GDN runtime concatenates heads
/// in TILED order (`h = r·n_k + k`, our `kh = h % n_k` mapping) —
/// permute tiled→grouped first, then sign+WHT. Mirrors the fork's
/// reshape/permute in build_lora_mm (prism-llama-graph.cpp:1561-67).
#[allow(clippy::too_many_arguments)]
fn hadamard_pre_ssm_out<'a>(
    had: Option<&HadamardState>,
    w: &Tensor,
    x: &'a [f32],
    head_v_dim: usize,
    n_qk_heads: usize,
    v_per_qk: usize,
    perm_scratch: &mut Vec<f32>,
    scratch: &'a mut Vec<f32>,
) -> &'a [f32] {
    let Some(h) = had else { return x };
    if no_hadamard_enabled() || !h.is_folded(&w.name) {
        return x;
    }
    let width = x.len();
    debug_assert_eq!(width, head_v_dim * n_qk_heads * v_per_qk);
    let src: &[f32] = if h.gdn_v_grouped && v_per_qk > 1 {
        if perm_scratch.len() < width {
            perm_scratch.resize(width, 0.0);
        }
        for k_grp in 0..n_qk_heads {
            for r in 0..v_per_qk {
                let src_off = (r * n_qk_heads + k_grp) * head_v_dim; // tiled h = r·n_k + k
                let dst_off = (k_grp * v_per_qk + r) * head_v_dim; // grouped h' = k·rep + r
                perm_scratch[dst_off..dst_off + head_v_dim]
                    .copy_from_slice(&x[src_off..src_off + head_v_dim]);
            }
        }
        &perm_scratch[..width]
    } else {
        x
    };
    if scratch.len() < width {
        scratch.resize(width, 0.0);
    }
    k::hadamard::hadamard_forward(src, h.signs_for(width), h.block_size, &mut scratch[..width]);
    &scratch[..width]
}

/// Row-batched [`hadamard_pre`] for the chunked prefill paths.
fn hadamard_pre_rows<'a>(
    had: Option<&HadamardState>,
    w: &Tensor,
    rows: &'a [f32],
    width: usize,
    n_rows: usize,
    scratch: &'a mut Vec<f32>,
) -> &'a [f32] {
    let Some(h) = had else { return rows };
    if no_hadamard_enabled() || !h.is_folded(&w.name) {
        return rows;
    }
    debug_assert_eq!(rows.len(), width * n_rows);
    if scratch.len() < rows.len() {
        scratch.resize(rows.len(), 0.0);
    }
    let signs = h.signs_for(width);
    for r in 0..n_rows {
        k::hadamard::hadamard_forward(
            &rows[r * width..(r + 1) * width],
            signs,
            h.block_size,
            &mut scratch[r * width..(r + 1) * width],
        );
    }
    &scratch[..rows.len()]
}

/// Dense SwiGLU FFN for hybrid layers (`HybridFfn::Dense`):
/// `out = w_down( silu(w_gate(x)) * w_up(x) )`. Scratch slices must
/// be at least `d_ff` long (the hybrid forwards size them to
/// `cfg.d_ff`, which is the dense FFN width on non-MoE hybrids).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dense_ffn_one_into(
    w_gate: &Tensor,
    w_up: &Tensor,
    w_down: &Tensor,
    x: &[f32],
    d: usize,
    d_ff: usize,
    out: &mut [f32],
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    ff_buf: &mut [f32],
    had: Option<&HadamardState>,
    had_buf: &mut Vec<f32>,
) {
    // The fused dispatch feeds ONE activation to both weights, so
    // gate/up must agree on foldedness (they always do on real
    // Bonsai files — both are in weight_names).
    debug_assert!(
        had.map_or(true, |h| h.is_folded(&w_gate.name)
            == h.is_folded(&w_up.name)),
        "hadamard: ffn_gate/ffn_up foldedness must match (fused dispatch)"
    );
    {
        let x_in = hadamard_pre(had, w_gate, x, had_buf);
        let gate = &mut gate_buf[..d_ff];
        let up = &mut up_buf[..d_ff];
        matvec_tensor_gate_up_fused_dispatch(w_gate, w_up, x_in, gate, up, d_ff, d);
    }
    k::silu_mul_f32(&gate_buf[..d_ff], &up_buf[..d_ff], &mut ff_buf[..d_ff]);
    // B1 (Kindi `k_smfold` spirit): the down-proj input is the SiLU-gate output
    // in `ff_buf` — locally owned and consumed only by the matmul below — so
    // rotate it IN PLACE instead of copying the rotation into `had_buf`.
    // Bit-identical to the former `hadamard_pre`; drops one scratch round-trip.
    hadamard_rotate_inplace(had, w_down, &mut ff_buf[..d_ff]);
    matvec_tensor_dispatch(w_down, &ff_buf[..d_ff], out, d, d_ff);
}

/// Chunked dense SwiGLU FFN for the batched hybrid prefill — the
/// dense sibling of [`crate::moe::moe_ffn_chunk_into_parts`]'s
/// shared-expert section. Scratch is allocated per call like the MoE
/// chunk path (once per layer per chunk; noise next to the matmuls).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dense_ffn_chunk_into(
    w_gate: &Tensor,
    w_up: &Tensor,
    w_down: &Tensor,
    hidden_rows: &[f32],
    n_tokens: usize,
    d: usize,
    d_ff: usize,
    out_rows: &mut [f32],
    had: Option<&HadamardState>,
    had_buf: &mut Vec<f32>,
) {
    debug_assert_eq!(hidden_rows.len(), n_tokens * d);
    debug_assert_eq!(out_rows.len(), n_tokens * d);
    if n_tokens == 0 {
        return;
    }
    debug_assert!(
        had.map_or(true, |h| h.is_folded(&w_gate.name)
            == h.is_folded(&w_up.name)),
        "hadamard: ffn_gate/ffn_up foldedness must match (shared rotated rows)"
    );
    let mut gate_rows = vec![0.0f32; n_tokens * d_ff];
    let mut up_rows = vec![0.0f32; n_tokens * d_ff];
    let mut ff_rows = vec![0.0f32; n_tokens * d_ff];
    {
        let x_rows = hadamard_pre_rows(had, w_gate, hidden_rows, d, n_tokens, had_buf);
        matvec_tensor_batched_dispatch(w_gate, x_rows, &mut gate_rows, d_ff, d, n_tokens);
        matvec_tensor_batched_dispatch(w_up, x_rows, &mut up_rows, d_ff, d, n_tokens);
    }
    for t in 0..n_tokens {
        k::silu_mul_f32(
            &gate_rows[t * d_ff..(t + 1) * d_ff],
            &up_rows[t * d_ff..(t + 1) * d_ff],
            &mut ff_rows[t * d_ff..(t + 1) * d_ff],
        );
    }
    let ff_in = hadamard_pre_rows(had, w_down, &ff_rows, d_ff, n_tokens, had_buf);
    matvec_tensor_batched_dispatch(w_down, ff_in, out_rows, d, d_ff, n_tokens);
}

/// Numerically-stable in-place softmax over `x` (subtract max, exp,
/// normalize). Used by the MLA attention paths.
#[inline]
fn mla_softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut m = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// M1 — ABSORBED MLA attention for one query position (DeepSeek-V2/V3).
///
/// Folds `attn_kv_b` into the per-head Q and output projections so the
/// score runs against the stored `[kv_lora_rank]` latent directly, never
/// materializing per-head K/V:
///   - nope score `q_nope · k_nope[p]` = `q_nope · (W_nope·c[p])`
///     = `(W_nopeᵀ·q_nope) · c[p]` = `q_abs · c[p]`
///   - output `Σ_p a[p]·v[p]` = `Σ_p a[p]·(W_v·c[p])` = `W_v·(Σ_p a[p]·c[p])`
///     = `W_v · c_bar`
///
/// `w_kv_b_f32` is the dequanted `attn_kv_b` laid out row-major
/// `[n_heads·(qk_nope+v_hd), kv_lora_rank]` (flat `o·kvr + k`). `q_all` is
/// the per-head `[nope | rope]` query (rope slice already rotated).
#[allow(clippy::too_many_arguments)]
fn mla_attend_absorbed(
    w_kv_b_f32: &[f32],
    q_all: &[f32],
    latent: &[f32],
    rope_key: &[f32],
    attn_out: &mut [f32],
    n_heads: usize,
    qk_nope: usize,
    qk_rope: usize,
    v_hd: usize,
    kvr: usize,
    kv_b_per_head: usize,
    kv_len: usize,
    scale: f32,
) {
    let qk = qk_nope + qk_rope;
    let mut q_abs = vec![0.0f32; kvr];
    let mut c_bar = vec![0.0f32; kvr];
    let mut scores = vec![0.0f32; kv_len];
    for h in 0..n_heads {
        let q_nope = &q_all[h * qk..h * qk + qk_nope];
        let q_rope = &q_all[h * qk + qk_nope..(h + 1) * qk];
        // Fold q into the latent basis: q_abs[k] = Σ_n W_nope[h][n][k]·q_nope[n].
        for v in q_abs.iter_mut() {
            *v = 0.0;
        }
        for (n, &qn) in q_nope.iter().enumerate() {
            let base = (h * kv_b_per_head + n) * kvr;
            let wrow = &w_kv_b_f32[base..base + kvr];
            for (acc, &w) in q_abs.iter_mut().zip(wrow.iter()) {
                *acc += w * qn;
            }
        }
        // Scores against each cached latent + decoupled rope key.
        for p in 0..kv_len {
            let c = &latent[p * kvr..p * kvr + kvr];
            let kr = &rope_key[p * qk_rope..p * qk_rope + qk_rope];
            let mut s = 0.0f32;
            for (a, b) in q_abs.iter().zip(c.iter()) {
                s += a * b;
            }
            for (a, b) in q_rope.iter().zip(kr.iter()) {
                s += a * b;
            }
            scores[p] = s * scale;
        }
        mla_softmax_inplace(&mut scores[..kv_len]);
        // c_bar = Σ_p a[p]·latent[p].
        for v in c_bar.iter_mut() {
            *v = 0.0;
        }
        for p in 0..kv_len {
            let a = scores[p];
            let c = &latent[p * kvr..p * kvr + kvr];
            for (acc, &cv) in c_bar.iter_mut().zip(c.iter()) {
                *acc += a * cv;
            }
        }
        // out[h][vd] = W_v[h][vd] · c_bar.
        for vd in 0..v_hd {
            let base = (h * kv_b_per_head + qk_nope + vd) * kvr;
            let wrow = &w_kv_b_f32[base..base + kvr];
            let mut acc = 0.0f32;
            for (&w, &cb) in wrow.iter().zip(c_bar.iter()) {
                acc += w * cb;
            }
            attn_out[h * v_hd + vd] = acc;
        }
    }
}

/// M1 — NAIVE MLA attention for one query position. Materializes per-head
/// K/V from the stored latent via the REAL `attn_kv_b` matvec, then runs
/// standard attention. Mathematically identical to [`mla_attend_absorbed`]
/// — the parity anchor that proves the absorbed weight-fold is correct.
#[allow(clippy::too_many_arguments)]
fn mla_attend_naive(
    block: &MlaBlockWeights,
    q_all: &[f32],
    latent: &[f32],
    rope_key: &[f32],
    attn_out: &mut [f32],
    n_heads: usize,
    qk_nope: usize,
    qk_rope: usize,
    v_hd: usize,
    kvr: usize,
    kv_b_per_head: usize,
    kv_len: usize,
    scale: f32,
) {
    let qk = qk_nope + qk_rope;
    let kv_b_out = n_heads * kv_b_per_head;
    // Materialize [nope-K | V] per head per cached position from the latent.
    let mut kvb_all = vec![0.0f32; kv_len * kv_b_out];
    for p in 0..kv_len {
        let c = &latent[p * kvr..p * kvr + kvr];
        matvec_tensor_dispatch(
            &block.w_kv_b,
            c,
            &mut kvb_all[p * kv_b_out..(p + 1) * kv_b_out],
            kv_b_out,
            kvr,
        );
    }
    let mut scores = vec![0.0f32; kv_len];
    for h in 0..n_heads {
        let q_nope = &q_all[h * qk..h * qk + qk_nope];
        let q_rope = &q_all[h * qk + qk_nope..(h + 1) * qk];
        for p in 0..kv_len {
            let off = p * kv_b_out + h * kv_b_per_head;
            let k_nope = &kvb_all[off..off + qk_nope];
            let kr = &rope_key[p * qk_rope..p * qk_rope + qk_rope];
            let mut s = 0.0f32;
            for (a, b) in q_nope.iter().zip(k_nope.iter()) {
                s += a * b;
            }
            for (a, b) in q_rope.iter().zip(kr.iter()) {
                s += a * b;
            }
            scores[p] = s * scale;
        }
        mla_softmax_inplace(&mut scores[..kv_len]);
        for vd in 0..v_hd {
            let mut acc = 0.0f32;
            for p in 0..kv_len {
                let vval = kvb_all[p * kv_b_out + h * kv_b_per_head + qk_nope + vd];
                acc += scores[p] * vval;
            }
            attn_out[h * v_hd + vd] = acc;
        }
    }
}

// Cached hot-path env levers ("set before the first forward"). The
// hybrid decode loop consulted several of these per layer per token;
// `std::env::var` takes the process-wide env lock and allocates, so
// per-call reads were measurable constant overhead. All remain A/B
// levers — they are simply latched at first use.

fn no_rope_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("RUSTLLAMA_NO_ROPE").is_ok())
}

fn no_qgate_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("RUSTLLAMA_NO_QGATE").is_ok())
}

fn debug_hybrid_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("RUSTLLAMA_DEBUG_HYBRID").is_ok())
}

fn skip_attn_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("RUSTLLAMA_SKIP_ATTN").is_ok())
}

/// `RUSTLLAMA_PROFILE_HYBRID_PREFILL=1`: accumulate per-phase wall
/// time inside `forward_prefill_hybrid_impl` and log one table per
/// call. Diagnostic lever for locating the prefill per-row floor.
fn hybrid_prefill_profile_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("RUSTLLAMA_PROFILE_HYBRID_PREFILL").is_ok())
}

fn skip_ffn_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("RUSTLLAMA_SKIP_FFN").is_ok())
}

fn rope_interleaved_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("RUSTLLAMA_ROPE_INTERLEAVED").is_ok())
}

/// Flash-decode KV-length threshold (`RUSTLLAMA_FLASH_KV_LEN_MIN`,
/// default 256). The per-call read also re-parsed the string every
/// layer every token.
fn flash_kv_len_min() -> usize {
    static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("RUSTLLAMA_FLASH_KV_LEN_MIN")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(256)
    })
}

fn include_mtp_layers() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("RUSTLLAMA_INCLUDE_MTP_LAYERS").is_ok())
}

fn dump_acts_dir() -> Option<&'static str> {
    static C: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    C.get_or_init(|| std::env::var("RUSTLLAMA_DUMP_ACTS_DIR").ok())
        .as_deref()
}

fn prefill_batched_enabled() -> bool {
    // T3: read the typed `LoadOptions`-backed accel knob (env-initialized
    // for back-compat, overridden by the engine at load) instead of a
    // per-process `std::env::var` parse cached here.
    crate::accel::prefill_batched()
}

/// A/B lever: `RUSTLLAMA_NO_HADAMARD=1` skips the Prism activation
/// rotation. On a real Bonsai model this MUST produce garbage — a
/// useful positive control when debugging (if output is unchanged,
/// the hooks aren't firing).
fn no_hadamard_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("RUSTLLAMA_NO_HADAMARD").is_ok())
}

/// KV whitening lever: `RUSTLLAMA_KV_WHITEN=0` disables the
/// quantized-KV Hadamard whitening (fork parity: `LLAMA_ATTN_ROT_DISABLE`).
/// Default ON — whitening only ever activates for quantized KV dtypes
/// via [`kv_whiten_active`], never for F32.
fn kv_whiten_disabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("RUSTLLAMA_KV_WHITEN").is_ok_and(|v| v == "0"))
}

/// Chunk width for KV whitening — the fork's measured-best V rotation
/// size. Chunks never cross head boundaries (head_dim % 64 == 0 is the
/// activation gate), so per-head Q·K dot products are exactly
/// preserved when both sides are whitened.
pub(crate) const KV_WHITEN_CHUNK: usize = 64;

/// Whether KV whitening applies inside a `KvLayer::Q4_0` compute arm
/// (the dtype condition is satisfied by the match). Mirrors the fork's
/// `attn_rot_v` gate: head_dim divisible by the chunk width AND not
/// disabled via env. v1 whitens Q4_0 KV only.
#[inline]
pub(crate) fn q4_0_whiten_active(head_dim: usize) -> bool {
    head_dim % KV_WHITEN_CHUNK == 0 && !kv_whiten_disabled()
}

/// Engine-facing predicate: is KV whitening active for this cache
/// configuration? Used to validate a KV-bias sidecar's recorded
/// calibration basis (`kv_mean_center.k_rot`) against the runtime.
pub fn kv_whitening_active_for(dtype: KvDtype, head_dim: usize) -> bool {
    matches!(dtype, KvDtype::Q4_0) && q4_0_whiten_active(head_dim)
}

#[cfg(test)]
mod env_lever_cache_tests {
    /// Every OnceLock-cached env lever must initialize by READING ITS
    /// ENV VAR — never by calling itself. A self-recursive
    /// `get_or_init` closure re-enters `std::sync::Once` and
    /// deadlocks; this shipped once (the first hybrid decode of a
    /// release build froze at the embed step with zero CPU, every
    /// request queued behind it). None of the forward-pass tests
    /// touched these helpers, so the suite stayed green while the
    /// binary hung. Calling each helper here makes a regression hang
    /// the suite instead of slipping through.
    #[test]
    fn cached_env_levers_initialize_without_recursion() {
        for _ in 0..2 {
            let _ = super::no_rope_enabled();
            let _ = super::no_qgate_enabled();
            let _ = super::debug_hybrid_enabled();
            let _ = super::include_mtp_layers();
            let _ = super::dump_acts_dir();
            let _ = super::prefill_batched_enabled();
            let _ = super::skip_attn_enabled();
            let _ = super::skip_ffn_enabled();
            let _ = super::rope_interleaved_enabled();
            let _ = super::flash_kv_len_min();
            let _ = super::hybrid_prefill_profile_enabled();
        }
    }
}

/// Debug instrumentation gated by environment variables. Cheap when
/// the vars are unset, but note `std::env::var` is NOT free — it
/// takes the process env lock. Hot paths use the cached accessors
/// above instead.
mod debug {
    fn stats(s: &[f32]) -> String {
        let mx = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mn = s.iter().cloned().fold(f32::INFINITY, f32::min);
        let rms = (s.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / s.len() as f64).sqrt();
        let nan = s.iter().any(|v| v.is_nan());
        let inf = s.iter().any(|v| v.is_infinite());
        format!("min={mn:.3e} max={mx:.3e} rms={rms:.3e} nan={nan} inf={inf}")
    }

    fn dbg_layer() -> usize {
        std::env::var("RUSTLLAMA_DEBUG_LAYER")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ffn_stats(
        layer_idx: usize,
        hidden: &[f32],
        h_norm: &[f32],
        gate: &[f32],
        up: &[f32],
        ffn_buf: &[f32],
        ffn_out: &[f32],
    ) {
        if std::env::var("RUSTLLAMA_DEBUG_FFN").is_err() || layer_idx != dbg_layer() {
            return;
        }
        eprintln!("[L{layer_idx} FFN] hidden    {}", stats(hidden));
        eprintln!("[L{layer_idx} FFN] h_norm    {}", stats(h_norm));
        eprintln!("[L{layer_idx} FFN] gate_buf  {}", stats(gate));
        eprintln!("[L{layer_idx} FFN] up_buf    {}", stats(up));
        eprintln!("[L{layer_idx} FFN] ffn_buf   {}", stats(ffn_buf));
        eprintln!("[L{layer_idx} FFN] ffn_out   {}", stats(ffn_out));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_stats(
        layer_idx: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        attn_out: &[f32],
        attn_proj: &[f32],
        hidden: &[f32],
    ) {
        if std::env::var("RUSTLLAMA_DEBUG_ATTN").is_err() {
            return;
        }
        eprintln!("[L{layer_idx} ATTN] q          {}", stats(q));
        eprintln!("[L{layer_idx} ATTN] k          {}", stats(k));
        eprintln!("[L{layer_idx} ATTN] v          {}", stats(v));
        eprintln!("[L{layer_idx} ATTN] attn_out   {}", stats(attn_out));
        eprintln!("[L{layer_idx} ATTN] attn_proj  {}", stats(attn_proj));
        eprintln!("[L{layer_idx} ATTN] hidden+a   {}", stats(hidden));
    }

    pub fn trace_layer(layer_idx: usize, hidden: &[f32]) {
        if std::env::var("RUSTLLAMA_TRACE_LAYERS").is_err() {
            return;
        }
        let rms =
            (hidden.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / hidden.len() as f64).sqrt();
        let nan = hidden.iter().any(|v| v.is_nan());
        eprintln!("[layer {layer_idx}] hidden_rms={rms:.4e} nan={nan}");
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LlamaLoadError {
    #[error("config: {0}")]
    Config(#[from] ConfigError),
    #[error("tensor: {0}")]
    Tensor(#[from] TensorError),
    #[error("expected dtype F32 for weight `{0}`")]
    NotF32(String),
    /// P10b — a companion EAGLE draft-head GGUF (`[model].eagle_head` /
    /// `serve --eagle-head`) is present but malformed: not tagged
    /// `adapter.type = "eagle"`, missing the required `eagle.fc.weight`, or a
    /// fusion-projection shape that disagrees with the base `d_model`. Hard
    /// error so a misconfigured head can't silently degrade (or mis-draft)
    /// decode — the fail-closed contract.
    #[error("eagle head: {0}")]
    Eagle(String),
    /// `prism.hadamard.*` metadata present but inconsistent with the
    /// model's tensors, or present on a model shape whose forward
    /// path hasn't wired the activation transform. Hard error: a
    /// Hadamard-folded model run without (or with wrong) rotation
    /// math produces silent garbage, so refusing to load is the only
    /// safe behavior — mirrors the reference implementation.
    #[error("prism.hadamard: {0}")]
    Hadamard(String),
    /// A hybrid (attention + SSM/DeltaNet) model whose DeltaNet head
    /// geometry can't be derived consistently from its tensor shapes
    /// (e.g. `ssm_inner_size` not divisible by the v-head count, a zero
    /// head count, or a layer-count mismatch). Community merges /
    /// fine-tunes that deviate from the base `qwen35moe` head layout the
    /// DeltaNet cache assumes land here. Returned instead of panicking so
    /// a load fails with a clear diagnostic rather than terminating the
    /// whole process mid-decode.
    #[error("unsupported hybrid DeltaNet geometry: {0}")]
    UnsupportedHybridGeometry(String),
}

/// One transformer block's weights, stored as F16.
///
/// F16 halves memory pressure vs F32 — the matvec hot path is memory-bound
/// on a CPU runtime, so this is roughly a 2× end-to-end speedup. Conversion
/// to F32 for FMA happens inline via the AVX2+F16C `matvec_f16_w_f32_a`
/// kernel (one `vcvtph2ps` per 8-lane chunk).
#[derive(Debug)]
pub struct LlamaBlockWeights {
    pub attn_norm: Vec<f32>, // [d_model]
    pub w_q: Tensor,         // [n_heads * head_dim, d_model] F32
    pub w_k: Tensor,         // [n_kv_heads * head_dim, d_model] F32
    pub w_v: Tensor,         // [n_kv_heads * head_dim, d_model] F32
    /// H3: fused QKV weight, populated at load (default-on, A4) whenever
    /// all three of `w_q` / `w_k` / `w_v` are F32 and share `d_model`
    /// (the contraction dim). `concat_qkv_f32` packs them into a single
    /// row-major `[d_q + 2*d_kv, d_model]` tensor; the forward path
    /// dispatches one matvec into a combined output buffer instead of
    /// three separate dispatches — bit-exact to the three matvecs.
    /// Quantized w_q/w_k/w_v aren't yet supported (bit-packed concat is a
    /// future variant); the optional stays `None` and the forward path
    /// falls through to the per-tensor matvecs.
    pub w_qkv_fused: Option<Tensor>,
    pub w_o: Tensor, // [d_model, n_heads * head_dim] F32
    /// Optional attention projection biases. Qwen2 has them; vanilla Llama
    /// and most other Llama-family architectures don't.
    pub b_q: Option<Vec<f32>>, // [n_heads * head_dim]
    pub b_k: Option<Vec<f32>>, // [n_kv_heads * head_dim]
    pub b_v: Option<Vec<f32>>, // [n_kv_heads * head_dim]
    pub ffn_norm: Vec<f32>, // [d_model]
    pub w_gate: Tensor, // [d_ff, d_model] F32
    pub w_up: Tensor, // [d_ff, d_model] F32
    pub w_down: Tensor, // [d_model, d_ff] F32
    /// Per-head Q/K RMSNorm (Qwen3 / Qwen3-MoE family). Shape
    /// `[head_dim]`, applied per-head to Q and K after their
    /// projection and *before* RoPE — the same op the hybrid
    /// `HybridAttnBlockWeights` carries. `None` on every other
    /// Llama-family arch (Qwen2 / Llama / Mistral / …), which ships
    /// no `attn_q_norm` / `attn_k_norm` tensors; the forward then
    /// skips the op entirely (zero overhead for non-Qwen3 models).
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
}

/// Uniform read-only view over the attention tensors of any
/// transformer block. Both [`LlamaBlockWeights`] (dense) and
/// [`LlamaMoeBlockWeights`] (MoE) implement this so the per-layer
/// attention loop in `forward_one` can be written once.
///
/// FFN access is intentionally NOT in this trait — dense and MoE
/// FFN shapes differ enough that we branch explicitly at the FFN
/// call site (dense `w_gate`/`w_up`/`w_down` matvecs vs MoE's
/// `moe::moe_ffn_one`).
pub trait AttnBlock {
    fn attn_norm(&self) -> &[f32];
    fn w_q(&self) -> &Tensor;
    fn w_k(&self) -> &Tensor;
    fn w_v(&self) -> &Tensor;
    fn w_o(&self) -> &Tensor;
    fn b_q(&self) -> Option<&Vec<f32>>;
    fn b_k(&self) -> Option<&Vec<f32>>;
    fn b_v(&self) -> Option<&Vec<f32>>;
    fn ffn_norm(&self) -> &[f32];
    /// Qwen3 per-head Q/K RMSNorm weights (`[head_dim]`). Default
    /// `None` keeps non-Qwen3 impls (and any future ones) untouched;
    /// the two concrete blocks override to expose their field.
    fn q_norm(&self) -> Option<&[f32]> {
        None
    }
    fn k_norm(&self) -> Option<&[f32]> {
        None
    }
}

impl AttnBlock for LlamaBlockWeights {
    fn attn_norm(&self) -> &[f32] {
        &self.attn_norm
    }
    fn w_q(&self) -> &Tensor {
        &self.w_q
    }
    fn w_k(&self) -> &Tensor {
        &self.w_k
    }
    fn w_v(&self) -> &Tensor {
        &self.w_v
    }
    fn w_o(&self) -> &Tensor {
        &self.w_o
    }
    fn b_q(&self) -> Option<&Vec<f32>> {
        self.b_q.as_ref()
    }
    fn b_k(&self) -> Option<&Vec<f32>> {
        self.b_k.as_ref()
    }
    fn b_v(&self) -> Option<&Vec<f32>> {
        self.b_v.as_ref()
    }
    fn ffn_norm(&self) -> &[f32] {
        &self.ffn_norm
    }
    fn q_norm(&self) -> Option<&[f32]> {
        self.q_norm.as_deref()
    }
    fn k_norm(&self) -> Option<&[f32]> {
        self.k_norm.as_deref()
    }
}

impl AttnBlock for LlamaMoeBlockWeights {
    fn attn_norm(&self) -> &[f32] {
        &self.attn_norm
    }
    fn w_q(&self) -> &Tensor {
        &self.w_q
    }
    fn w_k(&self) -> &Tensor {
        &self.w_k
    }
    fn w_v(&self) -> &Tensor {
        &self.w_v
    }
    fn w_o(&self) -> &Tensor {
        &self.w_o
    }
    fn b_q(&self) -> Option<&Vec<f32>> {
        self.b_q.as_ref()
    }
    fn b_k(&self) -> Option<&Vec<f32>> {
        self.b_k.as_ref()
    }
    fn b_v(&self) -> Option<&Vec<f32>> {
        self.b_v.as_ref()
    }
    fn ffn_norm(&self) -> &[f32] {
        &self.ffn_norm
    }
    fn q_norm(&self) -> Option<&[f32]> {
        self.q_norm.as_deref()
    }
    fn k_norm(&self) -> Option<&[f32]> {
        self.k_norm.as_deref()
    }
}

/// One MoE transformer block's weights — same attention path as
/// [`LlamaBlockWeights`], but the FFN is replaced by a router +
/// per-expert gated FFNs (Qwen3-MoE / Mixtral / DeepSeek-V3 shape).
///
/// The expert tensors are stored as single 3D tensors with shape
/// `[n_experts, d_ff, d_model]` (or transposed for down). The
/// forward pass slices per-expert views during compute. This
/// keeps load cost low (one mmap copy per layer × 3, not
/// per-expert × 3) and lets us add per-expert SIMD / SYCL
/// dispatch later without changing the on-disk layout.
#[derive(Debug)]
pub struct LlamaMoeBlockWeights {
    pub attn_norm: Vec<f32>, // [d_model]
    pub w_q: Tensor,         // [n_heads * head_dim, d_model]
    pub w_k: Tensor,         // [n_kv_heads * head_dim, d_model]
    pub w_v: Tensor,         // [n_kv_heads * head_dim, d_model]
    pub w_o: Tensor,         // [d_model, n_heads * head_dim]
    pub b_q: Option<Vec<f32>>,
    pub b_k: Option<Vec<f32>>,
    pub b_v: Option<Vec<f32>>,
    /// Per-head Q/K RMSNorm (Qwen3-MoE family). Shape `[head_dim]`,
    /// applied per-head to Q and K after projection and before RoPE.
    /// `None` on non-Qwen3 MoE archs (Qwen2-MoE / Mixtral / OLMoE /
    /// DeepSeek-V3), which carry no such tensors — the forward then
    /// skips the op. Mirrors [`LlamaBlockWeights::q_norm`].
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>, // [d_model]
    /// Router projection. Shape `[n_experts, d_model]` — produces
    /// one logit per expert which the forward pass softmax+top-K's
    /// to pick the routed experts.
    pub router: Tensor,
    /// D4+M4: per-arch routing policy (gating function, top-K
    /// renormalization, weight scale, group-limited routing). Derived
    /// from `cfg.moe` at load. Default (Mixtral / Qwen3-MoE) reduces to
    /// softmax → top-K → renormalize.
    pub router_spec: crate::moe::RouterSpec,
    /// DeepSeek-V3 routed-expert selection bias (`blk.N.exp_probs_b`).
    /// Length `n_experts`; added to the routing scores for SELECTION only
    /// (not the final weight). `None` on Mixtral / Qwen-MoE.
    pub exp_probs_b: Option<Vec<f32>>,
    /// Gate projection across all experts.
    /// Shape `[n_experts, d_ff, d_model]`.
    pub w_gate_exps: Tensor,
    /// Up projection across all experts. Same shape as gate.
    pub w_up_exps: Tensor,
    /// Down projection across all experts.
    /// Shape `[n_experts, d_model, d_ff]`.
    pub w_down_exps: Tensor,
    /// Shared-expert FFN tensors. Only present on DeepSeek-V3
    /// family models that always-route a fixed FFN in addition to
    /// the top-K routed experts. Tensor names on disk:
    /// `ffn_gate_shexp.weight` / `ffn_up_shexp.weight` /
    /// `ffn_down_shexp.weight`. `None` for Qwen3-MoE / Mixtral.
    pub w_gate_shared: Option<Tensor>,
    pub w_up_shared: Option<Tensor>,
    pub w_down_shared: Option<Tensor>,
    /// Shared-expert sigmoid gate. `Some` only on Qwen2-MoE
    /// (Qwen1.5-MoE), whose always-on shared expert is scaled per
    /// token by `sigmoid(shared_router @ hidden)` (a `[1, d_model]`
    /// projection). DeepSeek-V3 runs its shared expert ungated, so
    /// this is `None` there and the shared contribution is added at
    /// weight 1.0. Matches the `HybridFfn::Moe::shared_router` field
    /// the qwen35moe hybrid path already carries.
    pub shared_router: Option<Tensor>,
    /// Pre-computed per-expert tensor views into `w_gate_exps` /
    /// `w_up_exps` / `w_down_exps`. Each view is a 2D `[d_ff,
    /// d_model]` (or `[d_model, d_ff]` for down) tensor whose
    /// `Storage::CpuOwnedSlice` shares the parent's `Arc<[u8]>`
    /// — no byte duplication, just a struct + cheap Arc clone
    /// per entry. Populated by [`LlamaWeights::from_gguf`] when
    /// loading a MoE GGUF so the forward path can index in O(1)
    /// instead of recomputing the view per token.
    ///
    /// Length always equals `cfg.moe.n_experts`. Empty Vecs would
    /// be a load-time bug — pin via assertions in tests.
    pub gate_per_expert: Vec<Tensor>,
    pub up_per_expert: Vec<Tensor>,
    pub down_per_expert: Vec<Tensor>,
}

/// Hybrid attention+SSM layer for `qwen35moe` and related
/// Mamba/Transformer hybrid models. Differs from
/// [`LlamaMoeBlockWeights`] in two ways:
///   1. The "attention" side is a fused `attn_qkv` projection plus
///      a separate `attn_gate` matrix (instead of separate Q/K/V
///      and an `attn_output` projection). This is a small,
///      cheap attention-flavored side-channel that gates the SSM
///      output — not a full multi-head attention block.
///   2. An SSM (selective state space) block runs in parallel:
///      conv1d over the inner-dim projection, then a selective
///      scan parameterized by `alpha` (B-projection), `beta`
///      (C-projection), and the learned decay `ssm_a`.
///
/// The loader binds the tensors; hybrid models run their forward
/// pass through `forward_one_hybrid` (full-attention layers + the
/// DeltaNet/SSM recurrence), not the dense forward path.
/// Per-layer FFN of a hybrid model. Qwen3.6-family hybrids
/// (`qwen35moe`) carry a routed-expert MoE FFN on every layer;
/// Qwen3.8-family hybrids (Bonsai 2 / dense `qwen35` files without
/// expert metadata) carry a plain dense SwiGLU. One enum instead of
/// two near-duplicate struct families keeps every forward's FFN
/// call site an explicit two-arm match.
#[derive(Debug)]
pub enum HybridFfn {
    /// Routed experts + optional router-gated shared expert
    /// (field-for-field what the two hybrid structs carried before
    /// this enum existed — the MoE forward path is unchanged).
    Moe {
        router: Tensor,
        w_gate_exps: Tensor,
        w_up_exps: Tensor,
        w_down_exps: Tensor,
        w_gate_shared: Option<Tensor>,
        w_up_shared: Option<Tensor>,
        w_down_shared: Option<Tensor>,
        shared_router: Option<Tensor>,
        gate_per_expert: Vec<Tensor>,
        up_per_expert: Vec<Tensor>,
        down_per_expert: Vec<Tensor>,
        /// M1-tail: DeepSeek-V3 routed-expert selection bias
        /// (`blk.N.exp_probs_b`, aka `e_score_correction_bias`). Added to
        /// router scores for expert SELECTION only (never to the routing
        /// weights). `[n_experts]` long when present. `None` on
        /// DeepSeek-V2-Lite (no such tensor) and on qwen35moe hybrids, so
        /// those paths route byte-identically to before.
        exp_probs_b: Option<Vec<f32>>,
    },
    /// Plain dense SwiGLU: `down( silu(gate(x)) * up(x) )` at
    /// `cfg.d_ff` (17408 on Bonsai-2-27B). No expert cache, no
    /// router.
    Dense {
        w_gate: Tensor,
        w_up: Tensor,
        w_down: Tensor,
    },
}

#[derive(Debug)]
pub struct SsmBlockWeights {
    /// Pre-attention RMSNorm. Same name + role as
    /// `LlamaMoeBlockWeights::attn_norm`.
    pub attn_norm: Vec<f32>, // [d_model]
    /// Fused QKV projection. Shape `[3 * heads * head_dim, d_model]`
    /// on disk; the forward pass splits per-head at runtime.
    pub attn_qkv: Tensor,
    /// Attention-side gate, applied multiplicatively after the
    /// attention computation. Shape `[2 * d_model, d_model]` on
    /// disk for `qwen35moe` (4096×2048).
    pub attn_gate: Tensor,
    /// Per-block post-attention RMSNorm (between the attention/SSM
    /// merge and the MoE FFN). Note: this is `post_attention_norm`
    /// on disk, distinct from the standard `ffn_norm`.
    pub post_attention_norm: Vec<f32>, // [d_model]
    /// SSM 1D-conv weight. Shape `[conv_kernel, 2 * ssm_inner]` on
    /// disk for `qwen35moe` (4×8192). The 2x split is because conv
    /// runs over both the data path and the gate path simultaneously.
    pub ssm_conv1d: Tensor,
    /// Pre-dequanted f32 view of `ssm_conv1d`. Populated once at
    /// load time so the per-decode forward doesn't have to dequant
    /// the conv weight on every call. The `kernels-cpu::delta_net::
    /// conv1d_depthwise_step_f32` kernel needs raw f32; the
    /// quantized tensor format isn't kernel-friendly. Memory cost
    /// per SSM layer: `conv_kernel * 2 * ssm_inner * 4 bytes`
    /// = 4 × 8192 × 4 = 128 KB on `qwen35moe`. Across 30 SSM
    /// layers: ~3.8 MB, negligible against the model footprint.
    pub ssm_conv1d_f32: Vec<f32>,
    /// Selective-scan dt bias, used to anchor the time-step
    /// projection. Shape `[ssm_time_step_rank]` (32 for `qwen35moe`).
    pub ssm_dt_bias: Vec<f32>,
    /// Learned per-state decay parameter (the `A` in the Mamba
    /// recurrence `h = A * h + B * x`). Shape `[ssm_time_step_rank]`
    /// (32). Stored as raw values; the forward pass takes `-exp(a)`
    /// per the standard Mamba parameterization.
    pub ssm_a: Vec<f32>,
    /// B-projection (input → state contribution). Shape
    /// `[ssm_time_step_rank, d_model]` (32×2048 for `qwen35moe`).
    pub ssm_alpha: Tensor,
    /// C-projection (state → output contribution). Shape
    /// `[ssm_time_step_rank, d_model]` (32×2048).
    pub ssm_beta: Tensor,
    /// Groupwise RMSNorm applied to the SSM output before the
    /// final out-projection. Shape `[ssm_inner / ssm_group_count]`
    /// (128 = 4096 / 32 for `qwen35moe` — though the dump shows
    /// shape 128 directly, matching the per-group dim).
    pub ssm_norm: Vec<f32>,
    /// Final SSM output projection. Shape `[d_model, ssm_inner]`
    /// (4096×2048 stored as-is for `qwen35moe`; the matvec uses
    /// the transpose).
    pub ssm_out: Tensor,
    /// Layer FFN: routed-expert MoE (qwen35moe) or dense SwiGLU
    /// (Bonsai 2 / dense qwen35-family).
    pub ffn: HybridFfn,
}

/// Full-attention layer used at every `full_attention_interval`-th
/// position in a hybrid model. Structurally identical to
/// [`LlamaMoeBlockWeights`] but reuses the hybrid post-attention
/// RMSNorm key (`post_attention_norm`) instead of `ffn_norm`, and
/// adds the shared-expert router that hybrid layers also carry.
///
/// Functionally this is "MoE block with extra QK norm tensors
/// and a shared-expert router" — Qwen3 family uses per-head
/// `attn_q_norm` / `attn_k_norm` on full-attention layers.
#[derive(Debug)]
pub struct HybridAttnBlockWeights {
    pub attn_norm: Vec<f32>, // [d_model]
    pub w_q: Tensor,
    pub w_k: Tensor,
    pub w_v: Tensor,
    pub w_o: Tensor,
    /// Per-head Q norm (Qwen3 family). Shape `[head_dim]`.
    pub q_norm: Option<Vec<f32>>,
    /// Per-head K norm (Qwen3 family). Shape `[head_dim]`.
    pub k_norm: Option<Vec<f32>>,
    /// Post-attention (pre-FFN) RMSNorm. Stored under the
    /// `post_attention_norm.weight` tensor key, not `ffn_norm.weight`.
    pub post_attention_norm: Vec<f32>,
    /// Layer FFN: routed-expert MoE (qwen35moe) or dense SwiGLU
    /// (Bonsai 2 / dense qwen35-family).
    pub ffn: HybridFfn,
}

/// One layer of a hybrid model. The loader stamps each layer's
/// variant based on `(i + 1) % full_attention_interval == 0`.
#[derive(Debug)]
pub enum HybridLayer {
    /// `(i + 1) % full_attention_interval == 0` → full attention.
    FullAttention(HybridAttnBlockWeights),
    /// All other layers → SSM + gated-attention side-channel.
    Ssm(SsmBlockWeights),
}

/// One Multi-head Latent Attention (MLA) transformer block
/// (DeepSeek-V2/V3, `deepseek2` arch). Clean-room from the public
/// DeepSeek-V2 paper + the GGUF tensor layout.
///
/// Attention tensors (GGUF `[in, out]` order):
///   - **Q**: either the low-rank path `attn_q_a` `[d_model, q_lora_rank]`
///     → RMSNorm `attn_q_a_norm` `[q_lora_rank]` → `attn_q_b`
///     `[q_lora_rank, n_heads·qk_head_dim]`, OR a single direct `attn_q`
///     `[d_model, n_heads·qk_head_dim]` (DeepSeek-V2-Lite). `qk_head_dim
///     = qk_nope_head_dim + qk_rope_head_dim`.
///   - **KV**: `attn_kv_a_mqa` `[d_model, kv_lora_rank + qk_rope_head_dim]`
///     produces the compressed latent (first `kv_lora_rank`, RMSNorm'd by
///     `attn_kv_a_norm`) + the single decoupled-RoPE key (last
///     `qk_rope_head_dim`, shared across heads). `attn_kv_b`
///     `[kv_lora_rank, n_heads·(qk_nope_head_dim + v_head_dim)]`
///     up-projects the latent to per-head non-rotary K and V.
///   - **Out**: `attn_output` `[n_heads·v_head_dim, d_model]`.
///
/// The FFN is a dense SwiGLU or a routed-expert MoE per layer (reusing
/// [`HybridFfn`]) — DeepSeek stacks a few dense layers before the MoE
/// layers (`first_k_dense_replace`).
#[derive(Debug)]
pub struct MlaBlockWeights {
    pub attn_norm: Vec<f32>, // [d_model]
    /// Low-rank Q down-projection `[d_model, q_lora_rank]`. `None` on the
    /// direct-Q variant (V2-Lite), where `w_q` is `Some` instead.
    pub w_q_a: Option<Tensor>,
    /// RMSNorm on the Q latent (`[q_lora_rank]`). `Some` iff `w_q_a` is.
    pub q_a_norm: Option<Vec<f32>>,
    /// Q up-projection `[q_lora_rank, n_heads·qk_head_dim]`. `Some` iff
    /// `w_q_a` is.
    pub w_q_b: Option<Tensor>,
    /// Direct Q projection `[d_model, n_heads·qk_head_dim]` (V2-Lite).
    /// `Some` iff the low-rank Q tensors are absent.
    pub w_q: Option<Tensor>,
    /// Fused KV down-projection → `[kv_lora_rank | qk_rope_head_dim]`.
    pub w_kv_a_mqa: Tensor,
    /// RMSNorm on the compressed KV latent (`[kv_lora_rank]`).
    pub kv_a_norm: Vec<f32>,
    /// KV up-projection `[kv_lora_rank, n_heads·(qk_nope + v_head_dim)]`.
    pub w_kv_b: Tensor,
    /// Dequanted f32 copy of `w_kv_b`, laid out row-major `[out, in]` =
    /// `[n_heads·(qk_nope + v_head_dim), kv_lora_rank]`. Built once at load
    /// so the ABSORBED decode can fold `w_kv_b` into the per-head Q/output
    /// projections (score against the latent directly). The NAIVE path
    /// materializes per-head K/V from the latent via the regular matvec
    /// dispatch on `w_kv_b`, so it doesn't consult this.
    pub w_kv_b_f32: Vec<f32>,
    /// Attention output projection `[n_heads·v_head_dim, d_model]`.
    pub w_o: Tensor,
    pub ffn_norm: Vec<f32>, // [d_model]
    /// Dense SwiGLU or routed-expert MoE FFN (per layer).
    pub ffn: HybridFfn,
}

/// MTP head as it appears on `qwen35moe` (and likely other Qwen
/// hybrid variants going forward). On-disk tensor layout:
/// `blk.{N}.nextn.{eh_proj, enorm, hnorm, shared_head_norm}.weight`
/// where `N` is the last layer index (not a separate `mtp.{i}.*`
/// namespace as DeepSeek-V3 uses).
///
/// Functionally this projects from `concat([embed_t+1, hidden_t])`
/// (a `[2 * d_model]` vector) down to `[d_model]` and feeds the
/// model's main LM head. Phase 6 wires the forward; Phase 2 only
/// binds the four tensors.
#[derive(Debug)]
pub struct NextNHead {
    pub eh_proj: Tensor,            // [d_model, 2 * d_model]
    pub embed_norm: Vec<f32>,       // [d_model]
    pub hidden_norm: Vec<f32>,      // [d_model]
    pub shared_head_norm: Vec<f32>, // [d_model]
}

/// One Multi-Token Prediction (MTP) head — the DeepSeek-V3-style
/// "extra transformer block at the tail" that predicts the +k-th
/// token from the final hidden state. Forward semantics: re-norm
/// the hidden, run a full attention + FFN block, and project to
/// vocab via an optional per-head LM head (`lm_head: None` means
/// the head ties to the model's main `output` / `token_embd`).
///
/// Phase 4a wires loader + storage only — `forward_one_with_mtp_logits`
/// (phase 4b) and the MTP-drafter engine driver (phase 4c) come next.
#[derive(Debug)]
pub struct MtpHead {
    /// Same shape + role as `LlamaBlockWeights.attn_norm` etc — one
    /// MTP head is structurally a transformer block.
    pub block: LlamaBlockWeights,
    /// Per-head LM head. `None` => tied to the model's main `output`
    /// (or `token_embd` if the main head is itself tied). Some DeepSeek
    /// variants emit a dedicated `mtp.{i}.output.weight` per head; most
    /// share with the main LM head.
    pub lm_head: Option<Tensor>,
}

/// P10b — a trained EAGLE draft head (Li et al. 2024, "EAGLE: Speculative
/// Sampling Requires Rethinking Feature Uncertainty").
///
/// Loaded from a COMPANION GGUF (not the base file — detected via
/// `[model].eagle_head` / `serve --eagle-head`, mirroring P11's `--lora`
/// wiring). An EAGLE head is a small autoregressive module that drafts a
/// continuation from the base model's top hidden state + the embedding of the
/// last token. Its structure mirrors the NextN/MTP sidecar ([`MtpHead`] /
/// [`NextNHead`]): a fusion of `concat([token_embed, hidden])` down to
/// `d_model`, one transformer-ish block, and an LM head tied (by default) to
/// the base model's output.
///
/// Unlike MTP's per-position heads, EAGLE runs this SINGLE head REPEATEDLY
/// (see [`LlamaModel::forward_eagle_head`]) to grow a draft of arbitrary depth:
/// each step fuses the running feature with the last drafted token's embedding,
/// runs the block, and emits a next-token distribution whose argmax feeds the
/// following step. The base model then verifies the resulting draft tree with
/// the SHARED tree-spec stack ([`LlamaModel::forward_tree_verify_f32`] +
/// `tree_accept_reject`), so EAGLE only replaces the DRAFTING.
///
/// On-disk tensor namespace (clean-room; `general.type = "adapter"` +
/// `adapter.type = "eagle"`):
///   - `eagle.fc.weight`     `[d_model, 2·d_model]` (required) — fusion proj.
///   - `eagle.fc.bias`       `[d_model]`            (optional).
///   - `eagle.enorm.weight`  `[d_model]`  (optional) — pre-fusion embed norm.
///   - `eagle.hnorm.weight`  `[d_model]`  (optional) — pre-fusion hidden norm.
///   - `eagle.{attn_norm,attn_q,attn_k,attn_v,attn_output,ffn_norm,ffn_gate,
///      ffn_up,ffn_down}.weight` — the decoder block (same shapes as a base
///      `blk.{i}.*` layer); `attn_{q,k,v}.bias` optional.
///   - `eagle.output_norm.weight` `[d_model]` (optional) — else reuse the base
///      `output_norm`.
///   - `eagle.token_embd.weight`  `[vocab, d_model]` (optional) — else reuse
///      the base `token_embd` (the common case: EAGLE ties the embedding).
///   - `eagle.output.weight`      `[vocab, d_model]` (optional) — else tie to
///      the base `output` / `token_embd`.
#[derive(Debug)]
pub struct EagleHead {
    /// Fusion projection of `concat([token_embed, hidden])` `[2·d_model]` down
    /// to `[d_model]` — the block's input hidden. Mirrors `NextNHead::eh_proj`.
    pub fc: Tensor,
    /// Optional fusion bias `[d_model]`.
    pub fc_bias: Option<Vec<f32>>,
    /// Optional RMSNorm applied to the token embedding before fusion.
    pub embed_norm: Option<Vec<f32>>,
    /// Optional RMSNorm applied to the base hidden before fusion.
    pub hidden_norm: Option<Vec<f32>>,
    /// The draft decoder block (attn + FFN), structurally one transformer layer
    /// — reuses [`LlamaBlockWeights`] exactly like [`MtpHead::block`].
    pub block: LlamaBlockWeights,
    /// Optional own output norm. `None` => reuse the base `output_norm`.
    pub output_norm: Option<Vec<f32>>,
    /// Optional own token embedding `[vocab, d_model]`. `None` => reuse the
    /// base `token_embd` (the common, tied case).
    pub token_embd: Option<Tensor>,
    /// Optional own LM head. `None` => tie to the base `output` (or
    /// `token_embd` if the base head is itself tied).
    pub lm_head: Option<Tensor>,
}

impl EagleHead {
    /// Parse an EAGLE draft head from an already-open companion GGUF. `cfg` is
    /// the BASE model's config (the head is validated against the base
    /// `d_model` / `vocab_size`). Fail-closed: a file lacking
    /// `adapter.type = "eagle"` or the required `eagle.fc.weight` is rejected
    /// so a misconfigured path can never silently degrade decode.
    pub fn from_gguf(gguf: &Gguf, cfg: &LlamaConfig) -> Result<Self, LlamaLoadError> {
        let atype = gguf
            .metadata_get("adapter.type")
            .and_then(|v| v.as_string())
            .map(|s| s.to_string());
        if atype.as_deref() != Some("eagle") {
            return Err(LlamaLoadError::Eagle(format!(
                "not an EAGLE head GGUF: `adapter.type` is {atype:?} (want \"eagle\")"
            )));
        }
        let d = cfg.d_model;
        let fc = load_weight(gguf, "eagle.fc.weight")?;
        // The fusion maps [2·d_model] → [d_model]; verify before it reaches the
        // matvec (C1a would panic otherwise — fail loudly here with context).
        if (fc.out_dim(), fc.in_dim()) != (d, 2 * d) {
            return Err(LlamaLoadError::Eagle(format!(
                "eagle.fc.weight is ({}, {}), expected (d_model={d}, 2·d_model={})",
                fc.out_dim(),
                fc.in_dim(),
                2 * d
            )));
        }
        let block = LlamaBlockWeights {
            attn_norm: load_norm(gguf, "eagle.attn_norm.weight")?,
            w_q: load_weight(gguf, "eagle.attn_q.weight")?,
            w_k: load_weight(gguf, "eagle.attn_k.weight")?,
            w_v: load_weight(gguf, "eagle.attn_v.weight")?,
            // The draft head isn't on the base hot path; skip the load-time
            // QKV concat (the forward falls through to three matvecs).
            w_qkv_fused: None,
            w_o: load_weight(gguf, "eagle.attn_output.weight")?,
            b_q: load_optional_norm(gguf, "eagle.attn_q.bias"),
            b_k: load_optional_norm(gguf, "eagle.attn_k.bias"),
            b_v: load_optional_norm(gguf, "eagle.attn_v.bias"),
            ffn_norm: load_norm(gguf, "eagle.ffn_norm.weight")?,
            w_gate: load_weight(gguf, "eagle.ffn_gate.weight")?,
            w_up: load_weight(gguf, "eagle.ffn_up.weight")?,
            w_down: load_weight(gguf, "eagle.ffn_down.weight")?,
            q_norm: load_optional_norm(gguf, "eagle.attn_q_norm.weight"),
            k_norm: load_optional_norm(gguf, "eagle.attn_k_norm.weight"),
        };
        Ok(Self {
            fc,
            fc_bias: load_optional_norm(gguf, "eagle.fc.bias"),
            embed_norm: load_optional_norm(gguf, "eagle.enorm.weight"),
            hidden_norm: load_optional_norm(gguf, "eagle.hnorm.weight"),
            block,
            output_norm: load_optional_norm(gguf, "eagle.output_norm.weight"),
            token_embd: load_optional_weight(gguf, "eagle.token_embd.weight"),
            lm_head: load_optional_weight(gguf, "eagle.output.weight"),
        })
    }

    /// Open + parse an EAGLE head from a companion GGUF path, validated against
    /// the base model's `cfg`.
    pub fn from_path(path: &std::path::Path, cfg: &LlamaConfig) -> Result<Self, LlamaLoadError> {
        let gguf = Gguf::open(path)
            .map_err(|e| LlamaLoadError::Eagle(format!("gguf open failed: {e}")))?;
        Self::from_gguf(&gguf, cfg)
    }
}

#[derive(Debug)]
pub struct LlamaWeights {
    pub token_embd: Tensor, // [vocab, d_model] F32
    /// Dense transformer blocks. Empty when the model is MoE
    /// (use [`Self::moe_blocks`] instead) or hybrid (use
    /// [`Self::hybrid_layers`] instead).
    pub blocks: Vec<LlamaBlockWeights>,
    /// MoE transformer blocks. `None` for dense / hybrid models;
    /// `Some` when the GGUF carries `{arch}.expert_count > 0`
    /// and is not also hybrid (no `full_attention_interval`).
    /// Mutually exclusive with a non-empty `blocks` /
    /// `hybrid_layers`.
    pub moe_blocks: Option<Vec<LlamaMoeBlockWeights>>,
    /// Hybrid attention+SSM transformer blocks. `None` unless the
    /// GGUF declares both `{arch}.expert_count` and
    /// `{arch}.full_attention_interval`. Per-layer variant
    /// (FullAttention vs Ssm) is stamped by the loader based on
    /// `(i + 1) % full_attention_interval == 0`. Mutually
    /// exclusive with `blocks` / `moe_blocks`.
    pub hybrid_layers: Option<Vec<HybridLayer>>,
    /// M1: Multi-head Latent Attention transformer blocks (DeepSeek-V2/V3,
    /// `deepseek2` arch). `None` unless the GGUF declares
    /// `{arch}.attention.kv_lora_rank`. Each block carries the low-rank
    /// Q/KV projections + the compressed-latent KV path + a dense or MoE
    /// FFN. Mutually exclusive with `blocks` / `moe_blocks` /
    /// `hybrid_layers`; the forward routes through `forward_one_mla`.
    pub mla_blocks: Option<Vec<MlaBlockWeights>>,
    pub output_norm: Vec<f32>, // [d_model]
    /// LM head. `None` => use tied `token_embd`.
    pub output: Option<Tensor>, // F32
    /// MTP heads. `None` when `cfg.n_mtp_heads == 0`; `Some` of
    /// length `cfg.n_mtp_heads` when the loader found `mtp.{i}.*`
    /// tensors. Models without MTP load + run through the unchanged
    /// single-head path.
    pub mtp_heads: Option<Vec<MtpHead>>,
    /// `qwen35moe`-style NextN/MTP head (one per model, attached
    /// to the last layer's `blk.{N}.nextn.*` tensors). `None` for
    /// models that don't declare `nextn_predict_layers` under
    /// the hybrid metadata. Phase 2 binds; Phase 6 forwards.
    pub nextn_head: Option<NextNHead>,
    /// PrismML Hadamard rotation state — `Some` on Bonsai-family
    /// ternary models. Validated against the actual tensor table at
    /// load; the forwards consult it before every matmul against a
    /// folded weight.
    pub hadamard: Option<HadamardState>,
}

/// Load-validated Hadamard rotation state (see
/// [`crate::llama_config::HadamardConfig`] for the raw metadata and
/// `rustllama_kernels_cpu::hadamard` for the transform itself).
#[derive(Debug)]
pub struct HadamardState {
    pub block_size: usize,
    /// Names of weights stored in the rotated basis — their input
    /// activation gets the forward rotation right before the matmul.
    pub folded: std::collections::HashSet<String>,
    /// Full-width ±1 sign vectors keyed by activation width. For
    /// `sign_mode = "identity"` these are all-ones vectors,
    /// synthesized at load so the hook path is uniform.
    pub signs_by_width: std::collections::HashMap<usize, std::sync::Arc<Vec<f32>>>,
    /// `token_embd.weight` rows are stored rotated — apply the
    /// inverse rotation to every embedding-lookup result.
    pub embd_inverse: bool,
    /// `ssm_out` inputs need the tiled→grouped V-head permutation
    /// before rotation (fork `gdn_v_grouped`). The permutation dims
    /// come from the GDN geometry at forward time.
    pub gdn_v_grouped: bool,
}

impl HadamardState {
    #[inline]
    pub fn is_folded(&self, name: &str) -> bool {
        self.folded.contains(name)
    }

    /// Sign slice for an activation width. Load-time validation
    /// guarantees presence for every folded weight's input width,
    /// so a miss here is a programming error, not a model error.
    #[inline]
    pub fn signs_for(&self, width: usize) -> &[f32] {
        self.signs_by_width
            .get(&width)
            .unwrap_or_else(|| panic!("hadamard: no sign vector for activation width {width}"))
            .as_slice()
    }
}

impl LlamaWeights {
    /// `true` when this model's transformer blocks are MoE-shaped
    /// (router + expert tensors loaded). `false` for dense models.
    /// Forward-pass entry points consult this to decide between the
    /// dense forward and the MoE forward (routed-expert gating +
    /// per-expert FFN).
    pub fn is_hybrid(&self) -> bool {
        self.hybrid_layers.is_some()
    }

    pub fn is_moe(&self) -> bool {
        self.moe_blocks.is_some()
    }

    /// `true` when this model uses Multi-head Latent Attention
    /// (DeepSeek-V2/V3). The forward entry points route MLA models to
    /// [`LlamaModel::forward_one_mla`] / the serial MLA prefill instead
    /// of the dense/GQA path.
    pub fn is_mla(&self) -> bool {
        self.mla_blocks.is_some()
    }

    /// P11 Phase 1 — merge a LoRA adapter's delta into the F32/F16/BF16 base
    /// weights in place (`W += s·B·A`). Covers the attention projections +
    /// dense FFN + LM head of dense and MoE blocks. Quantized bases are left
    /// packed (counted in [`crate::lora::MergeStats::skipped_quant`]) for the
    /// Phase-2 delta path; MoE expert stacks are 3-D and not LoRA targets.
    /// The fused QKV copy (A4) is dropped on any Q/K/V merge so the forward
    /// re-dispatches the merged per-tensor projections.
    pub fn merge_lora(&mut self, adapter: &crate::lora::LoraAdapter) -> crate::lora::MergeStats {
        let mut stats = crate::lora::MergeStats::default();
        fn apply(
            adapter: &crate::lora::LoraAdapter,
            t: &mut Tensor,
            stats: &mut crate::lora::MergeStats,
        ) {
            match crate::lora::merge_tensor_in_place(adapter, t) {
                crate::lora::MergeOutcome::Merged => stats.merged += 1,
                crate::lora::MergeOutcome::SkippedQuant => stats.skipped_quant += 1,
                crate::lora::MergeOutcome::ShapeMismatch => stats.shape_mismatch += 1,
                crate::lora::MergeOutcome::NotTargeted => {}
            }
        }
        for b in &mut self.blocks {
            let before = stats.merged;
            apply(adapter, &mut b.w_q, &mut stats);
            apply(adapter, &mut b.w_k, &mut stats);
            apply(adapter, &mut b.w_v, &mut stats);
            if stats.merged != before {
                b.w_qkv_fused = None;
            }
            apply(adapter, &mut b.w_o, &mut stats);
            apply(adapter, &mut b.w_gate, &mut stats);
            apply(adapter, &mut b.w_up, &mut stats);
            apply(adapter, &mut b.w_down, &mut stats);
        }
        if let Some(moe) = &mut self.moe_blocks {
            for b in moe {
                apply(adapter, &mut b.w_q, &mut stats);
                apply(adapter, &mut b.w_k, &mut stats);
                apply(adapter, &mut b.w_v, &mut stats);
                apply(adapter, &mut b.w_o, &mut stats);
            }
        }
        if let Some(o) = &mut self.output {
            apply(adapter, o, &mut stats);
        }
        stats
    }

    /// MoE tiered-expert engine (Phase 1c): promote the hottest experts'
    /// gate/up/down weights into the ACTIVE GPU backend's device memory (CUDA
    /// managed / SYCL USM, via [`crate::accel::upload_bytes_to_device`]) up to
    /// `budget_bytes`, so the per-expert matvec reads them device-resident — no
    /// per-call host→device re-upload (the whole point of the device tier). The
    /// matching dispatch fast path lives in
    /// [`crate::accel::try_matvec_tensor_usm_f32`] (it feeds `device_ptr()`
    /// straight to the kernel when a tensor is `Storage::Device`).
    ///
    /// Returns `(experts_promoted, bytes_promoted)`.
    ///
    /// `ranked` is a usage-ordered expert list (hottest first, from
    /// `UsageLearner::ranked_keys` reading the persisted per-model sidecar):
    /// those experts are promoted FIRST so a tight budget buys the experts the
    /// model actually routes to most, then the remainder fills in `(layer,
    /// expert)` index order. An empty `ranked` (cold start — no sidecar yet) is
    /// pure index order.
    ///
    /// **Correctness is safe by construction:** the device allocation is
    /// CPU-mappable (`cpu_readable` — managed/USM/unified), so `as_bytes` stays
    /// valid everywhere and every non-GPU code path still sees the same bytes.
    /// The worst case of a blind or absent device tier is "no speedup", never a
    /// wrong answer. Already-device-resident experts and non-MoE models are
    /// skipped; a `budget_bytes` of 0, or no active GPU producer
    /// (`upload_bytes_to_device` → `None`, e.g. this unified-memory dev box with
    /// no dedicated VRAM), promotes nothing. Benefit is on dedicated-VRAM GPUs
    /// (the pod) + Apple; it is write-blind here and validated on that HW.
    pub fn promote_experts_to_device(
        &mut self,
        budget_bytes: u64,
        ranked: &[crate::accel::ExpertKey],
    ) -> (usize, u64) {
        if budget_bytes == 0 {
            return (0, 0);
        }
        let Some(blocks) = self.moe_blocks.as_mut() else {
            return (0, 0);
        };
        let n_layers = blocks.len();
        // Per-layer expert count (min of the three per-expert Vecs — always
        // equal; the `min` just guards a malformed load).
        let n_exp: Vec<usize> = blocks
            .iter()
            .map(|b| {
                b.gate_per_expert
                    .len()
                    .min(b.up_per_expert.len())
                    .min(b.down_per_expert.len())
            })
            .collect();
        // NO-DOUBLE, WHOLE-BLOCK promotion. Promote ALL of a block's experts,
        // then FREE the host-side parent `w_*_exps` tensors — the per-expert
        // views now hold their own device (managed) copies, so nothing else
        // references the parent `Arc`, and dropping it releases its host bytes.
        // This caps the transient host overhead at ~ONE block instead of
        // doubling the entire expert set (CUDA managed memory is host-backed),
        // which is the difference between "can't promote a 30B on a RAM-tight
        // host" and "promotes block-by-block, freeing as it goes". Blocks are
        // taken in order of their hottest expert's usage rank (from the
        // persisted sidecar), then layer-index order for the cold tail.
        let mut block_rank = vec![usize::MAX; n_layers];
        for (i, key) in ranked.iter().enumerate() {
            let l = key.layer as usize;
            if l < n_layers && block_rank[l] == usize::MAX {
                block_rank[l] = i;
            }
        }
        let mut block_order: Vec<usize> = (0..n_layers).collect();
        block_order.sort_by_key(|&l| (block_rank[l], l));

        let mut used: u64 = 0;
        let mut promoted = 0usize;
        'outer: for l in block_order {
            let ne = n_exp[l];
            if ne == 0 {
                continue;
            }
            let block = &mut blocks[l];
            // Already-promoted block (first view device-resident) → skip.
            if block.gate_per_expert[0].storage.residency() != rustllama_tensor::DeviceBackend::Cpu
            {
                continue;
            }
            // This block's expert bytes = the three parents (all experts).
            let block_bytes = block.w_gate_exps.storage.len_bytes() as u64
                + block.w_up_exps.storage.len_bytes() as u64
                + block.w_down_exps.storage.len_bytes() as u64;
            if block_bytes == 0 || used + block_bytes > budget_bytes {
                // Doesn't fit the remaining budget; a colder/smaller block still
                // might, so keep scanning rather than abort.
                continue;
            }
            // Promote every expert of the block. On producer failure (no active
            // backend, or the host can't back the managed alloc), stop entirely
            // — leaving this block's parents intact so it still runs correctly.
            let mut ok = true;
            for e in 0..ne {
                if !Self::promote_tensor(&mut block.gate_per_expert[e])
                    || !Self::promote_tensor(&mut block.up_per_expert[e])
                    || !Self::promote_tensor(&mut block.down_per_expert[e])
                {
                    ok = false;
                    break;
                }
            }
            if !ok {
                break 'outer;
            }
            // Free the now-unreferenced host-side parents (every view of them is
            // device-resident) — this is the "no-double": host bytes released
            // as managed bytes are added.
            Self::drop_expert_parent(&mut block.w_gate_exps);
            Self::drop_expert_parent(&mut block.w_up_exps);
            Self::drop_expert_parent(&mut block.w_down_exps);
            used += block_bytes;
            promoted += ne;
        }
        // Drain the async managed-memory prefetches each promotion issued
        // (`upload_bytes_to_cuda_managed` → `prefetch_managed`), so the promoted
        // expert pages are device-resident before inference instead of
        // fault-migrating during the first decode steps. One sync for the batch.
        if promoted > 0 {
            crate::accel::cuda_sync_device();
        }
        (promoted, used)
    }

    /// Replace a fully-promoted expert parent (`w_*_exps`) with an empty tensor,
    /// freeing its host bytes. Safe ONLY after every per-expert view of it is
    /// device-resident (they hold independent managed copies, so no view
    /// references the parent `Arc` any more). Shape → `[0]` + empty storage so
    /// every reader — placement byte-count (`per_layer_weight_bytes`, which also
    /// excludes experts when the tier is active), multi-GPU tensor info, the
    /// pagelock walk — sees a consistent 0-byte tensor.
    fn drop_expert_parent(t: &mut Tensor) {
        t.shape = vec![0];
        t.strides = vec![1];
        t.storage = rustllama_tensor::Storage::CpuOwned(std::sync::Arc::from(
            Vec::<u8>::new().into_boxed_slice(),
        ));
    }

    /// Swap one tensor's storage to the active GPU backend's device memory.
    /// Returns `false` (leaving the tensor untouched on CPU) when no producer is
    /// available or the upload fails; a no-op `true` for already-device tensors.
    /// Correctness-preserving: the new storage is CPU-mappable, so `as_bytes`
    /// still yields identical bytes.
    fn promote_tensor(t: &mut Tensor) -> bool {
        if t.storage.residency() != rustllama_tensor::DeviceBackend::Cpu {
            return true;
        }
        // `bytes` borrows `t.storage`; the producer copies them into device
        // memory synchronously and returns an owned Storage, so the borrow ends
        // before the reassignment (NLL).
        let bytes = t.storage.as_bytes();
        let Some(dev) = crate::accel::upload_bytes_to_device(bytes) else {
            return false;
        };
        t.storage = dev;
        true
    }

    /// Byte split for the MoE VRAM-budget projection (Phase 4): `(expert_bytes,
    /// non_expert_bytes)`.
    ///
    /// `expert_bytes` is the total of the per-expert gate/up/down weight views —
    /// exactly the bytes [`Self::promote_experts_to_device`] can move to the
    /// device, so it caps the budget. (The parent `w_*_exps` tensors they view
    /// stay in host RAM and are NOT counted — they never go to VRAM.)
    ///
    /// `non_expert_bytes` is everything else that normal layer placement can put
    /// in VRAM — token embeddings, LM head, and per-block attention
    /// (`w_q/k/v/o`), router, and shared-expert projections. The tiny `Vec<f32>`
    /// norms are omitted (negligible). The auto budget reserves this (assuming it
    /// is all GPU-resident — conservative) plus a margin, and gives the rest to
    /// experts. Non-MoE models report `(0, …)`.
    pub fn vram_byte_breakdown(&self) -> (u64, u64) {
        let b = |t: &Tensor| t.storage.len_bytes() as u64;
        let mut expert = 0u64;
        let mut non_expert = b(&self.token_embd);
        if let Some(out) = self.output.as_ref() {
            non_expert += b(out);
        }
        if let Some(blocks) = self.moe_blocks.as_ref() {
            for blk in blocks {
                non_expert +=
                    b(&blk.w_q) + b(&blk.w_k) + b(&blk.w_v) + b(&blk.w_o) + b(&blk.router);
                for sh in [&blk.w_gate_shared, &blk.w_up_shared, &blk.w_down_shared]
                    .into_iter()
                    .flatten()
                {
                    non_expert += b(sh);
                }
                for e in &blk.gate_per_expert {
                    expert += b(e);
                }
                for e in &blk.up_per_expert {
                    expert += b(e);
                }
                for e in &blk.down_per_expert {
                    expert += b(e);
                }
            }
        }
        (expert, non_expert)
    }

    pub fn from_gguf(gguf: &Gguf, cfg: &LlamaConfig) -> Result<Self, LlamaLoadError> {
        let token_embd = load_weight(gguf, "token_embd.weight")?;

        let output = if cfg.tie_word_embeddings {
            None
        } else {
            Some(load_weight(gguf, "output.weight")?)
        };

        let output_norm = load_norm(gguf, "output_norm.weight")?;

        // Branch by hybrid / MoE / dense. Hybrid wins whenever
        // `full_attention_interval > 0` metadata is present — with
        // OR without expert metadata (qwen35moe is a MoE hybrid;
        // Bonsai-2/Qwen3.8-family files are DENSE hybrids: same GDN
        // + full-attention layer pattern, plain SwiGLU FFN).
        // Populate exactly one of `blocks` / `moe_blocks` /
        // `hybrid_layers`; the other two stay empty / None.
        let (blocks, moe_blocks, hybrid_layers, mla_blocks) = if let Some(hyb_cfg) =
            cfg.hybrid.as_ref()
        {
            let layers = load_hybrid_layers(gguf, cfg, cfg.moe.as_ref(), hyb_cfg)?;
            (Vec::new(), None, Some(layers), None)
        } else if let Some(mla_cfg) = cfg.mla.as_ref() {
            // M1: Multi-head Latent Attention (DeepSeek-V2/V3). The MLA
            // attention tensor set replaces the GQA Q/K/V projections; the
            // FFN stays dense or MoE (per layer — DeepSeek's first-k-dense).
            let layers = load_mla_layers(gguf, cfg, mla_cfg, cfg.moe.as_ref())?;
            (Vec::new(), None, None, Some(layers))
        } else if let Some(moe_cfg) = cfg.moe.as_ref() {
            let n_experts = moe_cfg.n_experts as usize;
            let d_model = cfg.d_model;
            let d_ff = cfg.d_ff;
            let mut mbs = Vec::with_capacity(cfg.n_layers);
            for i in 0..cfg.n_layers {
                let prefix = format!("blk.{i}");
                let w_gate_exps = load_weight(gguf, &format!("{prefix}.ffn_gate_exps.weight"))?;
                let w_up_exps = load_weight(gguf, &format!("{prefix}.ffn_up_exps.weight"))?;
                let w_down_exps = load_weight(gguf, &format!("{prefix}.ffn_down_exps.weight"))?;
                // Pre-compute per-expert tensor views. Each share-
                // storage with the parent 3D tensor via
                // Storage::CpuOwnedSlice — no byte duplication,
                // just struct construction + Arc clone. This is
                // the phase 2-D perf-win: the MoE forward path
                // can index into pre-built handles instead of
                // recomputing per token.
                let gate_per_expert: Vec<Tensor> = (0..n_experts)
                    .map(|e| crate::moe::expert_view(&w_gate_exps, e, d_ff, d_model))
                    .collect();
                let up_per_expert: Vec<Tensor> = (0..n_experts)
                    .map(|e| crate::moe::expert_view(&w_up_exps, e, d_ff, d_model))
                    .collect();
                let down_per_expert: Vec<Tensor> = (0..n_experts)
                    .map(|e| crate::moe::expert_view(&w_down_exps, e, d_model, d_ff))
                    .collect();
                // Expert-cache registry: record each expert's mmap
                // ranges so the readahead thread / learning-cache
                // pre-pin can address experts by (layer, expert)
                // without holding tensors. No-op unless zero-copy.
                crate::accel::register_layer_experts(
                    i as u32,
                    &gate_per_expert,
                    &up_per_expert,
                    &down_per_expert,
                );
                mbs.push(LlamaMoeBlockWeights {
                    attn_norm: load_norm(gguf, &format!("{prefix}.attn_norm.weight"))?,
                    w_q: load_weight(gguf, &format!("{prefix}.attn_q.weight"))?,
                    w_k: load_weight(gguf, &format!("{prefix}.attn_k.weight"))?,
                    w_v: load_weight(gguf, &format!("{prefix}.attn_v.weight"))?,
                    w_o: load_weight(gguf, &format!("{prefix}.attn_output.weight"))?,
                    b_q: load_optional_norm(gguf, &format!("{prefix}.attn_q.bias")),
                    b_k: load_optional_norm(gguf, &format!("{prefix}.attn_k.bias")),
                    b_v: load_optional_norm(gguf, &format!("{prefix}.attn_v.bias")),
                    // Qwen3-MoE per-head Q/K norm; absent on other MoE
                    // archs (returns None → forward skips the op).
                    q_norm: load_optional_norm(gguf, &format!("{prefix}.attn_q_norm.weight")),
                    k_norm: load_optional_norm(gguf, &format!("{prefix}.attn_k_norm.weight")),
                    ffn_norm: load_norm(gguf, &format!("{prefix}.ffn_norm.weight"))?,
                    router: load_weight(gguf, &format!("{prefix}.ffn_gate_inp.weight"))?,
                    // D4+M4: per-arch routing policy from the MoE metadata.
                    router_spec: crate::moe::RouterSpec::from(moe_cfg),
                    // DeepSeek-V3 routed-expert selection bias. Tensor name
                    // varies across converters — accept the common spellings.
                    // Only kept when it is a `[n_experts]` vector (so a
                    // same-named tensor of another shape can't be mis-read
                    // and then index out of range in the router).
                    exp_probs_b: load_optional_norm(gguf, &format!("{prefix}.exp_probs_b.bias"))
                        .or_else(|| {
                            load_optional_norm(gguf, &format!("{prefix}.exp_probs_b.weight"))
                        })
                        .or_else(|| {
                            load_optional_norm(gguf, &format!("{prefix}.ffn_gate_inp.bias"))
                        })
                        .filter(|v| v.len() == n_experts),
                    w_gate_exps,
                    w_up_exps,
                    w_down_exps,
                    // Shared-expert tensors are optional — present
                    // only on DeepSeek-V3 family. `load_optional`
                    // returns None when the tensor is absent.
                    w_gate_shared: load_optional_weight(
                        gguf,
                        &format!("{prefix}.ffn_gate_shexp.weight"),
                    ),
                    w_up_shared: load_optional_weight(
                        gguf,
                        &format!("{prefix}.ffn_up_shexp.weight"),
                    ),
                    w_down_shared: load_optional_weight(
                        gguf,
                        &format!("{prefix}.ffn_down_shexp.weight"),
                    ),
                    // GGUF MoE loads (DeepSeek-V3 / Qwen3-MoE) keep the
                    // shared expert ungated; the Qwen2-MoE sigmoid gate
                    // is wired on the MLX/safetensors load path, which
                    // populates this. Preserve existing GGUF behavior.
                    shared_router: None,
                    gate_per_expert,
                    up_per_expert,
                    down_per_expert,
                });
            }
            (Vec::new(), Some(mbs), None, None)
        } else {
            let mut blocks = Vec::with_capacity(cfg.n_layers);
            for i in 0..cfg.n_layers {
                let prefix = format!("blk.{i}");
                let w_q = load_weight(gguf, &format!("{prefix}.attn_q.weight"))?;
                let w_k = load_weight(gguf, &format!("{prefix}.attn_k.weight"))?;
                let w_v = load_weight(gguf, &format!("{prefix}.attn_v.weight"))?;
                // H3 (A4): fuse w_q / w_k / w_v at load time whenever the
                // shapes allow (all three F32 + matching d_model). Default-on,
                // no env gate — the concat is bit-exact to the three separate
                // matvecs. Returns None for quantized weights (the common
                // case; bit-packed concat is a follow-up), so the forward
                // path falls through to the per-tensor matvecs there.
                let w_qkv_fused = concat_qkv_f32(&w_q, &w_k, &w_v);
                blocks.push(LlamaBlockWeights {
                    attn_norm: load_norm(gguf, &format!("{prefix}.attn_norm.weight"))?,
                    w_q,
                    w_k,
                    w_v,
                    w_qkv_fused,
                    w_o: load_weight(gguf, &format!("{prefix}.attn_output.weight"))?,
                    b_q: load_optional_norm(gguf, &format!("{prefix}.attn_q.bias")),
                    b_k: load_optional_norm(gguf, &format!("{prefix}.attn_k.bias")),
                    b_v: load_optional_norm(gguf, &format!("{prefix}.attn_v.bias")),
                    ffn_norm: load_norm(gguf, &format!("{prefix}.ffn_norm.weight"))?,
                    w_gate: load_weight(gguf, &format!("{prefix}.ffn_gate.weight"))?,
                    w_up: load_weight(gguf, &format!("{prefix}.ffn_up.weight"))?,
                    w_down: load_weight(gguf, &format!("{prefix}.ffn_down.weight"))?,
                    // Qwen3 per-head Q/K norm; absent on other dense
                    // archs (returns None → forward skips the op).
                    q_norm: load_optional_norm(gguf, &format!("{prefix}.attn_q_norm.weight")),
                    k_norm: load_optional_norm(gguf, &format!("{prefix}.attn_k_norm.weight")),
                });
            }
            (blocks, None, None, None)
        };

        // MTP heads. Pure-additive: models without MTP metadata have
        // `cfg.n_mtp_heads == 0` and skip the loop entirely. When the
        // model declares MTP heads but the tensors aren't actually
        // present on disk we surface `NotF32` (re-uses the existing
        // missing-tensor error path).
        let mtp_heads = if cfg.n_mtp_heads > 0 {
            let mut heads = Vec::with_capacity(cfg.n_mtp_heads as usize);
            for i in 0..cfg.n_mtp_heads as usize {
                let prefix = format!("mtp.{i}");
                heads.push(MtpHead {
                    block: LlamaBlockWeights {
                        attn_norm: load_norm(gguf, &format!("{prefix}.attn_norm.weight"))?,
                        w_q: load_weight(gguf, &format!("{prefix}.attn_q.weight"))?,
                        w_k: load_weight(gguf, &format!("{prefix}.attn_k.weight"))?,
                        w_v: load_weight(gguf, &format!("{prefix}.attn_v.weight"))?,
                        // H3: MTP heads aren't on the hot path; skip
                        // the load-time concat (could opt-in later).
                        w_qkv_fused: None,
                        w_o: load_weight(gguf, &format!("{prefix}.attn_output.weight"))?,
                        b_q: load_optional_norm(gguf, &format!("{prefix}.attn_q.bias")),
                        b_k: load_optional_norm(gguf, &format!("{prefix}.attn_k.bias")),
                        b_v: load_optional_norm(gguf, &format!("{prefix}.attn_v.bias")),
                        ffn_norm: load_norm(gguf, &format!("{prefix}.ffn_norm.weight"))?,
                        w_gate: load_weight(gguf, &format!("{prefix}.ffn_gate.weight"))?,
                        w_up: load_weight(gguf, &format!("{prefix}.ffn_up.weight"))?,
                        w_down: load_weight(gguf, &format!("{prefix}.ffn_down.weight"))?,
                        // MTP heads (DeepSeek-V3 style) carry no per-head
                        // Q/K norm — Qwen3 uses the NextN head, not MTP.
                        q_norm: None,
                        k_norm: None,
                    },
                    // Per-head LM head is optional — most DeepSeek-V3
                    // variants tie all MTP heads to the main LM head.
                    lm_head: load_optional_weight(gguf, &format!("{prefix}.output.weight")),
                });
            }
            Some(heads)
        } else {
            None
        };

        // NextN/MTP head for hybrid models. On disk it lives at
        // `blk.{N}.nextn.*` where N is the last layer index
        // (40 on `qwen35moe` with n_layers=41). We don't try to
        // load this on non-hybrid models because the standard
        // DeepSeek-V3 MTP path uses a different prefix
        // (`mtp.{i}.*`, handled above).
        let nextn_head = match cfg.hybrid.as_ref() {
            Some(hyb) if hyb.nextn_predict_layers > 0 => {
                let n = cfg.n_layers - 1;
                let prefix = format!("blk.{n}.nextn");
                // All four tensors are required when the count is
                // > 0; if any is missing it's a malformed GGUF.
                Some(NextNHead {
                    eh_proj: load_weight(gguf, &format!("{prefix}.eh_proj.weight"))?,
                    embed_norm: load_norm(gguf, &format!("{prefix}.enorm.weight"))?,
                    hidden_norm: load_norm(gguf, &format!("{prefix}.hnorm.weight"))?,
                    shared_head_norm: load_norm(
                        gguf,
                        &format!("{prefix}.shared_head_norm.weight"),
                    )?,
                })
            }
            _ => None,
        };

        let hadamard = build_hadamard_state(gguf, cfg)?;

        // Dense layer-ahead readahead registry: record each hybrid
        // layer's file-backed weight spans so `note_prefill_layer`
        // can prefetch layer L+1 while L computes (MoE expert pools
        // have their own registry; shared/router tensors register
        // here). No-op unless weights are zero-copy mmap.
        for (i, layer) in hybrid_layers.iter().flatten().enumerate() {
            let mut ts: Vec<&Tensor> = Vec::with_capacity(12);
            let ffn = match layer {
                HybridLayer::Ssm(b) => {
                    ts.extend([
                        &b.attn_qkv,
                        &b.attn_gate,
                        &b.ssm_conv1d,
                        &b.ssm_alpha,
                        &b.ssm_beta,
                        &b.ssm_out,
                    ]);
                    &b.ffn
                }
                HybridLayer::FullAttention(b) => {
                    ts.extend([&b.w_q, &b.w_k, &b.w_v, &b.w_o]);
                    &b.ffn
                }
            };
            match ffn {
                HybridFfn::Dense {
                    w_gate,
                    w_up,
                    w_down,
                } => {
                    ts.extend([w_gate, w_up, w_down]);
                }
                HybridFfn::Moe {
                    router,
                    w_gate_shared,
                    w_up_shared,
                    w_down_shared,
                    shared_router,
                    ..
                } => {
                    ts.push(router);
                    ts.extend(w_gate_shared.iter());
                    ts.extend(w_up_shared.iter());
                    ts.extend(w_down_shared.iter());
                    ts.extend(shared_router.iter());
                }
            }
            crate::accel::register_layer_dense_ranges(i as u32, &ts);
        }
        if hybrid_layers.is_some() {
            tracing::info!(
                layers = crate::accel::dense_registry_len(),
                "dense layer-ahead readahead spans registered"
            );
        }

        Ok(Self {
            token_embd,
            blocks,
            moe_blocks,
            hybrid_layers,
            mla_blocks,
            output_norm,
            output,
            mtp_heads,
            nextn_head,
            hadamard,
        })
    }
}

/// Validate `prism.hadamard.*` against the model's actual tensor
/// table and produce the runtime state. Every failure is a hard
/// [`LlamaLoadError::Hadamard`] — see that variant's doc for why.
fn build_hadamard_state(
    gguf: &Gguf,
    cfg: &LlamaConfig,
) -> Result<Option<HadamardState>, LlamaLoadError> {
    let Some(h) = cfg.hadamard.as_ref() else {
        return Ok(None);
    };
    let err = |msg: String| LlamaLoadError::Hadamard(msg);
    // Forward-path coverage: the transform is wired into the hybrid
    // forwards (and the LM head / embedding they share). A folded
    // MoE hybrid (exps tensors in weight_names) or a non-hybrid
    // folded model would run un-hooked paths — refuse.
    if cfg.hybrid.is_none() {
        return Err(err(
            "hadamard metadata on a non-hybrid model — no wired forward path".into(),
        ));
    }
    if cfg.moe.is_some() {
        return Err(err(
            "hadamard metadata on a MoE hybrid — the expert FFN path has no \
             rotation hook wired (no such model is known to exist)"
                .into(),
        ));
    }
    let identity_signs = h.signs_by_width.is_empty();
    let mut signs_by_width = h.signs_by_width.clone();
    let mut folded = std::collections::HashSet::with_capacity(h.weight_names.len());
    for name in &h.weight_names {
        let info = gguf
            .tensor(name)
            .ok_or_else(|| err(format!("folded weight `{name}` not found in GGUF")))?;
        // GGUF dims are [ne0 = input, ne1 = output]; the rotation
        // runs along the input axis (`axis = input-last-dimension`).
        let width = info.dims.first().copied().unwrap_or(0) as usize;
        if width == 0 || width % h.block_size != 0 {
            return Err(err(format!(
                "folded weight `{name}` input width {width} is not a \
                 multiple of block_size {}",
                h.block_size
            )));
        }
        if identity_signs {
            signs_by_width
                .entry(width)
                .or_insert_with(|| std::sync::Arc::new(vec![1.0f32; width]));
        } else if !signs_by_width.contains_key(&width) {
            return Err(err(format!(
                "folded weight `{name}` input width {width} has no sign vector"
            )));
        }
        folded.insert(name.clone());
    }
    let embd_inverse = h
        .inverse_weight_names
        .iter()
        .any(|n| n == "token_embd.weight");
    if embd_inverse {
        let d = cfg.d_model;
        if d % h.block_size != 0 {
            return Err(err(format!(
                "token_embd inverse: d_model {d} not a multiple of block_size {}",
                h.block_size
            )));
        }
        if identity_signs {
            signs_by_width
                .entry(d)
                .or_insert_with(|| std::sync::Arc::new(vec![1.0f32; d]));
        } else if !signs_by_width.contains_key(&d) {
            return Err(err(format!(
                "token_embd inverse: no sign vector for width {d}"
            )));
        }
    }
    tracing::info!(
        block_size = h.block_size,
        folded = folded.len(),
        widths = ?signs_by_width.keys().collect::<Vec<_>>(),
        embd_inverse,
        gdn_v_grouped = h.gdn_v_grouped,
        "prism.hadamard validated — activation rotation active for this model"
    );
    Ok(Some(HadamardState {
        block_size: h.block_size,
        folded,
        signs_by_width,
        embd_inverse,
        gdn_v_grouped: h.gdn_v_grouped,
    }))
}

/// Per-layer dispatch for hybrid attention+SSM models. Layer `i`
/// is full-attention when `(i + 1) % full_attention_interval == 0`;
/// otherwise it's an SSM block with a gated-attention side-channel.
/// Returns one `HybridLayer` per `cfg.n_layers`.
fn load_hybrid_layers(
    gguf: &Gguf,
    cfg: &LlamaConfig,
    moe_cfg: Option<&crate::llama_config::MoeConfig>,
    hyb_cfg: &crate::llama_config::HybridConfig,
) -> Result<Vec<HybridLayer>, LlamaLoadError> {
    let d_model = cfg.d_model;
    let d_ff = cfg.d_ff;
    let interval = hyb_cfg.full_attention_interval as usize;
    let mut out = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let prefix = format!("blk.{i}");
        // Per-layer FFN. MoE hybrids (qwen35moe) carry routed-expert
        // tensors on every layer (SSM and full-attention alike);
        // dense hybrids (Bonsai 2 / Qwen3.8-family) carry a plain
        // SwiGLU triple. Which one we get is decided by the model's
        // expert metadata, not per layer.
        let ffn = if let Some(moe_cfg) = moe_cfg {
            let n_experts = moe_cfg.n_experts as usize;
            let w_gate_exps = load_weight(gguf, &format!("{prefix}.ffn_gate_exps.weight"))?;
            let w_up_exps = load_weight(gguf, &format!("{prefix}.ffn_up_exps.weight"))?;
            let w_down_exps = load_weight(gguf, &format!("{prefix}.ffn_down_exps.weight"))?;
            let gate_per_expert: Vec<Tensor> = (0..n_experts)
                .map(|e| crate::moe::expert_view(&w_gate_exps, e, d_ff, d_model))
                .collect();
            let up_per_expert: Vec<Tensor> = (0..n_experts)
                .map(|e| crate::moe::expert_view(&w_up_exps, e, d_ff, d_model))
                .collect();
            let down_per_expert: Vec<Tensor> = (0..n_experts)
                .map(|e| crate::moe::expert_view(&w_down_exps, e, d_model, d_ff))
                .collect();
            // Expert-cache registry (see the dense-MoE branch of
            // `from_gguf`). Dense hybrids never get here — the
            // expert cache stays empty and every expert-cache
            // consumer no-ops, exactly like dense Llama models.
            crate::accel::register_layer_experts(
                i as u32,
                &gate_per_expert,
                &up_per_expert,
                &down_per_expert,
            );
            let w_gate_shared =
                load_optional_weight(gguf, &format!("{prefix}.ffn_gate_shexp.weight"));
            // A1-warn: the shared-expert FFN width is DERIVED from the gate
            // weight's element count (`shared_expert_width`), which is what the
            // forward path sizes its matvecs to. If the model also declares
            // `expert_shared_feed_forward_length` and it disagrees with that
            // derivation, the metadata is wrong (or a convention we don't
            // handle) — warn so a mis-sized shared expert is diagnosable rather
            // than silently truncating. The derived width always wins (it can't
            // overflow the tensor); this only surfaces the discrepancy.
            if let (Some(w_g), Some(declared)) =
                (&w_gate_shared, hyb_cfg.shared_expert_feed_forward_length)
            {
                let derived = crate::moe::shared_expert_width(w_g, d_model);
                if derived != declared as usize {
                    tracing::warn!(
                        layer = i,
                        declared = declared,
                        derived = derived,
                        "hybrid MoE loader: shared_expert_feed_forward_length \
                         disagrees with the gate-weight-derived width; using the \
                         derived width"
                    );
                }
            }
            HybridFfn::Moe {
                router: load_weight(gguf, &format!("{prefix}.ffn_gate_inp.weight"))?,
                w_gate_exps,
                w_up_exps,
                w_down_exps,
                w_gate_shared,
                w_up_shared: load_optional_weight(gguf, &format!("{prefix}.ffn_up_shexp.weight")),
                w_down_shared: load_optional_weight(
                    gguf,
                    &format!("{prefix}.ffn_down_shexp.weight"),
                ),
                shared_router: load_optional_weight(
                    gguf,
                    &format!("{prefix}.ffn_gate_inp_shexp.weight"),
                ),
                gate_per_expert,
                up_per_expert,
                down_per_expert,
                // M1-tail: qwen35moe hybrids carry no routed selection bias
                // (`exp_probs_b` is DeepSeek-V3 only) ⇒ `None`, so routing is
                // byte-identical.
                exp_probs_b: None,
            }
        } else {
            HybridFfn::Dense {
                w_gate: load_weight(gguf, &format!("{prefix}.ffn_gate.weight"))?,
                w_up: load_weight(gguf, &format!("{prefix}.ffn_up.weight"))?,
                w_down: load_weight(gguf, &format!("{prefix}.ffn_down.weight"))?,
            }
        };

        // Decide layer kind by tensor presence, not just by the
        // modulo rule. The modulo `(i + 1) % interval == 0` matches
        // most layers (3, 7, …, 39 on the 41-layer test model with
        // interval=4) — but the **final** layer is always full-
        // attention regardless of modulo, because it carries the
        // NextN/MTP head and the LM-head readout needs to come out
        // of a transformer attention block, not an SSM block.
        // Detecting via `attn_q.weight` (only present on full-
        // attention layers) makes us robust to that exception plus
        // any future model that places full-attention layers at
        // non-modulo positions.
        let has_separate_q = gguf.tensor(&format!("{prefix}.attn_q.weight")).is_some();
        let modulo_says_full_attn = (i + 1) % interval == 0;
        let is_full_attn = has_separate_q || modulo_says_full_attn;
        // Sanity-warn when the two disagree (helps catch new model
        // conventions before they silently misroute).
        if has_separate_q != modulo_says_full_attn {
            tracing::debug!(
                layer = i,
                interval = interval,
                has_attn_q = has_separate_q,
                modulo_full_attn = modulo_says_full_attn,
                "hybrid loader: layer kind disagrees with modulo rule; trusting tensor presence"
            );
        }
        if is_full_attn {
            out.push(HybridLayer::FullAttention(HybridAttnBlockWeights {
                attn_norm: load_norm(gguf, &format!("{prefix}.attn_norm.weight"))?,
                w_q: load_weight(gguf, &format!("{prefix}.attn_q.weight"))?,
                w_k: load_weight(gguf, &format!("{prefix}.attn_k.weight"))?,
                w_v: load_weight(gguf, &format!("{prefix}.attn_v.weight"))?,
                w_o: load_weight(gguf, &format!("{prefix}.attn_output.weight"))?,
                q_norm: load_optional_norm(gguf, &format!("{prefix}.attn_q_norm.weight")),
                k_norm: load_optional_norm(gguf, &format!("{prefix}.attn_k_norm.weight")),
                post_attention_norm: load_norm(
                    gguf,
                    &format!("{prefix}.post_attention_norm.weight"),
                )?,
                ffn,
            }));
        } else {
            // Load + cache the conv1d as f32 once at load time so
            // the per-decode kernel can read it as a raw slice.
            // Phase 3.7c perf win: dropped 256M ops/decode/layer down
            // to a fixed 30 dequants at engine load.
            let ssm_conv1d = load_weight(gguf, &format!("{prefix}.ssm_conv1d.weight"))?;
            let conv1d_len = ssm_conv1d.shape.iter().product::<u64>() as usize;
            let mut ssm_conv1d_f32 = vec![0.0f32; conv1d_len];
            dequant_or_copy_to_f32(&ssm_conv1d, &mut ssm_conv1d_f32);
            out.push(HybridLayer::Ssm(SsmBlockWeights {
                attn_norm: load_norm(gguf, &format!("{prefix}.attn_norm.weight"))?,
                attn_qkv: load_weight(gguf, &format!("{prefix}.attn_qkv.weight"))?,
                attn_gate: load_weight(gguf, &format!("{prefix}.attn_gate.weight"))?,
                post_attention_norm: load_norm(
                    gguf,
                    &format!("{prefix}.post_attention_norm.weight"),
                )?,
                ssm_conv1d,
                ssm_conv1d_f32,
                ssm_dt_bias: load_norm(gguf, &format!("{prefix}.ssm_dt.bias"))?,
                ssm_a: load_norm(gguf, &format!("{prefix}.ssm_a"))?,
                ssm_alpha: load_weight(gguf, &format!("{prefix}.ssm_alpha.weight"))?,
                ssm_beta: load_weight(gguf, &format!("{prefix}.ssm_beta.weight"))?,
                ssm_norm: load_norm(gguf, &format!("{prefix}.ssm_norm.weight"))?,
                ssm_out: load_weight(gguf, &format!("{prefix}.ssm_out.weight"))?,
                ffn,
            }));
        }
    }
    Ok(out)
}

/// Load the per-layer weights for a Multi-head Latent Attention
/// (DeepSeek-V2/V3) model. The Q path is low-rank (`attn_q_a`/`_a_norm`/
/// `_q_b`) or direct (`attn_q`); the KV path is always
/// `attn_kv_a_mqa`/`_a_norm`/`_kv_b`. The FFN is decided PER LAYER by
/// tensor presence — DeepSeek stacks a few dense SwiGLU layers before the
/// MoE layers (`first_k_dense_replace`), so a model-wide dense-vs-MoE
/// decision (as the hybrid loader makes) would be wrong here.
fn load_mla_layers(
    gguf: &Gguf,
    cfg: &LlamaConfig,
    mla_cfg: &crate::llama_config::MlaConfig,
    moe_cfg: Option<&crate::llama_config::MoeConfig>,
) -> Result<Vec<MlaBlockWeights>, LlamaLoadError> {
    let d_model = cfg.d_model;
    let d_ff = cfg.d_ff;
    let n_heads = cfg.n_heads;
    let kv_lora_rank = mla_cfg.kv_lora_rank;
    let kv_b_out = n_heads * (mla_cfg.qk_nope_head_dim + mla_cfg.v_head_dim);
    let mut out = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let prefix = format!("blk.{i}");
        // FFN: dense or MoE, decided per LAYER by tensor presence.
        let is_moe_layer = moe_cfg.is_some()
            && gguf
                .tensor(&format!("{prefix}.ffn_gate_exps.weight"))
                .is_some();
        let ffn = if is_moe_layer {
            let moe_cfg = moe_cfg.unwrap();
            let n_experts = moe_cfg.n_experts as usize;
            let w_gate_exps = load_weight(gguf, &format!("{prefix}.ffn_gate_exps.weight"))?;
            let w_up_exps = load_weight(gguf, &format!("{prefix}.ffn_up_exps.weight"))?;
            let w_down_exps = load_weight(gguf, &format!("{prefix}.ffn_down_exps.weight"))?;
            let gate_per_expert: Vec<Tensor> = (0..n_experts)
                .map(|e| crate::moe::expert_view(&w_gate_exps, e, d_ff, d_model))
                .collect();
            let up_per_expert: Vec<Tensor> = (0..n_experts)
                .map(|e| crate::moe::expert_view(&w_up_exps, e, d_ff, d_model))
                .collect();
            let down_per_expert: Vec<Tensor> = (0..n_experts)
                .map(|e| crate::moe::expert_view(&w_down_exps, e, d_model, d_ff))
                .collect();
            crate::accel::register_layer_experts(
                i as u32,
                &gate_per_expert,
                &up_per_expert,
                &down_per_expert,
            );
            HybridFfn::Moe {
                router: load_weight(gguf, &format!("{prefix}.ffn_gate_inp.weight"))?,
                w_gate_exps,
                w_up_exps,
                w_down_exps,
                w_gate_shared: load_optional_weight(
                    gguf,
                    &format!("{prefix}.ffn_gate_shexp.weight"),
                ),
                w_up_shared: load_optional_weight(gguf, &format!("{prefix}.ffn_up_shexp.weight")),
                w_down_shared: load_optional_weight(
                    gguf,
                    &format!("{prefix}.ffn_down_shexp.weight"),
                ),
                // DeepSeek-V3 runs its shared expert ungated (shared_router None);
                // Qwen2-MoE's sigmoid shared gate isn't an MLA arch.
                shared_router: load_optional_weight(
                    gguf,
                    &format!("{prefix}.ffn_gate_inp_shexp.weight"),
                ),
                gate_per_expert,
                up_per_expert,
                down_per_expert,
                // M1-tail: DeepSeek-V3 routed-expert selection bias. Mirrors
                // the dense MoE loader's spellings; only kept when it is the
                // `[n_experts]` vector the router indexes (a same-named tensor
                // of another shape is rejected). Absent on DeepSeek-V2-Lite ⇒
                // `None` ⇒ the MLA MoE FFN routes exactly as before.
                exp_probs_b: load_optional_norm(gguf, &format!("{prefix}.exp_probs_b.bias"))
                    .or_else(|| load_optional_norm(gguf, &format!("{prefix}.exp_probs_b.weight")))
                    .or_else(|| load_optional_norm(gguf, &format!("{prefix}.ffn_gate_inp.bias")))
                    .filter(|v| v.len() == n_experts),
            }
        } else {
            HybridFfn::Dense {
                w_gate: load_weight(gguf, &format!("{prefix}.ffn_gate.weight"))?,
                w_up: load_weight(gguf, &format!("{prefix}.ffn_up.weight"))?,
                w_down: load_weight(gguf, &format!("{prefix}.ffn_down.weight"))?,
            }
        };

        // Q path: low-rank (q_a→norm→q_b) or direct (q). Keyed on both the
        // config's `q_lora_rank` AND tensor presence so a mislabeled GGUF
        // falls back cleanly.
        let has_q_lora = mla_cfg.q_lora_rank.is_some()
            && gguf.tensor(&format!("{prefix}.attn_q_a.weight")).is_some();
        let (w_q_a, q_a_norm, w_q_b, w_q) = if has_q_lora {
            (
                Some(load_weight(gguf, &format!("{prefix}.attn_q_a.weight"))?),
                Some(load_norm(gguf, &format!("{prefix}.attn_q_a_norm.weight"))?),
                Some(load_weight(gguf, &format!("{prefix}.attn_q_b.weight"))?),
                None,
            )
        } else {
            (
                None,
                None,
                None,
                Some(load_weight(gguf, &format!("{prefix}.attn_q.weight"))?),
            )
        };

        // KV path. Dequant `attn_kv_b` to f32 once for the absorbed-decode
        // weight fold (row-major `[out, in]`: flat `o*kv_lora_rank + k`).
        let w_kv_b = load_weight(gguf, &format!("{prefix}.attn_kv_b.weight"))?;
        let n_kv_b = w_kv_b.element_count() as usize;
        debug_assert_eq!(
            n_kv_b,
            kv_b_out * kv_lora_rank,
            "attn_kv_b element count {} != n_heads*(qk_nope+v)*kv_lora_rank = {}",
            n_kv_b,
            kv_b_out * kv_lora_rank
        );
        let mut w_kv_b_f32 = vec![0.0f32; n_kv_b];
        dequant_or_copy_to_f32(&w_kv_b, &mut w_kv_b_f32);

        out.push(MlaBlockWeights {
            attn_norm: load_norm(gguf, &format!("{prefix}.attn_norm.weight"))?,
            w_q_a,
            q_a_norm,
            w_q_b,
            w_q,
            w_kv_a_mqa: load_weight(gguf, &format!("{prefix}.attn_kv_a_mqa.weight"))?,
            kv_a_norm: load_norm(gguf, &format!("{prefix}.attn_kv_a_norm.weight"))?,
            w_kv_b,
            w_kv_b_f32,
            w_o: load_weight(gguf, &format!("{prefix}.attn_output.weight"))?,
            ffn_norm: load_norm(gguf, &format!("{prefix}.ffn_norm.weight"))?,
            ffn,
        });
    }
    Ok(out)
}

/// Choose the in-memory dtype for a weight tensor.
///
/// Q8_0 source tensors are kept as `Q8_0Raw` (raw block bytes) so the
/// matvec dispatcher can dequant + FMA in one pass — halves memory
/// bandwidth vs an F16 round-trip. All other quant formats (Q4_K, Q5_0,
/// Q5_K, Q6_K) currently dequant to F16 at load. F16 / F32 source tensors
/// are kept as-is.
/// Heuristic size threshold below which we dequant Q5_0 to F16 on load
/// (compute path wins because matrices fit in L3 cache and the AVX2 F16
/// matvec is faster than the bit-extraction-heavy Q5_0 direct kernel).
/// Above this size, we keep the raw Q5_0 bytes — matvec becomes memory-
/// bound and the ~65% byte reduction pays for the extra compute.
///
/// 32 MB is calibrated empirically against Qwen 0.5B on a desktop Intel
/// CPU (~16 MB L3). For larger models (7B+) all layer weights exceed this
/// threshold and benefit. Q8_0 always stays raw — its kernel is light
/// enough to be win-win regardless.
const L3_HINT_BYTES: u64 = 32 * 1024 * 1024;

/// Load a tensor by name if present; `None` when the tensor is
/// absent (the GGUF doesn't carry it). Used for shared-expert
/// tensors that only appear on DeepSeek-V3-family MoE models.
/// NextN/MTP composition for the `qwen35moe`-family hybrid models.
///
/// Given the post-block hidden state and the just-predicted next-token
/// id, run the NextN head's projection + LM head and write the
/// predicted-token-after-next logits.
///
/// Math (per the DeepSeek-V3 MTP paper, adapted to qwen35moe):
/// 1. `e_next = token_embd[next_token_id]`
/// 2. `e_n = rmsnorm(e_next, head.embed_norm)`
/// 3. `h_n = rmsnorm(hidden, head.hidden_norm)`
/// 4. `concat = [e_n; h_n]`  (length `2 * d_model`)
/// 5. `projected = head.eh_proj @ concat`  (length `d_model`)
/// 6. `shared = rmsnorm(projected, head.shared_head_norm)`
/// 7. `logits = lm_head @ shared`
///
/// Public so `tests/nextn_python_parity.rs` can validate it in
/// isolation against a numpy reference without standing up the full
/// hybrid forward.
#[allow(clippy::too_many_arguments)]
pub fn nextn_compose_logits_f32(
    hidden: &[f32],
    next_token_id: i32,
    token_embd: &Tensor,
    head: &NextNHead,
    lm_head: &Tensor,
    rms_eps: f32,
    logits_out: &mut [f32],
) {
    let d = hidden.len();
    let vocab = logits_out.len();

    let mut e_next = vec![0.0f32; d];
    k::embed_lookup_tensor(token_embd, &[next_token_id], &mut e_next, d);

    let mut e_norm_buf = vec![0.0f32; d];
    k::rmsnorm_f32_row(&e_next, &head.embed_norm, &mut e_norm_buf, rms_eps);
    let mut h_norm_buf = vec![0.0f32; d];
    k::rmsnorm_f32_row(hidden, &head.hidden_norm, &mut h_norm_buf, rms_eps);

    let mut concat = vec![0.0f32; 2 * d];
    concat[..d].copy_from_slice(&e_norm_buf);
    concat[d..].copy_from_slice(&h_norm_buf);

    let mut projected = vec![0.0f32; d];
    matvec_tensor_dispatch(&head.eh_proj, &concat, &mut projected, d, 2 * d);

    let mut shared_norm = vec![0.0f32; d];
    k::rmsnorm_f32_row(
        &projected,
        &head.shared_head_norm,
        &mut shared_norm,
        rms_eps,
    );

    matvec_tensor_dispatch(lm_head, &shared_norm, logits_out, vocab, d);
}

/// Partial-rotary RoPE applied per-head. The first `rope_dim`
/// elements of each `head_dim`-sized head get the standard Neox
/// rotation; the trailing `(head_dim - rope_dim)` elements pass
/// through unchanged. For `partial_rotary_factor < 1.0` models
/// (qwen35moe with factor 0.25 → rope_dim=64, head_dim=128) this
/// is the correct behavior; the full-rotation `rope_inplace_neox`
/// would corrupt the non-rope tail.
///
/// `buf` is laid out as `[n_heads, head_dim]` row-major.
fn apply_partial_rope_per_head(
    buf: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos: u32,
    theta: f32,
) {
    assert_eq!(buf.len(), n_heads * head_dim);
    assert!(rope_dim <= head_dim);
    assert_eq!(rope_dim % 2, 0, "rope_dim must be even");
    if rope_dim == head_dim {
        // No partial: standard kernel is identical and faster.
        k::rope_inplace_neox(buf, n_heads, head_dim, pos, theta);
        return;
    }
    // Per-head: rotate the first `rope_dim` elements as if they
    // were a `1 × rope_dim` head; leave the rest alone.
    let pair_count = rope_dim / 2;
    let pos_f = pos as f32;
    for h in 0..n_heads {
        let head = &mut buf[h * head_dim..h * head_dim + rope_dim];
        // Neox layout: first half is the "x" lanes, second half is "y".
        // Standard Neox rotation: pairs (i, i + pair_count) rotated by
        // angle = pos * theta^(-2i / rope_dim).
        for i in 0..pair_count {
            let freq = theta.powf(-(2.0 * i as f32) / rope_dim as f32);
            let angle = pos_f * freq;
            let (s, c) = angle.sin_cos();
            let x = head[i];
            let y = head[i + pair_count];
            head[i] = x * c - y * s;
            head[i + pair_count] = x * s + y * c;
        }
    }
}

/// Qwen3 per-head Q/K RMSNorm, applied in place to a single token's
/// Q and K projection rows *before* RoPE.
///
/// Qwen3 (and Qwen3-MoE) insert a per-head RMSNorm on the query and
/// key vectors after their linear projection and before the rotary
/// embedding — tensors `self_attn.q_norm.weight` / `k_norm.weight`,
/// each `[head_dim]`, shared across all heads of that projection.
/// This is the exact op the hybrid `HybridAttnBlockWeights` path
/// already performs inline; this free helper lets the dense + MoE
/// (`&dyn AttnBlock`) forward paths reuse the identical math.
///
/// **No-op for every non-Qwen3 arch**: when both `q_norm` and
/// `k_norm` are `None` (Qwen2 / Llama / Mistral / Mixtral / …) the
/// function returns immediately, so threading the call through the
/// shared forward paths costs those models only two `Option` checks.
///
/// `q_row` is `[n_heads * head_dim]`, `k_row` is `[n_kv_heads *
/// head_dim]` — the same per-head row layout RoPE consumes.
#[inline]
fn apply_qk_head_norm(
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    q_row: &mut [f32],
    k_row: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    if q_norm.is_none() && k_norm.is_none() {
        return;
    }
    // One scratch row of `head_dim` reused across every head (the
    // rmsnorm kernel writes to a separate output buffer, so we copy
    // it back over the source after each head).
    let mut tmp = vec![0.0f32; head_dim];
    if let Some(qn) = q_norm {
        for h in 0..n_heads {
            let s = &mut q_row[h * head_dim..(h + 1) * head_dim];
            k::rmsnorm_f32_row(s, qn, &mut tmp, eps);
            s.copy_from_slice(&tmp);
        }
    }
    if let Some(kn) = k_norm {
        for h in 0..n_kv_heads {
            let s = &mut k_row[h * head_dim..(h + 1) * head_dim];
            k::rmsnorm_f32_row(s, kn, &mut tmp, eps);
            s.copy_from_slice(&tmp);
        }
    }
}

/// Stable softplus used by the hybrid forward (mirrors the helper
/// in `kernels-cpu::delta_net::softplus_f32`; inlined here so this
/// module doesn't need a fresh kernel-cpu import).
fn delta_net_softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Materialize a tensor's contents as f32 in the provided buffer.
/// Dispatches to the per-dtype dequant kernel in
/// [`rustllama_gguf::dequant`] — O(n) work, single pass over the
/// tensor bytes.
///
/// Used by `forward_one_hybrid`'s load-time cache of the DeltaNet
/// conv weights (the kernel-cpu conv1d requires raw f32 weights,
/// unlike `matvec_tensor_dispatch` which handles every dtype
/// natively).
fn dequant_or_copy_to_f32(t: &Tensor, out: &mut [f32]) {
    use rustllama_gguf::dequant as deq;
    use rustllama_tensor::Dtype;
    let bytes = rustllama_tensor::as_bytes(t);
    match t.dtype {
        Dtype::F32 => out.copy_from_slice(rustllama_tensor::as_slice_f32(t)),
        Dtype::F16 => deq::dequant_f16(bytes, out),
        Dtype::Bf16Raw => deq::dequant_bf16(bytes, out),
        Dtype::Q4_0Raw => deq::dequant_q4_0(bytes, out),
        Dtype::Q4_1Raw => deq::dequant_q4_1(bytes, out),
        Dtype::Q5_0Raw => deq::dequant_q5_0(bytes, out),
        Dtype::Q5_1Raw => deq::dequant_q5_1(bytes, out),
        Dtype::Q8_0Raw => deq::dequant_q8_0(bytes, out),
        Dtype::Q2_KRaw => deq::dequant_q2_k(bytes, out),
        Dtype::Q3_KRaw => deq::dequant_q3_k(bytes, out),
        Dtype::Q4_KRaw => deq::dequant_q4_k(bytes, out),
        Dtype::Q5_KRaw => deq::dequant_q5_k(bytes, out),
        Dtype::Q6_KRaw => deq::dequant_q6_k(bytes, out),
        Dtype::Q8_KRaw => deq::dequant_q8_k(bytes, out),
        Dtype::IQ4_NLRaw => deq::dequant_iq4_nl(bytes, out),
        Dtype::IQ4_XSRaw => deq::dequant_iq4_xs(bytes, out),
        Dtype::IQ1_SRaw => deq::dequant_iq1_s(bytes, out),
        Dtype::IQ1_MRaw => deq::dequant_iq1_m(bytes, out),
        Dtype::PQ2_0Raw => deq::dequant_pq2_0(bytes, out),
        Dtype::PTQ1_0Raw => deq::dequant_ptq1_0(bytes, out),
        Dtype::IQ2_XXSRaw => deq::dequant_iq2_xxs(bytes, out),
        Dtype::IQ2_XSRaw => deq::dequant_iq2_xs(bytes, out),
        Dtype::IQ2_SRaw => deq::dequant_iq2_s(bytes, out),
        Dtype::IQ3_XXSRaw => deq::dequant_iq3_xxs(bytes, out),
        Dtype::IQ3_SRaw => deq::dequant_iq3_s(bytes, out),
        other => panic!(
            "dequant_or_copy_to_f32: dtype {:?} not yet handled (extend the match arm + add the dequant_* helper in rustllama-gguf)",
            other
        ),
    }
}

fn load_optional_weight(gguf: &Gguf, name: &str) -> Option<Tensor> {
    if gguf.tensor(name).is_some() {
        load_weight(gguf, name).ok()
    } else {
        None
    }
}

fn load_weight(gguf: &Gguf, name: &str) -> Result<Tensor, LlamaLoadError> {
    let info = gguf
        .tensor(name)
        .ok_or_else(|| LlamaLoadError::NotF32(name.into()))?;

    // Estimate the tensor's F16-equivalent size; if it would fit in L3 the
    // F16 dequant-on-load path is usually faster (no inline bit-extraction
    // each call). Above that, raw quantized storage wins because the
    // matvec is memory-bound and the smaller footprint reduces DRAM traffic.
    //
    // Low-memory hosts can flip the trade by setting
    // `RUSTLLAMA_KEEP_QUANT_RAW=1` (CLI's `serve` sets this from
    // `[inference].keep_quant_raw`). With that set, every quantized
    // tensor stays in its on-disk Raw form — typically saves 1-3 GB
    // on a 7B-24B model at a modest matvec slowdown on the smaller
    // attention projections.
    let n_elements = info.element_count();
    let f16_bytes = n_elements * 2;
    let keep_raw_global = std::env::var("RUSTLLAMA_KEEP_QUANT_RAW")
        .map(|v| !v.is_empty() && v != "0" && v.to_ascii_lowercase() != "false")
        .unwrap_or(false);
    let fits_in_l3 = !keep_raw_global && f16_bytes <= L3_HINT_BYTES;

    let target = match info.dtype {
        GgmlType::F32 => Dtype::F32,
        // BF16 stays native: same 2B/elem footprint as F16 with a
        // wider dynamic range. The matvec widens BF16 → F32 via a
        // single zero-fill left-shift in SIMD, so there's no compute
        // penalty for keeping the original bit pattern. Going
        // through F16 would lossy-clamp values outside `|x| > 65504`
        // or `|x| < 6.1e-5`.
        GgmlType::Bf16 => Dtype::Bf16Raw,
        // Q8_0 is small per-block compute (i8 → f32, single scale) — keep
        // raw regardless of size; the kernel is fast enough that the
        // memory savings always net out.
        GgmlType::Q8_0 => Dtype::Q8_0Raw,
        // PrismML ternary (Bonsai): ALWAYS raw, no size threshold and
        // no RUSTLLAMA_KEEP_QUANT_RAW opt-out — these formats cover a
        // whole 27B model in ~6 GB and dequanting to F16 would be a
        // 54 GB owned-heap copy. Zero-copy mmap borrow applies via
        // raw_passthrough_source.
        GgmlType::PQ2_0 => Dtype::PQ2_0Raw,
        GgmlType::PTQ1_0 => Dtype::PTQ1_0Raw,
        // Q5_0 / Q4_K have nontrivial bit-extraction overhead per block.
        // Only worth keeping raw when the tensor exceeds L3 (LM heads, big
        // FFN). Below the threshold the F16 path (dequant once at load,
        // then AVX2 F16 matvec) is faster.
        // Q4_0: legacy 4-bit format (32-weight blocks, single per-block
        // scale, no min). Cheap enough to dequant per-load below L3 but
        // worth the memory win above.
        GgmlType::Q4_0 if !fits_in_l3 => Dtype::Q4_0Raw,
        GgmlType::Q5_0 if !fits_in_l3 => Dtype::Q5_0Raw,
        // Q4_1 / Q5_1: legacy 4/5-bit asymmetric formats. Same block-
        // shape as Q4_0/Q5_0 with an extra f16 min/offset; matvec is a
        // one-fmadd extension. Same L3 threshold as the symmetric pair.
        GgmlType::Q4_1 if !fits_in_l3 => Dtype::Q4_1Raw,
        GgmlType::Q5_1 if !fits_in_l3 => Dtype::Q5_1Raw,
        // Q8_K: 9.125 bpw, usually shipped as an intermediate
        // activation dtype rather than weight storage; rare in
        // practice. Same threshold logic as the rest.
        GgmlType::Q8_K if !fits_in_l3 => Dtype::Q8_KRaw,
        // Q2_K (2.625 bpw): the smallest K-quant. Same threshold logic
        // as the rest of the K-family — F16-on-load wins below L3
        // because of the AVX2 F16 matvec; raw-bytes win above L3
        // because the matvec becomes memory-bound and the ~84%
        // memory reduction more than pays for the per-block decode.
        GgmlType::Q2_K if !fits_in_l3 => Dtype::Q2_KRaw,
        // Q3_K (3.4375 bpw): smaller than Q4_K, slower per-tensor dequant
        // because of the 12-byte packed-scale layout, but the memory win
        // above L3 still beats a F16-dequant-on-load.
        GgmlType::Q3_K if !fits_in_l3 => Dtype::Q3_KRaw,
        GgmlType::Q4_K if !fits_in_l3 => Dtype::Q4_KRaw,
        GgmlType::Q5_K if !fits_in_l3 => Dtype::Q5_KRaw,
        GgmlType::Q6_K if !fits_in_l3 => Dtype::Q6_KRaw,
        // IQ4_XS: codebook-indexed 4-bit + sub-block scales. The
        // matvec is scalar (no AVX path yet) so we lean a bit harder
        // on F16-from-load below the threshold. Above, keep raw for
        // the memory win.
        GgmlType::IQ4_XS if !fits_in_l3 => Dtype::IQ4_XSRaw,
        // IQ4_NL has the same codebook overhead as IQ4_XS but with a
        // simpler per-block scale; same L3-threshold heuristic applies.
        GgmlType::IQ4_NL if !fits_in_l3 => Dtype::IQ4_NLRaw,
        // IQ3_S uses a 9-bit codebook + per-element sign bits. Raw matvec
        // is scalar-only for now but still beats a full F16 dequant copy.
        GgmlType::IQ3_S if !fits_in_l3 => Dtype::IQ3_SRaw,
        // IQ3_XXS (3.0625 bpw): 256-entry × 4-i8 codebook + packed
        // scale+sign u32 per sub-block. Same L3 heuristic — raw
        // matvec is scalar-only for v1 (same SIMD-gather concern as
        // IQ3_S) but the ~5× memory win below F16 keeps it raw above
        // the cutoff.
        GgmlType::IQ3_XXS if !fits_in_l3 => Dtype::IQ3_XXSRaw,
        // IQ2_XXS / IQ2_XS sit even lower on the bpw curve (~2.0/2.3). The
        // codebook overhead is similar to IQ3_S; matvec is scalar-only
        // for v1 but the F16 dequant cost dominates above the L3 cutoff.
        GgmlType::IQ2_XXS if !fits_in_l3 => Dtype::IQ2_XXSRaw,
        GgmlType::IQ2_XS if !fits_in_l3 => Dtype::IQ2_XSRaw,
        // IQ2_S (2.5625 bpw): 1024-entry codebook with split qs/signs and
        // qh-bits for the top of the index. Same L3 heuristic as the rest.
        GgmlType::IQ2_S if !fits_in_l3 => Dtype::IQ2_SRaw,
        // IQ1_S / IQ1_M (1.5625 / 1.75 bpw): tiniest quants in
        // common circulation. Scalar matvec only for v1 (the 11-bit
        // codebook gather doesn't map cleanly onto SIMD gathers); the
        // huge memory win above the L3 cutoff still beats a full F16
        // dequant copy.
        GgmlType::IQ1_S if !fits_in_l3 => Dtype::IQ1_SRaw,
        GgmlType::IQ1_M if !fits_in_l3 => Dtype::IQ1_MRaw,
        // OCP Microscaling weights kept raw (packed matvec on
        // CPU/SYCL/CUDA). Without these arms a big MXFP* tensor falls to
        // the F16 fallback below and silently dequants to an owned F16
        // slab — no packed path ever runs. (FP8 with its per-tensor
        // metadata scale is handled by its own load path, not here.)
        GgmlType::Mxfp4 if !fits_in_l3 => Dtype::Mxfp4Raw,
        // Upstream ggml MXFP4 (gpt-oss, id 39): request Mxfp4Raw so the
        // load-time path repacks it to our layout (lossless) and the
        // packed matvec runs on it exactly like a native-MXFP4 tensor.
        GgmlType::Mxfp4Ggml if !fits_in_l3 => Dtype::Mxfp4Raw,
        GgmlType::Mxfp6 if !fits_in_l3 => Dtype::Mxfp6Raw,
        GgmlType::Mxfp8 if !fits_in_l3 => Dtype::Mxfp8Raw,
        other => {
            // Every raw-passthrough quant above accepts big tensors, so a
            // BIG tensor can only land here when its GGUF type has no raw
            // arm at all (F16 itself excepted — that's a native copy).
            // The silent consequence is a full F16 dequant copy on the
            // heap: a 2-bit 100 MiB expert tensor becomes an 800 MiB F16
            // slab, and a freshly downloaded quant "mysteriously"
            // explodes RAM. Make it loud once per load path.
            if matches!(other, GgmlType::Fp8) {
                // FP8 (E4M3, per-tensor scale) is intentionally
                // transcoded to F16 at load: its scale is out-of-band
                // (GGUF metadata), which doesn't fit the packed matvec's
                // in-band block model, so we dequant (applying the
                // metadata scale) to F16 and run the standard F16 path.
                // This costs 2x the FP8 file's bytes in RAM; MXFP8 gives
                // the same 8-bit precision at ~8 bpw if memory matters.
                tracing::info!(
                    tensor = name,
                    f16_mib = f16_bytes / (1024 * 1024),
                    "FP8 tensor transcoded to F16 at load (per-tensor scale \
                     applied); use MXFP8 for memory-efficient 8-bit weights"
                );
            } else if !fits_in_l3 && !matches!(other, GgmlType::F16) {
                tracing::warn!(
                    tensor = name,
                    ggml_type = ?other,
                    f16_mib = f16_bytes / (1024 * 1024),
                    "large tensor's GGUF type has no raw-quant arm — \
                     dequanting to an owned F16 copy at load ({} MiB heap). \
                     If several tensors log this, the model file uses a \
                     format rustllama can't keep raw yet and RAM use will \
                     be far above the file size.",
                    f16_bytes / (1024 * 1024)
                );
            }
            Dtype::F16
        }
    };
    // Stage 8 zero-copy: when enabled and the target is a raw passthrough
    // of the source dtype, borrow the bytes straight from the GGUF mmap
    // (clean + file-backed → never pagefiled, no owned-heap duplicate)
    // instead of copying. Transformed targets (F16) always copy.
    let t = if zerocopy_weights_enabled()
        && rustllama_tensor::raw_passthrough_source(target).is_some()
    {
        let bytes = gguf.tensor(name).map(|i| i.byte_size).unwrap_or(0);
        ZEROCOPY_BORROWED_TENSORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ZEROCOPY_BORROWED_BYTES.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        Tensor::from_gguf_borrowed(gguf, name, target, gguf.mmap_backing())?
    } else {
        Tensor::from_gguf(gguf, name, target)?
    };
    Ok(t)
}

/// Process-wide tally of tensors borrowed zero-copy from the GGUF mmap
/// (Stage 8). Surfaced via [`zerocopy_borrow_stats`] so the engine can
/// log a one-line confirmation after load that the zero-copy path
/// actually engaged (vs. silently copying).
static ZEROCOPY_BORROWED_TENSORS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static ZEROCOPY_BORROWED_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(tensors_borrowed, bytes_borrowed)` — cumulative count of weights
/// loaded zero-copy (mmap-borrowed) since process start. `(0, 0)` means
/// the zero-copy path never engaged (disabled, or no raw-passthrough
/// tensors). Read by the engine to log a post-load confirmation.
pub fn zerocopy_borrow_stats() -> (u64, u64) {
    use std::sync::atomic::Ordering;
    (
        ZEROCOPY_BORROWED_TENSORS.load(Ordering::Relaxed),
        ZEROCOPY_BORROWED_BYTES.load(Ordering::Relaxed),
    )
}

/// `RUSTLLAMA_ZEROCOPY_WEIGHTS=1` — borrow raw quant tensors from the
/// GGUF mmap rather than copying to owned heap. Read once. Pairs with
/// `RUSTLLAMA_KEEP_QUANT_RAW=1` (only raw-kept tensors are borrowable).
fn zerocopy_weights_enabled() -> bool {
    // T3: typed `LoadOptions`-backed accel knob (env-initialized for
    // back-compat, overridden by the engine at load). The engine also
    // auto-enables it in-process when the MoE expert cache / memory budget
    // needs file-backed, pinnable experts.
    crate::accel::zerocopy_weights()
}

fn load_norm(gguf: &Gguf, name: &str) -> Result<Vec<f32>, LlamaLoadError> {
    let t = Tensor::from_gguf(gguf, name, Dtype::F32)?;
    Ok(rustllama_tensor::as_slice_f32(&t).to_vec())
}

/// H3: concat F32 `w_q` / `w_k` / `w_v` into one
/// `[d_q + 2*d_kv, d_model]` fused tensor. Returns `None` when any
/// input isn't F32 (quant-format concat is a future variant — the
/// bit-packed layouts don't share the same per-row stride math) or the
/// contraction (input / d_model) dim doesn't match across the three.
///
/// A4: the gate compares `in_dim()` (the contraction dim, d_model) —
/// NOT the output width. On a GQA model the OUTPUT widths differ
/// (`n_heads*head_dim` for Q vs `n_kv*head_dim` for K/V), so the old
/// "columns match" gate — which read the GGUF `[in, out]` tensor's LAST
/// dim (the output) — rejected every GQA model and the fusion never
/// fired. The physical byte layout is row-major `[out, in]` regardless
/// of the shape-vector order, so concatenating the raw f32 slices stacks
/// the three weights on the OUTPUT axis; the result is tagged
/// [`rustllama_tensor::DimOrder::RowMajor`] with shape `[sum_out, d_model]`.
fn concat_qkv_f32(w_q: &Tensor, w_k: &Tensor, w_v: &Tensor) -> Option<Tensor> {
    if w_q.dtype != Dtype::F32 || w_k.dtype != Dtype::F32 || w_v.dtype != Dtype::F32 {
        return None;
    }
    if w_q.shape.len() != 2 || w_k.shape.len() != 2 || w_v.shape.len() != 2 {
        return None;
    }
    // The contraction (input) dim must match for the stacked matvec to
    // share one activation vector; GQA makes the output widths differ.
    let d_model = w_q.in_dim();
    if w_k.in_dim() != d_model || w_v.in_dim() != d_model {
        return None;
    }
    let total_out = w_q.out_dim() + w_k.out_dim() + w_v.out_dim();
    let q_slice = rustllama_tensor::as_slice_f32(w_q);
    let k_slice = rustllama_tensor::as_slice_f32(w_k);
    let v_slice = rustllama_tensor::as_slice_f32(w_v);
    let mut fused = Vec::with_capacity(total_out * d_model);
    fused.extend_from_slice(q_slice);
    fused.extend_from_slice(k_slice);
    fused.extend_from_slice(v_slice);
    // from_vec_f32 already tags RowMajor; shape is physical `[out, in]`.
    Some(Tensor::from_vec_f32(
        format!("{}_qkv_fused", w_q.name),
        vec![total_out as u64, d_model as u64],
        fused,
    ))
}

fn load_optional_norm(gguf: &Gguf, name: &str) -> Option<Vec<f32>> {
    if gguf.tensor(name).is_none() {
        return None;
    }
    let t = Tensor::from_gguf(gguf, name, Dtype::F32).ok()?;
    Some(rustllama_tensor::as_slice_f32(&t).to_vec())
}

/// Per-DeltaNet-layer recurrent state for `qwen35moe`-family hybrid
/// models. Carried in parallel with [`KvCache`] (which still
/// services the full-attention layers) so the load-bearing
/// attention-math hot paths stay free of DeltaNet-specific match
/// arms. The forward pass picks `KvCache` for full-attention layers
/// and `DeltaNetCache` for SSM layers via the `HybridLayer` enum.
///
/// **Per-layer state:**
/// - `conv_state`: `[(conv_kernel - 1) * 2 * ssm_inner]` carries the
///   prior `kernel-1` conv1d input rows (depthwise, kernel-typically-4).
/// - `recurrent_state`: `[n_v_heads * head_qk_dim * head_v_dim]` is
///   the per-V-head Delta-Rule recurrent state matrix.
///
/// **Layer indexing:** `layers.len() == cfg.n_layers`. Full-attention
/// layers carry empty (zero-length) `conv_state` / `recurrent_state`
/// — they're sentinels that the forward dispatch skips. This keeps
/// the per-layer index in sync with `LlamaWeights::hybrid_layers`
/// without introducing a separate sparse index.
#[derive(Debug, Clone)]
pub struct DeltaNetLayerState {
    pub conv_state: Vec<f32>,
    pub recurrent_state: Vec<f32>,
}

impl DeltaNetLayerState {
    /// Empty sentinel for full-attention layers in a hybrid model.
    pub fn empty() -> Self {
        Self {
            conv_state: Vec::new(),
            recurrent_state: Vec::new(),
        }
    }

    /// Zero-init storage for an SSM layer with the given dims.
    /// `qkv_dim` is the fused QKV projection's output width — the
    /// same channel count the depthwise conv runs over
    /// (`2 * n_qk_heads * head_qk_dim + ssm_inner` in general; on
    /// qwen35moe that happens to equal `2 * ssm_inner`, which the
    /// old hardcode relied on and Bonsai-2/Qwen3.8 breaks).
    pub fn new_for_ssm(
        qkv_dim: usize,
        n_v_heads: usize,
        head_qk_dim: usize,
        head_v_dim: usize,
        conv_kernel: usize,
    ) -> Self {
        Self {
            conv_state: vec![0.0; (conv_kernel - 1) * qkv_dim],
            recurrent_state: vec![0.0; n_v_heads * head_qk_dim * head_v_dim],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.conv_state.is_empty() && self.recurrent_state.is_empty()
    }

    pub fn reset(&mut self) {
        for v in self.conv_state.iter_mut() {
            *v = 0.0;
        }
        for v in self.recurrent_state.iter_mut() {
            *v = 0.0;
        }
    }
}

#[derive(Debug)]
pub struct DeltaNetCache {
    /// One entry per layer. SSM layers carry allocated buffers;
    /// full-attention layers carry [`DeltaNetLayerState::empty`].
    pub layers: Vec<DeltaNetLayerState>,
}

impl DeltaNetCache {
    /// Build a cache shaped for a hybrid model. Pass `hybrid_layers`
    /// from the loaded weights so the constructor can decide which
    /// layers are SSM (allocate) vs full-attention (sentinel). Returns
    /// `None` for non-hybrid models — the caller should not construct
    /// a DeltaNetCache in that case.
    pub fn new_for_hybrid(
        cfg: &LlamaConfig,
        hybrid: &[HybridLayer],
    ) -> Result<Self, LlamaLoadError> {
        // The weights say "hybrid" but the config block is missing: a
        // genuine inconsistency. Error instead of silently returning
        // `None` (which the caller would then `.expect()` on → panic).
        let hyb = cfg.hybrid.as_ref().ok_or_else(|| {
            LlamaLoadError::UnsupportedHybridGeometry(
                "hybrid weights present but the [hybrid] config block is missing".to_string(),
            )
        })?;
        if hybrid.len() != cfg.n_layers {
            return Err(LlamaLoadError::UnsupportedHybridGeometry(format!(
                "hybrid_layers has {} entries but the model declares {} layers",
                hybrid.len(),
                cfg.n_layers
            )));
        }
        let ssm_inner = hyb.ssm_inner_size as usize;
        let conv_kernel = hyb.ssm_conv_kernel as usize;
        if conv_kernel == 0 {
            return Err(LlamaLoadError::UnsupportedHybridGeometry(
                "ssm_conv_kernel is 0".to_string(),
            ));
        }
        // n_v_heads / head_qk_dim / head_v_dim aren't first-class
        // config fields (they're encoded into the tensor shapes).
        // For `qwen35moe`: n_v_heads=32, n_qk_heads=16, head_qk_dim=
        // d_model/n_qk_heads=128, head_v_dim=ssm_inner/n_v_heads=128.
        // We derive them here from the tensor shapes of the first
        // SSM layer (which all SSM layers share).
        let first_ssm = hybrid
            .iter()
            .find_map(|l| match l {
                HybridLayer::Ssm(s) => Some(s),
                _ => None,
            })
            .ok_or_else(|| {
                LlamaLoadError::UnsupportedHybridGeometry(
                    "model is marked hybrid but has no SSM/DeltaNet layers".to_string(),
                )
            })?;
        let n_v_heads = first_ssm.ssm_a.len();
        if n_v_heads == 0 {
            return Err(LlamaLoadError::UnsupportedHybridGeometry(
                "first SSM layer's `ssm_a` is empty (zero v-heads)".to_string(),
            ));
        }
        // The DeltaNet state buffers are sized `n_v_heads * head_v_dim`
        // with `head_v_dim = ssm_inner / n_v_heads`, so `ssm_inner` must
        // divide evenly by the v-head count. A community merge / fine-
        // tune that changes the head layout from the base `qwen35moe`
        // (ssm_inner=4096, n_v_heads=32) would otherwise truncate here
        // and index out of bounds in the DeltaNet forward → process
        // crash. Fail cleanly at load instead (Part 2 — deriving the QK
        // dims from the tensor shapes — is the real fix for such models).
        if ssm_inner % n_v_heads != 0 {
            return Err(LlamaLoadError::UnsupportedHybridGeometry(format!(
                "ssm_inner_size ({ssm_inner}) is not divisible by the v-head count \
                 ({n_v_heads}); this model's DeltaNet head layout isn't supported"
            )));
        }
        let head_v_dim = ssm_inner / n_v_heads;
        // QK heads share a head_dim with V heads on `qwen35moe`. We
        // recover n_qk_heads from `attn_qkv` shape: that tensor is
        // [qkv_dim, d_model] with qkv_dim = 2*ssm_inner. Split as
        // q (n_qk_heads * head_qk_dim) + k (n_qk_heads * head_qk_dim)
        // + v (n_v_heads * head_v_dim). We assume head_qk_dim ==
        // head_v_dim (true for `qwen35moe`); a future model that
        // diverges needs the dims derived from tensor shapes (Part 2).
        let head_qk_dim = head_v_dim; // qwen35moe convention
        let _ = head_qk_dim; // suppress unused if future divergence
        let layers: Vec<DeltaNetLayerState> = hybrid
            .iter()
            .map(|l| match l {
                HybridLayer::Ssm(s) => {
                    // Conv channel count straight from the tensor:
                    // `ssm_conv1d` is `[conv_kernel, qkv_dim]`, and
                    // qkv_dim = 2·(n_qk·head_qk) + ssm_inner. Reading
                    // the shape sidesteps the `2 * ssm_inner`
                    // hardcode that only held on qwen35moe.
                    let qkv_dim = s
                        .ssm_conv1d
                        .shape
                        .last()
                        .map(|&v| v as usize)
                        .unwrap_or(2 * ssm_inner);
                    DeltaNetLayerState::new_for_ssm(
                        qkv_dim,
                        n_v_heads,
                        head_qk_dim,
                        head_v_dim,
                        conv_kernel,
                    )
                }
                HybridLayer::FullAttention(_) => DeltaNetLayerState::empty(),
            })
            .collect();
        Ok(Self { layers })
    }

    pub fn reset(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.reset();
        }
    }

    /// Total bytes of recurrent state held by this cache. Feeds the
    /// engine's memory-budget planner (roadmap Phase 3): honest peak
    /// projection so cache budgets never overcommit RAM.
    pub fn approx_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| (l.conv_state.len() + l.recurrent_state.len()) * std::mem::size_of::<f32>())
            .sum()
    }

    /// Clone the full recurrent state (roadmap Phase 5: semantic
    /// anchor snapshots). Unlike KV, recurrent state has no
    /// per-position addressing — a snapshot is only valid at exactly
    /// the token position it was captured, which is why hybrid prefix
    /// reuse restores whole anchors and re-prefills the suffix.
    pub fn snapshot(&self) -> DeltaNetSnapshot {
        DeltaNetSnapshot {
            layers: self.layers.clone(),
        }
    }

    /// Restore a previously captured snapshot. Panics on a shape
    /// mismatch (snapshot from a different model) — callers gate on
    /// model identity before restoring.
    pub fn restore(&mut self, snap: &DeltaNetSnapshot) {
        assert_eq!(
            snap.layers.len(),
            self.layers.len(),
            "DeltaNetSnapshot layer count mismatch"
        );
        for (live, src) in self.layers.iter_mut().zip(snap.layers.iter()) {
            assert_eq!(live.conv_state.len(), src.conv_state.len());
            assert_eq!(live.recurrent_state.len(), src.recurrent_state.len());
            live.conv_state.copy_from_slice(&src.conv_state);
            live.recurrent_state.copy_from_slice(&src.recurrent_state);
        }
    }
}

/// Point-in-time copy of a [`DeltaNetCache`] — the recurrent half of
/// a hybrid model's sequence state (KV is the other half, captured
/// by [`KvSnapshot`]). Full-attention layers carry empty sentinels,
/// so the size is the SSM layers' state only.
#[derive(Debug, Clone)]
pub struct DeltaNetSnapshot {
    pub layers: Vec<DeltaNetLayerState>,
}

impl DeltaNetSnapshot {
    pub fn approx_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| (l.conv_state.len() + l.recurrent_state.len()) * std::mem::size_of::<f32>())
            .sum()
    }
}

/// Per-layer KV cache. Storage layout depends on [`KvDtype`]:
///   - `F32`: dense f32 arrays sized `[n_kv_heads, max_ctx, head_dim]`.
///   - `Q8_0`: `i8` rows + one `f32` scale per `(head, slot)` pair. Each
///     stored row is `head_dim` ints (one byte per value) with the row's
///     scale at the matching index in `*_scales`. Cuts storage to ~26%
///     of the F32 footprint (1 byte + 4 bytes/`head_dim` vs 4 bytes per
///     value) — at the cost of one inline dequant per attention dot
///     product.
/// G2: storage abstraction for KV-cache buffers. `Host` is the legacy
/// host-`Vec` backing; `Usm` is a SYCL-shared allocation that the GPU
/// flash-attn USM kernels can read directly without a host→USM memcpy.
/// `Deref`/`DerefMut` to `[T]` so every existing slice/copy_from_slice/
/// iter call site continues to work without modification.
///
/// The USM variant requires a SYCL stream that outlives the buffer.
/// `KvCache` owns the stream via `Box<SyclStream>` and field drop
/// order (layers first, stream last) ensures buffers are freed before
/// the stream goes away. The `'static` lifetime on the buffer is a
/// transmute over the actual `Box`'s lifetime — safe under that drop
/// order invariant.
pub enum KvBuf<T: Copy> {
    Host(Vec<T>),
    Usm(rustllama_kernels_sycl::SyclSharedBuffer<'static, T>),
}

// SAFETY: USM-shared memory on integrated GPU lives in host RAM and is
// safe to read/write from any host thread. The underlying `SyclStream`
// reference is internally thread-safe per the SYCL spec (queues are
// thread-safe by default). The `&'static SyclStream` is a transmuted
// borrow over `KvCache`'s `Box<SyclStream>`, which outlives the buffers
// via field drop order. `Vec<T>` is already Send/Sync when `T: Send/Sync`.
unsafe impl<T: Copy + Send> Send for KvBuf<T> {}
unsafe impl<T: Copy + Sync> Sync for KvBuf<T> {}

impl<T: Copy> std::fmt::Debug for KvBuf<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvBuf::Host(v) => write!(f, "KvBuf::Host(len={})", v.len()),
            KvBuf::Usm(b) => write!(f, "KvBuf::Usm(len={})", b.len()),
        }
    }
}

/// Clone always produces a Host-resident copy. USM buffers are
/// single-owner (no clone on `SyclSharedBuffer`) and snapshot/restore
/// always uses host-side storage; the live cache may be USM but its
/// snapshot lives in host memory.
impl<T: Copy> Clone for KvBuf<T> {
    fn clone(&self) -> Self {
        // Deref to &[T], then copy into a fresh Vec.
        KvBuf::Host((**self).to_vec())
    }
}

impl<T: Copy + Default> KvBuf<T> {
    /// Construct a host-resident buffer of `len` elements, zero-initialized.
    pub fn host_zeroed(len: usize) -> Self {
        KvBuf::Host(vec![T::default(); len])
    }
}

impl<T: Copy> KvBuf<T> {
    /// Wrap an existing `Vec<T>` as a host-resident KvBuf.
    pub fn from_vec(v: Vec<T>) -> Self {
        KvBuf::Host(v)
    }

    pub fn len(&self) -> usize {
        match self {
            KvBuf::Host(v) => v.len(),
            KvBuf::Usm(b) => b.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Raw USM pointer when USM-backed; `None` for host backing.
    /// Used by the flash-attn USM dispatch to skip the host→USM
    /// memcpy when KV is already USM-resident.
    pub fn usm_ptr(&self) -> Option<*const T> {
        match self {
            KvBuf::Usm(b) => Some(b.as_ptr()),
            KvBuf::Host(_) => None,
        }
    }
}

impl<T: Copy> std::ops::Deref for KvBuf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            KvBuf::Host(v) => v.as_slice(),
            KvBuf::Usm(b) => b.as_slice(),
        }
    }
}

impl<T: Copy> std::ops::DerefMut for KvBuf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            KvBuf::Host(v) => v.as_mut_slice(),
            KvBuf::Usm(b) => b.as_mut_slice(),
        }
    }
}

pub enum KvLayer {
    F32 {
        k: KvBuf<f32>,
        v: KvBuf<f32>,
    },
    Q8_0 {
        k_q: KvBuf<i8>,
        k_scales: KvBuf<f32>,
        v_q: KvBuf<i8>,
        v_scales: KvBuf<f32>,
    },
    /// TurboQuant variant. `bits ∈ {1, 2, 4, 8}` picks the per-element
    /// budget; `k_packed` / `v_packed` carry the bit-packed codes for
    /// all rows (length = `total_rows × bytes_per_block(head_dim, bits)`),
    /// and `*_scales` carry the per-row f32 scale (length = `total_rows`).
    /// Attention dequantizes whole K/V slabs to an f32 scratch buffer,
    /// then dispatches the existing F32 GQA kernel.
    TurboQuant {
        bits: u8,
        k_packed: KvBuf<u8>,
        k_scales: KvBuf<f32>,
        v_packed: KvBuf<u8>,
        v_scales: KvBuf<f32>,
    },
    /// NVFP4 (E2M1 codebook + per-block FP8 E4M3 scale). `head_dim` must
    /// be a multiple of 16. Each row's storage is a sequence of 9-byte
    /// blocks (`NVFP4_BLOCK_BYTES`); the per-block FP8 scale is embedded
    /// in the last byte of each block, so there's no separate scales
    /// vector here (unlike TurboQuant).
    Nvfp4 {
        k_packed: KvBuf<u8>,
        v_packed: KvBuf<u8>,
    },
    /// Q4_0: exact ggml `block_q4_0` 18-byte blocks (32 elems, embedded
    /// little-endian f16 scale). `head_dim` must be a multiple of 32.
    /// Byte-compatible with the Prism fork's 4-bit KV cache, so their
    /// calibration ecosystem's expectations about cache bytes hold here.
    /// Like NVFP4 the scale is embedded, so no separate scales vector.
    Q4_0 {
        k_q: KvBuf<u8>,
        v_q: KvBuf<u8>,
    },
    /// OCP Microscaling KV (E2M1/E3M2/E4M3 + embedded E8M0 scale per 32
    /// elems; 17/25/33 B/block). `head_dim % 32 == 0`. Scale is embedded
    /// per block, so no separate scales vector (like NVFP4).
    Mxfp4 {
        k_packed: KvBuf<u8>,
        v_packed: KvBuf<u8>,
    },
    Mxfp6 {
        k_packed: KvBuf<u8>,
        v_packed: KvBuf<u8>,
    },
    Mxfp8 {
        k_packed: KvBuf<u8>,
        v_packed: KvBuf<u8>,
    },
    /// Half-precision KV (M6): two bytes per element, no block scale.
    /// `bf16=false` stores IEEE f16 bit patterns, `bf16=true` stores
    /// bfloat16 bit patterns; either way `k`/`v` are `u16` slabs sized
    /// `[n_kv_heads, max_ctx, head_dim]` (one `u16` per value, same row
    /// stride math as the F32 slab). Attention dequantizes the live
    /// region to an f32 scratch and dispatches the existing F32 kernels,
    /// so output equals the F32 path over f16/bf16-rounded K/V.
    Half {
        bf16: bool,
        k: KvBuf<u16>,
        v: KvBuf<u16>,
    },
}

impl KvLayer {
    /// Bytes of buffer storage this layer holds (host or USM — USM on
    /// integrated GPUs is host RAM too). Feeds the memory-budget
    /// planner's peak projection; hybrid models' full-attention-only
    /// KV allocation is captured exactly rather than estimated from
    /// geometry.
    pub fn approx_bytes(&self) -> usize {
        fn b<T: Copy>(buf: &KvBuf<T>) -> usize {
            (**buf).len() * std::mem::size_of::<T>()
        }
        match self {
            KvLayer::F32 { k, v } => b(k) + b(v),
            KvLayer::Q8_0 {
                k_q,
                k_scales,
                v_q,
                v_scales,
            } => b(k_q) + b(k_scales) + b(v_q) + b(v_scales),
            KvLayer::TurboQuant {
                k_packed,
                k_scales,
                v_packed,
                v_scales,
                ..
            } => b(k_packed) + b(k_scales) + b(v_packed) + b(v_scales),
            KvLayer::Nvfp4 { k_packed, v_packed } => b(k_packed) + b(v_packed),
            KvLayer::Q4_0 { k_q, v_q } => b(k_q) + b(v_q),
            KvLayer::Mxfp4 { k_packed, v_packed }
            | KvLayer::Mxfp6 { k_packed, v_packed }
            | KvLayer::Mxfp8 { k_packed, v_packed } => b(k_packed) + b(v_packed),
            KvLayer::Half { k, v, .. } => b(k) + b(v),
        }
    }
}

impl std::fmt::Debug for KvLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvLayer::F32 { k, v } => f.debug_struct("F32").field("k", k).field("v", v).finish(),
            KvLayer::Q8_0 {
                k_q,
                k_scales,
                v_q,
                v_scales,
            } => f
                .debug_struct("Q8_0")
                .field("k_q", k_q)
                .field("k_scales", k_scales)
                .field("v_q", v_q)
                .field("v_scales", v_scales)
                .finish(),
            KvLayer::TurboQuant {
                bits,
                k_packed,
                k_scales,
                v_packed,
                v_scales,
            } => f
                .debug_struct("TurboQuant")
                .field("bits", bits)
                .field("k_packed", k_packed)
                .field("k_scales", k_scales)
                .field("v_packed", v_packed)
                .field("v_scales", v_scales)
                .finish(),
            KvLayer::Nvfp4 { k_packed, v_packed } => f
                .debug_struct("Nvfp4")
                .field("k_packed", k_packed)
                .field("v_packed", v_packed)
                .finish(),
            KvLayer::Q4_0 { k_q, v_q } => f
                .debug_struct("Q4_0")
                .field("k_q", k_q)
                .field("v_q", v_q)
                .finish(),
            KvLayer::Mxfp4 { k_packed, v_packed } => f
                .debug_struct("Mxfp4")
                .field("k_packed", k_packed)
                .field("v_packed", v_packed)
                .finish(),
            KvLayer::Mxfp6 { k_packed, v_packed } => f
                .debug_struct("Mxfp6")
                .field("k_packed", k_packed)
                .field("v_packed", v_packed)
                .finish(),
            KvLayer::Mxfp8 { k_packed, v_packed } => f
                .debug_struct("Mxfp8")
                .field("k_packed", k_packed)
                .field("v_packed", v_packed)
                .finish(),
            KvLayer::Half { bf16, k, v } => f
                .debug_struct("Half")
                .field("bf16", bf16)
                .field("k", k)
                .field("v", v)
                .finish(),
        }
    }
}

/// Dtype selector for the KV cache. `F32` keeps the historic dense
/// layout used through phase 2. `Q8_0` is v1.x's memory-saving option:
/// scalar K/V rows are quantized to 8-bit signed ints with a per-row
/// float scale, dequantized inline in the attention loop. Fits roughly
/// 4x more context in the same RAM, with measured greedy parity within
/// a few rounding-error bits on the synthetic Llama tests.
///
/// `Tq(bits)` (TurboQuant) applies a Walsh–Hadamard rotation to each
/// K/V row before per-element quantization; `bits ∈ {1, 2, 4, 8}`. The
/// rotation flattens the distribution so uniform bins are near-optimal
/// — pays off most at 4 bits and below where naive uniform quantization
/// loses too much accuracy.
///
/// `Nvfp4` is NVIDIA's E2M1 4-bit floating-point format with shared
/// FP8 E4M3 scale per 16-element block. On non-Blackwell hardware the
/// dequant runs in software at FP16-GEMM throughput — you get FP4's
/// *accuracy* (non-uniform spacing covers activation-like distributions
/// better than INT4) without the hardware throughput bump.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDtype {
    F32,
    /// IEEE-754 half (`f16`, 1-5-10). Two bytes per element, no block
    /// scale — the usual mid-range KV option between f32 and q8_0. K/V
    /// rows are stored as f16 bit patterns and dequantized to f32 on the
    /// attention read, so the compute is bit-for-bit the F32 kernel over
    /// f16-rounded inputs (M6). `head_dim`-agnostic (no block alignment).
    F16,
    /// Google bfloat16 (`bf16`, 1-8-7). Two bytes per element, no block
    /// scale. Same store-f16bits/dequant-to-f32 path as [`Self::F16`] but
    /// with bf16's wider exponent + 7-bit mantissa (M6).
    Bf16,
    Q8_0,
    Tq(u8),
    Nvfp4,
    /// ggml Q4_0 blocks — 4-bit uniform with embedded f16 scale per 32
    /// elements. The Prism-compatible 4-bit KV option; ~7× smaller than
    /// F32. Requires head_dim % 32 == 0.
    Q4_0,
    /// OCP Microscaling KV: E2M1 4-bit + E8M0 scale per 32 (17B/32).
    Mxfp4,
    /// OCP Microscaling KV: E3M2 6-bit + E8M0 scale per 32 (25B/32).
    Mxfp6,
    /// OCP Microscaling KV: E4M3 8-bit + E8M0 scale per 32 (33B/32).
    Mxfp8,
}

impl Default for KvDtype {
    fn default() -> Self {
        KvDtype::F32
    }
}

impl KvDtype {
    /// Parse a config-string into a `KvDtype`. Accepts `"f32"`,
    /// `"q8_0"`, `"q4_0"`, `"tq1"`/`"tq2"`/`"tq4"`/`"tq8"`, and `"nvfp4"`. Case
    /// insensitive. Returns `None` for unrecognized strings — callers
    /// should treat that as a configuration error (with a helpful
    /// list of what's supported).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f32" => Some(KvDtype::F32),
            "f16" | "fp16" | "float16" => Some(KvDtype::F16),
            "bf16" | "bfloat16" => Some(KvDtype::Bf16),
            "q8_0" => Some(KvDtype::Q8_0),
            "q4_0" => Some(KvDtype::Q4_0),
            "tq1" => Some(KvDtype::Tq(1)),
            "tq2" => Some(KvDtype::Tq(2)),
            "tq4" => Some(KvDtype::Tq(4)),
            "tq8" => Some(KvDtype::Tq(8)),
            "nvfp4" => Some(KvDtype::Nvfp4),
            "mxfp4" => Some(KvDtype::Mxfp4),
            "mxfp6" => Some(KvDtype::Mxfp6),
            "mxfp8" => Some(KvDtype::Mxfp8),
            _ => None,
        }
    }

    /// Approximate KV-cache memory cost in **bits per element**.
    /// Used by the autotuner's coherence-vs-memory ranking when
    /// picking among candidates that all pass the quality bar —
    /// smaller is better.
    ///
    /// The numbers fold in per-row scale-byte overhead so they
    /// reflect the actual on-cache footprint, not just the raw
    /// quantization width. Head_dim is assumed ~64–128 (typical
    /// for modern LLM heads); the per-row overhead amortizes to
    /// well under 1 bit/element so the ranking is stable across
    /// the supported head shapes.
    pub fn approx_bits_per_element(self) -> f32 {
        match self {
            // 4-byte float, no overhead.
            KvDtype::F32 => 32.0,
            // 2-byte half floats, no block scale.
            KvDtype::F16 | KvDtype::Bf16 => 16.0,
            // 1 byte/element + 1 f16 scale per 32 elements.
            // Overhead: 16 / 32 = 0.5 bit/element.
            KvDtype::Q8_0 => 8.5,
            // `bits` bits/element + 1 f32 scale per row. With
            // typical head_dim=64–128, scale overhead is
            // ~32/96 ≈ 0.3 bit/element.
            KvDtype::Tq(bits) => bits as f32 + 0.3,
            // 4 bits/element + 1 e4m3 (8-bit) scale per 16 elements.
            // Overhead: 8 / 16 = 0.5 bit/element.
            KvDtype::Nvfp4 => 4.5,
            // 4 bits/element + 1 f16 scale per 32 elements.
            // Overhead: 16 / 32 = 0.5 bit/element.
            KvDtype::Q4_0 => 4.5,
            // OCP MX: element bits + 1 E8M0 (8-bit) scale per 32 elems
            // (0.25 bit/elem overhead).
            KvDtype::Mxfp4 => 4.25,
            KvDtype::Mxfp6 => 6.25,
            KvDtype::Mxfp8 => 8.25,
        }
    }
}

/// M1: the Multi-head Latent Attention KV cache (DeepSeek-V2/V3). Unlike
/// the per-head K/V slabs the `KvLayer` variants hold, MLA stores ONLY
/// the compressed KV latent `[kv_lora_rank]` + the single decoupled-RoPE
/// key `[qk_rope_head_dim]` per token per layer — the memory win. Both
/// the naive and absorbed attention paths read from here (naive
/// materializes per-head K/V from the latent on the fly; absorbed scores
/// against the latent directly).
#[derive(Debug, Clone)]
pub struct MlaKvCache {
    /// Per layer: compressed-latent slab, row-major `[max_ctx, kv_lora_rank]`.
    pub latent: Vec<Vec<f32>>,
    /// Per layer: decoupled-RoPE key slab, row-major
    /// `[max_ctx, qk_rope_head_dim]`.
    pub rope_key: Vec<Vec<f32>>,
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub max_ctx: usize,
}

impl MlaKvCache {
    fn new(n_layers: usize, max_ctx: usize, kv_lora_rank: usize, qk_rope_head_dim: usize) -> Self {
        Self {
            latent: (0..n_layers)
                .map(|_| vec![0.0f32; max_ctx * kv_lora_rank])
                .collect(),
            rope_key: (0..n_layers)
                .map(|_| vec![0.0f32; max_ctx * qk_rope_head_dim])
                .collect(),
            kv_lora_rank,
            qk_rope_head_dim,
            max_ctx,
        }
    }

    /// Total bytes held across all layers (feeds the cache's
    /// `approx_bytes` so the memory planner accounts the MLA slabs).
    pub fn approx_bytes(&self) -> usize {
        let per = |v: &Vec<Vec<f32>>| v.iter().map(|s| s.len() * 4).sum::<usize>();
        per(&self.latent) + per(&self.rope_key)
    }
}

pub struct KvCache {
    /// **Field declaration order matters**: `layers` drops before
    /// `usm_stream`, so any `KvBuf::Usm` allocations are freed before
    /// the stream they reference goes away. The `'static` lifetime on
    /// the USM buffers is a transmute over `usm_stream`'s lifetime;
    /// this drop order is the safety guarantee.
    pub layers: Vec<KvLayer>,
    /// M1: `Some` on a Multi-head Latent Attention model (DeepSeek-V2/V3),
    /// where `layers` holds zero-length F32 placeholders and the real KV
    /// (the compressed latent + decoupled rope key) lives here. `None` on
    /// every other model.
    pub mla: Option<MlaKvCache>,
    pub seq_len: usize,
    pub max_ctx: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// F4: per-layer KV dtype (one entry per layer). The common case is
    /// all entries equal — then the cache is byte-for-byte the old
    /// single-dtype cache. A per-layer-GROUP spec (e.g. attention-sink
    /// layers at f32/q8_0, the rest at a cheaper dtype) stores distinct
    /// entries. The actual storage + attention dispatch keys off each
    /// layer's [`KvLayer`] variant, so per-layer dtype is honored
    /// automatically once the slabs are allocated this way; use
    /// [`Self::dtype`] for the representative (layer-0) dtype and
    /// [`Self::uniform_dtype`] to gate the uniform-only batched paths.
    pub layer_dtypes: Vec<KvDtype>,
    /// P2: per-layer physical KV capacity in POSITIONS (one entry per
    /// layer). `max_ctx` for ordinary (full-attention) layers — their
    /// slab is byte-for-byte the historic `max_ctx`-strided allocation.
    /// A sliding-window (SWA) **F32** layer whose window `w` satisfies
    /// `0 < w < max_ctx` instead gets a bounded capacity `w`: its K/V
    /// slab is allocated `[n_kv_heads, w, head_dim]` (actual memory
    /// reduction, not just masking) and the decode path addresses it as
    /// a RING — position `p` lives at slot `p % w` — gathering the live
    /// window into a compact scratch for attention. Non-F32 SWA layers
    /// keep `max_ctx` (their windowed kernels mask a full slab; the ring
    /// is F32-only for now). The per-head slab stride for layer `li` is
    /// therefore `layer_caps[li]`, NOT `max_ctx`, on a ring layer.
    pub layer_caps: Vec<usize>,
    /// K-cache mean-centering bias (fork `kv_bar` parity) — attached
    /// by the engine after load when a validated sidecar exists.
    /// Subtracted from K rows in the Q4_0 write arms only; exactly
    /// softmax-invariant, so no read side exists. `None` = uncentered.
    pub kv_bias: Option<std::sync::Arc<crate::kv_bias::KvBiasData>>,
    /// G2: optional SYCL stream owning the KV-layer USM allocations.
    /// `None` when the `kv:usm` verdict is OFF or SYCL is unavailable;
    /// the layers' `KvBuf`s are then all `Host`-backed.
    pub usm_stream: Option<Box<rustllama_kernels_sycl::SyclStream>>,
}

impl std::fmt::Debug for KvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvCache")
            .field("layers", &self.layers)
            .field("seq_len", &self.seq_len)
            .field("max_ctx", &self.max_ctx)
            .field("n_kv_heads", &self.n_kv_heads)
            .field("head_dim", &self.head_dim)
            .field("dtype", &self.dtype())
            .field("layer_dtypes", &self.layer_dtypes)
            .field("mla", &self.mla.is_some())
            .finish()
    }
}

// SAFETY: same rationale as KvBuf Send/Sync — SyclStream is internally
// thread-safe per the SYCL spec; USM-shared memory on iGPU lives in host
// RAM. The engine owns the KvCache and serializes forward calls via the
// per-request semaphore, so even the !Send raw pointers inside SyclStream
// are safe to ferry across the tokio runtime's thread pool.
unsafe impl Send for KvCache {}
unsafe impl Sync for KvCache {}

/// G2: USM-resident KV allocation helpers. When `stream` is `Some`,
/// attempts `SyclSharedBuffer::alloc` and zero-inits; on failure
/// (mock build, USM exhausted) falls back to host `Vec`. When
/// `stream` is `None`, always returns host backing.
fn alloc_kv_buf_f32(stream: Option<&rustllama_kernels_sycl::SyclStream>, len: usize) -> KvBuf<f32> {
    if let Some(s) = stream {
        match rustllama_kernels_sycl::SyclSharedBuffer::<f32>::alloc(s, len) {
            Ok(mut buf) => {
                for x in buf.as_mut_slice() {
                    *x = 0.0;
                }
                // SAFETY: caller (KvCache) owns the stream via Box; layers
                // field drops before usm_stream field (declared order).
                let buf: rustllama_kernels_sycl::SyclSharedBuffer<'static, f32> =
                    unsafe { std::mem::transmute(buf) };
                return KvBuf::Usm(buf);
            }
            Err(_) => {} // fall through to host
        }
    }
    KvBuf::host_zeroed(len)
}
fn alloc_kv_buf_i8(stream: Option<&rustllama_kernels_sycl::SyclStream>, len: usize) -> KvBuf<i8> {
    if let Some(s) = stream {
        match rustllama_kernels_sycl::SyclSharedBuffer::<i8>::alloc(s, len) {
            Ok(mut buf) => {
                for x in buf.as_mut_slice() {
                    *x = 0;
                }
                let buf: rustllama_kernels_sycl::SyclSharedBuffer<'static, i8> =
                    unsafe { std::mem::transmute(buf) };
                return KvBuf::Usm(buf);
            }
            Err(_) => {}
        }
    }
    KvBuf::host_zeroed(len)
}
fn alloc_kv_buf_u8(stream: Option<&rustllama_kernels_sycl::SyclStream>, len: usize) -> KvBuf<u8> {
    if let Some(s) = stream {
        match rustllama_kernels_sycl::SyclSharedBuffer::<u8>::alloc(s, len) {
            Ok(mut buf) => {
                for x in buf.as_mut_slice() {
                    *x = 0;
                }
                let buf: rustllama_kernels_sycl::SyclSharedBuffer<'static, u8> =
                    unsafe { std::mem::transmute(buf) };
                return KvBuf::Usm(buf);
            }
            Err(_) => {}
        }
    }
    KvBuf::host_zeroed(len)
}
fn alloc_kv_buf_u16(stream: Option<&rustllama_kernels_sycl::SyclStream>, len: usize) -> KvBuf<u16> {
    if let Some(s) = stream {
        match rustllama_kernels_sycl::SyclSharedBuffer::<u16>::alloc(s, len) {
            Ok(mut buf) => {
                for x in buf.as_mut_slice() {
                    *x = 0;
                }
                let buf: rustllama_kernels_sycl::SyclSharedBuffer<'static, u16> =
                    unsafe { std::mem::transmute(buf) };
                return KvBuf::Usm(buf);
            }
            Err(_) => {}
        }
    }
    KvBuf::host_zeroed(len)
}

// M6: half-precision (f16 / bf16) KV row conversions. Storage is one
// `u16` bit pattern per element; `bf16` selects bfloat16 vs IEEE f16.
// `encode` rounds an f32 row to half on write; `decode` is the exact
// widening back to f32 on the attention read.
#[inline]
fn half_encode_row(bf16: bool, src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len());
    if bf16 {
        for (d, &x) in dst.iter_mut().zip(src.iter()) {
            *d = half::bf16::from_f32(x).to_bits();
        }
    } else {
        for (d, &x) in dst.iter_mut().zip(src.iter()) {
            *d = half::f16::from_f32(x).to_bits();
        }
    }
}
#[inline]
fn half_decode_row(bf16: bool, src: &[u16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    if bf16 {
        for (d, &b) in dst.iter_mut().zip(src.iter()) {
            *d = half::bf16::from_bits(b).to_f32();
        }
    } else {
        for (d, &b) in dst.iter_mut().zip(src.iter()) {
            *d = half::f16::from_bits(b).to_f32();
        }
    }
}

/// Dequantize the live `[n_kv_heads, kv_len, head_dim]` region of a
/// half-precision KV slab (head-major, `max_ctx` stride) into COMPACT
/// f32 `k_out`/`v_out` slabs laid out `[n_kv_heads, kv_len, head_dim]`
/// (stride `kv_len`). The caller then runs the existing F32 attention
/// kernels passing `kv_len` as the `max_ctx` stride argument. Keeps the
/// scratch O(kv_len) rather than materializing the full `max_ctx` slab.
#[allow(clippy::too_many_arguments)]
fn half_dequant_region(
    bf16: bool,
    k: &[u16],
    v: &[u16],
    n_kv_heads: usize,
    max_ctx: usize,
    head_dim: usize,
    kv_len: usize,
    k_out: &mut [f32],
    v_out: &mut [f32],
) {
    for h in 0..n_kv_heads {
        let src_base = h * max_ctx * head_dim;
        let dst_base = h * kv_len * head_dim;
        let n = kv_len * head_dim;
        half_decode_row(
            bf16,
            &k[src_base..src_base + n],
            &mut k_out[dst_base..dst_base + n],
        );
        half_decode_row(
            bf16,
            &v[src_base..src_base + n],
            &mut v_out[dst_base..dst_base + n],
        );
    }
}

/// G2: try to construct a SYCL stream for KV-cache USM allocations.
/// Returns `Some(Box<stream>)` when the `kv:usm` verdict is ON and SYCL has
/// a device; `None` otherwise. Cached after first read so repeated
/// `KvCache::new_with_dtype` calls don't re-check.
///
/// P5b: the USM-resident KV path auto-enables from the on-device PERF verdict
/// `kv:usm` (banked by `tune --validate-kernels` only where USM memory beat a
/// host buffer), replacing the removed `RUSTLLAMA_USM_KV` env gate. Fail-closed:
/// on a unified-memory iGPU (USM IS host memory) this is a wash → no verdict →
/// host-resident KV, byte-identical to before.
fn try_open_usm_kv_stream() -> Option<Box<rustllama_kernels_sycl::SyclStream>> {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled =
        *ENABLED.get_or_init(|| crate::accel::kernel_verdict(rustllama_tuner::VERDICT_KV_USM));
    if !enabled {
        return None;
    }
    match rustllama_kernels_sycl::create_stream(0) {
        Ok(s) => Some(Box::new(s)),
        Err(_) => None,
    }
}

/// P2: the ring capacity (in positions) for layer `li`. An F32
/// sliding-window layer whose window `w` is strictly inside the context
/// (`0 < w < max_ctx`) gets a bounded slab of `w` positions, addressed as
/// a ring on write/read. Every other layer — full-attention layers, and
/// SWA layers on a non-F32 KV dtype (their windowed kernels mask a full
/// slab) — keeps `max_ctx`, so its allocation and indexing are byte-for-
/// byte unchanged. Single source of truth for both allocation sizing and
/// the decode-path ring arithmetic.
#[inline]
fn swa_ring_cap(cfg: &LlamaConfig, dtype: KvDtype, li: usize, max_ctx: usize) -> usize {
    if matches!(dtype, KvDtype::F32) {
        let w = cfg.swa_window(li);
        if w > 0 && w < max_ctx {
            return w;
        }
    }
    max_ctx
}

impl KvCache {
    pub fn new(cfg: &LlamaConfig, max_ctx: usize) -> Self {
        Self::new_with_dtype(cfg, max_ctx, KvDtype::F32)
    }

    /// P2: physical capacity (positions) of layer `li`'s KV slab — i.e. its
    /// per-head stride. Equals `max_ctx` for every ordinary layer; smaller
    /// (the window) for an F32 SWA ring layer.
    #[inline]
    pub fn layer_cap(&self, li: usize) -> usize {
        self.layer_caps.get(li).copied().unwrap_or(self.max_ctx)
    }

    /// P2: true iff layer `li` is a bounded SWA ring (cap below `max_ctx`).
    /// A ring layer's slab is addressed `pos % cap`, so the full-history
    /// fast paths (batched prefill, prefix snapshot/restore, tree-spec
    /// compaction, the device-resident decode blocks) must route around it.
    #[inline]
    pub fn is_ring_layer(&self, li: usize) -> bool {
        self.layer_cap(li) < self.max_ctx
    }

    /// P2: true iff ANY layer is a bounded SWA ring. The engine reads this
    /// to keep a ring model off the paths that assume a full `max_ctx`-
    /// strided contiguous history (see [`Self::is_ring_layer`]).
    #[inline]
    pub fn has_ring_layers(&self) -> bool {
        self.max_ctx > 0 && self.layer_caps.iter().any(|&c| c < self.max_ctx)
    }

    /// P1 CONTEXT SHIFT. Drop positions `[keep, keep + discard)` from the KV
    /// cache and slide the retained suffix down by `discard` so the sequence
    /// re-occupies contiguous positions `[0, seq_len - discard)`, re-RoPE-ing
    /// every moved K row by `-discard` (NeoX, via
    /// [`rustllama_kernels_cpu::rope_shift_inplace`]). After this the caller
    /// continues generating at `new_seq_len = seq_len - discard` exactly as if
    /// the retained tokens had been prefilled at the shifted positions — the
    /// state matches a fresh re-prefill of the retained sequence to within
    /// float rounding (the P1 anchor).
    ///
    /// Composition with the P2 per-layer ring:
    /// - **Full-attention (non-ring) F32 layers** get the classic left-shift +
    ///   re-RoPE of the retained suffix.
    /// - **SWA ring layers** hold only the recent `window`; those positions
    ///   are all in the moved suffix (validated below), so each ring row is
    ///   re-RoPE'd by `-discard` and re-slotted to its new `pos % cap` (the
    ///   same renumbering a re-prefill would produce). V rows move unrotated.
    ///
    /// Returns `false` WITHOUT mutating anything when it can't guarantee a
    /// correct result — a non-F32 (or placeholder) layer, out-of-range
    /// `keep`/`discard`, or a ring whose live window would straddle the kept
    /// head / dropped gap. The caller then keeps today's hard-stop behavior.
    /// `rope_theta` must be the model's base theta; context shift is only
    /// wired for un-scaled NeoX RoPE (the engine gates on that).
    #[must_use]
    pub fn context_shift(&mut self, keep: usize, discard: usize, rope_theta: f32) -> bool {
        let seq = self.seq_len;
        if discard == 0 || keep >= seq || keep + discard >= seq {
            return false;
        }
        let max_ctx = self.max_ctx;
        let hd = self.head_dim;
        let n_h = self.n_kv_heads;
        let caps = self.layer_caps.clone();
        // Pre-validate every layer is shiftable before mutating any of them.
        for (li, layer) in self.layers.iter().enumerate() {
            match layer {
                KvLayer::F32 { k, .. } if k.len() == n_h * caps[li] * hd => {}
                _ => return false, // non-F32, or a sparse/placeholder layer
            }
            let cap = caps[li];
            if cap < max_ctx {
                // Ring: the live window must lie entirely in the moved suffix
                // so the whole ring renumbers by a uniform `-discard`.
                let win_len = seq.min(cap);
                let first = seq - win_len;
                if first < keep + discard {
                    return false;
                }
            }
        }
        let shift = -(discard as i32);
        let new_seq = seq - discard;
        for (li, layer) in self.layers.iter_mut().enumerate() {
            let cap = caps[li];
            let KvLayer::F32 { k, v } = layer else {
                unreachable!("validated F32 above");
            };
            if cap >= max_ctx {
                // Full-attention layer: left-shift the retained suffix, re-RoPE
                // each moved K row. Ascending order is clobber-safe (dst < src),
                // and a moved+roped row's new index is below every later src, so
                // it is never re-read.
                for p in (keep + discard)..seq {
                    let np = p - discard;
                    for h in 0..n_h {
                        let src = (h * max_ctx + p) * hd;
                        let dst = (h * max_ctx + np) * hd;
                        k.copy_within(src..src + hd, dst);
                        v.copy_within(src..src + hd, dst);
                        k::rope_shift_inplace(&mut k[dst..dst + hd], 1, hd, shift, rope_theta);
                    }
                }
            } else {
                // Ring layer: gather the live window (re-RoPE'd) then write each
                // row back at its NEW slot `new_pos % cap`. Two passes via a temp
                // so an old slot isn't overwritten before it is read.
                let win_len = seq.min(cap);
                let first = seq - win_len;
                let mut tk = vec![0f32; n_h * win_len * hd];
                let mut tv = vec![0f32; n_h * win_len * hd];
                for j in 0..win_len {
                    let old_p = first + j;
                    for h in 0..n_h {
                        let src = (h * cap + (old_p % cap)) * hd;
                        let td = (h * win_len + j) * hd;
                        tk[td..td + hd].copy_from_slice(&k[src..src + hd]);
                        tv[td..td + hd].copy_from_slice(&v[src..src + hd]);
                        k::rope_shift_inplace(&mut tk[td..td + hd], 1, hd, shift, rope_theta);
                    }
                }
                for j in 0..win_len {
                    let new_p = (first + j) - discard;
                    for h in 0..n_h {
                        let td = (h * win_len + j) * hd;
                        let dst = (h * cap + (new_p % cap)) * hd;
                        k[dst..dst + hd].copy_from_slice(&tk[td..td + hd]);
                        v[dst..dst + hd].copy_from_slice(&tv[td..td + hd]);
                    }
                }
            }
        }
        self.seq_len = new_seq;
        // The rotated/moved rows invalidate any thread-local USM / device KV
        // mirror: force a re-upload so the GPU can't serve stale rows.
        crate::accel::usm_attn_kv_epoch_bump();
        true
    }

    /// F4: the representative KV dtype — the dtype of layer 0 (which, in
    /// the common all-same case, is every layer's dtype). Used for
    /// logging, the `kv_dtype()` backend report, and the F32-only gates
    /// that only need "is this cache uniformly F32?" (they pair this with
    /// [`Self::uniform_dtype`]). Empty caches report F32.
    pub fn dtype(&self) -> KvDtype {
        self.layer_dtypes.first().copied().unwrap_or(KvDtype::F32)
    }

    /// F4: `Some(d)` when every layer shares dtype `d`; `None` when the
    /// cache mixes dtypes per layer group. The uniform-only fast paths
    /// (the batched prefill entry points + the speculation batched path)
    /// gate on `Some` — a mixed cache falls through to the serial
    /// per-layer forward, which dispatches each layer on its own
    /// [`KvLayer`] variant and so is correct for any mix.
    pub fn uniform_dtype(&self) -> Option<KvDtype> {
        let first = *self.layer_dtypes.first()?;
        if self.layer_dtypes.iter().all(|&d| d == first) {
            Some(first)
        } else {
            None
        }
    }

    /// True iff every layer is F32 (the condition the F32-only snapshot /
    /// persistence paths actually want — not just "layer 0 is F32").
    pub fn is_uniform_f32(&self) -> bool {
        self.uniform_dtype() == Some(KvDtype::F32)
    }

    /// Compact scattered token K/V slots into contiguous positions starting at
    /// `base`, for speculative TREE verification (Feature 4). After
    /// [`Self::forward_tree_verify_f32`]-style verification writes every tree
    /// node's K/V to slot `base + node_index`, the accepted root→leaf path's
    /// nodes sit at scattered slots `base + src_slots[j]`; this moves each to its
    /// committed position `base + j` across every F32 layer. The caller then
    /// sets `seq_len = base + src_slots.len()` and keeps decoding as though the
    /// accepted path had been an ordinary contiguous prefill — so tree spec pays
    /// ONE forward per round (no linear-replay of the accepted path).
    ///
    /// F32-only (the tree-spec KV path). A depth-`j` node's flat index is always
    /// `>= j`, so `src_slots[j] >= j` and the ascending in-place copy never
    /// clobbers a not-yet-moved source; `src == j` slots are skipped.
    pub fn compact_tokens(&mut self, base: usize, src_slots: &[usize]) {
        // P2: tree-spec compaction assumes `max_ctx`-strided slabs. The engine
        // disables speculation for SWA ring models (see `swa_ring_decode`), so
        // this is never reached with a ring layer; assert it in debug builds.
        debug_assert!(
            !self.has_ring_layers(),
            "compact_tokens on a SWA ring cache (tree-spec must be disabled for ring models)"
        );
        let max_ctx = self.max_ctx;
        let head_dim = self.head_dim;
        let n_kv_heads = self.n_kv_heads;
        for layer in self.layers.iter_mut() {
            let KvLayer::F32 { k, v } = layer else {
                continue;
            };
            for (j, &src) in src_slots.iter().enumerate() {
                if src == j {
                    continue;
                }
                debug_assert!(src >= j, "compact source slot {src} must be >= dest {j}");
                for h in 0..n_kv_heads {
                    let src_off = (h * max_ctx + base + src) * head_dim;
                    let dst_off = (h * max_ctx + base + j) * head_dim;
                    k.copy_within(src_off..src_off + head_dim, dst_off);
                    v.copy_within(src_off..src_off + head_dim, dst_off);
                }
            }
        }
    }

    /// Total bytes of K/V buffer storage across all layers. Exact (it
    /// walks the allocated buffers). With sparse construction
    /// ([`Self::new_with_dtype_sparse`]) hybrid models report only
    /// their full-attention layers' slabs; a densely-constructed
    /// cache reports every slab, used or not. Feeds the
    /// memory-budget planner's peak projection.
    pub fn approx_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.approx_bytes()).sum::<usize>()
            + self.mla.as_ref().map(|m| m.approx_bytes()).unwrap_or(0)
    }

    pub fn new_with_dtype(cfg: &LlamaConfig, max_ctx: usize, dtype: KvDtype) -> Self {
        Self::new_with_dtype_impl(cfg, max_ctx, dtype, None)
    }

    /// Sparse construction (roadmap: memory reclaim): allocate real
    /// K/V slabs only for layers marked `keep`; the rest get
    /// zero-length host placeholders. Hybrid models pass the
    /// full-attention mask — on qwen35moe, 31 of 41 layers (30
    /// DeltaNet + the excluded MTP block) never write KV, and their
    /// full-size slabs were ~992 MiB of dead allocation that
    /// `reset()` also memset on every cache miss. `snapshot_prefix*`
    /// and `restore_prefix` treat zero-length layers as placeholders.
    pub fn new_with_dtype_sparse(
        cfg: &LlamaConfig,
        max_ctx: usize,
        dtype: KvDtype,
        keep: &[bool],
    ) -> Self {
        assert_eq!(
            keep.len(),
            cfg.n_layers,
            "keep mask must have one entry per layer"
        );
        Self::new_with_dtype_impl(cfg, max_ctx, dtype, Some(keep))
    }

    /// F4: construct a cache with an explicit per-layer dtype. `dtypes`
    /// must have one entry per layer. The common all-same case is
    /// byte-for-byte identical to [`Self::new_with_dtype`] with that
    /// dtype; distinct entries give a per-layer-group cache (e.g.
    /// attention-sink layers at f32/q8_0, the rest cheaper). Allocation
    /// and the attention dispatch key off each layer's dtype.
    pub fn new_with_layer_dtypes(cfg: &LlamaConfig, max_ctx: usize, dtypes: &[KvDtype]) -> Self {
        Self::new_with_layer_dtypes_impl(cfg, max_ctx, dtypes.to_vec(), None, true)
    }

    /// Sparse [`Self::new_with_layer_dtypes`] (hybrid keep-mask aware).
    pub fn new_with_layer_dtypes_sparse(
        cfg: &LlamaConfig,
        max_ctx: usize,
        dtypes: &[KvDtype],
        keep: &[bool],
    ) -> Self {
        assert_eq!(
            keep.len(),
            cfg.n_layers,
            "keep mask must have one entry per layer"
        );
        Self::new_with_layer_dtypes_impl(cfg, max_ctx, dtypes.to_vec(), Some(keep), true)
    }

    /// P2: build a cache for a sliding-window model WITHOUT the bounded ring
    /// — every layer gets a full `max_ctx` slab and the forward applies SWA
    /// by MASKING (the pre-P2 behavior). This is the reference the ring is
    /// graded against (ring ≡ masked-unbounded), and the shape a config
    /// could request to opt out of the ring's memory saving.
    pub fn new_with_dtype_no_ring(cfg: &LlamaConfig, max_ctx: usize, dtype: KvDtype) -> Self {
        Self::new_with_layer_dtypes_impl(cfg, max_ctx, vec![dtype; cfg.n_layers], None, false)
    }

    fn new_with_dtype_impl(
        cfg: &LlamaConfig,
        max_ctx: usize,
        dtype: KvDtype,
        keep: Option<&[bool]>,
    ) -> Self {
        // F4: a uniform cache is just a per-layer cache with every entry
        // equal — same allocation path, so an all-same cache is
        // byte-identical to the historic single-dtype construction.
        Self::new_with_layer_dtypes_impl(cfg, max_ctx, vec![dtype; cfg.n_layers], keep, true)
    }

    fn new_with_layer_dtypes_impl(
        cfg: &LlamaConfig,
        max_ctx: usize,
        layer_dtypes: Vec<KvDtype>,
        keep: Option<&[bool]>,
        ring_enabled: bool,
    ) -> Self {
        assert_eq!(
            layer_dtypes.len(),
            cfg.n_layers,
            "layer_dtypes must have one entry per layer"
        );
        // M1: MLA models store ONLY the compressed KV latent + the decoupled
        // rope key per token (the memory win), not per-head K/V. Build the
        // dedicated MLA cache; `layers` gets zero-length F32 placeholders so the
        // layer-walking bookkeeping (approx_bytes, reset, prefix snapshot) sees
        // uniform empty F32 layers and no-ops on them. `forward_one_mla` reads
        // `self.mla` exclusively. Ignores the requested `kv_dtype` (the latent
        // is always f32) and never uses USM.
        if let Some(mla) = cfg.mla.as_ref() {
            let layers = (0..cfg.n_layers)
                .map(|_| KvLayer::F32 {
                    k: KvBuf::from_vec(Vec::new()),
                    v: KvBuf::from_vec(Vec::new()),
                })
                .collect();
            return Self {
                layers,
                mla: Some(MlaKvCache::new(
                    cfg.n_layers,
                    max_ctx,
                    mla.kv_lora_rank,
                    mla.qk_rope_head_dim,
                )),
                seq_len: 0,
                max_ctx,
                n_kv_heads: cfg.n_kv_heads,
                head_dim: cfg.head_dim,
                layer_dtypes: vec![KvDtype::F32; cfg.n_layers],
                layer_caps: vec![max_ctx; cfg.n_layers],
                kv_bias: None,
                usm_stream: None,
            };
        }
        let total_rows = cfg.n_kv_heads * max_ctx;
        let row_floats = total_rows * cfg.head_dim;
        // P2: per-layer ring capacity. An F32 sliding-window layer with a
        // window strictly inside the context gets a bounded slab sized to
        // the window (ring-addressed on write/read); everything else keeps
        // the full `max_ctx` slab (byte-identical to before — the common
        // `cap == max_ctx` case allocates exactly as it always did).
        let layer_caps: Vec<usize> = (0..cfg.n_layers)
            .map(|li| {
                if ring_enabled {
                    swa_ring_cap(cfg, layer_dtypes[li], li, max_ctx)
                } else {
                    max_ctx
                }
            })
            .collect();
        let usm_stream = try_open_usm_kv_stream();
        let stream_ref = usm_stream.as_deref();
        let layers = (0..cfg.n_layers)
            .map(|li| {
                let dtype = layer_dtypes[li];
                // F32 real-slab size honors this layer's ring cap; for a
                // non-ring layer `cap == max_ctx` so this equals `row_floats`.
                let cap_floats = layer_caps[li] * cfg.n_kv_heads * cfg.head_dim;
                if let Some(keep) = keep {
                    if !keep[li] {
                        // Zero-length host placeholder — this layer's
                        // KV is never written (hybrid SSM / excluded
                        // MTP layers). Host-backed even in USM mode:
                        // a zero-byte USM alloc is driver-dependent,
                        // and nothing ever reads these.
                        return match dtype {
                            KvDtype::F32 => KvLayer::F32 {
                                k: KvBuf::from_vec(Vec::new()),
                                v: KvBuf::from_vec(Vec::new()),
                            },
                            KvDtype::Q8_0 => KvLayer::Q8_0 {
                                k_q: KvBuf::from_vec(Vec::new()),
                                k_scales: KvBuf::from_vec(Vec::new()),
                                v_q: KvBuf::from_vec(Vec::new()),
                                v_scales: KvBuf::from_vec(Vec::new()),
                            },
                            KvDtype::Tq(bits) => KvLayer::TurboQuant {
                                bits,
                                k_packed: KvBuf::from_vec(Vec::new()),
                                k_scales: KvBuf::from_vec(Vec::new()),
                                v_packed: KvBuf::from_vec(Vec::new()),
                                v_scales: KvBuf::from_vec(Vec::new()),
                            },
                            KvDtype::Nvfp4 => KvLayer::Nvfp4 {
                                k_packed: KvBuf::from_vec(Vec::new()),
                                v_packed: KvBuf::from_vec(Vec::new()),
                            },
                            KvDtype::Q4_0 => KvLayer::Q4_0 {
                                k_q: KvBuf::from_vec(Vec::new()),
                                v_q: KvBuf::from_vec(Vec::new()),
                            },
                            KvDtype::Mxfp4 => KvLayer::Mxfp4 {
                                k_packed: KvBuf::from_vec(Vec::new()),
                                v_packed: KvBuf::from_vec(Vec::new()),
                            },
                            KvDtype::Mxfp6 => KvLayer::Mxfp6 {
                                k_packed: KvBuf::from_vec(Vec::new()),
                                v_packed: KvBuf::from_vec(Vec::new()),
                            },
                            KvDtype::Mxfp8 => KvLayer::Mxfp8 {
                                k_packed: KvBuf::from_vec(Vec::new()),
                                v_packed: KvBuf::from_vec(Vec::new()),
                            },
                            KvDtype::F16 => KvLayer::Half {
                                bf16: false,
                                k: KvBuf::from_vec(Vec::new()),
                                v: KvBuf::from_vec(Vec::new()),
                            },
                            KvDtype::Bf16 => KvLayer::Half {
                                bf16: true,
                                k: KvBuf::from_vec(Vec::new()),
                                v: KvBuf::from_vec(Vec::new()),
                            },
                        };
                    }
                }
                match dtype {
                    KvDtype::F32 => KvLayer::F32 {
                        k: alloc_kv_buf_f32(stream_ref, cap_floats),
                        v: alloc_kv_buf_f32(stream_ref, cap_floats),
                    },
                    KvDtype::F16 => KvLayer::Half {
                        bf16: false,
                        k: alloc_kv_buf_u16(stream_ref, row_floats),
                        v: alloc_kv_buf_u16(stream_ref, row_floats),
                    },
                    KvDtype::Bf16 => KvLayer::Half {
                        bf16: true,
                        k: alloc_kv_buf_u16(stream_ref, row_floats),
                        v: alloc_kv_buf_u16(stream_ref, row_floats),
                    },
                    KvDtype::Q8_0 => KvLayer::Q8_0 {
                        k_q: alloc_kv_buf_i8(stream_ref, row_floats),
                        k_scales: alloc_kv_buf_f32(stream_ref, total_rows),
                        v_q: alloc_kv_buf_i8(stream_ref, row_floats),
                        v_scales: alloc_kv_buf_f32(stream_ref, total_rows),
                    },
                    KvDtype::Tq(bits) => {
                        let bytes_per_row =
                            rustllama_kernels_cpu::turboquant::bytes_per_block(cfg.head_dim, bits);
                        let packed_len = total_rows * bytes_per_row;
                        KvLayer::TurboQuant {
                            bits,
                            k_packed: alloc_kv_buf_u8(stream_ref, packed_len),
                            k_scales: alloc_kv_buf_f32(stream_ref, total_rows),
                            v_packed: alloc_kv_buf_u8(stream_ref, packed_len),
                            v_scales: alloc_kv_buf_f32(stream_ref, total_rows),
                        }
                    }
                    KvDtype::Nvfp4 => {
                        // NVFP4 packs 16 elements per 9-byte block. Asserting
                        // here keeps the error message close to the source
                        // of the misconfiguration rather than surfacing as
                        // a confusing slice-mismatch panic later.
                        assert!(
                            cfg.head_dim % rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS == 0,
                            "nvfp4 KV requires head_dim divisible by {}, got {}",
                            rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS,
                            cfg.head_dim,
                        );
                        let blocks_per_row =
                            cfg.head_dim / rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS;
                        let bytes_per_row =
                            blocks_per_row * rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES;
                        let packed_len = total_rows * bytes_per_row;
                        KvLayer::Nvfp4 {
                            k_packed: alloc_kv_buf_u8(stream_ref, packed_len),
                            v_packed: alloc_kv_buf_u8(stream_ref, packed_len),
                        }
                    }
                    KvDtype::Q4_0 => {
                        // ggml Q4_0 packs 32 elements per 18-byte block.
                        // Same rationale as the NVFP4 assert: fail at
                        // construction with a clear message, not later as
                        // a slice-length panic in the attention loop.
                        assert!(
                            cfg.head_dim % rustllama_kernels_cpu::q4_0_kv::Q4_0_BLOCK_ELEMS == 0,
                            "q4_0 KV requires head_dim divisible by {}, got {}",
                            rustllama_kernels_cpu::q4_0_kv::Q4_0_BLOCK_ELEMS,
                            cfg.head_dim,
                        );
                        let blocks_per_row =
                            cfg.head_dim / rustllama_kernels_cpu::q4_0_kv::Q4_0_BLOCK_ELEMS;
                        let bytes_per_row =
                            blocks_per_row * rustllama_kernels_cpu::q4_0_kv::Q4_0_BLOCK_BYTES;
                        let packed_len = total_rows * bytes_per_row;
                        KvLayer::Q4_0 {
                            k_q: alloc_kv_buf_u8(stream_ref, packed_len),
                            v_q: alloc_kv_buf_u8(stream_ref, packed_len),
                        }
                    }
                    KvDtype::Mxfp4 | KvDtype::Mxfp6 | KvDtype::Mxfp8 => {
                        // OCP MX packs 32 elements per 17/25/33-byte block.
                        // Same rationale as the NVFP4 assert: fail clearly at
                        // construction, not later in the attention loop.
                        use rustllama_kernels_cpu::mxfp;
                        assert!(
                            cfg.head_dim % 32 == 0,
                            "mxfp KV requires head_dim divisible by 32, got {}",
                            cfg.head_dim,
                        );
                        let blocks_per_row = cfg.head_dim / 32;
                        let blk_bytes = match dtype {
                            KvDtype::Mxfp4 => mxfp::MXFP4_BLOCK_BYTES,
                            KvDtype::Mxfp6 => mxfp::MXFP6_BLOCK_BYTES,
                            _ => mxfp::MXFP8_BLOCK_BYTES,
                        };
                        let packed_len = total_rows * blocks_per_row * blk_bytes;
                        let k = alloc_kv_buf_u8(stream_ref, packed_len);
                        let v = alloc_kv_buf_u8(stream_ref, packed_len);
                        match dtype {
                            KvDtype::Mxfp4 => KvLayer::Mxfp4 {
                                k_packed: k,
                                v_packed: v,
                            },
                            KvDtype::Mxfp6 => KvLayer::Mxfp6 {
                                k_packed: k,
                                v_packed: v,
                            },
                            _ => KvLayer::Mxfp8 {
                                k_packed: k,
                                v_packed: v,
                            },
                        }
                    }
                }
            })
            .collect();
        Self {
            layers,
            mla: None,
            seq_len: 0,
            max_ctx,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            layer_dtypes,
            layer_caps,
            kv_bias: None,
            usm_stream,
        }
    }

    pub fn reset(&mut self) {
        // Host KV identity changes wholesale: any thread-local USM
        // attention mirror must stop trusting its rows (see
        // `accel::usm_attn_kv_epoch_bump`).
        crate::accel::usm_attn_kv_epoch_bump();
        for layer in &mut self.layers {
            match layer {
                KvLayer::F32 { k, v } => {
                    for x in k.iter_mut() {
                        *x = 0.0;
                    }
                    for x in v.iter_mut() {
                        *x = 0.0;
                    }
                }
                KvLayer::Q8_0 {
                    k_q,
                    k_scales,
                    v_q,
                    v_scales,
                } => {
                    for x in k_q.iter_mut() {
                        *x = 0;
                    }
                    for x in k_scales.iter_mut() {
                        *x = 0.0;
                    }
                    for x in v_q.iter_mut() {
                        *x = 0;
                    }
                    for x in v_scales.iter_mut() {
                        *x = 0.0;
                    }
                }
                KvLayer::TurboQuant {
                    bits: _,
                    k_packed,
                    k_scales,
                    v_packed,
                    v_scales,
                } => {
                    for x in k_packed.iter_mut() {
                        *x = 0;
                    }
                    for x in v_packed.iter_mut() {
                        *x = 0;
                    }
                    for x in k_scales.iter_mut() {
                        *x = 0.0;
                    }
                    for x in v_scales.iter_mut() {
                        *x = 0.0;
                    }
                }
                KvLayer::Nvfp4 { k_packed, v_packed } => {
                    for x in k_packed.iter_mut() {
                        *x = 0;
                    }
                    for x in v_packed.iter_mut() {
                        *x = 0;
                    }
                }
                KvLayer::Q4_0 { k_q, v_q } => {
                    for x in k_q.iter_mut() {
                        *x = 0;
                    }
                    for x in v_q.iter_mut() {
                        *x = 0;
                    }
                }
                KvLayer::Mxfp4 { k_packed, v_packed }
                | KvLayer::Mxfp6 { k_packed, v_packed }
                | KvLayer::Mxfp8 { k_packed, v_packed } => {
                    for x in k_packed.iter_mut() {
                        *x = 0;
                    }
                    for x in v_packed.iter_mut() {
                        *x = 0;
                    }
                }
                KvLayer::Half { k, v, .. } => {
                    for x in k.iter_mut() {
                        *x = 0;
                    }
                    for x in v.iter_mut() {
                        *x = 0;
                    }
                }
            }
        }
        // M1: clear the MLA latent + rope-key slabs (reads are bounded by
        // seq_len, so this is belt-and-suspenders against stale rows).
        if let Some(mla) = self.mla.as_mut() {
            for s in mla.latent.iter_mut() {
                s.fill(0.0);
            }
            for s in mla.rope_key.iter_mut() {
                s.fill(0.0);
            }
        }
        self.seq_len = 0;
    }

    /// Snapshot the first `prefix_len` positions of every layer's K/V
    /// into a heap-allocated [`KvSnapshot`] sized to just that prefix.
    /// The on-disk layout is head-major `[n_kv_heads, prefix_len, head_dim]`,
    /// matching the live cache's stride math so [`Self::restore_prefix`]
    /// can write back without reshaping. Used by the prompt prefix cache.
    pub fn snapshot_prefix(&self, prefix_len: usize) -> KvSnapshot {
        assert!(
            prefix_len <= self.seq_len,
            "snapshot_prefix {prefix_len} exceeds seq_len {}",
            self.seq_len
        );
        // P2: a ring layer holds only the last `cap` positions, so it can't
        // snapshot a `max_ctx`-strided prefix. The engine disables prefix-
        // cache snapshot/restore for ring models (see `commit_snapshot`); a
        // ring cache must never reach here.
        debug_assert!(
            !self.has_ring_layers(),
            "snapshot_prefix on a SWA ring cache (prefix cache must be disabled for ring models)"
        );
        let n_h = self.n_kv_heads;
        let hd = self.head_dim;
        let row_len = hd; // bytes per (head, token) row, in elements
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                // Sparse-cache placeholder (zero-length slab — hybrid
                // SSM / excluded MTP layers): nothing to capture;
                // cloning an empty layer yields an empty placeholder
                // that `restore_prefix` skips symmetrically.
                if layer.approx_bytes() == 0 {
                    return layer.clone();
                }
                match layer {
                    KvLayer::F32 { k, v } => {
                        let mut snap_k = vec![0f32; n_h * prefix_len * hd];
                        let mut snap_v = vec![0f32; n_h * prefix_len * hd];
                        for h in 0..n_h {
                            let src_base = h * self.max_ctx * hd;
                            let dst_base = h * prefix_len * hd;
                            snap_k[dst_base..dst_base + prefix_len * hd]
                                .copy_from_slice(&k[src_base..src_base + prefix_len * hd]);
                            snap_v[dst_base..dst_base + prefix_len * hd]
                                .copy_from_slice(&v[src_base..src_base + prefix_len * hd]);
                        }
                        let _ = row_len;
                        KvLayer::F32 {
                            k: KvBuf::from_vec(snap_k),
                            v: KvBuf::from_vec(snap_v),
                        }
                    }
                    KvLayer::Q8_0 {
                        k_q,
                        k_scales,
                        v_q,
                        v_scales,
                    } => {
                        let mut snap_kq = vec![0i8; n_h * prefix_len * hd];
                        let mut snap_vq = vec![0i8; n_h * prefix_len * hd];
                        let mut snap_ks = vec![0f32; n_h * prefix_len];
                        let mut snap_vs = vec![0f32; n_h * prefix_len];
                        for h in 0..n_h {
                            let q_src = h * self.max_ctx * hd;
                            let q_dst = h * prefix_len * hd;
                            snap_kq[q_dst..q_dst + prefix_len * hd]
                                .copy_from_slice(&k_q[q_src..q_src + prefix_len * hd]);
                            snap_vq[q_dst..q_dst + prefix_len * hd]
                                .copy_from_slice(&v_q[q_src..q_src + prefix_len * hd]);
                            let s_src = h * self.max_ctx;
                            let s_dst = h * prefix_len;
                            snap_ks[s_dst..s_dst + prefix_len]
                                .copy_from_slice(&k_scales[s_src..s_src + prefix_len]);
                            snap_vs[s_dst..s_dst + prefix_len]
                                .copy_from_slice(&v_scales[s_src..s_src + prefix_len]);
                        }
                        KvLayer::Q8_0 {
                            k_q: KvBuf::from_vec(snap_kq),
                            k_scales: KvBuf::from_vec(snap_ks),
                            v_q: KvBuf::from_vec(snap_vq),
                            v_scales: KvBuf::from_vec(snap_vs),
                        }
                    }
                    KvLayer::TurboQuant {
                        bits,
                        k_packed,
                        k_scales,
                        v_packed,
                        v_scales,
                    } => {
                        let bytes_per_row =
                            rustllama_kernels_cpu::turboquant::bytes_per_block(hd, *bits);
                        let mut snap_kp = vec![0u8; n_h * prefix_len * bytes_per_row];
                        let mut snap_vp = vec![0u8; n_h * prefix_len * bytes_per_row];
                        let mut snap_ks = vec![0f32; n_h * prefix_len];
                        let mut snap_vs = vec![0f32; n_h * prefix_len];
                        for h in 0..n_h {
                            let p_src = h * self.max_ctx * bytes_per_row;
                            let p_dst = h * prefix_len * bytes_per_row;
                            snap_kp[p_dst..p_dst + prefix_len * bytes_per_row].copy_from_slice(
                                &k_packed[p_src..p_src + prefix_len * bytes_per_row],
                            );
                            snap_vp[p_dst..p_dst + prefix_len * bytes_per_row].copy_from_slice(
                                &v_packed[p_src..p_src + prefix_len * bytes_per_row],
                            );
                            let s_src = h * self.max_ctx;
                            let s_dst = h * prefix_len;
                            snap_ks[s_dst..s_dst + prefix_len]
                                .copy_from_slice(&k_scales[s_src..s_src + prefix_len]);
                            snap_vs[s_dst..s_dst + prefix_len]
                                .copy_from_slice(&v_scales[s_src..s_src + prefix_len]);
                        }
                        KvLayer::TurboQuant {
                            bits: *bits,
                            k_packed: KvBuf::from_vec(snap_kp),
                            k_scales: KvBuf::from_vec(snap_ks),
                            v_packed: KvBuf::from_vec(snap_vp),
                            v_scales: KvBuf::from_vec(snap_vs),
                        }
                    }
                    KvLayer::Nvfp4 { k_packed, v_packed } => {
                        let blocks_per_row = hd / rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS;
                        let bytes_per_row =
                            blocks_per_row * rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES;
                        let mut snap_kp = vec![0u8; n_h * prefix_len * bytes_per_row];
                        let mut snap_vp = vec![0u8; n_h * prefix_len * bytes_per_row];
                        for h in 0..n_h {
                            let p_src = h * self.max_ctx * bytes_per_row;
                            let p_dst = h * prefix_len * bytes_per_row;
                            snap_kp[p_dst..p_dst + prefix_len * bytes_per_row].copy_from_slice(
                                &k_packed[p_src..p_src + prefix_len * bytes_per_row],
                            );
                            snap_vp[p_dst..p_dst + prefix_len * bytes_per_row].copy_from_slice(
                                &v_packed[p_src..p_src + prefix_len * bytes_per_row],
                            );
                        }
                        KvLayer::Nvfp4 {
                            k_packed: KvBuf::from_vec(snap_kp),
                            v_packed: KvBuf::from_vec(snap_vp),
                        }
                    }
                    KvLayer::Q4_0 { k_q, v_q } => {
                        let blocks_per_row = hd / rustllama_kernels_cpu::q4_0_kv::Q4_0_BLOCK_ELEMS;
                        let bytes_per_row =
                            blocks_per_row * rustllama_kernels_cpu::q4_0_kv::Q4_0_BLOCK_BYTES;
                        let mut snap_kq = vec![0u8; n_h * prefix_len * bytes_per_row];
                        let mut snap_vq = vec![0u8; n_h * prefix_len * bytes_per_row];
                        for h in 0..n_h {
                            let p_src = h * self.max_ctx * bytes_per_row;
                            let p_dst = h * prefix_len * bytes_per_row;
                            snap_kq[p_dst..p_dst + prefix_len * bytes_per_row]
                                .copy_from_slice(&k_q[p_src..p_src + prefix_len * bytes_per_row]);
                            snap_vq[p_dst..p_dst + prefix_len * bytes_per_row]
                                .copy_from_slice(&v_q[p_src..p_src + prefix_len * bytes_per_row]);
                        }
                        KvLayer::Q4_0 {
                            k_q: KvBuf::from_vec(snap_kq),
                            v_q: KvBuf::from_vec(snap_vq),
                        }
                    }
                    KvLayer::Mxfp4 { k_packed, v_packed }
                    | KvLayer::Mxfp6 { k_packed, v_packed }
                    | KvLayer::Mxfp8 { k_packed, v_packed } => {
                        use rustllama_kernels_cpu::mxfp;
                        let blk_bytes = match layer {
                            KvLayer::Mxfp4 { .. } => mxfp::MXFP4_BLOCK_BYTES,
                            KvLayer::Mxfp6 { .. } => mxfp::MXFP6_BLOCK_BYTES,
                            _ => mxfp::MXFP8_BLOCK_BYTES,
                        };
                        let bytes_per_row = (hd / 32) * blk_bytes;
                        let mut snap_kp = vec![0u8; n_h * prefix_len * bytes_per_row];
                        let mut snap_vp = vec![0u8; n_h * prefix_len * bytes_per_row];
                        for h in 0..n_h {
                            let p_src = h * self.max_ctx * bytes_per_row;
                            let p_dst = h * prefix_len * bytes_per_row;
                            snap_kp[p_dst..p_dst + prefix_len * bytes_per_row].copy_from_slice(
                                &k_packed[p_src..p_src + prefix_len * bytes_per_row],
                            );
                            snap_vp[p_dst..p_dst + prefix_len * bytes_per_row].copy_from_slice(
                                &v_packed[p_src..p_src + prefix_len * bytes_per_row],
                            );
                        }
                        let k = KvBuf::from_vec(snap_kp);
                        let v = KvBuf::from_vec(snap_vp);
                        match layer {
                            KvLayer::Mxfp4 { .. } => KvLayer::Mxfp4 {
                                k_packed: k,
                                v_packed: v,
                            },
                            KvLayer::Mxfp6 { .. } => KvLayer::Mxfp6 {
                                k_packed: k,
                                v_packed: v,
                            },
                            _ => KvLayer::Mxfp8 {
                                k_packed: k,
                                v_packed: v,
                            },
                        }
                    }
                    KvLayer::Half { bf16, k, v } => {
                        // u16 half slab, head-major [n_h, prefix_len, head_dim].
                        let mut snap_k = vec![0u16; n_h * prefix_len * hd];
                        let mut snap_v = vec![0u16; n_h * prefix_len * hd];
                        for h in 0..n_h {
                            let src_base = h * self.max_ctx * hd;
                            let dst_base = h * prefix_len * hd;
                            snap_k[dst_base..dst_base + prefix_len * hd]
                                .copy_from_slice(&k[src_base..src_base + prefix_len * hd]);
                            snap_v[dst_base..dst_base + prefix_len * hd]
                                .copy_from_slice(&v[src_base..src_base + prefix_len * hd]);
                        }
                        KvLayer::Half {
                            bf16: *bf16,
                            k: KvBuf::from_vec(snap_k),
                            v: KvBuf::from_vec(snap_v),
                        }
                    }
                }
            })
            .collect();
        KvSnapshot {
            layers,
            prefix_len,
            n_kv_heads: n_h,
            head_dim: hd,
            dtype: self.dtype(),
        }
    }

    /// Like [`Self::snapshot_prefix`] but captures only the layers
    /// marked `keep`, emitting zero-length placeholders for the rest
    /// (roadmap Phase 5). Hybrid models pass the full-attention mask:
    /// SSM layers' KV slabs are allocated but never written, so
    /// capturing them would waste ~3/4 of a qwen35moe snapshot's
    /// bytes on garbage. [`Self::restore_prefix`] leaves placeholder
    /// layers untouched. F32-only fast path — hybrid models force
    /// F32 KV; other dtypes fall back to the full capture.
    pub fn snapshot_prefix_selective(&self, prefix_len: usize, keep: &[bool]) -> KvSnapshot {
        assert_eq!(keep.len(), self.layers.len());
        // F4: the selective fast path is F32-only; a cache that isn't
        // uniformly F32 (any quant/half layer, or a mixed per-layer
        // cache) falls back to the full capture, which handles every
        // layer dtype correctly.
        if !self.is_uniform_f32() {
            return self.snapshot_prefix(prefix_len);
        }
        assert!(
            prefix_len <= self.seq_len,
            "snapshot_prefix_selective {prefix_len} exceeds seq_len {}",
            self.seq_len
        );
        let n_h = self.n_kv_heads;
        let hd = self.head_dim;
        let layers = self
            .layers
            .iter()
            .zip(keep.iter())
            .map(|(layer, &kp)| match layer {
                KvLayer::F32 { k, v } => {
                    // `!kp` = the caller doesn't want this layer.
                    // `k.is_empty()` = the sparse cache never
                    // allocated it (e.g. the MTP block, which the
                    // dn-sentinel mask marks keep=true because it IS
                    // a full-attention layer — but the sparse cache
                    // skipped it as excluded-from-forward). Either
                    // way: placeholder.
                    if !kp || (**k).is_empty() {
                        return KvLayer::F32 {
                            k: KvBuf::from_vec(Vec::new()),
                            v: KvBuf::from_vec(Vec::new()),
                        };
                    }
                    let mut snap_k = vec![0f32; n_h * prefix_len * hd];
                    let mut snap_v = vec![0f32; n_h * prefix_len * hd];
                    for h in 0..n_h {
                        let src_base = h * self.max_ctx * hd;
                        let dst_base = h * prefix_len * hd;
                        snap_k[dst_base..dst_base + prefix_len * hd]
                            .copy_from_slice(&k[src_base..src_base + prefix_len * hd]);
                        snap_v[dst_base..dst_base + prefix_len * hd]
                            .copy_from_slice(&v[src_base..src_base + prefix_len * hd]);
                    }
                    KvLayer::F32 {
                        k: KvBuf::from_vec(snap_k),
                        v: KvBuf::from_vec(snap_v),
                    }
                }
                _ => unreachable!("dtype checked F32 above"),
            })
            .collect();
        KvSnapshot {
            layers,
            prefix_len,
            n_kv_heads: n_h,
            head_dim: hd,
            dtype: self.dtype(),
        }
    }

    /// Restore a [`KvSnapshot`] into the live cache and set `seq_len` to
    /// the snapshot's prefix length. The snapshot must match this cache's
    /// `n_kv_heads`, `head_dim`, and `dtype` exactly (the model
    /// architecture and KV dtype don't change at runtime).
    /// Zero-length placeholder layers (selective snapshots) are left
    /// untouched — for hybrid models those are SSM layers whose KV is
    /// never read.
    ///
    /// Bytes outside `[0..prefix_len]` in each head are left untouched —
    /// the cache's attention loop only reads up to `seq_len`, so stale
    /// data beyond that point can't corrupt the next request.
    pub fn restore_prefix(&mut self, snap: &KvSnapshot) {
        // Same contract as `reset`: the live host KV's content
        // identity changes without a prefill replay, so per-thread
        // USM attention mirrors must re-validate from scratch.
        crate::accel::usm_attn_kv_epoch_bump();
        assert_eq!(snap.n_kv_heads, self.n_kv_heads);
        assert_eq!(snap.head_dim, self.head_dim);
        // Representative-dtype sanity check; the per-layer `(live, src)`
        // KvLayer-variant match below enforces the exact per-layer dtype
        // and panics on any mismatch (so a mixed cache is fully covered).
        assert_eq!(snap.dtype, self.dtype());
        assert!(
            snap.prefix_len <= self.max_ctx,
            "snapshot prefix_len {} exceeds max_ctx {}",
            snap.prefix_len,
            self.max_ctx
        );
        assert_eq!(snap.layers.len(), self.layers.len());
        let n_h = self.n_kv_heads;
        let hd = self.head_dim;
        let pl = snap.prefix_len;
        for (live, src) in self.layers.iter_mut().zip(snap.layers.iter()) {
            // Selective-snapshot placeholder (hybrid SSM layers):
            // nothing was captured, nothing to restore.
            if pl > 0 && src.approx_bytes() == 0 {
                continue;
            }
            // A snapshot carrying data for a layer the (sparse) live
            // cache never allocated means the keep-masks diverged —
            // a construction bug, not a runtime condition.
            debug_assert!(
                !(pl > 0 && live.approx_bytes() == 0 && src.approx_bytes() != 0),
                "restore_prefix: snapshot has data for a layer the live cache \
                 never allocated (keep-mask mismatch)"
            );
            match (live, src) {
                (KvLayer::F32 { k, v }, KvLayer::F32 { k: sk, v: sv }) => {
                    for h in 0..n_h {
                        let dst_base = h * self.max_ctx * hd;
                        let src_base = h * pl * hd;
                        k[dst_base..dst_base + pl * hd]
                            .copy_from_slice(&sk[src_base..src_base + pl * hd]);
                        v[dst_base..dst_base + pl * hd]
                            .copy_from_slice(&sv[src_base..src_base + pl * hd]);
                    }
                }
                (
                    KvLayer::Q8_0 {
                        k_q,
                        k_scales,
                        v_q,
                        v_scales,
                    },
                    KvLayer::Q8_0 {
                        k_q: sk_q,
                        k_scales: sks,
                        v_q: sv_q,
                        v_scales: svs,
                    },
                ) => {
                    for h in 0..n_h {
                        let q_dst = h * self.max_ctx * hd;
                        let q_src = h * pl * hd;
                        k_q[q_dst..q_dst + pl * hd].copy_from_slice(&sk_q[q_src..q_src + pl * hd]);
                        v_q[q_dst..q_dst + pl * hd].copy_from_slice(&sv_q[q_src..q_src + pl * hd]);
                        let s_dst = h * self.max_ctx;
                        let s_src = h * pl;
                        k_scales[s_dst..s_dst + pl].copy_from_slice(&sks[s_src..s_src + pl]);
                        v_scales[s_dst..s_dst + pl].copy_from_slice(&svs[s_src..s_src + pl]);
                    }
                }
                (
                    KvLayer::TurboQuant {
                        bits: live_bits,
                        k_packed,
                        k_scales,
                        v_packed,
                        v_scales,
                    },
                    KvLayer::TurboQuant {
                        bits: snap_bits,
                        k_packed: skp,
                        k_scales: sks,
                        v_packed: svp,
                        v_scales: svs,
                    },
                ) => {
                    assert_eq!(*live_bits, *snap_bits, "tq bits mismatch on restore");
                    let bytes_per_row =
                        rustllama_kernels_cpu::turboquant::bytes_per_block(hd, *live_bits);
                    for h in 0..n_h {
                        let p_dst = h * self.max_ctx * bytes_per_row;
                        let p_src = h * pl * bytes_per_row;
                        k_packed[p_dst..p_dst + pl * bytes_per_row]
                            .copy_from_slice(&skp[p_src..p_src + pl * bytes_per_row]);
                        v_packed[p_dst..p_dst + pl * bytes_per_row]
                            .copy_from_slice(&svp[p_src..p_src + pl * bytes_per_row]);
                        let s_dst = h * self.max_ctx;
                        let s_src = h * pl;
                        k_scales[s_dst..s_dst + pl].copy_from_slice(&sks[s_src..s_src + pl]);
                        v_scales[s_dst..s_dst + pl].copy_from_slice(&svs[s_src..s_src + pl]);
                    }
                }
                (
                    KvLayer::Nvfp4 { k_packed, v_packed },
                    KvLayer::Nvfp4 {
                        k_packed: skp,
                        v_packed: svp,
                    },
                ) => {
                    let blocks_per_row = hd / rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS;
                    let bytes_per_row =
                        blocks_per_row * rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES;
                    for h in 0..n_h {
                        let p_dst = h * self.max_ctx * bytes_per_row;
                        let p_src = h * pl * bytes_per_row;
                        k_packed[p_dst..p_dst + pl * bytes_per_row]
                            .copy_from_slice(&skp[p_src..p_src + pl * bytes_per_row]);
                        v_packed[p_dst..p_dst + pl * bytes_per_row]
                            .copy_from_slice(&svp[p_src..p_src + pl * bytes_per_row]);
                    }
                }
                (KvLayer::Q4_0 { k_q, v_q }, KvLayer::Q4_0 { k_q: skq, v_q: svq }) => {
                    let blocks_per_row = hd / rustllama_kernels_cpu::q4_0_kv::Q4_0_BLOCK_ELEMS;
                    let bytes_per_row =
                        blocks_per_row * rustllama_kernels_cpu::q4_0_kv::Q4_0_BLOCK_BYTES;
                    for h in 0..n_h {
                        let p_dst = h * self.max_ctx * bytes_per_row;
                        let p_src = h * pl * bytes_per_row;
                        k_q[p_dst..p_dst + pl * bytes_per_row]
                            .copy_from_slice(&skq[p_src..p_src + pl * bytes_per_row]);
                        v_q[p_dst..p_dst + pl * bytes_per_row]
                            .copy_from_slice(&svq[p_src..p_src + pl * bytes_per_row]);
                    }
                }
                (
                    KvLayer::Mxfp4 { k_packed, v_packed },
                    KvLayer::Mxfp4 {
                        k_packed: skp,
                        v_packed: svp,
                    },
                ) => {
                    restore_mxfp_prefix(
                        k_packed,
                        v_packed,
                        skp,
                        svp,
                        hd,
                        n_h,
                        self.max_ctx,
                        pl,
                        rustllama_kernels_cpu::mxfp::MXFP4_BLOCK_BYTES,
                    );
                }
                (
                    KvLayer::Mxfp6 { k_packed, v_packed },
                    KvLayer::Mxfp6 {
                        k_packed: skp,
                        v_packed: svp,
                    },
                ) => {
                    restore_mxfp_prefix(
                        k_packed,
                        v_packed,
                        skp,
                        svp,
                        hd,
                        n_h,
                        self.max_ctx,
                        pl,
                        rustllama_kernels_cpu::mxfp::MXFP6_BLOCK_BYTES,
                    );
                }
                (
                    KvLayer::Mxfp8 { k_packed, v_packed },
                    KvLayer::Mxfp8 {
                        k_packed: skp,
                        v_packed: svp,
                    },
                ) => {
                    restore_mxfp_prefix(
                        k_packed,
                        v_packed,
                        skp,
                        svp,
                        hd,
                        n_h,
                        self.max_ctx,
                        pl,
                        rustllama_kernels_cpu::mxfp::MXFP8_BLOCK_BYTES,
                    );
                }
                (KvLayer::Half { k, v, .. }, KvLayer::Half { k: sk, v: sv, .. }) => {
                    for h in 0..n_h {
                        let dst_base = h * self.max_ctx * hd;
                        let src_base = h * pl * hd;
                        k[dst_base..dst_base + pl * hd]
                            .copy_from_slice(&sk[src_base..src_base + pl * hd]);
                        v[dst_base..dst_base + pl * hd]
                            .copy_from_slice(&sv[src_base..src_base + pl * hd]);
                    }
                }
                _ => panic!("KvLayer dtype mismatch between live cache and snapshot"),
            }
        }
        self.seq_len = pl;
    }
}

/// A clipped K/V cache snapshot used by the prompt prefix cache. Bytes
/// are laid out head-major with the same `(head, pos, dim)` indexing as
/// the live cache, but sized to `prefix_len` along the position axis
/// instead of `max_ctx`. Created by [`KvCache::snapshot_prefix`] and
/// consumed by [`KvCache::restore_prefix`].
#[derive(Debug, Clone)]
pub struct KvSnapshot {
    pub layers: Vec<KvLayer>,
    pub prefix_len: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub dtype: KvDtype,
}

/// Copy a snapshot's `prefix_len` positions back into the live MXFP KV
/// cache (shared by the 3 MXFP restore arms; only the block byte size
/// differs). Head-major `[n_h, max_ctx, bytes_per_row]` live vs
/// `[n_h, pl, bytes_per_row]` snapshot layout.
#[allow(clippy::too_many_arguments)]
fn restore_mxfp_prefix(
    k_packed: &mut KvBuf<u8>,
    v_packed: &mut KvBuf<u8>,
    skp: &KvBuf<u8>,
    svp: &KvBuf<u8>,
    hd: usize,
    n_h: usize,
    max_ctx: usize,
    pl: usize,
    blk_bytes: usize,
) {
    let bytes_per_row = (hd / 32) * blk_bytes;
    for h in 0..n_h {
        let p_dst = h * max_ctx * bytes_per_row;
        let p_src = h * pl * bytes_per_row;
        k_packed[p_dst..p_dst + pl * bytes_per_row]
            .copy_from_slice(&skp[p_src..p_src + pl * bytes_per_row]);
        v_packed[p_dst..p_dst + pl * bytes_per_row]
            .copy_from_slice(&svp[p_src..p_src + pl * bytes_per_row]);
    }
}

/// Quantize one new K/V row (all `n_kv_heads`) into the MXFP host cache
/// at position `cur_pos`. `fmt` picks the block bytes + quantizer.
#[allow(clippy::too_many_arguments)]
fn mxfp_kv_quantize_row(
    fmt: KvDtype,
    k_buf: &[f32],
    v_buf: &[f32],
    k_packed: &mut KvBuf<u8>,
    v_packed: &mut KvBuf<u8>,
    cur_pos: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
) {
    use rustllama_kernels_cpu::{mxfp, mxfp_kv};
    let (blk_bytes, qfn): (usize, fn(&[f32], &mut [u8])) = match fmt {
        KvDtype::Mxfp4 => (mxfp::MXFP4_BLOCK_BYTES, mxfp_kv::quantize_block_mxfp4),
        KvDtype::Mxfp6 => (mxfp::MXFP6_BLOCK_BYTES, mxfp_kv::quantize_block_mxfp6),
        _ => (mxfp::MXFP8_BLOCK_BYTES, mxfp_kv::quantize_block_mxfp8),
    };
    let blocks_per_row = head_dim / 32;
    let bytes_per_row = blocks_per_row * blk_bytes;
    for h in 0..n_kv_heads {
        let p_dst = (h * max_ctx + cur_pos) * bytes_per_row;
        for b in 0..blocks_per_row {
            let eo = h * head_dim + b * 32;
            let bd = p_dst + b * blk_bytes;
            qfn(&k_buf[eo..eo + 32], &mut k_packed[bd..bd + blk_bytes]);
            qfn(&v_buf[eo..eo + 32], &mut v_packed[bd..bd + blk_bytes]);
        }
    }
}

/// MXFP KV flash-attention DECODE: quantize the new K/V row into the
/// host mirror at `cur_pos`, try the GPU (CUDA→SYCL), else the CPU
/// kernel. `fmt` selects the per-format block bytes + kernels. Shared by
/// every MXFP decode arm in the forward path.
#[allow(clippy::too_many_arguments)]
fn mxfp_kv_decode(
    fmt: KvDtype,
    q_buf: &[f32],
    k_buf: &[f32],
    v_buf: &[f32],
    k_packed: &mut KvBuf<u8>,
    v_packed: &mut KvBuf<u8>,
    attn_out: &mut [f32],
    li: usize,
    cur_pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
    n_layers: usize,
    // Per-layer SWA window (0=full) + Gemma2 attn-logit soft-cap (0.0=off),
    // threaded into the GPU dispatch AND the CPU `_capped_windowed` fallback.
    // Dense callers pass real values; hybrid callers pass 0 / 0.0 (passthrough
    // == byte-identical to the base kernel).
    window: u32,
    softcap: f32,
) {
    use rustllama_kernels_cpu::{mxfp, mxfp_kv};
    let (blk_bytes, qfn): (usize, fn(&[f32], &mut [u8])) = match fmt {
        KvDtype::Mxfp4 => (mxfp::MXFP4_BLOCK_BYTES, mxfp_kv::quantize_block_mxfp4),
        KvDtype::Mxfp6 => (mxfp::MXFP6_BLOCK_BYTES, mxfp_kv::quantize_block_mxfp6),
        _ => (mxfp::MXFP8_BLOCK_BYTES, mxfp_kv::quantize_block_mxfp8),
    };
    let blocks_per_row = head_dim / 32;
    let bytes_per_row = blocks_per_row * blk_bytes;
    for h in 0..n_kv_heads {
        let p_dst = (h * max_ctx + cur_pos) * bytes_per_row;
        for b in 0..blocks_per_row {
            let eo = h * head_dim + b * 32;
            let bd = p_dst + b * blk_bytes;
            qfn(&k_buf[eo..eo + 32], &mut k_packed[bd..bd + blk_bytes]);
            qfn(&v_buf[eo..eo + 32], &mut v_packed[bd..bd + blk_bytes]);
        }
    }
    let k_rows = &k_buf[..n_kv_heads * head_dim];
    let v_rows = &v_buf[..n_kv_heads * head_dim];
    // Seed the CUDA decode mirror with the prefill history on the first decode
    // step (no-op once resident); host k_packed/v_packed are byte-identical to
    // the mirror, so a direct copy is correct. Without it the mirror gap
    // declines and decode runs on CPU for the whole generation.
    #[allow(clippy::type_complexity)]
    let seed_fn: fn(&[u8], &[u8], usize, u32, u32, u32, u32, u32, u32) -> bool = match fmt {
        KvDtype::Mxfp4 => crate::accel::decode_seed_kv_mxfp4,
        KvDtype::Mxfp6 => crate::accel::decode_seed_kv_mxfp6,
        _ => crate::accel::decode_seed_kv_mxfp8,
    };
    seed_fn(
        &k_packed[..],
        &v_packed[..],
        li,
        cur_pos as u32,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        max_ctx as u32,
        n_layers as u32,
    );
    #[allow(clippy::type_complexity)]
    let gpu_fn: fn(
        &[f32],
        &[f32],
        &[f32],
        &mut [f32],
        usize,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        f32,
    ) -> bool = match fmt {
        KvDtype::Mxfp4 => crate::accel::try_flash_attn_decode_gpu_mxfp4,
        KvDtype::Mxfp6 => crate::accel::try_flash_attn_decode_gpu_mxfp6,
        _ => crate::accel::try_flash_attn_decode_gpu_mxfp8,
    };
    let gpu_ok = gpu_fn(
        q_buf,
        k_rows,
        v_rows,
        attn_out,
        li,
        cur_pos as u32,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        max_ctx as u32,
        n_layers as u32,
        window,
        softcap,
    );
    if !gpu_ok {
        #[allow(clippy::type_complexity)]
        let cpu_fn: fn(
            &[f32],
            &[u8],
            &[u8],
            &mut [f32],
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            f32,
        ) = match fmt {
            KvDtype::Mxfp4 => mxfp_kv::gqa_attention_flash_decode_mxfp4_capped_windowed,
            KvDtype::Mxfp6 => mxfp_kv::gqa_attention_flash_decode_mxfp6_capped_windowed,
            _ => mxfp_kv::gqa_attention_flash_decode_mxfp8_capped_windowed,
        };
        cpu_fn(
            q_buf,
            k_packed,
            v_packed,
            attn_out,
            n_heads,
            n_kv_heads,
            head_dim,
            max_ctx,
            kv_len,
            window as usize,
            softcap,
        );
    }
}

/// MXFP KV flash-attention PREFILL: quantize `n_new` new K/V rows into
/// the host mirror at `[kv_len_base, kv_len_base+n_new)`, try GPU, else
/// CPU. `k_buf`/`v_buf` are laid out `[n_new, n_kv_heads, head_dim]`
/// (stride `d_kv = n_kv_heads*head_dim` per new position).
#[allow(clippy::too_many_arguments)]
fn mxfp_kv_prefill(
    fmt: KvDtype,
    q_buf: &[f32],
    k_buf: &[f32],
    v_buf: &[f32],
    k_packed: &mut KvBuf<u8>,
    v_packed: &mut KvBuf<u8>,
    attn_out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
    // Per-layer SWA window (0=full) + Gemma2 attn-logit soft-cap (0.0=off),
    // threaded into the GPU dispatch AND the CPU `_capped_windowed` fallback.
    // Dense callers pass real values; hybrid callers pass 0 / 0.0 (passthrough
    // == byte-identical to the base kernel).
    window: u32,
    softcap: f32,
) {
    use rustllama_kernels_cpu::{mxfp, mxfp_kv};
    let (blk_bytes, qfn): (usize, fn(&[f32], &mut [u8])) = match fmt {
        KvDtype::Mxfp4 => (mxfp::MXFP4_BLOCK_BYTES, mxfp_kv::quantize_block_mxfp4),
        KvDtype::Mxfp6 => (mxfp::MXFP6_BLOCK_BYTES, mxfp_kv::quantize_block_mxfp6),
        _ => (mxfp::MXFP8_BLOCK_BYTES, mxfp_kv::quantize_block_mxfp8),
    };
    let blocks_per_row = head_dim / 32;
    let bytes_per_row = blocks_per_row * blk_bytes;
    let d_kv = n_kv_heads * head_dim;
    for i in 0..n_new {
        let pos_i = kv_len_base + i;
        for h in 0..n_kv_heads {
            let p_dst = (h * max_ctx + pos_i) * bytes_per_row;
            for b in 0..blocks_per_row {
                let eo = i * d_kv + h * head_dim + b * 32;
                let bd = p_dst + b * blk_bytes;
                qfn(&k_buf[eo..eo + 32], &mut k_packed[bd..bd + blk_bytes]);
                qfn(&v_buf[eo..eo + 32], &mut v_packed[bd..bd + blk_bytes]);
            }
        }
    }
    #[allow(clippy::type_complexity)]
    let gpu_fn: fn(
        &[f32],
        &[u8],
        &[u8],
        &mut [f32],
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        u32,
        f32,
    ) -> bool = match fmt {
        KvDtype::Mxfp4 => crate::accel::try_flash_attn_prefill_gpu_mxfp4,
        KvDtype::Mxfp6 => crate::accel::try_flash_attn_prefill_gpu_mxfp6,
        _ => crate::accel::try_flash_attn_prefill_gpu_mxfp8,
    };
    let gpu_ok = gpu_fn(
        q_buf,
        k_packed,
        v_packed,
        attn_out,
        n_heads,
        n_kv_heads,
        head_dim,
        max_ctx,
        kv_len_base,
        n_new,
        window,
        softcap,
    );
    if !gpu_ok {
        #[allow(clippy::type_complexity)]
        let cpu_fn: fn(
            &[f32],
            &[u8],
            &[u8],
            &mut [f32],
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            f32,
        ) = match fmt {
            KvDtype::Mxfp4 => mxfp_kv::gqa_attention_flash_prefill_mxfp4_capped_windowed,
            KvDtype::Mxfp6 => mxfp_kv::gqa_attention_flash_prefill_mxfp6_capped_windowed,
            _ => mxfp_kv::gqa_attention_flash_prefill_mxfp8_capped_windowed,
        };
        cpu_fn(
            q_buf,
            k_packed,
            v_packed,
            attn_out,
            n_heads,
            n_kv_heads,
            head_dim,
            max_ctx,
            kv_len_base,
            n_new,
            window as usize,
            softcap,
        );
    }
}

impl KvLayer {
    fn clone_layer(&self) -> Self {
        match self {
            KvLayer::F32 { k, v } => KvLayer::F32 {
                k: k.clone(),
                v: v.clone(),
            },
            KvLayer::Q8_0 {
                k_q,
                k_scales,
                v_q,
                v_scales,
            } => KvLayer::Q8_0 {
                k_q: k_q.clone(),
                k_scales: k_scales.clone(),
                v_q: v_q.clone(),
                v_scales: v_scales.clone(),
            },
            KvLayer::TurboQuant {
                bits,
                k_packed,
                k_scales,
                v_packed,
                v_scales,
            } => KvLayer::TurboQuant {
                bits: *bits,
                k_packed: k_packed.clone(),
                k_scales: k_scales.clone(),
                v_packed: v_packed.clone(),
                v_scales: v_scales.clone(),
            },
            KvLayer::Nvfp4 { k_packed, v_packed } => KvLayer::Nvfp4 {
                k_packed: k_packed.clone(),
                v_packed: v_packed.clone(),
            },
            KvLayer::Q4_0 { k_q, v_q } => KvLayer::Q4_0 {
                k_q: k_q.clone(),
                v_q: v_q.clone(),
            },
            KvLayer::Mxfp4 { k_packed, v_packed } => KvLayer::Mxfp4 {
                k_packed: k_packed.clone(),
                v_packed: v_packed.clone(),
            },
            KvLayer::Mxfp6 { k_packed, v_packed } => KvLayer::Mxfp6 {
                k_packed: k_packed.clone(),
                v_packed: v_packed.clone(),
            },
            KvLayer::Mxfp8 { k_packed, v_packed } => KvLayer::Mxfp8 {
                k_packed: k_packed.clone(),
                v_packed: v_packed.clone(),
            },
            KvLayer::Half { bf16, k, v } => KvLayer::Half {
                bf16: *bf16,
                k: k.clone(),
                v: v.clone(),
            },
        }
    }
}

impl Clone for KvLayer {
    fn clone(&self) -> Self {
        self.clone_layer()
    }
}

/// Quantize one `head_dim`-long row into `out_q` + return the row's
/// scale. Per-row absmax → 8-bit signed ints. The dequant is exact for
/// the pre-quant max value (gives back ±scale) and within `scale/2` for
/// every other value.
pub(crate) fn quantize_row_q8_0(row: &[f32], out_q: &mut [i8]) -> f32 {
    debug_assert_eq!(row.len(), out_q.len());
    let mut max_abs = 0f32;
    for &x in row {
        let a = x.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if max_abs == 0.0 {
        for q in out_q.iter_mut() {
            *q = 0;
        }
        return 1.0;
    }
    let scale = max_abs / 127.0;
    let inv = 1.0 / scale;
    for (q, &x) in out_q.iter_mut().zip(row.iter()) {
        let r = (x * inv).round();
        *q = r.clamp(-128.0, 127.0) as i8;
    }
    scale
}

/// Compiled Llama-family model: weights + config.
pub struct LlamaModel {
    pub cfg: LlamaConfig,
    pub weights: LlamaWeights,
    /// Precomputed RoPE inverse-frequency table (`head_dim/2` entries) with
    /// context-extension scaling applied (`cfg.rope_scaling`), built ONCE at
    /// construction so the per-token forward doesn't recompute it. For
    /// non-scaled models this equals the base NeoX table, so the scaled path is
    /// byte-identical. The single source of truth fed to every backend's RoPE.
    pub rope_inv_freq: Vec<f32>,
    /// YaRN attention `mscale` (`1.0` off YaRN); multiplies the query after RoPE
    /// so attention scores carry the YaRN temperature.
    pub yarn_mscale: f32,
}

/// Build the scaled RoPE table + YaRN mscale from a config (see
/// [`crate::accel::rope_inv_freq_table_scaled`]). Used at every [`LlamaModel`]
/// construction so the tables are always consistent with `cfg.rope_scaling`.
pub fn rope_tables_from_cfg(cfg: &LlamaConfig) -> (Vec<f32>, f32) {
    (
        crate::accel::rope_inv_freq_table_scaled(
            cfg.head_dim,
            cfg.rope_theta,
            cfg.rope_scaling.as_ref(),
        ),
        crate::accel::rope_yarn_mscale(cfg.rope_scaling.as_ref()),
    )
}

impl LlamaModel {
    /// F9-tail — build a minimal transformer SHELL (`cfg` + EMPTY weights) for
    /// a CANONICAL Mamba/SSM model, so a `ContiguousEngine` can carry it in its
    /// `LlamaModel` slot while dispatching the actual forward to the
    /// selective-scan [`crate::mamba_arch::MambaModel`]. The cfg mirrors the
    /// SSM dims the engine's transformer-shaped accessors read (vocab /
    /// n_layers / d_model / eos / rms_eps); a single attention head of width
    /// `d_model` keeps the `n_heads · head_dim == d_model` invariant while the
    /// vestigial KV cache the engine still allocates stays a few KB. The
    /// weights are EMPTY and NEVER executed — any transformer forward on this
    /// shell is a bug (the engine routes Mamba models to the selective scan
    /// before any such call); it exists only so the Llama-shaped accessors and
    /// KV sizing return the right numbers.
    pub fn mamba_shell(
        cfg: &crate::mamba_arch::MambaConfig,
        eos_token_id: Option<u32>,
    ) -> LlamaModel {
        let lcfg = LlamaConfig {
            arch: "mamba".to_string(),
            n_layers: cfg.n_layers,
            n_heads: 1,
            n_kv_heads: 1,
            d_model: cfg.d_model,
            d_ff: cfg.d_inner,
            head_dim: cfg.d_model,
            rope_dim: cfg.d_model,
            rope_theta: 10_000.0,
            rms_eps: cfg.eps,
            vocab_size: cfg.vocab_size,
            ctx_train: 1 << 20,
            bos_token_id: None,
            eos_token_id,
            tie_word_embeddings: true,
            ..LlamaConfig::default()
        };
        let weights = LlamaWeights {
            token_embd: Tensor::zeros_cpu(
                Dtype::F32,
                vec![cfg.d_model as u64, cfg.vocab_size as u64],
            ),
            blocks: Vec::new(),
            moe_blocks: None,
            hybrid_layers: None,
            mla_blocks: None,
            output_norm: vec![0.0; cfg.d_model],
            output: None,
            mtp_heads: None,
            nextn_head: None,
            hadamard: None,
        };
        let (rope_inv_freq, yarn_mscale) = rope_tables_from_cfg(&lcfg);
        LlamaModel {
            cfg: lcfg,
            weights,
            rope_inv_freq,
            yarn_mscale,
        }
    }
}

/// Tree-attention descriptor threaded into the batched F32 forward for
/// speculative TREE verification (Feature 4). Borrowed views over the caller's
/// per-node arrays; `depths[i]` = node `i`'s tree depth (its RoPE position is
/// `start_pos + depths[i]`), `parent[i]` = tree-local parent index (negative =
/// root level). See [`LlamaModel::forward_tree_verify_f32`].
#[derive(Clone, Copy)]
struct TreeAttn<'a> {
    depths: &'a [u32],
    parent: &'a [i32],
}

// ============================================================
// Recurrent-state-under-tree verification (Feature 4, hybrid path)
// ============================================================
//
// WHY A SEPARATE WALK (the subtle part):
//   For a TRANSFORMER layer, tree verification is a single batched forward
//   with an ancestor attention mask — node `j` attends only its ancestors,
//   and the KV rows are order-independent (attention is a weighted sum over a
//   *set*). That is exactly what [`LlamaModel::forward_tree_verify_f32`] does
//   via [`TreeAttn`].
//
//   A RECURRENT layer (DeltaNet / Mamba / SSM) has no such freedom. Its output
//   at a token is a function of a SEQUENTIAL state `s` that was folded one
//   token at a time: `s_t = f(s_{t-1}, x_t)`. Two tree branches that share a
//   root diverge into DIFFERENT state evolutions after the branch point, so a
//   single linear scan over the flattened node list would advance one shared
//   state THROUGH every branch and compute garbage for all but the first.
//
//   The correct construction is the "caterpillar" walk: snapshot the recurrent
//   state at the verified branch root, then visit the tree in parent-before-
//   child order; at each node, RESTORE the state its PARENT left, advance it by
//   that node's single token, and record the post-step state for the node's own
//   children. Siblings each restart from the shared parent state, so branches
//   never contaminate one another.
//
// WHY A DEPTH-FIRST ORDER (and how it keeps the ATTENTION layers correct too):
//   The hybrid forward interleaves recurrent and full-attention layers on the
//   SAME residual stream, so one walk must satisfy both. We place node `j`'s K/V
//   at cache slot `start_pos + depth(j)` (NOT its flattened index). Because each
//   node has exactly one parent, a node's ancestor chain occupies exactly one
//   slot per depth `0..depth(j)` — a CONTIGUOUS causal window `[0, start_pos +
//   depth(j)]`. Visiting depth-first guarantees that at the instant node `j` is
//   stepped, slots `start_pos + 0 ..= start_pos + depth(j)` hold precisely `j`'s
//   root→`j` path (the descent just wrote them, and a node's subtree only ever
//   writes STRICTLY DEEPER slots, so an ancestor slot is never clobbered while
//   one of its descendants is live). So the ordinary causal single-token forward
//   `[0, pos+1)` *is* "prefix + ancestors + self" for the attention layers, with
//   zero masking — the recurrent snapshot/restore is the only added machinery.
//
// DEGENERATE CASE (the parity anchor): a CHAIN-shaped tree (`parent[j] == j-1`,
//   `depth[j] == j`) makes every restore a no-op (we restore the state we just
//   snapshotted) and every step sequential, so the walk reduces BIT-FOR-BIT to
//   the linear per-token forward. `toy_recurrent_tree_walk_chain_matches_serial`
//   pins this on the shared walk code; the real hybrid forward inherits it
//   because `step` is the already-validated single-token hybrid forward.

/// Driver the [`tree_recurrent_walk`] calls into for each node. Splitting the
/// walk (pure ordering + snapshot bookkeeping) from the per-node work (the
/// model forward, or a toy recurrence in tests) lets the SAME walk code be
/// exercised bit-exactly by a unit test and by the real hybrid verify — the
/// write-blind safety property, since no public MTP-hybrid GGUF exists to
/// validate end-to-end.
pub(crate) trait RecurrentTreeDriver {
    /// An immutable copy of the recurrent state, cheap enough to clone once per
    /// node that has children (the caterpillar's main chain).
    type Snapshot: Clone;
    /// Capture the current live recurrent state.
    fn snapshot(&self) -> Self::Snapshot;
    /// Overwrite the live recurrent state with a previously captured snapshot.
    fn restore(&mut self, snap: &Self::Snapshot);
    /// Advance the live recurrent state by node `j`'s single token and record
    /// node `j`'s output. The walk guarantees it called [`Self::restore`] with
    /// node `j`'s parent snapshot (or the root snapshot for a root-level node)
    /// immediately before this call, and that `j`'s ancestors were stepped
    /// earlier in the current descent (so any ancestor-dependent state — KV
    /// slots for the attention layers — is already in place).
    fn step(&mut self, j: usize);
}

/// Walk a speculative draft TREE in parent-before-child, depth-first order,
/// driving the recurrent state through each branch independently (see the
/// module comment above for the full rationale).
///
/// `parent[j]` is node `j`'s tree-local parent index, or negative for a
/// root-level node; it MUST be topologically ordered (`parent[j] < j`). The
/// driver's live state must enter at the verified branch root; it is left in an
/// UNSPECIFIED (last-visited-leaf) state on return, so a caller that needs a
/// particular node's post-state must snapshot it itself or replay.
pub(crate) fn tree_recurrent_walk<D: RecurrentTreeDriver>(driver: &mut D, parent: &[i32]) {
    let n = parent.len();
    if n == 0 {
        return;
    }
    // Build child adjacency + the root set in one pass.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut roots: Vec<usize> = Vec::new();
    for (j, &p) in parent.iter().enumerate() {
        if p < 0 {
            roots.push(j);
        } else {
            let pu = p as usize;
            debug_assert!(
                pu < j,
                "tree_recurrent_walk: parent {pu} must be topologically before child {j}"
            );
            children[pu].push(j);
        }
    }
    // The recurrent state as it enters = the branch-root state. Every root-level
    // node restarts from it.
    let root_snap = driver.snapshot();
    // Explicit DFS stack of (node, snapshot-to-restore-before-stepping-it). We
    // push children in reverse so they pop in natural (slice) order, which keeps
    // the main (argmax-first) branch of a caterpillar on top and matches the
    // flattened node order a chain produces.
    let mut stack: Vec<(usize, D::Snapshot)> = Vec::with_capacity(n);
    for &r in roots.iter().rev() {
        stack.push((r, root_snap.clone()));
    }
    while let Some((j, snap)) = stack.pop() {
        driver.restore(&snap);
        driver.step(j);
        if !children[j].is_empty() {
            // Only clone when there are children to seed — the caterpillar's
            // leaf siblings never reach here, so the all-main-chain case clones
            // exactly once per depth.
            let my_snap = driver.snapshot();
            for &c in children[j].iter().rev() {
                stack.push((c, my_snap.clone()));
            }
        }
    }
}

/// [`RecurrentTreeDriver`] for the real hybrid verify forward. `step(j)` is the
/// single-token hybrid forward at `pos = start_pos + depths[j]` (which both
/// writes node `j`'s K/V at that slot and advances the DeltaNet recurrence by
/// one token), capturing node `j`'s logits; snapshot/restore round-trip the
/// DeltaNet recurrent + conv state. See [`LlamaModel::forward_tree_verify_hybrid_f32`].
struct HybridTreeVerifyDriver<'a> {
    model: &'a LlamaModel,
    kv: &'a mut KvCache,
    dn: &'a mut DeltaNetCache,
    tokens: &'a [i32],
    start_pos: u32,
    depths: &'a [u32],
    logits_out: &'a mut [f32],
    vocab: usize,
}

impl RecurrentTreeDriver for HybridTreeVerifyDriver<'_> {
    type Snapshot = DeltaNetSnapshot;
    fn snapshot(&self) -> DeltaNetSnapshot {
        self.dn.snapshot()
    }
    fn restore(&mut self, snap: &DeltaNetSnapshot) {
        self.dn.restore(snap);
    }
    fn step(&mut self, j: usize) {
        let pos = self.start_pos + self.depths[j];
        let out = &mut self.logits_out[j * self.vocab..(j + 1) * self.vocab];
        // `forward_one_hybrid` writes K/V at slot `pos`, attends `[0, pos+1)`,
        // and folds the token into the DeltaNet state — exactly one node's worth
        // of recurrent advance. The walk has already restored this node's parent
        // state and (via the depth-first descent) laid down its ancestor KV, so
        // this call sees precisely "prefix + ancestors + self".
        self.model
            .forward_one_hybrid(self.tokens[j], pos, self.kv, self.dn, out);
    }
}

/// A CPU-backed weight range the Windows page-lock path may pin into
/// the working set. `addr`/`len` describe the *backing buffer* (for
/// `CpuOwnedSlice` this is the parent Arc, so dedup-by-`addr` locks a
/// shared parent once). `tier` is the importance class: 0 = always-hot
/// (embeddings, attention, norms, dense FFN, router); 1 = MoE shared
/// experts; 2 = routed-expert parent blobs (biggest + coldest, usually
/// left pageable). Produced by [`LlamaModel::collect_lock_targets`].
#[derive(Debug, Clone)]
pub struct LockTarget {
    pub tier: u8,
    pub name: String,
    pub addr: usize,
    pub len: usize,
}

/// Input to the per-token forward pass. The normal path looks up
/// the embedding from the token embedding table; the V-6b-3 vision
/// splice path supplies a pre-populated `d_model`-long row (the
/// projected image-patch embedding from V-5's
/// [`splice_image_embeddings`]).
#[derive(Copy, Clone)]
enum EmbedInput<'a> {
    Token(i32),
    Embed(&'a [f32]),
}

/// Final-LM-head dispatch mode for the batched-prefill body.
///
/// The same N-position forward body services two callers:
///
/// - **Prefill** wants only the last position's logits — that's the
///   distribution the next-token sampler reads.
/// - **Speculation** wants ALL N positions' logits — the verification
///   step in `accept_reject` reads each draft's predicted-next-token
///   distribution + the bonus position when all drafts accept.
///
/// Sharing the body keeps prefill and speculation bit-identical
/// through the per-layer math, with only the final LM-head step
/// differing. Pinned by the
/// `forward_speculation_batched_f32_matches_serial_for_same_input`
/// parity test.
enum LmHeadMode<'a> {
    /// Write logits for the LAST position only; `out` is `[vocab_size]`.
    Last(&'a mut [f32]),
    /// Write logits for ALL positions; `out` is `[n_new × vocab_size]`,
    /// row-major.
    All(&'a mut [f32]),
}

/// One slot's input to [`LlamaModel::forward_decode_paged_batched_f32`].
/// Carries the token to decode, its absolute KV position, the slot's
/// own paged cache (which the call mutates in-place by appending one
/// new K/V row), and the per-slot logits output buffer (one row of
/// `vocab_size` floats).
///
/// Per-slot borrows must be **disjoint** — the caller is responsible
/// for ensuring no two `DecodeSlot`s share the same
/// `&mut PagedKvCache` or overlapping `&mut [f32]`. Rust's borrow
/// checker enforces this naturally when slots are built from separate
/// `let mut` bindings; constructing slots by splitting a
/// `Vec<PagedKvCache>` requires `split_at_mut` and friends.
pub struct DecodeSlot<'a> {
    pub token_id: i32,
    pub pos: u32,
    pub cache: &'a mut crate::paged_kv_cache::PagedKvCache,
    pub logits_out: &'a mut [f32],
}

/// One slot's contribution to a RAGGED mixed-batch forward
/// ([`LlamaModel::forward_ragged_paged_f32`], Feature 3b). A ragged batch
/// concatenates several slots' query tokens end-to-end into ONE token buffer so
/// the input projections + FFN batch over the whole concatenated token dim in a
/// single kernel launch, while attention loops per slot (each slot has its own
/// positions, causal mask, and paged KV history).
///
/// Per-slot descriptor semantics — this is the subtle bookkeeping:
///   - `n_query` is how many query tokens this slot contributes. `n_query == 1`
///     ⇒ a **DECODE** row (one new token; gather `[0, pos+1)` + flash-decode).
///     `n_query > 1` ⇒ a **PREFILL CHUNK** (several new prompt tokens;
///     flash-prefill with varlen causal masking). One ragged batch may freely
///     mix both kinds — that is the whole point of the feature.
///   - `start_pos` is the ABSOLUTE KV position of this slot's FIRST query token.
///     Query `j` (`j in 0..n_query`) sits at absolute position `start_pos + j`,
///     which drives three things at once: its RoPE rotation angle, the paged-KV
///     write slot, and the causal-mask base (`kv_len_base = start_pos`, so query
///     `j` attends `[0, start_pos + j]`). For a decode row this is just the
///     slot's current `pos`.
///   - The slot's rows occupy the CONTIGUOUS range `[row_off, row_off+n_query)`
///     of the concatenated buffers, where `row_off` is the running sum of the
///     preceding slots' `n_query` — so the `tokens` buffer handed to the forward
///     MUST be laid out in the same slot order as the descriptor slice. The
///     forward recomputes `row_off` internally and asserts
///     `sum(n_query) == tokens.len()`.
///   - `cache` is this slot's paged KV cache (the `kv_page_table` of the plan):
///     the forward appends `n_query` new K/V rows at `[start_pos,
///     start_pos+n_query)` and gathers `[0, start_pos+n_query)` for attention.
///   - `logits_out` is a `[vocab_size]` buffer for this slot's LAST query
///     token's logits, OR an EMPTY slice to SKIP the LM head for this slot. A
///     decode row passes a real buffer (it samples its next token); a prefill
///     chunk that has not yet reached the prompt's end passes an empty slice
///     (its logits are unused — exactly as the serial paged prefill discards
///     them), so the LM head only runs for the rows that need it.
///
/// As with [`DecodeSlot`], the per-slot `cache` / `logits_out` borrows must be
/// disjoint across slots — the caller builds them from by-index disjoint
/// borrows.
pub struct RaggedSlot<'a> {
    pub n_query: usize,
    pub start_pos: u32,
    pub cache: &'a mut crate::paged_kv_cache::PagedKvCache,
    pub logits_out: &'a mut [f32],
}

/// P3 cascade / shared-prefix decode: a group of DECODE slots (indices into the
/// `slots` slice handed to [`LlamaModel::forward_decode_paged_batched_cascade_f32`]
/// / [`LlamaModel::forward_ragged_paged_cascade_f32`]) that share a page-aligned
/// APC prefix of `prefix_len` tokens. The cascade decode path computes the
/// shared-prefix attention ONCE and merges each slot's suffix via the
/// online-softmax merge in [`rustllama_kernels_cpu::cascade`], instead of every
/// slot re-attending the shared prefix independently — numerically equivalent
/// (~1e-6) to the per-slot path. The paged driver (`PagedBatchEngine`) builds
/// these each tick from the slots' shared pages; `None` / an empty slice ⇒ the
/// byte-identical per-slot path. Engaged only on full-attention layers
/// (`swa_window == 0`) since the cascade kernel has no windowed variant.
#[derive(Clone, Debug)]
pub struct CascadeGroup {
    /// Page-aligned shared prefix length, in tokens (same for every member).
    pub prefix_len: usize,
    /// Member slot indices into the forward's `slots` slice. Must all be
    /// decode rows (`n_query == 1`) sharing the prefix pages.
    pub members: Vec<usize>,
}

/// P3 cascade: compute decode-row attention for one [`CascadeGroup`] and write
/// each member's `[n_heads, head_dim]` attention output into `out`
/// (`[n_seq, n_heads, head_dim]`, member order). Gathers the shared prefix K/V
/// ONCE from `caches[0]` (identical physical pages across the group) and each
/// sequence's suffix `[prefix_len, kv_len)` separately, then runs the batched
/// online-softmax cascade merge. `queries` is `[n_seq, n_heads, head_dim]` in
/// member order; `softcap > 0` applies the Gemma2 attention-logit cap.
#[allow(clippy::too_many_arguments)]
fn cascade_decode_attention_group(
    shared: &crate::shared_paged_kv::SharedPagedKv,
    layer: u32,
    prefix_len: usize,
    caches: &[&crate::paged_kv_cache::PagedKvCache],
    kv_lens: &[usize],
    queries: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    softcap: f32,
) {
    let n_seq = caches.len();
    debug_assert_eq!(kv_lens.len(), n_seq);
    debug_assert_eq!(queries.len(), n_seq * n_heads * head_dim);
    debug_assert_eq!(out.len(), n_seq * n_heads * head_dim);
    // Shared prefix K/V, gathered ONCE (the P3 memory win: one buffer instead
    // of n_seq copies streamed independently).
    let mut prefix_k = vec![0f32; n_kv_heads * prefix_len * head_dim];
    let mut prefix_v = vec![0f32; n_kv_heads * prefix_len * head_dim];
    caches[0]
        .gather_layer_range_shared(
            shared,
            layer,
            0,
            prefix_len as u32,
            &mut prefix_k,
            &mut prefix_v,
        )
        .expect("cascade prefix gather in range");
    // Each sequence's private suffix [prefix_len, kv_len).
    let mut suffix_k: Vec<Vec<f32>> = Vec::with_capacity(n_seq);
    let mut suffix_v: Vec<Vec<f32>> = Vec::with_capacity(n_seq);
    for (s, &c) in caches.iter().enumerate() {
        let suf = kv_lens[s] - prefix_len;
        let mut sk = vec![0f32; n_kv_heads * suf * head_dim];
        let mut sv = vec![0f32; n_kv_heads * suf * head_dim];
        c.gather_layer_range_shared(
            shared,
            layer,
            prefix_len as u32,
            kv_lens[s] as u32,
            &mut sk,
            &mut sv,
        )
        .expect("cascade suffix gather in range");
        suffix_k.push(sk);
        suffix_v.push(sv);
    }
    let suffixes: Vec<rustllama_kernels_cpu::cascade::CascadeSuffix> = (0..n_seq)
        .map(|s| rustllama_kernels_cpu::cascade::CascadeSuffix {
            k: &suffix_k[s],
            v: &suffix_v[s],
            len: kv_lens[s] - prefix_len,
        })
        .collect();
    rustllama_kernels_cpu::cascade::cascade_attention_f32(
        &prefix_k, &prefix_v, prefix_len, queries, &suffixes, out, n_seq, n_heads, n_kv_heads,
        head_dim, softcap,
    );
}

// ============================================================
// T2: BatchedKv — one generic batched-prefill body over the KV dtype.
// ============================================================
//
// The six `forward_prefill_batched_*` entry points (F32 / Q8_0 / TQ /
// NVFP4 / Q4_0 / MXFP4·6·8) each carried ~350 lines of IDENTICAL per-layer
// dataflow — embed, pre-attn RMSNorm, batched Q/K/V projections, bias +
// per-head norm + RoPE, output projection, residual, FFN (dense batched or
// MoE per-token), and the LM-head dispatch — differing ONLY in the KV write
// (quantize-on-write) and the attention read (dequantize-on-attend).
// `BatchedKv` abstracts exactly that per-dtype delta so a single generic
// `LlamaModel::forward_prefill_batched_inner` carries the shared body
// (precedent: `forward_one_paged_f32_inner` + `PagedKvStoreOps`).
//
// `write_and_attend` is ONE method rather than a write_rows/attend pair
// because the MXFP path fuses quantize + flash-prefill inside the shared
// `mxfp_kv_prefill` helper (also used by the contiguous MXFP path), and the
// Q4_0 path threads a whitening rotation through BOTH the K/V write and the
// attn-out read — so the two phases can't be cleanly severed. Each impl
// lifts its dtype's original KV block VERBATIM, so the collapse is
// byte-identical (pinned by `tests/golden_batched_prefill.rs`).
//
// `window` / `softcap` (the D1b sliding-window + Gemma2 attn-logit soft-cap
// uniforms) and the optional `tree` descriptor are threaded in. Every impl —
// F32 and the quant dtypes (F1) — now threads `window`/`softcap` into both its
// GPU dispatch and its CPU `_capped_windowed` fallback (passthrough at 0 / 0.0,
// byte-identical to the base kernel for ordinary models). Only `tree` is
// F32-only; the quant impls always receive tree=None (they never supported
// tree verify) and `debug_assert!` it.
trait BatchedKv {
    #[allow(clippy::too_many_arguments)]
    fn write_and_attend(
        &self,
        kv: &mut KvCache,
        layer_idx: usize,
        q_buf: &mut [f32],
        k_buf: &mut [f32],
        v_buf: &mut [f32],
        attn_out: &mut [f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_len_base: usize,
        n_new: usize,
        window: usize,
        softcap: f32,
        tree: Option<&TreeAttn<'_>>,
    );
}

/// F32 KV: write K/V rows verbatim into the contiguous slab, then the tree
/// (ancestor-masked) or causal (SWA/softcap, GPU→CPU) flash-prefill.
struct F32Kv;
impl BatchedKv for F32Kv {
    fn write_and_attend(
        &self,
        kv: &mut KvCache,
        layer_idx: usize,
        q_buf: &mut [f32],
        k_buf: &mut [f32],
        v_buf: &mut [f32],
        attn_out: &mut [f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_len_base: usize,
        n_new: usize,
        window: usize,
        softcap: f32,
        tree: Option<&TreeAttn<'_>>,
    ) {
        let d_kv = n_kv_heads * head_dim;
        let kv_layer = &mut kv.layers[layer_idx];
        match kv_layer {
            KvLayer::F32 { k, v } => {
                for i in 0..n_new {
                    let pos_i = kv_len_base + i;
                    for h in 0..n_kv_heads {
                        let dst = (h * max_ctx + pos_i) * head_dim;
                        let src_k = &k_buf[i * d_kv + h * head_dim..i * d_kv + (h + 1) * head_dim];
                        let src_v = &v_buf[i * d_kv + h * head_dim..i * d_kv + (h + 1) * head_dim];
                        k[dst..dst + head_dim].copy_from_slice(src_k);
                        v[dst..dst + head_dim].copy_from_slice(src_v);
                    }
                }
                if let Some(t) = tree {
                    // Tree verify: each node attends only its ancestor chain
                    // (+ the committed prefix). CPU primitive; a chain tree
                    // reduces bit-for-bit to the causal flash-prefill below.
                    k::gqa_attention_tree_prefill(
                        q_buf,
                        k,
                        v,
                        attn_out,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_len_base,
                        n_new,
                        t.parent,
                    );
                } else {
                    // GPU prefill attention via SYCL/CUDA/MLX — carries the SWA
                    // window + Gemma2 soft-cap (bit-exact to the CPU reference),
                    // CPU capped+windowed fallback. Passthrough at window 0 /
                    // softcap 0.0 is byte-identical for ordinary models.
                    let used_gpu = crate::accel::try_flash_attn_prefill_gpu_f32(
                        q_buf,
                        k,
                        v,
                        attn_out,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_len_base,
                        n_new,
                        window as u32,
                        softcap,
                    );
                    if !used_gpu {
                        k::gqa_attention_flash_prefill_capped_windowed(
                            q_buf,
                            k,
                            v,
                            attn_out,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_ctx,
                            kv_len_base,
                            n_new,
                            window,
                            softcap,
                        );
                    }
                }
            }
            _ => unreachable!("forward_prefill_batched_f32 entered with non-F32 KV"),
        }
    }
}

/// TurboQuant KV: quantize each K/V row via `turboquant::quantize_row` into
/// the packed slabs, then the batched TQ flash-prefill (GPU→CPU).
struct TqKv;
impl BatchedKv for TqKv {
    fn write_and_attend(
        &self,
        kv: &mut KvCache,
        layer_idx: usize,
        q_buf: &mut [f32],
        k_buf: &mut [f32],
        v_buf: &mut [f32],
        attn_out: &mut [f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_len_base: usize,
        n_new: usize,
        window: usize,
        softcap: f32,
        tree: Option<&TreeAttn<'_>>,
    ) {
        debug_assert!(tree.is_none(), "TQ batched prefill has no tree-verify path");
        let d_kv = n_kv_heads * head_dim;
        let kv_layer = &mut kv.layers[layer_idx];
        match kv_layer {
            KvLayer::TurboQuant {
                bits,
                k_packed,
                k_scales,
                v_packed,
                v_scales,
            } => {
                let bytes_per_row =
                    rustllama_kernels_cpu::turboquant::bytes_per_block(head_dim, *bits);
                let mut scratch = vec![0f32; head_dim];
                for i in 0..n_new {
                    let pos_i = kv_len_base + i;
                    for h in 0..n_kv_heads {
                        let row_idx = h * max_ctx + pos_i;
                        let p_dst = row_idx * bytes_per_row;
                        // K row.
                        scratch.copy_from_slice(
                            &k_buf[i * d_kv + h * head_dim..i * d_kv + (h + 1) * head_dim],
                        );
                        k_scales[row_idx] = rustllama_kernels_cpu::turboquant::quantize_row(
                            &mut scratch,
                            *bits,
                            &mut k_packed[p_dst..p_dst + bytes_per_row],
                        );
                        // V row.
                        scratch.copy_from_slice(
                            &v_buf[i * d_kv + h * head_dim..i * d_kv + (h + 1) * head_dim],
                        );
                        v_scales[row_idx] = rustllama_kernels_cpu::turboquant::quantize_row(
                            &mut scratch,
                            *bits,
                            &mut v_packed[p_dst..p_dst + bytes_per_row],
                        );
                    }
                }
                if !crate::accel::try_flash_attn_prefill_gpu_tq(
                    q_buf,
                    k_packed,
                    k_scales,
                    v_packed,
                    v_scales,
                    *bits,
                    attn_out,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_ctx,
                    kv_len_base,
                    n_new,
                    window as u32,
                    softcap,
                ) {
                    rustllama_kernels_cpu::turboquant::gqa_attention_flash_prefill_tq_capped_windowed(
                        q_buf,
                        k_packed,
                        k_scales,
                        v_packed,
                        v_scales,
                        *bits,
                        attn_out,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_len_base,
                        n_new,
                        window,
                        softcap,
                    );
                }
            }
            _ => unreachable!("forward_prefill_batched_tq entered with non-TQ KV"),
        }
    }
}

/// Q8_0 KV: quantize each K/V row via `quantize_row_q8_0` into i8 slabs +
/// per-row f32 scale, then the batched Q8_0 flash-prefill (GPU→CPU).
struct Q8_0Kv;
impl BatchedKv for Q8_0Kv {
    fn write_and_attend(
        &self,
        kv: &mut KvCache,
        layer_idx: usize,
        q_buf: &mut [f32],
        k_buf: &mut [f32],
        v_buf: &mut [f32],
        attn_out: &mut [f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_len_base: usize,
        n_new: usize,
        window: usize,
        softcap: f32,
        tree: Option<&TreeAttn<'_>>,
    ) {
        debug_assert!(
            tree.is_none(),
            "Q8_0 batched prefill has no tree-verify path"
        );
        let d_kv = n_kv_heads * head_dim;
        let kv_layer = &mut kv.layers[layer_idx];
        match kv_layer {
            KvLayer::Q8_0 {
                k_q,
                k_scales,
                v_q,
                v_scales,
            } => {
                for i in 0..n_new {
                    let pos_i = kv_len_base + i;
                    for h in 0..n_kv_heads {
                        let row_idx = h * max_ctx + pos_i;
                        let dst = row_idx * head_dim;
                        let k_src = &k_buf[i * d_kv + h * head_dim..i * d_kv + (h + 1) * head_dim];
                        k_scales[row_idx] = quantize_row_q8_0(k_src, &mut k_q[dst..dst + head_dim]);
                        let v_src = &v_buf[i * d_kv + h * head_dim..i * d_kv + (h + 1) * head_dim];
                        v_scales[row_idx] = quantize_row_q8_0(v_src, &mut v_q[dst..dst + head_dim]);
                    }
                }
                if !crate::accel::try_flash_attn_prefill_gpu_q8_0(
                    q_buf,
                    k_q,
                    k_scales,
                    v_q,
                    v_scales,
                    attn_out,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_ctx,
                    kv_len_base,
                    n_new,
                    window as u32,
                    softcap,
                ) {
                    k::gqa_attention_flash_prefill_q8_0_capped_windowed(
                        q_buf,
                        k_q,
                        k_scales,
                        v_q,
                        v_scales,
                        attn_out,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_len_base,
                        n_new,
                        window,
                        softcap,
                    );
                }
            }
            _ => unreachable!("forward_prefill_batched_q8_0 entered with non-Q8_0 KV"),
        }
    }
}

/// NVFP4 KV: quantize each K/V row per 16-elem block via `nvfp4::quantize_block`
/// into packed slabs, then the batched NVFP4 flash-prefill (GPU→CPU).
struct Nvfp4Kv;
impl BatchedKv for Nvfp4Kv {
    fn write_and_attend(
        &self,
        kv: &mut KvCache,
        layer_idx: usize,
        q_buf: &mut [f32],
        k_buf: &mut [f32],
        v_buf: &mut [f32],
        attn_out: &mut [f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_len_base: usize,
        n_new: usize,
        window: usize,
        softcap: f32,
        tree: Option<&TreeAttn<'_>>,
    ) {
        debug_assert!(
            tree.is_none(),
            "NVFP4 batched prefill has no tree-verify path"
        );
        let d_kv = n_kv_heads * head_dim;
        let kv_layer = &mut kv.layers[layer_idx];
        match kv_layer {
            KvLayer::Nvfp4 { k_packed, v_packed } => {
                use rustllama_kernels_cpu::nvfp4::{
                    quantize_block, NVFP4_BLOCK_BYTES, NVFP4_BLOCK_ELEMS,
                };
                let blocks_per_row = head_dim / NVFP4_BLOCK_ELEMS;
                let bytes_per_row = blocks_per_row * NVFP4_BLOCK_BYTES;
                for i in 0..n_new {
                    let pos_i = kv_len_base + i;
                    for h in 0..n_kv_heads {
                        let row_idx = h * max_ctx + pos_i;
                        let p_dst = row_idx * bytes_per_row;
                        for b in 0..blocks_per_row {
                            let elem_off = i * d_kv + h * head_dim + b * NVFP4_BLOCK_ELEMS;
                            let blk_dst = p_dst + b * NVFP4_BLOCK_BYTES;
                            quantize_block(
                                &k_buf[elem_off..elem_off + NVFP4_BLOCK_ELEMS],
                                &mut k_packed[blk_dst..blk_dst + NVFP4_BLOCK_BYTES],
                            );
                            quantize_block(
                                &v_buf[elem_off..elem_off + NVFP4_BLOCK_ELEMS],
                                &mut v_packed[blk_dst..blk_dst + NVFP4_BLOCK_BYTES],
                            );
                        }
                    }
                }
                if !crate::accel::try_flash_attn_prefill_gpu_nvfp4(
                    q_buf,
                    k_packed,
                    v_packed,
                    attn_out,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_ctx,
                    kv_len_base,
                    n_new,
                    window as u32,
                    softcap,
                ) {
                    rustllama_kernels_cpu::nvfp4::gqa_attention_flash_prefill_nvfp4_capped_windowed(
                        q_buf,
                        k_packed,
                        v_packed,
                        attn_out,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_len_base,
                        n_new,
                        window,
                        softcap,
                    );
                }
            }
            _ => unreachable!("forward_prefill_batched_nvfp4 entered with non-NVFP4 KV"),
        }
    }
}

/// MXFP4/6/8 KV: `fmt` selects the per-format block bytes + kernels; the KV
/// write + batched flash prefill both route through the shared
/// `mxfp_kv_prefill` helper (same one the contiguous MXFP prefill uses).
struct MxfpKv {
    fmt: KvDtype,
}
impl BatchedKv for MxfpKv {
    fn write_and_attend(
        &self,
        kv: &mut KvCache,
        layer_idx: usize,
        q_buf: &mut [f32],
        k_buf: &mut [f32],
        v_buf: &mut [f32],
        attn_out: &mut [f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_len_base: usize,
        n_new: usize,
        window: usize,
        softcap: f32,
        tree: Option<&TreeAttn<'_>>,
    ) {
        debug_assert!(
            tree.is_none(),
            "MXFP batched prefill has no tree-verify path"
        );
        let kv_layer = &mut kv.layers[layer_idx];
        match kv_layer {
            KvLayer::Mxfp4 { k_packed, v_packed }
            | KvLayer::Mxfp6 { k_packed, v_packed }
            | KvLayer::Mxfp8 { k_packed, v_packed } => {
                mxfp_kv_prefill(
                    self.fmt,
                    q_buf,
                    k_buf,
                    v_buf,
                    k_packed,
                    v_packed,
                    attn_out,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_ctx,
                    kv_len_base,
                    n_new,
                    window as u32,
                    softcap,
                );
            }
            _ => unreachable!("forward_prefill_batched_mxfp entered with non-MXFP KV"),
        }
    }
}

/// Q4_0 KV: ggml `block_q4_0` 18-byte blocks. Whitens Q/K/V (and un-whitens
/// the attention output) when `head_dim % 64 == 0`, applies the KV-bias
/// calibration observe + K-bias subtract, quantizes, then the batched Q4_0
/// flash-prefill (GPU→CPU).
struct Q4_0Kv;
impl BatchedKv for Q4_0Kv {
    fn write_and_attend(
        &self,
        kv: &mut KvCache,
        layer_idx: usize,
        q_buf: &mut [f32],
        k_buf: &mut [f32],
        v_buf: &mut [f32],
        attn_out: &mut [f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_len_base: usize,
        n_new: usize,
        window: usize,
        softcap: f32,
        tree: Option<&TreeAttn<'_>>,
    ) {
        debug_assert!(
            tree.is_none(),
            "Q4_0 batched prefill has no tree-verify path"
        );
        let d_kv = n_kv_heads * head_dim;
        let kv_bias_l = kv.kv_bias.clone();
        let kv_layer = &mut kv.layers[layer_idx];
        match kv_layer {
            KvLayer::Q4_0 { k_q, v_q } => {
                use rustllama_kernels_cpu::hadamard::whiten_chunks_inplace;
                use rustllama_kernels_cpu::q4_0_kv;
                let blocks_per_row = head_dim / q4_0_kv::Q4_0_BLOCK_ELEMS;
                let bytes_per_row = blocks_per_row * q4_0_kv::Q4_0_BLOCK_BYTES;
                // Whitening (fork attn_rot parity). Whole-buffer calls are
                // safe: rows are contiguous multiples of head_dim and
                // head_dim % 64 == 0, so no chunk crosses a head or token
                // boundary. Output is un-rotated after the batched attention.
                let whiten = q4_0_whiten_active(head_dim);
                if whiten {
                    whiten_chunks_inplace(q_buf, KV_WHITEN_CHUNK);
                    whiten_chunks_inplace(k_buf, KV_WHITEN_CHUNK);
                    whiten_chunks_inplace(v_buf, KV_WHITEN_CHUNK);
                }
                // Calibration observe + K bias subtract, per new row
                // (softmax-invariant; see kv_bias docs).
                if crate::kv_bias::calib_active() {
                    for i in 0..n_new {
                        crate::kv_bias::calib_observe(layer_idx, &k_buf[i * d_kv..(i + 1) * d_kv]);
                    }
                }
                if let Some(b) = kv_bias_l.as_ref().and_then(|b| b.layer(layer_idx)) {
                    for i in 0..n_new {
                        for (x, bb) in k_buf[i * d_kv..(i + 1) * d_kv].iter_mut().zip(b) {
                            *x -= *bb;
                        }
                    }
                }
                for i in 0..n_new {
                    let pos_i = kv_len_base + i;
                    for h in 0..n_kv_heads {
                        let row_idx = h * max_ctx + pos_i;
                        let p_dst = row_idx * bytes_per_row;
                        let elem_off = i * d_kv + h * head_dim;
                        q4_0_kv::quantize_row(
                            &k_buf[elem_off..elem_off + head_dim],
                            &mut k_q[p_dst..p_dst + bytes_per_row],
                        );
                        q4_0_kv::quantize_row(
                            &v_buf[elem_off..elem_off + head_dim],
                            &mut v_q[p_dst..p_dst + bytes_per_row],
                        );
                    }
                }
                if !crate::accel::try_flash_attn_prefill_gpu_q4_0(
                    q_buf,
                    k_q,
                    v_q,
                    attn_out,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_ctx,
                    kv_len_base,
                    n_new,
                    window as u32,
                    softcap,
                ) {
                    q4_0_kv::gqa_attention_flash_prefill_q4_0_capped_windowed(
                        q_buf,
                        k_q,
                        v_q,
                        attn_out,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_len_base,
                        n_new,
                        window,
                        softcap,
                    );
                }
                if whiten {
                    whiten_chunks_inplace(attn_out, KV_WHITEN_CHUNK);
                }
            }
            _ => unreachable!("forward_prefill_batched_q4_0 entered with non-Q4_0 KV"),
        }
    }
}

/// Half-precision KV (f16 / bf16): write each new K/V row as half bit
/// patterns, then dequantize the live `[0, kv_len_base + n_new)` region
/// to a COMPACT f32 scratch and run the existing F32 flash-prefill
/// (GPU→CPU) with the compact `kv_len` as the stride. Byte-for-byte the
/// F32 batched path over f16/bf16-rounded K/V. `window`/`softcap` are
/// threaded exactly like `F32Kv` (they're always 0/0.0 here because the
/// engine coerces SWA / soft-cap models to F32 KV), and tree-verify is
/// unsupported (half KV is never the tree-spec cache, which is F32).
struct HalfKv {
    bf16: bool,
}
impl BatchedKv for HalfKv {
    fn write_and_attend(
        &self,
        kv: &mut KvCache,
        layer_idx: usize,
        q_buf: &mut [f32],
        k_buf: &mut [f32],
        v_buf: &mut [f32],
        attn_out: &mut [f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_len_base: usize,
        n_new: usize,
        window: usize,
        softcap: f32,
        tree: Option<&TreeAttn<'_>>,
    ) {
        debug_assert!(
            tree.is_none(),
            "Half batched prefill has no tree-verify path"
        );
        let d_kv = n_kv_heads * head_dim;
        let kv_layer = &mut kv.layers[layer_idx];
        match kv_layer {
            KvLayer::Half { bf16, k, v } => {
                debug_assert_eq!(*bf16, self.bf16);
                // 1. Write the new rows as half bit patterns.
                for i in 0..n_new {
                    let pos_i = kv_len_base + i;
                    for h in 0..n_kv_heads {
                        let dst = (h * max_ctx + pos_i) * head_dim;
                        let src = i * d_kv + h * head_dim;
                        half_encode_row(
                            *bf16,
                            &k_buf[src..src + head_dim],
                            &mut k[dst..dst + head_dim],
                        );
                        half_encode_row(
                            *bf16,
                            &v_buf[src..src + head_dim],
                            &mut v[dst..dst + head_dim],
                        );
                    }
                }
                // 2. Dequant the live region to compact f32 scratch and
                //    run the F32 flash-prefill over it (stride = kv_len).
                let kv_len = kv_len_base + n_new;
                let mut k_f32 = vec![0f32; n_kv_heads * kv_len * head_dim];
                let mut v_f32 = vec![0f32; n_kv_heads * kv_len * head_dim];
                half_dequant_region(
                    *bf16, k, v, n_kv_heads, max_ctx, head_dim, kv_len, &mut k_f32, &mut v_f32,
                );
                let used_gpu = crate::accel::try_flash_attn_prefill_gpu_f32(
                    q_buf,
                    &k_f32,
                    &v_f32,
                    attn_out,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    /* max_ctx (compact stride) */ kv_len,
                    kv_len_base,
                    n_new,
                    window as u32,
                    softcap,
                );
                if !used_gpu {
                    k::gqa_attention_flash_prefill_capped_windowed(
                        q_buf,
                        &k_f32,
                        &v_f32,
                        attn_out,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        kv_len,
                        kv_len_base,
                        n_new,
                        window,
                        softcap,
                    );
                }
            }
            _ => unreachable!("forward_prefill_batched_half entered with non-Half KV"),
        }
    }
}

impl LlamaModel {
    /// Enumerate the model's CPU-backed weight ranges in page-lock
    /// priority order (tier 0 first). The Windows page-lock path
    /// ([`rustllama_engine`]'s `pagelock`) walks this, dedups by
    /// backing address, and `VirtualLock`s within a RAM budget. Only
    /// CPU-backed storage is returned (`SyclUsm` tensors are pinned
    /// separately at USM-alloc time); per-expert `CpuOwnedSlice` views
    /// resolve to their shared parent so each parent locks once.
    pub fn collect_lock_targets(&self) -> Vec<LockTarget> {
        let w = &self.weights;
        let mut out: Vec<LockTarget> = Vec::new();

        let push_tensor = |out: &mut Vec<LockTarget>, tier: u8, name: &str, t: &Tensor| {
            if let Some((ptr, len)) = t.storage.cpu_backing_ptr_len() {
                out.push(LockTarget {
                    tier,
                    name: name.to_string(),
                    addr: ptr as usize,
                    len,
                });
            }
        };
        let push_vec = |out: &mut Vec<LockTarget>, tier: u8, name: &str, v: &[f32]| {
            if !v.is_empty() {
                out.push(LockTarget {
                    tier,
                    name: name.to_string(),
                    addr: v.as_ptr() as usize,
                    len: std::mem::size_of_val(v),
                });
            }
        };

        // --- Tier 0: model-level always-hot tensors ---
        push_tensor(&mut out, 0, "token_embd", &w.token_embd);
        if let Some(o) = &w.output {
            push_tensor(&mut out, 0, "output", o);
        }
        push_vec(&mut out, 0, "output_norm", &w.output_norm);

        // Dense blocks: attention + norms + dense FFN are all Tier 0.
        for (i, b) in w.blocks.iter().enumerate() {
            push_vec(&mut out, 0, &format!("blk{i}.attn_norm"), &b.attn_norm);
            push_vec(&mut out, 0, &format!("blk{i}.ffn_norm"), &b.ffn_norm);
            if let Some(v) = &b.b_q {
                push_vec(&mut out, 0, &format!("blk{i}.b_q"), v);
            }
            if let Some(v) = &b.b_k {
                push_vec(&mut out, 0, &format!("blk{i}.b_k"), v);
            }
            if let Some(v) = &b.b_v {
                push_vec(&mut out, 0, &format!("blk{i}.b_v"), v);
            }
            if let Some(t) = &b.w_qkv_fused {
                push_tensor(&mut out, 0, &format!("blk{i}.w_qkv_fused"), t);
            } else {
                push_tensor(&mut out, 0, &format!("blk{i}.w_q"), &b.w_q);
                push_tensor(&mut out, 0, &format!("blk{i}.w_k"), &b.w_k);
                push_tensor(&mut out, 0, &format!("blk{i}.w_v"), &b.w_v);
            }
            push_tensor(&mut out, 0, &format!("blk{i}.w_o"), &b.w_o);
            push_tensor(&mut out, 0, &format!("blk{i}.w_gate"), &b.w_gate);
            push_tensor(&mut out, 0, &format!("blk{i}.w_up"), &b.w_up);
            push_tensor(&mut out, 0, &format!("blk{i}.w_down"), &b.w_down);
        }

        // Shared closure for the MoE FFN tiers (shared experts = Tier 1,
        // routed-expert parents = Tier 2; per-expert views are skipped —
        // they alias the parents already pushed at Tier 2).
        let push_moe_ffn = |out: &mut Vec<LockTarget>,
                            pfx: &str,
                            router: &Tensor,
                            gate_exps: &Tensor,
                            up_exps: &Tensor,
                            down_exps: &Tensor,
                            gate_shared: &Option<Tensor>,
                            up_shared: &Option<Tensor>,
                            down_shared: &Option<Tensor>| {
            push_tensor(out, 0, &format!("{pfx}.router"), router);
            if let Some(t) = gate_shared {
                push_tensor(out, 1, &format!("{pfx}.w_gate_shared"), t);
            }
            if let Some(t) = up_shared {
                push_tensor(out, 1, &format!("{pfx}.w_up_shared"), t);
            }
            if let Some(t) = down_shared {
                push_tensor(out, 1, &format!("{pfx}.w_down_shared"), t);
            }
            push_tensor(out, 2, &format!("{pfx}.w_gate_exps"), gate_exps);
            push_tensor(out, 2, &format!("{pfx}.w_up_exps"), up_exps);
            push_tensor(out, 2, &format!("{pfx}.w_down_exps"), down_exps);
        };

        // MoE blocks.
        if let Some(blocks) = &w.moe_blocks {
            for (i, b) in blocks.iter().enumerate() {
                push_vec(&mut out, 0, &format!("moe{i}.attn_norm"), &b.attn_norm);
                push_vec(&mut out, 0, &format!("moe{i}.ffn_norm"), &b.ffn_norm);
                if let Some(v) = &b.b_q {
                    push_vec(&mut out, 0, &format!("moe{i}.b_q"), v);
                }
                if let Some(v) = &b.b_k {
                    push_vec(&mut out, 0, &format!("moe{i}.b_k"), v);
                }
                if let Some(v) = &b.b_v {
                    push_vec(&mut out, 0, &format!("moe{i}.b_v"), v);
                }
                push_tensor(&mut out, 0, &format!("moe{i}.w_q"), &b.w_q);
                push_tensor(&mut out, 0, &format!("moe{i}.w_k"), &b.w_k);
                push_tensor(&mut out, 0, &format!("moe{i}.w_v"), &b.w_v);
                push_tensor(&mut out, 0, &format!("moe{i}.w_o"), &b.w_o);
                push_moe_ffn(
                    &mut out,
                    &format!("moe{i}"),
                    &b.router,
                    &b.w_gate_exps,
                    &b.w_up_exps,
                    &b.w_down_exps,
                    &b.w_gate_shared,
                    &b.w_up_shared,
                    &b.w_down_shared,
                );
            }
        }

        // Hybrid layers (full-attention + SSM variants).
        if let Some(layers) = &w.hybrid_layers {
            for (i, layer) in layers.iter().enumerate() {
                match layer {
                    HybridLayer::FullAttention(b) => {
                        push_vec(&mut out, 0, &format!("hyb{i}.attn_norm"), &b.attn_norm);
                        push_vec(
                            &mut out,
                            0,
                            &format!("hyb{i}.post_attn_norm"),
                            &b.post_attention_norm,
                        );
                        if let Some(v) = &b.q_norm {
                            push_vec(&mut out, 0, &format!("hyb{i}.q_norm"), v);
                        }
                        if let Some(v) = &b.k_norm {
                            push_vec(&mut out, 0, &format!("hyb{i}.k_norm"), v);
                        }
                        push_tensor(&mut out, 0, &format!("hyb{i}.w_q"), &b.w_q);
                        push_tensor(&mut out, 0, &format!("hyb{i}.w_k"), &b.w_k);
                        push_tensor(&mut out, 0, &format!("hyb{i}.w_v"), &b.w_v);
                        push_tensor(&mut out, 0, &format!("hyb{i}.w_o"), &b.w_o);
                        match &b.ffn {
                            HybridFfn::Moe {
                                router,
                                w_gate_exps,
                                w_up_exps,
                                w_down_exps,
                                w_gate_shared,
                                w_up_shared,
                                w_down_shared,
                                shared_router,
                                ..
                            } => {
                                if let Some(t) = shared_router {
                                    push_tensor(&mut out, 0, &format!("hyb{i}.shared_router"), t);
                                }
                                push_moe_ffn(
                                    &mut out,
                                    &format!("hyb{i}"),
                                    router,
                                    w_gate_exps,
                                    w_up_exps,
                                    w_down_exps,
                                    w_gate_shared,
                                    w_up_shared,
                                    w_down_shared,
                                );
                            }
                            HybridFfn::Dense {
                                w_gate,
                                w_up,
                                w_down,
                            } => {
                                push_tensor(&mut out, 0, &format!("hyb{i}.ffn_gate"), w_gate);
                                push_tensor(&mut out, 0, &format!("hyb{i}.ffn_up"), w_up);
                                push_tensor(&mut out, 0, &format!("hyb{i}.ffn_down"), w_down);
                            }
                        }
                    }
                    HybridLayer::Ssm(b) => {
                        push_vec(&mut out, 0, &format!("ssm{i}.attn_norm"), &b.attn_norm);
                        push_vec(
                            &mut out,
                            0,
                            &format!("ssm{i}.post_attn_norm"),
                            &b.post_attention_norm,
                        );
                        push_vec(&mut out, 0, &format!("ssm{i}.ssm_norm"), &b.ssm_norm);
                        push_tensor(&mut out, 0, &format!("ssm{i}.attn_qkv"), &b.attn_qkv);
                        push_tensor(&mut out, 0, &format!("ssm{i}.attn_gate"), &b.attn_gate);
                        push_tensor(&mut out, 0, &format!("ssm{i}.ssm_conv1d"), &b.ssm_conv1d);
                        push_tensor(&mut out, 0, &format!("ssm{i}.ssm_alpha"), &b.ssm_alpha);
                        push_tensor(&mut out, 0, &format!("ssm{i}.ssm_beta"), &b.ssm_beta);
                        push_tensor(&mut out, 0, &format!("ssm{i}.ssm_out"), &b.ssm_out);
                        match &b.ffn {
                            HybridFfn::Moe {
                                router,
                                w_gate_exps,
                                w_up_exps,
                                w_down_exps,
                                w_gate_shared,
                                w_up_shared,
                                w_down_shared,
                                shared_router,
                                ..
                            } => {
                                if let Some(t) = shared_router {
                                    push_tensor(&mut out, 0, &format!("ssm{i}.shared_router"), t);
                                }
                                push_moe_ffn(
                                    &mut out,
                                    &format!("ssm{i}"),
                                    router,
                                    w_gate_exps,
                                    w_up_exps,
                                    w_down_exps,
                                    w_gate_shared,
                                    w_up_shared,
                                    w_down_shared,
                                );
                            }
                            HybridFfn::Dense {
                                w_gate,
                                w_up,
                                w_down,
                            } => {
                                push_tensor(&mut out, 0, &format!("ssm{i}.ffn_gate"), w_gate);
                                push_tensor(&mut out, 0, &format!("ssm{i}.ffn_up"), w_up);
                                push_tensor(&mut out, 0, &format!("ssm{i}.ffn_down"), w_down);
                            }
                        }
                    }
                }
            }
        }

        // NextN / MTP heads (small, hot when speculative decode is on).
        if let Some(h) = &w.nextn_head {
            push_tensor(&mut out, 0, "nextn.eh_proj", &h.eh_proj);
            push_vec(&mut out, 0, "nextn.embed_norm", &h.embed_norm);
            push_vec(&mut out, 0, "nextn.hidden_norm", &h.hidden_norm);
            push_vec(&mut out, 0, "nextn.shared_head_norm", &h.shared_head_norm);
        }
        if let Some(heads) = &w.mtp_heads {
            for (i, h) in heads.iter().enumerate() {
                push_vec(
                    &mut out,
                    0,
                    &format!("mtp{i}.attn_norm"),
                    &h.block.attn_norm,
                );
                push_tensor(&mut out, 0, &format!("mtp{i}.w_q"), &h.block.w_q);
                push_tensor(&mut out, 0, &format!("mtp{i}.w_k"), &h.block.w_k);
                push_tensor(&mut out, 0, &format!("mtp{i}.w_v"), &h.block.w_v);
                push_tensor(&mut out, 0, &format!("mtp{i}.w_o"), &h.block.w_o);
                if let Some(t) = &h.lm_head {
                    push_tensor(&mut out, 0, &format!("mtp{i}.lm_head"), t);
                }
            }
        }

        out
    }

    /// Assemble a model from its parts, precomputing the scaled RoPE table +
    /// YaRN mscale ONCE here so every construction path (GGUF load, allow-moe,
    /// synth, and the safetensors AWQ/GPTQ/MLX builders in `rustllama-safetensors`)
    /// gets consistent tables and the per-token forward never recomputes them.
    /// Public so the out-of-crate safetensors loaders can build through it rather
    /// than the struct literal (which would silently skip the table precompute).
    pub fn from_parts(cfg: LlamaConfig, weights: LlamaWeights) -> Self {
        let (rope_inv_freq, yarn_mscale) = rope_tables_from_cfg(&cfg);
        Self {
            cfg,
            weights,
            rope_inv_freq,
            yarn_mscale,
        }
    }

    pub fn load(gguf: &Gguf) -> Result<Self, LlamaLoadError> {
        let cfg = LlamaConfig::from_gguf(gguf)?;
        let weights = LlamaWeights::from_gguf(gguf, &cfg)?;
        // MoE: phase 2-B-2 wired `forward_one` through `moe::moe_ffn_one`
        // and phase 2-C routed `forward_prefill` around the dense-only
        // batched paths (MoE falls back to the serial `forward_one`
        // loop, which already supports MoE). Real Qwen3-MoE /
        // Mixtral / DeepSeek-V3 GGUFs now load and generate through
        // the regular engine boundary. Performance: scalar per-expert
        // matvec, no batched MoE prefill — phase 2-D adds those.
        Ok(Self::from_parts(cfg, weights))
    }

    /// Like [`Self::load`], but does NOT reject MoE architectures.
    /// Phase 2-B-2 wires MoE through `forward_one` (decode-mode);
    /// prefill MoE lands in phase 2-C. Until then the engine
    /// boundary uses `load` (which rejects) because the chat path
    /// needs both prefill + decode — but this bypass lets the
    /// MoE-forward tests construct a `LlamaModel` to drive
    /// `forward_one` directly.
    ///
    /// Marked `#[doc(hidden)]` so it doesn't show in user-facing
    /// docs but stays callable from integration tests + future
    /// MoE-specific engines.
    #[doc(hidden)]
    pub fn load_allow_moe(gguf: &Gguf) -> Result<Self, LlamaLoadError> {
        let cfg = LlamaConfig::from_gguf(gguf)?;
        let weights = LlamaWeights::from_gguf(gguf, &cfg)?;
        Ok(Self::from_parts(cfg, weights))
    }

    /// P11 Phase 1 — load a base GGUF and merge one LoRA adapter GGUF at load
    /// (`W += s·B·A` for F32/F16 base tensors; quantized bases stay packed for
    /// the Phase-2 delta path). Returns the merged model plus the merge tally.
    pub fn load_with_lora(
        base: &Gguf,
        lora: &Gguf,
    ) -> Result<(Self, crate::lora::MergeStats), crate::lora::LoraError> {
        let mut model = Self::load_allow_moe(base)
            .map_err(|e| crate::lora::LoraError::BaseLoad(e.to_string()))?;
        let adapter = crate::lora::LoraAdapter::from_gguf(lora, "default")?;
        let stats = model.weights.merge_lora(&adapter);
        Ok((model, stats))
    }

    /// P11 Phase 2/3 — run [`Self::forward_one`] with `adapter` selected for
    /// this call only (the thread's previous selection is restored on return,
    /// even on panic). `None` selects the un-adapted base, which is
    /// byte-identical to a plain `forward_one`.
    pub fn forward_one_with_lora(
        &self,
        token_id: i32,
        pos: u32,
        kv: &mut KvCache,
        logits_out: &mut [f32],
        adapter: Option<&std::sync::Arc<crate::lora::LoraAdapter>>,
    ) {
        let _scope = crate::lora::scope_active_lora(adapter.cloned());
        self.forward_one(token_id, pos, kv, logits_out);
    }

    /// P11 Phase 2/3 — [`Self::forward_prefill`] with a scoped adapter
    /// selection (see [`Self::forward_one_with_lora`]).
    pub fn forward_prefill_with_lora(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        adapter: Option<&std::sync::Arc<crate::lora::LoraAdapter>>,
    ) -> Vec<f32> {
        let _scope = crate::lora::scope_active_lora(adapter.cloned());
        self.forward_prefill(tokens, start_pos, kv)
    }

    /// Centralized RoPE application for one Q or K activation buffer
    /// (`n_heads × head_dim` contiguous, per-head NeoX split-halves layout).
    ///
    /// Reproduces the existing theta-based NeoX / interleaved dispatch
    /// **byte-identically** for models WITHOUT context-extension scaling (the
    /// common case — `rope_scaling` is `None` ⇒ the base theta path, and
    /// `yarn_mscale == 1.0` ⇒ no post-scale), so coherence on e.g. Qwen2.5 is
    /// unchanged. When the model carries `rope.scaling.*` (YaRN / NTK / linear)
    /// and this is the full-rotation NeoX path, it drives RoPE from the
    /// precomputed scaled inverse-frequency table instead of the scalar theta.
    ///
    /// The YaRN attention `mscale` multiplies the rotated vector for BOTH Q and
    /// K (matching the reference: each is scaled by `mscale`, so the `q·k`
    /// attention logit carries `mscale²` = the YaRN `1/t` temperature). Linear
    /// and NTK leave `mscale == 1.0`.
    ///
    /// Interleaved (GPT-J pairing) + scaling is a documented gap: interleaved
    /// stays on the scalar-theta kernel (no interleaved table variant); the
    /// `mscale` post-scale still applies. Models shipping YaRN use NeoX.
    #[inline]
    fn rope_inplace(&self, buf: &mut [f32], n_heads: usize, head_dim: usize, pos: u32) {
        self.rope_rotate_inplace(buf, n_heads, head_dim, pos);
        // YaRN temperature: scale the rotated vector (applied to BOTH Q and K,
        // so the q·k logit carries mscale² = the YaRN 1/t). `1.0` off YaRN ⇒
        // no-op, keeping non-scaled models byte-identical.
        if self.yarn_mscale != 1.0 {
            for x in buf.iter_mut() {
                *x *= self.yarn_mscale;
            }
        }
    }

    /// Rotation-only RoPE — the pure positional rotation WITHOUT the YaRN
    /// `mscale`. Used where mscale must not be (re)applied: CacheBlend's `delta`
    /// re-rope, where the stored K already carries mscale from its isolated
    /// prefill and the re-base only adds the positional offset (RoPE is
    /// angle-additive per frequency, so rotating by `delta` composes with the
    /// chunk's local rotation; the scalar mscale does not and must stay single).
    #[inline]
    fn rope_rotate_inplace(&self, buf: &mut [f32], n_heads: usize, head_dim: usize, pos: u32) {
        let cfg = &self.cfg;
        if rope_interleaved_enabled() {
            k::rope_inplace_interleaved(buf, n_heads, head_dim, pos, cfg.rope_theta);
        } else if cfg.rope_scaling.is_some() && self.rope_inv_freq.len() * 2 == head_dim {
            k::rope_inplace_neox_invfreq(buf, n_heads, head_dim, pos, &self.rope_inv_freq);
        } else {
            k::rope_inplace_neox(buf, n_heads, head_dim, pos, cfg.rope_theta);
        }
    }

    /// Apply model-level post-processing to raw LM-head logits before they feed
    /// sampling / acceptance: Cohere `logit_scale` (multiply) then Gemma2
    /// `final_logit_softcapping` (`c·tanh(l/c)`). No-op — and byte-identical —
    /// when the model declares neither (the common case). Elementwise, so it
    /// works on a single `[vocab]` row or a batched `[m·vocab]` buffer alike.
    /// Called at every LM-head output site that produces committed-sampling or
    /// speculation-target logits. (Note: Gemma2 ALSO needs attention-logit
    /// softcapping, a kernel change not yet wired — so this alone fully fixes
    /// Cohere but only partially Gemma2.)
    #[inline]
    pub(crate) fn finalize_logits(&self, logits: &mut [f32]) {
        if let Some(s) = self.cfg.logit_scale {
            if s != 1.0 {
                for l in logits.iter_mut() {
                    *l *= s;
                }
            }
        }
        if let Some(c) = self.cfg.final_logit_softcapping {
            if c != 0.0 {
                let inv = 1.0 / c;
                for l in logits.iter_mut() {
                    *l = c * (*l * inv).tanh();
                }
            }
        }
    }

    /// Eagerly upload every packed-quant weight tensor in the model
    /// to the per-thread USM weight cache. Non-packed weights
    /// (F32 / F16 / Q*_K dequant-to-F16 variants) are skipped.
    /// Returns `(uploaded, skipped, bytes)` — diagnostic only.
    ///
    /// Called from the engine after `Engine::load` to shift the
    /// first-prefill weight-upload cost to model-load time. On
    /// Iris Xe shared memory each upload is a memcpy + page-mapping
    /// setup; a 7B Q4_K_M model adds ~1-2 s to load but makes the
    /// first chat as fast as steady-state chats.
    ///
    /// Safe to call multiple times — already-cached weights are a
    /// no-op. Safe to call when USM is disabled — returns
    /// `(0, n_weights, 0)` and the engine continues on the lazy
    /// upload path.
    pub fn preload_packed_weights_to_usm(&self) -> (usize, usize, usize) {
        // Default: upload every layer's weights. Hybrid-placement
        // callers should use `preload_packed_weights_to_usm_with_cutoff`
        // to skip the layers that won't run on GPU.
        self.preload_packed_weights_to_usm_with_cutoff(u32::MAX)
    }

    /// Same as [`Self::preload_packed_weights_to_usm`] but skips
    /// transformer blocks at `layer_idx >= n_gpu_layers`. Used by
    /// the engine when `[inference].n_gpu_layers < model.n_layers`
    /// so VRAM isn't wasted on weights that'll never be touched by
    /// a GPU kernel.
    ///
    /// `token_embd` and `output` (LM head) are always uploaded —
    /// they sit outside the transformer-layer cutoff (matches the
    /// dispatch behavior in [`Self::forward_one`], where post-
    /// layer kernels follow the last layer's GPU/CPU decision).
    pub fn preload_packed_weights_to_usm_with_cutoff(
        &self,
        n_gpu_layers: u32,
    ) -> (usize, usize, usize) {
        use crate::accel::preload_packed_tensor_to_usm;
        let mut uploaded = 0usize;
        let mut skipped = 0usize;
        let mut total_bytes = 0usize;
        let tally = |t: &Tensor, uploaded: &mut usize, skipped: &mut usize, total: &mut usize| {
            let bytes_before = *total;
            if preload_packed_tensor_to_usm(t) {
                *uploaded += 1;
                // Tensor byte size approximation — only the row-bytes
                // portion is actually uploaded, but for diagnostics
                // the raw bytes give a useful order-of-magnitude.
                let n_bytes = t.shape.iter().product::<u64>() as usize;
                *total = bytes_before + n_bytes;
            } else {
                *skipped += 1;
            }
        };
        tally(
            &self.weights.token_embd,
            &mut uploaded,
            &mut skipped,
            &mut total_bytes,
        );
        if let Some(out) = &self.weights.output {
            tally(out, &mut uploaded, &mut skipped, &mut total_bytes);
        }
        for (layer_idx, block) in self.weights.blocks.iter().enumerate() {
            if (layer_idx as u32) >= n_gpu_layers {
                // CPU-resident layer per the placement cutoff —
                // skip every per-block weight. The 7 per-block
                // tensors get charged to `skipped` so the
                // diagnostic accounting stays balanced.
                skipped += 7;
                continue;
            }
            tally(&block.w_q, &mut uploaded, &mut skipped, &mut total_bytes);
            tally(&block.w_k, &mut uploaded, &mut skipped, &mut total_bytes);
            tally(&block.w_v, &mut uploaded, &mut skipped, &mut total_bytes);
            tally(&block.w_o, &mut uploaded, &mut skipped, &mut total_bytes);
            tally(&block.w_gate, &mut uploaded, &mut skipped, &mut total_bytes);
            tally(&block.w_up, &mut uploaded, &mut skipped, &mut total_bytes);
            tally(&block.w_down, &mut uploaded, &mut skipped, &mut total_bytes);
        }
        // MoE + hybrid models keep their per-layer weights in `moe_blocks` /
        // `hybrid_layers`, NOT `blocks` (which is empty for them). Without
        // these loops the warmup uploaded ZERO layer weights for a hybrid
        // model — every matvec then re-decoded packed bytes on CPU each token
        // (the <1 tok/s qwen35/Ornith regression). Mirrors `collect_lock_targets`.
        if let Some(moe) = self.weights.moe_blocks.as_ref() {
            for (layer_idx, b) in moe.iter().enumerate() {
                if (layer_idx as u32) >= n_gpu_layers {
                    skipped += 8;
                    continue;
                }
                for t in [
                    &b.w_q,
                    &b.w_k,
                    &b.w_v,
                    &b.w_o,
                    &b.router,
                    &b.w_gate_exps,
                    &b.w_up_exps,
                    &b.w_down_exps,
                ] {
                    tally(t, &mut uploaded, &mut skipped, &mut total_bytes);
                }
                for t in [&b.w_gate_shared, &b.w_up_shared, &b.w_down_shared]
                    .into_iter()
                    .flatten()
                {
                    tally(t, &mut uploaded, &mut skipped, &mut total_bytes);
                }
            }
        }
        if let Some(hyb) = self.weights.hybrid_layers.as_ref() {
            for (layer_idx, layer) in hyb.iter().enumerate() {
                if (layer_idx as u32) >= n_gpu_layers {
                    continue;
                }
                let (heads, ffn): (Vec<&Tensor>, &HybridFfn) = match layer {
                    HybridLayer::FullAttention(b) => (vec![&b.w_q, &b.w_k, &b.w_v, &b.w_o], &b.ffn),
                    HybridLayer::Ssm(b) => (
                        vec![
                            &b.attn_qkv,
                            &b.attn_gate,
                            &b.ssm_conv1d,
                            &b.ssm_alpha,
                            &b.ssm_beta,
                            &b.ssm_out,
                        ],
                        &b.ffn,
                    ),
                };
                for t in heads {
                    tally(t, &mut uploaded, &mut skipped, &mut total_bytes);
                }
                match ffn {
                    HybridFfn::Moe {
                        router,
                        w_gate_exps,
                        w_up_exps,
                        w_down_exps,
                        w_gate_shared,
                        w_up_shared,
                        w_down_shared,
                        ..
                    } => {
                        for t in [router, w_gate_exps, w_up_exps, w_down_exps] {
                            tally(t, &mut uploaded, &mut skipped, &mut total_bytes);
                        }
                        for t in [w_gate_shared, w_up_shared, w_down_shared]
                            .into_iter()
                            .flatten()
                        {
                            tally(t, &mut uploaded, &mut skipped, &mut total_bytes);
                        }
                    }
                    HybridFfn::Dense {
                        w_gate,
                        w_up,
                        w_down,
                    } => {
                        for t in [w_gate, w_up, w_down] {
                            tally(t, &mut uploaded, &mut skipped, &mut total_bytes);
                        }
                    }
                }
            }
        }
        (uploaded, skipped, total_bytes)
    }

    /// Storage-swapping variant of [`Self::preload_packed_weights_to_usm_with_cutoff`].
    /// Walks every eligible weight tensor and *replaces* its
    /// `Storage::CpuOwned` with `Storage::SyclUsm`, copying the bytes
    /// into a fresh USM allocation that the tensor now owns.
    ///
    /// The difference from `preload_packed_weights_to_usm_with_cutoff`:
    /// that variant pre-warms the engine's per-thread upload cache
    /// (`accel::USM_ATTN::packed_weight_cache`) keyed by host pointer.
    /// This one moves the source of truth — the tensor itself carries
    /// the USM pointer, so kernel call sites consult
    /// [`rustllama_tensor::Storage::sycl_usm_ptr`] and skip the cache
    /// lookup entirely.
    ///
    /// Returns `(swapped, skipped, total_bytes)` for diagnostics.
    /// Skips:
    ///   - Tensors above the `n_gpu_layers` cutoff
    ///   - Tensors whose dtype isn't supported by the GPU dispatch
    ///     paths (`as_bytes` is fine, but the kernel won't use the
    ///     USM pointer — so uploading wastes VRAM)
    ///   - Any tensor where `accel::upload_bytes_to_usm` returns None
    ///     (USM disabled, alloc failed, etc.)
    ///
    /// Idempotent: calling on a model whose weights are already
    /// `SyclUsm` is a no-op for those tensors (the swap function
    /// short-circuits on existing SyclUsm storage).
    pub fn preload_weights_to_usm_swap(&mut self, n_gpu_layers: u32) -> (usize, usize, usize) {
        let mut swapped = 0usize;
        let mut skipped = 0usize;
        let mut total_bytes = 0usize;
        // Plain function so each call site can borrow the counters
        // freshly — a `&mut FnMut` closure would hold mutable borrows
        // of the counters across the whole for-loop, blocking the
        // `skipped += 7` skip-row path.
        fn swap_one(
            t: &mut crate::Tensor,
            swapped: &mut usize,
            skipped: &mut usize,
            total_bytes: &mut usize,
        ) {
            if t.storage.is_sycl_usm() {
                *swapped += 1;
                *total_bytes += t.storage.len_bytes();
                return;
            }
            let bytes = rustllama_tensor::as_bytes(t);
            match crate::accel::upload_bytes_to_usm(bytes) {
                Some(new_storage) => {
                    t.storage = new_storage;
                    *swapped += 1;
                    *total_bytes += t.storage.len_bytes();
                }
                None => {
                    *skipped += 1;
                }
            }
        }
        swap_one(
            &mut self.weights.token_embd,
            &mut swapped,
            &mut skipped,
            &mut total_bytes,
        );
        if let Some(out) = self.weights.output.as_mut() {
            swap_one(out, &mut swapped, &mut skipped, &mut total_bytes);
        }
        for (layer_idx, block) in self.weights.blocks.iter_mut().enumerate() {
            if (layer_idx as u32) >= n_gpu_layers {
                skipped += 7;
                continue;
            }
            swap_one(&mut block.w_q, &mut swapped, &mut skipped, &mut total_bytes);
            swap_one(&mut block.w_k, &mut swapped, &mut skipped, &mut total_bytes);
            swap_one(&mut block.w_v, &mut swapped, &mut skipped, &mut total_bytes);
            swap_one(&mut block.w_o, &mut swapped, &mut skipped, &mut total_bytes);
            swap_one(
                &mut block.w_gate,
                &mut swapped,
                &mut skipped,
                &mut total_bytes,
            );
            swap_one(
                &mut block.w_up,
                &mut swapped,
                &mut skipped,
                &mut total_bytes,
            );
            swap_one(
                &mut block.w_down,
                &mut swapped,
                &mut skipped,
                &mut total_bytes,
            );
        }
        tracing::info!(
            swapped,
            skipped,
            mib = (total_bytes as f64) / (1024.0 * 1024.0),
            n_gpu_layers,
            "USM weight storage swap complete"
        );
        (swapped, skipped, total_bytes)
    }

    /// Convenience: swap every layer's weights (uses `u32::MAX` as
    /// the cutoff). Matches [`Self::preload_packed_weights_to_usm`]'s
    /// "default to all layers on GPU" shape.
    pub fn preload_weights_to_usm_swap_all(&mut self) -> (usize, usize, usize) {
        self.preload_weights_to_usm_swap(u32::MAX)
    }

    /// Forward-pass entry point for the **vision splice path**:
    /// run one transformer step starting from a pre-populated `[d_model]`
    /// embedding row instead of a token id. Used by the engine's
    /// VLM prefill path, where image-placeholder positions carry
    /// projected patch embeddings (from V-5's
    /// [`splice_image_embeddings`]) rather than text-token lookups.
    ///
    /// Equivalent to `forward_one(token_id, ...)` for the no-image
    /// case — pass the row that `embed_tokens(&[token_id])` would
    /// have produced and the output is bit-identical.
    pub fn forward_one_from_embed(
        &self,
        embed_row: &[f32],
        pos: u32,
        kv: &mut KvCache,
        logits_out: &mut [f32],
    ) {
        self.forward_one_from_embed_mrope(embed_row, pos, None, kv, logits_out);
    }

    /// M5-tail: [`forward_one_from_embed`] with an explicit multimodal-RoPE
    /// 3-D `[t, h, w]` position for THIS token. When `mrope_pos` is `Some`
    /// AND the model carries `rope.dimension_sections` (Qwen2-VL /
    /// Qwen2.5-VL), the attention RoPE uses the sectioned `(t,h,w)` rotation;
    /// the scalar `pos` still drives the KV-cache slot + causal window (the
    /// PHYSICAL token order — mRoPE compresses rotation positions, not cache
    /// slots). For a text-only / non-sectioned model, or `mrope_pos == None`,
    /// this is byte-identical to [`forward_one_from_embed`].
    pub fn forward_one_from_embed_mrope(
        &self,
        embed_row: &[f32],
        pos: u32,
        mrope_pos: Option<[u32; 3]>,
        kv: &mut KvCache,
        logits_out: &mut [f32],
    ) {
        // M1: MLA path (see `forward_one`). MLA archs (DeepSeek) are not the
        // mRoPE VLM family, so the sectioned 3-D position does not apply.
        if self.weights.is_mla() {
            self.forward_one_mla_impl(EmbedInput::Embed(embed_row), pos, kv, logits_out, true);
            return;
        }
        let cfg = &self.cfg;
        // Hybrid (attention + SSM/DeltaNet) models run through
        // `forward_one_hybrid`, not this dense path. Guard against a
        // mis-dispatch: routing a hybrid here would run ZERO SSM layers
        // and silently emit garbage, so fail loudly instead.
        assert!(
            !self.weights.is_hybrid(),
            "hybrid attention+SSM model (arch `{}`) reached the dense \
             forward path — hybrids must route through the hybrid forward \
             (`forward_one_hybrid` / `forward_prefill_hybrid_impl`). \
             This is a dispatch bug.",
            cfg.arch
        );
        assert_eq!(
            embed_row.len(),
            cfg.d_model,
            "embed_row must be d_model long"
        );
        assert_eq!(logits_out.len(), cfg.vocab_size);
        assert_eq!(kv.layers.len(), cfg.n_layers);
        assert!((pos as usize) < kv.max_ctx, "pos exceeds max_ctx");

        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = cfg.d_ff;
        let (n_experts, top_k) = cfg
            .moe
            .as_ref()
            .map(|m| (m.n_experts as usize, m.n_experts_used as usize))
            .unwrap_or((0, 0));
        crate::accel::with_forward_scratch(
            d,
            d_q,
            d_kv,
            d_ff,
            head_dim,
            cfg.rope_theta,
            n_experts,
            top_k,
            |scratch| {
                self.forward_one_with_scratch_inner(
                    EmbedInput::Embed(embed_row),
                    pos,
                    mrope_pos,
                    kv,
                    logits_out,
                    scratch,
                );
            },
        );
    }

    /// Serial prefill over a `[seq_len, d_model]` row-major embedding
    /// buffer (the splice output from V-5). Returns the final logit
    /// vector — the same shape `forward_prefill` returns.
    ///
    /// Token-by-token loop calling [`forward_one_from_embed`]. The
    /// batched / paged prefill variants don't currently accept
    /// pre-embedded inputs; image-bearing requests route through this
    /// serial path until a future batched-from-embed implementation
    /// lands. Per-token attention is the dominant cost on long-context
    /// images, but at v1 image token counts (~576 patches for LLaVA)
    /// the serial loop is acceptable.
    pub fn forward_prefill_from_embeds(
        &self,
        embeds: &[f32],
        start_pos: u32,
        kv: &mut KvCache,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        assert_eq!(
            embeds.len() % d,
            0,
            "embeds.len()={} not a multiple of d_model={}",
            embeds.len(),
            d
        );
        let n_new = embeds.len() / d;
        let mut logits = vec![0.0f32; cfg.vocab_size];
        for i in 0..n_new {
            let row = &embeds[i * d..(i + 1) * d];
            let pos = start_pos + i as u32;
            self.forward_one_from_embed(row, pos, kv, &mut logits);
        }
        logits
    }

    /// M5-tail: multimodal-RoPE serial prefill over a `[seq_len, d_model]`
    /// spliced embedding buffer (the Qwen2-VL / Qwen2.5-VL vision splice).
    ///
    /// Identical to [`forward_prefill_from_embeds`] except each token `i`
    /// also carries its 3-D `[t, h, w]` M-RoPE position (`mrope_positions[i]`,
    /// one per token, from
    /// [`VlmPrefillInputs::mrope_positions_for_sequence`]). The scalar
    /// cache/causal position stays the PHYSICAL `start_pos + i` (mRoPE
    /// compresses rotation positions, not cache slots); the 3-D triple drives
    /// only the attention RoPE, and only when the model carries
    /// `rope.dimension_sections` — so for a model without sections this is
    /// byte-identical to [`forward_prefill_from_embeds`].
    pub fn forward_prefill_from_embeds_mrope(
        &self,
        embeds: &[f32],
        start_pos: u32,
        mrope_positions: &[[u32; 3]],
        kv: &mut KvCache,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        assert_eq!(
            embeds.len() % d,
            0,
            "embeds.len()={} not a multiple of d_model={}",
            embeds.len(),
            d
        );
        let n_new = embeds.len() / d;
        assert_eq!(
            mrope_positions.len(),
            n_new,
            "mrope_positions ({}) must have one [t,h,w] per spliced token ({n_new})",
            mrope_positions.len()
        );
        let mut logits = vec![0.0f32; cfg.vocab_size];
        for i in 0..n_new {
            let row = &embeds[i * d..(i + 1) * d];
            let pos = start_pos + i as u32;
            self.forward_one_from_embed_mrope(row, pos, Some(mrope_positions[i]), kv, &mut logits);
        }
        logits
    }

    /// Look up the token-embedding-table rows for a batch of token ids.
    ///
    /// Returns a row-major `[ids.len(), d_model]` f32 buffer — the same
    /// shape and dtype the transformer's first layer consumes
    /// internally inside [`forward_one`] / [`forward_prefill`].
    /// Exposed as a public surface so the V-5 / V-6b-1 vision splice
    /// can produce input embeddings on the engine side:
    ///
    /// ```text
    ///   text_embeds = model.embed_tokens(&prompt_ids);
    ///   vlm_inputs  = prepare_vlm_inputs(&vision, &prompt_ids_u32,
    ///                                    image_token_id, &image_bytes)?;
    ///   spliced     = splice_image_embeddings(
    ///                     &text_embeds, &vlm_inputs.positions,
    ///                     &vlm_inputs.feature_slices(), d_model)?;
    ///   logits      = model.forward_prefill_from_embeds(&spliced, 0, kv);
    ///                                       // ^ V-6b-3b (next slice)
    /// ```
    ///
    /// Routes through the same `embed_lookup_tensor` kernel the
    /// per-token forward uses, so the output is bit-for-bit
    /// identical to a `forward_one` call's internal `hidden` for
    /// each id.
    pub fn embed_tokens(&self, token_ids: &[i32]) -> Vec<f32> {
        let d = self.cfg.d_model;
        let mut out = vec![0.0f32; token_ids.len() * d];
        // Tolerate the empty-input case explicitly — the kernel
        // shouldn't be called with `ids.len() == 0` because some
        // dispatch paths panic on a zero-length slice.
        if !token_ids.is_empty() {
            // GPU dispatch first: F16 embedding tables run through
            // the SYCL gather kernel. CPU fallback covers Q-tables
            // (where dequant happens inline) and any path where the
            // USM context isn't ready. The GPU path returns `false`
            // cleanly on every "not eligible" case.
            if !crate::accel::try_embedding_lookup_usm_f32(
                &self.weights.token_embd,
                token_ids,
                &mut out,
                d,
            ) {
                k::embed_lookup_tensor(&self.weights.token_embd, token_ids, &mut out, d);
            }
        }
        out
    }

    /// Forward pass on **one** token. Caller advances position via the KV cache.
    /// Returns the logit vector of length `vocab_size`.
    pub fn forward_one(&self, token_id: i32, pos: u32, kv: &mut KvCache, logits_out: &mut [f32]) {
        // M1: MLA models (DeepSeek-V2/V3) run the dedicated low-rank /
        // compressed-latent attention path. Branch before the generic
        // dense/MoE body (which would die on the absent Q/K/V tensors).
        if self.weights.is_mla() {
            self.forward_one_mla_impl(EmbedInput::Token(token_id), pos, kv, logits_out, true);
            return;
        }
        let cfg = &self.cfg;
        // Guard the dense forward against a mis-dispatched hybrid model
        // (mirrors `forward_one_from_embed`): hybrids run through
        // `forward_one_hybrid`; routed here they would run ZERO layers
        // and emit gibberish instead of failing loudly.
        assert!(
            !self.weights.is_hybrid(),
            "hybrid attention+SSM model (arch `{}`) reached the dense \
             forward path — hybrids must route through the hybrid forward \
             (`forward_one_hybrid` / `forward_prefill_hybrid_impl`). \
             This is a dispatch bug.",
            cfg.arch
        );
        assert_eq!(logits_out.len(), cfg.vocab_size);
        assert_eq!(kv.layers.len(), cfg.n_layers);
        // KV-cache geometry must match this model's config — a cache
        // built for a different model would corrupt the per-head KV
        // writes (indexed by `head_dim`/`n_kv_heads`). Replaces a former
        // tautological `assert_eq!(kv.max_ctx, kv.max_ctx)`.
        assert_eq!(
            kv.n_kv_heads, cfg.n_kv_heads,
            "kv cache n_kv_heads mismatch"
        );
        assert_eq!(kv.head_dim, cfg.head_dim, "kv cache head_dim mismatch");
        assert!((pos as usize) < kv.max_ctx, "pos exceeds max_ctx");

        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = cfg.d_ff;

        // Borrow per-thread reusable scratch. `prepare` zeros every
        // buffer and resizes if model dims changed (e.g., a new
        // model was loaded in this binary). Caches `rope_inv_freq`
        // keyed on (head_dim, rope_theta) so it doesn't recompute
        // every call — but still re-uses the buffer storage.
        //
        // Pass `n_experts`/`top_k` from the MoE config so the
        // scratch's MoE-only buffers are pre-sized; dense models
        // pass 0/0 so the MoE buffers stay empty (no memory cost).
        let (n_experts, top_k) = cfg
            .moe
            .as_ref()
            .map(|m| (m.n_experts as usize, m.n_experts_used as usize))
            .unwrap_or((0, 0));
        crate::accel::with_forward_scratch(
            d,
            d_q,
            d_kv,
            d_ff,
            head_dim,
            cfg.rope_theta,
            n_experts,
            top_k,
            |scratch| {
                self.forward_one_with_scratch_inner(
                    EmbedInput::Token(token_id),
                    pos,
                    None, // scalar-pos token forward — no multimodal RoPE
                    kv,
                    logits_out,
                    scratch,
                );
            },
        );
    }

    // ============================================================
    // M1 — Multi-head Latent Attention (DeepSeek-V2/V3)
    // ============================================================

    /// Single-token MLA forward with the ABSORBED decode (score against
    /// the compressed latent directly). Public entry used by `forward_one`
    /// for DeepSeek-V2/V3 (`deepseek2`) models.
    pub fn forward_one_mla(
        &self,
        token_id: i32,
        pos: u32,
        kv: &mut KvCache,
        logits_out: &mut [f32],
    ) {
        self.forward_one_mla_impl(EmbedInput::Token(token_id), pos, kv, logits_out, true);
    }

    /// Single-token MLA forward with the NAIVE decode (materialize per-head
    /// K/V from the stored latent via the real `attn_kv_b` matvec, then
    /// standard attention). Mathematically identical to
    /// [`Self::forward_one_mla`] — the parity anchor that proves the
    /// absorbed weight-fold is correct. Exposed for the parity test.
    #[doc(hidden)]
    pub fn forward_one_mla_naive(
        &self,
        token_id: i32,
        pos: u32,
        kv: &mut KvCache,
        logits_out: &mut [f32],
    ) {
        self.forward_one_mla_impl(EmbedInput::Token(token_id), pos, kv, logits_out, false);
    }

    /// The shared MLA forward body. `absorbed` picks the decode form:
    /// `true` folds `attn_kv_b` into the per-head Q/output projections and
    /// scores against the `[kv_lora_rank]` latent directly (the
    /// memory-saving path); `false` materializes per-head K/V from the
    /// latent (naive). Both store ONLY the compressed latent + the
    /// decoupled RoPE key per token (the MLA KV-memory win).
    fn forward_one_mla_impl(
        &self,
        input: EmbedInput<'_>,
        pos: u32,
        kv: &mut KvCache,
        logits_out: &mut [f32],
        absorbed: bool,
    ) {
        let cfg = &self.cfg;
        let mla = cfg
            .mla
            .as_ref()
            .expect("forward_one_mla_impl on a non-MLA model");
        let blocks = self
            .weights
            .mla_blocks
            .as_ref()
            .expect("forward_one_mla_impl: mla_blocks not loaded");
        assert_eq!(logits_out.len(), cfg.vocab_size);
        assert_eq!(blocks.len(), cfg.n_layers);

        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let qk_nope = mla.qk_nope_head_dim;
        let qk_rope = mla.qk_rope_head_dim;
        let qk = qk_nope + qk_rope; // combined per-head Q/K width
        let v_hd = mla.v_head_dim;
        let kvr = mla.kv_lora_rank;
        let q_lora = mla.q_lora_rank.unwrap_or(0);
        let d_ff = cfg.d_ff;
        let kv_b_per_head = qk_nope + v_hd; // attn_kv_b per-head output width
        let (n_experts, top_k) = cfg
            .moe
            .as_ref()
            .map(|m| (m.n_experts as usize, m.n_experts_used as usize))
            .unwrap_or((0, 0));

        let cur_pos = pos as usize;
        let kv_len = cur_pos + 1;
        {
            let mla_kv = kv
                .mla
                .as_ref()
                .expect("forward_one_mla_impl: MLA KV cache not allocated");
            assert!(cur_pos < mla_kv.max_ctx, "pos {cur_pos} exceeds max_ctx");
        }

        // Decoupled-RoPE inv-freq table for the `qk_rope` slice. DeepSeek
        // uses YaRN, so build the SCALED table (no-op / base table when the
        // model declares no scaling). The YaRN temperature folds into the
        // softmax scale below (mscale² over every score dim), matching the
        // reference — not a per-element q/k post-scale.
        let rope_tab = crate::accel::rope_inv_freq_table_scaled(
            qk_rope,
            cfg.rope_theta,
            cfg.rope_scaling.as_ref(),
        );
        let use_tab = rope_tab.len() * 2 == qk_rope;
        let mscale = self.yarn_mscale;
        let scale = cfg.attention_scale.unwrap_or(1.0 / (qk as f32).sqrt()) * mscale * mscale;

        // ---- embed ----
        let mut hidden = vec![0.0f32; d];
        match input {
            EmbedInput::Token(t) => {
                k::embed_lookup_tensor(&self.weights.token_embd, &[t], &mut hidden, d);
            }
            EmbedInput::Embed(row) => {
                assert_eq!(row.len(), d, "embed_row must be d_model long");
                hidden.copy_from_slice(row);
            }
        }

        // Reusable per-layer scratch.
        let mut h_norm = vec![0.0f32; d];
        let mut q_all = vec![0.0f32; n_heads * qk]; // per-head [nope | rope]
        let mut q_lat = vec![0.0f32; q_lora.max(1)];
        let mut q_lat_n = vec![0.0f32; q_lora.max(1)];
        let mut kv_a = vec![0.0f32; kvr + qk_rope];
        let mut latent_n = vec![0.0f32; kvr];
        let mut rope_pack = vec![0.0f32; qk_rope.max(1)];
        let mut attn_out = vec![0.0f32; n_heads * v_hd];
        let mut o_proj = vec![0.0f32; d];
        // FFN scratch.
        let mut gate_buf = vec![0.0f32; d_ff.max(1)];
        let mut up_buf = vec![0.0f32; d_ff.max(1)];
        let mut ff_buf = vec![0.0f32; d_ff.max(1)];
        let mut down_buf = vec![0.0f32; d];
        let mut ffn_out = vec![0.0f32; d];
        let mut expert_logits = vec![0.0f32; n_experts.max(1)];
        let mut picks: Vec<(usize, f32)> = Vec::new();
        let mut had_buf: Vec<f32> = Vec::new();

        for (layer_idx, block) in blocks.iter().enumerate() {
            // 1. Pre-attention RMSNorm.
            k::rmsnorm_f32_row(&hidden, &block.attn_norm, &mut h_norm, cfg.rms_eps);

            // 2. Q projection → per-head [nope | rope].
            if let (Some(w_q_a), Some(q_a_norm), Some(w_q_b)) =
                (&block.w_q_a, &block.q_a_norm, &block.w_q_b)
            {
                matvec_tensor_dispatch(w_q_a, &h_norm, &mut q_lat[..q_lora], q_lora, d);
                k::rmsnorm_f32_row(
                    &q_lat[..q_lora],
                    q_a_norm,
                    &mut q_lat_n[..q_lora],
                    cfg.rms_eps,
                );
                matvec_tensor_dispatch(w_q_b, &q_lat_n[..q_lora], &mut q_all, n_heads * qk, q_lora);
            } else {
                let w_q = block
                    .w_q
                    .as_ref()
                    .expect("MLA block has neither low-rank nor direct Q");
                matvec_tensor_dispatch(w_q, &h_norm, &mut q_all, n_heads * qk, d);
            }

            // 3. Decoupled RoPE on the `qk_rope` slice of each Q head.
            for h in 0..n_heads {
                let s = h * qk + qk_nope;
                rope_pack.copy_from_slice(&q_all[s..s + qk_rope]);
                self.mla_rope(&mut rope_pack, qk_rope, pos, use_tab, &rope_tab);
                q_all[s..s + qk_rope].copy_from_slice(&rope_pack);
            }

            // 4. KV down-projection → [latent | decoupled rope key].
            matvec_tensor_dispatch(&block.w_kv_a_mqa, &h_norm, &mut kv_a, kvr + qk_rope, d);
            // RMSNorm the compressed latent.
            k::rmsnorm_f32_row(&kv_a[..kvr], &block.kv_a_norm, &mut latent_n, cfg.rms_eps);
            // RoPE the single decoupled key (shared across heads).
            rope_pack.copy_from_slice(&kv_a[kvr..kvr + qk_rope]);
            self.mla_rope(&mut rope_pack, qk_rope, pos, use_tab, &rope_tab);

            // 5. Store ONLY the latent + decoupled rope key at `cur_pos`.
            {
                let mla_kv = kv.mla.as_mut().unwrap();
                mla_kv.latent[layer_idx][cur_pos * kvr..(cur_pos + 1) * kvr]
                    .copy_from_slice(&latent_n);
                mla_kv.rope_key[layer_idx][cur_pos * qk_rope..(cur_pos + 1) * qk_rope]
                    .copy_from_slice(&rope_pack);
            }

            // 6. Attention over the cached latents → attn_out [n_heads * v_hd].
            {
                let mla_kv = kv.mla.as_ref().unwrap();
                let latent = &mla_kv.latent[layer_idx];
                let rope_key = &mla_kv.rope_key[layer_idx];
                if absorbed {
                    mla_attend_absorbed(
                        &block.w_kv_b_f32,
                        &q_all,
                        latent,
                        rope_key,
                        &mut attn_out,
                        n_heads,
                        qk_nope,
                        qk_rope,
                        v_hd,
                        kvr,
                        kv_b_per_head,
                        kv_len,
                        scale,
                    );
                } else {
                    mla_attend_naive(
                        block,
                        &q_all,
                        latent,
                        rope_key,
                        &mut attn_out,
                        n_heads,
                        qk_nope,
                        qk_rope,
                        v_hd,
                        kvr,
                        kv_b_per_head,
                        kv_len,
                        scale,
                    );
                }
            }

            // 7. Output projection + residual.
            matvec_tensor_dispatch(&block.w_o, &attn_out, &mut o_proj, d, n_heads * v_hd);
            k::add_inplace_f32(&mut hidden, &o_proj);

            // 8. Post-attention RMSNorm → FFN → residual.
            k::rmsnorm_f32_row(&hidden, &block.ffn_norm, &mut h_norm, cfg.rms_eps);
            match &block.ffn {
                HybridFfn::Moe {
                    router,
                    gate_per_expert,
                    up_per_expert,
                    down_per_expert,
                    w_gate_shared,
                    w_up_shared,
                    w_down_shared,
                    shared_router,
                    exp_probs_b,
                    ..
                } => crate::moe::moe_ffn_one_into_parts(
                    &h_norm,
                    router,
                    &crate::moe::router_spec_for(cfg),
                    // M1-tail: DeepSeek-V3's routed selection bias now steers
                    // which experts win in the MLA MoE FFN (was dropped here).
                    exp_probs_b.as_deref(),
                    gate_per_expert,
                    up_per_expert,
                    down_per_expert,
                    w_gate_shared.as_ref(),
                    w_up_shared.as_ref(),
                    w_down_shared.as_ref(),
                    shared_router.as_ref(),
                    d,
                    d_ff,
                    n_experts,
                    top_k,
                    &mut ffn_out,
                    &mut gate_buf,
                    &mut up_buf,
                    &mut ff_buf,
                    &mut down_buf,
                    &mut expert_logits,
                    &mut picks,
                ),
                HybridFfn::Dense {
                    w_gate,
                    w_up,
                    w_down,
                } => dense_ffn_one_into(
                    w_gate,
                    w_up,
                    w_down,
                    &h_norm,
                    d,
                    d_ff,
                    &mut ffn_out,
                    &mut gate_buf,
                    &mut up_buf,
                    &mut ff_buf,
                    None,
                    &mut had_buf,
                ),
            }
            k::add_inplace_f32(&mut hidden, &ffn_out);
        }

        // ---- final norm + LM head ----
        let mut final_norm = vec![0.0f32; d];
        k::rmsnorm_f32_row(
            &hidden,
            &self.weights.output_norm,
            &mut final_norm,
            cfg.rms_eps,
        );
        let lm_head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        matvec_tensor_dispatch(lm_head, &final_norm, logits_out, cfg.vocab_size, d);
        self.finalize_logits(logits_out);

        // Advance the MLA sequence length (mirrors the KvCache bookkeeping
        // the dense path does via the engine).
        kv.seq_len = kv.seq_len.max(kv_len);
    }

    /// NeoX half-split RoPE on one `qk_rope`-wide vector (the decoupled
    /// rope key, or one Q head's rope slice). Uses the scaled inv-freq
    /// table when it matches `qk_rope`, else the scalar-theta kernel.
    #[inline]
    fn mla_rope(&self, buf: &mut [f32], qk_rope: usize, pos: u32, use_tab: bool, tab: &[f32]) {
        if use_tab {
            k::rope_inplace_neox_invfreq(buf, 1, qk_rope, pos, tab);
        } else {
            k::rope_inplace_neox(buf, 1, qk_rope, pos, self.cfg.rope_theta);
        }
    }

    /// Phase 3.7b: single-token forward for hybrid attention+DeltaNet
    /// models (`qwen35moe` family).
    ///
    /// **F32-only, CPU-only, decode-only.** This is the "execution
    /// harness" pass — it iterates `hybrid_layers`, dispatches each
    /// layer to the right kernel (DeltaNet for SSM layers, standard
    /// attention for full-attn layers), and routes the FFN through
    /// the MoE path. Output may be subtly wrong vs reference (Phase 8
    /// will fix that); the bar here is mechanical execution without
    /// panic.
    ///
    /// **Known divergences from reference (to fix in later phases):**
    /// - Full-attention layers use the standard RoPE rotation rather
    ///   than multi-section RoPE (`rope.dimension_sections` array).
    ///   Phase 5.
    /// - Full-attention layers compute attention against only the
    ///   current K/V (not the historical KV cache via paged or
    ///   contiguous backends). Single-position decode against a fresh
    ///   sequence works; multi-token attention with history doesn't.
    /// - The DeltaNet kernel uses `delta_net_layer_forward_f32` whose
    ///   correctness vs Python reference is not yet validated.
    /// - NextN/MTP head ignored (Phase 6).
    ///
    /// Panics if the model isn't hybrid — caller must check
    /// `weights.is_hybrid()` first.
    pub fn forward_one_hybrid(
        &self,
        token_id: i32,
        pos: u32,
        kv: &mut KvCache,
        dn_cache: &mut DeltaNetCache,
        logits_out: &mut [f32],
    ) {
        let hidden = self.forward_one_hybrid_to_hidden(token_id, pos, kv, dn_cache);
        // Final output norm + LM head.
        let cfg = &self.cfg;
        let d = cfg.d_model;
        assert_eq!(logits_out.len(), cfg.vocab_size);
        let mut final_norm = vec![0.0f32; d];
        k::rmsnorm_f32_row(
            &hidden,
            &self.weights.output_norm,
            &mut final_norm,
            cfg.rms_eps,
        );
        let lm_head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        {
            let mut had_buf: Vec<f32> = Vec::new();
            let x_in = hadamard_pre(
                self.weights.hadamard.as_ref(),
                lm_head,
                &final_norm,
                &mut had_buf,
            );
            matvec_tensor_dispatch(lm_head, x_in, logits_out, cfg.vocab_size, d);
        }
        // Final-state instrumentation: with RUSTLLAMA_DEBUG_HYBRID,
        // dump RMS stats of the post-block hidden, the post-norm
        // hidden, and the logits + top-5 argmax. Tells us at a
        // glance whether the bug is in the layers (hidden RMS
        // already wrong) or downstream (hidden OK but logits
        // degenerate).
        if debug_hybrid_enabled() {
            let rms = |x: &[f32]| -> f64 {
                (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
            };
            eprintln!(
                "  hybrid[final]    post-block: rms={:.3e} | post-norm: rms={:.3e}",
                rms(&hidden),
                rms(&final_norm)
            );
            // Top-5 logits (id, value).
            let mut top: Vec<(usize, f32)> = logits_out.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top5: Vec<String> = top
                .iter()
                .take(5)
                .map(|(i, v)| format!("({i}, {v:+.3})"))
                .collect();
            let lmax = logits_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lmin = logits_out.iter().cloned().fold(f32::INFINITY, f32::min);
            let any_nan = logits_out.iter().any(|v| v.is_nan());
            eprintln!(
                "  hybrid[final]    logits: rms={:.3e} min={:+.3e} max={:+.3e} nan={} top5={}",
                rms(logits_out),
                lmin,
                lmax,
                any_nan,
                top5.join(" ")
            );
        }
    }

    /// Inner-loop helper: run every hybrid layer and return the
    /// final post-block hidden state. Used by both
    /// [`Self::forward_one_hybrid`] (which applies output norm +
    /// LM head) and
    /// [`Self::forward_one_hybrid_with_nextn_logits`] (which uses
    /// the hidden for both the main LM head AND the NextN/MTP
    /// projection). Side effects: extends `kv` history and mutates
    /// `dn_cache` per SSM-layer state evolution.
    fn forward_one_hybrid_to_hidden(
        &self,
        token_id: i32,
        pos: u32,
        kv: &mut KvCache,
        dn_cache: &mut DeltaNetCache,
    ) -> Vec<f32> {
        assert!((pos as usize) < kv.max_ctx, "pos exceeds max_ctx");
        let cfg = &self.cfg;
        assert!(
            self.weights.is_hybrid(),
            "forward_one_hybrid called on non-hybrid model — engine dispatch bug"
        );
        let hyb = cfg.hybrid.as_ref().expect("hybrid cfg present");
        let layers = self
            .weights
            .hybrid_layers
            .as_ref()
            .expect("hybrid_layers present");
        assert_eq!(layers.len(), cfg.n_layers);
        assert_eq!(dn_cache.layers.len(), cfg.n_layers);
        assert_eq!(kv.layers.len(), cfg.n_layers);

        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let ssm_inner = hyb.ssm_inner_size as usize;
        let conv_kernel = hyb.ssm_conv_kernel as usize;
        // Derive head counts from tensor shapes of the first SSM layer
        // (mirrors DeltaNetCache::new_for_hybrid).
        let first_ssm = layers
            .iter()
            .find_map(|l| match l {
                HybridLayer::Ssm(s) => Some(s),
                _ => None,
            })
            .expect("at least one SSM layer required");
        let n_v_heads = first_ssm.ssm_a.len();
        let head_v_dim = ssm_inner / n_v_heads;
        let head_qk_dim = head_v_dim;
        // QK head count: `ssm.group_count` metadata when present
        // (Bonsai-2/Qwen3.8: 16 groups at d=5120, where the legacy
        // `d / head_qk_dim` derivation would wrongly give 40). The
        // legacy derivation stays as the fallback and yields the
        // SAME value on qwen35moe (2048/128 = 16 = group_count), so
        // Qwen3.6-family models are byte-identical either way.
        let n_qk_heads = if hyb.ssm_group_count > 1 {
            hyb.ssm_group_count as usize
        } else {
            d / head_qk_dim
        };

        // Per-expert FFN width: pure-MoE models declare it via
        // `expert_feed_forward_length`. cfg.d_ff already falls back
        // to that key (Phase 1 fallback).
        let d_ff = cfg.d_ff;
        // MoE geometry is optional: qwen35moe hybrids route experts,
        // Bonsai-2/dense-qwen35 hybrids run a plain SwiGLU (HybridFfn
        // decides per layer; (0, 0) is never read on the Dense arm).
        let (n_experts, top_k) = cfg
            .moe
            .as_ref()
            .map(|m| (m.n_experts as usize, m.n_experts_used as usize))
            .unwrap_or((0, 0));

        // Prism Hadamard rotation state + scratch (Bonsai-family
        // ternary models; None everywhere else — the hooks are
        // no-ops then).
        let had = self.weights.hadamard.as_ref();
        let mut had_buf: Vec<f32> = Vec::new();
        let mut had_perm_buf: Vec<f32> = Vec::new();

        // --- Embed lookup ---
        let mut hidden = vec![0.0f32; d];
        k::embed_lookup_tensor(&self.weights.token_embd, &[token_id], &mut hidden, d);
        // Rotated embedding table: restore the primal basis right
        // after the lookup (`h = s ⊙ (H z) / √block`).
        if let Some(h) = had {
            if h.embd_inverse && !no_hadamard_enabled() {
                k::hadamard::hadamard_inverse_inplace(&mut hidden, h.signs_for(d), h.block_size);
            }
        }
        if debug_hybrid_enabled() {
            let h_rms = (hidden.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
                / hidden.len() as f64)
                .sqrt();
            let h_first: Vec<f32> = hidden.iter().take(6).copied().collect();
            let tbl_dtype = format!("{:?}", self.weights.token_embd.dtype);
            let tbl_shape = format!("{:?}", self.weights.token_embd.shape);
            let tbl_bytes_slice = rustllama_tensor::as_bytes(&self.weights.token_embd);
            let tbl_bytes = tbl_bytes_slice.len();
            // First 16 bytes of THIS token's row in the table, to
            // rule out "the loader returned zero-filled bytes":
            const BLOCK_BYTES: usize = 74; // IQ2_XS super-block
            let row_bytes = (d / 256) * BLOCK_BYTES;
            let row_start = (token_id as usize) * row_bytes;
            let row_preview: Vec<u8> = tbl_bytes_slice
                .get(row_start..row_start + 16)
                .unwrap_or(&[])
                .to_vec();
            // Also independently dequant the first super-block via
            // the gguf dequant function to compare:
            let mut indep_dequant = vec![0.0f32; 256];
            if let Some(block) = tbl_bytes_slice.get(row_start..row_start + BLOCK_BYTES) {
                rustllama_gguf::dequant::dequant_iq2_xs(block, &mut indep_dequant);
            }
            let indep_rms = (indep_dequant
                .iter()
                .map(|v| (*v as f64).powi(2))
                .sum::<f64>()
                / indep_dequant.len() as f64)
                .sqrt();
            eprintln!(
                "  embed_lookup: token_id={token_id} d={d} table=({tbl_dtype}, shape={tbl_shape}, bytes={tbl_bytes}, row_bytes={row_bytes}, row_start={row_start}) → hidden_rms={h_rms:.3e} first6={h_first:?}"
            );
            eprintln!(
                "  embed_lookup: row_preview_bytes={row_preview:02x?} indep_dequant_rms={indep_rms:.3e} indep_first4={:?}",
                &indep_dequant[..4]
            );
            // Probe several row offsets to localize whether the table
            // is "mostly empty" (file/encoder bug) or my row-offset
            // formula is wrong (layout bug).
            for sample_id in [0u32, 1, 100, 5834, 100000, 200000, 248319].iter() {
                let off = (*sample_id as usize) * row_bytes;
                if let Some(blk) = tbl_bytes_slice.get(off..off + BLOCK_BYTES) {
                    let nonzero = blk.iter().filter(|b| **b != 0).count();
                    let first8: Vec<String> =
                        blk.iter().take(8).map(|b| format!("{b:02x}")).collect();
                    eprintln!(
                        "    sample id={sample_id} off={off} nonzero_bytes={nonzero}/74 first8={}",
                        first8.join(" ")
                    );
                }
            }
            // Also peek at well-known offsets: file start, mid, end.
            for label in ["start", "mid", "end-row"].iter() {
                let off = match *label {
                    "start" => 0usize,
                    "mid" => tbl_bytes / 2,
                    "end-row" => tbl_bytes - row_bytes,
                    _ => unreachable!(),
                };
                if let Some(blk) = tbl_bytes_slice.get(off..off + 8) {
                    let nonzero = blk.iter().filter(|b| **b != 0).count();
                    let first8: Vec<String> =
                        blk.iter().take(8).map(|b| format!("{b:02x}")).collect();
                    eprintln!(
                        "    file_{label} off={off} nonzero={nonzero}/8 first8={}",
                        first8.join(" ")
                    );
                }
            }
        }

        // Per-layer scratch (allocated once per call; not pooled —
        // Phase 3.7c follow-up if perf matters).
        let mut tmp_d = vec![0.0f32; d];
        let mut layer_out = vec![0.0f32; d];

        // FFN scratch.
        let mut gate_buf = vec![0.0f32; d_ff];
        let mut up_buf = vec![0.0f32; d_ff];
        let mut ff_buf = vec![0.0f32; d_ff];
        let mut down_buf = vec![0.0f32; d];
        let mut expert_logits = vec![0.0f32; n_experts];
        let mut picks: Vec<(usize, f32)> = Vec::with_capacity(top_k);

        // Attention scratch (used by full-attention layers only).
        let mut q_buf = vec![0.0f32; d_q];
        let mut k_buf = vec![0.0f32; d_kv];
        let mut v_buf = vec![0.0f32; d_kv];
        let mut attn_out = vec![0.0f32; d_q];
        let mut attn_proj = vec![0.0f32; d];

        // Hoisted per-layer / per-head scratch — these were allocated
        // INSIDE the layer (and per-V-head) loops, ~4,100 heap
        // allocations per decoded token. Mirrors the prefill twin's
        // hoisting.
        // Fused-QKV width: q + k (n_qk heads each) + v (ssm_inner).
        // On qwen35moe this equals 2·ssm_inner (the old hardcode);
        // on Bonsai-2/Qwen3.8 (16 qk-heads at d=5120) it does not.
        let qkv_dim = 2 * n_qk_heads * head_qk_dim + ssm_inner;
        let mut qkv_pre = vec![0.0f32; qkv_dim];
        let mut qkv_conv = vec![0.0f32; qkv_dim];
        let mut gate_path = vec![0.0f32; ssm_inner];
        let mut alpha = vec![0.0f32; n_v_heads];
        let mut beta = vec![0.0f32; n_v_heads];
        let mut v_out_concat = vec![0.0f32; ssm_inner];
        let mut qg_buf = vec![0.0f32; 2 * d_q];
        let mut attn_gate_buf = vec![0.0f32; d_q];
        let mut head_tmp = vec![0.0f32; head_dim];
        let mut q_head = vec![0.0f32; head_qk_dim];
        let mut k_head = vec![0.0f32; head_qk_dim];
        let mut out_head = vec![0.0f32; head_v_dim];
        let mut v_tilde = vec![0.0f32; head_v_dim];

        // **Bug-fix from the HF Qwen3_5MoE config audit**: the GGUF
        // declares `block_count = 41` for a model whose
        // `num_hidden_layers = 40` because layer 40 holds the NextN
        // head's reserved transformer block (the one whose
        // `blk.40.nextn.*` tensors are bound separately as
        // [`LlamaWeights::nextn_head`]). The MAIN forward must
        // iterate only the first `num_hidden_layers` layers;
        // running layer 40 as a regular layer doubles work on the
        // wrong state and was the source of the empirically-garbage
        // output observed in the end-to-end smoke test. Layer 40
        // gets consumed by `forward_one_hybrid_with_nextn_logits`
        // (Phase 6 follow-up that hasn't yet wired this skip).
        // The qwen35moe converter (`_Qwen35MtpMixin`) extends block_count
        // by `mtp_num_hidden_layers`, packing the MTP draft head as the
        // last block. The main forward MUST exclude those layers — the
        // MTP block's attn + MoE FFN is for speculative decoding only;
        // running it on the residual stream adds garbage to the final
        // hidden state. Default to `cfg.n_layers - nextn_predict_layers`.
        // RUSTLLAMA_INCLUDE_MTP_LAYERS=1 keeps the old "iterate all"
        // behavior for A/B testing.
        let n_mtp_block_layers = cfg
            .hybrid
            .as_ref()
            .map(|h| h.nextn_predict_layers as usize)
            .unwrap_or(0);
        let main_layer_count = if include_mtp_layers() {
            cfg.n_layers
        } else {
            cfg.n_layers.saturating_sub(n_mtp_block_layers)
        };
        // Partial-rotary RoPE dim: HF config carries
        // `partial_rotary_factor = 0.25` for qwen35moe — only the
        // first `rope_dim/2` pairs of each `head_dim`-sized head
        // get rotated; the trailing `(head_dim - rope_dim)` elements
        // pass through unchanged. For Qwen2.5 / Mistral / Llama
        // family `rope_dim == head_dim` so the partial reduces to
        // the full rotation the existing `rope_inplace_neox` does.
        let rope_dim = cfg.rope_dim;

        // RUSTLLAMA_DEBUG_HYBRID: per-layer hidden-state stats.
        // Localizes where output coherence breaks down.
        let debug_hybrid = debug_hybrid_enabled();
        let dump_acts_dir = dump_acts_dir();
        let log_hidden = |stage: &str, li: usize, h: &[f32]| {
            if let Some(dir) = dump_acts_dir.as_ref() {
                // Per-layer activation dump for HF-vs-Rust cosine diff.
                // Slug the stage so spaces don't break filenames.
                let stage_slug: String = stage
                    .trim()
                    .chars()
                    .map(|c| if c.is_alphanumeric() { c } else { '_' })
                    .collect();
                let path = format!("{dir}/layer_{li:02}_{stage_slug}.bin");
                if let Ok(mut f) = std::fs::File::create(&path) {
                    use std::io::Write;
                    // SAFETY: f32 has stable LE bytes on x86_64; converting
                    // &[f32] to &[u8] of size 4× len is sound for write.
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            h.as_ptr() as *const u8,
                            std::mem::size_of_val(h),
                        )
                    };
                    let _ = f.write_all(bytes);
                }
            }
            if !debug_hybrid {
                return;
            }
            let rms = (h.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / h.len() as f64).sqrt();
            let max = h.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = h.iter().cloned().fold(f32::INFINITY, f32::min);
            let any_nan = h.iter().any(|v| v.is_nan());
            let any_inf = h.iter().any(|v| v.is_infinite());
            eprintln!(
                "  hybrid[L{li:02}] {stage:>16}: rms={rms:9.3e} min={min:+9.2e} max={max:+9.2e} nan={any_nan} inf={any_inf}"
            );
        };
        if debug_hybrid {
            log_hidden("after embed", 0, &hidden);
        }
        for (li, layer) in layers.iter().take(main_layer_count).enumerate() {
            // Publish the layer index to the per-thread TLS slot that
            // `accel::gpu_active_for_current_layer()` (n_gpu_layers
            // cutoff), the expert-pin cache's (layer, expert) keys,
            // and the decode readahead all consult. Every dense/paged
            // forward does this; its absence here silently disabled
            // all three on hybrid models.
            crate::accel::set_current_layer_idx(li as u32);
            // NB: no layer-ahead readahead on the DECODE path. A
            // single decoded token spends little time per layer, so
            // prefetching L+1 gives negligible overlap while costing
            // a channel send + async PrefetchVirtualMemory syscall
            // per layer per token — measured net-NEGATIVE on a
            // resident model. Readahead lives on the PREFILL path
            // only, where a chunk of tokens spends real time per
            // layer and the prefetch genuinely overlaps compute.
            match layer {
                HybridLayer::Ssm(block) => {
                    // 1. Pre-DeltaNet RMSNorm.
                    k::rmsnorm_f32_row(&hidden, &block.attn_norm, &mut tmp_d, cfg.rms_eps);

                    // 2. Projections via matvec_tensor_dispatch (handles
                    //    every quantized dtype the kernel-cpu/sycl
                    //    paths cover). The delta_net kernel module's
                    //    `delta_net_layer_forward_f32` requires raw
                    //    f32 weights, so for quantized hybrid models
                    //    we inline the projection + kernel composition
                    //    here.
                    // H3: hybrid (Qwen3.5-MoE DeltaNet) path uses
                    // the fused `attn_qkv` tensor at load time, so
                    // QKV projection is already a single matvec
                    // dispatch — H3-compliant. Standard Llama-family
                    // models ship separate w_q/w_k/w_v tensors and
                    // do 3 dispatches; load-time concatenation into
                    // a fused tensor (memory-equivalent, dispatch
                    // 2/3 reduction) is the remaining H3 follow-up,
                    // tracked as load-path refactor.
                    {
                        let x_in = hadamard_pre(had, &block.attn_qkv, &tmp_d, &mut had_buf);
                        matvec_tensor_dispatch(&block.attn_qkv, x_in, &mut qkv_pre, qkv_dim, d);
                    }
                    {
                        let x_in = hadamard_pre(had, &block.attn_gate, &tmp_d, &mut had_buf);
                        matvec_tensor_dispatch(
                            &block.attn_gate,
                            x_in,
                            &mut gate_path,
                            ssm_inner,
                            d,
                        );
                    }
                    matvec_tensor_dispatch(&block.ssm_alpha, &tmp_d, &mut alpha, n_v_heads, d);
                    matvec_tensor_dispatch(&block.ssm_beta, &tmp_d, &mut beta, n_v_heads, d);
                    // Sigmoid on beta (numerically stable).
                    for b in beta.iter_mut() {
                        let s = sigmoid_stable(*b);
                        *b = s;
                    }

                    // 3. Depthwise conv1d on qkv. The conv weight is
                    //    pre-dequanted to f32 at load time
                    //    (`SsmBlockWeights::ssm_conv1d_f32`) so the
                    //    per-decode hot path is a single conv call.
                    let dn = &mut dn_cache.layers[li];
                    k::delta_net::conv1d_depthwise_step_f32(
                        &qkv_pre,
                        &block.ssm_conv1d_f32,
                        &mut dn.conv_state,
                        &mut qkv_conv,
                        qkv_dim,
                        conv_kernel,
                    );
                    log_hidden("ssm-qkv-preconv", li, &qkv_pre);
                    log_hidden("ssm-qkv-postconv", li, &qkv_conv);
                    k::delta_net::silu_f32_inplace(&mut qkv_conv);
                    log_hidden("ssm-qkv-postsilu", li, &qkv_conv);
                    log_hidden("ssm-gate", li, &gate_path);
                    log_hidden("ssm-alpha", li, &alpha);
                    log_hidden("ssm-beta-sigmoid", li, &beta);

                    // 4. FLAT [all_Q | all_K | all_V] split per token.
                    //    This is the qwen35moe (NOT qwen3next) convention:
                    //    llama.cpp's `models/qwen35moe.cpp::build_layer_attn_linear`
                    //    extracts q/k/v from contiguous channel blocks:
                    //      q: offset 0, size head_k_dim × num_k_heads
                    //      k: offset Q_block_size, size head_k_dim × num_k_heads
                    //      v: offset 2 × Q_block_size, size head_v_dim × num_v_heads
                    //    Earlier I applied a "per-K-head interleaved" fix
                    //    that was sourced from qwen3next reference — that
                    //    convention is correct for qwen3next but the WRONG
                    //    layout for qwen35moe arch (which is what this model
                    //    actually is). Reverting to flat split.
                    let _v_per_qk = n_v_heads / n_qk_heads;
                    let q_block_size = head_qk_dim * n_qk_heads; // = d
                    let k_block_size = head_qk_dim * n_qk_heads; // = d
                    debug_assert_eq!(
                        q_block_size + k_block_size + head_v_dim * n_v_heads,
                        qkv_dim,
                        "DeltaNet flat qkv split must sum to qkv_dim"
                    );

                    // 5. Per V-head Delta Rule.
                    //
                    // V-head storage layout: **TILED** per ggml broadcast.
                    // Qwen3.5/3.6 conversion (`_LinearAttentionVReorderBase`)
                    // reorders all V-head-indexed components (ssm_alpha/beta,
                    // ssm_a, dt_bias, conv1d V channels, attn_gate Z, V part
                    // of attn_qkv, ssm_out columns) from HF native GROUPED
                    // `[G0_v0..v(r-1), G1_v0..v(r-1), ...]` to ggml TILED
                    // `[G0_v0, G1_v0, ..., G(N-1)_v0, G0_v1, ..., G(N-1)_v(r-1)]`
                    // so `ggml_repeat_4d` can broadcast `[Q0, Q1, ..., Q(N-1)]`
                    // across V positions as `[Q0, Q1, ..., Q(N-1), Q0, ...]`.
                    //
                    // In TILED order, V-slot s belongs to K-group (s % n_k_heads).
                    // Our previous `kh = h / v_per_qk` indexing was correct for
                    // the GROUPED layout used by `Qwen3NextModel`'s base
                    // converter (tiny qwen3next reference); the in-wild
                    // qwen35moe arch uses TILED and requires `kh = h % n_qk_heads`.
                    // DeltaNet-input dumps (mirror HF dn_NN_{q_pre,k_pre,v,g,beta}).
                    let dump_dn = dump_acts_dir.is_some();
                    let mut dn_q_pre = if dump_dn {
                        vec![0.0f32; n_v_heads * head_qk_dim]
                    } else {
                        vec![]
                    };
                    let mut dn_k_pre = if dump_dn {
                        vec![0.0f32; n_v_heads * head_qk_dim]
                    } else {
                        vec![]
                    };
                    let mut dn_v = if dump_dn {
                        vec![0.0f32; n_v_heads * head_v_dim]
                    } else {
                        vec![]
                    };
                    let mut dn_g = if dump_dn {
                        vec![0.0f32; n_v_heads]
                    } else {
                        vec![]
                    };
                    let mut dn_beta = if dump_dn {
                        vec![0.0f32; n_v_heads]
                    } else {
                        vec![]
                    };
                    if dump_dn {
                        // Debug-dump arm: keep the original serial loop so
                        // activation logs stay in deterministic head order.
                        for h in 0..n_v_heads {
                            // Decay formula: `ssm_a * softplus(alpha + dt_bias)`.
                            // The GGUF converter pre-bakes `-exp(ssm_a_log)` into
                            // the stored `ssm_a` tensor (see llama.cpp's
                            // `models/qwen35moe.cpp` line ~402), so we multiply
                            // raw — NOT `-exp(ssm_a)`. The earlier code applied
                            // `-exp` to an already-negated-and-exponentiated
                            // value, producing per-head systematic bias that
                            // compounded across the 30 SSM layers.
                            let decay = block.ssm_a[h]
                                * delta_net_softplus(alpha[h] + block.ssm_dt_bias[h]);
                            // TILED V-head layout: V-slot h belongs to
                            // K-group `h % n_qk_heads`.
                            let kh = h % n_qk_heads;
                            let q_off = kh * head_qk_dim;
                            let k_off = q_block_size + kh * head_qk_dim;
                            let v_off = q_block_size + k_block_size + h * head_v_dim;
                            // HF passes `use_qk_l2norm_in_kernel=True` → L2-
                            // normalize Q and K per head BEFORE the delta-rule
                            // step. Without this the recurrent state magnitude
                            // grows unboundedly.
                            q_head.copy_from_slice(&qkv_conv[q_off..q_off + head_qk_dim]);
                            k_head.copy_from_slice(&qkv_conv[k_off..k_off + head_qk_dim]);
                            dn_q_pre[h * head_qk_dim..(h + 1) * head_qk_dim]
                                .copy_from_slice(&q_head);
                            dn_k_pre[h * head_qk_dim..(h + 1) * head_qk_dim]
                                .copy_from_slice(&k_head);
                            dn_v[h * head_v_dim..(h + 1) * head_v_dim]
                                .copy_from_slice(&qkv_conv[v_off..v_off + head_v_dim]);
                            dn_g[h] = decay;
                            dn_beta[h] = beta[h];
                            k::delta_net::l2norm_f32_inplace(&mut q_head, 1e-6);
                            k::delta_net::l2norm_f32_inplace(&mut k_head, 1e-6);
                            // llama.cpp's delta-net autoregressive kernel
                            // (`delta-net-base.cpp::build_delta_net_autoregressive`)
                            // scales Q by 1/sqrt(S_k) where S_k = head_qk_dim
                            // BEFORE the recurrence. Without this, our delta-
                            // rule output is ~sqrt(head_qk_dim) ≈ 11× too large
                            // per V-head, compounding through the 30 SSM layers.
                            let q_scale = (head_qk_dim as f32).sqrt().recip();
                            for v in q_head.iter_mut() {
                                *v *= q_scale;
                            }
                            let v_head = &qkv_conv[v_off..v_off + head_v_dim];
                            let state_off = h * head_qk_dim * head_v_dim;
                            let state_head = &mut dn.recurrent_state
                                [state_off..state_off + head_qk_dim * head_v_dim];
                            k::delta_net::delta_rule_step_f32_with_scratch(
                                &q_head,
                                &k_head,
                                v_head,
                                decay,
                                beta[h],
                                state_head,
                                &mut out_head,
                                head_qk_dim,
                                head_v_dim,
                                &mut v_tilde,
                            );
                            // Gated RMSNorm per head.
                            let gate_head = &gate_path[h * head_v_dim..(h + 1) * head_v_dim];
                            k::delta_net::gated_rmsnorm_f32_inplace(
                                &mut out_head,
                                gate_head,
                                &block.ssm_norm,
                                cfg.rms_eps,
                            );
                            v_out_concat[h * head_v_dim..(h + 1) * head_v_dim]
                                .copy_from_slice(&out_head);
                        }
                    } else {
                        // Parallel arm: V-heads touch disjoint recurrent-state
                        // and output slabs, and heads never reduce into a
                        // shared accumulator, so per-head arithmetic order is
                        // unchanged — results are bitwise-identical to the
                        // serial loop. The closure calls pure `k::delta_net`
                        // kernels only (no accel dispatch → the rayon
                        // workers' default placement TLS is never consulted).
                        // See the serial arm above for the decay / TILED-
                        // layout / l2norm / q-scale derivations.
                        use rayon::prelude::*;
                        let state_stride = head_qk_dim * head_v_dim;
                        let q_scale = (head_qk_dim as f32).sqrt().recip();
                        dn.recurrent_state[..n_v_heads * state_stride]
                            .par_chunks_mut(state_stride)
                            .zip(v_out_concat[..n_v_heads * head_v_dim].par_chunks_mut(head_v_dim))
                            .enumerate()
                            .for_each_init(
                                || {
                                    (
                                        vec![0.0f32; head_qk_dim],
                                        vec![0.0f32; head_qk_dim],
                                        vec![0.0f32; head_v_dim],
                                        vec![0.0f32; head_v_dim],
                                        vec![0.0f32; head_v_dim],
                                    )
                                },
                                |(q_head, k_head, out_head, v_tilde, norm_scratch),
                                 (h, (state_head, v_out_head))| {
                                    let decay = block.ssm_a[h]
                                        * delta_net_softplus(alpha[h] + block.ssm_dt_bias[h]);
                                    let kh = h % n_qk_heads;
                                    let q_off = kh * head_qk_dim;
                                    let k_off = q_block_size + kh * head_qk_dim;
                                    let v_off = q_block_size + k_block_size + h * head_v_dim;
                                    q_head.copy_from_slice(&qkv_conv[q_off..q_off + head_qk_dim]);
                                    k_head.copy_from_slice(&qkv_conv[k_off..k_off + head_qk_dim]);
                                    k::delta_net::l2norm_f32_inplace(q_head, 1e-6);
                                    k::delta_net::l2norm_f32_inplace(k_head, 1e-6);
                                    for v in q_head.iter_mut() {
                                        *v *= q_scale;
                                    }
                                    let v_head = &qkv_conv[v_off..v_off + head_v_dim];
                                    k::delta_net::delta_rule_step_f32_with_scratch(
                                        q_head,
                                        k_head,
                                        v_head,
                                        decay,
                                        beta[h],
                                        state_head,
                                        out_head,
                                        head_qk_dim,
                                        head_v_dim,
                                        v_tilde,
                                    );
                                    let gate_head =
                                        &gate_path[h * head_v_dim..(h + 1) * head_v_dim];
                                    k::delta_net::gated_rmsnorm_f32_scratch(
                                        out_head,
                                        gate_head,
                                        &block.ssm_norm,
                                        cfg.rms_eps,
                                        norm_scratch,
                                    );
                                    v_out_head.copy_from_slice(out_head);
                                },
                            );
                    }

                    if dump_dn {
                        log_hidden("dn_q_pre", li, &dn_q_pre);
                        log_hidden("dn_k_pre", li, &dn_k_pre);
                        log_hidden("dn_v", li, &dn_v);
                        log_hidden("dn_g", li, &dn_g);
                        log_hidden("dn_beta", li, &dn_beta);
                    }
                    // Intra-SSM dumps for HF-vs-Rust diff (only when dump dir set).
                    log_hidden("ssm-prenorm", li, &tmp_d);
                    log_hidden("ssm-vout", li, &v_out_concat);
                    // 6. Output projection back to d_model.
                    {
                        let x_in = hadamard_pre_ssm_out(
                            had,
                            &block.ssm_out,
                            &v_out_concat,
                            head_v_dim,
                            n_qk_heads,
                            n_v_heads / n_qk_heads,
                            &mut had_perm_buf,
                            &mut had_buf,
                        );
                        matvec_tensor_dispatch(&block.ssm_out, x_in, &mut layer_out, d, ssm_inner);
                    }
                    log_hidden("ssm-branch-out", li, &layer_out);
                    // Residual.
                    k::add_inplace_f32(&mut hidden, &layer_out);
                    log_hidden("post-ssm", li, &hidden);

                    // 3. Post-attention RMSNorm → MoE FFN.
                    k::rmsnorm_f32_row(
                        &hidden,
                        &block.post_attention_norm,
                        &mut tmp_d,
                        cfg.rms_eps,
                    );
                    log_hidden("ssm-moe-input", li, &tmp_d);
                    match &block.ffn {
                        HybridFfn::Moe {
                            router,
                            gate_per_expert,
                            up_per_expert,
                            down_per_expert,
                            w_gate_shared,
                            w_up_shared,
                            w_down_shared,
                            shared_router,
                            ..
                        } => crate::moe::moe_ffn_one_into_parts(
                            &tmp_d,
                            router,
                            &crate::moe::router_spec_for(cfg),
                            None, // M1-tail: qwen35moe hybrids carry no routed bias
                            gate_per_expert,
                            up_per_expert,
                            down_per_expert,
                            w_gate_shared.as_ref(),
                            w_up_shared.as_ref(),
                            w_down_shared.as_ref(),
                            shared_router.as_ref(),
                            d,
                            d_ff,
                            n_experts,
                            top_k,
                            &mut layer_out,
                            &mut gate_buf,
                            &mut up_buf,
                            &mut ff_buf,
                            &mut down_buf,
                            &mut expert_logits,
                            &mut picks,
                        ),
                        HybridFfn::Dense {
                            w_gate,
                            w_up,
                            w_down,
                        } => dense_ffn_one_into(
                            w_gate,
                            w_up,
                            w_down,
                            &tmp_d,
                            d,
                            d_ff,
                            &mut layer_out,
                            &mut gate_buf,
                            &mut up_buf,
                            &mut ff_buf,
                            had,
                            &mut had_buf,
                        ),
                    }
                    log_hidden("ssm-moe-out", li, &layer_out);
                    k::add_inplace_f32(&mut hidden, &layer_out);
                    log_hidden("post-ssm-ffn", li, &hidden);
                }
                HybridLayer::FullAttention(block) => {
                    // 1. Pre-attention RMSNorm.
                    k::rmsnorm_f32_row(&hidden, &block.attn_norm, &mut tmp_d, cfg.rms_eps);

                    // B1 (Kindi `k_rmsfold` spirit): q/k/v share this attn-norm
                    // input AND its Hadamard rotation (the WHT depends only on
                    // `tmp_d` + its width, not on the weight), so rotate ONCE in
                    // place and let all three matmuls read it — instead of
                    // re-running the WHT per projection via `hadamard_pre`.
                    // Bit-identical; saves 2 of 3 WHTs per full-attention layer.
                    debug_assert!(
                        had.map_or(true, |h| {
                            let q = h.is_folded(&block.w_q.name);
                            q == h.is_folded(&block.w_k.name) && q == h.is_folded(&block.w_v.name)
                        }),
                        "hadamard: q/k/v foldedness must match for shared-input rotate-once"
                    );
                    hadamard_rotate_inplace(had, &block.w_q, &mut tmp_d);

                    // 2. Q projection — qwen35moe doubles q_proj's
                    //    output dim to carry a per-head gate that's
                    //    chunked off here and multiplied into the
                    //    attention output below. See HF
                    //    Qwen3_5MoeAttention.forward:
                    //        qg = self.q_proj(hidden).view(..., n_h, head_dim * 2)
                    //        query, gate = qg.chunk(2, dim=-1)
                    //        attn_output = attn_output * sigmoid(gate)
                    //    Per-head layout: `[h0.q, h0.g, h1.q, h1.g, …]`
                    //    flattened. Total dim = 2 * d_q.
                    matvec_tensor_dispatch(&block.w_q, &tmp_d, &mut qg_buf, 2 * d_q, d);
                    // Deinterleave per-head into separate query / gate
                    // buffers (hoisted scratch).
                    for h in 0..n_heads {
                        let src_base = h * 2 * head_dim;
                        q_buf[h * head_dim..(h + 1) * head_dim]
                            .copy_from_slice(&qg_buf[src_base..src_base + head_dim]);
                        attn_gate_buf[h * head_dim..(h + 1) * head_dim]
                            .copy_from_slice(&qg_buf[src_base + head_dim..src_base + 2 * head_dim]);
                    }
                    matvec_tensor_dispatch(&block.w_k, &tmp_d, &mut k_buf, d_kv, d);
                    matvec_tensor_dispatch(&block.w_v, &tmp_d, &mut v_buf, d_kv, d);

                    // 3. Per-head Q/K norm (Qwen3 family).
                    if let Some(qn) = block.q_norm.as_ref() {
                        for h in 0..n_heads {
                            let s = &mut q_buf[h * head_dim..(h + 1) * head_dim];
                            k::rmsnorm_f32_row(s, qn, &mut head_tmp, cfg.rms_eps);
                            s.copy_from_slice(&head_tmp);
                        }
                    }
                    if let Some(kn) = block.k_norm.as_ref() {
                        for h in 0..n_kv_heads {
                            let s = &mut k_buf[h * head_dim..(h + 1) * head_dim];
                            k::rmsnorm_f32_row(s, kn, &mut head_tmp, cfg.rms_eps);
                            s.copy_from_slice(&head_tmp);
                        }
                    }

                    // 4. RoPE on Q, K with **partial rotary factor**.
                    //    qwen35moe declares `partial_rotary_factor = 0.25`
                    //    so only the first `rope_dim` elements of each
                    //    `head_dim`-sized head get rotated; the rest
                    //    pass through. For text-only inference the
                    //    multi-section `[11, 11, 10, 0]` layout
                    //    degenerates to standard Neox RoPE on the
                    //    rotated prefix.
                    // RUSTLLAMA_NO_ROPE=1 skips RoPE for A/B testing.
                    if !no_rope_enabled() {
                        apply_partial_rope_per_head(
                            &mut q_buf,
                            n_heads,
                            head_dim,
                            rope_dim,
                            pos,
                            cfg.rope_theta,
                        );
                        apply_partial_rope_per_head(
                            &mut k_buf,
                            n_kv_heads,
                            head_dim,
                            rope_dim,
                            pos,
                            cfg.rope_theta,
                        );
                    }

                    // 5. Append K, V to the layer's KV cache at slot
                    //    `pos`. The hybrid `KvCache` uses the same
                    //    layer-indexed layout as the standard
                    //    transformer path — for SSM layers the slot
                    //    is allocated but unused.
                    let cur_pos = pos as usize;
                    let max_ctx = kv.max_ctx;
                    let kv_len = cur_pos + 1;
                    let kv_bias_l = kv.kv_bias.clone();
                    let kv_layer = &mut kv.layers[li];
                    match kv_layer {
                        KvLayer::F32 { k, v } => {
                            for h in 0..n_kv_heads {
                                let dst = (h * max_ctx + cur_pos) * head_dim;
                                k[dst..dst + head_dim]
                                    .copy_from_slice(&k_buf[h * head_dim..(h + 1) * head_dim]);
                                v[dst..dst + head_dim]
                                    .copy_from_slice(&v_buf[h * head_dim..(h + 1) * head_dim]);
                            }
                            // 6. Attention over K/V history. Try the GPU
                            //    flash-decode first (CUDA → SYCL): the
                            //    helper re-writes the new K/V row into its
                            //    own resident F32 mirror from k_buf/v_buf
                            //    (exactly as the quant hybrid arms do), so
                            //    it does NOT depend on the hybrid host K/V
                            //    slab being uploaded. Decline → the
                            //    existing CPU GQA flash-decode kernel, so
                            //    F32 behavior is unchanged when no GPU
                            //    path engages.
                            // Seed the CUDA decode KV mirror with the prefill
                            // history on the first decode step after a prefill
                            // (a no-op once resident). Without it the resident-
                            // mirror gap check declines and decode attention
                            // runs on the CPU for the whole generation.
                            crate::accel::decode_seed_kv_f32(
                                k,
                                v,
                                li,
                                cur_pos as u32,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                max_ctx as u32,
                                cfg.n_layers as u32,
                            );
                            // This is the gated-attention (sigmoid Q-gate) decode
                            // path — NOT a Gemma2 / Mistral-SWA arch, so there is no
                            // sliding window or attention-logit soft-cap here (the
                            // CPU twin is the plain `gqa_attention_flash_decode`).
                            // Pass window=0 / softcap=0.0 (full attention, no cap).
                            if !crate::accel::try_flash_attn_decode_gpu_f32(
                                &q_buf,
                                &k_buf[..n_kv_heads * head_dim],
                                &v_buf[..n_kv_heads * head_dim],
                                &mut attn_out,
                                li,
                                cur_pos as u32,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                max_ctx as u32,
                                cfg.n_layers as u32,
                                0,
                                0.0,
                            ) {
                                k::gqa_attention_flash_decode(
                                    &q_buf,
                                    k,
                                    v,
                                    &mut attn_out,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    max_ctx,
                                    kv_len,
                                );
                            }
                            // 7. Apply the per-head sigmoid gate
                            //    (chunked from q_proj above). The
                            //    multiply is element-wise across the
                            //    flat `[n_heads * head_dim]` attn_out.
                            // RUSTLLAMA_NO_QGATE=1 skips the gate
                            // multiplication for A/B testing.
                            if !no_qgate_enabled() {
                                for (a, g) in attn_out.iter_mut().zip(attn_gate_buf.iter()) {
                                    let s = sigmoid_stable(*g);
                                    *a *= s;
                                }
                            }
                        }
                        KvLayer::Q4_0 { k_q, v_q } => {
                            // Q4_0 KV — the Prism-style 4-bit cache.
                            // Quantize the new K/V rows into ggml
                            // blocks, flash-decode with per-kv-head
                            // dequant scratch, then the same sigmoid
                            // Q-gate as the F32 arm.
                            use rustllama_kernels_cpu::hadamard::whiten_chunks_inplace;
                            use rustllama_kernels_cpu::q4_0_kv;
                            let blocks_per_row = head_dim / q4_0_kv::Q4_0_BLOCK_ELEMS;
                            let bytes_per_row = blocks_per_row * q4_0_kv::Q4_0_BLOCK_BYTES;
                            // Whitening (fork attn_rot parity): rotate
                            // Q + K with the same self-inverse chunked
                            // WHT (scores exactly preserved) and V
                            // pre-write; the attention output gets the
                            // transform again, which un-rotates the V
                            // basis. Must happen BEFORE the Q-gate
                            // (gate is elementwise in the primal basis).
                            let whiten = q4_0_whiten_active(head_dim);
                            if whiten {
                                whiten_chunks_inplace(&mut q_buf, KV_WHITEN_CHUNK);
                                whiten_chunks_inplace(
                                    &mut k_buf[..n_kv_heads * head_dim],
                                    KV_WHITEN_CHUNK,
                                );
                                whiten_chunks_inplace(
                                    &mut v_buf[..n_kv_heads * head_dim],
                                    KV_WHITEN_CHUNK,
                                );
                            }
                            // Calibration observe (pre-bias, target
                            // basis) + K mean-centering subtract —
                            // exactly softmax-invariant (kv_bias docs).
                            if crate::kv_bias::calib_active() {
                                crate::kv_bias::calib_observe(li, &k_buf[..n_kv_heads * head_dim]);
                            }
                            if let Some(b) = kv_bias_l.as_ref().and_then(|b| b.layer(li)) {
                                for (x, bb) in k_buf[..n_kv_heads * head_dim].iter_mut().zip(b) {
                                    *x -= *bb;
                                }
                            }
                            for h in 0..n_kv_heads {
                                let p_dst = (h * max_ctx + cur_pos) * bytes_per_row;
                                q4_0_kv::quantize_row(
                                    &k_buf[h * head_dim..(h + 1) * head_dim],
                                    &mut k_q[p_dst..p_dst + bytes_per_row],
                                );
                                q4_0_kv::quantize_row(
                                    &v_buf[h * head_dim..(h + 1) * head_dim],
                                    &mut v_q[p_dst..p_dst + bytes_per_row],
                                );
                            }
                            // GPU quant-KV flash decode first (USM). k_buf/
                            // v_buf are already whitened + K-bias-subtracted,
                            // matching the host slab the helper's mirror is
                            // re-quantized from. Decline → CPU. The un-whiten
                            // below applies to either output.
                            // Seed the CUDA decode mirror with the prefill
                            // history (no-op once resident); host k_q/v_q are
                            // byte-identical to the mirror.
                            crate::accel::decode_seed_kv_q4_0(
                                &k_q[..],
                                &v_q[..],
                                li,
                                cur_pos as u32,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                max_ctx as u32,
                                cfg.n_layers as u32,
                            );
                            if !crate::accel::try_flash_attn_decode_gpu_q4_0(
                                &q_buf,
                                &k_buf[..n_kv_heads * head_dim],
                                &v_buf[..n_kv_heads * head_dim],
                                &mut attn_out,
                                li,
                                cur_pos as u32,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                max_ctx as u32,
                                cfg.n_layers as u32,
                                // Hybrid (SSM/attn) layers never carry SWA/soft-cap: pass 0 / 0.0.
                                0,
                                0.0,
                            ) {
                                q4_0_kv::gqa_attention_flash_decode_q4_0(
                                    &q_buf,
                                    k_q,
                                    v_q,
                                    &mut attn_out,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    max_ctx,
                                    kv_len,
                                );
                            }
                            if whiten {
                                whiten_chunks_inplace(&mut attn_out, KV_WHITEN_CHUNK);
                            }
                            if !no_qgate_enabled() {
                                for (a, g) in attn_out.iter_mut().zip(attn_gate_buf.iter()) {
                                    let s = sigmoid_stable(*g);
                                    *a *= s;
                                }
                            }
                        }
                        KvLayer::Q8_0 {
                            k_q,
                            k_scales,
                            v_q,
                            v_scales,
                        } => {
                            // All-KV-quant support for hybrids. Mirrors
                            // the dense Q8_0 arm (per-row i8 + absmax
                            // scale), then applies the hybrid per-head
                            // sigmoid Q-gate. No whitening/kv-bias — the
                            // 8-bit path is high-quality without them.
                            for h in 0..n_kv_heads {
                                let row_idx = h * max_ctx + cur_pos;
                                let dst = row_idx * head_dim;
                                k_scales[row_idx] = quantize_row_q8_0(
                                    &k_buf[h * head_dim..(h + 1) * head_dim],
                                    &mut k_q[dst..dst + head_dim],
                                );
                                v_scales[row_idx] = quantize_row_q8_0(
                                    &v_buf[h * head_dim..(h + 1) * head_dim],
                                    &mut v_q[dst..dst + head_dim],
                                );
                            }
                            // GPU quant-KV flash decode first (USM); host
                            // slab written above, so decline → CPU.
                            // Seed the CUDA decode mirror with the prefill
                            // history (no-op once resident); host k_q/k_scales
                            // are byte-identical to the mirror.
                            crate::accel::decode_seed_kv_q8_0(
                                unsafe {
                                    std::slice::from_raw_parts(k_q.as_ptr() as *const u8, k_q.len())
                                },
                                unsafe {
                                    std::slice::from_raw_parts(v_q.as_ptr() as *const u8, v_q.len())
                                },
                                &k_scales[..],
                                &v_scales[..],
                                li,
                                cur_pos as u32,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                max_ctx as u32,
                                cfg.n_layers as u32,
                            );
                            if !crate::accel::try_flash_attn_decode_gpu_q8_0(
                                &q_buf,
                                &k_buf[..n_kv_heads * head_dim],
                                &v_buf[..n_kv_heads * head_dim],
                                &mut attn_out,
                                li,
                                cur_pos as u32,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                max_ctx as u32,
                                cfg.n_layers as u32,
                                // Hybrid (SSM/attn) layers never carry SWA/soft-cap: pass 0 / 0.0.
                                0,
                                0.0,
                            ) {
                                k::gqa_attention_flash_decode_q8_0(
                                    &q_buf,
                                    k_q,
                                    k_scales,
                                    v_q,
                                    v_scales,
                                    &mut attn_out,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    max_ctx,
                                    kv_len,
                                );
                            }
                            if !no_qgate_enabled() {
                                for (a, g) in attn_out.iter_mut().zip(attn_gate_buf.iter()) {
                                    let s = sigmoid_stable(*g);
                                    *a *= s;
                                }
                            }
                        }
                        KvLayer::TurboQuant {
                            bits,
                            k_packed,
                            k_scales,
                            v_packed,
                            v_scales,
                        } => {
                            // TurboQuant (low-bit) KV for hybrids,
                            // mirroring the dense TQ arm. quantize_row
                            // mutates its input in place, so copy each
                            // head row into scratch first.
                            let bytes_per_row =
                                rustllama_kernels_cpu::turboquant::bytes_per_block(head_dim, *bits);
                            let mut tq_row = vec![0f32; head_dim];
                            for h in 0..n_kv_heads {
                                let row_idx = h * max_ctx + cur_pos;
                                let p_dst = row_idx * bytes_per_row;
                                tq_row.copy_from_slice(&k_buf[h * head_dim..(h + 1) * head_dim]);
                                k_scales[row_idx] = rustllama_kernels_cpu::turboquant::quantize_row(
                                    &mut tq_row,
                                    *bits,
                                    &mut k_packed[p_dst..p_dst + bytes_per_row],
                                );
                                tq_row.copy_from_slice(&v_buf[h * head_dim..(h + 1) * head_dim]);
                                v_scales[row_idx] = rustllama_kernels_cpu::turboquant::quantize_row(
                                    &mut tq_row,
                                    *bits,
                                    &mut v_packed[p_dst..p_dst + bytes_per_row],
                                );
                            }
                            // GPU quant-KV flash decode first (USM); host
                            // slab written above, so decline → CPU.
                            // Seed the CUDA decode mirror with the prefill
                            // history (no-op once resident); host
                            // k_packed/k_scales are byte-identical to the mirror.
                            crate::accel::decode_seed_kv_tq(
                                *bits,
                                &k_packed[..],
                                &v_packed[..],
                                &k_scales[..],
                                &v_scales[..],
                                li,
                                cur_pos as u32,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                max_ctx as u32,
                                cfg.n_layers as u32,
                            );
                            if !crate::accel::try_flash_attn_decode_gpu_tq(
                                &q_buf,
                                &k_buf[..n_kv_heads * head_dim],
                                &v_buf[..n_kv_heads * head_dim],
                                &mut attn_out,
                                *bits,
                                li,
                                cur_pos as u32,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                max_ctx as u32,
                                cfg.n_layers as u32,
                                // Hybrid (SSM/attn) layers never carry SWA/soft-cap: pass 0 / 0.0.
                                0,
                                0.0,
                            ) {
                                rustllama_kernels_cpu::turboquant::gqa_attention_flash_decode_tq(
                                    &q_buf,
                                    k_packed,
                                    k_scales,
                                    v_packed,
                                    v_scales,
                                    *bits,
                                    &mut attn_out,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    max_ctx,
                                    kv_len,
                                );
                            }
                            if !no_qgate_enabled() {
                                for (a, g) in attn_out.iter_mut().zip(attn_gate_buf.iter()) {
                                    let s = sigmoid_stable(*g);
                                    *a *= s;
                                }
                            }
                        }
                        KvLayer::Nvfp4 { k_packed, v_packed } => {
                            // NVFP4 (4-bit float, 16 elems/block) KV for
                            // hybrids, mirroring the dense NVFP4 arm.
                            let blocks_per_row =
                                head_dim / rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS;
                            let bytes_per_row =
                                blocks_per_row * rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES;
                            for h in 0..n_kv_heads {
                                let p_dst = (h * max_ctx + cur_pos) * bytes_per_row;
                                for b in 0..blocks_per_row {
                                    let elem_off = h * head_dim
                                        + b * rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS;
                                    let blk_dst =
                                        p_dst + b * rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES;
                                    rustllama_kernels_cpu::nvfp4::quantize_block(
                                        &k_buf[elem_off
                                            ..elem_off
                                                + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS],
                                        &mut k_packed[blk_dst
                                            ..blk_dst
                                                + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES],
                                    );
                                    rustllama_kernels_cpu::nvfp4::quantize_block(
                                        &v_buf[elem_off
                                            ..elem_off
                                                + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS],
                                        &mut v_packed[blk_dst
                                            ..blk_dst
                                                + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES],
                                    );
                                }
                            }
                            // GPU quant-KV flash decode first (USM); host
                            // slab written above, so decline → CPU.
                            // Seed the CUDA decode mirror with the prefill
                            // history (no-op once resident); host k_packed/
                            // v_packed are byte-identical to the mirror.
                            crate::accel::decode_seed_kv_nvfp4(
                                &k_packed[..],
                                &v_packed[..],
                                li,
                                cur_pos as u32,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                max_ctx as u32,
                                cfg.n_layers as u32,
                            );
                            if !crate::accel::try_flash_attn_decode_gpu_nvfp4(
                                &q_buf,
                                &k_buf[..n_kv_heads * head_dim],
                                &v_buf[..n_kv_heads * head_dim],
                                &mut attn_out,
                                li,
                                cur_pos as u32,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                max_ctx as u32,
                                cfg.n_layers as u32,
                                // Hybrid (SSM/attn) layers never carry SWA/soft-cap: pass 0 / 0.0.
                                0,
                                0.0,
                            ) {
                                rustllama_kernels_cpu::nvfp4::gqa_attention_flash_decode_nvfp4(
                                    &q_buf,
                                    k_packed,
                                    v_packed,
                                    &mut attn_out,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    max_ctx,
                                    kv_len,
                                );
                            }
                            if !no_qgate_enabled() {
                                for (a, g) in attn_out.iter_mut().zip(attn_gate_buf.iter()) {
                                    let s = sigmoid_stable(*g);
                                    *a *= s;
                                }
                            }
                        }
                        KvLayer::Mxfp4 { k_packed, v_packed } => {
                            mxfp_kv_decode(
                                KvDtype::Mxfp4,
                                &q_buf,
                                &k_buf,
                                &v_buf,
                                k_packed,
                                v_packed,
                                &mut attn_out,
                                li,
                                cur_pos,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                max_ctx,
                                kv_len,
                                cfg.n_layers,
                                // Hybrid (SSM/attn) layers never carry SWA/soft-cap: pass 0 / 0.0.
                                0,
                                0.0,
                            );
                            if !no_qgate_enabled() {
                                for (a, g) in attn_out.iter_mut().zip(attn_gate_buf.iter()) {
                                    *a *= sigmoid_stable(*g);
                                }
                            }
                        }
                        KvLayer::Mxfp6 { k_packed, v_packed } => {
                            mxfp_kv_decode(
                                KvDtype::Mxfp6,
                                &q_buf,
                                &k_buf,
                                &v_buf,
                                k_packed,
                                v_packed,
                                &mut attn_out,
                                li,
                                cur_pos,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                max_ctx,
                                kv_len,
                                cfg.n_layers,
                                // Hybrid (SSM/attn) layers never carry SWA/soft-cap: pass 0 / 0.0.
                                0,
                                0.0,
                            );
                            if !no_qgate_enabled() {
                                for (a, g) in attn_out.iter_mut().zip(attn_gate_buf.iter()) {
                                    *a *= sigmoid_stable(*g);
                                }
                            }
                        }
                        KvLayer::Mxfp8 { k_packed, v_packed } => {
                            mxfp_kv_decode(
                                KvDtype::Mxfp8,
                                &q_buf,
                                &k_buf,
                                &v_buf,
                                k_packed,
                                v_packed,
                                &mut attn_out,
                                li,
                                cur_pos,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                max_ctx,
                                kv_len,
                                cfg.n_layers,
                                // Hybrid (SSM/attn) layers never carry SWA/soft-cap: pass 0 / 0.0.
                                0,
                                0.0,
                            );
                            if !no_qgate_enabled() {
                                for (a, g) in attn_out.iter_mut().zip(attn_gate_buf.iter()) {
                                    *a *= sigmoid_stable(*g);
                                }
                            }
                        }
                        // Hybrid models coerce f16/bf16 KV to F32 at load
                        // (the windowed/soft-cap hybrid kernels are f32),
                        // so a half-KV layer can never reach the hybrid
                        // forward.
                        KvLayer::Half { .. } => {
                            unreachable!(
                                "hybrid forward: half (f16/bf16) KV is coerced to F32 at load"
                            )
                        }
                    }
                    {
                        let x_in = hadamard_pre(had, &block.w_o, &attn_out, &mut had_buf);
                        matvec_tensor_dispatch(&block.w_o, x_in, &mut attn_proj, d, d_q);
                    }
                    k::add_inplace_f32(&mut hidden, &attn_proj);
                    log_hidden("post-attn", li, &hidden);

                    // 5. Post-attention RMSNorm → MoE FFN.
                    k::rmsnorm_f32_row(
                        &hidden,
                        &block.post_attention_norm,
                        &mut tmp_d,
                        cfg.rms_eps,
                    );
                    log_hidden("attn-moe-input", li, &tmp_d);
                    match &block.ffn {
                        HybridFfn::Moe {
                            router,
                            gate_per_expert,
                            up_per_expert,
                            down_per_expert,
                            w_gate_shared,
                            w_up_shared,
                            w_down_shared,
                            shared_router,
                            ..
                        } => crate::moe::moe_ffn_one_into_parts(
                            &tmp_d,
                            router,
                            &crate::moe::router_spec_for(cfg),
                            None, // M1-tail: qwen35moe hybrids carry no routed bias
                            gate_per_expert,
                            up_per_expert,
                            down_per_expert,
                            w_gate_shared.as_ref(),
                            w_up_shared.as_ref(),
                            w_down_shared.as_ref(),
                            shared_router.as_ref(),
                            d,
                            d_ff,
                            n_experts,
                            top_k,
                            &mut layer_out,
                            &mut gate_buf,
                            &mut up_buf,
                            &mut ff_buf,
                            &mut down_buf,
                            &mut expert_logits,
                            &mut picks,
                        ),
                        HybridFfn::Dense {
                            w_gate,
                            w_up,
                            w_down,
                        } => dense_ffn_one_into(
                            w_gate,
                            w_up,
                            w_down,
                            &tmp_d,
                            d,
                            d_ff,
                            &mut layer_out,
                            &mut gate_buf,
                            &mut up_buf,
                            &mut ff_buf,
                            had,
                            &mut had_buf,
                        ),
                    }
                    log_hidden("attn-moe-out", li, &layer_out);
                    k::add_inplace_f32(&mut hidden, &layer_out);
                    log_hidden("post-attn-ffn", li, &hidden);
                }
            }
        }

        // Bump seq_len so subsequent forward calls extend the KV
        // history. Mirrors the standard transformer forward.
        kv.seq_len = (pos as usize) + 1;

        hidden
    }

    /// Chunked hybrid prefill (roadmap Phase 4): forward a chunk of
    /// tokens through the hybrid stack with **grouped-expert MoE
    /// execution**. Attention and DeltaNet run per token inside the
    /// chunk with exactly the serial step semantics (batched
    /// flash-prefill attention for the full-attention layers is the
    /// planned next optimization); the MoE FFN runs once per layer
    /// per chunk via [`crate::moe::moe_ffn_chunk_into_parts`] — each
    /// selected expert's weights are read once per chunk instead of
    /// once per token, the dominant prefill cost on a big MoE. Each
    /// layer also hints the readahead thread to stream the *next*
    /// layer's whole expert pool from disk while this one computes
    /// ([`crate::accel::note_prefill_layer`] — FreeToken's
    /// double-buffered prefill streaming, expressed as page-cache
    /// warming on a shared-memory host).
    ///
    /// Returns the **last** token's logits (standard prefill
    /// contract, mirroring [`Self::forward_prefill`]).
    ///
    /// Reference implementation: [`Self::forward_one_hybrid`] — the
    /// per-token branch bodies here mirror its layer steps minus the
    /// debug instrumentation. Numerical relationship: per-(token,
    /// expert) products use the same kernels; only the accumulation
    /// order over a token's experts changes (fp-association
    /// tolerance, same class as the dense batched-prefill paths).
    pub fn forward_prefill_hybrid(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        dn_cache: &mut DeltaNetCache,
    ) -> Vec<f32> {
        self.forward_prefill_hybrid_impl(tokens, None, start_pos, kv, dn_cache, None)
    }

    /// Chunked hybrid prefill over PRE-EMBEDDED rows — the VLM splice
    /// path (`[n, d_model]` flattened; text rows from
    /// [`Self::embed_tokens_primal`], image rows from the vision
    /// projector). CONTRACT: rows are in the PRIMAL (activation)
    /// basis — for Hadamard-folded models the embedding inverse is
    /// already applied; this entry does NOT apply it again.
    pub fn forward_prefill_hybrid_from_embeds(
        &self,
        embeds: &[f32],
        start_pos: u32,
        kv: &mut KvCache,
        dn_cache: &mut DeltaNetCache,
    ) -> Vec<f32> {
        assert_eq!(
            embeds.len() % self.cfg.d_model,
            0,
            "forward_prefill_hybrid_from_embeds: embeds must be n × d_model"
        );
        self.forward_prefill_hybrid_impl(&[], Some(embeds), start_pos, kv, dn_cache, None)
    }

    /// Token embedding lookup in the PRIMAL basis: `embed_tokens` +
    /// the per-row Hadamard embedding inverse when the model's table
    /// is rotated (identity for non-Hadamard models). Rows from here
    /// match what the hybrid forwards feed their first layer, so they
    /// can be spliced alongside vision-projector output and run
    /// through [`Self::forward_prefill_hybrid_from_embeds`].
    pub fn embed_tokens_primal(&self, token_ids: &[i32]) -> Vec<f32> {
        let d = self.cfg.d_model;
        let mut rows = self.embed_tokens(token_ids);
        if let Some(h) = self.weights.hadamard.as_ref() {
            if h.embd_inverse && !no_hadamard_enabled() {
                let signs = h.signs_for(d);
                for t in 0..token_ids.len() {
                    k::hadamard::hadamard_inverse_inplace(
                        &mut rows[t * d..(t + 1) * d],
                        signs,
                        h.block_size,
                    );
                }
            }
        }
        rows
    }

    /// E1 speculation primitive for HYBRID models: one chunked hybrid
    /// prefill over `tokens`, capturing **per-position** LM-head
    /// logits into `logits_out` (`tokens.len() * vocab_size`). The
    /// per-layer body is byte-identical to [`Self::forward_prefill_hybrid`]
    /// — only the LM-head tail differs (all rows instead of the last).
    ///
    /// NOTE: this advances BOTH the KV cache and the DeltaNet
    /// recurrent state through every position. Speculative callers
    /// must snapshot the DeltaNet state beforehand
    /// ([`DeltaNetCache::snapshot`]) and restore + replay on partial
    /// acceptance — attention KV rewinds by `seq_len` truncation, the
    /// SSM state does not.
    pub fn forward_speculation_batched_hybrid(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        dn_cache: &mut DeltaNetCache,
        logits_out: &mut [f32],
    ) {
        assert_eq!(
            logits_out.len(),
            tokens.len() * self.cfg.vocab_size,
            "forward_speculation_batched_hybrid: logits_out must be tokens.len() * vocab_size"
        );
        let _ = self.forward_prefill_hybrid_impl(
            tokens,
            None,
            start_pos,
            kv,
            dn_cache,
            Some(logits_out),
        );
    }

    fn forward_prefill_hybrid_impl(
        &self,
        tokens: &[i32],
        embeds: Option<&[f32]>,
        start_pos: u32,
        kv: &mut KvCache,
        dn_cache: &mut DeltaNetCache,
        all_logits: Option<&mut [f32]>,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let n = match embeds {
            Some(rows) => rows.len() / cfg.d_model,
            None => tokens.len(),
        };
        assert!(n > 0, "forward_prefill_hybrid: empty chunk");
        assert!(
            start_pos as usize + n <= kv.max_ctx,
            "chunk exceeds max_ctx"
        );
        assert!(
            self.weights.is_hybrid(),
            "forward_prefill_hybrid called on non-hybrid model — engine dispatch bug"
        );
        let hyb = cfg.hybrid.as_ref().expect("hybrid cfg present");
        let layers = self
            .weights
            .hybrid_layers
            .as_ref()
            .expect("hybrid_layers present");
        assert_eq!(layers.len(), cfg.n_layers);
        assert_eq!(dn_cache.layers.len(), cfg.n_layers);
        assert_eq!(kv.layers.len(), cfg.n_layers);

        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let ssm_inner = hyb.ssm_inner_size as usize;
        let conv_kernel = hyb.ssm_conv_kernel as usize;
        let first_ssm = layers
            .iter()
            .find_map(|l| match l {
                HybridLayer::Ssm(s) => Some(s),
                _ => None,
            })
            .expect("at least one SSM layer required");
        let n_v_heads = first_ssm.ssm_a.len();
        let head_v_dim = ssm_inner / n_v_heads;
        let head_qk_dim = head_v_dim;
        // QK head count: `ssm.group_count` metadata when present
        // (Bonsai-2/Qwen3.8: 16 groups at d=5120, where the legacy
        // `d / head_qk_dim` derivation would wrongly give 40). The
        // legacy derivation stays as the fallback and yields the
        // SAME value on qwen35moe (2048/128 = 16 = group_count), so
        // Qwen3.6-family models are byte-identical either way.
        let n_qk_heads = if hyb.ssm_group_count > 1 {
            hyb.ssm_group_count as usize
        } else {
            d / head_qk_dim
        };
        let d_ff = cfg.d_ff;
        // MoE geometry is optional: qwen35moe hybrids route experts,
        // Bonsai-2/dense-qwen35 hybrids run a plain SwiGLU (HybridFfn
        // decides per layer; (0, 0) is never read on the Dense arm).
        let (n_experts, top_k) = cfg
            .moe
            .as_ref()
            .map(|m| (m.n_experts as usize, m.n_experts_used as usize))
            .unwrap_or((0, 0));
        let n_mtp_block_layers = cfg
            .hybrid
            .as_ref()
            .map(|h| h.nextn_predict_layers as usize)
            .unwrap_or(0);
        let main_layer_count = if include_mtp_layers() {
            cfg.n_layers
        } else {
            cfg.n_layers.saturating_sub(n_mtp_block_layers)
        };
        let rope_dim = cfg.rope_dim;

        // --- Embed the whole chunk ---
        // Token path: lookup + (for rotated tables) the Hadamard
        // embedding inverse. From-embeds path: rows arrive ALREADY in
        // the primal basis (`forward_prefill_hybrid_from_embeds`
        // contract) — copy verbatim, never re-apply the inverse.
        let mut hidden_rows = vec![0.0f32; n * d];
        if let Some(rows) = embeds {
            hidden_rows.copy_from_slice(rows);
        } else {
            for (t, &tok) in tokens.iter().enumerate() {
                k::embed_lookup_tensor(
                    &self.weights.token_embd,
                    &[tok],
                    &mut hidden_rows[t * d..(t + 1) * d],
                    d,
                );
            }
            if let Some(h) = self.weights.hadamard.as_ref() {
                if h.embd_inverse && !no_hadamard_enabled() {
                    let signs = h.signs_for(d);
                    for t in 0..n {
                        k::hadamard::hadamard_inverse_inplace(
                            &mut hidden_rows[t * d..(t + 1) * d],
                            signs,
                            h.block_size,
                        );
                    }
                }
            }
        }

        // Per-token scratch, hoisted out of the token loops. (The
        // projection inputs/outputs that used to live here — tmp_d,
        // qkv_pre, gate_path, alpha, beta, layer_out, qg_buf,
        // attn_proj — moved into the hoist_* row buffers below when
        // the projections became sub-chunk batched matvecs.)
        // Fused-QKV width: q + k (n_qk heads each) + v (ssm_inner).
        // On qwen35moe this equals 2·ssm_inner (the old hardcode);
        // on Bonsai-2/Qwen3.8 (16 qk-heads at d=5120) it does not.
        let qkv_dim = 2 * n_qk_heads * head_qk_dim + ssm_inner;
        let mut qkv_conv = vec![0.0f32; qkv_dim];
        let mut v_out_concat = vec![0.0f32; ssm_inner];
        let mut q_buf = vec![0.0f32; d_q];
        let mut k_buf = vec![0.0f32; d_kv];
        let mut v_buf = vec![0.0f32; d_kv];
        let mut attn_gate_buf = vec![0.0f32; d_q];
        // Chunk-level FFN buffers.
        let mut moe_in_rows = vec![0.0f32; n * d];
        let mut moe_out_rows = vec![0.0f32; n * d];
        // Per-head attention scratch (hoisted). The DeltaNet per-head
        // scratch moved into the rayon `for_each_init` of the parallel
        // head loop (one set per pool worker).
        let mut head_tmp = vec![0.0f32; head_dim];
        // Projection-hoist buffers: the hybrid prefill batches each
        // layer's time-independent projections over sub-chunks of
        // PREFILL_HOIST_SUB tokens (weight rows decode once per
        // sub-chunk instead of once per token). Sized once here,
        // reused across layers and sub-chunks.
        const PREFILL_HOIST_SUB: usize = 64;
        let hoist_cap = PREFILL_HOIST_SUB.min(n.max(1));
        let mut hoist_norm = vec![0.0f32; hoist_cap * d];
        let mut hoist_rot: Vec<f32> = Vec::new();
        let mut hoist_qkv = vec![0.0f32; hoist_cap * qkv_dim];
        let mut hoist_gate = vec![0.0f32; hoist_cap * ssm_inner];
        let mut hoist_alpha = vec![0.0f32; hoist_cap * n_v_heads];
        let mut hoist_beta = vec![0.0f32; hoist_cap * n_v_heads];
        let mut hoist_qg = vec![0.0f32; hoist_cap * 2 * d_q];
        let mut hoist_kr = vec![0.0f32; hoist_cap * d_kv];
        let mut hoist_vr = vec![0.0f32; hoist_cap * d_kv];
        let mut hoist_vrot = vec![0.0f32; hoist_cap * ssm_inner];
        let mut hoist_lout = vec![0.0f32; hoist_cap * d];
        let mut hoist_attn_out = vec![0.0f32; hoist_cap * d_q];
        let mut hoist_qrows = vec![0.0f32; hoist_cap * d_q];
        let mut hoist_gaterows = vec![0.0f32; hoist_cap * d_q];
        // RUSTLLAMA_PROFILE_HYBRID_PREFILL: coarse per-phase wall
        // accumulators (ms), dumped once per call. Diagnostic lever
        // for locating the prefill per-row floor.
        let pp_on = hybrid_prefill_profile_enabled();
        let mut pp = [0f64; 8];
        let mut pp_mark = std::time::Instant::now();
        let pp_call = std::time::Instant::now();

        let q_block_size = head_qk_dim * n_qk_heads;
        let k_block_size = head_qk_dim * n_qk_heads;
        debug_assert_eq!(
            q_block_size + k_block_size + head_v_dim * n_v_heads,
            qkv_dim,
            "DeltaNet flat qkv split must sum to qkv_dim"
        );
        let no_rope = no_rope_enabled();
        let no_qgate = no_qgate_enabled();
        // ADDITIVE, default-off SSM prefill accel: when
        // RUSTLLAMA_SSM_PREFILL_CHUNKED is set, the SSM prefill arm
        // computes the per-head delta recurrence with the chunked-parallel
        // kernel (`delta_rule_prefill_chunked`) instead of nb sequential
        // per-token `delta_rule_step` calls. Read ONCE here, outside the
        // token loops. Off ⇒ the byte-identical per-token path runs.
        // T3: typed `LoadOptions`-backed accel knob (env-initialized for
        // back-compat; the tuner's SSM sweep toggles it via the setter).
        let ssm_prefill_chunked = crate::accel::ssm_prefill_chunked();
        // Prism Hadamard rotation state + scratch (see decode twin).
        let had = self.weights.hadamard.as_ref();
        let mut had_buf: Vec<f32> = Vec::new();
        let mut had_perm_buf: Vec<f32> = Vec::new();

        for (li, layer) in layers.iter().take(main_layer_count).enumerate() {
            // Publish the layer index for the GPU cutoff / expert-pin
            // keys — see the decode twin. Must precede any dispatch.
            crate::accel::set_current_layer_idx(li as u32);
            // Layer-ahead readahead: stream the NEXT layer's whole
            // expert pool from disk while this layer computes.
            crate::accel::note_prefill_layer(li as u32 + 1);
            match layer {
                HybridLayer::Ssm(block) => {
                    // The conv + delta-rule recurrence is inherently
                    // sequential over tokens, but this layer's
                    // projections are not (they read only the layer's
                    // input rows), so they hoist into batched matvecs
                    // over PREFILL_HOIST_SUB-token sub-chunks: each
                    // ternary weight row decodes once per sub-chunk
                    // instead of once per token — the structural fix
                    // for the per-token prefill weight re-stream.
                    // Per-token math is unchanged (same rmsnorm, same
                    // per-tensor Hadamard fold per row, bitwise-equal
                    // batched matvec kernel, same sigmoid), so model
                    // output is identical to the serial form.
                    let dn = &mut dn_cache.layers[li];
                    for t0 in (0..n).step_by(PREFILL_HOIST_SUB) {
                        let t1 = (t0 + PREFILL_HOIST_SUB).min(n);
                        let nb = t1 - t0;
                        if pp_on {
                            pp_mark = std::time::Instant::now();
                        }
                        for bi in 0..nb {
                            k::rmsnorm_f32_row(
                                &hidden_rows[(t0 + bi) * d..(t0 + bi + 1) * d],
                                &block.attn_norm,
                                &mut hoist_norm[bi * d..(bi + 1) * d],
                                cfg.rms_eps,
                            );
                        }
                        {
                            let xs = hadamard_pre_rows(
                                had,
                                &block.attn_qkv,
                                &hoist_norm[..nb * d],
                                d,
                                nb,
                                &mut hoist_rot,
                            );
                            matvec_tensor_batched_dispatch(
                                &block.attn_qkv,
                                xs,
                                &mut hoist_qkv[..nb * qkv_dim],
                                qkv_dim,
                                d,
                                nb,
                            );
                        }
                        {
                            let xs = hadamard_pre_rows(
                                had,
                                &block.attn_gate,
                                &hoist_norm[..nb * d],
                                d,
                                nb,
                                &mut hoist_rot,
                            );
                            matvec_tensor_batched_dispatch(
                                &block.attn_gate,
                                xs,
                                &mut hoist_gate[..nb * ssm_inner],
                                ssm_inner,
                                d,
                                nb,
                            );
                        }
                        // alpha/beta are unfolded (no Hadamard) — batched
                        // directly over the norm rows, then the same
                        // element-wise sigmoid the per-token path applied.
                        matvec_tensor_batched_dispatch(
                            &block.ssm_alpha,
                            &hoist_norm[..nb * d],
                            &mut hoist_alpha[..nb * n_v_heads],
                            n_v_heads,
                            d,
                            nb,
                        );
                        matvec_tensor_batched_dispatch(
                            &block.ssm_beta,
                            &hoist_norm[..nb * d],
                            &mut hoist_beta[..nb * n_v_heads],
                            n_v_heads,
                            d,
                            nb,
                        );
                        for b in hoist_beta[..nb * n_v_heads].iter_mut() {
                            let s = sigmoid_stable(*b);
                            *b = s;
                        }
                        if pp_on {
                            pp[0] += pp_mark.elapsed().as_secs_f64() * 1e3;
                            pp_mark = std::time::Instant::now();
                        }
                        if ssm_prefill_chunked {
                            // ADDITIVE chunked-prefill path (default-off). Same
                            // per-(token,head) math as the per-token `else`
                            // branch below — same conv+silu, same q/k/v build,
                            // same decay, same gate, same output projection — but
                            // the delta recurrence over this sub-chunk's `nb`
                            // tokens is computed by ONE `delta_rule_prefill_chunked`
                            // call per V-head instead of `nb` sequential
                            // `delta_rule_step` calls. `dn.recurrent_state` after
                            // this branch equals what the sequential branch would
                            // leave, so later sub-chunks / decode continue
                            // correctly; `dn.conv_state` is advanced by the
                            // still-sequential conv exactly as before.
                            use rayon::prelude::*;
                            let state_stride = head_qk_dim * head_v_dim;
                            let q_scale = (head_qk_dim as f32).sqrt().recip();
                            // 1. Sequential conv1d + silu for all nb tokens (the
                            //    conv recurrence over dn.conv_state cannot chunk).
                            //    Persist each token's conv'd qkv row for the scan.
                            let mut qkv_conv_sub = vec![0.0f32; nb * qkv_dim];
                            for bi in 0..nb {
                                k::delta_net::conv1d_depthwise_step_f32(
                                    &hoist_qkv[bi * qkv_dim..(bi + 1) * qkv_dim],
                                    &block.ssm_conv1d_f32,
                                    &mut dn.conv_state,
                                    &mut qkv_conv_sub[bi * qkv_dim..(bi + 1) * qkv_dim],
                                    qkv_dim,
                                    conv_kernel,
                                );
                                k::delta_net::silu_f32_inplace(
                                    &mut qkv_conv_sub[bi * qkv_dim..(bi + 1) * qkv_dim],
                                );
                            }
                            // 2. Per V-head chunked delta scan — rayon over heads
                            //    on disjoint recurrent-state slabs (the SAME
                            //    parallelism axis as the per-token path). For each
                            //    head, gather the nb-length token-major q/k/v/g/beta
                            //    sequences (q: l2norm then *q_scale; k: l2norm; v:
                            //    verbatim; g: this head's per-token decay; beta:
                            //    the already-sigmoid'd hoist_beta), then thread the
                            //    head's recurrent_state slab IN/OUT through one
                            //    chunked call. Outputs are collected head-major
                            //    ([n_v_heads][nb][head_v_dim]) and gated-RMSNorm'd
                            //    per token per head, exactly as the per-token path.
                            let mut chunk_out = vec![0.0f32; n_v_heads * nb * head_v_dim];
                            // P17: attempt the WHOLE-batch GPU DeltaNet chunked scan
                            // (all V-heads in one `ssm:delta_net` kernel). Fail-closed:
                            // `try_delta_net_chunk_prefill` returns false unless the
                            // parity verdict passed AND this is a dedicated-VRAM SYCL
                            // GPU (off on the Iris Xe box / any non-SYCL host), in which
                            // case the CPU rayon path below runs byte-identically to
                            // pre-P17. When it DOES run, it packs head-major q/k/v/g/beta
                            // with the IDENTICAL per-head gather the CPU closure uses
                            // (l2norm + q-scale on q, l2norm on k, decay =
                            // ssm_a·softplus(alpha+dt_bias), sigmoid'd beta), threads
                            // dn.recurrent_state through the kernel, and leaves the raw
                            // (pre-gate) scan in `chunk_out` — the gated RMSNorm is then
                            // applied below exactly as the per-token path does.
                            let gpu_scan = {
                                let mut q_pk = vec![0.0f32; n_v_heads * nb * head_qk_dim];
                                let mut k_pk = vec![0.0f32; n_v_heads * nb * head_qk_dim];
                                let mut v_pk = vec![0.0f32; n_v_heads * nb * head_v_dim];
                                let mut g_pk = vec![0.0f32; n_v_heads * nb];
                                let mut beta_pk = vec![0.0f32; n_v_heads * nb];
                                for h in 0..n_v_heads {
                                    let kh = h % n_qk_heads;
                                    let q_off = kh * head_qk_dim;
                                    let k_off = q_block_size + kh * head_qk_dim;
                                    let v_off = q_block_size + k_block_size + h * head_v_dim;
                                    for bi in 0..nb {
                                        let conv = &qkv_conv_sub[bi * qkv_dim..(bi + 1) * qkv_dim];
                                        let base_qk = (h * nb + bi) * head_qk_dim;
                                        let base_v = (h * nb + bi) * head_v_dim;
                                        let qd = &mut q_pk[base_qk..base_qk + head_qk_dim];
                                        qd.copy_from_slice(&conv[q_off..q_off + head_qk_dim]);
                                        k::delta_net::l2norm_f32_inplace(qd, 1e-6);
                                        for x in qd.iter_mut() {
                                            *x *= q_scale;
                                        }
                                        let kd = &mut k_pk[base_qk..base_qk + head_qk_dim];
                                        kd.copy_from_slice(&conv[k_off..k_off + head_qk_dim]);
                                        k::delta_net::l2norm_f32_inplace(kd, 1e-6);
                                        v_pk[base_v..base_v + head_v_dim]
                                            .copy_from_slice(&conv[v_off..v_off + head_v_dim]);
                                        g_pk[h * nb + bi] = block.ssm_a[h]
                                            * delta_net_softplus(
                                                hoist_alpha[bi * n_v_heads + h]
                                                    + block.ssm_dt_bias[h],
                                            );
                                        beta_pk[h * nb + bi] = hoist_beta[bi * n_v_heads + h];
                                    }
                                }
                                crate::accel::try_delta_net_chunk_prefill(
                                    &q_pk,
                                    &k_pk,
                                    &v_pk,
                                    &g_pk,
                                    &beta_pk,
                                    &mut dn.recurrent_state[..n_v_heads * state_stride],
                                    &mut chunk_out,
                                    n_v_heads,
                                    nb,
                                    head_qk_dim,
                                    head_v_dim,
                                    k::delta_net::ssm_chunk_current(),
                                )
                            };
                            if gpu_scan {
                                // Gated RMSNorm per (head, token) on the raw GPU scan.
                                // (rayon prelude is already in scope for this branch.)
                                chunk_out
                                    .par_chunks_mut(nb * head_v_dim)
                                    .enumerate()
                                    .for_each(|(h, out_head_all)| {
                                        let mut norm_scratch = vec![0.0f32; head_v_dim];
                                        for bi in 0..nb {
                                            let gate_head = &hoist_gate[bi * ssm_inner
                                                + h * head_v_dim
                                                ..bi * ssm_inner + (h + 1) * head_v_dim];
                                            let out_bi = &mut out_head_all
                                                [bi * head_v_dim..(bi + 1) * head_v_dim];
                                            k::delta_net::gated_rmsnorm_f32_scratch(
                                                out_bi,
                                                gate_head,
                                                &block.ssm_norm,
                                                cfg.rms_eps,
                                                &mut norm_scratch,
                                            );
                                        }
                                    });
                            } else {
                                dn.recurrent_state[..n_v_heads * state_stride]
                                    .par_chunks_mut(state_stride)
                                    .zip(chunk_out.par_chunks_mut(nb * head_v_dim))
                                    .enumerate()
                                    .for_each(|(h, (state_head, out_head_all))| {
                                        let kh = h % n_qk_heads;
                                        let q_off = kh * head_qk_dim;
                                        let k_off = q_block_size + kh * head_qk_dim;
                                        let v_off = q_block_size + k_block_size + h * head_v_dim;
                                        let mut q_seq = vec![0.0f32; nb * head_qk_dim];
                                        let mut k_seq = vec![0.0f32; nb * head_qk_dim];
                                        let mut v_seq = vec![0.0f32; nb * head_v_dim];
                                        let mut g_seq = vec![0.0f32; nb];
                                        let mut beta_seq = vec![0.0f32; nb];
                                        for bi in 0..nb {
                                            let conv =
                                                &qkv_conv_sub[bi * qkv_dim..(bi + 1) * qkv_dim];
                                            let qd = &mut q_seq
                                                [bi * head_qk_dim..(bi + 1) * head_qk_dim];
                                            qd.copy_from_slice(&conv[q_off..q_off + head_qk_dim]);
                                            k::delta_net::l2norm_f32_inplace(qd, 1e-6);
                                            for x in qd.iter_mut() {
                                                *x *= q_scale;
                                            }
                                            let kd = &mut k_seq
                                                [bi * head_qk_dim..(bi + 1) * head_qk_dim];
                                            kd.copy_from_slice(&conv[k_off..k_off + head_qk_dim]);
                                            k::delta_net::l2norm_f32_inplace(kd, 1e-6);
                                            v_seq[bi * head_v_dim..(bi + 1) * head_v_dim]
                                                .copy_from_slice(&conv[v_off..v_off + head_v_dim]);
                                            g_seq[bi] = block.ssm_a[h]
                                                * delta_net_softplus(
                                                    hoist_alpha[bi * n_v_heads + h]
                                                        + block.ssm_dt_bias[h],
                                                );
                                            beta_seq[bi] = hoist_beta[bi * n_v_heads + h];
                                        }
                                        k::delta_net::delta_rule_prefill_chunked(
                                            &q_seq,
                                            &k_seq,
                                            &v_seq,
                                            &g_seq,
                                            &beta_seq,
                                            state_head,
                                            out_head_all,
                                            nb,
                                            head_qk_dim,
                                            head_v_dim,
                                        );
                                        let mut norm_scratch = vec![0.0f32; head_v_dim];
                                        for bi in 0..nb {
                                            let gate_head = &hoist_gate[bi * ssm_inner
                                                + h * head_v_dim
                                                ..bi * ssm_inner + (h + 1) * head_v_dim];
                                            let out_bi = &mut out_head_all
                                                [bi * head_v_dim..(bi + 1) * head_v_dim];
                                            k::delta_net::gated_rmsnorm_f32_scratch(
                                                out_bi,
                                                gate_head,
                                                &block.ssm_norm,
                                                cfg.rms_eps,
                                                &mut norm_scratch,
                                            );
                                        }
                                    });
                            }
                            // 3. Per-token assembly: gather this token's heads into
                            //    v_out_concat (head-major), then the SAME output-
                            //    projection hoist (tiled→grouped perm + Hadamard)
                            //    the per-token path applies before the batched
                            //    ssm_out matvec below.
                            for bi in 0..nb {
                                for h in 0..n_v_heads {
                                    v_out_concat[h * head_v_dim..(h + 1) * head_v_dim]
                                        .copy_from_slice(
                                            &chunk_out[h * nb * head_v_dim + bi * head_v_dim
                                                ..h * nb * head_v_dim + (bi + 1) * head_v_dim],
                                        );
                                }
                                let x_in = hadamard_pre_ssm_out(
                                    had,
                                    &block.ssm_out,
                                    &v_out_concat,
                                    head_v_dim,
                                    n_qk_heads,
                                    n_v_heads / n_qk_heads,
                                    &mut had_perm_buf,
                                    &mut had_buf,
                                );
                                hoist_vrot[bi * ssm_inner..(bi + 1) * ssm_inner]
                                    .copy_from_slice(x_in);
                            }
                        } else {
                            for bi in 0..nb {
                                let alpha = &hoist_alpha[bi * n_v_heads..(bi + 1) * n_v_heads];
                                let beta = &hoist_beta[bi * n_v_heads..(bi + 1) * n_v_heads];
                                let gate_path = &hoist_gate[bi * ssm_inner..(bi + 1) * ssm_inner];
                                k::delta_net::conv1d_depthwise_step_f32(
                                    &hoist_qkv[bi * qkv_dim..(bi + 1) * qkv_dim],
                                    &block.ssm_conv1d_f32,
                                    &mut dn.conv_state,
                                    &mut qkv_conv,
                                    qkv_dim,
                                    conv_kernel,
                                );
                                k::delta_net::silu_f32_inplace(&mut qkv_conv);
                                // Per V-head Delta Rule, parallel across heads —
                                // disjoint state/output slabs, per-head arithmetic
                                // order unchanged (bitwise-identical to the serial
                                // loop; the decode arm above documents the decay /
                                // TILED-layout / l2norm / q-scale derivations).
                                // Recurrence stays serial over TIME (the outer
                                // per-token loop); only heads fan out.
                                {
                                    use rayon::prelude::*;
                                    let state_stride = head_qk_dim * head_v_dim;
                                    let q_scale = (head_qk_dim as f32).sqrt().recip();
                                    let qkv_conv = &qkv_conv;
                                    let alpha = &alpha;
                                    let beta = &beta;
                                    let gate_path = &gate_path;
                                    dn.recurrent_state[..n_v_heads * state_stride]
                                .par_chunks_mut(state_stride)
                                .zip(
                                    v_out_concat[..n_v_heads * head_v_dim]
                                        .par_chunks_mut(head_v_dim),
                                )
                                .enumerate()
                                .for_each_init(
                                    || {
                                        (
                                            vec![0.0f32; head_qk_dim],
                                            vec![0.0f32; head_qk_dim],
                                            vec![0.0f32; head_v_dim],
                                            vec![0.0f32; head_v_dim],
                                            vec![0.0f32; head_v_dim],
                                        )
                                    },
                                    |(q_head, k_head, out_head, v_tilde, norm_scratch),
                                     (h, (state_head, v_out_head))| {
                                        let decay = block.ssm_a[h]
                                            * delta_net_softplus(
                                                alpha[h] + block.ssm_dt_bias[h],
                                            );
                                        let kh = h % n_qk_heads;
                                        let q_off = kh * head_qk_dim;
                                        let k_off = q_block_size + kh * head_qk_dim;
                                        let v_off =
                                            q_block_size + k_block_size + h * head_v_dim;
                                        q_head.copy_from_slice(
                                            &qkv_conv[q_off..q_off + head_qk_dim],
                                        );
                                        k_head.copy_from_slice(
                                            &qkv_conv[k_off..k_off + head_qk_dim],
                                        );
                                        k::delta_net::l2norm_f32_inplace(q_head, 1e-6);
                                        k::delta_net::l2norm_f32_inplace(k_head, 1e-6);
                                        for v in q_head.iter_mut() {
                                            *v *= q_scale;
                                        }
                                        let v_head = &qkv_conv[v_off..v_off + head_v_dim];
                                        k::delta_net::delta_rule_step_f32_with_scratch(
                                            q_head, k_head, v_head, decay, beta[h],
                                            state_head, out_head, head_qk_dim,
                                            head_v_dim, v_tilde,
                                        );
                                        let gate_head = &gate_path
                                            [h * head_v_dim..(h + 1) * head_v_dim];
                                        k::delta_net::gated_rmsnorm_f32_scratch(
                                            out_head, gate_head, &block.ssm_norm,
                                            cfg.rms_eps, norm_scratch,
                                        );
                                        v_out_head.copy_from_slice(out_head);
                                    },
                                );
                                }
                                // Output-projection hoist: the tiled-to-
                                // grouped permutation + rotation is a per-row
                                // transform, so apply it here and collect the
                                // rotated row; ssm_out -- the last big
                                // per-token weight sweep in this arm -- then
                                // runs as ONE batched matvec below.
                                {
                                    let x_in = hadamard_pre_ssm_out(
                                        had,
                                        &block.ssm_out,
                                        &v_out_concat,
                                        head_v_dim,
                                        n_qk_heads,
                                        n_v_heads / n_qk_heads,
                                        &mut had_perm_buf,
                                        &mut had_buf,
                                    );
                                    hoist_vrot[bi * ssm_inner..(bi + 1) * ssm_inner]
                                        .copy_from_slice(x_in);
                                }
                            }
                        }
                        if pp_on {
                            pp[1] += pp_mark.elapsed().as_secs_f64() * 1e3;
                            pp_mark = std::time::Instant::now();
                        }
                        matvec_tensor_batched_dispatch(
                            &block.ssm_out,
                            &hoist_vrot[..nb * ssm_inner],
                            &mut hoist_lout[..nb * d],
                            d,
                            ssm_inner,
                            nb,
                        );
                        for bi in 0..nb {
                            k::add_inplace_f32(
                                &mut hidden_rows[(t0 + bi) * d..(t0 + bi + 1) * d],
                                &hoist_lout[bi * d..(bi + 1) * d],
                            );
                        }
                        if pp_on {
                            pp[2] += pp_mark.elapsed().as_secs_f64() * 1e3;
                            pp_mark = std::time::Instant::now();
                        }
                    }
                    // Chunked MoE FFN over the whole chunk.
                    for t in 0..n {
                        k::rmsnorm_f32_row(
                            &hidden_rows[t * d..(t + 1) * d],
                            &block.post_attention_norm,
                            &mut moe_in_rows[t * d..(t + 1) * d],
                            cfg.rms_eps,
                        );
                    }
                    match &block.ffn {
                        HybridFfn::Moe {
                            router,
                            gate_per_expert,
                            up_per_expert,
                            down_per_expert,
                            w_gate_shared,
                            w_up_shared,
                            w_down_shared,
                            shared_router,
                            ..
                        } => crate::moe::moe_ffn_chunk_into_parts(
                            &moe_in_rows,
                            n,
                            router,
                            &crate::moe::router_spec_for(cfg),
                            gate_per_expert,
                            up_per_expert,
                            down_per_expert,
                            w_gate_shared.as_ref(),
                            w_up_shared.as_ref(),
                            w_down_shared.as_ref(),
                            shared_router.as_ref(),
                            d,
                            d_ff,
                            n_experts,
                            top_k,
                            &mut moe_out_rows,
                        ),
                        HybridFfn::Dense {
                            w_gate,
                            w_up,
                            w_down,
                        } => dense_ffn_chunk_into(
                            w_gate,
                            w_up,
                            w_down,
                            &moe_in_rows,
                            n,
                            d,
                            d_ff,
                            &mut moe_out_rows,
                            had,
                            &mut had_buf,
                        ),
                    }
                    for t in 0..n {
                        k::add_inplace_f32(
                            &mut hidden_rows[t * d..(t + 1) * d],
                            &moe_out_rows[t * d..(t + 1) * d],
                        );
                    }
                    if pp_on {
                        pp[7] += pp_mark.elapsed().as_secs_f64() * 1e3;
                        pp_mark = std::time::Instant::now();
                    }
                }
                HybridLayer::FullAttention(block) => {
                    // Same hoist as the SSM arm: rmsnorm and the
                    // Q/K/V projections are time-independent, so they
                    // batch over sub-chunks (ternary rows decode once
                    // per sub-chunk). RoPE, q/k norms, whitening, the
                    // KV write, attention, and the output projection
                    // stay per-token and byte-identical.
                    //
                    // `kv_bias` is constant for the whole layer forward
                    // (read-only), so clone the `Option<Arc<_>>` once here
                    // rather than once per token inside the KV-write loop.
                    let kv_bias_l = kv.kv_bias.clone();
                    for t0 in (0..n).step_by(PREFILL_HOIST_SUB) {
                        let t1 = (t0 + PREFILL_HOIST_SUB).min(n);
                        let nb = t1 - t0;
                        if pp_on {
                            pp_mark = std::time::Instant::now();
                        }
                        for bi in 0..nb {
                            k::rmsnorm_f32_row(
                                &hidden_rows[(t0 + bi) * d..(t0 + bi + 1) * d],
                                &block.attn_norm,
                                &mut hoist_norm[bi * d..(bi + 1) * d],
                                cfg.rms_eps,
                            );
                        }
                        {
                            let xs = hadamard_pre_rows(
                                had,
                                &block.w_q,
                                &hoist_norm[..nb * d],
                                d,
                                nb,
                                &mut hoist_rot,
                            );
                            matvec_tensor_batched_dispatch(
                                &block.w_q,
                                xs,
                                &mut hoist_qg[..nb * 2 * d_q],
                                2 * d_q,
                                d,
                                nb,
                            );
                        }
                        {
                            let xs = hadamard_pre_rows(
                                had,
                                &block.w_k,
                                &hoist_norm[..nb * d],
                                d,
                                nb,
                                &mut hoist_rot,
                            );
                            matvec_tensor_batched_dispatch(
                                &block.w_k,
                                xs,
                                &mut hoist_kr[..nb * d_kv],
                                d_kv,
                                d,
                                nb,
                            );
                        }
                        {
                            let xs = hadamard_pre_rows(
                                had,
                                &block.w_v,
                                &hoist_norm[..nb * d],
                                d,
                                nb,
                                &mut hoist_rot,
                            );
                            matvec_tensor_batched_dispatch(
                                &block.w_v,
                                xs,
                                &mut hoist_vr[..nb * d_kv],
                                d_kv,
                                d,
                                nb,
                            );
                        }
                        if pp_on {
                            pp[3] += pp_mark.elapsed().as_secs_f64() * 1e3;
                            pp_mark = std::time::Instant::now();
                        }
                        for bi in 0..nb {
                            let t = t0 + bi;
                            let pos = start_pos + t as u32;
                            // Doubled q_proj carries a per-head sigmoid
                            // gate (qwen35moe) — deinterleave from the
                            // hoisted row; K/V rows copy into the existing
                            // per-token buffers the norm/rope/write code
                            // mutates in place.
                            let qg_row = &hoist_qg[bi * 2 * d_q..(bi + 1) * 2 * d_q];
                            k_buf.copy_from_slice(&hoist_kr[bi * d_kv..(bi + 1) * d_kv]);
                            v_buf.copy_from_slice(&hoist_vr[bi * d_kv..(bi + 1) * d_kv]);
                            for h in 0..n_heads {
                                let src_base = h * 2 * head_dim;
                                q_buf[h * head_dim..(h + 1) * head_dim]
                                    .copy_from_slice(&qg_row[src_base..src_base + head_dim]);
                                attn_gate_buf[h * head_dim..(h + 1) * head_dim].copy_from_slice(
                                    &qg_row[src_base + head_dim..src_base + 2 * head_dim],
                                );
                            }
                            if let Some(qn) = block.q_norm.as_ref() {
                                for h in 0..n_heads {
                                    let s = &mut q_buf[h * head_dim..(h + 1) * head_dim];
                                    k::rmsnorm_f32_row(s, qn, &mut head_tmp, cfg.rms_eps);
                                    s.copy_from_slice(&head_tmp);
                                }
                            }
                            if let Some(kn) = block.k_norm.as_ref() {
                                for h in 0..n_kv_heads {
                                    let s = &mut k_buf[h * head_dim..(h + 1) * head_dim];
                                    k::rmsnorm_f32_row(s, kn, &mut head_tmp, cfg.rms_eps);
                                    s.copy_from_slice(&head_tmp);
                                }
                            }
                            if !no_rope {
                                apply_partial_rope_per_head(
                                    &mut q_buf,
                                    n_heads,
                                    head_dim,
                                    rope_dim,
                                    pos,
                                    cfg.rope_theta,
                                );
                                apply_partial_rope_per_head(
                                    &mut k_buf,
                                    n_kv_heads,
                                    head_dim,
                                    rope_dim,
                                    pos,
                                    cfg.rope_theta,
                                );
                            }
                            let cur_pos = pos as usize;
                            let max_ctx = kv.max_ctx;
                            let kv_layer = &mut kv.layers[li];
                            match kv_layer {
                                KvLayer::F32 { k, v } => {
                                    for h in 0..n_kv_heads {
                                        let dst = (h * max_ctx + cur_pos) * head_dim;
                                        k[dst..dst + head_dim].copy_from_slice(
                                            &k_buf[h * head_dim..(h + 1) * head_dim],
                                        );
                                        v[dst..dst + head_dim].copy_from_slice(
                                            &v_buf[h * head_dim..(h + 1) * head_dim],
                                        );
                                    }
                                }
                                KvLayer::Q4_0 { k_q, v_q } => {
                                    use rustllama_kernels_cpu::hadamard::whiten_chunks_inplace;
                                    use rustllama_kernels_cpu::q4_0_kv;
                                    let blocks_per_row = head_dim / q4_0_kv::Q4_0_BLOCK_ELEMS;
                                    let bytes_per_row = blocks_per_row * q4_0_kv::Q4_0_BLOCK_BYTES;
                                    // Whitening — same contract as the
                                    // hybrid decode arm: Q+K+V rotated
                                    // pre-quantize, output un-rotated
                                    // before the (primal-basis) Q-gate.
                                    let whiten = q4_0_whiten_active(head_dim);
                                    if whiten {
                                        whiten_chunks_inplace(&mut q_buf, KV_WHITEN_CHUNK);
                                        whiten_chunks_inplace(
                                            &mut k_buf[..n_kv_heads * head_dim],
                                            KV_WHITEN_CHUNK,
                                        );
                                        whiten_chunks_inplace(
                                            &mut v_buf[..n_kv_heads * head_dim],
                                            KV_WHITEN_CHUNK,
                                        );
                                    }
                                    // Calibration observe + K bias
                                    // subtract (softmax-invariant).
                                    if crate::kv_bias::calib_active() {
                                        crate::kv_bias::calib_observe(
                                            li,
                                            &k_buf[..n_kv_heads * head_dim],
                                        );
                                    }
                                    if let Some(b) = kv_bias_l.as_ref().and_then(|b| b.layer(li)) {
                                        for (x, bb) in
                                            k_buf[..n_kv_heads * head_dim].iter_mut().zip(b)
                                        {
                                            *x -= *bb;
                                        }
                                    }
                                    for h in 0..n_kv_heads {
                                        let p_dst = (h * max_ctx + cur_pos) * bytes_per_row;
                                        q4_0_kv::quantize_row(
                                            &k_buf[h * head_dim..(h + 1) * head_dim],
                                            &mut k_q[p_dst..p_dst + bytes_per_row],
                                        );
                                        q4_0_kv::quantize_row(
                                            &v_buf[h * head_dim..(h + 1) * head_dim],
                                            &mut v_q[p_dst..p_dst + bytes_per_row],
                                        );
                                    }
                                }
                                KvLayer::Q8_0 {
                                    k_q,
                                    k_scales,
                                    v_q,
                                    v_scales,
                                } => {
                                    // All-KV-quant prefill store (attention
                                    // deferred to the batched flash-prefill
                                    // call below). Mirrors the dense Q8_0
                                    // store.
                                    for h in 0..n_kv_heads {
                                        let row_idx = h * max_ctx + cur_pos;
                                        let dst = row_idx * head_dim;
                                        k_scales[row_idx] = quantize_row_q8_0(
                                            &k_buf[h * head_dim..(h + 1) * head_dim],
                                            &mut k_q[dst..dst + head_dim],
                                        );
                                        v_scales[row_idx] = quantize_row_q8_0(
                                            &v_buf[h * head_dim..(h + 1) * head_dim],
                                            &mut v_q[dst..dst + head_dim],
                                        );
                                    }
                                }
                                KvLayer::TurboQuant {
                                    bits,
                                    k_packed,
                                    k_scales,
                                    v_packed,
                                    v_scales,
                                } => {
                                    let bytes_per_row =
                                        rustllama_kernels_cpu::turboquant::bytes_per_block(
                                            head_dim, *bits,
                                        );
                                    let mut tq_row = vec![0f32; head_dim];
                                    for h in 0..n_kv_heads {
                                        let row_idx = h * max_ctx + cur_pos;
                                        let p_dst = row_idx * bytes_per_row;
                                        tq_row.copy_from_slice(
                                            &k_buf[h * head_dim..(h + 1) * head_dim],
                                        );
                                        k_scales[row_idx] =
                                            rustllama_kernels_cpu::turboquant::quantize_row(
                                                &mut tq_row,
                                                *bits,
                                                &mut k_packed[p_dst..p_dst + bytes_per_row],
                                            );
                                        tq_row.copy_from_slice(
                                            &v_buf[h * head_dim..(h + 1) * head_dim],
                                        );
                                        v_scales[row_idx] =
                                            rustllama_kernels_cpu::turboquant::quantize_row(
                                                &mut tq_row,
                                                *bits,
                                                &mut v_packed[p_dst..p_dst + bytes_per_row],
                                            );
                                    }
                                }
                                KvLayer::Nvfp4 { k_packed, v_packed } => {
                                    let blocks_per_row =
                                        head_dim / rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS;
                                    let bytes_per_row = blocks_per_row
                                        * rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES;
                                    for h in 0..n_kv_heads {
                                        let p_dst = (h * max_ctx + cur_pos) * bytes_per_row;
                                        for b in 0..blocks_per_row {
                                            let elem_off = h * head_dim
                                            + b * rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS;
                                            let blk_dst =
                                            p_dst + b * rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES;
                                            rustllama_kernels_cpu::nvfp4::quantize_block(
                                            &k_buf[elem_off
                                                ..elem_off + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS],
                                            &mut k_packed[blk_dst
                                                ..blk_dst + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES],
                                        );
                                            rustllama_kernels_cpu::nvfp4::quantize_block(
                                            &v_buf[elem_off
                                                ..elem_off + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS],
                                            &mut v_packed[blk_dst
                                                ..blk_dst + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES],
                                        );
                                        }
                                    }
                                }
                                // MXFP KV: quantize-only (attention deferred to the
                                // sub-chunk flash-prefill pass below).
                                KvLayer::Mxfp4 { k_packed, v_packed } => mxfp_kv_quantize_row(
                                    KvDtype::Mxfp4,
                                    &k_buf,
                                    &v_buf,
                                    k_packed,
                                    v_packed,
                                    cur_pos,
                                    n_kv_heads,
                                    head_dim,
                                    max_ctx,
                                ),
                                KvLayer::Mxfp6 { k_packed, v_packed } => mxfp_kv_quantize_row(
                                    KvDtype::Mxfp6,
                                    &k_buf,
                                    &v_buf,
                                    k_packed,
                                    v_packed,
                                    cur_pos,
                                    n_kv_heads,
                                    head_dim,
                                    max_ctx,
                                ),
                                KvLayer::Mxfp8 { k_packed, v_packed } => mxfp_kv_quantize_row(
                                    KvDtype::Mxfp8,
                                    &k_buf,
                                    &v_buf,
                                    k_packed,
                                    v_packed,
                                    cur_pos,
                                    n_kv_heads,
                                    head_dim,
                                    max_ctx,
                                ),
                                KvLayer::Half { .. } => {
                                    unreachable!("hybrid forward: half (f16/bf16) KV is coerced to F32 at load")
                                }
                            }
                            // Attention is DEFERRED: stash this token's
                            // (whitened, for Q4_0) query row and its
                            // per-head gate row; the whole sub-chunk then
                            // attends via ONE flash-prefill call below —
                            // the per-token flash-decode calls re-walked
                            // the growing KV window once per token and
                            // were the last per-token kernel in this arm.
                            hoist_qrows[bi * d_q..(bi + 1) * d_q].copy_from_slice(&q_buf);
                            hoist_gaterows[bi * d_q..(bi + 1) * d_q]
                                .copy_from_slice(&attn_gate_buf);
                        }
                        if pp_on {
                            pp[4] += pp_mark.elapsed().as_secs_f64() * 1e3;
                            pp_mark = std::time::Instant::now();
                        }
                        // Sub-chunk attention: every query row attends its
                        // causal window `[0, kv_len_base + i + 1)` — the
                        // same per-(query, head) online-softmax math the
                        // per-token kernels ran (the Q4_0 pair literally
                        // shares the inner function), so outputs are
                        // bitwise-identical.
                        let sub_whiten;
                        {
                            use rustllama_kernels_cpu::q4_0_kv;
                            let kv_len_base = start_pos as usize + t0;
                            let max_ctx = kv.max_ctx;
                            let kv_layer = &mut kv.layers[li];
                            match kv_layer {
                                KvLayer::F32 { k, v } => {
                                    sub_whiten = false;
                                    // SWA span for this layer (0 ⇒ full causal) +
                                    // Gemma2 attention-logit soft-cap (0.0 ⇒ no cap).
                                    // Both are now THREADED into the GPU prefill kernel
                                    // (bit-exact to the CPU capped+windowed path), so an
                                    // SWA / Gemma2 layer runs on the GPU too — no more
                                    // fail-closed decline. Passthrough at 0/0.0.
                                    let swa_win = self.cfg.swa_window(li);
                                    let softcap = self.cfg.attn_logit_softcapping.unwrap_or(0.0);
                                    let used_gpu = crate::accel::try_flash_attn_prefill_gpu_f32(
                                        &hoist_qrows[..nb * d_q],
                                        k,
                                        v,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads,
                                        n_kv_heads,
                                        head_dim,
                                        max_ctx,
                                        kv_len_base,
                                        nb,
                                        swa_win as u32,
                                        softcap,
                                    );
                                    if !used_gpu {
                                        k::gqa_attention_flash_prefill_capped_windowed(
                                            &hoist_qrows[..nb * d_q],
                                            k,
                                            v,
                                            &mut hoist_attn_out[..nb * d_q],
                                            n_heads,
                                            n_kv_heads,
                                            head_dim,
                                            max_ctx,
                                            kv_len_base,
                                            nb,
                                            swa_win,
                                            softcap,
                                        );
                                    }
                                }
                                KvLayer::Q4_0 { k_q, v_q } => {
                                    sub_whiten = q4_0_whiten_active(head_dim);
                                    // SWA span (0 ⇒ full) + Gemma2 soft-cap (0.0 ⇒ none)
                                    // for this layer, threaded into the GPU + CPU paths.
                                    let swa_win = self.cfg.swa_window(li);
                                    let softcap = self.cfg.attn_logit_softcapping.unwrap_or(0.0);
                                    if !crate::accel::try_flash_attn_prefill_gpu_q4_0(
                                        &hoist_qrows[..nb * d_q],
                                        k_q,
                                        v_q,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads,
                                        n_kv_heads,
                                        head_dim,
                                        max_ctx,
                                        kv_len_base,
                                        nb,
                                        swa_win as u32,
                                        softcap,
                                    ) {
                                        q4_0_kv::gqa_attention_flash_prefill_q4_0_capped_windowed(
                                            &hoist_qrows[..nb * d_q],
                                            k_q,
                                            v_q,
                                            &mut hoist_attn_out[..nb * d_q],
                                            n_heads,
                                            n_kv_heads,
                                            head_dim,
                                            max_ctx,
                                            kv_len_base,
                                            nb,
                                            swa_win,
                                            softcap,
                                        );
                                    }
                                }
                                KvLayer::Q8_0 {
                                    k_q,
                                    k_scales,
                                    v_q,
                                    v_scales,
                                } => {
                                    sub_whiten = false;
                                    // SWA span (0 ⇒ full) + Gemma2 soft-cap (0.0 ⇒ none)
                                    // for this layer, threaded into the GPU + CPU paths.
                                    let swa_win = self.cfg.swa_window(li);
                                    let softcap = self.cfg.attn_logit_softcapping.unwrap_or(0.0);
                                    if !crate::accel::try_flash_attn_prefill_gpu_q8_0(
                                        &hoist_qrows[..nb * d_q],
                                        k_q,
                                        k_scales,
                                        v_q,
                                        v_scales,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads,
                                        n_kv_heads,
                                        head_dim,
                                        max_ctx,
                                        kv_len_base,
                                        nb,
                                        swa_win as u32,
                                        softcap,
                                    ) {
                                        k::gqa_attention_flash_prefill_q8_0_capped_windowed(
                                            &hoist_qrows[..nb * d_q],
                                            k_q,
                                            k_scales,
                                            v_q,
                                            v_scales,
                                            &mut hoist_attn_out[..nb * d_q],
                                            n_heads,
                                            n_kv_heads,
                                            head_dim,
                                            max_ctx,
                                            kv_len_base,
                                            nb,
                                            swa_win,
                                            softcap,
                                        );
                                    }
                                }
                                KvLayer::TurboQuant {
                                    bits,
                                    k_packed,
                                    k_scales,
                                    v_packed,
                                    v_scales,
                                } => {
                                    sub_whiten = false;
                                    // SWA span (0 ⇒ full) + Gemma2 soft-cap (0.0 ⇒ none)
                                    // for this layer, threaded into the GPU + CPU paths.
                                    let swa_win = self.cfg.swa_window(li);
                                    let softcap = self.cfg.attn_logit_softcapping.unwrap_or(0.0);
                                    if !crate::accel::try_flash_attn_prefill_gpu_tq(
                                        &hoist_qrows[..nb * d_q],
                                        k_packed,
                                        k_scales,
                                        v_packed,
                                        v_scales,
                                        *bits,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads,
                                        n_kv_heads,
                                        head_dim,
                                        max_ctx,
                                        kv_len_base,
                                        nb,
                                        swa_win as u32,
                                        softcap,
                                    ) {
                                        rustllama_kernels_cpu::turboquant::gqa_attention_flash_prefill_tq_capped_windowed(
                                        &hoist_qrows[..nb * d_q], k_packed, k_scales, v_packed, v_scales,
                                        *bits, &mut hoist_attn_out[..nb * d_q],
                                        n_heads, n_kv_heads, head_dim, max_ctx,
                                        kv_len_base, nb, swa_win, softcap,
                                    );
                                    }
                                }
                                KvLayer::Nvfp4 { k_packed, v_packed } => {
                                    sub_whiten = false;
                                    // SWA span (0 ⇒ full) + Gemma2 soft-cap (0.0 ⇒ none)
                                    // for this layer, threaded into the GPU + CPU paths.
                                    let swa_win = self.cfg.swa_window(li);
                                    let softcap = self.cfg.attn_logit_softcapping.unwrap_or(0.0);
                                    if !crate::accel::try_flash_attn_prefill_gpu_nvfp4(
                                        &hoist_qrows[..nb * d_q],
                                        k_packed,
                                        v_packed,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads,
                                        n_kv_heads,
                                        head_dim,
                                        max_ctx,
                                        kv_len_base,
                                        nb,
                                        swa_win as u32,
                                        softcap,
                                    ) {
                                        rustllama_kernels_cpu::nvfp4::gqa_attention_flash_prefill_nvfp4_capped_windowed(
                                        &hoist_qrows[..nb * d_q], k_packed, v_packed,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads, n_kv_heads, head_dim, max_ctx,
                                        kv_len_base, nb, swa_win, softcap,
                                    );
                                    }
                                }
                                // MXFP KV: flash-only (cache already populated in the
                                // per-position quantize pass); no whitening.
                                KvLayer::Mxfp4 { k_packed, v_packed } => {
                                    sub_whiten = false;
                                    // SWA span (0 ⇒ full) + Gemma2 soft-cap (0.0 ⇒ none)
                                    // for this layer, threaded into the GPU + CPU paths.
                                    let swa_win = self.cfg.swa_window(li);
                                    let softcap = self.cfg.attn_logit_softcapping.unwrap_or(0.0);
                                    if !crate::accel::try_flash_attn_prefill_gpu_mxfp4(
                                        &hoist_qrows[..nb * d_q],
                                        k_packed,
                                        v_packed,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads,
                                        n_kv_heads,
                                        head_dim,
                                        max_ctx,
                                        kv_len_base,
                                        nb,
                                        swa_win as u32,
                                        softcap,
                                    ) {
                                        rustllama_kernels_cpu::mxfp_kv::gqa_attention_flash_prefill_mxfp4_capped_windowed(
                                        &hoist_qrows[..nb * d_q], k_packed, v_packed,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, nb,
                                        swa_win, softcap,
                                    );
                                    }
                                }
                                KvLayer::Mxfp6 { k_packed, v_packed } => {
                                    sub_whiten = false;
                                    // SWA span (0 ⇒ full) + Gemma2 soft-cap (0.0 ⇒ none)
                                    // for this layer, threaded into the GPU + CPU paths.
                                    let swa_win = self.cfg.swa_window(li);
                                    let softcap = self.cfg.attn_logit_softcapping.unwrap_or(0.0);
                                    if !crate::accel::try_flash_attn_prefill_gpu_mxfp6(
                                        &hoist_qrows[..nb * d_q],
                                        k_packed,
                                        v_packed,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads,
                                        n_kv_heads,
                                        head_dim,
                                        max_ctx,
                                        kv_len_base,
                                        nb,
                                        swa_win as u32,
                                        softcap,
                                    ) {
                                        rustllama_kernels_cpu::mxfp_kv::gqa_attention_flash_prefill_mxfp6_capped_windowed(
                                        &hoist_qrows[..nb * d_q], k_packed, v_packed,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, nb,
                                        swa_win, softcap,
                                    );
                                    }
                                }
                                KvLayer::Mxfp8 { k_packed, v_packed } => {
                                    sub_whiten = false;
                                    // SWA span (0 ⇒ full) + Gemma2 soft-cap (0.0 ⇒ none)
                                    // for this layer, threaded into the GPU + CPU paths.
                                    let swa_win = self.cfg.swa_window(li);
                                    let softcap = self.cfg.attn_logit_softcapping.unwrap_or(0.0);
                                    if !crate::accel::try_flash_attn_prefill_gpu_mxfp8(
                                        &hoist_qrows[..nb * d_q],
                                        k_packed,
                                        v_packed,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads,
                                        n_kv_heads,
                                        head_dim,
                                        max_ctx,
                                        kv_len_base,
                                        nb,
                                        swa_win as u32,
                                        softcap,
                                    ) {
                                        rustllama_kernels_cpu::mxfp_kv::gqa_attention_flash_prefill_mxfp8_capped_windowed(
                                        &hoist_qrows[..nb * d_q], k_packed, v_packed,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, nb,
                                        swa_win, softcap,
                                    );
                                    }
                                }
                                KvLayer::Half { .. } => {
                                    unreachable!("hybrid forward: half (f16/bf16) KV is coerced to F32 at load")
                                }
                            }
                        }
                        for bi in 0..nb {
                            let row = &mut hoist_attn_out[bi * d_q..(bi + 1) * d_q];
                            if sub_whiten {
                                rustllama_kernels_cpu::hadamard::whiten_chunks_inplace(
                                    row,
                                    KV_WHITEN_CHUNK,
                                );
                            }
                            if !no_qgate {
                                let gate = &hoist_gaterows[bi * d_q..(bi + 1) * d_q];
                                for (a, g) in row.iter_mut().zip(gate.iter()) {
                                    let s = sigmoid_stable(*g);
                                    *a *= s;
                                }
                            }
                        }
                        if pp_on {
                            pp[5] += pp_mark.elapsed().as_secs_f64() * 1e3;
                            pp_mark = std::time::Instant::now();
                        }
                        {
                            let xs = hadamard_pre_rows(
                                had,
                                &block.w_o,
                                &hoist_attn_out[..nb * d_q],
                                d_q,
                                nb,
                                &mut hoist_rot,
                            );
                            matvec_tensor_batched_dispatch(
                                &block.w_o,
                                xs,
                                &mut hoist_lout[..nb * d],
                                d,
                                d_q,
                                nb,
                            );
                        }
                        for bi in 0..nb {
                            k::add_inplace_f32(
                                &mut hidden_rows[(t0 + bi) * d..(t0 + bi + 1) * d],
                                &hoist_lout[bi * d..(bi + 1) * d],
                            );
                        }
                        if pp_on {
                            pp[6] += pp_mark.elapsed().as_secs_f64() * 1e3;
                            pp_mark = std::time::Instant::now();
                        }
                    }
                    // Chunked MoE FFN over the whole chunk.
                    for t in 0..n {
                        k::rmsnorm_f32_row(
                            &hidden_rows[t * d..(t + 1) * d],
                            &block.post_attention_norm,
                            &mut moe_in_rows[t * d..(t + 1) * d],
                            cfg.rms_eps,
                        );
                    }
                    match &block.ffn {
                        HybridFfn::Moe {
                            router,
                            gate_per_expert,
                            up_per_expert,
                            down_per_expert,
                            w_gate_shared,
                            w_up_shared,
                            w_down_shared,
                            shared_router,
                            ..
                        } => crate::moe::moe_ffn_chunk_into_parts(
                            &moe_in_rows,
                            n,
                            router,
                            &crate::moe::router_spec_for(cfg),
                            gate_per_expert,
                            up_per_expert,
                            down_per_expert,
                            w_gate_shared.as_ref(),
                            w_up_shared.as_ref(),
                            w_down_shared.as_ref(),
                            shared_router.as_ref(),
                            d,
                            d_ff,
                            n_experts,
                            top_k,
                            &mut moe_out_rows,
                        ),
                        HybridFfn::Dense {
                            w_gate,
                            w_up,
                            w_down,
                        } => dense_ffn_chunk_into(
                            w_gate,
                            w_up,
                            w_down,
                            &moe_in_rows,
                            n,
                            d,
                            d_ff,
                            &mut moe_out_rows,
                            had,
                            &mut had_buf,
                        ),
                    }
                    for t in 0..n {
                        k::add_inplace_f32(
                            &mut hidden_rows[t * d..(t + 1) * d],
                            &moe_out_rows[t * d..(t + 1) * d],
                        );
                    }
                    if pp_on {
                        pp[7] += pp_mark.elapsed().as_secs_f64() * 1e3;
                        pp_mark = std::time::Instant::now();
                    }
                }
            }
        }

        kv.seq_len = start_pos as usize + n;

        if pp_on {
            let wall = pp_call.elapsed().as_secs_f64() * 1e3;
            let accounted: f64 = pp.iter().sum();
            tracing::info!(
                n_tokens = n,
                wall_ms = wall as u64,
                accounted_ms = accounted as u64,
                ssm_proj_ms = pp[0] as u64,
                ssm_serial_ms = pp[1] as u64,
                ssm_out_ms = pp[2] as u64,
                attn_proj_ms = pp[3] as u64,
                attn_tokens_ms = pp[4] as u64,
                attn_flash_ms = pp[5] as u64,
                attn_wo_ms = pp[6] as u64,
                ffn_ms = pp[7] as u64,
                "hybrid prefill phase profile"
            );
        }
        let lm_head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);

        // Speculation mode: LM head over EVERY position (batched
        // matvec + rows-Hadamard), plus the standard last-token
        // return value copied out of the batch.
        if let Some(out) = all_logits {
            let mut final_norm_all = vec![0.0f32; n * d];
            for t in 0..n {
                k::rmsnorm_f32_row(
                    &hidden_rows[t * d..(t + 1) * d],
                    &self.weights.output_norm,
                    &mut final_norm_all[t * d..(t + 1) * d],
                    cfg.rms_eps,
                );
            }
            let mut had_rows_buf = Vec::new();
            let x_in = hadamard_pre_rows(had, lm_head, &final_norm_all, d, n, &mut had_rows_buf);
            matvec_tensor_batched_dispatch(lm_head, x_in, out, cfg.vocab_size, d, n);
            return out[(n - 1) * cfg.vocab_size..n * cfg.vocab_size].to_vec();
        }

        // Final output norm + LM head for the LAST token only.
        let last = &hidden_rows[(n - 1) * d..n * d];
        let mut final_norm = vec![0.0f32; d];
        k::rmsnorm_f32_row(
            last,
            &self.weights.output_norm,
            &mut final_norm,
            cfg.rms_eps,
        );
        let mut logits = vec![0.0f32; cfg.vocab_size];
        {
            let x_in = hadamard_pre(had, lm_head, &final_norm, &mut had_buf);
            matvec_tensor_dispatch(lm_head, x_in, &mut logits, cfg.vocab_size, d);
        }
        logits
    }

    /// Phase 6: hybrid forward + NextN/MTP head logits.
    ///
    /// Runs the standard hybrid forward to populate `main_logits_out`
    /// (the next-token distribution from the main LM head) and ALSO
    /// runs the NextN head — the `blk.{N}.nextn.*` tensor module
    /// `qwen35moe` family models attach to the last layer to predict
    /// the token AFTER `next_token_id`.
    ///
    /// **Semantics** (per the DeepSeek-V3 MTP paper, adapted to
    /// the `qwen35moe` NextN layout):
    /// 1. Embed `next_token_id` via `token_embd` → `e_next`
    ///    (the embedding of the token the main forward just predicted
    ///    OR was given by the speculation driver)
    /// 2. Apply `nextn.enorm` to `e_next`, `nextn.hnorm` to
    ///    the post-block hidden state
    /// 3. Concatenate `[enorm(e_next); hnorm(hidden)]` → 2*d_model
    ///    vector
    /// 4. Project via `nextn.eh_proj` → d_model
    /// 5. Apply `nextn.shared_head_norm`
    /// 6. LM head → `nextn_logits_out`
    ///
    /// **Use case**: speculative decoding. The main forward predicts
    /// token t+1; the NextN forward predicts token t+2. A spec
    /// driver verifies both in parallel on the next pass. Phase 6 of
    /// the roadmap ships this primitive; the spec driver wiring is
    /// a follow-up.
    ///
    /// Panics if the model isn't hybrid OR doesn't carry a NextN
    /// head. Caller must check `weights.nextn_head.is_some()`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_one_hybrid_with_nextn_logits(
        &self,
        token_id: i32,
        next_token_id: i32,
        pos: u32,
        kv: &mut KvCache,
        dn_cache: &mut DeltaNetCache,
        main_logits_out: &mut [f32],
        nextn_logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        let head = self
            .weights
            .nextn_head
            .as_ref()
            .expect("forward_one_hybrid_with_nextn_logits called but model has no NextN head");
        assert_eq!(main_logits_out.len(), cfg.vocab_size);
        assert_eq!(nextn_logits_out.len(), cfg.vocab_size);
        let d = cfg.d_model;
        let lm_head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);

        // Run inner forward once — both heads share the same
        // post-block hidden state. Single hidden compute = no
        // duplicated forward work.
        let hidden = self.forward_one_hybrid_to_hidden(token_id, pos, kv, dn_cache);

        // --- Main LM head ---
        let mut final_norm = vec![0.0f32; d];
        k::rmsnorm_f32_row(
            &hidden,
            &self.weights.output_norm,
            &mut final_norm,
            cfg.rms_eps,
        );
        {
            let mut had_buf: Vec<f32> = Vec::new();
            let x_in = hadamard_pre(
                self.weights.hadamard.as_ref(),
                lm_head,
                &final_norm,
                &mut had_buf,
            );
            matvec_tensor_dispatch(lm_head, x_in, main_logits_out, cfg.vocab_size, d);
        }

        // NextN composition extracted to a public free function so
        // it can be parity-tested in isolation against a numpy
        // reference (see tests/nextn_python_parity.rs).
        nextn_compose_logits_f32(
            &hidden,
            next_token_id,
            &self.weights.token_embd,
            head,
            lm_head,
            cfg.rms_eps,
            nextn_logits_out,
        );
    }

    /// Hybrid TIP forward for tree speculation (Feature 4, hybrid path):
    /// advances the KV + DeltaNet caches by the single `token_id` at `pos` and
    /// returns `main_logits_out` (the next-token distribution) plus one
    /// full-vocab logit row per available DRAFT head.
    ///
    /// Draft-head policy (fail-closed — no draft head ⇒ empty vec ⇒ the caller's
    /// round is inert, same as the non-hybrid driver without MTP heads):
    ///   * `mtp_heads` (DeepSeek-V3 style, multiple depths) is preferred when
    ///     present — one sidecar pass per head, yielding a multi-DEPTH draft
    ///     tree via [`crate::speculative::build_mtp_tree`].
    ///   * else the single `nextn_head` (qwen35moe) produces ONE draft row,
    ///     conditioned on the main head's argmax (the best available guess for
    ///     the `+1` token the NextN composition consumes) — a single-depth,
    ///     width-`widths[0]` tree. A poor guess only lowers draft acceptance,
    ///     never correctness (every node is re-verified by the tree walk).
    ///
    /// Reuses [`Self::forward_one_hybrid_to_hidden`] (so the recurrent + KV state
    /// advance is byte-identical to the normal hybrid decode) and the already-
    /// tested [`Self::forward_mtp_head_sidecar`] / [`nextn_compose_logits_f32`]
    /// head compositions. Panics if the model isn't hybrid.
    pub fn forward_one_hybrid_with_draft_heads(
        &self,
        token_id: i32,
        pos: u32,
        kv: &mut KvCache,
        dn_cache: &mut DeltaNetCache,
        main_logits_out: &mut [f32],
    ) -> Vec<Vec<f32>> {
        let cfg = &self.cfg;
        assert!(
            self.weights.is_hybrid(),
            "forward_one_hybrid_with_draft_heads called on non-hybrid model"
        );
        assert_eq!(main_logits_out.len(), cfg.vocab_size);
        let d = cfg.d_model;
        let hidden = self.forward_one_hybrid_to_hidden(token_id, pos, kv, dn_cache);

        // Main LM head (identical to `forward_one_hybrid`'s tail).
        let mut final_norm = vec![0.0f32; d];
        k::rmsnorm_f32_row(
            &hidden,
            &self.weights.output_norm,
            &mut final_norm,
            cfg.rms_eps,
        );
        let lm_head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        {
            let mut had_buf: Vec<f32> = Vec::new();
            let x_in = hadamard_pre(
                self.weights.hadamard.as_ref(),
                lm_head,
                &final_norm,
                &mut had_buf,
            );
            matvec_tensor_dispatch(lm_head, x_in, main_logits_out, cfg.vocab_size, d);
        }

        // Draft heads.
        if let Some(heads) = self.weights.mtp_heads.as_ref() {
            let mut out = Vec::with_capacity(heads.len());
            for head in heads {
                let mut row = vec![0.0f32; cfg.vocab_size];
                self.forward_mtp_head_sidecar(&hidden, head, &mut row);
                out.push(row);
            }
            out
        } else if let Some(head) = self.weights.nextn_head.as_ref() {
            // argmax of the main head = the most likely `+1` token, which the
            // NextN head conditions on to predict `+2`.
            let next_tok = main_logits_out
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                })
                .0 as i32;
            let mut row = vec![0.0f32; cfg.vocab_size];
            nextn_compose_logits_f32(
                &hidden,
                next_tok,
                &self.weights.token_embd,
                head,
                lm_head,
                cfg.rms_eps,
                &mut row,
            );
            vec![row]
        } else {
            Vec::new()
        }
    }

    /// E4 phase 4b: forward + per-MTP-head logits.
    ///
    /// Runs the standard `forward_one` to produce the main next-
    /// token logits (`main_logits_out`) and the final pre-LM-head
    /// hidden state, then for each Multi-Token Prediction head
    /// (DeepSeek-V3 style) runs a sidecar transformer block on
    /// that hidden state to produce per-head "+k-th token" logits
    /// (`mtp_logits_out[k]`).
    ///
    /// Bit-identical to `forward_one` when `n_mtp_heads == 0` —
    /// `mtp_logits_out` must be empty in that case; the function
    /// short-circuits after the main forward.
    ///
    /// Sidecar topology (placeholder pending real-GGUF
    /// validation): single-position self-attention transformer
    /// block reusing the loaded head weights. With softmax over a
    /// single position the attention reduces to `attn_out_h =
    /// V_kv_head(h)` for each query head `h`, so RoPE is a no-op
    /// (cos(p)·V + sin(p)·V_rot cancels for single-position
    /// self-attention). This is the simplest correct forward
    /// against the loaded weights; the engine driver (phase 4c)
    /// consumes the logits as a drafter probability distribution
    /// — if the topology is slightly off, drafts get rejected by
    /// the existing `accept_reject` verifier and the engine
    /// gracefully falls back to n-gram drafting. Switch to a
    /// model-specific topology when a real DeepSeek-V3-MTP GGUF
    /// is available on disk to validate against.
    pub fn forward_one_with_mtp_logits(
        &self,
        token_id: i32,
        pos: u32,
        kv: &mut KvCache,
        main_logits_out: &mut [f32],
        mtp_logits_out: &mut [Vec<f32>],
    ) {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = cfg.d_ff;

        // Run the main forward and snapshot the final hidden
        // state out of the scratch buffer. The inner method
        // populates `scratch.hidden` with the post-last-layer
        // hidden right before the output_norm + LM head step,
        // which is exactly what every MTP head consumes.
        let mut captured_hidden: Vec<f32> = Vec::new();
        let (n_experts, top_k) = cfg
            .moe
            .as_ref()
            .map(|m| (m.n_experts as usize, m.n_experts_used as usize))
            .unwrap_or((0, 0));
        crate::accel::with_forward_scratch(
            d,
            d_q,
            d_kv,
            d_ff,
            head_dim,
            cfg.rope_theta,
            n_experts,
            top_k,
            |scratch| {
                self.forward_one_with_scratch_inner(
                    EmbedInput::Token(token_id),
                    pos,
                    None, // scalar-pos token forward — no multimodal RoPE
                    kv,
                    main_logits_out,
                    scratch,
                );
                captured_hidden = scratch.hidden.clone();
            },
        );

        let heads = match self.weights.mtp_heads.as_ref() {
            Some(h) => h,
            None => {
                assert!(
                    mtp_logits_out.is_empty(),
                    "forward_one_with_mtp_logits: model has no MTP heads but caller requested {} \
                     head logits — pass an empty slice for non-MTP models",
                    mtp_logits_out.len()
                );
                return;
            }
        };
        assert_eq!(
            mtp_logits_out.len(),
            heads.len(),
            "forward_one_with_mtp_logits: mtp_logits_out length must equal n_mtp_heads"
        );

        for (k_idx, head) in heads.iter().enumerate() {
            self.forward_mtp_head_sidecar(&captured_hidden, head, &mut mtp_logits_out[k_idx]);
        }
    }

    /// F7 — paged analog of [`Self::forward_one_with_mtp_logits`]. Runs ONE
    /// paged decode step for `token_id` at `pos` over the slot's paged KV
    /// (`shared` pool), writing the tip token's K/V and the MAIN distribution,
    /// then runs each MTP head's sidecar over the captured post-last-layer
    /// hidden to produce the per-head draft logits the tree builder consumes.
    ///
    /// The main forward REUSES the fully-validated single-token paged inner
    /// ([`Self::forward_one_paged_f32_inner`], driven under the shared store
    /// lock), so the tip's main logits are identical to a classic paged decode
    /// at the same `(token_id, pos)` — only the MTP sidecars are added on top.
    /// Bit-identical to a classic paged decode when the model has no MTP heads
    /// (`mtp_logits_out` must then be empty, matching the contiguous twin). The
    /// caller must `ensure_capacity_shared` the cache to at least `pos + 1`.
    pub fn forward_one_with_mtp_logits_paged(
        &self,
        token_id: i32,
        pos: u32,
        cache: &mut crate::paged_kv_cache::PagedKvCache,
        shared: &crate::shared_paged_kv::SharedPagedKv,
        main_logits_out: &mut [f32],
        mtp_logits_out: &mut [Vec<f32>],
    ) {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = cfg.d_ff;
        let (n_experts, top_k) = cfg
            .moe
            .as_ref()
            .map(|m| (m.n_experts as usize, m.n_experts_used as usize))
            .unwrap_or((0, 0));
        // Decode semantics: the tip attends `[0, pos + 1)` (its own row + the
        // committed prefix), exactly like a single-token paged decode.
        let kv_len = pos as usize + 1;
        let mut captured_hidden: Vec<f32> = Vec::new();
        crate::accel::with_forward_scratch(
            d,
            d_q,
            d_kv,
            d_ff,
            head_dim,
            cfg.rope_theta,
            n_experts,
            top_k,
            |scratch| {
                // The single-token paged inner takes an already-locked store;
                // the batch driver reaches the store through `shared`.
                shared.with_store_mut(|store| {
                    self.forward_one_paged_f32_inner(
                        token_id,
                        pos,
                        cache,
                        store,
                        main_logits_out,
                        scratch,
                        kv_len,
                    );
                });
                // `forward_one_paged_f32_inner` leaves `scratch.hidden` holding
                // the post-last-layer hidden (pre-output-norm) — exactly what
                // every MTP head consumes (same contract the contiguous
                // `forward_one_with_mtp_logits` relies on).
                captured_hidden = scratch.hidden.clone();
            },
        );

        let heads = match self.weights.mtp_heads.as_ref() {
            Some(h) => h,
            None => {
                assert!(
                    mtp_logits_out.is_empty(),
                    "forward_one_with_mtp_logits_paged: model has no MTP heads but caller \
                     requested {} head logits — pass an empty slice for non-MTP models",
                    mtp_logits_out.len()
                );
                return;
            }
        };
        assert_eq!(
            mtp_logits_out.len(),
            heads.len(),
            "forward_one_with_mtp_logits_paged: mtp_logits_out length must equal n_mtp_heads"
        );
        for (k_idx, head) in heads.iter().enumerate() {
            self.forward_mtp_head_sidecar(&captured_hidden, head, &mut mtp_logits_out[k_idx]);
        }
    }

    /// Single-position sidecar transformer block for one MTP head.
    /// See `forward_one_with_mtp_logits` for the topology rationale.
    fn forward_mtp_head_sidecar(&self, hidden_in: &[f32], head: &MtpHead, logits_out: &mut [f32]) {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = cfg.d_ff;
        let block = &head.block;

        let mut h_norm = vec![0f32; d];
        k::rmsnorm_f32_row(hidden_in, &block.attn_norm, &mut h_norm, cfg.rms_eps);

        let mut q_buf = vec![0f32; d_q];
        let mut k_buf = vec![0f32; d_kv];
        let mut v_buf = vec![0f32; d_kv];
        // H3: when `w_qkv_fused` is populated (load-time concat of
        // F32 w_q/w_k/w_v — default-on, A4), do ONE matvec into a
        // stacked output buffer and slice into the three per-head
        // buffers. Saves ~2/3 of the per-attention dispatch overhead.
        // Falls through to three matvecs when the fused tensor isn't
        // available (quantized weights — most models today).
        if let Some(w_qkv) = block.w_qkv_fused.as_ref() {
            let mut qkv_out = vec![0f32; d_q + 2 * d_kv];
            matvec_tensor_dispatch(w_qkv, &h_norm, &mut qkv_out, d_q + 2 * d_kv, d);
            q_buf.copy_from_slice(&qkv_out[..d_q]);
            k_buf.copy_from_slice(&qkv_out[d_q..d_q + d_kv]);
            v_buf.copy_from_slice(&qkv_out[d_q + d_kv..]);
        } else {
            matvec_tensor_dispatch(&block.w_q, &h_norm, &mut q_buf, d_q, d);
            matvec_tensor_dispatch(&block.w_k, &h_norm, &mut k_buf, d_kv, d);
            matvec_tensor_dispatch(&block.w_v, &h_norm, &mut v_buf, d_kv, d);
        }
        if let Some(bq) = block.b_q.as_ref() {
            for (q, b) in q_buf.iter_mut().zip(bq.iter()) {
                *q += *b;
            }
        }
        if let Some(bk) = block.b_k.as_ref() {
            for (k, b) in k_buf.iter_mut().zip(bk.iter()) {
                *k += *b;
            }
        }
        if let Some(bv) = block.b_v.as_ref() {
            for (v, b) in v_buf.iter_mut().zip(bv.iter()) {
                *v += *b;
            }
        }

        // Single-position self-attention reduces to broadcasting V
        // across the GQA query heads: softmax over one position is
        // 1.0, so attn_out_h = V_{kv_head(h)} verbatim.
        let mut attn_out = vec![0f32; d_q];
        let n_gqa = n_heads / n_kv_heads.max(1);
        for h in 0..n_heads {
            let kv_h = h / n_gqa.max(1);
            let src = &v_buf[kv_h * head_dim..(kv_h + 1) * head_dim];
            let dst = &mut attn_out[h * head_dim..(h + 1) * head_dim];
            dst.copy_from_slice(src);
        }

        let mut attn_proj = vec![0f32; d];
        matvec_tensor_dispatch(&block.w_o, &attn_out, &mut attn_proj, d, d_q);

        let mut hidden_mtp: Vec<f32> = hidden_in.to_vec();
        k::add_inplace_f32(&mut hidden_mtp, &attn_proj);

        let mut h_norm_ffn = vec![0f32; d];
        k::rmsnorm_f32_row(&hidden_mtp, &block.ffn_norm, &mut h_norm_ffn, cfg.rms_eps);

        let mut gate_buf = vec![0f32; d_ff];
        let mut up_buf = vec![0f32; d_ff];
        let mut ffn_buf = vec![0f32; d_ff];
        let mut ffn_out = vec![0f32; d];
        matvec_tensor_dispatch(&block.w_gate, &h_norm_ffn, &mut gate_buf, d_ff, d);
        matvec_tensor_dispatch(&block.w_up, &h_norm_ffn, &mut up_buf, d_ff, d);
        k::silu_mul_f32(&gate_buf, &up_buf, &mut ffn_buf);
        matvec_tensor_dispatch(&block.w_down, &ffn_buf, &mut ffn_out, d, d_ff);
        k::add_inplace_f32(&mut hidden_mtp, &ffn_out);

        let mut final_norm = vec![0f32; d];
        k::rmsnorm_f32_row(
            &hidden_mtp,
            &self.weights.output_norm,
            &mut final_norm,
            cfg.rms_eps,
        );
        let lm_head = head
            .lm_head
            .as_ref()
            .or(self.weights.output.as_ref())
            .unwrap_or(&self.weights.token_embd);
        matvec_tensor_dispatch(lm_head, &final_norm, logits_out, cfg.vocab_size, d);
        self.finalize_logits(logits_out);
    }

    /// P10b — one dense/MoE decode step that ALSO captures the post-last-layer
    /// hidden state (pre-output-norm) into `hidden_out`. This is the base
    /// feature an EAGLE draft head fuses with the last token's embedding — the
    /// SAME hidden [`Self::forward_one_with_mtp_logits`] snapshots for the MTP
    /// sidecars (the "second-to-top" feature in EAGLE's terms). Writes the
    /// tip token's K/V at `pos` and the main distribution into
    /// `main_logits_out`, exactly like a classic decode at `(token_id, pos)`,
    /// so the tip forward is bit-identical to a plain decode — only the hidden
    /// capture is added on top. Not valid for hybrid / MLA models (which take
    /// different forward bodies); the EAGLE dispatch gates those out.
    pub fn forward_one_with_hidden(
        &self,
        token_id: i32,
        pos: u32,
        kv: &mut KvCache,
        main_logits_out: &mut [f32],
        hidden_out: &mut Vec<f32>,
    ) {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let d_q = cfg.n_heads * cfg.head_dim;
        let d_kv = cfg.n_kv_heads * cfg.head_dim;
        let d_ff = cfg.d_ff;
        let head_dim = cfg.head_dim;
        let (n_experts, top_k) = cfg
            .moe
            .as_ref()
            .map(|m| (m.n_experts as usize, m.n_experts_used as usize))
            .unwrap_or((0, 0));
        crate::accel::with_forward_scratch(
            d,
            d_q,
            d_kv,
            d_ff,
            head_dim,
            cfg.rope_theta,
            n_experts,
            top_k,
            |scratch| {
                self.forward_one_with_scratch_inner(
                    EmbedInput::Token(token_id),
                    pos,
                    None, // scalar-pos token forward — no multimodal RoPE
                    kv,
                    main_logits_out,
                    scratch,
                );
                // `forward_one_with_scratch_inner` leaves `scratch.hidden`
                // holding the post-last-layer hidden (pre-output-norm) — the
                // EAGLE head's base feature.
                *hidden_out = scratch.hidden.clone();
            },
        );
    }

    /// P10b — run the EAGLE draft head AUTOREGRESSIVELY to grow `depth` draft
    /// positions, returning one full-vocab logit row per position. This is the
    /// drop-in replacement for the MTP path's per-head `mtp_logits_out`: the
    /// returned rows feed [`crate::speculative::build_mtp_tree`] exactly as the
    /// MTP head logits do, so the tree shape + verify + accept are the shared
    /// tree-spec stack and EAGLE only changes the DRAFTING.
    ///
    /// Each step (Li et al. 2024):
    ///   1. Embed the current token (the head's own `token_embd` if present,
    ///      else the base's — the common tied case).
    ///   2. Fuse `concat([embed, feature])` → `h_in` via `fc` (+ optional
    ///      pre-fusion norms + bias). `feature` is the base hidden at step 0,
    ///      then the draft block's OWN output hidden thereafter — EAGLE's
    ///      autoregressive feature recurrence.
    ///   3. Run the single-position decoder block → `(block_hidden, logits)`.
    ///   4. Record `logits`; advance `feature = block_hidden` and `token` to
    ///      the block's argmax (the main-path token `build_mtp_tree` also takes
    ///      as the main node of this level).
    ///
    /// Single-position self-attention reduces to broadcasting V across the GQA
    /// query heads (softmax over one position is 1.0), the same simplification
    /// [`Self::forward_mtp_head_sidecar`] uses — the draft head carries no KV
    /// of its own in this clean-room port, so its autoregressive signal flows
    /// entirely through the fused feature. Fully deterministic (argmax advance,
    /// no RNG).
    pub fn forward_eagle_head(
        &self,
        head: &EagleHead,
        base_hidden: &[f32],
        last_token: i32,
        depth: usize,
    ) -> Vec<Vec<f32>> {
        let d = self.cfg.d_model;
        let vocab = self.cfg.vocab_size;
        assert_eq!(
            base_hidden.len(),
            d,
            "forward_eagle_head: base_hidden must be d_model ({d}), got {}",
            base_hidden.len()
        );
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(depth);
        let mut feature = base_hidden.to_vec();
        let mut token = last_token;
        for _ in 0..depth {
            let embed = self.eagle_embed(head, token);
            let h_in = self.eagle_fuse(head, &embed, &feature);
            let mut logits = vec![0f32; vocab];
            let block_hidden = self.eagle_block_forward(head, &h_in, &mut logits);
            // Advance by the (penalty-free) argmax — the main-path token.
            let mut best_i = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best_i = i;
                }
            }
            rows.push(logits);
            feature = block_hidden;
            token = best_i as i32;
        }
        rows
    }

    /// Embed one token for the EAGLE head: its own `token_embd` table when
    /// present, else the base embedding (the tied case). Returns a `d_model`
    /// row.
    fn eagle_embed(&self, head: &EagleHead, token: i32) -> Vec<f32> {
        let d = self.cfg.d_model;
        match head.token_embd.as_ref() {
            Some(tbl) => {
                let mut out = vec![0f32; d];
                k::embed_lookup_tensor(tbl, &[token], &mut out, d);
                out
            }
            // Reuse the fully-validated base embedding lookup.
            None => self.embed_tokens(&[token]),
        }
    }

    /// Fuse `concat([embed, feature])` → the block's input hidden via the EAGLE
    /// `fc` projection (+ optional pre-fusion RMSNorms and bias). Mirrors the
    /// NextN head's `eh_proj` over `concat([embed_{t+1}, hidden_t])`.
    fn eagle_fuse(&self, head: &EagleHead, embed: &[f32], feature: &[f32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        // Optional pre-fusion norms (applied to a copy so the inputs are
        // untouched for the caller's own bookkeeping).
        let embed_n = match head.embed_norm.as_ref() {
            Some(w) => {
                let mut o = vec![0f32; d];
                k::rmsnorm_f32_row(embed, w, &mut o, cfg.rms_eps);
                o
            }
            None => embed.to_vec(),
        };
        let feat_n = match head.hidden_norm.as_ref() {
            Some(w) => {
                let mut o = vec![0f32; d];
                k::rmsnorm_f32_row(feature, w, &mut o, cfg.rms_eps);
                o
            }
            None => feature.to_vec(),
        };
        // concat([embed, feature]) — embed first, matching eh_proj's order.
        let mut cat = vec![0f32; 2 * d];
        cat[..d].copy_from_slice(&embed_n);
        cat[d..].copy_from_slice(&feat_n);
        let mut h_in = vec![0f32; d];
        matvec_tensor_dispatch(&head.fc, &cat, &mut h_in, d, 2 * d);
        if let Some(b) = head.fc_bias.as_ref() {
            for (h, bv) in h_in.iter_mut().zip(b.iter()) {
                *h += *bv;
            }
        }
        h_in
    }

    /// Single-position EAGLE decoder block over the fused hidden `h_in`
    /// (residual base). Runs attn (GQA V-broadcast) + SwiGLU FFN, writes the
    /// head's logits into `logits_out`, and returns the post-block hidden
    /// (pre-output-norm) — the feature the next autoregressive step fuses.
    /// Same block topology as [`Self::forward_mtp_head_sidecar`], differing
    /// only in that the residual base is the fused `h_in` and the output
    /// norm / LM head fall back to the head's own tensors when present.
    fn eagle_block_forward(
        &self,
        head: &EagleHead,
        h_in: &[f32],
        logits_out: &mut [f32],
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = cfg.d_ff;
        let block = &head.block;

        let mut h_norm = vec![0f32; d];
        k::rmsnorm_f32_row(h_in, &block.attn_norm, &mut h_norm, cfg.rms_eps);

        let mut q_buf = vec![0f32; d_q];
        let mut k_buf = vec![0f32; d_kv];
        let mut v_buf = vec![0f32; d_kv];
        matvec_tensor_dispatch(&block.w_q, &h_norm, &mut q_buf, d_q, d);
        matvec_tensor_dispatch(&block.w_k, &h_norm, &mut k_buf, d_kv, d);
        matvec_tensor_dispatch(&block.w_v, &h_norm, &mut v_buf, d_kv, d);
        if let Some(bq) = block.b_q.as_ref() {
            for (q, b) in q_buf.iter_mut().zip(bq.iter()) {
                *q += *b;
            }
        }
        if let Some(bk) = block.b_k.as_ref() {
            for (kk, b) in k_buf.iter_mut().zip(bk.iter()) {
                *kk += *b;
            }
        }
        if let Some(bv) = block.b_v.as_ref() {
            for (v, b) in v_buf.iter_mut().zip(bv.iter()) {
                *v += *b;
            }
        }

        // Single-position self-attention = broadcast V across the GQA query
        // heads (softmax over one position is 1.0).
        let mut attn_out = vec![0f32; d_q];
        let n_gqa = n_heads / n_kv_heads.max(1);
        for h in 0..n_heads {
            let kv_h = h / n_gqa.max(1);
            let src = &v_buf[kv_h * head_dim..(kv_h + 1) * head_dim];
            let dst = &mut attn_out[h * head_dim..(h + 1) * head_dim];
            dst.copy_from_slice(src);
        }

        let mut attn_proj = vec![0f32; d];
        matvec_tensor_dispatch(&block.w_o, &attn_out, &mut attn_proj, d, d_q);

        let mut hidden = h_in.to_vec();
        k::add_inplace_f32(&mut hidden, &attn_proj);

        let mut h_norm_ffn = vec![0f32; d];
        k::rmsnorm_f32_row(&hidden, &block.ffn_norm, &mut h_norm_ffn, cfg.rms_eps);

        let mut gate_buf = vec![0f32; d_ff];
        let mut up_buf = vec![0f32; d_ff];
        let mut ffn_buf = vec![0f32; d_ff];
        let mut ffn_out = vec![0f32; d];
        matvec_tensor_dispatch(&block.w_gate, &h_norm_ffn, &mut gate_buf, d_ff, d);
        matvec_tensor_dispatch(&block.w_up, &h_norm_ffn, &mut up_buf, d_ff, d);
        k::silu_mul_f32(&gate_buf, &up_buf, &mut ffn_buf);
        matvec_tensor_dispatch(&block.w_down, &ffn_buf, &mut ffn_out, d, d_ff);
        k::add_inplace_f32(&mut hidden, &ffn_out);

        // Project to vocab: the head's own output norm / LM head when present,
        // else tie to the base (output_norm, then output / token_embd).
        let out_norm = head
            .output_norm
            .as_ref()
            .unwrap_or(&self.weights.output_norm);
        let mut final_norm = vec![0f32; d];
        k::rmsnorm_f32_row(&hidden, out_norm, &mut final_norm, cfg.rms_eps);
        let lm_head = head
            .lm_head
            .as_ref()
            .or(self.weights.output.as_ref())
            .unwrap_or(&self.weights.token_embd);
        matvec_tensor_dispatch(lm_head, &final_norm, logits_out, cfg.vocab_size, d);
        self.finalize_logits(logits_out);
        // Return the pre-output-norm hidden — the next step's fused feature.
        hidden
    }

    /// Multi-position forward returning per-position logits.
    ///
    /// Used by speculative decoding: given K+1 candidate tokens
    /// starting at `start_pos`, run a forward at each position and
    /// record the logits at every step. Output `logits_out` is
    /// row-major `[tokens.len(), vocab_size]` — row `i` is the
    /// logits AFTER processing `tokens[i]`, i.e. the distribution
    /// the model would sample the (i+1)-th token from.
    ///
    /// Implementation today is a sequential loop over `forward_one`
    /// — same KV writes, same logits, same correctness as the
    /// normal decode path. Works for any KV dtype that `forward_one`
    /// supports.
    ///
    /// Perf note: this is the **serial-verify** variant. The
    /// speculation win comes from doing all K+1 forwards as ONE
    /// batched call (sharing the attention QK^T-V multiply and
    /// matvec setup). The batched variants per KV dtype are a
    /// follow-up; this primitive lets the engine driver be written
    /// + tested end-to-end against any KV dtype today.
    pub fn forward_speculation(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        assert_eq!(
            logits_out.len(),
            tokens.len() * cfg.vocab_size,
            "forward_speculation: logits_out must be sized to tokens.len() * vocab_size"
        );
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = start_pos + i as u32;
            let lo = i * cfg.vocab_size;
            let hi = lo + cfg.vocab_size;
            self.forward_one(tok, pos, kv, &mut logits_out[lo..hi]);
        }
    }

    /// Forward-pass body that operates against caller-provided
    /// scratch buffers (so the outer call can reuse them across
    /// many tokens, e.g. during prefill). Public-but-undocumented
    /// inherent so the engine layer can call it with its own
    /// pre-sized scratch in future batched-prefill work without
    /// re-entering the TLS path.
    ///
    /// `input` is either a [`EmbedInput::Token`] (the normal path:
    /// look up the embedding from `token_embd` for the given id)
    /// or a [`EmbedInput::Embed`] (the V-6b-3 vision splice path:
    /// the caller has already produced the d_model embedding row
    /// — e.g. it's a projected image-patch vector — and we should
    /// use it verbatim).
    fn forward_one_with_scratch_inner(
        &self,
        input: EmbedInput<'_>,
        pos: u32,
        mrope_pos: Option<[u32; 3]>,
        kv: &mut KvCache,
        logits_out: &mut [f32],
        scratch: &mut crate::accel::ForwardScratch,
    ) {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = cfg.d_ff;

        // Take ownership of each scratch buffer for the duration
        // of this call so the body's `&mut buf` / `&buf` patterns
        // resolve cleanly through `Vec<f32>` auto-deref. Swap back
        // at the bottom of the function. (On panic the scratch is
        // left empty and the next call from this thread allocates
        // fresh — fine, panic in forward_one is not the normal
        // path anyway.)
        let mut hidden = std::mem::take(&mut scratch.hidden);
        let mut h_norm = std::mem::take(&mut scratch.h_norm);
        let mut q_buf = std::mem::take(&mut scratch.q_buf);
        let mut k_buf = std::mem::take(&mut scratch.k_buf);
        let mut v_buf = std::mem::take(&mut scratch.v_buf);
        let mut attn_out = std::mem::take(&mut scratch.attn_out);
        let mut attn_proj = std::mem::take(&mut scratch.attn_proj);
        let mut gate_buf = std::mem::take(&mut scratch.gate_buf);
        let mut up_buf = std::mem::take(&mut scratch.up_buf);
        let mut ffn_buf = std::mem::take(&mut scratch.ffn_buf);
        let mut ffn_out = std::mem::take(&mut scratch.ffn_out);
        let mut final_norm = std::mem::take(&mut scratch.final_norm);
        // MoE-only scratch (zero-sized on dense models). Taking
        // them out unconditionally keeps the put-back loop at the
        // bottom symmetric; for dense models these stay zero-len
        // and the MoE branch never runs.
        let mut moe_down = std::mem::take(&mut scratch.moe_down);
        let mut moe_expert_logits = std::mem::take(&mut scratch.moe_expert_logits);
        let mut moe_routed_picks = std::mem::take(&mut scratch.moe_routed_picks);
        // TurboQuant KV-append scratch — see ForwardScratch::tq_scratch.
        let mut tq_scratch = std::mem::take(&mut scratch.tq_scratch);
        // P2: SWA-ring gather slabs, taken as locals (like the others) so the
        // ring decode arm can grow them without re-borrowing `scratch` — the
        // `rope_inv_freq` reference below holds an immutable borrow of `scratch`
        // for the whole layer loop, so a `scratch.*` mutable call there won't
        // compile. Put back at the bottom.
        let mut swa_ring_k = std::mem::take(&mut scratch.swa_ring_k);
        let mut swa_ring_v = std::mem::take(&mut scratch.swa_ring_v);

        match input {
            EmbedInput::Token(token_id) => {
                k::embed_lookup_tensor(&self.weights.token_embd, &[token_id], &mut hidden, d);
            }
            EmbedInput::Embed(row) => {
                // Pre-populated input row (vision splice path). Length
                // is asserted at the public entry point so this just
                // copies. `hidden` was sized to `d` by the scratch
                // allocator above.
                debug_assert_eq!(row.len(), d);
                hidden.clear();
                hidden.extend_from_slice(row);
            }
        }

        // Pre-computed RoPE inv_freq table is cached on the scratch
        // by `prepare` keyed on (head_dim, rope_theta) — invariant
        // for a loaded model so first call computes, the rest reuse.
        let rope_inv_freq = &scratch.rope_inv_freq;

        let cur_pos = pos as usize;

        // Total layer count works the same for dense + MoE: only
        // one of `blocks` / `moe_blocks` is populated at a time.
        let n_blocks_total = if let Some(mbs) = self.weights.moe_blocks.as_ref() {
            mbs.len()
        } else {
            self.weights.blocks.len()
        };
        // Layer count for THIS forward: the T5d early-exit draft depth (set by the
        // engine's self-speculation draft phase) takes precedence, else the
        // optional `RUSTLLAMA_MAX_LAYERS` debug cap, else all layers. Early-exit
        // runs layers `0..L` then the usual final-norm + lm_head below, yielding
        // draft logits from an intermediate hidden; `None`/unset ⇒ byte-identical
        // full-depth forward. Only this dense forward honors it.
        let n_layers_used = crate::accel::early_exit_layer()
            .or_else(crate::accel::max_layers_cap)
            .unwrap_or(n_blocks_total)
            .min(n_blocks_total);

        // Cache MoE config locally so the FFN branch below has
        // n_experts + n_experts_used without re-resolving cfg.moe
        // per layer.
        let moe_cfg = self.cfg.moe.clone();

        // Per-layer `&dyn AttnBlock` is materialized inside the loop
        // (a single 16-byte fat-pointer assembly per layer per token,
        // no allocation). The previous shape collected these into a
        // `Vec<&dyn AttnBlock>` per call, which allocated `n_layers
        // * 16` bytes + the Vec header on every token. Eliminating
        // that allocation gains ~32 micro-allocs per generated
        // token on a Llama-7B-class model.
        let moe_blocks = self.weights.moe_blocks.as_ref();
        // #2 Phase 2 (cross-layer hidden residency): true while the device
        // activation slab holds the current `hidden` and the HOST `hidden` is
        // stale. Only the pure-dense combined device path keeps it resident
        // across layers; a drain to host happens on a fallback to the host path
        // and once after the loop (before the final norm / LM head).
        let mut hidden_dev_resident = false;

        // T2 (B1) — whole-decode-step CUDA graph: for a PURE-DENSE model with all
        // preconditions met and the step verdict (or `RUSTLLAMA_DEV_STEP_GRAPH`)
        // on, capture EVERY layer's attention+FFN chain into ONE cudaGraph and
        // replay it with ONE launch + ONE sync for the whole token — vs the
        // per-layer loop below (one launch + one sync PER layer). Declines (runs
        // the per-layer loop unchanged) on ANY unmet precondition; the captured
        // kernels are the SAME as the per-layer graph, so it's fail-soft +
        // parity-probed on-device. On success `hidden` (post-stack) + the F32 host
        // KV caches are synced, and the per-layer loop is skipped (bound → 0).
        let step_graph_ran = moe_cfg.is_none()
            && moe_blocks.is_none()
            && !self.weights.is_hybrid()
            && !self.weights.is_mla()
            && self.yarn_mscale == 1.0
            && n_layers_used == n_blocks_total
            && crate::accel::stream_window() == 0
            && crate::accel::dev_step_cuda_graph_enabled()
            && (0..n_layers_used).all(|i| {
                kv.layer_cap(i) == kv.max_ctx && matches!(kv.layers[i], KvLayer::F32 { .. })
            })
            && {
                let dev_softcap = self.cfg.attn_logit_softcapping.unwrap_or(0.0);
                let mut sls: Vec<rustllama_kernels_cuda::StepLayer> =
                    Vec::with_capacity(n_layers_used);
                let mut gathered = true;
                for i in 0..n_layers_used {
                    let b = &self.weights.blocks[i];
                    if b.b_q().is_some() || b.b_k().is_some() || b.b_v().is_some() {
                        gathered = false;
                        break;
                    }
                    match crate::accel::make_step_layer(
                        b.w_q(),
                        b.w_k(),
                        b.w_v(),
                        b.w_o(),
                        Some((&b.w_gate, &b.w_up, &b.w_down, d_ff)),
                        b.attn_norm(),
                        b.ffn_norm(),
                        b.q_norm(),
                        b.k_norm(),
                        &self.rope_inv_freq,
                        self.cfg.swa_window(i) as u32,
                        dev_softcap,
                    ) {
                        Some(sl) => sls.push(sl),
                        None => {
                            gathered = false;
                            break;
                        }
                    }
                }
                if !gathered {
                    false
                } else {
                    let mctx = kv.max_ctx;
                    let mut kc: Vec<&mut [f32]> = Vec::with_capacity(n_layers_used);
                    let mut vc: Vec<&mut [f32]> = Vec::with_capacity(n_layers_used);
                    for l in kv.layers.iter_mut().take(n_layers_used) {
                        if let KvLayer::F32 { k, v } = l {
                            kc.push(&mut k[..]);
                            vc.push(&mut v[..]);
                        }
                    }
                    crate::accel::try_decode_step_cuda_graph_f32(
                        &sls,
                        &mut hidden,
                        &mut kc,
                        &mut vc,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        d,
                        cur_pos,
                        mctx,
                        cfg.rms_eps,
                    )
                }
            };

        for layer_idx in 0..(if step_graph_ran { 0 } else { n_layers_used }) {
            let block: &dyn AttnBlock = if let Some(mbs) = moe_blocks {
                &mbs[layer_idx]
            } else {
                &self.weights.blocks[layer_idx]
            };
            // Hybrid placement: tell the `try_*_usm_f32` dispatch
            // ladder which layer we're on. Helpers consult
            // `accel::gpu_active_for_current_layer()` (= layer_idx
            // < accel::n_gpu_layers()) to short-circuit the GPU
            // path for layers past the `[inference].n_gpu_layers`
            // cutoff.
            crate::accel::set_current_layer_idx(layer_idx as u32);
            // AirLLM streaming: tag the CUDA weight cache with this layer so its
            // bounded-window LRU groups + ages this layer's weights for
            // whole-layer eviction, then drive the prefetch double-buffer. Gated
            // on `stream_window() > 0` so the normal path never takes the extra
            // cache lock.
            if crate::accel::stream_window() > 0 {
                crate::accel::cuda_set_layer_tag(layer_idx as u32);
                // Phase 2: wait for THIS layer's prefetch (issued last iteration)
                // to land on the copy stream, then issue the NEXT layer's prefetch
                // so its H2D overlaps this layer's compute. Dense blocks only
                // (MoE experts stream per-token, Phase 5); the next block's QKV is
                // fused or split, and `cuda_prefetch_layer` skips any non-packable
                // / owned tensor.
                crate::accel::cuda_wait_prefetch();
                if layer_idx + 1 < n_layers_used {
                    if let Some(nb) = self.weights.blocks.get(layer_idx + 1) {
                        let mut ts: Vec<&Tensor> = vec![&nb.w_o, &nb.w_gate, &nb.w_up, &nb.w_down];
                        if let Some(qkv) = &nb.w_qkv_fused {
                            ts.push(qkv);
                        } else {
                            ts.extend([&nb.w_q, &nb.w_k, &nb.w_v]);
                        }
                        crate::accel::cuda_prefetch_layer((layer_idx + 1) as u32, &ts);
                    }
                }
            }
            // P2: is this an F32 SWA ring layer? A ring layer's slab is
            // `cap`-strided and ring-addressed (`pos % cap`), so the
            // device-resident decode blocks + the USM/CUDA GPU flash hooks
            // (which assume a `max_ctx`-strided contiguous slab written at
            // `cur_pos`) MUST be declined for it; its attention runs through
            // the ring write + compact gather + CPU flash below. Non-ring
            // layers (`cap == max_ctx`) keep every existing path unchanged.
            let cap_l = kv.layer_cap(layer_idx);
            let is_ring_l = cap_l < kv.max_ctx;
            // ---- attention ----
            // Dispatch ladder: USM-resident SYCL (shared stream
            // with the attention kernel, no per-call alloc) →
            // host-pointer SYCL (per-call alloc + memcpy) →
            // CPU kernel. The `try_*` functions return `true` on
            // success and we short-circuit; on `false` we fall
            // through to the next tier.
            //
            // #2 on-device activation-residency attention (perf-gated auto, no
            // env var — on only where `tune --dev-attn` measured it faster, via
            // `accel::dev_attn_enabled` → verdict `attn:dev_resident_block`): run
            // the WHOLE attention block on the GPU with the activation resident
            // on device (rmsnorm→qkv→qk-head-norm→rope→flash→o→residual+norm, no
            // per-op host round-trip). On success
            // `hidden` is updated (residual done), `h_norm` holds the post-attn
            // RMSNorm (the FFN input), and the host KV cache is appended +
            // coherent — so the whole host attention path below is skipped. F32
            // KV + no qkv bias only; otherwise falls through unchanged.
            let kv_max_ctx = kv.max_ctx;
            // #2 Phase 1 — dense FFN residency: for a PURE-dense layer (no MoE,
            // not hybrid-SSM) also hand the device block the SwiGLU FFN weights so
            // gate/up/silu/down + the FFN residual run GPU-resident too. MoE /
            // hybrid layers pass `None` (attention-only; their FFN stays on the
            // host / grouped path). `.get` avoids indexing a MoE model's empty
            // `blocks`. When the FFN ran on device, `device_dense_ffn` skips the
            // host FFN below.
            let dense_ffn_arg: Option<(&Tensor, &Tensor, &Tensor, usize)> =
                if moe_cfg.is_none() && !self.weights.is_hybrid() {
                    self.weights
                        .blocks
                        .get(layer_idx)
                        .map(|db| (&db.w_gate, &db.w_up, &db.w_down, d_ff))
                } else {
                    None
                };
            let mut device_dense_ffn = false;
            // Cross-layer residency (#2 Phase 2) applies only to the pure-dense
            // combined path (every layer device-resident). Upload hidden only
            // when it isn't already resident from the previous layer; keep it
            // resident (no download) across dense layers. The MoE / attention-
            // only path stays per-layer (`true, true`) since its host FFN needs
            // host hidden each layer.
            let cross_layer = dense_ffn_arg.is_some();
            let upload_hidden = if cross_layer {
                !hidden_dev_resident
            } else {
                true
            };
            let download_hidden = !cross_layer;
            // Feature 1a (SYCL sibling of the #2 CUDA block): set when the SYCL
            // device-resident block handled this layer. It is attention-only and
            // per-layer (no cross-layer hidden residency), so it must NOT set
            // `hidden_dev_resident` below.
            let mut device_attn_is_sycl = false;
            // Feature 1c (Metal/MLX sibling): set when the MLX device-resident
            // block handled this layer. Attention-only + per-layer exactly like
            // the SYCL block, so it likewise must NOT set `hidden_dev_resident`.
            let mut device_attn_is_mlx = false;
            // The fused CUDA decode block (and its SYCL sibling) rotate with the
            // inv-freq table but do NOT apply the YaRN attention `mscale` (they
            // fuse the Q·K score on-device, so Q can't be post-scaled). Run them
            // only when `mscale` is a no-op; YaRN models fall to the software path
            // below, which applies it. Non-scaled + linear/NTK keep the device
            // blocks, fed `self.rope_inv_freq` — the scaled table, byte-identical
            // to the base table when the model carries no scaling.
            //
            // Gemma2 attention-logit soft-capping AND sliding-window attention are
            // now THREADED into the device blocks (the flash-decode inside each
            // block applies `s = c*tanh(s/c)` after the scale and trims to the last
            // `window` positions, bit-exact to the CPU capped+windowed kernel), so a
            // Gemma2 / Mistral-SWA model runs device-resident too. The per-layer
            // `dev_window` (0 on a global/non-SWA layer) + `dev_softcap` (0 when the
            // model has no cap) are passed to every block below; passthrough at 0/0
            // keeps non-SWA / non-Gemma2 layers byte-identical.
            //
            // The YaRN attention `mscale` is still a no-go for the device blocks:
            // they fuse the Q·K score on-device, so Q can't be post-scaled. Run them
            // only when `mscale` is a no-op; YaRN models fall to the software path
            // below, which applies it. Non-scaled + linear/NTK keep the device
            // blocks, fed `self.rope_inv_freq` — the scaled table, byte-identical to
            // the base table when the model carries no scaling.
            let device_attn = if self.yarn_mscale == 1.0 && !is_ring_l {
                if let KvLayer::F32 { k, v } = &mut kv.layers[layer_idx] {
                    // Per-layer SWA window (0 ⇒ full causal) + Gemma2 attention-logit
                    // soft-cap (0.0 ⇒ no cap), threaded into every device block's
                    // internal flash-decode (see the block fns). Before this, an SWA
                    // model reaching the device block would have attended the FULL
                    // history (no window arg) — WRONG output; the window arg closes
                    // that (the engine already coerces SWA/softcap models to F32 KV,
                    // which is the only KV dtype the device block accepts, so they
                    // genuinely reach here).
                    let dev_window = self.cfg.swa_window(layer_idx) as u32;
                    let dev_softcap = self.cfg.attn_logit_softcapping.unwrap_or(0.0);
                    // Feature 1b: PREFER the CUDA-GRAPH block — it captures this
                    // layer's whole device chain into a cudaGraph and replays it
                    // as ONE launch (gated on `attn:dev_resident_block_cuda_graph`).
                    // It declines (false) when its verdict is off, or falls back
                    // internally + returns false if a capture/launch fails, leaving
                    // the host buffers untouched — in which case the eager CUDA
                    // block (itself gated on `attn:dev_resident_block`) runs next.
                    // AirLLM streaming: SKIP the CUDA-graph block — it caches this
                    // layer's whole device chain with CAPTURED weight DEVICE
                    // POINTERS, which the streaming LRU evicts + re-uploads
                    // (pointers dangle → illegal access on replay). The eager block
                    // below re-reads the cache's CURRENT device pointers every call,
                    // so it is safe under eviction; fall straight through to it.
                    let (mut attn_ran, mut ffn_ran) = if crate::accel::stream_window() > 0 {
                        (false, false)
                    } else {
                        crate::accel::try_decode_attn_block_cuda_graph_f32(
                            block.w_q(),
                            block.w_k(),
                            block.w_v(),
                            block.w_o(),
                            block.attn_norm(),
                            block.ffn_norm(),
                            block.q_norm(),
                            block.k_norm(),
                            block.b_q().is_some() || block.b_k().is_some() || block.b_v().is_some(),
                            &self.rope_inv_freq,
                            &mut hidden,
                            &mut h_norm,
                            &mut k_buf,
                            &mut v_buf,
                            k,
                            v,
                            dense_ffn_arg,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            d,
                            cur_pos,
                            kv_max_ctx,
                            cfg.n_layers,
                            layer_idx,
                            cfg.rms_eps,
                            dev_window,
                            dev_softcap,
                            upload_hidden,
                            download_hidden,
                        )
                    };
                    if !attn_ran {
                        let (a_eager, f_eager) = crate::accel::try_decode_attn_block_cuda_f32(
                            block.w_q(),
                            block.w_k(),
                            block.w_v(),
                            block.w_o(),
                            block.attn_norm(),
                            block.ffn_norm(),
                            block.q_norm(),
                            block.k_norm(),
                            block.b_q().is_some() || block.b_k().is_some() || block.b_v().is_some(),
                            &self.rope_inv_freq,
                            &mut hidden,
                            &mut h_norm,
                            &mut k_buf,
                            &mut v_buf,
                            k,
                            v,
                            dense_ffn_arg,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            d,
                            cur_pos,
                            kv_max_ctx,
                            cfg.n_layers,
                            layer_idx,
                            cfg.rms_eps,
                            dev_window,
                            dev_softcap,
                            upload_hidden,
                            download_hidden,
                        );
                        attn_ran = a_eager;
                        ffn_ran = f_eager;
                    }
                    device_dense_ffn = ffn_ran;
                    if attn_ran {
                        true
                    } else {
                        // Feature 1a: the SYCL device-resident DECODE block, tried
                        // when the CUDA block didn't run (no CUDA, or it declined).
                        // Inert off SYCL (the gate returns false → byte-identical
                        // path). Now also handed `dense_ffn_arg` so a pure-dense
                        // layer's SwiGLU FFN runs USM-resident too (gate/up/silu/
                        // down/residual on the same submit-chain); `sycl_ffn` then
                        // drives `device_dense_ffn` so the host FFN below is
                        // skipped. It stays attention-only (sycl_ffn=false, host
                        // FFN runs) for MoE/hybrid layers or non-dispatchable FFN
                        // weights. Fed the same scaled rope table as the CUDA block.
                        let (sycl_ran, sycl_ffn) = crate::accel::try_decode_attn_block_sycl_f32(
                            block.w_q(),
                            block.w_k(),
                            block.w_v(),
                            block.w_o(),
                            block.attn_norm(),
                            block.ffn_norm(),
                            block.q_norm(),
                            block.k_norm(),
                            block.b_q().is_some() || block.b_k().is_some() || block.b_v().is_some(),
                            &self.rope_inv_freq,
                            &mut hidden,
                            &mut h_norm,
                            &mut k_buf,
                            &mut v_buf,
                            k,
                            v,
                            dense_ffn_arg,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            d,
                            cur_pos,
                            kv_max_ctx,
                            cfg.n_layers,
                            layer_idx,
                            cfg.rms_eps,
                            dev_window,
                            dev_softcap,
                        );
                        device_attn_is_sycl = sycl_ran;
                        if sycl_ran {
                            device_dense_ffn = sycl_ffn;
                            true
                        } else {
                            // Feature 1c: the Metal/MLX device-resident DECODE
                            // block, tried when neither CUDA nor SYCL ran. Inert
                            // off Apple Silicon (the gate returns false →
                            // byte-identical path). Now also handed `dense_ffn_arg`
                            // so a pure-dense layer's SwiGLU FFN runs in the same
                            // Metal command buffer; `mlx_ffn` then drives
                            // `device_dense_ffn` so the host FFN below is skipped.
                            // Attention-only (mlx_ffn=false, host FFN runs) for
                            // MoE/hybrid layers or non-dispatchable FFN weights.
                            // Fed the same scaled rope table as the CUDA/SYCL blocks.
                            let (mlx_ran, mlx_ffn) = crate::accel::try_decode_attn_block_mlx_f32(
                                block.w_q(),
                                block.w_k(),
                                block.w_v(),
                                block.w_o(),
                                block.attn_norm(),
                                block.ffn_norm(),
                                block.q_norm(),
                                block.k_norm(),
                                block.b_q().is_some()
                                    || block.b_k().is_some()
                                    || block.b_v().is_some(),
                                &self.rope_inv_freq,
                                &mut hidden,
                                &mut h_norm,
                                &mut k_buf,
                                &mut v_buf,
                                k,
                                v,
                                dense_ffn_arg,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                d,
                                cur_pos,
                                kv_max_ctx,
                                cfg.n_layers,
                                layer_idx,
                                cfg.rms_eps,
                                dev_window,
                                dev_softcap,
                            );
                            device_attn_is_mlx = mlx_ran;
                            if mlx_ran {
                                device_dense_ffn = mlx_ffn;
                            }
                            mlx_ran
                        }
                    }
                } else {
                    false
                }
            } else {
                false
            };
            if device_attn {
                // Cross-layer: hidden stays on device (host stale); per-layer:
                // it was downloaded, so host is authoritative. The SYCL + MLX
                // blocks are always per-layer, so they never keep hidden
                // resident.
                hidden_dev_resident = cross_layer && !device_attn_is_sycl && !device_attn_is_mlx;
            } else if hidden_dev_resident {
                // The device path declined this layer but the slab holds the
                // current hidden (a prior dense layer left it resident); refresh
                // the host copy before the host attention/FFN path runs.
                crate::accel::fetch_resident_hidden_cuda_f32(&mut hidden, d);
                hidden_dev_resident = false;
            }
            if !device_attn {
                if !crate::accel::try_rmsnorm_usm_f32(
                    &hidden,
                    block.attn_norm(),
                    &mut h_norm,
                    d,
                    cfg.rms_eps,
                ) && !crate::accel::try_rmsnorm_f32(
                    &hidden,
                    block.attn_norm(),
                    &mut h_norm,
                    d,
                    cfg.rms_eps,
                ) {
                    k::rmsnorm_f32_row(&hidden, block.attn_norm(), &mut h_norm, cfg.rms_eps);
                }
                matvec_tensor_dispatch(block.w_q(), &h_norm, &mut q_buf, d_q, d);
                matvec_tensor_dispatch(block.w_k(), &h_norm, &mut k_buf, d_kv, d);
                matvec_tensor_dispatch(block.w_v(), &h_norm, &mut v_buf, d_kv, d);
                if let Some(bq) = block.b_q() {
                    k::add_inplace_f32(&mut q_buf, bq);
                }
                if let Some(bk) = block.b_k() {
                    k::add_inplace_f32(&mut k_buf, bk);
                }
                if let Some(bv) = block.b_v() {
                    k::add_inplace_f32(&mut v_buf, bv);
                }
                // Qwen3 per-head Q/K RMSNorm, before RoPE. No-op on every
                // non-Qwen3 arch (q_norm/k_norm are None).
                apply_qk_head_norm(
                    block.q_norm(),
                    block.k_norm(),
                    &mut q_buf,
                    &mut k_buf,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    cfg.rms_eps,
                );
                // Pick RoPE convention by environment variable so we can A/B
                // test against models without rebuilding (default = neox).
                // Interleaved layout has no SYCL kernel today (the GPU
                // kernel ships only the neox/half-split convention), so
                // that path stays CPU-only.
                if let (Some(sections), Some(triple)) =
                    (self.cfg.rope_dimension_sections.as_deref(), mrope_pos)
                {
                    // M5-tail: multimodal sectioned RoPE (Qwen2-VL / Qwen2.5-VL).
                    // The per-token 3-D (t,h,w) position drives a sectioned NeoX
                    // rotation: head-dim pairs split across the [t,h,w] axes by
                    // `sections`. Image tokens carry their patch-grid coords;
                    // text tokens carry (p,p,p), which `rope_inplace_neox_
                    // sectioned` makes BIT-identical to 1-D RoPE. CPU-only like
                    // the rope_scaling detour below (the GPU rope kernels are
                    // scalar-pos) — RoPE is a tiny slice of forward compute, so
                    // the detour is negligible. Gated on BOTH sections present
                    // (VLM only) AND a 3-D position supplied (the VLM mRoPE
                    // prefill path), so every non-VLM / text-only forward — and
                    // every scalar-`pos` caller — is byte-identical.
                    k::rope_inplace_neox_sectioned(
                        &mut q_buf,
                        n_heads,
                        head_dim,
                        sections,
                        &triple,
                        cfg.rope_theta,
                    );
                    k::rope_inplace_neox_sectioned(
                        &mut k_buf,
                        n_kv_heads,
                        head_dim,
                        sections,
                        &triple,
                        cfg.rope_theta,
                    );
                    // Match `rope_inplace`'s YaRN post-scale (no-op at 1.0, which
                    // is the mRoPE-VLM case) so the two paths stay consistent.
                    if self.yarn_mscale != 1.0 {
                        for x in q_buf.iter_mut() {
                            *x *= self.yarn_mscale;
                        }
                        for x in k_buf.iter_mut() {
                            *x *= self.yarn_mscale;
                        }
                    }
                } else if self.cfg.rope_scaling.is_some() {
                    // Context-extension scaling (YaRN/NTK/linear): drive RoPE from
                    // the precomputed scaled table on the CPU path. Correctness
                    // first — the GPU rope ladder below is scalar-theta only (the
                    // cached `scratch.rope_inv_freq` is the UNSCALED base table), so
                    // threading the scaled table through the USM kernel is a
                    // write-blind follow-up. RoPE is a tiny slice of decode compute,
                    // so the CPU detour here is negligible.
                    self.rope_inplace(&mut q_buf, n_heads, head_dim, pos);
                    self.rope_inplace(&mut k_buf, n_kv_heads, head_dim, pos);
                } else if rope_interleaved_enabled() {
                    k::rope_inplace_interleaved(&mut q_buf, n_heads, head_dim, pos, cfg.rope_theta);
                    k::rope_inplace_interleaved(
                        &mut k_buf,
                        n_kv_heads,
                        head_dim,
                        pos,
                        cfg.rope_theta,
                    );
                } else {
                    // Same USM → host-pointer → CPU dispatch ladder as
                    // rmsnorm. The USM path uses the cached inv-freq
                    // table on its own stream; the host-pointer path
                    // takes the local `rope_inv_freq` we built earlier.
                    if !crate::accel::try_rope_usm_f32(
                        &mut q_buf,
                        n_heads,
                        head_dim,
                        pos,
                        cfg.rope_theta,
                    ) && !crate::accel::try_rope_f32(
                        &mut q_buf,
                        n_heads,
                        head_dim,
                        pos,
                        &rope_inv_freq,
                    ) {
                        k::rope_inplace_neox(&mut q_buf, n_heads, head_dim, pos, cfg.rope_theta);
                    }
                    if !crate::accel::try_rope_usm_f32(
                        &mut k_buf,
                        n_kv_heads,
                        head_dim,
                        pos,
                        cfg.rope_theta,
                    ) && !crate::accel::try_rope_f32(
                        &mut k_buf,
                        n_kv_heads,
                        head_dim,
                        pos,
                        &rope_inv_freq,
                    ) {
                        k::rope_inplace_neox(&mut k_buf, n_kv_heads, head_dim, pos, cfg.rope_theta);
                    }
                }

                // Append K, V to the cache at slot `cur_pos`. The Q8_0 path
                // quantizes per-row using the row's absmax.
                let max_ctx = kv.max_ctx;
                let kv_bias_l = kv.kv_bias.clone();
                let kv_layer = &mut kv.layers[layer_idx];
                let kv_len = cur_pos + 1;
                match kv_layer {
                    KvLayer::F32 { k, v } if is_ring_l => {
                        // P2 SWA RING decode. Slab is `cap`-strided, ring-addressed.
                        // `cap == swa_window` here (a ring exists only when the window
                        // is strictly inside the context), so the live window is the
                        // whole ring. Write the new row at `cur_pos % cap`, gather the
                        // live window [cur_pos-win_len+1, cur_pos] into the compact
                        // scratch in ASCENDING position order, then run the CPU flash /
                        // one-step kernel over it with stride = `win_len` and window 0.
                        // Every gathered position is in-window, so window 0 (full over
                        // the compact set) reproduces the unbounded masked SWA path's
                        // attended set in the same order ⇒ bit-identical logits, while
                        // the slab itself is only `cap` positions (the memory win).
                        // The device-resident blocks + GPU flash hooks + CUDA mirror
                        // seed are all declined above for a ring layer (they assume a
                        // `max_ctx` slab written at `cur_pos`), so none run here.
                        let cap = cap_l;
                        let softcap = cfg.attn_logit_softcapping.unwrap_or(0.0);
                        let slot = cur_pos % cap;
                        for h in 0..n_kv_heads {
                            let dst = (h * cap + slot) * head_dim;
                            k[dst..dst + head_dim]
                                .copy_from_slice(&k_buf[h * head_dim..(h + 1) * head_dim]);
                            v[dst..dst + head_dim]
                                .copy_from_slice(&v_buf[h * head_dim..(h + 1) * head_dim]);
                        }
                        let win_len = kv_len.min(cap);
                        let first_pos = kv_len - win_len; // = cur_pos + 1 - win_len
                        let used = n_kv_heads * win_len * head_dim;
                        // Grow the (locally-owned) gather slabs; the gather fully
                        // overwrites `[..used]` so no zero-fill is needed.
                        if swa_ring_k.len() < used {
                            swa_ring_k.resize(used, 0.0);
                            swa_ring_v.resize(used, 0.0);
                        }
                        for h in 0..n_kv_heads {
                            for j in 0..win_len {
                                let pos = first_pos + j;
                                let src = (h * cap + (pos % cap)) * head_dim;
                                let dst = (h * win_len + j) * head_dim;
                                swa_ring_k[dst..dst + head_dim]
                                    .copy_from_slice(&k[src..src + head_dim]);
                                swa_ring_v[dst..dst + head_dim]
                                    .copy_from_slice(&v[src..src + head_dim]);
                            }
                        }
                        let flash_min = flash_kv_len_min();
                        if crate::accel::flash_attention_enabled() && win_len >= flash_min {
                            k::gqa_attention_flash_decode_capped_windowed(
                                &q_buf,
                                &swa_ring_k[..used],
                                &swa_ring_v[..used],
                                &mut attn_out,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                win_len,
                                win_len,
                                0,
                                softcap,
                            );
                        } else {
                            k::gqa_attention_one_step_capped_windowed(
                                &q_buf,
                                &swa_ring_k[..used],
                                &swa_ring_v[..used],
                                &mut attn_out,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                win_len,
                                win_len,
                                0,
                                softcap,
                            );
                        }
                    }
                    KvLayer::F32 { k, v } => {
                        for h in 0..n_kv_heads {
                            let dst = (h * max_ctx + cur_pos) * head_dim;
                            k[dst..dst + head_dim]
                                .copy_from_slice(&k_buf[h * head_dim..(h + 1) * head_dim]);
                            v[dst..dst + head_dim]
                                .copy_from_slice(&v_buf[h * head_dim..(h + 1) * head_dim]);
                        }
                        // Seed the CUDA decode KV mirror with the prefill
                        // history on the first decode step after a prefill (a
                        // no-op once resident). Without it the native-CUDA
                        // decode path's resident-mirror gap check declines for
                        // the whole generation (decode starts at
                        // pos = prompt_len against an empty mirror), so decode
                        // attention silently runs on the CPU while the matvec
                        // stays on the GPU.
                        crate::accel::decode_seed_kv_f32(
                            k,
                            v,
                            layer_idx,
                            cur_pos as u32,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            max_ctx as u32,
                            cfg.n_layers as u32,
                        );
                        // USM-resident SYCL path first (opt-in via
                        // `RUSTLLAMA_USM_ATTN=1`). On integrated GPUs
                        // (Iris Xe + shared LPDDR) the K/V cache stays
                        // in USM across the entire generate() call —
                        // the host writes we did above are page-mapped
                        // to the GPU. The kernel reads them in place.
                        // Returns true if the kernel ran; false falls
                        // through to the CPU flash/standard path below.
                        // Sliding-window span for this layer (0 ⇒ full attention) +
                        // Gemma2 attention-logit soft-cap (0.0 ⇒ no cap). Both are now
                        // THREADED into the GPU flash kernel (bit-exact to the CPU
                        // capped+windowed path), so an SWA / Gemma2 layer runs on the
                        // GPU too (no more fail-closed decline). Computed here so they
                        // feed both the GPU hook and the CPU fallback below.
                        let swa_win = cfg.swa_window(layer_idx);
                        let softcap = cfg.attn_logit_softcapping.unwrap_or(0.0);
                        let gpu_handled = crate::accel::try_flash_attn_decode_gpu_f32(
                            &q_buf,
                            &k_buf,
                            &v_buf,
                            &mut attn_out,
                            layer_idx,
                            cur_pos as u32,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            max_ctx as u32,
                            cfg.n_layers as u32,
                            swa_win as u32,
                            softcap,
                        );
                        if gpu_handled {
                            // attn_out already populated by the SYCL
                            // kernel; skip CPU flash + standard paths.
                        } else {
                            // Flash-decode wins on memory + cache behavior at
                            // long contexts; the existing 3-pass impl wins on
                            // x86_64 perf at short contexts because it's
                            // SIMD-specialized (AVX-512 / AVX-2) and the
                            // flash-decode path is currently scalar-only.
                            // Threshold picked from the cross-over where the
                            // scalar flash overtakes the AVX-512 standard:
                            // around kv_len=4096 on a modern desktop x86 core.
                            // Tune via `RUSTLLAMA_FLASH_KV_LEN_MIN` if your CPU
                            // is different.
                            // With AVX-2 SIMD inside the F32 flash-decode
                            // kernel, flash beats the standard impl across
                            // all kv_len. Threshold is mostly insurance for
                            // ARM / non-AVX-2 x86 where flash is still
                            // scalar — bump it on those targets via the
                            // `RUSTLLAMA_FLASH_KV_LEN_MIN` env. 256 is a
                            // conservative floor that keeps fp arithmetic
                            // settled before the recurrence becomes the bulk
                            // of the work.
                            let flash_min = flash_kv_len_min();
                            // `swa_win` / `softcap` for this layer are computed above (they
                            // also feed the GPU hook); reused here for the CPU fallback's
                            // fused capped+windowed kernel (0/0.0 ⇒ bit-identical).
                            if crate::accel::flash_attention_enabled() && kv_len >= flash_min {
                                k::gqa_attention_flash_decode_capped_windowed(
                                    &q_buf,
                                    k,
                                    v,
                                    &mut attn_out,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    max_ctx,
                                    kv_len,
                                    swa_win,
                                    softcap,
                                );
                            } else {
                                k::gqa_attention_one_step_capped_windowed(
                                    &q_buf,
                                    k,
                                    v,
                                    &mut attn_out,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    max_ctx,
                                    kv_len,
                                    swa_win,
                                    softcap,
                                );
                            }
                        } // end `else` block for non-USM CPU attention
                    }
                    KvLayer::Q8_0 {
                        k_q,
                        k_scales,
                        v_q,
                        v_scales,
                    } => {
                        for h in 0..n_kv_heads {
                            let row_idx = h * max_ctx + cur_pos;
                            let dst = row_idx * head_dim;
                            let k_row = &k_buf[h * head_dim..(h + 1) * head_dim];
                            k_scales[row_idx] =
                                quantize_row_q8_0(k_row, &mut k_q[dst..dst + head_dim]);
                            let v_row = &v_buf[h * head_dim..(h + 1) * head_dim];
                            v_scales[row_idx] =
                                quantize_row_q8_0(v_row, &mut v_q[dst..dst + head_dim]);
                        }
                        // Same flash-vs-standard split as the F32 / TQ
                        // paths. With AVX-2 inside the Q8_0 flash kernel
                        // (and AVX-512 in the standard kernel) the
                        // cross-over moves down significantly; 256 is
                        // the same floor we use for F32 with AVX-2. On
                        // a CPU without AVX-2+FMA the threshold should
                        // be raised via the env var.
                        // GPU quant-KV flash-attention decode first (USM,
                        // opt-in via RUSTLLAMA_USM_ATTN). The host slab was
                        // quantized above, so a decline falls cleanly back
                        // to the CPU flash/standard split below. The helper
                        // re-quantizes k_buf/v_buf into its own USM mirror
                        // with the same quantizer, so its bytes match k_q/v_q.
                        // Seed the CUDA decode mirror with the prefill history on
                        // the first decode step (no-op once resident); without it
                        // the mirror gap declines and quant-KV decode runs on CPU.
                        // Host k_q/k_scales are byte-identical to the mirror.
                        crate::accel::decode_seed_kv_q8_0(
                            unsafe {
                                std::slice::from_raw_parts(k_q.as_ptr() as *const u8, k_q.len())
                            },
                            unsafe {
                                std::slice::from_raw_parts(v_q.as_ptr() as *const u8, v_q.len())
                            },
                            &k_scales[..],
                            &v_scales[..],
                            layer_idx,
                            cur_pos as u32,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            max_ctx as u32,
                            cfg.n_layers as u32,
                        );
                        // SWA span for this layer (0 ⇒ full causal) + Gemma2
                        // attn-logit soft-cap (0.0 ⇒ no cap), threaded into the GPU
                        // dispatch AND the CPU capped+windowed fallback below
                        // (passthrough at 0 / 0.0 ⇒ bit-identical to the base path).
                        let swa_win = cfg.swa_window(layer_idx);
                        let softcap = cfg.attn_logit_softcapping.unwrap_or(0.0);
                        if !crate::accel::try_flash_attn_decode_gpu_q8_0(
                            &q_buf,
                            &k_buf,
                            &v_buf,
                            &mut attn_out,
                            layer_idx,
                            cur_pos as u32,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            max_ctx as u32,
                            cfg.n_layers as u32,
                            swa_win as u32,
                            softcap,
                        ) {
                            let flash_min = flash_kv_len_min();
                            if crate::accel::flash_attention_enabled() && kv_len >= flash_min {
                                k::gqa_attention_flash_decode_q8_0_capped_windowed(
                                    &q_buf,
                                    k_q,
                                    k_scales,
                                    v_q,
                                    v_scales,
                                    &mut attn_out,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    max_ctx,
                                    kv_len,
                                    swa_win,
                                    softcap,
                                );
                            } else {
                                k::gqa_attention_one_step_q8_0_capped_windowed(
                                    &q_buf,
                                    k_q,
                                    k_scales,
                                    v_q,
                                    v_scales,
                                    &mut attn_out,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    max_ctx,
                                    kv_len,
                                    swa_win,
                                    softcap,
                                );
                            }
                        } // end else: CPU Q8_0 attention (GPU declined)
                    }
                    KvLayer::TurboQuant {
                        bits,
                        k_packed,
                        k_scales,
                        v_packed,
                        v_scales,
                    } => {
                        // 1. Quantize the newly-arrived row at `cur_pos` for
                        //    each head. The quantize_row helper mutates its
                        //    input (in-place WHT); reuse the pooled
                        //    `tq_scratch` (sized to head_dim by
                        //    `ForwardScratch::prepare`) to keep `k_buf` /
                        //    `v_buf` intact for later paths and avoid the
                        //    per-layer-per-token allocation that the
                        //    inline `vec![0f32; head_dim]` was doing.
                        let bytes_per_row =
                            rustllama_kernels_cpu::turboquant::bytes_per_block(head_dim, *bits);
                        let tq_scratch_slice = &mut tq_scratch[..head_dim];
                        for h in 0..n_kv_heads {
                            let row_idx = h * max_ctx + cur_pos;
                            let p_dst = row_idx * bytes_per_row;
                            // K row.
                            tq_scratch_slice
                                .copy_from_slice(&k_buf[h * head_dim..(h + 1) * head_dim]);
                            k_scales[row_idx] = rustllama_kernels_cpu::turboquant::quantize_row(
                                tq_scratch_slice,
                                *bits,
                                &mut k_packed[p_dst..p_dst + bytes_per_row],
                            );
                            // V row.
                            tq_scratch_slice
                                .copy_from_slice(&v_buf[h * head_dim..(h + 1) * head_dim]);
                            v_scales[row_idx] = rustllama_kernels_cpu::turboquant::quantize_row(
                                tq_scratch_slice,
                                *bits,
                                &mut v_packed[p_dst..p_dst + bytes_per_row],
                            );
                        }
                        // 2. Fused attention path. Dequantizes one kv-head's
                        //    K/V slab at a time into a scratch buffer sized
                        //    `kv_len × head_dim` (NOT the full `n_kv_heads
                        //    × max_ctx × head_dim` slab the prior approach
                        //    materialized). Peak memory drops from ~16 MB
                        //    per attention call at 7B-config defaults to a
                        //    few hundred KB.
                        //
                        //    Flash-decode TQ variant kicks in at the same
                        //    `kv_len >= flash_min` threshold as the F32
                        //    path (default 4096). Both paths share the
                        //    dequant scratch; flash replaces the inner
                        //    3-pass Q-head softmax with online softmax,
                        //    fewer passes over kv_len at long contexts.
                        //    TQ flash now uses the shared AVX-2-dispatched
                        //    inner loop (the dequant scratch is f32 — same
                        //    shape as the F32 flash kernel). Threshold drops
                        //    to 256 to match.
                        // GPU quant-KV flash decode first (USM). Decline →
                        // CPU flash/standard split below over the host slab.
                        // Seed the CUDA decode mirror with the prefill history on
                        // the first decode step (no-op once resident); host
                        // k_packed/k_scales are byte-identical to the mirror.
                        crate::accel::decode_seed_kv_tq(
                            *bits,
                            &k_packed[..],
                            &v_packed[..],
                            &k_scales[..],
                            &v_scales[..],
                            layer_idx,
                            cur_pos as u32,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            max_ctx as u32,
                            cfg.n_layers as u32,
                        );
                        // SWA span for this layer (0 ⇒ full causal) + Gemma2
                        // attn-logit soft-cap (0.0 ⇒ no cap), threaded into the GPU
                        // dispatch AND the CPU capped+windowed fallback below
                        // (passthrough at 0 / 0.0 ⇒ bit-identical to the base path).
                        let swa_win = cfg.swa_window(layer_idx);
                        let softcap = cfg.attn_logit_softcapping.unwrap_or(0.0);
                        if !crate::accel::try_flash_attn_decode_gpu_tq(
                            &q_buf,
                            &k_buf,
                            &v_buf,
                            &mut attn_out,
                            *bits,
                            layer_idx,
                            cur_pos as u32,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            max_ctx as u32,
                            cfg.n_layers as u32,
                            swa_win as u32,
                            softcap,
                        ) {
                            let flash_min = flash_kv_len_min();
                            if crate::accel::flash_attention_enabled() && kv_len >= flash_min {
                                rustllama_kernels_cpu::turboquant::gqa_attention_flash_decode_tq_capped_windowed(
                            &q_buf,
                            k_packed,
                            k_scales,
                            v_packed,
                            v_scales,
                            *bits,
                            &mut attn_out,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_ctx,
                            kv_len,
                            swa_win,
                            softcap,
                        );
                            } else {
                                rustllama_kernels_cpu::turboquant::gqa_attention_one_step_tq_capped_windowed(
                            &q_buf,
                            k_packed,
                            k_scales,
                            v_packed,
                            v_scales,
                            *bits,
                            &mut attn_out,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_ctx,
                            kv_len,
                            swa_win,
                            softcap,
                        );
                            }
                        } // end else: CPU TurboQuant attention (GPU declined)
                    }
                    KvLayer::Nvfp4 { k_packed, v_packed } => {
                        let blocks_per_row =
                            head_dim / rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS;
                        let bytes_per_row =
                            blocks_per_row * rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES;
                        // 1. Quantize new row for each head.
                        for h in 0..n_kv_heads {
                            let row_idx = h * max_ctx + cur_pos;
                            let p_dst = row_idx * bytes_per_row;
                            for b in 0..blocks_per_row {
                                let elem_off = h * head_dim
                                    + b * rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS;
                                let blk_dst =
                                    p_dst + b * rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES;
                                rustllama_kernels_cpu::nvfp4::quantize_block(
                                    &k_buf[elem_off
                                        ..elem_off
                                            + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS],
                                    &mut k_packed[blk_dst
                                        ..blk_dst
                                            + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES],
                                );
                                rustllama_kernels_cpu::nvfp4::quantize_block(
                                    &v_buf[elem_off
                                        ..elem_off
                                            + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS],
                                    &mut v_packed[blk_dst
                                        ..blk_dst
                                            + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES],
                                );
                            }
                        }
                        // 2. Attention. The flash variant dequantizes one
                        //    kv_h's slab at a time AND uses online softmax
                        //    — peak scratch drops from `2 × n_kv_heads ×
                        //    max_ctx × head_dim` floats to `2 × kv_len ×
                        //    head_dim` floats per call. Below the
                        //    `kv_len >= flash_min` threshold we fall back
                        //    to the slab-dequant + standard F32 attn path
                        //    so the SIMD F32 kernels still apply at short
                        //    contexts.
                        // NVFP4 flash also routes through the shared
                        // AVX-2 inner loop (after per-kv_h block-dequant
                        // into f32 scratch), so the same 256 threshold
                        // applies as F32 / Q8_0 / TQ.
                        // GPU quant-KV flash decode first (USM). Decline →
                        // CPU flash/slab split below over the host slab.
                        // Seed the CUDA decode mirror with the prefill history on
                        // the first decode step (no-op once resident); host
                        // k_packed/v_packed are byte-identical to the mirror.
                        crate::accel::decode_seed_kv_nvfp4(
                            &k_packed[..],
                            &v_packed[..],
                            layer_idx,
                            cur_pos as u32,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            max_ctx as u32,
                            cfg.n_layers as u32,
                        );
                        // SWA span for this layer (0 ⇒ full causal) + Gemma2
                        // attn-logit soft-cap (0.0 ⇒ no cap), threaded into the GPU
                        // dispatch AND both CPU fallbacks below (the flash kernel and
                        // the slab-dequant one_step); passthrough at 0 / 0.0.
                        let swa_win = cfg.swa_window(layer_idx);
                        let softcap = cfg.attn_logit_softcapping.unwrap_or(0.0);
                        if !crate::accel::try_flash_attn_decode_gpu_nvfp4(
                            &q_buf,
                            &k_buf,
                            &v_buf,
                            &mut attn_out,
                            layer_idx,
                            cur_pos as u32,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            max_ctx as u32,
                            cfg.n_layers as u32,
                            swa_win as u32,
                            softcap,
                        ) {
                            let flash_min = flash_kv_len_min();
                            if crate::accel::flash_attention_enabled() && kv_len >= flash_min {
                                rustllama_kernels_cpu::nvfp4::gqa_attention_flash_decode_nvfp4_capped_windowed(
                            &q_buf,
                            k_packed,
                            v_packed,
                            &mut attn_out,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_ctx,
                            kv_len,
                            swa_win,
                            softcap,
                        );
                            } else {
                                let mut k_f32 = vec![0f32; n_kv_heads * max_ctx * head_dim];
                                let mut v_f32 = vec![0f32; n_kv_heads * max_ctx * head_dim];
                                for h in 0..n_kv_heads {
                                    for t in 0..kv_len {
                                        let row_idx = h * max_ctx + t;
                                        let p_off = row_idx * bytes_per_row;
                                        let out_off = row_idx * head_dim;
                                        for b in 0..blocks_per_row {
                                            let blk_src = p_off
                                        + b * rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES;
                                            let blk_dst = out_off
                                        + b * rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS;
                                            rustllama_kernels_cpu::nvfp4::dequantize_block(
                                        &k_packed[blk_src
                                            ..blk_src
                                                + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES],
                                        &mut k_f32[blk_dst
                                            ..blk_dst
                                                + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS],
                                    );
                                            rustllama_kernels_cpu::nvfp4::dequantize_block(
                                        &v_packed[blk_src
                                            ..blk_src
                                                + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES],
                                        &mut v_f32[blk_dst
                                            ..blk_dst
                                                + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS],
                                    );
                                        }
                                    }
                                }
                                k::gqa_attention_one_step_capped_windowed(
                                    &q_buf,
                                    &k_f32,
                                    &v_f32,
                                    &mut attn_out,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    max_ctx,
                                    kv_len,
                                    swa_win,
                                    softcap,
                                );
                            }
                        } // end else: CPU NVFP4 attention (GPU declined)
                    }
                    // Dense MXFP decode: SWA window + Gemma2 soft-cap for this layer
                    // thread straight into the helper (passthrough at 0 / 0.0).
                    KvLayer::Mxfp4 { k_packed, v_packed } => mxfp_kv_decode(
                        KvDtype::Mxfp4,
                        &q_buf,
                        &k_buf,
                        &v_buf,
                        k_packed,
                        v_packed,
                        &mut attn_out,
                        layer_idx,
                        cur_pos,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_len,
                        cfg.n_layers,
                        cfg.swa_window(layer_idx) as u32,
                        cfg.attn_logit_softcapping.unwrap_or(0.0),
                    ),
                    KvLayer::Mxfp6 { k_packed, v_packed } => mxfp_kv_decode(
                        KvDtype::Mxfp6,
                        &q_buf,
                        &k_buf,
                        &v_buf,
                        k_packed,
                        v_packed,
                        &mut attn_out,
                        layer_idx,
                        cur_pos,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_len,
                        cfg.n_layers,
                        cfg.swa_window(layer_idx) as u32,
                        cfg.attn_logit_softcapping.unwrap_or(0.0),
                    ),
                    KvLayer::Mxfp8 { k_packed, v_packed } => mxfp_kv_decode(
                        KvDtype::Mxfp8,
                        &q_buf,
                        &k_buf,
                        &v_buf,
                        k_packed,
                        v_packed,
                        &mut attn_out,
                        layer_idx,
                        cur_pos,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_len,
                        cfg.n_layers,
                        cfg.swa_window(layer_idx) as u32,
                        cfg.attn_logit_softcapping.unwrap_or(0.0),
                    ),
                    KvLayer::Q4_0 { k_q, v_q } => {
                        use rustllama_kernels_cpu::hadamard::whiten_chunks_inplace;
                        use rustllama_kernels_cpu::q4_0_kv;
                        let blocks_per_row = head_dim / q4_0_kv::Q4_0_BLOCK_ELEMS;
                        let bytes_per_row = blocks_per_row * q4_0_kv::Q4_0_BLOCK_BYTES;
                        // 0. Whitening (fork attn_rot parity): rotate Q+K
                        //    (scores exactly preserved — same self-inverse
                        //    chunked WHT on both sides) and V pre-write;
                        //    the attention output is rotated again after,
                        //    un-doing the V basis. Only the cached bytes'
                        //    distribution changes.
                        let whiten = q4_0_whiten_active(head_dim);
                        if whiten {
                            whiten_chunks_inplace(&mut q_buf, KV_WHITEN_CHUNK);
                            whiten_chunks_inplace(
                                &mut k_buf[..n_kv_heads * head_dim],
                                KV_WHITEN_CHUNK,
                            );
                            whiten_chunks_inplace(
                                &mut v_buf[..n_kv_heads * head_dim],
                                KV_WHITEN_CHUNK,
                            );
                        }
                        // 0b. Calibration observe (pre-bias, target basis)
                        //     + K mean-centering subtract — exactly
                        //     softmax-invariant (see kv_bias module docs).
                        if crate::kv_bias::calib_active() {
                            crate::kv_bias::calib_observe(
                                layer_idx,
                                &k_buf[..n_kv_heads * head_dim],
                            );
                        }
                        if let Some(b) = kv_bias_l.as_ref().and_then(|b| b.layer(layer_idx)) {
                            for (x, bb) in k_buf[..n_kv_heads * head_dim].iter_mut().zip(b) {
                                *x -= *bb;
                            }
                        }
                        // 1. Quantize the newly-arrived K/V row at `cur_pos`
                        //    for each head — whole rows at once (the row
                        //    quantizer walks the 32-elem blocks internally).
                        for h in 0..n_kv_heads {
                            let row_idx = h * max_ctx + cur_pos;
                            let p_dst = row_idx * bytes_per_row;
                            q4_0_kv::quantize_row(
                                &k_buf[h * head_dim..(h + 1) * head_dim],
                                &mut k_q[p_dst..p_dst + bytes_per_row],
                            );
                            q4_0_kv::quantize_row(
                                &v_buf[h * head_dim..(h + 1) * head_dim],
                                &mut v_q[p_dst..p_dst + bytes_per_row],
                            );
                        }
                        // 2. Attention — same flash-vs-slab split as NVFP4:
                        //    flash dequantizes one kv_h's live rows at a
                        //    time with online softmax; below the threshold,
                        //    slab-dequant + the SIMD F32 kernel.
                        // GPU quant-KV flash decode first (USM). k_buf/v_buf
                        // are already whitened + K-bias-subtracted (steps 0/0b),
                        // exactly what the host slab was quantized from, so the
                        // helper's mirror matches k_q/v_q. Decline → CPU below.
                        // The un-whiten in step 3 applies to either output.
                        // Seed the CUDA decode mirror with the prefill history on
                        // the first decode step (no-op once resident); host k_q/v_q
                        // (quantized from the whitened k_buf) are byte-identical to
                        // the mirror, so a direct copy is correct.
                        crate::accel::decode_seed_kv_q4_0(
                            &k_q[..],
                            &v_q[..],
                            layer_idx,
                            cur_pos as u32,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            max_ctx as u32,
                            cfg.n_layers as u32,
                        );
                        // SWA span for this layer (0 ⇒ full causal) + Gemma2
                        // attn-logit soft-cap (0.0 ⇒ no cap), threaded into the GPU
                        // dispatch AND both CPU fallbacks below (the flash kernel and
                        // the slab-dequant one_step); passthrough at 0 / 0.0.
                        let swa_win = cfg.swa_window(layer_idx);
                        let softcap = cfg.attn_logit_softcapping.unwrap_or(0.0);
                        if !crate::accel::try_flash_attn_decode_gpu_q4_0(
                            &q_buf,
                            &k_buf,
                            &v_buf,
                            &mut attn_out,
                            layer_idx,
                            cur_pos as u32,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            max_ctx as u32,
                            cfg.n_layers as u32,
                            swa_win as u32,
                            softcap,
                        ) {
                            let flash_min = flash_kv_len_min();
                            if crate::accel::flash_attention_enabled() && kv_len >= flash_min {
                                q4_0_kv::gqa_attention_flash_decode_q4_0_capped_windowed(
                                    &q_buf,
                                    k_q,
                                    v_q,
                                    &mut attn_out,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    max_ctx,
                                    kv_len,
                                    swa_win,
                                    softcap,
                                );
                            } else {
                                let mut k_f32 = vec![0f32; n_kv_heads * max_ctx * head_dim];
                                let mut v_f32 = vec![0f32; n_kv_heads * max_ctx * head_dim];
                                for h in 0..n_kv_heads {
                                    for t in 0..kv_len {
                                        let row_idx = h * max_ctx + t;
                                        let p_off = row_idx * bytes_per_row;
                                        let out_off = row_idx * head_dim;
                                        q4_0_kv::dequantize_row(
                                            &k_q[p_off..p_off + bytes_per_row],
                                            &mut k_f32[out_off..out_off + head_dim],
                                        );
                                        q4_0_kv::dequantize_row(
                                            &v_q[p_off..p_off + bytes_per_row],
                                            &mut v_f32[out_off..out_off + head_dim],
                                        );
                                    }
                                }
                                k::gqa_attention_one_step_capped_windowed(
                                    &q_buf,
                                    &k_f32,
                                    &v_f32,
                                    &mut attn_out,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    max_ctx,
                                    kv_len,
                                    swa_win,
                                    softcap,
                                );
                            }
                        } // end else: CPU Q4_0 attention (GPU declined)
                          // 3. Un-whiten the attention output (self-inverse
                          //    transform re-applied — see step 0).
                        if whiten {
                            whiten_chunks_inplace(&mut attn_out, KV_WHITEN_CHUNK);
                        }
                    }
                    KvLayer::Half { bf16, k, v } => {
                        // M6: write the new row as half bits, then dequant the
                        // live region to a compact f32 scratch and run the F32
                        // decode kernel (stride = kv_len). Output equals the F32
                        // path over f16/bf16-rounded K/V. `swa_win`/`softcap`
                        // are computed like the F32 arm (0/0.0 for ordinary
                        // models; SWA/soft-cap models are coerced to F32 KV at
                        // load, so half KV never sees a nonzero value here).
                        let bf16 = *bf16;
                        for h in 0..n_kv_heads {
                            let dst = (h * max_ctx + cur_pos) * head_dim;
                            half_encode_row(
                                bf16,
                                &k_buf[h * head_dim..(h + 1) * head_dim],
                                &mut k[dst..dst + head_dim],
                            );
                            half_encode_row(
                                bf16,
                                &v_buf[h * head_dim..(h + 1) * head_dim],
                                &mut v[dst..dst + head_dim],
                            );
                        }
                        let swa_win = cfg.swa_window(layer_idx);
                        let softcap = cfg.attn_logit_softcapping.unwrap_or(0.0);
                        let mut k_f32 = vec![0f32; n_kv_heads * kv_len * head_dim];
                        let mut v_f32 = vec![0f32; n_kv_heads * kv_len * head_dim];
                        half_dequant_region(
                            bf16, k, v, n_kv_heads, max_ctx, head_dim, kv_len, &mut k_f32,
                            &mut v_f32,
                        );
                        let flash_min = flash_kv_len_min();
                        if crate::accel::flash_attention_enabled() && kv_len >= flash_min {
                            k::gqa_attention_flash_decode_capped_windowed(
                                &q_buf,
                                &k_f32,
                                &v_f32,
                                &mut attn_out,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                /* max_ctx (compact stride) */ kv_len,
                                kv_len,
                                swa_win,
                                softcap,
                            );
                        } else {
                            k::gqa_attention_one_step_capped_windowed(
                                &q_buf,
                                &k_f32,
                                &v_f32,
                                &mut attn_out,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                kv_len,
                                kv_len,
                                swa_win,
                                softcap,
                            );
                        }
                    }
                }

                // H6: try the fully-fused output-proj + residual + norm
                // kernel first (one GPU dispatch for matvec(W_o) + add +
                // rmsnorm). P5b: `fused_out_proj_norm_enabled()` is fail-closed
                // OFF until the composite `fused:out_proj_norm` verdict lands (P6)
                // + dtype coverage; on success it writes the
                // residual sum into `hidden` AND the normalized output into
                // `h_norm`, so we skip the separate out-proj matvec and the
                // add_rmsnorm ladder below entirely.
                let h6_fused = !skip_attn_enabled()
                    && crate::accel::try_matvec_out_proj_add_rmsnorm_usm_f32(
                        block.w_o(),
                        &attn_out,
                        &mut hidden,
                        block.ffn_norm(),
                        &mut h_norm,
                        d,
                        d_q,
                        cfg.rms_eps,
                    );
                if h6_fused {
                    debug::attn_stats(
                        layer_idx, &q_buf, &k_buf, &v_buf, &attn_out, &attn_proj, &hidden,
                    );
                } else {
                    // Output projection: attn_proj = W_o · attn_out
                    matvec_tensor_dispatch(block.w_o(), &attn_out, &mut attn_proj, d, d_q);

                    debug::attn_stats(
                        layer_idx, &q_buf, &k_buf, &v_buf, &attn_out, &attn_proj, &hidden,
                    );

                    // ---- FFN ----
                    // Fused dispatch ladder: GPU `add_rmsnorm_usm` (one
                    // kernel does the residual add + post-attn norm) → CPU
                    // `add_inplace_f32` then GPU `try_rmsnorm_usm_f32` →
                    // host-pointer SYCL → CPU rmsnorm. `RUSTLLAMA_SKIP_ATTN`
                    // bypasses the residual entirely (debugging).
                    if skip_attn_enabled() {
                        if !crate::accel::try_rmsnorm_usm_f32(
                            &hidden,
                            block.ffn_norm(),
                            &mut h_norm,
                            d,
                            cfg.rms_eps,
                        ) && !crate::accel::try_rmsnorm_f32(
                            &hidden,
                            block.ffn_norm(),
                            &mut h_norm,
                            d,
                            cfg.rms_eps,
                        ) {
                            k::rmsnorm_f32_row(&hidden, block.ffn_norm(), &mut h_norm, cfg.rms_eps);
                        }
                    } else if {
                        // H6: `try_add_rmsnorm_usm_f32` already fuses
                        // `residual + norm` into one GPU dispatch (added in
                        // Tier F's `add_rmsnorm_usm` fusion). The remaining
                        // un-fused op is the OUTPUT PROJECTION matvec that
                        // produces `attn_proj` upstream. Fully fusing
                        // `out_proj_matvec + residual + norm` requires a new
                        // `matvec_add_rmsnorm_packed_f32_usm` kernel per
                        // weight dtype (~12 variants: Q8_0, Q4_K, IQ1_S, …).
                        // Tracked as Tier H follow-up; today's two-op fusion
                        // already saves the residual-add CPU loop.
                        !crate::accel::try_add_rmsnorm_usm_f32(
                            &mut hidden,
                            &attn_proj,
                            block.ffn_norm(),
                            &mut h_norm,
                            d,
                            cfg.rms_eps,
                        )
                    } {
                        k::add_inplace_f32(&mut hidden, &attn_proj);
                        if !crate::accel::try_rmsnorm_usm_f32(
                            &hidden,
                            block.ffn_norm(),
                            &mut h_norm,
                            d,
                            cfg.rms_eps,
                        ) && !crate::accel::try_rmsnorm_f32(
                            &hidden,
                            block.ffn_norm(),
                            &mut h_norm,
                            d,
                            cfg.rms_eps,
                        ) {
                            k::rmsnorm_f32_row(&hidden, block.ffn_norm(), &mut h_norm, cfg.rms_eps);
                        }
                    }
                } // end else (H6 fused out-proj+norm not taken)
            } // end `if !device_attn` (host attention path; device path done above)

            // FFN dispatch: dense (existing matvec ladder) or MoE
            // (router + top-K expert SwiGLU). The MoE path uses
            // pre-allocated scratch from `ForwardScratch` so the
            // per-layer call is allocation-free — gate/up/ff
            // buffers are shared with the dense ladder (same
            // shape), down_buf / expert_logits / routed_picks are
            // dedicated MoE scratch slots.
            // Skipped entirely when the device block already ran the dense FFN
            // (#2 Phase 1): `hidden` then already carries the FFN residual.
            if !device_dense_ffn {
                // P6-fusions: set when the dense FFN below fuses the down-proj
                // + residual-add into one USM dispatch (f32 down weight only,
                // `fused:down_residual` verdict on). When set, `ffn_out` already
                // holds `hidden + W_down·ffn_buf`, so the separate residual add
                // below is replaced by adopting it — byte-identical to the two
                // ops when the verdict is off/absent.
                let mut down_residual_fused = false;
                if let Some(moe) = moe_cfg.as_ref() {
                    let mb = &self.weights.moe_blocks.as_ref().unwrap()[layer_idx];
                    crate::moe::moe_ffn_one_into(
                        &h_norm,
                        mb,
                        d,
                        d_ff,
                        moe.n_experts as usize,
                        moe.n_experts_used as usize,
                        &mut ffn_out,
                        &mut gate_buf,
                        &mut up_buf,
                        &mut ffn_buf,
                        &mut moe_down,
                        &mut moe_expert_logits,
                        &mut moe_routed_picks,
                    );
                } else {
                    let dense_block = &self.weights.blocks[layer_idx];
                    matvec_tensor_dispatch(&dense_block.w_gate, &h_norm, &mut gate_buf, d_ff, d);
                    matvec_tensor_dispatch(&dense_block.w_up, &h_norm, &mut up_buf, d_ff, d);
                    // USM → host-pointer SYCL → CPU dispatch ladder.
                    if !crate::accel::try_silu_mul_usm_f32(&gate_buf, &up_buf, &mut ffn_buf)
                        && !crate::accel::try_silu_mul_f32(&gate_buf, &up_buf, &mut ffn_buf)
                    {
                        k::silu_mul_f32(&gate_buf, &up_buf, &mut ffn_buf);
                    }
                    // P6-fusions: FFN down-proj + residual-add in one USM
                    // dispatch (f32 down weight only; fail-closed via the
                    // `fused:down_residual` verdict → byte-identical fallback to
                    // the separate down matvec + `add_inplace_f32` below). On
                    // success `ffn_out` = `hidden + W_down·ffn_buf`.
                    let w_down = &dense_block.w_down;
                    if !skip_ffn_enabled()
                        && w_down.dtype == Dtype::F32
                        && crate::accel::try_matvec_down_residual_usm_f32(
                            &ffn_buf,
                            rustllama_tensor::as_slice_f32(w_down),
                            &hidden,
                            &mut ffn_out,
                            1,
                            d_ff,
                            d,
                        )
                    {
                        down_residual_fused = true;
                    } else {
                        matvec_tensor_dispatch(w_down, &ffn_buf, &mut ffn_out, d, d_ff);
                    }
                }

                debug::ffn_stats(
                    layer_idx, &hidden, &h_norm, &gate_buf, &up_buf, &ffn_buf, &ffn_out,
                );

                if !skip_ffn_enabled() {
                    if down_residual_fused {
                        // `ffn_out` already carries `hidden + W_down·ffn_buf`
                        // (the P6 down+residual fusion), so adopt it as the new
                        // hidden instead of re-adding — same result, one op.
                        hidden.copy_from_slice(&ffn_out);
                    } else {
                        k::add_inplace_f32(&mut hidden, &ffn_out);
                    }
                }
            }

            debug::trace_layer(layer_idx, &hidden);
        }

        // #2 Phase 2: drain the cross-layer-resident hidden back to the host once
        // the layer loop is done (the dense device path kept it on device across
        // layers), so the final norm + LM head below read the current state.
        if hidden_dev_resident {
            crate::accel::fetch_resident_hidden_cuda_f32(&mut hidden, d);
        }

        // ---- output norm + LM head ----
        // `final_norm` is already pre-sized + zeroed from the
        // ForwardScratch take at the top of this function; we
        // overwrite it fully in the rmsnorm call below.
        let lm_head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        // P6-fusions: fold the final RMSNorm into the LM-head matvec in one USM
        // dispatch (f32 LM-head weight only; fail-closed via the
        // `fused:rmsnorm_matvec` verdict). On success `logits_out` holds
        // `matvec(lm_head, rmsnorm(hidden)·output_norm)` directly and BOTH the
        // separate rmsnorm ladder + the matvec are skipped — BYTE-IDENTICAL
        // fallback to those two ops when the verdict is off/absent (or the
        // LM-head weight is not f32, the common quantized case).
        let rmsnorm_matvec_fused = lm_head.dtype == Dtype::F32
            && self.weights.output_norm.len() >= d
            && crate::accel::try_rmsnorm_matvec_usm_f32(
                &hidden,
                &self.weights.output_norm,
                rustllama_tensor::as_slice_f32(lm_head),
                logits_out,
                1,
                d,
                cfg.vocab_size,
                cfg.rms_eps,
            );
        if !rmsnorm_matvec_fused {
            // USM → host-pointer SYCL → CPU dispatch ladder.
            if !crate::accel::try_rmsnorm_usm_f32(
                &hidden,
                &self.weights.output_norm,
                &mut final_norm,
                d,
                cfg.rms_eps,
            ) && !crate::accel::try_rmsnorm_f32(
                &hidden,
                &self.weights.output_norm,
                &mut final_norm,
                d,
                cfg.rms_eps,
            ) {
                k::rmsnorm_f32_row(
                    &hidden,
                    &self.weights.output_norm,
                    &mut final_norm,
                    cfg.rms_eps,
                );
            }
            matvec_tensor_dispatch(lm_head, &final_norm, logits_out, cfg.vocab_size, d);
        }
        self.finalize_logits(logits_out);

        kv.seq_len = kv.seq_len.max(cur_pos + 1);

        // Swap the locally-owned scratch back into the per-thread
        // ForwardScratch so the next forward_one on this thread
        // reuses the same heap storage. `mem::take` left those
        // fields as empty `Vec`s; placing the populated ones back
        // restores the cache.
        scratch.hidden = hidden;
        scratch.h_norm = h_norm;
        scratch.q_buf = q_buf;
        scratch.k_buf = k_buf;
        scratch.v_buf = v_buf;
        scratch.attn_out = attn_out;
        scratch.attn_proj = attn_proj;
        scratch.gate_buf = gate_buf;
        scratch.up_buf = up_buf;
        scratch.ffn_buf = ffn_buf;
        scratch.ffn_out = ffn_out;
        scratch.final_norm = final_norm;
        scratch.moe_down = moe_down;
        scratch.moe_expert_logits = moe_expert_logits;
        scratch.moe_routed_picks = moe_routed_picks;
        scratch.tq_scratch = tq_scratch;
        scratch.swa_ring_k = swa_ring_k;
        scratch.swa_ring_v = swa_ring_v;
    }

    /// Fused multi-slot decode: advance M independent slots by one
    /// token each in a single forward pass. Batches every matmul
    /// kernel (Q/K/V/O/gate/up/down) across the M tokens so the
    /// per-launch overhead amortizes over all slots — that's the
    /// continuous-batching throughput win on integrated GPUs where
    /// kernel launch is a sizeable fraction of decode time.
    /// Attention is per-slot serial (each slot reads its own KV
    /// history under its own gather); a future fused-attention
    /// kernel would close the remaining gap.
    ///
    /// Per-slot semantics are identical to
    /// [`Self::forward_one_paged_f32`]: same RMSNorm / RoPE / FFN
    /// kernels, same flash-vs-standard attention threshold at
    /// `kv_len ≥ 256`. The parity test
    /// `forward_decode_paged_batched_matches_serial_forward_one_paged`
    /// asserts bit-for-bit equality between fused and serial-per-slot.
    ///
    /// Each slot's cache must have capacity for `slot.pos + 1`
    /// tokens before calling (caller's job — typically the
    /// scheduler grows caches at admit time). Each slot's
    /// `logits_out` must be exactly `cfg.vocab_size` floats.
    ///
    /// Returns immediately on `slots.is_empty()`. Single-slot
    /// (`slots.len() == 1`) is supported but has no batching
    /// advantage — the engine's scheduler skips this path when
    /// only one slot is decoding and uses `forward_one_paged_f32`
    /// directly.
    pub fn forward_decode_paged_batched_f32(
        &self,
        slots: &mut [crate::llama_arch::DecodeSlot<'_>],
        shared: &crate::shared_paged_kv::SharedPagedKv,
    ) {
        self.forward_decode_paged_batched_cascade_f32(slots, shared, None)
    }

    /// P3 cascade-aware variant of [`Self::forward_decode_paged_batched_f32`].
    /// `cascade` groups decode slots that share a page-aligned APC prefix; each
    /// engaged group computes the shared-prefix attention once and merges per
    /// slot (full-attention layers only), numerically equivalent (~1e-6) to the
    /// per-slot path. `None` / an empty slice is byte-identical to the plain
    /// batched decode.
    pub fn forward_decode_paged_batched_cascade_f32(
        &self,
        slots: &mut [crate::llama_arch::DecodeSlot<'_>],
        shared: &crate::shared_paged_kv::SharedPagedKv,
        cascade: Option<&[CascadeGroup]>,
    ) {
        let m = slots.len();
        if m == 0 {
            return;
        }
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = cfg.d_ff;
        let vocab = cfg.vocab_size;

        // Per-slot sanity. Cheap; we want the panic location to
        // point at the misshapen slot, not at a wrong-size memcpy
        // 200 lines later.
        for (i, slot) in slots.iter().enumerate() {
            assert_eq!(
                slot.logits_out.len(),
                vocab,
                "slot {i}: logits_out len {} != vocab_size {vocab}",
                slot.logits_out.len(),
            );
            debug_assert!(
                slot.pos < slot.cache.capacity_tokens(),
                "slot {i}: paged decode pos {} >= cache capacity {} \
                 (caller must ensure_capacity_shared first)",
                slot.pos,
                slot.cache.capacity_tokens(),
            );
        }

        // Batched scratch sized to `m` rows. Same shapes the
        // prefill mirror uses for its `n_new` rows; we just
        // reinterpret rows as slots instead of tokens-in-a-prompt.
        let mut hidden = vec![0f32; m * d];
        let mut h_norm = vec![0f32; m * d];
        let mut q_buf = vec![0f32; m * d_q];
        let mut k_buf = vec![0f32; m * d_kv];
        let mut v_buf = vec![0f32; m * d_kv];
        let mut attn_out = vec![0f32; m * d_q];
        let mut attn_proj = vec![0f32; m * d];
        let mut gate_buf = vec![0f32; m * d_ff];
        let mut up_buf = vec![0f32; m * d_ff];
        let mut ffn_buf = vec![0f32; m * d_ff];
        let mut ffn_out = vec![0f32; m * d];

        // Per-slot per-layer attention scratch. Reused across all
        // layers (and across all slots within a layer); sized to
        // the largest slot's kv_len so a single buffer covers
        // every slot's gather. Each gather overwrites the buffer
        // before the per-slot attention call reads it.
        let max_kv_len = slots.iter().map(|s| s.pos as usize + 1).max().unwrap_or(0);
        let slab_len = n_kv_heads * max_kv_len * head_dim;
        let mut k_slab = vec![0f32; slab_len];
        let mut v_slab = vec![0f32; slab_len];

        // Step 1: embed each slot's input token into its hidden row.
        for (i, slot) in slots.iter().enumerate() {
            k::embed_lookup_tensor(
                &self.weights.token_embd,
                &[slot.token_id],
                &mut hidden[i * d..(i + 1) * d],
                d,
            );
        }

        // MoE per-token scratch (continuous-batching decode loops the
        // MoE FFN per slot because routing is per-token).
        let moe_cfg = cfg.moe.clone();
        let (mut moe_gate_one, mut moe_up_one, mut moe_ff_one, mut moe_down_one) =
            if moe_cfg.is_some() {
                (
                    vec![0f32; d_ff],
                    vec![0f32; d_ff],
                    vec![0f32; d_ff],
                    vec![0f32; d],
                )
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };
        let (mut moe_expert_logits, mut moe_routed_picks) = if let Some(m) = moe_cfg.as_ref() {
            (
                vec![0f32; m.n_experts as usize],
                Vec::with_capacity(m.n_experts_used as usize),
            )
        } else {
            (Vec::new(), Vec::new())
        };

        let attn_views: Vec<&dyn AttnBlock> = if let Some(mbs) = self.weights.moe_blocks.as_ref() {
            mbs.iter().map(|b| b as &dyn AttnBlock).collect()
        } else {
            self.weights
                .blocks
                .iter()
                .map(|b| b as &dyn AttnBlock)
                .collect()
        };

        let n_layers_used = crate::accel::max_layers_cap()
            .unwrap_or(attn_views.len())
            .min(attn_views.len());

        for (layer_idx, block) in attn_views.iter().take(n_layers_used).enumerate() {
            // Hybrid placement: tell the `try_*_usm_f32` dispatch
            // ladder which layer we're on. Helpers consult
            // `accel::gpu_active_for_current_layer()` (= layer_idx
            // < accel::n_gpu_layers()) to short-circuit the GPU
            // path for layers past the `[inference].n_gpu_layers`
            // cutoff.
            crate::accel::set_current_layer_idx(layer_idx as u32);
            // Per-slot pre-attention rmsnorm. Cheap per-row CPU op;
            // batching across slots gives no real win here.
            for i in 0..m {
                k::rmsnorm_f32_row(
                    &hidden[i * d..(i + 1) * d],
                    block.attn_norm(),
                    &mut h_norm[i * d..(i + 1) * d],
                    cfg.rms_eps,
                );
            }
            // Batched Q / K / V projections — one launch each
            // across all M slots' hidden rows. This is the
            // headline win vs M serial calls to
            // `forward_one_paged_f32`.
            matvec_tensor_batched_dispatch(block.w_q(), &h_norm, &mut q_buf, d_q, d, m);
            matvec_tensor_batched_dispatch(block.w_k(), &h_norm, &mut k_buf, d_kv, d, m);
            matvec_tensor_batched_dispatch(block.w_v(), &h_norm, &mut v_buf, d_kv, d, m);
            // Per-slot bias add + RoPE — bias is shared across
            // slots but the RoPE position is per-slot, so this
            // can't be batched without a position-vector kernel.
            for (i, slot) in slots.iter().enumerate() {
                let q_row = &mut q_buf[i * d_q..(i + 1) * d_q];
                let k_row = &mut k_buf[i * d_kv..(i + 1) * d_kv];
                let v_row = &mut v_buf[i * d_kv..(i + 1) * d_kv];
                if let Some(bq) = block.b_q() {
                    k::add_inplace_f32(q_row, bq);
                }
                if let Some(bk) = block.b_k() {
                    k::add_inplace_f32(k_row, bk);
                }
                if let Some(bv) = block.b_v() {
                    k::add_inplace_f32(v_row, bv);
                }
                // Qwen3 per-head Q/K RMSNorm, before RoPE. No-op on
                // every non-Qwen3 arch (q_norm/k_norm are None).
                apply_qk_head_norm(
                    block.q_norm(),
                    block.k_norm(),
                    q_row,
                    k_row,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    cfg.rms_eps,
                );
                self.rope_inplace(q_row, n_heads, head_dim, slot.pos);
                self.rope_inplace(k_row, n_kv_heads, head_dim, slot.pos);
            }

            // Per-slot: write the new K/V row to that slot's cache
            // (under the shared store lock), gather the slot's full
            // KV history for this layer, run serial attention.
            // Attention is per-slot because each slot has its own
            // kv_len and its own KV history — fused attention
            // requires a paged-attention kernel that we'll add later.
            // D1b: per-layer SWA window (0 ⇒ full attention) + Gemma2 attn-logit
            // soft-cap (0.0 ⇒ none), threaded into the capped+windowed decode
            // kernels so an SWA / Gemma2-softcap model is correct on the paged
            // batched decode path. At 0 / 0.0 the capped+windowed kernels delegate
            // to the plain decode dispatchers, byte-identical for every ordinary
            // (non-SWA, non-softcap) model.
            let swa_win = cfg.swa_window(layer_idx);
            let softcap = cfg.attn_logit_softcapping.unwrap_or(0.0);
            // Pass 1: write every slot's new K/V row for this layer. Split from
            // the attention pass so the P3 cascade can gather a shared prefix
            // AFTER all slots have written (a slot only writes its OWN pages, so
            // this reordering is byte-identical to the interleaved loop).
            for (i, slot) in slots.iter_mut().enumerate() {
                let k_row = &k_buf[i * d_kv..(i + 1) * d_kv];
                let v_row = &v_buf[i * d_kv..(i + 1) * d_kv];
                slot.cache
                    .write_token_shared(shared, layer_idx as u32, slot.pos, k_row, v_row)
                    .expect("paged write_token_shared: pos < capacity (asserted above)");
            }

            // Pass 2a: P3 cascade — for a group of >=2 slots sharing a
            // page-aligned APC prefix, compute the shared-prefix attention ONCE
            // and merge each slot's suffix (full-attention layers only; the
            // cascade kernel has no windowed variant). Numerically equivalent
            // (online-softmax merge, ~1e-6) to the per-slot path below.
            let mut handled = vec![false; m];
            if swa_win == 0 {
                if let Some(groups) = cascade {
                    for g in groups {
                        if g.prefix_len == 0 || g.members.len() < 2 {
                            continue;
                        }
                        // Every member is a decode row (kv_len = pos + 1); skip
                        // the group if any member can't host the shared prefix
                        // or was already handled.
                        if g.members.iter().any(|&mi| {
                            mi >= m || handled[mi] || (slots[mi].pos as usize + 1) < g.prefix_len
                        }) {
                            continue;
                        }
                        let nm = g.members.len();
                        let caches: Vec<&crate::paged_kv_cache::PagedKvCache> =
                            g.members.iter().map(|&mi| &*slots[mi].cache).collect();
                        let kv_lens: Vec<usize> = g
                            .members
                            .iter()
                            .map(|&mi| slots[mi].pos as usize + 1)
                            .collect();
                        let mut queries = vec![0f32; nm * d_q];
                        for (gi, &mi) in g.members.iter().enumerate() {
                            queries[gi * d_q..(gi + 1) * d_q]
                                .copy_from_slice(&q_buf[mi * d_q..(mi + 1) * d_q]);
                        }
                        let mut group_out = vec![0f32; nm * d_q];
                        cascade_decode_attention_group(
                            shared,
                            layer_idx as u32,
                            g.prefix_len,
                            &caches,
                            &kv_lens,
                            &queries,
                            &mut group_out,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            softcap,
                        );
                        for (gi, &mi) in g.members.iter().enumerate() {
                            attn_out[mi * d_q..(mi + 1) * d_q]
                                .copy_from_slice(&group_out[gi * d_q..(gi + 1) * d_q]);
                            handled[mi] = true;
                        }
                    }
                }
            }

            // Pass 2b: per-slot gather + attention for every slot NOT handled by
            // a cascade group (byte-identical to the pre-P3 path).
            let flash_min = flash_kv_len_min();
            for (i, slot) in slots.iter().enumerate() {
                if handled[i] {
                    continue;
                }
                let kv_len = slot.pos as usize + 1;
                // Phase D/E/F (GPU paged attention): the block-table flash-decode
                // reads a device KV-pool mirror directly, so a hit SKIPS the host
                // gather below. The combinator tiers CUDA → MLX → SYCL; inert /
                // false off-GPU or without the per-backend verdict => the gather +
                // CPU attention path. `PageId` is repr(transparent), so `pages()`
                // reinterprets as the `&[u32]` block table zero-copy.
                {
                    let pages = slot.cache.pages();
                    let block_table: &[u32] = unsafe {
                        std::slice::from_raw_parts(pages.as_ptr() as *const u32, pages.len())
                    };
                    if crate::accel::try_flash_attn_decode_paged_gpu_f32(
                        &q_buf[i * d_q..(i + 1) * d_q],
                        &mut attn_out[i * d_q..(i + 1) * d_q],
                        block_table,
                        layer_idx as u32,
                        kv_len as u32,
                        n_heads as u32,
                        n_kv_heads as u32,
                        head_dim as u32,
                        swa_win as i32,
                        softcap,
                    ) {
                        continue;
                    }
                }
                // Slice the scratch to this slot's kv_len (the gather only writes
                // that prefix; the buffer is sized to the max kv_len so it fits).
                let slab_used = n_kv_heads * kv_len * head_dim;
                let k_slice = &mut k_slab[..slab_used];
                let v_slice = &mut v_slab[..slab_used];
                slot.cache
                    .gather_layer_shared(shared, layer_idx as u32, k_slice, v_slice)
                    .expect("paged gather_layer_shared: cache covers kv_len positions");
                // Match the single-slot path's flash-vs-standard
                // threshold so per-slot logits are bit-identical to
                // `forward_one_paged_f32`.
                let q_row = &q_buf[i * d_q..(i + 1) * d_q];
                let out_row = &mut attn_out[i * d_q..(i + 1) * d_q];
                if crate::accel::flash_attention_enabled() && kv_len >= flash_min {
                    k::gqa_attention_flash_decode_capped_windowed(
                        q_row, k_slice, v_slice, out_row, n_heads, n_kv_heads, head_dim, kv_len,
                        kv_len, swa_win, softcap,
                    );
                } else {
                    k::gqa_attention_one_step_capped_windowed(
                        q_row, k_slice, v_slice, out_row, n_heads, n_kv_heads, head_dim, kv_len,
                        kv_len, swa_win, softcap,
                    );
                }
            }

            // Batched output projection.
            matvec_tensor_batched_dispatch(block.w_o(), &attn_out, &mut attn_proj, d, d_q, m);
            // Per-slot residual + post-attn rmsnorm.
            for i in 0..m {
                let proj_row = &attn_proj[i * d..(i + 1) * d];
                let h_row = &mut hidden[i * d..(i + 1) * d];
                if !skip_attn_enabled() {
                    k::add_inplace_f32(h_row, proj_row);
                }
                let n_row = &mut h_norm[i * d..(i + 1) * d];
                k::rmsnorm_f32_row(h_row, block.ffn_norm(), n_row, cfg.rms_eps);
            }
            // FFN: dense batched matvec ladder OR per-slot MoE.
            if let Some(moe) = moe_cfg.as_ref() {
                let mb = &self.weights.moe_blocks.as_ref().unwrap()[layer_idx];
                for i in 0..m {
                    let n_row = &h_norm[i * d..(i + 1) * d];
                    let out_row = &mut ffn_out[i * d..(i + 1) * d];
                    crate::moe::moe_ffn_one_into(
                        n_row,
                        mb,
                        d,
                        d_ff,
                        moe.n_experts as usize,
                        moe.n_experts_used as usize,
                        out_row,
                        &mut moe_gate_one,
                        &mut moe_up_one,
                        &mut moe_ff_one,
                        &mut moe_down_one,
                        &mut moe_expert_logits,
                        &mut moe_routed_picks,
                    );
                }
            } else {
                let dense_block = &self.weights.blocks[layer_idx];
                // Batched FFN gate / up.
                matvec_tensor_batched_dispatch(
                    &dense_block.w_gate,
                    &h_norm,
                    &mut gate_buf,
                    d_ff,
                    d,
                    m,
                );
                matvec_tensor_batched_dispatch(&dense_block.w_up, &h_norm, &mut up_buf, d_ff, d, m);
                // Per-slot silu_mul (small element-wise op; batching
                // doesn't help).
                for i in 0..m {
                    k::silu_mul_f32(
                        &gate_buf[i * d_ff..(i + 1) * d_ff],
                        &up_buf[i * d_ff..(i + 1) * d_ff],
                        &mut ffn_buf[i * d_ff..(i + 1) * d_ff],
                    );
                }
                // Batched FFN down.
                matvec_tensor_batched_dispatch(
                    &dense_block.w_down,
                    &ffn_buf,
                    &mut ffn_out,
                    d,
                    d_ff,
                    m,
                );
            }
            // Per-slot residual.
            if !skip_ffn_enabled() {
                for i in 0..m {
                    k::add_inplace_f32(
                        &mut hidden[i * d..(i + 1) * d],
                        &ffn_out[i * d..(i + 1) * d],
                    );
                }
            }
        }

        // F2: batched final norm + LM head. The LM head matmul is the
        // largest per-call kernel for vocab-heavy models (vocab ≥ 100K
        // on Qwen2.5 / Llama 3); running it M times serially was the
        // single largest unrealized continuous-batching tok/s gain.
        // Replace with one batched RMSNorm-then-matvec: write all M
        // pre-LM-head normalized rows into one contiguous
        // `[M × d]` buffer, run a single `matvec_tensor_batched_dispatch`
        // into a `[M × vocab]` output buffer, then scatter each
        // row into the slot's caller-owned `logits_out`. The scatter
        // is `M × vocab` f32 writes — same byte volume as the M
        // serial matvecs would have produced; the savings are the
        // matmul dispatch overhead × M and the better cache
        // residency of one large GEMM.
        let lm_head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        let mut final_norm_all = vec![0f32; m * d];
        // H10: batched RMSNorm across M slots via the existing
        // `rsl_rmsnorm_usm` kernel (which already iterates over
        // `n_rows`). Falls back to per-row CPU loop when GPU is
        // unavailable or returns false. Win at high concurrency
        // (M=16): saves 16× per-row dispatch + lets the GPU
        // parallelize the per-row scan; at M=1 the heuristic skips
        // the GPU path to avoid USM marshal overhead.
        let used_gpu_norm = if m > 1
            && crate::accel::try_rmsnorm_usm_f32(
                &hidden,
                &self.weights.output_norm,
                &mut final_norm_all,
                d,
                cfg.rms_eps,
            ) {
            true
        } else {
            for i in 0..m {
                let src = &hidden[i * d..(i + 1) * d];
                let dst = &mut final_norm_all[i * d..(i + 1) * d];
                k::rmsnorm_f32_row(src, &self.weights.output_norm, dst, cfg.rms_eps);
            }
            false
        };
        let _ = used_gpu_norm;
        let mut logits_all = vec![0f32; m * vocab];
        matvec_tensor_batched_dispatch(lm_head, &final_norm_all, &mut logits_all, vocab, d, m);
        self.finalize_logits(&mut logits_all);
        for (i, slot) in slots.iter_mut().enumerate() {
            slot.logits_out
                .copy_from_slice(&logits_all[i * vocab..(i + 1) * vocab]);
        }
    }

    /// Feature 3b: a TRUE RAGGED mixed prefill+decode forward over a paged KV
    /// pool (F32 KV). ONE forward pass advances several slots at once, where
    /// each slot is EITHER mid-prompt-prefill (contributing `n_query > 1` chunk
    /// tokens) OR decoding (contributing a single `n_query == 1` query row) —
    /// freely mixed in the same batch. This unifies what the driver previously
    /// did as two separate forwards ([`Self::forward_prefill_paged_f32`] then
    /// [`Self::forward_decode_paged_batched_f32`]); the budget-0 driver path
    /// still uses those two directly, so this is the single path only when the
    /// per-tick batched-token budget is on.
    ///
    /// # How the batch is laid out
    ///
    /// `tokens` is the CONCATENATED query-token buffer: slot 0's `n_query`
    /// tokens, then slot 1's, … in `slots` order. Slot `s`'s rows occupy
    /// `[row_off(s), row_off(s)+n_query(s))` of every per-token scratch buffer,
    /// where `row_off(s)` is the running sum of the preceding slots' `n_query`.
    /// `total = Σ n_query` is the batched row count and MUST equal
    /// `tokens.len()`.
    ///
    /// # What batches vs what loops (the core of the feature)
    ///
    /// * **Input projections + FFN batch over the concatenated row dim.** The
    ///   Q/K/V projections, the attention-output projection, and the dense/MoE
    ///   FFN each run ONCE over all `total` rows via
    ///   [`matvec_tensor_batched_dispatch`] (or the per-row MoE loop). These ops
    ///   are per-row INDEPENDENT — row `r`'s output depends only on row `r`'s
    ///   input and the (shared, read-only) weights — so batching `n_a` prefill
    ///   rows together with `n_b` decode rows produces, for each row, EXACTLY
    ///   the value that row would get if projected alone. That independence is
    ///   what makes the ragged pass bit-identical (up to fp reduction order) to
    ///   running the slots' prefills and decodes as separate forwards, and it is
    ///   the whole throughput win (one kernel launch instead of two).
    ///
    /// * **RoPE, the paged KV write, the gather, and attention loop PER SLOT.**
    ///   Position is per-token (`start_pos + j`), causal masking is per-slot
    ///   (`kv_len_base = start_pos`), and each slot reads its own paged KV
    ///   history — none of that can be shared across slots, so these stay in a
    ///   per-slot loop. A slot with `n_query > 1` routes through the varlen
    ///   causal `gqa_attention_flash_prefill*` (identical mask to the serial
    ///   paged prefill); a slot with `n_query == 1` does the decode gather +
    ///   `gqa_attention_flash_decode*` / `gqa_attention_one_step*` at the same
    ///   `flash_kv_len_min()` threshold the batched decoder uses — so each
    ///   slot's attention output is bit-identical to its standalone forward.
    ///
    /// # SWA / softcap
    ///
    /// The per-layer sliding-window span and Gemma2 attention-logit soft-cap are
    /// threaded into the attention calls (the `*_capped_windowed` kernels + the
    /// window/softcap args of the GPU prefill hook), so an SWA / Gemma2 model
    /// stays correct inside a mixed batch. At `window == 0` / `softcap == 0.0`
    /// (every ordinary model) these delegate to the plain kernels and are
    /// byte-identical to the non-ragged paged forwards.
    ///
    /// # Logits
    ///
    /// Only slots with a non-empty `logits_out` get the final norm + LM head
    /// (decode rows that need to sample); a prefill chunk mid-prompt passes an
    /// empty slice and its LM head is skipped — matching the serial paged
    /// prefill, which discards all but the last token's logits (and here even
    /// those, since the driver samples a prefilled slot on the NEXT tick's
    /// decode row, not from its final prefill chunk).
    ///
    /// # Contract
    ///
    /// F32 paged KV only (the paged v1 dtype). The caller must have
    /// `ensure_capacity_shared`'d each slot's cache to at least
    /// `start_pos + n_query`. Per-slot `cache` / `logits_out` borrows must be
    /// disjoint. Parity is pinned by
    /// `forward_ragged_paged_f32_mixed_matches_sequential_prefill_then_decode`.
    ///
    /// F6-B — image-embed-row source: `embed_rows` optionally supplies the
    /// `[Σ n_query × d_model]` row-major INPUT embeddings for the concatenated
    /// token buffer instead of the per-token embedding-table lookup. `None`
    /// (the driver's text path) is byte-identical to the token-id embed. `Some`
    /// is the VLM path — a prompt's image-placeholder positions carry projected
    /// patch embeddings (from the vision splice, exactly like the contiguous
    /// `forward_prefill_from_embeds`), so the spliced rows get written into the
    /// paged KV. The rows still RoPE + attend at their sequential `start_pos +
    /// j` positions, identical to the token path; only the first embed step
    /// differs. See `forward_prefill_paged_from_embeds_f32`.
    pub fn forward_ragged_paged_f32(
        &self,
        tokens: &[i32],
        slots: &mut [crate::llama_arch::RaggedSlot<'_>],
        shared: &crate::shared_paged_kv::SharedPagedKv,
        embed_rows: Option<&[f32]>,
    ) {
        self.forward_ragged_paged_cascade_f32(tokens, slots, shared, embed_rows, None)
    }

    /// P3 cascade-aware variant of [`Self::forward_ragged_paged_f32`] — the
    /// paged driver's decode tick. `cascade` groups DECODE rows (`n_query == 1`)
    /// that share a page-aligned APC prefix so the shared-prefix attention is
    /// computed once and merged per slot (full-attention layers only). Prefill
    /// chunks always take the per-slot path. `None` / empty ⇒ byte-identical to
    /// the plain ragged forward.
    pub fn forward_ragged_paged_cascade_f32(
        &self,
        tokens: &[i32],
        slots: &mut [crate::llama_arch::RaggedSlot<'_>],
        shared: &crate::shared_paged_kv::SharedPagedKv,
        embed_rows: Option<&[f32]>,
        cascade: Option<&[CascadeGroup]>,
    ) {
        let n_slots = slots.len();
        if n_slots == 0 {
            return;
        }
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = cfg.d_ff;
        let vocab = cfg.vocab_size;

        // Running-sum row offset per slot into the concatenated `[total]` row
        // dim. `tokens` MUST be laid out in this same slot order; we assert the
        // total matches so a misbuilt batch panics at the boundary, not inside a
        // wrong-size gather 300 lines down.
        let total: usize = slots.iter().map(|s| s.n_query).sum();
        assert_eq!(
            total,
            tokens.len(),
            "ragged forward: concatenated token count {} != Σ n_query {}",
            tokens.len(),
            total,
        );
        if total == 0 {
            return;
        }
        let mut row_off = vec![0usize; n_slots];
        {
            let mut acc = 0usize;
            for (i, s) in slots.iter().enumerate() {
                row_off[i] = acc;
                acc += s.n_query;
            }
        }
        // Per-slot sanity (cheap; pins the panic site to the bad slot).
        for (i, s) in slots.iter().enumerate() {
            assert!(s.n_query >= 1, "ragged slot {i}: n_query must be >= 1");
            debug_assert!(
                (s.start_pos as usize + s.n_query) as u32 <= s.cache.capacity_tokens(),
                "ragged slot {i}: end pos {} > cache capacity {} \
                 (caller must ensure_capacity_shared first)",
                s.start_pos as usize + s.n_query,
                s.cache.capacity_tokens(),
            );
            debug_assert!(
                s.logits_out.is_empty() || s.logits_out.len() == vocab,
                "ragged slot {i}: logits_out len {} must be 0 (skip LM head) or vocab {vocab}",
                s.logits_out.len(),
            );
        }

        // Batched scratch over the concatenated row dim — same shapes the
        // prefill / batched-decode mirrors use, just sized to `total` rows that
        // may belong to different slots.
        let mut hidden = vec![0f32; total * d];
        let mut h_norm = vec![0f32; total * d];
        let mut q_buf = vec![0f32; total * d_q];
        let mut k_buf = vec![0f32; total * d_kv];
        let mut v_buf = vec![0f32; total * d_kv];
        let mut attn_out = vec![0f32; total * d_q];
        let mut attn_proj = vec![0f32; total * d];
        let mut gate_buf = vec![0f32; total * d_ff];
        let mut up_buf = vec![0f32; total * d_ff];
        let mut ffn_buf = vec![0f32; total * d_ff];
        let mut ffn_out = vec![0f32; total * d];

        // Per-layer gather slab, reused across slots within a layer (each slot's
        // gather overwrites it before that slot's attention reads it). Sized to
        // the largest slot's kv_len so one buffer covers every slot.
        let max_kv_len = slots
            .iter()
            .map(|s| s.start_pos as usize + s.n_query)
            .max()
            .unwrap_or(0);
        let slab_len = n_kv_heads * max_kv_len * head_dim;
        let mut k_slab = vec![0f32; slab_len];
        let mut v_slab = vec![0f32; slab_len];

        // Embed every concatenated token into its hidden row. F6-B: when
        // `embed_rows` is supplied (the VLM splice path), copy the pre-computed
        // input embeddings verbatim instead of the token-id lookup — the
        // image-placeholder rows carry projected patch features. `None` (the
        // text path) keeps the byte-identical token-embed lookup.
        match embed_rows {
            Some(rows) => {
                assert_eq!(
                    rows.len(),
                    total * d,
                    "forward_ragged_paged_f32: embed_rows len {} != Σ n_query × d_model {}",
                    rows.len(),
                    total * d,
                );
                hidden.copy_from_slice(rows);
            }
            None => {
                for (r, &tok) in tokens.iter().enumerate() {
                    k::embed_lookup_tensor(
                        &self.weights.token_embd,
                        &[tok],
                        &mut hidden[r * d..(r + 1) * d],
                        d,
                    );
                }
            }
        }

        // MoE per-token scratch (per-token routing ⇒ a per-row loop, exactly
        // like the batched decoder).
        let moe_cfg = cfg.moe.clone();
        let (mut moe_gate_one, mut moe_up_one, mut moe_ff_one, mut moe_down_one) =
            if moe_cfg.is_some() {
                (
                    vec![0f32; d_ff],
                    vec![0f32; d_ff],
                    vec![0f32; d_ff],
                    vec![0f32; d],
                )
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };
        let (mut moe_expert_logits, mut moe_routed_picks) = if let Some(m) = moe_cfg.as_ref() {
            (
                vec![0f32; m.n_experts as usize],
                Vec::with_capacity(m.n_experts_used as usize),
            )
        } else {
            (Vec::new(), Vec::new())
        };

        let attn_views: Vec<&dyn AttnBlock> = if let Some(mbs) = self.weights.moe_blocks.as_ref() {
            mbs.iter().map(|b| b as &dyn AttnBlock).collect()
        } else {
            self.weights
                .blocks
                .iter()
                .map(|b| b as &dyn AttnBlock)
                .collect()
        };

        let n_layers_used = crate::accel::max_layers_cap()
            .unwrap_or(attn_views.len())
            .min(attn_views.len());

        for (layer_idx, block) in attn_views.iter().take(n_layers_used).enumerate() {
            crate::accel::set_current_layer_idx(layer_idx as u32);
            // Per-row pre-attention rmsnorm.
            for r in 0..total {
                k::rmsnorm_f32_row(
                    &hidden[r * d..(r + 1) * d],
                    block.attn_norm(),
                    &mut h_norm[r * d..(r + 1) * d],
                    cfg.rms_eps,
                );
            }
            // Batched Q / K / V projections over the whole concatenated row dim.
            matvec_tensor_batched_dispatch(block.w_q(), &h_norm, &mut q_buf, d_q, d, total);
            matvec_tensor_batched_dispatch(block.w_k(), &h_norm, &mut k_buf, d_kv, d, total);
            matvec_tensor_batched_dispatch(block.w_v(), &h_norm, &mut v_buf, d_kv, d, total);

            // Per-slot / per-token bias + Qwen3 q/k-norm + RoPE. Position is
            // `start_pos + j` for query `j` in the slot (for a decode row,
            // `start_pos` itself). q/k/v live in separate buffers so the three
            // disjoint `&mut` row slices coexist.
            for (i, s) in slots.iter().enumerate() {
                let base = row_off[i];
                for j in 0..s.n_query {
                    let r = base + j;
                    let q_row = &mut q_buf[r * d_q..(r + 1) * d_q];
                    let k_row = &mut k_buf[r * d_kv..(r + 1) * d_kv];
                    let v_row = &mut v_buf[r * d_kv..(r + 1) * d_kv];
                    if let Some(bq) = block.b_q() {
                        k::add_inplace_f32(q_row, bq);
                    }
                    if let Some(bk) = block.b_k() {
                        k::add_inplace_f32(k_row, bk);
                    }
                    if let Some(bv) = block.b_v() {
                        k::add_inplace_f32(v_row, bv);
                    }
                    apply_qk_head_norm(
                        block.q_norm(),
                        block.k_norm(),
                        q_row,
                        k_row,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        cfg.rms_eps,
                    );
                    let pos = s.start_pos + j as u32;
                    self.rope_inplace(q_row, n_heads, head_dim, pos);
                    self.rope_inplace(k_row, n_kv_heads, head_dim, pos);
                }
            }

            // Per-slot: write this slot's new K/V rows, gather its full KV
            // history for the layer, run the slot-kind-appropriate attention.
            let swa_win = cfg.swa_window(layer_idx);
            let softcap = cfg.attn_logit_softcapping.unwrap_or(0.0);
            // Pass 1: write every slot's new K/V rows for this layer (split from
            // the attention pass so the P3 cascade can gather a shared prefix
            // after all writes — each slot writes only its OWN pages, so this is
            // byte-identical to the interleaved loop).
            for (i, s) in slots.iter_mut().enumerate() {
                let base = row_off[i];
                for j in 0..s.n_query {
                    let r = base + j;
                    let pos = s.start_pos + j as u32;
                    let k_row = &k_buf[r * d_kv..(r + 1) * d_kv];
                    let v_row = &v_buf[r * d_kv..(r + 1) * d_kv];
                    s.cache
                        .write_token_shared(shared, layer_idx as u32, pos, k_row, v_row)
                        .expect("ragged write_token_shared: pos < capacity (asserted above)");
                }
            }

            // Pass 2a: P3 cascade over DECODE rows (n_query == 1) sharing a
            // page-aligned APC prefix (full-attention layers only). Same
            // numeric-equivalence + gating as the batched decoder; prefill
            // chunks always take the per-slot path in pass 2b.
            let mut handled = vec![false; n_slots];
            if swa_win == 0 {
                if let Some(groups) = cascade {
                    for g in groups {
                        if g.prefix_len == 0 || g.members.len() < 2 {
                            continue;
                        }
                        if g.members.iter().any(|&mi| {
                            mi >= n_slots
                                || slots[mi].n_query != 1
                                || handled[mi]
                                || (slots[mi].start_pos as usize + 1) < g.prefix_len
                        }) {
                            continue;
                        }
                        let nm = g.members.len();
                        let caches: Vec<&crate::paged_kv_cache::PagedKvCache> =
                            g.members.iter().map(|&mi| &*slots[mi].cache).collect();
                        let kv_lens: Vec<usize> = g
                            .members
                            .iter()
                            .map(|&mi| slots[mi].start_pos as usize + 1)
                            .collect();
                        let mut queries = vec![0f32; nm * d_q];
                        for (gi, &mi) in g.members.iter().enumerate() {
                            let base = row_off[mi];
                            queries[gi * d_q..(gi + 1) * d_q]
                                .copy_from_slice(&q_buf[base * d_q..(base + 1) * d_q]);
                        }
                        let mut group_out = vec![0f32; nm * d_q];
                        cascade_decode_attention_group(
                            shared,
                            layer_idx as u32,
                            g.prefix_len,
                            &caches,
                            &kv_lens,
                            &queries,
                            &mut group_out,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            softcap,
                        );
                        for (gi, &mi) in g.members.iter().enumerate() {
                            let base = row_off[mi];
                            attn_out[base * d_q..(base + 1) * d_q]
                                .copy_from_slice(&group_out[gi * d_q..(gi + 1) * d_q]);
                            handled[mi] = true;
                        }
                    }
                }
            }

            // Pass 2b: per-slot gather + attention for the rest (prefill chunks +
            // non-grouped decode rows), byte-identical to the pre-P3 path.
            for (i, s) in slots.iter().enumerate() {
                if handled[i] {
                    continue;
                }
                let base = row_off[i];
                let kv_len = s.start_pos as usize + s.n_query;
                // Phase D/E/F (GPU paged attention): for a DECODE row (n_query ==
                // 1), try the block-table flash-decode against a device KV-pool
                // mirror first — a hit SKIPS the host gather + CPU attention below.
                // The combinator tiers CUDA → MLX → SYCL; inert / false off-GPU or
                // without the per-backend verdict => the gather + CPU path (prefill
                // chunks, n_query > 1, always gather). `PageId` is
                // repr(transparent) => zero-copy `&[u32]` block table.
                if s.n_query == 1 {
                    let pages = s.cache.pages();
                    let block_table: &[u32] = unsafe {
                        std::slice::from_raw_parts(pages.as_ptr() as *const u32, pages.len())
                    };
                    if crate::accel::try_flash_attn_decode_paged_gpu_f32(
                        &q_buf[base * d_q..(base + 1) * d_q],
                        &mut attn_out[base * d_q..(base + 1) * d_q],
                        block_table,
                        layer_idx as u32,
                        kv_len as u32,
                        n_heads as u32,
                        n_kv_heads as u32,
                        head_dim as u32,
                        swa_win as i32,
                        softcap,
                    ) {
                        continue;
                    }
                }
                // Gather `[0, kv_len)` into the shared slab prefix.
                let slab_used = n_kv_heads * kv_len * head_dim;
                let k_slice = &mut k_slab[..slab_used];
                let v_slice = &mut v_slab[..slab_used];
                s.cache
                    .gather_layer_shared(shared, layer_idx as u32, k_slice, v_slice)
                    .expect("ragged gather_layer_shared: cache covers kv_len positions");

                let q_slice = &q_buf[base * d_q..(base + s.n_query) * d_q];
                let out_slice = &mut attn_out[base * d_q..(base + s.n_query) * d_q];
                if s.n_query > 1 {
                    // PREFILL CHUNK — varlen causal flash-prefill. `kv_len_base
                    // = start_pos`, `n_new = n_query`: query `j` attends
                    // `[0, start_pos + j]`, exactly the serial paged prefill's
                    // mask. GPU hook first (self-contained copy-in/out; inert
                    // off-GPU), CPU capped+windowed fallback otherwise.
                    let used_gpu = crate::accel::try_flash_attn_prefill_gpu_f32(
                        q_slice,
                        k_slice,
                        v_slice,
                        out_slice,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        kv_len,
                        s.start_pos as usize,
                        s.n_query,
                        swa_win as u32,
                        softcap,
                    );
                    if !used_gpu {
                        k::gqa_attention_flash_prefill_capped_windowed(
                            q_slice,
                            k_slice,
                            v_slice,
                            out_slice,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            kv_len,
                            s.start_pos as usize,
                            s.n_query,
                            swa_win,
                            softcap,
                        );
                    }
                } else {
                    // DECODE ROW (CPU fallback) — gather + flash-decode / standard
                    // at the same threshold the batched decoder uses, so this row
                    // is bit-identical to its standalone decode. The GPU block-
                    // table flash-decode was tried above (Phase D); this runs only
                    // when it is off / unavailable (fail-closed).
                    let flash_min = flash_kv_len_min();
                    if crate::accel::flash_attention_enabled() && kv_len >= flash_min {
                        k::gqa_attention_flash_decode_capped_windowed(
                            q_slice, k_slice, v_slice, out_slice, n_heads, n_kv_heads, head_dim,
                            kv_len, kv_len, swa_win, softcap,
                        );
                    } else {
                        k::gqa_attention_one_step_capped_windowed(
                            q_slice, k_slice, v_slice, out_slice, n_heads, n_kv_heads, head_dim,
                            kv_len, kv_len, swa_win, softcap,
                        );
                    }
                }
            }

            // Batched output projection + per-row residual + post-attn rmsnorm.
            matvec_tensor_batched_dispatch(block.w_o(), &attn_out, &mut attn_proj, d, d_q, total);
            for r in 0..total {
                let proj_row = &attn_proj[r * d..(r + 1) * d];
                let h_row = &mut hidden[r * d..(r + 1) * d];
                if !skip_attn_enabled() {
                    k::add_inplace_f32(h_row, proj_row);
                }
                let n_row = &mut h_norm[r * d..(r + 1) * d];
                k::rmsnorm_f32_row(h_row, block.ffn_norm(), n_row, cfg.rms_eps);
            }
            // FFN: per-row MoE loop OR dense batched matvec ladder.
            if let Some(moe) = moe_cfg.as_ref() {
                let mb = &self.weights.moe_blocks.as_ref().unwrap()[layer_idx];
                for r in 0..total {
                    let n_row = &h_norm[r * d..(r + 1) * d];
                    let out_row = &mut ffn_out[r * d..(r + 1) * d];
                    crate::moe::moe_ffn_one_into(
                        n_row,
                        mb,
                        d,
                        d_ff,
                        moe.n_experts as usize,
                        moe.n_experts_used as usize,
                        out_row,
                        &mut moe_gate_one,
                        &mut moe_up_one,
                        &mut moe_ff_one,
                        &mut moe_down_one,
                        &mut moe_expert_logits,
                        &mut moe_routed_picks,
                    );
                }
            } else {
                let dense_block = &self.weights.blocks[layer_idx];
                matvec_tensor_batched_dispatch(
                    &dense_block.w_gate,
                    &h_norm,
                    &mut gate_buf,
                    d_ff,
                    d,
                    total,
                );
                matvec_tensor_batched_dispatch(
                    &dense_block.w_up,
                    &h_norm,
                    &mut up_buf,
                    d_ff,
                    d,
                    total,
                );
                for r in 0..total {
                    k::silu_mul_f32(
                        &gate_buf[r * d_ff..(r + 1) * d_ff],
                        &up_buf[r * d_ff..(r + 1) * d_ff],
                        &mut ffn_buf[r * d_ff..(r + 1) * d_ff],
                    );
                }
                matvec_tensor_batched_dispatch(
                    &dense_block.w_down,
                    &ffn_buf,
                    &mut ffn_out,
                    d,
                    d_ff,
                    total,
                );
            }
            // Per-row FFN residual.
            if !skip_ffn_enabled() {
                for r in 0..total {
                    k::add_inplace_f32(
                        &mut hidden[r * d..(r + 1) * d],
                        &ffn_out[r * d..(r + 1) * d],
                    );
                }
            }
        }

        // Final norm + LM head for the rows that want logits only: the LAST
        // query token of each slot carrying a non-empty `logits_out` (decode
        // rows). Prefill chunks mid-prompt are skipped — their logits are unused
        // (the driver samples a prefilled slot on its next tick's decode row).
        // Per-row CPU rmsnorm here (bit-identical to the batched decoder's
        // CPU-fallback final norm); the LM-head matmul still goes through the
        // batched dispatch (one GEMM over the logit rows), row-independent so
        // each row equals its standalone LM head.
        let logit_rows: Vec<(usize, usize)> = slots
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.logits_out.is_empty())
            .map(|(i, s)| (i, row_off[i] + s.n_query - 1))
            .collect();
        if logit_rows.is_empty() {
            return;
        }
        let n_logit = logit_rows.len();
        let lm_head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        let mut final_norm = vec![0f32; n_logit * d];
        for (p, &(_, row)) in logit_rows.iter().enumerate() {
            k::rmsnorm_f32_row(
                &hidden[row * d..(row + 1) * d],
                &self.weights.output_norm,
                &mut final_norm[p * d..(p + 1) * d],
                cfg.rms_eps,
            );
        }
        let mut logits_all = vec![0f32; n_logit * vocab];
        // P11 (LoRA): the LM head runs over a SUBSET of rows (`logit_rows`) in
        // their own order, not the concatenated row order the per-row adapter
        // table is indexed by. Remap the table to logit-row order for this one
        // batched dispatch (row `p` uses the adapter at concatenated row
        // `logit_rows[p].1`). `None` (no row table active) ⇒ no scope ⇒
        // byte-identical. The guard restores the concatenated table on drop.
        let _lm_lora = crate::lora::active_row_adapters().map(|t| {
            let sub: Vec<_> = logit_rows.iter().map(|&(_, row)| t.get(row)).collect();
            crate::lora::scope_row_adapters(Some(std::sync::Arc::new(
                crate::lora::RowAdapters::new(sub),
            )))
        });
        matvec_tensor_batched_dispatch(lm_head, &final_norm, &mut logits_all, vocab, d, n_logit);
        drop(_lm_lora);
        self.finalize_logits(&mut logits_all);
        for (p, &(slot_i, _)) in logit_rows.iter().enumerate() {
            slots[slot_i]
                .logits_out
                .copy_from_slice(&logits_all[p * vocab..(p + 1) * vocab]);
        }
    }

    /// F6-B — paged VLM prefill from a pre-spliced INPUT-embedding buffer.
    /// The paged analog of the contiguous [`Self::forward_prefill_from_embeds`]:
    /// given a `[seq_len × d_model]` row-major embedding sequence (the vision
    /// splice output — text-token rows with image-placeholder positions replaced
    /// by projected patch rows), write every row's K/V into the slot's paged
    /// cache at positions `[start_pos, start_pos + seq_len)` and return the FINAL
    /// position's logits (same shape `forward_prefill_paged_f32` returns).
    ///
    /// Routes through [`Self::forward_ragged_paged_f32`] with a single prefill
    /// slot and the `embed_rows` source, so it shares the batched layer body
    /// (bit-tolerant to the serial contiguous `forward_prefill_from_embeds` up to
    /// flash-prefill reduction order, exactly like the text batched-vs-serial
    /// prefill). The caller must `ensure_capacity_shared` the cache to at least
    /// `start_pos + seq_len` first.
    pub fn forward_prefill_paged_from_embeds_f32(
        &self,
        embeds: &[f32],
        start_pos: u32,
        cache: &mut crate::paged_kv_cache::PagedKvCache,
        shared: &crate::shared_paged_kv::SharedPagedKv,
    ) -> Vec<f32> {
        let d = self.cfg.d_model;
        assert_eq!(
            embeds.len() % d,
            0,
            "forward_prefill_paged_from_embeds_f32: embeds.len()={} not a multiple of d_model={}",
            embeds.len(),
            d,
        );
        let n_new = embeds.len() / d;
        let mut logits = vec![0f32; self.cfg.vocab_size];
        if n_new == 0 {
            return logits;
        }
        // One ragged PREFILL slot covering the whole spliced sequence; the
        // ragged forward writes each row's K/V and emits the LAST row's logits
        // (its `logit_rows` picks the final query of a slot with a non-empty
        // `logits_out`), matching the contiguous prefill's "return final logits".
        let dummy_tokens: Vec<i32> = vec![0; n_new];
        let mut slot = crate::llama_arch::RaggedSlot {
            n_query: n_new,
            start_pos,
            cache,
            logits_out: &mut logits,
        };
        self.forward_ragged_paged_f32(
            &dummy_tokens,
            std::slice::from_mut(&mut slot),
            shared,
            Some(embeds),
        );
        logits
    }

    /// F7 — paged tree-structured speculative verification forward (F32 KV).
    /// The paged analog of [`Self::forward_tree_verify_f32`]: scores a draft
    /// TREE in ONE forward over the slot's paged KV. Node `i`'s token is written
    /// to paged slot `start_pos + i`, RoPE'd at its DEPTH (`start_pos +
    /// depths[i]`), and attends only the committed prefix `[0, start_pos)` plus
    /// its own ancestor chain among the tree slots (via the shared
    /// [`rustllama_kernels_cpu::gqa_attention_tree_prefill`] over the gathered
    /// slab). Writes per-node logits into `logits_out` (`[nodes × vocab]`).
    ///
    /// `parent[i]` is the tree-local parent index (negative = root level),
    /// topologically ordered (`parent[i] < i`). A CHAIN-shaped tree
    /// (`depths[i] == i`, `parent[i] == i-1`) reduces to `n` sequential paged
    /// decodes — bit-exact up to the flash-vs-tree attention reduction order,
    /// the same tolerance the contiguous chain guard uses — pinned by
    /// `forward_tree_verify_paged_f32_chain_matches_linear`. Reuses
    /// `gqa_attention_tree_prefill` (no duplicated mask math) and handles both
    /// dense and MoE blocks like [`Self::forward_ragged_paged_f32`].
    ///
    /// Contract: F32 paged KV only; the caller must `ensure_capacity_shared` the
    /// cache to at least `start_pos + nodes.len()`, and the committed prefix
    /// `[0, start_pos)` (including the tip written by
    /// [`Self::forward_one_with_mtp_logits_paged`]) must already be resident.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tree_verify_paged_f32(
        &self,
        tokens: &[i32],
        start_pos: u32,
        depths: &[u32],
        parent: &[i32],
        cache: &mut crate::paged_kv_cache::PagedKvCache,
        shared: &crate::shared_paged_kv::SharedPagedKv,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        let n_new = tokens.len();
        assert_eq!(depths.len(), n_new, "paged tree verify: one depth per node");
        assert_eq!(
            parent.len(),
            n_new,
            "paged tree verify: one parent per node"
        );
        assert_eq!(
            logits_out.len(),
            n_new * cfg.vocab_size,
            "forward_tree_verify_paged_f32: logits_out must be nodes * vocab_size"
        );
        if n_new == 0 {
            return;
        }
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = cfg.d_ff;
        let vocab = cfg.vocab_size;
        let kv_len_base = start_pos as usize;
        // Total positions attended = committed prefix + the tree nodes. The
        // gather slab is strided by this (it is the `max_ctx` the tree kernel
        // sees), and `kv_len_base + n_new == kv_len` holds exactly.
        let kv_len = kv_len_base + n_new;
        debug_assert!(
            (kv_len as u32) <= cache.capacity_tokens(),
            "paged tree verify: cache capacity {} < kv_len {kv_len} \
             (caller must ensure_capacity_shared first)",
            cache.capacity_tokens(),
        );

        let mut hidden = vec![0f32; n_new * d];
        let mut h_norm = vec![0f32; n_new * d];
        let mut q_buf = vec![0f32; n_new * d_q];
        let mut k_buf = vec![0f32; n_new * d_kv];
        let mut v_buf = vec![0f32; n_new * d_kv];
        let mut attn_out = vec![0f32; n_new * d_q];
        let mut attn_proj = vec![0f32; n_new * d];
        let mut gate_buf = vec![0f32; n_new * d_ff];
        let mut up_buf = vec![0f32; n_new * d_ff];
        let mut ffn_buf = vec![0f32; n_new * d_ff];
        let mut ffn_out = vec![0f32; n_new * d];

        let slab_len = n_kv_heads * kv_len * head_dim;
        let mut k_slab = vec![0f32; slab_len];
        let mut v_slab = vec![0f32; slab_len];

        for (i, &tok) in tokens.iter().enumerate() {
            k::embed_lookup_tensor(
                &self.weights.token_embd,
                &[tok],
                &mut hidden[i * d..(i + 1) * d],
                d,
            );
        }

        let moe_cfg = cfg.moe.clone();
        let (mut moe_gate_one, mut moe_up_one, mut moe_ff_one, mut moe_down_one) =
            if moe_cfg.is_some() {
                (
                    vec![0f32; d_ff],
                    vec![0f32; d_ff],
                    vec![0f32; d_ff],
                    vec![0f32; d],
                )
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };
        let (mut moe_expert_logits, mut moe_routed_picks) = if let Some(m) = moe_cfg.as_ref() {
            (
                vec![0f32; m.n_experts as usize],
                Vec::with_capacity(m.n_experts_used as usize),
            )
        } else {
            (Vec::new(), Vec::new())
        };

        let attn_views: Vec<&dyn AttnBlock> = if let Some(mbs) = self.weights.moe_blocks.as_ref() {
            mbs.iter().map(|b| b as &dyn AttnBlock).collect()
        } else {
            self.weights
                .blocks
                .iter()
                .map(|b| b as &dyn AttnBlock)
                .collect()
        };
        let n_layers_used = crate::accel::max_layers_cap()
            .unwrap_or(attn_views.len())
            .min(attn_views.len());

        for (layer_idx, block) in attn_views.iter().take(n_layers_used).enumerate() {
            crate::accel::set_current_layer_idx(layer_idx as u32);
            for i in 0..n_new {
                k::rmsnorm_f32_row(
                    &hidden[i * d..(i + 1) * d],
                    block.attn_norm(),
                    &mut h_norm[i * d..(i + 1) * d],
                    cfg.rms_eps,
                );
            }
            matvec_tensor_batched_dispatch(block.w_q(), &h_norm, &mut q_buf, d_q, d, n_new);
            matvec_tensor_batched_dispatch(block.w_k(), &h_norm, &mut k_buf, d_kv, d, n_new);
            matvec_tensor_batched_dispatch(block.w_v(), &h_norm, &mut v_buf, d_kv, d, n_new);
            // Per-node bias + Qwen3 q/k-norm + RoPE at the node's DEPTH position
            // (siblings share a position), so a chain tree matches sequential.
            for i in 0..n_new {
                let q_row = &mut q_buf[i * d_q..(i + 1) * d_q];
                let k_row = &mut k_buf[i * d_kv..(i + 1) * d_kv];
                let v_row = &mut v_buf[i * d_kv..(i + 1) * d_kv];
                if let Some(bq) = block.b_q() {
                    k::add_inplace_f32(q_row, bq);
                }
                if let Some(bk) = block.b_k() {
                    k::add_inplace_f32(k_row, bk);
                }
                if let Some(bv) = block.b_v() {
                    k::add_inplace_f32(v_row, bv);
                }
                apply_qk_head_norm(
                    block.q_norm(),
                    block.k_norm(),
                    q_row,
                    k_row,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    cfg.rms_eps,
                );
                let pos = start_pos + depths[i];
                self.rope_inplace(q_row, n_heads, head_dim, pos);
                self.rope_inplace(k_row, n_kv_heads, head_dim, pos);
            }
            // Write each node's K/V at its FLAT paged position `start_pos + i`.
            for i in 0..n_new {
                let pos = start_pos + i as u32;
                let k_row = &k_buf[i * d_kv..(i + 1) * d_kv];
                let v_row = &v_buf[i * d_kv..(i + 1) * d_kv];
                cache
                    .write_token_shared(shared, layer_idx as u32, pos, k_row, v_row)
                    .expect("paged tree verify write_token_shared: pos < capacity");
            }
            // Gather `[0, kv_len)` (committed prefix + tree nodes) into the slab.
            cache
                .gather_layer_shared(shared, layer_idx as u32, &mut k_slab, &mut v_slab)
                .expect("paged tree verify gather_layer_shared: cache covers kv_len");
            // Ancestor-masked attention: the committed prefix `[0, start_pos)` is
            // seen by every node; a tree slot only by nodes whose ancestor chain
            // contains it. Slab stride == `kv_len`, tree slots at `[start_pos,
            // start_pos + n_new)`. A chain `parent` reduces this to causal.
            k::gqa_attention_tree_prefill(
                &q_buf,
                &k_slab,
                &v_slab,
                &mut attn_out,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_len,
                kv_len_base,
                n_new,
                parent,
            );
            matvec_tensor_batched_dispatch(block.w_o(), &attn_out, &mut attn_proj, d, d_q, n_new);
            for i in 0..n_new {
                let proj_row = &attn_proj[i * d..(i + 1) * d];
                let h_row = &mut hidden[i * d..(i + 1) * d];
                if !skip_attn_enabled() {
                    k::add_inplace_f32(h_row, proj_row);
                }
                let n_row = &mut h_norm[i * d..(i + 1) * d];
                k::rmsnorm_f32_row(h_row, block.ffn_norm(), n_row, cfg.rms_eps);
            }
            if let Some(moe) = moe_cfg.as_ref() {
                let mb = &self.weights.moe_blocks.as_ref().unwrap()[layer_idx];
                for i in 0..n_new {
                    let n_row = &h_norm[i * d..(i + 1) * d];
                    let out_row = &mut ffn_out[i * d..(i + 1) * d];
                    crate::moe::moe_ffn_one_into(
                        n_row,
                        mb,
                        d,
                        d_ff,
                        moe.n_experts as usize,
                        moe.n_experts_used as usize,
                        out_row,
                        &mut moe_gate_one,
                        &mut moe_up_one,
                        &mut moe_ff_one,
                        &mut moe_down_one,
                        &mut moe_expert_logits,
                        &mut moe_routed_picks,
                    );
                }
            } else {
                let dense_block = &self.weights.blocks[layer_idx];
                matvec_tensor_batched_dispatch(
                    &dense_block.w_gate,
                    &h_norm,
                    &mut gate_buf,
                    d_ff,
                    d,
                    n_new,
                );
                matvec_tensor_batched_dispatch(
                    &dense_block.w_up,
                    &h_norm,
                    &mut up_buf,
                    d_ff,
                    d,
                    n_new,
                );
                for i in 0..n_new {
                    k::silu_mul_f32(
                        &gate_buf[i * d_ff..(i + 1) * d_ff],
                        &up_buf[i * d_ff..(i + 1) * d_ff],
                        &mut ffn_buf[i * d_ff..(i + 1) * d_ff],
                    );
                }
                matvec_tensor_batched_dispatch(
                    &dense_block.w_down,
                    &ffn_buf,
                    &mut ffn_out,
                    d,
                    d_ff,
                    n_new,
                );
            }
            if !skip_ffn_enabled() {
                for i in 0..n_new {
                    k::add_inplace_f32(
                        &mut hidden[i * d..(i + 1) * d],
                        &ffn_out[i * d..(i + 1) * d],
                    );
                }
            }
        }

        // Final norm + LM head for EVERY node (per-node logits for the verifier).
        let lm_head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        let mut final_norm = vec![0f32; n_new * d];
        for i in 0..n_new {
            k::rmsnorm_f32_row(
                &hidden[i * d..(i + 1) * d],
                &self.weights.output_norm,
                &mut final_norm[i * d..(i + 1) * d],
                cfg.rms_eps,
            );
        }
        matvec_tensor_batched_dispatch(lm_head, &final_norm, logits_out, vocab, d, n_new);
        self.finalize_logits(logits_out);
    }

    /// Paged-KV variant of [`Self::forward_one`] (single decode step).
    /// Mirrors the F32 path of the contiguous decoder but routes the
    /// new K/V row through [`crate::paged_kv_cache::PagedKvCache`]
    /// and gathers the per-layer slab before attention. CPU-only for
    /// the per-token kernels (matches the prefill mirror's contract);
    /// matmuls still go through `matvec_tensor_dispatch` which has
    /// its own SYCL ladder.
    ///
    /// Bit-identical to `forward_one` for F32 KV when both run on the
    /// CPU per-token path — the gather just reconstructs the same
    /// slab the contiguous path would have indexed into.
    ///
    /// The caller must call
    /// [`crate::paged_kv_cache::PagedKvCache::ensure_capacity`] with
    /// at least `pos + 1` tokens before invoking — this function
    /// asserts the cache has room but won't grow it (scheduler's job).
    pub fn forward_one_paged_f32(
        &self,
        token_id: i32,
        pos: u32,
        cache: &mut crate::paged_kv_cache::PagedKvCache,
        store: &mut crate::paged_kv_store::PagedKvStore,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        assert_eq!(logits_out.len(), cfg.vocab_size);
        debug_assert!(
            pos < cache.capacity_tokens(),
            "paged decode: cache capacity {} <= pos {pos}; caller must ensure_capacity first",
            cache.capacity_tokens(),
        );

        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = cfg.d_ff;
        // Env-gated KV eviction (sliding-window / heavy-hitter): when a
        // budget is set and engaged, the cache retains fewer than
        // `pos + 1` positions and `retained_len_for_pos` reports the
        // compacted count the gather slab + attention run over. With no
        // budget set this is exactly `pos + 1` (byte-identical).
        let kv_len = cache.retained_len_for_pos(pos);

        // Pool the per-token scratch into the thread-local
        // ForwardScratch — same pattern the contiguous forward_one
        // uses. The whole body runs inside `with_forward_scratch`
        // so the TLS cell holds the heap allocations across calls;
        // on the second+ token the buffers are reused with zero
        // realloc.
        let cfg_clone = self.cfg.clone();
        let (moe_n_experts, moe_top_k) = match cfg_clone.moe.as_ref() {
            Some(m) => (m.n_experts as usize, m.n_experts_used as usize),
            None => (0, 0),
        };
        crate::accel::with_forward_scratch(
            d,
            d_q,
            d_kv,
            d_ff,
            head_dim,
            cfg.rope_theta,
            moe_n_experts,
            moe_top_k,
            |scratch| {
                self.forward_one_paged_f32_inner(
                    token_id,
                    pos,
                    cache,
                    store as &mut dyn crate::paged_kv_store::PagedKvStoreOps,
                    logits_out,
                    scratch,
                    kv_len,
                )
            },
        );
    }

    /// H9b: Paged-KV TurboQuant forward dispatch.
    pub fn forward_one_paged_tq(
        &self,
        token_id: i32,
        pos: u32,
        cache: &mut crate::paged_kv_cache::PagedKvCache,
        store: &mut crate::paged_kv_store::PagedKvStoreTQ,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        assert_eq!(logits_out.len(), cfg.vocab_size);
        debug_assert!(pos < cache.capacity_tokens());
        let d = cfg.d_model;
        let head_dim = cfg.head_dim;
        let d_q = cfg.n_heads * head_dim;
        let d_kv = cfg.n_kv_heads * head_dim;
        let kv_len = cache.retained_len_for_pos(pos); // env-gated KV eviction; == pos+1 when off
        let cfg_clone = self.cfg.clone();
        let (moe_n_experts, moe_top_k) = match cfg_clone.moe.as_ref() {
            Some(m) => (m.n_experts as usize, m.n_experts_used as usize),
            None => (0, 0),
        };
        crate::accel::with_forward_scratch(
            d,
            d_q,
            d_kv,
            cfg.d_ff,
            head_dim,
            cfg.rope_theta,
            moe_n_experts,
            moe_top_k,
            |scratch| {
                self.forward_one_paged_f32_inner(
                    token_id,
                    pos,
                    cache,
                    store as &mut dyn crate::paged_kv_store::PagedKvStoreOps,
                    logits_out,
                    scratch,
                    kv_len,
                )
            },
        );
    }

    /// H9b: Paged-KV NVFP4 forward dispatch.
    pub fn forward_one_paged_nvfp4(
        &self,
        token_id: i32,
        pos: u32,
        cache: &mut crate::paged_kv_cache::PagedKvCache,
        store: &mut crate::paged_kv_store::PagedKvStoreNvfp4,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        assert_eq!(logits_out.len(), cfg.vocab_size);
        debug_assert!(pos < cache.capacity_tokens());
        let d = cfg.d_model;
        let head_dim = cfg.head_dim;
        let d_q = cfg.n_heads * head_dim;
        let d_kv = cfg.n_kv_heads * head_dim;
        let kv_len = cache.retained_len_for_pos(pos); // env-gated KV eviction; == pos+1 when off
        let cfg_clone = self.cfg.clone();
        let (moe_n_experts, moe_top_k) = match cfg_clone.moe.as_ref() {
            Some(m) => (m.n_experts as usize, m.n_experts_used as usize),
            None => (0, 0),
        };
        crate::accel::with_forward_scratch(
            d,
            d_q,
            d_kv,
            cfg.d_ff,
            head_dim,
            cfg.rope_theta,
            moe_n_experts,
            moe_top_k,
            |scratch| {
                self.forward_one_paged_f32_inner(
                    token_id,
                    pos,
                    cache,
                    store as &mut dyn crate::paged_kv_store::PagedKvStoreOps,
                    logits_out,
                    scratch,
                    kv_len,
                )
            },
        );
    }

    /// Wave 2: Paged-KV MXFP4 forward dispatch. Same code path as
    /// `forward_one_paged_nvfp4` — the MXFP4 store quantizes K/V rows
    /// on write and dequantizes on gather, so the forward function only
    /// ever sees F32 slabs. `head_dim` must be a multiple of 32.
    pub fn forward_one_paged_mxfp4(
        &self,
        token_id: i32,
        pos: u32,
        cache: &mut crate::paged_kv_cache::PagedKvCache,
        store: &mut crate::paged_kv_store::PagedKvStoreMxfp4,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        assert_eq!(logits_out.len(), cfg.vocab_size);
        debug_assert!(pos < cache.capacity_tokens());
        let d = cfg.d_model;
        let head_dim = cfg.head_dim;
        let d_q = cfg.n_heads * head_dim;
        let d_kv = cfg.n_kv_heads * head_dim;
        let kv_len = cache.retained_len_for_pos(pos); // env-gated KV eviction; == pos+1 when off
        let cfg_clone = self.cfg.clone();
        let (moe_n_experts, moe_top_k) = match cfg_clone.moe.as_ref() {
            Some(m) => (m.n_experts as usize, m.n_experts_used as usize),
            None => (0, 0),
        };
        crate::accel::with_forward_scratch(
            d,
            d_q,
            d_kv,
            cfg.d_ff,
            head_dim,
            cfg.rope_theta,
            moe_n_experts,
            moe_top_k,
            |scratch| {
                self.forward_one_paged_f32_inner(
                    token_id,
                    pos,
                    cache,
                    store as &mut dyn crate::paged_kv_store::PagedKvStoreOps,
                    logits_out,
                    scratch,
                    kv_len,
                )
            },
        );
    }

    /// Wave 2: Paged-KV MXFP6 forward dispatch. See
    /// [`Self::forward_one_paged_mxfp4`].
    pub fn forward_one_paged_mxfp6(
        &self,
        token_id: i32,
        pos: u32,
        cache: &mut crate::paged_kv_cache::PagedKvCache,
        store: &mut crate::paged_kv_store::PagedKvStoreMxfp6,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        assert_eq!(logits_out.len(), cfg.vocab_size);
        debug_assert!(pos < cache.capacity_tokens());
        let d = cfg.d_model;
        let head_dim = cfg.head_dim;
        let d_q = cfg.n_heads * head_dim;
        let d_kv = cfg.n_kv_heads * head_dim;
        let kv_len = cache.retained_len_for_pos(pos); // env-gated KV eviction; == pos+1 when off
        let cfg_clone = self.cfg.clone();
        let (moe_n_experts, moe_top_k) = match cfg_clone.moe.as_ref() {
            Some(m) => (m.n_experts as usize, m.n_experts_used as usize),
            None => (0, 0),
        };
        crate::accel::with_forward_scratch(
            d,
            d_q,
            d_kv,
            cfg.d_ff,
            head_dim,
            cfg.rope_theta,
            moe_n_experts,
            moe_top_k,
            |scratch| {
                self.forward_one_paged_f32_inner(
                    token_id,
                    pos,
                    cache,
                    store as &mut dyn crate::paged_kv_store::PagedKvStoreOps,
                    logits_out,
                    scratch,
                    kv_len,
                )
            },
        );
    }

    /// Wave 2: Paged-KV MXFP8 forward dispatch. See
    /// [`Self::forward_one_paged_mxfp4`].
    pub fn forward_one_paged_mxfp8(
        &self,
        token_id: i32,
        pos: u32,
        cache: &mut crate::paged_kv_cache::PagedKvCache,
        store: &mut crate::paged_kv_store::PagedKvStoreMxfp8,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        assert_eq!(logits_out.len(), cfg.vocab_size);
        debug_assert!(pos < cache.capacity_tokens());
        let d = cfg.d_model;
        let head_dim = cfg.head_dim;
        let d_q = cfg.n_heads * head_dim;
        let d_kv = cfg.n_kv_heads * head_dim;
        let kv_len = cache.retained_len_for_pos(pos); // env-gated KV eviction; == pos+1 when off
        let cfg_clone = self.cfg.clone();
        let (moe_n_experts, moe_top_k) = match cfg_clone.moe.as_ref() {
            Some(m) => (m.n_experts as usize, m.n_experts_used as usize),
            None => (0, 0),
        };
        crate::accel::with_forward_scratch(
            d,
            d_q,
            d_kv,
            cfg.d_ff,
            head_dim,
            cfg.rope_theta,
            moe_n_experts,
            moe_top_k,
            |scratch| {
                self.forward_one_paged_f32_inner(
                    token_id,
                    pos,
                    cache,
                    store as &mut dyn crate::paged_kv_store::PagedKvStoreOps,
                    logits_out,
                    scratch,
                    kv_len,
                )
            },
        );
    }

    /// H9a: Paged-KV variant of `forward_one` with Q8_0 storage.
    /// Same code path as `forward_one_paged_f32` — both delegate to
    /// `forward_one_paged_f32_inner` via the `PagedKvStoreOps` trait
    /// object. The Q8_0 store quantizes K/V rows on write and
    /// dequantizes on gather; from the forward function's perspective
    /// the data still arrives as F32 slabs for the attention kernel.
    pub fn forward_one_paged_q8_0(
        &self,
        token_id: i32,
        pos: u32,
        cache: &mut crate::paged_kv_cache::PagedKvCache,
        store: &mut crate::paged_kv_store::PagedKvStoreQ8_0,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        assert_eq!(logits_out.len(), cfg.vocab_size);
        debug_assert!(
            pos < cache.capacity_tokens(),
            "forward_one_paged_q8_0: pos {pos} >= capacity {}",
            cache.capacity_tokens(),
        );

        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = cfg.d_ff;
        let kv_len = cache.retained_len_for_pos(pos); // env-gated KV eviction; == pos+1 when off
        let cfg_clone = self.cfg.clone();
        let (moe_n_experts, moe_top_k) = match cfg_clone.moe.as_ref() {
            Some(m) => (m.n_experts as usize, m.n_experts_used as usize),
            None => (0, 0),
        };
        let _ = (n_heads, d_kv, d_q); // silence unused warnings; used inside _inner
        crate::accel::with_forward_scratch(
            d,
            d_q,
            d_kv,
            d_ff,
            head_dim,
            cfg.rope_theta,
            moe_n_experts,
            moe_top_k,
            |scratch| {
                self.forward_one_paged_f32_inner(
                    token_id,
                    pos,
                    cache,
                    store as &mut dyn crate::paged_kv_store::PagedKvStoreOps,
                    logits_out,
                    scratch,
                    kv_len,
                )
            },
        );
    }

    /// M6: half-precision (f16 / bf16) paged forward dispatch. Like
    /// [`Self::forward_one_paged_q8_0`], routes through the shared
    /// `forward_one_paged_f32_inner` via the `PagedKvStoreOps` trait
    /// object — the store widens its half cells to f32 on gather, so the
    /// inner f32 attention path runs unchanged.
    pub fn forward_one_paged_f16(
        &self,
        token_id: i32,
        pos: u32,
        cache: &mut crate::paged_kv_cache::PagedKvCache,
        store: &mut crate::paged_kv_store::PagedKvStoreF16,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        assert_eq!(logits_out.len(), cfg.vocab_size);
        debug_assert!(
            pos < cache.capacity_tokens(),
            "forward_one_paged_f16: pos {pos} >= capacity {}",
            cache.capacity_tokens(),
        );

        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = cfg.d_ff;
        let kv_len = cache.retained_len_for_pos(pos);
        let cfg_clone = self.cfg.clone();
        let (moe_n_experts, moe_top_k) = match cfg_clone.moe.as_ref() {
            Some(m) => (m.n_experts as usize, m.n_experts_used as usize),
            None => (0, 0),
        };
        let _ = (n_heads, d_kv, d_q); // silence unused warnings; used inside _inner
        crate::accel::with_forward_scratch(
            d,
            d_q,
            d_kv,
            d_ff,
            head_dim,
            cfg.rope_theta,
            moe_n_experts,
            moe_top_k,
            |scratch| {
                self.forward_one_paged_f32_inner(
                    token_id,
                    pos,
                    cache,
                    store as &mut dyn crate::paged_kv_store::PagedKvStoreOps,
                    logits_out,
                    scratch,
                    kv_len,
                )
            },
        );
    }

    /// Inner body of [`Self::forward_one_paged_f32`] — the scratch-
    /// borrowing wrapper. Same per-layer dataflow as the public
    /// entry; sourcing buffers from `ForwardScratch` instead of
    /// fresh `vec![0f32; ...]` per call.
    #[allow(clippy::too_many_arguments)]
    fn forward_one_paged_f32_inner(
        &self,
        token_id: i32,
        pos: u32,
        cache: &mut crate::paged_kv_cache::PagedKvCache,
        store: &mut dyn crate::paged_kv_store::PagedKvStoreOps,
        logits_out: &mut [f32],
        scratch: &mut crate::accel::ForwardScratch,
        kv_len: usize,
    ) {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = cfg.d_ff;
        let cur_pos = pos as usize;
        let _ = cur_pos;

        let mut hidden = std::mem::take(&mut scratch.hidden);
        let mut h_norm = std::mem::take(&mut scratch.h_norm);
        let mut q_buf = std::mem::take(&mut scratch.q_buf);
        let mut k_buf = std::mem::take(&mut scratch.k_buf);
        let mut v_buf = std::mem::take(&mut scratch.v_buf);
        let mut attn_out = std::mem::take(&mut scratch.attn_out);
        let mut attn_proj = std::mem::take(&mut scratch.attn_proj);
        let mut gate_buf = std::mem::take(&mut scratch.gate_buf);
        let mut up_buf = std::mem::take(&mut scratch.up_buf);
        let mut ffn_buf = std::mem::take(&mut scratch.ffn_buf);
        let mut ffn_out = std::mem::take(&mut scratch.ffn_out);
        let mut final_norm = std::mem::take(&mut scratch.final_norm);
        let mut moe_expert_logits = std::mem::take(&mut scratch.moe_expert_logits);
        let mut moe_routed_picks = std::mem::take(&mut scratch.moe_routed_picks);

        // H2: per-call gather scratch sized to kv_len, pulled from
        // the pooled `ForwardScratch::paged_{k,v}_slab`. Grows once
        // to the worst-case kv_len in this request's lifetime;
        // subsequent same-or-smaller calls just zero the prefix.
        // Saves ~100-200 MB of allocator pressure per 1000 generated
        // tokens vs the prior per-call `vec![0f32; slab_len]` path.
        let slab_len = n_kv_heads * kv_len * head_dim;
        scratch.ensure_paged_kv_slab(n_kv_heads, kv_len, head_dim);
        let mut paged_k_slab = std::mem::take(&mut scratch.paged_k_slab);
        let mut paged_v_slab = std::mem::take(&mut scratch.paged_v_slab);

        k::embed_lookup_tensor(&self.weights.token_embd, &[token_id], &mut hidden, d);

        // MoE per-layer scratch is taken from ForwardScratch above;
        // `with_forward_scratch` sized it via `n_experts` / `top_k`.
        // For dense models the buffers stay zero-length so the MoE
        // branch never runs (same invariant as forward_one).
        let moe_cfg = self.cfg.moe.clone();

        // Same alloc-free `&dyn AttnBlock` iteration as
        // forward_one_with_scratch_inner. See that site's comment
        // for the rationale.
        let moe_blocks = self.weights.moe_blocks.as_ref();
        let n_blocks_total = if let Some(mbs) = moe_blocks {
            mbs.len()
        } else {
            self.weights.blocks.len()
        };

        let n_layers_used = crate::accel::max_layers_cap()
            .unwrap_or(n_blocks_total)
            .min(n_blocks_total);

        for layer_idx in 0..n_layers_used {
            let block: &dyn AttnBlock = if let Some(mbs) = moe_blocks {
                &mbs[layer_idx]
            } else {
                &self.weights.blocks[layer_idx]
            };
            // Hybrid placement: tell the `try_*_usm_f32` dispatch
            // ladder which layer we're on. Helpers consult
            // `accel::gpu_active_for_current_layer()` (= layer_idx
            // < accel::n_gpu_layers()) to short-circuit the GPU
            // path for layers past the `[inference].n_gpu_layers`
            // cutoff.
            crate::accel::set_current_layer_idx(layer_idx as u32);
            // Pre-attention RMSNorm (CPU; matches prefill mirror).
            k::rmsnorm_f32_row(&hidden, block.attn_norm(), &mut h_norm, cfg.rms_eps);
            // Q / K / V projections via the standard matmul ladder.
            matvec_tensor_dispatch(block.w_q(), &h_norm, &mut q_buf, d_q, d);
            matvec_tensor_dispatch(block.w_k(), &h_norm, &mut k_buf, d_kv, d);
            matvec_tensor_dispatch(block.w_v(), &h_norm, &mut v_buf, d_kv, d);
            if let Some(bq) = block.b_q() {
                k::add_inplace_f32(&mut q_buf, bq);
            }
            if let Some(bk) = block.b_k() {
                k::add_inplace_f32(&mut k_buf, bk);
            }
            if let Some(bv) = block.b_v() {
                k::add_inplace_f32(&mut v_buf, bv);
            }
            // Qwen3 per-head Q/K RMSNorm, before RoPE. No-op on every
            // non-Qwen3 arch (q_norm/k_norm are None).
            apply_qk_head_norm(
                block.q_norm(),
                block.k_norm(),
                &mut q_buf,
                &mut k_buf,
                n_heads,
                n_kv_heads,
                head_dim,
                cfg.rms_eps,
            );
            self.rope_inplace(&mut q_buf, n_heads, head_dim, pos);
            self.rope_inplace(&mut k_buf, n_kv_heads, head_dim, pos);

            // Paged write: append this token's K/V row, then gather
            // [0, kv_len) for attention.
            // H9a: dyn-trait dispatch — handles both F32 and Q8_0
            // paged storage variants without code duplication.
            cache
                .write_token_dyn(store, layer_idx as u32, pos, &k_buf, &v_buf)
                .expect("paged write_token: pos < capacity (ensured above)");
            cache
                .gather_layer_dyn(
                    store,
                    layer_idx as u32,
                    &mut paged_k_slab[..slab_len],
                    &mut paged_v_slab[..slab_len],
                )
                .expect("paged gather_layer: cache covers kv_len positions");
            // Attention — `max_ctx = kv_len` because the gather
            // sized the slab exactly there. Mirror the contiguous
            // F32 path's flash-vs-standard threshold so paged and
            // contiguous produce bit-identical output for the same
            // kv_len (different accumulation order between the two
            // kernels would otherwise diverge by a few rounding
            // bits).
            // D1b: per-layer SWA window (0 ⇒ full) + Gemma2 attn-logit soft-cap
            // (0.0 ⇒ none), threaded into the capped+windowed decode kernels so
            // an SWA / softcap model is correct on the single-token paged decode
            // path. At 0 / 0.0 these delegate to the plain decode dispatchers —
            // byte-identical to the pre-D1b path for every ordinary model.
            let swa_win = cfg.swa_window(layer_idx);
            let softcap = cfg.attn_logit_softcapping.unwrap_or(0.0);
            let flash_min = flash_kv_len_min();
            if crate::accel::flash_attention_enabled() && kv_len >= flash_min {
                k::gqa_attention_flash_decode_capped_windowed(
                    &q_buf,
                    &paged_k_slab[..slab_len],
                    &paged_v_slab[..slab_len],
                    &mut attn_out,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_len,
                    kv_len,
                    swa_win,
                    softcap,
                );
            } else {
                k::gqa_attention_one_step_capped_windowed(
                    &q_buf,
                    &paged_k_slab[..slab_len],
                    &paged_v_slab[..slab_len],
                    &mut attn_out,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_len,
                    kv_len,
                    swa_win,
                    softcap,
                );
            }

            // Output projection + residual.
            matvec_tensor_dispatch(block.w_o(), &attn_out, &mut attn_proj, d, d_q);
            if !skip_attn_enabled() {
                k::add_inplace_f32(&mut hidden, &attn_proj);
            }
            // FFN — dense ladder or per-token MoE.
            k::rmsnorm_f32_row(&hidden, block.ffn_norm(), &mut h_norm, cfg.rms_eps);
            if let Some(moe) = moe_cfg.as_ref() {
                let mb = &self.weights.moe_blocks.as_ref().unwrap()[layer_idx];
                crate::moe::moe_ffn_one_into(
                    &h_norm,
                    mb,
                    d,
                    d_ff,
                    moe.n_experts as usize,
                    moe.n_experts_used as usize,
                    &mut ffn_out,
                    &mut gate_buf,
                    &mut up_buf,
                    &mut ffn_buf,
                    &mut attn_proj, // reuse as down-scratch (size d, unused below)
                    &mut moe_expert_logits,
                    &mut moe_routed_picks,
                );
            } else {
                let dense_block = &self.weights.blocks[layer_idx];
                matvec_tensor_dispatch(&dense_block.w_gate, &h_norm, &mut gate_buf, d_ff, d);
                matvec_tensor_dispatch(&dense_block.w_up, &h_norm, &mut up_buf, d_ff, d);
                k::silu_mul_f32(&gate_buf, &up_buf, &mut ffn_buf);
                matvec_tensor_dispatch(&dense_block.w_down, &ffn_buf, &mut ffn_out, d, d_ff);
            }
            if !skip_ffn_enabled() {
                k::add_inplace_f32(&mut hidden, &ffn_out);
            }
        }

        // Final norm + LM head.
        k::rmsnorm_f32_row(
            &hidden,
            &self.weights.output_norm,
            &mut final_norm,
            cfg.rms_eps,
        );
        let lm_head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        matvec_tensor_dispatch(lm_head, &final_norm, logits_out, cfg.vocab_size, d);
        self.finalize_logits(logits_out);

        // Return the scratch buffers to the TLS cell so the next
        // paged decode on this thread reuses the same heap storage
        // — same put-back convention as `forward_one_with_scratch_inner`.
        scratch.hidden = hidden;
        scratch.h_norm = h_norm;
        scratch.q_buf = q_buf;
        scratch.k_buf = k_buf;
        scratch.v_buf = v_buf;
        scratch.attn_out = attn_out;
        scratch.attn_proj = attn_proj;
        scratch.gate_buf = gate_buf;
        scratch.up_buf = up_buf;
        scratch.ffn_buf = ffn_buf;
        scratch.ffn_out = ffn_out;
        scratch.final_norm = final_norm;
        scratch.moe_expert_logits = moe_expert_logits;
        scratch.moe_routed_picks = moe_routed_picks;
        // H2: put back the pooled paged-KV gather slabs.
        scratch.paged_k_slab = paged_k_slab;
        scratch.paged_v_slab = paged_v_slab;
    }

    /// Run a forward over a sequence of tokens, writing K/V cache for each
    /// and returning only the logits for the *final* token (typical for
    /// prefill — the intermediate logits are not needed).
    pub fn forward_prefill(&self, tokens: &[i32], start_pos: u32, kv: &mut KvCache) -> Vec<f32> {
        // Guard the dense prefill against a mis-dispatched hybrid model
        // (mirrors `forward_one_from_embed`): hybrid models have their
        // own `forward_prefill_hybrid_impl`, and routing one through the
        // dense path here would run ZERO layers and emit gibberish
        // instead of failing loudly.
        assert!(
            !self.weights.is_hybrid(),
            "hybrid attention+SSM model (arch `{}`) reached the dense \
             forward path — hybrids must route through the hybrid forward \
             (`forward_one_hybrid` / `forward_prefill_hybrid_impl`). \
             This is a dispatch bug.",
            self.cfg.arch
        );
        // Opt-in batched path. Uses the new multi-query flash-prefill
        // kernel for the attention compute; the F32 KV path is
        // supported today, quantized KV paths fall back to the serial
        // loop. Gate is per-call via env so users can A/B against the
        // serial baseline without recompiling.
        //
        // MoE models share the batched attention path via the
        // `AttnBlock` trait across all KV dtypes; the FFN section
        // per-token-loops through `moe::moe_ffn_one_into`. Phase 2-D-4
        // wired this for F32 KV; phase 2-D-5 extended it to the
        // Q8_0 / TurboQuant / NVFP4 variants so any KV dtype goes
        // through the batched path for MoE GGUFs.
        let batched_on = prefill_batched_enabled();
        // F4: the batched prefill entry points assume a UNIFORM KV dtype
        // (one `BatchedKv` impl for every layer). A mixed per-layer-group
        // cache (`uniform_dtype() == None`) skips them and uses the serial
        // `forward_one` loop below, which dispatches each layer on its own
        // `KvLayer` variant and is correct for any mix.
        // P2: a cache with any SWA ring layer uses the serial `forward_one`
        // loop below. The batched prefill kernels stride every layer's slab
        // by `max_ctx` and compute all N queries against the FINAL cache
        // state — both wrong for a `cap`-strided ring that overwrites older
        // positions as it fills. The serial loop drives each token through
        // the ring-aware decode arm, which is correct (and O(n·window), not
        // O(n²), on the windowed layers).
        if batched_on && !tokens.is_empty() && kv.uniform_dtype().is_some()
            && !kv.has_ring_layers()
            // M1: MLA has no batched-prefill kernel — its placeholder F32
            // `layers` would route to the dense batched path and run ZERO
            // real layers. Fall through to the serial `forward_one` loop,
            // which dispatches each token through `forward_one_mla`.
            && !self.weights.is_mla()
        {
            match kv.layers.first() {
                Some(KvLayer::F32 { .. }) => {
                    return self.forward_prefill_batched_f32(tokens, start_pos, kv);
                }
                Some(KvLayer::Q8_0 { .. }) => {
                    return self.forward_prefill_batched_q8_0(tokens, start_pos, kv);
                }
                Some(KvLayer::TurboQuant { .. }) => {
                    return self.forward_prefill_batched_tq(tokens, start_pos, kv);
                }
                Some(KvLayer::Nvfp4 { .. }) => {
                    return self.forward_prefill_batched_nvfp4(tokens, start_pos, kv);
                }
                Some(KvLayer::Q4_0 { .. }) => {
                    return self.forward_prefill_batched_q4_0(tokens, start_pos, kv);
                }
                Some(KvLayer::Half { .. }) => {
                    // Pattern binds nothing (mirrors the other arms) so no
                    // borrow of `kv` persists into the `&mut kv` call;
                    // `uniform_dtype()` returns an owned value.
                    let bf16 = kv.uniform_dtype() == Some(KvDtype::Bf16);
                    return self.forward_prefill_batched_half(bf16, tokens, start_pos, kv);
                }
                _ => {}
            }
        }
        let mut logits = vec![0.0f32; self.cfg.vocab_size];
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = start_pos + i as u32;
            self.forward_one(tok, pos, kv, &mut logits);
        }
        logits
    }

    /// Multi-query batched prefill — F32 KV only. Calls
    /// [`rustllama_kernels_cpu::gqa_attention_flash_prefill`] once
    /// per layer for the attention compute, batching all `tokens`
    /// queries against the existing K/V cache. Non-attention ops
    /// (rmsnorm, matvec, rope, FFN) run per-token in a tight loop;
    /// the per-token cost of those is small compared to attention
    /// at long prefill, so the win comes from collapsing attention
    /// from N calls to one.
    ///
    /// Numerical guarantee: bit-for-bit identical to N serial
    /// `forward_one` calls up to floating-point reduction order —
    /// each layer's K/V is appended to the cache before the
    /// attention call, just like the serial path. Parity tests
    /// gate this via `RUSTLLAMA_PREFILL_BATCHED=1` in the engine
    /// integration suite, or by calling this method directly (which
    /// is what the in-tree parity test does to avoid env-var
    /// manipulation racing with parallel tests).
    pub fn forward_prefill_batched_f32(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
    ) -> Vec<f32> {
        let n_new = tokens.len();
        if n_new == 0 {
            return vec![0.0f32; self.cfg.vocab_size];
        }
        let mut logits = vec![0.0f32; self.cfg.vocab_size];
        self.forward_prefill_batched_inner(
            &F32Kv,
            tokens,
            start_pos,
            kv,
            LmHeadMode::Last(&mut logits),
            None,
        );
        logits
    }

    /// Batched F32-KV multi-position forward returning per-position
    /// logits — the E1 speculation primitive. Shares the entire layer
    /// body with [`Self::forward_prefill_batched_f32`] via the
    /// `LmHeadMode::All` dispatch at the final LM-head step. The
    /// caller pre-allocates `logits_out` as a row-major
    /// `[tokens.len() × vocab_size]` buffer.
    ///
    /// Numerical guarantee: bit-identical per-position logits to the
    /// serial-loop [`Self::forward_speculation`] for the same inputs.
    /// Pinned by `forward_speculation_batched_f32_matches_serial`.
    pub fn forward_speculation_batched_f32(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        let n_new = tokens.len();
        assert_eq!(
            logits_out.len(),
            n_new * cfg.vocab_size,
            "forward_speculation_batched_f32: logits_out must be sized to tokens.len() * vocab_size"
        );
        if n_new == 0 {
            return;
        }
        self.forward_prefill_batched_inner(
            &F32Kv,
            tokens,
            start_pos,
            kv,
            LmHeadMode::All(logits_out),
            None,
        );
    }

    /// Tree-structured speculative verification forward (F32 KV). Scores a draft
    /// TREE in ONE batched forward: node `i`'s token is at cache slot
    /// `start_pos + i`, RoPE'd at its DEPTH (`depths[i]`), and attends only the
    /// committed prefix `[0, start_pos)` plus its own ancestor chain (via
    /// [`rustllama_kernels_cpu::gqa_attention_tree_prefill`]). Writes per-node
    /// logits into `logits_out` (`[nodes × vocab]`, row `i` = node `i`), which
    /// [`crate::speculative`-style tree accept/reject] then walks.
    ///
    /// `parent[i]` is the tree-local parent index (negative = depth-1 / root
    /// level), topologically ordered (`parent[i] < i`). A CHAIN-shaped tree
    /// (`depths[i] == i`, `parent[i] == i-1`) reduces to
    /// [`Self::forward_speculation_batched_f32`] (within attention-reduction
    /// tolerance) — pinned by `forward_tree_verify_f32_chain_matches_speculation`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tree_verify_f32(
        &self,
        tokens: &[i32],
        start_pos: u32,
        depths: &[u32],
        parent: &[i32],
        kv: &mut KvCache,
        logits_out: &mut [f32],
    ) {
        let n_new = tokens.len();
        assert_eq!(depths.len(), n_new, "tree verify: one depth per node");
        assert_eq!(parent.len(), n_new, "tree verify: one parent per node");
        assert_eq!(
            logits_out.len(),
            n_new * self.cfg.vocab_size,
            "forward_tree_verify_f32: logits_out must be nodes * vocab_size"
        );
        if n_new == 0 {
            return;
        }
        self.forward_prefill_batched_inner(
            &F32Kv,
            tokens,
            start_pos,
            kv,
            LmHeadMode::All(logits_out),
            Some(TreeAttn { depths, parent }),
        );
    }

    /// HYBRID tree-structured speculative verification forward (F32 KV) — the
    /// recurrent-model counterpart of [`Self::forward_tree_verify_f32`].
    ///
    /// A hybrid model interleaves full-attention layers with recurrent
    /// (DeltaNet / SSM) layers whose state is SEQUENTIAL, so the single batched
    /// ancestor-masked forward the dense tree path uses is *wrong* for the
    /// recurrent layers (see the [`tree_recurrent_walk`] module comment). This
    /// method instead walks the tree node-by-node in parent-before-child,
    /// depth-first order, reusing the fully-validated single-token hybrid
    /// forward ([`Self::forward_one_hybrid`]) as the per-node step:
    ///
    ///   * **Recurrent layers** get the caterpillar treatment — the walk
    ///     snapshots `dn_cache` at the branch root and, for each node, restores
    ///     its parent's post-step snapshot before stepping, so every branch
    ///     evolves its own recurrent state from the shared ancestor.
    ///   * **Attention layers** are correct *for free*: node `j` is stepped at
    ///     `pos = start_pos + depths[j]`, writing its K/V to slot `start_pos +
    ///     depths[j]` and attending the ordinary causal window `[0, pos]`.
    ///     Because each node's unique ancestor chain occupies one slot per depth
    ///     and the depth-first order never clobbers a live ancestor slot, that
    ///     window is exactly "prefix + ancestors + self".
    ///
    /// `parent[j]` is the tree-local parent index (negative = root level),
    /// topologically ordered (`parent[j] < j`). Writes per-node logits into
    /// `logits_out` (`[nodes × vocab]`, row `j` = node `j`).
    ///
    /// CONTRACT: on return, `kv.seq_len` and `dn_cache` are in an UNSPECIFIED
    /// (last-DFS-leaf) state — the walk overwrites shared depth slots and the
    /// live recurrent state as it explores. A caller that commits an accepted
    /// path MUST re-establish it (the engine snapshots `dn_cache` beforehand and
    /// replays the accepted tokens — a recurrence can't be compacted in place
    /// the way dense KV can via [`KvCache::compact_tokens`]).
    ///
    /// A CHAIN-shaped tree (`depths[j] == j`, `parent[j] == j-1`) reduces this to
    /// `n` sequential [`Self::forward_one_hybrid`] calls — i.e. the linear hybrid
    /// decode — bit-for-bit, because every snapshot/restore round-trips the state
    /// we just produced. That reduction is the write-blind correctness anchor
    /// (`toy_recurrent_tree_walk_chain_matches_serial` pins the shared walk;
    /// `forward_tree_verify_hybrid_chain_matches_serial_on_real_model` validates
    /// the whole forward on real weights when a GGUF is supplied).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tree_verify_hybrid_f32(
        &self,
        tokens: &[i32],
        start_pos: u32,
        depths: &[u32],
        parent: &[i32],
        kv: &mut KvCache,
        dn_cache: &mut DeltaNetCache,
        logits_out: &mut [f32],
    ) {
        let n = tokens.len();
        assert_eq!(depths.len(), n, "hybrid tree verify: one depth per node");
        assert_eq!(parent.len(), n, "hybrid tree verify: one parent per node");
        assert_eq!(
            logits_out.len(),
            n * self.cfg.vocab_size,
            "forward_tree_verify_hybrid_f32: logits_out must be nodes * vocab_size"
        );
        assert!(
            self.weights.is_hybrid(),
            "forward_tree_verify_hybrid_f32 called on non-hybrid model — use forward_tree_verify_f32"
        );
        if n == 0 {
            return;
        }
        let vocab = self.cfg.vocab_size;
        let mut driver = HybridTreeVerifyDriver {
            model: self,
            kv,
            dn: dn_cache,
            tokens,
            start_pos,
            depths,
            logits_out,
            vocab,
        };
        tree_recurrent_walk(&mut driver, parent);
    }

    /// CacheBlend selective-recompute prefill (F32 KV, dense + MoE, full-rope).
    ///
    /// `tokens` is the full assembled context — the concatenated chunk tokens
    /// followed by the query suffix. `placements` give each chunk's isolated KV
    /// snapshot and its absolute offset; they must be contiguous and cover
    /// `[0, suffix_start)`, where `suffix_start` is the first suffix position
    /// (`= Σ chunk lens`). The suffix `[suffix_start, tokens.len())` has no
    /// cached KV and is always recomputed.
    ///
    /// Schedule (see [`crate::cacheblend`]): inject each chunk's cached KV at its
    /// offset with K re-rotated to that position; recompute every token through
    /// layers `0..=selection_layer`; measure per-token K deviation at
    /// `selection_layer` (fresh vs isolated-cached); keep the top
    /// `ceil(recompute_ratio · n_chunk)` chunk tokens (the HKVD set) plus the
    /// whole suffix; for layers above `selection_layer`, recompute only that
    /// subset and reuse the injected cached KV for the rest. At
    /// `recompute_ratio >= 1.0` (or no chunks) the subset never shrinks, so the
    /// pass is a full prefill — the correctness anchor the parity test pins.
    ///
    /// Returns the next-token logits for the final position (always in the
    /// recomputed set, since the suffix is). Leaves `kv` holding the blended KV
    /// (fresh for recomputed positions, isolated-cached for the rest) so decode
    /// can continue from `seq_len = tokens.len()`.
    ///
    /// SWA / softcap: the per-layer sliding-window span and Gemma2 attention-logit
    /// soft-cap are threaded into both attention branches (D2b), so a windowed /
    /// capped model re-assembles correctly. F32 KV only (asserted via the
    /// `KvLayer::F32` match arms); the engine gates CacheBlend to F32 KV.
    pub fn forward_prefill_cacheblend(
        &self,
        tokens: &[i32],
        placements: &[crate::cacheblend::ChunkPlacement],
        suffix_start: usize,
        blend: crate::cacheblend::CacheBlendConfig,
        kv: &mut KvCache,
    ) -> Vec<f32> {
        assert!(
            !self.weights.is_hybrid(),
            "CacheBlend prefill not supported for hybrid arch `{}`",
            self.cfg.arch
        );
        let cfg = &self.cfg;
        let n = tokens.len();
        let vocab = cfg.vocab_size;
        if n == 0 {
            return vec![0.0f32; vocab];
        }
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = cfg.d_ff;
        let max_ctx = kv.max_ctx;
        assert!(
            n <= max_ctx,
            "CacheBlend context {n} exceeds max_ctx {max_ctx}"
        );
        let suffix_start = suffix_start.min(n);

        // --- 1. Inject each chunk's isolated KV at its offset (K re-roped). ---
        for pl in placements {
            self.inject_chunk_reroped(kv, pl);
        }
        kv.seq_len = kv.seq_len.max(n);

        // The attention/FFN block list (dense or MoE), uniform via AttnBlock.
        let attn_views: Vec<&dyn AttnBlock> = if let Some(mbs) = self.weights.moe_blocks.as_ref() {
            mbs.iter().map(|b| b as &dyn AttnBlock).collect()
        } else {
            self.weights
                .blocks
                .iter()
                .map(|b| b as &dyn AttnBlock)
                .collect()
        };
        let n_layers = attn_views.len();
        let sel_layer = blend.selection_layer.min(n_layers.saturating_sub(1));
        // `exact` ⇒ never shrink the recompute set ⇒ the pass is a full prefill.
        let exact = blend.recompute_ratio >= 1.0 || suffix_start == 0;

        // Capture the isolated-cached K at the selection layer (post-inject,
        // pre-forward) so the HKVD deviation can compare against it after the
        // forward overwrites that layer's slab. Only chunk positions carry
        // cached KV; suffix positions stay zero (always recomputed anyway).
        let mut cached_sel_k = vec![0f32; n * d_kv];
        if !exact {
            if let KvLayer::F32 { k, .. } = &kv.layers[sel_layer] {
                for p in 0..suffix_start {
                    for h in 0..n_kv_heads {
                        let s = (h * max_ctx + p) * head_dim;
                        let dpos = p * d_kv + h * head_dim;
                        cached_sel_k[dpos..dpos + head_dim].copy_from_slice(&k[s..s + head_dim]);
                    }
                }
            }
        }

        // MoE scratch (single-token slots; untouched for dense models).
        let moe_cfg = cfg.moe.clone();
        let (mut moe_gate_one, mut moe_up_one, mut moe_ff_one, mut moe_down_one) =
            if moe_cfg.is_some() {
                (
                    vec![0f32; d_ff],
                    vec![0f32; d_ff],
                    vec![0f32; d_ff],
                    vec![0f32; d],
                )
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };
        let (mut moe_expert_logits, mut moe_routed_picks) = if let Some(m) = moe_cfg.as_ref() {
            (
                vec![0f32; m.n_experts as usize],
                Vec::with_capacity(m.n_experts_used as usize),
            )
        } else {
            (Vec::new(), Vec::new())
        };

        // --- 2. Recompute set. Starts as every position; shrinks after the
        //        selection layer to HKVD ∪ suffix (unless `exact`). ---
        let mut r_idx: Vec<usize> = (0..n).collect();
        // True while the recompute set is still every position (the warm-up
        // layers 0..=selection_layer, and the whole `exact` path): the queries
        // then form a contiguous 0..n ramp, so one batched GPU/CPU flash-prefill
        // call handles attention. Flipped false once the set shrinks to the
        // scattered HKVD subset.
        let mut r_is_full = true;
        // Hidden state for the recompute set, compacted row-major `[|R|, d]`.
        let mut h_r = vec![0f32; n * d];
        for (i, &tok) in tokens.iter().enumerate() {
            k::embed_lookup_tensor(
                &self.weights.token_embd,
                &[tok],
                &mut h_r[i * d..(i + 1) * d],
                d,
            );
        }

        for layer_idx in 0..n_layers {
            crate::accel::set_current_layer_idx(layer_idx as u32);
            let block = attn_views[layer_idx];
            let nr = r_idx.len();

            // Stage 1: pre-attn rmsnorm (per compact row).
            let mut h_norm = vec![0f32; nr * d];
            for j in 0..nr {
                k::rmsnorm_f32_row(
                    &h_r[j * d..(j + 1) * d],
                    block.attn_norm(),
                    &mut h_norm[j * d..(j + 1) * d],
                    cfg.rms_eps,
                );
            }
            // Stage 2: batched Q/K/V projections over the compact set.
            let mut q_r = vec![0f32; nr * d_q];
            let mut k_rr = vec![0f32; nr * d_kv];
            let mut v_r = vec![0f32; nr * d_kv];
            matvec_tensor_batched_dispatch(block.w_q(), &h_norm, &mut q_r, d_q, d, nr);
            matvec_tensor_batched_dispatch(block.w_k(), &h_norm, &mut k_rr, d_kv, d, nr);
            matvec_tensor_batched_dispatch(block.w_v(), &h_norm, &mut v_r, d_kv, d, nr);

            // Stage 3: bias + qk-head-norm + RoPE at each row's REAL position,
            // then write the fresh K/V into the live slab at that position.
            {
                let (kslab, vslab) = match &mut kv.layers[layer_idx] {
                    KvLayer::F32 { k, v } => (k, v),
                    _ => unreachable!("CacheBlend requires F32 KV"),
                };
                for j in 0..nr {
                    let pos = r_idx[j] as u32;
                    let q_row = &mut q_r[j * d_q..(j + 1) * d_q];
                    let k_row = &mut k_rr[j * d_kv..(j + 1) * d_kv];
                    let v_row = &mut v_r[j * d_kv..(j + 1) * d_kv];
                    if let Some(bq) = block.b_q() {
                        k::add_inplace_f32(q_row, bq);
                    }
                    if let Some(bk) = block.b_k() {
                        k::add_inplace_f32(k_row, bk);
                    }
                    if let Some(bv) = block.b_v() {
                        k::add_inplace_f32(v_row, bv);
                    }
                    apply_qk_head_norm(
                        block.q_norm(),
                        block.k_norm(),
                        q_row,
                        k_row,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        cfg.rms_eps,
                    );
                    self.rope_inplace(q_row, n_heads, head_dim, pos);
                    self.rope_inplace(k_row, n_kv_heads, head_dim, pos);
                    let posu = r_idx[j];
                    for h in 0..n_kv_heads {
                        let dst = (h * max_ctx + posu) * head_dim;
                        kslab[dst..dst + head_dim]
                            .copy_from_slice(&k_row[h * head_dim..(h + 1) * head_dim]);
                        vslab[dst..dst + head_dim]
                            .copy_from_slice(&v_row[h * head_dim..(h + 1) * head_dim]);
                    }
                }
            }

            // Stage 4: attention over the FULL assembled KV (cached for the
            // non-recomputed positions, fresh for the rest). A recomputed query
            // at real position `p` attends causally over `0..=p`.
            //
            // While `r_is_full` (warm-up layers 0..=selection_layer + the exact
            // path), the queries are the contiguous ramp 0..nr, so ONE batched
            // flash-prefill call serves them — routed through the shared
            // CUDA/Metal/SYCL GPU hook (single slab copy), CPU fallback. These
            // are the expensive O(n²) layers, so this is where GPU attention
            // pays off. Once the set is the scattered HKVD subset, each query
            // attends over `0..=its_pos` via a per-row CPU step: the blended KV
            // is host-resident so a per-row GPU copy would re-upload the whole
            // slab each row (a loss), and no scatter-position GPU kernel exists.
            // D2b: per-layer SWA window (0 ⇒ full attention) + Gemma2 attn-logit
            // soft-cap (0.0 ⇒ none), threaded into BOTH CacheBlend attention
            // branches below so a sliding-window / Gemma2-softcap model is
            // re-assembled with the correct mask + logit cap. At 0 / 0.0 these
            // delegate to the plain `gqa_attention_flash_prefill` /
            // `gqa_attention_one_step`, byte-identical to the pre-D2b path for
            // every ordinary model (CacheBlend is F32-only; the engine still gates
            // it off for non-F32 KV, so window/softcap always ride the F32 path).
            let swa_win = cfg.swa_window(layer_idx);
            let softcap = cfg.attn_logit_softcapping.unwrap_or(0.0);
            let mut attn_out = vec![0f32; nr * d_q];
            {
                let (kslab, vslab) = match &kv.layers[layer_idx] {
                    KvLayer::F32 { k, v } => (&k[..], &v[..]),
                    _ => unreachable!("CacheBlend requires F32 KV"),
                };
                if r_is_full {
                    // Contiguous queries 0..nr (row j is at position j),
                    // kv_len_base = 0. Shared GPU prefill hook (carries
                    // window/softcap), CPU capped+windowed fallback.
                    let used_gpu = crate::accel::try_flash_attn_prefill_gpu_f32(
                        &q_r,
                        kslab,
                        vslab,
                        &mut attn_out,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        0,
                        nr,
                        swa_win as u32,
                        softcap,
                    );
                    if !used_gpu {
                        k::gqa_attention_flash_prefill_capped_windowed(
                            &q_r,
                            kslab,
                            vslab,
                            &mut attn_out,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_ctx,
                            0,
                            nr,
                            swa_win,
                            softcap,
                        );
                    }
                } else {
                    // Scattered HKVD subset: query j at real position `r_idx[j]`
                    // attends `[0, r_idx[j]]` (kv_len = r_idx[j] + 1). The
                    // capped+windowed decode kernel trims to `[kv_len-window,
                    // kv_len)` = `[pos-window+1, pos+1)`, matching the prefill
                    // window semantics.
                    for j in 0..nr {
                        k::gqa_attention_one_step_capped_windowed(
                            &q_r[j * d_q..(j + 1) * d_q],
                            kslab,
                            vslab,
                            &mut attn_out[j * d_q..(j + 1) * d_q],
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_ctx,
                            r_idx[j] + 1,
                            swa_win,
                            softcap,
                        );
                    }
                }
            }

            // Stage 5: output projection + residual + post-attn rmsnorm.
            let mut attn_proj = vec![0f32; nr * d];
            matvec_tensor_batched_dispatch(block.w_o(), &attn_out, &mut attn_proj, d, d_q, nr);
            for j in 0..nr {
                let proj_row = &attn_proj[j * d..(j + 1) * d];
                let h_row = &mut h_r[j * d..(j + 1) * d];
                if !skip_attn_enabled() {
                    k::add_inplace_f32(h_row, proj_row);
                }
                k::rmsnorm_f32_row(
                    h_row,
                    block.ffn_norm(),
                    &mut h_norm[j * d..(j + 1) * d],
                    cfg.rms_eps,
                );
            }

            // Stage 6-8: FFN (dense batched / MoE per-row).
            let mut ffn_out = vec![0f32; nr * d];
            if let Some(moe) = moe_cfg.as_ref() {
                let mb = &self.weights.moe_blocks.as_ref().unwrap()[layer_idx];
                for j in 0..nr {
                    crate::moe::moe_ffn_one_into(
                        &h_norm[j * d..(j + 1) * d],
                        mb,
                        d,
                        d_ff,
                        moe.n_experts as usize,
                        moe.n_experts_used as usize,
                        &mut ffn_out[j * d..(j + 1) * d],
                        &mut moe_gate_one,
                        &mut moe_up_one,
                        &mut moe_ff_one,
                        &mut moe_down_one,
                        &mut moe_expert_logits,
                        &mut moe_routed_picks,
                    );
                }
            } else {
                let dense_block = &self.weights.blocks[layer_idx];
                let mut gate_buf = vec![0f32; nr * d_ff];
                let mut up_buf = vec![0f32; nr * d_ff];
                let mut ffn_buf = vec![0f32; nr * d_ff];
                matvec_tensor_batched_dispatch(
                    &dense_block.w_gate,
                    &h_norm,
                    &mut gate_buf,
                    d_ff,
                    d,
                    nr,
                );
                matvec_tensor_batched_dispatch(
                    &dense_block.w_up,
                    &h_norm,
                    &mut up_buf,
                    d_ff,
                    d,
                    nr,
                );
                for j in 0..nr {
                    k::silu_mul_f32(
                        &gate_buf[j * d_ff..(j + 1) * d_ff],
                        &up_buf[j * d_ff..(j + 1) * d_ff],
                        &mut ffn_buf[j * d_ff..(j + 1) * d_ff],
                    );
                }
                matvec_tensor_batched_dispatch(
                    &dense_block.w_down,
                    &ffn_buf,
                    &mut ffn_out,
                    d,
                    d_ff,
                    nr,
                );
            }
            // Stage 9: FFN residual.
            if !skip_ffn_enabled() {
                for j in 0..nr {
                    let h_row = &mut h_r[j * d..(j + 1) * d];
                    k::add_inplace_f32(h_row, &ffn_out[j * d..(j + 1) * d]);
                }
            }

            // --- 3. After the selection layer, shrink to HKVD ∪ suffix. ---
            if !exact && layer_idx == sel_layer {
                // At the selection layer `r_idx == 0..n`, so compact row j == pos
                // j and `k_rr` holds fresh (post-RoPE) K for every position.
                let n_rec = (((suffix_start as f32) * blend.recompute_ratio).ceil() as usize)
                    .min(suffix_start);
                let mut dev: Vec<(f32, usize)> = (0..suffix_start)
                    .map(|p| {
                        let a = &k_rr[p * d_kv..(p + 1) * d_kv];
                        let b = &cached_sel_k[p * d_kv..(p + 1) * d_kv];
                        let s: f32 = a
                            .iter()
                            .zip(b)
                            .map(|(x, y)| {
                                let e = x - y;
                                e * e
                            })
                            .sum();
                        (s, p)
                    })
                    .collect();
                // Highest deviation first.
                dev.sort_unstable_by(|x, y| {
                    y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut keep: Vec<usize> = dev[..n_rec].iter().map(|&(_, p)| p).collect();
                keep.extend(suffix_start..n);
                keep.sort_unstable();
                keep.dedup();
                // Gather the surviving rows' hidden state (old row == pos here).
                let mut new_h = vec![0f32; keep.len() * d];
                for (nj, &p) in keep.iter().enumerate() {
                    new_h[nj * d..(nj + 1) * d].copy_from_slice(&h_r[p * d..(p + 1) * d]);
                }
                h_r = new_h;
                r_idx = keep;
                r_is_full = false;
            }
        }

        // --- 4. LM head on the final position (always the last recomputed row,
        //        because the suffix is always recomputed). ---
        let last = r_idx.len() - 1;
        debug_assert_eq!(
            r_idx[last],
            n - 1,
            "CacheBlend: final position must be recomputed"
        );
        let mut final_norm = vec![0f32; d];
        k::rmsnorm_f32_row(
            &h_r[last * d..(last + 1) * d],
            &self.weights.output_norm,
            &mut final_norm,
            cfg.rms_eps,
        );
        let lm_head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        let mut logits = vec![0.0f32; vocab];
        matvec_tensor_batched_dispatch(lm_head, &final_norm, &mut logits, vocab, d, 1);
        self.finalize_logits(&mut logits);
        logits
    }

    /// Inject one chunk's isolated KV snapshot into the live cache at
    /// `placement.offset`, re-rotating K from the chunk-local positions it was
    /// computed at (`0..chunk_len`) to the assembled positions (`off..off+len`).
    ///
    /// The snapshot's K is already RoPE-rotated by its within-chunk index `j`
    /// (a standalone prefill advances the position per token). RoPE is additive
    /// in angle, so one extra rotation by the constant `off` turns `rope-at-j`
    /// into `rope-at-(j+off)` — exactly what a fresh prefill at the assembled
    /// offset would produce (per-head Q/K RMSNorm runs before RoPE and is
    /// position-independent, so it is unaffected). V carries no RoPE and is
    /// copied verbatim.
    fn inject_chunk_reroped(
        &self,
        kv: &mut KvCache,
        placement: &crate::cacheblend::ChunkPlacement,
    ) {
        let cfg = &self.cfg;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_kv = n_kv_heads * head_dim;
        let max_ctx = kv.max_ctx;
        let snap = &placement.snapshot;
        let pl = snap.prefix_len;
        let off = placement.offset;
        assert_eq!(snap.n_kv_heads, n_kv_heads);
        assert_eq!(snap.head_dim, head_dim);
        assert_eq!(snap.layers.len(), kv.layers.len());
        assert!(off + pl <= max_ctx, "CacheBlend chunk overflows max_ctx");
        // Delta rotation to re-base the chunk from local → assembled positions.
        let delta = off as u32;
        let mut k_scratch = vec![0f32; d_kv];
        for (live, src) in kv.layers.iter_mut().zip(snap.layers.iter()) {
            let (kslab, vslab, sk, sv) = match (live, src) {
                (KvLayer::F32 { k, v }, KvLayer::F32 { k: sk, v: sv }) => (k, v, sk, sv),
                _ => panic!("CacheBlend requires F32 KV snapshots"),
            };
            for j in 0..pl {
                // Gather this position's K across heads into contiguous [d_kv].
                for h in 0..n_kv_heads {
                    let s = (h * pl + j) * head_dim;
                    k_scratch[h * head_dim..(h + 1) * head_dim]
                        .copy_from_slice(&sk[s..s + head_dim]);
                }
                // Re-base by the constant offset (identity when off == 0).
                if delta != 0 {
                    self.rope_rotate_inplace(&mut k_scratch, n_kv_heads, head_dim, delta);
                }
                // Scatter K (re-roped) + V (verbatim) into the live slab.
                let posu = off + j;
                for h in 0..n_kv_heads {
                    let dst = (h * max_ctx + posu) * head_dim;
                    kslab[dst..dst + head_dim]
                        .copy_from_slice(&k_scratch[h * head_dim..(h + 1) * head_dim]);
                    let ss = (h * pl + j) * head_dim;
                    vslab[dst..dst + head_dim].copy_from_slice(&sv[ss..ss + head_dim]);
                }
            }
        }
        crate::accel::usm_attn_kv_epoch_bump();
    }

    /// T2: the shared batched-prefill body for EVERY KV dtype. Lifted
    /// verbatim from the former `forward_prefill_batched_f32_inner`, it
    /// delegates ONLY the per-layer KV write + attention to `kv_impl` (a
    /// `BatchedKv`); the six public entry points each construct the matching
    /// marker and call this.
    ///
    /// `tree` (the speculative tree-verify descriptor — `depths[i]` is node
    /// `i`'s depth, its RoPE position is `start_pos + depths[i]`; `parent[i]`
    /// drives the ancestor mask) is honoured only by the F32 impl; quant
    /// dtypes always pass `None` ⇒ the ordinary sequential causal prefill.
    /// `window`/`softcap` (D1b) are computed per layer and threaded into
    /// `write_and_attend`.
    #[allow(clippy::too_many_arguments)]
    fn forward_prefill_batched_inner<K: BatchedKv>(
        &self,
        kv_impl: &K,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        mode: LmHeadMode<'_>,
        tree: Option<TreeAttn<'_>>,
    ) {
        let cfg = &self.cfg;
        let n_new = tokens.len();
        if n_new == 0 {
            return;
        }
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = cfg.d_ff;
        let max_ctx = kv.max_ctx;
        let kv_len_base = start_pos as usize;
        assert!(
            kv_len_base + n_new <= max_ctx,
            "prefill batch overflows max_ctx: {kv_len_base} + {n_new} > {max_ctx}"
        );

        // Per-token embed → hidden state `[n_new, d]`. The body
        // below treats this as a contiguous row-major buffer; row
        // `i` lives at `hidden[i*d..(i+1)*d]`.
        let mut hidden = vec![0f32; n_new * d];
        for (i, &tok) in tokens.iter().enumerate() {
            k::embed_lookup_tensor(
                &self.weights.token_embd,
                &[tok],
                &mut hidden[i * d..(i + 1) * d],
                d,
            );
        }

        // Scratch sized for the batched layer body.
        let mut h_norm = vec![0f32; n_new * d];
        let mut q_buf = vec![0f32; n_new * d_q];
        let mut k_buf = vec![0f32; n_new * d_kv];
        let mut v_buf = vec![0f32; n_new * d_kv];
        let mut attn_out = vec![0f32; n_new * d_q];
        let mut attn_proj = vec![0f32; n_new * d];
        let mut gate_buf = vec![0f32; n_new * d_ff];
        let mut up_buf = vec![0f32; n_new * d_ff];
        let mut ffn_buf = vec![0f32; n_new * d_ff];
        let mut ffn_out = vec![0f32; n_new * d];

        // MoE FFN runs per-token through `moe::moe_ffn_one_into`. Its
        // signature consumes `&mut [f32]` slices the size of ONE token's
        // worth of activations, so we size single-token scratch slots
        // here. Dense models leave these untouched.
        let moe_cfg = cfg.moe.clone();
        let (mut moe_gate_one, mut moe_up_one, mut moe_ff_one, mut moe_down_one) =
            if moe_cfg.is_some() {
                (
                    vec![0f32; d_ff],
                    vec![0f32; d_ff],
                    vec![0f32; d_ff],
                    vec![0f32; d],
                )
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };
        let (mut moe_expert_logits, mut moe_routed_picks) = if let Some(m) = moe_cfg.as_ref() {
            (
                vec![0f32; m.n_experts as usize],
                Vec::with_capacity(m.n_experts_used as usize),
            )
        } else {
            (Vec::new(), Vec::new())
        };

        // Collect `&dyn AttnBlock` per layer so the attention body is
        // a single uniform loop regardless of dense vs MoE — same
        // pattern as `forward_one_with_scratch`.
        let attn_views: Vec<&dyn AttnBlock> = if let Some(mbs) = self.weights.moe_blocks.as_ref() {
            mbs.iter().map(|b| b as &dyn AttnBlock).collect()
        } else {
            self.weights
                .blocks
                .iter()
                .map(|b| b as &dyn AttnBlock)
                .collect()
        };

        let n_layers_used = crate::accel::max_layers_cap()
            .unwrap_or(attn_views.len())
            .min(attn_views.len());

        for (layer_idx, block) in attn_views.iter().take(n_layers_used).enumerate() {
            // Hybrid placement: tell the `try_*_usm_f32` dispatch
            // ladder which layer we're on. Helpers consult
            // `accel::gpu_active_for_current_layer()` (= layer_idx
            // < accel::n_gpu_layers()) to short-circuit the GPU
            // path for layers past the `[inference].n_gpu_layers`
            // cutoff.
            crate::accel::set_current_layer_idx(layer_idx as u32);
            // ---- attention ----
            // Stage 1 (per-token): pre-attention rmsnorm. Cheap and
            // hard to batch across tokens cleanly, so it stays serial.
            for i in 0..n_new {
                let h_row = &hidden[i * d..(i + 1) * d];
                let n_row = &mut h_norm[i * d..(i + 1) * d];
                k::rmsnorm_f32_row(h_row, block.attn_norm(), n_row, cfg.rms_eps);
            }
            // Stage 2 (batched): Q / K / V projections in one kernel
            // launch each, instead of N per projection.
            matvec_tensor_batched_dispatch(block.w_q(), &h_norm, &mut q_buf, d_q, d, n_new);
            matvec_tensor_batched_dispatch(block.w_k(), &h_norm, &mut k_buf, d_kv, d, n_new);
            matvec_tensor_batched_dispatch(block.w_v(), &h_norm, &mut v_buf, d_kv, d, n_new);
            // Stage 3 (per-token): bias add + RoPE — both depend on
            // per-token position, so they stay serial. q_buf/k_buf/v_buf
            // are separate Vecs, so three mutable disjoint borrows
            // (one per token slice) coexist without conflict.
            for i in 0..n_new {
                let q_row = &mut q_buf[i * d_q..(i + 1) * d_q];
                let k_row = &mut k_buf[i * d_kv..(i + 1) * d_kv];
                let v_row = &mut v_buf[i * d_kv..(i + 1) * d_kv];
                if let Some(bq) = block.b_q() {
                    k::add_inplace_f32(q_row, bq);
                }
                if let Some(bk) = block.b_k() {
                    k::add_inplace_f32(k_row, bk);
                }
                if let Some(bv) = block.b_v() {
                    k::add_inplace_f32(v_row, bv);
                }
                // Qwen3 per-head Q/K RMSNorm, before RoPE. No-op (two
                // Option checks) on every non-Qwen3 arch.
                apply_qk_head_norm(
                    block.q_norm(),
                    block.k_norm(),
                    q_row,
                    k_row,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    cfg.rms_eps,
                );
                // RoPE position: sequential for ordinary prefill; the tree
                // node's DEPTH for tree verify (siblings share a position), so a
                // chain tree (`depths[i] == i`) matches the sequential path.
                let pos_i = match &tree {
                    Some(t) => kv_len_base as u32 + t.depths[i],
                    None => (kv_len_base + i) as u32,
                };
                // Same neox/half-split convention as forward_one; batched
                // prefill stays on the CPU path (scaled table applied for
                // rope-scaled models, byte-identical theta otherwise).
                self.rope_inplace(q_row, n_heads, head_dim, pos_i);
                self.rope_inplace(k_row, n_kv_heads, head_dim, pos_i);
            }

            // KV write + attention — delegated to the dtype's BatchedKv impl
            // (quantize-on-write + dequantize-on-attend). The D1b per-layer SWA
            // window (0 ⇒ full causal) + Gemma2 attn-logit soft-cap (0.0 ⇒ no
            // cap) and the optional tree descriptor are threaded in; quant
            // dtypes ignore all three (they only ever get 0 / 0.0 / None).
            let window = cfg.swa_window(layer_idx);
            let softcap = cfg.attn_logit_softcapping.unwrap_or(0.0);
            kv_impl.write_and_attend(
                kv,
                layer_idx,
                &mut q_buf,
                &mut k_buf,
                &mut v_buf,
                &mut attn_out,
                n_heads,
                n_kv_heads,
                head_dim,
                max_ctx,
                kv_len_base,
                n_new,
                window,
                softcap,
                tree.as_ref(),
            );

            // Stage 4 (batched): output projection in one kernel.
            matvec_tensor_batched_dispatch(block.w_o(), &attn_out, &mut attn_proj, d, d_q, n_new);
            // Stage 5 (per-token): residual add + post-attn rmsnorm.
            for i in 0..n_new {
                let proj_row = &attn_proj[i * d..(i + 1) * d];
                let h_row = &mut hidden[i * d..(i + 1) * d];
                if !skip_attn_enabled() {
                    k::add_inplace_f32(h_row, proj_row);
                }
                let n_row = &mut h_norm[i * d..(i + 1) * d];
                k::rmsnorm_f32_row(h_row, block.ffn_norm(), n_row, cfg.rms_eps);
            }
            // Stage 6-8: FFN. Dense path matvec-batches gate/up/down
            // across all `n_new` tokens; MoE path loops `moe_ffn_one_into`
            // per-token because each token routes to a different
            // subset of experts.
            if let Some(moe) = moe_cfg.as_ref() {
                let mb = &self.weights.moe_blocks.as_ref().unwrap()[layer_idx];
                for i in 0..n_new {
                    let n_row = &h_norm[i * d..(i + 1) * d];
                    let out_row = &mut ffn_out[i * d..(i + 1) * d];
                    crate::moe::moe_ffn_one_into(
                        n_row,
                        mb,
                        d,
                        d_ff,
                        moe.n_experts as usize,
                        moe.n_experts_used as usize,
                        out_row,
                        &mut moe_gate_one,
                        &mut moe_up_one,
                        &mut moe_ff_one,
                        &mut moe_down_one,
                        &mut moe_expert_logits,
                        &mut moe_routed_picks,
                    );
                }
            } else {
                let dense_block = &self.weights.blocks[layer_idx];
                // Stage 6 (batched): FFN gate + up projections.
                matvec_tensor_batched_dispatch(
                    &dense_block.w_gate,
                    &h_norm,
                    &mut gate_buf,
                    d_ff,
                    d,
                    n_new,
                );
                matvec_tensor_batched_dispatch(
                    &dense_block.w_up,
                    &h_norm,
                    &mut up_buf,
                    d_ff,
                    d,
                    n_new,
                );
                // Stage 7 (per-token): silu_mul element-wise op.
                for i in 0..n_new {
                    let g_row = &gate_buf[i * d_ff..(i + 1) * d_ff];
                    let u_row = &up_buf[i * d_ff..(i + 1) * d_ff];
                    let f_row = &mut ffn_buf[i * d_ff..(i + 1) * d_ff];
                    k::silu_mul_f32(g_row, u_row, f_row);
                }
                // Stage 8 (batched): FFN down projection.
                matvec_tensor_batched_dispatch(
                    &dense_block.w_down,
                    &ffn_buf,
                    &mut ffn_out,
                    d,
                    d_ff,
                    n_new,
                );
            }
            // Stage 9 (per-token): residual add for FFN output.
            if !skip_ffn_enabled() {
                for i in 0..n_new {
                    let h_row = &mut hidden[i * d..(i + 1) * d];
                    let o_row = &ffn_out[i * d..(i + 1) * d];
                    k::add_inplace_f32(h_row, o_row);
                }
            }

            // Update kv.seq_len to reflect that this layer's K/V is
            // populated through position kv_len_base + n_new - 1.
            // The serial path bumps `kv.seq_len` per token inside
            // forward_one (at the very end); we do it once after
            // the batched layer so the invariants line up.
        }
        kv.seq_len = kv.seq_len.max(kv_len_base + n_new);

        // Final norm + LM head — dispatch on `mode`. `Last` writes
        // logits for only the final position (matches the prefill
        // contract: caller wants the distribution for the next-to-
        // sample token). `All` writes one logit row per position
        // (the speculation contract: caller verifies each draft).
        let lm_head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        match mode {
            LmHeadMode::Last(out) => {
                let last_off = (n_new - 1) * d;
                let last_hidden = &hidden[last_off..last_off + d];
                let mut final_norm = vec![0f32; d];
                k::rmsnorm_f32_row(
                    last_hidden,
                    &self.weights.output_norm,
                    &mut final_norm,
                    cfg.rms_eps,
                );
                matvec_tensor_dispatch(lm_head, &final_norm, out, cfg.vocab_size, d);
                self.finalize_logits(out);
            }
            LmHeadMode::All(out) => {
                // Norm all `n_new` rows of hidden into a contiguous
                // `[n_new, d]` scratch, then run a single batched LM
                // head over all positions. matvec_tensor_batched_dispatch
                // handles the dtype-aware GPU/CPU dispatch.
                let mut final_norm_all = vec![0f32; n_new * d];
                for i in 0..n_new {
                    let in_row = &hidden[i * d..(i + 1) * d];
                    let out_row = &mut final_norm_all[i * d..(i + 1) * d];
                    k::rmsnorm_f32_row(in_row, &self.weights.output_norm, out_row, cfg.rms_eps);
                }
                matvec_tensor_batched_dispatch(
                    lm_head,
                    &final_norm_all,
                    out,
                    cfg.vocab_size,
                    d,
                    n_new,
                );
            }
        }
    }

    /// Multi-query batched prefill — TurboQuant KV variant.
    /// Paged-KV variant of [`Self::forward_prefill_batched_f32`].
    /// Identical layer body — same RMSNorm, Q/K/V projections, RoPE,
    /// attention, output projection, FFN — but the KV cache is fed
    /// through the [`crate::paged_kv_cache::PagedKvCache`] indirection
    /// instead of one contiguous `[n_kv_heads * max_ctx * head_dim]`
    /// slab per layer.
    ///
    /// The data-path delta is two operations per layer:
    ///   1. **Write**: for each new token `i`, write that token's
    ///      head-major K/V row through
    ///      [`crate::paged_kv_cache::PagedKvCache::write_token`] —
    ///      the cache decomposes the absolute position into
    ///      `(page_index, pos_in_page)` and dispatches to the right
    ///      page in the shared store.
    ///   2. **Gather**: before calling the attention kernel, gather
    ///      all `kv_len_base + n_new` positions into a contiguous
    ///      `[n_kv_heads, kv_len, head_dim]` scratch slab that
    ///      matches the kernel's expected layout. The kernel itself
    ///      is unchanged — we just feed it a freshly-built slab
    ///      every call instead of indexing into a fixed `max_ctx`-
    ///      sized one.
    ///
    /// `kv_len` (`= kv_len_base + n_new`) is passed as the kernel's
    /// `max_ctx` argument too: the gathered slab is sized exactly
    /// to `kv_len`, so the kernel's per-head stride equals `kv_len *
    /// head_dim` and `t in 0..kv_len` walks every cell exactly once.
    ///
    /// The caller is responsible for calling
    /// [`crate::paged_kv_cache::PagedKvCache::ensure_capacity`] with
    /// `kv_len_base + n_new` before invoking — this function asserts
    /// (in debug) that the cache has enough pages but won't grow it.
    /// The split lets the scheduler's admission path handle OOM
    /// (back-off / queue) rather than burying it inside the forward
    /// pass.
    ///
    /// Per-layer `kv_len_base + n_new` MUST fit `cfg.ctx_train` or
    /// whatever the engine's effective context cap is — the existing
    /// `kv_len_base + n_new <= max_ctx` assert is replaced here by
    /// the cache's `capacity_tokens()` check (an over-grown cache
    /// is fine; we only walk `kv_len` positions regardless).
    pub fn forward_prefill_paged_f32(
        &self,
        tokens: &[i32],
        start_pos: u32,
        cache: &mut crate::paged_kv_cache::PagedKvCache,
        store: &mut crate::paged_kv_store::PagedKvStore,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let n_new = tokens.len();
        if n_new == 0 {
            return vec![0.0f32; cfg.vocab_size];
        }
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = cfg.d_ff;
        let kv_len_base = start_pos as usize;
        let kv_len = kv_len_base + n_new;
        debug_assert!(
            (kv_len as u32) <= cache.capacity_tokens(),
            "paged prefill: cache capacity {} < kv_len {kv_len}; \
             caller must ensure_capacity first",
            cache.capacity_tokens(),
        );

        let mut hidden = vec![0f32; n_new * d];
        for (i, &tok) in tokens.iter().enumerate() {
            k::embed_lookup_tensor(
                &self.weights.token_embd,
                &[tok],
                &mut hidden[i * d..(i + 1) * d],
                d,
            );
        }

        let mut h_norm = vec![0f32; n_new * d];
        let mut q_buf = vec![0f32; n_new * d_q];
        let mut k_buf = vec![0f32; n_new * d_kv];
        let mut v_buf = vec![0f32; n_new * d_kv];
        let mut attn_out = vec![0f32; n_new * d_q];
        let mut attn_proj = vec![0f32; n_new * d];
        let mut gate_buf = vec![0f32; n_new * d_ff];
        let mut up_buf = vec![0f32; n_new * d_ff];
        let mut ffn_buf = vec![0f32; n_new * d_ff];
        let mut ffn_out = vec![0f32; n_new * d];

        // Reusable gather scratch — sized once to the max kv_len for
        // this call. Same layout as the contiguous path's K/V slabs
        // would have been, but only for the positions we'll actually
        // attend to (no max_ctx slack).
        let slab_len = n_kv_heads * kv_len * head_dim;
        let mut k_slab = vec![0f32; slab_len];
        let mut v_slab = vec![0f32; slab_len];

        let n_layers_used = crate::accel::max_layers_cap()
            .unwrap_or(self.weights.blocks.len())
            .min(self.weights.blocks.len());

        for (layer_idx, block) in self.weights.blocks.iter().take(n_layers_used).enumerate() {
            // Hybrid placement: tell the `try_*_usm_f32` dispatch
            // ladder which layer we're on. Helpers consult
            // `accel::gpu_active_for_current_layer()` (= layer_idx
            // < accel::n_gpu_layers()) to short-circuit the GPU
            // path for layers past the `[inference].n_gpu_layers`
            // cutoff.
            crate::accel::set_current_layer_idx(layer_idx as u32);
            // Stage 1 (per-token): pre-attention rmsnorm.
            for i in 0..n_new {
                let h_row = &hidden[i * d..(i + 1) * d];
                let n_row = &mut h_norm[i * d..(i + 1) * d];
                k::rmsnorm_f32_row(h_row, &block.attn_norm, n_row, cfg.rms_eps);
            }
            // Stage 2 (batched): Q / K / V projections.
            matvec_tensor_batched_dispatch(&block.w_q, &h_norm, &mut q_buf, d_q, d, n_new);
            matvec_tensor_batched_dispatch(&block.w_k, &h_norm, &mut k_buf, d_kv, d, n_new);
            matvec_tensor_batched_dispatch(&block.w_v, &h_norm, &mut v_buf, d_kv, d, n_new);
            // Stage 3 (per-token): bias add + RoPE.
            for i in 0..n_new {
                let q_row = &mut q_buf[i * d_q..(i + 1) * d_q];
                let k_row = &mut k_buf[i * d_kv..(i + 1) * d_kv];
                let v_row = &mut v_buf[i * d_kv..(i + 1) * d_kv];
                if let Some(bq) = &block.b_q {
                    k::add_inplace_f32(q_row, bq);
                }
                if let Some(bk) = &block.b_k {
                    k::add_inplace_f32(k_row, bk);
                }
                if let Some(bv) = &block.b_v {
                    k::add_inplace_f32(v_row, bv);
                }
                // Qwen3 per-head Q/K RMSNorm, before RoPE. No-op on
                // every non-Qwen3 arch (q_norm/k_norm are None).
                apply_qk_head_norm(
                    block.q_norm.as_deref(),
                    block.k_norm.as_deref(),
                    q_row,
                    k_row,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    cfg.rms_eps,
                );
                let pos_i = (kv_len_base + i) as u32;
                self.rope_inplace(q_row, n_heads, head_dim, pos_i);
                self.rope_inplace(k_row, n_kv_heads, head_dim, pos_i);
            }

            // Paged write: append each new token's K/V row to the
            // cache through the page indirection. Cache bumps
            // `seq_len` to `kv_len` after the last write.
            for i in 0..n_new {
                let pos_i = (kv_len_base + i) as u32;
                let k_row = &k_buf[i * d_kv..(i + 1) * d_kv];
                let v_row = &v_buf[i * d_kv..(i + 1) * d_kv];
                cache
                    .write_token(store, layer_idx as u32, pos_i, k_row, v_row)
                    .expect("paged write_token: pos < capacity (ensured above)");
            }
            // Gather into the per-call scratch slab. Layout matches
            // what the existing kernel expects: `[h, pos, d]` flat.
            cache
                .gather_layer(store, layer_idx as u32, &mut k_slab, &mut v_slab)
                .expect("paged gather_layer: cache covers kv_len positions");

            // Attention — try the SYCL F32 prefill path first
            // (passes max_ctx = kv_len because the slab is sized
            // exactly there); fall back to CPU if SYCL is off /
            // hook returns false. This is the PAGED prefill path.
            // D1b: per-layer SWA window (0 ⇒ full causal) + Gemma2 attn-logit
            // soft-cap (0.0 ⇒ none) are threaded into BOTH the F32 GPU prefill
            // hook (which carries them, bit-exact to the CPU reference) and the
            // CPU `gqa_attention_flash_prefill_capped_windowed` fallback — so a
            // dense SWA (Mistral-SWA) / Gemma2-softcap model prefilled through
            // this path (e.g. `generate_batched_token_ids`) masks and caps
            // correctly. At 0 / 0.0 both delegate to the plain prefill, so an
            // ordinary model is byte-identical to the pre-D1b path.
            let swa_win = cfg.swa_window(layer_idx);
            let softcap = cfg.attn_logit_softcapping.unwrap_or(0.0);
            let used_gpu = crate::accel::try_flash_attn_prefill_gpu_f32(
                &q_buf,
                &k_slab,
                &v_slab,
                &mut attn_out,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_len,
                kv_len_base,
                n_new,
                swa_win as u32,
                softcap,
            );
            if !used_gpu {
                k::gqa_attention_flash_prefill_capped_windowed(
                    &q_buf,
                    &k_slab,
                    &v_slab,
                    &mut attn_out,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_len,
                    kv_len_base,
                    n_new,
                    swa_win,
                    softcap,
                );
            }

            // Stage 4 (batched): output projection.
            matvec_tensor_batched_dispatch(&block.w_o, &attn_out, &mut attn_proj, d, d_q, n_new);
            // Stage 5 (per-token): residual add + post-attn rmsnorm.
            for i in 0..n_new {
                let proj_row = &attn_proj[i * d..(i + 1) * d];
                let h_row = &mut hidden[i * d..(i + 1) * d];
                if !skip_attn_enabled() {
                    k::add_inplace_f32(h_row, proj_row);
                }
                let n_row = &mut h_norm[i * d..(i + 1) * d];
                k::rmsnorm_f32_row(h_row, &block.ffn_norm, n_row, cfg.rms_eps);
            }
            // Stage 6 (batched): FFN gate + up.
            matvec_tensor_batched_dispatch(&block.w_gate, &h_norm, &mut gate_buf, d_ff, d, n_new);
            matvec_tensor_batched_dispatch(&block.w_up, &h_norm, &mut up_buf, d_ff, d, n_new);
            // Stage 7 (per-token): silu_mul.
            for i in 0..n_new {
                let g_row = &gate_buf[i * d_ff..(i + 1) * d_ff];
                let u_row = &up_buf[i * d_ff..(i + 1) * d_ff];
                let f_row = &mut ffn_buf[i * d_ff..(i + 1) * d_ff];
                k::silu_mul_f32(g_row, u_row, f_row);
            }
            // Stage 8 (batched): FFN down.
            matvec_tensor_batched_dispatch(&block.w_down, &ffn_buf, &mut ffn_out, d, d_ff, n_new);
            // Stage 9 (per-token): residual add for FFN output.
            if !skip_ffn_enabled() {
                for i in 0..n_new {
                    let h_row = &mut hidden[i * d..(i + 1) * d];
                    let o_row = &ffn_out[i * d..(i + 1) * d];
                    k::add_inplace_f32(h_row, o_row);
                }
            }
        }

        // Final norm + LM head on the last token only.
        let last_off = (n_new - 1) * d;
        let last_hidden = &hidden[last_off..last_off + d];
        let mut final_norm = vec![0f32; d];
        k::rmsnorm_f32_row(
            last_hidden,
            &self.weights.output_norm,
            &mut final_norm,
            cfg.rms_eps,
        );
        let lm_head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        let mut logits = vec![0f32; cfg.vocab_size];
        matvec_tensor_dispatch(lm_head, &final_norm, &mut logits, cfg.vocab_size, d);
        self.finalize_logits(&mut logits);
        logits
    }

    /// Mirrors [`Self::forward_prefill_batched_f32`] structurally
    /// but writes each new K/V row through `quantize_row` into the
    /// packed `KvLayer::TurboQuant` slabs, then calls
    /// [`rustllama_kernels_cpu::turboquant::gqa_attention_flash_prefill_tq`]
    /// for the batched attention compute. Greedy parity vs the
    /// serial forward_one loop is bit-for-bit (up to FP reduction
    /// order); the parity test in `tests/forward.rs` covers F32 KV
    /// and the kernels-cpu suite covers the TQ kernel itself.
    pub fn forward_prefill_batched_tq(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
    ) -> Vec<f32> {
        let n_new = tokens.len();
        if n_new == 0 {
            return vec![0.0f32; self.cfg.vocab_size];
        }
        let mut logits = vec![0.0f32; self.cfg.vocab_size];
        self.forward_prefill_batched_inner(
            &TqKv,
            tokens,
            start_pos,
            kv,
            LmHeadMode::Last(&mut logits),
            None,
        );
        logits
    }

    /// Batched TurboQuant-KV multi-position forward returning
    /// per-position logits — the E1 speculation primitive for TQ KV.
    /// Shares the per-layer body with
    /// [`Self::forward_prefill_batched_tq`] via the `LmHeadMode::All`
    /// dispatch at the final LM-head step.
    pub fn forward_speculation_batched_tq(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        let n_new = tokens.len();
        assert_eq!(
            logits_out.len(),
            n_new * cfg.vocab_size,
            "forward_speculation_batched_tq: logits_out must be sized to tokens.len() * vocab_size"
        );
        if n_new == 0 {
            return;
        }
        self.forward_prefill_batched_inner(
            &TqKv,
            tokens,
            start_pos,
            kv,
            LmHeadMode::All(logits_out),
            None,
        );
    }

    /// Multi-query batched prefill — Q8_0 KV variant. Quantizes each
    /// new K/V row per-token via [`quantize_row_q8_0`] into the
    /// `KvLayer::Q8_0` i8 slabs, then calls
    /// [`rustllama_kernels_cpu::gqa_attention_flash_prefill_q8_0`]
    /// once per layer for the batched attention compute. Same
    /// non-attention structure as [`Self::forward_prefill_batched_f32`].
    pub fn forward_prefill_batched_q8_0(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
    ) -> Vec<f32> {
        let n_new = tokens.len();
        if n_new == 0 {
            return vec![0.0f32; self.cfg.vocab_size];
        }
        let mut logits = vec![0.0f32; self.cfg.vocab_size];
        self.forward_prefill_batched_inner(
            &Q8_0Kv,
            tokens,
            start_pos,
            kv,
            LmHeadMode::Last(&mut logits),
            None,
        );
        logits
    }

    /// Batched Q8_0-KV multi-position forward returning per-position
    /// logits — the E1 speculation primitive for Q8_0 KV. Shares the
    /// per-layer body with [`Self::forward_prefill_batched_q8_0`] via
    /// the `LmHeadMode::All` dispatch at the final LM-head step.
    /// `logits_out` is row-major `[tokens.len() × vocab_size]`.
    pub fn forward_speculation_batched_q8_0(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        let n_new = tokens.len();
        assert_eq!(
            logits_out.len(),
            n_new * cfg.vocab_size,
            "forward_speculation_batched_q8_0: logits_out must be sized to tokens.len() * vocab_size"
        );
        if n_new == 0 {
            return;
        }
        self.forward_prefill_batched_inner(
            &Q8_0Kv,
            tokens,
            start_pos,
            kv,
            LmHeadMode::All(logits_out),
            None,
        );
    }

    /// Multi-query batched prefill — NVFP4 KV variant. Quantizes
    /// each new K/V row per-block via
    /// [`rustllama_kernels_cpu::nvfp4::quantize_block`] into the
    /// `KvLayer::Nvfp4` packed slabs, then calls
    /// [`rustllama_kernels_cpu::nvfp4::gqa_attention_flash_prefill_nvfp4`]
    /// for the batched attention. Closes out the KV-dtype matrix
    /// — all four dtypes (F32, Q8_0, TQ, NVFP4) now batch-prefill.
    pub fn forward_prefill_batched_nvfp4(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
    ) -> Vec<f32> {
        let n_new = tokens.len();
        if n_new == 0 {
            return vec![0.0f32; self.cfg.vocab_size];
        }
        let mut logits = vec![0.0f32; self.cfg.vocab_size];
        self.forward_prefill_batched_inner(
            &Nvfp4Kv,
            tokens,
            start_pos,
            kv,
            LmHeadMode::Last(&mut logits),
            None,
        );
        logits
    }

    /// Batched NVFP4-KV multi-position forward returning per-position
    /// logits — the E1 speculation primitive for NVFP4 KV. Shares the
    /// per-layer body with [`Self::forward_prefill_batched_nvfp4`] via
    /// the `LmHeadMode::All` dispatch at the final LM-head step.
    pub fn forward_speculation_batched_nvfp4(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        let n_new = tokens.len();
        assert_eq!(
            logits_out.len(),
            n_new * cfg.vocab_size,
            "forward_speculation_batched_nvfp4: logits_out must be sized to tokens.len() * vocab_size"
        );
        if n_new == 0 {
            return;
        }
        self.forward_prefill_batched_inner(
            &Nvfp4Kv,
            tokens,
            start_pos,
            kv,
            LmHeadMode::All(logits_out),
            None,
        );
    }

    /// Batched MXFP4-KV multi-position forward returning per-position
    /// logits — the E1 speculation primitive for MXFP4 KV. Mirrors
    /// [`Self::forward_speculation_batched_nvfp4`] modulo the 32-elem MX
    /// block geometry; shares the per-layer body with the MXFP6/MXFP8
    /// spec entries via the generic `forward_prefill_batched_inner` with a
    /// [`MxfpKv`] marker (its `fmt` selects the block bytes + kernels).
    pub fn forward_speculation_batched_mxfp4(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        let n_new = tokens.len();
        assert_eq!(
            logits_out.len(),
            n_new * cfg.vocab_size,
            "forward_speculation_batched_mxfp4: logits_out must be sized to tokens.len() * vocab_size"
        );
        if n_new == 0 {
            return;
        }
        self.forward_prefill_batched_inner(
            &MxfpKv {
                fmt: KvDtype::Mxfp4,
            },
            tokens,
            start_pos,
            kv,
            LmHeadMode::All(logits_out),
            None,
        );
    }

    /// Batched MXFP6-KV spec primitive. See
    /// [`Self::forward_speculation_batched_mxfp4`].
    pub fn forward_speculation_batched_mxfp6(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        let n_new = tokens.len();
        assert_eq!(
            logits_out.len(),
            n_new * cfg.vocab_size,
            "forward_speculation_batched_mxfp6: logits_out must be sized to tokens.len() * vocab_size"
        );
        if n_new == 0 {
            return;
        }
        self.forward_prefill_batched_inner(
            &MxfpKv {
                fmt: KvDtype::Mxfp6,
            },
            tokens,
            start_pos,
            kv,
            LmHeadMode::All(logits_out),
            None,
        );
    }

    /// Batched MXFP8-KV spec primitive. See
    /// [`Self::forward_speculation_batched_mxfp4`].
    pub fn forward_speculation_batched_mxfp8(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        let n_new = tokens.len();
        assert_eq!(
            logits_out.len(),
            n_new * cfg.vocab_size,
            "forward_speculation_batched_mxfp8: logits_out must be sized to tokens.len() * vocab_size"
        );
        if n_new == 0 {
            return;
        }
        self.forward_prefill_batched_inner(
            &MxfpKv {
                fmt: KvDtype::Mxfp8,
            },
            tokens,
            start_pos,
            kv,
            LmHeadMode::All(logits_out),
            None,
        );
    }

    /// Multi-query batched prefill — Q4_0 KV variant. Quantizes each
    /// new K/V row into ggml Q4_0 blocks via
    /// [`rustllama_kernels_cpu::q4_0_kv::quantize_row`], then calls
    /// [`rustllama_kernels_cpu::q4_0_kv::gqa_attention_flash_prefill_q4_0`]
    /// for the batched attention.
    pub fn forward_prefill_batched_q4_0(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
    ) -> Vec<f32> {
        let n_new = tokens.len();
        if n_new == 0 {
            return vec![0.0f32; self.cfg.vocab_size];
        }
        let mut logits = vec![0.0f32; self.cfg.vocab_size];
        self.forward_prefill_batched_inner(
            &Q4_0Kv,
            tokens,
            start_pos,
            kv,
            LmHeadMode::Last(&mut logits),
            None,
        );
        logits
    }

    /// Batched prefill for half-precision KV (M6). `bf16` selects
    /// bfloat16 vs IEEE f16 storage; the write/attend body is in
    /// [`HalfKv`] (store half bits, dequant the live region to f32, run
    /// the F32 flash-prefill). Output equals the F32 batched path over
    /// f16/bf16-rounded K/V.
    pub fn forward_prefill_batched_half(
        &self,
        bf16: bool,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
    ) -> Vec<f32> {
        let n_new = tokens.len();
        if n_new == 0 {
            return vec![0.0f32; self.cfg.vocab_size];
        }
        let mut logits = vec![0.0f32; self.cfg.vocab_size];
        self.forward_prefill_batched_inner(
            &HalfKv { bf16 },
            tokens,
            start_pos,
            kv,
            LmHeadMode::Last(&mut logits),
            None,
        );
        logits
    }

    /// Batched Q4_0-KV multi-position forward returning per-position
    /// logits — the E1 speculation primitive for Q4_0 KV. Shares the
    /// per-layer body with [`Self::forward_prefill_batched_q4_0`] via
    /// the `LmHeadMode::All` dispatch at the final LM-head step.
    pub fn forward_speculation_batched_q4_0(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        logits_out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        let n_new = tokens.len();
        assert_eq!(
            logits_out.len(),
            n_new * cfg.vocab_size,
            "forward_speculation_batched_q4_0: logits_out must be sized to tokens.len() * vocab_size"
        );
        if n_new == 0 {
            return;
        }
        self.forward_prefill_batched_inner(
            &Q4_0Kv,
            tokens,
            start_pos,
            kv,
            LmHeadMode::All(logits_out),
            None,
        );
    }
}

// Adapter to the generic Model trait.
impl crate::Model for LlamaModel {
    fn n_layers(&self) -> usize {
        self.cfg.n_layers
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn device(&self) -> &rustllama_tensor::Device {
        &rustllama_tensor::Device::Cpu
    }
    fn forward(
        &self,
        _batch: &crate::Batch,
        _kv: &mut crate::KvCache,
        _out: &mut rustllama_tensor::Tensor,
    ) -> crate::Result<()> {
        // The trait was the phase-0 placeholder shape; the real call surface
        // is `LlamaModel::forward_one` / `forward_prefill` above. We'll widen
        // the trait in a follow-up once `ContiguousEngine` is wired.
        Ok(())
    }
}

#[cfg(test)]
mod tests;
