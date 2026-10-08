//! Forward pass for the Llama family (Llama / Mistral / Qwen2 / DeepSeek /
//! Phi-3 / Yi). All these models share the same block structure:
//!   pre-attn RMSNorm → Q/K/V proj → RoPE → GQA attention → out proj → residual
//!   pre-ffn  RMSNorm → SwiGLU (gate * up → down) → residual
//! ending in a final RMSNorm + LM head.
//!
//! Phase 1 keeps all weights as F16 and runs activations in F32 on the CPU.
//! The same code path will accept SYCL-allocated weights via the `Engine`
//! trait once phase 3 lands.

use rustllama_gguf::{Gguf, GgmlType};
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

    // H8: F16-input mixed-precision matvec (gated by
    // RUSTLLAMA_MIXED_PRECISION_MATVEC, default off). Tried before the
    // F32 USM path; on miss falls straight through with no cost.
    if crate::accel::try_matvec_tensor_f16in_usm_f32(w, x, out, m, k_dim) {
        return;
    }
    if crate::accel::try_matvec_tensor_usm_f32(w, x, out, m, k_dim) {
        return;
    }
    if crate::accel::try_matvec_f16_usm_f32(w, x, out, m, k_dim) {
        return;
    }
    k::matvec_tensor(w, x, out, m, k_dim);
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
    if crate::accel::try_matvec_tensor_batched_usm_f32(w, x, out, m_dim, k_dim, n_rows) {
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
    if w.dtype == rustllama_tensor::Dtype::PTQ1_0Raw && n_rows > 1 {
        k::matvec_tensor_batched(w, x, out, m_dim, k_dim, n_rows);
        return;
    }
    // Fallback: per-row dispatch via the single-row hook, which
    // tries single-row USM then CPU. Identical to the pre-batched
    // behavior.
    for i in 0..n_rows {
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
    // During imatrix calibration, force the two-matvec base path so
    // each tensor is recorded exactly once via `matvec_tensor_dispatch`
    // (the fused USM kernel bypasses that hook). No double-count: only
    // the base dispatch records.
    if !crate::imatrix_collect::is_collecting()
        && crate::accel::try_matvec_tensor_gate_up_fused_usm_f32(
            w_gate, w_up, x, gate_out, up_out, m, k_dim,
        )
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
        had.map_or(true, |h| h.is_folded(&w_gate.name) == h.is_folded(&w_up.name)),
        "hadamard: ffn_gate/ffn_up foldedness must match (fused dispatch)"
    );
    {
        let x_in = hadamard_pre(had, w_gate, x, had_buf);
        let gate = &mut gate_buf[..d_ff];
        let up = &mut up_buf[..d_ff];
        matvec_tensor_gate_up_fused_dispatch(w_gate, w_up, x_in, gate, up, d_ff, d);
    }
    k::silu_mul_f32(&gate_buf[..d_ff], &up_buf[..d_ff], &mut ff_buf[..d_ff]);
    let ff_in = hadamard_pre(had, w_down, &ff_buf[..d_ff], had_buf);
    matvec_tensor_dispatch(w_down, ff_in, out, d, d_ff);
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
        had.map_or(true, |h| h.is_folded(&w_gate.name) == h.is_folded(&w_up.name)),
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
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("RUSTLLAMA_PREFILL_BATCHED")
            .map(|v| !v.is_empty() && v != "0" && v.to_ascii_lowercase() != "false")
            .unwrap_or(false)
    })
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
        let rms =
            (s.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / s.len() as f64).sqrt();
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
    /// `prism.hadamard.*` metadata present but inconsistent with the
    /// model's tensors, or present on a model shape whose forward
    /// path hasn't wired the activation transform. Hard error: a
    /// Hadamard-folded model run without (or with wrong) rotation
    /// math produces silent garbage, so refusing to load is the only
    /// safe behavior — mirrors the reference implementation.
    #[error("prism.hadamard: {0}")]
    Hadamard(String),
    /// MoE (mixture-of-experts) GGUFs are detected at load time but
    /// the forward pass isn't implemented yet. Surfaced before the
    /// loader tries to bind dense FFN tensors that don't exist on
    /// MoE models — without this, users get a confusing "missing
    /// `ffn_gate.weight`" error instead of a clear "MoE not yet
    /// supported" diagnostic.
    #[error(
        "MoE (mixture-of-experts) architecture `{arch}` detected \
         ({n_experts} experts, top-{n_experts_used} routed{shared_msg}). \
         MoE forward pass is a follow-up — use a dense Llama-family \
         GGUF (Qwen2.5-Coder, DeepSeek-Coder-V2, Mistral, Phi-3) for now."
    )]
    UnsupportedMoe {
        arch: String,
        n_experts: u32,
        n_experts_used: u32,
        /// Pre-formatted ", N shared" suffix (or empty for non-shared
        /// MoE variants). Inlined here so the `thiserror`-derived
        /// `Display` impl stays declarative.
        shared_msg: String,
    },
    #[error(
        "model loaded but forward pass not yet implemented: hybrid \
         attention+SSM architecture `{arch}` (Mamba-style state-space \
         layers + full-attention every {interval}th layer). Phase 2 \
         of the qwen35moe roadmap bound all tensors; Phase 3 implements \
         the SSM forward kernel. See docs/qwen35moe-roadmap.md."
    )]
    UnsupportedHybridForward { arch: String, interval: u32 },
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
    /// H3: optionally-populated fused QKV weight. When all three of
    /// `w_q`, `w_k`, `w_v` are F32 + same `d_model` column and
    /// `RUSTLLAMA_QKV_FUSED=1` is set, `concat_qkv_f32` packs them
    /// into a single `[d_q + 2*d_kv, d_model]` tensor at load time.
    /// The forward path dispatches one matvec into a combined output
    /// buffer instead of three separate dispatches — saves ~2/3 of
    /// the per-attention dispatch overhead on supported models.
    /// Quantized w_q/w_k/w_v aren't yet supported (bit-packed concat
    /// is a future variant); the optional stays `None` and the
    /// forward path falls through to the per-tensor matvecs.
    pub w_qkv_fused: Option<Tensor>,
    pub w_o: Tensor,         // [d_model, n_heads * head_dim] F32
    /// Optional attention projection biases. Qwen2 has them; vanilla Llama
    /// and most other Llama-family architectures don't.
    pub b_q: Option<Vec<f32>>, // [n_heads * head_dim]
    pub b_k: Option<Vec<f32>>, // [n_kv_heads * head_dim]
    pub b_v: Option<Vec<f32>>, // [n_kv_heads * head_dim]
    pub ffn_norm: Vec<f32>,  // [d_model]
    pub w_gate: Tensor,      // [d_ff, d_model] F32
    pub w_up: Tensor,        // [d_ff, d_model] F32
    pub w_down: Tensor,      // [d_model, d_ff] F32
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
    fn attn_norm(&self) -> &[f32] { &self.attn_norm }
    fn w_q(&self) -> &Tensor { &self.w_q }
    fn w_k(&self) -> &Tensor { &self.w_k }
    fn w_v(&self) -> &Tensor { &self.w_v }
    fn w_o(&self) -> &Tensor { &self.w_o }
    fn b_q(&self) -> Option<&Vec<f32>> { self.b_q.as_ref() }
    fn b_k(&self) -> Option<&Vec<f32>> { self.b_k.as_ref() }
    fn b_v(&self) -> Option<&Vec<f32>> { self.b_v.as_ref() }
    fn ffn_norm(&self) -> &[f32] { &self.ffn_norm }
    fn q_norm(&self) -> Option<&[f32]> { self.q_norm.as_deref() }
    fn k_norm(&self) -> Option<&[f32]> { self.k_norm.as_deref() }
}

impl AttnBlock for LlamaMoeBlockWeights {
    fn attn_norm(&self) -> &[f32] { &self.attn_norm }
    fn w_q(&self) -> &Tensor { &self.w_q }
    fn w_k(&self) -> &Tensor { &self.w_k }
    fn w_v(&self) -> &Tensor { &self.w_v }
    fn w_o(&self) -> &Tensor { &self.w_o }
    fn b_q(&self) -> Option<&Vec<f32>> { self.b_q.as_ref() }
    fn b_k(&self) -> Option<&Vec<f32>> { self.b_k.as_ref() }
    fn b_v(&self) -> Option<&Vec<f32>> { self.b_v.as_ref() }
    fn ffn_norm(&self) -> &[f32] { &self.ffn_norm }
    fn q_norm(&self) -> Option<&[f32]> { self.q_norm.as_deref() }
    fn k_norm(&self) -> Option<&[f32]> { self.k_norm.as_deref() }
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
    pub ffn_norm: Vec<f32>,  // [d_model]
    /// Router projection. Shape `[n_experts, d_model]` — produces
    /// one logit per expert which the forward pass softmax+top-K's
    /// to pick the routed experts.
    pub router: Tensor,
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
/// Phase 2 (this file) binds the tensors. Phase 3 wires the
/// forward pass; until then `forward_*` returns
/// `LlamaLoadError::UnsupportedHybridForward`.
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
            .unwrap_or_else(|| {
                panic!("hadamard: no sign vector for activation width {width}")
            })
            .as_slice()
    }
}

impl LlamaWeights {
    /// `true` when this model's transformer blocks are MoE-shaped
    /// (router + expert tensors loaded). `false` for dense models.
    /// Forward-pass entry points consult this to decide between the
    /// dense forward and the MoE forward (the latter currently
    /// errors with `UnsupportedMoe` pending phase-2-B routing
    /// math).
    pub fn is_hybrid(&self) -> bool {
        self.hybrid_layers.is_some()
    }

    pub fn is_moe(&self) -> bool {
        self.moe_blocks.is_some()
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
        let (blocks, moe_blocks, hybrid_layers) = if let Some(hyb_cfg) = cfg.hybrid.as_ref() {
            let layers = load_hybrid_layers(gguf, cfg, cfg.moe.as_ref(), hyb_cfg)?;
            (Vec::new(), None, Some(layers))
        } else if let Some(moe_cfg) = cfg.moe.as_ref() {
            let n_experts = moe_cfg.n_experts as usize;
            let d_model = cfg.d_model;
            let d_ff = cfg.d_ff;
            let mut mbs = Vec::with_capacity(cfg.n_layers);
            for i in 0..cfg.n_layers {
                let prefix = format!("blk.{i}");
                let w_gate_exps = load_weight(
                    gguf,
                    &format!("{prefix}.ffn_gate_exps.weight"),
                )?;
                let w_up_exps =
                    load_weight(gguf, &format!("{prefix}.ffn_up_exps.weight"))?;
                let w_down_exps = load_weight(
                    gguf,
                    &format!("{prefix}.ffn_down_exps.weight"),
                )?;
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
            (Vec::new(), Some(mbs), None)
        } else {
            let mut blocks = Vec::with_capacity(cfg.n_layers);
            for i in 0..cfg.n_layers {
                let prefix = format!("blk.{i}");
                let w_q = load_weight(gguf, &format!("{prefix}.attn_q.weight"))?;
                let w_k = load_weight(gguf, &format!("{prefix}.attn_k.weight"))?;
                let w_v = load_weight(gguf, &format!("{prefix}.attn_v.weight"))?;
                // H3: try to fuse w_q / w_k / w_v at load time. Only
                // takes effect when env-var is set and all three are
                // F32 (quantized concat is a follow-up).
                let w_qkv_fused = if crate::accel::qkv_fused_enabled() {
                    concat_qkv_f32(&w_q, &w_k, &w_v)
                } else {
                    None
                };
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
            (blocks, None, None)
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
                        &b.attn_qkv, &b.attn_gate, &b.ssm_conv1d,
                        &b.ssm_alpha, &b.ssm_beta, &b.ssm_out,
                    ]);
                    &b.ffn
                }
                HybridLayer::FullAttention(b) => {
                    ts.extend([&b.w_q, &b.w_k, &b.w_v, &b.w_o]);
                    &b.ffn
                }
            };
            match ffn {
                HybridFfn::Dense { w_gate, w_up, w_down } => {
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
            HybridFfn::Moe {
                router: load_weight(gguf, &format!("{prefix}.ffn_gate_inp.weight"))?,
                w_gate_exps,
                w_up_exps,
                w_down_exps,
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
                shared_router: load_optional_weight(
                    gguf,
                    &format!("{prefix}.ffn_gate_inp_shexp.weight"),
                ),
                gate_per_expert,
                up_per_expert,
                down_per_expert,
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
        let has_separate_q = gguf
            .tensor(&format!("{prefix}.attn_q.weight"))
            .is_some();
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
    k::rmsnorm_f32_row(&projected, &head.shared_head_norm, &mut shared_norm, rms_eps);

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
static ZEROCOPY_BORROWED_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

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
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("RUSTLLAMA_ZEROCOPY_WEIGHTS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

fn load_norm(gguf: &Gguf, name: &str) -> Result<Vec<f32>, LlamaLoadError> {
    let t = Tensor::from_gguf(gguf, name, Dtype::F32)?;
    Ok(rustllama_tensor::as_slice_f32(&t).to_vec())
}

/// H3: concat F32 `w_q` / `w_k` / `w_v` into one
/// `[d_q + 2*d_kv, d_model]` fused tensor. Returns `None` when any
/// input isn't F32 or the column dims don't match (quant-format
/// concat is a future variant — the bit-packed layouts don't share
/// the same per-row stride math).
///
/// GGUF stores weight matrices as `[out_rows, d_model]` in row-major;
/// concat is a simple per-row stacking on the leading axis.
fn concat_qkv_f32(w_q: &Tensor, w_k: &Tensor, w_v: &Tensor) -> Option<Tensor> {
    if w_q.dtype != Dtype::F32 || w_k.dtype != Dtype::F32 || w_v.dtype != Dtype::F32 {
        return None;
    }
    // Shape is `[rows, cols]`; the `cols` dim must match (= d_model)
    // for the concat to be meaningful.
    if w_q.shape.len() != 2 || w_k.shape.len() != 2 || w_v.shape.len() != 2 {
        return None;
    }
    let cols = w_q.shape[w_q.shape.len() - 1];
    if w_k.shape[w_k.shape.len() - 1] != cols || w_v.shape[w_v.shape.len() - 1] != cols {
        return None;
    }
    let rows_q = w_q.shape[0];
    let rows_k = w_k.shape[0];
    let rows_v = w_v.shape[0];
    let total_rows = rows_q + rows_k + rows_v;
    let q_slice = rustllama_tensor::as_slice_f32(w_q);
    let k_slice = rustllama_tensor::as_slice_f32(w_k);
    let v_slice = rustllama_tensor::as_slice_f32(w_v);
    let mut fused = Vec::with_capacity((total_rows * cols) as usize);
    fused.extend_from_slice(q_slice);
    fused.extend_from_slice(k_slice);
    fused.extend_from_slice(v_slice);
    Some(Tensor::from_vec_f32(
        format!("{}_qkv_fused", w_q.name),
        vec![total_rows, cols],
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
            KvLayer::Q8_0 { k_q, k_scales, v_q, v_scales } => {
                b(k_q) + b(k_scales) + b(v_q) + b(v_scales)
            }
            KvLayer::TurboQuant { k_packed, k_scales, v_packed, v_scales, .. } => {
                b(k_packed) + b(k_scales) + b(v_packed) + b(v_scales)
            }
            KvLayer::Nvfp4 { k_packed, v_packed } => b(k_packed) + b(v_packed),
            KvLayer::Q4_0 { k_q, v_q } => b(k_q) + b(v_q),
            KvLayer::Mxfp4 { k_packed, v_packed }
            | KvLayer::Mxfp6 { k_packed, v_packed }
            | KvLayer::Mxfp8 { k_packed, v_packed } => b(k_packed) + b(v_packed),
        }
    }
}

impl std::fmt::Debug for KvLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvLayer::F32 { k, v } => f.debug_struct("F32").field("k", k).field("v", v).finish(),
            KvLayer::Q8_0 { k_q, k_scales, v_q, v_scales } => f.debug_struct("Q8_0")
                .field("k_q", k_q).field("k_scales", k_scales)
                .field("v_q", v_q).field("v_scales", v_scales).finish(),
            KvLayer::TurboQuant { bits, k_packed, k_scales, v_packed, v_scales } => f.debug_struct("TurboQuant")
                .field("bits", bits).field("k_packed", k_packed).field("k_scales", k_scales)
                .field("v_packed", v_packed).field("v_scales", v_scales).finish(),
            KvLayer::Nvfp4 { k_packed, v_packed } => f.debug_struct("Nvfp4")
                .field("k_packed", k_packed).field("v_packed", v_packed).finish(),
            KvLayer::Q4_0 { k_q, v_q } => f.debug_struct("Q4_0")
                .field("k_q", k_q).field("v_q", v_q).finish(),
            KvLayer::Mxfp4 { k_packed, v_packed } => f.debug_struct("Mxfp4")
                .field("k_packed", k_packed).field("v_packed", v_packed).finish(),
            KvLayer::Mxfp6 { k_packed, v_packed } => f.debug_struct("Mxfp6")
                .field("k_packed", k_packed).field("v_packed", v_packed).finish(),
            KvLayer::Mxfp8 { k_packed, v_packed } => f.debug_struct("Mxfp8")
                .field("k_packed", k_packed).field("v_packed", v_packed).finish(),
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

pub struct KvCache {
    /// **Field declaration order matters**: `layers` drops before
    /// `usm_stream`, so any `KvBuf::Usm` allocations are freed before
    /// the stream they reference goes away. The `'static` lifetime on
    /// the USM buffers is a transmute over `usm_stream`'s lifetime;
    /// this drop order is the safety guarantee.
    pub layers: Vec<KvLayer>,
    pub seq_len: usize,
    pub max_ctx: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub dtype: KvDtype,
    /// K-cache mean-centering bias (fork `kv_bar` parity) — attached
    /// by the engine after load when a validated sidecar exists.
    /// Subtracted from K rows in the Q4_0 write arms only; exactly
    /// softmax-invariant, so no read side exists. `None` = uncentered.
    pub kv_bias: Option<std::sync::Arc<crate::kv_bias::KvBiasData>>,
    /// G2: optional SYCL stream owning the KV-layer USM allocations.
    /// `None` when `RUSTLLAMA_USM_KV` is unset/0 or SYCL is unavailable;
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
            .field("dtype", &self.dtype)
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
fn alloc_kv_buf_f32(
    stream: Option<&rustllama_kernels_sycl::SyclStream>,
    len: usize,
) -> KvBuf<f32> {
    if let Some(s) = stream {
        match rustllama_kernels_sycl::SyclSharedBuffer::<f32>::alloc(s, len) {
            Ok(mut buf) => {
                for x in buf.as_mut_slice() { *x = 0.0; }
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
fn alloc_kv_buf_i8(
    stream: Option<&rustllama_kernels_sycl::SyclStream>,
    len: usize,
) -> KvBuf<i8> {
    if let Some(s) = stream {
        match rustllama_kernels_sycl::SyclSharedBuffer::<i8>::alloc(s, len) {
            Ok(mut buf) => {
                for x in buf.as_mut_slice() { *x = 0; }
                let buf: rustllama_kernels_sycl::SyclSharedBuffer<'static, i8> =
                    unsafe { std::mem::transmute(buf) };
                return KvBuf::Usm(buf);
            }
            Err(_) => {}
        }
    }
    KvBuf::host_zeroed(len)
}
fn alloc_kv_buf_u8(
    stream: Option<&rustllama_kernels_sycl::SyclStream>,
    len: usize,
) -> KvBuf<u8> {
    if let Some(s) = stream {
        match rustllama_kernels_sycl::SyclSharedBuffer::<u8>::alloc(s, len) {
            Ok(mut buf) => {
                for x in buf.as_mut_slice() { *x = 0; }
                let buf: rustllama_kernels_sycl::SyclSharedBuffer<'static, u8> =
                    unsafe { std::mem::transmute(buf) };
                return KvBuf::Usm(buf);
            }
            Err(_) => {}
        }
    }
    KvBuf::host_zeroed(len)
}


/// G2: try to construct a SYCL stream for KV-cache USM allocations.
/// Returns `Some(Box<stream>)` when `RUSTLLAMA_USM_KV=1` and SYCL has
/// a device; `None` otherwise. Cached after first parse so repeated
/// `KvCache::new_with_dtype` calls don't re-parse the env var.
fn try_open_usm_kv_stream() -> Option<Box<rustllama_kernels_sycl::SyclStream>> {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        std::env::var("RUSTLLAMA_USM_KV")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    });
    if !enabled {
        return None;
    }
    match rustllama_kernels_sycl::create_stream(0) {
        Ok(s) => Some(Box::new(s)),
        Err(_) => None,
    }
}

impl KvCache {
    pub fn new(cfg: &LlamaConfig, max_ctx: usize) -> Self {
        Self::new_with_dtype(cfg, max_ctx, KvDtype::F32)
    }

    /// Total bytes of K/V buffer storage across all layers. Exact (it
    /// walks the allocated buffers). With sparse construction
    /// ([`Self::new_with_dtype_sparse`]) hybrid models report only
    /// their full-attention layers' slabs; a densely-constructed
    /// cache reports every slab, used or not. Feeds the
    /// memory-budget planner's peak projection.
    pub fn approx_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.approx_bytes()).sum()
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

    fn new_with_dtype_impl(
        cfg: &LlamaConfig,
        max_ctx: usize,
        dtype: KvDtype,
        keep: Option<&[bool]>,
    ) -> Self {
        let total_rows = cfg.n_kv_heads * max_ctx;
        let row_floats = total_rows * cfg.head_dim;
        let usm_stream = try_open_usm_kv_stream();
        let stream_ref = usm_stream.as_deref();
        let layers = (0..cfg.n_layers)
            .map(|li| {
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
                        };
                    }
                }
                match dtype {
                KvDtype::F32 => KvLayer::F32 {
                    k: alloc_kv_buf_f32(stream_ref, row_floats),
                    v: alloc_kv_buf_f32(stream_ref, row_floats),
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
                        KvDtype::Mxfp4 => KvLayer::Mxfp4 { k_packed: k, v_packed: v },
                        KvDtype::Mxfp6 => KvLayer::Mxfp6 { k_packed: k, v_packed: v },
                        _ => KvLayer::Mxfp8 { k_packed: k, v_packed: v },
                    }
                }
                }
            })
            .collect();
        Self {
            layers,
            seq_len: 0,
            max_ctx,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            dtype,
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
                        snap_kp[p_dst..p_dst + prefix_len * bytes_per_row]
                            .copy_from_slice(&k_packed[p_src..p_src + prefix_len * bytes_per_row]);
                        snap_vp[p_dst..p_dst + prefix_len * bytes_per_row]
                            .copy_from_slice(&v_packed[p_src..p_src + prefix_len * bytes_per_row]);
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
                        snap_kp[p_dst..p_dst + prefix_len * bytes_per_row]
                            .copy_from_slice(&k_packed[p_src..p_src + prefix_len * bytes_per_row]);
                        snap_vp[p_dst..p_dst + prefix_len * bytes_per_row]
                            .copy_from_slice(&v_packed[p_src..p_src + prefix_len * bytes_per_row]);
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
                        snap_kp[p_dst..p_dst + prefix_len * bytes_per_row]
                            .copy_from_slice(&k_packed[p_src..p_src + prefix_len * bytes_per_row]);
                        snap_vp[p_dst..p_dst + prefix_len * bytes_per_row]
                            .copy_from_slice(&v_packed[p_src..p_src + prefix_len * bytes_per_row]);
                    }
                    let k = KvBuf::from_vec(snap_kp);
                    let v = KvBuf::from_vec(snap_vp);
                    match layer {
                        KvLayer::Mxfp4 { .. } => KvLayer::Mxfp4 { k_packed: k, v_packed: v },
                        KvLayer::Mxfp6 { .. } => KvLayer::Mxfp6 { k_packed: k, v_packed: v },
                        _ => KvLayer::Mxfp8 { k_packed: k, v_packed: v },
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
            dtype: self.dtype,
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
        if self.dtype != KvDtype::F32 {
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
            dtype: self.dtype,
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
        assert_eq!(snap.dtype, self.dtype);
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
                (
                    KvLayer::F32 { k, v },
                    KvLayer::F32 {
                        k: sk,
                        v: sv,
                    },
                ) => {
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
                        k_q[q_dst..q_dst + pl * hd]
                            .copy_from_slice(&sk_q[q_src..q_src + pl * hd]);
                        v_q[q_dst..q_dst + pl * hd]
                            .copy_from_slice(&sv_q[q_src..q_src + pl * hd]);
                        let s_dst = h * self.max_ctx;
                        let s_src = h * pl;
                        k_scales[s_dst..s_dst + pl]
                            .copy_from_slice(&sks[s_src..s_src + pl]);
                        v_scales[s_dst..s_dst + pl]
                            .copy_from_slice(&svs[s_src..s_src + pl]);
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
                        k_scales[s_dst..s_dst + pl]
                            .copy_from_slice(&sks[s_src..s_src + pl]);
                        v_scales[s_dst..s_dst + pl]
                            .copy_from_slice(&svs[s_src..s_src + pl]);
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
                (
                    KvLayer::Q4_0 { k_q, v_q },
                    KvLayer::Q4_0 {
                        k_q: skq,
                        v_q: svq,
                    },
                ) => {
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
                (KvLayer::Mxfp4 { k_packed, v_packed }, KvLayer::Mxfp4 { k_packed: skp, v_packed: svp }) => {
                    restore_mxfp_prefix(k_packed, v_packed, skp, svp, hd, n_h, self.max_ctx, pl,
                        rustllama_kernels_cpu::mxfp::MXFP4_BLOCK_BYTES);
                }
                (KvLayer::Mxfp6 { k_packed, v_packed }, KvLayer::Mxfp6 { k_packed: skp, v_packed: svp }) => {
                    restore_mxfp_prefix(k_packed, v_packed, skp, svp, hd, n_h, self.max_ctx, pl,
                        rustllama_kernels_cpu::mxfp::MXFP6_BLOCK_BYTES);
                }
                (KvLayer::Mxfp8 { k_packed, v_packed }, KvLayer::Mxfp8 { k_packed: skp, v_packed: svp }) => {
                    restore_mxfp_prefix(k_packed, v_packed, skp, svp, hd, n_h, self.max_ctx, pl,
                        rustllama_kernels_cpu::mxfp::MXFP8_BLOCK_BYTES);
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
        KvDtype::Mxfp4 => crate::accel::cuda_decode_seed_kv_mxfp4,
        KvDtype::Mxfp6 => crate::accel::cuda_decode_seed_kv_mxfp6,
        _ => crate::accel::cuda_decode_seed_kv_mxfp8,
    };
    seed_fn(
        &k_packed[..], &v_packed[..], li, cur_pos as u32,
        n_heads as u32, n_kv_heads as u32, head_dim as u32, max_ctx as u32, n_layers as u32,
    );
    #[allow(clippy::type_complexity)]
    let gpu_fn: fn(&[f32], &[f32], &[f32], &mut [f32], usize, u32, u32, u32, u32, u32, u32) -> bool =
        match fmt {
            KvDtype::Mxfp4 => crate::accel::try_flash_attn_decode_gpu_mxfp4,
            KvDtype::Mxfp6 => crate::accel::try_flash_attn_decode_gpu_mxfp6,
            _ => crate::accel::try_flash_attn_decode_gpu_mxfp8,
        };
    let gpu_ok = gpu_fn(
        q_buf, k_rows, v_rows, attn_out, li, cur_pos as u32,
        n_heads as u32, n_kv_heads as u32, head_dim as u32, max_ctx as u32, n_layers as u32,
    );
    if !gpu_ok {
        #[allow(clippy::type_complexity)]
        let cpu_fn: fn(&[f32], &[u8], &[u8], &mut [f32], usize, usize, usize, usize, usize) =
            match fmt {
                KvDtype::Mxfp4 => mxfp_kv::gqa_attention_flash_decode_mxfp4,
                KvDtype::Mxfp6 => mxfp_kv::gqa_attention_flash_decode_mxfp6,
                _ => mxfp_kv::gqa_attention_flash_decode_mxfp8,
            };
        cpu_fn(q_buf, k_packed, v_packed, attn_out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len);
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
    let gpu_fn: fn(&[f32], &[u8], &[u8], &mut [f32], usize, usize, usize, usize, usize, usize) -> bool =
        match fmt {
            KvDtype::Mxfp4 => crate::accel::try_flash_attn_prefill_gpu_mxfp4,
            KvDtype::Mxfp6 => crate::accel::try_flash_attn_prefill_gpu_mxfp6,
            _ => crate::accel::try_flash_attn_prefill_gpu_mxfp8,
        };
    let gpu_ok = gpu_fn(
        q_buf, k_packed, v_packed, attn_out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base,
        n_new,
    );
    if !gpu_ok {
        #[allow(clippy::type_complexity)]
        let cpu_fn: fn(&[f32], &[u8], &[u8], &mut [f32], usize, usize, usize, usize, usize, usize) =
            match fmt {
                KvDtype::Mxfp4 => mxfp_kv::gqa_attention_flash_prefill_mxfp4,
                KvDtype::Mxfp6 => mxfp_kv::gqa_attention_flash_prefill_mxfp6,
                _ => mxfp_kv::gqa_attention_flash_prefill_mxfp8,
            };
        cpu_fn(
            q_buf, k_packed, v_packed, attn_out, n_heads, n_kv_heads, head_dim, max_ctx,
            kv_len_base, n_new,
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

        let mut push_tensor = |out: &mut Vec<LockTarget>, tier: u8, name: &str, t: &Tensor| {
            if let Some((ptr, len)) = t.storage.cpu_backing_ptr_len() {
                out.push(LockTarget { tier, name: name.to_string(), addr: ptr as usize, len });
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
            if let Some(v) = &b.b_q { push_vec(&mut out, 0, &format!("blk{i}.b_q"), v); }
            if let Some(v) = &b.b_k { push_vec(&mut out, 0, &format!("blk{i}.b_k"), v); }
            if let Some(v) = &b.b_v { push_vec(&mut out, 0, &format!("blk{i}.b_v"), v); }
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
        let mut push_moe_ffn = |out: &mut Vec<LockTarget>,
                                pfx: &str,
                                router: &Tensor,
                                gate_exps: &Tensor,
                                up_exps: &Tensor,
                                down_exps: &Tensor,
                                gate_shared: &Option<Tensor>,
                                up_shared: &Option<Tensor>,
                                down_shared: &Option<Tensor>| {
            push_tensor(out, 0, &format!("{pfx}.router"), router);
            if let Some(t) = gate_shared { push_tensor(out, 1, &format!("{pfx}.w_gate_shared"), t); }
            if let Some(t) = up_shared { push_tensor(out, 1, &format!("{pfx}.w_up_shared"), t); }
            if let Some(t) = down_shared { push_tensor(out, 1, &format!("{pfx}.w_down_shared"), t); }
            push_tensor(out, 2, &format!("{pfx}.w_gate_exps"), gate_exps);
            push_tensor(out, 2, &format!("{pfx}.w_up_exps"), up_exps);
            push_tensor(out, 2, &format!("{pfx}.w_down_exps"), down_exps);
        };

        // MoE blocks.
        if let Some(blocks) = &w.moe_blocks {
            for (i, b) in blocks.iter().enumerate() {
                push_vec(&mut out, 0, &format!("moe{i}.attn_norm"), &b.attn_norm);
                push_vec(&mut out, 0, &format!("moe{i}.ffn_norm"), &b.ffn_norm);
                if let Some(v) = &b.b_q { push_vec(&mut out, 0, &format!("moe{i}.b_q"), v); }
                if let Some(v) = &b.b_k { push_vec(&mut out, 0, &format!("moe{i}.b_k"), v); }
                if let Some(v) = &b.b_v { push_vec(&mut out, 0, &format!("moe{i}.b_v"), v); }
                push_tensor(&mut out, 0, &format!("moe{i}.w_q"), &b.w_q);
                push_tensor(&mut out, 0, &format!("moe{i}.w_k"), &b.w_k);
                push_tensor(&mut out, 0, &format!("moe{i}.w_v"), &b.w_v);
                push_tensor(&mut out, 0, &format!("moe{i}.w_o"), &b.w_o);
                push_moe_ffn(
                    &mut out, &format!("moe{i}"), &b.router,
                    &b.w_gate_exps, &b.w_up_exps, &b.w_down_exps,
                    &b.w_gate_shared, &b.w_up_shared, &b.w_down_shared,
                );
            }
        }

        // Hybrid layers (full-attention + SSM variants).
        if let Some(layers) = &w.hybrid_layers {
            for (i, layer) in layers.iter().enumerate() {
                match layer {
                    HybridLayer::FullAttention(b) => {
                        push_vec(&mut out, 0, &format!("hyb{i}.attn_norm"), &b.attn_norm);
                        push_vec(&mut out, 0, &format!("hyb{i}.post_attn_norm"), &b.post_attention_norm);
                        if let Some(v) = &b.q_norm { push_vec(&mut out, 0, &format!("hyb{i}.q_norm"), v); }
                        if let Some(v) = &b.k_norm { push_vec(&mut out, 0, &format!("hyb{i}.k_norm"), v); }
                        push_tensor(&mut out, 0, &format!("hyb{i}.w_q"), &b.w_q);
                        push_tensor(&mut out, 0, &format!("hyb{i}.w_k"), &b.w_k);
                        push_tensor(&mut out, 0, &format!("hyb{i}.w_v"), &b.w_v);
                        push_tensor(&mut out, 0, &format!("hyb{i}.w_o"), &b.w_o);
                        match &b.ffn {
                            HybridFfn::Moe {
                                router, w_gate_exps, w_up_exps, w_down_exps,
                                w_gate_shared, w_up_shared, w_down_shared,
                                shared_router, ..
                            } => {
                                if let Some(t) = shared_router {
                                    push_tensor(&mut out, 0, &format!("hyb{i}.shared_router"), t);
                                }
                                push_moe_ffn(
                                    &mut out, &format!("hyb{i}"), router,
                                    w_gate_exps, w_up_exps, w_down_exps,
                                    w_gate_shared, w_up_shared, w_down_shared,
                                );
                            }
                            HybridFfn::Dense { w_gate, w_up, w_down } => {
                                push_tensor(&mut out, 0, &format!("hyb{i}.ffn_gate"), w_gate);
                                push_tensor(&mut out, 0, &format!("hyb{i}.ffn_up"), w_up);
                                push_tensor(&mut out, 0, &format!("hyb{i}.ffn_down"), w_down);
                            }
                        }
                    }
                    HybridLayer::Ssm(b) => {
                        push_vec(&mut out, 0, &format!("ssm{i}.attn_norm"), &b.attn_norm);
                        push_vec(&mut out, 0, &format!("ssm{i}.post_attn_norm"), &b.post_attention_norm);
                        push_vec(&mut out, 0, &format!("ssm{i}.ssm_norm"), &b.ssm_norm);
                        push_tensor(&mut out, 0, &format!("ssm{i}.attn_qkv"), &b.attn_qkv);
                        push_tensor(&mut out, 0, &format!("ssm{i}.attn_gate"), &b.attn_gate);
                        push_tensor(&mut out, 0, &format!("ssm{i}.ssm_conv1d"), &b.ssm_conv1d);
                        push_tensor(&mut out, 0, &format!("ssm{i}.ssm_alpha"), &b.ssm_alpha);
                        push_tensor(&mut out, 0, &format!("ssm{i}.ssm_beta"), &b.ssm_beta);
                        push_tensor(&mut out, 0, &format!("ssm{i}.ssm_out"), &b.ssm_out);
                        match &b.ffn {
                            HybridFfn::Moe {
                                router, w_gate_exps, w_up_exps, w_down_exps,
                                w_gate_shared, w_up_shared, w_down_shared,
                                shared_router, ..
                            } => {
                                if let Some(t) = shared_router {
                                    push_tensor(&mut out, 0, &format!("ssm{i}.shared_router"), t);
                                }
                                push_moe_ffn(
                                    &mut out, &format!("ssm{i}"), router,
                                    w_gate_exps, w_up_exps, w_down_exps,
                                    w_gate_shared, w_up_shared, w_down_shared,
                                );
                            }
                            HybridFfn::Dense { w_gate, w_up, w_down } => {
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
                push_vec(&mut out, 0, &format!("mtp{i}.attn_norm"), &h.block.attn_norm);
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
        Ok(Self { cfg, weights })
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
        Ok(Self { cfg, weights })
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
        tally(&self.weights.token_embd, &mut uploaded, &mut skipped, &mut total_bytes);
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
                for t in [&b.w_q, &b.w_k, &b.w_v, &b.w_o, &b.router, &b.w_gate_exps, &b.w_up_exps, &b.w_down_exps] {
                    tally(t, &mut uploaded, &mut skipped, &mut total_bytes);
                }
                for t in [&b.w_gate_shared, &b.w_up_shared, &b.w_down_shared].into_iter().flatten() {
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
                        vec![&b.attn_qkv, &b.attn_gate, &b.ssm_conv1d, &b.ssm_alpha, &b.ssm_beta, &b.ssm_out],
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
                        for t in [w_gate_shared, w_up_shared, w_down_shared].into_iter().flatten() {
                            tally(t, &mut uploaded, &mut skipped, &mut total_bytes);
                        }
                    }
                    HybridFfn::Dense { w_gate, w_up, w_down } => {
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
    pub fn preload_weights_to_usm_swap(
        &mut self,
        n_gpu_layers: u32,
    ) -> (usize, usize, usize) {
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
            swap_one(&mut block.w_gate, &mut swapped, &mut skipped, &mut total_bytes);
            swap_one(&mut block.w_up, &mut swapped, &mut skipped, &mut total_bytes);
            swap_one(&mut block.w_down, &mut swapped, &mut skipped, &mut total_bytes);
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
        let cfg = &self.cfg;
        // Phase 2 of the qwen35moe roadmap binds hybrid tensors but
        // does not implement the SSM forward kernel; bail with a
        // clean panic message instead of silently producing garbage
        // by routing through the dense forward path. Phase 3 lifts
        // this guard.
        assert!(
            !self.weights.is_hybrid(),
            "hybrid attention+SSM forward (arch `{}`) not yet implemented \
             — Phase 3 of the qwen35moe roadmap (see docs/qwen35moe-roadmap.md). \
             Phase 2 binds all tensors; the model loaded successfully but \
             cannot inference yet.",
            cfg.arch
        );
        assert_eq!(embed_row.len(), cfg.d_model, "embed_row must be d_model long");
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
            d, d_q, d_kv, d_ff, head_dim, cfg.rope_theta, n_experts, top_k,
            |scratch| {
                self.forward_one_with_scratch_inner(
                    EmbedInput::Embed(embed_row),
                    pos,
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
        let cfg = &self.cfg;
        // Guard the dense forward against a mis-dispatched hybrid model
        // (mirrors `forward_one_from_embed`): a hybrid model routed here
        // would run ZERO layers and emit gibberish instead of failing
        // loudly. Phase 3 of the qwen35moe roadmap lifts this.
        assert!(
            !self.weights.is_hybrid(),
            "hybrid attention+SSM forward (arch `{}`) not yet implemented \
             — Phase 3 of the qwen35moe roadmap (see docs/qwen35moe-roadmap.md). \
             Phase 2 binds all tensors; the model loaded successfully but \
             cannot inference yet.",
            cfg.arch
        );
        assert_eq!(logits_out.len(), cfg.vocab_size);
        assert_eq!(kv.layers.len(), cfg.n_layers);
        // KV-cache geometry must match this model's config — a cache
        // built for a different model would corrupt the per-head KV
        // writes (indexed by `head_dim`/`n_kv_heads`). Replaces a former
        // tautological `assert_eq!(kv.max_ctx, kv.max_ctx)`.
        assert_eq!(kv.n_kv_heads, cfg.n_kv_heads, "kv cache n_kv_heads mismatch");
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
                    EmbedInput::Token(token_id), pos, kv, logits_out, scratch,
                );
            },
        );
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
        k::rmsnorm_f32_row(&hidden, &self.weights.output_norm, &mut final_norm, cfg.rms_eps);
        let lm_head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        {
            let mut had_buf: Vec<f32> = Vec::new();
            let x_in = hadamard_pre(self.weights.hadamard.as_ref(), lm_head, &final_norm, &mut had_buf);
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
                rms(&hidden), rms(&final_norm)
            );
            // Top-5 logits (id, value).
            let mut top: Vec<(usize, f32)> =
                logits_out.iter().copied().enumerate().collect();
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
                rms(logits_out), lmin, lmax, any_nan, top5.join(" ")
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
            let h_rms = (hidden.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / hidden.len() as f64).sqrt();
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
            let indep_rms = (indep_dequant.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / indep_dequant.len() as f64).sqrt();
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
                    let first8: Vec<String> = blk.iter().take(8).map(|b| format!("{b:02x}")).collect();
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
                    let first8: Vec<String> = blk.iter().take(8).map(|b| format!("{b:02x}")).collect();
                    eprintln!("    file_{label} off={off} nonzero={nonzero}/8 first8={}", first8.join(" "));
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
                        matvec_tensor_dispatch(&block.attn_gate, x_in, &mut gate_path, ssm_inner, d);
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
                    let mut dn_q_pre = if dump_dn { vec![0.0f32; n_v_heads * head_qk_dim] } else { vec![] };
                    let mut dn_k_pre = if dump_dn { vec![0.0f32; n_v_heads * head_qk_dim] } else { vec![] };
                    let mut dn_v = if dump_dn { vec![0.0f32; n_v_heads * head_v_dim] } else { vec![] };
                    let mut dn_g = if dump_dn { vec![0.0f32; n_v_heads] } else { vec![] };
                    let mut dn_beta = if dump_dn { vec![0.0f32; n_v_heads] } else { vec![] };
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
                            dn_q_pre[h * head_qk_dim..(h + 1) * head_qk_dim].copy_from_slice(&q_head);
                            dn_k_pre[h * head_qk_dim..(h + 1) * head_qk_dim].copy_from_slice(&k_head);
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
                                &q_head, &k_head, v_head, decay, beta[h],
                                state_head, &mut out_head, head_qk_dim, head_v_dim,
                                &mut v_tilde,
                            );
                            // Gated RMSNorm per head.
                            let gate_head = &gate_path[h * head_v_dim..(h + 1) * head_v_dim];
                            k::delta_net::gated_rmsnorm_f32_inplace(
                                &mut out_head, gate_head, &block.ssm_norm, cfg.rms_eps,
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
                                        q_head, k_head, v_head, decay, beta[h],
                                        state_head, out_head, head_qk_dim, head_v_dim,
                                        v_tilde,
                                    );
                                    let gate_head =
                                        &gate_path[h * head_v_dim..(h + 1) * head_v_dim];
                                    k::delta_net::gated_rmsnorm_f32_scratch(
                                        out_head, gate_head, &block.ssm_norm, cfg.rms_eps,
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
                            had, &block.ssm_out, &v_out_concat,
                            head_v_dim, n_qk_heads, n_v_heads / n_qk_heads,
                            &mut had_perm_buf, &mut had_buf,
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
                            router, gate_per_expert, up_per_expert, down_per_expert,
                            w_gate_shared, w_up_shared, w_down_shared, shared_router, ..
                        } => crate::moe::moe_ffn_one_into_parts(
                            &tmp_d,
                            router,
                            gate_per_expert,
                            up_per_expert,
                            down_per_expert,
                            w_gate_shared.as_ref(),
                            w_up_shared.as_ref(),
                            w_down_shared.as_ref(),
                            shared_router.as_ref(),
                            d, d_ff, n_experts, top_k,
                            &mut layer_out,
                            &mut gate_buf, &mut up_buf, &mut ff_buf, &mut down_buf,
                            &mut expert_logits, &mut picks,
                        ),
                        HybridFfn::Dense { w_gate, w_up, w_down } => dense_ffn_one_into(
                            w_gate, w_up, w_down, &tmp_d, d, d_ff,
                            &mut layer_out, &mut gate_buf, &mut up_buf, &mut ff_buf,
                            had, &mut had_buf,
                        ),
                    }
                    log_hidden("ssm-moe-out", li, &layer_out);
                    k::add_inplace_f32(&mut hidden, &layer_out);
                    log_hidden("post-ssm-ffn", li, &hidden);
                }
                HybridLayer::FullAttention(block) => {
                    // 1. Pre-attention RMSNorm.
                    k::rmsnorm_f32_row(&hidden, &block.attn_norm, &mut tmp_d, cfg.rms_eps);

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
                    {
                        let x_in = hadamard_pre(had, &block.w_q, &tmp_d, &mut had_buf);
                        matvec_tensor_dispatch(&block.w_q, x_in, &mut qg_buf, 2 * d_q, d);
                    }
                    // Deinterleave per-head into separate query / gate
                    // buffers (hoisted scratch).
                    for h in 0..n_heads {
                        let src_base = h * 2 * head_dim;
                        q_buf[h * head_dim..(h + 1) * head_dim]
                            .copy_from_slice(&qg_buf[src_base..src_base + head_dim]);
                        attn_gate_buf[h * head_dim..(h + 1) * head_dim]
                            .copy_from_slice(&qg_buf[src_base + head_dim..src_base + 2 * head_dim]);
                    }
                    {
                        let x_in = hadamard_pre(had, &block.w_k, &tmp_d, &mut had_buf);
                        matvec_tensor_dispatch(&block.w_k, x_in, &mut k_buf, d_kv, d);
                    }
                    {
                        let x_in = hadamard_pre(had, &block.w_v, &tmp_d, &mut had_buf);
                        matvec_tensor_dispatch(&block.w_v, x_in, &mut v_buf, d_kv, d);
                    }

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
                            &mut q_buf, n_heads, head_dim, rope_dim, pos, cfg.rope_theta,
                        );
                        apply_partial_rope_per_head(
                            &mut k_buf, n_kv_heads, head_dim, rope_dim, pos, cfg.rope_theta,
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
                            crate::accel::cuda_decode_seed_kv_f32(
                                k, v, li, cur_pos as u32,
                                n_heads as u32, n_kv_heads as u32,
                                head_dim as u32, max_ctx as u32,
                                cfg.n_layers as u32,
                            );
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
                            ) {
                                k::gqa_attention_flash_decode(
                                    &q_buf, k, v, &mut attn_out,
                                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
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
                            use rustllama_kernels_cpu::q4_0_kv;
                            use rustllama_kernels_cpu::hadamard::whiten_chunks_inplace;
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
                                whiten_chunks_inplace(&mut k_buf[..n_kv_heads * head_dim], KV_WHITEN_CHUNK);
                                whiten_chunks_inplace(&mut v_buf[..n_kv_heads * head_dim], KV_WHITEN_CHUNK);
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
                            crate::accel::cuda_decode_seed_kv_q4_0(
                                &k_q[..], &v_q[..], li, cur_pos as u32,
                                n_heads as u32, n_kv_heads as u32,
                                head_dim as u32, max_ctx as u32, cfg.n_layers as u32,
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
                            ) {
                                q4_0_kv::gqa_attention_flash_decode_q4_0(
                                    &q_buf, k_q, v_q, &mut attn_out,
                                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
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
                        KvLayer::Q8_0 { k_q, k_scales, v_q, v_scales } => {
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
                            crate::accel::cuda_decode_seed_kv_q8_0(
                                unsafe {
                                    std::slice::from_raw_parts(k_q.as_ptr() as *const u8, k_q.len())
                                },
                                unsafe {
                                    std::slice::from_raw_parts(v_q.as_ptr() as *const u8, v_q.len())
                                },
                                &k_scales[..], &v_scales[..], li, cur_pos as u32,
                                n_heads as u32, n_kv_heads as u32,
                                head_dim as u32, max_ctx as u32, cfg.n_layers as u32,
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
                            ) {
                                k::gqa_attention_flash_decode_q8_0(
                                    &q_buf, k_q, k_scales, v_q, v_scales, &mut attn_out,
                                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                                );
                            }
                            if !no_qgate_enabled() {
                                for (a, g) in attn_out.iter_mut().zip(attn_gate_buf.iter()) {
                                    let s = sigmoid_stable(*g);
                                    *a *= s;
                                }
                            }
                        }
                        KvLayer::TurboQuant { bits, k_packed, k_scales, v_packed, v_scales } => {
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
                                    &mut tq_row, *bits, &mut k_packed[p_dst..p_dst + bytes_per_row],
                                );
                                tq_row.copy_from_slice(&v_buf[h * head_dim..(h + 1) * head_dim]);
                                v_scales[row_idx] = rustllama_kernels_cpu::turboquant::quantize_row(
                                    &mut tq_row, *bits, &mut v_packed[p_dst..p_dst + bytes_per_row],
                                );
                            }
                            // GPU quant-KV flash decode first (USM); host
                            // slab written above, so decline → CPU.
                            // Seed the CUDA decode mirror with the prefill
                            // history (no-op once resident); host
                            // k_packed/k_scales are byte-identical to the mirror.
                            crate::accel::cuda_decode_seed_kv_tq(
                                *bits, &k_packed[..], &v_packed[..],
                                &k_scales[..], &v_scales[..], li, cur_pos as u32,
                                n_heads as u32, n_kv_heads as u32,
                                head_dim as u32, max_ctx as u32, cfg.n_layers as u32,
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
                            ) {
                                rustllama_kernels_cpu::turboquant::gqa_attention_flash_decode_tq(
                                    &q_buf, k_packed, k_scales, v_packed, v_scales, *bits, &mut attn_out,
                                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
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
                            // GPU quant-KV flash decode first (USM); host
                            // slab written above, so decline → CPU.
                            // Seed the CUDA decode mirror with the prefill
                            // history (no-op once resident); host k_packed/
                            // v_packed are byte-identical to the mirror.
                            crate::accel::cuda_decode_seed_kv_nvfp4(
                                &k_packed[..], &v_packed[..], li, cur_pos as u32,
                                n_heads as u32, n_kv_heads as u32,
                                head_dim as u32, max_ctx as u32, cfg.n_layers as u32,
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
                            ) {
                                rustllama_kernels_cpu::nvfp4::gqa_attention_flash_decode_nvfp4(
                                    &q_buf, k_packed, v_packed, &mut attn_out,
                                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
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
                                KvDtype::Mxfp4, &q_buf, &k_buf, &v_buf, k_packed, v_packed,
                                &mut attn_out, li, cur_pos, n_heads, n_kv_heads, head_dim, max_ctx,
                                kv_len, cfg.n_layers,
                            );
                            if !no_qgate_enabled() {
                                for (a, g) in attn_out.iter_mut().zip(attn_gate_buf.iter()) {
                                    *a *= sigmoid_stable(*g);
                                }
                            }
                        }
                        KvLayer::Mxfp6 { k_packed, v_packed } => {
                            mxfp_kv_decode(
                                KvDtype::Mxfp6, &q_buf, &k_buf, &v_buf, k_packed, v_packed,
                                &mut attn_out, li, cur_pos, n_heads, n_kv_heads, head_dim, max_ctx,
                                kv_len, cfg.n_layers,
                            );
                            if !no_qgate_enabled() {
                                for (a, g) in attn_out.iter_mut().zip(attn_gate_buf.iter()) {
                                    *a *= sigmoid_stable(*g);
                                }
                            }
                        }
                        KvLayer::Mxfp8 { k_packed, v_packed } => {
                            mxfp_kv_decode(
                                KvDtype::Mxfp8, &q_buf, &k_buf, &v_buf, k_packed, v_packed,
                                &mut attn_out, li, cur_pos, n_heads, n_kv_heads, head_dim, max_ctx,
                                kv_len, cfg.n_layers,
                            );
                            if !no_qgate_enabled() {
                                for (a, g) in attn_out.iter_mut().zip(attn_gate_buf.iter()) {
                                    *a *= sigmoid_stable(*g);
                                }
                            }
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
                            router, gate_per_expert, up_per_expert, down_per_expert,
                            w_gate_shared, w_up_shared, w_down_shared, shared_router, ..
                        } => crate::moe::moe_ffn_one_into_parts(
                            &tmp_d,
                            router,
                            gate_per_expert,
                            up_per_expert,
                            down_per_expert,
                            w_gate_shared.as_ref(),
                            w_up_shared.as_ref(),
                            w_down_shared.as_ref(),
                            shared_router.as_ref(),
                            d, d_ff, n_experts, top_k,
                            &mut layer_out,
                            &mut gate_buf, &mut up_buf, &mut ff_buf, &mut down_buf,
                            &mut expert_logits, &mut picks,
                        ),
                        HybridFfn::Dense { w_gate, w_up, w_down } => dense_ffn_one_into(
                            w_gate, w_up, w_down, &tmp_d, d, d_ff,
                            &mut layer_out, &mut gate_buf, &mut up_buf, &mut ff_buf,
                            had, &mut had_buf,
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
        let _ = self.forward_prefill_hybrid_impl(tokens, None, start_pos, kv, dn_cache, Some(logits_out));
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
        // token loops. Unset ⇒ the byte-identical per-token path runs.
        let ssm_prefill_chunked = std::env::var_os("RUSTLLAMA_SSM_PREFILL_CHUNKED").is_some();
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
                    if pp_on { pp_mark = std::time::Instant::now(); }
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
                            had, &block.attn_qkv, &hoist_norm[..nb * d], d, nb,
                            &mut hoist_rot,
                        );
                        matvec_tensor_batched_dispatch(
                            &block.attn_qkv, xs, &mut hoist_qkv[..nb * qkv_dim],
                            qkv_dim, d, nb,
                        );
                    }
                    {
                        let xs = hadamard_pre_rows(
                            had, &block.attn_gate, &hoist_norm[..nb * d], d, nb,
                            &mut hoist_rot,
                        );
                        matvec_tensor_batched_dispatch(
                            &block.attn_gate, xs, &mut hoist_gate[..nb * ssm_inner],
                            ssm_inner, d, nb,
                        );
                    }
                    // alpha/beta are unfolded (no Hadamard) — batched
                    // directly over the norm rows, then the same
                    // element-wise sigmoid the per-token path applied.
                    matvec_tensor_batched_dispatch(
                        &block.ssm_alpha, &hoist_norm[..nb * d],
                        &mut hoist_alpha[..nb * n_v_heads], n_v_heads, d, nb,
                    );
                    matvec_tensor_batched_dispatch(
                        &block.ssm_beta, &hoist_norm[..nb * d],
                        &mut hoist_beta[..nb * n_v_heads], n_v_heads, d, nb,
                    );
                    for b in hoist_beta[..nb * n_v_heads].iter_mut() {
                        let s = sigmoid_stable(*b);
                        *b = s;
                    }
                    if pp_on { pp[0] += pp_mark.elapsed().as_secs_f64() * 1e3; pp_mark = std::time::Instant::now(); }
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
                                had, &block.ssm_out, &v_out_concat,
                                head_v_dim, n_qk_heads, n_v_heads / n_qk_heads,
                                &mut had_perm_buf, &mut had_buf,
                            );
                            hoist_vrot[bi * ssm_inner..(bi + 1) * ssm_inner]
                                .copy_from_slice(x_in);
                        }
                    }
                    }
                    if pp_on { pp[1] += pp_mark.elapsed().as_secs_f64() * 1e3; pp_mark = std::time::Instant::now(); }
                    matvec_tensor_batched_dispatch(
                        &block.ssm_out, &hoist_vrot[..nb * ssm_inner],
                        &mut hoist_lout[..nb * d], d, ssm_inner, nb,
                    );
                    for bi in 0..nb {
                        k::add_inplace_f32(
                            &mut hidden_rows[(t0 + bi) * d..(t0 + bi + 1) * d],
                            &hoist_lout[bi * d..(bi + 1) * d],
                        );
                    }
                    if pp_on { pp[2] += pp_mark.elapsed().as_secs_f64() * 1e3; pp_mark = std::time::Instant::now(); }
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
                            router, gate_per_expert, up_per_expert, down_per_expert,
                            w_gate_shared, w_up_shared, w_down_shared, shared_router, ..
                        } => crate::moe::moe_ffn_chunk_into_parts(
                            &moe_in_rows,
                            n,
                            router,
                            gate_per_expert,
                            up_per_expert,
                            down_per_expert,
                            w_gate_shared.as_ref(),
                            w_up_shared.as_ref(),
                            w_down_shared.as_ref(),
                            shared_router.as_ref(),
                            d, d_ff, n_experts, top_k,
                            &mut moe_out_rows,
                        ),
                        HybridFfn::Dense { w_gate, w_up, w_down } => dense_ffn_chunk_into(
                            w_gate, w_up, w_down, &moe_in_rows, n, d, d_ff, &mut moe_out_rows,
                            had, &mut had_buf,
                        ),
                    }
                    for t in 0..n {
                        k::add_inplace_f32(
                            &mut hidden_rows[t * d..(t + 1) * d],
                            &moe_out_rows[t * d..(t + 1) * d],
                        );
                    }
                    if pp_on { pp[7] += pp_mark.elapsed().as_secs_f64() * 1e3; pp_mark = std::time::Instant::now(); }
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
                    if pp_on { pp_mark = std::time::Instant::now(); }
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
                            had, &block.w_q, &hoist_norm[..nb * d], d, nb, &mut hoist_rot,
                        );
                        matvec_tensor_batched_dispatch(
                            &block.w_q, xs, &mut hoist_qg[..nb * 2 * d_q], 2 * d_q, d, nb,
                        );
                    }
                    {
                        let xs = hadamard_pre_rows(
                            had, &block.w_k, &hoist_norm[..nb * d], d, nb, &mut hoist_rot,
                        );
                        matvec_tensor_batched_dispatch(
                            &block.w_k, xs, &mut hoist_kr[..nb * d_kv], d_kv, d, nb,
                        );
                    }
                    {
                        let xs = hadamard_pre_rows(
                            had, &block.w_v, &hoist_norm[..nb * d], d, nb, &mut hoist_rot,
                        );
                        matvec_tensor_batched_dispatch(
                            &block.w_v, xs, &mut hoist_vr[..nb * d_kv], d_kv, d, nb,
                        );
                    }
                    if pp_on { pp[3] += pp_mark.elapsed().as_secs_f64() * 1e3; pp_mark = std::time::Instant::now(); }
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
                                &mut q_buf, n_heads, head_dim, rope_dim, pos, cfg.rope_theta,
                            );
                            apply_partial_rope_per_head(
                                &mut k_buf, n_kv_heads, head_dim, rope_dim, pos, cfg.rope_theta,
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
                                use rustllama_kernels_cpu::q4_0_kv;
                                use rustllama_kernels_cpu::hadamard::whiten_chunks_inplace;
                                let blocks_per_row = head_dim / q4_0_kv::Q4_0_BLOCK_ELEMS;
                                let bytes_per_row =
                                    blocks_per_row * q4_0_kv::Q4_0_BLOCK_BYTES;
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
                                if let Some(b) =
                                    kv_bias_l.as_ref().and_then(|b| b.layer(li))
                                {
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
                            KvLayer::Q8_0 { k_q, k_scales, v_q, v_scales } => {
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
                            KvLayer::TurboQuant { bits, k_packed, k_scales, v_packed, v_scales } => {
                                let bytes_per_row =
                                    rustllama_kernels_cpu::turboquant::bytes_per_block(head_dim, *bits);
                                let mut tq_row = vec![0f32; head_dim];
                                for h in 0..n_kv_heads {
                                    let row_idx = h * max_ctx + cur_pos;
                                    let p_dst = row_idx * bytes_per_row;
                                    tq_row.copy_from_slice(&k_buf[h * head_dim..(h + 1) * head_dim]);
                                    k_scales[row_idx] = rustllama_kernels_cpu::turboquant::quantize_row(
                                        &mut tq_row, *bits, &mut k_packed[p_dst..p_dst + bytes_per_row],
                                    );
                                    tq_row.copy_from_slice(&v_buf[h * head_dim..(h + 1) * head_dim]);
                                    v_scales[row_idx] = rustllama_kernels_cpu::turboquant::quantize_row(
                                        &mut tq_row, *bits, &mut v_packed[p_dst..p_dst + bytes_per_row],
                                    );
                                }
                            }
                            KvLayer::Nvfp4 { k_packed, v_packed } => {
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
                                KvDtype::Mxfp4, &k_buf, &v_buf, k_packed, v_packed, cur_pos,
                                n_kv_heads, head_dim, max_ctx,
                            ),
                            KvLayer::Mxfp6 { k_packed, v_packed } => mxfp_kv_quantize_row(
                                KvDtype::Mxfp6, &k_buf, &v_buf, k_packed, v_packed, cur_pos,
                                n_kv_heads, head_dim, max_ctx,
                            ),
                            KvLayer::Mxfp8 { k_packed, v_packed } => mxfp_kv_quantize_row(
                                KvDtype::Mxfp8, &k_buf, &v_buf, k_packed, v_packed, cur_pos,
                                n_kv_heads, head_dim, max_ctx,
                            ),
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
                    if pp_on { pp[4] += pp_mark.elapsed().as_secs_f64() * 1e3; pp_mark = std::time::Instant::now(); }
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
                                if !crate::accel::try_flash_attn_prefill_gpu_f32(
                                    &hoist_qrows[..nb * d_q], k, v,
                                    &mut hoist_attn_out[..nb * d_q],
                                    n_heads, n_kv_heads, head_dim, max_ctx,
                                    kv_len_base, nb,
                                ) {
                                    k::gqa_attention_flash_prefill(
                                        &hoist_qrows[..nb * d_q], k, v,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads, n_kv_heads, head_dim, max_ctx,
                                        kv_len_base, nb,
                                    );
                                }
                            }
                            KvLayer::Q4_0 { k_q, v_q } => {
                                sub_whiten = q4_0_whiten_active(head_dim);
                                if !crate::accel::try_flash_attn_prefill_gpu_q4_0(
                                    &hoist_qrows[..nb * d_q], k_q, v_q,
                                    &mut hoist_attn_out[..nb * d_q],
                                    n_heads, n_kv_heads, head_dim, max_ctx,
                                    kv_len_base, nb,
                                ) {
                                    q4_0_kv::gqa_attention_flash_prefill_q4_0(
                                        &hoist_qrows[..nb * d_q], k_q, v_q,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads, n_kv_heads, head_dim, max_ctx,
                                        kv_len_base, nb,
                                    );
                                }
                            }
                            KvLayer::Q8_0 { k_q, k_scales, v_q, v_scales } => {
                                sub_whiten = false;
                                if !crate::accel::try_flash_attn_prefill_gpu_q8_0(
                                    &hoist_qrows[..nb * d_q], k_q, k_scales, v_q, v_scales,
                                    &mut hoist_attn_out[..nb * d_q],
                                    n_heads, n_kv_heads, head_dim, max_ctx,
                                    kv_len_base, nb,
                                ) {
                                    k::gqa_attention_flash_prefill_q8_0(
                                        &hoist_qrows[..nb * d_q], k_q, k_scales, v_q, v_scales,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads, n_kv_heads, head_dim, max_ctx,
                                        kv_len_base, nb,
                                    );
                                }
                            }
                            KvLayer::TurboQuant { bits, k_packed, k_scales, v_packed, v_scales } => {
                                sub_whiten = false;
                                if !crate::accel::try_flash_attn_prefill_gpu_tq(
                                    &hoist_qrows[..nb * d_q], k_packed, k_scales, v_packed, v_scales,
                                    *bits, &mut hoist_attn_out[..nb * d_q],
                                    n_heads, n_kv_heads, head_dim, max_ctx,
                                    kv_len_base, nb,
                                ) {
                                    rustllama_kernels_cpu::turboquant::gqa_attention_flash_prefill_tq(
                                        &hoist_qrows[..nb * d_q], k_packed, k_scales, v_packed, v_scales,
                                        *bits, &mut hoist_attn_out[..nb * d_q],
                                        n_heads, n_kv_heads, head_dim, max_ctx,
                                        kv_len_base, nb,
                                    );
                                }
                            }
                            KvLayer::Nvfp4 { k_packed, v_packed } => {
                                sub_whiten = false;
                                if !crate::accel::try_flash_attn_prefill_gpu_nvfp4(
                                    &hoist_qrows[..nb * d_q], k_packed, v_packed,
                                    &mut hoist_attn_out[..nb * d_q],
                                    n_heads, n_kv_heads, head_dim, max_ctx,
                                    kv_len_base, nb,
                                ) {
                                    rustllama_kernels_cpu::nvfp4::gqa_attention_flash_prefill_nvfp4(
                                        &hoist_qrows[..nb * d_q], k_packed, v_packed,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads, n_kv_heads, head_dim, max_ctx,
                                        kv_len_base, nb,
                                    );
                                }
                            }
                            // MXFP KV: flash-only (cache already populated in the
                            // per-position quantize pass); no whitening.
                            KvLayer::Mxfp4 { k_packed, v_packed } => {
                                sub_whiten = false;
                                if !crate::accel::try_flash_attn_prefill_gpu_mxfp4(
                                    &hoist_qrows[..nb * d_q], k_packed, v_packed,
                                    &mut hoist_attn_out[..nb * d_q],
                                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, nb,
                                ) {
                                    rustllama_kernels_cpu::mxfp_kv::gqa_attention_flash_prefill_mxfp4(
                                        &hoist_qrows[..nb * d_q], k_packed, v_packed,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, nb,
                                    );
                                }
                            }
                            KvLayer::Mxfp6 { k_packed, v_packed } => {
                                sub_whiten = false;
                                if !crate::accel::try_flash_attn_prefill_gpu_mxfp6(
                                    &hoist_qrows[..nb * d_q], k_packed, v_packed,
                                    &mut hoist_attn_out[..nb * d_q],
                                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, nb,
                                ) {
                                    rustllama_kernels_cpu::mxfp_kv::gqa_attention_flash_prefill_mxfp6(
                                        &hoist_qrows[..nb * d_q], k_packed, v_packed,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, nb,
                                    );
                                }
                            }
                            KvLayer::Mxfp8 { k_packed, v_packed } => {
                                sub_whiten = false;
                                if !crate::accel::try_flash_attn_prefill_gpu_mxfp8(
                                    &hoist_qrows[..nb * d_q], k_packed, v_packed,
                                    &mut hoist_attn_out[..nb * d_q],
                                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, nb,
                                ) {
                                    rustllama_kernels_cpu::mxfp_kv::gqa_attention_flash_prefill_mxfp8(
                                        &hoist_qrows[..nb * d_q], k_packed, v_packed,
                                        &mut hoist_attn_out[..nb * d_q],
                                        n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, nb,
                                    );
                                }
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
                    if pp_on { pp[5] += pp_mark.elapsed().as_secs_f64() * 1e3; pp_mark = std::time::Instant::now(); }
                    {
                        let xs = hadamard_pre_rows(
                            had, &block.w_o, &hoist_attn_out[..nb * d_q], d_q, nb,
                            &mut hoist_rot,
                        );
                        matvec_tensor_batched_dispatch(
                            &block.w_o, xs, &mut hoist_lout[..nb * d], d, d_q, nb,
                        );
                    }
                    for bi in 0..nb {
                        k::add_inplace_f32(
                            &mut hidden_rows[(t0 + bi) * d..(t0 + bi + 1) * d],
                            &hoist_lout[bi * d..(bi + 1) * d],
                        );
                    }
                    if pp_on { pp[6] += pp_mark.elapsed().as_secs_f64() * 1e3; pp_mark = std::time::Instant::now(); }
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
                            router, gate_per_expert, up_per_expert, down_per_expert,
                            w_gate_shared, w_up_shared, w_down_shared, shared_router, ..
                        } => crate::moe::moe_ffn_chunk_into_parts(
                            &moe_in_rows,
                            n,
                            router,
                            gate_per_expert,
                            up_per_expert,
                            down_per_expert,
                            w_gate_shared.as_ref(),
                            w_up_shared.as_ref(),
                            w_down_shared.as_ref(),
                            shared_router.as_ref(),
                            d, d_ff, n_experts, top_k,
                            &mut moe_out_rows,
                        ),
                        HybridFfn::Dense { w_gate, w_up, w_down } => dense_ffn_chunk_into(
                            w_gate, w_up, w_down, &moe_in_rows, n, d, d_ff, &mut moe_out_rows,
                            had, &mut had_buf,
                        ),
                    }
                    for t in 0..n {
                        k::add_inplace_f32(
                            &mut hidden_rows[t * d..(t + 1) * d],
                            &moe_out_rows[t * d..(t + 1) * d],
                        );
                    }
                    if pp_on { pp[7] += pp_mark.elapsed().as_secs_f64() * 1e3; pp_mark = std::time::Instant::now(); }
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
            let x_in =
                hadamard_pre_rows(had, lm_head, &final_norm_all, d, n, &mut had_rows_buf);
            matvec_tensor_batched_dispatch(lm_head, x_in, out, cfg.vocab_size, d, n);
            return out[(n - 1) * cfg.vocab_size..n * cfg.vocab_size].to_vec();
        }

        // Final output norm + LM head for the LAST token only.
        let last = &hidden_rows[(n - 1) * d..n * d];
        let mut final_norm = vec![0.0f32; d];
        k::rmsnorm_f32_row(last, &self.weights.output_norm, &mut final_norm, cfg.rms_eps);
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
        k::rmsnorm_f32_row(&hidden, &self.weights.output_norm, &mut final_norm, cfg.rms_eps);
        {
            let mut had_buf: Vec<f32> = Vec::new();
            let x_in =
                hadamard_pre(self.weights.hadamard.as_ref(), lm_head, &final_norm, &mut had_buf);
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
            d, d_q, d_kv, d_ff, head_dim, cfg.rope_theta, n_experts, top_k,
            |scratch| {
                self.forward_one_with_scratch_inner(
                    EmbedInput::Token(token_id),
                    pos,
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

    /// Single-position sidecar transformer block for one MTP head.
    /// See `forward_one_with_mtp_logits` for the topology rationale.
    fn forward_mtp_head_sidecar(
        &self,
        hidden_in: &[f32],
        head: &MtpHead,
        logits_out: &mut [f32],
    ) {
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
        // F32 w_q/w_k/w_v with `RUSTLLAMA_QKV_FUSED=1`), do ONE
        // matvec into a stacked output buffer and slice into the
        // three per-head buffers. Saves ~2/3 of the per-attention
        // dispatch overhead. Falls through to three matvecs when
        // the fused tensor isn't available (most models today).
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

        match input {
            EmbedInput::Token(token_id) => {
                k::embed_lookup_tensor(
                    &self.weights.token_embd,
                    &[token_id],
                    &mut hidden,
                    d,
                );
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
        // Optional debug: limit to first N layers via RUSTLLAMA_MAX_LAYERS env.
        let n_layers_used = std::env::var("RUSTLLAMA_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
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
            // ---- attention ----
            // Dispatch ladder: USM-resident SYCL (shared stream
            // with the attention kernel, no per-call alloc) →
            // host-pointer SYCL (per-call alloc + memcpy) →
            // CPU kernel. The `try_*` functions return `true` on
            // success and we short-circuit; on `false` we fall
            // through to the next tier.
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
            if rope_interleaved_enabled() {
                k::rope_inplace_interleaved(&mut q_buf, n_heads, head_dim, pos, cfg.rope_theta);
                k::rope_inplace_interleaved(&mut k_buf, n_kv_heads, head_dim, pos, cfg.rope_theta);
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
                    crate::accel::cuda_decode_seed_kv_f32(
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
                    if crate::accel::flash_attention_enabled() && kv_len >= flash_min {
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
                    } else {
                        k::gqa_attention_one_step(
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
                    crate::accel::cuda_decode_seed_kv_q8_0(
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
                    ) {
                    let flash_min = flash_kv_len_min();
                    if crate::accel::flash_attention_enabled() && kv_len >= flash_min {
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
                    } else {
                        k::gqa_attention_one_step_q8_0(
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
                    crate::accel::cuda_decode_seed_kv_tq(
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
                    ) {
                    let flash_min = flash_kv_len_min();
                    if crate::accel::flash_attention_enabled() && kv_len >= flash_min {
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
                    } else {
                        rustllama_kernels_cpu::turboquant::gqa_attention_one_step_tq(
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
                            let elem_off =
                                h * head_dim + b * rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS;
                            let blk_dst = p_dst
                                + b * rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES;
                            rustllama_kernels_cpu::nvfp4::quantize_block(
                                &k_buf[elem_off
                                    ..elem_off
                                        + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS],
                                &mut k_packed[blk_dst
                                    ..blk_dst + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES],
                            );
                            rustllama_kernels_cpu::nvfp4::quantize_block(
                                &v_buf[elem_off
                                    ..elem_off
                                        + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS],
                                &mut v_packed[blk_dst
                                    ..blk_dst + rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES],
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
                    crate::accel::cuda_decode_seed_kv_nvfp4(
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
                    ) {
                    let flash_min = flash_kv_len_min();
                    if crate::accel::flash_attention_enabled() && kv_len >= flash_min {
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
                        k::gqa_attention_one_step(
                            &q_buf,
                            &k_f32,
                            &v_f32,
                            &mut attn_out,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_ctx,
                            kv_len,
                        );
                    }
                    } // end else: CPU NVFP4 attention (GPU declined)
                }
                KvLayer::Mxfp4 { k_packed, v_packed } => mxfp_kv_decode(
                    KvDtype::Mxfp4, &q_buf, &k_buf, &v_buf, k_packed, v_packed, &mut attn_out,
                    layer_idx, cur_pos, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                    cfg.n_layers,
                ),
                KvLayer::Mxfp6 { k_packed, v_packed } => mxfp_kv_decode(
                    KvDtype::Mxfp6, &q_buf, &k_buf, &v_buf, k_packed, v_packed, &mut attn_out,
                    layer_idx, cur_pos, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                    cfg.n_layers,
                ),
                KvLayer::Mxfp8 { k_packed, v_packed } => mxfp_kv_decode(
                    KvDtype::Mxfp8, &q_buf, &k_buf, &v_buf, k_packed, v_packed, &mut attn_out,
                    layer_idx, cur_pos, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                    cfg.n_layers,
                ),
                KvLayer::Q4_0 { k_q, v_q } => {
                    use rustllama_kernels_cpu::q4_0_kv;
                    use rustllama_kernels_cpu::hadamard::whiten_chunks_inplace;
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
                        whiten_chunks_inplace(&mut k_buf[..n_kv_heads * head_dim], KV_WHITEN_CHUNK);
                        whiten_chunks_inplace(&mut v_buf[..n_kv_heads * head_dim], KV_WHITEN_CHUNK);
                    }
                    // 0b. Calibration observe (pre-bias, target basis)
                    //     + K mean-centering subtract — exactly
                    //     softmax-invariant (see kv_bias module docs).
                    if crate::kv_bias::calib_active() {
                        crate::kv_bias::calib_observe(layer_idx, &k_buf[..n_kv_heads * head_dim]);
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
                    crate::accel::cuda_decode_seed_kv_q4_0(
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
                    ) {
                    let flash_min = flash_kv_len_min();
                    if crate::accel::flash_attention_enabled() && kv_len >= flash_min {
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
                        k::gqa_attention_one_step(
                            &q_buf,
                            &k_f32,
                            &v_f32,
                            &mut attn_out,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_ctx,
                            kv_len,
                        );
                    }
                    } // end else: CPU Q4_0 attention (GPU declined)
                    // 3. Un-whiten the attention output (self-inverse
                    //    transform re-applied — see step 0).
                    if whiten {
                        whiten_chunks_inplace(&mut attn_out, KV_WHITEN_CHUNK);
                    }
                }
            }

            // H6: try the fully-fused output-proj + residual + norm
            // kernel first (one GPU dispatch for matvec(W_o) + add +
            // rmsnorm). Gated behind RUSTLLAMA_FUSED_OUT_PROJ_NORM
            // (default off) + dtype coverage; on success it writes the
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
                debug::attn_stats(layer_idx, &q_buf, &k_buf, &v_buf, &attn_out, &attn_proj, &hidden);
            } else {

            // Output projection: attn_proj = W_o · attn_out
            matvec_tensor_dispatch(block.w_o(), &attn_out, &mut attn_proj, d, d_q);

            debug::attn_stats(layer_idx, &q_buf, &k_buf, &v_buf, &attn_out, &attn_proj, &hidden);

            // ---- FFN ----
            // Fused dispatch ladder: GPU `add_rmsnorm_usm` (one
            // kernel does the residual add + post-attn norm) → CPU
            // `add_inplace_f32` then GPU `try_rmsnorm_usm_f32` →
            // host-pointer SYCL → CPU rmsnorm. `RUSTLLAMA_SKIP_ATTN`
            // bypasses the residual entirely (debugging).
            if skip_attn_enabled() {
                if !crate::accel::try_rmsnorm_usm_f32(
                    &hidden, block.ffn_norm(), &mut h_norm, d, cfg.rms_eps,
                ) && !crate::accel::try_rmsnorm_f32(
                    &hidden, block.ffn_norm(), &mut h_norm, d, cfg.rms_eps,
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
                    &hidden, block.ffn_norm(), &mut h_norm, d, cfg.rms_eps,
                ) && !crate::accel::try_rmsnorm_f32(
                    &hidden, block.ffn_norm(), &mut h_norm, d, cfg.rms_eps,
                ) {
                    k::rmsnorm_f32_row(&hidden, block.ffn_norm(), &mut h_norm, cfg.rms_eps);
                }
            }
            } // end else (H6 fused out-proj+norm not taken)

            // FFN dispatch: dense (existing matvec ladder) or MoE
            // (router + top-K expert SwiGLU). The MoE path uses
            // pre-allocated scratch from `ForwardScratch` so the
            // per-layer call is allocation-free — gate/up/ff
            // buffers are shared with the dense ladder (same
            // shape), down_buf / expert_logits / routed_picks are
            // dedicated MoE scratch slots.
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
                matvec_tensor_dispatch(&dense_block.w_down, &ffn_buf, &mut ffn_out, d, d_ff);
            }

            debug::ffn_stats(
                layer_idx,
                &hidden,
                &h_norm,
                &gate_buf,
                &up_buf,
                &ffn_buf,
                &ffn_out,
            );

            if !skip_ffn_enabled() {
                k::add_inplace_f32(&mut hidden, &ffn_out);
            }

            debug::trace_layer(layer_idx, &hidden);
        }

        // ---- output norm + LM head ----
        // `final_norm` is already pre-sized + zeroed from the
        // ForwardScratch take at the top of this function; we
        // overwrite it fully in the rmsnorm call below.
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

        let lm_head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        matvec_tensor_dispatch(lm_head, &final_norm, logits_out, cfg.vocab_size, d);

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
                (vec![0f32; d_ff], vec![0f32; d_ff], vec![0f32; d_ff], vec![0f32; d])
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };
        let (mut moe_expert_logits, mut moe_routed_picks) = if let Some(m) = moe_cfg.as_ref() {
            (vec![0f32; m.n_experts as usize], Vec::with_capacity(m.n_experts_used as usize))
        } else {
            (Vec::new(), Vec::new())
        };

        let attn_views: Vec<&dyn AttnBlock> =
            if let Some(mbs) = self.weights.moe_blocks.as_ref() {
                mbs.iter().map(|b| b as &dyn AttnBlock).collect()
            } else {
                self.weights
                    .blocks
                    .iter()
                    .map(|b| b as &dyn AttnBlock)
                    .collect()
            };

        let n_layers_used = std::env::var("RUSTLLAMA_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
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
                if rope_interleaved_enabled() {
                    k::rope_inplace_interleaved(q_row, n_heads, head_dim, slot.pos, cfg.rope_theta);
                    k::rope_inplace_interleaved(k_row, n_kv_heads, head_dim, slot.pos, cfg.rope_theta);
                } else {
                    k::rope_inplace_neox(q_row, n_heads, head_dim, slot.pos, cfg.rope_theta);
                    k::rope_inplace_neox(k_row, n_kv_heads, head_dim, slot.pos, cfg.rope_theta);
                }
            }

            // Per-slot: write the new K/V row to that slot's cache
            // (under the shared store lock), gather the slot's full
            // KV history for this layer, run serial attention.
            // Attention is per-slot because each slot has its own
            // kv_len and its own KV history — fused attention
            // requires a paged-attention kernel that we'll add later.
            for (i, slot) in slots.iter_mut().enumerate() {
                let kv_len = slot.pos as usize + 1;
                let k_row = &k_buf[i * d_kv..(i + 1) * d_kv];
                let v_row = &v_buf[i * d_kv..(i + 1) * d_kv];
                slot.cache
                    .write_token_shared(shared, layer_idx as u32, slot.pos, k_row, v_row)
                    .expect("paged write_token_shared: pos < capacity (asserted above)");
                // Slice the scratch to this slot's kv_len (the
                // gather only writes that prefix; later slots may
                // have a different kv_len but we always size to max
                // so the buffer fits all).
                let slab_used = n_kv_heads * kv_len * head_dim;
                let k_slice = &mut k_slab[..slab_used];
                let v_slice = &mut v_slab[..slab_used];
                slot.cache
                    .gather_layer_shared(shared, layer_idx as u32, k_slice, v_slice)
                    .expect("paged gather_layer_shared: cache covers kv_len positions");
                // Match the single-slot path's flash-vs-standard
                // threshold so per-slot logits are bit-identical to
                // `forward_one_paged_f32`.
                let flash_min = flash_kv_len_min();
                let q_row = &q_buf[i * d_q..(i + 1) * d_q];
                let out_row = &mut attn_out[i * d_q..(i + 1) * d_q];
                if crate::accel::flash_attention_enabled() && kv_len >= flash_min {
                    k::gqa_attention_flash_decode(
                        q_row, k_slice, v_slice, out_row,
                        n_heads, n_kv_heads, head_dim, kv_len, kv_len,
                    );
                } else {
                    k::gqa_attention_one_step(
                        q_row, k_slice, v_slice, out_row,
                        n_heads, n_kv_heads, head_dim, kv_len, kv_len,
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
                matvec_tensor_batched_dispatch(
                    &dense_block.w_up,
                    &h_norm,
                    &mut up_buf,
                    d_ff,
                    d,
                    m,
                );
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
                &hidden, &self.weights.output_norm, &mut final_norm_all, d, cfg.rms_eps,
            )
        {
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
        for (i, slot) in slots.iter_mut().enumerate() {
            slot.logits_out
                .copy_from_slice(&logits_all[i * vocab..(i + 1) * vocab]);
        }
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
            d, d_q, d_kv, d_ff, head_dim, cfg.rope_theta, moe_n_experts, moe_top_k,
            |scratch| self.forward_one_paged_f32_inner(
                token_id, pos, cache,
                store as &mut dyn crate::paged_kv_store::PagedKvStoreOps,
                logits_out, scratch, kv_len,
            ),
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
            d, d_q, d_kv, cfg.d_ff, head_dim, cfg.rope_theta, moe_n_experts, moe_top_k,
            |scratch| self.forward_one_paged_f32_inner(
                token_id, pos, cache,
                store as &mut dyn crate::paged_kv_store::PagedKvStoreOps,
                logits_out, scratch, kv_len,
            ),
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
            d, d_q, d_kv, cfg.d_ff, head_dim, cfg.rope_theta, moe_n_experts, moe_top_k,
            |scratch| self.forward_one_paged_f32_inner(
                token_id, pos, cache,
                store as &mut dyn crate::paged_kv_store::PagedKvStoreOps,
                logits_out, scratch, kv_len,
            ),
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
            d, d_q, d_kv, cfg.d_ff, head_dim, cfg.rope_theta, moe_n_experts, moe_top_k,
            |scratch| self.forward_one_paged_f32_inner(
                token_id, pos, cache,
                store as &mut dyn crate::paged_kv_store::PagedKvStoreOps,
                logits_out, scratch, kv_len,
            ),
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
            d, d_q, d_kv, cfg.d_ff, head_dim, cfg.rope_theta, moe_n_experts, moe_top_k,
            |scratch| self.forward_one_paged_f32_inner(
                token_id, pos, cache,
                store as &mut dyn crate::paged_kv_store::PagedKvStoreOps,
                logits_out, scratch, kv_len,
            ),
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
            d, d_q, d_kv, cfg.d_ff, head_dim, cfg.rope_theta, moe_n_experts, moe_top_k,
            |scratch| self.forward_one_paged_f32_inner(
                token_id, pos, cache,
                store as &mut dyn crate::paged_kv_store::PagedKvStoreOps,
                logits_out, scratch, kv_len,
            ),
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
            d, d_q, d_kv, d_ff, head_dim, cfg.rope_theta, moe_n_experts, moe_top_k,
            |scratch| self.forward_one_paged_f32_inner(
                token_id, pos, cache,
                store as &mut dyn crate::paged_kv_store::PagedKvStoreOps,
                logits_out, scratch, kv_len,
            ),
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

        let n_layers_used = std::env::var("RUSTLLAMA_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
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
            if rope_interleaved_enabled() {
                k::rope_inplace_interleaved(&mut q_buf, n_heads, head_dim, pos, cfg.rope_theta);
                k::rope_inplace_interleaved(&mut k_buf, n_kv_heads, head_dim, pos, cfg.rope_theta);
            } else {
                k::rope_inplace_neox(&mut q_buf, n_heads, head_dim, pos, cfg.rope_theta);
                k::rope_inplace_neox(&mut k_buf, n_kv_heads, head_dim, pos, cfg.rope_theta);
            }

            // Paged write: append this token's K/V row, then gather
            // [0, kv_len) for attention.
            // H9a: dyn-trait dispatch — handles both F32 and Q8_0
            // paged storage variants without code duplication.
            cache
                .write_token_dyn(store, layer_idx as u32, pos, &k_buf, &v_buf)
                .expect("paged write_token: pos < capacity (ensured above)");
            cache
                .gather_layer_dyn(store, layer_idx as u32, &mut paged_k_slab[..slab_len], &mut paged_v_slab[..slab_len])
                .expect("paged gather_layer: cache covers kv_len positions");
            // Attention — `max_ctx = kv_len` because the gather
            // sized the slab exactly there. Mirror the contiguous
            // F32 path's flash-vs-standard threshold so paged and
            // contiguous produce bit-identical output for the same
            // kv_len (different accumulation order between the two
            // kernels would otherwise diverge by a few rounding
            // bits).
            let flash_min = flash_kv_len_min();
            if crate::accel::flash_attention_enabled() && kv_len >= flash_min {
                k::gqa_attention_flash_decode(
                    &q_buf, &paged_k_slab[..slab_len], &paged_v_slab[..slab_len], &mut attn_out,
                    n_heads, n_kv_heads, head_dim, kv_len, kv_len,
                );
            } else {
                k::gqa_attention_one_step(
                    &q_buf, &paged_k_slab[..slab_len], &paged_v_slab[..slab_len], &mut attn_out,
                    n_heads, n_kv_heads, head_dim, kv_len, kv_len,
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
        k::rmsnorm_f32_row(&hidden, &self.weights.output_norm, &mut final_norm, cfg.rms_eps);
        let lm_head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        matvec_tensor_dispatch(lm_head, &final_norm, logits_out, cfg.vocab_size, d);

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
        // instead of failing loudly. Phase 3 of the qwen35moe roadmap
        // lifts this.
        assert!(
            !self.weights.is_hybrid(),
            "hybrid attention+SSM forward (arch `{}`) not yet implemented \
             — Phase 3 of the qwen35moe roadmap (see docs/qwen35moe-roadmap.md). \
             Phase 2 binds all tensors; the model loaded successfully but \
             cannot inference yet.",
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
        if batched_on && !tokens.is_empty() {
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
        self.forward_prefill_batched_f32_inner(
            tokens, start_pos, kv,
            LmHeadMode::Last(&mut logits),
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
        self.forward_prefill_batched_f32_inner(
            tokens, start_pos, kv,
            LmHeadMode::All(logits_out),
        );
    }

    fn forward_prefill_batched_f32_inner(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        mode: LmHeadMode<'_>,
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

        // Pre-compute RoPE inv-freq once.
        let rope_inv_freq = crate::accel::rope_inv_freq_table(head_dim, cfg.rope_theta);

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
            (vec![0f32; m.n_experts as usize], Vec::with_capacity(m.n_experts_used as usize))
        } else {
            (Vec::new(), Vec::new())
        };

        // Collect `&dyn AttnBlock` per layer so the attention body is
        // a single uniform loop regardless of dense vs MoE — same
        // pattern as `forward_one_with_scratch`.
        let attn_views: Vec<&dyn AttnBlock> =
            if let Some(mbs) = self.weights.moe_blocks.as_ref() {
                mbs.iter().map(|b| b as &dyn AttnBlock).collect()
            } else {
                self.weights
                    .blocks
                    .iter()
                    .map(|b| b as &dyn AttnBlock)
                    .collect()
            };

        let n_layers_used = std::env::var("RUSTLLAMA_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
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
                let pos_i = (kv_len_base + i) as u32;
                if rope_interleaved_enabled() {
                    k::rope_inplace_interleaved(q_row, n_heads, head_dim, pos_i, cfg.rope_theta);
                    k::rope_inplace_interleaved(k_row, n_kv_heads, head_dim, pos_i, cfg.rope_theta);
                } else {
                    // Use the same neox/half-split convention as
                    // forward_one. Skip SYCL routing for now —
                    // batched prefill stays on the CPU path.
                    let _ = &rope_inv_freq;
                    k::rope_inplace_neox(q_row, n_heads, head_dim, pos_i, cfg.rope_theta);
                    k::rope_inplace_neox(k_row, n_kv_heads, head_dim, pos_i, cfg.rope_theta);
                }
            }

            // Append all n_new K/V rows to the cache, then run one
            // batched attention call.
            let kv_layer = &mut kv.layers[layer_idx];
            match kv_layer {
                KvLayer::F32 { k, v } => {
                    for i in 0..n_new {
                        let pos_i = kv_len_base + i;
                        for h in 0..n_kv_heads {
                            let dst = (h * max_ctx + pos_i) * head_dim;
                            let src_k =
                                &k_buf[i * d_kv + h * head_dim..i * d_kv + (h + 1) * head_dim];
                            let src_v =
                                &v_buf[i * d_kv + h * head_dim..i * d_kv + (h + 1) * head_dim];
                            k[dst..dst + head_dim].copy_from_slice(src_k);
                            v[dst..dst + head_dim].copy_from_slice(src_v);
                        }
                    }
                    // GPU prefill attention via SYCL — falls back
                    // to the CPU path below if USM is disabled, the
                    // GPU is unavailable, or any sanity check fails.
                    // The hook is internally a one-shot copy in,
                    // kernel, copy out — see `try_flash_attn_prefill_usm_f32`
                    // for the layout contract.
                    let used_gpu = crate::accel::try_flash_attn_prefill_gpu_f32(
                        &q_buf,
                        k,
                        v,
                        &mut attn_out,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_len_base,
                        n_new,
                    );
                    if !used_gpu {
                        k::gqa_attention_flash_prefill(
                            &q_buf,
                            k,
                            v,
                            &mut attn_out,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_ctx,
                            kv_len_base,
                            n_new,
                        );
                    }
                }
                _ => unreachable!("forward_prefill_batched_f32 entered with non-F32 KV"),
            }

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
                    last_hidden, &self.weights.output_norm, &mut final_norm, cfg.rms_eps,
                );
                matvec_tensor_dispatch(lm_head, &final_norm, out, cfg.vocab_size, d);
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
                    k::rmsnorm_f32_row(
                        in_row, &self.weights.output_norm, out_row, cfg.rms_eps,
                    );
                }
                matvec_tensor_batched_dispatch(
                    lm_head, &final_norm_all, out, cfg.vocab_size, d, n_new,
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

        let rope_inv_freq = crate::accel::rope_inv_freq_table(head_dim, cfg.rope_theta);

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

        let n_layers_used = std::env::var("RUSTLLAMA_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
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
                if rope_interleaved_enabled() {
                    k::rope_inplace_interleaved(q_row, n_heads, head_dim, pos_i, cfg.rope_theta);
                    k::rope_inplace_interleaved(k_row, n_kv_heads, head_dim, pos_i, cfg.rope_theta);
                } else {
                    let _ = &rope_inv_freq;
                    k::rope_inplace_neox(q_row, n_heads, head_dim, pos_i, cfg.rope_theta);
                    k::rope_inplace_neox(k_row, n_kv_heads, head_dim, pos_i, cfg.rope_theta);
                }
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
            // hook returns false.
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
            );
            if !used_gpu {
                k::gqa_attention_flash_prefill(
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
        k::rmsnorm_f32_row(last_hidden, &self.weights.output_norm, &mut final_norm, cfg.rms_eps);
        let lm_head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        let mut logits = vec![0f32; cfg.vocab_size];
        matvec_tensor_dispatch(lm_head, &final_norm, &mut logits, cfg.vocab_size, d);
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
        self.forward_prefill_batched_tq_inner(
            tokens, start_pos, kv,
            LmHeadMode::Last(&mut logits),
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
        self.forward_prefill_batched_tq_inner(
            tokens, start_pos, kv,
            LmHeadMode::All(logits_out),
        );
    }

    fn forward_prefill_batched_tq_inner(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        mode: LmHeadMode<'_>,
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

        // MoE single-token scratch — see forward_prefill_batched_f32.
        let moe_cfg = cfg.moe.clone();
        let (mut moe_gate_one, mut moe_up_one, mut moe_ff_one, mut moe_down_one) =
            if moe_cfg.is_some() {
                (vec![0f32; d_ff], vec![0f32; d_ff], vec![0f32; d_ff], vec![0f32; d])
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };
        let (mut moe_expert_logits, mut moe_routed_picks) = if let Some(m) = moe_cfg.as_ref() {
            (vec![0f32; m.n_experts as usize], Vec::with_capacity(m.n_experts_used as usize))
        } else {
            (Vec::new(), Vec::new())
        };

        let attn_views: Vec<&dyn AttnBlock> =
            if let Some(mbs) = self.weights.moe_blocks.as_ref() {
                mbs.iter().map(|b| b as &dyn AttnBlock).collect()
            } else {
                self.weights
                    .blocks
                    .iter()
                    .map(|b| b as &dyn AttnBlock)
                    .collect()
            };

        let n_layers_used = std::env::var("RUSTLLAMA_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
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
            // Stage 1 (per-token): pre-attention rmsnorm.
            for i in 0..n_new {
                let h_row = &hidden[i * d..(i + 1) * d];
                let n_row = &mut h_norm[i * d..(i + 1) * d];
                k::rmsnorm_f32_row(h_row, block.attn_norm(), n_row, cfg.rms_eps);
            }
            // Stage 2 (batched): Q / K / V projections.
            matvec_tensor_batched_dispatch(block.w_q(), &h_norm, &mut q_buf, d_q, d, n_new);
            matvec_tensor_batched_dispatch(block.w_k(), &h_norm, &mut k_buf, d_kv, d, n_new);
            matvec_tensor_batched_dispatch(block.w_v(), &h_norm, &mut v_buf, d_kv, d, n_new);
            // Stage 3 (per-token): bias + RoPE.
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
                let pos_i = (kv_len_base + i) as u32;
                if rope_interleaved_enabled() {
                    k::rope_inplace_interleaved(q_row, n_heads, head_dim, pos_i, cfg.rope_theta);
                    k::rope_inplace_interleaved(k_row, n_kv_heads, head_dim, pos_i, cfg.rope_theta);
                } else {
                    k::rope_inplace_neox(q_row, n_heads, head_dim, pos_i, cfg.rope_theta);
                    k::rope_inplace_neox(k_row, n_kv_heads, head_dim, pos_i, cfg.rope_theta);
                }
            }

            // Quantize all n_new K/V rows into the TQ packed cache,
            // then run the batched TQ flash-prefill kernel.
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
                            k_scales[row_idx] =
                                rustllama_kernels_cpu::turboquant::quantize_row(
                                    &mut scratch,
                                    *bits,
                                    &mut k_packed[p_dst..p_dst + bytes_per_row],
                                );
                            // V row.
                            scratch.copy_from_slice(
                                &v_buf[i * d_kv + h * head_dim..i * d_kv + (h + 1) * head_dim],
                            );
                            v_scales[row_idx] =
                                rustllama_kernels_cpu::turboquant::quantize_row(
                                    &mut scratch,
                                    *bits,
                                    &mut v_packed[p_dst..p_dst + bytes_per_row],
                                );
                        }
                    }
                    if !crate::accel::try_flash_attn_prefill_gpu_tq(
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
                        kv_len_base,
                        n_new,
                    ) {
                        rustllama_kernels_cpu::turboquant::gqa_attention_flash_prefill_tq(
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
                            kv_len_base,
                            n_new,
                        );
                    }
                }
                _ => unreachable!("forward_prefill_batched_tq entered with non-TQ KV"),
            }

            // Stage 4 (batched): output projection.
            matvec_tensor_batched_dispatch(block.w_o(), &attn_out, &mut attn_proj, d, d_q, n_new);
            // Stage 5 (per-token): residual + post-attn rmsnorm.
            for i in 0..n_new {
                let proj_row = &attn_proj[i * d..(i + 1) * d];
                let h_row = &mut hidden[i * d..(i + 1) * d];
                if !skip_attn_enabled() {
                    k::add_inplace_f32(h_row, proj_row);
                }
                let n_row = &mut h_norm[i * d..(i + 1) * d];
                k::rmsnorm_f32_row(h_row, block.ffn_norm(), n_row, cfg.rms_eps);
            }
            // Stage 6-8: FFN dense matvec ladder OR per-token MoE.
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
                    let g_row = &gate_buf[i * d_ff..(i + 1) * d_ff];
                    let u_row = &up_buf[i * d_ff..(i + 1) * d_ff];
                    let f_row = &mut ffn_buf[i * d_ff..(i + 1) * d_ff];
                    k::silu_mul_f32(g_row, u_row, f_row);
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
            // Stage 9 (per-token): residual.
            if !skip_ffn_enabled() {
                for i in 0..n_new {
                    let h_row = &mut hidden[i * d..(i + 1) * d];
                    let o_row = &ffn_out[i * d..(i + 1) * d];
                    k::add_inplace_f32(h_row, o_row);
                }
            }
        }
        kv.seq_len = kv.seq_len.max(kv_len_base + n_new);

        // LM-head dispatch — see `LmHeadMode` doc.
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
                    last_hidden, &self.weights.output_norm, &mut final_norm, cfg.rms_eps,
                );
                matvec_tensor_dispatch(lm_head, &final_norm, out, cfg.vocab_size, d);
            }
            LmHeadMode::All(out) => {
                let mut final_norm_all = vec![0f32; n_new * d];
                for i in 0..n_new {
                    let in_row = &hidden[i * d..(i + 1) * d];
                    let out_row = &mut final_norm_all[i * d..(i + 1) * d];
                    k::rmsnorm_f32_row(
                        in_row, &self.weights.output_norm, out_row, cfg.rms_eps,
                    );
                }
                matvec_tensor_batched_dispatch(
                    lm_head, &final_norm_all, out, cfg.vocab_size, d, n_new,
                );
            }
        }
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
        self.forward_prefill_batched_q8_0_inner(
            tokens, start_pos, kv,
            LmHeadMode::Last(&mut logits),
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
        self.forward_prefill_batched_q8_0_inner(
            tokens, start_pos, kv,
            LmHeadMode::All(logits_out),
        );
    }

    fn forward_prefill_batched_q8_0_inner(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        mode: LmHeadMode<'_>,
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

        // MoE single-token scratch — see forward_prefill_batched_f32
        // for the rationale (per-token routing means per-token FFN).
        let moe_cfg = cfg.moe.clone();
        let (mut moe_gate_one, mut moe_up_one, mut moe_ff_one, mut moe_down_one) =
            if moe_cfg.is_some() {
                (vec![0f32; d_ff], vec![0f32; d_ff], vec![0f32; d_ff], vec![0f32; d])
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };
        let (mut moe_expert_logits, mut moe_routed_picks) = if let Some(m) = moe_cfg.as_ref() {
            (vec![0f32; m.n_experts as usize], Vec::with_capacity(m.n_experts_used as usize))
        } else {
            (Vec::new(), Vec::new())
        };

        let attn_views: Vec<&dyn AttnBlock> =
            if let Some(mbs) = self.weights.moe_blocks.as_ref() {
                mbs.iter().map(|b| b as &dyn AttnBlock).collect()
            } else {
                self.weights
                    .blocks
                    .iter()
                    .map(|b| b as &dyn AttnBlock)
                    .collect()
            };

        let n_layers_used = std::env::var("RUSTLLAMA_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
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
            // Stage 1 (per-token): pre-attention rmsnorm.
            for i in 0..n_new {
                let h_row = &hidden[i * d..(i + 1) * d];
                let n_row = &mut h_norm[i * d..(i + 1) * d];
                k::rmsnorm_f32_row(h_row, block.attn_norm(), n_row, cfg.rms_eps);
            }
            // Stage 2 (batched): Q / K / V projections.
            matvec_tensor_batched_dispatch(block.w_q(), &h_norm, &mut q_buf, d_q, d, n_new);
            matvec_tensor_batched_dispatch(block.w_k(), &h_norm, &mut k_buf, d_kv, d, n_new);
            matvec_tensor_batched_dispatch(block.w_v(), &h_norm, &mut v_buf, d_kv, d, n_new);
            // Stage 3 (per-token): bias + RoPE.
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
                let pos_i = (kv_len_base + i) as u32;
                if rope_interleaved_enabled() {
                    k::rope_inplace_interleaved(q_row, n_heads, head_dim, pos_i, cfg.rope_theta);
                    k::rope_inplace_interleaved(k_row, n_kv_heads, head_dim, pos_i, cfg.rope_theta);
                } else {
                    k::rope_inplace_neox(q_row, n_heads, head_dim, pos_i, cfg.rope_theta);
                    k::rope_inplace_neox(k_row, n_kv_heads, head_dim, pos_i, cfg.rope_theta);
                }
            }

            // Quantize all n_new K/V rows into the Q8_0 cache, then
            // run the batched Q8_0 flash-prefill kernel.
            let kv_layer = &mut kv.layers[layer_idx];
            match kv_layer {
                KvLayer::Q8_0 { k_q, k_scales, v_q, v_scales } => {
                    for i in 0..n_new {
                        let pos_i = kv_len_base + i;
                        for h in 0..n_kv_heads {
                            let row_idx = h * max_ctx + pos_i;
                            let dst = row_idx * head_dim;
                            let k_src =
                                &k_buf[i * d_kv + h * head_dim..i * d_kv + (h + 1) * head_dim];
                            k_scales[row_idx] =
                                quantize_row_q8_0(k_src, &mut k_q[dst..dst + head_dim]);
                            let v_src =
                                &v_buf[i * d_kv + h * head_dim..i * d_kv + (h + 1) * head_dim];
                            v_scales[row_idx] =
                                quantize_row_q8_0(v_src, &mut v_q[dst..dst + head_dim]);
                        }
                    }
                    if !crate::accel::try_flash_attn_prefill_gpu_q8_0(
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
                        kv_len_base,
                        n_new,
                    ) {
                        k::gqa_attention_flash_prefill_q8_0(
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
                            kv_len_base,
                            n_new,
                        );
                    }
                }
                _ => unreachable!("forward_prefill_batched_q8_0 entered with non-Q8_0 KV"),
            }

            // Stage 4 (batched): output projection.
            matvec_tensor_batched_dispatch(block.w_o(), &attn_out, &mut attn_proj, d, d_q, n_new);
            // Stage 5 (per-token): residual + post-attn rmsnorm.
            for i in 0..n_new {
                let proj_row = &attn_proj[i * d..(i + 1) * d];
                let h_row = &mut hidden[i * d..(i + 1) * d];
                if !skip_attn_enabled() {
                    k::add_inplace_f32(h_row, proj_row);
                }
                let n_row = &mut h_norm[i * d..(i + 1) * d];
                k::rmsnorm_f32_row(h_row, block.ffn_norm(), n_row, cfg.rms_eps);
            }
            // Stage 6-8: FFN. Dense path matvec-batches; MoE loops
            // per-token through `moe_ffn_one_into`.
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
                    let g_row = &gate_buf[i * d_ff..(i + 1) * d_ff];
                    let u_row = &up_buf[i * d_ff..(i + 1) * d_ff];
                    let f_row = &mut ffn_buf[i * d_ff..(i + 1) * d_ff];
                    k::silu_mul_f32(g_row, u_row, f_row);
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
            // Stage 9 (per-token): residual.
            if !skip_ffn_enabled() {
                for i in 0..n_new {
                    let h_row = &mut hidden[i * d..(i + 1) * d];
                    let o_row = &ffn_out[i * d..(i + 1) * d];
                    k::add_inplace_f32(h_row, o_row);
                }
            }
        }
        kv.seq_len = kv.seq_len.max(kv_len_base + n_new);

        // LM-head dispatch — see `LmHeadMode` doc. Same shape as
        // the F32 path's final step.
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
                    last_hidden, &self.weights.output_norm, &mut final_norm, cfg.rms_eps,
                );
                matvec_tensor_dispatch(lm_head, &final_norm, out, cfg.vocab_size, d);
            }
            LmHeadMode::All(out) => {
                let mut final_norm_all = vec![0f32; n_new * d];
                for i in 0..n_new {
                    let in_row = &hidden[i * d..(i + 1) * d];
                    let out_row = &mut final_norm_all[i * d..(i + 1) * d];
                    k::rmsnorm_f32_row(
                        in_row, &self.weights.output_norm, out_row, cfg.rms_eps,
                    );
                }
                matvec_tensor_batched_dispatch(
                    lm_head, &final_norm_all, out, cfg.vocab_size, d, n_new,
                );
            }
        }
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
        self.forward_prefill_batched_nvfp4_inner(
            tokens, start_pos, kv,
            LmHeadMode::Last(&mut logits),
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
        self.forward_prefill_batched_nvfp4_inner(
            tokens, start_pos, kv,
            LmHeadMode::All(logits_out),
        );
    }

    fn forward_prefill_batched_nvfp4_inner(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        mode: LmHeadMode<'_>,
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

        // MoE single-token scratch — see forward_prefill_batched_f32.
        let moe_cfg = cfg.moe.clone();
        let (mut moe_gate_one, mut moe_up_one, mut moe_ff_one, mut moe_down_one) =
            if moe_cfg.is_some() {
                (vec![0f32; d_ff], vec![0f32; d_ff], vec![0f32; d_ff], vec![0f32; d])
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };
        let (mut moe_expert_logits, mut moe_routed_picks) = if let Some(m) = moe_cfg.as_ref() {
            (vec![0f32; m.n_experts as usize], Vec::with_capacity(m.n_experts_used as usize))
        } else {
            (Vec::new(), Vec::new())
        };

        let attn_views: Vec<&dyn AttnBlock> =
            if let Some(mbs) = self.weights.moe_blocks.as_ref() {
                mbs.iter().map(|b| b as &dyn AttnBlock).collect()
            } else {
                self.weights
                    .blocks
                    .iter()
                    .map(|b| b as &dyn AttnBlock)
                    .collect()
            };

        let n_layers_used = std::env::var("RUSTLLAMA_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
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
            // Stage 1 (per-token): pre-attention rmsnorm.
            for i in 0..n_new {
                let h_row = &hidden[i * d..(i + 1) * d];
                let n_row = &mut h_norm[i * d..(i + 1) * d];
                k::rmsnorm_f32_row(h_row, block.attn_norm(), n_row, cfg.rms_eps);
            }
            // Stage 2 (batched): Q / K / V projections.
            matvec_tensor_batched_dispatch(block.w_q(), &h_norm, &mut q_buf, d_q, d, n_new);
            matvec_tensor_batched_dispatch(block.w_k(), &h_norm, &mut k_buf, d_kv, d, n_new);
            matvec_tensor_batched_dispatch(block.w_v(), &h_norm, &mut v_buf, d_kv, d, n_new);
            // Stage 3 (per-token): bias + RoPE.
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
                let pos_i = (kv_len_base + i) as u32;
                if rope_interleaved_enabled() {
                    k::rope_inplace_interleaved(q_row, n_heads, head_dim, pos_i, cfg.rope_theta);
                    k::rope_inplace_interleaved(k_row, n_kv_heads, head_dim, pos_i, cfg.rope_theta);
                } else {
                    k::rope_inplace_neox(q_row, n_heads, head_dim, pos_i, cfg.rope_theta);
                    k::rope_inplace_neox(k_row, n_kv_heads, head_dim, pos_i, cfg.rope_theta);
                }
            }

            // Quantize all n_new K/V rows per-block into the NVFP4
            // cache, then call the batched NVFP4 prefill kernel.
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
                                let elem_off =
                                    i * d_kv + h * head_dim + b * NVFP4_BLOCK_ELEMS;
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
                        &q_buf,
                        k_packed,
                        v_packed,
                        &mut attn_out,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_len_base,
                        n_new,
                    ) {
                        rustllama_kernels_cpu::nvfp4::gqa_attention_flash_prefill_nvfp4(
                            &q_buf,
                            k_packed,
                            v_packed,
                            &mut attn_out,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_ctx,
                            kv_len_base,
                            n_new,
                        );
                    }
                }
                _ => unreachable!("forward_prefill_batched_nvfp4 entered with non-NVFP4 KV"),
            }

            // Stage 4 (batched): output projection.
            matvec_tensor_batched_dispatch(block.w_o(), &attn_out, &mut attn_proj, d, d_q, n_new);
            // Stage 5 (per-token): residual + post-attn rmsnorm.
            for i in 0..n_new {
                let proj_row = &attn_proj[i * d..(i + 1) * d];
                let h_row = &mut hidden[i * d..(i + 1) * d];
                if !skip_attn_enabled() {
                    k::add_inplace_f32(h_row, proj_row);
                }
                let n_row = &mut h_norm[i * d..(i + 1) * d];
                k::rmsnorm_f32_row(h_row, block.ffn_norm(), n_row, cfg.rms_eps);
            }
            // Stage 6-8: FFN dense matvec ladder OR per-token MoE.
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
                    let g_row = &gate_buf[i * d_ff..(i + 1) * d_ff];
                    let u_row = &up_buf[i * d_ff..(i + 1) * d_ff];
                    let f_row = &mut ffn_buf[i * d_ff..(i + 1) * d_ff];
                    k::silu_mul_f32(g_row, u_row, f_row);
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
            // Stage 9 (per-token): residual.
            if !skip_ffn_enabled() {
                for i in 0..n_new {
                    let h_row = &mut hidden[i * d..(i + 1) * d];
                    let o_row = &ffn_out[i * d..(i + 1) * d];
                    k::add_inplace_f32(h_row, o_row);
                }
            }
        }
        kv.seq_len = kv.seq_len.max(kv_len_base + n_new);

        // LM-head dispatch — see `LmHeadMode` doc.
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
                    last_hidden, &self.weights.output_norm, &mut final_norm, cfg.rms_eps,
                );
                matvec_tensor_dispatch(lm_head, &final_norm, out, cfg.vocab_size, d);
            }
            LmHeadMode::All(out) => {
                let mut final_norm_all = vec![0f32; n_new * d];
                for i in 0..n_new {
                    let in_row = &hidden[i * d..(i + 1) * d];
                    let out_row = &mut final_norm_all[i * d..(i + 1) * d];
                    k::rmsnorm_f32_row(
                        in_row, &self.weights.output_norm, out_row, cfg.rms_eps,
                    );
                }
                matvec_tensor_batched_dispatch(
                    lm_head, &final_norm_all, out, cfg.vocab_size, d, n_new,
                );
            }
        }
    }

    /// Batched MXFP4-KV multi-position forward returning per-position
    /// logits — the E1 speculation primitive for MXFP4 KV. Mirrors
    /// [`Self::forward_speculation_batched_nvfp4`] modulo the 32-elem MX
    /// block geometry; shares the per-layer body with the MXFP6/MXFP8
    /// spec entries via the generic `forward_prefill_batched_mxfp_inner`
    /// (the `fmt` arg selects the block bytes + kernels).
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
        self.forward_prefill_batched_mxfp_inner(
            KvDtype::Mxfp4, tokens, start_pos, kv,
            LmHeadMode::All(logits_out),
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
        self.forward_prefill_batched_mxfp_inner(
            KvDtype::Mxfp6, tokens, start_pos, kv,
            LmHeadMode::All(logits_out),
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
        self.forward_prefill_batched_mxfp_inner(
            KvDtype::Mxfp8, tokens, start_pos, kv,
            LmHeadMode::All(logits_out),
        );
    }

    /// Shared per-layer body for the three MXFP batched-spec entries.
    /// Byte-identical to `forward_prefill_batched_nvfp4_inner` except
    /// the attention stage routes through the MXFP KV path: `fmt` picks
    /// the per-format block bytes (17/25/33) + quant/attention kernels,
    /// and the KV write + batched flash prefill (GPU→CPU) go through the
    /// shared `mxfp_kv_prefill` helper — the same helper the contiguous
    /// MXFP prefill uses, so paged/contiguous stay consistent.
    fn forward_prefill_batched_mxfp_inner(
        &self,
        fmt: KvDtype,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        mode: LmHeadMode<'_>,
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

        // MoE single-token scratch — see forward_prefill_batched_f32.
        let moe_cfg = cfg.moe.clone();
        let (mut moe_gate_one, mut moe_up_one, mut moe_ff_one, mut moe_down_one) =
            if moe_cfg.is_some() {
                (vec![0f32; d_ff], vec![0f32; d_ff], vec![0f32; d_ff], vec![0f32; d])
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };
        let (mut moe_expert_logits, mut moe_routed_picks) = if let Some(m) = moe_cfg.as_ref() {
            (vec![0f32; m.n_experts as usize], Vec::with_capacity(m.n_experts_used as usize))
        } else {
            (Vec::new(), Vec::new())
        };

        let attn_views: Vec<&dyn AttnBlock> =
            if let Some(mbs) = self.weights.moe_blocks.as_ref() {
                mbs.iter().map(|b| b as &dyn AttnBlock).collect()
            } else {
                self.weights
                    .blocks
                    .iter()
                    .map(|b| b as &dyn AttnBlock)
                    .collect()
            };

        let n_layers_used = std::env::var("RUSTLLAMA_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
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
            // Stage 1 (per-token): pre-attention rmsnorm.
            for i in 0..n_new {
                let h_row = &hidden[i * d..(i + 1) * d];
                let n_row = &mut h_norm[i * d..(i + 1) * d];
                k::rmsnorm_f32_row(h_row, block.attn_norm(), n_row, cfg.rms_eps);
            }
            // Stage 2 (batched): Q / K / V projections.
            matvec_tensor_batched_dispatch(block.w_q(), &h_norm, &mut q_buf, d_q, d, n_new);
            matvec_tensor_batched_dispatch(block.w_k(), &h_norm, &mut k_buf, d_kv, d, n_new);
            matvec_tensor_batched_dispatch(block.w_v(), &h_norm, &mut v_buf, d_kv, d, n_new);
            // Stage 3 (per-token): bias + RoPE.
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
                let pos_i = (kv_len_base + i) as u32;
                if rope_interleaved_enabled() {
                    k::rope_inplace_interleaved(q_row, n_heads, head_dim, pos_i, cfg.rope_theta);
                    k::rope_inplace_interleaved(k_row, n_kv_heads, head_dim, pos_i, cfg.rope_theta);
                } else {
                    k::rope_inplace_neox(q_row, n_heads, head_dim, pos_i, cfg.rope_theta);
                    k::rope_inplace_neox(k_row, n_kv_heads, head_dim, pos_i, cfg.rope_theta);
                }
            }

            // MXFP KV write + batched flash prefill. `mxfp_kv_prefill`
            // quantizes all n_new K/V rows per-block into the packed
            // slabs at `[kv_len_base, kv_len_base+n_new)` (layout
            // `[h, pos]` with stride `head_dim/32 × blk_bytes`), then
            // runs the GPU flash prefill or the CPU kernel — identical
            // to the NVFP4 inline block, factored into the helper the
            // contiguous MXFP prefill already shares.
            let kv_layer = &mut kv.layers[layer_idx];
            match kv_layer {
                KvLayer::Mxfp4 { k_packed, v_packed }
                | KvLayer::Mxfp6 { k_packed, v_packed }
                | KvLayer::Mxfp8 { k_packed, v_packed } => {
                    mxfp_kv_prefill(
                        fmt, &q_buf, &k_buf, &v_buf, k_packed, v_packed, &mut attn_out,
                        n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
                    );
                }
                _ => unreachable!("forward_prefill_batched_mxfp entered with non-MXFP KV"),
            }

            // Stage 4 (batched): output projection.
            matvec_tensor_batched_dispatch(block.w_o(), &attn_out, &mut attn_proj, d, d_q, n_new);
            // Stage 5 (per-token): residual + post-attn rmsnorm.
            for i in 0..n_new {
                let proj_row = &attn_proj[i * d..(i + 1) * d];
                let h_row = &mut hidden[i * d..(i + 1) * d];
                if !skip_attn_enabled() {
                    k::add_inplace_f32(h_row, proj_row);
                }
                let n_row = &mut h_norm[i * d..(i + 1) * d];
                k::rmsnorm_f32_row(h_row, block.ffn_norm(), n_row, cfg.rms_eps);
            }
            // Stage 6-8: FFN dense matvec ladder OR per-token MoE.
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
                    let g_row = &gate_buf[i * d_ff..(i + 1) * d_ff];
                    let u_row = &up_buf[i * d_ff..(i + 1) * d_ff];
                    let f_row = &mut ffn_buf[i * d_ff..(i + 1) * d_ff];
                    k::silu_mul_f32(g_row, u_row, f_row);
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
            // Stage 9 (per-token): residual.
            if !skip_ffn_enabled() {
                for i in 0..n_new {
                    let h_row = &mut hidden[i * d..(i + 1) * d];
                    let o_row = &ffn_out[i * d..(i + 1) * d];
                    k::add_inplace_f32(h_row, o_row);
                }
            }
        }
        kv.seq_len = kv.seq_len.max(kv_len_base + n_new);

        // LM-head dispatch — see `LmHeadMode` doc.
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
                    last_hidden, &self.weights.output_norm, &mut final_norm, cfg.rms_eps,
                );
                matvec_tensor_dispatch(lm_head, &final_norm, out, cfg.vocab_size, d);
            }
            LmHeadMode::All(out) => {
                let mut final_norm_all = vec![0f32; n_new * d];
                for i in 0..n_new {
                    let in_row = &hidden[i * d..(i + 1) * d];
                    let out_row = &mut final_norm_all[i * d..(i + 1) * d];
                    k::rmsnorm_f32_row(
                        in_row, &self.weights.output_norm, out_row, cfg.rms_eps,
                    );
                }
                matvec_tensor_batched_dispatch(
                    lm_head, &final_norm_all, out, cfg.vocab_size, d, n_new,
                );
            }
        }
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
        self.forward_prefill_batched_q4_0_inner(
            tokens, start_pos, kv,
            LmHeadMode::Last(&mut logits),
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
        self.forward_prefill_batched_q4_0_inner(
            tokens, start_pos, kv,
            LmHeadMode::All(logits_out),
        );
    }

    fn forward_prefill_batched_q4_0_inner(
        &self,
        tokens: &[i32],
        start_pos: u32,
        kv: &mut KvCache,
        mode: LmHeadMode<'_>,
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

        // MoE single-token scratch — see forward_prefill_batched_f32.
        let moe_cfg = cfg.moe.clone();
        let (mut moe_gate_one, mut moe_up_one, mut moe_ff_one, mut moe_down_one) =
            if moe_cfg.is_some() {
                (vec![0f32; d_ff], vec![0f32; d_ff], vec![0f32; d_ff], vec![0f32; d])
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };
        let (mut moe_expert_logits, mut moe_routed_picks) = if let Some(m) = moe_cfg.as_ref() {
            (vec![0f32; m.n_experts as usize], Vec::with_capacity(m.n_experts_used as usize))
        } else {
            (Vec::new(), Vec::new())
        };

        let attn_views: Vec<&dyn AttnBlock> =
            if let Some(mbs) = self.weights.moe_blocks.as_ref() {
                mbs.iter().map(|b| b as &dyn AttnBlock).collect()
            } else {
                self.weights
                    .blocks
                    .iter()
                    .map(|b| b as &dyn AttnBlock)
                    .collect()
            };

        let n_layers_used = std::env::var("RUSTLLAMA_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(attn_views.len())
            .min(attn_views.len());

        for (layer_idx, block) in attn_views.iter().take(n_layers_used).enumerate() {
            crate::accel::set_current_layer_idx(layer_idx as u32);
            // Stage 1 (per-token): pre-attention rmsnorm.
            for i in 0..n_new {
                let h_row = &hidden[i * d..(i + 1) * d];
                let n_row = &mut h_norm[i * d..(i + 1) * d];
                k::rmsnorm_f32_row(h_row, block.attn_norm(), n_row, cfg.rms_eps);
            }
            // Stage 2 (batched): Q / K / V projections.
            matvec_tensor_batched_dispatch(block.w_q(), &h_norm, &mut q_buf, d_q, d, n_new);
            matvec_tensor_batched_dispatch(block.w_k(), &h_norm, &mut k_buf, d_kv, d, n_new);
            matvec_tensor_batched_dispatch(block.w_v(), &h_norm, &mut v_buf, d_kv, d, n_new);
            // Stage 3 (per-token): bias + RoPE.
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
                let pos_i = (kv_len_base + i) as u32;
                if rope_interleaved_enabled() {
                    k::rope_inplace_interleaved(q_row, n_heads, head_dim, pos_i, cfg.rope_theta);
                    k::rope_inplace_interleaved(k_row, n_kv_heads, head_dim, pos_i, cfg.rope_theta);
                } else {
                    k::rope_inplace_neox(q_row, n_heads, head_dim, pos_i, cfg.rope_theta);
                    k::rope_inplace_neox(k_row, n_kv_heads, head_dim, pos_i, cfg.rope_theta);
                }
            }

            // Quantize all n_new K/V rows into the Q4_0 cache, then
            // call the batched Q4_0 prefill kernel.
            let kv_bias_l = kv.kv_bias.clone();
            let kv_layer = &mut kv.layers[layer_idx];
            match kv_layer {
                KvLayer::Q4_0 { k_q, v_q } => {
                    use rustllama_kernels_cpu::q4_0_kv;
                    use rustllama_kernels_cpu::hadamard::whiten_chunks_inplace;
                    let blocks_per_row = head_dim / q4_0_kv::Q4_0_BLOCK_ELEMS;
                    let bytes_per_row = blocks_per_row * q4_0_kv::Q4_0_BLOCK_BYTES;
                    // Whitening (fork attn_rot parity). Whole-buffer
                    // calls are safe: rows are contiguous multiples of
                    // head_dim and head_dim % 64 == 0, so no chunk
                    // crosses a head or token boundary. Output is
                    // un-rotated after the batched attention.
                    let whiten = q4_0_whiten_active(head_dim);
                    if whiten {
                        whiten_chunks_inplace(&mut q_buf, KV_WHITEN_CHUNK);
                        whiten_chunks_inplace(&mut k_buf, KV_WHITEN_CHUNK);
                        whiten_chunks_inplace(&mut v_buf, KV_WHITEN_CHUNK);
                    }
                    // Calibration observe + K bias subtract, per new
                    // row (softmax-invariant; see kv_bias docs).
                    if crate::kv_bias::calib_active() {
                        for i in 0..n_new {
                            crate::kv_bias::calib_observe(
                                layer_idx,
                                &k_buf[i * d_kv..(i + 1) * d_kv],
                            );
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
                        &q_buf,
                        k_q,
                        v_q,
                        &mut attn_out,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_len_base,
                        n_new,
                    ) {
                        q4_0_kv::gqa_attention_flash_prefill_q4_0(
                            &q_buf,
                            k_q,
                            v_q,
                            &mut attn_out,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_ctx,
                            kv_len_base,
                            n_new,
                        );
                    }
                    if whiten {
                        whiten_chunks_inplace(&mut attn_out, KV_WHITEN_CHUNK);
                    }
                }
                _ => unreachable!("forward_prefill_batched_q4_0 entered with non-Q4_0 KV"),
            }

            // Stage 4 (batched): output projection.
            matvec_tensor_batched_dispatch(block.w_o(), &attn_out, &mut attn_proj, d, d_q, n_new);
            // Stage 5 (per-token): residual + post-attn rmsnorm.
            for i in 0..n_new {
                let proj_row = &attn_proj[i * d..(i + 1) * d];
                let h_row = &mut hidden[i * d..(i + 1) * d];
                if !skip_attn_enabled() {
                    k::add_inplace_f32(h_row, proj_row);
                }
                let n_row = &mut h_norm[i * d..(i + 1) * d];
                k::rmsnorm_f32_row(h_row, block.ffn_norm(), n_row, cfg.rms_eps);
            }
            // Stage 6-8: FFN dense matvec ladder OR per-token MoE.
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
                    let g_row = &gate_buf[i * d_ff..(i + 1) * d_ff];
                    let u_row = &up_buf[i * d_ff..(i + 1) * d_ff];
                    let f_row = &mut ffn_buf[i * d_ff..(i + 1) * d_ff];
                    k::silu_mul_f32(g_row, u_row, f_row);
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
            // Stage 9 (per-token): residual.
            if !skip_ffn_enabled() {
                for i in 0..n_new {
                    let h_row = &mut hidden[i * d..(i + 1) * d];
                    let o_row = &ffn_out[i * d..(i + 1) * d];
                    k::add_inplace_f32(h_row, o_row);
                }
            }
        }
        kv.seq_len = kv.seq_len.max(kv_len_base + n_new);

        // LM-head dispatch — see `LmHeadMode` doc.
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
                    last_hidden, &self.weights.output_norm, &mut final_norm, cfg.rms_eps,
                );
                matvec_tensor_dispatch(lm_head, &final_norm, out, cfg.vocab_size, d);
            }
            LmHeadMode::All(out) => {
                let mut final_norm_all = vec![0f32; n_new * d];
                for i in 0..n_new {
                    let in_row = &hidden[i * d..(i + 1) * d];
                    let out_row = &mut final_norm_all[i * d..(i + 1) * d];
                    k::rmsnorm_f32_row(
                        in_row, &self.weights.output_norm, out_row, cfg.rms_eps,
                    );
                }
                matvec_tensor_batched_dispatch(
                    lm_head, &final_norm_all, out, cfg.vocab_size, d, n_new,
                );
            }
        }
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
        // the trait in a follow-up once `CpuEngine` is wired.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Diagnostic (V-4): on a REAL hybrid GGUF named by
    /// RUSTLLAMA_REAL_HYBRID_GGUF, the from-embeds hybrid prefill fed
    /// `embed_tokens_primal` rows must reproduce the token prefill's
    /// logits BITWISE — pinning the primal-basis contract that the
    /// VLM splice path relies on (Hadamard inverse applied exactly
    /// once).
    #[test]
    #[ignore]
    fn hybrid_from_embeds_matches_token_prefill_on_real_model() {
        let Ok(path) = std::env::var("RUSTLLAMA_REAL_HYBRID_GGUF") else {
            eprintln!("RUSTLLAMA_REAL_HYBRID_GGUF not set — skipping");
            return;
        };
        let gguf = Gguf::open(&path).expect("open gguf");
        let cfg = crate::llama_config::LlamaConfig::from_gguf(&gguf).expect("config");
        assert!(cfg.hybrid.is_some(), "fixture must be a hybrid model");
        let model = LlamaModel {
            weights: LlamaWeights::from_gguf(&gguf, &cfg).expect("weights"),
            cfg,
        };
        let hybrid = model.weights.hybrid_layers.as_ref().expect("hybrid layers");
        let tokens: Vec<i32> = vec![3, 1000, 25, 4096, 77];

        let mut kv_a = KvCache::new_with_dtype(&model.cfg, 64, KvDtype::F32);
        let mut dn_a = DeltaNetCache::new_for_hybrid(&model.cfg, hybrid).expect("dn");
        let logits_a = model.forward_prefill_hybrid(&tokens, 0, &mut kv_a, &mut dn_a);

        let mut kv_b = KvCache::new_with_dtype(&model.cfg, 64, KvDtype::F32);
        let mut dn_b = DeltaNetCache::new_for_hybrid(&model.cfg, hybrid).expect("dn");
        let rows = model.embed_tokens_primal(&tokens);
        let logits_b =
            model.forward_prefill_hybrid_from_embeds(&rows, 0, &mut kv_b, &mut dn_b);

        assert_eq!(
            logits_a, logits_b,
            "from-embeds prefill must be bitwise-identical to token prefill"
        );
        eprintln!("hybrid from-embeds parity OK ({} logits)", logits_a.len());
    }

    fn synth_cfg(n_layers: usize, n_kv_heads: usize, head_dim: usize) -> LlamaConfig {
        LlamaConfig {
            arch: "llama".into(),
            n_layers,
            n_heads: n_kv_heads,
            n_kv_heads,
            d_model: head_dim * n_kv_heads,
            d_ff: head_dim * n_kv_heads * 2,
            head_dim,
            rope_dim: head_dim,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            vocab_size: 256,
            ctx_train: 32,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: false,
            n_mtp_heads: 0,
            moe: None,
            hybrid: None,
            hadamard: None,
        }
    }

    #[test]
    fn q8_0_storage_is_roughly_one_quarter_of_f32() {
        // 4 layers × 4 kv-heads × 32 ctx × 64 head_dim = 32 768 entries
        // F32: 32k * 4 bytes * 2 (K+V) = 256 KiB per layer × 4 = 1 MiB
        // Q8_0: 32k * 1 byte (q) + 4k * 4 bytes (scales) = 32k+16k = 48 KiB,
        //   times 2 (K+V) = 96 KiB per layer × 4 = 384 KiB.
        // Ratio: 384/1024 = 0.375 — a bit over a quarter due to the
        // per-row scale overhead. Still a 2.6x reduction in storage.
        let cfg = synth_cfg(4, 4, 64);
        let max_ctx = 32;

        let f32_cache = KvCache::new(&cfg, max_ctx);
        let q8_cache = KvCache::new_with_dtype(&cfg, max_ctx, KvDtype::Q8_0);

        let f32_bytes: usize = f32_cache
            .layers
            .iter()
            .map(|l| match l {
                KvLayer::F32 { k, v } => k.len() * 4 + v.len() * 4,
                _ => unreachable!(),
            })
            .sum();
        let q8_bytes: usize = q8_cache
            .layers
            .iter()
            .map(|l| match l {
                KvLayer::Q8_0 {
                    k_q,
                    k_scales,
                    v_q,
                    v_scales,
                } => k_q.len() + k_scales.len() * 4 + v_q.len() + v_scales.len() * 4,
                _ => unreachable!(),
            })
            .sum();

        // For head_dim=64, expected ratio is ~(1 + 4/64) / 4 = 0.266.
        let ratio = q8_bytes as f64 / f32_bytes as f64;
        assert!(
            ratio < 0.30,
            "Q8_0 should be < 30% of F32 size for head_dim=64, got ratio={ratio:.3}"
        );
        assert!(
            ratio > 0.20,
            "Q8_0 shouldn't be impossibly small; got ratio={ratio:.3}"
        );
    }

    #[test]
    fn quantize_row_q8_0_dequant_round_trips_extremes() {
        let mut q = vec![0i8; 8];
        let row = vec![-3.0, -1.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
        let scale = quantize_row_q8_0(&row, &mut q);
        // The absmax in the row is 3.0; scale should be 3/127.
        let expected_scale = 3.0_f32 / 127.0;
        assert!((scale - expected_scale).abs() < 1e-6);
        // Dequantize and confirm every value is within scale/2 of the
        // original. Endpoints (-3, 3) should dequant exactly.
        let dq: Vec<f32> = q.iter().map(|&x| x as f32 * scale).collect();
        assert!((dq[0] - (-3.0)).abs() < 1e-5, "expected -3.0, got {}", dq[0]);
        assert!((dq[7] - 3.0).abs() < 1e-5, "expected 3.0, got {}", dq[7]);
        for (i, (&orig, &deq)) in row.iter().zip(dq.iter()).enumerate() {
            assert!(
                (orig - deq).abs() < scale,
                "i={i}: |{orig} - {deq}| = {} >= scale={scale}",
                (orig - deq).abs()
            );
        }
    }

    #[test]
    fn quantize_row_q8_0_handles_all_zeros() {
        let mut q = vec![99i8; 8];
        let row = vec![0.0; 8];
        let scale = quantize_row_q8_0(&row, &mut q);
        // Scale convention for the all-zero row is 1.0 (any nonzero
        // value would do; the i8 row is all-zeros either way).
        assert_eq!(scale, 1.0);
        assert!(q.iter().all(|&x| x == 0));
    }

    /// Build a tiny synthetic Llama model with deterministic F32
    /// weights. Used by the paged-vs-contiguous parity test below.
    /// All values come from a cheap hash so any silently re-ordered
    /// or skipped axis would surface as a non-trivial diff.
    fn synth_llama_model(cfg: &LlamaConfig) -> LlamaModel {
        use rustllama_tensor::{Shape, Tensor};
        fn fill(n: usize, seed: u32) -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let x = (seed as u64).wrapping_mul(1_000_003).wrapping_add(i as u64);
                    // Map u64 → small f32 in [-0.5, 0.5] with enough
                    // variation per cell to make any silent axis swap
                    // produce a different bit pattern.
                    let bits = (x ^ (x >> 17)) & 0xFFF;
                    (bits as f32 / 4096.0) - 0.5
                })
                .collect()
        }
        fn mat(name: &str, rows: u64, cols: u64, seed: u32) -> Tensor {
            let n = (rows * cols) as usize;
            Tensor::from_vec_f32(name, vec![rows, cols] as Shape, fill(n, seed))
        }
        let d = cfg.d_model as u64;
        let d_ff = cfg.d_ff as u64;
        let d_q = (cfg.n_heads * cfg.head_dim) as u64;
        let d_kv = (cfg.n_kv_heads * cfg.head_dim) as u64;
        let v = cfg.vocab_size as u64;
        let token_embd = mat("token_embd", v, d, 1);
        let output_norm: Vec<f32> = (0..d).map(|i| 1.0 + (i as f32) * 0.001).collect();
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let base = 100 + (i as u32) * 13;
            let attn_norm: Vec<f32> =
                (0..d).map(|j| 1.0 + ((i as u32 + j as u32) % 7) as f32 * 0.01).collect();
            let ffn_norm: Vec<f32> =
                (0..d).map(|j| 1.0 + ((i as u32 + j as u32) % 5) as f32 * 0.01).collect();
            blocks.push(LlamaBlockWeights {
                attn_norm,
                w_q: mat("w_q", d_q, d, base),
                w_k: mat("w_k", d_kv, d, base + 1),
                w_v: mat("w_v", d_kv, d, base + 2),
                w_o: mat("w_o", d, d_q, base + 3),
                b_q: None,
                b_k: None,
                b_v: None,
                ffn_norm,
                w_gate: mat("w_gate", d_ff, d, base + 4),
                w_up: mat("w_up", d_ff, d, base + 5),
                w_down: mat("w_down", d, d_ff, base + 6),
                w_qkv_fused: None,
                q_norm: None,
                k_norm: None,
            });
        }
        LlamaModel {
            cfg: cfg.clone(),
            weights: LlamaWeights {
                token_embd,
                blocks,
                moe_blocks: None,
                hybrid_layers: None,
                output_norm,
                output: None, // tie embeddings
                mtp_heads: None,
                nextn_head: None,
                hadamard: None,
            },
        }
    }

    /// Parity probe: `forward_prefill_paged_f32` returns bit-
    /// identical logits to `forward_prefill_batched_f32` for the
    /// same model + tokens. Proves the gather path's data layout
    /// agrees with the contiguous slab the existing attention
    /// kernels expect — any axis swap inside `PagedKvCache::write_token`
    /// or `gather_layer` would surface as a non-zero diff somewhere
    /// in the 256-wide logits vector.
    ///
    /// Runs in mock mode (no SYCL); both paths fall through to the
    /// CPU `gqa_attention_flash_prefill`, so the bit-for-bit
    /// equality is the strict assertion the test wants.
    #[test]
    fn forward_prefill_paged_matches_contiguous_for_same_input() {
        // Small model: 2 layers, 2 heads (no GQA grouping for v1
        // test simplicity — covered separately by the page store's
        // gather tests, which exercise n_kv_heads=3 too).
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        let tokens: Vec<i32> = vec![1, 4, 17, 9, 33];
        let n_new = tokens.len();
        let max_ctx = 16;

        // Contiguous path — uses the existing F32 KvCache layout.
        let mut contig_kv = KvCache::new(&cfg, max_ctx);
        let logits_contig = model.forward_prefill_batched_f32(&tokens, 0, &mut contig_kv);

        // Paged path — fresh store + cache, grown to fit n_new
        // tokens for one slot.
        let page_size = 4;
        let pages_for_kv = (n_new as u32).div_ceil(page_size);
        let mut store = crate::paged_kv_store::PagedKvStore::new(
            pages_for_kv,
            cfg.n_layers as u32,
            cfg.n_kv_heads as u32,
            page_size,
            cfg.head_dim as u32,
        )
        .expect("paged store");
        let mut table = crate::page_table::PageTable::new(pages_for_kv, page_size);
        let mut paged_cache = crate::paged_kv_cache::PagedKvCache::new_for(&store);
        paged_cache
            .ensure_capacity(&mut table, n_new as u32)
            .expect("ensure_capacity");
        let logits_paged =
            model.forward_prefill_paged_f32(&tokens, 0, &mut paged_cache, &mut store);

        // Bit-identical: same inputs, same kernels — gather just
        // routes the data through a different storage layout.
        assert_eq!(logits_contig.len(), logits_paged.len(), "logits length");
        for (i, (&a, &b)) in logits_contig.iter().zip(logits_paged.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "logit[{i}] diverges: contiguous={a} paged={b}"
            );
        }
    }

    /// Parity probe across prefill + multiple decode steps:
    /// `forward_one_paged_f32` produces bit-identical logits to
    /// `forward_one` (F32 path) at every decode step after a paged
    /// prefill. This exercises the full single-request generation
    /// flow through paged storage — write+gather correctness
    /// across both the batched-prefill code path AND the per-step
    /// decode code path, with the cache growing as more tokens
    /// land.
    #[test]
    fn forward_one_paged_matches_contiguous_across_decode_steps() {
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        let prompt: Vec<i32> = vec![3, 1, 4, 1, 5];
        let n_prompt = prompt.len();
        let n_decode = 4; // generate 4 more tokens after prefill
        let total = n_prompt + n_decode;
        let max_ctx = 16;
        let mut decode_inputs: Vec<i32> = vec![7, 11, 13, 17];

        // Contiguous reference: prefill + n_decode forward_ones.
        let mut contig_kv = KvCache::new(&cfg, max_ctx);
        let _ = model.forward_prefill_batched_f32(&prompt, 0, &mut contig_kv);
        let mut contig_decode_logits: Vec<Vec<f32>> = Vec::with_capacity(n_decode);
        for (i, &tok) in decode_inputs.iter().enumerate() {
            let mut logits = vec![0f32; cfg.vocab_size];
            model.forward_one(tok, (n_prompt + i) as u32, &mut contig_kv, &mut logits);
            contig_decode_logits.push(logits);
        }

        // Paged path: same model, same inputs.
        let page_size = 4;
        let pages_needed = (total as u32).div_ceil(page_size);
        let mut store = crate::paged_kv_store::PagedKvStore::new(
            pages_needed,
            cfg.n_layers as u32,
            cfg.n_kv_heads as u32,
            page_size,
            cfg.head_dim as u32,
        )
        .expect("paged store");
        let mut table = crate::page_table::PageTable::new(pages_needed, page_size);
        let mut paged_cache = crate::paged_kv_cache::PagedKvCache::new_for(&store);
        paged_cache
            .ensure_capacity(&mut table, total as u32)
            .expect("ensure_capacity");
        let _ = model.forward_prefill_paged_f32(&prompt, 0, &mut paged_cache, &mut store);
        let mut paged_decode_logits: Vec<Vec<f32>> = Vec::with_capacity(n_decode);
        for (i, &tok) in decode_inputs.iter().enumerate() {
            let mut logits = vec![0f32; cfg.vocab_size];
            model.forward_one_paged_f32(
                tok,
                (n_prompt + i) as u32,
                &mut paged_cache,
                &mut store,
                &mut logits,
            );
            paged_decode_logits.push(logits);
        }

        // Bit-identical at every decode step.
        for (step, (a_logits, b_logits)) in contig_decode_logits
            .iter()
            .zip(paged_decode_logits.iter())
            .enumerate()
        {
            for (i, (&a, &b)) in a_logits.iter().zip(b_logits.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "step {step} logit[{i}] diverges: contiguous={a} paged={b}"
                );
            }
        }

        // Borrow checker silencer: make sure we used decode_inputs
        // after the parity assertions so the test's intent is clear.
        decode_inputs.clear();
        assert!(decode_inputs.is_empty());
    }

    /// Parity probe: one fused M-slot decode call produces bit-
    /// identical logits to M independent `forward_one_paged_f32`
    /// calls (one per slot). Each slot is independently prefilled
    /// with a different prompt so the test exercises the actual
    /// "M slots with different KV histories" case, not just the
    /// trivial "same prompt M times" case.
    ///
    /// This is the milestone test for Item 3.7c: proves the
    /// batched-matmul path doesn't cross-contaminate slot state,
    /// and that the per-slot serial attention reads the right
    /// slot's KV history from the shared store.
    #[test]
    fn forward_decode_paged_batched_matches_serial_forward_one_paged() {
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        // Two slots with different prompts so their KV state
        // diverges before the decode step under test.
        let prompt_a: Vec<i32> = vec![5, 1, 4, 2];
        let prompt_b: Vec<i32> = vec![8, 0, 3, 7];
        assert_eq!(prompt_a.len(), prompt_b.len(), "prompts equal length to share scratch");
        let n_prompt = prompt_a.len();
        let total_capacity = (n_prompt + 1) as u32; // prefill + 1 decode step

        // ---- Reference path: two independent single-owner caches ----
        // Each slot prefills + decodes one token through
        // `forward_one_paged_f32` (no batching, no sharing). The
        // logits this produces are the ground truth the fused
        // call must match.
        let page_size = 4;
        let pages_per_cache = total_capacity.div_ceil(page_size);
        let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(2);
        for prompt in [&prompt_a, &prompt_b] {
            let mut store = crate::paged_kv_store::PagedKvStore::new(
                pages_per_cache,
                cfg.n_layers as u32,
                cfg.n_kv_heads as u32,
                page_size,
                cfg.head_dim as u32,
            )
            .expect("ref store");
            let mut table = crate::page_table::PageTable::new(pages_per_cache, page_size);
            let mut cache = crate::paged_kv_cache::PagedKvCache::new_for(&store);
            cache
                .ensure_capacity(&mut table, total_capacity)
                .expect("ensure_capacity");
            let _ = model.forward_prefill_paged_f32(prompt, 0, &mut cache, &mut store);
            // One decode step: the last token of the prompt is
            // the "next input" the sampler would have produced.
            // We use prompt[n_prompt-1] for determinism in the
            // test; the engine's real generate loop samples
            // here.
            let mut logits = vec![0f32; cfg.vocab_size];
            model.forward_one_paged_f32(
                prompt[n_prompt - 1],
                n_prompt as u32,
                &mut cache,
                &mut store,
                &mut logits,
            );
            ref_logits.push(logits);
        }

        // ---- Fused path: shared store, two PagedKvCaches, one
        // forward_decode_paged_batched_f32 call ----
        // Build a shared store sized to hold both slots'
        // capacity. `pages_per_cache * 2` ensures the table can
        // satisfy both ensure_capacity_shared calls without
        // contention.
        let total_pages_shared = pages_per_cache * 2;
        let store = crate::paged_kv_store::PagedKvStore::new(
            total_pages_shared,
            cfg.n_layers as u32,
            cfg.n_kv_heads as u32,
            page_size,
            cfg.head_dim as u32,
        )
        .expect("shared store");
        let table = crate::page_table::PageTable::new(total_pages_shared, page_size);
        let shared = crate::shared_paged_kv::SharedPagedKv::new(store, table);

        // Per-slot caches that all point at `shared`. Build a
        // throwaway PagedKvStore just for the geometry borrow
        // `PagedKvCache::new_for` needs — it only reads page_size,
        // so the throwaway is cheap (4 pages × 2 layers × 2 heads
        // × 4 page_size × 8 head_dim = 6 KiB).
        let geom_store = crate::paged_kv_store::PagedKvStore::new(
            total_pages_shared,
            cfg.n_layers as u32,
            cfg.n_kv_heads as u32,
            page_size,
            cfg.head_dim as u32,
        )
        .unwrap();
        let mut cache_a = crate::paged_kv_cache::PagedKvCache::new_for(&geom_store);
        let mut cache_b = crate::paged_kv_cache::PagedKvCache::new_for(&geom_store);
        cache_a
            .ensure_capacity_shared(&shared, total_capacity)
            .expect("ensure A");
        cache_b
            .ensure_capacity_shared(&shared, total_capacity)
            .expect("ensure B");

        // Prefill both slots through the shared store using the
        // shared-access methods directly. `forward_prefill_paged_f32`
        // only takes single-owner `&mut PagedKvStore`, so we
        // bridge by locking the shared store for the entire
        // prefill of each slot. Prefill is per-slot serial here —
        // that's fine, this is test setup, not the path under
        // test.
        for (prompt, cache) in [(&prompt_a, &mut cache_a), (&prompt_b, &mut cache_b)] {
            // The shared store's with_store_mut helper hands us a
            // &mut PagedKvStore for the duration of the closure.
            shared.with_store_mut(|store| {
                let _ = model.forward_prefill_paged_f32(prompt, 0, cache, store);
            });
        }

        // The decode step under test: two slots, one fused call.
        let mut logits_a = vec![0f32; cfg.vocab_size];
        let mut logits_b = vec![0f32; cfg.vocab_size];
        let mut slots = vec![
            crate::llama_arch::DecodeSlot {
                token_id: prompt_a[n_prompt - 1],
                pos: n_prompt as u32,
                cache: &mut cache_a,
                logits_out: &mut logits_a,
            },
            crate::llama_arch::DecodeSlot {
                token_id: prompt_b[n_prompt - 1],
                pos: n_prompt as u32,
                cache: &mut cache_b,
                logits_out: &mut logits_b,
            },
        ];
        model.forward_decode_paged_batched_f32(&mut slots, &shared);
        drop(slots);

        // Bit-identical per-slot.
        let fused_logits = [&logits_a, &logits_b];
        for (slot_idx, (&fused, reference)) in
            fused_logits.iter().zip(ref_logits.iter()).enumerate()
        {
            for (i, (&a, &b)) in fused.iter().zip(reference.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "slot {slot_idx} logit[{i}] diverges: fused={a} ref={b}"
                );
            }
        }
    }

    /// Single-slot edge: `forward_decode_paged_batched_f32` with
    /// `m=1` produces bit-identical output to a single
    /// `forward_one_paged_f32` call. Catches off-by-one bugs in
    /// the batched-scratch sizing (size depends on `m`) that
    /// wouldn't show up in the multi-slot case.
    #[test]
    fn forward_decode_paged_batched_single_slot_matches_forward_one_paged() {
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        let prompt: Vec<i32> = vec![3, 14, 1, 5];
        let n_prompt = prompt.len();
        let total_capacity = (n_prompt + 1) as u32;
        let page_size = 4;
        let pages = total_capacity.div_ceil(page_size);

        // Reference via single-owner forward_one_paged_f32.
        let mut ref_store = crate::paged_kv_store::PagedKvStore::new(
            pages,
            cfg.n_layers as u32,
            cfg.n_kv_heads as u32,
            page_size,
            cfg.head_dim as u32,
        )
        .unwrap();
        let mut ref_table = crate::page_table::PageTable::new(pages, page_size);
        let mut ref_cache = crate::paged_kv_cache::PagedKvCache::new_for(&ref_store);
        ref_cache.ensure_capacity(&mut ref_table, total_capacity).unwrap();
        let _ = model.forward_prefill_paged_f32(&prompt, 0, &mut ref_cache, &mut ref_store);
        let mut ref_logits = vec![0f32; cfg.vocab_size];
        model.forward_one_paged_f32(
            prompt[n_prompt - 1],
            n_prompt as u32,
            &mut ref_cache,
            &mut ref_store,
            &mut ref_logits,
        );

        // Fused single-slot path.
        let shared_store = crate::paged_kv_store::PagedKvStore::new(
            pages,
            cfg.n_layers as u32,
            cfg.n_kv_heads as u32,
            page_size,
            cfg.head_dim as u32,
        )
        .unwrap();
        let shared_table = crate::page_table::PageTable::new(pages, page_size);
        let shared = crate::shared_paged_kv::SharedPagedKv::new(shared_store, shared_table);
        let geom_store = crate::paged_kv_store::PagedKvStore::new(
            pages,
            cfg.n_layers as u32,
            cfg.n_kv_heads as u32,
            page_size,
            cfg.head_dim as u32,
        )
        .unwrap();
        let mut cache = crate::paged_kv_cache::PagedKvCache::new_for(&geom_store);
        cache
            .ensure_capacity_shared(&shared, total_capacity)
            .unwrap();
        shared.with_store_mut(|store| {
            let _ = model.forward_prefill_paged_f32(&prompt, 0, &mut cache, store);
        });
        let mut fused_logits = vec![0f32; cfg.vocab_size];
        let mut slots = vec![crate::llama_arch::DecodeSlot {
            token_id: prompt[n_prompt - 1],
            pos: n_prompt as u32,
            cache: &mut cache,
            logits_out: &mut fused_logits,
        }];
        model.forward_decode_paged_batched_f32(&mut slots, &shared);
        drop(slots);

        for (i, (&a, &b)) in fused_logits.iter().zip(ref_logits.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "single-slot fused logit[{i}] diverges: fused={a} ref={b}"
            );
        }
    }

    /// Same parity probe at a non-zero `start_pos`: the paged path
    /// must also handle the case where the cache already holds
    /// previously-prefilled tokens and the new batch extends it.
    /// Catches off-by-one bugs in `kv_len_base` / `seq_len`
    /// handling that wouldn't show up in the start_pos=0 case.
    #[test]
    fn forward_prefill_paged_matches_contiguous_at_nonzero_start_pos() {
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        let first_batch: Vec<i32> = vec![5, 11, 2];
        let second_batch: Vec<i32> = vec![19, 7];
        let total = first_batch.len() + second_batch.len();
        let max_ctx = 16;

        let mut contig_kv = KvCache::new(&cfg, max_ctx);
        let _ = model.forward_prefill_batched_f32(&first_batch, 0, &mut contig_kv);
        let logits_contig = model.forward_prefill_batched_f32(
            &second_batch,
            first_batch.len() as u32,
            &mut contig_kv,
        );

        let page_size = 4;
        let pages_needed = (total as u32).div_ceil(page_size);
        let mut store = crate::paged_kv_store::PagedKvStore::new(
            pages_needed,
            cfg.n_layers as u32,
            cfg.n_kv_heads as u32,
            page_size,
            cfg.head_dim as u32,
        )
        .expect("paged store");
        let mut table = crate::page_table::PageTable::new(pages_needed, page_size);
        let mut paged_cache = crate::paged_kv_cache::PagedKvCache::new_for(&store);
        paged_cache
            .ensure_capacity(&mut table, total as u32)
            .expect("ensure_capacity");
        let _ = model.forward_prefill_paged_f32(&first_batch, 0, &mut paged_cache, &mut store);
        let logits_paged = model.forward_prefill_paged_f32(
            &second_batch,
            first_batch.len() as u32,
            &mut paged_cache,
            &mut store,
        );

        for (i, (&a, &b)) in logits_contig.iter().zip(logits_paged.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "logit[{i}] diverges at start_pos>0: contiguous={a} paged={b}"
            );
        }
    }

    /// Hybrid placement parity: the forward output is invariant to
    /// `n_gpu_layers` in mock mode. Both `n_gpu_layers=u32::MAX`
    /// (all GPU, all USM helpers run their full body) and
    /// `n_gpu_layers=0` (all CPU, every helper short-circuits and
    /// falls through) must produce bit-identical logits, because
    /// in mock mode the GPU helpers also return false (no SYCL
    /// device) → they fall through to the same CPU kernels. This
    /// pins the contract that the placement gate is a *routing*
    /// decision, not a numerical one.
    ///
    /// Real-hardware perf wins from the cutoff land independently
    /// — they don't change correctness.
    #[test]
    fn forward_one_invariant_to_n_gpu_layers_in_mock_mode() {
        let cfg = synth_cfg(3, 2, 8);
        let model = synth_llama_model(&cfg);
        let prompt: Vec<i32> = vec![2, 9, 4, 1];
        let max_ctx = 16;

        // All-CPU run: n_gpu_layers = 0.
        crate::accel::set_n_gpu_layers(0);
        let mut kv = KvCache::new(&cfg, max_ctx);
        let logits_cpu = model.forward_prefill_batched_f32(&prompt, 0, &mut kv);

        // All-GPU run (well, all "would be GPU if a device existed"):
        // in mock mode the same CPU fallbacks fire.
        crate::accel::set_n_gpu_layers(u32::MAX);
        let mut kv = KvCache::new(&cfg, max_ctx);
        let logits_gpu = model.forward_prefill_batched_f32(&prompt, 0, &mut kv);

        // Half-and-half: 1 GPU layer, 2 CPU layers.
        crate::accel::set_n_gpu_layers(1);
        let mut kv = KvCache::new(&cfg, max_ctx);
        let logits_hybrid = model.forward_prefill_batched_f32(&prompt, 0, &mut kv);

        for (i, (&a, &b)) in logits_cpu.iter().zip(logits_gpu.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "logit[{i}] differs between n_gpu_layers=0 and n_gpu_layers=MAX",
            );
        }
        for (i, (&a, &b)) in logits_cpu.iter().zip(logits_hybrid.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "logit[{i}] differs between n_gpu_layers=0 and n_gpu_layers=1",
            );
        }
        // Restore TLS for any sibling tests on this thread.
        crate::accel::set_n_gpu_layers(u32::MAX);
        crate::accel::set_current_layer_idx(0);
    }

    /// `preload_packed_weights_to_usm_with_cutoff` returns a
    /// consistent total tensor count across every cutoff: in mock
    /// mode every tensor reports as "skipped" (no real USM
    /// upload happens), so the function should report the same
    /// `uploaded + skipped` total regardless of cutoff. The
    /// **real**-hardware test for "cutoff actually shrinks the
    /// upload count" requires SYCL and lives in `real_model.rs`-
    /// style gated tests — that's the perf-measurement side and
    /// not what this unit test asserts. What this test guards is
    /// that the cutoff branch doesn't double-count or miss a
    /// tensor.
    #[test]
    fn preload_cutoff_keeps_total_tensor_count_consistent() {
        let cfg = synth_cfg(4, 2, 8); // 4 transformer blocks
        let model = synth_llama_model(&cfg);
        // 4 blocks × 7 per-block tensors + token_embd = 29 tensors.
        // (Tied output weights — no separate `output` tensor.)
        const EXPECTED_TOTAL: usize = 4 * 7 + 1;
        for cutoff in [0u32, 1, 2, 3, 4, 999, u32::MAX] {
            let (uploaded, skipped, _) =
                model.preload_packed_weights_to_usm_with_cutoff(cutoff);
            assert_eq!(
                uploaded + skipped,
                EXPECTED_TOTAL,
                "cutoff={cutoff}: uploaded={uploaded} + skipped={skipped} != expected {EXPECTED_TOTAL}",
            );
        }
        // And the default `preload_packed_weights_to_usm()`
        // behaves identically to `with_cutoff(u32::MAX)`.
        let (a_up, a_sk, _) = model.preload_packed_weights_to_usm();
        let (b_up, b_sk, _) = model.preload_packed_weights_to_usm_with_cutoff(u32::MAX);
        assert_eq!(a_up, b_up);
        assert_eq!(a_sk, b_sk);
    }

    // ---- V-6b-3a: embed_tokens public surface --------------------

    /// `embed_tokens` returns `[ids.len(), d_model]` row-major f32 —
    /// pin the contract the V-5 splice depends on.
    #[test]
    fn embed_tokens_returns_d_model_row_per_id() {
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        let ids: Vec<i32> = vec![3, 7, 2, 11];
        let out = model.embed_tokens(&ids);
        assert_eq!(out.len(), ids.len() * cfg.d_model);
    }

    /// Empty input yields empty output — no kernel call, no panic.
    /// A VLM-aware engine can call this unconditionally for every
    /// chat request (text-only requests just produce nothing).
    #[test]
    fn embed_tokens_empty_input_returns_empty() {
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        let out = model.embed_tokens(&[]);
        assert!(out.is_empty());
    }

    /// Each row matches the same token's per-id lookup. Guarantees
    /// the batched dispatch produces the same rows as the per-token
    /// path the transformer body uses internally.
    #[test]
    fn embed_tokens_rows_match_per_token_lookup() {
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        let d = cfg.d_model;
        let ids: Vec<i32> = vec![5, 100, 0, 42];
        let batched = model.embed_tokens(&ids);
        for (i, &id) in ids.iter().enumerate() {
            let single = model.embed_tokens(&[id]);
            assert_eq!(single.len(), d);
            assert_eq!(
                &batched[i * d..(i + 1) * d],
                &single[..],
                "row {i} (token {id}) diverges from per-token lookup",
            );
        }
    }

    /// Distinct token ids must yield distinct rows (otherwise the
    /// splice loses information when an image-placeholder embedding
    /// is silently identical to a text token).
    #[test]
    fn embed_tokens_distinct_ids_yield_distinct_rows() {
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        let d = cfg.d_model;
        let out_a = model.embed_tokens(&[3]);
        let out_b = model.embed_tokens(&[7]);
        assert_eq!(out_a.len(), d);
        assert_eq!(out_b.len(), d);
        assert_ne!(out_a, out_b, "embedding table rows should differ per id");
    }

    /// `embed_tokens` is deterministic: calling twice with the same
    /// ids produces bit-identical output. (No hidden RNG, no thread-
    /// local state that mutates across calls.)
    #[test]
    fn embed_tokens_is_deterministic() {
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        let ids: Vec<i32> = vec![1, 2, 3, 4];
        let a = model.embed_tokens(&ids);
        let b = model.embed_tokens(&ids);
        assert_eq!(a, b);
    }

    // ---- V-6b-3b: forward_one_from_embed / forward_prefill_from_embeds ----

    /// `forward_one_from_embed(embed_tokens(&[id])[..])` produces
    /// bit-identical logits to `forward_one(id, ...)`. This is the
    /// core correctness claim of V-6b-3b: feeding a pre-embedded row
    /// through the splice path matches the token-id path for the
    /// no-image case, so V-5's splice can mix text and image rows
    /// without bias for text tokens.
    #[test]
    fn forward_one_from_embed_matches_forward_one_for_text_token() {
        let cfg = synth_cfg(3, 2, 8);
        let model = synth_llama_model(&cfg);
        let token: i32 = 17;
        let max_ctx = 8;

        let mut kv_tok = KvCache::new(&cfg, max_ctx);
        let mut logits_tok = vec![0.0f32; cfg.vocab_size];
        model.forward_one(token, 0, &mut kv_tok, &mut logits_tok);

        let embed = model.embed_tokens(&[token]);
        let mut kv_emb = KvCache::new(&cfg, max_ctx);
        let mut logits_emb = vec![0.0f32; cfg.vocab_size];
        model.forward_one_from_embed(&embed, 0, &mut kv_emb, &mut logits_emb);

        for (i, (&a, &b)) in logits_tok.iter().zip(logits_emb.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "logit[{i}] differs between forward_one and forward_one_from_embed",
            );
        }
    }

    /// Prefill parity: feeding the same prompt either as token ids
    /// (forward_prefill) or as pre-embedded rows (forward_prefill_from_embeds)
    /// must produce the same final logits.
    #[test]
    fn forward_prefill_from_embeds_matches_forward_prefill_for_text_prompt() {
        let cfg = synth_cfg(3, 2, 8);
        let model = synth_llama_model(&cfg);
        let prompt: Vec<i32> = vec![3, 14, 1, 5, 9];
        let max_ctx = 16;

        let mut kv_tok = KvCache::new(&cfg, max_ctx);
        let logits_tok = model.forward_prefill(&prompt, 0, &mut kv_tok);

        let embeds = model.embed_tokens(&prompt);
        let mut kv_emb = KvCache::new(&cfg, max_ctx);
        let logits_emb = model.forward_prefill_from_embeds(&embeds, 0, &mut kv_emb);

        for (i, (&a, &b)) in logits_tok.iter().zip(logits_emb.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "logit[{i}] differs between forward_prefill and forward_prefill_from_embeds",
            );
        }
    }

    /// DeltaNetCache: SSM layers get sized buffers; full-attention
    /// layers get empty sentinels; layer count matches `cfg.n_layers`.
    /// Pins the Phase 3.7a contract.
    #[test]
    fn deltanet_cache_sizes_ssm_layers_only() {
        // Build a tiny synth model that's `qwen35moe`-shaped: 4 layers
        // with full_attention_interval=2 → layer 1 and layer 3 are
        // full attention; layers 0 and 2 are SSM. We bypass the GGUF
        // path by constructing config + a stub `hybrid_layers` vec
        // directly — the cache only inspects shapes, not weights.
        let mut cfg = synth_cfg(4, 2, 4);
        cfg.hybrid = Some(crate::llama_config::HybridConfig {
            full_attention_interval: 2,
            ssm_state_size: 16,
            ssm_conv_kernel: 4,
            ssm_group_count: 1,
            ssm_time_step_rank: 2,
            ssm_inner_size: 16, // 2 V heads × 8
            shared_expert_feed_forward_length: None,
            nextn_predict_layers: 0,
        });
        let n_v_heads = 2_usize;
        // Build stub HybridLayer entries. SSM layers need `ssm_a` of
        // length n_v_heads for the cache's head-count derivation.
        let mk_dense_ffn = || HybridFfn::Dense {
            w_gate: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            w_up: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            w_down: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
        };
        let mk_ssm = || HybridLayer::Ssm(SsmBlockWeights {
            attn_norm: vec![],
            attn_qkv: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            attn_gate: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            post_attention_norm: vec![],
            // The cache derives the conv channel count (= fused-QKV
            // width) from this tensor's last dim: kernel 4 over
            // 2·ssm_inner = 32 channels for the legacy-shaped stub.
            ssm_conv1d: Tensor::zeros_cpu(Dtype::F32, vec![4u64, 32u64]),
            ssm_conv1d_f32: vec![],
            ssm_dt_bias: vec![],
            ssm_a: vec![0.0; n_v_heads],
            ssm_alpha: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            ssm_beta: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            ssm_norm: vec![],
            ssm_out: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            ffn: mk_dense_ffn(),
        });
        let mk_attn = || HybridLayer::FullAttention(HybridAttnBlockWeights {
            attn_norm: vec![],
            w_q: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            w_k: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            w_v: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            w_o: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            q_norm: None,
            k_norm: None,
            post_attention_norm: vec![],
            ffn: mk_dense_ffn(),
        });
        // Layers 0, 2 = SSM; layers 1, 3 = full attention (per
        // (i + 1) % 2 == 0 → layers 1, 3).
        let hybrid_layers = vec![mk_ssm(), mk_attn(), mk_ssm(), mk_attn()];
        let cache = DeltaNetCache::new_for_hybrid(&cfg, &hybrid_layers)
            .expect("hybrid config present");
        assert_eq!(cache.layers.len(), 4, "one entry per layer");
        // SSM layers (0, 2) have sized buffers.
        for &i in &[0_usize, 2] {
            let l = &cache.layers[i];
            assert!(!l.is_empty(), "SSM layer {i} should have allocated state");
            // conv_state size: (kernel-1) * 2*ssm_inner = 3 * 32 = 96
            assert_eq!(l.conv_state.len(), 3 * 32);
            // recurrent_state: n_v_heads * head_qk_dim * head_v_dim
            // = 2 * 8 * 8 = 128
            assert_eq!(l.recurrent_state.len(), 128);
        }
        // Full-attention layers (1, 3) carry empty sentinels.
        for &i in &[1_usize, 3] {
            let l = &cache.layers[i];
            assert!(l.is_empty(), "full-attention layer {i} should be empty sentinel");
        }
    }

    /// DeltaNetCache::reset zeros all buffers.
    #[test]
    fn deltanet_cache_reset_zeros_state() {
        // First param is the fused-QKV width (32 = 2·ssm_inner for
        // this legacy-shaped fixture), not ssm_inner.
        let mut state = DeltaNetLayerState::new_for_ssm(32, 2, 8, 8, 4);
        state.conv_state[0] = 1.0;
        state.recurrent_state[5] = 2.5;
        state.reset();
        assert!(state.conv_state.iter().all(|&v| v == 0.0));
        assert!(state.recurrent_state.iter().all(|&v| v == 0.0));
    }

    /// DeltaNetCache::new_for_hybrid errors on a non-hybrid cfg. The
    /// "non-hybrid → no cache" guard now lives in the engine's
    /// `build_delta_net_cache` (it checks the weights first and never
    /// calls this on a dense model), so reaching here without a [hybrid]
    /// config block is an inconsistency — surfaced as a clean error
    /// instead of a panic.
    #[test]
    fn deltanet_cache_errors_for_non_hybrid() {
        let cfg = synth_cfg(2, 2, 8);
        assert!(cfg.hybrid.is_none());
        let cache = DeltaNetCache::new_for_hybrid(&cfg, &[]);
        assert!(cache.is_err());
    }

    /// Misshapen embed buffer panics (debug build) or errors clearly.
    /// Pins the contract that the splice output length must be a
    /// multiple of d_model.
    #[test]
    #[should_panic(expected = "not a multiple of d_model")]
    fn forward_prefill_from_embeds_rejects_misshapen_buffer() {
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        let max_ctx = 8;
        let mut kv = KvCache::new(&cfg, max_ctx);
        // d_model = 16; 17 floats is not a multiple of 16.
        let bad = vec![0.0f32; 17];
        let _ = model.forward_prefill_from_embeds(&bad, 0, &mut kv);
    }

    /// Empty embedding buffer yields a zero-logit vector — same as
    /// calling forward_prefill with an empty token slice. Lets the
    /// caller unconditionally route through the embed path for the
    /// empty-prompt edge case without special-casing.
    #[test]
    fn forward_prefill_from_embeds_empty_input_returns_zero_logits() {
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        let max_ctx = 8;
        let mut kv = KvCache::new(&cfg, max_ctx);
        let logits = model.forward_prefill_from_embeds(&[], 0, &mut kv);
        assert_eq!(logits.len(), cfg.vocab_size);
        assert!(logits.iter().all(|&x| x == 0.0));
    }

    /// Pin the contract for [`LlamaModel::forward_speculation`]:
    /// per-position logits must be **bit-identical** to a manual
    /// loop of `forward_one` calls. The serial-verify path's
    /// implementation IS the loop, so this is mostly a smoke test
    /// for the API shape — but the bit-identical assertion pins
    /// the baseline so future batched implementations (the actual
    /// perf win for D1) have a strict parity gate.
    #[test]
    fn forward_speculation_matches_forward_one_loop() {
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        let tokens: Vec<i32> = vec![5, 7, 2, 14, 3];
        let max_ctx = 16;
        let vocab = cfg.vocab_size;

        // Baseline: drive forward_one in a manual loop.
        let mut kv_ref = KvCache::new(&cfg, max_ctx);
        let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
        for (i, &t) in tokens.iter().enumerate() {
            let mut l = vec![0f32; vocab];
            model.forward_one(t, i as u32, &mut kv_ref, &mut l);
            ref_logits.push(l);
        }

        // forward_speculation should produce the concatenation.
        let mut kv_spec = KvCache::new(&cfg, max_ctx);
        let mut spec_logits = vec![0f32; tokens.len() * vocab];
        model.forward_speculation(&tokens, 0, &mut kv_spec, &mut spec_logits);

        for (i, ref_row) in ref_logits.iter().enumerate() {
            let lo = i * vocab;
            let hi = lo + vocab;
            for (j, (&a, &b)) in spec_logits[lo..hi].iter().zip(ref_row.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "forward_speculation logit[{i},{j}] diverges from forward_one loop: \
                     spec={a} ref={b}"
                );
            }
        }
        // KV state should also match (both wrote `tokens.len()` positions
        // sequentially, in the same order).
        assert_eq!(kv_ref.seq_len, kv_spec.seq_len);
    }

    /// After `forward_speculation` populates K+1 positions, dropping
    /// `kv.seq_len` back (the rewind primitive that `KvBackend::
    /// set_seq_len` exposes in the engine crate) must leave the
    /// cache in a state where a subsequent `forward_one` at the
    /// rewound position produces logits identical to walking the
    /// same prefix on a fresh cache — i.e., the K/V rows past the
    /// watermark are correctly invisible to attention.
    /// E1: per-position logits from the batched F32-KV speculation
    /// method must match the serial-loop primitive within numerical
    /// noise. They are NOT bit-identical because the serial path
    /// computes attention per-position via the regular GQA kernel
    /// while the batched path uses `gqa_attention_flash_prefill`
    /// which has a different floating-point reduction order — same
    /// math, different accumulation. The end-to-end sampler is
    /// invariant to ≤ 1-ULP logit differences, so the tolerance
    /// (1e-3 abs, conservative) is well above the worst-case
    /// reduction-order drift on a 2-layer synthetic model.
    #[test]
    fn forward_speculation_batched_f32_matches_serial_for_same_input() {
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        let tokens: Vec<i32> = vec![5, 7, 2, 14, 3];
        let max_ctx = 16;
        let vocab = cfg.vocab_size;

        // Baseline: serial `forward_speculation` (driven by
        // forward_one per position).
        let mut kv_ref = KvCache::new(&cfg, max_ctx);
        let mut ref_logits = vec![0f32; tokens.len() * vocab];
        model.forward_speculation(&tokens, 0, &mut kv_ref, &mut ref_logits);

        // New: batched F32-KV speculation. Shares the per-layer
        // body with forward_prefill_batched_f32.
        let mut kv_batched = KvCache::new(&cfg, max_ctx);
        let mut batched_logits = vec![0f32; tokens.len() * vocab];
        model.forward_speculation_batched_f32(
            &tokens, 0, &mut kv_batched, &mut batched_logits,
        );

        let mut max_diff = 0f32;
        for (i, (&a, &b)) in batched_logits.iter().zip(ref_logits.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(
                d < 1e-3,
                "forward_speculation_batched_f32 logit[{i}] drift {d}: \
                 batched={a} serial={b}"
            );
        }
        // Document the observed drift so a future tightening of
        // the tolerance has a baseline. On the synthetic 2-layer
        // model the drift is < 1e-5 in practice.
        let _ = max_diff;
        // KV state matches: same writes, same seq_len.
        assert_eq!(kv_ref.seq_len, kv_batched.seq_len);
    }

    /// E1 Q8_0: per-position logits from the batched Q8_0-KV
    /// speculation method match the serial-loop primitive within
    /// numerical noise. Same tolerance rationale as the F32 parity
    /// test — the two paths differ in attention kernel reduction
    /// order. The KV append + inline-dequant in the batched path
    /// also accumulates Q8_0 rows in a slightly different order
    /// than the per-token forward_one loop.
    #[test]
    fn forward_speculation_batched_q8_0_matches_serial_for_same_input() {
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        let tokens: Vec<i32> = vec![5, 7, 2, 14, 3];
        let max_ctx = 16;
        let vocab = cfg.vocab_size;

        // Baseline: serial path on a Q8_0 KV cache.
        let mut kv_ref = KvCache::new_with_dtype(&cfg, max_ctx, KvDtype::Q8_0);
        let mut ref_logits = vec![0f32; tokens.len() * vocab];
        model.forward_speculation(&tokens, 0, &mut kv_ref, &mut ref_logits);

        // Batched Q8_0 path.
        let mut kv_batched = KvCache::new_with_dtype(&cfg, max_ctx, KvDtype::Q8_0);
        let mut batched_logits = vec![0f32; tokens.len() * vocab];
        model.forward_speculation_batched_q8_0(
            &tokens, 0, &mut kv_batched, &mut batched_logits,
        );

        let mut max_diff = 0f32;
        for (i, (&a, &b)) in batched_logits.iter().zip(ref_logits.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(
                d < 1e-2,
                "forward_speculation_batched_q8_0 logit[{i}] drift {d}: \
                 batched={a} serial={b}"
            );
        }
        let _ = max_diff;
        assert_eq!(kv_ref.seq_len, kv_batched.seq_len);
    }

    /// E1 TurboQuant: per-position logits from the batched TQ-KV
    /// speculation method match the serial-loop primitive within
    /// numerical noise. TQ has a 4-bit quantize-on-write path that
    /// accumulates differently from the per-token forward_one loop.
    #[test]
    fn forward_speculation_batched_tq_matches_serial_for_same_input() {
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        let tokens: Vec<i32> = vec![5, 7, 2, 14, 3];
        let max_ctx = 16;
        let vocab = cfg.vocab_size;

        let mut kv_ref = KvCache::new_with_dtype(&cfg, max_ctx, KvDtype::Tq(4));
        let mut ref_logits = vec![0f32; tokens.len() * vocab];
        model.forward_speculation(&tokens, 0, &mut kv_ref, &mut ref_logits);

        let mut kv_batched = KvCache::new_with_dtype(&cfg, max_ctx, KvDtype::Tq(4));
        let mut batched_logits = vec![0f32; tokens.len() * vocab];
        model.forward_speculation_batched_tq(
            &tokens, 0, &mut kv_batched, &mut batched_logits,
        );

        // TQ tolerance: 4-bit quantization injects more noise per
        // row than Q8_0 (8-bit); allow 5e-2 abs to cover the
        // worst-case attention-reduction-order drift on a
        // 4-bit-quantized cache.
        let mut max_diff = 0f32;
        for (i, (&a, &b)) in batched_logits.iter().zip(ref_logits.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(
                d < 5e-2,
                "forward_speculation_batched_tq logit[{i}] drift {d}: \
                 batched={a} serial={b}"
            );
        }
        let _ = max_diff;
        assert_eq!(kv_ref.seq_len, kv_batched.seq_len);
    }

    /// E1 NVFP4: per-position logits from the batched NVFP4-KV
    /// speculation method match the serial-loop primitive within
    /// numerical noise. NVFP4 is the second FP-quant variant
    /// (alongside F32) — 4-bit FP with FP8 scales per 16-element
    /// block; the batched path uses a per-block dequant
    /// concurrent with the attention compute.
    #[test]
    fn forward_speculation_batched_nvfp4_matches_serial_for_same_input() {
        // NVFP4 requires head_dim % 16 == 0 (block size).
        let cfg = synth_cfg(2, 2, 16);
        let model = synth_llama_model(&cfg);
        let tokens: Vec<i32> = vec![5, 7, 2, 14, 3];
        let max_ctx = 16;
        let vocab = cfg.vocab_size;

        let mut kv_ref = KvCache::new_with_dtype(&cfg, max_ctx, KvDtype::Nvfp4);
        let mut ref_logits = vec![0f32; tokens.len() * vocab];
        model.forward_speculation(&tokens, 0, &mut kv_ref, &mut ref_logits);

        let mut kv_batched = KvCache::new_with_dtype(&cfg, max_ctx, KvDtype::Nvfp4);
        let mut batched_logits = vec![0f32; tokens.len() * vocab];
        model.forward_speculation_batched_nvfp4(
            &tokens, 0, &mut kv_batched, &mut batched_logits,
        );

        let mut max_diff = 0f32;
        for (i, (&a, &b)) in batched_logits.iter().zip(ref_logits.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(
                d < 5e-2,
                "forward_speculation_batched_nvfp4 logit[{i}] drift {d}: \
                 batched={a} serial={b}"
            );
        }
        let _ = max_diff;
        assert_eq!(kv_ref.seq_len, kv_batched.seq_len);
    }

    #[test]
    fn forward_speculation_rewind_then_forward_one_matches_fresh_path() {
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        let prefix: Vec<i32> = vec![3, 1, 4];
        let drafts: Vec<i32> = vec![9, 9, 9, 9]; // rejected drafts
        let replacement: i32 = 5; // the real next token (chosen post-rewind)
        let vocab = cfg.vocab_size;
        let max_ctx = 16;

        // Path A: walk prefix + drafts via forward_speculation,
        // then rewind to end-of-prefix, then forward_one(replacement).
        let mut full: Vec<i32> = prefix.clone();
        full.extend_from_slice(&drafts);
        let mut kv_a = KvCache::new(&cfg, max_ctx);
        let mut logits = vec![0f32; full.len() * vocab];
        model.forward_speculation(&full, 0, &mut kv_a, &mut logits);
        // Rewind: drop seq_len back to end-of-prefix. The K/V cells
        // at positions [prefix.len() .. full.len()) keep their data
        // but become invisible to attention because the kernels
        // walk only `seq_len` positions.
        kv_a.seq_len = prefix.len();
        let mut a_logits = vec![0f32; vocab];
        model.forward_one(replacement, prefix.len() as u32, &mut kv_a, &mut a_logits);

        // Path B: walk prefix + replacement directly (no draft
        // detour, no rewind).
        let mut kv_b = KvCache::new(&cfg, max_ctx);
        for (i, &t) in prefix.iter().enumerate() {
            let mut l = vec![0f32; vocab];
            model.forward_one(t, i as u32, &mut kv_b, &mut l);
        }
        let mut b_logits = vec![0f32; vocab];
        model.forward_one(replacement, prefix.len() as u32, &mut kv_b, &mut b_logits);

        // Bit-identical: the rewind drops rejected drafts' K/V
        // rows from attention's view (seq_len gate); the
        // forward_one at position `prefix.len()` overwrites
        // whatever was there from `drafts[0]` with the
        // replacement's actual K/V, so subsequent attention reads
        // are correct.
        for (i, (&a, &b)) in a_logits.iter().zip(b_logits.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "logit[{i}]: rewind+overwrite path = {a}, fresh path = {b}"
            );
        }
    }

    /// E4 phase 4b parity gate: `forward_one_with_mtp_logits` on a
    /// model with `n_mtp_heads == 0` produces bit-identical main
    /// logits to `forward_one`. The MTP path is purely additive —
    /// for non-MTP models, the new entry point is a no-overhead
    /// passthrough that emits only the main logits.
    #[test]
    fn forward_one_with_mtp_logits_matches_forward_one_when_no_heads() {
        let cfg = synth_cfg(2, 2, 8);
        let model = synth_llama_model(&cfg);
        let max_ctx = 16;
        let prefix: Vec<i32> = vec![3, 1, 4, 1];
        let probe: i32 = 5;

        let mut kv_a = KvCache::new(&cfg, max_ctx);
        for (i, &t) in prefix.iter().enumerate() {
            let mut tmp = vec![0f32; cfg.vocab_size];
            model.forward_one(t, i as u32, &mut kv_a, &mut tmp);
        }
        let mut logits_a = vec![0f32; cfg.vocab_size];
        model.forward_one(probe, prefix.len() as u32, &mut kv_a, &mut logits_a);

        let mut kv_b = KvCache::new(&cfg, max_ctx);
        for (i, &t) in prefix.iter().enumerate() {
            let mut tmp = vec![0f32; cfg.vocab_size];
            model.forward_one(t, i as u32, &mut kv_b, &mut tmp);
        }
        let mut logits_b = vec![0f32; cfg.vocab_size];
        let mut mtp_logits: Vec<Vec<f32>> = Vec::new();
        model.forward_one_with_mtp_logits(
            probe,
            prefix.len() as u32,
            &mut kv_b,
            &mut logits_b,
            &mut mtp_logits,
        );
        assert!(
            mtp_logits.is_empty(),
            "non-MTP model must leave mtp_logits empty"
        );
        for (i, (&a, &b)) in logits_a.iter().zip(logits_b.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "logit[{i}]: forward_one = {a}, forward_one_with_mtp_logits = {b} \
                 — non-MTP path must be bit-identical"
            );
        }
    }

    /// E4 phase 4a parity gate: a model whose config does not declare
    /// MTP heads loads with `mtp_heads == None`. This is the
    /// no-behavior-change pin — every non-MTP model in the wild
    /// continues to flow through the unchanged single-head path.
    #[test]
    fn synth_model_without_mtp_metadata_has_no_mtp_heads() {
        let cfg = synth_cfg(2, 2, 8);
        assert_eq!(
            cfg.n_mtp_heads, 0,
            "synth_cfg sets the no-MTP default; if this regresses, every \
             synthetic model in the test suite would start trying to \
             materialize phantom MTP heads"
        );
        let model = synth_llama_model(&cfg);
        assert!(
            model.weights.mtp_heads.is_none(),
            "models without MTP metadata must load with mtp_heads = None"
        );
    }
}
