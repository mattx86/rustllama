//! Mixture-of-experts forward-pass primitives.
//!
//! This is MoE phase 2-B-1: the routing math + per-expert SwiGLU FFN
//! as standalone helpers. Phase 2-B-2 wires these into
//! [`crate::llama_arch::LlamaModel::forward_one`] so MoE models can
//! actually generate tokens.
//!
//! ## Routing
//!
//! [`route_topk`] projects the hidden state through the router matrix
//! `[n_experts, d_model]`, applies softmax over expert logits, picks
//! the top-K experts, and renormalizes the chosen K weights so they
//! sum to 1.0. This is the canonical Mixtral / Qwen3-MoE recipe.
//! DeepSeek-V3's "auxiliary-loss-free load balancing" variant differs
//! in how the routing scores are computed but the post-selection
//! weighted-sum shape is the same.
//!
//! ## Per-expert FFN
//!
//! [`moe_ffn_one`] runs the full MoE FFN for one token position:
//!   1. Route through top-K experts (above).
//!   2. For each selected expert, slice its `[d_ff, d_model]` matrix
//!      out of the 3D `w_*_exps` tensor, run the standard SwiGLU
//!      FFN (`silu(gate) * up → down`), and accumulate into the
//!      output with the router weight as the scaling factor.
//!   3. If shared-expert tensors are present (DeepSeek-V3 family),
//!      run the shared FFN unconditionally and add to the output
//!      with weight 1.0 (the shared expert is always-active, not
//!      weighted by the router).
//!
//! ## Performance
//!
//! Scalar-only for phase 2-B-1. Per-expert slicing allocates a fresh
//! `Vec<u8>` per matvec call (copy the expert's bytes out of the 3D
//! tensor); a follow-up turn caches per-expert `Tensor` handles at
//! load time so the call-site reuses pre-sliced views. The F16 fast
//! path runs at roughly 1/4 the throughput of a same-size dense
//! model in this scalar mode — adequate for correctness validation
//! but not production inference.

use rustllama_kernels_cpu as k;
use rustllama_tensor::Tensor;

use crate::llama_arch::LlamaMoeBlockWeights;

// Cached hot-path env levers. These are consulted 40+ times per
// decoded token (once per MoE layer); the previous per-call
// `std::env::var` reads took the process-wide env lock each time.
// All are "set before the first forward" A/B levers — notably
// `RUSTLLAMA_NO_NORM_TOPK_PROB`, which is the production-required
// configuration for the qwen35moe APEX variants (see the empirical
// note in `route_topk_into`).

fn debug_router_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("RUSTLLAMA_DEBUG_ROUTER").is_ok())
}

fn no_norm_topk_prob() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("RUSTLLAMA_NO_NORM_TOPK_PROB").is_ok())
}

fn debug_moe_split_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("RUSTLLAMA_DEBUG_MOE_SPLIT").is_ok())
}

fn no_shared_expert() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("RUSTLLAMA_NO_SHARED_EXPERT").is_ok())
}

/// Result of routing one token through the MoE router.
#[derive(Debug, Clone)]
pub struct RoutedExperts {
    /// `(expert_idx, weight)` for each of the K selected experts,
    /// ordered by descending weight. Weights are renormalized so
    /// they sum to 1.0 (over the K picks, not over all N experts).
    pub picks: Vec<(usize, f32)>,
}

/// Project `hidden` through the router, softmax over all experts,
/// pick the top-K, renormalize. Allocates a fresh `RoutedExperts`
/// per call — convenient for tests / one-shot callers. The
/// production forward path uses [`route_topk_into`] which writes
/// into caller-provided scratch.
pub fn route_topk(
    hidden: &[f32],
    router: &Tensor,
    n_experts: usize,
    top_k: usize,
) -> RoutedExperts {
    let mut expert_logits = vec![0f32; n_experts];
    let mut picks: Vec<(usize, f32)> = Vec::with_capacity(top_k);
    route_topk_into(hidden, router, n_experts, top_k, &mut expert_logits, &mut picks);
    RoutedExperts { picks }
}

/// Same routing as [`route_topk`] but writes into caller-provided
/// scratch buffers — zero allocations in steady state. Used by
/// [`moe_ffn_one_into`] which gets its scratch from
/// [`crate::accel::ForwardScratch`].
///
/// - `expert_logits`: must be sized `n_experts` on entry; receives
///   the softmax probabilities (over all experts, before top-K
///   selection).
/// - `picks`: cleared on entry and refilled with the top-K
///   `(expert_idx, weight)` pairs (renormalized so weights sum to
///   1.0). Caller's `Vec::capacity` is preserved across calls so
///   re-fills don't allocate.
pub fn route_topk_into(
    hidden: &[f32],
    router: &Tensor,
    n_experts: usize,
    top_k: usize,
    expert_logits: &mut [f32],
    picks: &mut Vec<(usize, f32)>,
) {
    debug_assert!(top_k > 0, "top_k must be at least 1");
    debug_assert!(
        top_k <= n_experts,
        "top_k ({top_k}) cannot exceed n_experts ({n_experts})"
    );
    debug_assert_eq!(expert_logits.len(), n_experts);
    let d_model = hidden.len();
    // RUSTLLAMA_DEBUG_ROUTER: dump per-call router input stats so we
    // can localize whether full-attn-layer MoE inputs are
    // qualitatively different from SSM-layer ones (agent finding:
    // the 32× MoE explosion on full-attn layers is the signature of
    // a near-uniform router softmax — i.e. the router sees inputs
    // it can't distinguish, suggesting upstream-of-router corruption
    // in the full-attention block's output).
    let debug_router = debug_router_enabled();

    // Step 1: router @ hidden → expert_logits.
    k::matvec_tensor(router, hidden, expert_logits, n_experts, d_model);

    // Step 2: softmax over all experts. Subtract max for numerical
    // stability (standard recipe).
    let max = expert_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0f32;
    for v in expert_logits.iter_mut() {
        *v = (*v - max).exp();
        exp_sum += *v;
    }
    if exp_sum > 0.0 {
        let inv = 1.0 / exp_sum;
        for v in expert_logits.iter_mut() {
            *v *= inv;
        }
    }

    // Step 3: top-K selection. Fill `picks` (clearing first) with
    // (idx, weight) pairs, sort descending, truncate. The pre-
    // allocated capacity means clear+push doesn't realloc.
    picks.clear();
    picks.extend(expert_logits.iter().copied().enumerate());
    // Partial selection: O(n) select of the top-K boundary, then a
    // sort of only the K winners — vs the previous full O(n log n)
    // sort of all `n_experts` entries, 40+ times per decoded token
    // on a 256-expert model. Same ordering contract (descending
    // weight, index tiebreak — the total order makes the selected
    // set deterministic); same pattern the sampler's top-k uses.
    let cmp = |a: &(usize, f32), b: &(usize, f32)| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    };
    if top_k < picks.len() {
        picks.select_nth_unstable_by(top_k - 1, cmp);
        picks.truncate(top_k);
    }
    picks.sort_by(cmp);
    if debug_router {
        let top_max = picks.first().map(|p| p.1).unwrap_or(0.0);
        let top_sum: f32 = picks.iter().map(|(_, w)| *w).sum();
        let renorm_factor = if top_sum > 0.0 { 1.0 / top_sum } else { 0.0 };
        let hidden_rms = (hidden.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
            / hidden.len() as f64)
            .sqrt();
        eprintln!(
            "  router: hidden_rms={:.3e} top_max={:.4} top_sum={:.4} renorm_factor={:.2}× picks=[{}]",
            hidden_rms,
            top_max,
            top_sum,
            renorm_factor,
            picks
                .iter()
                .take(3)
                .map(|(i, w)| format!("({i}, {w:.4})"))
                .collect::<Vec<_>>()
                .join(", "),
        );
    }

    // Step 4: renormalize over the top-K so weights sum to 1.0.
    // Mixtral / Qwen3-MoE / Qwen3-Next convention (the latter pins
    // `norm_topk_prob=True` in both default config and the shipped
    // 80B-A3B-Instruct config). Some MoE variants skip
    // renormalization — the impact is a constant scaling factor
    // baked into the down projection, so models trained with vs
    // without renorm aren't drop-in compatible.
    //
    // Empirical note: on some qwen35moe variants, enabling
    // renormalization causes the full-attention layers' MoE to
    // amplify the residual ~30× per layer with a
    // uniformly-negative bias — the residual stream explodes after
    // L03 and the model produces gibberish. Disabling renorm
    // produces a bounded mixed-sign residual stream and partial
    // coherence (English words appearing in the output). Reason
    // unclear: HF reference uses renorm; possible explanations
    // include a quantization artifact in the router producing
    // near-uniform top-K probabilities (where renorm's factor
    // approaches `1/raw_sum` ~30×), or this fine-tuned variant
    // being trained without renorm despite the upstream default.
    // `RUSTLLAMA_NO_NORM_TOPK_PROB=1` skips renorm and is the
    // working configuration for this model until the root cause
    // is pinned.
    let skip_renorm = no_norm_topk_prob();
    if !skip_renorm {
        let topk_sum: f32 = picks.iter().map(|(_, w)| *w).sum();
        if topk_sum > 0.0 {
            let inv = 1.0 / topk_sum;
            for (_, w) in picks.iter_mut() {
                *w *= inv;
            }
        }
    }
}

/// Full MoE FFN for one token. Allocates an output `Vec<f32>` per
/// call — kept as a convenience for tests / one-shot callers. The
/// production forward path uses [`moe_ffn_one_into`] which writes
/// into a caller-provided buffer and uses pre-allocated scratch.
/// MoE disk-spill read-redirect (completes the opt-in spill feature #5).
///
/// When the disk-spill store is armed (`accel::moe_spill_enabled()`) and
/// the routed expert whose gate weight is `w_gate_e` was evicted — and
/// therefore spilled — reconstruct its three weight parts from the
/// secondary store and hand back owned CPU [`Tensor`]s to matvec against,
/// instead of re-faulting the weight pages from the GGUF mmap.
///
/// Returns `None` (caller then uses the mmap tensors, byte-identically)
/// when the expert has no spill record (never evicted) or its gate weight
/// isn't file-backed (`mmap_borrowed_ptr_len` is `None`, so it was never
/// spill-eligible).
///
/// With the default `RUSTLLAMA_MOE_STORE_BITS=0` (native spill) the
/// reconstructed bytes are bit-identical to the mmap, so matvec output is
/// unchanged; a nonzero store-bits requantizes the cold tier (lossy — the
/// caller's explicit opt-in).
///
/// The caller MUST gate this behind `accel::moe_spill_enabled()` so the
/// default (feature-off) path never allocates or touches the store.
///
/// PERF: reconstruction reads the spill file + rebuilds tensors, so the
/// caller gates it on a true residency MISS — it samples
/// `accel::expert_resident_by_gate` BEFORE `expert_pin_touch` pins the
/// expert and only redirects when the expert was non-resident (cold /
/// evicted). A resident (hot) expert keeps its RAM-locked mmap pages, so
/// it skips reconstruction entirely.
fn moe_spill_redirect_expert(
    w_gate_e: &Tensor,
    w_up_e: &Tensor,
    w_down_e: &Tensor,
) -> Option<(Tensor, Tensor, Tensor)> {
    // Key = the gate weight's mmap start address (the exact key the pin
    // cache + spill store use). Non-mmap experts are never spilled.
    let gate_key = w_gate_e.storage.mmap_borrowed_ptr_len()?.0 as usize;
    let spilled = crate::accel::moe_spill_reconstruct(gate_key)?;
    // Parts are (gate, up, down) in that order (see `SpilledExpert`).
    let mut parts = spilled.parts.into_iter();
    let g = spill_part_to_tensor(parts.next()?, w_gate_e)?;
    let u = spill_part_to_tensor(parts.next()?, w_up_e)?;
    let d = spill_part_to_tensor(parts.next()?, w_down_e)?;
    Some((g, u, d))
}

/// Turn one reconstructed [`crate::accel::SpilledPart`] into an owned CPU
/// [`Tensor`] shaped like `orig` (the mmap tensor it stands in for).
fn spill_part_to_tensor(part: crate::accel::SpilledPart, orig: &Tensor) -> Option<Tensor> {
    match part {
        // Verbatim native bytes: reuse the ORIGINAL dtype + shape (the
        // record's own `dtype` is a placeholder for native spill). Bytes
        // equal the mmap bytes, so matvec output is unchanged.
        crate::accel::SpilledPart::Native { bytes, .. } => {
            if bytes.len() != orig.storage.len_bytes() {
                return None;
            }
            Some(Tensor {
                device: orig.device.clone(),
                dtype: orig.dtype,
                shape: orig.shape.clone(),
                strides: orig.strides.clone(),
                storage: rustllama_tensor::Storage::CpuOwned(bytes.into()),
                name: orig.name.clone(),
            })
        }
        // Low-bit spill decoded back to f32 (lossy vs the source): build
        // an F32 tensor with the original shape.
        crate::accel::SpilledPart::Dequant { values, .. } => {
            if values.len() as u64 != orig.element_count() {
                return None;
            }
            Some(Tensor::from_vec_f32(orig.name.clone(), orig.shape.clone(), values))
        }
    }
}

pub fn moe_ffn_one(
    hidden: &[f32],
    block: &LlamaMoeBlockWeights,
    d_model: usize,
    d_ff: usize,
    n_experts: usize,
    top_k: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; d_model];
    let mut gate_buf = vec![0f32; d_ff];
    let mut up_buf = vec![0f32; d_ff];
    let mut ff_buf = vec![0f32; d_ff];
    let mut down_buf = vec![0f32; d_model];
    let mut expert_logits = vec![0f32; n_experts];
    let mut picks: Vec<(usize, f32)> = Vec::with_capacity(top_k);
    moe_ffn_one_into(
        hidden,
        block,
        d_model,
        d_ff,
        n_experts,
        top_k,
        &mut out,
        &mut gate_buf,
        &mut up_buf,
        &mut ff_buf,
        &mut down_buf,
        &mut expert_logits,
        &mut picks,
    );
    out
}

/// Allocation-free MoE FFN: writes into caller-provided output +
/// scratch buffers. Wired by `LlamaModel::forward_one` from
/// [`crate::accel::ForwardScratch`] so the per-layer call reuses
/// pre-allocated buffers across all tokens.
///
/// Buffer sizing contract:
/// - `out`, `down_buf`: `[d_model]`
/// - `gate_buf`, `up_buf`, `ff_buf`: `[d_ff]`
/// - `expert_logits`: `[n_experts]`
/// - `picks`: capacity ≥ `top_k` (cleared on entry, refilled)
///
/// `out` is zeroed at the top — caller doesn't need to pre-clear.
#[allow(clippy::too_many_arguments)]
pub fn moe_ffn_one_into(
    hidden: &[f32],
    block: &LlamaMoeBlockWeights,
    d_model: usize,
    d_ff: usize,
    n_experts: usize,
    top_k: usize,
    out: &mut [f32],
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    ff_buf: &mut [f32],
    down_buf: &mut [f32],
    expert_logits: &mut [f32],
    picks: &mut Vec<(usize, f32)>,
) {
    debug_assert_eq!(out.len(), d_model);
    debug_assert_eq!(gate_buf.len(), d_ff);
    debug_assert_eq!(up_buf.len(), d_ff);
    debug_assert_eq!(ff_buf.len(), d_ff);
    debug_assert_eq!(down_buf.len(), d_model);

    // Zero the output: subsequent expert loops accumulate into it.
    for v in out.iter_mut() {
        *v = 0.0;
    }

    route_topk_into(hidden, &block.router, n_experts, top_k, expert_logits, picks);
    // Hint the readahead thread before the first expert matvec: the
    // remaining picks (and the same indices on the next layers) start
    // faulting in from disk while pick 1 computes.
    crate::accel::note_routed_experts(picks);

    // MoE tiered-expert grouped FFN (decode fast path): when every routed
    // expert is a promoted, device-resident Q4_K buffer, run the whole top-k
    // FFN in one on-device pass (no per-expert host↔device round-trip / CPU
    // silu migration). Verdict + fallback gated; on success `out` holds the
    // routed-expert sum and the shared expert (below) still adds in. A miss
    // (not all promoted, dtype ≠ Q4_K, verdict off) falls through unchanged.
    if crate::accel::try_moe_ffn_grouped_q4k_dev_resident(
        hidden,
        &block.gate_per_expert,
        &block.up_per_expert,
        &block.down_per_expert,
        picks,
        d_model,
        d_ff,
        out,
    ) {
        // Routed experts done on-device. Fall through to the shared-expert
        // block, which accumulates into `out` exactly as in the CPU path.
    } else {
    // q★ co-execution (tuner-resolved): with a nonzero split on a
    // GPU-active layer, part of the picks dispatch to the GPU while
    // the rest run CPU-concurrently. Pointless when experts are
    // pinned to CPU anyway (the split would race two CPU branches
    // over the same cores the inner matvecs already parallelize).
    let split = crate::accel::moe_gpu_split_permille();
    if split > 0
        && picks.len() >= 2
        && !crate::accel::moe_experts_cpu_enabled()
        && crate::accel::gpu_active_for_current_layer()
    {
        moe_ffn_routed_split_exec(
            hidden,
            &block.gate_per_expert,
            &block.up_per_expert,
            &block.down_per_expert,
            picks,
            d_model,
            d_ff,
            out,
            gate_buf,
            up_buf,
            ff_buf,
            down_buf,
            split,
        );
    } else {
        for (expert_idx, weight) in picks.iter() {
            // Per-expert views are pre-computed at load time
            // (`LlamaWeights::from_gguf` MoE branch) and share storage
            // with the parent 3D tensor via Storage::CpuOwnedSlice.
            let w_gate_e = &block.gate_per_expert[*expert_idx];
            let w_up_e = &block.up_per_expert[*expert_idx];
            let w_down_e = &block.down_per_expert[*expert_idx];

            // Residency-aware spill gate: sample the pin cache BEFORE the
            // touch below pins this expert (→ makes it resident). A hot
            // (resident) expert keeps its pages RAM-locked, so serving it
            // from the spill store would be pure overhead — reconstruct
            // only on a true residency MISS. Default-OFF no-op.
            let spill_miss = crate::accel::moe_spill_enabled()
                && !crate::accel::expert_resident_by_gate(w_gate_e);

            // MoE LRU expert-pin cache: VirtualLock this hot expert's
            // weight pages in RAM (when the cache is active + weights are
            // zero-copy file-backed) so they stop re-faulting from disk.
            // No-op when RUSTLLAMA_MOE_EXPERT_CACHE_MB=0. Released after
            // the matvecs so eviction can reclaim it once cold.
            let pin = crate::accel::expert_pin_touch(*expert_idx, w_gate_e, w_up_e, w_down_e);

            // MoE disk-spill read-redirect (feature #5, opt-in). Default-
            // OFF: `moe_spill_enabled()` is false → this whole block is
            // skipped and the mmap path below runs byte-identically. When
            // armed, an evicted+spilled expert (residency miss) is served
            // from the secondary store instead of re-faulting its GGUF
            // pages; a resident expert uses its hot mmap pages.
            let spilled = if spill_miss {
                moe_spill_redirect_expert(w_gate_e, w_up_e, w_down_e)
            } else {
                None
            };
            if let Some((sg, su, sd)) = spilled.as_ref() {
                // Reconstructed tensors are freshly-allocated CPU buffers
                // whose address changes per token; routing them through
                // the USM dispatch (host-pointer-keyed weight cache) could
                // serve a stale upload, so run the CPU kernels directly.
                k::matvec_tensor(sg, hidden, gate_buf, d_ff, d_model);
                k::matvec_tensor(su, hidden, up_buf, d_ff, d_model);
                k::silu_mul_f32(gate_buf, up_buf, ff_buf);
                k::matvec_tensor(sd, ff_buf, down_buf, d_model, d_ff);
            } else {
                // H4: fused gate+up matvec when the dtype has a fused
                // kernel (currently Q4_K, Q8_0); otherwise falls back to
                // two dispatches inside the helper. Saves one Level-Zero
                // dispatch + activation re-load per expert.
                crate::llama_arch::matvec_tensor_gate_up_fused_dispatch(
                    w_gate_e, w_up_e, hidden, gate_buf, up_buf, d_ff, d_model,
                );
                k::silu_mul_f32(gate_buf, up_buf, ff_buf);
                crate::llama_arch::matvec_tensor_dispatch(w_down_e, ff_buf, down_buf, d_model, d_ff);
            }

            for j in 0..d_model {
                out[j] += weight * down_buf[j];
            }
            crate::accel::expert_pin_release(pin);
        }
    }
    } // end grouped-FFN fallback (per-expert / split path)

    // Shared expert. DeepSeek-V3 runs it always-on at the routed-
    // expert width (`d_ff`); Qwen2-MoE (Qwen1.5-MoE) runs a WIDER
    // shared FFN (`shared_expert_intermediate_size` != d_ff) scaled
    // per token by `sigmoid(shared_router @ hidden)`. The shared
    // width is read from the tensor itself, and the sigmoid gate is
    // applied only when `block.shared_router` is present (None →
    // weight 1.0, the DeepSeek-V3 path, byte-identical to before).
    if let (Some(w_g), Some(w_u), Some(w_d)) = (
        &block.w_gate_shared,
        &block.w_up_shared,
        &block.w_down_shared,
    ) {
        let shared_weight = match block.shared_router.as_ref() {
            Some(sr) => {
                let mut s = [0.0f32; 1];
                k::matvec_tensor(sr, hidden, &mut s, 1, d_model);
                // Numerically stable sigmoid (matches moe_ffn_one_into_parts).
                if s[0] >= 0.0 {
                    1.0 / (1.0 + (-s[0]).exp())
                } else {
                    let e = s[0].exp();
                    e / (1.0 + e)
                }
            }
            None => 1.0,
        };
        // Shared FFN width = w_gate_shared rows. Equals d_ff for
        // DeepSeek-V3 (reuse the routed-expert scratch); wider for
        // Qwen2-MoE, where the d_ff-sized scratch would overflow, so
        // allocate per-call shared buffers.
        let shared_d_ff = w_g.shape.first().copied().unwrap_or(d_ff as u64) as usize;
        if shared_d_ff <= d_ff {
            crate::llama_arch::matvec_tensor_gate_up_fused_dispatch(
                w_g,
                w_u,
                hidden,
                &mut gate_buf[..shared_d_ff],
                &mut up_buf[..shared_d_ff],
                shared_d_ff,
                d_model,
            );
            k::silu_mul_f32(
                &gate_buf[..shared_d_ff],
                &up_buf[..shared_d_ff],
                &mut ff_buf[..shared_d_ff],
            );
            crate::llama_arch::matvec_tensor_dispatch(
                w_d,
                &ff_buf[..shared_d_ff],
                down_buf,
                d_model,
                shared_d_ff,
            );
        } else {
            let mut sg = vec![0.0f32; shared_d_ff];
            let mut su = vec![0.0f32; shared_d_ff];
            let mut sf = vec![0.0f32; shared_d_ff];
            crate::llama_arch::matvec_tensor_gate_up_fused_dispatch(
                w_g, w_u, hidden, &mut sg, &mut su, shared_d_ff, d_model,
            );
            k::silu_mul_f32(&sg, &su, &mut sf);
            crate::llama_arch::matvec_tensor_dispatch(w_d, &sf, down_buf, d_model, shared_d_ff);
        }
        for j in 0..d_model {
            out[j] += shared_weight * down_buf[j];
        }
    }
}

/// Generalized variant of [`moe_ffn_one_into`] that takes tensor
/// references directly instead of a `&LlamaMoeBlockWeights`. Lets
/// the hybrid-model forward (Phase 3.7b) reuse the MoE FFN logic
/// for `SsmBlockWeights` and `HybridAttnBlockWeights`, which carry
/// the same FFN tensor layout but live in different block types.
///
/// Hybrid models can optionally pass a `shared_router` tensor; when
/// `Some`, the shared expert contribution is sigmoid-gated by the
/// per-token scalar `sigmoid(shared_router @ hidden)` (qwen35moe
/// convention). When `None`, the shared expert is always-on
/// (DeepSeek-V3 convention) — matches the existing
/// [`moe_ffn_one_into`] behavior.
#[allow(clippy::too_many_arguments)]
pub fn moe_ffn_one_into_parts(
    hidden: &[f32],
    router: &Tensor,
    gate_per_expert: &[Tensor],
    up_per_expert: &[Tensor],
    down_per_expert: &[Tensor],
    w_gate_shared: Option<&Tensor>,
    w_up_shared: Option<&Tensor>,
    w_down_shared: Option<&Tensor>,
    shared_router: Option<&Tensor>,
    d_model: usize,
    d_ff: usize,
    n_experts: usize,
    top_k: usize,
    out: &mut [f32],
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    ff_buf: &mut [f32],
    down_buf: &mut [f32],
    expert_logits: &mut [f32],
    picks: &mut Vec<(usize, f32)>,
) {
    debug_assert_eq!(out.len(), d_model);
    debug_assert_eq!(gate_buf.len(), d_ff);
    debug_assert_eq!(up_buf.len(), d_ff);
    debug_assert_eq!(ff_buf.len(), d_ff);
    debug_assert_eq!(down_buf.len(), d_model);

    for v in out.iter_mut() {
        *v = 0.0;
    }
    route_topk_into(hidden, router, n_experts, top_k, expert_logits, picks);
    // Readahead hint — see moe_ffn_one_into.
    crate::accel::note_routed_experts(picks);

    // q★ co-execution gate — see moe_ffn_one_into.
    let split = crate::accel::moe_gpu_split_permille();
    if split > 0
        && picks.len() >= 2
        && !crate::accel::moe_experts_cpu_enabled()
        && crate::accel::gpu_active_for_current_layer()
    {
        moe_ffn_routed_split_exec(
            hidden,
            gate_per_expert,
            up_per_expert,
            down_per_expert,
            picks,
            d_model,
            d_ff,
            out,
            gate_buf,
            up_buf,
            ff_buf,
            down_buf,
            split,
        );
    } else {
        for (expert_idx, weight) in picks.iter() {
            let w_gate_e = &gate_per_expert[*expert_idx];
            let w_up_e = &up_per_expert[*expert_idx];
            let w_down_e = &down_per_expert[*expert_idx];

            // Residency-aware spill gate: sample BEFORE the touch pins the
            // expert (see moe_ffn_one_into). Reconstruct only on a miss.
            let spill_miss = crate::accel::moe_spill_enabled()
                && !crate::accel::expert_resident_by_gate(w_gate_e);

            // MoE LRU expert-pin cache (see moe_ffn_one_into). No-op when
            // RUSTLLAMA_MOE_EXPERT_CACHE_MB=0 or weights aren't file-backed.
            let pin = crate::accel::expert_pin_touch(*expert_idx, w_gate_e, w_up_e, w_down_e);

            // MoE disk-spill read-redirect (feature #5, opt-in). Default-
            // OFF (`moe_spill_enabled()` == false) → the mmap path below
            // runs byte-identically. See `moe_ffn_one_into` for the full
            // rationale (CPU-only kernels avoid the USM stale-upload race).
            let spilled = if spill_miss {
                moe_spill_redirect_expert(w_gate_e, w_up_e, w_down_e)
            } else {
                None
            };
            if let Some((sg, su, sd)) = spilled.as_ref() {
                k::matvec_tensor(sg, hidden, gate_buf, d_ff, d_model);
                k::matvec_tensor(su, hidden, up_buf, d_ff, d_model);
                k::silu_mul_f32(gate_buf, up_buf, ff_buf);
                k::matvec_tensor(sd, ff_buf, down_buf, d_model, d_ff);
            } else {
                crate::llama_arch::matvec_tensor_gate_up_fused_dispatch(
                    w_gate_e, w_up_e, hidden, gate_buf, up_buf, d_ff, d_model,
                );
                k::silu_mul_f32(gate_buf, up_buf, ff_buf);
                crate::llama_arch::matvec_tensor_dispatch(w_down_e, ff_buf, down_buf, d_model, d_ff);
            }

            for j in 0..d_model {
                out[j] += weight * down_buf[j];
            }
            crate::accel::expert_pin_release(pin);
        }
    }

    // RUSTLLAMA_DEBUG_MOE_SPLIT: log routed-only vs total separately
    // so we can isolate routed-expert magnitude from shared-expert
    // magnitude. Critical for diagnosing the layer-3 MoE explosion.
    if debug_moe_split_enabled() {
        let routed_rms = (out.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
            / out.len() as f64)
            .sqrt();
        let routed_min = out.iter().cloned().fold(f32::INFINITY, f32::min);
        let routed_max = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!(
            "  moe-split routed: rms={:.3e} min={:+.2e} max={:+.2e}",
            routed_rms, routed_min, routed_max
        );
    }
    // RUSTLLAMA_NO_SHARED_EXPERT: skip shared expert entirely so we
    // can see whether the explosion comes from routed or shared.
    let skip_shared = no_shared_expert();
    // Shared expert. Optionally router-gated (qwen35moe) — when
    // `shared_router` is present, multiply the shared expert's
    // contribution by sigmoid(shared_router @ hidden). The router
    // is shape [1, d_model], yielding a single scalar per token.
    if !skip_shared {
        if let (Some(w_g), Some(w_u), Some(w_d)) = (w_gate_shared, w_up_shared, w_down_shared) {
            let shared_weight = if let Some(sr) = shared_router {
                let mut s = [0.0f32; 1];
                k::matvec_tensor(sr, hidden, &mut s, 1, d_model);
                // Numerically stable sigmoid.
                if s[0] >= 0.0 {
                    1.0 / (1.0 + (-s[0]).exp())
                } else {
                    let e = s[0].exp();
                    e / (1.0 + e)
                }
            } else {
                1.0
            };
            crate::llama_arch::matvec_tensor_gate_up_fused_dispatch(
                w_g, w_u, hidden, gate_buf, up_buf, d_ff, d_model,
            );
            k::silu_mul_f32(gate_buf, up_buf, ff_buf);
            crate::llama_arch::matvec_tensor_dispatch(w_d, ff_buf, down_buf, d_model, d_ff);
            for j in 0..d_model {
                out[j] += shared_weight * down_buf[j];
            }
        }
    }
}

/// Chunk-level MoE FFN with **grouped-expert execution** — the core
/// of the batched hybrid prefill (roadmap Phase 4, FreeToken's
/// prefill-streaming insight): route every token of the chunk first,
/// group the (token, pick) pairs by expert, then run **one pass per
/// unique expert** over all of its assigned tokens via the batched
/// matvec dispatch. At chunk=64 on a 256-expert model, each selected
/// expert's weights are read once per chunk instead of up to 64× —
/// on a streaming (bigger-than-RAM) model that cuts prefill expert
/// I/O by an order of magnitude, and on a resident model it turns 64
/// weight re-reads into one cache-warm pass.
///
/// Semantics match `n_tokens` sequential [`moe_ffn_one_into_parts`]
/// calls up to fp reduction order: per-(token, expert) products are
/// computed by the same kernels; only the accumulation order into a
/// token's output row changes (ascending expert index instead of
/// router-weight order). Experts execute in ascending index order —
/// deterministic run to run.
///
/// `hidden_rows` / `out_rows` are row-major `[n_tokens][d_model]`;
/// `out_rows` is zeroed here. Scratch is allocated per call — the
/// chunk path runs once per layer per chunk, so allocation cost is
/// noise next to the expert matvecs (pooling is a follow-up with the
/// batched-prefill scratch story).
#[allow(clippy::too_many_arguments)]
pub fn moe_ffn_chunk_into_parts(
    hidden_rows: &[f32],
    n_tokens: usize,
    router: &Tensor,
    gate_per_expert: &[Tensor],
    up_per_expert: &[Tensor],
    down_per_expert: &[Tensor],
    w_gate_shared: Option<&Tensor>,
    w_up_shared: Option<&Tensor>,
    w_down_shared: Option<&Tensor>,
    shared_router: Option<&Tensor>,
    d_model: usize,
    d_ff: usize,
    n_experts: usize,
    top_k: usize,
    out_rows: &mut [f32],
) {
    debug_assert_eq!(hidden_rows.len(), n_tokens * d_model);
    debug_assert_eq!(out_rows.len(), n_tokens * d_model);
    for v in out_rows.iter_mut() {
        *v = 0.0;
    }
    if n_tokens == 0 {
        return;
    }

    // 1. Route every token. Same math as the per-token path — the
    //    routing itself stays per token (tiny matvec).
    let mut expert_logits = vec![0.0f32; n_experts];
    let mut picks: Vec<(usize, f32)> = Vec::with_capacity(top_k);
    let mut assignments: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_experts];
    let mut max_group = 0usize;
    for t in 0..n_tokens {
        let hidden = &hidden_rows[t * d_model..(t + 1) * d_model];
        route_topk_into(hidden, router, n_experts, top_k, &mut expert_logits, &mut picks);
        for (e, w) in picks.iter() {
            assignments[*e].push((t, *w));
            max_group = max_group.max(assignments[*e].len());
        }
    }

    // 2. One pass per unique expert over its assigned tokens.
    let mut x_gather = vec![0.0f32; max_group * d_model];
    let mut gate_rows = vec![0.0f32; max_group * d_ff];
    let mut up_rows = vec![0.0f32; max_group * d_ff];
    let mut ff_rows = vec![0.0f32; max_group * d_ff];
    let mut down_rows = vec![0.0f32; max_group * d_model];
    for (e, assigned) in assignments.iter().enumerate() {
        if assigned.is_empty() {
            continue;
        }
        let g = assigned.len();
        let w_gate_e = &gate_per_expert[e];
        let w_up_e = &up_per_expert[e];
        let w_down_e = &down_per_expert[e];
        // One pin-cache access per (expert, chunk) — that's one real
        // read of the expert's weights, which is what the learning
        // cache should count.
        let pin = crate::accel::expert_pin_touch(e, w_gate_e, w_up_e, w_down_e);
        for (i, (t, _)) in assigned.iter().enumerate() {
            x_gather[i * d_model..(i + 1) * d_model]
                .copy_from_slice(&hidden_rows[t * d_model..(t + 1) * d_model]);
        }
        crate::llama_arch::matvec_tensor_batched_dispatch(
            w_gate_e, &x_gather[..g * d_model], &mut gate_rows[..g * d_ff], d_ff, d_model, g,
        );
        crate::llama_arch::matvec_tensor_batched_dispatch(
            w_up_e, &x_gather[..g * d_model], &mut up_rows[..g * d_ff], d_ff, d_model, g,
        );
        for i in 0..g {
            k::silu_mul_f32(
                &gate_rows[i * d_ff..(i + 1) * d_ff],
                &up_rows[i * d_ff..(i + 1) * d_ff],
                &mut ff_rows[i * d_ff..(i + 1) * d_ff],
            );
        }
        crate::llama_arch::matvec_tensor_batched_dispatch(
            w_down_e, &ff_rows[..g * d_ff], &mut down_rows[..g * d_model], d_model, d_ff, g,
        );
        for (i, (t, w)) in assigned.iter().enumerate() {
            let dst = &mut out_rows[t * d_model..(t + 1) * d_model];
            let src = &down_rows[i * d_model..(i + 1) * d_model];
            for j in 0..d_model {
                dst[j] += w * src[j];
            }
        }
        crate::accel::expert_pin_release(pin);
    }

    // 3. Shared expert — batched over the whole chunk, optionally
    //    sigmoid-gated per token by the shared router (qwen35moe
    //    convention; always-on when absent — DeepSeek-V3 convention).
    //    Honors the same RUSTLLAMA_NO_SHARED_EXPERT A/B lever as the
    //    per-token path.
    if no_shared_expert() {
        return;
    }
    if let (Some(w_g), Some(w_u), Some(w_d)) = (w_gate_shared, w_up_shared, w_down_shared) {
        let mut sh_gate = vec![0.0f32; n_tokens * d_ff];
        let mut sh_up = vec![0.0f32; n_tokens * d_ff];
        let mut sh_ff = vec![0.0f32; n_tokens * d_ff];
        let mut sh_down = vec![0.0f32; n_tokens * d_model];
        crate::llama_arch::matvec_tensor_batched_dispatch(
            w_g, hidden_rows, &mut sh_gate, d_ff, d_model, n_tokens,
        );
        crate::llama_arch::matvec_tensor_batched_dispatch(
            w_u, hidden_rows, &mut sh_up, d_ff, d_model, n_tokens,
        );
        for i in 0..n_tokens {
            k::silu_mul_f32(
                &sh_gate[i * d_ff..(i + 1) * d_ff],
                &sh_up[i * d_ff..(i + 1) * d_ff],
                &mut sh_ff[i * d_ff..(i + 1) * d_ff],
            );
        }
        crate::llama_arch::matvec_tensor_batched_dispatch(
            w_d, &sh_ff, &mut sh_down, d_model, d_ff, n_tokens,
        );
        for t in 0..n_tokens {
            let shared_weight = if let Some(sr) = shared_router {
                let mut s = [0.0f32; 1];
                k::matvec_tensor(sr, &hidden_rows[t * d_model..(t + 1) * d_model], &mut s, 1, d_model);
                // Numerically stable sigmoid (same branch as the
                // per-token path).
                if s[0] >= 0.0 {
                    1.0 / (1.0 + (-s[0]).exp())
                } else {
                    let e = s[0].exp();
                    e / (1.0 + e)
                }
            } else {
                1.0
            };
            let dst = &mut out_rows[t * d_model..(t + 1) * d_model];
            let src = &sh_down[t * d_model..(t + 1) * d_model];
            for j in 0..d_model {
                dst[j] += shared_weight * src[j];
            }
        }
    }
}

thread_local! {
    /// Scratch for the CPU branch of the split-exec path. Lives in
    /// the *executing* thread's TLS (a rayon worker), so repeated
    /// layer calls reuse the same heap storage without contending
    /// with the engine thread's `ForwardScratch`.
    static SPLIT_CPU_SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())) };
    /// Engine-thread-side accumulator handed to the CPU branch; taken
    /// out (mem::take) for the duration of the join so the borrow can
    /// cross into the spawned closure.
    static SPLIT_CPU_OUT: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// q★-style CPU/GPU co-execution over the routed-expert loop
/// (FreeToken's bandwidth-adaptive execution, tuner-resolved): the
/// first `⌈K·split‰⌉` picks dispatch through the (GPU-eligible)
/// matvec helpers on the calling engine thread while the remaining
/// picks run pure-CPU kernels on a rayon worker concurrently. Each
/// branch accumulates into its own buffer and the buffers combine in
/// a fixed order, so the result is deterministic regardless of thread
/// timing (both branches preserve pick order; only the final add is
/// reordered vs. the serial loop — same fp-tolerance class as the
/// batched prefill paths).
///
/// Callers gate on: split > 0, ≥2 picks, experts not forced to CPU,
/// and the current layer being GPU-active — otherwise the serial loop
/// is strictly better (no join overhead).
#[allow(clippy::too_many_arguments)]
fn moe_ffn_routed_split_exec(
    hidden: &[f32],
    gate_per_expert: &[Tensor],
    up_per_expert: &[Tensor],
    down_per_expert: &[Tensor],
    picks: &[(usize, f32)],
    d_model: usize,
    d_ff: usize,
    out: &mut [f32],
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    ff_buf: &mut [f32],
    down_buf: &mut [f32],
    split_permille: u32,
) {
    debug_assert!(picks.len() >= 2);
    let n_gpu = ((picks.len() as u32 * split_permille + 500) / 1000)
        .clamp(1, picks.len() as u32 - 1) as usize;
    let (gpu_picks, cpu_picks) = picks.split_at(n_gpu);

    let mut cpu_out = SPLIT_CPU_OUT.with(|c| std::mem::take(&mut *c.borrow_mut()));
    cpu_out.clear();
    cpu_out.resize(d_model, 0.0);

    // `join(a, b)`: `a` is guaranteed to run on the calling thread —
    // that's the GPU branch, which needs this thread's SYCL stream
    // TLS. `b` (the CPU branch) is stolen by a pool worker; its
    // kernels never consult dispatch TLS, so any thread is fine.
    rayon::join(
        || {
            for (expert_idx, weight) in gpu_picks.iter() {
                let w_gate_e = &gate_per_expert[*expert_idx];
                let w_up_e = &up_per_expert[*expert_idx];
                let w_down_e = &down_per_expert[*expert_idx];
                let pin = crate::accel::expert_pin_touch(*expert_idx, w_gate_e, w_up_e, w_down_e);
                crate::llama_arch::matvec_tensor_gate_up_fused_dispatch(
                    w_gate_e, w_up_e, hidden, gate_buf, up_buf, d_ff, d_model,
                );
                k::silu_mul_f32(gate_buf, up_buf, ff_buf);
                crate::llama_arch::matvec_tensor_dispatch(w_down_e, ff_buf, down_buf, d_model, d_ff);
                for j in 0..d_model {
                    out[j] += weight * down_buf[j];
                }
                crate::accel::expert_pin_release(pin);
            }
        },
        || {
            SPLIT_CPU_SCRATCH.with(|cell| {
                let mut s = cell.borrow_mut();
                let (gate_s, up_s, ff_s, down_s) = &mut *s;
                gate_s.resize(d_ff, 0.0);
                up_s.resize(d_ff, 0.0);
                ff_s.resize(d_ff, 0.0);
                down_s.resize(d_model, 0.0);
                for (expert_idx, weight) in cpu_picks.iter() {
                    let w_gate_e = &gate_per_expert[*expert_idx];
                    let w_up_e = &up_per_expert[*expert_idx];
                    let w_down_e = &down_per_expert[*expert_idx];
                    let pin =
                        crate::accel::expert_pin_touch(*expert_idx, w_gate_e, w_up_e, w_down_e);
                    // Direct CPU kernels — no dispatch, so this branch
                    // is device-deterministic on any thread. Pinned to
                    // the fully-serial variants: this closure already
                    // occupies a rayon worker, and stealing more pool
                    // threads here would oversubscribe against the
                    // concurrent GPU branch.
                    k::matvec_tensor_serial(w_gate_e, hidden, gate_s, d_ff, d_model);
                    k::matvec_tensor_serial(w_up_e, hidden, up_s, d_ff, d_model);
                    k::silu_mul_f32(gate_s, up_s, ff_s);
                    k::matvec_tensor_serial(w_down_e, ff_s, down_s, d_model, d_ff);
                    for j in 0..d_model {
                        cpu_out[j] += weight * down_s[j];
                    }
                    crate::accel::expert_pin_release(pin);
                }
            });
        },
    );

    for j in 0..d_model {
        out[j] += cpu_out[j];
    }
    SPLIT_CPU_OUT.with(|c| *c.borrow_mut() = cpu_out);
}

/// Carve a per-expert view out of a 3D MoE expert tensor.
///
/// Real-world MoE GGUFs store the per-expert weights as a single
/// 3D tensor with dims `[d_in, d_out, n_experts]` (GGUF column-
/// major, so the rightmost dim varies fastest in memory). Expert
/// `e` occupies the byte range computed from
/// [`Dtype::byte_size`] applied to `d_out * d_in` — which handles
/// both flat dtypes (F16/F32/BF16: simple multiplication) AND
/// block-quantized formats (Q4_K_M: `(d_out*d_in/256) * 144`,
/// IQ4_XS: `(d_out*d_in/256) * 136`, etc.). The expert slice
/// forms a row-major `[d_out, d_in]` matrix that
/// `matvec_tensor_dispatch` consumes via the appropriate per-
/// dtype kernel.
///
/// Storage is shared with the parent via [`Storage::slice`] — no
/// byte copy, just an Arc clone + offset. The returned `Tensor`
/// keeps the parent Arc alive through the slice variant.
///
/// Panics if `d_out * d_in` isn't aligned to the dtype's block
/// size (real GGUFs always are; the panic surfaces a configuration
/// error rather than letting the slice produce garbage).
pub fn expert_view(big: &Tensor, expert_idx: usize, d_out: usize, d_in: usize) -> Tensor {
    // byte_size handles all supported dtypes (F16/F32/BF16 +
    // every Raw quant variant). Per-expert byte count = byte
    // count for one `[d_out, d_in]` matrix of this dtype.
    let bytes_per_expert = big.dtype.byte_size((d_out * d_in) as u64) as usize;
    let offset = expert_idx * bytes_per_expert;
    Tensor {
        device: big.device.clone(),
        dtype: big.dtype,
        shape: vec![d_out as u64, d_in as u64],
        strides: vec![d_in as i64, 1],
        storage: big.storage.slice(offset, bytes_per_expert),
        name: format!("{}.e{expert_idx}", big.name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use rustllama_tensor::{Device, Dtype, Storage as TStorage};

    /// q★ split-exec parity: the concurrent CPU/GPU-partitioned loop
    /// must match the serial routed loop within fp-association
    /// tolerance (the two branches accumulate separately, so the
    /// final add re-associates the sum — same class of difference as
    /// the batched prefill paths). Calls the helper directly with an
    /// explicit split so no process-global knob is touched (tests run
    /// in parallel).
    #[test]
    fn split_exec_matches_serial_routed_loop() {
        let d_model = 8usize;
        let d_ff = 4usize;
        let n_experts = 3usize;
        // Distinct per-expert weights: expert e's matrix entry (r, c)
        // = small deterministic value varying by (e, r, c).
        let mk = |e: usize, rows: usize, cols: usize, tag: &str| {
            let vals: Vec<f32> = (0..rows * cols)
                .map(|i| ((e * 31 + i * 7) % 13) as f32 * 0.05 - 0.3)
                .collect();
            tensor_f16_from_rowmajor(&vals, rows, cols, &format!("{tag}.e{e}"))
        };
        let gates: Vec<Tensor> = (0..n_experts).map(|e| mk(e, d_ff, d_model, "g")).collect();
        let ups: Vec<Tensor> = (0..n_experts).map(|e| mk(e, d_ff, d_model, "u")).collect();
        let downs: Vec<Tensor> = (0..n_experts).map(|e| mk(e, d_model, d_ff, "d")).collect();
        let hidden: Vec<f32> = (0..d_model).map(|i| (i as f32 * 0.17) - 0.5).collect();
        let picks: Vec<(usize, f32)> = vec![(0, 0.5), (1, 0.3), (2, 0.2)];

        // Serial reference with direct CPU kernels.
        let mut expect = vec![0.0f32; d_model];
        let mut g = vec![0.0f32; d_ff];
        let mut u = vec![0.0f32; d_ff];
        let mut f = vec![0.0f32; d_ff];
        let mut d = vec![0.0f32; d_model];
        for (e, w) in &picks {
            k::matvec_tensor(&gates[*e], &hidden, &mut g, d_ff, d_model);
            k::matvec_tensor(&ups[*e], &hidden, &mut u, d_ff, d_model);
            k::silu_mul_f32(&g, &u, &mut f);
            k::matvec_tensor(&downs[*e], &f, &mut d, d_model, d_ff);
            for j in 0..d_model {
                expect[j] += w * d[j];
            }
        }

        // Split path: 334‰ of 3 picks → 1 pick on the "GPU" branch
        // (which falls back to the same CPU kernels here), 2 on the
        // CPU branch.
        let mut out = vec![0.0f32; d_model];
        let mut gate_buf = vec![0.0f32; d_ff];
        let mut up_buf = vec![0.0f32; d_ff];
        let mut ff_buf = vec![0.0f32; d_ff];
        let mut down_buf = vec![0.0f32; d_model];
        moe_ffn_routed_split_exec(
            &hidden, &gates, &ups, &downs, &picks, d_model, d_ff, &mut out, &mut gate_buf,
            &mut up_buf, &mut ff_buf, &mut down_buf, 334,
        );

        for j in 0..d_model {
            assert!(
                (out[j] - expect[j]).abs() <= 1e-4 * (1.0 + expect[j].abs()),
                "lane {j}: split {} vs serial {}",
                out[j],
                expect[j]
            );
        }
    }

    /// Grouped-expert chunk FFN parity: `moe_ffn_chunk_into_parts`
    /// must match `n_tokens` sequential `moe_ffn_one_into_parts`
    /// calls within fp-association tolerance (same kernels; only the
    /// per-token accumulation order over experts changes). Exercises
    /// routing, grouping, the batched dispatch fallback, shared
    /// expert, and the sigmoid shared-router gate.
    #[test]
    fn chunk_ffn_matches_per_token_loop() {
        let d_model = 8usize;
        let d_ff = 4usize;
        let n_experts = 4usize;
        let top_k = 2usize;
        let n_tokens = 5usize;
        let mk = |e: usize, rows: usize, cols: usize, tag: &str| {
            let vals: Vec<f32> = (0..rows * cols)
                .map(|i| ((e * 17 + i * 5) % 11) as f32 * 0.07 - 0.35)
                .collect();
            tensor_f16_from_rowmajor(&vals, rows, cols, &format!("{tag}.e{e}"))
        };
        let router_vals: Vec<f32> = (0..n_experts * d_model)
            .map(|i| ((i * 13) % 7) as f32 * 0.11 - 0.3)
            .collect();
        let router = tensor_f16_from_rowmajor(&router_vals, n_experts, d_model, "router");
        let gates: Vec<Tensor> = (0..n_experts).map(|e| mk(e, d_ff, d_model, "g")).collect();
        let ups: Vec<Tensor> = (0..n_experts).map(|e| mk(e, d_ff, d_model, "u")).collect();
        let downs: Vec<Tensor> = (0..n_experts).map(|e| mk(e, d_model, d_ff, "d")).collect();
        let sh_gate = mk(9, d_ff, d_model, "shg");
        let sh_up = mk(10, d_ff, d_model, "shu");
        let sh_down = mk(11, d_model, d_ff, "shd");
        let sh_router = tensor_f16_from_rowmajor(
            &(0..d_model).map(|i| (i as f32) * 0.03 - 0.1).collect::<Vec<_>>(),
            1,
            d_model,
            "shr",
        );

        let hidden_rows: Vec<f32> = (0..n_tokens * d_model)
            .map(|i| ((i * 23) % 19) as f32 * 0.04 - 0.35)
            .collect();

        // Serial reference: per-token moe_ffn_one_into_parts.
        let mut expect = vec![0.0f32; n_tokens * d_model];
        let mut out = vec![0.0f32; d_model];
        let mut gate_buf = vec![0.0f32; d_ff];
        let mut up_buf = vec![0.0f32; d_ff];
        let mut ff_buf = vec![0.0f32; d_ff];
        let mut down_buf = vec![0.0f32; d_model];
        let mut logits = vec![0.0f32; n_experts];
        let mut picks: Vec<(usize, f32)> = Vec::with_capacity(top_k);
        for t in 0..n_tokens {
            moe_ffn_one_into_parts(
                &hidden_rows[t * d_model..(t + 1) * d_model],
                &router,
                &gates,
                &ups,
                &downs,
                Some(&sh_gate),
                Some(&sh_up),
                Some(&sh_down),
                Some(&sh_router),
                d_model,
                d_ff,
                n_experts,
                top_k,
                &mut out,
                &mut gate_buf,
                &mut up_buf,
                &mut ff_buf,
                &mut down_buf,
                &mut logits,
                &mut picks,
            );
            expect[t * d_model..(t + 1) * d_model].copy_from_slice(&out);
        }

        // Chunked path.
        let mut got = vec![0.0f32; n_tokens * d_model];
        moe_ffn_chunk_into_parts(
            &hidden_rows,
            n_tokens,
            &router,
            &gates,
            &ups,
            &downs,
            Some(&sh_gate),
            Some(&sh_up),
            Some(&sh_down),
            Some(&sh_router),
            d_model,
            d_ff,
            n_experts,
            top_k,
            &mut got,
        );

        for i in 0..n_tokens * d_model {
            assert!(
                (got[i] - expect[i]).abs() <= 1e-4 * (1.0 + expect[i].abs()),
                "elem {i}: chunk {} vs serial {}",
                got[i],
                expect[i]
            );
        }
    }

    /// The split's GPU-set sizing clamps to leave at least one pick
    /// on each branch for every legal permille value.
    #[test]
    fn split_exec_gpu_set_sizing_clamps() {
        for (len, split, expect_gpu) in [
            (2usize, 1u32, 1usize),   // rounds to 0 → clamped up
            (2, 999, 1),              // rounds to 2 → clamped down
            (8, 250, 2),
            (8, 500, 4),
            (3, 334, 1),
        ] {
            let n_gpu = ((len as u32 * split + 500) / 1000).clamp(1, len as u32 - 1) as usize;
            assert_eq!(n_gpu, expect_gpu, "len={len} split={split}");
        }
    }

    /// Build a small F16 `[m, k]` Tensor from explicit row-major f32 values.
    fn tensor_f16_from_rowmajor(values: &[f32], m: usize, k: usize, name: &str) -> Tensor {
        assert_eq!(values.len(), m * k);
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for v in values {
            bytes.extend_from_slice(&f16::from_f32(*v).to_le_bytes());
        }
        Tensor {
            device: Device::Cpu,
            dtype: Dtype::F16,
            shape: vec![m as u64, k as u64],
            strides: vec![k as i64, 1],
            storage: TStorage::CpuOwned(bytes.into()),
            name: name.into(),
        }
    }

    // ----- route_topk -----------------------------------------------------

    #[test]
    fn route_topk_uniform_logits_yields_uniform_weights() {
        // 4 experts, router is all-zeros → all logits = 0 → softmax = 0.25
        // each → top-2 renormalize → 0.5 each.
        let router = tensor_f16_from_rowmajor(&vec![0.0; 4 * 3], 4, 3, "router");
        let hidden = vec![1.0, 2.0, 3.0];
        let routed = route_topk(&hidden, &router, 4, 2);
        assert_eq!(routed.picks.len(), 2);
        for (_, w) in &routed.picks {
            assert!(
                (*w - 0.5).abs() < 1e-2,
                "uniform → 0.5 each after renorm, got {w}"
            );
        }
        // Sum to 1.0.
        let sum: f32 = routed.picks.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 1e-4, "top-K weights must sum to 1.0");
    }

    #[test]
    fn route_topk_picks_highest_router_logit_expert_first() {
        // Build a router where expert 2 has all-1.0 weights, others
        // are all-0.0. With hidden = [1, 1, 1] → logits = [0, 0, 3, 0].
        // softmax → expert 2 dominates → picked first.
        let mut router_vals = vec![0.0f32; 4 * 3];
        for j in 0..3 {
            router_vals[2 * 3 + j] = 1.0;
        }
        let router = tensor_f16_from_rowmajor(&router_vals, 4, 3, "router");
        let hidden = vec![1.0, 1.0, 1.0];
        let routed = route_topk(&hidden, &router, 4, 2);
        assert_eq!(routed.picks[0].0, 2, "highest-logit expert picked first");
        // First pick's weight must be the larger of the two.
        assert!(
            routed.picks[0].1 > routed.picks[1].1,
            "ordering must be descending"
        );
    }

    #[test]
    fn route_topk_top1_yields_single_weight_one() {
        let router = tensor_f16_from_rowmajor(&vec![0.0; 4 * 3], 4, 3, "router");
        let hidden = vec![1.0, 2.0, 3.0];
        let routed = route_topk(&hidden, &router, 4, 1);
        assert_eq!(routed.picks.len(), 1);
        assert!(
            (routed.picks[0].1 - 1.0).abs() < 1e-4,
            "top-1 renormalizes to 1.0"
        );
    }

    #[test]
    fn route_topk_full_k_equals_full_softmax() {
        // top_k = n_experts → the renormalized weights equal the
        // original softmax probabilities.
        let mut router_vals = vec![0.0f32; 3 * 2];
        router_vals[1 * 2] = 1.0; // expert 1 gets a boost
        let router = tensor_f16_from_rowmajor(&router_vals, 3, 2, "router");
        let hidden = vec![1.0, 1.0];
        let routed = route_topk(&hidden, &router, 3, 3);
        assert_eq!(routed.picks.len(), 3);
        let sum: f32 = routed.picks.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    // ----- expert_view ----------------------------------------------------

    #[test]
    fn expert_view_q4k_uses_block_geometry_for_offset() {
        // Q4_K_M is the most common Mixtral quant. Per-expert byte
        // count = (d_out * d_in / 256) * 144. Construct a fake
        // [n_experts=2, d_out=8, d_in=256] tensor (so each expert
        // is 8*256/256 * 144 = 8 * 144 = 1152 bytes) and verify
        // expert 0 / expert 1 slices land at the correct offsets.
        let n_experts = 2usize;
        let d_out = 8usize;
        let d_in = 256usize;
        let bytes_per_expert =
            Dtype::Q4_KRaw.byte_size((d_out * d_in) as u64) as usize;
        assert_eq!(bytes_per_expert, 1152);

        // Build bytes where expert 0's first byte is 0x42, expert
        // 1's first byte is 0x99 — lets us pin the slice math by
        // checking the first byte of each view.
        let total = n_experts * bytes_per_expert;
        let mut bytes = vec![0u8; total];
        bytes[0] = 0x42;
        bytes[bytes_per_expert] = 0x99;
        let big = Tensor {
            device: Device::Cpu,
            dtype: Dtype::Q4_KRaw,
            shape: vec![d_in as u64, d_out as u64, n_experts as u64],
            strides: vec![(d_out * d_in) as i64, d_in as i64, 1],
            storage: TStorage::CpuOwned(bytes.into()),
            name: "q4k_exps".into(),
        };

        let e0 = expert_view(&big, 0, d_out, d_in);
        assert_eq!(e0.dtype, Dtype::Q4_KRaw);
        assert_eq!(e0.shape, vec![d_out as u64, d_in as u64]);
        assert_eq!(e0.storage.len_bytes(), bytes_per_expert);
        assert_eq!(e0.storage.as_bytes()[0], 0x42);

        let e1 = expert_view(&big, 1, d_out, d_in);
        assert_eq!(e1.storage.len_bytes(), bytes_per_expert);
        assert_eq!(e1.storage.as_bytes()[0], 0x99);
    }

    #[test]
    fn expert_view_iq4_xs_uses_block_geometry_for_offset() {
        // IQ4_XS: 256-element block, 136 bytes per block. A
        // [4 experts, 4 out, 256 in] tensor has bytes_per_expert =
        // (4*256/256) * 136 = 4*136 = 544.
        let n_experts = 4usize;
        let d_out = 4usize;
        let d_in = 256usize;
        let bytes_per_expert =
            Dtype::IQ4_XSRaw.byte_size((d_out * d_in) as u64) as usize;
        assert_eq!(bytes_per_expert, 544);
        let mut bytes = vec![0u8; n_experts * bytes_per_expert];
        for (e, marker) in [(0, 0x11u8), (1, 0x22), (2, 0x33), (3, 0x44)] {
            bytes[e * bytes_per_expert] = marker;
        }
        let big = Tensor {
            device: Device::Cpu,
            dtype: Dtype::IQ4_XSRaw,
            shape: vec![d_in as u64, d_out as u64, n_experts as u64],
            strides: vec![(d_out * d_in) as i64, d_in as i64, 1],
            storage: TStorage::CpuOwned(bytes.into()),
            name: "iq4xs_exps".into(),
        };
        for (e, marker) in [(0, 0x11u8), (1, 0x22), (2, 0x33), (3, 0x44)] {
            let v = expert_view(&big, e, d_out, d_in);
            assert_eq!(
                v.storage.as_bytes()[0],
                marker,
                "expert {e} sliced to wrong offset"
            );
        }
    }

    #[test]
    fn expert_view_slices_correct_bytes() {
        // 3D tensor with 2 experts × 2×2 each. Bytes:
        //   expert 0: [a, b, c, d]
        //   expert 1: [e, f, g, h]
        let vals = vec![
            1.0, 2.0, 3.0, 4.0, // expert 0
            5.0, 6.0, 7.0, 8.0, // expert 1
        ];
        let mut bytes = Vec::with_capacity(vals.len() * 2);
        for v in &vals {
            bytes.extend_from_slice(&f16::from_f32(*v).to_le_bytes());
        }
        let big = Tensor {
            device: Device::Cpu,
            dtype: Dtype::F16,
            shape: vec![2, 2, 2],
            strides: vec![4, 2, 1],
            storage: TStorage::CpuOwned(bytes.into()),
            name: "exps".into(),
        };
        let e0 = expert_view(&big, 0, 2, 2);
        let e0_slice = e0.storage.cast_slice::<u8>();
        // First 8 bytes (4 × f16 = 8 bytes) match expert 0.
        let expected_0: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect();
        assert_eq!(e0_slice, expected_0.as_slice());

        let e1 = expert_view(&big, 1, 2, 2);
        let e1_slice = e1.storage.cast_slice::<u8>();
        let expected_1: Vec<u8> = [5.0f32, 6.0, 7.0, 8.0]
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect();
        assert_eq!(e1_slice, expected_1.as_slice());
    }

    // ----- moe_ffn_one ----------------------------------------------------

    /// Build a complete MoE block from explicit weight values. Per-
    /// expert FFNs are constructed by concatenating expert 0's bytes
    /// then expert 1's bytes (etc.) in the same layout as
    /// [`expert_view`] expects.
    fn build_synth_moe_block(
        d_model: usize,
        d_ff: usize,
        n_experts: usize,
        router_vals: Vec<f32>,
        gate_per_expert: Vec<Vec<f32>>,
        up_per_expert: Vec<Vec<f32>>,
        down_per_expert: Vec<Vec<f32>>,
    ) -> LlamaMoeBlockWeights {
        assert_eq!(router_vals.len(), n_experts * d_model);
        assert_eq!(gate_per_expert.len(), n_experts);
        assert_eq!(up_per_expert.len(), n_experts);
        assert_eq!(down_per_expert.len(), n_experts);

        // Concatenate per-expert F16 bytes for w_gate_exps:
        // expert 0 (d_ff × d_model) then expert 1, etc.
        let make_exps =
            |per_expert: &Vec<Vec<f32>>, d_out: usize, d_in: usize, name: &str| -> Tensor {
                let mut bytes = Vec::with_capacity(n_experts * d_out * d_in * 2);
                for e in per_expert {
                    assert_eq!(e.len(), d_out * d_in);
                    for v in e {
                        bytes.extend_from_slice(&f16::from_f32(*v).to_le_bytes());
                    }
                }
                Tensor {
                    device: Device::Cpu,
                    dtype: Dtype::F16,
                    shape: vec![d_in as u64, d_out as u64, n_experts as u64],
                    strides: vec![(d_out * d_in) as i64, d_in as i64, 1],
                    storage: TStorage::CpuOwned(bytes.into()),
                    name: name.into(),
                }
            };

        // Construct with empty per-expert vecs first, then slice
        // them from the 3D parents — field initializers can't see
        // each other so this two-step is unavoidable. Mirrors the
        // load-time wiring in `LlamaWeights::from_gguf`.
        let mut block = LlamaMoeBlockWeights {
            attn_norm: vec![1.0; d_model],
            w_q: tensor_f16_from_rowmajor(&vec![0.0; d_model * d_model], d_model, d_model, "wq"),
            w_k: tensor_f16_from_rowmajor(&vec![0.0; d_model * d_model], d_model, d_model, "wk"),
            w_v: tensor_f16_from_rowmajor(&vec![0.0; d_model * d_model], d_model, d_model, "wv"),
            w_o: tensor_f16_from_rowmajor(&vec![0.0; d_model * d_model], d_model, d_model, "wo"),
            b_q: None,
            b_k: None,
            b_v: None,
            q_norm: None,
            k_norm: None,
            ffn_norm: vec![1.0; d_model],
            router: tensor_f16_from_rowmajor(&router_vals, n_experts, d_model, "router"),
            w_gate_exps: make_exps(&gate_per_expert, d_ff, d_model, "gate_exps"),
            w_up_exps: make_exps(&up_per_expert, d_ff, d_model, "up_exps"),
            w_down_exps: make_exps(&down_per_expert, d_model, d_ff, "down_exps"),
            w_gate_shared: None,
            w_up_shared: None,
            w_down_shared: None,
            shared_router: None,
            gate_per_expert: Vec::new(),
            up_per_expert: Vec::new(),
            down_per_expert: Vec::new(),
        };
        block.gate_per_expert = (0..n_experts)
            .map(|e| crate::moe::expert_view(&block.w_gate_exps, e, d_ff, d_model))
            .collect();
        block.up_per_expert = (0..n_experts)
            .map(|e| crate::moe::expert_view(&block.w_up_exps, e, d_ff, d_model))
            .collect();
        block.down_per_expert = (0..n_experts)
            .map(|e| crate::moe::expert_view(&block.w_down_exps, e, d_model, d_ff))
            .collect();
        block
    }

    /// Reference implementation of one expert's SwiGLU FFN —
    /// gate @ h → silu → * up @ h → down @ . Matches the
    /// canonical Llama-family FFN.
    fn ref_swiglu(
        hidden: &[f32],
        gate: &[f32],
        up: &[f32],
        down: &[f32],
        d_model: usize,
        d_ff: usize,
    ) -> Vec<f32> {
        let mut gate_out = vec![0f32; d_ff];
        let mut up_out = vec![0f32; d_ff];
        for i in 0..d_ff {
            for j in 0..d_model {
                gate_out[i] += gate[i * d_model + j] * hidden[j];
                up_out[i] += up[i * d_model + j] * hidden[j];
            }
        }
        // SiLU: x * sigmoid(x) = x / (1 + e^-x)
        let mut ff = vec![0f32; d_ff];
        for i in 0..d_ff {
            let s = gate_out[i] / (1.0 + (-gate_out[i]).exp());
            ff[i] = s * up_out[i];
        }
        let mut out = vec![0f32; d_model];
        for i in 0..d_model {
            for j in 0..d_ff {
                out[i] += down[i * d_ff + j] * ff[j];
            }
        }
        out
    }

    #[test]
    fn moe_ffn_one_top1_matches_single_expert_reference() {
        // 2 experts, top-1. Build the router so expert 1 wins
        // decisively. moe_ffn_one's output should equal a direct
        // single-expert SwiGLU on expert 1's weights.
        let d_model = 4;
        let d_ff = 3;
        let n_experts = 2;
        // Router: expert 0 zeros, expert 1 large positive → expert 1 wins.
        let mut router_vals = vec![0f32; n_experts * d_model];
        for j in 0..d_model {
            router_vals[1 * d_model + j] = 5.0;
        }
        // Per-expert weights (small distinct values so we can spot drift).
        let gate0: Vec<f32> = (0..d_ff * d_model).map(|i| 0.01 * i as f32).collect();
        let gate1: Vec<f32> = (0..d_ff * d_model).map(|i| 0.02 * i as f32 + 0.1).collect();
        let up0: Vec<f32> = (0..d_ff * d_model).map(|i| 0.005 * i as f32).collect();
        let up1: Vec<f32> = (0..d_ff * d_model).map(|i| 0.015 * i as f32 + 0.05).collect();
        let down0: Vec<f32> = (0..d_model * d_ff).map(|i| 0.001 * i as f32).collect();
        let down1: Vec<f32> = (0..d_model * d_ff).map(|i| 0.003 * i as f32 + 0.02).collect();
        let block = build_synth_moe_block(
            d_model,
            d_ff,
            n_experts,
            router_vals,
            vec![gate0, gate1.clone()],
            vec![up0, up1.clone()],
            vec![down0, down1.clone()],
        );

        let hidden = vec![0.1, 0.2, 0.3, 0.4];
        let got = moe_ffn_one(&hidden, &block, d_model, d_ff, n_experts, 1);
        // top-1 + renorm → weight 1.0 on expert 1 → output = ref_swiglu(expert 1).
        let expected = ref_swiglu(&hidden, &gate1, &up1, &down1, d_model, d_ff);

        // F16 round-trip + scalar f32 accumulation: tolerance ~1e-2.
        for i in 0..d_model {
            assert!(
                (got[i] - expected[i]).abs() < 1e-2,
                "out[{i}]: got {} vs expected {}",
                got[i],
                expected[i]
            );
        }
    }

    #[test]
    fn moe_ffn_one_top2_yields_weighted_sum_of_experts() {
        // 2 experts, top-2, uniform router → both weights = 0.5
        // → output = 0.5 * expert0_swiglu + 0.5 * expert1_swiglu.
        let d_model = 2;
        let d_ff = 2;
        let n_experts = 2;
        let router_vals = vec![0f32; n_experts * d_model]; // uniform
        let gate0 = vec![1.0, 0.0, 0.0, 1.0]; // identity-ish
        let gate1 = vec![0.5, 0.5, 0.5, 0.5];
        let up0 = vec![1.0, 1.0, 1.0, 1.0];
        let up1 = vec![2.0, 0.0, 0.0, 2.0];
        let down0 = vec![1.0, 0.0, 0.0, 1.0];
        let down1 = vec![0.5, 0.5, 0.5, 0.5];
        let block = build_synth_moe_block(
            d_model,
            d_ff,
            n_experts,
            router_vals,
            vec![gate0.clone(), gate1.clone()],
            vec![up0.clone(), up1.clone()],
            vec![down0.clone(), down1.clone()],
        );
        let hidden = vec![0.5, 0.5];

        let got = moe_ffn_one(&hidden, &block, d_model, d_ff, n_experts, 2);
        let ref0 = ref_swiglu(&hidden, &gate0, &up0, &down0, d_model, d_ff);
        let ref1 = ref_swiglu(&hidden, &gate1, &up1, &down1, d_model, d_ff);
        // Uniform router → 0.5/0.5 split.
        for i in 0..d_model {
            let expected = 0.5 * ref0[i] + 0.5 * ref1[i];
            assert!(
                (got[i] - expected).abs() < 5e-2,
                "out[{i}]: got {} vs 0.5*({} + {}) = {}",
                got[i],
                ref0[i],
                ref1[i],
                expected
            );
        }
    }

    #[test]
    fn moe_ffn_one_into_matches_allocating_variant() {
        // Parity: `moe_ffn_one_into` writing into caller scratch
        // must produce bit-identical output to `moe_ffn_one`.
        // Pins the allocation-free path against the convenience
        // wrapper that tests + one-shot callers use.
        let d_model = 4;
        let d_ff = 6;
        let n_experts = 4;
        let top_k = 2;
        let mut router_vals = vec![0f32; n_experts * d_model];
        for j in 0..d_model {
            router_vals[1 * d_model + j] = 0.7;
            router_vals[3 * d_model + j] = 0.3;
        }
        let per_expert = |seed: f32, n: usize| -> Vec<f32> {
            (0..n).map(|i| seed + i as f32 * 0.01).collect()
        };
        let block = build_synth_moe_block(
            d_model,
            d_ff,
            n_experts,
            router_vals,
            vec![
                per_expert(0.1, d_ff * d_model),
                per_expert(0.2, d_ff * d_model),
                per_expert(0.3, d_ff * d_model),
                per_expert(0.4, d_ff * d_model),
            ],
            vec![
                per_expert(0.05, d_ff * d_model),
                per_expert(0.15, d_ff * d_model),
                per_expert(0.25, d_ff * d_model),
                per_expert(0.35, d_ff * d_model),
            ],
            vec![
                per_expert(0.02, d_model * d_ff),
                per_expert(0.12, d_model * d_ff),
                per_expert(0.22, d_model * d_ff),
                per_expert(0.32, d_model * d_ff),
            ],
        );
        let hidden = vec![0.1, 0.2, 0.3, 0.4];

        let allocating = moe_ffn_one(&hidden, &block, d_model, d_ff, n_experts, top_k);

        let mut into_out = vec![0f32; d_model];
        let mut gate = vec![0f32; d_ff];
        let mut up = vec![0f32; d_ff];
        let mut ff = vec![0f32; d_ff];
        let mut down = vec![0f32; d_model];
        let mut logits = vec![0f32; n_experts];
        let mut picks: Vec<(usize, f32)> = Vec::with_capacity(top_k);
        moe_ffn_one_into(
            &hidden,
            &block,
            d_model,
            d_ff,
            n_experts,
            top_k,
            &mut into_out,
            &mut gate,
            &mut up,
            &mut ff,
            &mut down,
            &mut logits,
            &mut picks,
        );

        for i in 0..d_model {
            assert!(
                (allocating[i] - into_out[i]).abs() < 1e-6,
                "out[{i}]: allocating={} into={}",
                allocating[i],
                into_out[i]
            );
        }
    }

    #[test]
    fn moe_ffn_one_into_reuse_across_calls_is_deterministic() {
        // Calling _into multiple times with the same scratch must
        // produce identical results — i.e. the scratch zero-init
        // at the top of the function actually wipes stale values
        // from the previous call. Pin via two back-to-back calls
        // with the same inputs.
        let d_model = 2;
        let d_ff = 2;
        let n_experts = 2;
        let top_k = 2;
        let router_vals = vec![0f32; n_experts * d_model];
        let block = build_synth_moe_block(
            d_model,
            d_ff,
            n_experts,
            router_vals,
            vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.5, 0.5, 0.5, 0.5]],
            vec![vec![1.0, 1.0, 1.0, 1.0], vec![2.0, 0.0, 0.0, 2.0]],
            vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.5, 0.5, 0.5, 0.5]],
        );
        let hidden = vec![0.5, 0.5];

        let mut out_a = vec![0f32; d_model];
        let mut gate = vec![0f32; d_ff];
        let mut up = vec![0f32; d_ff];
        let mut ff = vec![0f32; d_ff];
        let mut down = vec![0f32; d_model];
        let mut logits = vec![0f32; n_experts];
        let mut picks: Vec<(usize, f32)> = Vec::with_capacity(top_k);
        // Pre-poison the output to verify the zero-init at the top
        // of _into actually fires (otherwise stale values would
        // leak into the result).
        out_a[0] = 999.0;
        out_a[1] = -999.0;
        moe_ffn_one_into(
            &hidden,
            &block,
            d_model,
            d_ff,
            n_experts,
            top_k,
            &mut out_a,
            &mut gate,
            &mut up,
            &mut ff,
            &mut down,
            &mut logits,
            &mut picks,
        );

        let mut out_b = vec![0f32; d_model];
        moe_ffn_one_into(
            &hidden,
            &block,
            d_model,
            d_ff,
            n_experts,
            top_k,
            &mut out_b,
            &mut gate,
            &mut up,
            &mut ff,
            &mut down,
            &mut logits,
            &mut picks,
        );

        assert_eq!(
            out_a, out_b,
            "scratch reuse must produce deterministic output"
        );
        // Sanity: not the poisoned values.
        assert!(out_a[0].abs() < 100.0);
    }

    #[test]
    fn moe_ffn_one_with_shared_expert_adds_unweighted_contribution() {
        // 2 experts, top-1 → expert 1 selected with weight 1.0.
        // Add a shared expert. Output = ref_swiglu(expert 1) +
        // ref_swiglu(shared).
        let d_model = 2;
        let d_ff = 2;
        let n_experts = 2;
        let mut router_vals = vec![0f32; n_experts * d_model];
        for j in 0..d_model {
            router_vals[1 * d_model + j] = 5.0;
        }
        let gate_picked = vec![1.0, 0.0, 0.0, 1.0];
        let up_picked = vec![1.0, 1.0, 1.0, 1.0];
        let down_picked = vec![1.0, 0.0, 0.0, 1.0];
        let mut block = build_synth_moe_block(
            d_model,
            d_ff,
            n_experts,
            router_vals,
            vec![vec![0.0; d_ff * d_model], gate_picked.clone()],
            vec![vec![0.0; d_ff * d_model], up_picked.clone()],
            vec![vec![0.0; d_model * d_ff], down_picked.clone()],
        );

        let gate_shared = vec![0.5, 0.5, 0.5, 0.5];
        let up_shared = vec![1.0, 1.0, 1.0, 1.0];
        let down_shared = vec![0.5, 0.5, 0.5, 0.5];
        block.w_gate_shared = Some(tensor_f16_from_rowmajor(&gate_shared, d_ff, d_model, "g_sh"));
        block.w_up_shared = Some(tensor_f16_from_rowmajor(&up_shared, d_ff, d_model, "u_sh"));
        block.w_down_shared = Some(tensor_f16_from_rowmajor(&down_shared, d_model, d_ff, "d_sh"));

        let hidden = vec![0.5, 0.5];
        let got = moe_ffn_one(&hidden, &block, d_model, d_ff, n_experts, 1);

        let ref_routed = ref_swiglu(&hidden, &gate_picked, &up_picked, &down_picked, d_model, d_ff);
        let ref_shared = ref_swiglu(&hidden, &gate_shared, &up_shared, &down_shared, d_model, d_ff);
        for i in 0..d_model {
            let expected = ref_routed[i] + ref_shared[i];
            assert!(
                (got[i] - expected).abs() < 5e-2,
                "out[{i}]: got {} vs expected {} (routed {} + shared {})",
                got[i],
                expected,
                ref_routed[i],
                ref_shared[i]
            );
        }
    }
}
