//! Placement-sweep candidate generator. Given a model's per-layer
//! shape + the bytes-per-weight cost of its quantization, enumerates
//! the `n_gpu_layers` values worth measuring, filters to those that
//! fit a VRAM budget (with a configurable headroom), and returns
//! them in "most GPU first" order so the engine can stop measuring
//! as soon as a candidate beats its currently-best generation tok/s.
//!
//! No measurement is done here — that lives behind a callback once
//! the engine integration lands. This module is the dependency-free
//! deterministic half: pure arithmetic over the model's dimensions
//! and the cached `DeviceFingerprint`, fully unit-testable in mock
//! mode without an Intel GPU.
//!
//! The arithmetic mirrors the per-layer Llama-family tensor shapes
//! (Q / K / V / O projections + gate / up / down FFN + attn / FFN
//! norms). Weight counts are computed in float weights, then scaled
//! by the format's bytes-per-weight ratio.

use crate::PlacementPlan;

/// Compact per-model dimensions the placement sweep needs. Mirror
/// of `rustllama_models::llama_config::LlamaConfig` minus tokenizer
/// and runtime fields. We keep this small + local rather than
/// depending on `rustllama-models` (which pulls in the kernel
/// crates) — placement is a pure-arithmetic module.
///
/// MoE awareness: `n_experts`, `n_experts_used`, and
/// `n_experts_shared` describe Qwen3-MoE / Mixtral / DeepSeek-V3
/// routing. `n_experts == 0` means a dense transformer block;
/// `n_experts >= 2` means the FFN is replaced by a router + N
/// per-expert FFNs, all of which must be resident regardless of
/// `n_experts_used` (top-K is a compute optimization, not a memory
/// one). Construct via [`ModelDims::dense`] for dense models or set
/// the MoE fields manually for MoE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelDims {
    pub n_layers: u32,
    pub d_model: u32,
    pub d_ff: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub vocab_size: u32,
    /// MoE routed expert count. `0` for dense models. Each expert
    /// carries a full gate / up / down weight triplet of the same
    /// shape a dense FFN would have.
    pub n_experts: u32,
    /// Top-K routed experts per token. `0` for dense models. Does
    /// not affect VRAM (all experts resident); recorded so the
    /// tuner can bucket compute-cost predictions per (n_experts,
    /// n_experts_used) shape and so cache entries survive cleanly
    /// across architecture variants with the same n_experts but
    /// different top-K.
    pub n_experts_used: u32,
    /// DeepSeek-V3-style always-active shared experts. `0` for
    /// dense models and Mixtral / Qwen3-MoE; `1+` adds a parallel
    /// FFN of the same shape as a routed expert that runs every
    /// token regardless of router output.
    pub n_experts_shared: u32,
}

impl ModelDims {
    /// Construct dense-model dimensions. MoE fields zero-init. New
    /// callers should prefer this over field-init so adding future
    /// MoE-awareness fields doesn't break existing sites.
    pub fn dense(
        n_layers: u32,
        d_model: u32,
        d_ff: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        vocab_size: u32,
    ) -> Self {
        Self {
            n_layers,
            d_model,
            d_ff,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            n_experts: 0,
            n_experts_used: 0,
            n_experts_shared: 0,
        }
    }

    /// Q-projection inner dim (`n_heads * head_dim`). Equals
    /// `d_model` for standard MHA; smaller for GQA models like
    /// Llama 3 / Qwen2.5-Coder where K/V heads are grouped.
    pub fn d_q(self) -> u32 {
        self.n_heads * self.head_dim
    }

    /// K/V projection inner dim (`n_kv_heads * head_dim`).
    pub fn d_kv(self) -> u32 {
        self.n_kv_heads * self.head_dim
    }

    /// `true` when this block participates in MoE routing.
    pub fn is_moe(self) -> bool {
        self.n_experts >= 2
    }

    /// Float weight count per transformer block: Q + K + V + O
    /// projection matrices, plus the FFN — either dense (3 matrices
    /// of `(d_ff, d_model)`) or MoE (router `(n_experts, d_model)`
    /// + `n_experts + n_experts_shared` parallel FFNs each carrying
    /// the same 3-matrix structure as a dense FFN). Norms are f32
    /// and small (`2 * d_model`), counted separately in
    /// [`block_vram_bytes`].
    pub fn block_weights(self) -> u64 {
        let d = self.d_model as u64;
        let d_q = self.d_q() as u64;
        let d_kv = self.d_kv() as u64;
        let d_ff = self.d_ff as u64;
        // Projections: shape (out, in) — count is out*in.
        let proj_qkvo = d_q * d + 2 * d_kv * d + d * d_q;
        if self.is_moe() {
            // Router: (n_experts, d_model). Tiny — for Mixtral
            // d_model=4096 / n_experts=8 it's 32K weights total.
            let router = self.n_experts as u64 * d;
            // Routed experts: each is a 3-matrix FFN of dense shape.
            // All must be VRAM-resident even though only `n_experts_used`
            // activate per token.
            let routed_experts = self.n_experts as u64 * 3 * d_ff * d;
            // Shared experts (DeepSeek-V3): same shape, always-active.
            let shared_experts = self.n_experts_shared as u64 * 3 * d_ff * d;
            proj_qkvo + router + routed_experts + shared_experts
        } else {
            // FFN: gate/up are (d_ff, d); down is (d, d_ff).
            let ffn = 3 * d_ff * d;
            proj_qkvo + ffn
        }
    }

    /// Float weight count of the LM head: `[d_model, vocab_size]`.
    /// Tied embeddings reuse the input embedding matrix, but for
    /// placement budget purposes treat the head as a distinct VRAM
    /// cost — it's almost always the single biggest tensor in a
    /// modern coding model (vocab_size 128K × d_model 4096 dominates).
    pub fn lm_head_weights(self) -> u64 {
        self.d_model as u64 * self.vocab_size as u64
    }

    /// Float weight count of the input embedding table:
    /// `[vocab_size, d_model]`. Same byte count as `lm_head_weights`;
    /// kept separate so a future "embedding on CPU" override can
    /// price it independently of the LM head.
    pub fn embedding_weights(self) -> u64 {
        self.vocab_size as u64 * self.d_model as u64
    }
}

/// Bytes per weight for the v1-locked quantization formats. The
/// values for K-quants are the on-disk block-byte ratios — what
/// the engine's `Dtype::*Raw` storage actually consumes. F16 is
/// dequantized to 16 bits at rest (the engine keeps activations in
/// f32 but doesn't dequantize weights on load).
#[allow(non_camel_case_types)] // names mirror the GGUF / Dtype naming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightQuant {
    /// 16-bit float, 2 bytes per weight.
    F16,
    /// Q8_0: 34-byte blocks of 32 weights = 1.0625 bytes per weight.
    Q8_0,
    /// Q4_K_M: 144-byte super-blocks of 256 weights ≈ 0.5625 bytes per weight.
    Q4_K_M,
    /// Q5_K_M: 176-byte super-blocks of 256 weights ≈ 0.6875 bytes per weight.
    Q5_K_M,
}

impl WeightQuant {
    /// Bytes per individual weight, as a float (because K-quants
    /// don't land on integer byte boundaries). Used scaled by the
    /// total weight count to estimate VRAM.
    pub fn bytes_per_weight(self) -> f32 {
        match self {
            WeightQuant::F16 => 2.0,
            WeightQuant::Q8_0 => 34.0 / 32.0,
            WeightQuant::Q4_K_M => 144.0 / 256.0,
            WeightQuant::Q5_K_M => 176.0 / 256.0,
        }
    }

    /// Parse the quant name as the engine / GGUF reports it (e.g.
    /// "Q4_K_M", "Q4_K_S" — we treat both as `Q4_K_M`-cost for
    /// placement purposes since per-weight bytes are identical).
    /// Returns `None` for unsupported names.
    pub fn from_name(s: &str) -> Option<Self> {
        let s = s.trim();
        match s.to_ascii_uppercase().as_str() {
            "F16" | "FP16" => Some(WeightQuant::F16),
            "Q8_0" => Some(WeightQuant::Q8_0),
            "Q4_K" | "Q4_K_M" | "Q4_K_S" => Some(WeightQuant::Q4_K_M),
            "Q5_K" | "Q5_K_M" | "Q5_K_S" => Some(WeightQuant::Q5_K_M),
            _ => None,
        }
    }
}

/// Convert a float weight count to a byte cost given the format.
fn bytes_for(weights: u64, quant: WeightQuant) -> u64 {
    // f32 conversion is fine here — the worst-case `vocab_size *
    // d_model` for a 128K-vocab 4096-dim model fits comfortably in
    // the f32 mantissa (2^29 ≈ 5e8 < 2^24 * 2^53 ULP envelope).
    ((weights as f64) * (quant.bytes_per_weight() as f64)).ceil() as u64
}

/// Per-block VRAM cost given a quantization choice. Includes the
/// 8 weight matrices (Q/K/V/O + gate/up/down + attn/ffn norms);
/// norms are tiny `f32` scalars (`2 * d_model * 4` bytes) so they
/// don't get the quant scaling.
pub fn block_vram_bytes(dims: ModelDims, quant: WeightQuant) -> u64 {
    let quantized = bytes_for(dims.block_weights(), quant);
    // attn_norm + ffn_norm: 2 vectors of `d_model` f32 values each.
    let norms = 2u64 * dims.d_model as u64 * 4;
    quantized + norms
}

/// LM head VRAM cost: `[d_model, vocab_size]` weights at the quant
/// scaling. Note: GGUF models often keep `output.weight` in F16
/// even when the rest of the model is Q4_K_M — for v1 we don't
/// model that subtlety; placement assumes uniform quant. Adding
/// `output_quant: Option<WeightQuant>` is a v1.1 follow-up.
pub fn lm_head_vram_bytes(dims: ModelDims, quant: WeightQuant) -> u64 {
    bytes_for(dims.lm_head_weights(), quant)
}

/// Input embedding-table VRAM cost: same shape as the LM head but
/// independently sized so a future "embeddings on CPU" override
/// can charge it to system RAM instead.
pub fn embedding_vram_bytes(dims: ModelDims, quant: WeightQuant) -> u64 {
    bytes_for(dims.embedding_weights(), quant)
}

/// KV-cache VRAM cost for a given context window. For each layer
/// we keep `2 * n_kv_heads * max_ctx * head_dim` f16 elements
/// (the default KV dtype). `f16` = 2 bytes. Multiplied across all
/// `n_layers` regardless of `n_gpu_layers` because the engine
/// keeps the full KV cache device-resident even for CPU-placed
/// layers — the alternative (per-layer dtype-juggling) is messy
/// and the savings are small at v1 context sizes.
pub fn kv_cache_vram_bytes(dims: ModelDims, max_ctx: u32) -> u64 {
    let per_layer =
        2u64 * dims.n_kv_heads as u64 * max_ctx as u64 * dims.head_dim as u64 * 2;
    per_layer * dims.n_layers as u64
}

/// Default sweep granularity for `n_gpu_layers`. Step `8` is what
/// the plan prescribes; smaller steps balloon the search-space
/// without much benefit on real models where adjacent layer counts
/// produce near-identical tok/s.
pub const DEFAULT_SWEEP_STEP: u32 = 8;

/// Generate the candidate `PlacementPlan` set to measure. Each
/// candidate is `{ n_gpu_layers: N, overrides: [] }` for N from
/// `n_layers` down to `0` in steps of `step`, plus `0` itself, plus
/// the all-on-GPU sentinel `n_layers + 1` (which conventionally
/// also puts the LM head on the GPU).
///
/// Filters to candidates whose estimated VRAM cost stays within
/// `vram_budget_bytes - headroom_bytes`. Returns the filtered list
/// in "most GPU first" order: an empty list signals "no candidate
/// fits the budget" (caller falls back to fully-CPU placement).
///
/// The `step` argument lets callers do a `Quick` sweep (large step
/// — 4 candidates for a 32-layer model) vs. a `Thorough` sweep
/// (step `1` — every layer count). `step == 0` is treated as `1`.
pub fn candidate_placements(
    dims: ModelDims,
    quant: WeightQuant,
    vram_budget_bytes: u64,
    headroom_bytes: u64,
    max_ctx: u32,
    step: u32,
) -> Vec<PlacementPlan> {
    let step = step.max(1);
    let budget = vram_budget_bytes.saturating_sub(headroom_bytes);
    let block = block_vram_bytes(dims, quant);
    let lm_head = lm_head_vram_bytes(dims, quant);
    let embed = embedding_vram_bytes(dims, quant);
    let kv = kv_cache_vram_bytes(dims, max_ctx);

    // Build the candidate `n_gpu_layers` set in "most GPU first"
    // order:
    //   - `n_layers + 1` (sentinel: all blocks + LM head on GPU)
    //   - `n_layers` (all blocks, LM head + embeddings on CPU)
    //   - then step down from `n_layers - step` to `0` on the step grid
    //   - always include `0` as the all-CPU baseline if the grid skips it
    let mut counts: Vec<u32> = Vec::new();
    counts.push(dims.n_layers + 1);
    counts.push(dims.n_layers);
    let mut n = dims.n_layers;
    while n > step {
        n -= step;
        counts.push(n);
    }
    if *counts.last().unwrap() != 0 {
        counts.push(0);
    }
    counts.dedup();

    counts
        .into_iter()
        .filter_map(|n_gpu| {
            let blocks_on_gpu = n_gpu.min(dims.n_layers) as u64;
            let weights_cost = blocks_on_gpu * block
                + if n_gpu > dims.n_layers { lm_head + embed } else { 0 };
            // KV cache always device-resident regardless of which
            // layers are on GPU — see `kv_cache_vram_bytes`.
            let total = weights_cost + kv;
            if total <= budget {
                Some(PlacementPlan {
                    n_gpu_layers: n_gpu,
                    overrides: Vec::new(),
                })
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Llama-2-7B-style dimensions, used as the canonical sanity
    /// shape across these tests (32 layers, 4096 hidden, 11008 ffn,
    /// 32 heads, GQA off).
    fn llama2_7b_dims() -> ModelDims {
        ModelDims::dense(32, 4096, 11008, 32, 32, 128, 32000)
    }

    /// Mixtral-8x7B-style MoE: same per-block attn/FFN footprint
    /// as Llama-2-7B but with 8 routed experts top-2, no shared.
    /// Used to pin the MoE VRAM accounting.
    fn mixtral_8x7b_dims() -> ModelDims {
        ModelDims {
            n_layers: 32,
            d_model: 4096,
            d_ff: 14336,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            vocab_size: 32000,
            n_experts: 8,
            n_experts_used: 2,
            n_experts_shared: 0,
        }
    }

    #[test]
    fn bytes_per_weight_v1_quants() {
        assert!((WeightQuant::F16.bytes_per_weight() - 2.0).abs() < 1e-6);
        assert!((WeightQuant::Q8_0.bytes_per_weight() - 34.0 / 32.0).abs() < 1e-6);
        assert!((WeightQuant::Q4_K_M.bytes_per_weight() - 144.0 / 256.0).abs() < 1e-6);
        assert!((WeightQuant::Q5_K_M.bytes_per_weight() - 176.0 / 256.0).abs() < 1e-6);
    }

    #[test]
    fn quant_from_name_handles_common_variants() {
        assert_eq!(WeightQuant::from_name("F16"), Some(WeightQuant::F16));
        assert_eq!(WeightQuant::from_name("f16"), Some(WeightQuant::F16));
        assert_eq!(WeightQuant::from_name("Q8_0"), Some(WeightQuant::Q8_0));
        assert_eq!(WeightQuant::from_name("Q4_K_M"), Some(WeightQuant::Q4_K_M));
        assert_eq!(WeightQuant::from_name("Q4_K_S"), Some(WeightQuant::Q4_K_M));
        assert_eq!(WeightQuant::from_name("Q5_K_M"), Some(WeightQuant::Q5_K_M));
        assert_eq!(WeightQuant::from_name("IQ3_S"), None);
    }

    #[test]
    fn block_weights_matches_hand_calculation() {
        // Llama-2-7B per-block: 4 projections (each 4096*4096) +
        // 3 FFN matrices (each 4096*11008).
        let d: u64 = 4096;
        let d_ff: u64 = 11008;
        let expected = 4 * d * d + 3 * d * d_ff;
        assert_eq!(llama2_7b_dims().block_weights(), expected);
    }

    #[test]
    fn block_vram_q4km_is_reasonable_for_7b() {
        // Q4_K_M on a 7B Llama2 block: weights only ≈
        // (4*4096^2 + 3*4096*11008) * 144/256 ≈ 113 MB. Plus 2*4096*4
        // bytes of f32 norms ≈ 32 KB. Floor & ceil envelope.
        let bytes = block_vram_bytes(llama2_7b_dims(), WeightQuant::Q4_K_M);
        let mib = bytes / (1024 * 1024);
        // Should be in the [100, 130] MB range.
        assert!(
            (100..130).contains(&mib),
            "Q4_K_M 7B block VRAM ≈ {mib} MiB outside [100, 130]"
        );
    }

    #[test]
    fn candidate_placements_empty_when_budget_zero() {
        let dims = llama2_7b_dims();
        let plans = candidate_placements(
            dims,
            WeightQuant::Q4_K_M,
            0,
            0,
            2048,
            DEFAULT_SWEEP_STEP,
        );
        // Even 0 GPU layers carries the KV cache cost, so a zero
        // budget rejects every candidate.
        assert!(plans.is_empty(), "expected empty plans, got {plans:?}");
    }

    #[test]
    fn candidate_placements_all_fit_with_generous_budget() {
        let dims = llama2_7b_dims();
        // 32 GiB is more than enough for Q4_K_M 7B at 2K context.
        let plans = candidate_placements(
            dims,
            WeightQuant::Q4_K_M,
            32 * 1024 * 1024 * 1024,
            256 * 1024 * 1024,
            2048,
            DEFAULT_SWEEP_STEP,
        );
        // With step=8: {33, 32, 24, 16, 8, 0} should all fit.
        let counts: Vec<u32> = plans.iter().map(|p| p.n_gpu_layers).collect();
        assert_eq!(counts, vec![33, 32, 24, 16, 8, 0]);
    }

    #[test]
    fn candidate_placements_step_1_is_thorough() {
        let dims = llama2_7b_dims();
        let plans = candidate_placements(
            dims,
            WeightQuant::Q4_K_M,
            32 * 1024 * 1024 * 1024,
            256 * 1024 * 1024,
            2048,
            1,
        );
        // step=1 yields every layer count from n_layers+1 down to 0.
        let counts: Vec<u32> = plans.iter().map(|p| p.n_gpu_layers).collect();
        let expected: Vec<u32> = (0..=33u32).rev().collect();
        assert_eq!(counts, expected);
    }

    #[test]
    fn candidate_placements_zero_step_clamps_to_one() {
        let dims = llama2_7b_dims();
        // step=0 should behave identically to step=1.
        let plans_zero = candidate_placements(
            dims,
            WeightQuant::Q4_K_M,
            32 * 1024 * 1024 * 1024,
            256 * 1024 * 1024,
            2048,
            0,
        );
        let plans_one = candidate_placements(
            dims,
            WeightQuant::Q4_K_M,
            32 * 1024 * 1024 * 1024,
            256 * 1024 * 1024,
            2048,
            1,
        );
        assert_eq!(plans_zero.len(), plans_one.len());
    }

    #[test]
    fn candidate_placements_filters_oversize_layers() {
        // Iris Xe-style budget: 4 GiB shared LPDDR. Q4_K_M 7B with
        // 2K context wants ~3.8 GiB weights + ~512 MiB KV cache —
        // most-GPU candidates won't fit, but the lower layer counts
        // and full-CPU should.
        let dims = llama2_7b_dims();
        let plans = candidate_placements(
            dims,
            WeightQuant::Q4_K_M,
            4 * 1024 * 1024 * 1024,
            256 * 1024 * 1024,
            2048,
            DEFAULT_SWEEP_STEP,
        );
        assert!(!plans.is_empty(), "Iris-Xe budget should fit some plans");
        // Sorted descending.
        for w in plans.windows(2) {
            assert!(w[0].n_gpu_layers > w[1].n_gpu_layers, "not descending: {plans:?}");
        }
        // n=0 always fits (just the KV cache).
        let last = plans.last().unwrap();
        assert_eq!(last.n_gpu_layers, 0);
    }

    #[test]
    fn kv_cache_scales_with_context() {
        let dims = llama2_7b_dims();
        let kv_2k = kv_cache_vram_bytes(dims, 2048);
        let kv_8k = kv_cache_vram_bytes(dims, 8192);
        assert_eq!(kv_8k, kv_2k * 4);
    }

    #[test]
    fn dense_constructor_zeros_moe_fields() {
        let dims = ModelDims::dense(32, 4096, 11008, 32, 32, 128, 32000);
        assert_eq!(dims.n_experts, 0);
        assert_eq!(dims.n_experts_used, 0);
        assert_eq!(dims.n_experts_shared, 0);
        assert!(!dims.is_moe(), "n_experts=0 must report not-MoE");
    }

    #[test]
    fn is_moe_treats_single_expert_as_dense() {
        // A few synthetic test GGUFs set expert_count=1 to exercise
        // the router code path without ballooning weights. For VRAM
        // accounting we conservatively bucket those as dense — one
        // "expert" carries the same weight footprint as a dense FFN.
        let mut dims = ModelDims::dense(32, 4096, 11008, 32, 32, 128, 32000);
        dims.n_experts = 1;
        assert!(!dims.is_moe());
    }

    #[test]
    fn block_weights_grows_with_expert_count() {
        let mixtral = mixtral_8x7b_dims();
        // A "dense-shaped" reference using mixtral's same per-expert FFN size.
        let dense_ref =
            ModelDims::dense(32, 4096, 14336, 32, 8, 128, 32000);
        let moe = mixtral.block_weights();
        let dense = dense_ref.block_weights();
        // 8 experts → FFN portion is ~8× dense; the projection
        // portion is identical; the router is tiny (4096*8 = 32K).
        // Total MoE/dense ratio for Mixtral is dominated by FFN.
        let ratio = moe as f64 / dense as f64;
        assert!(
            ratio > 6.0 && ratio < 8.5,
            "8-expert MoE block weights / dense ratio = {ratio}; expected ~7-8"
        );
    }

    #[test]
    fn block_weights_includes_router() {
        // Router contributes exactly n_experts * d_model weights.
        let mut moe = mixtral_8x7b_dims();
        let with_router = moe.block_weights();
        // Removing the router (artificially zero n_experts → falls
        // into dense branch — not what we want), instead compute by
        // hand and verify additivity.
        let d = moe.d_model as u64;
        let d_q = moe.d_q() as u64;
        let d_kv = moe.d_kv() as u64;
        let d_ff = moe.d_ff as u64;
        let proj = d_q * d + 2 * d_kv * d + d * d_q;
        let router = moe.n_experts as u64 * d;
        let experts = moe.n_experts as u64 * 3 * d_ff * d;
        assert_eq!(with_router, proj + router + experts);
        // Adding shared experts adds exactly n_shared FFNs.
        moe.n_experts_shared = 2;
        let with_shared = moe.block_weights();
        assert_eq!(with_shared, with_router + 2 * 3 * d_ff * d);
    }

    #[test]
    fn block_vram_q4km_mixtral_block_is_reasonable() {
        // Mixtral-8x7B-style block at Q4_K_M:
        //   proj_qkvo  ≈ 4096*4096 + 2*(1024*4096) + 4096*4096 ≈ 41.9M weights
        //   router     ≈ 8 * 4096 = 32K weights
        //   8 experts × 3 × 14336 × 4096 ≈ 1.41B weights
        //   total      ≈ 1.45B weights × 0.5625 B/wt ≈ 815 MiB
        // Plus 2 norm vectors (tiny). Expect the result in the
        // 750-900 MiB envelope per block.
        let bytes = block_vram_bytes(mixtral_8x7b_dims(), WeightQuant::Q4_K_M);
        let mib = bytes / (1024 * 1024);
        assert!(
            (750..900).contains(&mib),
            "Mixtral block VRAM ≈ {mib} MiB outside [750, 900]"
        );
    }

    #[test]
    fn candidate_placements_mixtral_q4km_needs_high_vram() {
        // Mixtral Q4_K_M weights total: ~26 GiB. On a 16 GiB GPU,
        // most-GPU candidates must be rejected — only low-layer
        // counts (or 0) survive.
        let dims = mixtral_8x7b_dims();
        let plans = candidate_placements(
            dims,
            WeightQuant::Q4_K_M,
            16 * 1024 * 1024 * 1024,
            256 * 1024 * 1024,
            2048,
            DEFAULT_SWEEP_STEP,
        );
        assert!(!plans.is_empty(), "n=0 must always fit");
        // The all-on-GPU and full-layers sentinels must be filtered out.
        let counts: Vec<u32> = plans.iter().map(|p| p.n_gpu_layers).collect();
        assert!(
            !counts.contains(&(dims.n_layers + 1)),
            "Mixtral all-on-GPU must not fit 16 GiB: counts={counts:?}"
        );
        assert!(
            !counts.contains(&dims.n_layers),
            "Mixtral all-layers (LM head CPU) still oversize for 16 GiB"
        );
        // n=0 is always present.
        assert_eq!(counts.last(), Some(&0));
    }

    #[test]
    fn candidate_placements_mixtral_fits_with_generous_vram() {
        // 48 GiB workstation card (A6000-class) easily fits Mixtral Q4_K_M.
        let dims = mixtral_8x7b_dims();
        let plans = candidate_placements(
            dims,
            WeightQuant::Q4_K_M,
            48u64 * 1024 * 1024 * 1024,
            256 * 1024 * 1024,
            8192,
            DEFAULT_SWEEP_STEP,
        );
        let counts: Vec<u32> = plans.iter().map(|p| p.n_gpu_layers).collect();
        // All-on-GPU sentinel + all-layers candidate both fit.
        assert!(counts.contains(&(dims.n_layers + 1)));
        assert!(counts.contains(&dims.n_layers));
        // Descending order preserved.
        for w in plans.windows(2) {
            assert!(w[0].n_gpu_layers > w[1].n_gpu_layers);
        }
    }
}
