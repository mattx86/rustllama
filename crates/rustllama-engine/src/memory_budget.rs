//! Memory-budget planner (roadmap Phase 3): one honest division of
//! physical RAM instead of several independently-OOM-capable knobs.
//!
//! Colibrì's insight, adopted here: derive cache budgets from what is
//! *actually available* minus a real projection of everything the
//! engine still has to allocate — KV cache, recurrent state, scratch,
//! an OS reserve — so the box never thrashes or OOMs because two
//! budgets each "fit" on their own.
//!
//! The planner runs in `CpuEngine::load_inner` after the model and KV
//! backend exist (their true byte counts are known, not estimated)
//! and before `lock_model_into_ram` / the expert-cache pre-pin (its
//! output feeds both). It is only consulted when
//! `[inference].memory_budget = "auto"` (promoted to
//! `RUSTLLAMA_MEMORY_BUDGET`); manual mode keeps the explicit knobs
//! authoritative.
//!
//! v1 scope: the planner drives the **expert-cache budget** (the knob
//! with the widest damage range) and reports the projection it used.
//! Driving the pagelock tier and KV pool sizing from the same plan is
//! follow-up work — pagelock already self-clamps against available
//! RAM *minus the expert budget* (see `pagelock.rs`), so the two
//! never jointly overcommit.

/// Inputs to [`plan`]. All byte counts; gather them *after* model +
/// KV construction so they're measured, not estimated.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryFacts {
    /// Physical RAM installed.
    pub total_phys: u64,
    /// Physical RAM available right now (post model load).
    pub avail_phys: u64,
    /// KV-cache buffers as allocated (mostly not yet faulted in —
    /// treated as still-to-come, which double-counts any already
    /// faulted pages: a deliberate safety margin).
    pub kv_bytes: u64,
    /// DeltaNet recurrent state (hybrid models; 0 otherwise).
    pub dn_bytes: u64,
    /// Forward-pass scratch projection (logits rows, FFN buffers,
    /// sampler state). See [`scratch_projection`].
    pub scratch_bytes: u64,
    /// Total bytes of the registered routed-expert pool — the most
    /// the expert cache could ever usefully pin.
    pub expert_pool_bytes: u64,
    /// Prefix-cache pool reserve: worst-case bytes the snapshot pool
    /// may grow to (snapshot count × per-anchor size). Without this
    /// the pool grew AFTER the expert budget was granted, on RAM the
    /// planner had already given away.
    pub pool_reserve_bytes: u64,
    /// Explicit `moe_expert_cache_mb` in bytes; in auto mode a
    /// non-zero value acts as a *cap* on the computed budget.
    pub explicit_cap_bytes: u64,
}

/// Output of [`plan`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryPlan {
    /// Expert-pin cache budget to install.
    pub expert_cache_bytes: u64,
    /// OS / other-process reserve kept untouched.
    pub reserve_bytes: u64,
    /// What was left to divide after reserve + projections.
    pub spendable_bytes: u64,
    /// True when the projection left nothing to spend — the caller
    /// should warn that the box is already at its limit.
    pub exhausted: bool,
}

/// Fraction of spendable RAM the expert cache may claim. The
/// remainder is deliberately left to the OS page cache — cold
/// experts still stream through it, and starving it would turn every
/// cache miss into a raw disk read.
const EXPERT_SHARE_NUM: u64 = 3;
const EXPERT_SHARE_DEN: u64 = 4;

/// Divide available RAM into an expert-cache budget plus reserves.
/// Pure function — unit-tested against synthetic memory profiles.
pub fn plan(f: &MemoryFacts) -> MemoryPlan {
    let two_gb: u64 = 2 * 1024 * 1024 * 1024;
    // Reserve: at least 2 GiB or 1/8 of the box for the OS, other
    // processes, and the page cache's own metadata.
    let reserve = two_gb.max(f.total_phys / 8);
    let committed = f
        .kv_bytes
        .saturating_add(f.dn_bytes)
        .saturating_add(f.scratch_bytes)
        .saturating_add(f.pool_reserve_bytes);
    let spendable = f.avail_phys.saturating_sub(reserve.saturating_add(committed));
    let mut expert = (spendable / EXPERT_SHARE_DEN).saturating_mul(EXPERT_SHARE_NUM);
    expert = expert.min(f.expert_pool_bytes);
    if f.explicit_cap_bytes > 0 {
        expert = expert.min(f.explicit_cap_bytes);
    }
    MemoryPlan {
        expert_cache_bytes: expert,
        reserve_bytes: reserve,
        spendable_bytes: spendable,
        exhausted: spendable == 0,
    }
}

/// Project the forward pass's scratch footprint from model geometry:
/// a handful of vocab-sized logits/softmax rows, the per-layer FFN
/// and attention scratch, plus a fixed allowance for tokenizer /
/// sampler / channel buffers. Deliberately generous — the planner
/// prefers to under-spend by ~100 MB over paging mid-decode.
pub fn scratch_projection(vocab_size: usize, d_model: usize, d_ff: usize) -> u64 {
    let f32b = 4u64;
    let vocab_rows = 6 * vocab_size as u64 * f32b;
    let ffn_rows = 8 * d_ff as u64 * f32b;
    let model_rows = 32 * d_model as u64 * f32b;
    let fixed = 128 * 1024 * 1024;
    vocab_rows + ffn_rows + model_rows + fixed
}

/// `RUSTLLAMA_MEMORY_BUDGET` — `"auto"` enables the planner at engine
/// load. Anything else (or unset) keeps manual budgets. Read per call
/// (load-time only, not hot-path).
pub fn auto_memory_budget_enabled() -> bool {
    std::env::var("RUSTLLAMA_MEMORY_BUDGET")
        .map(|v| v.trim().eq_ignore_ascii_case("auto"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;

    fn base_facts() -> MemoryFacts {
        // A 16 GB box after loading a big MoE: ~5 GB still available,
        // 1 GB of KV to come, 8 GB expert pool on disk.
        MemoryFacts {
            total_phys: 16 * GB,
            avail_phys: 5 * GB,
            kv_bytes: GB,
            dn_bytes: 60 * 1024 * 1024,
            scratch_bytes: 300 * 1024 * 1024,
            expert_pool_bytes: 8 * GB,
            pool_reserve_bytes: 0,
            explicit_cap_bytes: 0,
        }
    }

    #[test]
    fn pool_reserve_shrinks_spendable() {
        let without = plan(&base_facts());
        let with = plan(&MemoryFacts {
            pool_reserve_bytes: GB,
            ..base_facts()
        });
        assert!(with.spendable_bytes + GB == without.spendable_bytes);
        assert!(with.expert_cache_bytes < without.expert_cache_bytes);
    }

    #[test]
    fn tight_box_spends_three_quarters_of_headroom() {
        let f = base_facts();
        let p = plan(&f);
        // reserve = max(2 GB, 16/8 GB) = 2 GB;
        // spendable = 5 GB − 2 GB − ~1.36 GB ≈ 1.65 GB; expert = 3/4 of that.
        assert!(!p.exhausted);
        assert!(p.expert_cache_bytes > GB, "got {}", p.expert_cache_bytes);
        assert!(p.expert_cache_bytes < 2 * GB, "got {}", p.expert_cache_bytes);
        // Never exceeds what remains after reserve + projections.
        assert!(p.expert_cache_bytes <= p.spendable_bytes);
    }

    #[test]
    fn expert_budget_capped_by_pool_size() {
        // Roomy box, tiny expert pool: budget is the pool, not 3/4 of RAM.
        let f = MemoryFacts {
            total_phys: 128 * GB,
            avail_phys: 100 * GB,
            expert_pool_bytes: 3 * GB,
            ..base_facts()
        };
        let p = plan(&f);
        assert_eq!(p.expert_cache_bytes, 3 * GB);
    }

    #[test]
    fn explicit_value_acts_as_cap_in_auto_mode() {
        let f = MemoryFacts {
            explicit_cap_bytes: 512 * 1024 * 1024,
            ..base_facts()
        };
        let p = plan(&f);
        assert_eq!(p.expert_cache_bytes, 512 * 1024 * 1024);
    }

    #[test]
    fn exhausted_box_yields_zero_budget_and_flags_it() {
        // Available barely covers the reserve: nothing to spend, and
        // the plan says so instead of overcommitting.
        let f = MemoryFacts {
            avail_phys: 2 * GB,
            ..base_facts()
        };
        let p = plan(&f);
        assert_eq!(p.expert_cache_bytes, 0);
        assert!(p.exhausted);
    }

    #[test]
    fn reserve_scales_with_big_boxes() {
        let f = MemoryFacts {
            total_phys: 128 * GB,
            avail_phys: 100 * GB,
            ..base_facts()
        };
        let p = plan(&f);
        assert_eq!(p.reserve_bytes, 16 * GB); // 128/8 > 2 GB floor
    }

    #[test]
    fn scratch_projection_scales_with_geometry() {
        let small = scratch_projection(32_000, 2048, 5632);
        let big = scratch_projection(152_000, 4096, 12288);
        assert!(big > small);
        // Fixed allowance keeps even tiny models honest.
        assert!(small > 128 * 1024 * 1024);
    }
}
