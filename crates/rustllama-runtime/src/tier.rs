//! Heterogeneous compute-tier scheduling: an ordered tier list with
//! capacity-based spillover (GPU/Metal → NPU → CPU).
//!
//! The pure policy core of the N-tier scheduler from
//! `docs/npu-heterogeneous.md`. It generalizes the VRAM-fit placement (fill the
//! GPU by VRAM, spill the rest to the CPU) to ≥3 tiers: fill the highest-
//! priority tier until its capacity is exhausted, spill the remainder to the
//! next, and let the last tier (CPU) absorb everything left. The engine's
//! `placement_auto` consumes it with the measured per-tier capacities (free
//! VRAM, NPU memory, "the rest" for CPU); the config crate re-exports it for
//! `[inference.placement].tier_order`. Pure + unit-tested, no device dependency.

use serde::{Deserialize, Serialize};

/// A compute tier in the heterogeneous scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TierKind {
    /// A GPU (CUDA / SYCL) or Apple Metal device.
    #[serde(alias = "gpu", alias = "metal", alias = "gpu/metal")]
    GpuMetal,
    /// An NPU (Intel / AMD), reached through its vendor runtime.
    Npu,
    /// The CPU / host-RAM tier (the unbounded fallback — always last).
    Cpu,
}

/// The default tier priority order: throughput-first (GPU fills first), then
/// the efficient NPU offload, then the CPU fallback.
pub fn default_tier_order() -> Vec<TierKind> {
    vec![TierKind::GpuMetal, TierKind::Npu, TierKind::Cpu]
}

/// Assign each placeable unit (e.g. a model layer's weight bytes, in placement
/// order) to a tier by capacity-based spillover: fill tier 0 until the next
/// unit would exceed `tier_caps[0]`, then tier 1, and so on; the LAST tier is
/// unbounded and absorbs everything remaining. Returns the tier index per unit.
///
/// A unit larger than a bounded tier's remaining room advances to the next tier
/// (and ultimately the unbounded last tier), so a single oversized layer never
/// wedges placement. With one tier, everything lands on it. `tier_caps` is in
/// priority order and should align with the active [`TierKind`] order.
pub fn spillover_assign(unit_bytes: &[u64], tier_caps: &[u64]) -> Vec<usize> {
    let n_tiers = tier_caps.len().max(1);
    let last = n_tiers - 1;
    let mut out = Vec::with_capacity(unit_bytes.len());
    let mut tier = 0usize;
    let mut used = 0u64;
    for &b in unit_bytes {
        // Advance past any bounded tier that can't hold this unit.
        while tier < last && used.saturating_add(b) > tier_caps[tier] {
            tier += 1;
            used = 0;
        }
        out.push(tier);
        used = used.saturating_add(b);
    }
    out
}

/// Order tiers fastest→slowest from MEASURED throughput, the spillover priority
/// the autotuner sets after benching each tier (CPU + NPU + GPU/Metal) in tok/s.
/// Descending by tok/s; a stable sort keeps the input order for ties. This
/// replaces the static `default_tier_order` assumption with the measured
/// ranking — e.g. a weak iGPU that benches slower than the CPU drops below it,
/// and an NPU slots wherever its measured speed lands.
pub fn order_tiers_by_perf(measured: &[(TierKind, f64)]) -> Vec<TierKind> {
    let mut v: Vec<(TierKind, f64)> = measured.to_vec();
    // Descending tok/s; NaN sinks to the bottom. Stable for ties.
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v.into_iter().map(|(k, _)| k).collect()
}

/// Count how many units landed on each tier (len `n_tiers`).
pub fn tier_counts(assignment: &[usize], n_tiers: usize) -> Vec<usize> {
    let mut counts = vec![0usize; n_tiers.max(1)];
    for &t in assignment {
        if t < counts.len() {
            counts[t] += 1;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_in_priority_order_then_spills() {
        let units = vec![10u64; 6];
        let a = spillover_assign(&units, &[25, 20, 0]);
        assert_eq!(a, vec![0, 0, 1, 1, 2, 2]);
        assert_eq!(tier_counts(&a, 3), vec![2, 2, 2]);
    }

    #[test]
    fn last_tier_is_unbounded() {
        let units = vec![100u64; 4];
        let a = spillover_assign(&units, &[0, 0, 0]);
        assert_eq!(a, vec![2, 2, 2, 2]);
    }

    #[test]
    fn single_tier_takes_all() {
        let a = spillover_assign(&[1, 2, 3], &[0]);
        assert_eq!(a, vec![0, 0, 0]);
    }

    #[test]
    fn oversized_unit_advances_but_still_places() {
        let a = spillover_assign(&[20, 50, 10], &[40, 100, 0]);
        assert_eq!(a, vec![0, 1, 1]);
    }

    #[test]
    fn default_order_is_gpu_npu_cpu() {
        assert_eq!(default_tier_order(), vec![TierKind::GpuMetal, TierKind::Npu, TierKind::Cpu]);
    }

    #[test]
    fn perf_ranking_orders_fastest_first() {
        // A weak GPU (30 tok/s) benches below the CPU (45) and NPU (60) → the
        // measured order is NPU > CPU > GPU, not the static GPU-first default.
        let measured = vec![
            (TierKind::GpuMetal, 30.0),
            (TierKind::Cpu, 45.0),
            (TierKind::Npu, 60.0),
        ];
        assert_eq!(
            order_tiers_by_perf(&measured),
            vec![TierKind::Npu, TierKind::Cpu, TierKind::GpuMetal]
        );
    }

    /// Two-tier case must match the GPU-fill-then-CPU cutoff the engine's
    /// placement uses: tier-0 count = largest prefix whose bytes fit the budget.
    #[test]
    fn two_tier_matches_vram_cutoff() {
        let per_layer = vec![10u64, 10, 10, 10, 10];
        let budget = 25u64;
        let a = spillover_assign(&per_layer, &[budget, 0]);
        let cutoff = a.iter().take_while(|&&t| t == 0).count();
        assert_eq!(cutoff, 2); // 20<=25, 30>25
        assert!(a[cutoff..].iter().all(|&t| t == 1)); // rest on CPU
    }
}
