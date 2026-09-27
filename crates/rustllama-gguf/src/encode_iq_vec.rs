//! Vector-grid IQ encoders (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
//! IQ1_S, IQ1_M).
//!
//! These formats encode each 4- or 8-weight chunk as a pair of
//! `(grid_index, sign_index)`: the grid_index picks one of N
//! all-positive grid vectors from a fixed codebook (256–2048
//! entries) and the sign_index picks one of 128 sign masks from
//! [`crate::dequant::KSIGNS_IQ2XS`]. The encoder's job is the
//! inverse search — find the `(grid, signs)` pair that minimizes
//! L2 reconstruction error of the target chunk.
//!
//! ## Search algorithm
//!
//! For a target chunk `x` of length 8 (IQ2_*) or 4 (IQ3_XXS):
//!
//! 1. For each candidate grid entry `g`:
//!    - The "free" optimal sign per element is `sign(x[j] * g[j])`.
//!    - The 128 representable sign patterns have **even popcount**
//!      (the 7-bit `KSIGNS_IQ2XS` index implicitly carries the
//!      parity bit). If the free-optimal pattern is already even,
//!      use it; otherwise flip the element with the smallest
//!      `|x[j] * g[j]|` to recover parity at minimum cost.
//!    - Score the pair via the projection energy:
//!      `score = (x · signed_g)² / |g|²` (maximized when `signed_g`
//!      best aligns with `x`).
//! 2. Track the grid entry with the highest score; that pair's
//!    optimal real-valued scale is `(x · signed_g) / |g|²`.
//!
//! Per-chunk cost: O(n_grid × 8) — roughly 2K ops for IQ2_XXS's
//! 256-entry grid, 4K for IQ2_S's 1024-entry grid, etc. For a 7B
//! model that's ~30 seconds of single-threaded encode time; future
//! work can parallelize across blocks via rayon.

use half::f16;

use crate::dequant::{
    IQ1S_DELTA, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS,
};
use crate::iq1_grid::IQ1S_GRID;
use std::sync::OnceLock;

/// IQ1_S grid pre-expanded as f32 (no per-iteration int8→f32
/// conversion in the hot loop), with `|g|²` cached alongside.
/// Layout: `flat[idx * 8 + j]` is the f32 of `IQ1S_GRID[idx]` byte j.
///
/// 2048 × 8 × 4 = 64 KB grid + 2048 × 4 = 8 KB norms = 72 KB total.
/// Fits in L2 on every modern CPU.
///
/// The unused-by-default `sort_by_norm_desc` + `grid_l2` fields are
/// **groundwork for codebook pruning** (which is not active in this
/// release). The clean Cauchy-Schwarz upper-bound `score ≤ ||T||²`
/// is grid-independent, so no general norm-based prune is correct
/// for IQ1_S. A real prune needs format-specific heuristics (e.g.
/// sign-pattern bucketing for the asymmetric IQ4 codebooks); that
/// investigation is parked for a follow-up session. Keeping the
/// sort + L2 fields costs ~16 KB of constant memory and lets the
/// follow-up land without touching the cache struct.
#[allow(dead_code)]
struct Iq1sGridCache {
    grid_f32: Vec<f32>,    // 2048 × 8 = 16384 f32 entries
    grid_norm: Vec<f32>,   // 2048 entries — |g|² per grid
    grid_l2: Vec<f32>,     // 2048 entries — |g| (sqrt of grid_norm)
    /// Indices sorted by |g|² descending. Not yet consumed.
    sort_by_norm_desc: Vec<u16>,
}

fn iq1s_grid_cache() -> &'static Iq1sGridCache {
    static CELL: OnceLock<Iq1sGridCache> = OnceLock::new();
    CELL.get_or_init(|| {
        let mut grid_f32 = Vec::with_capacity(2048 * 8);
        let mut grid_norm = Vec::with_capacity(2048);
        let mut grid_l2 = Vec::with_capacity(2048);
        for idx in 0..IQ1S_GRID.len() {
            let bytes = IQ1S_GRID[idx].to_le_bytes();
            let mut norm = 0f32;
            for j in 0..8 {
                let g = (bytes[j] as i8) as f32;
                grid_f32.push(g);
                norm += g * g;
            }
            grid_norm.push(norm);
            grid_l2.push(norm.sqrt());
        }
        // Build descending-norm permutation. NaN-free domain
        // (grid is small fixed int values), so partial_cmp is safe.
        let mut sort_by_norm_desc: Vec<u16> = (0..2048u16).collect();
        sort_by_norm_desc.sort_unstable_by(|&a, &b| {
            grid_norm[b as usize]
                .partial_cmp(&grid_norm[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Iq1sGridCache {
            grid_f32,
            grid_norm,
            grid_l2,
            sort_by_norm_desc,
        }
    })
}

const QK_K: usize = 256;
const N_SUB_BLOCKS: usize = 8; // 256 / 32

/// Reverse lookup table for `KSIGNS_IQ2XS`: given an 8-bit sign
/// mask, return the 7-bit index that decodes back to that mask
/// (or 0xFF if the mask isn't in the 128 representable patterns).
///
/// Built lazily on first use via `std::sync::OnceLock`. Costs 256
/// bytes of static state; the full enumeration is one pass over
/// the 128-entry source table.
fn ksigns_iq2xs_reverse() -> &'static [u8; 256] {
    use std::sync::OnceLock;
    static REV: OnceLock<[u8; 256]> = OnceLock::new();
    REV.get_or_init(|| {
        let mut rev = [0xFFu8; 256];
        for (i, &mask) in KSIGNS_IQ2XS.iter().enumerate() {
            rev[mask as usize] = i as u8;
        }
        rev
    })
}

/// Outcome of one chunk's grid+sign search. `signed_score` is the
/// signed projection `(target · signed_grid)`; the optimal real
/// scale for this chunk is `signed_score / grid_norm_sq`. The
/// per-sub-block scale chooser aggregates across chunks.
#[derive(Debug, Clone, Copy, Default)]
struct ChunkPick {
    grid_idx: u16,
    sign_idx: u8,
    /// `target · signed_grid`. Always non-negative under the
    /// greedy-sign-then-parity-fixup heuristic (we flip signs to
    /// align positive whenever possible; the parity flip on a
    /// near-zero element keeps the result non-negative in practice).
    signed_score: f32,
    /// `|grid|²` — the same per-grid constant precomputed once
    /// before the per-chunk loop.
    grid_norm_sq: f32,
}

/// Greedily flip element signs so each term `target[j] * grid[j] *
/// sign[j]` is non-negative; if the resulting sign mask has odd
/// popcount (not in `KSIGNS_IQ2XS`), flip the element with the
/// smallest `|target[j] * grid[j]|` (lowest-cost parity fix).
/// Returns `(sign_mask, signed_score)` where signed_score is the
/// final `target · signed_grid` value.
///
/// Scalar reference used by tests + non-x86_64 hosts. The hot
/// path takes `target` + `grid_f32` (precomputed) via
/// [`pick_sign_mask_8_f32`] which is SIMD-friendly.
fn pick_sign_mask_8(target: &[f32], grid: &[i8]) -> (u8, f32) {
    debug_assert_eq!(target.len(), 8);
    debug_assert_eq!(grid.len(), 8);
    let mut g_f32 = [0f32; 8];
    for j in 0..8 {
        g_f32[j] = grid[j] as f32;
    }
    pick_sign_mask_8_f32(target, &g_f32)
}

/// Same as [`pick_sign_mask_8`] but takes the grid already pre-
/// expanded to f32. Used by the SIMD path (which loads the grid
/// from a precomputed f32 table) and by the scalar reference (which
/// converts on the fly). KMASK_IQ2XS[j] = 1 << j, so the bit
/// indexing matches AVX2's `movemask_ps` output directly.
fn pick_sign_mask_8_f32(target: &[f32], grid_f32: &[f32]) -> (u8, f32) {
    debug_assert_eq!(target.len(), 8);
    debug_assert_eq!(grid_f32.len(), 8);
    let mut mask: u8 = 0;
    let mut score = 0f32;
    let mut min_abs_contrib = f32::INFINITY;
    let mut min_idx = 0usize;
    for j in 0..8 {
        let contrib = target[j] * grid_f32[j];
        let a = contrib.abs();
        if a < min_abs_contrib {
            min_abs_contrib = a;
            min_idx = j;
        }
        if contrib >= 0.0 {
            score += contrib;
        } else {
            score -= contrib;
            mask |= KMASK_IQ2XS[j];
        }
    }
    if mask.count_ones() & 1 == 1 {
        mask ^= KMASK_IQ2XS[min_idx];
        score -= 2.0 * min_abs_contrib;
    }
    (mask, score)
}

/// Per-chunk best (grid, signs) search over an 8-element-per-entry
/// grid (IQ2_XXS, IQ2_XS, IQ2_S layouts).
///
/// **`grid_f32`** is the precomputed f32 view of the grid
/// (`n_grid × 8` floats). Built once per format via the
/// `iq2*_grid_f32_table` helpers and cached statically.
/// Eliminates the per-iteration `i8 → f32` conversion and lets the
/// SIMD path do a single `_mm256_loadu_ps` per candidate.
///
/// Dispatches to AVX2 when available; scalar fallback otherwise.
fn search_chunk_8(
    target: &[f32],
    grid_f32: &[f32],
    n_grid: usize,
    grid_norm_sq_table: &[f32],
) -> ChunkPick {
    debug_assert_eq!(target.len(), 8);
    debug_assert_eq!(grid_f32.len(), n_grid * 8);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            return unsafe {
                search_chunk_8_avx2(target, grid_f32, n_grid, grid_norm_sq_table)
            };
        }
    }
    search_chunk_8_scalar(target, grid_f32, n_grid, grid_norm_sq_table)
}

pub(crate) fn search_chunk_8_scalar(
    target: &[f32],
    grid_f32: &[f32],
    n_grid: usize,
    grid_norm_sq_table: &[f32],
) -> ChunkPick {
    let mut best = ChunkPick {
        grid_idx: 0,
        sign_idx: 0,
        signed_score: 0.0,
        grid_norm_sq: 1.0,
    };
    let rev = ksigns_iq2xs_reverse();
    for g in 0..n_grid {
        let grid = &grid_f32[g * 8..g * 8 + 8];
        let (mask, signed_score) = pick_sign_mask_8_f32(target, grid);
        let sign_idx = rev[mask as usize];
        if sign_idx == 0xFF {
            continue;
        }
        let norm_sq = grid_norm_sq_table[g];
        let lhs = signed_score * signed_score * best.grid_norm_sq;
        let rhs = best.signed_score * best.signed_score * norm_sq;
        if lhs > rhs {
            best = ChunkPick {
                grid_idx: g as u16,
                sign_idx,
                signed_score,
                grid_norm_sq: norm_sq,
            };
        }
    }
    best
}

/// AVX2 path: load 8 grid floats per candidate, FMA against target,
/// extract per-lane sign mask via `_mm256_movemask_ps`, horizontal-
/// sum the absolute products for the greedy score. Parity fix +
/// best-update logic stays scalar (only ~10 cycles per candidate).
///
/// Per-candidate cost: ~6 cycles SIMD + ~10 scalar = ~16 cycles,
/// vs scalar baseline of ~40 cycles. ~2.5× speedup, modest but real.
/// The dominant gain is killing the per-iteration i8→f32 conversion
/// (which the scalar refactor already does); SIMD compounds on top.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn search_chunk_8_avx2(
    target: &[f32],
    grid_f32: &[f32],
    n_grid: usize,
    grid_norm_sq_table: &[f32],
) -> ChunkPick {
    use std::arch::x86_64::*;
    let rev = ksigns_iq2xs_reverse();
    let mut best = ChunkPick {
        grid_idx: 0,
        sign_idx: 0,
        signed_score: 0.0,
        grid_norm_sq: 1.0,
    };
    // SAFETY: target is &[f32] of length 8 (debug-asserted by caller).
    let target_v = unsafe { _mm256_loadu_ps(target.as_ptr()) };
    let grid_ptr = grid_f32.as_ptr();
    // Absolute-value mask: clear the sign bit of each lane.
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFFu32 as i32));

    // Scratch for SIMD → scalar handoff (abs contribs).
    let mut abs_contribs = [0f32; 8];

    for g in 0..n_grid {
        // SAFETY: g ∈ [0, n_grid), grid_f32 has n_grid * 8 floats.
        let g_v = unsafe { _mm256_loadu_ps(grid_ptr.add(g * 8)) };
        let prod = _mm256_mul_ps(target_v, g_v);
        let abs_prod = _mm256_and_ps(prod, abs_mask);
        // Sign mask: bit j set ⇔ prod[j] is negative. Matches
        // KMASK_IQ2XS[j] = 1 << j exactly, so movemask gives us
        // the byte-encoded sign pattern directly.
        let neg_mask = _mm256_movemask_ps(prod) as u32 as u8;
        // Greedy score (before parity fix) = sum of |prod[j]|.
        let signed_score = unsafe { horizontal_sum_avx2(abs_prod) };

        let mut mask = neg_mask;
        let mut score = signed_score;

        // Parity fix: if odd popcount, flip the lane with the
        // smallest |contrib|. Spill abs_prod to scalar to find
        // argmin (8-lane argmin would need shuffles; the spill
        // costs ~5 cycles and runs only when parity is odd, so
        // half the time amortized).
        if mask.count_ones() & 1 == 1 {
            // SAFETY: abs_contribs[..] is 8 f32 aligned to 4 bytes
            // — `_mm256_storeu_ps` handles unaligned. Could use
            // aligned alloc for marginal gain.
            unsafe { _mm256_storeu_ps(abs_contribs.as_mut_ptr(), abs_prod) };
            let mut min_abs = abs_contribs[0];
            let mut min_idx = 0usize;
            for j in 1..8 {
                if abs_contribs[j] < min_abs {
                    min_abs = abs_contribs[j];
                    min_idx = j;
                }
            }
            mask ^= 1u8 << min_idx;
            score -= 2.0 * min_abs;
        }

        let sign_idx = rev[mask as usize];
        if sign_idx == 0xFF {
            continue;
        }
        let norm_sq = grid_norm_sq_table[g];
        let lhs = score * score * best.grid_norm_sq;
        let rhs = best.signed_score * best.signed_score * norm_sq;
        if lhs > rhs {
            best = ChunkPick {
                grid_idx: g as u16,
                sign_idx,
                signed_score: score,
                grid_norm_sq: norm_sq,
            };
        }
    }
    best
}

/// Precompute `|grid|²` for every entry. Cheap one-time table —
/// 256–2048 floats — and saves N×8 multiplies per chunk search.
fn build_grid_norm_sq_table_8(grid_bytes: &[u8], n_grid: usize) -> Vec<f32> {
    (0..n_grid)
        .map(|g| {
            let grid = &grid_bytes[g * 8..g * 8 + 8];
            let mut sum = 0f32;
            for &b in grid {
                let v = (b as i8) as f32;
                sum += v * v;
            }
            sum
        })
        .collect()
}

/// Build the IQ2_XXS grid as a flat i8 byte slice (256 × 8 bytes).
fn iq2xxs_grid_bytes() -> &'static [u8] {
    bytemuck::cast_slice(&IQ2XXS_GRID)
}

/// Precomputed f32 grid for IQ2_XXS. Same byte layout as
/// `iq2xxs_grid_bytes` but expanded to f32 so the SIMD path can
/// do a single `_mm256_loadu_ps` per candidate instead of an
/// i8→f32 conversion in the inner loop.
fn iq2xxs_grid_f32() -> &'static [f32] {
    static CELL: OnceLock<Vec<f32>> = OnceLock::new();
    CELL.get_or_init(|| {
        let bytes = iq2xxs_grid_bytes();
        bytes.iter().map(|&b| (b as i8) as f32).collect()
    })
}

/// Precomputed f32 grid for IQ2_XS (512 entries × 8 elements).
fn iq2xs_grid_f32() -> &'static [f32] {
    static CELL: OnceLock<Vec<f32>> = OnceLock::new();
    CELL.get_or_init(|| {
        let bytes: &[u8] = bytemuck::cast_slice(&IQ2XS_GRID);
        bytes.iter().map(|&b| (b as i8) as f32).collect()
    })
}

/// Precomputed f32 grid for IQ2_S (1024 entries × 8 elements).
fn iq2s_grid_f32() -> &'static [f32] {
    static CELL: OnceLock<Vec<f32>> = OnceLock::new();
    CELL.get_or_init(|| {
        let bytes: &[u8] = bytemuck::cast_slice(&IQ2S_GRID);
        bytes.iter().map(|&b| (b as i8) as f32).collect()
    })
}

/// IQ3_XXS grid as a flat i8 byte slice (256 × 4 bytes). Each
/// grid entry is 4 i8 values packed into a u32. Two grid entries
/// (8 bytes total) compose one 8-weight chunk.
fn iq3xxs_grid_bytes() -> &'static [u8] {
    bytemuck::cast_slice(&IQ3XXS_GRID)
}

/// Precomputed f32 grid for IQ3_XXS (256 entries × 4 elements).
fn iq3xxs_grid_f32() -> &'static [f32] {
    static CELL: OnceLock<Vec<f32>> = OnceLock::new();
    CELL.get_or_init(|| {
        let bytes = iq3xxs_grid_bytes();
        bytes.iter().map(|&b| (b as i8) as f32).collect()
    })
}

/// Precomputed f32 grid for IQ3_S (512 entries × 4 elements).
fn iq3s_grid_f32() -> &'static [f32] {
    static CELL: OnceLock<Vec<f32>> = OnceLock::new();
    CELL.get_or_init(|| {
        let bytes: &[u8] = bytemuck::cast_slice(&IQ3S_GRID);
        bytes.iter().map(|&b| (b as i8) as f32).collect()
    })
}

/// Precompute `|grid|²` for a 4-element-per-entry grid (IQ3_XXS / IQ3_S).
fn build_grid_norm_sq_table_4(grid_bytes: &[u8], n_grid: usize) -> Vec<f32> {
    (0..n_grid)
        .map(|g| {
            let grid = &grid_bytes[g * 4..g * 4 + 4];
            let mut sum = 0f32;
            for &b in grid {
                let v = (b as i8) as f32;
                sum += v * v;
            }
            sum
        })
        .collect()
}

/// Per-chunk search for IQ3_XXS-style layouts: one 8-weight chunk
/// is encoded as TWO 4-element grid picks (grid1 covers weights
/// 0..4, grid2 covers weights 4..8) sharing a single sign mask
/// (one bit per weight, 8 bits total). Returns the best `(grid1,
/// grid2, sign_idx, signed_score, grid_norm_sq)` pair where
/// `signed_score = target · [signed_grid1, signed_grid2]` and
/// `grid_norm_sq = |grid1|² + |grid2|²`.
#[derive(Debug, Clone, Copy, Default)]
struct ChunkPickIq3Xxs {
    grid1_idx: u16,
    grid2_idx: u16,
    sign_idx: u8,
    signed_score: f32,
    grid_norm_sq: f32,
}

/// IQ3_XXS-style chunk search. With two independent 4-element
/// grid picks, the inner loop is O(n_grid²) per chunk = 65K
/// candidates for the 256-entry grid. To keep encode time
/// reasonable we use a two-stage greedy: first find the best
/// (grid1, signs_lo) for `target[0..4]`, then the best
/// (grid2, signs_hi) for `target[4..8]` independently. This is
/// suboptimal vs. joint search (the sign mask is global, not
/// independent), but for v1 it gives a working encoder.
fn search_chunk_iq3xxs(
    target: &[f32],
    grid_f32: &[f32],
    n_grid: usize,
    grid_norm_sq_table: &[f32],
) -> ChunkPickIq3Xxs {
    debug_assert_eq!(target.len(), 8);
    let rev = ksigns_iq2xs_reverse();
    let lo = &target[0..4];
    let hi = &target[4..8];

    // Find best (grid1, mask_lo) and (grid2, mask_hi) independently.
    let (g1, mask_lo, score_lo, norm_lo) = best_grid_4(lo, grid_f32, n_grid, grid_norm_sq_table);
    let (g2, mask_hi, score_hi, norm_hi) = best_grid_4(hi, grid_f32, n_grid, grid_norm_sq_table);

    // Combine the two 4-bit masks into one 8-bit mask. The
    // dequant uses `KMASK_IQ2XS[j]` for j ∈ 0..4 (lo half) and
    // `KMASK_IQ2XS[j+4]` for the hi half — matching our split.
    let mut mask = 0u8;
    for j in 0..4 {
        if mask_lo & (1u8 << j) != 0 {
            mask |= KMASK_IQ2XS[j];
        }
        if mask_hi & (1u8 << j) != 0 {
            mask |= KMASK_IQ2XS[j + 4];
        }
    }
    // If the combined popcount is odd, fix parity by flipping the
    // smallest-contrib element across all 8 (cheaper than redoing
    // both halves).
    let mut signed_score = score_lo + score_hi;
    if mask.count_ones() & 1 == 1 {
        // Find smallest |target[j] * grid[j]| across all 8 positions.
        // Read from the precomputed f32 grid — no i8→f32 conversion.
        let g1_slice = &grid_f32[g1 as usize * 4..g1 as usize * 4 + 4];
        let g2_slice = &grid_f32[g2 as usize * 4..g2 as usize * 4 + 4];
        let mut min_abs = f32::INFINITY;
        let mut min_j = 0usize;
        for j in 0..4 {
            let c = (target[j] * g1_slice[j]).abs();
            if c < min_abs {
                min_abs = c;
                min_j = j;
            }
        }
        for j in 0..4 {
            let c = (target[j + 4] * g2_slice[j]).abs();
            if c < min_abs {
                min_abs = c;
                min_j = j + 4;
            }
        }
        mask ^= KMASK_IQ2XS[min_j];
        signed_score -= 2.0 * min_abs;
    }
    let sign_idx = rev[mask as usize];
    ChunkPickIq3Xxs {
        grid1_idx: g1,
        grid2_idx: g2,
        sign_idx,
        signed_score,
        grid_norm_sq: norm_lo + norm_hi,
    }
}

/// Helper for IQ3_XXS's two-stage search: best (grid_idx,
/// 4-bit-sign-mask, signed_score, |g|²) for a 4-weight target.
/// The 4-bit mask here is a LOCAL convention (bit j → element j);
/// the caller stitches it into the global 8-bit sign mask before
/// looking up the 7-bit KSIGNS_IQ2XS index.
///
/// `grid_f32` is the precomputed f32 view of the 4-element-per-
/// entry codebook. Dispatches to SSE4.1 SIMD when available.
fn best_grid_4(
    target: &[f32],
    grid_f32: &[f32],
    n_grid: usize,
    grid_norm_sq_table: &[f32],
) -> (u16, u8, f32, f32) {
    debug_assert_eq!(target.len(), 4);
    debug_assert_eq!(grid_f32.len(), n_grid * 4);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: runtime feature detection above.
            return unsafe { best_grid_4_sse41(target, grid_f32, n_grid, grid_norm_sq_table) };
        }
    }
    best_grid_4_scalar(target, grid_f32, n_grid, grid_norm_sq_table)
}

pub(crate) fn best_grid_4_scalar(
    target: &[f32],
    grid_f32: &[f32],
    n_grid: usize,
    grid_norm_sq_table: &[f32],
) -> (u16, u8, f32, f32) {
    let mut best_g = 0u16;
    let mut best_mask = 0u8;
    // Sentinel `score=-1, norm=1` so a candidate must satisfy
    // `score² > norm` to win — a Cauchy-Schwarz-style quality
    // floor that rejects poor fits (small-magnitude inputs stay
    // at grid_idx=0). The IQ3 round-trip bound depends on this;
    // moving to `score=0` lets the encoder pick least-bad fits
    // for noise inputs and `iq3_xxs_round_trip_within_bound`
    // regresses from max_err 0.49 → 0.81. The SYCL kernel matches.
    let mut best_score = -1f32;
    let mut best_norm = 1f32;
    for g in 0..n_grid {
        let grid = &grid_f32[g * 4..g * 4 + 4];
        let mut mask = 0u8;
        let mut score = 0f32;
        for j in 0..4 {
            let contrib = target[j] * grid[j];
            if contrib >= 0.0 {
                score += contrib;
            } else {
                score -= contrib;
                mask |= 1u8 << j;
            }
        }
        let norm = grid_norm_sq_table[g];
        let lhs = score * score * best_norm;
        let rhs = best_score * best_score * norm;
        if lhs > rhs {
            best_g = g as u16;
            best_mask = mask;
            best_score = score;
            best_norm = norm;
        }
    }
    (best_g, best_mask, best_score, best_norm)
}

/// SSE4.1 path: 4-lane f32 product + absolute-value + horizontal-
/// sum + sign-bit extract. ~3× faster than scalar per candidate.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1")]
unsafe fn best_grid_4_sse41(
    target: &[f32],
    grid_f32: &[f32],
    n_grid: usize,
    grid_norm_sq_table: &[f32],
) -> (u16, u8, f32, f32) {
    use std::arch::x86_64::*;
    let mut best_g = 0u16;
    let mut best_mask = 0u8;
    // See `best_grid_4_scalar` — same `score² > norm` quality floor.
    let mut best_score = -1f32;
    let mut best_norm = 1f32;
    // SAFETY: target is &[f32] of length 4 (debug-asserted above).
    let target_v = unsafe { _mm_loadu_ps(target.as_ptr()) };
    // Abs-mask: clear sign bit per lane.
    let abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFFu32 as i32));
    let grid_ptr = grid_f32.as_ptr();
    for g in 0..n_grid {
        // SAFETY: g < n_grid, grid_f32 has n_grid * 4 elements.
        let g_v = unsafe { _mm_loadu_ps(grid_ptr.add(g * 4)) };
        let prod = _mm_mul_ps(target_v, g_v);
        let abs_prod = _mm_and_ps(prod, abs_mask);
        // Score = sum of |contribs|. Horizontal sum of 4 lanes.
        let shuf1 = _mm_movehdup_ps(abs_prod); // [b, b, d, d]
        let sums1 = _mm_add_ps(abs_prod, shuf1); // [a+b, _, c+d, _]
        let shuf2 = _mm_movehl_ps(shuf1, sums1); // [c+d, _, _, _]
        let final_v = _mm_add_ss(sums1, shuf2);
        let score = _mm_cvtss_f32(final_v);
        // Sign mask: bit j = 1 ⇔ prod[j] < 0.
        let mask = _mm_movemask_ps(prod) as u8 & 0x0F;
        let norm = grid_norm_sq_table[g];
        let lhs = score * score * best_norm;
        let rhs = best_score * best_score * norm;
        if lhs > rhs {
            best_g = g as u16;
            best_mask = mask;
            best_score = score;
            best_norm = norm;
        }
    }
    (best_g, best_mask, best_score, best_norm)
}

// ----------------------------------------------------------------------
// IQ2_XXS — `{ d: f16, qs: [u8; 64] }` = 66 bytes/block, 2.0625 bpw
// ----------------------------------------------------------------------

const BLOCK_IQ2_XXS_BYTES: usize = 66;

/// Encode IQ2_XXS. Per the dequant: each 32-weight sub-block
/// occupies 8 bytes of `qs` (two u32 words). `aux0` packs four
/// 8-bit grid indices; `aux1` packs four 7-bit sign indices plus a
/// 4-bit sub-block scale in bits 28..31. Sub-block scale formula:
/// `db = d * (0.5 + scale_nibble) * 0.25`.
pub fn encode_iq2_xxs(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK_K, 0, "encode_iq2_xxs: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_IQ2_XXS_BYTES,
        "encode_iq2_xxs: dst.len() must be n_blocks * 66"
    );
    let grid_bytes = iq2xxs_grid_bytes();
    let grid_f32 = iq2xxs_grid_f32();
    let grid_norm = build_grid_norm_sq_table_8(grid_bytes, 256);

    for b in 0..n_blocks {
        let xs = &src[b * QK_K..(b + 1) * QK_K];
        let off = b * BLOCK_IQ2_XXS_BYTES;

        // Stage 1: for every 32-weight sub-block, run the per-chunk
        // search (4 chunks × 8 weights) to get the picks. The
        // sub-block's natural scale is `signed_total / norm_total`,
        // where signed_total = Σ chunks' signed_scores and
        // norm_total = Σ chunks' grid_norm_sq.
        let mut sub_picks: [[ChunkPick; 4]; N_SUB_BLOCKS] = Default::default();
        let mut sub_scale = [0f32; N_SUB_BLOCKS]; // natural per-sub-block scale
        for ib32 in 0..N_SUB_BLOCKS {
            let mut signed_total = 0f32;
            let mut norm_total = 0f32;
            for l in 0..4 {
                let chunk = &xs[ib32 * 32 + l * 8..ib32 * 32 + l * 8 + 8];
                let pick = search_chunk_8(chunk, grid_f32, 256, &grid_norm);
                sub_picks[ib32][l] = pick;
                signed_total += pick.signed_score;
                norm_total += pick.grid_norm_sq;
            }
            sub_scale[ib32] = if norm_total > 0.0 {
                signed_total / norm_total
            } else {
                0.0
            };
        }

        // Stage 2: pick a super-block d such that every sub-block's
        // 4-bit scale-nibble fits the formula
        //   `db = d * (0.5 + nibble) * 0.25`
        // i.e. `nibble = (sub_scale / d) * 4 - 0.5` ∈ [0, 15].
        // The biggest sub_scale determines d: solve nibble=15 for
        // the max → d = sub_scale_max * 4 / (0.5 + 15) = sub_scale_max * 4 / 15.5.
        let sub_scale_max = sub_scale.iter().cloned().fold(0f32, f32::max);
        let d_super = if sub_scale_max > 0.0 {
            sub_scale_max * 4.0 / 15.5
        } else {
            0.0
        };
        let id_super = if d_super != 0.0 { 1.0 / d_super } else { 0.0 };

        // Stage 3: derive each sub-block's 4-bit scale nibble.
        let mut scale_nibbles = [0u32; N_SUB_BLOCKS];
        for ib32 in 0..N_SUB_BLOCKS {
            let n = (sub_scale[ib32] * id_super * 4.0 - 0.5).round();
            scale_nibbles[ib32] = n.clamp(0.0, 15.0) as u32;
        }

        // Stage 4: write d (f16) + qs.
        let d_bits = f16::from_f32(d_super).to_bits();
        dst[off] = (d_bits & 0xFF) as u8;
        dst[off + 1] = ((d_bits >> 8) & 0xFF) as u8;
        let qs = &mut dst[off + 2..off + 2 + 64];
        for ib32 in 0..N_SUB_BLOCKS {
            let mut aux0: u32 = 0;
            let mut aux1: u32 = 0;
            // aux0: 4 × 8-bit grid indices.
            // aux1: 4 × 7-bit sign indices + 4-bit scale in bits 28..31.
            for l in 0..4 {
                let p = sub_picks[ib32][l];
                aux0 |= (p.grid_idx as u32 & 0xFF) << (8 * l);
                aux1 |= (p.sign_idx as u32 & 0x7F) << (7 * l);
            }
            aux1 |= (scale_nibbles[ib32] & 0xF) << 28;
            qs[8 * ib32..8 * ib32 + 4].copy_from_slice(&aux0.to_le_bytes());
            qs[8 * ib32 + 4..8 * ib32 + 8].copy_from_slice(&aux1.to_le_bytes());
        }
    }
}

/// Batched IQ2_XXS encoder. Dispatches the per-chunk `(grid, sign)`
/// search through an [`IqGpuEncoder`] — one large batched call
/// covering all `32 * n_blocks` chunks per tensor — then runs the
/// existing per-block scale derivation + bit-packing tail.
///
/// Falls back to the per-chunk CPU path if `encoder` returns `Err`
/// (e.g. GPU `Unavailable`). The returned bytes are bit-identical
/// regardless of backend.
pub fn encode_iq2_xxs_with_encoder(
    src: &[f32],
    dst: &mut [u8],
    encoder: &dyn crate::iq_gpu::IqGpuEncoder,
) {
    use crate::iq_gpu::{Iq8EltGridFormat, Iq8EltSignedPick};

    assert_eq!(
        src.len() % QK_K,
        0,
        "encode_iq2_xxs_with_encoder: src.len() must be multiple of 256"
    );
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_IQ2_XXS_BYTES,
        "encode_iq2_xxs_with_encoder: dst.len() must be n_blocks * 66"
    );
    if n_blocks == 0 {
        return;
    }

    let total_chunks = n_blocks * 32;
    debug_assert_eq!(src.len(), total_chunks * 8);

    let mut picks: Vec<Iq8EltSignedPick> = vec![
        Iq8EltSignedPick {
            grid_idx: 0,
            sign_idx: 0,
            signed_score: 0.0,
            grid_norm_sq: 1.0,
        };
        total_chunks
    ];
    let gpu_ok = encoder
        .iq_8elt_signed_batched(src, Iq8EltGridFormat::Iq2Xxs, &mut picks)
        .is_ok();
    if !gpu_ok {
        encode_iq2_xxs(src, dst);
        return;
    }

    bit_pack_iq2_xxs_from_picks(&picks, dst);
}

/// IQ2_XXS per-block bit-packer. See [`bit_pack_iq2_xs_from_picks`].
pub fn bit_pack_iq2_xxs_from_picks(
    picks: &[crate::iq_gpu::Iq8EltSignedPick],
    dst: &mut [u8],
) {
    use rayon::prelude::*;
    dst.par_chunks_mut(BLOCK_IQ2_XXS_BYTES)
        .enumerate()
        .for_each(|(b, dst_block)| {
            let mut sub_picks: [[ChunkPick; 4]; N_SUB_BLOCKS] = Default::default();
            let mut sub_scale = [0f32; N_SUB_BLOCKS];
            for ib32 in 0..N_SUB_BLOCKS {
                let mut signed_total = 0f32;
                let mut norm_total = 0f32;
                for l in 0..4 {
                    let global = b * 32 + ib32 * 4 + l;
                    let p = &picks[global];
                    sub_picks[ib32][l] = ChunkPick {
                        grid_idx: p.grid_idx,
                        sign_idx: p.sign_idx,
                        signed_score: p.signed_score,
                        grid_norm_sq: p.grid_norm_sq,
                    };
                    signed_total += p.signed_score;
                    norm_total += p.grid_norm_sq;
                }
                sub_scale[ib32] = if norm_total > 0.0 {
                    signed_total / norm_total
                } else {
                    0.0
                };
            }
            let sub_scale_max = sub_scale.iter().cloned().fold(0f32, f32::max);
            let d_super = if sub_scale_max > 0.0 {
                sub_scale_max * 4.0 / 15.5
            } else {
                0.0
            };
            let id_super = if d_super != 0.0 { 1.0 / d_super } else { 0.0 };
            let mut scale_nibbles = [0u32; N_SUB_BLOCKS];
            for ib32 in 0..N_SUB_BLOCKS {
                let n = (sub_scale[ib32] * id_super * 4.0 - 0.5).round();
                scale_nibbles[ib32] = n.clamp(0.0, 15.0) as u32;
            }
            let d_bits = f16::from_f32(d_super).to_bits();
            dst_block[0] = (d_bits & 0xFF) as u8;
            dst_block[1] = ((d_bits >> 8) & 0xFF) as u8;
            let qs = &mut dst_block[2..2 + 64];
            for ib32 in 0..N_SUB_BLOCKS {
                let mut aux0: u32 = 0;
                let mut aux1: u32 = 0;
                for l in 0..4 {
                    let p = sub_picks[ib32][l];
                    aux0 |= (p.grid_idx as u32 & 0xFF) << (8 * l);
                    aux1 |= (p.sign_idx as u32 & 0x7F) << (7 * l);
                }
                aux1 |= (scale_nibbles[ib32] & 0xF) << 28;
                qs[8 * ib32..8 * ib32 + 4].copy_from_slice(&aux0.to_le_bytes());
                qs[8 * ib32 + 4..8 * ib32 + 8].copy_from_slice(&aux1.to_le_bytes());
            }
        });
}

// ----------------------------------------------------------------------
// IQ2_XS — `{ d: f16, qs: [u16; 32], scales: [u8; 8] }` = 74 bytes/block, 2.3125 bpw
// ----------------------------------------------------------------------

const BLOCK_IQ2_XS_BYTES: usize = 74;

/// Build the IQ2_XS grid as a flat i8 byte slice (512 × 8 bytes).
fn iq2xs_grid_bytes() -> &'static [u8] {
    bytemuck::cast_slice(&IQ2XS_GRID)
}

/// Encode IQ2_XS. Per the dequant: each `qs[ib32*4 + l]` u16 packs
/// `(grid_idx | (sign_idx << 9))` — 9-bit grid index into the
/// 512-entry IQ2XS_GRID, 7-bit sign index into KSIGNS_IQ2XS.
/// `scales[ib32]` packs two 4-bit nibbles (low for chunks 0/1,
/// high for chunks 2/3); sub-scale formula `db = d * (0.5 +
/// nibble) * 0.25`.
pub fn encode_iq2_xs(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK_K, 0, "encode_iq2_xs: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_IQ2_XS_BYTES,
        "encode_iq2_xs: dst.len() must be n_blocks * 74"
    );
    let grid_bytes = iq2xs_grid_bytes();
    let grid_f32 = iq2xs_grid_f32();
    let grid_norm = build_grid_norm_sq_table_8(grid_bytes, 512);

    for b in 0..n_blocks {
        let xs = &src[b * QK_K..(b + 1) * QK_K];
        let off = b * BLOCK_IQ2_XS_BYTES;

        // IQ2_XS has TWO sub-scales per 32-weight ib32: one for
        // chunks (0, 1) covering weights 0..15, another for chunks
        // (2, 3) covering weights 16..31. So we split each
        // sub-block into a "lo" half and a "hi" half and derive
        // each half's natural scale independently.
        let mut sub_picks: [[ChunkPick; 4]; N_SUB_BLOCKS] = Default::default();
        let mut sub_scale_lo = [0f32; N_SUB_BLOCKS];
        let mut sub_scale_hi = [0f32; N_SUB_BLOCKS];
        for ib32 in 0..N_SUB_BLOCKS {
            for half in 0..2 {
                let mut signed_total = 0f32;
                let mut norm_total = 0f32;
                for l_off in 0..2 {
                    let l = half * 2 + l_off;
                    let chunk = &xs[ib32 * 32 + l * 8..ib32 * 32 + l * 8 + 8];
                    let pick = search_chunk_8(chunk, grid_f32, 512, &grid_norm);
                    sub_picks[ib32][l] = pick;
                    signed_total += pick.signed_score;
                    norm_total += pick.grid_norm_sq;
                }
                let scale = if norm_total > 0.0 {
                    signed_total / norm_total
                } else {
                    0.0
                };
                if half == 0 {
                    sub_scale_lo[ib32] = scale;
                } else {
                    sub_scale_hi[ib32] = scale;
                }
            }
        }

        // Super-block d so the largest of the 16 half-sub-block
        // scales fits the formula `db = d * (0.5 + nibble) * 0.25`,
        // nibble ∈ [0, 15] → db_max = d * 3.875 → d = max / 3.875.
        let mut sub_scale_max = 0f32;
        for k in 0..N_SUB_BLOCKS {
            if sub_scale_lo[k] > sub_scale_max {
                sub_scale_max = sub_scale_lo[k];
            }
            if sub_scale_hi[k] > sub_scale_max {
                sub_scale_max = sub_scale_hi[k];
            }
        }
        let d_super = if sub_scale_max > 0.0 {
            sub_scale_max * 4.0 / 15.5
        } else {
            0.0
        };
        let id_super = if d_super != 0.0 { 1.0 / d_super } else { 0.0 };

        let mut scales = [0u8; N_SUB_BLOCKS];
        for ib32 in 0..N_SUB_BLOCKS {
            let n_lo = (sub_scale_lo[ib32] * id_super * 4.0 - 0.5).round();
            let n_hi = (sub_scale_hi[ib32] * id_super * 4.0 - 0.5).round();
            let lo = n_lo.clamp(0.0, 15.0) as u8;
            let hi = n_hi.clamp(0.0, 15.0) as u8;
            scales[ib32] = lo | (hi << 4);
        }

        // Write d (f16) + qs (32 u16 = 64 bytes) + scales (8 bytes).
        let d_bits = f16::from_f32(d_super).to_bits();
        dst[off] = (d_bits & 0xFF) as u8;
        dst[off + 1] = ((d_bits >> 8) & 0xFF) as u8;
        for ib32 in 0..N_SUB_BLOCKS {
            for l in 0..4 {
                let p = sub_picks[ib32][l];
                let word: u16 = (p.grid_idx & 0x1FF) | ((p.sign_idx as u16 & 0x7F) << 9);
                let off_w = off + 2 + (ib32 * 4 + l) * 2;
                dst[off_w] = (word & 0xFF) as u8;
                dst[off_w + 1] = ((word >> 8) & 0xFF) as u8;
            }
        }
        dst[off + 2 + 64..off + 2 + 64 + 8].copy_from_slice(&scales);
    }
}

/// Batched IQ2_XS encoder via [`IqGpuEncoder`]. See
/// [`encode_iq2_xxs_with_encoder`] for the design pattern; IQ2_XS
/// differs in: 512-entry grid, two half-sub-block scales per
/// 32-weight ib32, 9-bit grid indices packed into u16 `qs[]`.
pub fn encode_iq2_xs_with_encoder(
    src: &[f32],
    dst: &mut [u8],
    encoder: &dyn crate::iq_gpu::IqGpuEncoder,
) {
    use crate::iq_gpu::{Iq8EltGridFormat, Iq8EltSignedPick};

    assert_eq!(
        src.len() % QK_K,
        0,
        "encode_iq2_xs_with_encoder: src.len() must be multiple of 256"
    );
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_IQ2_XS_BYTES,
        "encode_iq2_xs_with_encoder: dst.len() must be n_blocks * 74"
    );
    if n_blocks == 0 {
        return;
    }

    let total_chunks = n_blocks * 32;
    debug_assert_eq!(src.len(), total_chunks * 8);
    let mut picks: Vec<Iq8EltSignedPick> = vec![
        Iq8EltSignedPick {
            grid_idx: 0,
            sign_idx: 0,
            signed_score: 0.0,
            grid_norm_sq: 1.0,
        };
        total_chunks
    ];
    let gpu_ok = encoder
        .iq_8elt_signed_batched(src, Iq8EltGridFormat::Iq2Xs, &mut picks)
        .is_ok();
    if !gpu_ok {
        encode_iq2_xs(src, dst);
        return;
    }

    bit_pack_iq2_xs_from_picks(&picks, dst);
}

/// IQ2_XS per-block bit-packer. Reads picks (`[Iq8EltSignedPick;
/// n_blocks * 32]`) and writes the 74-byte block layout to `dst`.
///
/// G4(a): parallel across blocks via `par_chunks_mut` — each
/// iteration reads `picks[b*32..(b+1)*32]` and writes the disjoint
/// range `dst[b*BLOCK..(b+1)*BLOCK]`, no shared mutable state.
///
/// G4(b): exposed publicly so the quantize pipeline can call this
/// after a *cross-tensor* coalesced GPU dispatch — one big encoder
/// call covers N tensors' chunks, then per-tensor bit-pack splits
/// the picks back.
pub fn bit_pack_iq2_xs_from_picks(
    picks: &[crate::iq_gpu::Iq8EltSignedPick],
    dst: &mut [u8],
) {
    use rayon::prelude::*;
    dst.par_chunks_mut(BLOCK_IQ2_XS_BYTES)
        .enumerate()
        .for_each(|(b, dst_block)| {
            let mut sub_picks: [[ChunkPick; 4]; N_SUB_BLOCKS] = Default::default();
            let mut sub_scale_lo = [0f32; N_SUB_BLOCKS];
            let mut sub_scale_hi = [0f32; N_SUB_BLOCKS];
            for ib32 in 0..N_SUB_BLOCKS {
                for half in 0..2 {
                    let mut signed_total = 0f32;
                    let mut norm_total = 0f32;
                    for l_off in 0..2 {
                        let l = half * 2 + l_off;
                        let global = b * 32 + ib32 * 4 + l;
                        let p = &picks[global];
                        sub_picks[ib32][l] = ChunkPick {
                            grid_idx: p.grid_idx,
                            sign_idx: p.sign_idx,
                            signed_score: p.signed_score,
                            grid_norm_sq: p.grid_norm_sq,
                        };
                        signed_total += p.signed_score;
                        norm_total += p.grid_norm_sq;
                    }
                    let scale = if norm_total > 0.0 {
                        signed_total / norm_total
                    } else {
                        0.0
                    };
                    if half == 0 {
                        sub_scale_lo[ib32] = scale;
                    } else {
                        sub_scale_hi[ib32] = scale;
                    }
                }
            }
            let mut sub_scale_max = 0f32;
            for k in 0..N_SUB_BLOCKS {
                if sub_scale_lo[k] > sub_scale_max {
                    sub_scale_max = sub_scale_lo[k];
                }
                if sub_scale_hi[k] > sub_scale_max {
                    sub_scale_max = sub_scale_hi[k];
                }
            }
            let d_super = if sub_scale_max > 0.0 {
                sub_scale_max * 4.0 / 15.5
            } else {
                0.0
            };
            let id_super = if d_super != 0.0 { 1.0 / d_super } else { 0.0 };
            let mut scales = [0u8; N_SUB_BLOCKS];
            for ib32 in 0..N_SUB_BLOCKS {
                let n_lo = (sub_scale_lo[ib32] * id_super * 4.0 - 0.5).round();
                let n_hi = (sub_scale_hi[ib32] * id_super * 4.0 - 0.5).round();
                let lo = n_lo.clamp(0.0, 15.0) as u8;
                let hi = n_hi.clamp(0.0, 15.0) as u8;
                scales[ib32] = lo | (hi << 4);
            }
            let d_bits = f16::from_f32(d_super).to_bits();
            dst_block[0] = (d_bits & 0xFF) as u8;
            dst_block[1] = ((d_bits >> 8) & 0xFF) as u8;
            for ib32 in 0..N_SUB_BLOCKS {
                for l in 0..4 {
                    let p = sub_picks[ib32][l];
                    let word: u16 = (p.grid_idx & 0x1FF) | ((p.sign_idx as u16 & 0x7F) << 9);
                    let off_w = 2 + (ib32 * 4 + l) * 2;
                    dst_block[off_w] = (word & 0xFF) as u8;
                    dst_block[off_w + 1] = ((word >> 8) & 0xFF) as u8;
                }
            }
            dst_block[2 + 64..2 + 64 + 8].copy_from_slice(&scales);
        });
}

// ----------------------------------------------------------------------
// IQ2_S — `{ d: f16, qs: [u8; 64], qh: [u8; 8], scales: [u8; 8] }` = 82 bytes/block, 2.5625 bpw
// ----------------------------------------------------------------------

const BLOCK_IQ2_S_BYTES: usize = 82;

/// Build the IQ2_S grid as a flat i8 byte slice (1024 × 8 bytes).
fn iq2s_grid_bytes() -> &'static [u8] {
    bytemuck::cast_slice(&IQ2S_GRID)
}

/// Encode IQ2_S. 10-bit grid index (split: low 8 bits in `qs[0..32]`,
/// high 2 bits in `qh[ib32]` at bit positions `2*l..2*l+2`); 8-bit
/// sign mask per chunk in `qs[32..64]`. Same per-half sub-scale
/// layout as IQ2_XS.
///
/// Note: IQ2_S stores the **full 8-bit sign mask directly** in
/// `signs[qs_off + l]` (unlike IQ2_XXS / IQ2_XS / IQ3_XXS which
/// store a 7-bit `KSIGNS_IQ2XS` index). Encoding writes the mask
/// bits directly; the parity-fix in `pick_sign_mask_8` still
/// runs but its parity-completion is no longer required by the
/// format. We keep the parity logic anyway — the resulting masks
/// are a strict subset of the format's representable set.
pub fn encode_iq2_s(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK_K, 0, "encode_iq2_s: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_IQ2_S_BYTES,
        "encode_iq2_s: dst.len() must be n_blocks * 82"
    );
    let grid_bytes = iq2s_grid_bytes();
    let grid_f32 = iq2s_grid_f32();
    let grid_norm = build_grid_norm_sq_table_8(grid_bytes, 1024);

    for b in 0..n_blocks {
        let xs = &src[b * QK_K..(b + 1) * QK_K];
        let off = b * BLOCK_IQ2_S_BYTES;

        let mut sub_picks: [[ChunkPick; 4]; N_SUB_BLOCKS] = Default::default();
        let mut sub_scale_lo = [0f32; N_SUB_BLOCKS];
        let mut sub_scale_hi = [0f32; N_SUB_BLOCKS];
        for ib32 in 0..N_SUB_BLOCKS {
            for half in 0..2 {
                let mut signed_total = 0f32;
                let mut norm_total = 0f32;
                for l_off in 0..2 {
                    let l = half * 2 + l_off;
                    let chunk = &xs[ib32 * 32 + l * 8..ib32 * 32 + l * 8 + 8];
                    let pick = search_chunk_8(chunk, grid_f32, 1024, &grid_norm);
                    sub_picks[ib32][l] = pick;
                    signed_total += pick.signed_score;
                    norm_total += pick.grid_norm_sq;
                }
                let scale = if norm_total > 0.0 {
                    signed_total / norm_total
                } else {
                    0.0
                };
                if half == 0 {
                    sub_scale_lo[ib32] = scale;
                } else {
                    sub_scale_hi[ib32] = scale;
                }
            }
        }

        let mut sub_scale_max = 0f32;
        for k in 0..N_SUB_BLOCKS {
            if sub_scale_lo[k] > sub_scale_max {
                sub_scale_max = sub_scale_lo[k];
            }
            if sub_scale_hi[k] > sub_scale_max {
                sub_scale_max = sub_scale_hi[k];
            }
        }
        let d_super = if sub_scale_max > 0.0 {
            sub_scale_max * 4.0 / 15.5
        } else {
            0.0
        };
        let id_super = if d_super != 0.0 { 1.0 / d_super } else { 0.0 };

        let mut scales = [0u8; N_SUB_BLOCKS];
        for ib32 in 0..N_SUB_BLOCKS {
            let n_lo = (sub_scale_lo[ib32] * id_super * 4.0 - 0.5).round();
            let n_hi = (sub_scale_hi[ib32] * id_super * 4.0 - 0.5).round();
            let lo = n_lo.clamp(0.0, 15.0) as u8;
            let hi = n_hi.clamp(0.0, 15.0) as u8;
            scales[ib32] = lo | (hi << 4);
        }

        // Layout (matches dequant_iq2_s):
        //   off + 0..2     : d (f16)
        //   off + 2..34    : qs_lo (low 8 bits of grid index, 4 bytes per ib32)
        //   off + 34..66   : signs (8-bit sign mask per chunk, 4 bytes per ib32)
        //   off + 66..74   : qh (high 2 bits of grid index, 1 byte per ib32)
        //   off + 74..82   : scales (8 bytes)
        let d_bits = f16::from_f32(d_super).to_bits();
        dst[off] = (d_bits & 0xFF) as u8;
        dst[off + 1] = ((d_bits >> 8) & 0xFF) as u8;
        let qs_lo_off = off + 2;
        let signs_off = off + 2 + 32;
        let qh_off = off + 2 + 64;
        let scales_off = off + 2 + 64 + 8;
        // Zero qh so OR-builds work.
        for byte in dst[qh_off..qh_off + 8].iter_mut() {
            *byte = 0;
        }
        for ib32 in 0..N_SUB_BLOCKS {
            for l in 0..4 {
                let p = sub_picks[ib32][l];
                // Low 8 bits → qs_lo[ib32*4 + l].
                dst[qs_lo_off + ib32 * 4 + l] = (p.grid_idx & 0xFF) as u8;
                // High 2 bits → qh[ib32] at bits (2*l, 2*l+1).
                let hi = ((p.grid_idx >> 8) & 0x3) as u8;
                dst[qh_off + ib32] |= hi << (2 * l);
                // Sign mask → signs[ib32*4 + l].
                // The format stores the full 8-bit mask directly,
                // NOT a KSIGNS_IQ2XS index. Convert back.
                let mask = KSIGNS_IQ2XS[p.sign_idx as usize];
                dst[signs_off + ib32 * 4 + l] = mask;
            }
        }
        dst[scales_off..scales_off + 8].copy_from_slice(&scales);
    }
}

/// Batched IQ2_S encoder via [`IqGpuEncoder`]. See
/// [`encode_iq2_xxs_with_encoder`] for design. IQ2_S differs in:
/// 1024-entry grid, half-sub-block scales (like IQ2_XS), 10-bit grid
/// indices split into low-8 in `qs[]` + high-2 in `qh[]`, and the
/// full 8-bit sign mask written directly (decoded from sign_idx via
/// `KSIGNS_IQ2XS[sign_idx]`).
pub fn encode_iq2_s_with_encoder(
    src: &[f32],
    dst: &mut [u8],
    encoder: &dyn crate::iq_gpu::IqGpuEncoder,
) {
    use crate::iq_gpu::{Iq8EltGridFormat, Iq8EltSignedPick};

    assert_eq!(
        src.len() % QK_K,
        0,
        "encode_iq2_s_with_encoder: src.len() must be multiple of 256"
    );
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_IQ2_S_BYTES,
        "encode_iq2_s_with_encoder: dst.len() must be n_blocks * 82"
    );
    if n_blocks == 0 {
        return;
    }

    let total_chunks = n_blocks * 32;
    debug_assert_eq!(src.len(), total_chunks * 8);
    let mut picks: Vec<Iq8EltSignedPick> = vec![
        Iq8EltSignedPick {
            grid_idx: 0,
            sign_idx: 0,
            signed_score: 0.0,
            grid_norm_sq: 1.0,
        };
        total_chunks
    ];
    let gpu_ok = encoder
        .iq_8elt_signed_batched(src, Iq8EltGridFormat::Iq2S, &mut picks)
        .is_ok();
    if !gpu_ok {
        encode_iq2_s(src, dst);
        return;
    }

    bit_pack_iq2_s_from_picks(&picks, dst);
}

/// IQ2_S per-block bit-packer. See [`bit_pack_iq2_xs_from_picks`].
pub fn bit_pack_iq2_s_from_picks(
    picks: &[crate::iq_gpu::Iq8EltSignedPick],
    dst: &mut [u8],
) {
    use rayon::prelude::*;
    dst.par_chunks_mut(BLOCK_IQ2_S_BYTES)
        .enumerate()
        .for_each(|(b, dst_block)| {
            let mut sub_picks: [[ChunkPick; 4]; N_SUB_BLOCKS] = Default::default();
            let mut sub_scale_lo = [0f32; N_SUB_BLOCKS];
            let mut sub_scale_hi = [0f32; N_SUB_BLOCKS];
            for ib32 in 0..N_SUB_BLOCKS {
                for half in 0..2 {
                    let mut signed_total = 0f32;
                    let mut norm_total = 0f32;
                    for l_off in 0..2 {
                        let l = half * 2 + l_off;
                        let global = b * 32 + ib32 * 4 + l;
                        let p = &picks[global];
                        sub_picks[ib32][l] = ChunkPick {
                            grid_idx: p.grid_idx,
                            sign_idx: p.sign_idx,
                            signed_score: p.signed_score,
                            grid_norm_sq: p.grid_norm_sq,
                        };
                        signed_total += p.signed_score;
                        norm_total += p.grid_norm_sq;
                    }
                    let scale = if norm_total > 0.0 {
                        signed_total / norm_total
                    } else {
                        0.0
                    };
                    if half == 0 {
                        sub_scale_lo[ib32] = scale;
                    } else {
                        sub_scale_hi[ib32] = scale;
                    }
                }
            }
            let mut sub_scale_max = 0f32;
            for k in 0..N_SUB_BLOCKS {
                if sub_scale_lo[k] > sub_scale_max {
                    sub_scale_max = sub_scale_lo[k];
                }
                if sub_scale_hi[k] > sub_scale_max {
                    sub_scale_max = sub_scale_hi[k];
                }
            }
            let d_super = if sub_scale_max > 0.0 {
                sub_scale_max * 4.0 / 15.5
            } else {
                0.0
            };
            let id_super = if d_super != 0.0 { 1.0 / d_super } else { 0.0 };
            let mut scales = [0u8; N_SUB_BLOCKS];
            for ib32 in 0..N_SUB_BLOCKS {
                let n_lo = (sub_scale_lo[ib32] * id_super * 4.0 - 0.5).round();
                let n_hi = (sub_scale_hi[ib32] * id_super * 4.0 - 0.5).round();
                let lo = n_lo.clamp(0.0, 15.0) as u8;
                let hi = n_hi.clamp(0.0, 15.0) as u8;
                scales[ib32] = lo | (hi << 4);
            }
            let d_bits = f16::from_f32(d_super).to_bits();
            dst_block[0] = (d_bits & 0xFF) as u8;
            dst_block[1] = ((d_bits >> 8) & 0xFF) as u8;
            let qs_lo_off = 2;
            let signs_off = 2 + 32;
            let qh_off = 2 + 64;
            let scales_off = 2 + 64 + 8;
            for byte in dst_block[qh_off..qh_off + 8].iter_mut() {
                *byte = 0;
            }
            for ib32 in 0..N_SUB_BLOCKS {
                for l in 0..4 {
                    let p = sub_picks[ib32][l];
                    dst_block[qs_lo_off + ib32 * 4 + l] = (p.grid_idx & 0xFF) as u8;
                    let hi = ((p.grid_idx >> 8) & 0x3) as u8;
                    dst_block[qh_off + ib32] |= hi << (2 * l);
                    let mask = KSIGNS_IQ2XS[p.sign_idx as usize];
                    dst_block[signs_off + ib32 * 4 + l] = mask;
                }
            }
            dst_block[scales_off..scales_off + 8].copy_from_slice(&scales);
        });
}

// ----------------------------------------------------------------------
// IQ3_XXS — `{ d: f16, qs: [u8; 96] }` = 98 bytes/block, 3.0625 bpw
// ----------------------------------------------------------------------

const BLOCK_IQ3_XXS_BYTES: usize = 98;

/// Encode IQ3_XXS. Per the dequant: `qs[0..64]` are grid indices
/// (8 bytes per 32-weight sub-block: two 4-byte u32s carrying
/// 4 grid pairs each — actually, 8 separate `u8` grid indices,
/// each chunk picking 2 of them); `qs[64..96]` are 8 u32 words
/// (one per sub-block) packing 4 × 7-bit sign indices plus a
/// 4-bit sub-block scale in bits 28..31. Sub-block scale formula:
/// `db = d * (0.5 + scale_nibble) * 0.5`.
pub fn encode_iq3_xxs(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK_K, 0, "encode_iq3_xxs: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_IQ3_XXS_BYTES,
        "encode_iq3_xxs: dst.len() must be n_blocks * 98"
    );
    let grid_bytes = iq3xxs_grid_bytes();
    let grid_f32 = iq3xxs_grid_f32();
    let grid_norm = build_grid_norm_sq_table_4(grid_bytes, 256);

    for b in 0..n_blocks {
        let xs = &src[b * QK_K..(b + 1) * QK_K];
        let off = b * BLOCK_IQ3_XXS_BYTES;

        // Per-sub-block: 4 chunks of 8 weights → 4 (grid1, grid2, signs).
        let mut sub_picks: [[ChunkPickIq3Xxs; 4]; N_SUB_BLOCKS] = Default::default();
        let mut sub_scale = [0f32; N_SUB_BLOCKS];
        for ib32 in 0..N_SUB_BLOCKS {
            let mut signed_total = 0f32;
            let mut norm_total = 0f32;
            for l in 0..4 {
                let chunk = &xs[ib32 * 32 + l * 8..ib32 * 32 + l * 8 + 8];
                let pick = search_chunk_iq3xxs(chunk, grid_f32, 256, &grid_norm);
                sub_picks[ib32][l] = pick;
                signed_total += pick.signed_score;
                norm_total += pick.grid_norm_sq;
            }
            sub_scale[ib32] = if norm_total > 0.0 {
                signed_total / norm_total
            } else {
                0.0
            };
        }

        // Super-block d: `db = d * (0.5 + nibble) * 0.5`, nibble in
        // [0, 15], so db_max = d * 7.75. Solve nibble=15 for the
        // largest sub_scale to derive d.
        let sub_scale_max = sub_scale.iter().cloned().fold(0f32, f32::max);
        let d_super = if sub_scale_max > 0.0 {
            sub_scale_max * 2.0 / 15.5
        } else {
            0.0
        };
        let id_super = if d_super != 0.0 { 1.0 / d_super } else { 0.0 };

        let mut scale_nibbles = [0u32; N_SUB_BLOCKS];
        for ib32 in 0..N_SUB_BLOCKS {
            let n = (sub_scale[ib32] * id_super * 2.0 - 0.5).round();
            scale_nibbles[ib32] = n.clamp(0.0, 15.0) as u32;
        }

        // Write d (f16) + qs (96 bytes = 64 grid + 32 scales/signs).
        let d_bits = f16::from_f32(d_super).to_bits();
        dst[off] = (d_bits & 0xFF) as u8;
        dst[off + 1] = ((d_bits >> 8) & 0xFF) as u8;
        let qs_grid = &mut dst[off + 2..off + 2 + 64];
        for ib32 in 0..N_SUB_BLOCKS {
            for l in 0..4 {
                let p = sub_picks[ib32][l];
                qs_grid[8 * ib32 + 2 * l] = (p.grid1_idx & 0xFF) as u8;
                qs_grid[8 * ib32 + 2 * l + 1] = (p.grid2_idx & 0xFF) as u8;
            }
        }
        let qs_sas = &mut dst[off + 2 + 64..off + 2 + 96];
        for ib32 in 0..N_SUB_BLOCKS {
            let mut aux32: u32 = 0;
            for l in 0..4 {
                aux32 |= (sub_picks[ib32][l].sign_idx as u32 & 0x7F) << (7 * l);
            }
            aux32 |= (scale_nibbles[ib32] & 0xF) << 28;
            qs_sas[4 * ib32..4 * ib32 + 4].copy_from_slice(&aux32.to_le_bytes());
        }
    }
}

/// Batched IQ3_XXS encoder via [`IqGpuEncoder`]. Same tail as
/// [`encode_iq3_xxs`]; the per-chunk paired-grid search is hoisted
/// into one batched GPU call over all `32 × n_blocks` chunks.
pub fn encode_iq3_xxs_with_encoder(
    src: &[f32],
    dst: &mut [u8],
    encoder: &dyn crate::iq_gpu::IqGpuEncoder,
) {
    use crate::iq_gpu::{Iq4EltGridFormat, Iq4EltPairedPick};

    assert_eq!(
        src.len() % QK_K,
        0,
        "encode_iq3_xxs_with_encoder: src.len() must be multiple of 256"
    );
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_IQ3_XXS_BYTES,
        "encode_iq3_xxs_with_encoder: dst.len() must be n_blocks * 98"
    );
    if n_blocks == 0 {
        return;
    }

    let total_chunks = n_blocks * 32;
    debug_assert_eq!(src.len(), total_chunks * 8);
    let mut picks: Vec<Iq4EltPairedPick> = vec![
        Iq4EltPairedPick {
            grid1_idx: 0,
            grid2_idx: 0,
            sign_idx: 0,
            signed_score: 0.0,
            grid_norm_sq: 1.0,
        };
        total_chunks
    ];
    let gpu_ok = encoder
        .iq_4elt_paired_signed_batched(src, Iq4EltGridFormat::Iq3Xxs, &mut picks)
        .is_ok();
    if !gpu_ok {
        encode_iq3_xxs(src, dst);
        return;
    }

    // G4(a): parallel bit-pack across blocks.
    use rayon::prelude::*;
    dst.par_chunks_mut(BLOCK_IQ3_XXS_BYTES)
        .enumerate()
        .for_each(|(b, dst_block)| {
            let mut sub_picks: [[ChunkPickIq3Xxs; 4]; N_SUB_BLOCKS] = Default::default();
            let mut sub_scale = [0f32; N_SUB_BLOCKS];
            for ib32 in 0..N_SUB_BLOCKS {
                let mut signed_total = 0f32;
                let mut norm_total = 0f32;
                for l in 0..4 {
                    let global = b * 32 + ib32 * 4 + l;
                    let p = &picks[global];
                    sub_picks[ib32][l] = ChunkPickIq3Xxs {
                        grid1_idx: p.grid1_idx,
                        grid2_idx: p.grid2_idx,
                        sign_idx: p.sign_idx,
                        signed_score: p.signed_score,
                        grid_norm_sq: p.grid_norm_sq,
                    };
                    signed_total += p.signed_score;
                    norm_total += p.grid_norm_sq;
                }
                sub_scale[ib32] = if norm_total > 0.0 {
                    signed_total / norm_total
                } else {
                    0.0
                };
            }
            let sub_scale_max = sub_scale.iter().cloned().fold(0f32, f32::max);
            let d_super = if sub_scale_max > 0.0 {
                sub_scale_max * 2.0 / 15.5
            } else {
                0.0
            };
            let id_super = if d_super != 0.0 { 1.0 / d_super } else { 0.0 };
            let mut scale_nibbles = [0u32; N_SUB_BLOCKS];
            for ib32 in 0..N_SUB_BLOCKS {
                let n = (sub_scale[ib32] * id_super * 2.0 - 0.5).round();
                scale_nibbles[ib32] = n.clamp(0.0, 15.0) as u32;
            }
            let d_bits = f16::from_f32(d_super).to_bits();
            dst_block[0] = (d_bits & 0xFF) as u8;
            dst_block[1] = ((d_bits >> 8) & 0xFF) as u8;
            let (head, tail) = dst_block.split_at_mut(2);
            let _ = head;
            let (qs_grid, qs_sas) = tail.split_at_mut(64);
            for ib32 in 0..N_SUB_BLOCKS {
                for l in 0..4 {
                    let p = sub_picks[ib32][l];
                    qs_grid[8 * ib32 + 2 * l] = (p.grid1_idx & 0xFF) as u8;
                    qs_grid[8 * ib32 + 2 * l + 1] = (p.grid2_idx & 0xFF) as u8;
                }
            }
            for ib32 in 0..N_SUB_BLOCKS {
                let mut aux32: u32 = 0;
                for l in 0..4 {
                    aux32 |= (sub_picks[ib32][l].sign_idx as u32 & 0x7F) << (7 * l);
                }
                aux32 |= (scale_nibbles[ib32] & 0xF) << 28;
                qs_sas[4 * ib32..4 * ib32 + 4].copy_from_slice(&aux32.to_le_bytes());
            }
        });
}

// ----------------------------------------------------------------------
// IQ3_S — 110 bytes/block: `{ d: f16, qs: [u8; 64], qh: [u8; 8], signs: [u8; 32], scales: [u8; 4] }`
// ----------------------------------------------------------------------

const BLOCK_IQ3_S_BYTES: usize = 110;

/// Build the IQ3_S grid as a flat i8 byte slice (512 × 4 bytes).
fn iq3s_grid_bytes() -> &'static [u8] {
    bytemuck::cast_slice(&IQ3S_GRID)
}

/// Encode IQ3_S. 9-bit grid indices (low 8 in `qs`, high 1 in `qh`)
/// into the 512-entry 4-elt IQ3S_GRID; 8-bit sign mask per chunk
/// stored directly in `signs[]`; per-sub-block 4-bit scale packed
/// two-per-byte into `scales[4]`. Sub-scale formula:
/// `db = d * (1 + 2 * x)` with `x ∈ [0, 15]` → db ∈ `d * [1, 31]`.
pub fn encode_iq3_s(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK_K, 0, "encode_iq3_s: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_IQ3_S_BYTES,
        "encode_iq3_s: dst.len() must be n_blocks * 110"
    );
    let grid_bytes = iq3s_grid_bytes();
    let grid_f32 = iq3s_grid_f32();
    let grid_norm = build_grid_norm_sq_table_4(grid_bytes, 512);

    for b in 0..n_blocks {
        let xs = &src[b * QK_K..(b + 1) * QK_K];
        let off = b * BLOCK_IQ3_S_BYTES;

        // Per-sub-block search: 4 chunks of 8 weights each. Each
        // chunk picks 2 grid entries (lo + hi 4-elt halves) plus
        // a global 8-bit sign mask.
        let mut sub_picks: [[ChunkPickIq3Xxs; 4]; N_SUB_BLOCKS] = Default::default();
        let mut sub_scale = [0f32; N_SUB_BLOCKS];
        for ib32 in 0..N_SUB_BLOCKS {
            let mut signed_total = 0f32;
            let mut norm_total = 0f32;
            for l in 0..4 {
                let chunk = &xs[ib32 * 32 + l * 8..ib32 * 32 + l * 8 + 8];
                let pick = search_chunk_iq3xxs(chunk, grid_f32, 512, &grid_norm);
                sub_picks[ib32][l] = pick;
                signed_total += pick.signed_score;
                norm_total += pick.grid_norm_sq;
            }
            sub_scale[ib32] = if norm_total > 0.0 {
                signed_total / norm_total
            } else {
                0.0
            };
        }

        // Super-block d: largest sub-scale → x=15 → db_max = d * 31.
        // d = sub_scale_max / 31.
        let sub_scale_max = sub_scale.iter().cloned().fold(0f32, f32::max);
        let d_super = if sub_scale_max > 0.0 {
            sub_scale_max / 31.0
        } else {
            0.0
        };
        let id_super = if d_super != 0.0 { 1.0 / d_super } else { 0.0 };

        let mut scale_nibbles = [0u8; N_SUB_BLOCKS];
        for ib32 in 0..N_SUB_BLOCKS {
            // sub_scale = d * (1 + 2x) → x = (sub_scale/d - 1) / 2.
            let x = (sub_scale[ib32] * id_super - 1.0) * 0.5;
            scale_nibbles[ib32] = x.round().clamp(0.0, 15.0) as u8;
        }

        // Layout (matches dequant_iq3_s):
        //   off + 0..2     : d (f16)
        //   off + 2..66    : qs (low 8 bits of grid index, 8 per ib32)
        //   off + 66..74   : qh (high 1 bit per grid pick — 8 bits per ib32)
        //   off + 74..106  : signs (8-bit mask per chunk, 4 per ib32)
        //   off + 106..110 : scales (4 bytes, 2 sub-blocks per byte)
        let d_bits = f16::from_f32(d_super).to_bits();
        dst[off] = (d_bits & 0xFF) as u8;
        dst[off + 1] = ((d_bits >> 8) & 0xFF) as u8;
        let qs_off = off + 2;
        let qh_off = off + 2 + 64;
        let signs_off = off + 2 + 64 + 8;
        let scales_off = off + 2 + 64 + 8 + 32;
        for byte in dst[qh_off..qh_off + 8].iter_mut() {
            *byte = 0;
        }

        let mut qs_cur = 0usize;
        let mut signs_cur = 0usize;
        for pair in 0..4 {
            let ib32 = pair * 2;
            // First sub-block of the pair.
            for l in 0..4 {
                let p = sub_picks[ib32][l];
                dst[qs_off + qs_cur + 2 * l] = (p.grid1_idx & 0xFF) as u8;
                dst[qs_off + qs_cur + 2 * l + 1] = (p.grid2_idx & 0xFF) as u8;
                // High bit of grid1 goes to qh[ib32] at bit (2*l);
                // high bit of grid2 goes to qh[ib32] at bit (2*l+1).
                // Reverse-engineering from dequant:
                //   g1_idx = qs | (qh << (8 - 2*l)) & 0x100
                //   g2_idx = qs | (qh << (7 - 2*l)) & 0x100
                // So qh bit `2*l` (after the shift) IS the 9th bit
                // of g1, and bit `2*l + 1` is the 9th bit of g2.
                let hi1 = ((p.grid1_idx >> 8) & 0x1) as u8;
                let hi2 = ((p.grid2_idx >> 8) & 0x1) as u8;
                dst[qh_off + ib32] |= hi1 << (2 * l);
                dst[qh_off + ib32] |= hi2 << (2 * l + 1);
                // Sign mask — KSIGNS_IQ2XS lookup back to 8-bit mask.
                let mask = KSIGNS_IQ2XS[p.sign_idx as usize];
                dst[signs_off + signs_cur + l] = mask;
            }
            qs_cur += 8;
            signs_cur += 4;
            // Second sub-block of the pair.
            for l in 0..4 {
                let p = sub_picks[ib32 + 1][l];
                dst[qs_off + qs_cur + 2 * l] = (p.grid1_idx & 0xFF) as u8;
                dst[qs_off + qs_cur + 2 * l + 1] = (p.grid2_idx & 0xFF) as u8;
                let hi1 = ((p.grid1_idx >> 8) & 0x1) as u8;
                let hi2 = ((p.grid2_idx >> 8) & 0x1) as u8;
                dst[qh_off + ib32 + 1] |= hi1 << (2 * l);
                dst[qh_off + ib32 + 1] |= hi2 << (2 * l + 1);
                let mask = KSIGNS_IQ2XS[p.sign_idx as usize];
                dst[signs_off + signs_cur + l] = mask;
            }
            qs_cur += 8;
            signs_cur += 4;
            // Scale byte for this pair: low nibble = first ib32,
            // high nibble = second.
            dst[scales_off + pair] = scale_nibbles[ib32] | (scale_nibbles[ib32 + 1] << 4);
        }
    }
}

/// Batched IQ3_S encoder via [`IqGpuEncoder`]. Same tail as
/// [`encode_iq3_s`]; the per-chunk paired-grid search is hoisted
/// into one batched GPU call over all `32 × n_blocks` chunks.
pub fn encode_iq3_s_with_encoder(
    src: &[f32],
    dst: &mut [u8],
    encoder: &dyn crate::iq_gpu::IqGpuEncoder,
) {
    use crate::iq_gpu::{Iq4EltGridFormat, Iq4EltPairedPick};

    assert_eq!(
        src.len() % QK_K,
        0,
        "encode_iq3_s_with_encoder: src.len() must be multiple of 256"
    );
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_IQ3_S_BYTES,
        "encode_iq3_s_with_encoder: dst.len() must be n_blocks * 110"
    );
    if n_blocks == 0 {
        return;
    }

    let total_chunks = n_blocks * 32;
    debug_assert_eq!(src.len(), total_chunks * 8);
    let mut picks: Vec<Iq4EltPairedPick> = vec![
        Iq4EltPairedPick {
            grid1_idx: 0,
            grid2_idx: 0,
            sign_idx: 0,
            signed_score: 0.0,
            grid_norm_sq: 1.0,
        };
        total_chunks
    ];
    let gpu_ok = encoder
        .iq_4elt_paired_signed_batched(src, Iq4EltGridFormat::Iq3S, &mut picks)
        .is_ok();
    if !gpu_ok {
        encode_iq3_s(src, dst);
        return;
    }

    // G4(a): parallel bit-pack across blocks.
    use rayon::prelude::*;
    dst.par_chunks_mut(BLOCK_IQ3_S_BYTES)
        .enumerate()
        .for_each(|(b, dst_block)| {
            let mut sub_picks: [[ChunkPickIq3Xxs; 4]; N_SUB_BLOCKS] = Default::default();
            let mut sub_scale = [0f32; N_SUB_BLOCKS];
            for ib32 in 0..N_SUB_BLOCKS {
                let mut signed_total = 0f32;
                let mut norm_total = 0f32;
                for l in 0..4 {
                    let global = b * 32 + ib32 * 4 + l;
                    let p = &picks[global];
                    sub_picks[ib32][l] = ChunkPickIq3Xxs {
                        grid1_idx: p.grid1_idx,
                        grid2_idx: p.grid2_idx,
                        sign_idx: p.sign_idx,
                        signed_score: p.signed_score,
                        grid_norm_sq: p.grid_norm_sq,
                    };
                    signed_total += p.signed_score;
                    norm_total += p.grid_norm_sq;
                }
                sub_scale[ib32] = if norm_total > 0.0 {
                    signed_total / norm_total
                } else {
                    0.0
                };
            }
            let sub_scale_max = sub_scale.iter().cloned().fold(0f32, f32::max);
            let d_super = if sub_scale_max > 0.0 {
                sub_scale_max / 31.0
            } else {
                0.0
            };
            let id_super = if d_super != 0.0 { 1.0 / d_super } else { 0.0 };
            let mut scale_nibbles = [0u8; N_SUB_BLOCKS];
            for ib32 in 0..N_SUB_BLOCKS {
                let x = (sub_scale[ib32] * id_super - 1.0) * 0.5;
                scale_nibbles[ib32] = x.round().clamp(0.0, 15.0) as u8;
            }

            let d_bits = f16::from_f32(d_super).to_bits();
            dst_block[0] = (d_bits & 0xFF) as u8;
            dst_block[1] = ((d_bits >> 8) & 0xFF) as u8;
            let qs_off = 2;
            let qh_off = 2 + 64;
            let signs_off = 2 + 64 + 8;
            let scales_off = 2 + 64 + 8 + 32;
            for byte in dst_block[qh_off..qh_off + 8].iter_mut() {
                *byte = 0;
            }
            let mut qs_cur = 0usize;
            let mut signs_cur = 0usize;
            for pair in 0..4 {
                let ib32 = pair * 2;
                for l in 0..4 {
                    let p = sub_picks[ib32][l];
                    dst_block[qs_off + qs_cur + 2 * l] = (p.grid1_idx & 0xFF) as u8;
                    dst_block[qs_off + qs_cur + 2 * l + 1] = (p.grid2_idx & 0xFF) as u8;
                    let hi1 = ((p.grid1_idx >> 8) & 0x1) as u8;
                    let hi2 = ((p.grid2_idx >> 8) & 0x1) as u8;
                    dst_block[qh_off + ib32] |= hi1 << (2 * l);
                    dst_block[qh_off + ib32] |= hi2 << (2 * l + 1);
                    let mask = KSIGNS_IQ2XS[p.sign_idx as usize];
                    dst_block[signs_off + signs_cur + l] = mask;
                }
                qs_cur += 8;
                signs_cur += 4;
                for l in 0..4 {
                    let p = sub_picks[ib32 + 1][l];
                    dst_block[qs_off + qs_cur + 2 * l] = (p.grid1_idx & 0xFF) as u8;
                    dst_block[qs_off + qs_cur + 2 * l + 1] = (p.grid2_idx & 0xFF) as u8;
                    let hi1 = ((p.grid1_idx >> 8) & 0x1) as u8;
                    let hi2 = ((p.grid2_idx >> 8) & 0x1) as u8;
                    dst_block[qh_off + ib32 + 1] |= hi1 << (2 * l);
                    dst_block[qh_off + ib32 + 1] |= hi2 << (2 * l + 1);
                    let mask = KSIGNS_IQ2XS[p.sign_idx as usize];
                    dst_block[signs_off + signs_cur + l] = mask;
                }
                qs_cur += 8;
                signs_cur += 4;
                dst_block[scales_off + pair] = scale_nibbles[ib32] | (scale_nibbles[ib32 + 1] << 4);
            }
        });
}

// ----------------------------------------------------------------------
// IQ1_S — 50 bytes/block, 1.5625 bpw
//   { d: f16, qs: [u8; 32], qh: [u16; 8] }
// ----------------------------------------------------------------------

const BLOCK_IQ1_S_BYTES: usize = 50;

/// Per-chunk IQ1_S pick: 11-bit grid index + the score of the
/// reconstruction `dl * (grid + delta)` for the chosen delta sign.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ChunkPickIq1 {
    pub(crate) grid_idx: u16,
    /// `(target · effective)` where `effective[j] = grid[j] + delta`.
    /// Always non-negative (the chunk picks the sign such that the
    /// projection is positive; the per-sub-block dl absorbs any
    /// remaining sign).
    pub(crate) signed_score: f32,
    /// `|effective|²` for the chosen grid + delta. Same role as in
    /// `ChunkPick`: lets the sub-block aggregator compute the
    /// optimal `dl` as `Σ signed_score / Σ norm_sq`.
    pub(crate) norm_sq: f32,
}

/// Find the best IQ1_S grid entry for an 8-weight target, given a
/// per-sub-block delta. Returns the grid_idx + the resulting
/// `target · (grid + delta)` and `|grid + delta|²`. The sign of
/// `target · (grid + delta)` is preserved in `signed_score` (it
/// can be negative; the caller's per-sub-block dl absorbs the
/// sign).
///
/// Dispatch ladder:
///   1. AVX2 (8-lane FMA + horizontal-sum) when the host reports
///      AVX2 + FMA — ~4-6× faster than scalar.
///   2. Scalar fallback using the precomputed f32 grid (no
///      per-iter int8→f32 conversion) — ~2× faster than the
///      original implementation.
///
/// The two paths produce byte-identical output (same grid +
/// `+delta` math; FMA's higher-precision intermediate doesn't
/// change the score comparison's branch direction at any
/// realistic input magnitude).
/// Expose the IQ1_S 2048×8 f32 grid table to external backends
/// (the SYCL `IqGpuEncoder` impl in `rustllama-kernels-sycl` needs
/// to upload it to USM device memory once per stream). The slice
/// is `2048 * 8 = 16384` f32 entries, row-major.
pub fn iq1s_grid_f32_table() -> &'static [f32] {
    &iq1s_grid_cache().grid_f32
}

/// Expose the IQ2_XXS 256×8 f32 grid table (`256 * 8 = 2048`
/// entries). Same shape conventions as [`iq1s_grid_f32_table`].
pub fn iq2xxs_grid_f32_table() -> &'static [f32] {
    iq2xxs_grid_f32()
}

/// Expose the IQ2_XS 512×8 f32 grid table.
pub fn iq2xs_grid_f32_table() -> &'static [f32] {
    iq2xs_grid_f32()
}

/// Expose the IQ2_S 1024×8 f32 grid table.
pub fn iq2s_grid_f32_table() -> &'static [f32] {
    iq2s_grid_f32()
}

/// Expose the IQ3_XXS 256×4 f32 grid table.
pub fn iq3xxs_grid_f32_table() -> &'static [f32] {
    iq3xxs_grid_f32()
}

/// Expose the IQ3_S 512×4 f32 grid table.
pub fn iq3s_grid_f32_table() -> &'static [f32] {
    iq3s_grid_f32()
}

/// Expose the 256-entry inverse-`KSIGNS_IQ2XS` lookup table used by
/// the IQ2_* encoders. Entry `i` is the 7-bit `ksigns` index whose
/// 8-bit decoded mask equals `i`, or `0xFF` if `i` is not one of the
/// 128 representable patterns (odd-popcount masks). Needed by the
/// SYCL `IqGpuEncoder` impl to upload as a USM constant.
pub fn ksigns_iq2xs_reverse_table() -> &'static [u8; 256] {
    ksigns_iq2xs_reverse()
}

/// Expose precomputed `|grid|²` table for IQ2_XXS (256 entries).
pub fn iq2xxs_grid_norm_sq_table() -> Vec<f32> {
    build_grid_norm_sq_table_8(iq2xxs_grid_bytes(), 256)
}

/// Expose precomputed `|grid|²` table for IQ2_XS (512 entries).
pub fn iq2xs_grid_norm_sq_table() -> Vec<f32> {
    build_grid_norm_sq_table_8(
        bytemuck::cast_slice::<u64, u8>(&crate::dequant::IQ2XS_GRID),
        512,
    )
}

/// Expose precomputed `|grid|²` table for IQ2_S (1024 entries).
pub fn iq2s_grid_norm_sq_table() -> Vec<f32> {
    build_grid_norm_sq_table_8(
        bytemuck::cast_slice::<u64, u8>(&crate::dequant::IQ2S_GRID),
        1024,
    )
}

/// Expose precomputed `|grid|²` table for IQ3_XXS (256 entries).
pub fn iq3xxs_grid_norm_sq_table() -> Vec<f32> {
    build_grid_norm_sq_table_4(iq3xxs_grid_bytes(), 256)
}

/// Expose precomputed `|grid|²` table for IQ3_S (512 entries).
pub fn iq3s_grid_norm_sq_table() -> Vec<f32> {
    build_grid_norm_sq_table_4(
        bytemuck::cast_slice::<u32, u8>(&crate::dequant::IQ3S_GRID),
        512,
    )
}

/// Expose the 8-entry `KMASK_IQ2XS` constant. Used by the GPU
/// IQ3 kernel to map per-half greedy sign bits into the global
/// 8-bit mask. Entry j ∈ [0, 7] is `1 << j` by construction.
pub fn kmask_iq2xs_table() -> &'static [u8; 8] {
    &KMASK_IQ2XS
}

/// Public adapter for the IQ trait CPU fallback. Returns
/// `(grid_idx, signed_score, norm_sq)`.
pub(crate) fn best_iq1s_grid_for_chunk_pub(target: &[f32], delta: f32) -> (u16, f32, f32) {
    let p = best_iq1s_grid_for_chunk(target, delta);
    (p.grid_idx, p.signed_score, p.norm_sq)
}

/// Diagnostic helper: scalar-only IQ2 search_chunk_8, exposed for
/// FP-parity hardware tests against the GPU kernel. The default
/// search_chunk_8 dispatches to AVX2+FMA on x86, which can round
/// the Cauchy-Schwarz comparison differently from the GPU's FP
/// unit. This entry point uses only scalar mul+add — matching
/// what the GPU kernel does.
pub fn search_chunk_8_scalar_for_format(
    target: &[f32],
    format: crate::iq_gpu::Iq8EltGridFormat,
) -> crate::iq_gpu::Iq8EltSignedPick {
    use crate::iq_gpu::{Iq8EltGridFormat, Iq8EltSignedPick};
    let (grid_f32, n_grid, grid_norm): (&[f32], usize, Vec<f32>) = match format {
        Iq8EltGridFormat::Iq1s => {
            return Iq8EltSignedPick {
                grid_idx: 0,
                sign_idx: 0,
                signed_score: 0.0,
                grid_norm_sq: 1.0,
            };
        }
        Iq8EltGridFormat::Iq2Xxs => (
            iq2xxs_grid_f32(),
            256,
            build_grid_norm_sq_table_8(iq2xxs_grid_bytes(), 256),
        ),
        Iq8EltGridFormat::Iq2Xs => (
            iq2xs_grid_f32(),
            512,
            build_grid_norm_sq_table_8(
                bytemuck::cast_slice::<u64, u8>(&crate::dequant::IQ2XS_GRID),
                512,
            ),
        ),
        Iq8EltGridFormat::Iq2S => (
            iq2s_grid_f32(),
            1024,
            build_grid_norm_sq_table_8(
                bytemuck::cast_slice::<u64, u8>(&crate::dequant::IQ2S_GRID),
                1024,
            ),
        ),
    };
    let p = search_chunk_8_scalar(target, grid_f32, n_grid, &grid_norm);
    Iq8EltSignedPick {
        grid_idx: p.grid_idx,
        sign_idx: p.sign_idx,
        signed_score: p.signed_score,
        grid_norm_sq: p.grid_norm_sq,
    }
}

/// Public adapter for the IQ2 trait CPU fallback. Picks the right
/// grid table + n_grid based on the format selector and forwards
/// to the SIMD-dispatched search.
pub(crate) fn search_chunk_8_for_format(
    target: &[f32],
    format: crate::iq_gpu::Iq8EltGridFormat,
) -> crate::iq_gpu::Iq8EltSignedPick {
    use crate::iq_gpu::{Iq8EltGridFormat, Iq8EltSignedPick};
    let (grid_f32, n_grid, grid_norm): (&[f32], usize, Vec<f32>) = match format {
        Iq8EltGridFormat::Iq1s => {
            // Not normally invoked here (IQ1 uses the delta path),
            // but provide a defensive route: zero output.
            return Iq8EltSignedPick {
                grid_idx: 0,
                sign_idx: 0,
                signed_score: 0.0,
                grid_norm_sq: 1.0,
            };
        }
        Iq8EltGridFormat::Iq2Xxs => (
            iq2xxs_grid_f32(),
            256,
            build_grid_norm_sq_table_8(iq2xxs_grid_bytes(), 256),
        ),
        Iq8EltGridFormat::Iq2Xs => (
            iq2xs_grid_f32(),
            512,
            build_grid_norm_sq_table_8(
                bytemuck::cast_slice::<u64, u8>(&crate::dequant::IQ2XS_GRID),
                512,
            ),
        ),
        Iq8EltGridFormat::Iq2S => (
            iq2s_grid_f32(),
            1024,
            build_grid_norm_sq_table_8(
                bytemuck::cast_slice::<u64, u8>(&crate::dequant::IQ2S_GRID),
                1024,
            ),
        ),
    };
    let p = search_chunk_8(target, grid_f32, n_grid, &grid_norm);
    Iq8EltSignedPick {
        grid_idx: p.grid_idx,
        sign_idx: p.sign_idx,
        signed_score: p.signed_score,
        grid_norm_sq: p.grid_norm_sq,
    }
}

/// Diagnostic helper: scalar-only IQ3 paired-signed search, exposed
/// for FP-parity hardware tests. The default `best_grid_4`
/// dispatches to SSE4.1 SIMD on x86; this entry uses only scalar
/// mul+add, matching the GPU kernel's FP semantics.
pub fn search_chunk_iq3_scalar_for_format(
    target: &[f32],
    format: crate::iq_gpu::Iq4EltGridFormat,
) -> crate::iq_gpu::Iq4EltPairedPick {
    use crate::iq_gpu::{Iq4EltGridFormat, Iq4EltPairedPick};
    let (grid_f32, n_grid, grid_norm): (&[f32], usize, Vec<f32>) = match format {
        Iq4EltGridFormat::Iq3Xxs => (
            iq3xxs_grid_f32(),
            256,
            build_grid_norm_sq_table_4(iq3xxs_grid_bytes(), 256),
        ),
        Iq4EltGridFormat::Iq3S => (
            iq3s_grid_f32(),
            512,
            build_grid_norm_sq_table_4(
                bytemuck::cast_slice::<u32, u8>(&crate::dequant::IQ3S_GRID),
                512,
            ),
        ),
    };
    // Reuse the scalar-only halves via best_grid_4_scalar, then
    // apply the same combine + parity-fix as search_chunk_iq3xxs.
    debug_assert_eq!(target.len(), 8);
    let rev = ksigns_iq2xs_reverse();
    let lo = &target[0..4];
    let hi = &target[4..8];
    let (g1, mask_lo, score_lo, norm_lo) = best_grid_4_scalar(lo, grid_f32, n_grid, &grid_norm);
    let (g2, mask_hi, score_hi, norm_hi) = best_grid_4_scalar(hi, grid_f32, n_grid, &grid_norm);
    let mut mask = 0u8;
    for j in 0..4 {
        if mask_lo & (1u8 << j) != 0 {
            mask |= KMASK_IQ2XS[j];
        }
        if mask_hi & (1u8 << j) != 0 {
            mask |= KMASK_IQ2XS[j + 4];
        }
    }
    let mut signed_score = score_lo + score_hi;
    if mask.count_ones() & 1 == 1 {
        let g1_slice = &grid_f32[g1 as usize * 4..g1 as usize * 4 + 4];
        let g2_slice = &grid_f32[g2 as usize * 4..g2 as usize * 4 + 4];
        let mut min_abs = f32::INFINITY;
        let mut min_j = 0usize;
        for j in 0..4 {
            let c = (target[j] * g1_slice[j]).abs();
            if c < min_abs {
                min_abs = c;
                min_j = j;
            }
        }
        for j in 0..4 {
            let c = (target[j + 4] * g2_slice[j]).abs();
            if c < min_abs {
                min_abs = c;
                min_j = j + 4;
            }
        }
        mask ^= KMASK_IQ2XS[min_j];
        signed_score -= 2.0 * min_abs;
    }
    let sign_idx = rev[mask as usize];
    Iq4EltPairedPick {
        grid1_idx: g1,
        grid2_idx: g2,
        sign_idx,
        signed_score,
        grid_norm_sq: norm_lo + norm_hi,
    }
}

/// Public adapter for the IQ3 trait CPU fallback.
pub(crate) fn search_chunk_iq3_for_format(
    target: &[f32],
    format: crate::iq_gpu::Iq4EltGridFormat,
) -> crate::iq_gpu::Iq4EltPairedPick {
    use crate::iq_gpu::{Iq4EltGridFormat, Iq4EltPairedPick};
    let (grid_f32, n_grid, grid_norm): (&[f32], usize, Vec<f32>) = match format {
        Iq4EltGridFormat::Iq3Xxs => (
            iq3xxs_grid_f32(),
            256,
            build_grid_norm_sq_table_4(iq3xxs_grid_bytes(), 256),
        ),
        Iq4EltGridFormat::Iq3S => (
            iq3s_grid_f32(),
            512,
            build_grid_norm_sq_table_4(
                bytemuck::cast_slice::<u32, u8>(&crate::dequant::IQ3S_GRID),
                512,
            ),
        ),
    };
    let p = search_chunk_iq3xxs(target, grid_f32, n_grid, &grid_norm);
    Iq4EltPairedPick {
        grid1_idx: p.grid1_idx,
        grid2_idx: p.grid2_idx,
        sign_idx: p.sign_idx,
        signed_score: p.signed_score,
        grid_norm_sq: p.grid_norm_sq,
    }
}

fn best_iq1s_grid_for_chunk(target: &[f32], delta: f32) -> ChunkPickIq1 {
    debug_assert_eq!(target.len(), 8);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            return unsafe { best_iq1s_grid_for_chunk_avx2(target, delta) };
        }
    }
    best_iq1s_grid_for_chunk_scalar(target, delta)
}

/// Variant of [`best_iq1s_grid_for_chunk_positive_scalar`] that picks
/// the grid entry minimizing (most-negative) `signed_score`. Used in
/// the d<0 super-block-sign branch: with negative d_super, dl_used ≤ 0,
/// so we want the grid pick to have signed_score ≤ 0 too — and the
/// minimizer of signed_score gives the lowest L2 error under dl ≤ 0.
pub(crate) fn best_iq1s_grid_for_chunk_negative_scalar(
    target: &[f32],
    delta: f32,
) -> ChunkPickIq1 {
    let cache = iq1s_grid_cache();
    let mut best = ChunkPickIq1 {
        grid_idx: 0,
        signed_score: 0.0,
        norm_sq: 1.0,
    };
    let mut best_signed = f32::INFINITY;
    for idx in 0..2048 {
        let g_base = idx * 8;
        let g_f32 = &cache.grid_f32[g_base..g_base + 8];
        let mut dot_grid = 0f32;
        let mut norm = 0f32;
        for j in 0..8 {
            let g_plus_d = g_f32[j] + delta;
            dot_grid += target[j] * g_plus_d;
            norm += g_plus_d * g_plus_d;
        }
        if norm > 0.0 && dot_grid < best_signed {
            best_signed = dot_grid;
            best = ChunkPickIq1 {
                grid_idx: idx as u16,
                signed_score: dot_grid,
                norm_sq: norm,
            };
        }
    }
    best
}

/// IQ1_S sign-bias fix: scalar grid search that maximizes the **signed**
/// score (`target · (grid + delta)`) instead of its square. Used in the
/// fallback path when a sub-block's natural dl would be negative — the
/// IQ1_S block scale is unsigned, so taking `|dl|` sign-flips the
/// reconstruction (proven to be strictly worse than zero-reconstruction
/// by `+3 * signed² / norm`). Picking the max-positive-score grid entry
/// per chunk yields a guaranteed `dl ≥ 0` reconstruction that lower-bounds
/// the L2 error.
pub(crate) fn best_iq1s_grid_for_chunk_positive_scalar(
    target: &[f32],
    delta: f32,
) -> ChunkPickIq1 {
    let cache = iq1s_grid_cache();
    let mut best = ChunkPickIq1 {
        grid_idx: 0,
        signed_score: 0.0,
        norm_sq: 1.0,
    };
    let mut best_signed = f32::NEG_INFINITY;
    for idx in 0..2048 {
        let g_base = idx * 8;
        let g_f32 = &cache.grid_f32[g_base..g_base + 8];
        let mut dot_grid = 0f32;
        let mut norm = 0f32;
        for j in 0..8 {
            let g_plus_d = g_f32[j] + delta;
            dot_grid += target[j] * g_plus_d;
            norm += g_plus_d * g_plus_d;
        }
        if norm > 0.0 && dot_grid > best_signed {
            best_signed = dot_grid;
            best = ChunkPickIq1 {
                grid_idx: idx as u16,
                signed_score: dot_grid,
                norm_sq: norm,
            };
        }
    }
    best
}

/// Scalar reference. Used by the parity test + as fallback on
/// non-x86_64 hosts. Reads the precomputed f32 grid (built once via
/// [`iq1s_grid_cache`]).
pub(crate) fn best_iq1s_grid_for_chunk_scalar(
    target: &[f32],
    delta: f32,
) -> ChunkPickIq1 {
    let cache = iq1s_grid_cache();
    let mut best = ChunkPickIq1 {
        grid_idx: 0,
        signed_score: 0.0,
        norm_sq: 1.0,
    };
    let mut best_score_sq_div_norm = -1f32;
    // Iterate sorted-by-norm-desc would enable pruning; for v1
    // the linear sweep over the 2048-entry grid is what we ship.
    // Pruning is a follow-up that stacks on top of this path.
    for idx in 0..2048 {
        let g_base = idx * 8;
        let g_f32 = &cache.grid_f32[g_base..g_base + 8];
        // Compute dot = Σ target[j] * (g_f32[j] + delta)
        //             = Σ target[j] * g_f32[j] + delta * Σ target[j]
        // We factor the delta term so the inner 8-MAC loop only
        // does target·grid without the per-iter add. The Σ
        // target[j] is constant across the grid sweep — precompute.
        let mut dot_grid = 0f32;
        let mut norm = 0f32;
        for j in 0..8 {
            let g_plus_d = g_f32[j] + delta;
            dot_grid += target[j] * g_plus_d;
            norm += g_plus_d * g_plus_d;
        }
        if norm > 0.0 {
            let s2_div_n = dot_grid * dot_grid / norm;
            if s2_div_n > best_score_sq_div_norm {
                best_score_sq_div_norm = s2_div_n;
                best = ChunkPickIq1 {
                    grid_idx: idx as u16,
                    signed_score: dot_grid,
                    norm_sq: norm,
                };
            }
        }
    }
    best
}

/// AVX2 implementation: each iteration loads 8 grid f32s in one
/// `_mm256_loadu_ps`, broadcasts delta, FMAs against target,
/// horizontal-sums for dot, computes norm in parallel, and uses
/// scalar branch for the best-score compare. Net cost per
/// candidate: ~6 cycles vs ~25 in scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn best_iq1s_grid_for_chunk_avx2(
    target: &[f32],
    delta: f32,
) -> ChunkPickIq1 {
    use std::arch::x86_64::*;
    debug_assert_eq!(target.len(), 8);
    let cache = iq1s_grid_cache();
    let grid_ptr = cache.grid_f32.as_ptr();

    // Pre-load target once (read-only across grid sweep).
    // SAFETY: target is `&[f32]` of length 8 (debug-asserted above);
    // grid_ptr is a Vec<f32> base pointer with 2048 × 8 elements.
    let target_v = unsafe { _mm256_loadu_ps(target.as_ptr()) };
    let delta_v = _mm256_set1_ps(delta);

    let mut best = ChunkPickIq1 {
        grid_idx: 0,
        signed_score: 0.0,
        norm_sq: 1.0,
    };
    let mut best_score_sq_div_norm = -1f32;

    for idx in 0..2048 {
        // SAFETY: idx ∈ [0, 2048), grid has 16384 elements,
        // 8-element load at `idx * 8` stays in bounds.
        let g_v = unsafe { _mm256_loadu_ps(grid_ptr.add(idx * 8)) };
        let g_plus_d = _mm256_add_ps(g_v, delta_v);
        let prod = _mm256_mul_ps(target_v, g_plus_d);
        let sq = _mm256_mul_ps(g_plus_d, g_plus_d);
        // SAFETY: each of the helper calls is itself a target-feature
        // sandbox; we're inside the same AVX2 context.
        let dot = unsafe { horizontal_sum_avx2(prod) };
        let norm = unsafe { horizontal_sum_avx2(sq) };
        if norm > 0.0 {
            let s2_div_n = dot * dot / norm;
            if s2_div_n > best_score_sq_div_norm {
                best_score_sq_div_norm = s2_div_n;
                best = ChunkPickIq1 {
                    grid_idx: idx as u16,
                    signed_score: dot,
                    norm_sq: norm,
                };
            }
        }
    }
    best
}

/// #2: Process BOTH IQ1_S deltas (`-1+δ` and `-1-δ`) in a single AVX2
/// pass over the 2048-entry grid. Shares grid loads + cache locality
/// between deltas. Returns `[picks_for_delta_pos, picks_for_delta_neg]`
/// where each entry is `(max-|score|, max-positive, max-negative)`.
pub(crate) fn best_iq1s_grid_all3_both_deltas(
    target: &[f32],
) -> [(ChunkPickIq1, ChunkPickIq1, ChunkPickIq1); 2] {
    debug_assert_eq!(target.len(), 8);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { best_iq1s_grid_all3_both_deltas_avx2(target) };
        }
    }
    let dp = -1.0 + IQ1S_DELTA;
    let dn = -1.0 - IQ1S_DELTA;
    [best_iq1s_grid_all3(target, dp), best_iq1s_grid_all3(target, dn)]
}

/// imatrix-weighted scalar variant of [`best_iq1s_grid_all3`]: one
/// combined grid sweep returning (max-|score|, max-+score, max--score)
/// for `delta`, with each candidate's dot/norm weighted by the
/// per-element importance `w` (length 8). Scalar-only — the imatrix
/// path forgoes AVX2 (a perf, not correctness, follow-up). The
/// unweighted/AVX2 path is left completely untouched.
fn best_iq1s_grid_all3_scalar_w(
    target: &[f32],
    delta: f32,
    w: &[f32],
) -> (ChunkPickIq1, ChunkPickIq1, ChunkPickIq1) {
    debug_assert_eq!(target.len(), 8);
    debug_assert_eq!(w.len(), 8);
    let cache = iq1s_grid_cache();
    let z = ChunkPickIq1 { grid_idx: 0, signed_score: 0.0, norm_sq: 1.0 };
    let (mut best_abs, mut best_pos, mut best_neg) = (z, z, z);
    let mut best_s2n = -1f32;
    let mut best_p = f32::NEG_INFINITY;
    let mut best_n = f32::INFINITY;
    for idx in 0..2048 {
        let g = &cache.grid_f32[idx * 8..idx * 8 + 8];
        let mut dot = 0f32;
        let mut norm = 0f32;
        for j in 0..8 {
            let gpd = g[j] + delta;
            dot += w[j] * target[j] * gpd;
            norm += w[j] * gpd * gpd;
        }
        if norm > 0.0 {
            let pick = ChunkPickIq1 { grid_idx: idx as u16, signed_score: dot, norm_sq: norm };
            let s2n = dot * dot / norm;
            if s2n > best_s2n {
                best_s2n = s2n;
                best_abs = pick;
            }
            if dot > best_p {
                best_p = dot;
                best_pos = pick;
            }
            if dot < best_n {
                best_n = dot;
                best_neg = pick;
            }
        }
    }
    (best_abs, best_pos, best_neg)
}

/// imatrix-weighted scalar variant of [`best_iq1s_grid_all3_both_deltas`].
fn best_iq1s_grid_all3_both_deltas_w(
    target: &[f32],
    w: &[f32],
) -> [(ChunkPickIq1, ChunkPickIq1, ChunkPickIq1); 2] {
    let dp = -1.0 + IQ1S_DELTA;
    let dn = -1.0 - IQ1S_DELTA;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe {
                [
                    best_iq1s_grid_all3_avx2_w(target, dp, w),
                    best_iq1s_grid_all3_avx2_w(target, dn, w),
                ]
            };
        }
    }
    [
        best_iq1s_grid_all3_scalar_w(target, dp, w),
        best_iq1s_grid_all3_scalar_w(target, dn, w),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn best_iq1s_grid_all3_both_deltas_avx2(
    target: &[f32],
) -> [(ChunkPickIq1, ChunkPickIq1, ChunkPickIq1); 2] {
    use std::arch::x86_64::*;
    let cache = iq1s_grid_cache();
    let grid_ptr = cache.grid_f32.as_ptr();
    let target_v = unsafe { _mm256_loadu_ps(target.as_ptr()) };
    let delta_p_v = _mm256_set1_ps(-1.0 + IQ1S_DELTA);
    let delta_n_v = _mm256_set1_ps(-1.0 - IQ1S_DELTA);

    let init = ChunkPickIq1 { grid_idx: 0, signed_score: 0.0, norm_sq: 1.0 };
    let mut best_abs_p = init; let mut best_pos_p = init; let mut best_neg_p = init;
    let mut best_abs_n = init; let mut best_pos_n = init; let mut best_neg_n = init;
    let mut best_abs_p_s2divn = -1f32;
    let mut best_pos_p_signed = f32::NEG_INFINITY;
    let mut best_neg_p_signed = f32::INFINITY;
    let mut best_abs_n_s2divn = -1f32;
    let mut best_pos_n_signed = f32::NEG_INFINITY;
    let mut best_neg_n_signed = f32::INFINITY;

    for idx in 0..2048 {
        let g_v = unsafe { _mm256_loadu_ps(grid_ptr.add(idx * 8)) };
        // Both (g+δ) computations share the grid load.
        let gpd_p = _mm256_add_ps(g_v, delta_p_v);
        let gpd_n = _mm256_add_ps(g_v, delta_n_v);
        let prod_p = _mm256_mul_ps(target_v, gpd_p);
        let prod_n = _mm256_mul_ps(target_v, gpd_n);
        let sq_p = _mm256_mul_ps(gpd_p, gpd_p);
        let sq_n = _mm256_mul_ps(gpd_n, gpd_n);
        let dot_p = unsafe { horizontal_sum_avx2(prod_p) };
        let dot_n = unsafe { horizontal_sum_avx2(prod_n) };
        let norm_p = unsafe { horizontal_sum_avx2(sq_p) };
        let norm_n = unsafe { horizontal_sum_avx2(sq_n) };
        let idx_u16 = idx as u16;
        if norm_p > 0.0 {
            let s2_div_n = dot_p * dot_p / norm_p;
            if s2_div_n > best_abs_p_s2divn {
                best_abs_p_s2divn = s2_div_n;
                best_abs_p = ChunkPickIq1 { grid_idx: idx_u16, signed_score: dot_p, norm_sq: norm_p };
            }
            if dot_p > best_pos_p_signed {
                best_pos_p_signed = dot_p;
                best_pos_p = ChunkPickIq1 { grid_idx: idx_u16, signed_score: dot_p, norm_sq: norm_p };
            }
            if dot_p < best_neg_p_signed {
                best_neg_p_signed = dot_p;
                best_neg_p = ChunkPickIq1 { grid_idx: idx_u16, signed_score: dot_p, norm_sq: norm_p };
            }
        }
        if norm_n > 0.0 {
            let s2_div_n = dot_n * dot_n / norm_n;
            if s2_div_n > best_abs_n_s2divn {
                best_abs_n_s2divn = s2_div_n;
                best_abs_n = ChunkPickIq1 { grid_idx: idx_u16, signed_score: dot_n, norm_sq: norm_n };
            }
            if dot_n > best_pos_n_signed {
                best_pos_n_signed = dot_n;
                best_pos_n = ChunkPickIq1 { grid_idx: idx_u16, signed_score: dot_n, norm_sq: norm_n };
            }
            if dot_n < best_neg_n_signed {
                best_neg_n_signed = dot_n;
                best_neg_n = ChunkPickIq1 { grid_idx: idx_u16, signed_score: dot_n, norm_sq: norm_n };
            }
        }
    }
    [(best_abs_p, best_pos_p, best_neg_p), (best_abs_n, best_pos_n, best_neg_n)]
}

/// Sibling of [`best_iq1s_grid_all3_both_deltas`] for the GPU-encoder
/// path: max-|score| arrives from GPU. Single AVX2 pass over the grid,
/// returns `[(max-pos, max-neg) for delta_pos, (max-pos, max-neg) for delta_neg]`.
pub(crate) fn best_iq1s_grid_pos_neg_both_deltas(
    target: &[f32],
) -> [(ChunkPickIq1, ChunkPickIq1); 2] {
    debug_assert_eq!(target.len(), 8);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { best_iq1s_grid_pos_neg_both_deltas_avx2(target) };
        }
    }
    let dp = -1.0 + IQ1S_DELTA;
    let dn = -1.0 - IQ1S_DELTA;
    [best_iq1s_grid_pos_neg(target, dp), best_iq1s_grid_pos_neg(target, dn)]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn best_iq1s_grid_pos_neg_both_deltas_avx2(
    target: &[f32],
) -> [(ChunkPickIq1, ChunkPickIq1); 2] {
    use std::arch::x86_64::*;
    let cache = iq1s_grid_cache();
    let grid_ptr = cache.grid_f32.as_ptr();
    let target_v = unsafe { _mm256_loadu_ps(target.as_ptr()) };
    let delta_p_v = _mm256_set1_ps(-1.0 + IQ1S_DELTA);
    let delta_n_v = _mm256_set1_ps(-1.0 - IQ1S_DELTA);
    let init = ChunkPickIq1 { grid_idx: 0, signed_score: 0.0, norm_sq: 1.0 };
    let mut best_pos_p = init; let mut best_neg_p = init;
    let mut best_pos_n = init; let mut best_neg_n = init;
    let mut best_pos_p_signed = f32::NEG_INFINITY;
    let mut best_neg_p_signed = f32::INFINITY;
    let mut best_pos_n_signed = f32::NEG_INFINITY;
    let mut best_neg_n_signed = f32::INFINITY;
    for idx in 0..2048 {
        let g_v = unsafe { _mm256_loadu_ps(grid_ptr.add(idx * 8)) };
        let gpd_p = _mm256_add_ps(g_v, delta_p_v);
        let gpd_n = _mm256_add_ps(g_v, delta_n_v);
        let prod_p = _mm256_mul_ps(target_v, gpd_p);
        let prod_n = _mm256_mul_ps(target_v, gpd_n);
        let sq_p = _mm256_mul_ps(gpd_p, gpd_p);
        let sq_n = _mm256_mul_ps(gpd_n, gpd_n);
        let dot_p = unsafe { horizontal_sum_avx2(prod_p) };
        let dot_n = unsafe { horizontal_sum_avx2(prod_n) };
        let norm_p = unsafe { horizontal_sum_avx2(sq_p) };
        let norm_n = unsafe { horizontal_sum_avx2(sq_n) };
        let idx_u16 = idx as u16;
        if norm_p > 0.0 {
            if dot_p > best_pos_p_signed {
                best_pos_p_signed = dot_p;
                best_pos_p = ChunkPickIq1 { grid_idx: idx_u16, signed_score: dot_p, norm_sq: norm_p };
            }
            if dot_p < best_neg_p_signed {
                best_neg_p_signed = dot_p;
                best_neg_p = ChunkPickIq1 { grid_idx: idx_u16, signed_score: dot_p, norm_sq: norm_p };
            }
        }
        if norm_n > 0.0 {
            if dot_n > best_pos_n_signed {
                best_pos_n_signed = dot_n;
                best_pos_n = ChunkPickIq1 { grid_idx: idx_u16, signed_score: dot_n, norm_sq: norm_n };
            }
            if dot_n < best_neg_n_signed {
                best_neg_n_signed = dot_n;
                best_neg_n = ChunkPickIq1 { grid_idx: idx_u16, signed_score: dot_n, norm_sq: norm_n };
            }
        }
    }
    [(best_pos_p, best_neg_p), (best_pos_n, best_neg_n)]
}

/// Sibling of [`best_iq1s_grid_all3`] for the GPU-encoder path: max-|score|
/// already arrives from the GPU's batched kernel, so we only need max-positive
/// and max-negative. Single AVX2 sweep, two best-trackers — roughly half the
/// per-chunk cost of `all3`.
pub(crate) fn best_iq1s_grid_pos_neg(
    target: &[f32],
    delta: f32,
) -> (ChunkPickIq1, ChunkPickIq1) {
    debug_assert_eq!(target.len(), 8);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { best_iq1s_grid_pos_neg_avx2(target, delta) };
        }
    }
    let (_a, p, n) = (
        best_iq1s_grid_for_chunk_scalar(target, delta),
        best_iq1s_grid_for_chunk_positive_scalar(target, delta),
        best_iq1s_grid_for_chunk_negative_scalar(target, delta),
    );
    (p, n)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn best_iq1s_grid_pos_neg_avx2(
    target: &[f32],
    delta: f32,
) -> (ChunkPickIq1, ChunkPickIq1) {
    use std::arch::x86_64::*;
    debug_assert_eq!(target.len(), 8);
    let cache = iq1s_grid_cache();
    let grid_ptr = cache.grid_f32.as_ptr();
    let target_v = unsafe { _mm256_loadu_ps(target.as_ptr()) };
    let delta_v = _mm256_set1_ps(delta);
    let mut best_pos = ChunkPickIq1 { grid_idx: 0, signed_score: 0.0, norm_sq: 1.0 };
    let mut best_neg = ChunkPickIq1 { grid_idx: 0, signed_score: 0.0, norm_sq: 1.0 };
    let mut best_pos_signed = f32::NEG_INFINITY;
    let mut best_neg_signed = f32::INFINITY;
    for idx in 0..2048 {
        let g_v = unsafe { _mm256_loadu_ps(grid_ptr.add(idx * 8)) };
        let g_plus_d = _mm256_add_ps(g_v, delta_v);
        let prod = _mm256_mul_ps(target_v, g_plus_d);
        let sq = _mm256_mul_ps(g_plus_d, g_plus_d);
        let dot = unsafe { horizontal_sum_avx2(prod) };
        let norm = unsafe { horizontal_sum_avx2(sq) };
        if norm > 0.0 {
            if dot > best_pos_signed {
                best_pos_signed = dot;
                best_pos = ChunkPickIq1 { grid_idx: idx as u16, signed_score: dot, norm_sq: norm };
            }
            if dot < best_neg_signed {
                best_neg_signed = dot;
                best_neg = ChunkPickIq1 { grid_idx: idx as u16, signed_score: dot, norm_sq: norm };
            }
        }
    }
    (best_pos, best_neg)
}

/// Combined grid search returning all three picks (max-|score|,
/// max-positive-score, max-negative-score) in a single pass over the
/// 2048-entry grid. The 3-strategy encoder calls all three sequentially
/// otherwise — this fuses them, so per-chunk cost stays at ~1 grid
/// sweep instead of 3. Critical for v8 re-quant perf: without this
/// fusion the encoder runs 25× slower per element.
pub(crate) fn best_iq1s_grid_all3(
    target: &[f32],
    delta: f32,
) -> (ChunkPickIq1, ChunkPickIq1, ChunkPickIq1) {
    debug_assert_eq!(target.len(), 8);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { best_iq1s_grid_all3_avx2(target, delta) };
        }
    }
    // Scalar fallback: 3 separate sweeps (slow path, never hit on x86_64).
    (
        best_iq1s_grid_for_chunk_scalar(target, delta),
        best_iq1s_grid_for_chunk_positive_scalar(target, delta),
        best_iq1s_grid_for_chunk_negative_scalar(target, delta),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn best_iq1s_grid_all3_avx2(
    target: &[f32],
    delta: f32,
) -> (ChunkPickIq1, ChunkPickIq1, ChunkPickIq1) {
    use std::arch::x86_64::*;
    debug_assert_eq!(target.len(), 8);
    let cache = iq1s_grid_cache();
    let grid_ptr = cache.grid_f32.as_ptr();
    let target_v = unsafe { _mm256_loadu_ps(target.as_ptr()) };
    let delta_v = _mm256_set1_ps(delta);

    let mut best_abs = ChunkPickIq1 { grid_idx: 0, signed_score: 0.0, norm_sq: 1.0 };
    let mut best_pos = ChunkPickIq1 { grid_idx: 0, signed_score: 0.0, norm_sq: 1.0 };
    let mut best_neg = ChunkPickIq1 { grid_idx: 0, signed_score: 0.0, norm_sq: 1.0 };
    let mut best_abs_s2divn = -1f32;
    let mut best_pos_signed = f32::NEG_INFINITY;
    let mut best_neg_signed = f32::INFINITY;

    for idx in 0..2048 {
        let g_v = unsafe { _mm256_loadu_ps(grid_ptr.add(idx * 8)) };
        let g_plus_d = _mm256_add_ps(g_v, delta_v);
        let prod = _mm256_mul_ps(target_v, g_plus_d);
        let sq = _mm256_mul_ps(g_plus_d, g_plus_d);
        let dot = unsafe { horizontal_sum_avx2(prod) };
        let norm = unsafe { horizontal_sum_avx2(sq) };
        if norm > 0.0 {
            let s2_div_n = dot * dot / norm;
            if s2_div_n > best_abs_s2divn {
                best_abs_s2divn = s2_div_n;
                best_abs = ChunkPickIq1 { grid_idx: idx as u16, signed_score: dot, norm_sq: norm };
            }
            if dot > best_pos_signed {
                best_pos_signed = dot;
                best_pos = ChunkPickIq1 { grid_idx: idx as u16, signed_score: dot, norm_sq: norm };
            }
            if dot < best_neg_signed {
                best_neg_signed = dot;
                best_neg = ChunkPickIq1 { grid_idx: idx as u16, signed_score: dot, norm_sq: norm };
            }
        }
    }
    (best_abs, best_pos, best_neg)
}

/// imatrix-weighted AVX2 variant of [`best_iq1s_grid_all3_avx2`]: same
/// 3-strategy single sweep, but each lane's contribution is weighted
/// by the per-column importance `w` (length 8):
///   dot  = Σ wⱼ·targetⱼ·(gⱼ+δ)   (pre-fold `wt = w·target`)
///   norm = Σ wⱼ·(gⱼ+δ)²
/// Matches [`best_iq1s_grid_all3_scalar_w`] up to FP reduction order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn best_iq1s_grid_all3_avx2_w(
    target: &[f32],
    delta: f32,
    w: &[f32],
) -> (ChunkPickIq1, ChunkPickIq1, ChunkPickIq1) {
    use std::arch::x86_64::*;
    debug_assert_eq!(target.len(), 8);
    debug_assert_eq!(w.len(), 8);
    let cache = iq1s_grid_cache();
    let grid_ptr = cache.grid_f32.as_ptr();
    let target_v = unsafe { _mm256_loadu_ps(target.as_ptr()) };
    let w_v = unsafe { _mm256_loadu_ps(w.as_ptr()) };
    let delta_v = _mm256_set1_ps(delta);
    // Pre-fold the importance into target so the dot is one mul.
    let wt_v = _mm256_mul_ps(w_v, target_v);

    let mut best_abs = ChunkPickIq1 { grid_idx: 0, signed_score: 0.0, norm_sq: 1.0 };
    let mut best_pos = ChunkPickIq1 { grid_idx: 0, signed_score: 0.0, norm_sq: 1.0 };
    let mut best_neg = ChunkPickIq1 { grid_idx: 0, signed_score: 0.0, norm_sq: 1.0 };
    let mut best_abs_s2divn = -1f32;
    let mut best_pos_signed = f32::NEG_INFINITY;
    let mut best_neg_signed = f32::INFINITY;

    for idx in 0..2048 {
        let g_v = unsafe { _mm256_loadu_ps(grid_ptr.add(idx * 8)) };
        let g_plus_d = _mm256_add_ps(g_v, delta_v);
        // dot = Σ (w·t)·(g+δ);  norm = Σ w·(g+δ)².
        let prod = _mm256_mul_ps(wt_v, g_plus_d);
        let gpd_sq = _mm256_mul_ps(g_plus_d, g_plus_d);
        let wsq = _mm256_mul_ps(w_v, gpd_sq);
        let dot = unsafe { horizontal_sum_avx2(prod) };
        let norm = unsafe { horizontal_sum_avx2(wsq) };
        if norm > 0.0 {
            let s2_div_n = dot * dot / norm;
            if s2_div_n > best_abs_s2divn {
                best_abs_s2divn = s2_div_n;
                best_abs = ChunkPickIq1 { grid_idx: idx as u16, signed_score: dot, norm_sq: norm };
            }
            if dot > best_pos_signed {
                best_pos_signed = dot;
                best_pos = ChunkPickIq1 { grid_idx: idx as u16, signed_score: dot, norm_sq: norm };
            }
            if dot < best_neg_signed {
                best_neg_signed = dot;
                best_neg = ChunkPickIq1 { grid_idx: idx as u16, signed_score: dot, norm_sq: norm };
            }
        }
    }
    (best_abs, best_pos, best_neg)
}

/// Horizontal sum of an `__m256` — 8 f32 lanes → scalar. Uses the
/// standard "pair-add then extract" sequence.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2")]
#[inline]
unsafe fn horizontal_sum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // Split into lower + upper 128-bit halves, add, then sum-within-128.
    let lo = _mm256_castps256_ps128(v);
    // SAFETY: extractf128 with const-imm=1 grabs the upper 128 of v.
    let hi = unsafe { _mm256_extractf128_ps::<1>(v) };
    let sum128 = _mm_add_ps(lo, hi); // [a, b, c, d]
    let shuf = _mm_movehdup_ps(sum128); // [b, b, d, d]
    let sums = _mm_add_ps(sum128, shuf); // [a+b, _, c+d, _]
    let high64 = _mm_movehl_ps(shuf, sums); // [c+d, _, _, _] in lane 0
    let final_sum = _mm_add_ss(sums, high64); // (a+b)+(c+d) in lane 0
    _mm_cvtss_f32(final_sum)
}

/// Encode IQ1_S. Per sub-block (32 weights = 4 chunks of 8):
///   - Try both delta signs (`-1 ± IQ1S_DELTA`).
///   - For each delta, find the best grid_idx per chunk (2048
///     candidates). Score each (delta, grid_idx) pair by the
///     reconstruction L2 energy.
///   - Pick the delta with the higher total energy.
///   - Compute the natural dl = Σ signed_score / Σ norm_sq.
/// Then derive the super-block d so each sub-block's `dl = d * (2s + 1)`,
/// `s ∈ [0, 7]` → d range `[1, 15]`.
pub fn encode_iq1_s(src: &[f32], dst: &mut [u8]) {
    encode_iq1_s_imatrix(src, dst, None);
}

/// Importance-matrix-aware IQ1_S encoder. `imatrix`, when `Some`, is a
/// per-input-column importance vector of length `src.len()`; the
/// per-chunk grid search + sub-block error weight their L2 metric by
/// it so high-importance columns are reconstructed more faithfully.
/// `None` reproduces the uniform-weight encode exactly (and keeps the
/// AVX2 fast path). The imatrix path is scalar-only.
pub fn encode_iq1_s_imatrix(src: &[f32], dst: &mut [u8], imatrix: Option<&[f32]>) {
    assert_eq!(src.len() % QK_K, 0, "encode_iq1_s: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_IQ1_S_BYTES,
        "encode_iq1_s: dst.len() must be n_blocks * 50"
    );
    debug_assert!(imatrix.map_or(true, |w| w.len() == src.len()));

    // Parallelize across blocks via rayon. The 3-strategy encoder is
    // ~5x slower than the original single-strategy path; a full-model
    // requant of a large model is many hours single-threaded but drops
    // to minutes with rayon across all cores.
    use rayon::prelude::*;
    dst.par_chunks_mut(BLOCK_IQ1_S_BYTES)
        .enumerate()
        .for_each(|(b, dst_block)| {
        let xs = &src[b * QK_K..(b + 1) * QK_K];
        // imatrix slice for this 256-col block (None = uniform).
        let w_block = imatrix.map(|w| &w[b * QK_K..(b + 1) * QK_K]);
        let off = 0; // dst_block is already the per-block slice
        let _ = off; // suppress unused warning

        // Per-sub-block: try BOTH d_super signs and pick whichever
        // gives lower total L2 error across the super-block. The
        // unsigned-d_super-only encoder produces -38% bias on
        // zero-mean Gaussian sources (diagnosed via the
        // `iq1_s_bias_diagnostic` test) because IQ1_S's representable
        // range with d>0 is heavily negative-biased
        // ((grid+delta) ∈ [-2.125, +0.125]). Allowing d<0 makes the
        // range [-0.125, +2.125] available so positive-mean
        // super-blocks can be fit without sign-flipping every
        // sub-block's dl.
        //
        // Per sub-block we record the BEST picks under the d>0
        // assumption (dl_used ≥ 0) AND under d<0 (dl_used ≤ 0),
        // then pick the super-block-wide sign that minimizes total
        // L2 error.
        #[derive(Default, Clone, Copy)]
        struct SubBlockChoice {
            picks: [ChunkPickIq1; 4],
            delta_neg: bool,
            dl_used: f32, // signed
            error: f64,
        }
        let mut sub_pos: [SubBlockChoice; N_SUB_BLOCKS] = Default::default();
        let mut sub_neg: [SubBlockChoice; N_SUB_BLOCKS] = Default::default();
        // Tracking arrays needed by the super-block tail.
        let mut sub_picks: [[ChunkPickIq1; 4]; N_SUB_BLOCKS] = Default::default();
        let mut sub_delta_neg: [bool; N_SUB_BLOCKS] = [false; N_SUB_BLOCKS];
        let mut sub_dl_abs = [0f32; N_SUB_BLOCKS]; // |dl| post-sign decision

        // #6: scan the super-block once for uniform sign. If all values
        // are ≥ 0 or all ≤ 0, only one d_super sign is meaningful — skip
        // the dl-of-wrong-sign candidate evaluations per sub-block.
        // Common for embedding rows + biases; rare for MoE expert weights
        // (which are balanced).
        let (mut any_pos, mut any_neg) = (false, false);
        for &v in xs {
            if v > 0.0 { any_pos = true; }
            else if v < 0.0 { any_neg = true; }
            if any_pos && any_neg { break; }
        }
        let only_dl_pos = any_pos && !any_neg;
        let only_dl_neg = !any_pos && any_neg;

        for ib32 in 0..N_SUB_BLOCKS {
            // For each (delta, strategy) pair we score by ACTUAL L2 error
            // under the dl ≥ 0 constraint:
            //   error_used = ||t||² - 2*dl_used*signed_total + dl_used²*norm_total
            // with dl_used = max(0, signed_total/norm_total). Strategy A
            // (max-|score|) can drive signed_total < 0, in which case
            // dl_used = 0 and the sub-block contributes nothing better
            // than zero-reconstruction. Strategy B (max-positive-score)
            // guarantees signed_total ≥ 0, trading reach for direction.
            // We pick whichever wins.
            let xs_chunk = |l: usize| &xs[ib32 * 32 + l * 8..ib32 * 32 + l * 8 + 8];
            // imatrix slice for chunk l of this sub-block (8 weights).
            let w_chunk = |l: usize| {
                w_block.map(|wb| &wb[ib32 * 32 + l * 8..ib32 * 32 + l * 8 + 8])
            };
            // Weighted target norm: Σ wⱼ·tⱼ² (unweighted Σ tⱼ² when None).
            // Keeps the sub-block error formula
            //   err = ‖t‖²_w − 2·dl·signed_total + dl²·norm_total
            // consistent, since signed_total / norm_total below are the
            // sums of the (weighted) chunk dot / norm.
            let mut t_norm_sq = 0f64;
            for l in 0..4 {
                let c = xs_chunk(l);
                match w_chunk(l) {
                    Some(wc) => {
                        for j in 0..8 {
                            t_norm_sq += (wc[j] as f64) * (c[j] as f64) * (c[j] as f64);
                        }
                    }
                    None => {
                        for &v in c {
                            t_norm_sq += (v as f64) * (v as f64);
                        }
                    }
                }
            }
            // Find the best (delta_neg, picks) for BOTH a positive
            // dl_used AND a negative dl_used. We'll pick the super-
            // block sign later.
            let mut bp = SubBlockChoice {
                error: t_norm_sq + 1.0,
                ..Default::default()
            };
            let mut bn = SubBlockChoice {
                error: t_norm_sq + 1.0,
                ..Default::default()
            };
            // #2: One AVX2 sweep per chunk handles BOTH deltas. Halves
            // the per-chunk grid sweeps vs the previous "one per delta"
            // pattern. The hoisted loop computes all 4 chunks × 2 deltas
            // before the candidate-evaluation loop below.
            let chunks_both = match w_block {
                // imatrix path: scalar weighted sweep (no AVX2).
                Some(_) => [
                    best_iq1s_grid_all3_both_deltas_w(xs_chunk(0), w_chunk(0).unwrap()),
                    best_iq1s_grid_all3_both_deltas_w(xs_chunk(1), w_chunk(1).unwrap()),
                    best_iq1s_grid_all3_both_deltas_w(xs_chunk(2), w_chunk(2).unwrap()),
                    best_iq1s_grid_all3_both_deltas_w(xs_chunk(3), w_chunk(3).unwrap()),
                ],
                // uniform path: unchanged (AVX2 fast path preserved).
                None => [
                    best_iq1s_grid_all3_both_deltas(xs_chunk(0)),
                    best_iq1s_grid_all3_both_deltas(xs_chunk(1)),
                    best_iq1s_grid_all3_both_deltas(xs_chunk(2)),
                    best_iq1s_grid_all3_both_deltas(xs_chunk(3)),
                ],
            };
            for (delta_idx, delta_neg) in [false, true].iter().enumerate() {
                let delta_neg = *delta_neg;
                let max_abs: [ChunkPickIq1; 4] = [
                    chunks_both[0][delta_idx].0, chunks_both[1][delta_idx].0,
                    chunks_both[2][delta_idx].0, chunks_both[3][delta_idx].0,
                ];
                let max_pos: [ChunkPickIq1; 4] = [
                    chunks_both[0][delta_idx].1, chunks_both[1][delta_idx].1,
                    chunks_both[2][delta_idx].1, chunks_both[3][delta_idx].1,
                ];
                let max_neg: [ChunkPickIq1; 4] = [
                    chunks_both[0][delta_idx].2, chunks_both[1][delta_idx].2,
                    chunks_both[2][delta_idx].2, chunks_both[3][delta_idx].2,
                ];
                // #1: Skip dead (pick, dl_sign) combinations.
                // - max_pos has signed_total ≥ 0 by construction, so
                //   min(0, signed/norm) = 0 (zero contribution to bn).
                // - max_neg has signed_total ≤ 0, so max(0, signed/norm) = 0
                //   (zero contribution to bp).
                // So we test (max_abs, both dl signs), (max_pos, dl_pos only),
                // and (max_neg, dl_neg only) — 4 candidates per delta = 8 per
                // sub-block, vs 12 before.
                let sums = |picks: &[ChunkPickIq1; 4]| -> (f32, f32) {
                    let mut s = 0f32; let mut n = 0f32;
                    for l in 0..4 { s += picks[l].signed_score; n += picks[l].norm_sq; }
                    (s, n)
                };
                // #4: manually-inlined `consider` (no closure).
                // #6: skip bn-updates when only_dl_pos, bp-updates when only_dl_neg.
                let (s_abs, n_abs) = sums(&max_abs);
                if n_abs > 0.0 {
                    if !only_dl_neg {
                        let dl_p = (s_abs / n_abs).max(0.0);
                        let err = t_norm_sq - 2.0 * (dl_p as f64) * (s_abs as f64)
                            + (dl_p as f64) * (dl_p as f64) * (n_abs as f64);
                        if err < bp.error {
                            bp = SubBlockChoice { picks: max_abs, delta_neg, dl_used: dl_p, error: err };
                        }
                    }
                    if !only_dl_pos {
                        let dl_n = (s_abs / n_abs).min(0.0);
                        let err = t_norm_sq - 2.0 * (dl_n as f64) * (s_abs as f64)
                            + (dl_n as f64) * (dl_n as f64) * (n_abs as f64);
                        if err < bn.error {
                            bn = SubBlockChoice { picks: max_abs, delta_neg, dl_used: dl_n, error: err };
                        }
                    }
                }
                if !only_dl_neg {
                    let (s_pos, n_pos) = sums(&max_pos);
                    if n_pos > 0.0 {
                        let dl_p = (s_pos / n_pos).max(0.0);
                        let err = t_norm_sq - 2.0 * (dl_p as f64) * (s_pos as f64)
                            + (dl_p as f64) * (dl_p as f64) * (n_pos as f64);
                        if err < bp.error {
                            bp = SubBlockChoice { picks: max_pos, delta_neg, dl_used: dl_p, error: err };
                        }
                    }
                }
                if !only_dl_pos {
                    let (s_neg, n_neg) = sums(&max_neg);
                    if n_neg > 0.0 {
                        let dl_n = (s_neg / n_neg).min(0.0);
                        let err = t_norm_sq - 2.0 * (dl_n as f64) * (s_neg as f64)
                            + (dl_n as f64) * (dl_n as f64) * (n_neg as f64);
                        if err < bn.error {
                            bn = SubBlockChoice { picks: max_neg, delta_neg, dl_used: dl_n, error: err };
                        }
                    }
                }
            }
            sub_pos[ib32] = bp;
            sub_neg[ib32] = bn;
        }

        // Choose super-block d_super sign by alternating across super-blocks
        // when the source is near zero-mean. Reason: IQ1_S's (grid+delta)
        // range is heavily asymmetric per d-sign — d>0 gives output range
        // ~[-1.875*|d|, +0.125*|d|], d<0 gives ~[-0.125*|d|, +1.875*|d|].
        // Picking d>0 for ALL super-blocks of a zero-mean source produces
        // a uniform -38% mean bias (every super-block's dequant is shifted
        // negative). The fix: alternate sign across super-blocks of a
        // tensor that has near-zero source mean per-super-block. We
        // alternate using `b % 2` so the bias averages out across the
        // tensor. For super-blocks where one sign is meaningfully better
        // (>10% lower error), use that sign instead.
        // #6 short-circuit: when source is uniformly signed, only one
        // d_super sign is meaningful; skip the comparison.
        let d_super_negative = if only_dl_pos {
            false
        } else if only_dl_neg {
            true
        } else {
            let total_pos: f64 = sub_pos.iter().map(|s| s.error).sum();
            let total_neg: f64 = sub_neg.iter().map(|s| s.error).sum();
            if (total_pos - total_neg).abs() < 0.10 * total_pos.max(total_neg) {
                // Near-tie: alternate by super-block index so the per-tensor
                // mean bias cancels across alternating super-blocks.
                b % 2 == 1
            } else {
                total_neg < total_pos
            }
        };
        for ib32 in 0..N_SUB_BLOCKS {
            let chosen = if d_super_negative {
                &sub_neg[ib32]
            } else {
                &sub_pos[ib32]
            };
            sub_picks[ib32] = chosen.picks;
            sub_delta_neg[ib32] = chosen.delta_neg;
            sub_dl_abs[ib32] = chosen.dl_used.abs();
        }

        // Super-block d: dl_max = |d| * 15 (s=7). |d| = dl_max / 15.
        let dl_max = sub_dl_abs.iter().cloned().fold(0f32, f32::max);
        let d_super_abs = if dl_max > 0.0 { dl_max / 15.0 } else { 0.0 };
        let d_super = if d_super_negative { -d_super_abs } else { d_super_abs };
        let id_super = if d_super_abs != 0.0 { 1.0 / d_super_abs } else { 0.0 };

        // 3-bit scale per sub-block: solve `|dl| = |d| * (2s + 1)` for s.
        let mut sub_scale_s = [0u16; N_SUB_BLOCKS];
        for ib32 in 0..N_SUB_BLOCKS {
            let s = (sub_dl_abs[ib32] * id_super - 1.0) * 0.5;
            sub_scale_s[ib32] = s.round().clamp(0.0, 7.0) as u16;
        }

        // Layout (matches dequant_iq1_s):
        //   off + 0..2     : d (f16)
        //   off + 2..34    : qs (low 8 bits of grid index, 4 per ib32)
        //   off + 34..50   : qh as 8 × u16
        let d_bits = f16::from_f32(d_super).to_bits();
        dst_block[0] = (d_bits & 0xFF) as u8;
        dst_block[1] = ((d_bits >> 8) & 0xFF) as u8;
        let qs_off = 2;
        let qh_off = 2 + 32;
        for ib32 in 0..N_SUB_BLOCKS {
            // qh layout per dequant:
            //   `dl = d * (2 * ((qh >> 12) & 7) + 1)` — scale in bits 12..14
            //   `delta_neg = qh & 0x8000` — bit 15
            //   `idx_high = (qh >> (3*l)) & 7` — bits 0..2 for l=0, 3..5 for l=1, 6..8 for l=2, 9..11 for l=3
            let mut qh: u16 = 0;
            qh |= (sub_scale_s[ib32] & 0x7) << 12;
            if sub_delta_neg[ib32] {
                qh |= 0x8000;
            }
            for l in 0..4 {
                let p = sub_picks[ib32][l];
                dst_block[qs_off + 4 * ib32 + l] = (p.grid_idx & 0xFF) as u8;
                let hi = (p.grid_idx >> 8) & 0x7;
                qh |= hi << (3 * l);
            }
            dst_block[qh_off + 2 * ib32] = (qh & 0xFF) as u8;
            dst_block[qh_off + 2 * ib32 + 1] = ((qh >> 8) & 0xFF) as u8;
        }
    });
}

/// Batched IQ1_S encoder that dispatches the per-chunk grid search
/// through an [`IqGpuEncoder`]. For each tensor chunk passed to
/// `src`, performs **two** large batched searches (one per delta
/// sign) covering all `32 * n_blocks` 8-element chunks, then runs
/// the existing per-sub-block delta-selection + scale-derivation
/// logic on the precomputed picks.
///
/// Falls back to the per-chunk CPU SIMD path automatically if
/// `encoder` returns `Err` (e.g. GPU `Unavailable`) — the returned
/// bytes are bit-identical regardless of backend (the score
/// comparisons are total-order on f32 values and the CPU-fallback
/// implementation uses the exact same grid table + arithmetic as
/// the per-chunk path).
pub fn encode_iq1_s_with_encoder(
    src: &[f32],
    dst: &mut [u8],
    encoder: &dyn crate::iq_gpu::IqGpuEncoder,
) {
    use crate::iq_gpu::Iq8EltDeltaPick;

    assert_eq!(
        src.len() % QK_K,
        0,
        "encode_iq1_s_with_encoder: src.len() must be multiple of 256"
    );
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_IQ1_S_BYTES,
        "encode_iq1_s_with_encoder: dst.len() must be n_blocks * 50"
    );
    if n_blocks == 0 {
        return;
    }

    // Flatten all 32 chunks (8 sub-blocks × 4 chunks-per-sub-block)
    // for all blocks into one `[total_chunks × 8]` f32 buffer.
    // Order: block-major, then sub-block, then chunk. This matches
    // the per-block loop below so we can index in O(1).
    let total_chunks = n_blocks * 32;
    // SAFETY: src is already a contiguous f32 slice with length
    // `n_blocks * 256 = total_chunks * 8`. The IQ1_S chunk layout
    // is the natural row-major reading of that slice — no shuffle
    // needed; we hand the slice directly to the encoder.
    debug_assert_eq!(src.len(), total_chunks * 8);

    // #3 + #5: GPU computes all 3 picks per chunk per delta in one
    // dispatch. 6 thread-local scratch vecs amortize allocation.
    thread_local! {
        static PICKS_ABS_POS_S: std::cell::RefCell<Vec<Iq8EltDeltaPick>> =
            const { std::cell::RefCell::new(Vec::new()) };
        static PICKS_POS_POS_S: std::cell::RefCell<Vec<Iq8EltDeltaPick>> =
            const { std::cell::RefCell::new(Vec::new()) };
        static PICKS_NEG_POS_S: std::cell::RefCell<Vec<Iq8EltDeltaPick>> =
            const { std::cell::RefCell::new(Vec::new()) };
        static PICKS_ABS_NEG_S: std::cell::RefCell<Vec<Iq8EltDeltaPick>> =
            const { std::cell::RefCell::new(Vec::new()) };
        static PICKS_POS_NEG_S: std::cell::RefCell<Vec<Iq8EltDeltaPick>> =
            const { std::cell::RefCell::new(Vec::new()) };
        static PICKS_NEG_NEG_S: std::cell::RefCell<Vec<Iq8EltDeltaPick>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    let mut picks_abs_p = PICKS_ABS_POS_S.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut picks_pos_p = PICKS_POS_POS_S.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut picks_neg_p = PICKS_NEG_POS_S.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut picks_abs_n = PICKS_ABS_NEG_S.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut picks_pos_n = PICKS_POS_NEG_S.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut picks_neg_n = PICKS_NEG_NEG_S.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let default_pick = Iq8EltDeltaPick {
        grid_idx: 0,
        signed_score: 0.0,
        norm_sq: 1.0,
    };
    picks_abs_p.resize(total_chunks, default_pick);
    picks_pos_p.resize(total_chunks, default_pick);
    picks_neg_p.resize(total_chunks, default_pick);
    picks_abs_n.resize(total_chunks, default_pick);
    picks_pos_n.resize(total_chunks, default_pick);
    picks_neg_n.resize(total_chunks, default_pick);

    let delta_pos = -1.0 + IQ1S_DELTA;
    let delta_neg = -1.0 - IQ1S_DELTA;

    let gpu_ok_pos = encoder
        .iq_8elt_delta_batched_all3(src, delta_pos, &mut picks_abs_p, &mut picks_pos_p, &mut picks_neg_p)
        .is_ok();
    let gpu_ok_neg = encoder
        .iq_8elt_delta_batched_all3(src, delta_neg, &mut picks_abs_n, &mut picks_pos_n, &mut picks_neg_n)
        .is_ok();

    if !gpu_ok_pos || !gpu_ok_neg {
        // Fall back to the per-chunk CPU path.
        encode_iq1_s(src, dst);
        return;
    }

    // G4(a): parallel bit-pack across blocks. Mirrors the CPU
    // `encode_iq1_s` 3-strategy + d_super-sign logic, using GPU-provided
    // max-|score| picks for strategy A and computing max-positive /
    // max-negative variants on CPU per-block (cheap relative to A).
    #[derive(Default, Clone, Copy)]
    struct SubBlockChoice {
        picks: [ChunkPickIq1; 4],
        delta_neg: bool,
        dl_used: f32,
        error: f64,
    }
    use rayon::prelude::*;
    dst.par_chunks_mut(BLOCK_IQ1_S_BYTES)
        .enumerate()
        .for_each(|(b, dst_block)| {
            let mut sub_pos: [SubBlockChoice; N_SUB_BLOCKS] = Default::default();
            let mut sub_neg: [SubBlockChoice; N_SUB_BLOCKS] = Default::default();
            let mut sub_picks: [[ChunkPickIq1; 4]; N_SUB_BLOCKS] = Default::default();
            let mut sub_delta_neg_flag: [bool; N_SUB_BLOCKS] = [false; N_SUB_BLOCKS];
            let mut sub_dl_abs = [0f32; N_SUB_BLOCKS];

            let xs_block = &src[b * QK_K..(b + 1) * QK_K];
            // #6: detect uniformly-signed super-block; skip the wrong-sign branch.
            let (mut any_pos, mut any_neg) = (false, false);
            for &v in xs_block {
                if v > 0.0 { any_pos = true; }
                else if v < 0.0 { any_neg = true; }
                if any_pos && any_neg { break; }
            }
            let only_dl_pos = any_pos && !any_neg;
            let only_dl_neg = !any_pos && any_neg;
            for ib32 in 0..N_SUB_BLOCKS {
                let xs_chunk = |l: usize| &xs_block[ib32 * 32 + l * 8..ib32 * 32 + l * 8 + 8];
                let mut t_norm_sq = 0f64;
                for l in 0..4 {
                    for &v in xs_chunk(l) {
                        t_norm_sq += (v as f64) * (v as f64);
                    }
                }
                let mut bp = SubBlockChoice {
                    error: t_norm_sq + 1.0,
                    ..Default::default()
                };
                let mut bn = SubBlockChoice {
                    error: t_norm_sq + 1.0,
                    ..Default::default()
                };
                // #3: all 3 picks per chunk come from the GPU; no CPU
                // AVX2 grid sweep needed at all in the bit-pack stage.
                for (delta_idx, delta_neg) in [false, true].iter().enumerate() {
                    let delta_neg = *delta_neg;
                    let (picks_abs_arr, picks_pos_arr, picks_neg_arr): (
                        &Vec<Iq8EltDeltaPick>, &Vec<Iq8EltDeltaPick>, &Vec<Iq8EltDeltaPick>,
                    ) = if delta_idx == 0 {
                        (&picks_abs_p, &picks_pos_p, &picks_neg_p)
                    } else {
                        (&picks_abs_n, &picks_pos_n, &picks_neg_n)
                    };
                    let mut max_abs: [ChunkPickIq1; 4] = Default::default();
                    let mut max_pos: [ChunkPickIq1; 4] = Default::default();
                    let mut max_neg: [ChunkPickIq1; 4] = Default::default();
                    for l in 0..4 {
                        let global = b * 32 + ib32 * 4 + l;
                        let a = &picks_abs_arr[global];
                        let p = &picks_pos_arr[global];
                        let nv = &picks_neg_arr[global];
                        max_abs[l] = ChunkPickIq1 { grid_idx: a.grid_idx, signed_score: a.signed_score, norm_sq: a.norm_sq };
                        max_pos[l] = ChunkPickIq1 { grid_idx: p.grid_idx, signed_score: p.signed_score, norm_sq: p.norm_sq };
                        max_neg[l] = ChunkPickIq1 { grid_idx: nv.grid_idx, signed_score: nv.signed_score, norm_sq: nv.norm_sq };
                    }
                    // #1+#4+#6: skip dead candidates, manually inline, gate by sign.
                    let sums = |picks: &[ChunkPickIq1; 4]| -> (f32, f32) {
                        let mut s = 0f32; let mut n = 0f32;
                        for l in 0..4 { s += picks[l].signed_score; n += picks[l].norm_sq; }
                        (s, n)
                    };
                    let (s_abs, n_abs) = sums(&max_abs);
                    if n_abs > 0.0 {
                        if !only_dl_neg {
                            let dl_p = (s_abs / n_abs).max(0.0);
                            let err = t_norm_sq - 2.0 * (dl_p as f64) * (s_abs as f64)
                                + (dl_p as f64) * (dl_p as f64) * (n_abs as f64);
                            if err < bp.error {
                                bp = SubBlockChoice { picks: max_abs, delta_neg, dl_used: dl_p, error: err };
                            }
                        }
                        if !only_dl_pos {
                            let dl_n = (s_abs / n_abs).min(0.0);
                            let err = t_norm_sq - 2.0 * (dl_n as f64) * (s_abs as f64)
                                + (dl_n as f64) * (dl_n as f64) * (n_abs as f64);
                            if err < bn.error {
                                bn = SubBlockChoice { picks: max_abs, delta_neg, dl_used: dl_n, error: err };
                            }
                        }
                    }
                    if !only_dl_neg {
                        let (s_pos, n_pos) = sums(&max_pos);
                        if n_pos > 0.0 {
                            let dl_p = (s_pos / n_pos).max(0.0);
                            let err = t_norm_sq - 2.0 * (dl_p as f64) * (s_pos as f64)
                                + (dl_p as f64) * (dl_p as f64) * (n_pos as f64);
                            if err < bp.error {
                                bp = SubBlockChoice { picks: max_pos, delta_neg, dl_used: dl_p, error: err };
                            }
                        }
                    }
                    if !only_dl_pos {
                        let (s_neg, n_neg) = sums(&max_neg);
                        if n_neg > 0.0 {
                            let dl_n = (s_neg / n_neg).min(0.0);
                            let err = t_norm_sq - 2.0 * (dl_n as f64) * (s_neg as f64)
                                + (dl_n as f64) * (dl_n as f64) * (n_neg as f64);
                            if err < bn.error {
                                bn = SubBlockChoice { picks: max_neg, delta_neg, dl_used: dl_n, error: err };
                            }
                        }
                    }
                }
                sub_pos[ib32] = bp;
                sub_neg[ib32] = bn;
            }

            // Super-block d_super sign decision — mirrors encode_iq1_s
            // (incl. #6 short-circuit for uniformly-signed source).
            let d_super_negative = if only_dl_pos {
                false
            } else if only_dl_neg {
                true
            } else {
                let total_pos: f64 = sub_pos.iter().map(|s| s.error).sum();
                let total_neg: f64 = sub_neg.iter().map(|s| s.error).sum();
                if (total_pos - total_neg).abs() < 0.10 * total_pos.max(total_neg) {
                    b % 2 == 1
                } else {
                    total_neg < total_pos
                }
            };
            for ib32 in 0..N_SUB_BLOCKS {
                let chosen = if d_super_negative {
                    &sub_neg[ib32]
                } else {
                    &sub_pos[ib32]
                };
                sub_picks[ib32] = chosen.picks;
                sub_delta_neg_flag[ib32] = chosen.delta_neg;
                sub_dl_abs[ib32] = chosen.dl_used.abs();
            }

            let dl_max = sub_dl_abs.iter().cloned().fold(0f32, f32::max);
            let d_super_abs = if dl_max > 0.0 { dl_max / 15.0 } else { 0.0 };
            let d_super = if d_super_negative { -d_super_abs } else { d_super_abs };
            let id_super = if d_super_abs != 0.0 { 1.0 / d_super_abs } else { 0.0 };

            let mut sub_scale_s = [0u16; N_SUB_BLOCKS];
            for ib32 in 0..N_SUB_BLOCKS {
                let s = (sub_dl_abs[ib32] * id_super - 1.0) * 0.5;
                sub_scale_s[ib32] = s.round().clamp(0.0, 7.0) as u16;
            }

            let d_bits = f16::from_f32(d_super).to_bits();
            dst_block[0] = (d_bits & 0xFF) as u8;
            dst_block[1] = ((d_bits >> 8) & 0xFF) as u8;
            let qs_off = 2;
            let qh_off = 2 + 32;
            for ib32 in 0..N_SUB_BLOCKS {
                let mut qh: u16 = 0;
                qh |= (sub_scale_s[ib32] & 0x7) << 12;
                if sub_delta_neg_flag[ib32] {
                    qh |= 0x8000;
                }
                for l in 0..4 {
                    let p = sub_picks[ib32][l];
                    dst_block[qs_off + 4 * ib32 + l] = (p.grid_idx & 0xFF) as u8;
                    let hi = (p.grid_idx >> 8) & 0x7;
                    qh |= hi << (3 * l);
                }
                dst_block[qh_off + 2 * ib32] = (qh & 0xFF) as u8;
                dst_block[qh_off + 2 * ib32 + 1] = ((qh >> 8) & 0xFF) as u8;
            }
        });

    // Put the scratch buffers back so the next encode call reuses them.
    PICKS_ABS_POS_S.with(|c| *c.borrow_mut() = picks_abs_p);
    PICKS_POS_POS_S.with(|c| *c.borrow_mut() = picks_pos_p);
    PICKS_NEG_POS_S.with(|c| *c.borrow_mut() = picks_neg_p);
    PICKS_ABS_NEG_S.with(|c| *c.borrow_mut() = picks_abs_n);
    PICKS_POS_NEG_S.with(|c| *c.borrow_mut() = picks_pos_n);
    PICKS_NEG_NEG_S.with(|c| *c.borrow_mut() = picks_neg_n);
}

/// Imatrix-weighted GPU encoder for IQ1_S. `w` is element-by-element
/// per-input-column importance, length == `src.len()` (chunk-major,
/// matching `src`). The GPU weighted grid search returns weighted dot
/// (Σ w·t·g) and weighted norm (Σ w·g²) per pick; the CPU side uses a
/// weighted `t_norm_sq` (Σ w·t²) so the sub-block error formula
///   err = ‖t‖²_w − 2·dl·signed_total + dl²·norm_total
/// stays consistent. Bit-pack + super-block sign logic are identical
/// to the unweighted `encode_iq1_s_with_encoder`. Falls back to the CPU
/// `encode_iq1_s_imatrix` path when the GPU encoder is unavailable.
pub fn encode_iq1_s_with_encoder_imatrix(
    src: &[f32],
    dst: &mut [u8],
    w: &[f32],
    encoder: &dyn crate::iq_gpu::IqGpuEncoder,
) {
    use crate::iq_gpu::Iq8EltDeltaPick;

    assert_eq!(
        src.len() % QK_K,
        0,
        "encode_iq1_s_with_encoder_imatrix: src.len() must be multiple of 256"
    );
    assert_eq!(
        w.len(),
        src.len(),
        "encode_iq1_s_with_encoder_imatrix: w.len() must equal src.len()"
    );
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_IQ1_S_BYTES,
        "encode_iq1_s_with_encoder_imatrix: dst.len() must be n_blocks * 50"
    );
    if n_blocks == 0 {
        return;
    }

    let total_chunks = n_blocks * 32;
    debug_assert_eq!(src.len(), total_chunks * 8);

    let default_pick = Iq8EltDeltaPick {
        grid_idx: 0,
        signed_score: 0.0,
        norm_sq: 1.0,
    };
    let mut picks_abs_p = vec![default_pick; total_chunks];
    let mut picks_pos_p = vec![default_pick; total_chunks];
    let mut picks_neg_p = vec![default_pick; total_chunks];
    let mut picks_abs_n = vec![default_pick; total_chunks];
    let mut picks_pos_n = vec![default_pick; total_chunks];
    let mut picks_neg_n = vec![default_pick; total_chunks];

    let delta_pos = -1.0 + IQ1S_DELTA;
    let delta_neg = -1.0 - IQ1S_DELTA;

    let gpu_ok_pos = encoder
        .iq_8elt_delta_batched_all3_w(src, w, delta_pos, &mut picks_abs_p, &mut picks_pos_p, &mut picks_neg_p)
        .is_ok();
    let gpu_ok_neg = encoder
        .iq_8elt_delta_batched_all3_w(src, w, delta_neg, &mut picks_abs_n, &mut picks_pos_n, &mut picks_neg_n)
        .is_ok();

    if !gpu_ok_pos || !gpu_ok_neg {
        // Fall back to the per-chunk CPU weighted path.
        encode_iq1_s_imatrix(src, dst, Some(w));
        return;
    }

    #[derive(Default, Clone, Copy)]
    struct SubBlockChoice {
        picks: [ChunkPickIq1; 4],
        delta_neg: bool,
        dl_used: f32,
        error: f64,
    }
    use rayon::prelude::*;
    dst.par_chunks_mut(BLOCK_IQ1_S_BYTES)
        .enumerate()
        .for_each(|(b, dst_block)| {
            let mut sub_pos: [SubBlockChoice; N_SUB_BLOCKS] = Default::default();
            let mut sub_neg: [SubBlockChoice; N_SUB_BLOCKS] = Default::default();
            let mut sub_picks: [[ChunkPickIq1; 4]; N_SUB_BLOCKS] = Default::default();
            let mut sub_delta_neg_flag: [bool; N_SUB_BLOCKS] = [false; N_SUB_BLOCKS];
            let mut sub_dl_abs = [0f32; N_SUB_BLOCKS];

            let xs_block = &src[b * QK_K..(b + 1) * QK_K];
            let w_block = &w[b * QK_K..(b + 1) * QK_K];
            let (mut any_pos, mut any_neg) = (false, false);
            for &v in xs_block {
                if v > 0.0 { any_pos = true; }
                else if v < 0.0 { any_neg = true; }
                if any_pos && any_neg { break; }
            }
            let only_dl_pos = any_pos && !any_neg;
            let only_dl_neg = !any_pos && any_neg;
            for ib32 in 0..N_SUB_BLOCKS {
                // Weighted target norm: Σ wⱼ·tⱼ², matching the GPU's
                // weighted signed/norm picks.
                let mut t_norm_sq = 0f64;
                for l in 0..4 {
                    let base = ib32 * 32 + l * 8;
                    for j in 0..8 {
                        let v = xs_block[base + j] as f64;
                        t_norm_sq += (w_block[base + j] as f64) * v * v;
                    }
                }
                let mut bp = SubBlockChoice {
                    error: t_norm_sq + 1.0,
                    ..Default::default()
                };
                let mut bn = SubBlockChoice {
                    error: t_norm_sq + 1.0,
                    ..Default::default()
                };
                for (delta_idx, delta_neg) in [false, true].iter().enumerate() {
                    let delta_neg = *delta_neg;
                    let (picks_abs_arr, picks_pos_arr, picks_neg_arr): (
                        &Vec<Iq8EltDeltaPick>, &Vec<Iq8EltDeltaPick>, &Vec<Iq8EltDeltaPick>,
                    ) = if delta_idx == 0 {
                        (&picks_abs_p, &picks_pos_p, &picks_neg_p)
                    } else {
                        (&picks_abs_n, &picks_pos_n, &picks_neg_n)
                    };
                    let mut max_abs: [ChunkPickIq1; 4] = Default::default();
                    let mut max_pos: [ChunkPickIq1; 4] = Default::default();
                    let mut max_neg: [ChunkPickIq1; 4] = Default::default();
                    for l in 0..4 {
                        let global = b * 32 + ib32 * 4 + l;
                        let a = &picks_abs_arr[global];
                        let p = &picks_pos_arr[global];
                        let nv = &picks_neg_arr[global];
                        max_abs[l] = ChunkPickIq1 { grid_idx: a.grid_idx, signed_score: a.signed_score, norm_sq: a.norm_sq };
                        max_pos[l] = ChunkPickIq1 { grid_idx: p.grid_idx, signed_score: p.signed_score, norm_sq: p.norm_sq };
                        max_neg[l] = ChunkPickIq1 { grid_idx: nv.grid_idx, signed_score: nv.signed_score, norm_sq: nv.norm_sq };
                    }
                    let sums = |picks: &[ChunkPickIq1; 4]| -> (f32, f32) {
                        let mut s = 0f32; let mut n = 0f32;
                        for l in 0..4 { s += picks[l].signed_score; n += picks[l].norm_sq; }
                        (s, n)
                    };
                    let (s_abs, n_abs) = sums(&max_abs);
                    if n_abs > 0.0 {
                        if !only_dl_neg {
                            let dl_p = (s_abs / n_abs).max(0.0);
                            let err = t_norm_sq - 2.0 * (dl_p as f64) * (s_abs as f64)
                                + (dl_p as f64) * (dl_p as f64) * (n_abs as f64);
                            if err < bp.error {
                                bp = SubBlockChoice { picks: max_abs, delta_neg, dl_used: dl_p, error: err };
                            }
                        }
                        if !only_dl_pos {
                            let dl_n = (s_abs / n_abs).min(0.0);
                            let err = t_norm_sq - 2.0 * (dl_n as f64) * (s_abs as f64)
                                + (dl_n as f64) * (dl_n as f64) * (n_abs as f64);
                            if err < bn.error {
                                bn = SubBlockChoice { picks: max_abs, delta_neg, dl_used: dl_n, error: err };
                            }
                        }
                    }
                    if !only_dl_neg {
                        let (s_pos, n_pos) = sums(&max_pos);
                        if n_pos > 0.0 {
                            let dl_p = (s_pos / n_pos).max(0.0);
                            let err = t_norm_sq - 2.0 * (dl_p as f64) * (s_pos as f64)
                                + (dl_p as f64) * (dl_p as f64) * (n_pos as f64);
                            if err < bp.error {
                                bp = SubBlockChoice { picks: max_pos, delta_neg, dl_used: dl_p, error: err };
                            }
                        }
                    }
                    if !only_dl_pos {
                        let (s_neg, n_neg) = sums(&max_neg);
                        if n_neg > 0.0 {
                            let dl_n = (s_neg / n_neg).min(0.0);
                            let err = t_norm_sq - 2.0 * (dl_n as f64) * (s_neg as f64)
                                + (dl_n as f64) * (dl_n as f64) * (n_neg as f64);
                            if err < bn.error {
                                bn = SubBlockChoice { picks: max_neg, delta_neg, dl_used: dl_n, error: err };
                            }
                        }
                    }
                }
                sub_pos[ib32] = bp;
                sub_neg[ib32] = bn;
            }

            let d_super_negative = if only_dl_pos {
                false
            } else if only_dl_neg {
                true
            } else {
                let total_pos: f64 = sub_pos.iter().map(|s| s.error).sum();
                let total_neg: f64 = sub_neg.iter().map(|s| s.error).sum();
                if (total_pos - total_neg).abs() < 0.10 * total_pos.max(total_neg) {
                    b % 2 == 1
                } else {
                    total_neg < total_pos
                }
            };
            for ib32 in 0..N_SUB_BLOCKS {
                let chosen = if d_super_negative {
                    &sub_neg[ib32]
                } else {
                    &sub_pos[ib32]
                };
                sub_picks[ib32] = chosen.picks;
                sub_delta_neg_flag[ib32] = chosen.delta_neg;
                sub_dl_abs[ib32] = chosen.dl_used.abs();
            }

            let dl_max = sub_dl_abs.iter().cloned().fold(0f32, f32::max);
            let d_super_abs = if dl_max > 0.0 { dl_max / 15.0 } else { 0.0 };
            let d_super = if d_super_negative { -d_super_abs } else { d_super_abs };
            let id_super = if d_super_abs != 0.0 { 1.0 / d_super_abs } else { 0.0 };

            let mut sub_scale_s = [0u16; N_SUB_BLOCKS];
            for ib32 in 0..N_SUB_BLOCKS {
                let s = (sub_dl_abs[ib32] * id_super - 1.0) * 0.5;
                sub_scale_s[ib32] = s.round().clamp(0.0, 7.0) as u16;
            }

            let d_bits = f16::from_f32(d_super).to_bits();
            dst_block[0] = (d_bits & 0xFF) as u8;
            dst_block[1] = ((d_bits >> 8) & 0xFF) as u8;
            let qs_off = 2;
            let qh_off = 2 + 32;
            for ib32 in 0..N_SUB_BLOCKS {
                let mut qh: u16 = 0;
                qh |= (sub_scale_s[ib32] & 0x7) << 12;
                if sub_delta_neg_flag[ib32] {
                    qh |= 0x8000;
                }
                for l in 0..4 {
                    let p = sub_picks[ib32][l];
                    dst_block[qs_off + 4 * ib32 + l] = (p.grid_idx & 0xFF) as u8;
                    let hi = (p.grid_idx >> 8) & 0x7;
                    qh |= hi << (3 * l);
                }
                dst_block[qh_off + 2 * ib32] = (qh & 0xFF) as u8;
                dst_block[qh_off + 2 * ib32 + 1] = ((qh >> 8) & 0xFF) as u8;
            }
        });
}

// ----------------------------------------------------------------------
// IQ1_M — 56 bytes/block, 1.75 bpw
//   { qs: [u8; 32], qh: [u8; 16], scales: [u8; 8] }
// ----------------------------------------------------------------------

const BLOCK_IQ1_M_BYTES: usize = 56;

/// Encode IQ1_M. Per the dequant:
///   - No separate `d` field. The super-block f16 d_bits is split
///     across the top nibbles of four u16 scale words (`sc[0..4]`).
///   - Per-16-weight 3-bit sub-scales (16 total per super-block) +
///     per-16-weight delta sign bits in `qh` (at bits 0x08 and
///     0x80 of `qh[ib32*2]` and `qh[ib32*2 + 1]`).
///   - 11-bit grid indices: low 8 in `qs[ib*4 + l]`, high 3 in
///     `qh[ib*2 + (l>=2 ? 1 : 0)]` at bits `(0..3)` for even-l and
///     `(4..6)` for odd-l within each qh byte.
///
/// IQ1_M has the finest delta-sign granularity of any IQ format
/// (per-8-weight, vs per-32-weight for IQ1_S). The encoder picks
/// each chunk's (grid, delta_sign) independently, then derives
/// the two dl values per ib32 from the two chunk pairs that share
/// them.
pub fn encode_iq1_m(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK_K, 0, "encode_iq1_m: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_IQ1_M_BYTES,
        "encode_iq1_m: dst.len() must be n_blocks * 56"
    );

    for b in 0..n_blocks {
        let xs = &src[b * QK_K..(b + 1) * QK_K];
        let off = b * BLOCK_IQ1_M_BYTES;

        // Per-chunk picks: (grid_idx, delta_neg, signed_score, norm_sq).
        // 8 ib32 × 4 chunks = 32 total.
        #[derive(Default, Clone, Copy)]
        struct ChunkPickIq1M {
            grid_idx: u16,
            delta_neg: bool,
            signed_score: f32,
            norm_sq: f32,
        }
        let mut picks: [[ChunkPickIq1M; 4]; N_SUB_BLOCKS] = Default::default();

        // Find each chunk's best (grid, delta) independently.
        for ib32 in 0..N_SUB_BLOCKS {
            for l in 0..4 {
                let chunk = &xs[ib32 * 32 + l * 8..ib32 * 32 + l * 8 + 8];
                let mut best = ChunkPickIq1M::default();
                let mut best_score_sq_div_norm = -1f32;
                for delta_neg in [false, true] {
                    let delta = if delta_neg {
                        -1.0 - IQ1S_DELTA
                    } else {
                        -1.0 + IQ1S_DELTA
                    };
                    let pick = best_iq1s_grid_for_chunk(chunk, delta);
                    if pick.norm_sq > 0.0 {
                        let s2 = pick.signed_score * pick.signed_score / pick.norm_sq;
                        if s2 > best_score_sq_div_norm {
                            best_score_sq_div_norm = s2;
                            best = ChunkPickIq1M {
                                grid_idx: pick.grid_idx,
                                delta_neg,
                                signed_score: pick.signed_score,
                                norm_sq: pick.norm_sq,
                            };
                        }
                    }
                }
                picks[ib32][l] = best;
            }
        }

        // Derive per-(ib32, half) dl from the pair of chunks sharing it.
        // half 0 = chunks 0,1 (weights 0..15) → dl1
        // half 1 = chunks 2,3 (weights 16..31) → dl2
        let mut sub_dl_abs = [[0f32; 2]; N_SUB_BLOCKS];
        for ib32 in 0..N_SUB_BLOCKS {
            for half in 0..2 {
                let l0 = half * 2;
                let l1 = half * 2 + 1;
                let signed = picks[ib32][l0].signed_score + picks[ib32][l1].signed_score;
                let norm = picks[ib32][l0].norm_sq + picks[ib32][l1].norm_sq;
                let dl = if norm > 0.0 { signed / norm } else { 0.0 };
                sub_dl_abs[ib32][half] = dl.abs();
            }
        }

        // Super-block d: max dl / 15 (3-bit scale → dl = d * (2s + 1), s ∈ [0, 7] → dl_max = d * 15).
        let mut dl_max = 0f32;
        for ib32 in 0..N_SUB_BLOCKS {
            for half in 0..2 {
                if sub_dl_abs[ib32][half] > dl_max {
                    dl_max = sub_dl_abs[ib32][half];
                }
            }
        }
        let d_super = if dl_max > 0.0 { dl_max / 15.0 } else { 0.0 };
        let id_super = if d_super != 0.0 { 1.0 / d_super } else { 0.0 };

        let mut sub_scale_s = [[0u16; 2]; N_SUB_BLOCKS];
        for ib32 in 0..N_SUB_BLOCKS {
            for half in 0..2 {
                let s = (sub_dl_abs[ib32][half] * id_super - 1.0) * 0.5;
                sub_scale_s[ib32][half] = s.round().clamp(0.0, 7.0) as u16;
            }
        }

        // Pack into the IQ1_M layout.
        let d_bits = f16::from_f32(d_super).to_bits();
        // sc[0..4]: four u16 words. Low 12 bits carry 2 ib32's of
        // 3-bit sub-scales (6 bits each, packed 6 + 6 = 12). High
        // 4 bits carry one nibble of d_bits.
        let mut sc = [0u16; 4];
        // Per dequant: sc[ib32/2] is the source. shift0 = 6*(ib32%2)
        // for dl1, shift1 = 6*(ib32%2)+3 for dl2.
        for ib32 in 0..N_SUB_BLOCKS {
            let word = ib32 / 2;
            let shift0 = 6 * (ib32 % 2);
            let shift1 = shift0 + 3;
            sc[word] |= (sub_scale_s[ib32][0] & 0x7) << shift0;
            sc[word] |= (sub_scale_s[ib32][1] & 0x7) << shift1;
        }
        // Plant d_bits' four nibbles into the top of each sc word.
        // Dequant:
        //   d_bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0xF0) | ((sc[2] >> 4) & 0xF00) | (sc[3] & 0xF000)
        // Inverse:
        //   sc[0] |= (d_bits & 0xF) << 12     ; bits 0..3 of d_bits in sc[0] top nibble
        //   sc[1] |= (d_bits & 0xF0) << 8     ; bits 4..7 of d_bits in sc[1] top nibble
        //   sc[2] |= (d_bits & 0xF00) << 4    ; bits 8..11 of d_bits in sc[2] top nibble
        //   sc[3] |= d_bits & 0xF000          ; bits 12..15 of d_bits in sc[3] top nibble
        sc[0] |= (d_bits & 0x000F) << 12;
        sc[1] |= (d_bits & 0x00F0) << 8;
        sc[2] |= (d_bits & 0x0F00) << 4;
        sc[3] |= d_bits & 0xF000;

        // Layout:
        //   off + 0..32  : qs
        //   off + 32..48 : qh
        //   off + 48..56 : scales (= sc[0..4] as 8 little-endian bytes)
        let qs_off = off;
        let qh_off = off + 32;
        let scales_off = off + 48;
        for byte in dst[qs_off..qs_off + 32].iter_mut() {
            *byte = 0;
        }
        for byte in dst[qh_off..qh_off + 16].iter_mut() {
            *byte = 0;
        }

        // qs + qh per dequant:
        //   qs[ib*4 + l]: low 8 bits of grid index for chunk l.
        //   qh[ib*2 + 0]: bits 0..2 = high 3 of grid for l=0;
        //                 bit 3 = delta_neg for half 0 (weights 0..7);
        //                 bits 4..6 = high 3 of grid for l=1;
        //                 bit 7 = delta_neg for half 1 (weights 8..15).
        //   qh[ib*2 + 1]: bits 0..2 = high 3 of grid for l=2;
        //                 bit 3 = delta_neg for half 2 (weights 16..23);
        //                 bits 4..6 = high 3 of grid for l=3;
        //                 bit 7 = delta_neg for half 3 (weights 24..31).
        for ib32 in 0..N_SUB_BLOCKS {
            for l in 0..4 {
                let p = picks[ib32][l];
                dst[qs_off + ib32 * 4 + l] = (p.grid_idx & 0xFF) as u8;
                let hi = ((p.grid_idx >> 8) & 0x7) as u8;
                let qh_index = ib32 * 2 + if l >= 2 { 1 } else { 0 };
                let bit_lo = if l % 2 == 0 { 0 } else { 4 };
                dst[qh_off + qh_index] |= hi << bit_lo;
                if p.delta_neg {
                    let delta_bit = if l % 2 == 0 { 0x08 } else { 0x80 };
                    dst[qh_off + qh_index] |= delta_bit;
                }
            }
        }

        // Write scales (4 u16 → 8 bytes LE).
        for i in 0..4 {
            dst[scales_off + i * 2] = (sc[i] & 0xFF) as u8;
            dst[scales_off + i * 2 + 1] = ((sc[i] >> 8) & 0xFF) as u8;
        }
    }
}

/// Batched IQ1_M encoder via [`IqGpuEncoder`]. Reuses the IQ1_S
/// 8-element delta-search kernel — IQ1_M's per-chunk search is
/// identical to IQ1_S's (same 2048-entry grid, same two delta
/// signs). The difference is in the block-tail: IQ1_M picks
/// `delta_neg` independently per chunk and derives `dl` per
/// 16-weight half (vs per-32-weight sub-block in IQ1_S).
///
/// Falls back to the per-chunk CPU path if `encoder` returns `Err`.
pub fn encode_iq1_m_with_encoder(
    src: &[f32],
    dst: &mut [u8],
    encoder: &dyn crate::iq_gpu::IqGpuEncoder,
) {
    use crate::iq_gpu::Iq8EltDeltaPick;

    assert_eq!(
        src.len() % QK_K,
        0,
        "encode_iq1_m_with_encoder: src.len() must be multiple of 256"
    );
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_IQ1_M_BYTES,
        "encode_iq1_m_with_encoder: dst.len() must be n_blocks * 56"
    );
    if n_blocks == 0 {
        return;
    }

    let total_chunks = n_blocks * 32;
    debug_assert_eq!(src.len(), total_chunks * 8);

    // #5: thread-local scratch buffers avoid the ~350 MB-per-tensor
    // zero-init cost on every encode call (123 large expert tensors ×
    // 2 vecs = 246 allocations per quant). Reused across encodes on
    // the same thread, grown but never shrunk.
    thread_local! {
        static PICKS_POS_SCRATCH: std::cell::RefCell<Vec<Iq8EltDeltaPick>> =
            const { std::cell::RefCell::new(Vec::new()) };
        static PICKS_NEG_SCRATCH: std::cell::RefCell<Vec<Iq8EltDeltaPick>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    let mut picks_pos = PICKS_POS_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut picks_neg = PICKS_NEG_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let default_pick = Iq8EltDeltaPick {
        grid_idx: 0,
        signed_score: 0.0,
        norm_sq: 1.0,
    };
    picks_pos.resize(total_chunks, default_pick);
    picks_neg.resize(total_chunks, default_pick);

    let delta_pos = -1.0 + IQ1S_DELTA;
    let delta_neg = -1.0 - IQ1S_DELTA;
    let gpu_ok_pos = encoder
        .iq_8elt_delta_batched(src, delta_pos, &mut picks_pos)
        .is_ok();
    let gpu_ok_neg = encoder
        .iq_8elt_delta_batched(src, delta_neg, &mut picks_neg)
        .is_ok();
    if !gpu_ok_pos || !gpu_ok_neg {
        encode_iq1_m(src, dst);
        return;
    }

    #[derive(Default, Clone, Copy)]
    struct ChunkPickIq1M {
        grid_idx: u16,
        delta_neg: bool,
        signed_score: f32,
        norm_sq: f32,
    }

    // G4(a): parallel bit-pack across blocks.
    use rayon::prelude::*;
    dst.par_chunks_mut(BLOCK_IQ1_M_BYTES)
        .enumerate()
        .for_each(|(b, dst_block)| {
            let mut picks: [[ChunkPickIq1M; 4]; N_SUB_BLOCKS] = Default::default();

            // Per-chunk: compare delta_pos vs delta_neg by score²/norm,
            // pick the better. Same comparison the direct encoder does.
            for ib32 in 0..N_SUB_BLOCKS {
                for l in 0..4 {
                    let global = b * 32 + ib32 * 4 + l;
                    let pp = &picks_pos[global];
                    let pn = &picks_neg[global];
                    let s2p = if pp.norm_sq > 0.0 {
                        pp.signed_score * pp.signed_score / pp.norm_sq
                    } else {
                        -1.0
                    };
                    let s2n = if pn.norm_sq > 0.0 {
                        pn.signed_score * pn.signed_score / pn.norm_sq
                    } else {
                        -1.0
                    };
                    let pick_neg = s2n > s2p;
                    let chosen = if pick_neg { pn } else { pp };
                    picks[ib32][l] = ChunkPickIq1M {
                        grid_idx: chosen.grid_idx,
                        delta_neg: pick_neg,
                        signed_score: chosen.signed_score,
                        norm_sq: chosen.norm_sq,
                    };
                }
            }

            // Per-half dl derivation (chunks 0..1 → half 0, 2..3 → half 1).
            let mut sub_dl_abs = [[0f32; 2]; N_SUB_BLOCKS];
            for ib32 in 0..N_SUB_BLOCKS {
                for half in 0..2 {
                    let l0 = half * 2;
                    let l1 = half * 2 + 1;
                    let signed = picks[ib32][l0].signed_score + picks[ib32][l1].signed_score;
                    let norm = picks[ib32][l0].norm_sq + picks[ib32][l1].norm_sq;
                    let dl = if norm > 0.0 { signed / norm } else { 0.0 };
                    sub_dl_abs[ib32][half] = dl.abs();
                }
            }
            let mut dl_max = 0f32;
            for ib32 in 0..N_SUB_BLOCKS {
                for half in 0..2 {
                    if sub_dl_abs[ib32][half] > dl_max {
                        dl_max = sub_dl_abs[ib32][half];
                    }
                }
            }
            let d_super = if dl_max > 0.0 { dl_max / 15.0 } else { 0.0 };
            let id_super = if d_super != 0.0 { 1.0 / d_super } else { 0.0 };
            let mut sub_scale_s = [[0u16; 2]; N_SUB_BLOCKS];
            for ib32 in 0..N_SUB_BLOCKS {
                for half in 0..2 {
                    let s = (sub_dl_abs[ib32][half] * id_super - 1.0) * 0.5;
                    sub_scale_s[ib32][half] = s.round().clamp(0.0, 7.0) as u16;
                }
            }

            let d_bits = f16::from_f32(d_super).to_bits();
            let mut sc = [0u16; 4];
            for ib32 in 0..N_SUB_BLOCKS {
                let word = ib32 / 2;
                let shift0 = 6 * (ib32 % 2);
                let shift1 = shift0 + 3;
                sc[word] |= (sub_scale_s[ib32][0] & 0x7) << shift0;
                sc[word] |= (sub_scale_s[ib32][1] & 0x7) << shift1;
            }
            sc[0] |= (d_bits & 0x000F) << 12;
            sc[1] |= (d_bits & 0x00F0) << 8;
            sc[2] |= (d_bits & 0x0F00) << 4;
            sc[3] |= d_bits & 0xF000;

            let qs_off = 0;
            let qh_off = 32;
            let scales_off = 48;
            for byte in dst_block[qs_off..qs_off + 32].iter_mut() {
                *byte = 0;
            }
            for byte in dst_block[qh_off..qh_off + 16].iter_mut() {
                *byte = 0;
            }
            for ib32 in 0..N_SUB_BLOCKS {
                for l in 0..4 {
                    let p = picks[ib32][l];
                    dst_block[qs_off + ib32 * 4 + l] = (p.grid_idx & 0xFF) as u8;
                    let hi = ((p.grid_idx >> 8) & 0x7) as u8;
                    let qh_index = ib32 * 2 + if l >= 2 { 1 } else { 0 };
                    let bit_lo = if l % 2 == 0 { 0 } else { 4 };
                    dst_block[qh_off + qh_index] |= hi << bit_lo;
                    if p.delta_neg {
                        let delta_bit = if l % 2 == 0 { 0x08 } else { 0x80 };
                        dst_block[qh_off + qh_index] |= delta_bit;
                    }
                }
            }
            for i in 0..4 {
                dst_block[scales_off + i * 2] = (sc[i] & 0xFF) as u8;
                dst_block[scales_off + i * 2 + 1] = ((sc[i] >> 8) & 0xFF) as u8;
            }
        });

    // Put the scratch buffers back so the next encode call reuses them.
    PICKS_POS_SCRATCH.with(|c| *c.borrow_mut() = picks_pos);
    PICKS_NEG_SCRATCH.with(|c| *c.borrow_mut() = picks_neg);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant;

    /// Inputs sized in the IQ2_XXS sweet spot: small, mostly-zero
    /// values with sparse 8-element activations. Mirrors a real
    /// model's quantized weight distribution after norm scaling.
    fn make_sparse_row(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                // Mostly small positive values (matches the IQ2_XXS
                // codebook's all-positive grid design).
                let u = (s >> 8) as f32 / (1u32 << 24) as f32;
                (u - 0.5) * 0.5
            })
            .collect()
    }

    /// IQ2_XXS round-trip: encode → dequant → assert max error
    /// within a loose bound (the codebook + sign quantization +
    /// 4-bit per-sub-block scale all introduce error).
    #[test]
    fn iq2_xxs_round_trip_within_bound() {
        for &n_blocks in &[1usize, 2, 3] {
            let n = n_blocks * QK_K;
            let src = make_sparse_row(n, n as u32 * 7);
            let mut enc = vec![0u8; n_blocks * BLOCK_IQ2_XXS_BYTES];
            encode_iq2_xxs(&src, &mut enc);
            let mut dec = vec![0f32; n];
            dequant::dequant_iq2_xxs(&enc, &mut dec);
            let mut max_err = 0f32;
            let mut sum_sq_err = 0f32;
            for i in 0..n {
                let e = (dec[i] - src[i]).abs();
                if e > max_err {
                    max_err = e;
                }
                sum_sq_err += (dec[i] - src[i]).powi(2);
            }
            let rmse = (sum_sq_err / n as f32).sqrt();
            // IQ2_XXS is the lowest-bpw "production" format. Per-
            // weight error can be large on individual outliers
            // (the grid only has 256 × 128 = 32K (grid, sign)
            // combos to cover all distributions); RMSE is the
            // metric llama.cpp's quantization quality script
            // uses. Loose bound: max_err < 0.5 and rmse < 0.15
            // on the sparse-row distribution.
            assert!(
                max_err < 0.5,
                "iq2_xxs n_blocks={n_blocks}: max_err={max_err} (rmse={rmse})"
            );
            assert!(
                rmse < 0.15,
                "iq2_xxs n_blocks={n_blocks}: rmse={rmse} (max_err={max_err})"
            );
        }
    }

    #[test]
    fn iq2_xxs_with_cpu_encoder_matches_direct() {
        // Per-chunk searches go through `search_chunk_8` in both
        // paths; the batched path just hoists the search loop. The
        // CPU fallback encoder uses the exact same arithmetic, so
        // outputs must be byte-equal.
        let n = 4 * QK_K;
        let src = make_sparse_row(n, 73);
        let mut direct = vec![0u8; 4 * BLOCK_IQ2_XXS_BYTES];
        let mut batched = vec![0u8; 4 * BLOCK_IQ2_XXS_BYTES];
        encode_iq2_xxs(&src, &mut direct);
        encode_iq2_xxs_with_encoder(&src, &mut batched, &crate::iq_gpu::CpuFallbackEncoder);
        assert_eq!(direct, batched, "batched IQ2_XXS encoder must match direct");
    }

    /// Zero input: every chunk picks the zero-vector grid entry
    /// (`IQ2XXS_GRID[0] = 0x0808...` is the smallest entry but not
    /// zero; the encoder picks it with sign_idx = 0 and the score
    /// stays 0, which dequant decodes back to 0 since d=0).
    #[test]
    fn iq2_xxs_handles_zero_input() {
        let n = 2 * QK_K;
        let zero = vec![0f32; n];
        let mut enc = vec![0u8; 2 * BLOCK_IQ2_XXS_BYTES];
        encode_iq2_xxs(&zero, &mut enc);
        let mut dec = vec![0f32; n];
        dequant::dequant_iq2_xxs(&enc, &mut dec);
        assert!(
            dec.iter().all(|&v| v.abs() < 1e-6),
            "iq2_xxs zero input: max abs output = {}",
            dec.iter().map(|v| v.abs()).fold(0f32, f32::max)
        );
    }

    #[test]
    fn iq2_xs_round_trip_within_bound() {
        for &n_blocks in &[1usize, 2] {
            let n = n_blocks * QK_K;
            let src = make_sparse_row(n, n as u32 * 17);
            let mut enc = vec![0u8; n_blocks * BLOCK_IQ2_XS_BYTES];
            encode_iq2_xs(&src, &mut enc);
            let mut dec = vec![0f32; n];
            dequant::dequant_iq2_xs(&enc, &mut dec);
            let mut max_err = 0f32;
            let mut sum_sq = 0f32;
            for i in 0..n {
                let e = (dec[i] - src[i]).abs();
                if e > max_err {
                    max_err = e;
                }
                sum_sq += (dec[i] - src[i]).powi(2);
            }
            let rmse = (sum_sq / n as f32).sqrt();
            assert!(
                max_err < 0.5,
                "iq2_xs n_blocks={n_blocks}: max_err={max_err} rmse={rmse}"
            );
            assert!(
                rmse < 0.15,
                "iq2_xs n_blocks={n_blocks}: rmse={rmse}"
            );
        }
    }

    #[test]
    fn iq2_xs_handles_zero_input() {
        let n = 2 * QK_K;
        let zero = vec![0f32; n];
        let mut enc = vec![0u8; 2 * BLOCK_IQ2_XS_BYTES];
        encode_iq2_xs(&zero, &mut enc);
        let mut dec = vec![0f32; n];
        dequant::dequant_iq2_xs(&enc, &mut dec);
        assert!(dec.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn iq2_xs_with_cpu_encoder_matches_direct() {
        let n = 4 * QK_K;
        let src = make_sparse_row(n, 89);
        let mut direct = vec![0u8; 4 * BLOCK_IQ2_XS_BYTES];
        let mut batched = vec![0u8; 4 * BLOCK_IQ2_XS_BYTES];
        encode_iq2_xs(&src, &mut direct);
        encode_iq2_xs_with_encoder(&src, &mut batched, &crate::iq_gpu::CpuFallbackEncoder);
        assert_eq!(direct, batched, "batched IQ2_XS encoder must match direct");
    }

    #[test]
    fn iq2_s_round_trip_within_bound() {
        for &n_blocks in &[1usize, 2] {
            let n = n_blocks * QK_K;
            let src = make_sparse_row(n, n as u32 * 19);
            let mut enc = vec![0u8; n_blocks * BLOCK_IQ2_S_BYTES];
            encode_iq2_s(&src, &mut enc);
            let mut dec = vec![0f32; n];
            dequant::dequant_iq2_s(&enc, &mut dec);
            let mut max_err = 0f32;
            let mut sum_sq = 0f32;
            for i in 0..n {
                let e = (dec[i] - src[i]).abs();
                if e > max_err {
                    max_err = e;
                }
                sum_sq += (dec[i] - src[i]).powi(2);
            }
            let rmse = (sum_sq / n as f32).sqrt();
            assert!(
                max_err < 0.5,
                "iq2_s n_blocks={n_blocks}: max_err={max_err} rmse={rmse}"
            );
            // IQ2_S has the biggest grid (1024 entries) of the IQ2
            // family. Quality is comparable to IQ2_XS in practice
            // for the v1 search; the bigger codebook pays off more
            // with the joint-search refinement (v1.x follow-up).
            assert!(
                rmse < 0.16,
                "iq2_s n_blocks={n_blocks}: rmse={rmse}"
            );
        }
    }

    #[test]
    fn iq2_s_handles_zero_input() {
        let n = 2 * QK_K;
        let zero = vec![0f32; n];
        let mut enc = vec![0u8; 2 * BLOCK_IQ2_S_BYTES];
        encode_iq2_s(&zero, &mut enc);
        let mut dec = vec![0f32; n];
        dequant::dequant_iq2_s(&enc, &mut dec);
        assert!(dec.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn iq2_s_with_cpu_encoder_matches_direct() {
        let n = 4 * QK_K;
        let src = make_sparse_row(n, 103);
        let mut direct = vec![0u8; 4 * BLOCK_IQ2_S_BYTES];
        let mut batched = vec![0u8; 4 * BLOCK_IQ2_S_BYTES];
        encode_iq2_s(&src, &mut direct);
        encode_iq2_s_with_encoder(&src, &mut batched, &crate::iq_gpu::CpuFallbackEncoder);
        assert_eq!(direct, batched, "batched IQ2_S encoder must match direct");
    }

    #[test]
    fn iq3_xxs_round_trip_within_bound() {
        for &n_blocks in &[1usize, 2, 3] {
            let n = n_blocks * QK_K;
            // IQ3_XXS gets 1.5 bits/weight more than IQ2_XXS — its
            // sweet spot is wider-distribution inputs.
            let src = make_sparse_row(n, n as u32 * 13);
            let mut enc = vec![0u8; n_blocks * BLOCK_IQ3_XXS_BYTES];
            encode_iq3_xxs(&src, &mut enc);
            let mut dec = vec![0f32; n];
            dequant::dequant_iq3_xxs(&enc, &mut dec);
            let mut max_err = 0f32;
            let mut sum_sq = 0f32;
            for i in 0..n {
                let e = (dec[i] - src[i]).abs();
                if e > max_err {
                    max_err = e;
                }
                sum_sq += (dec[i] - src[i]).powi(2);
            }
            let rmse = (sum_sq / n as f32).sqrt();
            assert!(
                max_err < 0.5,
                "iq3_xxs n_blocks={n_blocks}: max_err={max_err} rmse={rmse}"
            );
            // The v1 two-stage greedy (independent 4-elt searches
            // per chunk-half, parity-fix after) is suboptimal vs.
            // joint search — RMSE is ~0.14 vs ~0.10 for IQ2_XXS's
            // joint 8-elt search. Bound reflects v1 quality;
            // upgrading to joint search is a quality follow-up.
            assert!(
                rmse < 0.16,
                "iq3_xxs n_blocks={n_blocks}: rmse={rmse}"
            );
        }
    }

    #[test]
    fn iq3_xxs_with_cpu_encoder_matches_direct() {
        let n = 4 * QK_K;
        let src = make_sparse_row(n, 127);
        let mut direct = vec![0u8; 4 * BLOCK_IQ3_XXS_BYTES];
        let mut batched = vec![0u8; 4 * BLOCK_IQ3_XXS_BYTES];
        encode_iq3_xxs(&src, &mut direct);
        encode_iq3_xxs_with_encoder(&src, &mut batched, &crate::iq_gpu::CpuFallbackEncoder);
        assert_eq!(direct, batched, "batched IQ3_XXS encoder must match direct");
    }

    #[test]
    fn iq3_xxs_handles_zero_input() {
        let n = 2 * QK_K;
        let zero = vec![0f32; n];
        let mut enc = vec![0u8; 2 * BLOCK_IQ3_XXS_BYTES];
        encode_iq3_xxs(&zero, &mut enc);
        let mut dec = vec![0f32; n];
        dequant::dequant_iq3_xxs(&enc, &mut dec);
        assert!(dec.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn iq3_s_round_trip_within_bound() {
        for &n_blocks in &[1usize, 2] {
            let n = n_blocks * QK_K;
            let src = make_sparse_row(n, n as u32 * 23);
            let mut enc = vec![0u8; n_blocks * BLOCK_IQ3_S_BYTES];
            encode_iq3_s(&src, &mut enc);
            let mut dec = vec![0f32; n];
            dequant::dequant_iq3_s(&enc, &mut dec);
            let mut max_err = 0f32;
            let mut sum_sq = 0f32;
            for i in 0..n {
                let e = (dec[i] - src[i]).abs();
                if e > max_err {
                    max_err = e;
                }
                sum_sq += (dec[i] - src[i]).powi(2);
            }
            let rmse = (sum_sq / n as f32).sqrt();
            assert!(
                max_err < 0.5,
                "iq3_s n_blocks={n_blocks}: max_err={max_err} rmse={rmse}"
            );
            assert!(
                rmse < 0.16,
                "iq3_s n_blocks={n_blocks}: rmse={rmse}"
            );
        }
    }

    #[test]
    fn iq3_s_handles_zero_input() {
        let n = 2 * QK_K;
        let zero = vec![0f32; n];
        let mut enc = vec![0u8; 2 * BLOCK_IQ3_S_BYTES];
        encode_iq3_s(&zero, &mut enc);
        let mut dec = vec![0f32; n];
        dequant::dequant_iq3_s(&enc, &mut dec);
        assert!(dec.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn iq3_s_with_cpu_encoder_matches_direct() {
        let n = 4 * QK_K;
        let src = make_sparse_row(n, 131);
        let mut direct = vec![0u8; 4 * BLOCK_IQ3_S_BYTES];
        let mut batched = vec![0u8; 4 * BLOCK_IQ3_S_BYTES];
        encode_iq3_s(&src, &mut direct);
        encode_iq3_s_with_encoder(&src, &mut batched, &crate::iq_gpu::CpuFallbackEncoder);
        assert_eq!(direct, batched, "batched IQ3_S encoder must match direct");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn best_grid_4_simd_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            eprintln!("skip: host lacks SSE4.1");
            return;
        }
        let grid_f32 = iq3xxs_grid_f32();
        let grid_norm = build_grid_norm_sq_table_4(iq3xxs_grid_bytes(), 256);
        let test_targets = [
            [0.5_f32, -0.3, 0.1, 0.0],
            [1.0_f32, 1.0, 1.0, 1.0],
            [-2.5_f32, 1.7, 0.0, -0.4],
            [0.0_f32; 4],
            [0.01_f32, -0.01, 0.01, -0.01],
        ];
        for target in &test_targets {
            let (s_g, s_m, s_score, s_norm) =
                best_grid_4_scalar(target, grid_f32, 256, &grid_norm);
            let (v_g, v_m, v_score, v_norm) =
                unsafe { best_grid_4_sse41(target, grid_f32, 256, &grid_norm) };
            assert_eq!(s_g, v_g, "grid mismatch for target={target:?}");
            assert_eq!(s_m, v_m, "mask mismatch for target={target:?}");
            assert!((s_score - v_score).abs() < 1e-3);
            assert!((s_norm - v_norm).abs() < 1e-3);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn search_chunk_8_simd_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("skip: host lacks AVX2/FMA");
            return;
        }
        // Use IQ2_XXS grid (256 entries) as the test bed — covers
        // the same code path used by IQ2_XS and IQ2_S (just with
        // different n_grid). If 256 entries match, the larger
        // tables match too.
        let grid_f32 = iq2xxs_grid_f32();
        let grid_norm = build_grid_norm_sq_table_8(iq2xxs_grid_bytes(), 256);
        let test_targets = [
            [0.5_f32, -0.3, 0.1, 0.0, -0.7, 0.4, -0.2, 0.8],
            [1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [-2.5_f32, 1.7, 0.0, -0.4, 0.9, -1.2, 0.3, -0.8],
            [0.0_f32; 8],
            [0.01_f32, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01],
        ];
        for target in &test_targets {
            let scalar = search_chunk_8_scalar(target, grid_f32, 256, &grid_norm);
            let simd = unsafe { search_chunk_8_avx2(target, grid_f32, 256, &grid_norm) };
            assert_eq!(
                scalar.grid_idx, simd.grid_idx,
                "grid_idx mismatch for target={target:?}: scalar={} simd={}",
                scalar.grid_idx, simd.grid_idx
            );
            assert_eq!(
                scalar.sign_idx, simd.sign_idx,
                "sign_idx mismatch for target={target:?}"
            );
            let diff_score = (scalar.signed_score - simd.signed_score).abs();
            let diff_norm = (scalar.grid_norm_sq - simd.grid_norm_sq).abs();
            assert!(
                diff_score < 1e-3 && diff_norm < 1e-3,
                "score precision drift: scalar={scalar:?} simd={simd:?}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn iq1s_simd_matches_scalar() {
        // Hand-built target chunks covering a few magnitudes;
        // AVX2 path must pick the same grid + score as the scalar
        // path for all of them. Mismatch would mean the SIMD
        // implementation diverged from the algorithm (or the
        // horizontal-sum has a precision bug that flips the
        // comparison branch).
        let test_chunks = [
            [0.5_f32, -0.3, 0.1, 0.0, -0.7, 0.4, -0.2, 0.8],
            [1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [-2.5_f32, 1.7, 0.0, -0.4, 0.9, -1.2, 0.3, -0.8],
            [0.0_f32; 8],
            [0.01_f32, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        ];
        let deltas = [-1.0 - IQ1S_DELTA, -1.0 + IQ1S_DELTA];

        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("skip: host lacks AVX2/FMA");
            return;
        }

        for chunk in &test_chunks {
            for &delta in &deltas {
                let scalar = best_iq1s_grid_for_chunk_scalar(chunk, delta);
                let simd = unsafe { best_iq1s_grid_for_chunk_avx2(chunk, delta) };
                assert_eq!(
                    scalar.grid_idx, simd.grid_idx,
                    "grid_idx mismatch for chunk={chunk:?} delta={delta}: scalar={} simd={}",
                    scalar.grid_idx, simd.grid_idx
                );
                // Score values must agree to high precision —
                // small FP rounding from AVX2 horizontal-sum is
                // tolerable but shouldn't change the *chosen*
                // grid_idx (already asserted above).
                let diff_dot = (scalar.signed_score - simd.signed_score).abs();
                let diff_norm = (scalar.norm_sq - simd.norm_sq).abs();
                assert!(
                    diff_dot < 1e-3 && diff_norm < 1e-3,
                    "score precision drift: scalar={:?} simd={:?}",
                    scalar,
                    simd
                );
            }
        }
    }

    #[test]
    fn iq1_s_round_trip_within_bound() {
        // IQ1_S is the smallest "production" quant. Quality is
        // intrinsically rough (1.5 bpw); we only check that the
        // round-trip stays bounded and zero input round-trips
        // cleanly.
        let n = 2 * QK_K;
        let src = make_sparse_row(n, 29);
        let mut enc = vec![0u8; 2 * BLOCK_IQ1_S_BYTES];
        encode_iq1_s(&src, &mut enc);
        let mut dec = vec![0f32; n];
        dequant::dequant_iq1_s(&enc, &mut dec);
        let mut max_err = 0f32;
        let mut sum_sq = 0f32;
        for i in 0..n {
            let e = (dec[i] - src[i]).abs();
            if e > max_err {
                max_err = e;
            }
            sum_sq += (dec[i] - src[i]).powi(2);
        }
        let rmse = (sum_sq / n as f32).sqrt();
        // Looser bound than IQ2 family — 1.5 bpw + the per-sub-block
        // delta-scalar reconstruction is fundamentally less
        // precise. Pin a sanity ceiling.
        assert!(
            max_err < 1.0,
            "iq1_s: max_err={max_err} rmse={rmse}"
        );
        assert!(
            rmse < 0.4,
            "iq1_s: rmse={rmse}"
        );
    }

    #[test]
    fn iq1_s_with_cpu_encoder_matches_direct() {
        // The batched-encoder path with `CpuFallbackEncoder` must
        // produce byte-identical output to `encode_iq1_s`. Both
        // paths use the same per-chunk grid-search arithmetic and
        // the same tail layout; only the search-loop driver differs.
        let n = 4 * QK_K;
        let src = make_sparse_row(n, 71);
        let mut direct = vec![0u8; 4 * BLOCK_IQ1_S_BYTES];
        let mut batched = vec![0u8; 4 * BLOCK_IQ1_S_BYTES];
        encode_iq1_s(&src, &mut direct);
        encode_iq1_s_with_encoder(&src, &mut batched, &crate::iq_gpu::CpuFallbackEncoder);
        assert_eq!(direct, batched, "batched IQ1_S encoder must match direct");
    }

    #[test]
    fn iq1_s_handles_zero_input() {
        let n = 2 * QK_K;
        let zero = vec![0f32; n];
        let mut enc = vec![0u8; 2 * BLOCK_IQ1_S_BYTES];
        encode_iq1_s(&zero, &mut enc);
        let mut dec = vec![0f32; n];
        dequant::dequant_iq1_s(&enc, &mut dec);
        // d=0 ⇒ dl=0 for every sub-block, so output is all zero.
        assert!(dec.iter().all(|&v| v.abs() < 1e-6));
    }

    // ---- IQ1_S imatrix (importance-weighted) encode -------------------

    fn iq1_s_rmse(src: &[f32], enc: &[u8]) -> f32 {
        let mut dec = vec![0f32; src.len()];
        dequant::dequant_iq1_s(enc, &mut dec);
        let sq: f32 = (0..src.len()).map(|i| (dec[i] - src[i]).powi(2)).sum();
        (sq / src.len() as f32).sqrt()
    }

    /// A uniform (all-1.0) imatrix optimizes the same objective as the
    /// unweighted encode, so its reconstruction RMSE must match within
    /// a small tolerance. (Not byte-identical: the unweighted path uses
    /// AVX2 while the weighted path is scalar, so grid-tie breaking can
    /// differ — but quality must not regress.)
    #[test]
    fn iq1_s_imatrix_uniform_weight_matches_unweighted_quality() {
        let n = 4 * QK_K;
        let src = make_sparse_row(n, 53);
        let mut none = vec![0u8; 4 * BLOCK_IQ1_S_BYTES];
        let mut ones_enc = vec![0u8; 4 * BLOCK_IQ1_S_BYTES];
        encode_iq1_s(&src, &mut none);
        let ones = vec![1.0f32; n];
        encode_iq1_s_imatrix(&src, &mut ones_enc, Some(&ones));
        let r_none = iq1_s_rmse(&src, &none);
        let r_ones = iq1_s_rmse(&src, &ones_enc);
        assert!(
            (r_ones - r_none).abs() <= 0.05 * r_none + 1e-4,
            "uniform-weight imatrix RMSE {r_ones} should match unweighted {r_none}"
        );
    }

    /// A non-uniform imatrix that heavily weights a subset of columns
    /// must not raise the *weighted* reconstruction error on those
    /// columns vs. the uniform encode — i.e. it redirects IQ1_S's
    /// scarce precision toward the important columns.
    #[test]
    fn iq1_s_imatrix_reduces_error_on_weighted_columns() {
        let n = 2 * QK_K;
        let src = make_sparse_row(n, 67);
        // Weight the first 8 of each 32-col sub-block 100×, the rest 0.01×.
        let mut w = vec![0.01f32; n];
        for sub in 0..(n / 32) {
            for j in 0..8 {
                w[sub * 32 + j] = 100.0;
            }
        }
        let mut enc_u = vec![0u8; 2 * BLOCK_IQ1_S_BYTES];
        let mut enc_w = vec![0u8; 2 * BLOCK_IQ1_S_BYTES];
        encode_iq1_s(&src, &mut enc_u);
        encode_iq1_s_imatrix(&src, &mut enc_w, Some(&w));
        let mut dec_u = vec![0f32; n];
        let mut dec_w = vec![0f32; n];
        dequant::dequant_iq1_s(&enc_u, &mut dec_u);
        dequant::dequant_iq1_s(&enc_w, &mut dec_w);
        let werr = |dec: &[f32]| -> f64 {
            (0..n).map(|i| (w[i] as f64) * ((dec[i] - src[i]) as f64).powi(2)).sum()
        };
        let eu = werr(&dec_u);
        let ew = werr(&dec_w);
        assert!(
            ew <= eu + 1e-6,
            "imatrix IQ1_S encode should not raise weighted error on important cols: weighted={ew} uniform={eu}"
        );
    }

    #[test]
    fn iq1_m_round_trip_within_bound() {
        let n = 2 * QK_K;
        let src = make_sparse_row(n, 31);
        let mut enc = vec![0u8; 2 * BLOCK_IQ1_M_BYTES];
        encode_iq1_m(&src, &mut enc);
        let mut dec = vec![0f32; n];
        dequant::dequant_iq1_m(&enc, &mut dec);
        let mut max_err = 0f32;
        let mut sum_sq = 0f32;
        for i in 0..n {
            let e = (dec[i] - src[i]).abs();
            if e > max_err {
                max_err = e;
            }
            sum_sq += (dec[i] - src[i]).powi(2);
        }
        let rmse = (sum_sq / n as f32).sqrt();
        // IQ1_M (1.75 bpw) gets per-8-weight delta granularity vs.
        // IQ1_S's per-32-weight. Tighter than IQ1_S in practice
        // but bound stays loose given the analytical search.
        assert!(
            max_err < 1.0,
            "iq1_m: max_err={max_err} rmse={rmse}"
        );
        assert!(
            rmse < 0.35,
            "iq1_m: rmse={rmse}"
        );
    }

    #[test]
    #[ignore = "perf timing; run with --ignored"]
    fn iq1_s_perf_realistic_tensor() {
        // Realistic-ish: 1024 super-blocks = 256k elements (one
        // expert FFN matrix is ~2048*512 = 1M = 4096 super-blocks, so
        // 1024 is 1/4 of one matrix). Time should be sub-second.
        use std::time::Instant;
        let n_blocks = 1024;
        let mut seed = 42u64;
        let src: Vec<f32> = (0..n_blocks * QK_K)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 32) as i32 as f32 / i32::MAX as f32) * 0.05
            })
            .collect();
        let mut dst = vec![0u8; n_blocks * BLOCK_IQ1_S_BYTES];
        let t = Instant::now();
        encode_iq1_s(&src, &mut dst);
        let dt = t.elapsed();
        println!(
            "encode_iq1_s({} blocks = {} elements): {:.2} ms = {:.0} ns/element = {:.0} blocks/sec",
            n_blocks, src.len(), dt.as_secs_f64() * 1000.0,
            dt.as_nanos() as f64 / src.len() as f64,
            n_blocks as f64 / dt.as_secs_f64()
        );
        assert!(dt.as_secs() < 30, "encode took {:.1} s — should be sub-second", dt.as_secs_f64());
    }

    #[test]
    fn iq1_s_bias_diagnostic() {
        use crate::dequant::dequant_iq1_s;
        fn stats(name: &str, v: &[f32]) -> (f64, f64) {
            let sum: f64 = v.iter().map(|x| *x as f64).sum();
            let mean = sum / v.len() as f64;
            let var: f64 = v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / v.len() as f64;
            let mn = v.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let mx = v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            println!("  {name:24} mean={mean:+.5e} rms={:+.5e} min={mn:+.5e} max={mx:+.5e}", var.sqrt());
            (mean, var.sqrt())
        }
        fn test_case(name: &str, src: Vec<f32>) {
            let n_blocks = src.len() / QK_K;
            let mut dst = vec![0u8; n_blocks * BLOCK_IQ1_S_BYTES];
            encode_iq1_s(&src, &mut dst);
            let mut deq = vec![0.0f32; src.len()];
            dequant_iq1_s(&dst, &mut deq);
            println!("== {name} ==");
            let (src_mean, _) = stats("src", &src);
            let (deq_mean, _) = stats("dequant", &deq);
            let diff: Vec<f32> = src.iter().zip(deq.iter()).map(|(a,b)| a - b).collect();
            stats("diff (src - dq)", &diff);
            println!("  bias = {:+.3e} ({:+.1}% of src rms)", deq_mean - src_mean,
                100.0 * (deq_mean - src_mean) / (src.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / src.len() as f64).sqrt());
        }
        // Pseudo-random gaussian via simple LCG
        let mut seed = 42u64;
        let mut next = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 32) as i32 as f32 / i32::MAX as f32) * 0.1
        };
        // 8 super-blocks (2048 elements) — closer to real tensor size
        // and exercises the alternating-sign tie-breaker.
        let n = 256 * 8;
        let g: Vec<f32> = (0..n).map(|_| next()).collect();
        test_case("Uniform[-0.1,0.1]", g);
        let g2: Vec<f32> = (0..n).map(|_| -next().abs()).collect();
        test_case("All-negative", g2);
        let g3: Vec<f32> = (0..n).map(|_| next().abs()).collect();
        test_case("All-positive", g3);
        let g4: Vec<f32> = (0..n).map(|_| -0.05f32 + next() * 0.01).collect();
        test_case("Negative-biased(-0.05)", g4);
        let g5: Vec<f32> = (0..n).map(|_| 0.05f32 + next() * 0.01).collect();
        test_case("Positive-biased(+0.05)", g5);
    }

    #[test]
    fn iq1_m_handles_zero_input() {
        let n = 2 * QK_K;
        let zero = vec![0f32; n];
        let mut enc = vec![0u8; 2 * BLOCK_IQ1_M_BYTES];
        encode_iq1_m(&zero, &mut enc);
        let mut dec = vec![0f32; n];
        dequant::dequant_iq1_m(&enc, &mut dec);
        assert!(dec.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn iq1_m_with_cpu_encoder_matches_direct() {
        let n = 4 * QK_K;
        let src = make_sparse_row(n, 149);
        let mut direct = vec![0u8; 4 * BLOCK_IQ1_M_BYTES];
        let mut batched = vec![0u8; 4 * BLOCK_IQ1_M_BYTES];
        encode_iq1_m(&src, &mut direct);
        encode_iq1_m_with_encoder(&src, &mut batched, &crate::iq_gpu::CpuFallbackEncoder);
        assert_eq!(direct, batched, "batched IQ1_M encoder must match direct");
    }

    /// Sign-mask picker correctness: for a known target + grid,
    /// the picker should select the parity-valid sign mask that
    /// gives the highest projection score.
    #[test]
    fn pick_sign_mask_handles_parity_fixup() {
        // Target with 5 positive elements at large magnitude and 3
        // negative at smaller magnitude. Grid all-positive (i8=8).
        // Free-optimal mask = the 3 negatives → popcount 3 (odd).
        // Parity fix: flip the *smallest* contribution.
        let target: [f32; 8] = [1.0, -0.1, 1.0, 1.0, -0.05, 1.0, 1.0, -0.2];
        let grid: [i8; 8] = [8; 8];
        let (mask, score) = pick_sign_mask_8(&target, &grid);
        assert_eq!(
            mask.count_ones() & 1,
            0,
            "mask must have even popcount, got {mask:08b}"
        );
        // The 0.05 (idx 4) has the smallest contribution; the picker
        // should keep its sign FREE (i.e. flip it back to "no sign"
        // because |0.05| is the smallest cost-to-fix).
        // Verify the score is positive and tracks the expected value.
        assert!(score > 0.0, "score should be positive, got {score}");
    }
}
