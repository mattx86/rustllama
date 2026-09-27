//! IQ-quant encoders (v1: IQ4_NL + IQ4_XS only).
//!
//! These two share the 16-entry `KVALUES_IQ4NL` codebook (already
//! exported by [`crate::dequant`]). Each weight encodes as a 4-bit
//! index into the codebook — no per-vector grid search, no sign
//! tables, no `qh` high bits to manage. The asymmetric codebook
//! (biased toward zero) gives them the same bpw as Q4_0/Q4_K
//! while preserving more dynamic range for the small-magnitude
//! weights that dominate model parameter distributions.
//!
//! ## Out of scope (Q-D-extra)
//!
//! IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S all use
//! **vector-grid** codebooks where each codebook entry is an
//! 8-weight (or 4-weight) vector and a per-element sign table.
//! Encoding requires per-8-weight-sub-block exhaustive (or
//! heuristic-pruned) search over 256–2048 grid entries × 128 sign
//! tables — a fundamentally different algorithm. Tracked
//! separately; this module ships only the 4-bit per-weight
//! IQ-quants for v1.

use half::f16;

use crate::dequant::KVALUES_IQ4NL;

const QK_K: usize = 256;
const N_SUB_BLOCKS_IQ4XS: usize = 8; // 256 / 32

/// Nearest-neighbor codebook lookup: given a scalar `x` already
/// normalized to the codebook's value scale (i.e. `x = original /
/// sub_scale`), return the index in `0..16` that minimizes
/// `|x - KVALUES_IQ4NL[idx]|`. Ties break toward the lower index
/// (matches the natural argmin loop).
#[inline]
fn nearest_iq4nl(x: f32) -> usize {
    let mut best = 0usize;
    let mut best_err = (x - KVALUES_IQ4NL[0] as f32).abs();
    for (i, &c) in KVALUES_IQ4NL.iter().enumerate().skip(1) {
        let e = (x - c as f32).abs();
        if e < best_err {
            best_err = e;
            best = i;
        }
    }
    best
}

// ----------------------------------------------------------------------
// IQ4_NL — `{ d: f16, qs: [u8; 16] }` = 18 bytes/block, 4.5 bpw
// ----------------------------------------------------------------------

const BLOCK_IQ4_NL_BYTES: usize = 18;
const QK_NL: usize = 32;

/// Encode IQ4_NL. 32-weight blocks; one f16 scale + 16 bytes of 4-bit
/// indices into `KVALUES_IQ4NL`. The scale is `d = signed_amax /
/// MIN_CODEBOOK = max_abs_val / -127` (matches the
/// `KVALUES_IQ4NL[0] = -127` extreme), so `q[i] = round(x[i] / d)`
/// after rescaling lands roughly within the codebook range and the
/// nearest-neighbor lookup recovers `x[i]` to within half a step.
pub fn encode_iq4_nl(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK_NL, 0, "encode_iq4_nl: src.len() must be multiple of 32");
    let n_blocks = src.len() / QK_NL;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_IQ4_NL_BYTES,
        "encode_iq4_nl: dst.len() must be n_blocks * 18"
    );
    for b in 0..n_blocks {
        let xs = &src[b * QK_NL..(b + 1) * QK_NL];
        let off = b * BLOCK_IQ4_NL_BYTES;

        // Find the signed-absmax sample so `d` carries the sign of
        // the element that has the largest |x|. Reference: same
        // pattern as Q4_0 — `d = max_val / KVALUES_IQ4NL[0]` (where
        // KVALUES[0] = -127).
        let mut amax = 0f32;
        let mut max_val = 0f32;
        for &x in xs {
            let a = x.abs();
            if a > amax {
                amax = a;
                max_val = x;
            }
        }
        let d = if amax == 0.0 {
            0.0
        } else {
            max_val / (KVALUES_IQ4NL[0] as f32)
        };
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let d_bits = f16::from_f32(d).to_bits();
        dst[off] = (d_bits & 0xFF) as u8;
        dst[off + 1] = ((d_bits >> 8) & 0xFF) as u8;

        // qs[j]: low nibble = index for out[j], high nibble = index for out[j+16].
        for j in 0..16 {
            let lo = nearest_iq4nl(xs[j] * id) as u8;
            let hi = nearest_iq4nl(xs[j + 16] * id) as u8;
            dst[off + 2 + j] = lo | (hi << 4);
        }
    }
}

// ----------------------------------------------------------------------
// IQ4_XS — 136 bytes/super-block (256 weights), 4.25 bpw
//   { d: f16, scales_h: u16, scales_l: [u8; 4], qs: [u8; 128] }
// ----------------------------------------------------------------------

const BLOCK_IQ4_XS_BYTES: usize = 136;

/// Encode IQ4_XS. Super-block of 256 weights = 8 sub-blocks × 32
/// weights. Per sub-block: a 6-bit signed scale stored as
/// `(lo_nibble | (hi_2bits << 4))` ∈ `[0, 63]` (then bias-shifted
/// `-32` on decode); plus 16 bytes of 4-bit codebook indices.
pub fn encode_iq4_xs(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK_K, 0, "encode_iq4_xs: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_IQ4_XS_BYTES,
        "encode_iq4_xs: dst.len() must be n_blocks * 136"
    );
    for b in 0..n_blocks {
        let xs = &src[b * QK_K..(b + 1) * QK_K];
        let off = b * BLOCK_IQ4_XS_BYTES;

        // Stage 1: per-sub-block ideal signed scale.
        let mut sub_scales = [0f32; N_SUB_BLOCKS_IQ4XS];
        for k in 0..N_SUB_BLOCKS_IQ4XS {
            let sb = &xs[k * 32..(k + 1) * 32];
            let mut amax = 0f32;
            let mut max_val = 0f32;
            for &x in sb {
                let a = x.abs();
                if a > amax {
                    amax = a;
                    max_val = x;
                }
            }
            // The ideal scale puts `max_val` exactly at codebook
            // extreme; we use `KVALUES[0] = -127` (signed) so the
            // sub-scale absorbs the sign of `max_val`.
            sub_scales[k] = if amax == 0.0 {
                0.0
            } else {
                max_val / (KVALUES_IQ4NL[0] as f32)
            };
        }

        // Stage 2: super-block d so that quantized sub-scales fit
        // in signed 6-bit [-32, 31]. Use the same iscale = -32/max_signed
        // trick as Q6_K so the sign of the largest sub-scale lands
        // negative-most-quantized.
        let signed_max = sub_scales
            .iter()
            .fold(0f32, |acc, &s| if s.abs() > acc.abs() { s } else { acc });
        let iscale = if signed_max != 0.0 { -32.0 / signed_max } else { 0.0 };
        let d_super = if iscale != 0.0 { 1.0 / iscale } else { 0.0 };

        let mut scales_q = [0i8; N_SUB_BLOCKS_IQ4XS];
        for k in 0..N_SUB_BLOCKS_IQ4XS {
            let l = (iscale * sub_scales[k]).round().clamp(-32.0, 31.0) as i32;
            scales_q[k] = l as i8;
        }

        // Stage 3: write d.
        let d_bits = f16::from_f32(d_super).to_bits();
        dst[off] = (d_bits & 0xFF) as u8;
        dst[off + 1] = ((d_bits >> 8) & 0xFF) as u8;

        // Stage 4: pack 8 signed 6-bit sub-scales into scales_h (16 bits
        // total = 2 bits × 8) + scales_l (4 bytes = 4 bits × 8). Per the
        // dequant: for sub-block ib, `lo_nibble = scales_l[ib/2] & 0xF`
        // when ib%2==0, else `>> 4`; `hi_bits = (scales_h >> (2*ib)) &
        // 0x3`. Stored unsigned = signed + 32 (the dequant subtracts
        // 32 after combining).
        let mut scales_l = [0u8; 4];
        let mut scales_h: u16 = 0;
        for ib in 0..N_SUB_BLOCKS_IQ4XS {
            let stored = (scales_q[ib] as i32 + 32) as u32; // [0, 63]
            let lo_nibble = (stored & 0x0F) as u8;
            let hi_bits = ((stored >> 4) & 0x03) as u8;
            if ib % 2 == 0 {
                scales_l[ib / 2] |= lo_nibble; // low nibble of byte
            } else {
                scales_l[ib / 2] |= lo_nibble << 4; // high nibble of byte
            }
            scales_h |= (hi_bits as u16) << (2 * ib);
        }
        dst[off + 2] = (scales_h & 0xFF) as u8;
        dst[off + 3] = ((scales_h >> 8) & 0xFF) as u8;
        dst[off + 4..off + 8].copy_from_slice(&scales_l);

        // Stage 5: per-sub-block codebook lookup.
        let qs = &mut dst[off + 8..off + 8 + 128];
        for ib in 0..N_SUB_BLOCKS_IQ4XS {
            let s_q = scales_q[ib] as f32;
            let dl = d_super * s_q;
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            let sb = &xs[ib * 32..(ib + 1) * 32];
            let q_off = ib * 16;
            // 16 q-bytes per sub-block: low nibble for out[j],
            // high nibble for out[j+16].
            for j in 0..16 {
                let lo = nearest_iq4nl(sb[j] * idl) as u8;
                let hi = nearest_iq4nl(sb[j + 16] * idl) as u8;
                qs[q_off + j] = lo | (hi << 4);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant;

    fn make_row(n: usize, amp: f32, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / (1u32 << 24) as f32;
                (u * 2.0 - 1.0) * amp
            })
            .collect()
    }

    /// IQ4_NL: the 16-entry codebook spans `[-127, 113]` (signed,
    /// asymmetric). Max quantization step is the largest gap between
    /// consecutive codebook entries: 23 (between 89 and 113) → as a
    /// fraction of `d * 127`, that's ~0.18. Loose round-trip bound
    /// `0.25 * amp`.
    #[test]
    fn iq4_nl_round_trip_within_bound() {
        let amp = 4.0;
        for &n_blocks in &[1usize, 3, 8] {
            let n = n_blocks * QK_NL;
            let src = make_row(n, amp, n as u32 * 31);
            let mut enc = vec![0u8; n_blocks * BLOCK_IQ4_NL_BYTES];
            encode_iq4_nl(&src, &mut enc);
            let mut dec = vec![0f32; n];
            dequant::dequant_iq4_nl(&enc, &mut dec);
            let mut max_err = 0f32;
            for i in 0..n {
                let e = (dec[i] - src[i]).abs();
                if e > max_err {
                    max_err = e;
                }
            }
            assert!(
                max_err < 0.25 * amp,
                "iq4_nl n_blocks={n_blocks}: max_err={max_err}"
            );
        }
    }

    #[test]
    fn iq4_xs_round_trip_within_bound() {
        // Same codebook gap as IQ4_NL but the 6-bit sub-scale adds a
        // multiplicative rounding (~3% per sub-scale step). Bound
        // ~0.3 * amp.
        let amp = 4.0;
        for &n_blocks in &[1usize, 2, 4] {
            let n = n_blocks * QK_K;
            let src = make_row(n, amp, n as u32 * 31);
            let mut enc = vec![0u8; n_blocks * BLOCK_IQ4_XS_BYTES];
            encode_iq4_xs(&src, &mut enc);
            let mut dec = vec![0f32; n];
            dequant::dequant_iq4_xs(&enc, &mut dec);
            let mut max_err = 0f32;
            for i in 0..n {
                let e = (dec[i] - src[i]).abs();
                if e > max_err {
                    max_err = e;
                }
            }
            assert!(
                max_err < 0.3 * amp,
                "iq4_xs n_blocks={n_blocks}: max_err={max_err}"
            );
        }
    }

    #[test]
    fn iq4_formats_handle_zero_input() {
        let n_nl = 2 * QK_NL;
        let zero_nl = vec![0f32; n_nl];
        let mut q_nl = vec![0u8; 2 * BLOCK_IQ4_NL_BYTES];
        encode_iq4_nl(&zero_nl, &mut q_nl);
        let mut out_nl = vec![0f32; n_nl];
        dequant::dequant_iq4_nl(&q_nl, &mut out_nl);
        // Codebook[8] = 1 (not zero), but with d=0 the dequant
        // multiplies by zero so output is zero.
        assert!(out_nl.iter().all(|&v| v == 0.0));

        let n_xs = 2 * QK_K;
        let zero_xs = vec![0f32; n_xs];
        let mut q_xs = vec![0u8; 2 * BLOCK_IQ4_XS_BYTES];
        encode_iq4_xs(&zero_xs, &mut q_xs);
        let mut out_xs = vec![0f32; n_xs];
        dequant::dequant_iq4_xs(&q_xs, &mut out_xs);
        assert!(out_xs.iter().all(|&v| v == 0.0));
    }

    /// Codebook nearest-neighbor is monotonic in `x`. Pin that the
    /// helper picks the right bucket at codebook boundaries.
    #[test]
    fn nearest_iq4nl_boundary_behavior() {
        // Codebook value 8 = 1; codebook value 7 = -10. Midpoint
        // = -4.5. Inputs above -4.5 should pick 8 (=1), below -4.5
        // should pick 7 (=-10).
        assert_eq!(nearest_iq4nl(-127.0), 0);
        assert_eq!(nearest_iq4nl(113.0), 15);
        assert_eq!(nearest_iq4nl(0.0), 8); // codebook[8] = 1 is closest to 0
        assert_eq!(nearest_iq4nl(-5.0), 7); // -5 closer to -10 than to 1
        assert_eq!(nearest_iq4nl(-4.0), 8); // -4 closer to 1 than to -10
        assert_eq!(nearest_iq4nl(1000.0), 15); // saturate to max
        assert_eq!(nearest_iq4nl(-1000.0), 0); // saturate to min
    }
}
