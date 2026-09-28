//! Ternary quant encoders (TQ1_0, TQ2_0).
//!
//! Per-weight values: `{-d, 0, +d}`. The encoder picks a per-block
//! `d = max(|x|)` so each input snaps to its nearest trit; bit
//! packing differs between the two formats:
//!
//! - **TQ2_0** (2 bpw, 66 B/block) — 2-bit fields, 4 trits per byte
//!   across shift positions `0/2/4/6`.
//! - **TQ1_0** (1.6875 bpw, 54 B/block) — base-3 packing (5 trits
//!   per byte for the bulk; 4 trits per byte in the tail).
//!
//! Both round-trip exactly against [`crate::dequant`] because the
//! trit choice is unique once `d` is fixed (no quantization step
//! ambiguity).

use half::f16;

const QK_K: usize = 256;

/// Compute a per-block `d = max(|x|)` and map each input to a trit
/// in `{0, 1, 2}` (stored as `q + 1` where `q ∈ {-1, 0, +1}`).
fn build_trits(xs: &[f32; QK_K]) -> (f32, [u8; QK_K]) {
    let mut amax = 0f32;
    for &x in xs {
        let a = x.abs();
        if a > amax {
            amax = a;
        }
    }
    let mut trits = [1u8; QK_K]; // default to "0" trit ⇒ 1 in stored form
    if amax == 0.0 {
        return (0.0, trits);
    }
    let d = amax;
    let inv = 1.0 / d;
    for i in 0..QK_K {
        let r = (xs[i] * inv).round();
        // Clamp to {-1, 0, +1}; stored form = r + 1 ∈ {0, 1, 2}.
        let q = r.clamp(-1.0, 1.0) as i32;
        trits[i] = (q + 1) as u8;
    }
    (d, trits)
}

// ----------------------------------------------------------------------
// TQ2_0 — `{ qs: [u8; 64], d: f16 }` = 66 bytes/block, 2 bpw
// ----------------------------------------------------------------------

const BLOCK_TQ2_0_BYTES: usize = 66;

/// Encode TQ2_0. Each 256-weight super-block splits into two halves
/// of 128 weights; each half packs 4 shift-positions × 32 trits =
/// 128 weights into 32 bytes. Per-trit storage: 2 bits at the
/// shift position; reconstruction subtracts 1 to recover
/// `{-1, 0, +1}`.
pub fn encode_tq2_0(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK_K, 0, "encode_tq2_0: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_TQ2_0_BYTES,
        "encode_tq2_0: dst.len() must be n_blocks * 66"
    );
    for b in 0..n_blocks {
        let xs: &[f32; QK_K] = src[b * QK_K..(b + 1) * QK_K].try_into().unwrap();
        let off = b * BLOCK_TQ2_0_BYTES;
        let (d, trits) = build_trits(xs);

        // qs[0..64]: two halves of 32 bytes; within each half, 4
        // shift positions of 32 trits each. Inverse of the dequant's
        // walk: weight at output index `o` lives at qs[half_off + m]
        // bit `shift = (o_in_half / 32) * 2`, position `m = o_in_half % 32`.
        let qs = &mut dst[off..off + 64];
        for byte in qs.iter_mut() {
            *byte = 0;
        }
        for half in 0..2 {
            let half_off = half * 32;
            for l in 0..4 {
                let shift = (l as u32) * 2;
                for m in 0..32 {
                    let o = half * 128 + l * 32 + m;
                    qs[half_off + m] |= (trits[o] & 0x03) << shift;
                }
            }
        }
        let d_bits = f16::from_f32(d).to_bits();
        dst[off + 64] = (d_bits & 0xFF) as u8;
        dst[off + 65] = ((d_bits >> 8) & 0xFF) as u8;
    }
}

// ----------------------------------------------------------------------
// TQ1_0 — `{ qs: [u8; 48], qh: [u8; 4], d: f16 }` = 54 bytes/block, 1.6875 bpw
// ----------------------------------------------------------------------

const BLOCK_TQ1_0_BYTES: usize = 54;

/// Encode TQ1_0 in ggml's **fixed-point** wire layout (see
/// [`crate::dequant::dequant_tq1_0`] for the decode side and the format
/// rationale). Per byte we build the plain base-3 value
/// `v = t0*3^4 + t1*3^3 + … + t4` (most-significant trit first) and
/// then store `ceil(v * 256 / 243)` so the decoder's wrapping-multiply
/// trick recovers each trit. The 4-trit `qh` tail is shifted up by one
/// trit (`v *= 3`) before scaling so its digits land in the top 4
/// positions, matching `quantize_row_tq1_0_ref`.
pub fn encode_tq1_0(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK_K, 0, "encode_tq1_0: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_TQ1_0_BYTES,
        "encode_tq1_0: dst.len() must be n_blocks * 54"
    );
    // ceil(v * 256 / 243) — biases toward the base-3 digit boundaries
    // so `((byte * 3^n) * 3) >> 8` reads back the exact trit.
    let ceil_scale = |v: u32| ((v * 256 + 242) / 243) as u8;
    for b in 0..n_blocks {
        let xs: &[f32; QK_K] = src[b * QK_K..(b + 1) * QK_K].try_into().unwrap();
        let off = b * BLOCK_TQ1_0_BYTES;
        let (d, trits) = build_trits(xs);

        let mut qs = [0u8; 48];
        let mut qh = [0u8; 4];

        // Chunk 0: qs[0..32], 5 trits per byte → 160 weights
        // (output positions 0..160). Output element `n*32 + m` is the
        // digit with coefficient `3^(4-n)` of byte qs[m] (n=0 = MSB).
        for m in 0..32 {
            let mut v: u32 = 0;
            for n in 0..5 {
                v = v * 3 + trits[n * 32 + m] as u32;
            }
            qs[m] = ceil_scale(v);
        }
        // Chunk 1: qs[32..48], 5 trits per byte → 80 weights
        // (positions 160..240). Output element `160 + n*16 + m`.
        for m in 0..16 {
            let mut v: u32 = 0;
            for n in 0..5 {
                v = v * 3 + trits[160 + n * 16 + m] as u32;
            }
            qs[32 + m] = ceil_scale(v);
        }
        // Tail: qh[0..4], 4 trits per byte → 16 weights
        // (positions 240..256). Output element `240 + n*4 + h`. The
        // extra `* 3` shifts the 4 trits into the top 4 digit slots.
        for h in 0..4 {
            let mut v: u32 = 0;
            for n in 0..4 {
                v = v * 3 + trits[240 + n * 4 + h] as u32;
            }
            v *= 3;
            qh[h] = ceil_scale(v);
        }

        dst[off..off + 48].copy_from_slice(&qs);
        dst[off + 48..off + 52].copy_from_slice(&qh);
        let d_bits = f16::from_f32(d).to_bits();
        dst[off + 52] = (d_bits & 0xFF) as u8;
        dst[off + 53] = ((d_bits >> 8) & 0xFF) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant;

    /// Deterministic ternary-friendly row: produces values clustered
    /// near `{-amp, 0, +amp}` with small perturbations so the encoder
    /// has unambiguous trit choices.
    fn make_ternary_row(n: usize, amp: f32, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let pick = s % 3;
                let jitter = ((s >> 16) as f32 / (1u32 << 16) as f32 - 0.5) * 0.01 * amp;
                let base = match pick {
                    0 => -amp,
                    1 => 0.0,
                    _ => amp,
                };
                base + jitter
            })
            .collect()
    }

    #[test]
    fn tq2_0_round_trip_recovers_ternary_within_step() {
        // The trit picker snaps each input to its nearest of {-d, 0, +d}.
        // For inputs jittered ±0.01 around those values, the recovered
        // dequant must land within `1.5 * amp / 127` of the original
        // (f16 d rounding + the jitter itself).
        for &n_blocks in &[1usize, 2, 3] {
            let n = n_blocks * QK_K;
            let src = make_ternary_row(n, 4.0, n as u32 * 7);
            let mut enc = vec![0u8; n_blocks * BLOCK_TQ2_0_BYTES];
            encode_tq2_0(&src, &mut enc);
            let mut dec = vec![0f32; n];
            dequant::dequant_tq2_0(&enc, &mut dec);
            let mut max_err = 0f32;
            for i in 0..n {
                let e = (dec[i] - src[i]).abs();
                if e > max_err {
                    max_err = e;
                }
            }
            // Jitter is ≤ 0.04 (0.01 * amp); f16 d precision adds another
            // ~0.002 * amp. Loose bound: 0.05.
            assert!(
                max_err < 0.05,
                "tq2_0 n_blocks={n_blocks}: max_err={max_err}"
            );
        }
    }

    #[test]
    fn tq1_0_round_trip_recovers_ternary_within_step() {
        for &n_blocks in &[1usize, 2, 3] {
            let n = n_blocks * QK_K;
            let src = make_ternary_row(n, 4.0, n as u32 * 11);
            let mut enc = vec![0u8; n_blocks * BLOCK_TQ1_0_BYTES];
            encode_tq1_0(&src, &mut enc);
            let mut dec = vec![0f32; n];
            dequant::dequant_tq1_0(&enc, &mut dec);
            let mut max_err = 0f32;
            for i in 0..n {
                let e = (dec[i] - src[i]).abs();
                if e > max_err {
                    max_err = e;
                }
            }
            assert!(
                max_err < 0.05,
                "tq1_0 n_blocks={n_blocks}: max_err={max_err}"
            );
        }
    }

    /// Pins the TQ1_0 ggml fixed-point wire layout with a hand-computed
    /// reference. `amp = 4.0` is exactly f16-representable so `d == amp`
    /// and the round-trip is bit-exact. The two asserted bytes are
    /// derived directly from `ceil(v * 256 / 243)` where `v` is the
    /// plain base-3 value (MSB = first output element of the byte); the
    /// `qh` value carries the extra `* 3` top-shift.
    ///
    /// NOTE: this validates our encoder ↔ decoder pair against the ggml
    /// *formula*, not against a byte dump from a real llama.cpp TQ1_0
    /// GGUF (none is available in-tree). A live-file diff is still the
    /// gold standard for wire compatibility.
    #[test]
    fn tq1_0_matches_ggml_fixed_point_reference() {
        let amp = 4.0f32;
        let mut src = vec![0f32; QK_K]; // all-zero ⇒ trit "1" (ternary 0)

        // qs[0] packs output elements {0, 32, 64, 96, 128}, MSB = elem 0.
        // Trits [2, 1, 0, 2, 1] ⇒ +amp, 0, -amp, +amp, 0.
        src[0] = amp; // trit 2
        src[64] = -amp; // trit 0
        src[96] = amp; // trit 2
        // src[32], src[128] stay 0.0 ⇒ trit 1.

        // qh[0] packs output elements {240, 244, 248, 252}, MSB = 240.
        // Trits [0, 2, 1, 2] ⇒ -amp, +amp, 0, +amp.
        src[240] = -amp; // trit 0
        src[244] = amp; // trit 2
        src[252] = amp; // trit 2
        // src[248] stays 0.0 ⇒ trit 1.

        let mut enc = vec![0u8; BLOCK_TQ1_0_BYTES];
        encode_tq1_0(&src, &mut enc);

        // qs[0]: v = 2*81 + 1*27 + 0*9 + 2*3 + 1 = 196; ceil(196*256/243) = 207.
        assert_eq!(enc[0], 207, "qs[0] wire byte");
        // qh[0] (byte offset 48): v = (0*27 + 2*9 + 1*3 + 2) * 3 = 69;
        // ceil(69*256/243) = 73.
        assert_eq!(enc[48], 73, "qh[0] wire byte");

        // d little-endian f16 at bytes 52..54.
        let d = f16::from_le_bytes([enc[52], enc[53]]).to_f32();
        assert_eq!(d, amp, "d must equal amax exactly for f16-exact amp");

        // Full round-trip is bit-exact for this input.
        let mut dec = vec![0f32; QK_K];
        dequant::dequant_tq1_0(&enc, &mut dec);
        for i in 0..QK_K {
            assert_eq!(dec[i], src[i], "element {i} round-trip");
        }
    }

    /// Zero input encodes to all "1" trits (the stored form of the
    /// `0` ternary value) and dequants back to all zeros.
    #[test]
    fn ternary_handles_zero_input() {
        let n = 2 * QK_K;
        let zero = vec![0f32; n];

        let mut q2 = vec![0u8; 2 * BLOCK_TQ2_0_BYTES];
        encode_tq2_0(&zero, &mut q2);
        let mut out = vec![0f32; n];
        dequant::dequant_tq2_0(&q2, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));

        let mut q1 = vec![0u8; 2 * BLOCK_TQ1_0_BYTES];
        encode_tq1_0(&zero, &mut q1);
        dequant::dequant_tq1_0(&q1, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    /// Arbitrary smooth (non-ternary) input: the encoder snaps each
    /// weight to its nearest of `{-d, 0, +d}`. We only verify that
    /// the round-trip stays bounded by `d` (max possible step).
    #[test]
    fn ternary_handles_smooth_input_with_bounded_error() {
        let amp = 4.0;
        let n = QK_K;
        let mut s: u32 = 1;
        let src: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / (1u32 << 24) as f32;
                (u * 2.0 - 1.0) * amp
            })
            .collect();

        let mut enc = vec![0u8; BLOCK_TQ2_0_BYTES];
        encode_tq2_0(&src, &mut enc);
        let mut dec = vec![0f32; n];
        dequant::dequant_tq2_0(&enc, &mut dec);
        let mut max_err = 0f32;
        for i in 0..n {
            let e = (dec[i] - src[i]).abs();
            if e > max_err {
                max_err = e;
            }
        }
        // Ternary forces each weight to {-amp, 0, +amp}; max error
        // is amp / 2 (halfway between two trit values).
        assert!(
            max_err <= amp / 2.0 + 0.01,
            "tq2_0 smooth max_err={max_err} > bound={}",
            amp / 2.0 + 0.01
        );
    }
}
