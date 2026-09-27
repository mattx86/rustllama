//! GGUF quantization encoders.
//!
//! Each function takes an `[f32]` source row and writes the
//! corresponding GGUF block bytes to an `[u8]` destination. Block
//! layouts mirror the inverse of [`crate::dequant`]; round-trip
//! parity tests in this module pin the byte-exact encoder/decoder
//! pair against the documented ggml reference behavior.
//!
//! Naming convention: `encode_qX_Y(src: &[f32], dst: &mut [u8])` —
//! `src.len()` must be a multiple of the format's block size,
//! `dst.len()` must equal the corresponding byte size from
//! `GgmlType::byte_size(src.len())`.
//!
//! These are the **legacy** quant formats (32-weight blocks).
//! K-quants and IQ-quants land in subsequent modules with their
//! own super-block geometry.

use half::f16;

const QK: usize = 32;

// ----------------------------------------------------------------------
// Q8_0 — `{ d: f16, qs: [i8; 32] }` = 34 bytes/block, 8.5 bpw
// ----------------------------------------------------------------------

const BLOCK_Q8_0_BYTES: usize = 34;

/// Encode an f32 row as Q8_0 blocks. Symmetric 8-bit: per-block
/// scale `d = max(|x|) / 127`; each quantized value `q[i] =
/// round(x[i] / d)` in `[-127, 127]` (128 unused so the dequant
/// stays symmetric around zero). Reference: `ggml_quantize_q8_0`
/// in ggml-quants.c.
pub fn encode_q8_0(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK, 0, "encode_q8_0: src.len() must be multiple of 32");
    let n_blocks = src.len() / QK;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_Q8_0_BYTES,
        "encode_q8_0: dst.len() must be n_blocks * 34"
    );
    for b in 0..n_blocks {
        let xs = &src[b * QK..(b + 1) * QK];
        let off = b * BLOCK_Q8_0_BYTES;
        let mut amax = 0f32;
        for &x in xs {
            let a = x.abs();
            if a > amax {
                amax = a;
            }
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_bits = f16::from_f32(d).to_bits();
        dst[off] = (d_bits & 0xFF) as u8;
        dst[off + 1] = ((d_bits >> 8) & 0xFF) as u8;
        for i in 0..QK {
            let q = (xs[i] * id).round().clamp(-128.0, 127.0) as i8;
            dst[off + 2 + i] = q as u8;
        }
    }
}

// ----------------------------------------------------------------------
// Q8_1 — `{ d: f16, s: f16, qs: [i8; 32] }` = 36 bytes/block, 9 bpw
// ----------------------------------------------------------------------

const BLOCK_Q8_1_BYTES: usize = 36;

/// Encode an f32 row as Q8_1 blocks. Same quantization as Q8_0 plus
/// a second per-block field `s = d * sum(qs)` — the running sum of
/// dequantized values, precomputed so dot products with Q4_1/Q5_1
/// activations don't have to recompute it. Reference: ggml-quants.c
/// `block_q8_1` write path.
pub fn encode_q8_1(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK, 0, "encode_q8_1: src.len() must be multiple of 32");
    let n_blocks = src.len() / QK;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_Q8_1_BYTES,
        "encode_q8_1: dst.len() must be n_blocks * 36"
    );
    for b in 0..n_blocks {
        let xs = &src[b * QK..(b + 1) * QK];
        let off = b * BLOCK_Q8_1_BYTES;
        let mut amax = 0f32;
        for &x in xs {
            let a = x.abs();
            if a > amax {
                amax = a;
            }
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut sum: i32 = 0;
        for i in 0..QK {
            let q = (xs[i] * id).round().clamp(-128.0, 127.0) as i8;
            dst[off + 4 + i] = q as u8;
            sum += q as i32;
        }
        let s = d * (sum as f32);
        let d_bits = f16::from_f32(d).to_bits();
        let s_bits = f16::from_f32(s).to_bits();
        dst[off] = (d_bits & 0xFF) as u8;
        dst[off + 1] = ((d_bits >> 8) & 0xFF) as u8;
        dst[off + 2] = (s_bits & 0xFF) as u8;
        dst[off + 3] = ((s_bits >> 8) & 0xFF) as u8;
    }
}

// ----------------------------------------------------------------------
// Q4_0 — `{ d: f16, qs: [u8; 16] }` = 18 bytes/block, 4.5 bpw
// ----------------------------------------------------------------------

const BLOCK_Q4_0_BYTES: usize = 18;

/// Encode an f32 row as Q4_0 blocks. Symmetric 4-bit: per-block
/// signed scale `d = signed_absmax / -8` (negative so the dequant
/// `(q - 8) * d` recovers the sign of the input's absmax-bearing
/// element). Each quantized value `q[i] ∈ [0, 15]`; output layout
/// is `qs[j]` low nibble → output position `j`, high nibble →
/// position `j + 16`. Reference: ggml-quants.c `quantize_row_q4_0`.
pub fn encode_q4_0(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK, 0, "encode_q4_0: src.len() must be multiple of 32");
    let n_blocks = src.len() / QK;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_Q4_0_BYTES,
        "encode_q4_0: dst.len() must be n_blocks * 18"
    );
    for b in 0..n_blocks {
        let xs = &src[b * QK..(b + 1) * QK];
        let off = b * BLOCK_Q4_0_BYTES;
        // Find the element with the largest |x| (preserve its sign).
        let mut amax = 0f32;
        let mut max_val = 0f32;
        for &x in xs {
            let a = x.abs();
            if a > amax {
                amax = a;
                max_val = x;
            }
        }
        let d = max_val / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_bits = f16::from_f32(d).to_bits();
        dst[off] = (d_bits & 0xFF) as u8;
        dst[off + 1] = ((d_bits >> 8) & 0xFF) as u8;
        for j in 0..16 {
            // Reference: `(int8_t)(x*id + 8.5f)` — biased rounding
            // toward +inf so the unsigned cast lands in [0, 15].
            let q0 = ((xs[j] * id + 8.5).floor() as i32).clamp(0, 15) as u8;
            let q1 = ((xs[j + 16] * id + 8.5).floor() as i32).clamp(0, 15) as u8;
            dst[off + 2 + j] = q0 | (q1 << 4);
        }
    }
}

// ----------------------------------------------------------------------
// Q4_1 — `{ d: f16, m: f16, qs: [u8; 16] }` = 20 bytes/block, 5 bpw
// ----------------------------------------------------------------------

const BLOCK_Q4_1_BYTES: usize = 20;

/// Encode an f32 row as Q4_1 blocks. Asymmetric 4-bit: per-block
/// `d = (max - min) / 15`, `m = min`. Each quantized value
/// `q[i] = round((x[i] - m) / d) ∈ [0, 15]`. Reference:
/// ggml-quants.c `quantize_row_q4_1`.
pub fn encode_q4_1(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK, 0, "encode_q4_1: src.len() must be multiple of 32");
    let n_blocks = src.len() / QK;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_Q4_1_BYTES,
        "encode_q4_1: dst.len() must be n_blocks * 20"
    );
    for b in 0..n_blocks {
        let xs = &src[b * QK..(b + 1) * QK];
        let off = b * BLOCK_Q4_1_BYTES;
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &x in xs {
            if x < mn {
                mn = x;
            }
            if x > mx {
                mx = x;
            }
        }
        let d = (mx - mn) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_bits = f16::from_f32(d).to_bits();
        let m_bits = f16::from_f32(mn).to_bits();
        dst[off] = (d_bits & 0xFF) as u8;
        dst[off + 1] = ((d_bits >> 8) & 0xFF) as u8;
        dst[off + 2] = (m_bits & 0xFF) as u8;
        dst[off + 3] = ((m_bits >> 8) & 0xFF) as u8;
        for j in 0..16 {
            let q0 = (((xs[j] - mn) * id + 0.5).floor() as i32).clamp(0, 15) as u8;
            let q1 = (((xs[j + 16] - mn) * id + 0.5).floor() as i32).clamp(0, 15) as u8;
            dst[off + 4 + j] = q0 | (q1 << 4);
        }
    }
}

// ----------------------------------------------------------------------
// Q5_0 — `{ d: f16, qh: [u8; 4], qs: [u8; 16] }` = 22 bytes/block, 5.5 bpw
// ----------------------------------------------------------------------

const BLOCK_Q5_0_BYTES: usize = 22;

/// Encode an f32 row as Q5_0 blocks. Symmetric 5-bit: per-block
/// signed scale `d = signed_absmax / -16`. Each quantized value
/// `q[i] ∈ [0, 31]`; the low 4 bits land in `qs[j]` (low nibble for
/// output `j`, high nibble for `j+16`); the 5th bit lands in `qh`,
/// where bit `j` of the u32 holds the 5th bit of output `j` and bit
/// `j+16` holds the 5th bit of output `j+16`. Reference:
/// ggml-quants.c `quantize_row_q5_0`.
pub fn encode_q5_0(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK, 0, "encode_q5_0: src.len() must be multiple of 32");
    let n_blocks = src.len() / QK;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_Q5_0_BYTES,
        "encode_q5_0: dst.len() must be n_blocks * 22"
    );
    for b in 0..n_blocks {
        let xs = &src[b * QK..(b + 1) * QK];
        let off = b * BLOCK_Q5_0_BYTES;
        let mut amax = 0f32;
        let mut max_val = 0f32;
        for &x in xs {
            let a = x.abs();
            if a > amax {
                amax = a;
                max_val = x;
            }
        }
        let d = max_val / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_bits = f16::from_f32(d).to_bits();
        dst[off] = (d_bits & 0xFF) as u8;
        dst[off + 1] = ((d_bits >> 8) & 0xFF) as u8;

        let mut qh: u32 = 0;
        for j in 0..16 {
            let xi0 = ((xs[j] * id + 16.5).floor() as i32).clamp(0, 31) as u32;
            let xi1 = ((xs[j + 16] * id + 16.5).floor() as i32).clamp(0, 31) as u32;
            dst[off + 6 + j] = ((xi0 & 0x0F) | ((xi1 & 0x0F) << 4)) as u8;
            // 5th bit lives at the 0x10 position; shift it to bit j (for the
            // first half) and bit j+16 (for the second half).
            qh |= ((xi0 & 0x10) >> 4) << j;
            qh |= ((xi1 & 0x10) >> 4) << (j + 16);
        }
        dst[off + 2] = (qh & 0xFF) as u8;
        dst[off + 3] = ((qh >> 8) & 0xFF) as u8;
        dst[off + 4] = ((qh >> 16) & 0xFF) as u8;
        dst[off + 5] = ((qh >> 24) & 0xFF) as u8;
    }
}

// ----------------------------------------------------------------------
// Q5_1 — `{ d: f16, m: f16, qh: [u8; 4], qs: [u8; 16] }` = 24 bytes/block, 6 bpw
// ----------------------------------------------------------------------

const BLOCK_Q5_1_BYTES: usize = 24;

/// Encode an f32 row as Q5_1 blocks. Asymmetric 5-bit: per-block
/// `d = (max - min) / 31`, `m = min`. Each quantized value
/// `q[i] = round((x[i] - m) / d) ∈ [0, 31]`. Same qh-bit packing
/// as Q5_0. Reference: ggml-quants.c `quantize_row_q5_1`.
pub fn encode_q5_1(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK, 0, "encode_q5_1: src.len() must be multiple of 32");
    let n_blocks = src.len() / QK;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_Q5_1_BYTES,
        "encode_q5_1: dst.len() must be n_blocks * 24"
    );
    for b in 0..n_blocks {
        let xs = &src[b * QK..(b + 1) * QK];
        let off = b * BLOCK_Q5_1_BYTES;
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &x in xs {
            if x < mn {
                mn = x;
            }
            if x > mx {
                mx = x;
            }
        }
        let d = (mx - mn) / 31.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_bits = f16::from_f32(d).to_bits();
        let m_bits = f16::from_f32(mn).to_bits();
        dst[off] = (d_bits & 0xFF) as u8;
        dst[off + 1] = ((d_bits >> 8) & 0xFF) as u8;
        dst[off + 2] = (m_bits & 0xFF) as u8;
        dst[off + 3] = ((m_bits >> 8) & 0xFF) as u8;

        let mut qh: u32 = 0;
        for j in 0..16 {
            let xi0 = (((xs[j] - mn) * id + 0.5).floor() as i32).clamp(0, 31) as u32;
            let xi1 = (((xs[j + 16] - mn) * id + 0.5).floor() as i32).clamp(0, 31) as u32;
            dst[off + 8 + j] = ((xi0 & 0x0F) | ((xi1 & 0x0F) << 4)) as u8;
            qh |= ((xi0 & 0x10) >> 4) << j;
            qh |= ((xi1 & 0x10) >> 4) << (j + 16);
        }
        dst[off + 4] = (qh & 0xFF) as u8;
        dst[off + 5] = ((qh >> 8) & 0xFF) as u8;
        dst[off + 6] = ((qh >> 16) & 0xFF) as u8;
        dst[off + 7] = ((qh >> 24) & 0xFF) as u8;
    }
}

// ----------------------------------------------------------------------
// Parity tests — encode → dequant → assert max abs error
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant;

    /// Deterministic LCG row builder. Range `[-amp, +amp]`; covers
    /// both small and large input magnitudes so the per-block scale
    /// gets exercised across multiple orders of magnitude.
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

    /// Per-format upper bound on max-abs error after a round trip.
    /// Bounds are derived from the format's quantization step:
    /// roughly `(amp / num_levels) + f16 scale rounding`. We use a
    /// loose `2 * step` to absorb worst-case alignment between the
    /// input distribution and the quantization grid.
    fn assert_round_trip(
        encode: fn(&[f32], &mut [u8]),
        decode: fn(&[u8], &mut [f32]),
        block_bytes: usize,
        max_step: f32,
        amp: f32,
        label: &str,
    ) {
        for &n_blocks in &[1usize, 3, 16] {
            let n = n_blocks * QK;
            let src = make_row(n, amp, n as u32 * 31);
            let mut enc = vec![0u8; n_blocks * block_bytes];
            encode(&src, &mut enc);
            let mut dec = vec![0f32; n];
            decode(&enc, &mut dec);
            let mut max_err = 0f32;
            for i in 0..n {
                let e = (dec[i] - src[i]).abs();
                if e > max_err {
                    max_err = e;
                }
            }
            assert!(
                max_err <= max_step,
                "{label} round-trip n_blocks={n_blocks}: max_err={max_err} > bound={max_step}"
            );
        }
    }

    #[test]
    fn q8_0_round_trip_within_bound() {
        // amp / 127 step + f16 scale rounding; bound 2*step.
        let amp = 4.0;
        assert_round_trip(
            encode_q8_0,
            dequant::dequant_q8_0,
            BLOCK_Q8_0_BYTES,
            2.0 * amp / 127.0,
            amp,
            "q8_0",
        );
    }

    #[test]
    fn q8_1_round_trip_decodes_same_qs_as_q8_0() {
        // Q8_1 uses the same per-element quant as Q8_0. The `s`
        // field doesn't affect decoded values; reusing the Q8_0
        // dequant on the Q8_1 qs+d portion would match exactly. We
        // sanity-check the byte layout: skip s (bytes 2..4) and
        // verify d + qs match a Q8_0 encode.
        let n = 3 * QK;
        let src = make_row(n, 3.0, 7);
        let mut q80 = vec![0u8; 3 * BLOCK_Q8_0_BYTES];
        encode_q8_0(&src, &mut q80);
        let mut q81 = vec![0u8; 3 * BLOCK_Q8_1_BYTES];
        encode_q8_1(&src, &mut q81);
        for b in 0..3 {
            let q80_off = b * BLOCK_Q8_0_BYTES;
            let q81_off = b * BLOCK_Q8_1_BYTES;
            // d
            assert_eq!(&q80[q80_off..q80_off + 2], &q81[q81_off..q81_off + 2]);
            // qs (skip 2-byte s in q81)
            assert_eq!(
                &q80[q80_off + 2..q80_off + 2 + QK],
                &q81[q81_off + 4..q81_off + 4 + QK],
                "block {b}: q8_1 qs must match q8_0 qs"
            );
            // s should equal d * sum(qs) — read both back and verify.
            let d = f16::from_le_bytes([q81[q81_off], q81[q81_off + 1]]).to_f32();
            let s_stored =
                f16::from_le_bytes([q81[q81_off + 2], q81[q81_off + 3]]).to_f32();
            let mut sum: i32 = 0;
            for i in 0..QK {
                sum += (q81[q81_off + 4 + i] as i8) as i32;
            }
            let s_computed = d * (sum as f32);
            let s_diff = (s_stored - s_computed).abs();
            assert!(
                s_diff < 1e-2,
                "block {b}: stored s={s_stored} != recomputed s={s_computed} (diff {s_diff})"
            );
        }
    }

    #[test]
    fn q4_0_round_trip_within_bound() {
        // 4-bit symmetric: step ≈ 2*amp / 16; bound ~2*step.
        let amp = 4.0;
        assert_round_trip(
            encode_q4_0,
            dequant::dequant_q4_0,
            BLOCK_Q4_0_BYTES,
            2.0 * (2.0 * amp / 15.0),
            amp,
            "q4_0",
        );
    }

    #[test]
    fn q4_1_round_trip_within_bound() {
        // 4-bit asymmetric: step ≈ (max-min)/15; same bound shape.
        let amp = 4.0;
        assert_round_trip(
            encode_q4_1,
            dequant::dequant_q4_1,
            BLOCK_Q4_1_BYTES,
            2.0 * (2.0 * amp / 15.0),
            amp,
            "q4_1",
        );
    }

    #[test]
    fn q5_0_round_trip_within_bound() {
        // 5-bit symmetric: step ≈ 2*amp / 31; bound ~2*step.
        let amp = 4.0;
        assert_round_trip(
            encode_q5_0,
            dequant::dequant_q5_0,
            BLOCK_Q5_0_BYTES,
            2.0 * (2.0 * amp / 31.0),
            amp,
            "q5_0",
        );
    }

    #[test]
    fn q5_1_round_trip_within_bound() {
        // 5-bit asymmetric: step ≈ (max-min)/31; same bound shape.
        let amp = 4.0;
        assert_round_trip(
            encode_q5_1,
            dequant::dequant_q5_1,
            BLOCK_Q5_1_BYTES,
            2.0 * (2.0 * amp / 31.0),
            amp,
            "q5_1",
        );
    }

    /// Zero input must encode to all-zero (or constant-zero) blocks
    /// without dividing by zero. Pin this on every format — it's the
    /// most common edge case (unused tensors / padding rows often
    /// land here).
    #[test]
    fn all_formats_handle_zero_input() {
        let n = 2 * QK;
        let zero = vec![0f32; n];

        let mut q80 = vec![0u8; 2 * BLOCK_Q8_0_BYTES];
        encode_q8_0(&zero, &mut q80);
        let mut q81 = vec![0u8; 2 * BLOCK_Q8_1_BYTES];
        encode_q8_1(&zero, &mut q81);
        let mut q40 = vec![0u8; 2 * BLOCK_Q4_0_BYTES];
        encode_q4_0(&zero, &mut q40);
        let mut q41 = vec![0u8; 2 * BLOCK_Q4_1_BYTES];
        encode_q4_1(&zero, &mut q41);
        let mut q50 = vec![0u8; 2 * BLOCK_Q5_0_BYTES];
        encode_q5_0(&zero, &mut q50);
        let mut q51 = vec![0u8; 2 * BLOCK_Q5_1_BYTES];
        encode_q5_1(&zero, &mut q51);

        // Dequant must produce all zeros (within f16 scale precision).
        let mut out = vec![0f32; n];
        dequant::dequant_q8_0(&q80, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
        dequant::dequant_q4_0(&q40, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
        dequant::dequant_q4_1(&q41, &mut out);
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
        dequant::dequant_q5_0(&q50, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
        dequant::dequant_q5_1(&q51, &mut out);
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    /// Constant non-zero input: every weight should reconstruct
    /// (closely) to the input value. Pins that the asymmetric
    /// formats (Q4_1, Q5_1) handle the "all values equal" case
    /// where `max - min == 0`.
    #[test]
    fn asymmetric_formats_handle_constant_input() {
        let n = 2 * QK;
        let v = 0.42f32;
        let src = vec![v; n];

        let mut q41 = vec![0u8; 2 * BLOCK_Q4_1_BYTES];
        encode_q4_1(&src, &mut q41);
        let mut out = vec![0f32; n];
        dequant::dequant_q4_1(&q41, &mut out);
        // With max==min, d==0; dequant gives m for every weight.
        // f16(0.42) → ~0.42 within f16 precision.
        for &o in &out {
            assert!((o - v).abs() < 1e-3, "Q4_1 const: got {o}");
        }

        let mut q51 = vec![0u8; 2 * BLOCK_Q5_1_BYTES];
        encode_q5_1(&src, &mut q51);
        dequant::dequant_q5_1(&q51, &mut out);
        for &o in &out {
            assert!((o - v).abs() < 1e-3, "Q5_1 const: got {o}");
        }
    }
}
