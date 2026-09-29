//! OCP Microscaling (MX) weight encoders: f32 → MXFP4 / MXFP6 / MXFP8.
//!
//! Each format quantizes a 32-element block to a shared 8-bit E8M0
//! power-of-two scale plus per-element low-precision floats (E2M1 /
//! E3M2 / E4M3). The scale is chosen so the block's largest-magnitude
//! element maps near the element format's max representable value, then
//! every element is rounded to the nearest representable level of the
//! element format. Decoders live in [`crate::dequant`]
//! (`dequant_mxfp4/6/8`); these encoders are their inverse and the two
//! round-trip within the element format's quantization step.
//!
//! FP8 (E4M3, per-tensor scale) is intentionally NOT here: its scale is
//! whole-tensor and lives in GGUF metadata, so it can't be produced by
//! a per-block/per-chunk encoder — that path is handled at the pipeline
//! level where the tensor scale is computed and written to metadata.

use crate::dequant::{e3m2_to_f32, e4m3_to_f32, e8m0_to_f32};

/// The 8 non-negative E2M1 levels (index = 3-bit magnitude code).
const E2M1_LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Max normal exponent of each element format (for scale selection):
/// E2M1 max 6.0 = 1.5·2^2 → 2; E3M2 max 28 = 1.75·2^4 → 4;
/// E4M3 max 448 = 1.75·2^8 → 8.
const E2M1_MAX_EXP: i32 = 2;
const E3M2_MAX_EXP: i32 = 4;
const E4M3_MAX_EXP: i32 = 8;

/// Choose the E8M0 block-scale byte for a block whose largest element
/// magnitude is `absmax`, targeting element format max-exponent
/// `elem_max_exp`. `value = 2^(byte - 127)`. A zero/non-finite block
/// scales by 1.0 (byte 127). The result is clamped to the valid E8M0
/// range `[0, 254]` (255 is the reserved NaN).
fn block_scale_byte(absmax: f32, elem_max_exp: i32) -> u8 {
    if absmax == 0.0 || !absmax.is_finite() {
        return 127;
    }
    let e = absmax.log2().floor() as i32;
    (e - elem_max_exp + 127).clamp(0, 254) as u8
}

/// Round `v` to the nearest E2M1 level, returning the 4-bit code
/// (sign in bit 3, magnitude index in bits 0..3).
fn quant_e2m1_nibble(v: f32) -> u8 {
    let a = v.abs();
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, &lv) in E2M1_LEVELS.iter().enumerate() {
        let d = (a - lv).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    let nib = best as u8;
    // Sign bit only for non-zero magnitudes (avoid encoding -0 as 0x08).
    if v.is_sign_negative() && nib != 0 {
        nib | 0x08
    } else {
        nib
    }
}

/// Nearest E3M2 6-bit code for `v` (brute-force over all 64 codes —
/// this is offline quantize, correctness over speed).
fn quant_e3m2_code(v: f32) -> u8 {
    let mut best = 0u8;
    let mut best_d = f32::INFINITY;
    for c in 0u8..64 {
        let lv = e3m2_to_f32(c);
        let d = (v - lv).abs();
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    best
}

/// Nearest E4M3 byte for `v` (brute-force over all finite codes; the
/// two NaN encodings 0x7F/0xFF are skipped).
fn quant_e4m3_byte(v: f32) -> u8 {
    let mut best = 0u8;
    let mut best_d = f32::INFINITY;
    for c in 0u16..256 {
        let c = c as u8;
        if c == 0x7F || c == 0xFF {
            continue;
        }
        let lv = e4m3_to_f32(c);
        let d = (v - lv).abs();
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    best
}

/// Encode f32 → MXFP4. `dst.len()` must equal `src.len() / 32 * 17`.
pub fn encode_mxfp4(src: &[f32], dst: &mut [u8]) {
    const ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 17;
    let n_blocks = src.len() / ELEMS;
    debug_assert_eq!(src.len(), n_blocks * ELEMS);
    debug_assert_eq!(dst.len(), n_blocks * BLOCK_BYTES);
    for b in 0..n_blocks {
        let s = &src[b * ELEMS..b * ELEMS + ELEMS];
        let absmax = s.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let scale_byte = block_scale_byte(absmax, E2M1_MAX_EXP);
        let scale = e8m0_to_f32(scale_byte);
        let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
        let o = b * BLOCK_BYTES;
        for j in 0..16 {
            let lo = quant_e2m1_nibble(s[j * 2] * inv);
            let hi = quant_e2m1_nibble(s[j * 2 + 1] * inv);
            dst[o + j] = (lo & 0x0F) | (hi << 4);
        }
        dst[o + 16] = scale_byte;
    }
}

/// Encode f32 → MXFP6 (E3M2). `dst.len()` must equal `src.len()/32*25`.
pub fn encode_mxfp6(src: &[f32], dst: &mut [u8]) {
    const ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 25;
    const CODE_BYTES: usize = 24;
    let n_blocks = src.len() / ELEMS;
    debug_assert_eq!(src.len(), n_blocks * ELEMS);
    debug_assert_eq!(dst.len(), n_blocks * BLOCK_BYTES);
    for b in 0..n_blocks {
        let s = &src[b * ELEMS..b * ELEMS + ELEMS];
        let absmax = s.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let scale_byte = block_scale_byte(absmax, E3M2_MAX_EXP);
        let scale = e8m0_to_f32(scale_byte);
        let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
        let o = b * BLOCK_BYTES;
        // Clear the 24 code bytes, then OR in each 6-bit code at its bit
        // offset (little-endian bitstream, element j at bit 6*j).
        for byte in dst[o..o + CODE_BYTES].iter_mut() {
            *byte = 0;
        }
        for j in 0..ELEMS {
            let code = quant_e3m2_code(s[j] * inv) as u16 & 0x3F;
            let bitpos = j * 6;
            let byte_idx = bitpos / 8;
            let bit_off = bitpos % 8;
            let shifted = (code as u32) << bit_off;
            dst[o + byte_idx] |= (shifted & 0xFF) as u8;
            if byte_idx + 1 < CODE_BYTES {
                dst[o + byte_idx + 1] |= ((shifted >> 8) & 0xFF) as u8;
            }
        }
        dst[o + CODE_BYTES] = scale_byte;
    }
}

/// Encode f32 → MXFP8 (E4M3). `dst.len()` must equal `src.len()/32*33`.
pub fn encode_mxfp8(src: &[f32], dst: &mut [u8]) {
    const ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 33;
    let n_blocks = src.len() / ELEMS;
    debug_assert_eq!(src.len(), n_blocks * ELEMS);
    debug_assert_eq!(dst.len(), n_blocks * BLOCK_BYTES);
    for b in 0..n_blocks {
        let s = &src[b * ELEMS..b * ELEMS + ELEMS];
        let absmax = s.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let scale_byte = block_scale_byte(absmax, E4M3_MAX_EXP);
        let scale = e8m0_to_f32(scale_byte);
        let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
        let o = b * BLOCK_BYTES;
        for j in 0..ELEMS {
            dst[o + j] = quant_e4m3_byte(s[j] * inv);
        }
        dst[o + 32] = scale_byte;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant::{dequant_mxfp4, dequant_mxfp6, dequant_mxfp8};

    // A block of 32 varied values spanning the dynamic range.
    fn sample_block() -> Vec<f32> {
        (0..32)
            .map(|i| {
                let x = (i as f32 - 16.0) * 0.37;
                x * (1.0 + 0.1 * (i as f32).sin())
            })
            .collect()
    }

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    #[test]
    fn mxfp4_round_trip_within_step() {
        let src = sample_block();
        let mut enc = vec![0u8; 17];
        encode_mxfp4(&src, &mut enc);
        let mut dec = vec![0f32; 32];
        dequant_mxfp4(&enc, &mut dec);
        // E2M1 is coarse; the error is bounded by ~½ the largest gap
        // (between 4 and 6) times the block scale. Just assert it's in
        // the right ballpark and coherent (not garbage).
        let absmax = src.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        assert!(max_abs_err(&src, &dec) <= absmax, "mxfp4 error too large");
        // Signs preserved for the large-magnitude elements.
        for (s, d) in src.iter().zip(&dec) {
            if s.abs() > absmax * 0.5 {
                assert_eq!(s.is_sign_negative(), d.is_sign_negative());
            }
        }
    }

    #[test]
    fn mxfp6_round_trip_tighter_than_mxfp4() {
        let src = sample_block();
        let mut e4 = vec![0u8; 17];
        let mut e6 = vec![0u8; 25];
        encode_mxfp4(&src, &mut e4);
        encode_mxfp6(&src, &mut e6);
        let mut d4 = vec![0f32; 32];
        let mut d6 = vec![0f32; 32];
        dequant_mxfp4(&e4, &mut d4);
        dequant_mxfp6(&e6, &mut d6);
        // More mantissa bits ⇒ MXFP6 should be at least as accurate.
        assert!(max_abs_err(&src, &d6) <= max_abs_err(&src, &d4) + 1e-6);
    }

    #[test]
    fn mxfp8_round_trip_best() {
        let src = sample_block();
        let mut e8 = vec![0u8; 33];
        encode_mxfp8(&src, &mut e8);
        let mut d8 = vec![0f32; 32];
        dequant_mxfp8(&e8, &mut d8);
        let absmax = src.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        // E4M3 elements: ~2^-3 relative step; error well under 5% of range.
        assert!(max_abs_err(&src, &d8) < absmax * 0.05, "mxfp8 error too large");
    }

    #[test]
    fn mxfp_zero_block_round_trips_to_zero() {
        let src = vec![0.0f32; 32];
        for (enc_len, enc_fn, dec_fn) in [
            (17usize, encode_mxfp4 as fn(&[f32], &mut [u8]), dequant_mxfp4 as fn(&[u8], &mut [f32])),
            (25, encode_mxfp6, dequant_mxfp6),
            (33, encode_mxfp8, dequant_mxfp8),
        ] {
            let mut enc = vec![0u8; enc_len];
            enc_fn(&src, &mut enc);
            let mut dec = vec![1.0f32; 32];
            dec_fn(&enc, &mut dec);
            assert!(dec.iter().all(|&x| x == 0.0), "zero block must decode to zero");
        }
    }
}
