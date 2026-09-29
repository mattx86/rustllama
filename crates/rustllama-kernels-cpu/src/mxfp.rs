//! OCP Microscaling (MX) weight matvec kernels: MXFP4 / MXFP6 / MXFP8.
//!
//! Each format packs 32 weights per block sharing one 8-bit E8M0
//! power-of-two scale, with per-element low-precision floats:
//! MXFP4 = E2M1 (4-bit), MXFP6 = E3M2 (6-bit), MXFP8 = E4M3 (8-bit).
//! These are the scalar reference matvecs (`out[m] = Σ_k W[m,k]·x[k]`);
//! AVX2/AVX-512 fast paths slot in alongside the NVFP4 family in a
//! follow-up. The per-element decode is byte-exact with the GGUF
//! loader's `dequant_mxfp4/6/8` and with the SYCL/CUDA kernels — the
//! parity harness checks all three backends against this reference.

/// E2M1 4-bit codebook (index = nibble). Shared with NVFP4.
pub(crate) const E2M1_CODEBOOK: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

pub const MXFP4_BLOCK_ELEMS: usize = 32;
pub const MXFP4_BLOCK_BYTES: usize = 17;
pub const MXFP6_BLOCK_ELEMS: usize = 32;
pub const MXFP6_BLOCK_BYTES: usize = 25;
pub(crate) const MXFP6_CODE_BYTES: usize = 24;
pub const MXFP8_BLOCK_ELEMS: usize = 32;
pub const MXFP8_BLOCK_BYTES: usize = 33;

/// OCP E8M0 shared-scale byte → f32: `2^(byte-127)`, `0xFF` = NaN,
/// `0x00` = `2^-127`. A pure power of two.
#[inline]
pub(crate) fn e8m0_to_f32(b: u8) -> f32 {
    if b == 0xFF {
        return f32::NAN;
    }
    (2.0f32).powi(b as i32 - 127)
}

/// OCP E3M2 (MXFP6 element) 6-bit code → f32. 1 sign / 3 exp (bias 3) /
/// 2 mantissa; no Inf/NaN.
#[inline]
pub(crate) fn e3m2_to_f32(c: u8) -> f32 {
    let sign = (c & 0x20) != 0;
    let exp = (c >> 2) & 0x07;
    let mant = c & 0x03;
    let val = if exp == 0 {
        (mant as f32) / 4.0 * (2.0f32).powi(-2)
    } else {
        (1.0 + (mant as f32) / 4.0) * (2.0f32).powi(exp as i32 - 3)
    };
    if sign {
        -val
    } else {
        val
    }
}

/// FP8 E4M3 byte → f32. 1 sign / 4 exp (bias 7) / 3 mantissa; NaN at
/// `0x7F`/`0xFF`.
#[inline]
pub(crate) fn e4m3_to_f32(b: u8) -> f32 {
    let sign = (b & 0x80) != 0;
    let exp = (b >> 3) & 0x0F;
    let mant = b & 0x07;
    if exp == 0x0F && mant == 0x07 {
        return f32::NAN;
    }
    let val = if exp == 0 {
        (mant as f32) * (1.0 / 512.0)
    } else {
        (1.0 + (mant as f32) / 8.0) * (2.0f32).powi(exp as i32 - 7)
    };
    if sign {
        -val
    } else {
        val
    }
}

/// MXFP4 weight matvec (E2M1 + E8M0). `w_bytes` is `m` rows of
/// `k/32` 17-byte blocks; `k % 32 == 0`.
pub fn matvec_mxfp4_w_f32_a(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    assert_eq!(k % MXFP4_BLOCK_ELEMS, 0, "mxfp4 matvec requires k % 32 == 0");
    let bpr = k / MXFP4_BLOCK_ELEMS;
    assert_eq!(w_bytes.len(), m * bpr * MXFP4_BLOCK_BYTES);
    assert_eq!(x.len(), k);
    assert_eq!(out.len(), m);
    for i in 0..m {
        let row = i * bpr * MXFP4_BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..bpr {
            let off = row + b * MXFP4_BLOCK_BYTES;
            let scale = e8m0_to_f32(w_bytes[off + 16]);
            let xc = &x[b * MXFP4_BLOCK_ELEMS..(b + 1) * MXFP4_BLOCK_ELEMS];
            for j in 0..16 {
                let byte = w_bytes[off + j];
                let lo = (byte & 0x0F) as usize;
                let hi = ((byte >> 4) & 0x0F) as usize;
                acc += scale * E2M1_CODEBOOK[lo] * xc[j * 2];
                acc += scale * E2M1_CODEBOOK[hi] * xc[j * 2 + 1];
            }
        }
        out[i] = acc;
    }
}

/// MXFP6 weight matvec (E3M2 + E8M0). 25-byte blocks; the 24 code bytes
/// are a little-endian bitstream of 32 six-bit codes.
pub fn matvec_mxfp6_w_f32_a(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    assert_eq!(k % MXFP6_BLOCK_ELEMS, 0, "mxfp6 matvec requires k % 32 == 0");
    let bpr = k / MXFP6_BLOCK_ELEMS;
    assert_eq!(w_bytes.len(), m * bpr * MXFP6_BLOCK_BYTES);
    assert_eq!(x.len(), k);
    assert_eq!(out.len(), m);
    for i in 0..m {
        let row = i * bpr * MXFP6_BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..bpr {
            let off = row + b * MXFP6_BLOCK_BYTES;
            let scale = e8m0_to_f32(w_bytes[off + MXFP6_CODE_BYTES]);
            let codes = &w_bytes[off..off + MXFP6_CODE_BYTES];
            let xc = &x[b * MXFP6_BLOCK_ELEMS..(b + 1) * MXFP6_BLOCK_ELEMS];
            for j in 0..MXFP6_BLOCK_ELEMS {
                let bitpos = j * 6;
                let byte_idx = bitpos / 8;
                let bit_off = bitpos % 8;
                let lo = codes[byte_idx] as u16;
                let hi = if byte_idx + 1 < MXFP6_CODE_BYTES {
                    codes[byte_idx + 1] as u16
                } else {
                    0
                };
                let word = lo | (hi << 8);
                let code = ((word >> bit_off) & 0x3F) as u8;
                acc += scale * e3m2_to_f32(code) * xc[j];
            }
        }
        out[i] = acc;
    }
}

/// MXFP8 weight matvec (E4M3 + E8M0). 33-byte blocks: 32 E4M3 bytes + 1
/// scale byte.
pub fn matvec_mxfp8_w_f32_a(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    assert_eq!(k % MXFP8_BLOCK_ELEMS, 0, "mxfp8 matvec requires k % 32 == 0");
    let bpr = k / MXFP8_BLOCK_ELEMS;
    assert_eq!(w_bytes.len(), m * bpr * MXFP8_BLOCK_BYTES);
    assert_eq!(x.len(), k);
    assert_eq!(out.len(), m);
    for i in 0..m {
        let row = i * bpr * MXFP8_BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..bpr {
            let off = row + b * MXFP8_BLOCK_BYTES;
            let scale = e8m0_to_f32(w_bytes[off + 32]);
            let xc = &x[b * MXFP8_BLOCK_ELEMS..(b + 1) * MXFP8_BLOCK_ELEMS];
            for j in 0..MXFP8_BLOCK_ELEMS {
                acc += scale * e4m3_to_f32(w_bytes[off + j]) * xc[j];
            }
        }
        out[i] = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Decode a single block to 32 f32 values (reference for the dot).
    fn decode_mxfp4(block: &[u8]) -> Vec<f32> {
        let scale = e8m0_to_f32(block[16]);
        let mut out = vec![0f32; 32];
        for j in 0..16 {
            out[j * 2] = scale * E2M1_CODEBOOK[(block[j] & 0x0F) as usize];
            out[j * 2 + 1] = scale * E2M1_CODEBOOK[((block[j] >> 4) & 0x0F) as usize];
        }
        out
    }

    // Build one MXFP4 block from 32 nibbles + a scale byte.
    fn mk_mxfp4(nibbles: &[u8; 32], scale_byte: u8) -> Vec<u8> {
        let mut blk = vec![0u8; 17];
        for j in 0..16 {
            blk[j] = (nibbles[j * 2] & 0x0F) | (nibbles[j * 2 + 1] << 4);
        }
        blk[16] = scale_byte;
        blk
    }

    #[test]
    fn mxfp4_matvec_matches_decode_then_dot() {
        // One row, one block. nibbles cycle through the codebook.
        let nibbles: [u8; 32] = std::array::from_fn(|i| (i % 16) as u8);
        let blk = mk_mxfp4(&nibbles, 130); // scale 2^3 = 8
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.25 - 4.0).collect();
        let mut out = vec![0f32; 1];
        matvec_mxfp4_w_f32_a(&blk, &x, &mut out, 1, 32);
        let dec = decode_mxfp4(&blk);
        let ref_dot: f32 = dec.iter().zip(&x).map(|(w, a)| w * a).sum();
        assert!((out[0] - ref_dot).abs() < 1e-3, "got {} want {}", out[0], ref_dot);
    }

    #[test]
    fn mxfp6_matvec_matches_roundtrip_reference() {
        // Encode a known f32 block via the gguf encoder path is in
        // another crate; here we hand-build codes and check the matvec
        // equals decode-then-dot using the same e3m2 decoder.
        let mut blk = vec![0u8; 25];
        // element j gets code (j*2+1) % 64, packed little-endian.
        for j in 0..32usize {
            let code = ((j * 2 + 1) % 64) as u32;
            let bitpos = j * 6;
            let byte_idx = bitpos / 8;
            let bit_off = bitpos % 8;
            let shifted = code << bit_off;
            blk[byte_idx] |= (shifted & 0xFF) as u8;
            if byte_idx + 1 < 24 {
                blk[byte_idx + 1] |= ((shifted >> 8) & 0xFF) as u8;
            }
        }
        blk[24] = 127; // scale 1.0
        let x: Vec<f32> = (0..32).map(|i| (i as f32).cos()).collect();
        let mut out = vec![0f32; 1];
        matvec_mxfp6_w_f32_a(&blk, &x, &mut out, 1, 32);
        // reference: decode each code with e3m2 and dot
        let mut ref_dot = 0f32;
        for j in 0..32usize {
            let bitpos = j * 6;
            let byte_idx = bitpos / 8;
            let bit_off = bitpos % 8;
            let lo = blk[byte_idx] as u16;
            let hi = if byte_idx + 1 < 24 { blk[byte_idx + 1] as u16 } else { 0 };
            let code = (((lo | (hi << 8)) >> bit_off) & 0x3F) as u8;
            ref_dot += e3m2_to_f32(code) * x[j];
        }
        assert!((out[0] - ref_dot).abs() < 1e-4, "got {} want {}", out[0], ref_dot);
    }

    #[test]
    fn mxfp8_matvec_matches_decode_then_dot() {
        let mut blk = vec![0u8; 33];
        for j in 0..32 {
            blk[j] = (j as u8).wrapping_mul(5); // spread across E4M3 codes
            if blk[j] == 0x7F || blk[j] == 0xFF {
                blk[j] = 0x10;
            }
        }
        blk[32] = 124; // scale 2^-3
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let mut out = vec![0f32; 1];
        matvec_mxfp8_w_f32_a(&blk, &x, &mut out, 1, 32);
        let scale = e8m0_to_f32(124);
        let ref_dot: f32 = (0..32).map(|j| scale * e4m3_to_f32(blk[j]) * x[j]).sum();
        assert!((out[0] - ref_dot).abs() < 1e-5, "got {} want {}", out[0], ref_dot);
    }

    #[test]
    fn mxfp4_two_rows() {
        // Two rows, two blocks each (k=64) — checks row striding.
        let nib: [u8; 32] = std::array::from_fn(|i| (i % 16) as u8);
        let b0 = mk_mxfp4(&nib, 127);
        let mut w = Vec::new();
        for _ in 0..4 {
            w.extend_from_slice(&b0);
        }
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        let mut out = vec![0f32; 2];
        matvec_mxfp4_w_f32_a(&w, &x, &mut out, 2, 64);
        // manual reference: both rows share the same block bytes, so
        // each row's dot is Σ over its two blocks of decode·x.
        let dec = decode_mxfp4(&b0);
        let mut want = [0f32; 2];
        for r in 0..2 {
            for blk in 0..2 {
                for j in 0..32 {
                    want[r] += dec[j] * x[blk * 32 + j];
                }
            }
        }
        assert!((out[0] - want[0]).abs() < 1e-3);
        assert!((out[1] - want[1]).abs() < 1e-3);
    }
}
