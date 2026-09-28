//! Pure-Rust quantization decoders.
//!
//! Phase 0 ships reference (non-SIMD) implementations of F16, Q8_0, Q4_K, Q5_K
//! decoders. SIMD fast paths (AVX2 / AVX-512) land in phase 2 behind runtime
//! feature detection. The block layouts here are byte-for-byte compatible
//! with the ggml format definitions; see `docs/quant-formats.md`.

use half::f16;

/// Block sizes (number of weights per block) for each quantized dtype.
pub const QK8_0: usize = 32;
pub const QK_K: usize = 256;

/// TQ1_0: 256 weights / 54 bytes. qs = 48 B (ternary base-3 packing,
/// 5 digits per byte), qh = 4 B (4 digits per byte), d = f16.
pub const TQ1_0_BLOCK_BYTES: usize = 54;
/// TQ2_0: 256 weights / 66 bytes. qs = 64 B (2 bits per weight,
/// 4 weights per byte at shifts 0/2/4/6), d = f16.
pub const TQ2_0_BLOCK_BYTES: usize = 66;

/// Dequantize `bytes` interpreted as F16 into `out`. `out.len()` must equal
/// `bytes.len() / 2`.
pub fn dequant_f16(bytes: &[u8], out: &mut [f32]) {
    debug_assert_eq!(bytes.len(), out.len() * 2);
    for (i, dst) in out.iter_mut().enumerate() {
        let lo = bytes[i * 2];
        let hi = bytes[i * 2 + 1];
        *dst = f16::from_le_bytes([lo, hi]).to_f32();
    }
}

/// Dequantize `bytes` interpreted as BF16 (Google's "brain float" — 1
/// sign + 8 exponent + 7 mantissa bits) into f32. BF16 has the same
/// exponent range as f32 but ~3 decimal digits of precision; the
/// conversion is exact and lossless because BF16's bit pattern is
/// literally the top 16 bits of an IEEE 754 f32. `out.len()` must equal
/// `bytes.len() / 2`.
pub fn dequant_bf16(bytes: &[u8], out: &mut [f32]) {
    debug_assert_eq!(bytes.len(), out.len() * 2);
    for (i, dst) in out.iter_mut().enumerate() {
        let lo = bytes[i * 2];
        let hi = bytes[i * 2 + 1];
        let u = u16::from_le_bytes([lo, hi]);
        *dst = f32::from_bits((u as u32) << 16);
    }
}

/// Q8_0 block: `{ d: f16, qs: [i8; 32] }` = 34 bytes per 32 values.
///
/// `out.len()` must be a multiple of 32 equal to `bytes.len() / 34 * 32`.
pub fn dequant_q8_0(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 34;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK8_0);
    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        for i in 0..QK8_0 {
            let q = bytes[off + 2 + i] as i8;
            out[b * QK8_0 + i] = d * (q as f32);
        }
    }
}

/// Q8_1 block: `{ d: f16, s: f16, qs: [i8; 32] }` = 36 bytes per 32
/// values. Identical per-weight reconstruction to Q8_0 (`value = d *
/// q`); the extra `s` field is a *precomputed* per-block sum
/// (`s = d * sum(qs)`) that dot-product kernels consume — it carries
/// no information not already in `d`/`qs`, so the dequant simply skips
/// it. Matches ggml's `block_q8_1`.
///
/// `out.len()` must be a multiple of 32 equal to `bytes.len() / 36 * 32`.
pub fn dequant_q8_1(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 36;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK8_0);
    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        // bytes[off + 2 .. off + 4] = s (precomputed sum) — unused here.
        for i in 0..QK8_0 {
            let q = bytes[off + 4 + i] as i8;
            out[b * QK8_0 + i] = d * (q as f32);
        }
    }
}

/// Q8_K super-block: 292 bytes per 256 weights. ~9.125 bpw.
/// Layout (matching ggml's `block_q8_K`):
///   { d: f32, qs: [i8; 256], bsums: [i16; 16] }
///
/// Note the per-block scale is `float` (f32) — unlike Q8_0's `f16 d`,
/// because Q8_K is typically used as the *intermediate activation*
/// dtype when multiplying K-quants (e.g., Q4_K weights × Q8_K
/// activations), where the wider-precision scale matters. The
/// `bsums` field precomputes per-16-weight sub-block sums to
/// accelerate Q4_K × Q8_K matmuls; for our Q8_K × F32 matvec we
/// don't need it.
///
/// Per-weight reconstruction: `value = d * q[i]` where `q[i] ∈ [-128, 127]`.
pub fn dequant_q8_k(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 292;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK_K);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f32::from_le_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
        ]);
        let qs = &bytes[off + 4..off + 4 + 256];
        // bsums at off + 260..292 — unused here.
        let out_base = b * QK_K;
        for i in 0..256 {
            let q = qs[i] as i8 as f32;
            out[out_base + i] = d * q;
        }
    }
}

/// Q2_K super-block: 84 bytes per 256 weights. 2.625 bpw — the smallest
/// of the K-quant family. Layout (matching ggml's `block_q2_K`):
///   { scales: [u8; 16], qs: [u8; 64], d: f16, dmin: f16 }
///
/// Each scales byte packs two 4-bit sub-fields: low nibble = scale,
/// high nibble = min (both unsigned). 16 sub-blocks of 16 weights
/// per super-block — one scale byte per sub-block. The `qs` field
/// holds 2-bit unsigned weight values packed 4-per-byte: each byte
/// `qs[l]` contributes one 2-bit value at each of 4 shift positions
/// `0, 2, 4, 6`. The layout walks two 128-weight halves, each
/// consuming 32 q-bytes via 4 shift positions over 32 weights.
///
/// Per-sub-block reconstruction: `dl = d * (scale & 0xF)`,
/// `ml = dmin * (scale >> 4)`, then `value = dl * q - ml` where
/// `q ∈ [0, 3]`. The asymmetric `min`-style offset (same shape as
/// Q4_K/Q5_K) is what lets 2-bit weights still represent useful
/// model parameter distributions.
pub fn dequant_q2_k(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 84;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK_K);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let scales = &bytes[off..off + 16];
        let qs = &bytes[off + 16..off + 16 + 64];
        let d = f16::from_le_bytes([bytes[off + 80], bytes[off + 81]]).to_f32();
        let dmin = f16::from_le_bytes([bytes[off + 82], bytes[off + 83]]).to_f32();

        let mut y_cursor = b * QK_K;
        let mut is = 0usize;
        // QK_K = 256 = 2 chunks of 128 weights. Each chunk consumes 32
        // q-bytes via shift positions 0/2/4/6; each shift covers two
        // 16-weight sub-blocks (q[0..16] and q[16..32] at that shift).
        for chunk in 0..2 {
            let q = &qs[chunk * 32..chunk * 32 + 32];
            for shift in [0u32, 2, 4, 6] {
                // Sub-block A: q[0..16] >> shift
                let sc_a = scales[is];
                let dl_a = d * (sc_a & 0xF) as f32;
                let ml_a = dmin * (sc_a >> 4) as f32;
                for l in 0..16 {
                    let q_val = ((q[l] >> shift) & 3) as f32;
                    out[y_cursor] = dl_a * q_val - ml_a;
                    y_cursor += 1;
                }
                is += 1;
                // Sub-block B: q[16..32] >> shift
                let sc_b = scales[is];
                let dl_b = d * (sc_b & 0xF) as f32;
                let ml_b = dmin * (sc_b >> 4) as f32;
                for l in 0..16 {
                    let q_val = ((q[l + 16] >> shift) & 3) as f32;
                    out[y_cursor] = dl_b * q_val - ml_b;
                    y_cursor += 1;
                }
                is += 1;
            }
        }
    }
}

/// Q3_K super-block: 110 bytes per 256 weights. 3.4375 bpw.
/// Layout (matching ggml's `block_q3_K`):
///   { hmask: [u8; 32], qs: [u8; 64], scales: [u8; 12], d: f16 }
///
/// Each 3-bit weight splits its high bit into `hmask` (one bit per
/// weight) and its low 2 bits into `qs` (4 weights packed per byte
/// across 4 shift positions). The 12-byte `scales` packs 16 signed
/// 6-bit sub-scales (one per 16-weight sub-block) via a baroque layout
/// optimized for SIMD extraction in upstream ggml; we unpack with the
/// same `kmask1 = 0x03030303 / kmask2 = 0x0f0f0f0f` shuffle. Each
/// 16-weight sub-block uses `dl = d * (scale - 32)` (signed sub-scale)
/// and emits `dl * (low2 - (hi_bit_set ? 0 : 4))` per weight — the
/// 8-value range `-4..3` is the "3-bit signed" reconstruction.
pub fn dequant_q3_k(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 110;
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK_K);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let hmask = &bytes[off..off + 32];
        let qs = &bytes[off + 32..off + 32 + 64];
        let sc = &bytes[off + 32 + 64..off + 32 + 64 + 12];
        let d_all = f16::from_le_bytes([bytes[off + 108], bytes[off + 109]]).to_f32();

        // Unpack the 12 scale bytes into 16 i8 sub-scales. The layout is
        // a 3-word reshuffle from ggml's reference: each of `aux[0..3]`
        // holds 4 packed 6-bit values where the low 4 bits live in the
        // four bytes of one word and the high 2 bits are gathered from
        // a fourth word's nibbles.
        let mut aux = [0u32; 4];
        aux[0] = u32::from_le_bytes([sc[0], sc[1], sc[2], sc[3]]);
        aux[1] = u32::from_le_bytes([sc[4], sc[5], sc[6], sc[7]]);
        aux[2] = u32::from_le_bytes([sc[8], sc[9], sc[10], sc[11]]);
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let aux_bytes: [u8; 16] = bytemuck::cast(aux);
        let scales: [i8; 16] = aux_bytes.map(|x| x as i8);

        let mut q_cursor = 0usize;
        let mut y_cursor = b * QK_K;
        let mut m: u8 = 1;
        let mut is = 0usize;
        // QK_K = 256 = 2 chunks of 128 weights each. Each chunk uses
        // 32 bytes of `qs` (4 shift positions × 32 weights = 128) and
        // advances `m` through 4 successive single-bit positions.
        for _chunk in 0..2 {
            let mut shift: u32 = 0;
            for _j in 0..4 {
                let dl = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let lo = ((qs[q_cursor + l] >> shift) & 3) as i32;
                    let hi_sub = if hmask[l] & m != 0 { 0 } else { 4 };
                    out[y_cursor] = dl * ((lo - hi_sub) as f32);
                    y_cursor += 1;
                }
                let dl = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let lo = ((qs[q_cursor + l + 16] >> shift) & 3) as i32;
                    let hi_sub = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                    out[y_cursor] = dl * ((lo - hi_sub) as f32);
                    y_cursor += 1;
                }
                shift += 2;
                m = m.wrapping_shl(1);
            }
            q_cursor += 32;
        }
    }
}

/// Q4_K super-block: 144 bytes per 256 values.
///   { d: f16, dmin: f16, scales: [u8; 12], qs: [u8; 128] }
///
/// The 12-byte `scales` array packs eight (scale, min) 6-bit pairs.
///
/// Layout follows ggml's `dequantize_row_q4_K`: each 64-output group reads
/// 32 contiguous q bytes — 32 outputs are low nibbles (scaled by sc[2k]),
/// the next 32 are HIGH nibbles of the SAME 32 q bytes (scaled by sc[2k+1]).
pub fn dequant_q4_k(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 144;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK_K);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let dmin = f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]).to_f32();
        let scales_raw = &bytes[off + 4..off + 16];
        let qs = &bytes[off + 16..off + 16 + 128];

        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for j in 0..8 {
            if j < 4 {
                sc[j] = scales_raw[j] & 0x3F;
                mn[j] = scales_raw[j + 4] & 0x3F;
            } else {
                sc[j] = (scales_raw[j + 4] & 0x0F) | ((scales_raw[j - 4] >> 6) << 4);
                mn[j] = (scales_raw[j + 4] >> 4) | ((scales_raw[j] >> 6) << 4);
            }
        }

        let out_base = b * QK_K;
        // 4 groups of 64 outputs, each pulling from 32 q bytes.
        for group in 0..4 {
            let q_chunk = &qs[group * 32..(group + 1) * 32];
            let sc_lo = sc[group * 2] as f32;
            let mn_lo = mn[group * 2] as f32;
            let sc_hi = sc[group * 2 + 1] as f32;
            let mn_hi = mn[group * 2 + 1] as f32;
            let d_lo = d * sc_lo;
            let m_lo = dmin * mn_lo;
            let d_hi = d * sc_hi;
            let m_hi = dmin * mn_hi;
            let dst = out_base + group * 64;
            for l in 0..32 {
                let q = q_chunk[l];
                out[dst + l] = d_lo * (q & 0x0F) as f32 - m_lo;
                out[dst + 32 + l] = d_hi * (q >> 4) as f32 - m_hi;
            }
        }
    }
}

/// Q4_0 block: 18 bytes per 32 values.
///   { d: f16, qs: [u8; 16] }
///
/// `qs[j]` packs two 4-bit nibbles → outputs at positions `j` and `j+16`
/// within the block. Values are shifted to be symmetric around zero
/// (`- 8`). This is the "no min, no high-bit" baseline of the K-quant
/// family; for 5-bit precision with the same shape see [`dequant_q5_0`].
pub fn dequant_q4_0(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 18;
    const QK4_0: usize = 32;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK4_0);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let qs = &bytes[off + 2..off + 2 + 16];
        for j in 0..16 {
            let x0 = (qs[j] & 0x0F) as i32 - 8;
            let x1 = (qs[j] >> 4) as i32 - 8;
            out[b * QK4_0 + j] = d * x0 as f32;
            out[b * QK4_0 + j + 16] = d * x1 as f32;
        }
    }
}

/// Q4_1 block: 20 bytes per 32 values.
///   { d: f16, m: f16, qs: [u8; 16] }
///
/// Asymmetric 4-bit: `value = d * q + m`, where `q` is the unsigned
/// nibble in `[0, 15]`. Differs from [`dequant_q4_0`] in that the
/// per-block scale is paired with an explicit min/offset `m` so the
/// quantization range need not be centered on zero. The output layout
/// is identical to Q4_0: `qs[j]` low nibble → `out[j]`, high nibble →
/// `out[j + 16]`.
pub fn dequant_q4_1(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 20;
    const QK4_1: usize = 32;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK4_1);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let m = f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]).to_f32();
        let qs = &bytes[off + 4..off + 4 + 16];
        for j in 0..16 {
            let x0 = (qs[j] & 0x0F) as i32;
            let x1 = (qs[j] >> 4) as i32;
            out[b * QK4_1 + j] = d * x0 as f32 + m;
            out[b * QK4_1 + j + 16] = d * x1 as f32 + m;
        }
    }
}

/// Q5_0 block: 22 bytes per 32 values.
///   { d: f16, qh: [u8; 4], qs: [u8; 16] }
///
/// `qs[j]` packs two 4-bit nibbles → outputs at positions `j` and `j+16`
/// within the block. `qh` is a packed u32 of "5th bits": bit `j` is the high
/// bit of output `j`, bit `j+16` is the high bit of output `j+16`. Values
/// are shifted to be symmetric around zero (`- 16`).
pub fn dequant_q5_0(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 22;
    const QK5_0: usize = 32;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK5_0);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let qh = u32::from_le_bytes([
            bytes[off + 2],
            bytes[off + 3],
            bytes[off + 4],
            bytes[off + 5],
        ]);
        let qs = &bytes[off + 6..off + 22];
        for j in 0..16 {
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = ((qs[j] & 0x0F) as i32 | xh_0 as i32) - 16;
            let x1 = ((qs[j] >> 4) as i32 | xh_1 as i32) - 16;
            out[b * QK5_0 + j] = d * x0 as f32;
            out[b * QK5_0 + j + 16] = d * x1 as f32;
        }
    }
}

/// Q5_1 block: 24 bytes per 32 values.
///   { d: f16, m: f16, qh: [u8; 4], qs: [u8; 16] }
///
/// Asymmetric 5-bit: `value = d * q + m` where `q` is the unsigned
/// 5-bit value `(low4 | bit5 << 4)` in `[0, 31]`. The `qh` packed-u32
/// holds the 5th bits in the same arrangement as [`dequant_q5_0`]:
/// bit `j` is the 5th bit of output `j`, bit `j+16` is the 5th bit of
/// output `j+16`. Output layout matches Q5_0.
pub fn dequant_q5_1(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 24;
    const QK5_1: usize = 32;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK5_1);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let m = f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]).to_f32();
        let qh = u32::from_le_bytes([
            bytes[off + 4],
            bytes[off + 5],
            bytes[off + 6],
            bytes[off + 7],
        ]);
        let qs = &bytes[off + 8..off + 24];
        for j in 0..16 {
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = ((qs[j] & 0x0F) as i32) | (xh_0 as i32);
            let x1 = ((qs[j] >> 4) as i32) | (xh_1 as i32);
            out[b * QK5_1 + j] = d * x0 as f32 + m;
            out[b * QK5_1 + j + 16] = d * x1 as f32 + m;
        }
    }
}

/// Q6_K super-block: 210 bytes per 256 values.
///   { ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], d: f16 }
///
/// Per-weight: 4 low bits from `ql`, 2 high bits from `qh`, packed into a
/// signed 6-bit value (subtracting 32 to center). 16 sub-blocks of 16 each,
/// each with its own `i8` scale, all multiplied by the super-block `d`.
pub fn dequant_q6_k(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 210;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK_K);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let ql = &bytes[off..off + 128];
        let qh = &bytes[off + 128..off + 128 + 64];
        let scales_raw = &bytes[off + 128 + 64..off + 128 + 64 + 16];
        let d = f16::from_le_bytes([bytes[off + 208], bytes[off + 209]]).to_f32();

        // Layout follows ggml: each group of 128 outputs uses `ql[0..64]`
        // (low nibbles) and `qh[0..32]` (high 2-bit pairs); the next 128 use
        // the upper nibbles of `ql[0..64]` and `ql[64..128]` similarly.
        // Implementation below mirrors `dequantize_row_q6_K` for QK_K=256.
        for n in 0..2 {
            for l in 0..32 {
                let is = l / 16 + n * 8; // scale index
                let q1 = ((ql[64 * n + l] & 0x0F) as i32
                    | (((qh[32 * n + l] >> 0) & 0x03) << 4) as i32)
                    - 32;
                let q2 = ((ql[64 * n + l + 32] & 0x0F) as i32
                    | (((qh[32 * n + l] >> 2) & 0x03) << 4) as i32)
                    - 32;
                let q3 = ((ql[64 * n + l] >> 4) as i32
                    | (((qh[32 * n + l] >> 4) & 0x03) << 4) as i32)
                    - 32;
                let q4 = ((ql[64 * n + l + 32] >> 4) as i32
                    | (((qh[32 * n + l] >> 6) & 0x03) << 4) as i32)
                    - 32;
                let base = b * QK_K + n * 128 + l;
                // Q6_K scales are SIGNED int8 — sign-extend properly via i8.
                let s0 = (scales_raw[is] as i8) as f32;
                let s1 = (scales_raw[is + 2] as i8) as f32;
                let s2 = (scales_raw[is + 4] as i8) as f32;
                let s3 = (scales_raw[is + 6] as i8) as f32;
                out[base] = d * s0 * q1 as f32;
                out[base + 32] = d * s1 * q2 as f32;
                out[base + 64] = d * s2 * q3 as f32;
                out[base + 96] = d * s3 * q4 as f32;
            }
        }
    }
}

/// Q5_K super-block: 176 bytes per 256 values.
///   { d: f16, dmin: f16, scales: [u8; 12], qh: [u8; 32], qs: [u8; 128] }
///
/// Same group-of-64 layout as Q4_K, with one extra "5th bit" pulled from
/// the qh array. ggml's loop selects 5th-bit masks `u1`, `u2` that shift
/// left by 2 each iteration; ours uses an explicit `(qh[l] >> bit_pos) & 1`.
pub fn dequant_q5_k(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 176;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK_K);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let dmin = f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]).to_f32();
        let scales_raw = &bytes[off + 4..off + 16];
        let qh = &bytes[off + 16..off + 48];
        let qs = &bytes[off + 48..off + 48 + 128];

        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for j in 0..8 {
            if j < 4 {
                sc[j] = scales_raw[j] & 0x3F;
                mn[j] = scales_raw[j + 4] & 0x3F;
            } else {
                sc[j] = (scales_raw[j + 4] & 0x0F) | ((scales_raw[j - 4] >> 6) << 4);
                mn[j] = (scales_raw[j + 4] >> 4) | ((scales_raw[j] >> 6) << 4);
            }
        }

        let out_base = b * QK_K;
        // 4 groups of 64 outputs, 32 q bytes each. The 5th bit is selected
        // from qh[l]'s bit (group*2) for lows, (group*2 + 1) for highs.
        for group in 0..4 {
            let q_chunk = &qs[group * 32..(group + 1) * 32];
            let sc_lo = sc[group * 2] as f32;
            let mn_lo = mn[group * 2] as f32;
            let sc_hi = sc[group * 2 + 1] as f32;
            let mn_hi = mn[group * 2 + 1] as f32;
            let d_lo = d * sc_lo;
            let m_lo = dmin * mn_lo;
            let d_hi = d * sc_hi;
            let m_hi = dmin * mn_hi;
            let bit_lo = group * 2;
            let bit_hi = group * 2 + 1;
            let dst = out_base + group * 64;
            for l in 0..32 {
                let q_byte = q_chunk[l];
                let qh_byte = qh[l];
                let lo = q_byte & 0x0F;
                let hi = q_byte >> 4;
                let bit_lo_set = ((qh_byte >> bit_lo) & 1) != 0;
                let bit_hi_set = ((qh_byte >> bit_hi) & 1) != 0;
                let lo_full = lo | (if bit_lo_set { 16 } else { 0 });
                let hi_full = hi | (if bit_hi_set { 16 } else { 0 });
                out[dst + l] = d_lo * lo_full as f32 - m_lo;
                out[dst + 32 + l] = d_hi * hi_full as f32 - m_hi;
            }
        }
    }
}

/// IQ4_XS / IQ4_NL lookup table — non-linear 4-bit codebook. Each
/// 4-bit index in a `qs` byte maps to one of these 16 signed i8
/// values. Matches llama.cpp's `kvalues_iq4nl`. Shared by both IQ4
/// variants — the name reflects the upstream convention.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// IQ4_NL block: 32 weights per 18-byte block. Layout:
///   `{ d: f16, qs: [u8; 16] }`
/// Each `qs` byte encodes two 4-bit indices into [`KVALUES_IQ4NL`];
/// final value is `d * KVALUES_IQ4NL[idx]`. Simpler than IQ4_XS —
/// just one per-block scale, no sub-block hierarchy.
///
/// Output layout: `qs[j]`'s low nibble → `out[j]`,
/// high nibble → `out[j + 16]`. (Match upstream llama.cpp.)
/// NVFP4: 16 elements per 9-byte block (8 packed nibbles + 1 FP8
/// E4M3 scale). Each nibble indexes into the E2M1 codebook. The
/// per-block dequant + matvec is in `rustllama-kernels-cpu::nvfp4`;
/// this helper exists so the GGUF loader can produce a clean F32
/// view of the weights for the F16/F32 conversion path.
pub fn dequant_nvfp4(bytes: &[u8], out: &mut [f32]) {
    // Codebook + FP8 decode logic mirror `rustllama-kernels-cpu::nvfp4`.
    // Pinned here to avoid a circular dep — the gguf crate is the
    // foundation that kernels-cpu builds on.
    const CODEBOOK: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    const BLOCK_BYTES: usize = 9;
    const ELEMS_PER_BLOCK: usize = 16;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * ELEMS_PER_BLOCK);
    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let scale_byte = bytes[off + 8];
        let scale = e4m3_to_f32(scale_byte);
        let dst = b * ELEMS_PER_BLOCK;
        for j in 0..8 {
            let byte = bytes[off + j];
            let lo = (byte & 0x0F) as usize;
            let hi = ((byte >> 4) & 0x0F) as usize;
            out[dst + j * 2] = CODEBOOK[lo] * scale;
            out[dst + j * 2 + 1] = CODEBOOK[hi] * scale;
        }
    }
}

/// FP8 E4M3 → f32. Mirrors `rustllama-kernels-cpu::nvfp4::e4m3_to_f32`.
/// 4-bit exponent (bias 7) + 3-bit mantissa, NaN at `0x7F`/`0xFF`.
fn e4m3_to_f32(b: u8) -> f32 {
    let sign = (b & 0x80) != 0;
    let exp = (b >> 3) & 0x0F;
    let mant = b & 0x07;
    if exp == 0x0F && mant == 0x07 {
        return f32::NAN;
    }
    let val = if exp == 0 {
        (mant as f32) * (1.0 / 512.0)
    } else {
        let m = 1.0 + (mant as f32) / 8.0;
        let e = (exp as i32) - 7;
        m * (2.0f32).powi(e)
    };
    if sign {
        -val
    } else {
        val
    }
}

/// TQ2_0: 66 bytes per 256 weights (2.0625 bpw). The simpler of the
/// two ternary quants — each 2-bit field decodes to `{0, 1, 2}` which
/// becomes the trit `{-1, 0, +1}` after subtracting 1.
///
/// Layout (matching ggml's `block_tq2_0`):
///   { qs: [u8; 64], d: f16 }
///
/// Iteration order (mirrors `dequantize_row_tq2_0` in ggml-quants.c):
///   - The 64 `qs` bytes split into two 32-byte halves.
///   - Within each half, shift `l = 0..4` selects two bits per byte;
///     for each `l` we walk all 32 bytes, emitting one weight per byte.
///   - So weights 0..32 = shift 0 of half 0, weights 32..64 = shift 1
///     of half 0, …, weights 128..160 = shift 0 of half 1, etc.
///
/// `out.len()` must equal `bytes.len() / 66 * 256`.
pub fn dequant_tq2_0(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 66;
    const HALF_BYTES: usize = 32;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK_K);
    let mut o = 0usize;
    for b in 0..n_blocks {
        let block_off = b * BLOCK_BYTES;
        let qs = &bytes[block_off..block_off + 64];
        let d = f16::from_le_bytes([bytes[block_off + 64], bytes[block_off + 65]]).to_f32();
        // Two halves of 32 bytes each. Within each half we sweep
        // shift `l` from 0 to 3 (2-bit fields), and for each shift
        // emit the 32 trits from the 32 bytes of the half.
        for half in 0..2 {
            let half_off = half * HALF_BYTES;
            for l in 0..4 {
                let shift = (l as u32) * 2;
                for m in 0..32 {
                    let q = ((qs[half_off + m] >> shift) & 0x3) as i32;
                    out[o] = d * (q - 1) as f32;
                    o += 1;
                }
            }
        }
    }
    debug_assert_eq!(o, out.len());
}

/// TQ1_0: 54 bytes per 256 weights (1.6875 bpw). The tightest
/// production ternary quant — packs 5 trits into a single byte for the
/// bulk of the data, plus a smaller 4-trit packing for the tail.
///
/// Layout (matching ggml's `block_tq1_0`):
///   { qs: [u8; 48], qh: [u8; 4], d: f16 }
///
/// **Wire format (ggml fixed-point, not plain base-3).** ggml does
/// NOT store the raw base-3 value `v = t0*81 + t1*27 + … + t4`.
/// Instead each byte holds `ceil(v * 256 / 243)` — a *ceiling*
/// fixed-point scaling — so that individual trits can be recovered
/// without a division:
///   `q  = byte * 3^n`            (u8, wraps mod 256)
///   `xi = ((q as u16) * 3) >> 8` ∈ {0, 1, 2}
/// with `n = 0` yielding the most-significant trit. This is the exact
/// trick the in-tree PTQ1_0 codec ([`dequant_ptq1_0`]) uses, and it is
/// what makes us byte-compatible with `dequantize_row_tq1_0`. (The
/// earlier plain `(byte / 3^k) % 3` decode round-tripped against our
/// own encoder but produced garbage on real llama.cpp TQ1_0 files.)
///
/// The 4-trit `qh` tail is packed into the *top* 4 digit positions
/// (the encoder shifts the accumulated value up by one trit before the
/// ceiling scale), so it is extracted with `n = 0..4` too.
///
/// Iteration order (mirrors `dequantize_row_tq1_0` in ggml-quants.c):
///   - chunk 0: `qs[0..32]`, 5 trits each → 160 weights
///   - chunk 1: `qs[32..48]`, 5 trits each → 80 weights
///   - tail: `qh[0..4]`, 4 trits each → 16 weights
///
/// NOTE: byte layout matches ggml by construction/derivation but has
/// not been diffed against a real llama.cpp-produced TQ1_0 GGUF (we
/// have none in-tree) — see the module tests for the hand-computed
/// reference vector that pins the format.
///
/// `out.len()` must equal `bytes.len() / 54 * 256`.
pub fn dequant_tq1_0(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 54;
    // pow3[n] as u8 so the `byte * pow3[n]` product wraps mod 256,
    // exactly like ggml's `uint8_t q`.
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK_K);

    let mut o = 0usize;
    for b in 0..n_blocks {
        let block_off = b * BLOCK_BYTES;
        let qs = &bytes[block_off..block_off + 48];
        let qh = &bytes[block_off + 48..block_off + 52];
        let d = f16::from_le_bytes([bytes[block_off + 52], bytes[block_off + 53]]).to_f32();
        // Chunk 0: qs[0..32], 5 trits per byte → 160 weights.
        for &pow in POW3.iter() {
            for m in 0..32 {
                let q = qs[m].wrapping_mul(pow);
                let xi = ((q as u16) * 3) >> 8;
                out[o] = d * (xi as i32 - 1) as f32;
                o += 1;
            }
        }
        // Chunk 1: qs[32..48], 5 trits per byte → 80 weights.
        for &pow in POW3.iter() {
            for m in 0..16 {
                let q = qs[32 + m].wrapping_mul(pow);
                let xi = ((q as u16) * 3) >> 8;
                out[o] = d * (xi as i32 - 1) as f32;
                o += 1;
            }
        }
        // Tail: qh[0..4], 4 trits per byte → 16 weights. Only the top
        // 4 digit positions are used (POW3[0..4]).
        for &pow in POW3[..4].iter() {
            for m in 0..4 {
                let q = qh[m].wrapping_mul(pow);
                let xi = ((q as u16) * 3) >> 8;
                out[o] = d * (xi as i32 - 1) as f32;
                o += 1;
            }
        }
    }
    debug_assert_eq!(o, out.len());
}

/// PQ2_0 (PrismML Bonsai): 34 bytes per 128 weights (2.125 bpw).
/// Layout `{ d: f16, qs: [u8; 32] }` — one f16 scale for the whole
/// 128-weight group, then little-endian 2-bit codes
/// (`(qs[j/4] >> ((j%4)*2)) & 3`). Reconstruction `d * (code - 1)`:
/// `00`→−d, `01`→0, `10`→+d, `11`→+2d. The +2 level is part of the
/// codec (shared with Prism's group-64 Q2_0) even though their
/// ternary encoder never emits it — honor it on decode.
///
/// Ported from `dequantize_row_pq2_0` in the PrismML llama.cpp fork
/// (ggml-quants.c). `out.len()` must equal `bytes.len() / 34 * 128`.
pub fn dequant_pq2_0(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 34;
    const QK: usize = 128;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK);
    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let qs = &bytes[off + 2..off + 2 + 32];
        for j in 0..QK {
            let q = ((qs[j / 4] >> ((j % 4) * 2)) & 0x3) as i32;
            out[b * QK + j] = d * (q - 1) as f32;
        }
    }
}

/// PTQ1_0 (PrismML Bonsai): 28 bytes per 128 weights (1.75 bpw).
/// Layout `{ qs: [u8; 24], qh: [u8; 2], d: f16 }`. Base-3 CEILING
/// fixed-point packing (ggml TQ1_0 family): each qs byte holds 5
/// trits as `ceil(v * 256 / 243)` of the base-3 value `v` whose MOST
/// significant digit is the chunk's first element; each qh byte
/// holds 4 trits pre-shifted into the top digits. Extraction is the
/// wrapping-multiply trick — `digit_n = ((byte.wrapping_mul(3^n) as
/// u16) * 3) >> 8` — NOT plain div/mod (that is only valid for
/// non-ceiling packing; see `dequant_tq1_0`'s caveat).
///
/// The qs staging generalizes TQ1_0's fixed 32+16 to chunk sizes
/// {32, 16, 8}: for qs = 24 bytes that resolves to one 16-byte chunk
/// (80 weights) then one 8-byte chunk (40 weights); element order
/// within a chunk of size `c` is `n*c + m` (trit index outer, byte
/// inner). qh contributes the last 8 weights, element `n*2 + h`.
///
/// Ported from `dequantize_row_ptq1_0` in the PrismML llama.cpp fork
/// (ggml-quants.c:2255-2285, stages {32,16,8}).
/// `out.len()` must equal `bytes.len() / 28 * 128`.
pub fn dequant_ptq1_0(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 28;
    const QK: usize = 128;
    const QS_BYTES: usize = 24;
    const QH_BYTES: usize = 2;
    const STAGES: [usize; 3] = [32, 16, 8];
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK);
    let mut o = 0usize;
    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let qs = &bytes[off..off + QS_BYTES];
        let qh = &bytes[off + QS_BYTES..off + QS_BYTES + QH_BYTES];
        let d = f16::from_le_bytes([bytes[off + 26], bytes[off + 27]]).to_f32();
        let mut j = 0usize;
        for &c in STAGES.iter() {
            while j + c <= QS_BYTES {
                for n in 0..5 {
                    for m in 0..c {
                        let q = qs[j + m].wrapping_mul(POW3[n]);
                        let xi = ((q as u16) * 3) >> 8;
                        out[o] = d * (xi as i32 - 1) as f32;
                        o += 1;
                    }
                }
                j += c;
            }
        }
        for n in 0..4 {
            for h in 0..QH_BYTES {
                let q = qh[h].wrapping_mul(POW3[n]);
                let xi = ((q as u16) * 3) >> 8;
                out[o] = d * (xi as i32 - 1) as f32;
                o += 1;
            }
        }
    }
    debug_assert_eq!(o, out.len());
}

pub fn dequant_iq4_nl(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 18;
    const QK_NL: usize = 32;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK_NL);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let qs = &bytes[off + 2..off + 18];
        let dst_off = b * QK_NL;
        for j in 0..16 {
            let q = qs[j];
            let lo = (q & 0x0F) as usize;
            let hi = (q >> 4) as usize;
            out[dst_off + j] = d * (KVALUES_IQ4NL[lo] as f32);
            out[dst_off + 16 + j] = d * (KVALUES_IQ4NL[hi] as f32);
        }
    }
}

/// IQ3_S 9-bit codebook. Each u32 packs four i8 grid coordinates (one
/// per byte). The codebook is the specific learned table from ggml /
/// llama.cpp's `iq3s_grid`. Constants licensed under MIT (transcribed
/// from `ggml-common.h` in the ggml-org/llama.cpp repo at commit
/// `master` on the date this was added).
pub const IQ3S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
];

/// IQ3_S sign-bit selector mask. `kmask_iq2xs[j]` is the bit in a
/// `signs` byte that controls the sign of output position `j`
/// (j in 0..8 within a sign-byte group). Transcribed from
/// `kmask_iq2xs` in ggml-common.h.
pub const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// IQ1_S sub-block delta. The mid-point between two adjacent grid
/// codeword values; flipping the high `qh` bit shifts the entire
/// 32-weight sub-block by `±IQ1S_DELTA` relative to the grid. The
/// classical IQ1_S setting; matches llama.cpp's `IQ1S_DELTA`.
pub const IQ1S_DELTA: f32 = 0.125;

/// IQ1_S super-block: 50 bytes per 256 weights. 1.5625 bpw — the
/// smallest quant in common circulation.
///
/// Layout (matching ggml's `block_iq1_s`):
///   { d: f16, qs: [u8; 32], qh: [u16; 8] }
///
/// Eight ib32 sub-blocks per super-block (32 weights each). For
/// each ib32:
///   - `dl = d * (2*((qh[ib32] >> 12) & 7) + 1)` — per-sub-block scale
///     (odd values 1..15).
///   - `delta = if qh[ib32] & 0x8000 { -1 - IQ1S_DELTA } else { -1 + IQ1S_DELTA }`
///     — the per-sub-block additive offset that lets a 1-bit symbol
///     represent both "near +1" and "near -1" sides of the grid.
///   - For each `l ∈ 0..4`:
///     - `idx = qs[4*ib32 + l] | (((qh[ib32] >> (3*l)) & 7) << 8)` —
///       11-bit codebook index.
///     - The 8 packed-i8 grid coordinates from `IQ1S_GRID[idx]` map
///       onto outputs `32*ib32 + 8*l + j` for `j ∈ 0..8`:
///       `out = dl * (grid[j] + delta)`.
pub fn dequant_iq1_s(bytes: &[u8], out: &mut [f32]) {
    use crate::iq1_grid::IQ1S_GRID;
    const BLOCK_BYTES: usize = 50;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK_K);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let qs = &bytes[off + 2..off + 2 + 32];
        // qh starts at off + 34; eight u16 little-endian, 16 bytes total.
        let qh_bytes = &bytes[off + 34..off + 34 + 16];
        let mut out_off = b * QK_K;
        for ib32 in 0..8 {
            let qh = u16::from_le_bytes([qh_bytes[ib32 * 2], qh_bytes[ib32 * 2 + 1]]);
            let dl = d * (2.0 * ((qh >> 12) & 7) as f32 + 1.0);
            let delta = if qh & 0x8000 != 0 {
                -1.0 - IQ1S_DELTA
            } else {
                -1.0 + IQ1S_DELTA
            };
            for l in 0..4 {
                let idx = qs[4 * ib32 + l] as usize | ((((qh >> (3 * l)) & 7) as usize) << 8);
                let grid = IQ1S_GRID[idx].to_le_bytes();
                for j in 0..8 {
                    let g = grid[j] as i8 as f32;
                    out[out_off + 8 * l + j] = dl * (g + delta);
                }
            }
            out_off += 32;
        }
    }
}

/// IQ1_M super-block: 56 bytes per 256 weights. 1.75 bpw. Same
/// `IQ1S_GRID` codebook as IQ1_S, with a more elaborate per-16-weight
/// scale + delta layout.
///
/// Layout (matching ggml's `block_iq1_m`):
///   { qs: [u8; 32], qh: [u8; 16], scales: [u8; 8] }
///
/// Reconstruction follows ggml's reference: a 4×u16 view of `scales`
/// reassembles a u16 whose bits encode the super-block f16 `d` (top
/// nibbles of `sc[0..4]`); the same 4 u16s also pack 16 3-bit
/// sub-block scales (two per 16-weight half of each ib32 sub-block);
/// and `qh` supplies both the high 3 bits of every grid index AND
/// per-16-weight delta sign bits at bits 3 and 7.
pub fn dequant_iq1_m(bytes: &[u8], out: &mut [f32]) {
    use crate::iq1_grid::IQ1S_GRID;
    const BLOCK_BYTES: usize = 56;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK_K);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let qs = &bytes[off..off + 32];
        let qh = &bytes[off + 32..off + 32 + 16];
        let scales_bytes = &bytes[off + 48..off + 48 + 8];

        // Reassemble the four u16 scale words and the super-block
        // f16 d from their top-nibble contributions. Mirrors ggml's
        // `iq1m_scale_t` reconstruction.
        let mut sc = [0u16; 4];
        for i in 0..4 {
            sc[i] = u16::from_le_bytes([scales_bytes[i * 2], scales_bytes[i * 2 + 1]]);
        }
        let d_bits: u16 = (sc[0] >> 12)
            | ((sc[1] >> 8) & 0x00F0)
            | ((sc[2] >> 4) & 0x0F00)
            | (sc[3] & 0xF000);
        let d = f16::from_bits(d_bits).to_f32();

        let mut out_off = b * QK_K;
        // 8 ib32 sub-blocks; each consumes 4 bytes of qs, 2 bytes of qh,
        // and indexes 2 of the 4 sc words.
        for ib in 0..8 {
            // Per-16-weight scales: dl1 covers weights 0..15 of this
            // ib32, dl2 covers 16..31. Bit offsets within sc[ib/2]
            // depend on whether this is the even or odd member of
            // its sc pair.
            let s_word = sc[ib / 2];
            let shift0 = 6 * (ib % 2);
            let shift1 = 6 * (ib % 2) + 3;
            let dl1 = d * (2.0 * ((s_word >> shift0) & 0x7) as f32 + 1.0);
            let dl2 = d * (2.0 * ((s_word >> shift1) & 0x7) as f32 + 1.0);
            // Per-16-weight deltas. qh[ib*2 + 0] holds bits for the
            // first half; qh[ib*2 + 1] for the second.
            let qh0 = qh[ib * 2];
            let qh1 = qh[ib * 2 + 1];
            let delta1 = if qh0 & 0x08 != 0 {
                -1.0 - IQ1S_DELTA
            } else {
                -1.0 + IQ1S_DELTA
            };
            let delta2 = if qh0 & 0x80 != 0 {
                -1.0 - IQ1S_DELTA
            } else {
                -1.0 + IQ1S_DELTA
            };
            let delta3 = if qh1 & 0x08 != 0 {
                -1.0 - IQ1S_DELTA
            } else {
                -1.0 + IQ1S_DELTA
            };
            let delta4 = if qh1 & 0x80 != 0 {
                -1.0 - IQ1S_DELTA
            } else {
                -1.0 + IQ1S_DELTA
            };
            // 32 weights from 4 grid lookups. The high 3 bits of each
            // index come from `qh` at non-overlapping 3-bit positions
            // (matching ggml: bits 0..2 of qh0 → l=0, bits 4..6 of qh0
            // → l=1, bits 0..2 of qh1 → l=2, bits 4..6 of qh1 → l=3).
            // Weights 0..15 use dl1+delta1/2; weights 16..31 use dl2+delta3/4.
            let qs_chunk = &qs[ib * 4..ib * 4 + 4];
            let idx_l = [
                qs_chunk[0] as usize | (((qh0 & 0x07) as usize) << 8),
                qs_chunk[1] as usize | ((((qh0 >> 4) & 0x07) as usize) << 8),
                qs_chunk[2] as usize | (((qh1 & 0x07) as usize) << 8),
                qs_chunk[3] as usize | ((((qh1 >> 4) & 0x07) as usize) << 8),
            ];
            // l=0,1 → first 16 weights with dl1; l=0 uses delta1, l=1 uses delta2.
            // l=2,3 → next 16 weights with dl2; l=2 uses delta3, l=3 uses delta4.
            for (l, (dl, delta)) in [
                (dl1, delta1),
                (dl1, delta2),
                (dl2, delta3),
                (dl2, delta4),
            ]
            .into_iter()
            .enumerate()
            {
                let grid = IQ1S_GRID[idx_l[l]].to_le_bytes();
                for j in 0..8 {
                    let g = grid[j] as i8 as f32;
                    out[out_off + 8 * l + j] = dl * (g + delta);
                }
            }
            out_off += 32;
        }
    }
}

/// IQ3_S super-block: 110 bytes per 256 weights.
/// Layout (matching ggml's `block_iq3_s`, `IQ3S_N_SCALE = QK_K/64 = 4`):
///   { d: f16, qs: [u8; 64], qh: [u8; 8], signs: [u8; 32], scales: [u8; 4] }
///
/// Each weight is a 9-bit index (`qs[2l]` + 1 bit from `qh`) into the
/// 512-entry [`IQ3S_GRID`], yielding 4 i8 grid coordinates. The
/// per-element sign comes from the matching bit in `signs` (selector
/// from [`KMASK_IQ2XS`]). Scales: 4 bytes pack 8 sub-block 4-bit
/// values (2 per byte); effective sub-scale is `1 + 2*x` (odd numbers
/// in 1..=31). Two consecutive 32-element sub-blocks share one
/// `scales` byte (low nibble → first, high nibble → second).
pub fn dequant_iq3_s(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 110;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK_K);

    // The grid is u32 packing four i8 coordinates per entry. Reading
    // through `cast_slice::<u32, u8>` gives us little-endian bytes
    // where `grid_bytes[entry*4 + j]` is the j-th i8 coordinate (as
    // unsigned u8 — the values in the grid are all small positive
    // odd numbers like 1, 3, 5, ..., 15, so the unsigned interpretation
    // is fine; sign comes separately from `signs`).
    let grid_bytes: &[u8] = bytemuck::cast_slice(&IQ3S_GRID);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        // Layout offsets: d(2) + qs(64) + qh(8) + signs(32) + scales(4) = 110.
        let qs = &bytes[off + 2..off + 2 + 64];
        let qh = &bytes[off + 2 + 64..off + 2 + 64 + 8];
        let signs = &bytes[off + 2 + 64 + 8..off + 2 + 64 + 8 + 32];
        let scales = &bytes[off + 2 + 64 + 8 + 32..off + 2 + 64 + 8 + 32 + 4];
        let dst_off = b * QK_K;

        // Walk a `qs` cursor + `signs` cursor that advance per
        // sub-block (8 qs bytes + 4 signs bytes consumed per
        // sub-block). 8 sub-blocks total, processed in pairs that
        // share a scales byte and adjacent qh bytes.
        let mut qs_cur = 0usize; // index into qs (0..64)
        let mut signs_cur = 0usize; // index into signs (0..32)

        for pair in 0..4 {
            let ib32 = pair * 2;
            let scale_byte = scales[pair];
            let db1 = d * (1.0 + 2.0 * ((scale_byte & 0x0F) as f32));
            let db2 = d * (1.0 + 2.0 * ((scale_byte >> 4) as f32));

            // First sub-block of the pair.
            let mut y = dst_off + ib32 * 32;
            let qh_byte = qh[ib32];
            for l in 0..4 {
                let g1_idx = qs[qs_cur + 2 * l] as usize
                    | (((qh_byte as usize) << (8 - 2 * l)) & 0x100);
                let g2_idx = qs[qs_cur + 2 * l + 1] as usize
                    | (((qh_byte as usize) << (7 - 2 * l)) & 0x100);
                let g1 = &grid_bytes[g1_idx * 4..g1_idx * 4 + 4];
                let g2 = &grid_bytes[g2_idx * 4..g2_idx * 4 + 4];
                let sign_byte = signs[signs_cur + l];
                for j in 0..4 {
                    let s_lo = if sign_byte & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    let s_hi = if sign_byte & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                    out[y + j] = db1 * (g1[j] as f32) * s_lo;
                    out[y + j + 4] = db1 * (g2[j] as f32) * s_hi;
                }
                y += 8;
            }
            qs_cur += 8;
            signs_cur += 4;

            // Second sub-block of the pair.
            let qh_byte2 = qh[ib32 + 1];
            for l in 0..4 {
                let g1_idx = qs[qs_cur + 2 * l] as usize
                    | (((qh_byte2 as usize) << (8 - 2 * l)) & 0x100);
                let g2_idx = qs[qs_cur + 2 * l + 1] as usize
                    | (((qh_byte2 as usize) << (7 - 2 * l)) & 0x100);
                let g1 = &grid_bytes[g1_idx * 4..g1_idx * 4 + 4];
                let g2 = &grid_bytes[g2_idx * 4..g2_idx * 4 + 4];
                let sign_byte = signs[signs_cur + l];
                for j in 0..4 {
                    let s_lo = if sign_byte & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    let s_hi = if sign_byte & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                    out[y + j] = db2 * (g1[j] as f32) * s_lo;
                    out[y + j + 4] = db2 * (g2[j] as f32) * s_hi;
                }
                y += 8;
            }
            qs_cur += 8;
            signs_cur += 4;
        }
    }
}

/// IQ2_XXS / IQ2_XS sign-index lookup. The codebook here is a small
/// learned table that maps a 7-bit "sign index" (extracted from the
/// `qs` payload) onto an 8-bit pattern; bit `j` of the result tells
/// whether the j-th grid coordinate in that 8-byte chunk is negated.
/// `KMASK_IQ2XS` selects bit `j` out of the byte. Transcribed verbatim
/// from `ksigns_iq2xs` in ggml-common.h (MIT).
pub const KSIGNS_IQ2XS: [u8; 128] = [
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
];

/// IQ2_XXS codebook: 256 entries × 8 packed positive-i8 grid coordinates.
/// Stored as `u64` so each entry yields exactly 8 bytes when cast via
/// [`bytemuck::cast_slice`]; on a little-endian host that matches the
/// `(uint8_t*)(iq2xxs_grid + idx)` indexing in upstream ggml. Transcribed
/// verbatim from `iq2xxs_grid` in ggml-common.h (MIT).
pub const IQ2XXS_GRID: [u64; 256] = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
];

/// IQ3_XXS codebook: 256 entries × 4 packed positive-i8 grid coordinates
/// (one u32 per entry). Each grid coordinate is a small odd integer
/// in the set `{1, 3, 5, 7, 9, 11, 13, 15}` — sign is applied
/// separately via [`KSIGNS_IQ2XS`]. Transcribed verbatim from
/// `iq3xxs_grid` in ggml-common.h (MIT).
pub const IQ3XXS_GRID: [u32; 256] = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c34041c, 0x1c3e1404, 0x1c3e14ac, 0x1c3e434e, 0x24040424, 0x24040c3e, 0x24041c2c, 0x24041c3e,
    0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404, 0x24143404, 0x24143434,
    0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c, 0x242c241c, 0x242c3e04,
    0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c, 0x2c043e04, 0x2c0c0404,
    0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14, 0x2c1c0414, 0x2c1c2c1c,
    0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c, 0x2c342c04, 0x2c3e1424,
    0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c, 0x340c340c, 0x34140c3e,
    0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14, 0x34341c1c, 0x343e041c,
    0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14, 0x3e042c14, 0x3e0c1434,
    0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c, 0x3e1c1c1c, 0x3e1c3404,
    0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04, 0x3e3e2c04, 0x3e3e2c1c,
];

/// IQ2_XS codebook: 512 entries × 8 packed positive-i8 grid coordinates.
/// Same byte interpretation as [`IQ2XXS_GRID`] — see that doc comment.
/// Transcribed verbatim from `iq2xs_grid` in ggml-common.h (MIT).
pub const IQ2XS_GRID: [u64; 512] = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
];

/// IQ2_XXS super-block: 66 bytes per 256 weights. 2.0625 bpw.
/// Layout (matching ggml's `block_iq2_xxs`):
///   { d: f16, qs: [u16; 32] }
///
/// Sub-block decode (8 sub-blocks of 32 weights each): read two
/// little-endian u32s from `qs[4*ib32 ..]`. `aux32[0]` is four packed
/// 8-bit grid indices into [`IQ2XXS_GRID`] (each entry yields 8 grid
/// points). `aux32[1]` packs four 7-bit sign-indices into
/// [`KSIGNS_IQ2XS`] (bits 0–6, 7–13, 14–20, 21–27) plus a 4-bit scale
/// in bits 28–31. Sub-block float scale: `db = d * (0.5 + scale) * 0.25`.
pub fn dequant_iq2_xxs(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 66;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK_K);

    let grid_bytes: &[u8] = bytemuck::cast_slice(&IQ2XXS_GRID);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let qs = &bytes[off + 2..off + 2 + 64];

        let dst_base = b * QK_K;
        for ib32 in 0..8 {
            let aux0 = u32::from_le_bytes([
                qs[8 * ib32],
                qs[8 * ib32 + 1],
                qs[8 * ib32 + 2],
                qs[8 * ib32 + 3],
            ]);
            let aux1 = u32::from_le_bytes([
                qs[8 * ib32 + 4],
                qs[8 * ib32 + 5],
                qs[8 * ib32 + 6],
                qs[8 * ib32 + 7],
            ]);
            let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
            let aux8 = aux0.to_le_bytes();
            for l in 0..4 {
                let grid_idx = aux8[l] as usize;
                let grid = &grid_bytes[grid_idx * 8..grid_idx * 8 + 8];
                let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                let dst = dst_base + ib32 * 32 + l * 8;
                for j in 0..8 {
                    let s = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    out[dst + j] = db * (grid[j] as f32) * s;
                }
            }
        }
    }
}

/// IQ2_XS super-block: 74 bytes per 256 weights. 2.3125 bpw.
/// Layout (matching ggml's `block_iq2_xs`):
///   { d: f16, qs: [u16; 32], scales: [u8; 8] }
///
/// Sub-block decode (8 sub-blocks of 32 weights, 4 u16 per sub-block):
/// each u16 carries `idx = low 9 bits` (index into [`IQ2XS_GRID`]) and
/// `sign_idx = high 7 bits` (index into [`KSIGNS_IQ2XS`]). Each
/// `scales[ib32]` byte holds two 4-bit sub-scales: the low nibble drives
/// the first two grid lookups (l=0,1, weights 0..15), the high nibble
/// the next two (l=2,3, weights 16..31). Float scale per nibble:
/// `db = d * (0.5 + nibble) * 0.25`.
pub fn dequant_iq2_xs(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 74;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK_K);

    let grid_bytes: &[u8] = bytemuck::cast_slice(&IQ2XS_GRID);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let qs = &bytes[off + 2..off + 2 + 64];
        let scales = &bytes[off + 2 + 64..off + 2 + 64 + 8];

        let dst_base = b * QK_K;
        for ib32 in 0..8 {
            let scale_byte = scales[ib32];
            let db_lo = d * (0.5 + (scale_byte & 0x0F) as f32) * 0.25;
            let db_hi = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
            for l in 0..4 {
                let q = u16::from_le_bytes([qs[8 * ib32 + 2 * l], qs[8 * ib32 + 2 * l + 1]]);
                let grid_idx = (q & 511) as usize;
                let sign_idx = (q >> 9) as usize;
                let grid = &grid_bytes[grid_idx * 8..grid_idx * 8 + 8];
                let signs = KSIGNS_IQ2XS[sign_idx];
                let db = if l < 2 { db_lo } else { db_hi };
                let dst = dst_base + ib32 * 32 + l * 8;
                for j in 0..8 {
                    let s = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    out[dst + j] = db * (grid[j] as f32) * s;
                }
            }
        }
    }
}

/// IQ2_S codebook: 1024 entries × 8 packed positive-i8 grid coordinates.
/// Stored as `u64` so `bytemuck::cast_slice` yields `&[u8]` directly on
/// LE hosts (matching `(uint8_t*)(iq2s_grid + idx)` indexing in upstream
/// ggml). The 10-bit index space matches the qs[l] (low 8 bits) + qh
/// (high 2 bits) layout in [`dequant_iq2_s`]. Transcribed verbatim from
/// `iq2s_grid` in ggml-common.h (MIT).
pub const IQ2S_GRID: [u64; 1024] = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919,
    0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908,
    0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908,
    0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19,
    0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919,
    0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908,
    0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b,
    0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b,
    0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908,
    0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b,
    0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b,
    0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808,
    0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b,
    0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08,
    0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808,
    0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819,
    0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908,
    0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b,
    0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808,
    0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919,
    0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808,
    0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b,
    0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b,
    0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919,
    0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919,
    0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819,
    0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908,
    0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908,
    0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908,
    0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908,
    0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819,
    0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08,
    0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819,
    0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819,
    0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808,
    0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08,
    0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908,
    0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819,
    0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808,
    0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908,
    0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819,
    0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19,
    0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919,
    0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808,
    0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908,
    0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08,
    0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908,
    0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808,
    0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908,
    0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819,
    0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808,
    0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808,
    0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b,
    0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808,
    0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808,
    0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19,
    0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08,
    0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908,
    0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b,
    0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808,
    0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908,
    0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08,
    0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919,
    0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b,
    0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b,
    0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08,
    0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808,
    0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919,
    0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808,
    0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819,
    0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808,
    0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919,
    0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b,
    0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808,
    0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b,
    0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b,
    0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b,
    0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08,
    0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b,
    0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08,
    0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919,
    0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19,
    0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919,
    0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819,
    0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919,
    0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808,
    0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919,
    0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b,
    0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919,
    0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08,
    0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808,
    0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908,
    0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808,
    0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908,
    0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808,
    0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908,
    0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808,
    0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808,
    0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b,
    0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08,
    0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919,
    0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908,
    0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808,
    0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908,
    0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808,
    0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808,
    0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908,
    0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08,
    0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b,
    0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819,
    0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819,
    0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808,
    0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908,
    0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908,
    0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819,
    0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b,
    0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819,
    0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b,
    0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819,
    0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819,
    0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
];

/// IQ3_XXS super-block: 98 bytes per 256 weights. 3.0625 bpw.
/// Layout (matching ggml's `block_iq3_xxs`):
///   { d: f16, qs: [u8; 96] }
///
/// `qs` splits into two regions: `qs[0..64]` (`QK_K/4 = 64` bytes)
/// are grid indices (256 indices × 8 bits each, indexing the
/// 256-entry [`IQ3XXS_GRID`] codebook of 4 i8 coordinates each), and
/// `qs[64..96]` (`QK_K/8 = 32` bytes) are 8 little-endian u32 words
/// — one per 32-weight sub-block — packing the sub-block scale
/// (high 4 bits, range 0..=15) and four 7-bit sign-table indices
/// (low 28 bits, indexing [`KSIGNS_IQ2XS`]).
///
/// Each sub-block consumes 8 grid indices and produces 32 weights.
/// Per sub-block: `db = d * (0.5 + scale) * 0.5`. Per `l ∈ 0..4`:
/// two grid lookups (indices `qs[2*l]` and `qs[2*l+1]`) contribute
/// 4 weights each, with signs from `KSIGNS_IQ2XS[(aux32 >> 7l) & 127]`
/// applied via [`KMASK_IQ2XS`].
pub fn dequant_iq3_xxs(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 98;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK_K);

    // The grid is u32 packing four i8 coordinates per entry. Reading
    // through `cast_slice::<u32, u8>` gives little-endian bytes
    // where `grid_bytes[entry*4 + j]` is the j-th i8 coord (small
    // positive odd numbers — sign comes separately from KSIGNS_IQ2XS).
    let grid_bytes: &[u8] = bytemuck::cast_slice(&IQ3XXS_GRID);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        // qs region: 96 bytes split into grid_indices[0..64] +
        // scales_and_signs[64..96] (eight u32 words).
        let qs_grid = &bytes[off + 2..off + 2 + 64];
        let qs_sas = &bytes[off + 2 + 64..off + 2 + 96];

        let dst_base = b * QK_K;
        for ib32 in 0..8 {
            let aux32 = u32::from_le_bytes([
                qs_sas[4 * ib32],
                qs_sas[4 * ib32 + 1],
                qs_sas[4 * ib32 + 2],
                qs_sas[4 * ib32 + 3],
            ]);
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
            let qs_off = 8 * ib32;
            for l in 0..4 {
                let grid1_idx = qs_grid[qs_off + 2 * l] as usize;
                let grid2_idx = qs_grid[qs_off + 2 * l + 1] as usize;
                let grid1 = &grid_bytes[grid1_idx * 4..grid1_idx * 4 + 4];
                let grid2 = &grid_bytes[grid2_idx * 4..grid2_idx * 4 + 4];
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                let dst = dst_base + ib32 * 32 + l * 8;
                for j in 0..4 {
                    let s_lo = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    let s_hi = if signs & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                    out[dst + j] = db * (grid1[j] as f32) * s_lo;
                    out[dst + j + 4] = db * (grid2[j] as f32) * s_hi;
                }
            }
        }
    }
}

/// IQ2_S super-block: 82 bytes per 256 weights. 2.5625 bpw.
/// Layout (matching ggml's `block_iq2_s`):
///   { d: f16, qs: [u8; 64], qh: [u8; 8], scales: [u8; 8] }
///
/// `qs` is split into two contiguous halves: the first 32 bytes hold
/// the low-8-bit grid indices (one byte per 4-weight grid lookup, 4
/// indices per ib32 sub-block), the next 32 bytes are sign-index bytes
/// (also 4 per ib32 sub-block). `qh[ib32]` supplies the high two bits
/// of the 10-bit grid index per `l in 0..4` — `qh[ib32] << (8 - 2*l)`
/// places bits `(2*l, 2*l+1)` of `qh[ib32]` at bit positions 8-9 of the
/// index. Scales: 8 bytes, two 4-bit nibbles per sub-block (low nibble
/// drives `l=0,1`, high nibble drives `l=2,3`); each nibble feeds
/// `db = d * (0.5 + nibble) * 0.25`.
pub fn dequant_iq2_s(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 82;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK_K);

    let grid_bytes: &[u8] = bytemuck::cast_slice(&IQ2S_GRID);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        // qs region: 64 bytes split into qs_lo[0..32] (grid indices) +
        // signs[32..64] (sign-table indices).
        let qs_lo = &bytes[off + 2..off + 2 + 32];
        let signs = &bytes[off + 2 + 32..off + 2 + 64];
        let qh = &bytes[off + 2 + 64..off + 2 + 64 + 8];
        let scales = &bytes[off + 2 + 64 + 8..off + 2 + 64 + 8 + 8];

        let dst_base = b * QK_K;
        for ib32 in 0..8 {
            let scale_byte = scales[ib32];
            let db_lo = d * (0.5 + (scale_byte & 0x0F) as f32) * 0.25;
            let db_hi = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
            let qs_off = ib32 * 4;
            let qh_byte = qh[ib32];
            for l in 0..4 {
                let high_bits = ((qh_byte as usize) << (8 - 2 * l)) & 0x300;
                let grid_idx = (qs_lo[qs_off + l] as usize) | high_bits;
                let sign_byte = signs[qs_off + l];
                let grid = &grid_bytes[grid_idx * 8..grid_idx * 8 + 8];
                let db = if l < 2 { db_lo } else { db_hi };
                let dst = dst_base + ib32 * 32 + l * 8;
                for j in 0..8 {
                    let s = if sign_byte & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    out[dst + j] = db * (grid[j] as f32) * s;
                }
            }
        }
    }
}

/// IQ4_XS super-block: 136 bytes per 256 weights.
/// Layout (matching ggml's `block_iq4_xs`):
///   { d: f16, scales_h: u16, scales_l: [u8; QK_K/64], qs: [u8; QK_K/2] }
/// where `scales_h` and `scales_l` together pack eight 6-bit signed
/// sub-block scales (one per 32-weight sub-block).
///
/// Each 4-bit value in `qs` indexes [`KVALUES_IQ4NL`] (a non-linear
/// codebook biased toward zero). The final f32 is
/// `d * sub_scale * kvalues_iq4nl[idx]`.
pub fn dequant_iq4_xs(bytes: &[u8], out: &mut [f32]) {
    const BLOCK_BYTES: usize = 136;
    let n_blocks = bytes.len() / BLOCK_BYTES;
    debug_assert_eq!(bytes.len(), n_blocks * BLOCK_BYTES);
    debug_assert_eq!(out.len(), n_blocks * QK_K);

    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let scales_h = u16::from_le_bytes([bytes[off + 2], bytes[off + 3]]);
        let scales_l = &bytes[off + 4..off + 8]; // 4 bytes, 8 nibbles
        let qs = &bytes[off + 8..off + 8 + 128];

        // 8 sub-blocks of 32 weights each.
        for ib in 0..8 {
            // Reconstruct the 6-bit signed scale for this sub-block:
            //   low 4 bits from scales_l (nibble), high 2 bits from
            //   scales_h (bit-pair). Then subtract 32 for the signed
            //   bias.
            let lo_nibble = if ib % 2 == 0 {
                scales_l[ib / 2] & 0x0F
            } else {
                scales_l[ib / 2] >> 4
            };
            let hi_bits = ((scales_h >> (2 * ib)) & 0x03) as u8;
            let ls_raw = (lo_nibble | (hi_bits << 4)) as i8;
            let ls = ls_raw - 32; // signed range -32..+31
            let sub_d = d * (ls as f32);

            // 32 outputs from 16 q-bytes (each q-byte has 2 4-bit values).
            let q_off = ib * 16;
            let dst_off = b * QK_K + ib * 32;
            for j in 0..16 {
                let q = qs[q_off + j];
                let lo = (q & 0x0F) as usize;
                let hi = (q >> 4) as usize;
                out[dst_off + j] = sub_d * (KVALUES_IQ4NL[lo] as f32);
                out[dst_off + 16 + j] = sub_d * (KVALUES_IQ4NL[hi] as f32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct one Q4_K super-block from explicit (sc, mn) pairs.
    /// Returns the 144 bytes plus the decoded sc/mn arrays so tests can
    /// assert outputs without re-deriving the bit-packing.
    fn make_q4_k_block(
        d_f32: f32,
        dmin_f32: f32,
        sc: [u8; 8],
        mn: [u8; 8],
        qs: [u8; 128],
    ) -> Vec<u8> {
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&f16::from_f32(d_f32).to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(dmin_f32).to_le_bytes());
        // Inverse of get_scale_min_k4:
        //   For j<4: scales[j]  = sc[j] | ((sc[j+4] >> 4) << 6)
        //            scales[j+4] = mn[j] | ((mn[j+4] >> 4) << 6)
        //   For j>=4: scales[j+4] = (sc[j] & 0xF) | ((mn[j] & 0xF) << 4)
        for j in 0..4 {
            assert!(sc[j] < 64);
            assert!(mn[j] < 64);
            block[4 + j] = sc[j] | (((sc[j + 4] >> 4) & 0x3) << 6);
            block[4 + j + 4] = mn[j] | (((mn[j + 4] >> 4) & 0x3) << 6);
        }
        for j in 4..8 {
            block[4 + j + 4] = (sc[j] & 0x0F) | ((mn[j] & 0x0F) << 4);
        }
        block[16..16 + 128].copy_from_slice(&qs);
        block
    }

    #[test]
    fn iq1_s_decodes_a_simple_block_with_known_grid_entry() {
        // Construct one IQ1_S block where every ib32 sub-block points
        // at grid index 0 (which is `0xFFFFFFFFFFFFFFFF` —
        // eight `0xFF` bytes interpreted as i8 = `-1`).
        // - d = 1.0
        // - qh[ib32] = 0 means: scale = 1, delta = -1 + IQ1S_DELTA = -0.875.
        // - dl = 1 * 1 = 1; per-weight value = 1 * (-1 + (-0.875)) = -1.875.
        let mut block = [0u8; 50];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        // qs stays all 0 — combined with qh's low bits = 0, every
        // index is 0.
        // qh: 8 u16 all zero → scale=1 (from 2*0+1), delta=-1+0.125 = -0.875.
        let mut out = vec![0f32; 256];
        dequant_iq1_s(&block, &mut out);
        // Reconstruction: dl * (grid[j] + delta) where grid[j]=-1
        // (from 0xFF) and delta = -1 + IQ1S_DELTA = -0.875.
        // → 1 * (-1 + -0.875) = -1.875.
        let expected = -1.0 + (-1.0 + IQ1S_DELTA);
        for (i, v) in out.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-5,
                "out[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn iq1_s_sign_delta_flip_at_high_bit_of_qh() {
        // Same as above but with qh[0] = 0x8000 → delta = -1 - IQ1S_DELTA
        // for the first sub-block. Other sub-blocks keep the +delta side.
        let mut block = [0u8; 50];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        block[34] = 0x00;
        block[35] = 0x80; // qh[0] = 0x8000
        let mut out = vec![0f32; 256];
        dequant_iq1_s(&block, &mut out);
        // grid[j] = -1 throughout. With qh[0]=0x8000:
        //   delta = -1 - IQ1S_DELTA = -1.125 → out[0..32] = -1 + -1.125 = -2.125.
        // With qh[1..]=0: delta = -0.875 → out[32..] = -1 + -0.875 = -1.875.
        let neg = -1.0 + (-1.0 - IQ1S_DELTA);
        let pos = -1.0 + (-1.0 + IQ1S_DELTA);
        for i in 0..32 {
            assert!(
                (out[i] - neg).abs() < 1e-5,
                "first ib32 should use -delta, got out[{i}] = {}",
                out[i]
            );
        }
        for i in 32..64 {
            assert!(
                (out[i] - pos).abs() < 1e-5,
                "second ib32 should use +delta, got out[{i}] = {}",
                out[i]
            );
        }
    }

    #[test]
    fn iq1_m_sub_block_scale_unpacks_from_packed_bytes() {
        // Construct an IQ1_M block where all scales are at their
        // minimum and verify the dequant doesn't blow up. The
        // scale-from-packed reconstruction is what differentiates
        // IQ1_M from IQ1_S; this pins that the d-extraction reads
        // bits 12-15 of each sc word.
        //
        // scales[0..2] = sc[0] = 0xF000 → contributes 0xF to d_bits
        //                              and scale-pair (0, 0)
        // scales[2..8] = 0 → contributes nothing to d_bits.
        // Result: d_bits = 0xF.
        let mut block = [0u8; 56];
        block[48] = 0x00;
        block[49] = 0xF0; // sc[0] high nibble = 0xF
        let mut out = vec![0f32; 256];
        dequant_iq1_m(&block, &mut out);
        let d = half::f16::from_bits(0x000Fu16).to_f32();
        // With all scale bits = 0 → dl1 = dl2 = d. All qs/qh = 0 →
        // grid index 0 → grid bytes = 0xFF (= -1), delta = -1 + 0.125
        // (qh bits 0x08 / 0x80 not set).
        let expected = d * ((-1.0_f32) + (-1.0 + IQ1S_DELTA));
        // Just spot-check the first weight stays finite and matches.
        assert!(
            (out[0] - expected).abs() < 1e-5,
            "out[0] = {}, expected {expected}",
            out[0]
        );
    }

    #[test]
    fn q8_k_unit_scale_round_trips_all_signed_byte_values() {
        // d=1: each output equals the signed-i8 interpretation of its
        // quant byte. Walk the full -128..127 range across two
        // super-blocks (512 weights).
        let mut block = vec![0u8; 292 * 2];
        for blk in 0..2 {
            let off = blk * 292;
            // d = 1.0 (f32, four bytes).
            block[off..off + 4].copy_from_slice(&1.0f32.to_le_bytes());
            // qs[0..256]: byte j gets a known signed value.
            for j in 0..256 {
                // For block 0: j-128 maps full -128..127 once.
                // For block 1: same distribution offset by +/- 32 to
                // exercise both halves.
                let q: i8 = if blk == 0 {
                    (j as i32 - 128) as i8
                } else {
                    ((j as i32 + 32) - 128) as i8
                };
                block[off + 4 + j] = q as u8;
            }
            // bsums[16] at off+260..292 — left zero, decoder ignores.
        }
        let mut out = vec![0f32; 512];
        dequant_q8_k(&block, &mut out);
        for j in 0..256 {
            assert_eq!(out[j], (j as i32 - 128) as f32, "block 0 elem {j}");
            assert_eq!(
                out[256 + j],
                ((j as i32 + 32) - 128) as i8 as f32,
                "block 1 elem {j}"
            );
        }
    }

    #[test]
    fn q8_k_d_zero_yields_all_zeros() {
        let mut block = vec![0u8; 292];
        // d=0; qs filled with non-zero garbage.
        for j in 0..256 {
            block[4 + j] = ((j as u32 * 7) ^ 0x5A) as u8;
        }
        let mut out = vec![0f32; 256];
        dequant_q8_k(&block, &mut out);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn q2_k_unit_scale_zero_min_full_range() {
        // d=1, dmin=0: weights map straight through their 2-bit value
        // (0..3) scaled by the per-sub-block scale nibble. Set every
        // scale nibble's low to 1 and high (min) to 0 so each weight is
        // just its raw 2-bit value.
        let mut block = [0u8; 84];
        // scales[0..16]: low nibble=1 (scale), high nibble=0 (min).
        for s in 0..16 {
            block[s] = 0x01;
        }
        // qs[0..64]: byte j packs 4 weights at shifts 0,2,4,6. Put
        // distinct values in each 2-bit field so we can verify ordering.
        // For simplicity: qs[0] = 0b11_10_01_00 = 0xE4 → 4 values: 0,1,2,3
        // at shifts 0,2,4,6 respectively.
        block[16] = 0b1110_0100; // weight 0/16/32/48 = 0/1/2/3
        // d at offset 80..82, dmin at 82..84.
        block[80..82].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        block[82..84].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());

        let mut out = vec![0f32; 256];
        dequant_q2_k(&block, &mut out);
        // Each shift step contributes 32 outputs (sub-block A's 16 +
        // sub-block B's 16). qs[0]'s value at shift S lands at the
        // first output of that shift's sub-block A — output base
        // 32*(shift/2).
        // From qs[0] = 0b11_10_01_00:
        //   shift 0: q=0 → out[0]   = 1*0 - 0 = 0
        //   shift 2: q=1 → out[32]  = 1*1 - 0 = 1
        //   shift 4: q=2 → out[64]  = 1*2 - 0 = 2
        //   shift 6: q=3 → out[96]  = 1*3 - 0 = 3
        assert_eq!(out[0], 0.0, "shift 0");
        assert_eq!(out[32], 1.0, "shift 2");
        assert_eq!(out[64], 2.0, "shift 4");
        assert_eq!(out[96], 3.0, "shift 6");
    }

    #[test]
    fn q2_k_min_offset_subtracts_from_all_weights() {
        // d=2, dmin=4: scale_nibble=1, min_nibble=1 ⇒ dl=2, ml=4. With
        // q=0 a weight is dl*0 - ml = -4. With q=3 it's dl*3 - ml = 2.
        // Use two different qs bytes at the same shift (shift 0) so
        // both branches hit the same sub-block A scale (scales[0]).
        let mut block = [0u8; 84];
        block[0] = 0x11; // scales[0]: low=1 (scale), high=1 (min)
        block[16] = 0b0000_0000; // qs[0] low 2 bits = 0  → q=0
        block[17] = 0b0000_0011; // qs[1] low 2 bits = 3  → q=3
        block[80..82].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
        block[82..84].copy_from_slice(&half::f16::from_f32(4.0).to_le_bytes());
        let mut out = vec![0f32; 256];
        dequant_q2_k(&block, &mut out);
        // Both outputs are in sub-block A (16 weights at shift 0)
        // using scales[0].
        assert_eq!(out[0], -4.0, "q=0 → dl*0 - ml = -4");
        assert_eq!(out[1], 2.0, "q=3 → dl*3 - ml = 2");
    }

    #[test]
    fn q4_0_round_trip_unit_scale_full_range() {
        // Build one Q4_0 block where qs[j] = 0x{lo}{hi} with lo and hi
        // sweeping the 16-value 4-bit alphabet. With d = 1.0 the output
        // should be exactly the i8 value of the nibble minus 8.
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        // qs[j]: low nibble = j, high nibble = 15-j → outputs j-8 (low),
        // (15-j)-8 (high). Lets us validate both halves.
        for j in 0..16u8 {
            block[2 + j as usize] = (j & 0x0F) | ((15 - j) << 4);
        }
        let mut out = vec![0f32; 32];
        dequant_q4_0(&block, &mut out);
        for j in 0..16 {
            assert_eq!(out[j], (j as f32) - 8.0, "low nibble out[{j}]");
            assert_eq!(out[j + 16], (15 - j) as f32 - 8.0, "high nibble out[{}]", j + 16);
        }
    }

    #[test]
    fn bf16_lossless_from_known_f32_values() {
        // BF16 is the top 16 bits of an IEEE-754 f32. Picking values
        // whose mantissa fits in BF16's 7 mantissa bits means the
        // round-trip is exact. For values with extra mantissa bits, the
        // bottom 16 bits get truncated (we construct BF16 here by
        // truncation, same as the synth-builder side).
        for v in &[0.0f32, 1.0, -1.0, 0.5, -0.25, 2.0, 4.0, -8.0, 100.0] {
            let bf16_bits = (v.to_bits() >> 16) as u16;
            let bytes = bf16_bits.to_le_bytes();
            let mut out = [0f32; 1];
            dequant_bf16(&bytes, &mut out);
            assert_eq!(out[0], *v, "round-trip mismatch for {v}");
        }
    }

    #[test]
    fn bf16_preserves_sign_of_zero() {
        // +0 and -0 have different bit patterns; BF16 should preserve
        // the sign bit just like F16 does.
        let pos_zero_bits = (0.0f32.to_bits() >> 16) as u16;
        let neg_zero_bits = ((-0.0f32).to_bits() >> 16) as u16;
        let bytes: Vec<u8> = pos_zero_bits
            .to_le_bytes()
            .iter()
            .chain(neg_zero_bits.to_le_bytes().iter())
            .copied()
            .collect();
        let mut out = [0f32; 2];
        dequant_bf16(&bytes, &mut out);
        assert!(out[0].is_sign_positive(), "+0 lost sign");
        assert!(out[1].is_sign_negative(), "-0 lost sign");
    }

    #[test]
    fn bf16_handles_subnormal_and_inf() {
        // F32 +inf = 0x7F80_0000; top 16 bits = 0x7F80 → BF16 +inf.
        let inf_bits = (f32::INFINITY.to_bits() >> 16) as u16;
        let mut out = [0f32; 1];
        dequant_bf16(&inf_bits.to_le_bytes(), &mut out);
        assert!(out[0].is_infinite() && out[0] > 0.0);
    }

    #[test]
    fn q4_1_unit_scale_zero_min_full_range() {
        // d=1, m=0: nibbles map straight through (no -8 bias since Q4_1
        // is unsigned), so out[j] = nibble in [0, 15].
        let mut block = [0u8; 20];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
        for j in 0..16u8 {
            block[4 + j as usize] = (j & 0x0F) | ((15 - j) << 4);
        }
        let mut out = vec![0f32; 32];
        dequant_q4_1(&block, &mut out);
        for j in 0..16 {
            assert_eq!(out[j], j as f32, "low nibble out[{j}]");
            assert_eq!(out[j + 16], (15 - j) as f32, "high nibble out[{}]", j + 16);
        }
    }

    #[test]
    fn q4_1_min_shifts_all_values() {
        // d=1, m=3: every output is its nibble plus 3.
        let mut block = [0u8; 20];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(3.0).to_le_bytes());
        block[4] = 0x12; // low=2 → out[0]=5, high=1 → out[16]=4
        let mut out = vec![0f32; 32];
        dequant_q4_1(&block, &mut out);
        assert_eq!(out[0], 5.0);
        assert_eq!(out[16], 4.0);
    }

    #[test]
    fn q5_1_unit_scale_zero_min_full_range() {
        // d=1, m=0, all qh bits set: every output is its 5-bit value in
        // [16, 31] depending on the bit pattern in qs. Walk a small case.
        let mut block = [0u8; 24];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
        // qh = all 1s in the low 32 bits.
        block[4..8].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        // qs: each byte's low nibble = j (in [0..15]), high nibble = 15-j.
        for j in 0..16u8 {
            block[8 + j as usize] = (j & 0x0F) | ((15 - j) << 4);
        }
        let mut out = vec![0f32; 32];
        dequant_q5_1(&block, &mut out);
        for j in 0..16 {
            // With all qh bits set: out[j] = (j | 16) - 0 = j + 16.
            //                      out[j+16] = ((15-j) | 16) - 0 = (15-j) + 16.
            assert_eq!(out[j], (j + 16) as f32, "low nibble out[{j}]");
            assert_eq!(
                out[j + 16],
                ((15 - j) + 16) as f32,
                "high nibble out[{}]",
                j + 16
            );
        }
    }

    #[test]
    fn q5_1_min_offset_applies_to_5bit_values() {
        // d=2, m=-1, qh=0, qs[0]=0x53 (low nibble=3, high nibble=5).
        // Expect: out[0] = 2*3 + (-1) = 5, out[16] = 2*5 + (-1) = 9.
        let mut block = [0u8; 24];
        block[0..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(-1.0).to_le_bytes());
        block[4..8].copy_from_slice(&0u32.to_le_bytes());
        block[8] = 0x53;
        let mut out = vec![0f32; 32];
        dequant_q5_1(&block, &mut out);
        assert_eq!(out[0], 5.0);
        assert_eq!(out[16], 9.0);
    }

    #[test]
    fn q4_0_d_zero_yields_all_zeros() {
        // d = 0 → entire block is zero regardless of qs bits.
        let mut block = [0u8; 18];
        // qs filled with non-zero garbage.
        block[2..18].copy_from_slice(&[0xAB; 16]);
        let mut out = vec![0.0; 32];
        dequant_q4_0(&block, &mut out);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn q4_0_negative_d_flips_sign() {
        // d = -2 → outputs negated and doubled relative to d=1.
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&half::f16::from_f32(-2.0).to_le_bytes());
        block[2] = 0x07; // low=7 → 7-8=-1, high=0 → 0-8=-8
        let mut out = vec![0.0; 32];
        dequant_q4_0(&block, &mut out);
        assert_eq!(out[0], -2.0 * -1.0, "low: -2 * (-1) = 2");
        assert_eq!(out[16], -2.0 * -8.0, "high: -2 * (-8) = 16");
    }

    #[test]
    fn q4_k_layout_regression() {
        // This test would have caught the bug fixed mid-2026:
        // ggml's Q4_K dequant uses groups of 64 outputs from 32 q bytes
        // (32 lows + 32 highs of the same 32 bytes), with different scales.
        // The previous bug used groups of 32 from 16 q bytes (interleaved
        // low/high within the same 16 bytes), giving wrong element ordering.
        let mut qs = [0u8; 128];
        // Put a low nibble of 2 and a high nibble of 1 in qs[0].
        // In the CORRECT layout:
        //   out[0]   = sc[0] * 2 (low nibble of q_chunk[0])
        //   out[32]  = sc[1] * 1 (HIGH nibble of q_chunk[0], DIFFERENT scale)
        // In the BUGGY layout:
        //   out[0]   = sc[0] * 2
        //   out[1]   = sc[0] * 1 (HIGH nibble of qs[0], SAME scale as out[0])
        //   out[32]  = 0
        qs[0] = (1 << 4) | 2;
        // Mark qs[32] as well so we can verify group 1 uses qs[32..64].
        qs[32] = (3 << 4) | 4;

        let sc = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mn = [0u8; 8];
        let block = make_q4_k_block(1.0, 0.0, sc, mn, qs);

        let mut out = vec![0.0f32; QK_K];
        dequant_q4_k(&block, &mut out);

        // Group 0 (outs 0..63) reads q_chunk[0..32]:
        //   - out[0..32]:  low nibbles of qs[0..32] * sc[0]=1
        //   - out[32..64]: HIGH nibbles of qs[0..32] * sc[1]=2
        assert_eq!(out[0], 2.0, "low nibble of qs[0], scale sc[0]=1");
        assert_eq!(out[1], 0.0, "qs[1] is zero");
        assert_eq!(out[32], 2.0, "HIGH nibble of qs[0] = 1, scale sc[1]=2");
        assert_eq!(out[33], 0.0);

        // Group 1 (outs 64..127) reads q_chunk[32..64]:
        //   - out[64..96]:  low nibbles of qs[32..64] * sc[2]=3
        //   - out[96..128]: HIGH nibbles of qs[32..64] * sc[3]=4
        assert_eq!(out[64], 12.0, "low nibble of qs[32]=4, scale sc[2]=3");
        assert_eq!(out[96], 12.0, "HIGH nibble of qs[32]=3, scale sc[3]=4");

        // Outs in groups 2 and 3 should all be zero (qs[64..128] all zero).
        for i in 128..256 {
            assert_eq!(out[i], 0.0, "out[{i}] should be 0");
        }
    }

    #[test]
    fn q5_k_layout_regression() {
        // Same layout invariant as Q4_K but with the extra 5th bit from qh.
        let mut qs = [0u8; 128];
        let mut qh = [0u8; 32];
        // qs[0] low=2, high=1; qh[0] bit 0 set (extends low to 18), bit 1 clear.
        qs[0] = (1 << 4) | 2;
        qh[0] = 0b0000_0001;

        let sc = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mn = [0u8; 8];

        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(0.0).to_le_bytes());
        for j in 0..4 {
            block[4 + j] = sc[j] | (((sc[j + 4] >> 4) & 0x3) << 6);
            block[4 + j + 4] = mn[j] | (((mn[j + 4] >> 4) & 0x3) << 6);
        }
        for j in 4..8 {
            block[4 + j + 4] = (sc[j] & 0x0F) | ((mn[j] & 0x0F) << 4);
        }
        block[16..16 + 32].copy_from_slice(&qh);
        block[48..48 + 128].copy_from_slice(&qs);

        let mut out = vec![0.0f32; QK_K];
        dequant_q5_k(&block, &mut out);

        // Group 0 uses qh bit (0) for the LOW nibble of qs[0..32], bit (1) for the HIGH.
        //   out[0] = sc[0] * (low_nibble + bit_0?16:0) = 1 * (2 + 16) = 18
        //   out[32] = sc[1] * (high_nibble + bit_1?16:0) = 2 * (1 + 0) = 2
        assert_eq!(out[0], 18.0, "qs[0] low + qh bit 0 set");
        assert_eq!(out[32], 2.0, "qs[0] high, qh bit 1 clear");
        // Other positions zero
        assert_eq!(out[1], 0.0);
        assert_eq!(out[33], 0.0);
        for i in 64..256 {
            assert_eq!(out[i], 0.0, "out[{i}] should be 0");
        }
    }

    #[test]
    fn q8_0_single_block_zero() {
        // d = 0, qs = [0;32]  ->  all zeros
        let mut bytes = [0u8; 34];
        let d = f16::from_f32(0.0).to_le_bytes();
        bytes[0] = d[0];
        bytes[1] = d[1];
        let mut out = vec![1.0f32; 32];
        dequant_q8_0(&bytes, &mut out);
        assert_eq!(out, vec![0.0; 32]);
    }

    #[test]
    fn q8_0_single_block_unit_scale() {
        // d = 1, qs[i] = i  ->  out[i] = i
        let mut bytes = [0u8; 34];
        let d = f16::from_f32(1.0).to_le_bytes();
        bytes[0] = d[0];
        bytes[1] = d[1];
        for i in 0..32 {
            bytes[2 + i] = i as u8;
        }
        let mut out = vec![-1.0f32; 32];
        dequant_q8_0(&bytes, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, i as f32, "mismatch at {i}");
        }
    }

    /// Build one IQ4_XS super-block from explicit components. Inputs:
    ///   - `d`: top-level f16 scale
    ///   - `ls_signed`: 8 sub-block scales, each in -32..=31
    ///   - `qs_nibbles`: 256 4-bit codebook indices (one per output)
    fn build_iq4_xs_block(d: f32, ls_signed: [i8; 8], qs_nibbles: [u8; 256]) -> [u8; 136] {
        let mut block = [0u8; 136];
        let db = half::f16::from_f32(d).to_le_bytes();
        block[0] = db[0];
        block[1] = db[1];
        // Convert signed (-32..31) to raw 6-bit (add 32, range 0..63).
        let ls_raw: [u8; 8] = ls_signed.map(|v| (v + 32) as u8);
        // Pack low 4 bits of each sub-scale into scales_l (4 bytes, two nibbles each).
        for ib in 0..8 {
            let lo = ls_raw[ib] & 0x0F;
            if ib % 2 == 0 {
                block[4 + ib / 2] |= lo;
            } else {
                block[4 + ib / 2] |= lo << 4;
            }
        }
        // Pack the high 2 bits into scales_h (u16 LE, 2 bits per sub-block).
        let mut scales_h: u16 = 0;
        for ib in 0..8 {
            let hi = ((ls_raw[ib] >> 4) & 0x03) as u16;
            scales_h |= hi << (2 * ib);
        }
        block[2] = scales_h as u8;
        block[3] = (scales_h >> 8) as u8;
        // Pack 256 nibbles into 128 bytes — sub-block layout:
        //   for ib in 0..8:
        //     for j in 0..16:
        //       lo nibble → output (ib*32 + j)
        //       hi nibble → output (ib*32 + 16 + j)
        // Within each sub-block, qs_nibbles[ib*32 .. ib*32+16] are the low
        // nibbles and qs_nibbles[ib*32+16 .. ib*32+32] are the high nibbles.
        for ib in 0..8 {
            for j in 0..16 {
                let lo = qs_nibbles[ib * 32 + j] & 0x0F;
                let hi = qs_nibbles[ib * 32 + 16 + j] & 0x0F;
                block[8 + ib * 16 + j] = lo | (hi << 4);
            }
        }
        block
    }

    #[test]
    fn iq4_xs_round_trip_unit_scale() {
        // d = 1, ls = 1 for all sub-blocks, every nibble = 8 (idx 8 → 1).
        let mut qs = [0u8; 256];
        qs.fill(8);
        let block = build_iq4_xs_block(1.0, [1i8; 8], qs);
        let mut out = vec![0f32; 256];
        dequant_iq4_xs(&block, &mut out);
        // All outputs should equal d * ls * kvalues[8] = 1 * 1 * 1 = 1.
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 1.0, "out[{i}] = {v}");
        }
    }

    #[test]
    fn iq4_xs_alternating_sign_picks_correct_codebook() {
        // Use nibble idx 0 (kvalues[0] = -127) and nibble idx 15
        // (kvalues[15] = 113) alternately so a layout regression
        // (lo/hi swapped, sub-block order wrong) immediately surfaces.
        let mut qs = [0u8; 256];
        for i in 0..256 {
            qs[i] = if i % 2 == 0 { 0 } else { 15 };
        }
        let block = build_iq4_xs_block(1.0, [1i8; 8], qs);
        let mut out = vec![0f32; 256];
        dequant_iq4_xs(&block, &mut out);
        for (i, v) in out.iter().enumerate() {
            let expected = if i % 2 == 0 { -127.0 } else { 113.0 };
            assert_eq!(*v, expected, "out[{i}] = {v}, expected {expected}");
        }
    }

    #[test]
    fn iq4_xs_per_subblock_scales_apply_independently() {
        // Each sub-block has a distinct scale; check the boundaries.
        let scales = [1i8, 2, 3, 4, 5, 6, 7, 8];
        let mut qs = [0u8; 256];
        qs.fill(8); // kvalues[8] = 1
        let block = build_iq4_xs_block(1.0, scales, qs);
        let mut out = vec![0f32; 256];
        dequant_iq4_xs(&block, &mut out);
        // Outputs in sub-block `ib` should equal `scales[ib]` (since
        // d=1, kvalues[8]=1).
        for ib in 0..8 {
            for j in 0..32 {
                let v = out[ib * 32 + j];
                let expected = scales[ib] as f32;
                assert_eq!(v, expected, "sub-block {ib}, j={j}: got {v}");
            }
        }
    }

    /// Build one IQ4_NL block from `d` + 32 codebook indices.
    fn build_iq4_nl_block(d: f32, qs_nibbles: [u8; 32]) -> [u8; 18] {
        let mut block = [0u8; 18];
        let db = half::f16::from_f32(d).to_le_bytes();
        block[0] = db[0];
        block[1] = db[1];
        // Pack 32 nibbles into 16 bytes. Per upstream layout:
        //   qs[j].lo → out[j]      (j in 0..16)
        //   qs[j].hi → out[j+16]   (j in 0..16)
        for j in 0..16 {
            block[2 + j] = (qs_nibbles[j] & 0x0F) | ((qs_nibbles[16 + j] & 0x0F) << 4);
        }
        block
    }

    /// Build one Q3_K block with given d, all-zero hmask/qs, and a
    /// single scale value placed in slot 0 of the unpacked-scales view
    /// (the rest zero). Used by the tests below to isolate one scale.
    fn build_q3_k_block_single_scale(d: f32, scale0: i8) -> [u8; 110] {
        // Inverting the unpack:
        //   aux[0] low-nibble = scales_packed[0] low-nibble
        //   aux[0] high-nibble (in tmp >> 0 & kmask1) = scales_packed[8] bits 0-1
        // We want scales[0] (= aux byte 0) to equal scale0 (as u8), and
        // scales[1..16] = 0. Build scales_packed by clearing all and
        // setting scales_packed[0] = low 4 bits of scale0, scales_packed[8]
        // bit 0,1 = high 2 bits of scale0.
        let su = scale0 as u8;
        let lo = su & 0x0F;
        let hi = (su >> 4) & 0x03;
        let mut sc = [0u8; 12];
        sc[0] = lo;
        sc[8] = hi;
        let mut block = [0u8; 110];
        // hmask[0..32] = 0, qs[32..96] = 0
        block[96..108].copy_from_slice(&sc);
        block[108..110].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
        block
    }

    #[test]
    fn q3_k_all_zero_qs_with_hmask_zero_yields_minus_four_times_scale() {
        // qs = 0 → low2 = 0 for every weight.
        // hmask = 0 → for every weight, hi bit is unset → subtract 4.
        // So each weight value = (0 - 4) = -4.
        // With d=1.0 and scale[0]=33, dl = 1 * (33 - 32) = 1.0.
        // Sub-block 0 output (first 16 weights) = 1.0 * -4 = -4.0.
        // Other sub-blocks: scale = 0 → dl = -32.0, value = -4 → output = 128.
        // We only assert the first 16 since the others depend on the other
        // (uninteresting here) scales.
        let block = build_q3_k_block_single_scale(1.0, 33);
        let mut out = vec![0f32; 256];
        dequant_q3_k(&block, &mut out);
        for v in out.iter().take(16) {
            assert!((v - (-4.0)).abs() < 1e-6, "expected -4, got {v}");
        }
    }

    #[test]
    fn q3_k_hmask_bit_lifts_value_back_to_zero() {
        // Same as above but hmask[0] = 1 for sub-block 0 (m=1, l=0..15
        // uses hmask[0..16] though — we need all 16 bits across hmask[0..16]
        // set for the m=1 mask check to hit on every l in 0..16).
        // Setting hmask[0..16] = 0xFF makes all 16 bits set at every
        // m value, so for sub-block 0 specifically (m=1) the high bit is
        // set → subtract 0 instead of 4 → value = low2 - 0 = 0.
        let mut block = build_q3_k_block_single_scale(1.0, 33);
        for b in block.iter_mut().take(16) {
            *b = 0xFF;
        }
        let mut out = vec![0f32; 256];
        dequant_q3_k(&block, &mut out);
        for v in out.iter().take(16) {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn q3_k_signed_scale_offset_handles_negative() {
        // scale[0] = 31 → dl = d * (31 - 32) = -d. With d=1 we get dl=-1.
        // qs = 0, hmask = 0 → value = -4. Output = -1 * -4 = +4.
        let block = build_q3_k_block_single_scale(1.0, 31);
        let mut out = vec![0f32; 256];
        dequant_q3_k(&block, &mut out);
        for v in out.iter().take(16) {
            assert!((v - 4.0).abs() < 1e-6, "expected 4, got {v}");
        }
    }

    #[test]
    fn iq3_s_grid_entry_zero_yields_all_ones() {
        // grid[0] = 0x01010101 → bytes [1, 1, 1, 1].
        let g = &IQ3S_GRID[0].to_le_bytes();
        assert_eq!(g, &[1u8, 1, 1, 1]);
    }

    #[test]
    fn iq3_s_all_zeros_block_with_unit_d_yields_all_ones() {
        // Build one IQ3_S block where:
        //   - d = 1.0 (f16)
        //   - all qs = 0 → grid index 0 → grid[0] = [1,1,1,1]
        //   - all qh = 0 → 9th bit clear, indices stay at 0
        //   - all signs = 0 → all outputs positive
        //   - all scales = 0 → sub-scale = 1 + 2*0 = 1
        // Expected: 256 outputs all equal to 1.0.
        let mut block = vec![0u8; 110];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        // qs / qh / signs / scales all stay 0.
        let mut out = vec![0f32; 256];
        dequant_iq3_s(&block, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 1.0, "out[{i}] = {v}");
        }
    }

    #[test]
    fn iq3_s_sign_byte_flips_first_four_outputs() {
        // signs[0] = 0b00001111 → KMASK_IQ2XS[0..4] all match → first
        // 4 outputs of lane 0 are negated; high 4 unchanged.
        let mut block = vec![0u8; 110];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        // signs starts at offset 2 + 64 + 8 = 74.
        block[74] = 0b00001111;
        let mut out = vec![0f32; 256];
        dequant_iq3_s(&block, &mut out);
        // First 4 outputs of sub-block 0 should be -1.0, next 4 = +1.0.
        for i in 0..4 {
            assert_eq!(out[i], -1.0, "out[{i}] expected -1.0, got {}", out[i]);
        }
        for i in 4..8 {
            assert_eq!(out[i], 1.0, "out[{i}] expected 1.0, got {}", out[i]);
        }
        // Rest of the block stays +1.
        for i in 8..256 {
            assert_eq!(out[i], 1.0, "out[{i}] expected 1.0, got {}", out[i]);
        }
    }

    #[test]
    fn iq3_s_scale_nibble_picks_correct_sub_block_multiplier() {
        // scales[0] = 0x10 → low nibble = 0 (db1 = d*1), high nibble
        // = 1 (db2 = d*(1+2) = 3*d). With d=1, first 32 outputs use
        // db1=1, next 32 use db2=3.
        let mut block = vec![0u8; 110];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        // scales starts at offset 2 + 64 + 8 + 32 = 106.
        block[106] = 0x10;
        let mut out = vec![0f32; 256];
        dequant_iq3_s(&block, &mut out);
        // Sub-block 0 (outputs 0..32): db1 = 1, grid val = 1 → outputs = 1.0
        for i in 0..32 {
            assert_eq!(out[i], 1.0, "sub-block 0 out[{i}] = {}", out[i]);
        }
        // Sub-block 1 (outputs 32..64): db2 = 3, grid val = 1 → outputs = 3.0
        for i in 32..64 {
            assert_eq!(out[i], 3.0, "sub-block 1 out[{i}] = {}", out[i]);
        }
        // Sub-blocks 2..8 still use scales[1..4] which are 0 → 1.0.
        for i in 64..256 {
            assert_eq!(out[i], 1.0, "out[{i}] expected 1.0, got {}", out[i]);
        }
    }

    #[test]
    fn iq2_xxs_grid_entry_zero_yields_all_eights() {
        // grid[0] = 0x0808080808080808 → 8 bytes all == 8.
        let bytes = IQ2XXS_GRID[0].to_le_bytes();
        assert_eq!(bytes, [8u8; 8]);
    }

    #[test]
    fn iq2_xxs_all_zeros_block_with_unit_d_and_full_scale_yields_d_times_grid() {
        // Build one IQ2_XXS block where:
        //   - d = 1.0 (f16)
        //   - all qs = 0 → all four grid indices in each sub-block = 0,
        //     so each grid lookup returns [8, 8, 8, 8, 8, 8, 8, 8].
        //   - aux32[1] has scale = 0 in top nibble → db = 1*(0.5+0)*0.25 = 0.125
        //   - all sign indices = 0 → ksigns_iq2xs[0] = 0 → all positive.
        // Expected: 256 outputs all equal to 0.125 * 8 = 1.0.
        let mut block = vec![0u8; 66];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        let mut out = vec![0f32; 256];
        dequant_iq2_xxs(&block, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-6, "out[{i}] = {v}");
        }
    }

    #[test]
    fn iq2_xxs_top_nibble_of_aux1_picks_sub_block_scale() {
        // Set scale=15 in the top nibble of aux32[1] of sub-block 0
        // (and only sub-block 0). db = 1*(0.5+15)*0.25 = 3.875.
        // With grid byte 8, output = 3.875 * 8 = 31.0.
        let mut block = vec![0u8; 66];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        // qs offset = 2; sub-block 0 aux32[1] occupies bytes 2+4..2+8.
        // Top nibble of aux32[1] (LE) = high nibble of the highest byte.
        block[2 + 7] = 0xF0;
        let mut out = vec![0f32; 256];
        dequant_iq2_xxs(&block, &mut out);
        for i in 0..32 {
            assert!((out[i] - 31.0).abs() < 1e-4, "sub-block 0 out[{i}] = {}", out[i]);
        }
        // Sub-blocks 1..8 still have scale 0 → 1.0.
        for i in 32..256 {
            assert!((out[i] - 1.0).abs() < 1e-6, "sub-block !=0 out[{i}] = {}", out[i]);
        }
    }

    #[test]
    fn iq3_xxs_grid_entry_zero_yields_all_fours() {
        // grid[0] = 0x04040404 → 4 bytes all == 4.
        let bytes = IQ3XXS_GRID[0].to_le_bytes();
        assert_eq!(bytes, [4u8; 4]);
    }

    #[test]
    fn iq3_xxs_all_zeros_block_with_unit_d_yields_one() {
        // Build one IQ3_XXS block where:
        //   - d = 1.0 (f16)
        //   - all qs = 0 (grid indices) → grid[0] = [4, 4, 4, 4]
        //   - all qs_sas u32 words = 0 → scale = 0, sign-idx = 0
        // Per sub-block: db = 1.0 * (0.5 + 0) * 0.5 = 0.25
        // KSIGNS_IQ2XS[0] = 0 → all positive
        // Output: 0.25 * 4 = 1.0 everywhere.
        let mut block = vec![0u8; 98];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        let mut out = vec![0f32; 256];
        dequant_iq3_xxs(&block, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-6, "out[{i}] = {v}");
        }
    }

    #[test]
    fn iq3_xxs_top_nibble_of_aux32_picks_sub_block_scale() {
        // Set scale=15 in the top nibble of sub-block 0's aux32 (the
        // first u32 of the scales_and_signs region). db = 1*(0.5+15)*0.5 = 7.75
        // → output = 7.75 * 4 = 31.0 for the 32 weights of sub-block 0.
        let mut block = vec![0u8; 98];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        // scales_and_signs region starts at offset 2 + 64 = 66.
        // Sub-block 0's u32 occupies bytes 66..70. Top nibble of the
        // u32 = high nibble of byte 69 (little-endian).
        block[66 + 3] = 0xF0;
        let mut out = vec![0f32; 256];
        dequant_iq3_xxs(&block, &mut out);
        for i in 0..32 {
            assert!(
                (out[i] - 31.0).abs() < 1e-4,
                "sub-block 0 out[{i}] = {}",
                out[i]
            );
        }
        // Sub-blocks 1..8 still have scale=0 → 1.0.
        for i in 32..256 {
            assert!(
                (out[i] - 1.0).abs() < 1e-6,
                "sub-block !=0 out[{i}] = {}",
                out[i]
            );
        }
    }

    #[test]
    fn iq3_xxs_sign_idx_one_negates_first_four_outputs_of_lane_zero() {
        // KSIGNS_IQ2XS[1] = 0b1000_0001 (popcount-odd parity bit on
        // bit 7 + the input bit 0). bits 0..4 → 0001 → only bit 0
        // selects KMASK_IQ2XS[0]. Result: only out[0] is negated;
        // out[1..4] stay positive (s_lo path); out[4..8] stay
        // positive (s_hi path uses KMASK_IQ2XS[4..8] which see
        // 0b1000 → only bit 7 set → out[7] negated, out[4..7] +).
        // The expected pattern: out[0] = -1, out[1..7] = +1, out[7] = -1.
        let mut block = vec![0u8; 98];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        // Set scale to give db = 0.25 (zeros). Set sub-block 0's
        // sign-idx l=0 (bits 0..7 of aux32) = 1.
        block[66] = 0x01;
        let mut out = vec![0f32; 256];
        dequant_iq3_xxs(&block, &mut out);
        // Verify the per-byte negation pattern matches the KSIGNS
        // table for sign-idx=1.
        let signs = KSIGNS_IQ2XS[1];
        for j in 0..4 {
            let expected = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
            assert_eq!(out[j], expected, "out[{j}] negation by KSIGNS_IQ2XS[1]");
        }
        for j in 0..4 {
            let expected = if signs & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
            assert_eq!(out[4 + j], expected, "out[{}] (s_hi)", 4 + j);
        }
        // Rest of the block stays +1.
        for i in 8..256 {
            assert_eq!(out[i], 1.0, "out[{i}] should remain +1.0");
        }
    }

    #[test]
    fn iq2_xs_grid_entry_zero_yields_all_eights() {
        let bytes = IQ2XS_GRID[0].to_le_bytes();
        assert_eq!(bytes, [8u8; 8]);
    }

    #[test]
    fn iq2_xs_all_zeros_block_with_unit_d_yields_d_eighth_times_grid() {
        // d=1, qs=0, scales=0: each sub-scale db = 1*(0.5+0)*0.25 = 0.125.
        // All grid values = 8 (entry 0 is 0x0808...). signs idx = 0 → all +.
        // Output = 0.125 * 8 = 1.0 everywhere.
        let mut block = vec![0u8; 74];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        let mut out = vec![0f32; 256];
        dequant_iq2_xs(&block, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-6, "out[{i}] = {v}");
        }
    }

    #[test]
    fn iq2_s_grid_entry_zero_yields_all_eights() {
        let bytes = IQ2S_GRID[0].to_le_bytes();
        assert_eq!(bytes, [8u8; 8]);
    }

    #[test]
    fn iq2_s_all_zeros_block_with_unit_d_yields_d_eighth_times_grid() {
        // Block layout: d (2) | qs_lo (32) | signs (32) | qh (8) | scales (8) = 82.
        // qs_lo = 0 → low-bits of grid index = 0; qh = 0 → high bits = 0;
        // so all indices = 0 → grid[0] = [8;8].
        // signs = 0 → ksigns[0] = 0 → all positive.
        // scales = 0 → db = 1*(0.5+0)*0.25 = 0.125.
        // Output = 0.125 * 8 = 1.0 everywhere.
        let mut block = vec![0u8; 82];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        let mut out = vec![0f32; 256];
        dequant_iq2_s(&block, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-6, "out[{i}] = {v}");
        }
    }

    #[test]
    fn iq2_s_qh_bit_pair_picks_high_grid_bits() {
        // Set qh[0] = 0b00000011 → bits (0,1) = 11 → for l=0 these
        // become bits (8,9) of the index. Grid index = 0 | 0x300 = 768.
        // Other lanes (l=1,2,3) read bit-pairs (2-3,4-5,6-7) of qh, all 0,
        // so they still hit grid index 0.
        // d=1, scales=0 → db = 0.125. signs=0 → all positive.
        // Sub-block 0 outputs 0..8 use grid[768], rest of block use grid[0]=8.
        let mut block = vec![0u8; 82];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        // qh starts at offset 2 + 64 = 66.
        block[66] = 0b00000011;
        let mut out = vec![0f32; 256];
        dequant_iq2_s(&block, &mut out);
        let expected_grid_768 = IQ2S_GRID[768].to_le_bytes();
        for j in 0..8 {
            let want = 0.125 * (expected_grid_768[j] as f32);
            assert!((out[j] - want).abs() < 1e-4, "out[{j}] = {} want {}", out[j], want);
        }
        // Lanes l=1,2,3 of sub-block 0 still hit grid[0] → 1.0.
        for i in 8..32 {
            assert!((out[i] - 1.0).abs() < 1e-6, "out[{i}] = {}", out[i]);
        }
    }

    #[test]
    fn iq2_s_signs_byte_flips_lane_outputs() {
        // signs region starts at offset 2 + 32 = 34. Setting signs[0] = 0x0F
        // negates the first 4 outputs of lane 0 in sub-block 0 (j=0..3 match
        // KMASK_IQ2XS), while j=4..7 stay positive.
        let mut block = vec![0u8; 82];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        block[34] = 0x0F;
        let mut out = vec![0f32; 256];
        dequant_iq2_s(&block, &mut out);
        for i in 0..4 {
            assert!((out[i] + 1.0).abs() < 1e-6, "out[{i}] expected -1.0, got {}", out[i]);
        }
        for i in 4..8 {
            assert!((out[i] - 1.0).abs() < 1e-6, "out[{i}] expected +1.0, got {}", out[i]);
        }
    }

    #[test]
    fn iq2_s_scales_byte_picks_low_and_high_nibble_for_correct_halves() {
        // scales[0] = 0x21 → low nibble=1 (db_lo=1*(0.5+1)*0.25=0.375),
        // high nibble=2 (db_hi=1*(0.5+2)*0.25=0.625).
        // l=0,1 use db_lo (weights 0..15), l=2,3 use db_hi (weights 16..31).
        // Grid value = 8 everywhere → 0.375*8 = 3.0 / 0.625*8 = 5.0.
        let mut block = vec![0u8; 82];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        // scales start at offset 2 + 64 + 8 = 74.
        block[74] = 0x21;
        let mut out = vec![0f32; 256];
        dequant_iq2_s(&block, &mut out);
        for i in 0..16 {
            assert!((out[i] - 3.0).abs() < 1e-4, "low-nibble out[{i}] = {}", out[i]);
        }
        for i in 16..32 {
            assert!((out[i] - 5.0).abs() < 1e-4, "high-nibble out[{i}] = {}", out[i]);
        }
        // Other sub-blocks (scales = 0) still emit 1.0.
        for i in 32..256 {
            assert!((out[i] - 1.0).abs() < 1e-6, "out[{i}] = {}", out[i]);
        }
    }

    #[test]
    fn iq2_xs_scales_byte_picks_low_and_high_nibble_for_correct_halves() {
        // scales[0] = 0x21 → low nibble=1 (db_lo = 1*(0.5+1)*0.25 = 0.375),
        // high nibble=2 (db_hi = 1*(0.5+2)*0.25 = 0.625).
        // Sub-block 0 weights 0..15 use db_lo; weights 16..31 use db_hi.
        // Grid value = 8 everywhere → outputs = 0.375*8 = 3.0 / 0.625*8 = 5.0.
        let mut block = vec![0u8; 74];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        // scales start at offset 2 + 64 = 66.
        block[66] = 0x21;
        let mut out = vec![0f32; 256];
        dequant_iq2_xs(&block, &mut out);
        for i in 0..16 {
            assert!((out[i] - 3.0).abs() < 1e-4, "low-nibble out[{i}] = {}", out[i]);
        }
        for i in 16..32 {
            assert!((out[i] - 5.0).abs() < 1e-4, "high-nibble out[{i}] = {}", out[i]);
        }
        // Sub-blocks 1..8 untouched (scales = 0 → 1.0).
        for i in 32..256 {
            assert!((out[i] - 1.0).abs() < 1e-6, "out[{i}] = {}", out[i]);
        }
    }

    #[test]
    fn iq4_nl_round_trip_unit_scale() {
        let mut qs = [0u8; 32];
        qs.fill(8); // kvalues[8] = 1
        let block = build_iq4_nl_block(1.0, qs);
        let mut out = vec![0f32; 32];
        dequant_iq4_nl(&block, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 1.0, "out[{i}] = {v}");
        }
    }

    #[test]
    fn iq4_nl_alternating_codebook_endpoints() {
        let mut qs = [0u8; 32];
        for i in 0..32 {
            qs[i] = if i % 2 == 0 { 0 } else { 15 };
        }
        let block = build_iq4_nl_block(1.0, qs);
        let mut out = vec![0f32; 32];
        dequant_iq4_nl(&block, &mut out);
        for (i, v) in out.iter().enumerate() {
            let expected = if i % 2 == 0 { -127.0 } else { 113.0 };
            assert_eq!(*v, expected, "out[{i}] = {v}");
        }
    }

    #[test]
    fn iq4_nl_negative_d_flips_codebook_signs() {
        let mut qs = [0u8; 32];
        qs.fill(15); // kvalues[15] = 113
        let block = build_iq4_nl_block(-2.0, qs);
        let mut out = vec![0f32; 32];
        dequant_iq4_nl(&block, &mut out);
        for v in &out {
            assert_eq!(*v, -226.0);
        }
    }

    #[test]
    fn iq4_xs_negative_sub_scale_works() {
        // Sub-scale of -5 with nibble idx 1 (kvalues[1] = -104).
        // Expected output: 1.0 * -5 * -104 = 520.
        let mut qs = [0u8; 256];
        qs.fill(1);
        let block = build_iq4_xs_block(1.0, [-5i8; 8], qs);
        let mut out = vec![0f32; 256];
        dequant_iq4_xs(&block, &mut out);
        for v in &out {
            assert_eq!(*v, 520.0);
        }
    }

    // ---- TQ2_0 ----

    /// Build a TQ2_0 block (66 bytes) from a per-trit pattern.
    /// `trits` holds 256 values in `{0, 1, 2}`; `d` is the f16 scale.
    /// Layout follows llama.cpp's emission order — see
    /// [`dequant_tq2_0`] doc-comment for the mapping.
    fn build_tq2_0_block(d: f32, trits: &[u8; 256]) -> Vec<u8> {
        let mut block = vec![0u8; 66];
        // For each output position wi, find the (half, l, m) it came
        // from and write `trits[wi]` into qs[half*32 + m]'s 2-bit
        // field at shift l*2.
        let mut wi = 0usize;
        for half in 0..2 {
            for l in 0..4 {
                let shift = l * 2;
                for m in 0..32 {
                    let byte_idx = half * 32 + m;
                    let trit = trits[wi] & 0x3;
                    block[byte_idx] |= trit << shift;
                    wi += 1;
                }
            }
        }
        assert_eq!(wi, 256);
        let d_bits = half::f16::from_f32(d).to_le_bytes();
        block[64] = d_bits[0];
        block[65] = d_bits[1];
        block
    }

    #[test]
    fn tq2_0_roundtrip_all_ternary_values() {
        // Encode 256 outputs as the cyclic pattern 0, 1, 2, 0, 1, 2, ...
        // (skipping the never-emitted `q=3` value the spec leaves
        // undefined). Decode and verify each output is `d * (q - 1)`.
        let d = 0.25f32;
        let mut trits = [0u8; 256];
        for i in 0..256 {
            trits[i] = (i % 3) as u8;
        }
        let block = build_tq2_0_block(d, &trits);
        let mut out = vec![0f32; 256];
        dequant_tq2_0(&block, &mut out);
        for i in 0..256 {
            let expected = d * (trits[i] as f32 - 1.0);
            assert!(
                (out[i] - expected).abs() < 1e-6,
                "TQ2_0 mismatch at {i}: got {}, want {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn tq2_0_handles_all_three_codes_and_zero_scale() {
        // Edge case: d=0 collapses everything to 0 regardless of trit.
        let trits = [1u8; 256];
        let block = build_tq2_0_block(0.0, &trits);
        let mut out = vec![0f32; 256];
        dequant_tq2_0(&block, &mut out);
        for v in &out {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn tq2_0_emits_qk_k_outputs_per_block() {
        // Sanity: byte/elem accounting. Two consecutive blocks must
        // produce 512 outputs.
        let block_a = build_tq2_0_block(0.5, &[0u8; 256]);
        let block_b = build_tq2_0_block(1.5, &[2u8; 256]);
        let mut combined = Vec::with_capacity(132);
        combined.extend_from_slice(&block_a);
        combined.extend_from_slice(&block_b);
        let mut out = vec![0f32; 512];
        dequant_tq2_0(&combined, &mut out);
        // First block: every trit = 0 → value = d*(0-1) = -0.5.
        for v in &out[0..256] {
            assert_eq!(*v, -0.5);
        }
        // Second block: every trit = 2 → value = d*(2-1) = 1.5.
        for v in &out[256..512] {
            assert_eq!(*v, 1.5);
        }
    }

    // ---- TQ1_0 ----

    fn build_tq1_0_block(d: f32, trits: &[u8; 256]) -> Vec<u8> {
        // Inverse of the canonical `dequant_tq1_0` in ggml's
        // fixed-point wire layout: per byte, assemble the base-3 value
        // most-significant-trit first, then store `ceil(v * 256 / 243)`
        // so the decoder's wrapping-multiply extraction reads each trit
        // back. The 4-trit `qh` tail is shifted up one trit (`v *= 3`)
        // so its digits occupy the top 4 positions. `trits` holds the
        // stored ternary form {0,1,2} (decoded value = d * (trit - 1)).
        //
        // Chunk 0: qs[0..32], 5 trits each → outputs `n*32 + m`
        // Chunk 1: qs[32..48], 5 trits each → outputs `160 + n*16 + m`
        // Tail:    qh[0..4],   4 trits each → outputs `240 + n*4 + j`
        let mut qs = [0u8; 48];
        let mut qh = [0u8; 4];
        let ceil_scale = |v: u32| ((v * 256 + 242) / 243) as u8;
        for m in 0..32 {
            let mut v: u32 = 0;
            for n in 0..5 {
                v = v * 3 + (trits[n * 32 + m] as u32 & 0x3);
            }
            qs[m] = ceil_scale(v);
        }
        for m in 0..16 {
            let mut v: u32 = 0;
            for n in 0..5 {
                v = v * 3 + (trits[160 + n * 16 + m] as u32 & 0x3);
            }
            qs[32 + m] = ceil_scale(v);
        }
        for j in 0..4 {
            let mut v: u32 = 0;
            for n in 0..4 {
                v = v * 3 + (trits[240 + n * 4 + j] as u32 & 0x3);
            }
            v *= 3;
            qh[j] = ceil_scale(v);
        }
        let mut block = vec![0u8; 54];
        block[0..48].copy_from_slice(&qs);
        block[48..52].copy_from_slice(&qh);
        let d_bits = half::f16::from_f32(d).to_le_bytes();
        block[52] = d_bits[0];
        block[53] = d_bits[1];
        block
    }

    #[test]
    fn tq1_0_roundtrip_zero_block_decodes_to_negative_d() {
        // All trits = 0 → packed byte = 0 → decoded trit = 0 →
        // value = d * (0 - 1) = -d. Verifies the base-3 mul-extract
        // matches the inverse encoder above (key correctness gate).
        let d = 0.5f32;
        let trits = [0u8; 256];
        let block = build_tq1_0_block(d, &trits);
        let mut out = vec![0f32; 256];
        dequant_tq1_0(&block, &mut out);
        for v in &out {
            assert!(
                (*v - (-d)).abs() < 1e-6,
                "TQ1_0 all-zero block should decode to -d, got {v}"
            );
        }
    }

    #[test]
    fn tq1_0_roundtrip_handles_mixed_trits() {
        // Pattern: alternate 0/1/2 across all 256 outputs. The
        // encoder above is the algebraic inverse of the dequant —
        // this asserts the round-trip stays within the trit
        // alphabet without overflow or aliasing across positions.
        let d = 1.0f32;
        let mut trits = [0u8; 256];
        for i in 0..256 {
            trits[i] = (i % 3) as u8;
        }
        let block = build_tq1_0_block(d, &trits);
        let mut out = vec![0f32; 256];
        dequant_tq1_0(&block, &mut out);
        for i in 0..256 {
            let expected = d * (trits[i] as f32 - 1.0);
            assert!(
                (out[i] - expected).abs() < 1e-6,
                "TQ1_0 mismatch at {i}: got {}, want {expected}",
                out[i]
            );
        }
    }
}
