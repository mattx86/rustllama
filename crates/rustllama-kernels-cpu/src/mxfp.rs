//! OCP Microscaling (MX) weight matvec kernels: MXFP4 / MXFP6 / MXFP8.
//!
//! Each format packs 32 weights per block sharing one 8-bit E8M0
//! power-of-two scale, with per-element low-precision floats:
//! MXFP4 = E2M1 (4-bit), MXFP6 = E3M2 (6-bit), MXFP8 = E4M3 (8-bit).
//! The public matvecs (`out[m] = Σ_k W[m,k]·x[k]`) dispatch at runtime to
//! AVX-512 / AVX2 / scalar (mirroring the NVFP4 matvec family and the
//! TurboQuant dispatch idiom); the scalar bodies stay as the reference and
//! the non-x86 fallback. The per-element decode is byte-exact with the GGUF
//! loader's `dequant_mxfp4/6/8` and with the SYCL/CUDA kernels — the parity
//! harness checks all three backends against this reference, and the
//! `mxfp{4,6,8}_simd_matches_scalar` tests check the SIMD lanes against it.

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
/// `k/32` 17-byte blocks; `k % 32 == 0`. Dispatches to AVX-512 / AVX2 /
/// scalar at runtime; all three are numerically equivalent (only the FMA
/// reduction order differs — the E2M1 codebook decode is exact).
pub fn matvec_mxfp4_w_f32_a(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    assert_eq!(k % MXFP4_BLOCK_ELEMS, 0, "mxfp4 matvec requires k % 32 == 0");
    let bpr = k / MXFP4_BLOCK_ELEMS;
    assert_eq!(w_bytes.len(), m * bpr * MXFP4_BLOCK_BYTES);
    assert_eq!(x.len(), k);
    assert_eq!(out.len(), m);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection; the kernel reads 17-byte
            // blocks for `m × bpr` blocks and writes exactly `m` f32.
            unsafe { matvec_mxfp4_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection (bounds as above).
            unsafe { matvec_mxfp4_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    matvec_mxfp4_w_f32_a_scalar(w_bytes, x, out, m, k);
}

/// Scalar reference for the MXFP4 matvec — also the non-x86 fallback.
fn matvec_mxfp4_w_f32_a_scalar(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    let bpr = k / MXFP4_BLOCK_ELEMS;
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
/// are a little-endian bitstream of 32 six-bit codes. Dispatches to
/// AVX-512 / AVX2 / scalar at runtime (equivalent up to FMA order).
pub fn matvec_mxfp6_w_f32_a(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    assert_eq!(k % MXFP6_BLOCK_ELEMS, 0, "mxfp6 matvec requires k % 32 == 0");
    let bpr = k / MXFP6_BLOCK_ELEMS;
    assert_eq!(w_bytes.len(), m * bpr * MXFP6_BLOCK_BYTES);
    assert_eq!(x.len(), k);
    assert_eq!(out.len(), m);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection; reads 25-byte blocks, writes `m` f32.
            unsafe { matvec_mxfp6_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection (bounds as above).
            unsafe { matvec_mxfp6_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    matvec_mxfp6_w_f32_a_scalar(w_bytes, x, out, m, k);
}

/// Scalar reference for the MXFP6 matvec — also the non-x86 fallback.
fn matvec_mxfp6_w_f32_a_scalar(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    let bpr = k / MXFP6_BLOCK_ELEMS;
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
/// scale byte. Dispatches to AVX-512 / AVX2 / scalar at runtime
/// (equivalent up to FMA order; the E4M3 codebook decode is exact).
pub fn matvec_mxfp8_w_f32_a(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    assert_eq!(k % MXFP8_BLOCK_ELEMS, 0, "mxfp8 matvec requires k % 32 == 0");
    let bpr = k / MXFP8_BLOCK_ELEMS;
    assert_eq!(w_bytes.len(), m * bpr * MXFP8_BLOCK_BYTES);
    assert_eq!(x.len(), k);
    assert_eq!(out.len(), m);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection; reads 33-byte blocks, writes `m` f32.
            unsafe { matvec_mxfp8_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection (bounds as above).
            unsafe { matvec_mxfp8_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    matvec_mxfp8_w_f32_a_scalar(w_bytes, x, out, m, k);
}

/// Scalar reference for the MXFP8 matvec — also the non-x86 fallback.
fn matvec_mxfp8_w_f32_a_scalar(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    let bpr = k / MXFP8_BLOCK_ELEMS;
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

// ----------------------------------------------------------------------
// x86_64 SIMD fast paths.
//
// Shape mirrors the NVFP4 matvec family (`crate::nvfp4`) and the TurboQuant
// AVX dispatch idiom (`crate::turboquant`): `#[target_feature]` AVX-512 and
// AVX2 variants behind runtime `is_x86_feature_detected!`, with the scalar
// path as the reference + fallback. The per-block E8M0 scale (`e8m0_to_f32`)
// and the element codebooks decode identically to the scalar path, so only
// the FMA reduction order differs.
//
// Every block is exactly 32 elements and `k % 32 == 0`, so a block is two
// 16-lane AVX-512 FMAs or four 8-lane AVX2 FMAs with no intra-block tail.
// Partial sums are kept lane-wise in the accumulators across all of a row's
// blocks (each block j advances `x` by 32) and reduced once at row end —
// the same trick the NVFP4 kernel uses.
// ----------------------------------------------------------------------

/// Horizontal sum of a YMM (8 f32 lanes) → scalar, via the standard
/// NVFP4-style `extractf128` + two-stage `hadd` reduction.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx,avx2")]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let mut s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

/// 64-entry E3M2 (MXFP6) codebook, `code -> f32`. Byte-identical to
/// `e3m2_to_f32`; materialized once per matvec so the inner loop is a table
/// lookup instead of a per-element decode (branch + `powi`).
#[cfg(target_arch = "x86_64")]
fn e3m2_codebook() -> [f32; 64] {
    let mut cb = [0.0f32; 64];
    for (c, v) in cb.iter_mut().enumerate() {
        *v = e3m2_to_f32(c as u8);
    }
    cb
}

/// 256-entry E4M3 (MXFP8) codebook, `byte -> f32`. Byte-identical to
/// `e4m3_to_f32` (NaN in the `0x7F` / `0xFF` slots, which never occur in
/// valid weight data). Used as the AVX gather LUT.
#[cfg(target_arch = "x86_64")]
fn e4m3_codebook() -> [f32; 256] {
    let mut cb = [0.0f32; 256];
    for (c, v) in cb.iter_mut().enumerate() {
        *v = e4m3_to_f32(c as u8);
    }
    cb
}

/// AVX2 MXFP4 matvec. A 32-element block is two NVFP4-shaped 16-element
/// halves (bytes 0..8 → elems 0..16, bytes 8..16 → elems 16..32); each half
/// decodes exactly like `matvec_nvfp4_w_f32_a_avx2`: widen the 8 packed
/// bytes to i32 lanes, split lo/hi nibbles, gather the 16-entry E2M1 LUT,
/// interleave the even/odd code streams back to element order, scale, FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_mxfp4_w_f32_a_avx2(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    use std::arch::x86_64::*;
    let bpr = k / MXFP4_BLOCK_ELEMS;
    let cb = E2M1_CODEBOOK.as_ptr();
    let nibble_mask = _mm256_set1_epi32(0x0F);
    for i in 0..m {
        let row = i * bpr * MXFP4_BLOCK_BYTES;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        for b in 0..bpr {
            let off = row + b * MXFP4_BLOCK_BYTES;
            // E8M0 power-of-two scale, broadcast to all lanes.
            let scale = _mm256_set1_ps(e8m0_to_f32(w_bytes[off + 16]));
            for half in 0..2usize {
                let byte_off = off + half * 8;
                let x_base = b * MXFP4_BLOCK_ELEMS + half * 16;
                // 8 packed bytes → i32x8 (each lane one byte = two nibbles).
                let bytes_xmm = _mm_loadl_epi64(w_bytes.as_ptr().add(byte_off) as *const __m128i);
                let i32_ymm = _mm256_cvtepu8_epi32(bytes_xmm);
                let lo_idx = _mm256_and_si256(i32_ymm, nibble_mask);
                let hi_idx = _mm256_and_si256(_mm256_srli_epi32(i32_ymm, 4), nibble_mask);
                let lo_codes = _mm256_i32gather_ps::<4>(cb, lo_idx);
                let hi_codes = _mm256_i32gather_ps::<4>(cb, hi_idx);
                // Weave even (lo) / odd (hi) codes back to contiguous [0..16];
                // same unpack + permute2f128 recipe as the NVFP4 kernel.
                let ul = _mm256_unpacklo_ps(lo_codes, hi_codes);
                let uh = _mm256_unpackhi_ps(lo_codes, hi_codes);
                let codes_lo8 = _mm256_permute2f128_ps::<0x20>(ul, uh);
                let codes_hi8 = _mm256_permute2f128_ps::<0x31>(ul, uh);
                let x_ptr = x.as_ptr().add(x_base);
                acc0 = _mm256_fmadd_ps(_mm256_mul_ps(codes_lo8, scale), _mm256_loadu_ps(x_ptr), acc0);
                acc1 =
                    _mm256_fmadd_ps(_mm256_mul_ps(codes_hi8, scale), _mm256_loadu_ps(x_ptr.add(8)), acc1);
            }
        }
        out[i] = hsum256_ps(_mm256_add_ps(acc0, acc1));
    }
}

/// AVX-512 MXFP4 matvec. The 16-entry E2M1 codebook fits in one ZMM, so the
/// nibble decode is a register-resident `permutexvar` LUT — no memory
/// gather. Per 32-element block: widen all 16 code bytes to i32 lanes, look
/// up even (lo-nibble) and odd (hi-nibble) codes, then two `permutex2var`
/// weaves rebuild contiguous elements [0..16] / [16..32] for a pair of
/// 16-wide FMAs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_mxfp4_w_f32_a_avx512(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    use std::arch::x86_64::*;
    let bpr = k / MXFP4_BLOCK_ELEMS;
    let cb = _mm512_loadu_ps(E2M1_CODEBOOK.as_ptr());
    // permutex2var combined-source indices: 0..15 pick lo_codes (even
    // elements), 16..31 pick hi_codes (odd elements). low = [lo0,hi0,lo1,
    // hi1,…] = elems 0..16, high = [lo8,hi8,…] = elems 16..32.
    let idx_low = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    let idx_high = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
    let nibble_mask = _mm512_set1_epi32(0x0F);
    for i in 0..m {
        let row = i * bpr * MXFP4_BLOCK_BYTES;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        for b in 0..bpr {
            let off = row + b * MXFP4_BLOCK_BYTES;
            let scale = _mm512_set1_ps(e8m0_to_f32(w_bytes[off + 16]));
            // All 16 code bytes → i32x16 (each lane a byte = two nibbles).
            let bytes_xmm = _mm_loadu_si128(w_bytes.as_ptr().add(off) as *const __m128i);
            let i32_zmm = _mm512_cvtepu8_epi32(bytes_xmm);
            let lo_idx = _mm512_and_si512(i32_zmm, nibble_mask);
            let hi_idx = _mm512_and_si512(_mm512_srli_epi32::<4>(i32_zmm), nibble_mask);
            let lo_codes = _mm512_permutexvar_ps(lo_idx, cb);
            let hi_codes = _mm512_permutexvar_ps(hi_idx, cb);
            let codes_lo = _mm512_permutex2var_ps(lo_codes, idx_low, hi_codes);
            let codes_hi = _mm512_permutex2var_ps(lo_codes, idx_high, hi_codes);
            let x_ptr = x.as_ptr().add(b * MXFP4_BLOCK_ELEMS);
            acc0 = _mm512_fmadd_ps(_mm512_mul_ps(codes_lo, scale), _mm512_loadu_ps(x_ptr), acc0);
            acc1 = _mm512_fmadd_ps(_mm512_mul_ps(codes_hi, scale), _mm512_loadu_ps(x_ptr.add(16)), acc1);
        }
        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}

/// Decode one 24-byte MXFP6 code block into 32 f32 (`E3M2 value × scale`),
/// bit-identical to the scalar matvec's inner loop. The 6-bit little-endian
/// bitstream is inherently serial to unpack, so both SIMD MXFP6 paths share
/// this scalar decode into a register-width scratch and only vectorize the
/// subsequent multiply-accumulate.
#[cfg(target_arch = "x86_64")]
#[inline]
fn decode_mxfp6_block(codes: &[u8], cb: &[f32; 64], scale: f32, vals: &mut [f32; MXFP6_BLOCK_ELEMS]) {
    for (j, v) in vals.iter_mut().enumerate() {
        let bitpos = j * 6;
        let byte_idx = bitpos / 8;
        let bit_off = bitpos % 8;
        let lo = codes[byte_idx] as u16;
        let hi = if byte_idx + 1 < MXFP6_CODE_BYTES {
            codes[byte_idx + 1] as u16
        } else {
            0
        };
        let code = ((lo | (hi << 8)) >> bit_off) & 0x3F;
        *v = cb[code as usize] * scale;
    }
}

/// AVX2 MXFP6 matvec. Scalar E3M2 bitstream decode (scale pre-folded) into a
/// 32-wide scratch, then four 8-lane FMAs against the activations.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_mxfp6_w_f32_a_avx2(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    use std::arch::x86_64::*;
    let bpr = k / MXFP6_BLOCK_ELEMS;
    let cb = e3m2_codebook();
    let mut vals = [0.0f32; MXFP6_BLOCK_ELEMS];
    for i in 0..m {
        let row = i * bpr * MXFP6_BLOCK_BYTES;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        for b in 0..bpr {
            let off = row + b * MXFP6_BLOCK_BYTES;
            let scale = e8m0_to_f32(w_bytes[off + MXFP6_CODE_BYTES]);
            let codes = &w_bytes[off..off + MXFP6_CODE_BYTES];
            decode_mxfp6_block(codes, &cb, scale, &mut vals);
            let x_ptr = x.as_ptr().add(b * MXFP6_BLOCK_ELEMS);
            let vp = vals.as_ptr();
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(vp), _mm256_loadu_ps(x_ptr), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(vp.add(8)), _mm256_loadu_ps(x_ptr.add(8)), acc1);
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(vp.add(16)), _mm256_loadu_ps(x_ptr.add(16)), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(vp.add(24)), _mm256_loadu_ps(x_ptr.add(24)), acc1);
        }
        out[i] = hsum256_ps(_mm256_add_ps(acc0, acc1));
    }
}

/// AVX-512 MXFP6 matvec. Same serial E3M2 decode into a 32-wide scratch as
/// the AVX2 path; the accumulate is two 16-lane FMAs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_mxfp6_w_f32_a_avx512(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    use std::arch::x86_64::*;
    let bpr = k / MXFP6_BLOCK_ELEMS;
    let cb = e3m2_codebook();
    let mut vals = [0.0f32; MXFP6_BLOCK_ELEMS];
    for i in 0..m {
        let row = i * bpr * MXFP6_BLOCK_BYTES;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        for b in 0..bpr {
            let off = row + b * MXFP6_BLOCK_BYTES;
            let scale = e8m0_to_f32(w_bytes[off + MXFP6_CODE_BYTES]);
            let codes = &w_bytes[off..off + MXFP6_CODE_BYTES];
            decode_mxfp6_block(codes, &cb, scale, &mut vals);
            let x_ptr = x.as_ptr().add(b * MXFP6_BLOCK_ELEMS);
            let vp = vals.as_ptr();
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(vp), _mm512_loadu_ps(x_ptr), acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(vp.add(16)), _mm512_loadu_ps(x_ptr.add(16)), acc1);
        }
        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}

/// AVX2 MXFP8 matvec. Each E4M3 byte indexes the 256-entry LUT directly:
/// widen 8 bytes to i32 lanes, gather 8 codebook values, scale, FMA. Four
/// 8-lane groups cover the block; even/odd groups feed two accumulators to
/// relax the FMA dependency chain.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_mxfp8_w_f32_a_avx2(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    use std::arch::x86_64::*;
    let bpr = k / MXFP8_BLOCK_ELEMS;
    let cb = e4m3_codebook();
    let cb_base = cb.as_ptr();
    for i in 0..m {
        let row = i * bpr * MXFP8_BLOCK_BYTES;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        for b in 0..bpr {
            let off = row + b * MXFP8_BLOCK_BYTES;
            let scale = _mm256_set1_ps(e8m0_to_f32(w_bytes[off + 32]));
            let x_blk = x.as_ptr().add(b * MXFP8_BLOCK_ELEMS);
            let wp = w_bytes.as_ptr().add(off);
            let g0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(wp as *const __m128i));
            let g1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(wp.add(8) as *const __m128i));
            let g2 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(wp.add(16) as *const __m128i));
            let g3 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(wp.add(24) as *const __m128i));
            let c0 = _mm256_mul_ps(_mm256_i32gather_ps::<4>(cb_base, g0), scale);
            let c1 = _mm256_mul_ps(_mm256_i32gather_ps::<4>(cb_base, g1), scale);
            let c2 = _mm256_mul_ps(_mm256_i32gather_ps::<4>(cb_base, g2), scale);
            let c3 = _mm256_mul_ps(_mm256_i32gather_ps::<4>(cb_base, g3), scale);
            acc0 = _mm256_fmadd_ps(c0, _mm256_loadu_ps(x_blk), acc0);
            acc1 = _mm256_fmadd_ps(c1, _mm256_loadu_ps(x_blk.add(8)), acc1);
            acc0 = _mm256_fmadd_ps(c2, _mm256_loadu_ps(x_blk.add(16)), acc0);
            acc1 = _mm256_fmadd_ps(c3, _mm256_loadu_ps(x_blk.add(24)), acc1);
        }
        out[i] = hsum256_ps(_mm256_add_ps(acc0, acc1));
    }
}

/// AVX-512 MXFP8 matvec. Widen 16 E4M3 bytes to i32 lanes and gather 16
/// codebook values per half-block; two 16-lane FMAs cover the block. Note
/// the AVX-512 gather argument order is `(offsets, base)`, the reverse of
/// the AVX2 `(base, offsets)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_mxfp8_w_f32_a_avx512(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    use std::arch::x86_64::*;
    let bpr = k / MXFP8_BLOCK_ELEMS;
    let cb = e4m3_codebook();
    let cb_base = cb.as_ptr();
    for i in 0..m {
        let row = i * bpr * MXFP8_BLOCK_BYTES;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        for b in 0..bpr {
            let off = row + b * MXFP8_BLOCK_BYTES;
            let scale = _mm512_set1_ps(e8m0_to_f32(w_bytes[off + 32]));
            let x_blk = x.as_ptr().add(b * MXFP8_BLOCK_ELEMS);
            let wp = w_bytes.as_ptr().add(off);
            let idx0 = _mm512_cvtepu8_epi32(_mm_loadu_si128(wp as *const __m128i));
            let idx1 = _mm512_cvtepu8_epi32(_mm_loadu_si128(wp.add(16) as *const __m128i));
            let c0 = _mm512_mul_ps(_mm512_i32gather_ps::<4>(idx0, cb_base), scale);
            let c1 = _mm512_mul_ps(_mm512_i32gather_ps::<4>(idx1, cb_base), scale);
            acc0 = _mm512_fmadd_ps(c0, _mm512_loadu_ps(x_blk), acc0);
            acc1 = _mm512_fmadd_ps(c1, _mm512_loadu_ps(x_blk.add(16)), acc1);
        }
        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
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
        // Relative tolerance: the public matvec dispatches to SIMD, whose
        // lane-wise reduction differs from this sequential sum at the f32
        // level (the E4M3 codebook decode itself is exact).
        assert!(
            (out[0] - ref_dot).abs() < 1e-4 * ref_dot.abs().max(1.0),
            "got {} want {}",
            out[0],
            ref_dot,
        );
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

    // ---- SIMD parity: the SIMD lanes must match the scalar reference ----
    //
    // The codebook + E8M0 scale decode is exact; only the FMA reduction
    // order differs between scalar and SIMD, so we compare against the
    // scalar path with a rel(1e-4) + abs(3e-3) tolerance (the absolute
    // floor absorbs f32 accumulation noise without hiding a real bug, which
    // would move a result by ~its own magnitude). We exercise the public
    // dispatch (whatever this host has) plus, when detected, each
    // `#[target_feature]` variant directly, over several shapes.

    // Small deterministic LCG so inputs are reproducible across runs.
    fn lcg(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 32) as u32
    }

    fn rand_x(state: &mut u64, k: usize) -> Vec<f32> {
        (0..k)
            .map(|_| (lcg(state) as f32 / u32::MAX as f32) * 4.0 - 2.0)
            .collect()
    }

    // E8M0 scale byte kept in a well-conditioned range (2^-3 .. 2^3) and
    // clear of 0xFF (NaN), so accumulation stays numerically tame.
    fn rand_scale_byte(state: &mut u64) -> u8 {
        124 + (lcg(state) % 7) as u8
    }

    fn assert_close(simd: &[f32], scalar: &[f32], tag: &str) {
        for (i, (&a, &b)) in simd.iter().zip(scalar.iter()).enumerate() {
            let tol = 1e-4 * b.abs() + 3e-3;
            let err = (a - b).abs();
            assert!(
                err <= tol,
                "{tag} row {i}: simd {a} vs scalar {b} (err {err} > tol {tol})",
            );
        }
    }

    fn build_mxfp4(state: &mut u64, m: usize, k: usize) -> Vec<u8> {
        let bpr = k / MXFP4_BLOCK_ELEMS;
        let mut w = vec![0u8; m * bpr * MXFP4_BLOCK_BYTES];
        for blk in w.chunks_mut(MXFP4_BLOCK_BYTES) {
            for byte in blk[..16].iter_mut() {
                let lo = (lcg(state) & 0x0F) as u8;
                let hi = (lcg(state) & 0x0F) as u8;
                *byte = lo | (hi << 4);
            }
            blk[16] = rand_scale_byte(state);
        }
        w
    }

    fn build_mxfp6(state: &mut u64, m: usize, k: usize) -> Vec<u8> {
        let bpr = k / MXFP6_BLOCK_ELEMS;
        let mut w = vec![0u8; m * bpr * MXFP6_BLOCK_BYTES];
        for blk in w.chunks_mut(MXFP6_BLOCK_BYTES) {
            for j in 0..MXFP6_BLOCK_ELEMS {
                let code = (lcg(state) & 0x3F) as u32; // 6-bit E3M2 code
                let bitpos = j * 6;
                let byte_idx = bitpos / 8;
                let bit_off = bitpos % 8;
                let shifted = code << bit_off;
                blk[byte_idx] |= (shifted & 0xFF) as u8;
                if byte_idx + 1 < MXFP6_CODE_BYTES {
                    blk[byte_idx + 1] |= ((shifted >> 8) & 0xFF) as u8;
                }
            }
            blk[MXFP6_CODE_BYTES] = rand_scale_byte(state);
        }
        w
    }

    fn build_mxfp8(state: &mut u64, m: usize, k: usize) -> Vec<u8> {
        let bpr = k / MXFP8_BLOCK_ELEMS;
        let mut w = vec![0u8; m * bpr * MXFP8_BLOCK_BYTES];
        for blk in w.chunks_mut(MXFP8_BLOCK_BYTES) {
            for byte in blk[..32].iter_mut() {
                let mut c = (lcg(state) & 0xFF) as u8;
                if c == 0x7F || c == 0xFF {
                    c = 0x10; // avoid the E4M3 NaN slots
                }
                *byte = c;
            }
            blk[32] = rand_scale_byte(state);
        }
        w
    }

    // Shapes: single-row single-block, plus multi-row multi-block cases.
    const SIMD_SHAPES: [(usize, usize); 4] = [(1, 32), (4, 128), (3, 96), (2, 64)];

    #[test]
    fn mxfp4_simd_matches_scalar() {
        let mut st = 0x1234_5678_9abc_def0u64;
        for &(m, k) in &SIMD_SHAPES {
            let w = build_mxfp4(&mut st, m, k);
            let x = rand_x(&mut st, k);
            let mut scalar = vec![0f32; m];
            matvec_mxfp4_w_f32_a_scalar(&w, &x, &mut scalar, m, k);
            let mut disp = vec![0f32; m];
            matvec_mxfp4_w_f32_a(&w, &x, &mut disp, m, k);
            assert_close(&disp, &scalar, "mxfp4 dispatch");
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    let mut o = vec![0f32; m];
                    unsafe { matvec_mxfp4_w_f32_a_avx2(&w, &x, &mut o, m, k) };
                    assert_close(&o, &scalar, "mxfp4 avx2");
                }
                if is_x86_feature_detected!("avx512f") {
                    let mut o = vec![0f32; m];
                    unsafe { matvec_mxfp4_w_f32_a_avx512(&w, &x, &mut o, m, k) };
                    assert_close(&o, &scalar, "mxfp4 avx512");
                }
            }
        }
    }

    #[test]
    fn mxfp6_simd_matches_scalar() {
        let mut st = 0x0f0f_0f0f_dead_beefu64;
        for &(m, k) in &SIMD_SHAPES {
            let w = build_mxfp6(&mut st, m, k);
            let x = rand_x(&mut st, k);
            let mut scalar = vec![0f32; m];
            matvec_mxfp6_w_f32_a_scalar(&w, &x, &mut scalar, m, k);
            let mut disp = vec![0f32; m];
            matvec_mxfp6_w_f32_a(&w, &x, &mut disp, m, k);
            assert_close(&disp, &scalar, "mxfp6 dispatch");
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    let mut o = vec![0f32; m];
                    unsafe { matvec_mxfp6_w_f32_a_avx2(&w, &x, &mut o, m, k) };
                    assert_close(&o, &scalar, "mxfp6 avx2");
                }
                if is_x86_feature_detected!("avx512f") {
                    let mut o = vec![0f32; m];
                    unsafe { matvec_mxfp6_w_f32_a_avx512(&w, &x, &mut o, m, k) };
                    assert_close(&o, &scalar, "mxfp6 avx512");
                }
            }
        }
    }

    #[test]
    fn mxfp8_simd_matches_scalar() {
        let mut st = 0xcafe_babe_1357_9bdfu64;
        for &(m, k) in &SIMD_SHAPES {
            let w = build_mxfp8(&mut st, m, k);
            let x = rand_x(&mut st, k);
            let mut scalar = vec![0f32; m];
            matvec_mxfp8_w_f32_a_scalar(&w, &x, &mut scalar, m, k);
            let mut disp = vec![0f32; m];
            matvec_mxfp8_w_f32_a(&w, &x, &mut disp, m, k);
            assert_close(&disp, &scalar, "mxfp8 dispatch");
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    let mut o = vec![0f32; m];
                    unsafe { matvec_mxfp8_w_f32_a_avx2(&w, &x, &mut o, m, k) };
                    assert_close(&o, &scalar, "mxfp8 avx2");
                }
                if is_x86_feature_detected!("avx512f") {
                    let mut o = vec![0f32; m];
                    unsafe { matvec_mxfp8_w_f32_a_avx512(&w, &x, &mut o, m, k) };
                    assert_close(&o, &scalar, "mxfp8 avx512");
                }
            }
        }
    }
}
