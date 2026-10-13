//! Speculative bit-manipulation CPU extensions (VPOPCNTDQ / GFNI / VBMI2).
//!
//! These have **no current consumer** in the GGUF formats this runtime uses —
//! standard sub-byte unpack is already optimal with shifts+masks, and there is
//! no binary (BitNet-style) format here yet. They are built as self-contained,
//! self-check-gated primitives for FUTURE low-bit formats, per an explicit
//! request for complete Intel+AMD extension coverage. Each detects via CPUID
//! (the `is_x86_feature_detected!` arms are nightly-gated) and validates
//! against a scalar reference; GFNI's is validatable on the dev box (AVX2-GFNI),
//! the AVX-512 ones (VPOPCNTDQ/VBMI2) are write-blind + compile-verified.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;
use std::sync::OnceLock;

// ---- AVX-512 VPOPCNTDQ: binary (±1) dot product ----

/// AVX-512-VPOPCNTDQ via CPUID.(EAX=7,ECX=0):ECX[14] (+ F).
fn vpopcntdq_supported() -> bool {
    let l0 = unsafe { __cpuid_count(7, 0) };
    ((l0.ebx >> 16) & 1 != 0) && ((l0.ecx >> 14) & 1 != 0)
}

/// Scalar binary dot: weights/activations are ±1 packed as bits (1→+1, 0→−1).
/// `Σ sign·sign = n_bits − 2·popcount(w XOR x)`.
fn binary_dot_scalar(w: &[u64], x: &[u64]) -> i64 {
    let n = (w.len() * 64) as i64;
    let mut diff = 0i64;
    for (a, b) in w.iter().zip(x) {
        diff += (a ^ b).count_ones() as i64;
    }
    n - 2 * diff
}

/// VPOPCNTDQ binary dot over 8×u64 (512-bit) chunks + a scalar tail.
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn binary_dot_vpopcntdq(w: &[u64], x: &[u64]) -> i64 {
    let n = (w.len() * 64) as i64;
    let chunks = w.len() / 8;
    let mut acc = _mm512_setzero_si512();
    for c in 0..chunks {
        let wv = _mm512_loadu_si512(w.as_ptr().add(c * 8) as *const __m512i);
        let xv = _mm512_loadu_si512(x.as_ptr().add(c * 8) as *const __m512i);
        let pc = _mm512_popcnt_epi64(_mm512_xor_si512(wv, xv));
        acc = _mm512_add_epi64(acc, pc);
    }
    let mut diff = _mm512_reduce_add_epi64(acc);
    for i in (chunks * 8)..w.len() {
        diff += (w[i] ^ x[i]).count_ones() as i64;
    }
    n - 2 * diff
}

/// Whether the VPOPCNTDQ binary dot is usable — cached + self-checked.
#[allow(dead_code)]
pub fn binary_dot_usable() -> bool {
    static U: OnceLock<bool> = OnceLock::new();
    *U.get_or_init(|| {
        if !vpopcntdq_supported() {
            return false;
        }
        let w: Vec<u64> = (0..20).map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
        let x: Vec<u64> = (0..20).map(|i| (i as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)).collect();
        // SAFETY: vpopcntdq_supported() checked above.
        let got = unsafe { binary_dot_vpopcntdq(&w, &x) };
        got == binary_dot_scalar(&w, &x)
    })
}

/// Binary (±1) dot, picking VPOPCNTDQ where available. `#[allow(dead_code)]`:
/// no binary format consumes it yet.
#[allow(dead_code)]
pub fn binary_dot(w: &[u64], x: &[u64]) -> i64 {
    if binary_dot_usable() {
        // SAFETY: binary_dot_usable() confirmed the feature + self-check.
        unsafe { binary_dot_vpopcntdq(w, x) }
    } else {
        binary_dot_scalar(w, x)
    }
}

// ---- GFNI: reverse the 8 bits within each byte (AVX2 form) ----

/// GFNI via CPUID.(EAX=7,ECX=0):ECX[8].
fn gfni_supported() -> bool {
    let l0 = unsafe { __cpuid_count(7, 0) };
    (l0.ecx >> 8) & 1 != 0
}

/// Reverse the bit order within each byte of `bytes` using the GF(2) affine
/// transform (bit-reversal permutation matrix, no XOR). AVX2-GFNI (32 bytes /
/// op) + a scalar tail.
#[target_feature(enable = "avx,avx2,gfni")]
unsafe fn gfni_reverse_bits(bytes: &mut [u8]) {
    // Per-64-bit-lane 8x8 matrix whose row j selects input bit (7-j) → output
    // bit j = input bit 7-j, i.e. bit reversal.
    let mat = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
    let n32 = bytes.len() / 32;
    for c in 0..n32 {
        let p = bytes.as_mut_ptr().add(c * 32) as *mut __m256i;
        let v = _mm256_loadu_si256(p);
        let r = _mm256_gf2p8affine_epi64_epi8::<0>(v, mat);
        _mm256_storeu_si256(p, r);
    }
    for i in (n32 * 32)..bytes.len() {
        bytes[i] = bytes[i].reverse_bits();
    }
}

/// Whether the GFNI bit-reverse is usable — cached + self-checked (validatable
/// on the dev box: AVX2-GFNI is present on Alder/Raptor Lake).
#[allow(dead_code)]
pub fn gfni_reverse_usable() -> bool {
    static U: OnceLock<bool> = OnceLock::new();
    *U.get_or_init(|| {
        if !gfni_supported() {
            return false;
        }
        let mut v: Vec<u8> = (0..70u16).map(|i| (i as u8).wrapping_mul(37)).collect();
        let want: Vec<u8> = v.iter().map(|b| b.reverse_bits()).collect();
        // SAFETY: gfni_supported() checked above.
        unsafe { gfni_reverse_bits(&mut v) };
        v == want
    })
}

// ---- AVX-512 VBMI2: byte expand (sub-byte unpack primitive) ----

/// AVX-512-VBMI2 via CPUID.(EAX=7,ECX=0):ECX[6] (+ F/BW).
fn vbmi2_supported() -> bool {
    let l0 = unsafe { __cpuid_count(7, 0) };
    ((l0.ebx >> 16) & 1 != 0) && ((l0.ebx >> 30) & 1 != 0) && ((l0.ecx >> 6) & 1 != 0)
}

fn expand_bytes_scalar(src: &[u8], mask: u64, out: &mut [u8; 64]) {
    *out = [0u8; 64];
    let mut si = 0usize;
    for i in 0..64 {
        if (mask >> i) & 1 != 0 {
            out[i] = src.get(si).copied().unwrap_or(0);
            si += 1;
        }
    }
}

/// VBMI2 byte-expand: scatter the low `popcount(mask)` bytes of `src` to the
/// set bit positions of `mask` (the primitive sub-byte formats unpack with).
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi2")]
unsafe fn expand_bytes_vbmi2(src: &[u8], mask: u64, out: &mut [u8; 64]) {
    let mut buf = [0u8; 64];
    let n = src.len().min(64);
    buf[..n].copy_from_slice(&src[..n]);
    let sv = _mm512_loadu_si512(buf.as_ptr() as *const __m512i);
    let r = _mm512_maskz_expand_epi8(mask, sv);
    _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, r);
}

/// Whether the VBMI2 byte-expand is usable — cached + self-checked. Write-blind.
#[allow(dead_code)]
pub fn vbmi2_expand_usable() -> bool {
    static U: OnceLock<bool> = OnceLock::new();
    *U.get_or_init(|| {
        if !vbmi2_supported() {
            return false;
        }
        let src: Vec<u8> = (1..=40u8).collect();
        let mask = 0xDEAD_BEEF_1234_5678u64;
        let mut got = [0u8; 64];
        let mut want = [0u8; 64];
        expand_bytes_scalar(&src, mask, &mut want);
        // SAFETY: vbmi2_supported() checked above.
        unsafe { expand_bytes_vbmi2(&src, mask, &mut got) };
        got == want
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitops_ext_match_reference_or_skip() {
        // GFNI runs on the dev box (AVX2-GFNI); the AVX-512 ones SKIP here.
        if gfni_supported() {
            assert!(gfni_reverse_usable(), "GFNI present but bit-reverse self-check failed");
        } else {
            eprintln!("gfni: not supported — SKIP");
        }
        if vpopcntdq_supported() {
            assert!(binary_dot_usable(), "VPOPCNTDQ present but binary-dot self-check failed");
        } else {
            eprintln!("vpopcntdq: not supported — SKIP");
        }
        if vbmi2_supported() {
            assert!(vbmi2_expand_usable(), "VBMI2 present but expand self-check failed");
        } else {
            eprintln!("vbmi2: not supported — SKIP");
        }
    }
}
