// deliberate: incompatible_msrv — this path uses the AVX-512-FP16 intrinsics
// (`__m512h`, `_mm512_fmadd_ph`, …) that stabilized after the declared MSRV; it is
// runtime-feature-gated and self-check-gated, so it only executes on capable CPUs with
// a new-enough compiler. unused_unsafe — explicit `unsafe` blocks kept for edition-2024
// `unsafe_op_in_unsafe_fn` forward-compat; missing_transmute_annotations — the intrinsic
// register transmutes read clearly from context here.
#![allow(
    clippy::incompatible_msrv,
    unused_unsafe,
    clippy::missing_transmute_annotations
)]

//! AVX-512-FP16 native-f16 GEMM — a completeness CPU-extension path.
//!
//! CAVEAT (why it is not wired into the hot path): the throughput win of
//! AVX-512-FP16 comes from f16 FMA (`vfmadd…ph`), which **accumulates in f16**.
//! Over a long K dot that loses precision badly, so for LLM matmul the bf16
//! paths here (AMX / AVX-512-BF16), which accumulate in f32, are usually more
//! accurate. This kernel is therefore built for completeness, self-check-gated
//! against an f16-accumulate reference (so the check measures the kernel, not
//! the f16-vs-f32 precision gap), and left available but **not** auto-selected
//! in `gemm_f16_w_f32_a`.
//!
//! Stable-Rust note: the `f16` primitive used by `_mm512_loadu_ph` /
//! `_mm512_reduce_add_ph` is unstable, so loads go through `__m512i` +
//! transmute to `__m512h`, and the reduce converts the f16 accumulator to f32
//! lanes with `_mm512_cvtph_ps` — no `core::f16` is named.

#![cfg(target_arch = "x86_64")]

use half::f16;
use std::arch::x86_64::*;
use std::sync::OnceLock;

/// AVX-512-FP16 via CPUID.(EAX=7,ECX=0):EDX[23] (+ F for the base ops).
fn avx512fp16_supported() -> bool {
    let l0 = unsafe { __cpuid_count(7, 0) };
    let f = (l0.ebx >> 16) & 1 != 0;
    let fp16 = (l0.edx >> 23) & 1 != 0;
    f && fp16
}

fn pack_xt_f16(x: &[f32], n: usize, k: usize, xt: &mut [u16]) {
    for kk in 0..k {
        let row = &x[kk * n..kk * n + n];
        for (nn, &v) in row.iter().enumerate() {
            xt[nn * k + kk] = f16::from_f32(v).to_bits();
        }
    }
}

/// Scalar f16-accumulate reference (matches the kernel's f16 accumulation, so a
/// match validates the SIMD encoding, not the f16 precision gap).
fn gemm_f16_accum_ref(w: &[f16], x: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = f16::from_f32(0.0);
            for p in 0..k {
                let prod = f16::from_f32(w[i * k + p].to_f32() * f16::from_f32(x[p * n + j]).to_f32());
                acc = f16::from_f32(acc.to_f32() + prod.to_f32());
            }
            c[i * n + j] = acc.to_f32();
        }
    }
}

/// Reduce a 32-lane f16 accumulator (held as `__m512i` bits) to f32 via two
/// `cvtph_ps` halves — avoids naming `core::f16`.
#[target_feature(enable = "avx512f")]
unsafe fn reduce_ph_bits_to_f32(acc_bits: __m512i) -> f32 {
    let lo = _mm512_castsi512_si256(acc_bits);
    let hi = _mm512_extracti64x4_epi64::<1>(acc_bits);
    let flo = _mm512_cvtph_ps(lo);
    let fhi = _mm512_cvtph_ps(hi);
    _mm512_reduce_add_ps(_mm512_add_ps(flo, fhi))
}

/// AVX-512-FP16 GEMM (native f16 FMA, f16 accumulation). `xt` is the f16-packed
/// transpose (`xt[n*k + p]` bits = f16(x[p*n + n_idx])).
#[target_feature(enable = "avx512f,avx512bw,avx512fp16")]
unsafe fn gemm_f16_avx512_inner(w: &[f16], xt: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
    let k32 = k - (k % 32);
    for i in 0..m {
        let wrow = &w[i * k..i * k + k];
        for j in 0..n {
            let xcol = &xt[j * k..j * k + k];
            let mut acc: __m512h = std::mem::transmute(_mm512_setzero_si512());
            let mut p = 0usize;
            while p < k32 {
                let w_h: __m512h =
                    std::mem::transmute(_mm512_loadu_si512(wrow.as_ptr().add(p) as *const __m512i));
                let x_h: __m512h =
                    std::mem::transmute(_mm512_loadu_si512(xcol.as_ptr().add(p) as *const __m512i));
                acc = _mm512_fmadd_ph(w_h, x_h, acc);
                p += 32;
            }
            let mut sum = reduce_ph_bits_to_f32(std::mem::transmute(acc));
            while p < k {
                let prod = f16::from_f32(wrow[p].to_f32() * f16::from_bits(xcol[p]).to_f32());
                sum += prod.to_f32();
                p += 1;
            }
            c[i * n + j] = sum;
        }
    }
}

/// AVX-512-FP16 GEMM (same shape contract as `gemm_f16_w_f32_a`). Caller must
/// have confirmed [`avx512fp16_gemm_usable`].
#[allow(dead_code)]
pub fn gemm_f16_avx512fp16(w: &[f16], x: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let mut xt = vec![0u16; k * n];
    pack_xt_f16(x, n, k, &mut xt);
    // SAFETY: gated on avx512fp16_supported() via avx512fp16_gemm_usable().
    unsafe { gemm_f16_avx512_inner(w, &xt, c, m, n, k) };
}

/// Whether the AVX-512-FP16 GEMM is usable — cached. True only with AVX-512-FP16
/// AND a match to the f16-accumulate reference (fail-closed). A non-FP16 CPU
/// returns false and no `vfmadd…ph` runs.
#[allow(dead_code)]
pub fn avx512fp16_gemm_usable() -> bool {
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| {
        if !avx512fp16_supported() {
            return false;
        }
        let (m, n, k) = (6usize, 5usize, 40usize);
        let w: Vec<f16> = (0..m * k).map(|i| f16::from_f32(((i % 97) as f32 - 48.0) * 0.03)).collect();
        let x: Vec<f32> = (0..k * n).map(|i| ((i % 89) as f32 - 44.0) * 0.02).collect();
        let mut got = vec![0f32; m * n];
        let mut want = vec![0f32; m * n];
        gemm_f16_accum_ref(&w, &x, &mut want, m, n, k);
        gemm_f16_avx512fp16(&w, &x, &mut got, m, n, k);
        got.iter().zip(&want).all(|(a, b)| {
            let denom = b.abs().max(1e-3);
            (a - b).abs() / denom <= 0.05
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx512fp16_gemm_matches_reference_or_skips() {
        if !avx512fp16_supported() {
            eprintln!("avx512fp16: not supported on this host — SKIP");
            return;
        }
        assert!(avx512fp16_gemm_usable(), "AVX-512-FP16 present but self-check failed");
    }
}
