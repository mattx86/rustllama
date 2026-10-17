// rustc, deliberate: unused_unsafe — explicit `unsafe` blocks kept inside the
// `#[target_feature]` unsafe fns for edition-2024 `unsafe_op_in_unsafe_fn` forward-compat.
#![allow(unused_unsafe)]

//! AVX-512-BF16 GEMM — the AMD / older-Intel prefill-GEMM accelerator, the
//! SIMD companion to the AMX tile path (`amx.rs`). AMD has no AMX, so on
//! Zen 4/5 (and Intel server/HEDT without AMX) AVX-512-BF16's `vdpbf16ps` is
//! the bf16 matmul path.
//!
//! WRITE-BLIND: there is no AVX-512 (let alone AVX-512-BF16) on the author's
//! box (consumer Raptor Lake), so this is authored blind and ships behind the
//! same fail-closed runtime self-check as AMX ([`avx512bf16_gemm_usable`]):
//! it runs the kernel against a scalar bf16 reference once and enables it only
//! where the output matches, so a wrong encoding degrades to the scalar GEMM
//! instead of corrupting output, and a non-AVX512-BF16 CPU never reaches a
//! `vdpbf16ps`.
//!
//! Numerics: bf16 operands (8-bit mantissa), LOSSY vs the f16·f32 reference —
//! same trade as AMX and the GPU XMX/Hopper bf16 tensor-core paths, so it is
//! confined to prefill-shaped GEMMs.
//!
//! `C[M,N] = W[M,K]·X[K,N]`. X is transposed+packed to `xT[n][k]` once (so each
//! output column is a contiguous K-vector shared across all M rows), W rows are
//! widened f16→f32 per row, both are converted to bf16 with `cvtne2ps_pbh`, and
//! `dpbf16_ps` accumulates pairwise products into 16 f32 lanes, horizontally
//! reduced per output. A `K % 32` tail runs in scalar bf16.

#![cfg(target_arch = "x86_64")]

use half::{bf16, f16};
use std::arch::x86_64::*;
use std::sync::OnceLock;

/// AVX-512-BF16 (+ the F/BW it needs) via CPUID: 7.0 EBX[16]=AVX512F,
/// EBX[30]=AVX512BW; 7.1 EAX[5]=AVX512-BF16. (The `is_x86_feature_detected!`
/// arms for these are nightly-gated — same trap as AMX — so read CPUID.)
fn avx512bf16_supported() -> bool {
    // SAFETY: CPUID leaf 7 exists on all x86-64.
    let l0 = unsafe { __cpuid_count(7, 0) };
    let f = (l0.ebx >> 16) & 1 != 0;
    let bw = (l0.ebx >> 30) & 1 != 0;
    if !(f && bw) {
        return false;
    }
    let l1 = unsafe { __cpuid_count(7, 1) };
    (l1.eax >> 5) & 1 != 0
}

/// Transpose `x[K,N]` (row-major) into `xT[N,K]` (row-major) so each output
/// column n is a contiguous K-vector.
fn pack_xt(x: &[f32], n: usize, k: usize, xt: &mut [f32]) {
    for kk in 0..k {
        let row = &x[kk * n..kk * n + n];
        for (nn, &v) in row.iter().enumerate() {
            xt[nn * k + kk] = v;
        }
    }
}

/// Scalar bf16 reference GEMM (truncate both operands to bf16), for the
/// self-check — isolates kernel/encoding correctness from bf16 rounding.
fn gemm_bf16_ref(w: &[f16], x: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                let wv = bf16::from_f32(w[i * k + p].to_f32()).to_f32();
                let xv = bf16::from_f32(x[p * n + j]).to_f32();
                acc += wv * xv;
            }
            c[i * n + j] = acc;
        }
    }
}

/// AVX-512-BF16 GEMM. `xt` is the pre-packed transpose (`xt[n*k + p] =
/// x[p*n + n_idx]`); `wrow` scratch holds the widened f16→f32 current W row.
#[target_feature(enable = "avx512f,avx512bw,avx512bf16")]
unsafe fn gemm_bf16_avx512_inner(
    w: &[f16],
    xt: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let k32 = k - (k % 32);
    let mut wrow = vec![0f32; k];
    for i in 0..m {
        // Widen this W row f16 -> f32 once; reused across all N columns.
        for (p, wv) in wrow.iter_mut().enumerate() {
            *wv = w[i * k + p].to_f32();
        }
        for j in 0..n {
            let xcol = &xt[j * k..j * k + k];
            let mut acc = _mm512_setzero_ps();
            let mut p = 0usize;
            while p < k32 {
                let w_lo = _mm512_loadu_ps(wrow.as_ptr().add(p));
                let w_hi = _mm512_loadu_ps(wrow.as_ptr().add(p + 16));
                let x_lo = _mm512_loadu_ps(xcol.as_ptr().add(p));
                let x_hi = _mm512_loadu_ps(xcol.as_ptr().add(p + 16));
                // cvtne2ps_pbh(hi, lo): lanes 0..16 = lo, 16..32 = hi → 32 bf16
                // in k-order for both operands, so dpbf16's pairwise products
                // line up on matching k.
                let w_bh = _mm512_cvtne2ps_pbh(w_hi, w_lo);
                let x_bh = _mm512_cvtne2ps_pbh(x_hi, x_lo);
                acc = _mm512_dpbf16_ps(acc, w_bh, x_bh);
                p += 32;
            }
            let mut sum = _mm512_reduce_add_ps(acc);
            // Scalar bf16 tail for the K % 32 remainder.
            while p < k {
                let wv = bf16::from_f32(wrow[p]).to_f32();
                let xv = bf16::from_f32(xcol[p]).to_f32();
                sum += wv * xv;
                p += 1;
            }
            c[i * n + j] = sum;
        }
    }
}

/// AVX-512-BF16 GEMM public entry (same shape contract as `gemm_f16_w_f32_a`).
/// Caller must have confirmed [`avx512bf16_gemm_usable`].
pub fn gemm_bf16_avx512(w: &[f16], x: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let mut xt = vec![0f32; k * n];
    pack_xt(x, n, k, &mut xt);
    // SAFETY: gated on avx512bf16_supported() via avx512bf16_gemm_usable().
    unsafe { gemm_bf16_avx512_inner(w, &xt, c, m, n, k) };
}

/// Whether the AVX-512-BF16 GEMM is usable — cached. True only when the CPU has
/// AVX-512-BF16 AND the kernel reproduces the scalar bf16 reference within
/// tolerance on a probe (fail-closed). A non-AVX512-BF16 CPU returns false and
/// no `vdpbf16ps` is ever executed.
pub fn avx512bf16_gemm_usable() -> bool {
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| {
        if !avx512bf16_supported() {
            return false;
        }
        // Probe crossing the 32-wide SIMD step with a K%32 tail (K=40).
        let (m, n, k) = (6usize, 5usize, 40usize);
        let w: Vec<f16> = (0..m * k)
            .map(|i| f16::from_f32((((i as i64 * 2_654_435_761) % 97) as f32 - 48.0) * 0.03))
            .collect();
        let x: Vec<f32> = (0..k * n)
            .map(|i| (((i as i64 * 40_503) % 89) as f32 - 44.0) * 0.02)
            .collect();
        let mut got = vec![0f32; m * n];
        let mut want = vec![0f32; m * n];
        gemm_bf16_ref(&w, &x, &mut want, m, n, k);
        gemm_bf16_avx512(&w, &x, &mut got, m, n, k);
        got.iter().zip(&want).all(|(a, b)| {
            let denom = b.abs().max(1e-4);
            (a - b).abs() / denom <= 0.02
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx512bf16_gemm_matches_reference_or_skips() {
        if !avx512bf16_supported() {
            eprintln!("avx512bf16: not supported on this host — SKIP");
            return;
        }
        assert!(
            avx512bf16_gemm_usable(),
            "AVX-512-BF16 present but self-check failed"
        );
    }
}
