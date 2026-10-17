//! VNNI int8 (W8A8) matvec for Q8_0 weights — an Intel+AMD CPU extension.
//!
//! Q8_0 weights are already int8, so a VNNI dot-product instruction accumulates
//! `Σ wq·xq` four products at a time per 32-bit lane after the f32 activation is
//! quantized to int8 once (shared across all output rows). This is **W8A8 —
//! LOSSY** vs the default f32-activation path, so it is OPT-IN
//! (`set_vnni_w8a8`, default off) like the CUDA W4A8 matvec, and additionally
//! gated on a runtime self-check ([`vnni_q8_0_usable`]) that confirms the
//! vectorized kernel reproduces a scalar int8 reference bit-for-bit.
//!
//! Three instruction forms, picked by CPUID (the `is_x86_feature_detected!`
//! arms for these are nightly-gated, same trap as AMX, so we read CPUID):
//! - **AVX-VNNI-INT8** `vpdpbssd` (signed×signed) — the cleanest: the activation
//!   is quantized to signed i8 and dotted directly, no bias correction.
//! - **AVX-512-VNNI** / **AVX-VNNI** `vpdpbusd` (unsigned×signed) — the
//!   activation i8 is XOR-biased to u8 (`xq + 128`) at load and the +128 is
//!   corrected out with `Σ wq` (a second dot vs all-ones): `dpbusd(xq+128, wq)
//!   − 128·Σwq = Σ xq·wq`.
//! All three compute the same `Σ xq·wq`, so one scalar i8 reference validates
//! whichever is active.

#![cfg(target_arch = "x86_64")]

use half::f16;
use std::arch::x86_64::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

const QK: usize = 32;
const BLOCK_BYTES: usize = 34;

/// Opt-in enable for the lossy W8A8 VNNI path (default OFF — the lossless f32
/// path stays the default). A coherence-gated autotune stage can flip this on
/// per model; until then it is an explicit opt-in.
static VNNI_W8A8: AtomicBool = AtomicBool::new(false);

/// Enable/disable the W8A8 VNNI matvec for Q8_0 (opt-in; lossy).
pub fn set_vnni_w8a8(on: bool) {
    VNNI_W8A8.store(on, Ordering::Relaxed);
}

/// Whether the W8A8 VNNI path has been opted into.
pub fn vnni_w8a8_enabled() -> bool {
    VNNI_W8A8.load(Ordering::Relaxed)
}

/// AVX-VNNI (VEX 256-bit `vpdpbusd`) via CPUID.(EAX=7,ECX=1):EAX[4].
fn has_avxvnni() -> bool {
    let leaf = unsafe { __cpuid_count(7, 1) };
    (leaf.eax >> 4) & 1 != 0
}

/// AVX-VNNI-INT8 (`vpdpbssd`, signed×signed) via CPUID.(EAX=7,ECX=1):EDX[4].
fn has_avxvnniint8() -> bool {
    let leaf = unsafe { __cpuid_count(7, 1) };
    (leaf.edx >> 4) & 1 != 0
}

/// AVX-512-VNNI (+ F/BW/VL for the 256-bit EVEX `vpdpbusd`) via
/// CPUID.(EAX=7,ECX=0): EBX[16]/[30]/[31] + ECX[11].
fn has_avx512vnni() -> bool {
    let leaf = unsafe { __cpuid_count(7, 0) };
    let f = (leaf.ebx >> 16) & 1 != 0;
    let bw = (leaf.ebx >> 30) & 1 != 0;
    let vl = (leaf.ebx >> 31) & 1 != 0;
    let vnni = (leaf.ecx >> 11) & 1 != 0;
    f && bw && vl && vnni
}

/// Which VNNI instruction to use, cached. Prefer AVX-VNNI-INT8 (`vpdpbssd`, no
/// bias correction) > AVX-512-VNNI > AVX-VNNI > none.
#[derive(Clone, Copy, PartialEq)]
enum VnniKind {
    None,
    Avx256,
    Avx512,
    Avx256Int8,
}

fn vnni_kind() -> VnniKind {
    static K: OnceLock<VnniKind> = OnceLock::new();
    *K.get_or_init(|| {
        if has_avxvnniint8() {
            VnniKind::Avx256Int8
        } else if has_avx512vnni() {
            VnniKind::Avx512
        } else if has_avxvnni() {
            VnniKind::Avx256
        } else {
            VnniKind::None
        }
    })
}

/// Synthesize a valid Q8_0 weight (m rows × k, row-major 32-blocks: f16 scale +
/// 32 i8) for the self-check / tests — no gguf-encoder dependency. `pub(crate)`
/// so the AMX int8 prefill self-check / test reuses the same synth path.
pub(crate) fn synth_q8_0(m: usize, k: usize, seed: i64) -> Vec<u8> {
    let bpr = k / QK;
    let mut w = vec![0u8; m * bpr * BLOCK_BYTES];
    for r in 0..m {
        for b in 0..bpr {
            let off = (r * bpr + b) * BLOCK_BYTES;
            let d = f16::from_f32(0.02 + 0.001 * ((b % 7) as f32));
            let db = d.to_le_bytes();
            w[off] = db[0];
            w[off + 1] = db[1];
            for j in 0..QK {
                let v = (((r * k + b * QK + j) as i64 * seed) % 255 - 127) as i8;
                w[off + 2 + j] = v as u8;
            }
        }
    }
    w
}

/// Quantize the f32 activation to signed int8 per 32-block (symmetric, scale =
/// absmax/127), done ONCE per matvec and reused across every output row.
///
/// `pub(crate)` so the AMX int8 prefill GEMM ([`crate::amx::gemm_q8_0_w8a8_amx`])
/// quantizes activations with the IDENTICAL per-block scheme — a shared quant means
/// the AMX kernel grades bit-for-bit against [`matvec_q8_0_w8a8_scalar`], isolating
/// tile-encoding correctness from the (shared) activation-quant loss.
pub(crate) fn quantize_activation_i8(x: &[f32], k: usize, xq: &mut [i8], xscales: &mut [f32]) {
    let blocks = k / QK;
    for b in 0..blocks {
        let chunk = &x[b * QK..(b + 1) * QK];
        let amax = chunk.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        xscales[b] = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        for (j, &v) in chunk.iter().enumerate() {
            xq[b * QK + j] = (v * inv).round().clamp(-127.0, 127.0) as i32 as i8;
        }
    }
}

/// Horizontal sum of the 8 i32 lanes of a `__m256i`.
#[target_feature(enable = "avx,avx2")]
unsafe fn hsum_i32x8(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_hadd_epi32(s, s);
    let s = _mm_hadd_epi32(s, s);
    _mm_cvtsi128_si32(s)
}

/// Scalar int8 reference: `Σ xq·wq` per block, scaled by `d·x_scale`. The exact
/// value every VNNI form computes, so a match validates the vectorized kernel.
/// `pub(crate)`: also the parity reference for the AMX int8 prefill GEMM (one row
/// per token), since both share the per-block accumulation order (f32 add per
/// block) → the AMX tile result matches this to within int rounding.
pub(crate) fn matvec_q8_0_w8a8_scalar(w_bytes: &[u8], xq: &[i8], xscales: &[f32], out: &mut [f32], m: usize, k: usize) {
    let blocks_per_row = k / QK;
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let mut dot: i32 = 0;
            for j in 0..QK {
                dot += (xq[b * QK + j] as i32) * (w_bytes[off + 2 + j] as i8 as i32);
            }
            acc += d * xscales[b] * dot as f32;
        }
        out[i] = acc;
    }
}

/// `vpdpbusd` kernel body (unsigned×signed): XOR-bias the i8 activation to u8
/// (`xq + 128`) and correct the +128 with `Σ wq`. Shared by the AVX-VNNI and
/// AVX-512-VNNI forms; the `DP` closure picks the encoding.
#[inline(always)]
unsafe fn dpbusd_row<DP: Fn(__m256i, __m256i, __m256i) -> __m256i>(
    w_bytes: &[u8],
    xq: &[i8],
    xscales: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    dp: DP,
) {
    let blocks_per_row = k / QK;
    let ones = _mm256_set1_epi8(1);
    let bias = _mm256_set1_epi8(-128); // XOR 0x80: i8 -> biased u8 (+128)
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let wq = _mm256_loadu_si256(w_bytes.as_ptr().add(off + 2) as *const __m256i);
            let xi = _mm256_loadu_si256(xq.as_ptr().add(b * QK) as *const __m256i);
            let a = _mm256_xor_si256(xi, bias); // u8 = xq + 128
            let dot = dp(_mm256_setzero_si256(), a, wq);
            let wsum = dp(_mm256_setzero_si256(), ones, wq);
            let corrected = hsum_i32x8(dot) - 128 * hsum_i32x8(wsum);
            acc += d * xscales[b] * corrected as f32;
        }
        out[i] = acc;
    }
}

#[target_feature(enable = "avx,avx2,avxvnni")]
unsafe fn matvec_q8_0_w8a8_avxvnni(w_bytes: &[u8], xq: &[i8], xscales: &[f32], out: &mut [f32], m: usize, k: usize) {
    dpbusd_row(w_bytes, xq, xscales, out, m, k, |s, a, b| {
        _mm256_dpbusd_avx_epi32(s, a, b)
    });
}

#[target_feature(enable = "avx,avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn matvec_q8_0_w8a8_avx512vnni(w_bytes: &[u8], xq: &[i8], xscales: &[f32], out: &mut [f32], m: usize, k: usize) {
    dpbusd_row(w_bytes, xq, xscales, out, m, k, |s, a, b| {
        _mm256_dpbusd_epi32(s, a, b)
    });
}

/// `vpdpbssd` kernel (signed×signed) — the activation i8 is dotted directly, no
/// bias correction. WRITE-BLIND (AVX-VNNI-INT8 is newer than the author's box).
#[target_feature(enable = "avx,avx2,avxvnniint8")]
unsafe fn matvec_q8_0_w8a8_avxvnniint8(w_bytes: &[u8], xq: &[i8], xscales: &[f32], out: &mut [f32], m: usize, k: usize) {
    let blocks_per_row = k / QK;
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let wq = _mm256_loadu_si256(w_bytes.as_ptr().add(off + 2) as *const __m256i);
            let xi = _mm256_loadu_si256(xq.as_ptr().add(b * QK) as *const __m256i);
            let dot = _mm256_dpbssd_epi32(_mm256_setzero_si256(), xi, wq);
            acc += d * xscales[b] * hsum_i32x8(dot) as f32;
        }
        out[i] = acc;
    }
}

/// Run the active VNNI kernel on the pre-quantized i8 activation.
unsafe fn run_vnni_kernel(w_bytes: &[u8], xq: &[i8], xscales: &[f32], out: &mut [f32], m: usize, k: usize) {
    match vnni_kind() {
        VnniKind::Avx256Int8 => matvec_q8_0_w8a8_avxvnniint8(w_bytes, xq, xscales, out, m, k),
        VnniKind::Avx512 => matvec_q8_0_w8a8_avx512vnni(w_bytes, xq, xscales, out, m, k),
        VnniKind::Avx256 => matvec_q8_0_w8a8_avxvnni(w_bytes, xq, xscales, out, m, k),
        VnniKind::None => matvec_q8_0_w8a8_scalar(w_bytes, xq, xscales, out, m, k),
    }
}

/// W8A8 Q8_0 matvec (public entry): quantize the activation to i8 once, then run
/// the active VNNI kernel. Caller must have confirmed [`vnni_q8_0_usable`].
pub fn matvec_q8_0_w8a8(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(k % QK, 0);
    let blocks = k / QK;
    let mut xq = vec![0i8; k];
    let mut xscales = vec![0f32; blocks];
    quantize_activation_i8(x, k, &mut xq, &mut xscales);
    // SAFETY: a VNNI kind was confirmed present via vnni_q8_0_usable().
    unsafe { run_vnni_kernel(w_bytes, &xq, &xscales, out, m, k) };
}

/// Whether the W8A8 VNNI Q8_0 matvec is usable — cached. True only when the CPU
/// has some VNNI form AND the active kernel reproduces the scalar int8 reference
/// bit-for-bit on a probe (integer math is exact → a mismatch means a wrong
/// encoding, fail closed).
pub fn vnni_q8_0_usable() -> bool {
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| {
        if vnni_kind() == VnniKind::None {
            return false;
        }
        let (m, k) = (5usize, 128usize);
        let w = synth_q8_0(m, k, 2_654_435_761);
        let x: Vec<f32> = (0..k).map(|j| ((j % 17) as f32 - 8.0) * 0.07).collect();
        let mut xq = vec![0i8; k];
        let mut xscales = vec![0f32; k / QK];
        quantize_activation_i8(&x, k, &mut xq, &mut xscales);
        let mut got = vec![0f32; m];
        let mut want = vec![0f32; m];
        matvec_q8_0_w8a8_scalar(&w, &xq, &xscales, &mut want, m, k);
        // SAFETY: a VNNI kind is present (checked above).
        unsafe { run_vnni_kernel(&w, &xq, &xscales, &mut got, m, k) };
        got == want
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vnni_w8a8_matches_scalar_or_skips() {
        if vnni_kind() == VnniKind::None {
            eprintln!("vnni: no VNNI form on this host — SKIP");
            return;
        }
        assert!(vnni_q8_0_usable(), "VNNI present but self-check failed");
        let (m, k) = (37usize, 256usize);
        let w = synth_q8_0(m, k, 40_503);
        let x: Vec<f32> = (0..k).map(|j| ((j % 29) as f32 - 14.0) * 0.03).collect();
        let mut xq = vec![0i8; k];
        let mut xscales = vec![0f32; k / QK];
        quantize_activation_i8(&x, k, &mut xq, &mut xscales);
        let mut got = vec![0f32; m];
        let mut want = vec![0f32; m];
        matvec_q8_0_w8a8_scalar(&w, &xq, &xscales, &mut want, m, k);
        unsafe { run_vnni_kernel(&w, &xq, &xscales, &mut got, m, k) };
        assert_eq!(got, want, "VNNI W8A8 != scalar int8 reference");
    }
}
