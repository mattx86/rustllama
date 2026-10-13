//! AVX-VNNI int8 (W8A8) matvec for Q8_0 weights — an Intel+AMD CPU
//! performance extension.
//!
//! Q8_0 weights are already int8, so the VNNI dot-product instruction
//! (`vpdpbusd`) is a natural fit: quantize the f32 activation to int8 once
//! (shared across all output rows), then accumulate `Σ wq·xq` four products at
//! a time per 32-bit lane. This is **W8A8 — LOSSY** vs the default f32-
//! activation path (the activation picks up an int8 quantization step), so it
//! is OPT-IN (`set_vnni_w8a8`, default off) exactly like the CUDA W4A8 matvec,
//! and additionally gated on a runtime self-check ([`vnni_q8_0_usable`]) that
//! confirms the vectorized kernel reproduces a scalar W8A8 reference.
//!
//! `vpdpbusd` multiplies an UNSIGNED byte by a SIGNED byte, so the activation
//! is stored biased to u8 (`xq + 128`) and the +128 is corrected out with
//! `Σ wq` (a second `vpdpbusd` against an all-ones vector): the recovered dot
//! is `dpbusd((xq+128), wq) − 128·Σwq = Σ xq·wq`.
//!
//! Detection is via raw CPUID (the `is_x86_feature_detected!("avxvnni")` macro
//! arm is nightly-gated, same trap as AMX). This increment implements the
//! VEX-encoded 256-bit **AVX-VNNI** path (Intel Alder Lake+/AMD Zen5, and the
//! one validatable on the author's 13900H); the 512-bit **AVX-512-VNNI** path
//! (AMD Zen4/5, Intel server) is a follow-up.

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

/// AVX-VNNI (VEX 256-bit `vpdpbusd`) presence via CPUID.(EAX=7,ECX=1):EAX[4].
/// The `is_x86_feature_detected!("avxvnni")` arm is nightly-gated, so read the
/// bit directly (CPUID is always available on x86-64).
fn has_avxvnni() -> bool {
    // SAFETY: CPUID leaf 7 exists on all x86-64.
    let leaf = unsafe { __cpuid_count(7, 1) };
    (leaf.eax >> 4) & 1 != 0
}

/// AVX-512-VNNI (+ the F/BW/VL it needs for the 256-bit EVEX `vpdpbusd`) via
/// CPUID.(EAX=7,ECX=0): EBX[16]=AVX512F, EBX[30]=AVX512BW, EBX[31]=AVX512VL,
/// ECX[11]=AVX512VNNI. AMD Zen 4/5 + Intel server have this even where the VEX
/// AVX-VNNI bit is absent.
fn has_avx512vnni() -> bool {
    // SAFETY: CPUID leaf 7 exists on all x86-64.
    let leaf = unsafe { __cpuid_count(7, 0) };
    let f = (leaf.ebx >> 16) & 1 != 0;
    let bw = (leaf.ebx >> 30) & 1 != 0;
    let vl = (leaf.ebx >> 31) & 1 != 0;
    let vnni = (leaf.ecx >> 11) & 1 != 0;
    f && bw && vl && vnni
}

/// Which VNNI instruction set to use, cached. Prefer AVX-512-VNNI (EVEX) where
/// present, else the VEX AVX-VNNI, else none.
#[derive(Clone, Copy, PartialEq)]
enum VnniKind {
    None,
    Avx256,
    Avx512,
}

fn vnni_kind() -> VnniKind {
    static K: OnceLock<VnniKind> = OnceLock::new();
    *K.get_or_init(|| {
        if has_avx512vnni() {
            VnniKind::Avx512
        } else if has_avxvnni() {
            VnniKind::Avx256
        } else {
            VnniKind::None
        }
    })
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

/// Quantize the f32 activation to biased-u8 int8 per 32-block (symmetric,
/// scale = absmax/127), writing `xqu8[j] = clamp(round(x/scale), -127, 127) +
/// 128` and the per-block `xscales`. Done ONCE per matvec and reused across
/// every output row.
fn quantize_activation(x: &[f32], k: usize, xqu8: &mut [u8], xscales: &mut [f32]) {
    let blocks = k / QK;
    for b in 0..blocks {
        let chunk = &x[b * QK..(b + 1) * QK];
        let amax = chunk.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        xscales[b] = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        for (j, &v) in chunk.iter().enumerate() {
            let q = (v * inv).round().clamp(-127.0, 127.0) as i32;
            xqu8[b * QK + j] = (q + 128) as u8;
        }
    }
}

/// Synthesize a valid Q8_0 weight (m rows × k, row-major 32-blocks: f16 scale
/// + 32 i8) for the self-check / tests — no gguf-encoder dependency; the check
/// only needs identical bytes fed to both kernels.
fn synth_q8_0(m: usize, k: usize, seed: i64) -> Vec<u8> {
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

/// Scalar W8A8 reference — the EXACT integer math the VNNI kernel does
/// (`Σ (xq+128)·wq − 128·Σwq`, scaled by `d·x_scale` per block). Used by the
/// self-check so a match isolates the vectorized dpbusd + bias correction from
/// quantization loss.
fn matvec_q8_0_w8a8_scalar(w_bytes: &[u8], xqu8: &[u8], xscales: &[f32], out: &mut [f32], m: usize, k: usize) {
    let blocks_per_row = k / QK;
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let mut dot: i32 = 0;
            let mut wsum: i32 = 0;
            for j in 0..QK {
                let wq = w_bytes[off + 2 + j] as i8 as i32;
                let a = xqu8[b * QK + j] as i32;
                dot += a * wq;
                wsum += wq;
            }
            acc += d * xscales[b] * (dot - 128 * wsum) as f32;
        }
        out[i] = acc;
    }
}

/// AVX-VNNI 256-bit W8A8 Q8_0 matvec. One `vpdpbusd` handles a whole 32-wide
/// Q8_0 block (8 i32 lane-sums); a second against all-ones yields `Σwq` for the
/// +128 bias correction. Per-block scale `d·x_scale` folds in after the reduce.
#[target_feature(enable = "avx,avx2,avxvnni")]
unsafe fn matvec_q8_0_w8a8_avxvnni(
    w_bytes: &[u8],
    xqu8: &[u8],
    xscales: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    let blocks_per_row = k / QK;
    let ones = _mm256_set1_epi8(1);
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let wq = _mm256_loadu_si256(w_bytes.as_ptr().add(off + 2) as *const __m256i);
            let a = _mm256_loadu_si256(xqu8.as_ptr().add(b * QK) as *const __m256i);
            let dot = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), a, wq);
            let wsum = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ones, wq);
            let corrected = hsum_i32x8(dot) - 128 * hsum_i32x8(wsum);
            acc += d * xscales[b] * corrected as f32;
        }
        out[i] = acc;
    }
}

/// AVX-512-VNNI W8A8 Q8_0 matvec — identical algorithm to the AVX-VNNI path,
/// using the EVEX-encoded 256-bit `_mm256_dpbusd_epi32` so it runs on
/// AVX-512-VNNI CPUs that lack the VEX AVX-VNNI bit (AMD Zen 4/5, Intel
/// server). WRITE-BLIND (no AVX-512 on the author's box); a 512-bit-wide
/// two-block form is a later perf follow-up.
#[target_feature(enable = "avx,avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn matvec_q8_0_w8a8_avx512vnni(
    w_bytes: &[u8],
    xqu8: &[u8],
    xscales: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    let blocks_per_row = k / QK;
    let ones = _mm256_set1_epi8(1);
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let wq = _mm256_loadu_si256(w_bytes.as_ptr().add(off + 2) as *const __m256i);
            let a = _mm256_loadu_si256(xqu8.as_ptr().add(b * QK) as *const __m256i);
            let dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), a, wq);
            let wsum = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones, wq);
            let corrected = hsum_i32x8(dot) - 128 * hsum_i32x8(wsum);
            acc += d * xscales[b] * corrected as f32;
        }
        out[i] = acc;
    }
}

/// Run the active VNNI kernel on pre-quantized activations. Caller ensures a
/// VNNI kind is present ([`vnni_q8_0_usable`]).
unsafe fn run_vnni_kernel(w_bytes: &[u8], xqu8: &[u8], xscales: &[f32], out: &mut [f32], m: usize, k: usize) {
    match vnni_kind() {
        VnniKind::Avx512 => matvec_q8_0_w8a8_avx512vnni(w_bytes, xqu8, xscales, out, m, k),
        VnniKind::Avx256 => matvec_q8_0_w8a8_avxvnni(w_bytes, xqu8, xscales, out, m, k),
        VnniKind::None => matvec_q8_0_w8a8_scalar(w_bytes, xqu8, xscales, out, m, k),
    }
}

/// W8A8 Q8_0 matvec (public entry): quantize the activation once, then run the
/// widest available VNNI kernel. Caller must have confirmed [`vnni_q8_0_usable`].
pub fn matvec_q8_0_w8a8(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(k % QK, 0);
    let blocks = k / QK;
    let mut xqu8 = vec![0u8; k];
    let mut xscales = vec![0f32; blocks];
    quantize_activation(x, k, &mut xqu8, &mut xscales);
    // SAFETY: a VNNI kind was confirmed present via vnni_q8_0_usable().
    unsafe { run_vnni_kernel(w_bytes, &xqu8, &xscales, out, m, k) };
}

/// Whether the W8A8 VNNI Q8_0 matvec is usable — cached. True only when the
/// CPU has AVX-VNNI AND the vectorized kernel reproduces the scalar W8A8
/// reference bit-for-bit on a probe (the integer math is exact, so a mismatch
/// means a wrong dpbusd/bias encoding → fail closed).
pub fn vnni_q8_0_usable() -> bool {
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| {
        if vnni_kind() == VnniKind::None {
            return false;
        }
        // Probe across several blocks/rows with a valid Q8_0 weight.
        let (m, k) = (5usize, 128usize);
        let w = synth_q8_0(m, k, 2_654_435_761);
        let x: Vec<f32> = (0..k).map(|j| ((j % 17) as f32 - 8.0) * 0.07).collect();
        let mut xqu8 = vec![0u8; k];
        let mut xscales = vec![0f32; k / QK];
        quantize_activation(&x, k, &mut xqu8, &mut xscales);
        let mut got = vec![0f32; m];
        let mut want = vec![0f32; m];
        matvec_q8_0_w8a8_scalar(&w, &xqu8, &xscales, &mut want, m, k);
        // SAFETY: a VNNI kind is present (checked above).
        unsafe { run_vnni_kernel(&w, &xqu8, &xscales, &mut got, m, k) };
        // Integer math is identical → exact f32 match (both fold the same
        // per-block d·x_scale in the same order).
        got == want
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vnni_w8a8_matches_scalar_or_skips() {
        if vnni_kind() == VnniKind::None {
            eprintln!("vnni: no AVX-VNNI / AVX-512-VNNI on this host — SKIP");
            return;
        }
        assert!(vnni_q8_0_usable(), "VNNI present but self-check failed");
        // Larger shape, different data.
        let (m, k) = (37usize, 256usize);
        let w = synth_q8_0(m, k, 40_503);
        let x: Vec<f32> = (0..k).map(|j| ((j % 29) as f32 - 14.0) * 0.03).collect();
        let mut xqu8 = vec![0u8; k];
        let mut xscales = vec![0f32; k / QK];
        quantize_activation(&x, k, &mut xqu8, &mut xscales);
        let mut got = vec![0f32; m];
        let mut want = vec![0f32; m];
        matvec_q8_0_w8a8_scalar(&w, &xqu8, &xscales, &mut want, m, k);
        unsafe { run_vnni_kernel(&w, &xqu8, &xscales, &mut got, m, k) };
        assert_eq!(got, want, "VNNI W8A8 != scalar W8A8 reference");
    }
}
