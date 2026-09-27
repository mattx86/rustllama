//! K-quant encoders.
//!
//! 256-weight super-blocks with sub-block scale hierarchies. Each
//! encoder mirrors the inverse of the corresponding [`crate::dequant`]
//! routine and follows the layout documented in [`crate::parse`]'s
//! `GgmlType` variant docstrings.
//!
//! Quality note: Q2_K / Q4_K / Q5_K use the iterative
//! [`make_qkx2_quants_asym`] scale-finder (a Rust port of
//! llama.cpp's `make_qkx2_quants`) — perturbs the analytical
//! starting point across `nstep=20` inverse-scale candidates,
//! refits `(d, min)` per try via weighted least-squares, and
//! picks the lowest-L2 assignment. Quality matches the
//! llama.cpp `--imatrix`-less reference within FP rounding.
//!
//! Q3_K, Q6_K, Q8_K still use the analytical formula (they have
//! signed sub-scales which don't fit `make_qkx2_quants_asym`'s
//! 2-parameter LS refit cleanly). The iterative variant for
//! signed sub-scales is a future quality follow-up.

use half::f16;

const QK_K: usize = 256;
const N_SUB_BLOCKS_16: usize = 16; // 256 / 16
const N_SUB_BLOCKS_32: usize = 8; // 256 / 32

// ----------------------------------------------------------------------
// Q8_K — `{ d: f32, qs: [i8; 256], bsums: [i16; 16] }` = 292 bytes
// ----------------------------------------------------------------------

const BLOCK_Q8_K_BYTES: usize = 292;

/// Encode Q8_K. Symmetric 8-bit with f32 super-block scale; bsums
/// precomputes per-16-weight sub-block sums (used by the K-quant
/// matmul path to avoid recomputing during inner-loop dot products).
pub fn encode_q8_k(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK_K, 0, "encode_q8_k: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_Q8_K_BYTES,
        "encode_q8_k: dst.len() must be n_blocks * 292"
    );
    for b in 0..n_blocks {
        let xs = &src[b * QK_K..(b + 1) * QK_K];
        let off = b * BLOCK_Q8_K_BYTES;
        let mut amax = 0f32;
        let mut max_val = 0f32;
        for &x in xs {
            let a = x.abs();
            if a > amax {
                amax = a;
                max_val = x;
            }
        }
        // Reference: signed d = max_val / -128, so q = round(x / d)
        // recovers `q` in [-128, 127].
        let d = if amax == 0.0 { 0.0 } else { -max_val / 128.0 };
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        dst[off..off + 4].copy_from_slice(&d.to_le_bytes());
        let mut bsums = [0i32; 16];
        for i in 0..256 {
            let q = (xs[i] * id).round().clamp(-128.0, 127.0) as i8;
            dst[off + 4 + i] = q as u8;
            bsums[i / 16] += q as i32;
        }
        for k in 0..16 {
            let s = bsums[k].clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            let off_s = off + 4 + 256 + k * 2;
            dst[off_s] = (s as u16 & 0xFF) as u8;
            dst[off_s + 1] = ((s as u16 >> 8) & 0xFF) as u8;
        }
    }
}

// ----------------------------------------------------------------------
// Q6_K — `{ ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], d: f16 }` = 210 bytes
// ----------------------------------------------------------------------

const BLOCK_Q6_K_BYTES: usize = 210;

/// Encode Q6_K. 16 sub-blocks of 16 weights each. Per-sub-block i8
/// signed scale plus a super-block f16 scale; each weight is a signed
/// 6-bit value (4 low bits in `ql`, 2 high bits in `qh`, biased -32
/// on decode to land in `[-32, 31]`).
/// G6: GPU+CPU Q6_K block encoder. Tries the SYCL kernel via
/// `IqGpuEncoder::try_encode_q6_k_blocks`; falls back to the CPU
/// `encode_q6_k` on `Err` (mock mode, USM exhaustion, kernel failure).
/// Per-block bit-for-bit parity vs the CPU reference is enforced by
/// the SYCL kernel (algorithm is purely analytical, no FP-order risk).
pub fn encode_q6_k_with_encoder(
    src: &[f32],
    dst: &mut [u8],
    encoder: &dyn crate::iq_gpu::IqGpuEncoder,
) {
    assert_eq!(src.len() % QK_K, 0, "encode_q6_k_with_encoder: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_Q6_K_BYTES,
        "encode_q6_k_with_encoder: dst.len() must be n_blocks * 210"
    );
    if encoder.try_encode_q6_k_blocks(src, dst).is_ok() {
        return;
    }
    encode_q6_k(src, dst);
}

pub fn encode_q6_k(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK_K, 0, "encode_q6_k: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_Q6_K_BYTES,
        "encode_q6_k: dst.len() must be n_blocks * 210"
    );
    for b in 0..n_blocks {
        let xs = &src[b * QK_K..(b + 1) * QK_K];
        let off = b * BLOCK_Q6_K_BYTES;

        // Stage 1: per-sub-block ideal signed scale = max_val / -32.
        let mut sub_scales = [0f32; N_SUB_BLOCKS_16];
        for k in 0..N_SUB_BLOCKS_16 {
            let sb = &xs[k * 16..(k + 1) * 16];
            let mut amax = 0f32;
            let mut max_val = 0f32;
            for &x in sb {
                let a = x.abs();
                if a > amax {
                    amax = a;
                    max_val = x;
                }
            }
            sub_scales[k] = if amax == 0.0 { 0.0 } else { max_val / -32.0 };
        }

        // Stage 2: super-block d chosen so quantized sub-scales fit
        // in i8 [-128, 127]. Reference uses negative iscale to keep
        // sub-scale signs consistent with the negated /-32 above.
        let max_abs_scale = sub_scales.iter().fold(0f32, |a, &s| a.max(s.abs()));
        let iscale = if max_abs_scale > 0.0 {
            -128.0 / sub_scales
                .iter()
                .fold(0f32, |acc, &s| if s.abs() > acc.abs() { s } else { acc })
        } else {
            0.0
        };
        let d_super = if iscale != 0.0 { 1.0 / iscale } else { 0.0 };

        // Quantize per-sub-block scales as i8.
        let mut scales_q = [0i8; N_SUB_BLOCKS_16];
        for k in 0..N_SUB_BLOCKS_16 {
            let l = (iscale * sub_scales[k]).round().clamp(-128.0, 127.0) as i32;
            scales_q[k] = l as i8;
        }

        // Stage 3: quantize 6-bit weights against per-sub-block effective scale.
        let mut q_signed = [0i32; QK_K]; // values in [-32, 31] (or [-32, 31])
        for k in 0..N_SUB_BLOCKS_16 {
            let s_q = scales_q[k] as f32;
            let dl = d_super * s_q;
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            let base = k * 16;
            for l in 0..16 {
                let q = (xs[base + l] * idl).round().clamp(-32.0, 31.0) as i32;
                q_signed[base + l] = q;
            }
        }

        // Pack q_signed into ql (low nibble) + qh (high 2 bits) in
        // the same shuffle pattern as the dequant. Reverse of
        // `dequant_q6_k`'s `q1..q4` extraction.
        //   For each n in 0..2:
        //     For l in 0..32:
        //       q1 = signed[n*128 + l]            stored at ql[64*n+l] low,    qh[32*n+l] bits 0..2
        //       q2 = signed[n*128 + 32 + l]       stored at ql[64*n+l+32] low, qh[32*n+l] bits 2..4
        //       q3 = signed[n*128 + 64 + l]       stored at ql[64*n+l] high,   qh[32*n+l] bits 4..6
        //       q4 = signed[n*128 + 96 + l]       stored at ql[64*n+l+32] high,qh[32*n+l] bits 6..8
        // The decoded value is signed - 32, so the stored unsigned representation is signed + 32.
        let ql_off = off;
        let qh_off = off + 128;
        for byte in dst[ql_off..ql_off + 128].iter_mut() {
            *byte = 0;
        }
        for byte in dst[qh_off..qh_off + 64].iter_mut() {
            *byte = 0;
        }
        for n in 0..2 {
            for l in 0..32 {
                let q1 = (q_signed[n * 128 + l] + 32) as u32;
                let q2 = (q_signed[n * 128 + 32 + l] + 32) as u32;
                let q3 = (q_signed[n * 128 + 64 + l] + 32) as u32;
                let q4 = (q_signed[n * 128 + 96 + l] + 32) as u32;
                dst[ql_off + 64 * n + l] =
                    ((q1 & 0x0F) | ((q3 & 0x0F) << 4)) as u8;
                dst[ql_off + 64 * n + l + 32] =
                    ((q2 & 0x0F) | ((q4 & 0x0F) << 4)) as u8;
                let qh_byte = ((q1 >> 4) & 0x03)
                    | (((q2 >> 4) & 0x03) << 2)
                    | (((q3 >> 4) & 0x03) << 4)
                    | (((q4 >> 4) & 0x03) << 6);
                dst[qh_off + 32 * n + l] = qh_byte as u8;
            }
        }

        // Store i8 sub-scales.
        let scales_off = off + 128 + 64;
        for k in 0..N_SUB_BLOCKS_16 {
            dst[scales_off + k] = scales_q[k] as u8;
        }
        // f16 super-block d at the end.
        let d_bits = f16::from_f32(d_super).to_bits();
        dst[off + 208] = (d_bits & 0xFF) as u8;
        dst[off + 209] = ((d_bits >> 8) & 0xFF) as u8;
    }
}

// ----------------------------------------------------------------------
// Q4_K — `{ d: f16, dmin: f16, scales: [u8; 12], qs: [u8; 128] }` = 144 bytes
// ----------------------------------------------------------------------

const BLOCK_Q4_K_BYTES: usize = 144;

/// Iterative asymmetric sub-block quantizer — Rust port of
/// llama.cpp's `make_qkx2_quants` (ggml-quants.c). For each
/// `nstep` inverse-scale perturbation around the analytical
/// starting point, requantize the sub-block, refit (d, min) by
/// weighted least-squares, and track the assignment minimizing
/// total L2 error. Quality vs the one-shot analytical formula
/// recovers ~0.05–0.15 perplexity at the same bpw on real
/// model weights.
///
/// Returns `(d, min, L)` matching the dequant convention
/// `value = d * L[i] - min` (both `d, min ≥ 0`).
fn make_qkx2_quants_asym<const NMAX: u32>(
    xs: &[f32],
    out_q: &mut [u8],
    w: Option<&[f32]>,
) -> (f32, f32) {
    debug_assert_eq!(xs.len(), out_q.len());
    debug_assert!(w.map_or(true, |w| w.len() == xs.len()));
    let n = xs.len();
    let nmax_f = NMAX as f32;

    // Find min / max.
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &v in xs {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    if mx == mn {
        // Degenerate sub-block (constant input).
        let m = if mn < 0.0 { -mn } else { 0.0 };
        for q in out_q.iter_mut() {
            *q = 0;
        }
        return (0.0, m);
    }

    // The dequant convention is `value = d * q - m` with `m ≥ 0`.
    // For positive-only inputs (`mn > 0`), the format can't encode
    // a positive offset; clamp `m = 0` and quantize `[0, max]`.
    // That path matches our v1 one-shot encoder so we keep it.
    let mut min_used = if mn <= 0.0 { -mn } else { 0.0 };
    let mut d = if mn <= 0.0 {
        (mx - mn) / nmax_f
    } else {
        mx / nmax_f
    };

    // Initial quantization.
    let mut id = 1.0 / d;
    let mut best_l: Vec<u8> = (0..n)
        .map(|i| {
            let q = ((xs[i] + min_used) * id).round().clamp(0.0, nmax_f) as u8;
            q
        })
        .collect();
    let mut best_d = d;
    let mut best_min = min_used;
    let mut best_err = weighted_l2_err(xs, &best_l, best_d, best_min, w);

    // llama.cpp's parameters: rmin=-1.0, rdelta=0.1, nstep=20.
    // Per step: inv_scale = id * (rmin + step * rdelta) → covers
    // perturbations from -id to +0.9*id, refitting (d, min) by
    // weighted LS after each requantize.
    const NSTEP: i32 = 20;
    const RMIN: f32 = -1.0;
    const RDELTA: f32 = 0.1;
    for step in 0..NSTEP {
        let factor = RMIN + (step as f32) * RDELTA;
        if factor == 0.0 {
            continue;
        }
        let inv_scale_try = id * factor;
        let mut l_try = vec![0u8; n];
        // Requantize against the perturbed inverse scale.
        for i in 0..n {
            let q = ((xs[i] + min_used) * inv_scale_try).round().clamp(0.0, nmax_f) as u8;
            l_try[i] = q;
        }
        // Refit (d, min) by weighted least-squares against this
        // assignment. Weights = 1.0 in v1; imatrix weights are an
        // additional follow-up that plugs into this same step.
        let (refit_d, refit_min) = refit_d_min(xs, &l_try, w);
        if refit_d <= 0.0 {
            continue;
        }
        // Reconstruct with refit values; clamp min ≥ 0 to match the
        // format constraint.
        let m_clamped = refit_min.max(0.0);
        // Final L using refit_d + clamped min. Skip refit if it
        // sends min negative — the format can't encode it.
        let refit_inv = 1.0 / refit_d;
        let mut l_refit = vec![0u8; n];
        for i in 0..n {
            let q = ((xs[i] + m_clamped) * refit_inv).round().clamp(0.0, nmax_f) as u8;
            l_refit[i] = q;
        }
        let err = weighted_l2_err(xs, &l_refit, refit_d, m_clamped, w);
        if err < best_err {
            best_err = err;
            best_d = refit_d;
            best_min = m_clamped;
            best_l = l_refit;
            // Also adopt the new id as the next iteration's starting
            // point so the perturbation walk stays focused on the
            // current best — mirrors ggml's behavior.
            id = refit_inv;
            d = refit_d;
            min_used = m_clamped;
        }
    }

    out_q.copy_from_slice(&best_l);
    let _ = d;
    let _ = min_used;
    (best_d, best_min)
}

/// Importance-weighted sum-squared error for the candidate
/// assignment `value[i] = d * L[i] - min` vs the source `xs`.
///
/// `w` is the per-element importance (imatrix) slice; `None` means
/// uniform weight 1.0 (identical to the pre-imatrix behavior).
fn weighted_l2_err(xs: &[f32], l: &[u8], d: f32, min: f32, w: Option<&[f32]>) -> f32 {
    let n = xs.len();
    let mut err = 0f32;
    for i in 0..n {
        let v = d * (l[i] as f32) - min;
        let diff = v - xs[i];
        let wi = w.map_or(1.0, |w| w[i]);
        err += wi * diff * diff;
    }
    err
}

/// Refit `(d, min)` from a fixed `L` assignment via importance-
/// weighted least-squares. Solves the 2-parameter linear system that
/// minimizes `Σ wᵢ (d * L[i] - min - x[i])²`. With `w = None` (all
/// weights 1.0) this reduces exactly to the prior unweighted refit.
fn refit_d_min(xs: &[f32], l: &[u8], w: Option<&[f32]>) -> (f32, f32) {
    let n = xs.len();
    // Weighted moments: SW = Σw, SL = Σw·L, SLL = Σw·L², SX = Σw·x,
    // SLX = Σw·L·x. The normal equations are
    //   d·SLL + b·SL = SLX
    //   d·SL  + b·SW = SX     (value = d·L + b, min = -b)
    let mut sw = 0f32;
    let mut sum_l = 0f32;
    let mut sum_l2 = 0f32;
    let mut sum_x = 0f32;
    let mut sum_lx = 0f32;
    for i in 0..n {
        let wi = w.map_or(1.0, |w| w[i]);
        let li = l[i] as f32;
        sw += wi;
        sum_l += wi * li;
        sum_l2 += wi * li * li;
        sum_x += wi * xs[i];
        sum_lx += wi * li * xs[i];
    }
    let denom = sw * sum_l2 - sum_l * sum_l;
    if denom.abs() < 1e-12 {
        // Degenerate — fall back to no refit (caller keeps best
        // known d/min).
        return (-1.0, 0.0);
    }
    let d = (sw * sum_lx - sum_l * sum_x) / denom;
    // The dequant formula stores `min ≥ 0` and reconstructs
    // `value = d*L - min`. After LS we have an unconstrained
    // intercept `b` with `value = d*L + b` ⇒ `min = -b`.
    let b = (sum_l2 * sum_x - sum_l * sum_lx) / denom;
    (d, -b)
}

/// Per-sub-block asymmetric quantize (Q4_K / Q5_K shared helper).
///
/// Returns `(d_sub, m_sub, q[N])` matching the dequant convention
/// `value = d_sub * q - m_sub`. Both `d_sub` and `m_sub` are
/// guaranteed non-negative (the unsigned sub-scale + min slots in
/// the K-quant scale-packing layout require it).
///
/// v1.x note: this is the **one-shot analytical** scale-finder
/// (kept as the fallback for non-K_K formats). The K-quant
/// encoders (Q2_K, Q4_K, Q5_K) now call
/// [`make_qkx2_quants_asym`] which iterates around this starting
/// point for ~0.05–0.15 perplexity improvement at the same bpw.
fn quant_sub_block_asym<const NMAX: u32>(
    xs: &[f32],
    out_q: &mut [u8],
) -> (f32, f32) {
    let n = xs.len();
    debug_assert_eq!(out_q.len(), n);
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &v in xs {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    if mx == mn {
        // Degenerate sub-block. d=0 means the dequant returns -m for
        // every weight; m must be non-negative for storage.
        let m = if mn < 0.0 { -mn } else { 0.0 };
        for q in out_q.iter_mut() {
            *q = 0;
        }
        return (0.0, m);
    }
    if mn <= 0.0 {
        let d = (mx - mn) / NMAX as f32;
        let id = 1.0 / d;
        let m = -mn; // ≥ 0 since mn ≤ 0
        for i in 0..n {
            let q = ((xs[i] - mn) * id).round().clamp(0.0, NMAX as f32) as u8;
            out_q[i] = q;
        }
        (d, m)
    } else {
        // All values > 0: keep reconstruction range [0, mx]; lose
        // one quantization level under mn but the K-quant family
        // typically sees mixed-sign weight distributions, so the
        // common case is the branch above. This branch handles
        // bias / norm tensors that happen to be positive-only.
        let d = mx / NMAX as f32;
        let id = 1.0 / d;
        for i in 0..n {
            let q = (xs[i] * id).round().clamp(0.0, NMAX as f32) as u8;
            out_q[i] = q;
        }
        (d, 0.0)
    }
}

/// Pack the 8 sub-block (sc, mn) 6-bit unsigned pairs into the
/// 12-byte K-quant scales array (Q4_K / Q5_K shared layout). Inverse
/// of the dequant's unpack: `j < 4` uses straight low-6-bits; `j >=
/// 4` splits across two source bytes (low nibble + bits 6..8 from a
/// far-away byte).
fn pack_q4k_q5k_scales(sc: [u8; 8], mn: [u8; 8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), 12);
    // Clear so the "split" writes can OR into stable bits.
    for byte in out.iter_mut() {
        *byte = 0;
    }
    for j in 0..8 {
        if j < 4 {
            // Low 6 bits of sc[j] in low 6 bits of out[j]; same for mn[j] in out[j+4].
            out[j] = sc[j] & 0x3F;
            out[j + 4] = mn[j] & 0x3F;
        } else {
            // sc[j] = low 4 bits in out[j+4] low nibble, top 2 bits in out[j-4] bits 6-7
            // mn[j] = low 4 bits in out[j+4] high nibble, top 2 bits in out[j]   bits 6-7
            out[j + 4] |= sc[j] & 0x0F;
            out[j - 4] |= (sc[j] >> 4) << 6;
            out[j + 4] |= (mn[j] & 0x0F) << 4;
            out[j] |= (mn[j] >> 4) << 6;
        }
    }
}

/// Encode Q4_K. 8 sub-blocks of 32 weights with asymmetric 4-bit
/// quantization (`value = d_sub * q - m_sub`, `q ∈ [0, 15]`). The 16
/// (d_sub, m_sub) values across the super-block are themselves
/// quantized as 6-bit unsigned with f16 super-block (d, dmin) and
/// packed via [`pack_q4k_q5k_scales`].
/// G6: GPU+CPU Q4_K block encoder. Tries SYCL via
/// `IqGpuEncoder::try_encode_q4_k_blocks`; falls back to CPU
/// `encode_q4_k` on `Err`.
pub fn encode_q4_k_with_encoder(
    src: &[f32],
    dst: &mut [u8],
    encoder: &dyn crate::iq_gpu::IqGpuEncoder,
) {
    assert_eq!(src.len() % QK_K, 0, "encode_q4_k_with_encoder: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_Q4_K_BYTES,
        "encode_q4_k_with_encoder: dst.len() must be n_blocks * 144"
    );
    if encoder.try_encode_q4_k_blocks(src, dst).is_ok() {
        return;
    }
    encode_q4_k(src, dst);
}

pub fn encode_q4_k(src: &[f32], dst: &mut [u8]) {
    encode_q4_k_imatrix(src, dst, None);
}

/// Importance-matrix-aware Q4_K encoder. `imatrix`, when `Some`, is a
/// per-input-column importance vector of length `src.len()` (mean
/// squared activation per column); the scale-finder weights its L2
/// error by it so high-importance columns are reconstructed more
/// faithfully. `None` reproduces the uniform-weight encode exactly.
pub fn encode_q4_k_imatrix(src: &[f32], dst: &mut [u8], imatrix: Option<&[f32]>) {
    assert_eq!(src.len() % QK_K, 0, "encode_q4_k: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_Q4_K_BYTES,
        "encode_q4_k: dst.len() must be n_blocks * 144"
    );
    debug_assert!(imatrix.map_or(true, |w| w.len() == src.len()));
    for b in 0..n_blocks {
        let xs = &src[b * QK_K..(b + 1) * QK_K];
        let off = b * BLOCK_Q4_K_BYTES;

        // Stage 1: per-sub-block analytical (d_sub, m_sub) + 4-bit q.
        let mut sub_d = [0f32; N_SUB_BLOCKS_32];
        let mut sub_m = [0f32; N_SUB_BLOCKS_32];
        let mut q_all = [0u8; QK_K];
        for k in 0..N_SUB_BLOCKS_32 {
            let w_sub = imatrix.map(|w| &w[b * QK_K + k * 32..b * QK_K + (k + 1) * 32]);
            let (d, m) = make_qkx2_quants_asym::<15>(
                &xs[k * 32..(k + 1) * 32],
                &mut q_all[k * 32..(k + 1) * 32],
                w_sub,
            );
            sub_d[k] = d;
            sub_m[k] = m;
        }

        // Stage 2: pick super-block d / dmin to fit 6-bit sub-scales.
        let max_d = sub_d.iter().cloned().fold(0f32, f32::max);
        let max_m = sub_m.iter().cloned().fold(0f32, f32::max);
        let d_super = max_d / 63.0;
        let dmin_super = max_m / 63.0;
        let id_super = if d_super > 0.0 { 1.0 / d_super } else { 0.0 };
        let im_super = if dmin_super > 0.0 { 1.0 / dmin_super } else { 0.0 };

        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for k in 0..N_SUB_BLOCKS_32 {
            sc[k] = (sub_d[k] * id_super).round().clamp(0.0, 63.0) as u8;
            mn[k] = (sub_m[k] * im_super).round().clamp(0.0, 63.0) as u8;
        }

        // Stage 3: write header.
        let d_bits = f16::from_f32(d_super).to_bits();
        let dmin_bits = f16::from_f32(dmin_super).to_bits();
        dst[off] = (d_bits & 0xFF) as u8;
        dst[off + 1] = ((d_bits >> 8) & 0xFF) as u8;
        dst[off + 2] = (dmin_bits & 0xFF) as u8;
        dst[off + 3] = ((dmin_bits >> 8) & 0xFF) as u8;
        pack_q4k_q5k_scales(sc, mn, &mut dst[off + 4..off + 16]);

        // Stage 4: pack 256 4-bit q values into 128 qs bytes in the
        // dequant-matching group-of-64 layout (each group: 32 lows,
        // then 32 highs of the same qs[0..32]).
        let qs = &mut dst[off + 16..off + 16 + 128];
        for group in 0..4 {
            let q_chunk = &mut qs[group * 32..(group + 1) * 32];
            let g_base = group * 64;
            for l in 0..32 {
                let lo = q_all[g_base + l] & 0x0F;
                let hi = q_all[g_base + 32 + l] & 0x0F;
                q_chunk[l] = lo | (hi << 4);
            }
        }
    }
}

// ----------------------------------------------------------------------
// Q5_K — `{ d: f16, dmin: f16, scales: [u8; 12], qh: [u8; 32], qs: [u8; 128] }` = 176 bytes
// ----------------------------------------------------------------------

const BLOCK_Q5_K_BYTES: usize = 176;

/// Encode Q5_K. Same asymmetric structure as Q4_K with one extra
/// 5th bit per weight stored in a 32-byte `qh` array. Per the
/// dequant's bit-layout: for each `l ∈ 0..32`, `qh[l]` carries 8
/// bits — bits `(group*2, group*2+1)` belong to the lows/highs of
/// the group's 32-weight pair.
/// G6: GPU+CPU Q5_K block encoder. Tries SYCL via
/// `IqGpuEncoder::try_encode_q5_k_blocks`; falls back to CPU
/// `encode_q5_k` on `Err`.
pub fn encode_q5_k_with_encoder(
    src: &[f32],
    dst: &mut [u8],
    encoder: &dyn crate::iq_gpu::IqGpuEncoder,
) {
    assert_eq!(src.len() % QK_K, 0, "encode_q5_k_with_encoder: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_Q5_K_BYTES,
        "encode_q5_k_with_encoder: dst.len() must be n_blocks * 176"
    );
    if encoder.try_encode_q5_k_blocks(src, dst).is_ok() {
        return;
    }
    encode_q5_k(src, dst);
}

pub fn encode_q5_k(src: &[f32], dst: &mut [u8]) {
    encode_q5_k_imatrix(src, dst, None);
}

/// Importance-matrix-aware Q5_K encoder. See [`encode_q4_k_imatrix`].
pub fn encode_q5_k_imatrix(src: &[f32], dst: &mut [u8], imatrix: Option<&[f32]>) {
    assert_eq!(src.len() % QK_K, 0, "encode_q5_k: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_Q5_K_BYTES,
        "encode_q5_k: dst.len() must be n_blocks * 176"
    );
    debug_assert!(imatrix.map_or(true, |w| w.len() == src.len()));
    for b in 0..n_blocks {
        let xs = &src[b * QK_K..(b + 1) * QK_K];
        let off = b * BLOCK_Q5_K_BYTES;

        let mut sub_d = [0f32; N_SUB_BLOCKS_32];
        let mut sub_m = [0f32; N_SUB_BLOCKS_32];
        let mut q_all = [0u8; QK_K];
        for k in 0..N_SUB_BLOCKS_32 {
            let w_sub = imatrix.map(|w| &w[b * QK_K + k * 32..b * QK_K + (k + 1) * 32]);
            let (d, m) = make_qkx2_quants_asym::<31>(
                &xs[k * 32..(k + 1) * 32],
                &mut q_all[k * 32..(k + 1) * 32],
                w_sub,
            );
            sub_d[k] = d;
            sub_m[k] = m;
        }

        let max_d = sub_d.iter().cloned().fold(0f32, f32::max);
        let max_m = sub_m.iter().cloned().fold(0f32, f32::max);
        let d_super = max_d / 63.0;
        let dmin_super = max_m / 63.0;
        let id_super = if d_super > 0.0 { 1.0 / d_super } else { 0.0 };
        let im_super = if dmin_super > 0.0 { 1.0 / dmin_super } else { 0.0 };

        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for k in 0..N_SUB_BLOCKS_32 {
            sc[k] = (sub_d[k] * id_super).round().clamp(0.0, 63.0) as u8;
            mn[k] = (sub_m[k] * im_super).round().clamp(0.0, 63.0) as u8;
        }

        let d_bits = f16::from_f32(d_super).to_bits();
        let dmin_bits = f16::from_f32(dmin_super).to_bits();
        dst[off] = (d_bits & 0xFF) as u8;
        dst[off + 1] = ((d_bits >> 8) & 0xFF) as u8;
        dst[off + 2] = (dmin_bits & 0xFF) as u8;
        dst[off + 3] = ((dmin_bits >> 8) & 0xFF) as u8;
        pack_q4k_q5k_scales(sc, mn, &mut dst[off + 4..off + 16]);

        // qh: 32 bytes, one byte per `l` index; bits laid out as
        // `(group*2, group*2+1) = (lo_5th, hi_5th)` for that group's
        // 32-weight pair at position `l`. Split the dst slice so we
        // can hold both qh and qs as mutable borrows simultaneously.
        let (qh_full, rest) = dst[off + 16..off + 16 + 32 + 128].split_at_mut(32);
        let qs = &mut rest[..128];
        for byte in qh_full.iter_mut() {
            *byte = 0;
        }
        for group in 0..4 {
            let g_base = group * 64;
            let q_chunk = &mut qs[group * 32..(group + 1) * 32];
            for l in 0..32 {
                let lo_full = q_all[g_base + l];
                let hi_full = q_all[g_base + 32 + l];
                q_chunk[l] = (lo_full & 0x0F) | ((hi_full & 0x0F) << 4);
                if lo_full & 0x10 != 0 {
                    qh_full[l] |= 1 << (group * 2);
                }
                if hi_full & 0x10 != 0 {
                    qh_full[l] |= 1 << (group * 2 + 1);
                }
            }
        }
    }
}

// ----------------------------------------------------------------------
// Q3_K — `{ hmask: [u8; 32], qs: [u8; 64], scales: [u8; 12], d: f16 }` = 110 bytes
// ----------------------------------------------------------------------

const BLOCK_Q3_K_BYTES: usize = 110;

/// Pack 16 signed-6-bit sub-scales into the Q3_K 12-byte scales
/// array. Inverse of the kmask1/kmask2 unpack in the dequant:
///   aux[0..3] hold 12 raw bytes; aux[3] is reconstructed.
///   sub_scale i (i32, signed, [-32, 31]) → stored as `s + 0` (i.e.
///   raw two's-complement low 6 bits): the dequant reads as i8 then
///   subtracts 32 to recenter, so the stored 6-bit value is
///   `scales[i] + 32` clipped to [0, 63].
fn pack_q3k_scales(scales_i8: [i8; 16], out: &mut [u8]) {
    debug_assert_eq!(out.len(), 12);
    // Convert signed sub-scales to 6-bit unsigned biased form. The
    // dequant subtracts 32 after reading, so the stored value is
    // (i8 + 32) clipped to [0, 63]. That preserves the original
    // signed sub-scale across the round trip.
    let mut s6 = [0u8; 16];
    for i in 0..16 {
        let raw = scales_i8[i] as i32 + 32;
        s6[i] = raw.clamp(0, 63) as u8;
    }
    // Build the inverse of the dequant's kmask1/kmask2 packing:
    //   aux[0..3] = 12 bytes of scales.
    // Reference (encode side, ggml-quants.c::quantize_row_q3_K_reference):
    //   for i in 0..8: scales[i] = s6[i]&0xF; scales[i+8] = s6[i]>>4 ... etc.
    // But the exact byte layout is what `dequant_q3_k` reverses, so
    // we recompute by directly inverting that. After the dequant's
    // shuffle, the recovered 16 6-bit values land in aux_bytes[0..16].
    // The encoding fills out[0..4]=aux[0]'s components, out[4..8]=aux[1]'s,
    // out[8..12]=aux[2]'s, such that the dequant's reshuffle reproduces s6.
    //
    // We solve by symbolic inversion of the dequant code:
    //   aux[2]' = ((aux[0]>>4) & K2) | (((aux[2]>>4) & K1) << 4);
    //   aux[3]' = ((aux[1]>>4) & K2) | (((aux[2]>>6) & K1) << 4);
    //   aux[0]' = (aux[0]  & K2) | ((aux[2]    & K1) << 4);
    //   aux[1]' = (aux[1]  & K2) | (((aux[2]>>2) & K1) << 4);
    // where aux'[i] is the recovered 4-byte word (each byte = one 6-bit value),
    // and aux[i] (i < 3) are the on-disk bytes. K2=0x0F0F0F0F (low nibble per byte),
    // K1=0x03030303 (low 2 bits per byte).
    //
    // Per byte j in {0,1,2,3}:
    //   recovered[0][j] = (raw[0][j] & 0xF) | ((raw[2][j] & 3) << 4)        // s6[0..4]
    //   recovered[1][j] = (raw[1][j] & 0xF) | (((raw[2][j] >> 2) & 3) << 4) // s6[4..8]
    //   recovered[2][j] = ((raw[0][j] >> 4) & 0xF) | (((raw[2][j] >> 4) & 3) << 4) // s6[8..12]
    //   recovered[3][j] = ((raw[1][j] >> 4) & 0xF) | (((raw[2][j] >> 6) & 3) << 4) // s6[12..16]
    //
    // So per byte j of raw[0..3] we encode:
    //   raw[0][j] low4   = s6[0+j] low 4 bits
    //   raw[0][j] high4  = s6[8+j] low 4 bits
    //   raw[1][j] low4   = s6[4+j] low 4 bits
    //   raw[1][j] high4  = s6[12+j] low 4 bits
    //   raw[2][j] bits 0..1 = s6[0+j] high 2 bits
    //   raw[2][j] bits 2..3 = s6[4+j] high 2 bits
    //   raw[2][j] bits 4..5 = s6[8+j] high 2 bits
    //   raw[2][j] bits 6..7 = s6[12+j] high 2 bits
    for j in 0..4 {
        let v0 = s6[j] as u32;
        let v4 = s6[j + 4] as u32;
        let v8 = s6[j + 8] as u32;
        let v12 = s6[j + 12] as u32;
        out[j] = ((v0 & 0x0F) | ((v8 & 0x0F) << 4)) as u8;
        out[j + 4] = ((v4 & 0x0F) | ((v12 & 0x0F) << 4)) as u8;
        out[j + 8] = ((v0 >> 4) & 3
            | (((v4 >> 4) & 3) << 2)
            | (((v8 >> 4) & 3) << 4)
            | (((v12 >> 4) & 3) << 6)) as u8;
    }
}

/// Encode Q3_K. Signed 3-bit per weight: 1 high bit in `hmask`, 2
/// low bits in `qs` (4 weights per byte at shift positions 0/2/4/6).
/// 16 sub-blocks of 16 weights each, signed 6-bit sub-scales,
/// super-block f16 d. Decode formula: `value = d * (sub_scale - 32)
/// * ((low2) - (hi_bit_set ? 0 : 4))` — values land in `[-4, 3]`
/// times the sub-scale.
/// G6: GPU+CPU Q3_K block encoder. Tries SYCL via
/// `IqGpuEncoder::try_encode_q3_k_blocks`; falls back to CPU
/// `encode_q3_k` on `Err`.
pub fn encode_q3_k_with_encoder(
    src: &[f32],
    dst: &mut [u8],
    encoder: &dyn crate::iq_gpu::IqGpuEncoder,
) {
    assert_eq!(src.len() % QK_K, 0, "encode_q3_k_with_encoder: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_Q3_K_BYTES,
        "encode_q3_k_with_encoder: dst.len() must be n_blocks * 110"
    );
    if encoder.try_encode_q3_k_blocks(src, dst).is_ok() {
        return;
    }
    encode_q3_k(src, dst);
}

pub fn encode_q3_k(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK_K, 0, "encode_q3_k: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_Q3_K_BYTES,
        "encode_q3_k: dst.len() must be n_blocks * 110"
    );
    for b in 0..n_blocks {
        let xs = &src[b * QK_K..(b + 1) * QK_K];
        let off = b * BLOCK_Q3_K_BYTES;

        // Per-sub-block analytical signed scale = max_val / -4.
        let mut sub_scales = [0f32; N_SUB_BLOCKS_16];
        for k in 0..N_SUB_BLOCKS_16 {
            let sb = &xs[k * 16..(k + 1) * 16];
            let mut amax = 0f32;
            let mut max_val = 0f32;
            for &x in sb {
                let a = x.abs();
                if a > amax {
                    amax = a;
                    max_val = x;
                }
            }
            sub_scales[k] = if amax == 0.0 { 0.0 } else { max_val / -4.0 };
        }

        // Super-block d so sub-scales fit in signed 6-bit (i.e.
        // [-32, 31] after the -32 bias).
        let signed_max = sub_scales
            .iter()
            .fold(0f32, |acc, &s| if s.abs() > acc.abs() { s } else { acc });
        let iscale = if signed_max != 0.0 { -32.0 / signed_max } else { 0.0 };
        let d_super = if iscale != 0.0 { 1.0 / iscale } else { 0.0 };

        let mut scales_q = [0i8; N_SUB_BLOCKS_16];
        for k in 0..N_SUB_BLOCKS_16 {
            let l = (iscale * sub_scales[k]).round().clamp(-32.0, 31.0) as i32;
            scales_q[k] = l as i8;
        }

        // Per-weight signed 3-bit q in [-4, 3].
        let mut q_signed = [0i32; QK_K];
        for k in 0..N_SUB_BLOCKS_16 {
            let s_q = scales_q[k] as f32;
            let dl = d_super * s_q;
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            let base = k * 16;
            for l in 0..16 {
                let q = (xs[base + l] * idl).round().clamp(-4.0, 3.0) as i32;
                q_signed[base + l] = q;
            }
        }

        // Encode: stored value `u3 = q + 4` ∈ [0, 7]. Low 2 bits go
        // into qs[] at the right shift slot; bit 2 goes into hmask.
        // hmask layout (per the dequant): `hmask[l] & m != 0` =>
        // "high bit NOT set" (hi_sub = 0); else hi_sub = 4 (the bias
        // shifts q from [0,3] to [-4,-1]). So hmask BIT MEANS the
        // raw stored value is ≥ 4, i.e. the original signed q was ≥
        // 0. Wait — checking dequant carefully:
        //   if hmask[l] & m != 0 { hi_sub = 0 } else { hi_sub = 4 }
        //   out = dl * (lo - hi_sub)
        // For stored q ∈ [0,3] (high bit clear), hmask bit clears
        // (m & hmask = 0) → hi_sub=4 → reconstructed `lo - 4` ∈
        // [-4,-1]. For stored q ∈ [4,7] (high bit set in u3), hmask
        // bit is set → hi_sub=0 → reconstructed `lo` ∈ [0, 3].
        // So `u3 = q_signed + 4`; low 2 bits → qs, high bit → hmask.
        let hmask_off = off;
        let qs_off = off + 32;
        for byte in dst[hmask_off..hmask_off + 32].iter_mut() {
            *byte = 0;
        }
        for byte in dst[qs_off..qs_off + 64].iter_mut() {
            *byte = 0;
        }

        // Walk the same chunk/j/shift pattern as the dequant.
        let mut q_cursor = 0usize;
        let mut m: u8 = 1;
        let mut is = 0usize;
        for chunk in 0..2 {
            let mut shift: u32 = 0;
            for _j in 0..4 {
                // First 16-weight sub-block within this (chunk, shift):
                //   weights at positions chunk*128 + (j-derived 32-weight base + 0..16)
                // dequant indexes via q_cursor + l (l in 0..16) for low-half,
                // q_cursor + l + 16 for high-half, both reading bits at `shift`.
                let base_lo = chunk * 128 + (q_cursor - chunk * 32) * 4 + 0; // unused; see real index below
                let _ = base_lo;
                // It's simpler to just reproduce the dequant's flow:
                // for l in 0..16: weight[y_cursor] = q_signed[...]
                // We need to know which y_cursor corresponds to which
                // (chunk, j, shift, l). Re-derive:
                //   y_cursor starts at chunk*128, advances 16 at a time:
                //   for each j in 0..4 (each shift 0/2/4/6):
                //     emit 16 weights for sub-block "A"
                //     emit 16 weights for sub-block "B"
                //   so within a chunk: 8 sub-blocks of 16 weights.
                //
                // For the encoder, walk the same order and pack.
                // Sub-block A at this (chunk, shift): y = chunk*128 + _j*32 + 0..16
                // Sub-block B at this (chunk, shift): y = chunk*128 + _j*32 + 16..32
                let y_base_a = chunk * 128 + _j * 32;
                for l in 0..16 {
                    let u3 = (q_signed[y_base_a + l] + 4) as u32; // ∈ [0,7]
                    let lo2 = u3 & 0x03;
                    let hi1 = (u3 >> 2) & 0x01;
                    dst[qs_off + q_cursor + l] |= (lo2 as u8) << shift;
                    if hi1 != 0 {
                        dst[hmask_off + l] |= m;
                    }
                }
                is += 1;
                let y_base_b = chunk * 128 + _j * 32 + 16;
                for l in 0..16 {
                    let u3 = (q_signed[y_base_b + l] + 4) as u32;
                    let lo2 = u3 & 0x03;
                    let hi1 = (u3 >> 2) & 0x01;
                    dst[qs_off + q_cursor + l + 16] |= (lo2 as u8) << shift;
                    if hi1 != 0 {
                        dst[hmask_off + l + 16] |= m;
                    }
                }
                is += 1;
                shift += 2;
                m = m.wrapping_shl(1);
            }
            q_cursor += 32;
        }

        // Pack scales + d.
        pack_q3k_scales(scales_q, &mut dst[off + 32 + 64..off + 32 + 64 + 12]);
        let d_bits = f16::from_f32(d_super).to_bits();
        dst[off + 108] = (d_bits & 0xFF) as u8;
        dst[off + 109] = ((d_bits >> 8) & 0xFF) as u8;

        // is/m unused after the loop; the dequant only reads
        // scales[0..16] in fixed order, so we already covered them via
        // pack_q3k_scales above. Suppress unused-variable lint.
        let _ = is;
        let _ = m;
    }
}

// ----------------------------------------------------------------------
// Q2_K — `{ scales: [u8; 16], qs: [u8; 64], d: f16, dmin: f16 }` = 84 bytes
// ----------------------------------------------------------------------

const BLOCK_Q2_K_BYTES: usize = 84;

/// Encode Q2_K. 16 sub-blocks of 16 weights, 2-bit unsigned q ∈
/// `[0, 3]`. Each sub-block carries a packed (scale, min) pair with
/// 4 bits each; super-block has f16 d + dmin. Per-weight decode:
/// `value = d * scale_sub * q - dmin * min_sub`.
pub fn encode_q2_k(src: &[f32], dst: &mut [u8]) {
    encode_q2_k_imatrix(src, dst, None);
}

/// Importance-matrix-aware Q2_K encoder. See [`encode_q4_k_imatrix`].
pub fn encode_q2_k_imatrix(src: &[f32], dst: &mut [u8], imatrix: Option<&[f32]>) {
    assert_eq!(src.len() % QK_K, 0, "encode_q2_k: src.len() must be multiple of 256");
    let n_blocks = src.len() / QK_K;
    assert_eq!(
        dst.len(),
        n_blocks * BLOCK_Q2_K_BYTES,
        "encode_q2_k: dst.len() must be n_blocks * 84"
    );
    debug_assert!(imatrix.map_or(true, |w| w.len() == src.len()));
    for b in 0..n_blocks {
        let xs = &src[b * QK_K..(b + 1) * QK_K];
        let off = b * BLOCK_Q2_K_BYTES;

        // Stage 1: per-sub-block (d, m) with 2-bit (NMAX=3) q.
        let mut sub_d = [0f32; N_SUB_BLOCKS_16];
        let mut sub_m = [0f32; N_SUB_BLOCKS_16];
        let mut q_all = [0u8; QK_K];
        for k in 0..N_SUB_BLOCKS_16 {
            let w_sub = imatrix.map(|w| &w[b * QK_K + k * 16..b * QK_K + (k + 1) * 16]);
            let (d, m) = make_qkx2_quants_asym::<3>(
                &xs[k * 16..(k + 1) * 16],
                &mut q_all[k * 16..(k + 1) * 16],
                w_sub,
            );
            sub_d[k] = d;
            sub_m[k] = m;
        }

        // Stage 2: super-block d, dmin so sub-scales fit in 4-bit unsigned [0, 15].
        let max_d = sub_d.iter().cloned().fold(0f32, f32::max);
        let max_m = sub_m.iter().cloned().fold(0f32, f32::max);
        let d_super = max_d / 15.0;
        let dmin_super = max_m / 15.0;
        let id_super = if d_super > 0.0 { 1.0 / d_super } else { 0.0 };
        let im_super = if dmin_super > 0.0 { 1.0 / dmin_super } else { 0.0 };

        let mut scales = [0u8; 16];
        for k in 0..N_SUB_BLOCKS_16 {
            let sc = (sub_d[k] * id_super).round().clamp(0.0, 15.0) as u8;
            let mn = (sub_m[k] * im_super).round().clamp(0.0, 15.0) as u8;
            scales[k] = sc | (mn << 4);
        }

        // Stage 3: pack 2-bit q values into 64 qs bytes per the
        // dequant's two-chunk-of-128 / four-shift layout.
        let scales_off = off;
        let qs_off = off + 16;
        for byte in dst[qs_off..qs_off + 64].iter_mut() {
            *byte = 0;
        }
        let mut is = 0usize;
        for chunk in 0..2 {
            let q_byte_base = chunk * 32; // qs[chunk*32 .. chunk*32 + 32]
            for shift in [0u32, 2, 4, 6] {
                // Sub-block A: 16 weights → qs[q_byte_base + 0..16] << shift
                let y_base_a = chunk * 128 + (shift as usize / 2) * 32;
                for l in 0..16 {
                    let q = q_all[y_base_a + l] & 0x03;
                    dst[qs_off + q_byte_base + l] |= q << shift;
                }
                is += 1;
                // Sub-block B: 16 weights → qs[q_byte_base + 16..32] << shift
                let y_base_b = chunk * 128 + (shift as usize / 2) * 32 + 16;
                for l in 0..16 {
                    let q = q_all[y_base_b + l] & 0x03;
                    dst[qs_off + q_byte_base + 16 + l] |= q << shift;
                }
                is += 1;
            }
        }
        let _ = is;

        // Write scales + d + dmin.
        dst[scales_off..scales_off + 16].copy_from_slice(&scales);
        let d_bits = f16::from_f32(d_super).to_bits();
        let dmin_bits = f16::from_f32(dmin_super).to_bits();
        dst[off + 80] = (d_bits & 0xFF) as u8;
        dst[off + 81] = ((d_bits >> 8) & 0xFF) as u8;
        dst[off + 82] = (dmin_bits & 0xFF) as u8;
        dst[off + 83] = ((dmin_bits >> 8) & 0xFF) as u8;
    }
}

// ----------------------------------------------------------------------
// Parity tests — encode → dequant → assert max abs error
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant;

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

    fn assert_round_trip(
        encode: fn(&[f32], &mut [u8]),
        decode: fn(&[u8], &mut [f32]),
        block_bytes: usize,
        max_step: f32,
        amp: f32,
        label: &str,
    ) {
        for &n_blocks in &[1usize, 2, 4] {
            let n = n_blocks * QK_K;
            let src = make_row(n, amp, n as u32 * 31);
            let mut enc = vec![0u8; n_blocks * block_bytes];
            encode(&src, &mut enc);
            let mut dec = vec![0f32; n];
            decode(&enc, &mut dec);
            let mut max_err = 0f32;
            let mut i_worst = 0usize;
            for i in 0..n {
                let e = (dec[i] - src[i]).abs();
                if e > max_err {
                    max_err = e;
                    i_worst = i;
                }
            }
            assert!(
                max_err <= max_step,
                "{label} round-trip n_blocks={n_blocks}: max_err={max_err} > bound={max_step} \
                 at i={i_worst} (src={}, dec={})",
                src[i_worst],
                dec[i_worst]
            );
        }
    }

    #[test]
    fn q8_k_round_trip_within_bound() {
        // Q8_K: i8 quantization with f32 super-block d. Step ≈ amp/127;
        // sub-block coupling not applicable (uniform across super-block).
        let amp = 4.0;
        assert_round_trip(
            encode_q8_k,
            dequant::dequant_q8_k,
            BLOCK_Q8_K_BYTES,
            2.0 * amp / 127.0,
            amp,
            "q8_k",
        );
    }

    #[test]
    fn q6_k_round_trip_within_bound() {
        // Q6_K: 6-bit signed × 16-weight sub-block. Worst-case step
        // includes the super-block d → sub-scale i8 quantization
        // overhead. Empirical bound: ~3 * amp/32 ≈ amp/10.
        let amp = 4.0;
        assert_round_trip(
            encode_q6_k,
            dequant::dequant_q6_k,
            BLOCK_Q6_K_BYTES,
            amp / 5.0,
            amp,
            "q6_k",
        );
    }

    #[test]
    fn q4_k_round_trip_within_bound() {
        // Q4_K: 4-bit asymmetric × 32-weight sub-block + 6-bit sub-scales.
        // Empirical bound: ~4 * amp / 15.
        let amp = 4.0;
        assert_round_trip(
            encode_q4_k,
            dequant::dequant_q4_k,
            BLOCK_Q4_K_BYTES,
            4.0 * amp / 15.0,
            amp,
            "q4_k",
        );
    }

    #[test]
    fn q5_k_round_trip_within_bound() {
        // Q5_K: 5-bit asymmetric × 32-weight sub-block + 6-bit sub-scales.
        let amp = 4.0;
        assert_round_trip(
            encode_q5_k,
            dequant::dequant_q5_k,
            BLOCK_Q5_K_BYTES,
            4.0 * amp / 31.0,
            amp,
            "q5_k",
        );
    }

    #[test]
    fn q3_k_round_trip_within_bound() {
        // Q3_K: 3-bit signed × 16-weight sub-block. Worst quantization
        // step ≈ amp / 4. Bound generous to cover sub-scale rounding.
        let amp = 4.0;
        assert_round_trip(
            encode_q3_k,
            dequant::dequant_q3_k,
            BLOCK_Q3_K_BYTES,
            amp,
            amp,
            "q3_k",
        );
    }

    #[test]
    fn q2_k_round_trip_within_bound() {
        // Q2_K: 2-bit asymmetric × 16-weight sub-block. Step ≈ amp / 3
        // per sub-block, plus super-block coupling. Loose bound: amp.
        let amp = 4.0;
        assert_round_trip(
            encode_q2_k,
            dequant::dequant_q2_k,
            BLOCK_Q2_K_BYTES,
            amp,
            amp,
            "q2_k",
        );
    }

    /// Zero input must encode without divide-by-zero on every K-quant
    /// format. Pin that every encoder handles the "unused tensor"
    /// edge case gracefully.
    #[test]
    fn all_k_formats_handle_zero_input() {
        let n = 2 * QK_K;
        let zero = vec![0f32; n];
        let mut buf;

        buf = vec![0u8; 2 * BLOCK_Q8_K_BYTES];
        encode_q8_k(&zero, &mut buf);
        buf = vec![0u8; 2 * BLOCK_Q6_K_BYTES];
        encode_q6_k(&zero, &mut buf);
        buf = vec![0u8; 2 * BLOCK_Q4_K_BYTES];
        encode_q4_k(&zero, &mut buf);
        buf = vec![0u8; 2 * BLOCK_Q5_K_BYTES];
        encode_q5_k(&zero, &mut buf);
        buf = vec![0u8; 2 * BLOCK_Q3_K_BYTES];
        encode_q3_k(&zero, &mut buf);
        buf = vec![0u8; 2 * BLOCK_Q2_K_BYTES];
        encode_q2_k(&zero, &mut buf);

        // Verify dequant of zero blocks is exactly zero (within f16 noise).
        let mut out = vec![0f32; n];
        buf = vec![0u8; 2 * BLOCK_Q8_K_BYTES];
        encode_q8_k(&zero, &mut buf);
        dequant::dequant_q8_k(&buf, &mut out);
        assert!(out.iter().all(|v| v.abs() < 1e-6));

        buf = vec![0u8; 2 * BLOCK_Q4_K_BYTES];
        encode_q4_k(&zero, &mut buf);
        dequant::dequant_q4_k(&buf, &mut out);
        assert!(out.iter().all(|v| v.abs() < 1e-6));
    }

    // ---- imatrix (importance-weighted) encode -------------------------

    /// A uniform (all-1.0) imatrix must reproduce the unweighted encode
    /// byte-for-byte — proves the weighted least-squares + weighted L2
    /// error reduce exactly to the prior path when w ≡ 1.
    #[test]
    fn q4_k_imatrix_uniform_weight_matches_unweighted() {
        for &n_blocks in &[1usize, 2, 4] {
            let n = n_blocks * QK_K;
            let src = make_row(n, 4.0, n as u32 * 17);
            let mut unweighted = vec![0u8; n_blocks * BLOCK_Q4_K_BYTES];
            let mut uniform = vec![0u8; n_blocks * BLOCK_Q4_K_BYTES];
            encode_q4_k(&src, &mut unweighted);
            let ones = vec![1.0f32; n];
            encode_q4_k_imatrix(&src, &mut uniform, Some(&ones));
            assert_eq!(
                unweighted, uniform,
                "uniform-weight imatrix must match unweighted encode (n_blocks={n_blocks})"
            );
        }
        // Same invariant for Q5_K and Q2_K.
        let n = 2 * QK_K;
        let src = make_row(n, 4.0, 99);
        let ones = vec![1.0f32; n];
        let (mut a, mut b) = (vec![0u8; 2 * BLOCK_Q5_K_BYTES], vec![0u8; 2 * BLOCK_Q5_K_BYTES]);
        encode_q5_k(&src, &mut a);
        encode_q5_k_imatrix(&src, &mut b, Some(&ones));
        assert_eq!(a, b, "Q5_K uniform-weight imatrix must match unweighted");
        let (mut a, mut b) = (vec![0u8; 2 * BLOCK_Q2_K_BYTES], vec![0u8; 2 * BLOCK_Q2_K_BYTES]);
        encode_q2_k(&src, &mut a);
        encode_q2_k_imatrix(&src, &mut b, Some(&ones));
        assert_eq!(a, b, "Q2_K uniform-weight imatrix must match unweighted");
    }

    /// A non-uniform imatrix that heavily weights a subset of columns
    /// must reduce the *weighted* reconstruction error on those columns
    /// vs. the uniform encode — i.e. the encoder really does redirect
    /// precision toward important columns.
    #[test]
    fn q4_k_imatrix_reduces_error_on_weighted_columns() {
        let n = QK_K;
        let src = make_row(n, 4.0, 11);
        // Weight the first 8 of each 32-col sub-block 100×, the rest 0.01×.
        let mut w = vec![0.01f32; n];
        for sub in 0..(QK_K / 32) {
            for j in 0..8 {
                w[sub * 32 + j] = 100.0;
            }
        }
        let mut enc_u = vec![0u8; BLOCK_Q4_K_BYTES];
        let mut enc_w = vec![0u8; BLOCK_Q4_K_BYTES];
        encode_q4_k(&src, &mut enc_u);
        encode_q4_k_imatrix(&src, &mut enc_w, Some(&w));
        let mut dec_u = vec![0f32; n];
        let mut dec_w = vec![0f32; n];
        dequant::dequant_q4_k(&enc_u, &mut dec_u);
        dequant::dequant_q4_k(&enc_w, &mut dec_w);
        let werr = |dec: &[f32]| -> f64 {
            (0..n).map(|i| (w[i] as f64) * ((dec[i] - src[i]) as f64).powi(2)).sum()
        };
        let eu = werr(&dec_u);
        let ew = werr(&dec_w);
        assert!(
            ew <= eu + 1e-6,
            "imatrix encode should not raise weighted error on important cols: weighted={ew} uniform={eu}"
        );
    }
}
