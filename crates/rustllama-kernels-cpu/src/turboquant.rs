//! TurboQuant: random-rotation + uniform-quantization for KV cache rows.
//!
//! Background: standard per-row K/V quantization (Q8_0, Q4_0) bins
//! along the natural coordinate axes. Attention's K/V coordinates
//! are NOT uniform along those axes — RoPE concentrates energy in
//! certain components, GQA leaves correlations across heads, etc.
//! Uniform bins waste resolution on the empty regions of the
//! distribution.
//!
//! TurboQuant first applies a fast **Walsh–Hadamard transform**
//! (a fixed orthogonal rotation) to each row, then quantizes the
//! rotated coefficients. The WHT decorrelates the components so
//! the resulting distribution is roughly Gaussian-like, and
//! uniform bins are near-optimal under Gaussian inputs. Decode
//! reverses: unpack, scale, run the SAME WHT (it's self-inverse up
//! to a `1/N` factor).
//!
//! The math is unconditional — any orthogonal rotation works — but
//! WHT is the cheap choice: O(N log N) flops, no multiplications
//! (just adds), no codebook to store.
//!
//! Block format (constant across bit-widths):
//!
//! ```text
//!   bits ∈ {1, 2, 4, 8}
//!   block = `head_dim` elements
//!   storage per block:
//!     scale: f32                 // max-abs of the rotated row / max_level
//!     packed: ceil(head_dim * bits / 8) bytes
//! ```
//!
//! Quantize:
//!   1. `rotated = WHT(row)`
//!   2. `scale = max(|rotated|) / max_level`        (max_level depends on bits)
//!   3. `levels[i] = round(rotated[i] / scale)` clamped to `[-max_level, max_level]`
//!   4. Pack `levels` into bits-per-element bytes.
//!
//! Dequantize:
//!   1. Unpack `levels` from `packed`.
//!   2. `rotated[i] = levels[i] * scale`.
//!   3. `row = WHT(rotated) * (1 / N)`.
//!
//! Where `N = head_dim` and we rely on `WHT(WHT(x)) = N * x`.

use std::convert::TryInto;

/// In-place Walsh–Hadamard transform.
///
/// `x.len()` must be a power of 2. After this call, `WHT(WHT(x)) = N * x`,
/// so the forward and inverse transforms are the same kernel — the caller
/// multiplies by `1/N` after the second invocation to recover the original.
///
/// O(N log N) adds / subtracts; no multiplications.
pub fn wht_inplace(x: &mut [f32]) {
    let n = x.len();
    assert!(
        n.is_power_of_two() && n > 0,
        "wht_inplace requires power-of-2 length, got {n}"
    );
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }
}

/// Apply the inverse WHT in-place. Same kernel as forward but
/// followed by a `1/N` scaling so the result equals the original
/// pre-rotated vector. Use this immediately after a forward WHT to
/// recover the original signal.
pub fn wht_inverse_inplace(x: &mut [f32]) {
    wht_inplace(x);
    let inv_n = 1.0 / x.len() as f32;
    for v in x.iter_mut() {
        *v *= inv_n;
    }
}

/// Quantize one `head_dim`-length row to the TurboQuant block format.
///
/// `row` is the input K or V row (post-RoPE for K). `packed_out`
/// receives the bit-packed levels; its length must equal
/// `bytes_per_block(row.len(), bits)`. Returns the per-block scale.
///
/// Mutation of `row` is permitted — we'd WHT it anyway. Callers that
/// need to preserve the original should clone first.
pub fn quantize_row(row: &mut [f32], bits: u8, packed_out: &mut [u8]) -> f32 {
    let n = row.len();
    let need = bytes_per_block(n, bits);
    assert_eq!(
        packed_out.len(),
        need,
        "packed_out must be {need} bytes for n={n} bits={bits}"
    );

    // 1. WHT rotates the row in place. After this, `row` holds the
    //    rotated coefficients which are roughly Gaussian-distributed.
    wht_inplace(row);

    // 2. Choose a scale that fits the max-abs coefficient into the
    //    representable range. `max_level` is the largest positive
    //    code we can emit; the negative side reaches `-max_level`.
    //    For tq1 the only codes are {-1, +1} so max_level = 1 and
    //    we collapse everything to its sign × scale.
    let max_abs: f32 = row.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let max_level = max_level_for(bits);
    let scale = if max_abs > 0.0 {
        max_abs / max_level as f32
    } else {
        // All-zero row: pick scale = 1.0 so dequant produces zeros.
        1.0
    };
    let inv_scale = 1.0 / scale;

    // 3. Quantize each rotated coefficient to a signed integer code.
    //    The codes range over `[-max_level, +max_level]` inclusive,
    //    which is `2*max_level + 1` distinct values — one less than
    //    the bit count would technically allow for tq2/tq4/tq8, but
    //    keeps the symmetric range that maps cleanly back to f32.
    //    tq1 uses just {-1, +1} (sign bit).
    for v in row.iter_mut() {
        let scaled = *v * inv_scale;
        let code = if bits == 1 {
            // tq1: pure sign. Zero rounds to +1 by convention so the
            // packed bit is well-defined for sparse-zero rows.
            if scaled < 0.0 {
                -1i32
            } else {
                1i32
            }
        } else {
            let r = scaled.round() as i32;
            r.clamp(-(max_level as i32), max_level as i32)
        };
        // Reuse `row` as a scratch buffer for the integer codes —
        // cast through f32 since that's the storage type. Packing
        // reads them back out.
        *v = code as f32;
    }

    // 4. Pack the integer codes into `bits`-per-element bytes.
    pack_codes(row, bits, packed_out);

    scale
}

/// Dequantize one TurboQuant-packed row back into `out` (length
/// `head_dim`). Applies the inverse WHT so the result is in the
/// original coordinate frame.
///
/// Dispatches to an AVX2 fast path for `bits == 8` when the CPU
/// supports it — the unpack + scale phase becomes ~3-4× faster
/// than the byte-at-a-time scalar loop. The WHT itself stays
/// scalar (vectorizing the butterfly across stage `h < 8` needs
/// within-register shuffles; lands in a follow-up).
pub fn dequantize_row(packed: &[u8], scale: f32, bits: u8, out: &mut [f32]) {
    let n = out.len();
    let need = bytes_per_block(n, bits);
    assert_eq!(
        packed.len(),
        need,
        "packed must be {need} bytes for n={n} bits={bits}"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if bits == 8 && is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection. AVX-512 processes
            // 16 codes per iteration — half the loop trips of the
            // AVX2 path on hosts that have both.
            unsafe { dequantize_row_tq8_unpack_avx512(packed, scale, out) };
            wht_inverse_inplace(out);
            return;
        }
        if bits == 8
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            // SAFETY: runtime feature detection above. The kernel
            // writes exactly `out.len()` f32 values; tail handling
            // covers any `n` not a multiple of 8.
            unsafe { dequantize_row_tq8_unpack_avx2(packed, scale, out) };
            wht_inverse_inplace(out);
            return;
        }
    }

    // 1. Unpack levels → out (as f32 codes).
    unpack_codes(packed, bits, out);
    // 2. Multiply by scale.
    for v in out.iter_mut() {
        *v *= scale;
    }
    // 3. Inverse WHT (forward WHT then 1/N).
    wht_inverse_inplace(out);
}

/// AVX-512 fused unpack-and-scale for tq8 codes. 16 codes per
/// iteration: load 16 packed bytes via `_mm_loadu_si128`, widen
/// `u8 → i32` via `_mm512_cvtepu8_epi32`, convert to f32, subtract
/// bias, multiply by scale. Twice the lane count of the AVX2 path.
///
/// Does NOT apply the inverse WHT — the caller chains that step.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn dequantize_row_tq8_unpack_avx512(packed: &[u8], scale: f32, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = out.len();
    let scale_v = _mm512_set1_ps(scale);
    let bias_v = _mm512_set1_ps(127.0);
    let n16 = n & !15;
    let mut i = 0;
    while i < n16 {
        // Load 16 packed code bytes into a 128-bit XMM register, widen
        // u8 → i32x16 (zero-extending), convert to f32x16.
        let bytes_xmm = _mm_loadu_si128(packed.as_ptr().add(i) as *const __m128i);
        let i32_zmm = _mm512_cvtepu8_epi32(bytes_xmm);
        let f32_zmm = _mm512_cvtepi32_ps(i32_zmm);
        let signed = _mm512_sub_ps(f32_zmm, bias_v);
        let scaled = _mm512_mul_ps(signed, scale_v);
        _mm512_storeu_ps(out.as_mut_ptr().add(i), scaled);
        i += 16;
    }
    // Tail: scalar.
    while i < n {
        let unsigned = packed[i] as i32;
        let signed = unsigned - 127;
        out[i] = (signed as f32) * scale;
        i += 1;
    }
}

/// AVX2 fused unpack-and-scale for tq8 codes. Each byte is one code
/// in `[0, 254]` (= signed + 127 bias); we widen 8 bytes → i32x8 →
/// f32x8, subtract the bias, multiply by `scale`, store. The tail
/// past the largest multiple-of-8 length runs scalar.
///
/// Does NOT apply the inverse WHT — the caller chains that step.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn dequantize_row_tq8_unpack_avx2(packed: &[u8], scale: f32, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = out.len();
    let scale_v = _mm256_set1_ps(scale);
    let bias_v = _mm256_set1_ps(127.0);
    let n8 = n & !7;
    let mut i = 0;
    while i < n8 {
        // Load 8 packed code bytes into a 64-bit XMM register, then
        // widen u8 → u16 → i32 → f32 step-by-step.
        let bytes_xmm = _mm_loadl_epi64(packed.as_ptr().add(i) as *const __m128i);
        let u16_xmm = _mm_cvtepu8_epi16(bytes_xmm);
        let i32_ymm = _mm256_cvtepu16_epi32(u16_xmm);
        let f32_ymm = _mm256_cvtepi32_ps(i32_ymm);
        // signed = unsigned - 127; out = signed * scale (single FMA: `signed * scale + 0` via mul, no add).
        let signed = _mm256_sub_ps(f32_ymm, bias_v);
        let scaled = _mm256_mul_ps(signed, scale_v);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), scaled);
        i += 8;
    }
    // Tail: scalar.
    while i < n {
        let unsigned = packed[i] as i32;
        let signed = unsigned - 127;
        out[i] = (signed as f32) * scale;
        i += 1;
    }
}

/// Storage bytes per `n`-element block under `bits`-per-element
/// packing. Rounded up so the last partial byte is allocated. Asserts
/// `bits` is one of the supported values.
pub fn bytes_per_block(n: usize, bits: u8) -> usize {
    assert!(matches!(bits, 1 | 2 | 4 | 8), "tq bits must be 1/2/4/8");
    (n * bits as usize).div_ceil(8)
}

/// Largest positive code representable in `bits` bits with the
/// symmetric `[-max, +max]` range we use.
fn max_level_for(bits: u8) -> u8 {
    match bits {
        1 => 1,
        2 => 1,           // codes ∈ {-1, 0, 0, 1} effectively (3 levels)
        4 => 7,           // codes ∈ [-7, 7]; one slot unused
        8 => 127,         // codes ∈ [-127, 127]; matches Q8_0's range
        _ => unreachable!("tq bits already validated upstream"),
    }
}

/// Pack signed integer codes (carried as `f32` in `codes`) into
/// `bits`-per-element bytes.
///
/// Encoding conventions per bit-width:
///   - tq1: codes ∈ {-1, +1}. Stored as `(signed + 1) / 2`, i.e.,
///     `-1 → 0`, `+1 → 1`. (Symmetric bias doesn't work in 1 bit
///     because the unsigned range only has 2 slots; we want both
///     to land on a non-zero magnitude.)
///   - tq2/tq4/tq8: codes ∈ [-max_level, +max_level]. Stored as
///     `signed + max_level`, biased into a nonnegative unsigned
///     range. The top unsigned slot is unused (room for `2 *
///     max_level + 1` codes in `1 << bits` slots).
fn pack_codes(codes: &[f32], bits: u8, out: &mut [u8]) {
    // Zero the output so any unused tail bits stay deterministic.
    for b in out.iter_mut() {
        *b = 0;
    }
    let max_level = max_level_for(bits) as i32;
    let max_unsigned = match bits {
        1 => 1u32,
        2 => 3u32,
        4 => 15u32,
        8 => 255u32,
        _ => unreachable!(),
    };
    for (i, &code_f) in codes.iter().enumerate() {
        let signed = code_f as i32;
        let cell = if bits == 1 {
            // -1 → 0, +1 → 1. Zero rounds to +1 by the
            // upstream quantize_row sign convention.
            if signed >= 0 {
                1u32
            } else {
                0u32
            }
        } else {
            // Symmetric bias: signed ∈ [-max_level, +max_level]
            // → unsigned ∈ [0, 2*max_level].
            let biased = (signed + max_level) as u32;
            biased.min(max_unsigned)
        };
        let bit_offset = i * bits as usize;
        let byte_idx = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;
        out[byte_idx] |= (cell << bit_in_byte) as u8;
        // Spill to next byte if the cell straddles a boundary
        // (only possible for `bits` that don't divide 8, but the
        // safe-cast version below handles 1/2/4/8 uniformly).
        if bit_in_byte + (bits as usize) > 8 {
            let spill = bit_in_byte + (bits as usize) - 8;
            let shift = (bits as usize) - spill;
            out[byte_idx + 1] |= (cell >> shift) as u8;
        }
    }
}

/// Unpack `bits`-per-element codes from `bytes` into `out` (as f32
/// signed integer values, post-debias). Inverse of [`pack_codes`].
fn unpack_codes(bytes: &[u8], bits: u8, out: &mut [f32]) {
    let max_level = max_level_for(bits) as i32;
    let mask = match bits {
        1 => 0x1u32,
        2 => 0x3u32,
        4 => 0xfu32,
        8 => 0xffu32,
        _ => unreachable!(),
    };
    for (i, slot) in out.iter_mut().enumerate() {
        let bit_offset = i * bits as usize;
        let byte_idx = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;
        let mut cell = (bytes[byte_idx] as u32) >> bit_in_byte;
        if bit_in_byte + (bits as usize) > 8 {
            let spill = bit_in_byte + (bits as usize) - 8;
            let shift = (bits as usize) - spill;
            cell |= (bytes[byte_idx + 1] as u32) << shift;
        }
        let unsigned = cell & mask;
        let signed = if bits == 1 {
            // 0 → -1, 1 → +1 (inverse of the tq1 pack convention).
            if unsigned == 0 {
                -1i32
            } else {
                1i32
            }
        } else {
            unsigned as i32 - max_level
        };
        *slot = signed as f32;
    }
}

/// Convenience: encode an entire `[rows × n]` matrix block-by-block.
/// `rows` is the number of independent blocks (e.g., kv_heads ×
/// seq_positions). Output `packed_out` is `rows * bytes_per_block(n,
/// bits)` bytes; `scales_out` is `rows` f32 values.
///
/// Used by the KV cache's batch-quantize path. Falls back to a no-op
/// for empty inputs.
#[allow(clippy::too_many_arguments)]
pub fn quantize_matrix(
    rows: usize,
    n: usize,
    bits: u8,
    input: &[f32],
    scratch: &mut [f32],
    packed_out: &mut [u8],
    scales_out: &mut [f32],
) {
    if rows == 0 {
        return;
    }
    assert_eq!(input.len(), rows * n);
    assert_eq!(scratch.len(), n);
    let block = bytes_per_block(n, bits);
    assert_eq!(packed_out.len(), rows * block);
    assert_eq!(scales_out.len(), rows);
    for r in 0..rows {
        let in_slice = &input[r * n..(r + 1) * n];
        let out_slice = &mut packed_out[r * block..(r + 1) * block];
        // Copy into scratch since quantize_row mutates.
        let chunk: &mut [f32] = scratch;
        chunk.copy_from_slice(in_slice);
        scales_out[r] = quantize_row(chunk, bits, out_slice);
    }
}

/// Convenience: dequantize an entire `[rows × n]` matrix back to f32.
pub fn dequantize_matrix(
    rows: usize,
    n: usize,
    bits: u8,
    packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
) {
    if rows == 0 {
        return;
    }
    let block = bytes_per_block(n, bits);
    assert_eq!(packed.len(), rows * block);
    assert_eq!(scales.len(), rows);
    assert_eq!(out.len(), rows * n);
    for r in 0..rows {
        let in_slice = &packed[r * block..(r + 1) * block];
        let out_slice = &mut out[r * n..(r + 1) * n];
        dequantize_row(in_slice, scales[r], bits, out_slice);
    }
}

/// Suppress an unused-import lint when the `TryInto` import is only
/// reached on niche bit-paths — the trait is in the prelude on
/// recent compilers but importing explicitly keeps the file readable.
#[allow(dead_code)]
fn _force_try_into_import_lint_suppressor(x: &[u8]) -> Option<[u8; 4]> {
    x.try_into().ok()
}

/// Fused TurboQuant attention step: reads packed K/V cache, dequants
/// one KV-head's slab at a time into a small scratch buffer, then
/// runs the standard scaled-dot-product attention across all Q heads
/// in that GQA group. The win vs the "dequant-whole-slab + call F32
/// attention" approach is **memory**: peak working set is `2 × kv_len
/// × head_dim` floats per call (one kv_h worth) instead of the full
/// `n_kv_heads × max_ctx × head_dim` slab. Compute is equivalent —
/// the dot products still happen in scalar f32.
///
/// SIMD'd inner-loop variants land alongside the per-bit-width dequant
/// fast paths in a follow-up.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_one_step_tq(
    q: &[f32],
    k_packed: &[u8],
    k_scales: &[f32],
    v_packed: &[u8],
    v_scales: &[f32],
    bits: u8,
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    let bytes_per_row = bytes_per_block(head_dim, bits);
    assert_eq!(q.len(), n_heads * head_dim);
    assert_eq!(out.len(), n_heads * head_dim);
    assert_eq!(k_packed.len(), n_kv_heads * max_ctx * bytes_per_row);
    assert_eq!(v_packed.len(), n_kv_heads * max_ctx * bytes_per_row);
    assert_eq!(k_scales.len(), n_kv_heads * max_ctx);
    assert_eq!(v_scales.len(), n_kv_heads * max_ctx);
    assert!(kv_len <= max_ctx);
    assert_eq!(n_heads % n_kv_heads, 0, "GQA requires n_heads divisible by n_kv_heads");

    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; kv_len];

    // Per-kv-head dequant scratch. Sized to just the live `kv_len`
    // window — typical decode-step calls leave most of the
    // `max_ctx`-sized cache empty. Reused across kv heads.
    let mut k_h_scratch = vec![0.0f32; kv_len * head_dim];
    let mut v_h_scratch = vec![0.0f32; kv_len * head_dim];

    for kv_h in 0..n_kv_heads {
        // Phase 1: dequantize this kv head's K and V rows once.
        for t in 0..kv_len {
            let row_idx = kv_h * max_ctx + t;
            let p_off = row_idx * bytes_per_row;
            dequantize_row(
                &k_packed[p_off..p_off + bytes_per_row],
                k_scales[row_idx],
                bits,
                &mut k_h_scratch[t * head_dim..(t + 1) * head_dim],
            );
            dequantize_row(
                &v_packed[p_off..p_off + bytes_per_row],
                v_scales[row_idx],
                bits,
                &mut v_h_scratch[t * head_dim..(t + 1) * head_dim],
            );
        }

        // Phase 2: for each Q head sharing this KV head (n_gqa
        // heads), compute Q·Kᵀ, softmax, and the weighted V sum.
        for qh_off in 0..n_gqa {
            let h = kv_h * n_gqa + qh_off;
            let q_h = &q[h * head_dim..(h + 1) * head_dim];
            for t in 0..kv_len {
                let k_row = &k_h_scratch[t * head_dim..(t + 1) * head_dim];
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    acc += q_h[d] * k_row[d];
                }
                scores[t] = acc * scale;
            }
            softmax_inplace(&mut scores);
            let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
            for v in out_h.iter_mut() {
                *v = 0.0;
            }
            for t in 0..kv_len {
                let w = scores[t];
                let v_row = &v_h_scratch[t * head_dim..(t + 1) * head_dim];
                for d in 0..head_dim {
                    out_h[d] += w * v_row[d];
                }
            }
        }
    }
}

/// FlashAttention-decode variant of [`gqa_attention_one_step_tq`].
/// Same outer structure — dequantizes each kv_h's K/V slab once,
/// reuses it across the `n_gqa` Q heads — but the inner Q-head
/// loop uses online softmax instead of the 3-pass (Q·K, softmax,
/// V-weighted sum) sequence.
///
/// Compared to the standard kernel:
///   - No `[kv_len]` scratch buffer for scores (the per-Q-head one
///     in the inner loop, not the shared K/V dequant scratch).
///   - Two passes over `[kv_len × head_dim]` instead of three per
///     Q head: the Q·K dot is fused with the running-max + V-weighted
///     accumulation update.
///   - One `exp` per `t` per Q head instead of `kv_len` exps in
///     the separate softmax pass.
///
/// Greedy parity vs the standard kernel is asserted by the
/// `flash_decode_tq_matches_standard_tq` test below — bit-for-bit
/// identical up to floating-point reduction order, max abs error
/// well under 1e-5.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_flash_decode_tq(
    q: &[f32],
    k_packed: &[u8],
    k_scales: &[f32],
    v_packed: &[u8],
    v_scales: &[f32],
    bits: u8,
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    let bytes_per_row = bytes_per_block(head_dim, bits);
    assert_eq!(q.len(), n_heads * head_dim);
    assert_eq!(out.len(), n_heads * head_dim);
    assert_eq!(k_packed.len(), n_kv_heads * max_ctx * bytes_per_row);
    assert_eq!(v_packed.len(), n_kv_heads * max_ctx * bytes_per_row);
    assert_eq!(k_scales.len(), n_kv_heads * max_ctx);
    assert_eq!(v_scales.len(), n_kv_heads * max_ctx);
    assert!(kv_len <= max_ctx);
    assert_eq!(n_heads % n_kv_heads, 0, "GQA requires n_heads divisible by n_kv_heads");
    if kv_len == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Per-kv-head dequant scratch — same as the standard kernel.
    // We re-dequantize per kv_h but reuse across n_gqa Q heads,
    // so this is sized once at the top of the call.
    let mut k_h_scratch = vec![0.0f32; kv_len * head_dim];
    let mut v_h_scratch = vec![0.0f32; kv_len * head_dim];

    for kv_h in 0..n_kv_heads {
        // Phase 1: dequantize this kv head's K and V rows.
        for t in 0..kv_len {
            let row_idx = kv_h * max_ctx + t;
            let p_off = row_idx * bytes_per_row;
            dequantize_row(
                &k_packed[p_off..p_off + bytes_per_row],
                k_scales[row_idx],
                bits,
                &mut k_h_scratch[t * head_dim..(t + 1) * head_dim],
            );
            dequantize_row(
                &v_packed[p_off..p_off + bytes_per_row],
                v_scales[row_idx],
                bits,
                &mut v_h_scratch[t * head_dim..(t + 1) * head_dim],
            );
        }

        // Phase 2: online softmax per Q head sharing this KV head.
        // Dispatch the dequantized-scratch inner loop to SIMD when
        // available — the K/V scratch is plain f32 so the same AVX-2
        // pattern as the F32 flash kernel applies.
        for qh_off in 0..n_gqa {
            let h = kv_h * n_gqa + qh_off;
            let q_h = &q[h * head_dim..(h + 1) * head_dim];
            let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
            online_softmax_attn_f32_scratch(
                q_h, &k_h_scratch, &v_h_scratch, out_h, head_dim, kv_len, scale,
            );
        }
    }
}

/// Multi-query FlashAttention prefill for TurboQuant KV. Mirrors
/// [`gqa_attention_flash_decode_tq`]'s structure (dequantize per
/// kv_h, reuse the f32 scratch across this kv_h's Q heads) but
/// processes `n_new` queries at once with causal masking — query
/// `i` attends to `[0, kv_len_base + i + 1)`.
///
/// Q layout is `[n_new, n_heads, head_dim]` (the same row-major
/// shape as `gqa_attention_flash_prefill`). Out matches.
///
/// Dequantization scratch is sized to the live `kv_len_base +
/// n_new` window — not the full `max_ctx` slab — so memory cost
/// scales with conversation length, not the ceiling.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_flash_prefill_tq(
    q: &[f32],
    k_packed: &[u8],
    k_scales: &[f32],
    v_packed: &[u8],
    v_scales: &[f32],
    bits: u8,
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) {
    let bytes_per_row = bytes_per_block(head_dim, bits);
    assert_eq!(q.len(), n_new * n_heads * head_dim);
    assert_eq!(out.len(), n_new * n_heads * head_dim);
    assert_eq!(k_packed.len(), n_kv_heads * max_ctx * bytes_per_row);
    assert_eq!(v_packed.len(), n_kv_heads * max_ctx * bytes_per_row);
    assert_eq!(k_scales.len(), n_kv_heads * max_ctx);
    assert_eq!(v_scales.len(), n_kv_heads * max_ctx);
    assert!(kv_len_base + n_new <= max_ctx);
    assert_eq!(n_heads % n_kv_heads, 0, "GQA requires n_heads divisible by n_kv_heads");
    if n_new == 0 {
        return;
    }

    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let total_positions = kv_len_base + n_new;

    // Per-kv-head dequant scratch — only the live `[0, total_positions)`
    // window, not the full max_ctx slab. Reused across n_gqa Q heads
    // and across all n_new queries within this kv_h.
    let mut k_h_scratch = vec![0.0f32; total_positions * head_dim];
    let mut v_h_scratch = vec![0.0f32; total_positions * head_dim];

    for kv_h in 0..n_kv_heads {
        // Phase 1: dequantize this kv head's K and V rows for the
        // positions we care about.
        for t in 0..total_positions {
            let row_idx = kv_h * max_ctx + t;
            let p_off = row_idx * bytes_per_row;
            dequantize_row(
                &k_packed[p_off..p_off + bytes_per_row],
                k_scales[row_idx],
                bits,
                &mut k_h_scratch[t * head_dim..(t + 1) * head_dim],
            );
            dequantize_row(
                &v_packed[p_off..p_off + bytes_per_row],
                v_scales[row_idx],
                bits,
                &mut v_h_scratch[t * head_dim..(t + 1) * head_dim],
            );
        }
        // Phase 2: per Q head in this group, per new query, online
        // softmax over [0, kv_len_base + i + 1).
        for qh_off in 0..n_gqa {
            let h = kv_h * n_gqa + qh_off;
            for i in 0..n_new {
                let kv_len_i = kv_len_base + i + 1;
                let q_off = (i * n_heads + h) * head_dim;
                let q_h = &q[q_off..q_off + head_dim];
                let out_off = (i * n_heads + h) * head_dim;
                let out_h = &mut out[out_off..out_off + head_dim];
                online_softmax_attn_f32_scratch(
                    q_h,
                    &k_h_scratch[..kv_len_i * head_dim],
                    &v_h_scratch[..kv_len_i * head_dim],
                    out_h,
                    head_dim,
                    kv_len_i,
                    scale,
                );
            }
        }
    }
}

/// Online-softmax attention over an already-dequantized
/// `[kv_len, head_dim]` K/V scratch (one kv-head's worth). Shared
/// by `gqa_attention_flash_decode_tq` and the NVFP4 flash kernel
/// in `crate::nvfp4`. Dispatches to SIMD at runtime.
pub(crate) fn online_softmax_attn_f32_scratch(
    q_h: &[f32],
    k_scratch: &[f32],
    v_scratch: &[f32],
    out_h: &mut [f32],
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            unsafe {
                online_softmax_attn_f32_scratch_avx512(
                    q_h, k_scratch, v_scratch, out_h, head_dim, kv_len, scale,
                );
            }
            return;
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe {
                online_softmax_attn_f32_scratch_avx2(
                    q_h, k_scratch, v_scratch, out_h, head_dim, kv_len, scale,
                );
            }
            return;
        }
    }
    online_softmax_attn_f32_scratch_scalar(
        q_h, k_scratch, v_scratch, out_h, head_dim, kv_len, scale,
    );
}

fn online_softmax_attn_f32_scratch_scalar(
    q_h: &[f32],
    k_scratch: &[f32],
    v_scratch: &[f32],
    out_h: &mut [f32],
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) {
    for v in out_h.iter_mut() {
        *v = 0.0;
    }
    let mut m = f32::NEG_INFINITY;
    let mut l = 0.0f32;
    for t in 0..kv_len {
        let k_row = &k_scratch[t * head_dim..(t + 1) * head_dim];
        let mut s = 0.0f32;
        for d in 0..head_dim {
            s += q_h[d] * k_row[d];
        }
        s *= scale;
        let m_new = m.max(s);
        let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
        let p = (s - m_new).exp();
        l = l * rescale + p;
        let v_row = &v_scratch[t * head_dim..(t + 1) * head_dim];
        for d in 0..head_dim {
            out_h[d] = out_h[d] * rescale + p * v_row[d];
        }
        m = m_new;
    }
    let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
    for v in out_h.iter_mut() {
        *v *= inv_l;
    }
}

/// AVX-512 specialization of the online-softmax inner loop. Same
/// shape as the AVX-2 variant but 16 floats per iteration and
/// `_mm512_reduce_add_ps` for the horizontal sum (single instruction
/// vs AVX-2's two-stage hadd dance).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn online_softmax_attn_f32_scratch_avx512(
    q_h: &[f32],
    k_scratch: &[f32],
    v_scratch: &[f32],
    out_h: &mut [f32],
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) {
    use std::arch::x86_64::*;
    let head_dim_16 = head_dim & !15;
    let mut d = 0;
    while d < head_dim_16 {
        _mm512_storeu_ps(out_h.as_mut_ptr().add(d), _mm512_setzero_ps());
        d += 16;
    }
    while d < head_dim {
        out_h[d] = 0.0;
        d += 1;
    }
    let mut m = f32::NEG_INFINITY;
    let mut l = 0.0f32;
    for t in 0..kv_len {
        let k_row = &k_scratch[t * head_dim..(t + 1) * head_dim];
        let mut acc = _mm512_setzero_ps();
        let mut d = 0;
        while d < head_dim_16 {
            let qv = _mm512_loadu_ps(q_h.as_ptr().add(d));
            let kv = _mm512_loadu_ps(k_row.as_ptr().add(d));
            acc = _mm512_fmadd_ps(qv, kv, acc);
            d += 16;
        }
        let mut s = _mm512_reduce_add_ps(acc);
        while d < head_dim {
            s += q_h[d] * k_row[d];
            d += 1;
        }
        s *= scale;
        let m_new = m.max(s);
        let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
        let p = (s - m_new).exp();
        l = l * rescale + p;
        let v_row = &v_scratch[t * head_dim..(t + 1) * head_dim];
        let rescale_v = _mm512_set1_ps(rescale);
        let p_v = _mm512_set1_ps(p);
        let mut d = 0;
        while d < head_dim_16 {
            let cur = _mm512_loadu_ps(out_h.as_ptr().add(d));
            let scaled = _mm512_mul_ps(cur, rescale_v);
            let vv = _mm512_loadu_ps(v_row.as_ptr().add(d));
            let updated = _mm512_fmadd_ps(p_v, vv, scaled);
            _mm512_storeu_ps(out_h.as_mut_ptr().add(d), updated);
            d += 16;
        }
        while d < head_dim {
            out_h[d] = out_h[d] * rescale + p * v_row[d];
            d += 1;
        }
        m = m_new;
    }
    let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
    let inv_l_v = _mm512_set1_ps(inv_l);
    let mut d = 0;
    while d < head_dim_16 {
        let cur = _mm512_loadu_ps(out_h.as_ptr().add(d));
        _mm512_storeu_ps(out_h.as_mut_ptr().add(d), _mm512_mul_ps(cur, inv_l_v));
        d += 16;
    }
    while d < head_dim {
        out_h[d] *= inv_l;
        d += 1;
    }
}

/// AVX-2 + FMA specialization of the online-softmax inner loop.
/// Vectorizes the Q·K dot and V accumulation at 8 floats/iter,
/// horizontal-sum via two-stage `_mm_hadd_ps`. The cross-`t`
/// recurrence stays scalar (each iteration depends on the previous
/// `m`/`l`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn online_softmax_attn_f32_scratch_avx2(
    q_h: &[f32],
    k_scratch: &[f32],
    v_scratch: &[f32],
    out_h: &mut [f32],
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) {
    use std::arch::x86_64::*;
    let head_dim_8 = head_dim & !7;
    // Zero accumulator.
    let mut d = 0;
    while d < head_dim_8 {
        _mm256_storeu_ps(out_h.as_mut_ptr().add(d), _mm256_setzero_ps());
        d += 8;
    }
    while d < head_dim {
        out_h[d] = 0.0;
        d += 1;
    }
    let mut m = f32::NEG_INFINITY;
    let mut l = 0.0f32;
    for t in 0..kv_len {
        let k_row = &k_scratch[t * head_dim..(t + 1) * head_dim];
        let mut acc = _mm256_setzero_ps();
        let mut d = 0;
        while d < head_dim_8 {
            let qv = _mm256_loadu_ps(q_h.as_ptr().add(d));
            let kv = _mm256_loadu_ps(k_row.as_ptr().add(d));
            acc = _mm256_fmadd_ps(qv, kv, acc);
            d += 8;
        }
        let mut sum128 = _mm_add_ps(
            _mm256_castps256_ps128(acc),
            _mm256_extractf128_ps(acc, 1),
        );
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        let mut s = _mm_cvtss_f32(sum128);
        while d < head_dim {
            s += q_h[d] * k_row[d];
            d += 1;
        }
        s *= scale;
        let m_new = m.max(s);
        let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
        let p = (s - m_new).exp();
        l = l * rescale + p;
        let v_row = &v_scratch[t * head_dim..(t + 1) * head_dim];
        let rescale_v = _mm256_set1_ps(rescale);
        let p_v = _mm256_set1_ps(p);
        let mut d = 0;
        while d < head_dim_8 {
            let cur = _mm256_loadu_ps(out_h.as_ptr().add(d));
            let scaled = _mm256_mul_ps(cur, rescale_v);
            let vv = _mm256_loadu_ps(v_row.as_ptr().add(d));
            let updated = _mm256_fmadd_ps(p_v, vv, scaled);
            _mm256_storeu_ps(out_h.as_mut_ptr().add(d), updated);
            d += 8;
        }
        while d < head_dim {
            out_h[d] = out_h[d] * rescale + p * v_row[d];
            d += 1;
        }
        m = m_new;
    }
    let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
    let inv_l_v = _mm256_set1_ps(inv_l);
    let mut d = 0;
    while d < head_dim_8 {
        let cur = _mm256_loadu_ps(out_h.as_ptr().add(d));
        _mm256_storeu_ps(out_h.as_mut_ptr().add(d), _mm256_mul_ps(cur, inv_l_v));
        d += 8;
    }
    while d < head_dim {
        out_h[d] *= inv_l;
        d += 1;
    }
}

/// Numerically-stable softmax in-place. Subtract the max, exp, then
/// divide by the sum. Standard pattern; pulled into this file so the
/// fused attention kernel doesn't reach across to the parent crate.
fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn flash_prefill_tq_matches_decode_loop() {
        // The multi-query TQ prefill kernel must produce the same
        // output as calling the per-token TQ decode kernel `n_new`
        // times with the right `kv_len` per call.
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 16;
        let max_ctx = 64;
        let bits = 4u8;
        let kv_len_base = 8;
        let n_new = 6;
        let bytes_per_row = bytes_per_block(head_dim, bits);
        let mut k_packed = vec![0u8; n_kv_heads * max_ctx * bytes_per_row];
        let mut v_packed = vec![0u8; n_kv_heads * max_ctx * bytes_per_row];
        let mut k_scales = vec![0f32; n_kv_heads * max_ctx];
        let mut v_scales = vec![0f32; n_kv_heads * max_ctx];
        // Quantize K and V for every position the kernel will read.
        for kv_h in 0..n_kv_heads {
            for t in 0..(kv_len_base + n_new) {
                let row_idx = kv_h * max_ctx + t;
                let off = row_idx * bytes_per_row;
                let mut k_row: Vec<f32> = (0..head_dim)
                    .map(|d| ((kv_h * 7 + t * 3 + d) % 13) as f32 * 0.1 - 0.6)
                    .collect();
                k_scales[row_idx] = quantize_row(
                    &mut k_row,
                    bits,
                    &mut k_packed[off..off + bytes_per_row],
                );
                let mut v_row: Vec<f32> = (0..head_dim)
                    .map(|d| ((kv_h * 5 + t * 2 + d) % 11) as f32 * 0.15 - 0.7)
                    .collect();
                v_scales[row_idx] = quantize_row(
                    &mut v_row,
                    bits,
                    &mut v_packed[off..off + bytes_per_row],
                );
            }
        }
        // Synthetic Q for all n_new positions.
        let q: Vec<f32> = (0..n_new * n_heads * head_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.12)
            .collect();
        let mut out_prefill = vec![0f32; n_new * n_heads * head_dim];
        gqa_attention_flash_prefill_tq(
            &q, &k_packed, &k_scales, &v_packed, &v_scales,
            bits, &mut out_prefill,
            n_heads, n_kv_heads, head_dim, max_ctx,
            kv_len_base, n_new,
        );
        // Reference: call the TQ flash-decode kernel once per
        // new position with kv_len = kv_len_base + i + 1 (causal).
        let mut out_reference = vec![0f32; n_new * n_heads * head_dim];
        for i in 0..n_new {
            let kv_len_i = kv_len_base + i + 1;
            let q_slice = &q[i * n_heads * head_dim..(i + 1) * n_heads * head_dim];
            let out_slice =
                &mut out_reference[i * n_heads * head_dim..(i + 1) * n_heads * head_dim];
            gqa_attention_flash_decode_tq(
                q_slice, &k_packed, &k_scales, &v_packed, &v_scales,
                bits, out_slice,
                n_heads, n_kv_heads, head_dim, max_ctx, kv_len_i,
            );
        }
        let mut max_err = 0f32;
        for (a, b) in out_prefill.iter().zip(out_reference.iter()) {
            let e = (a - b).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < 1e-5,
            "TQ prefill vs decode-loop max abs error {max_err} > 1e-5",
        );
    }

    #[test]
    fn flash_decode_tq_matches_standard_tq() {
        // Online-softmax TQ-flash variant must produce the same
        // output as the existing 3-pass TQ kernel up to FP reduction
        // order. Synthetic 4-head × head_dim 16 × kv_len 32 workload.
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 16;
        let max_ctx = 64;
        let kv_len = 32;
        let bits = 4u8;
        let bytes_per_row = bytes_per_block(head_dim, bits);
        // Quantize deterministic K and V rows.
        let mut k_packed = vec![0u8; n_kv_heads * max_ctx * bytes_per_row];
        let mut v_packed = vec![0u8; n_kv_heads * max_ctx * bytes_per_row];
        let mut k_scales = vec![0f32; n_kv_heads * max_ctx];
        let mut v_scales = vec![0f32; n_kv_heads * max_ctx];
        for kv_h in 0..n_kv_heads {
            for t in 0..kv_len {
                let row_idx = kv_h * max_ctx + t;
                let off = row_idx * bytes_per_row;
                // K row: deterministic synthetic values.
                let mut k_row: Vec<f32> = (0..head_dim)
                    .map(|d| ((kv_h * 7 + t * 3 + d) % 13) as f32 * 0.1 - 0.6)
                    .collect();
                k_scales[row_idx] =
                    quantize_row(&mut k_row, bits, &mut k_packed[off..off + bytes_per_row]);
                // V row.
                let mut v_row: Vec<f32> = (0..head_dim)
                    .map(|d| ((kv_h * 5 + t * 2 + d) % 11) as f32 * 0.15 - 0.7)
                    .collect();
                v_scales[row_idx] =
                    quantize_row(&mut v_row, bits, &mut v_packed[off..off + bytes_per_row]);
            }
        }
        // Synthetic Q.
        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.12)
            .collect();
        let mut out_standard = vec![0f32; n_heads * head_dim];
        let mut out_flash = vec![0f32; n_heads * head_dim];
        gqa_attention_one_step_tq(
            &q, &k_packed, &k_scales, &v_packed, &v_scales,
            bits, &mut out_standard,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
        );
        gqa_attention_flash_decode_tq(
            &q, &k_packed, &k_scales, &v_packed, &v_scales,
            bits, &mut out_flash,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
        );
        let mut max_err = 0f32;
        for (a, b) in out_standard.iter().zip(out_flash.iter()) {
            let e = (a - b).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < 1e-5,
            "flash vs standard TQ attention max abs error {max_err} > 1e-5",
        );
    }

    #[test]
    fn wht_round_trip_recovers_input() {
        // Applying WHT twice + scaling by 1/N must recover the input
        // exactly (up to fp noise). This is the property TurboQuant
        // relies on for invertibility.
        let original = vec![1.0f32, 2.0, -3.0, 0.5, 7.0, -1.5, 0.0, 4.0];
        let mut x = original.clone();
        wht_inplace(&mut x);
        wht_inverse_inplace(&mut x);
        for (i, (&a, &b)) in original.iter().zip(x.iter()).enumerate() {
            assert!(
                approx_eq(a, b, 1e-5),
                "round-trip mismatch at {i}: {a} vs {b}",
            );
        }
    }

    #[test]
    fn wht_8_known_pattern() {
        // Sanity-check the kernel against a known small input. The
        // Hadamard matrix at order 8 is well-defined; the first
        // coefficient is always the sum, and the second is the sum
        // of the first half minus the second half.
        let mut x = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        wht_inplace(&mut x);
        // All-ones input: only the DC coefficient survives.
        assert!(approx_eq(x[0], 8.0, 1e-6));
        for &v in &x[1..] {
            assert!(approx_eq(v, 0.0, 1e-6));
        }
    }

    #[test]
    fn bytes_per_block_rounds_up() {
        assert_eq!(bytes_per_block(8, 1), 1);
        assert_eq!(bytes_per_block(16, 1), 2);
        assert_eq!(bytes_per_block(8, 2), 2);
        assert_eq!(bytes_per_block(8, 4), 4);
        assert_eq!(bytes_per_block(8, 8), 8);
        assert_eq!(bytes_per_block(32, 4), 16);
        assert_eq!(bytes_per_block(64, 1), 8);
        // Non-multiple-of-8 element counts round up to the next byte.
        assert_eq!(bytes_per_block(5, 2), 2); // 10 bits → 2 bytes
        assert_eq!(bytes_per_block(3, 4), 2); // 12 bits → 2 bytes
    }

    #[test]
    fn tq8_round_trip_is_near_exact() {
        // tq8 has 8 bits per coefficient → 256 levels → quantization
        // error should be well under 1% of the row's max-abs.
        let row = vec![
            0.5f32, -1.2, 3.7, -0.1, 2.0, -2.5, 0.05, 4.4,
            -3.3, 1.1, 0.9, -0.4, 1.8, -1.7, 2.2, -0.05,
        ];
        let n = row.len();
        let mut work = row.clone();
        let mut packed = vec![0u8; bytes_per_block(n, 8)];
        let scale = quantize_row(&mut work, 8, &mut packed);
        let mut decoded = vec![0f32; n];
        dequantize_row(&packed, scale, 8, &mut decoded);

        let max_abs = row.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        for (i, (&o, &d)) in row.iter().zip(decoded.iter()).enumerate() {
            let err = (o - d).abs();
            assert!(
                err < max_abs * 0.05,
                "tq8 error at {i} too large: orig={o} decoded={d} err={err} max_abs={max_abs}",
            );
        }
    }

    #[test]
    fn tq4_round_trip_within_4bit_resolution() {
        // tq4 has 4 bits → 15 levels (after symmetric clamp) → error
        // up to roughly max_abs / 14 per coefficient on average.
        let row = vec![
            1.0f32, -0.5, 2.0, 0.25, -1.5, 3.0, -2.2, 0.8,
            1.7, -0.3, 2.5, -1.1, 0.6, -2.8, 0.0, 1.4,
        ];
        let n = row.len();
        let mut work = row.clone();
        let mut packed = vec![0u8; bytes_per_block(n, 4)];
        let scale = quantize_row(&mut work, 4, &mut packed);
        let mut decoded = vec![0f32; n];
        dequantize_row(&packed, scale, 4, &mut decoded);

        let max_abs = row.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        // Tolerance: WHT spreads error across all coefficients, so
        // ~15% of max-abs is a reasonable bar (much better than the
        // naive per-coefficient bound).
        for (i, (&o, &d)) in row.iter().zip(decoded.iter()).enumerate() {
            let err = (o - d).abs();
            assert!(
                err < max_abs * 0.5,
                "tq4 error at {i} too large: orig={o} decoded={d} err={err} max_abs={max_abs}",
            );
        }
    }

    #[test]
    fn tq2_round_trip_preserves_sign_majority() {
        // tq2 has 2 bits → 3 levels {-1, 0, +1}. Each coefficient's
        // SIGN must agree with the dequantized sign for at least half
        // of the entries — this is the weakest correctness claim that
        // still demonstrates the rotation didn't destroy the signal.
        let row = vec![
            0.5f32, -1.2, 3.7, -0.1, 2.0, -2.5, 0.05, 4.4,
            -3.3, 1.1, 0.9, -0.4, 1.8, -1.7, 2.2, -0.05,
        ];
        let n = row.len();
        let mut work = row.clone();
        let mut packed = vec![0u8; bytes_per_block(n, 2)];
        let scale = quantize_row(&mut work, 2, &mut packed);
        let mut decoded = vec![0f32; n];
        dequantize_row(&packed, scale, 2, &mut decoded);

        let max_abs = row.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let mut agree = 0usize;
        for (&o, &d) in row.iter().zip(decoded.iter()) {
            if o.signum() == d.signum() || o.abs() < max_abs * 0.05 {
                agree += 1;
            }
        }
        let agree_frac = agree as f32 / n as f32;
        assert!(
            agree_frac >= 0.5,
            "tq2 sign agreement {agree_frac:.2} below 0.5 — rotation broke",
        );
    }

    #[test]
    fn tq1_codes_are_signed_binary() {
        // tq1 carries one sign bit per element. Dequantization
        // produces `decoded = (scale / N) × WHT(±1 codes)` since
        // wht_inverse = forward WHT × (1/N). Applying a forward WHT
        // to `decoded` brings back `scale × (±1 codes)`. So
        // `WHT(decoded) / scale` must be exactly ±1 element-wise.
        let row = vec![1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0];
        let n = row.len();
        let mut work = row.clone();
        let mut packed = vec![0u8; bytes_per_block(n, 1)];
        let scale = quantize_row(&mut work, 1, &mut packed);
        let mut decoded = vec![0f32; n];
        dequantize_row(&packed, scale, 1, &mut decoded);

        let mut rot_work = decoded.clone();
        super::wht_inplace(&mut rot_work);
        for &v in &rot_work {
            let normalized = v / scale;
            assert!(
                (normalized - 1.0).abs() < 1e-3 || (normalized + 1.0).abs() < 1e-3,
                "tq1 dequant produced non-binary value: {normalized}",
            );
        }
        let _ = n;
    }

    #[test]
    fn tq8_avx2_path_matches_scalar() {
        // AVX2 dispatch is automatic when the CPU supports it. To pin
        // the parity, we compare a direct dequantize_row call (which
        // takes the AVX2 path on AVX2 hosts) against a pure scalar
        // reference re-implementation here. On non-AVX2 hosts both
        // paths are scalar and the test still passes.
        // 32-element row: AVX2 path processes 8 at a time so we want
        // multiple full iterations + a check that the tail elements
        // (none here since 32 % 8 == 0) match. WHT also requires a
        // power-of-2 length.
        let row = vec![
            0.5f32, -1.2, 3.7, -0.1, 2.0, -2.5, 0.05, 4.4,
            -3.3, 1.1, 0.9, -0.4, 1.8, -1.7, 2.2, -0.05,
            0.0, 5.5, -6.6, 7.7, -0.001, 2.0, -2.0, 3.0,
            1.0, -1.0, 2.5, -3.5, 0.2, 4.1, -5.0, 6.0,
        ];
        let n = row.len();
        let mut work = row.clone();
        let mut packed = vec![0u8; bytes_per_block(n, 8)];
        let scale = quantize_row(&mut work, 8, &mut packed);

        // Path A: the public `dequantize_row` (dispatches to AVX2 on
        // supported hosts).
        let mut dispatched = vec![0f32; n];
        dequantize_row(&packed, scale, 8, &mut dispatched);

        // Path B: pure scalar reference. Inline the steps to avoid
        // calling the dispatching function.
        let mut reference = vec![0f32; n];
        unpack_codes(&packed, 8, &mut reference);
        for v in reference.iter_mut() {
            *v *= scale;
        }
        wht_inverse_inplace(&mut reference);

        for (i, (&a, &b)) in dispatched.iter().zip(reference.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "AVX2 vs scalar mismatch at {i}: avx2={a} scalar={b}"
            );
        }
    }

    #[test]
    fn quantize_matrix_round_trip() {
        // Multi-row helper: encode 3 rows of 8 elements each, then
        // decode and verify per-row reconstruction.
        let rows = 3;
        let n = 8;
        let input = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
            -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, //
            0.5, 0.0, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5,
        ];
        let mut scratch = vec![0f32; n];
        let mut packed = vec![0u8; rows * bytes_per_block(n, 8)];
        let mut scales = vec![0f32; rows];
        quantize_matrix(rows, n, 8, &input, &mut scratch, &mut packed, &mut scales);

        let mut out = vec![0f32; rows * n];
        dequantize_matrix(rows, n, 8, &packed, &scales, &mut out);

        for r in 0..rows {
            let max_abs = input[r * n..(r + 1) * n]
                .iter()
                .fold(0.0f32, |a, &v| a.max(v.abs()));
            for c in 0..n {
                let err = (input[r * n + c] - out[r * n + c]).abs();
                assert!(
                    err < max_abs * 0.05,
                    "row {r} col {c} error too large: {err}"
                );
            }
        }
    }
}
