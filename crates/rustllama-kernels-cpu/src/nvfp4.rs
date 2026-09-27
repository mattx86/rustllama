//! NVFP4: 4-bit floating-point quantization (E2M1) with shared FP8
//! (E4M3) scale per 16-element block.
//!
//! Originally from NVIDIA's Blackwell tensor-core ISA; the format
//! itself is hardware-neutral. On Intel/AMD GPUs (or any GPU without
//! dedicated FP4 tensor cores) the dequant runs in software at
//! FP16-GEMM throughput — you get FP4's *accuracy* (the non-uniform
//! spacing makes 4 bits go further than INT4 for activation-like
//! distributions) without the *throughput* bump Blackwell hardware
//! gives. Memory savings (8× over F32, 4× over F16) are real on
//! every backend.
//!
//! ## Block format
//!
//! ```text
//!   16 elements per block
//!   storage per block:
//!     packed: 8 bytes   // 16 × 4 bits, little-endian (lo nibble = element 0)
//!     scale:  1 byte    // FP8 E4M3 encoding of the per-block scale
//!   total:    9 bytes per 16-element block
//! ```
//!
//! ## Codebook (E2M1, signed)
//!
//! ```text
//!   sign | exp | mant | value
//!   ──────────────────────────
//!     0  | 00  |  0   |  0.0
//!     0  | 00  |  1   |  0.5
//!     0  | 01  |  0   |  1.0
//!     0  | 01  |  1   |  1.5
//!     0  | 10  |  0   |  2.0
//!     0  | 10  |  1   |  3.0
//!     0  | 11  |  0   |  4.0
//!     0  | 11  |  1   |  6.0
//!     1  | 00  |  0   | -0.0  (treated as +0 by the codebook)
//!     1  | 00  |  1   | -0.5
//!     1  | 01  |  0   | -1.0
//!     ...                ...
//!     1  | 11  |  1   | -6.0
//! ```
//!
//! The codebook is small and fixed, so we expose it as a `static`
//! array indexed by the 4-bit code. Encoding picks the closest entry
//! (Euclidean distance in the rotated frame is overkill — for 4
//! bits, linear search across 16 entries is fast enough).
//!
//! ## FP8 E4M3 scale
//!
//! The block's per-block scale is stored as one FP8 byte in the E4M3
//! encoding (1 sign + 4 exponent + 3 mantissa, exp bias 7, NaN at
//! 0x7F / 0xFF). This gives ~7 bits of mantissa precision over a
//! dynamic range from ~2⁻⁹ to ~448 — more than enough for the
//! per-block max-abs scales that arise in weight / KV quantization.
//!
//! ## Out of scope
//!
//! - GEMM kernels (this module ships encode/decode primitives only).
//!   The matvec kernel for weight quantization is the next step.
//! - KV-cache integration. Like TurboQuant, the storage variant +
//!   attention path land in a follow-up round.

/// E2M1 codebook: 16 signed values, indexed by the 4-bit code.
/// Hardware-equivalent to NVIDIA's NVFP4 lookup table.
pub const NVFP4_CODEBOOK: [f32; 16] = [
    // Positive half (sign bit clear).
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    // Negative half (sign bit set). The "negative zero" slot still
    // decodes to 0.0; treating it as a distinct code-point only
    // matters for the encoder's preferred output.
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Largest absolute value the codebook can represent. Used by the
/// per-block scale picker to set the dynamic range.
pub const NVFP4_MAX_ABS: f32 = 6.0;

/// Elements per NVFP4 block. Matches the standard 16 used by
/// Blackwell's tensor cores — keeps memory layouts directly
/// transferable.
pub const NVFP4_BLOCK_ELEMS: usize = 16;

/// Bytes per block: 16 × 4 bits / 8 = 8 bytes for codes + 1 byte
/// for the FP8 scale = 9.
pub const NVFP4_BLOCK_BYTES: usize = NVFP4_BLOCK_ELEMS / 2 + 1;

/// Encode an f32 to FP8 E4M3. Saturates on overflow (returns the
/// max-magnitude code with matching sign). Underflow and denormals
/// flush to ±0. NaN encodes as the canonical NaN code (`0x7F` for
/// positive, `0xFF` for negative — though there's only one NaN slot
/// in E4M3 traditionally; we pick `0x7F`).
pub fn f32_to_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign_bit: u8 = if x.is_sign_negative() { 0x80 } else { 0x00 };
    let mag = x.abs();
    // E4M3 max representable magnitude with bias-7 exponent + 3-bit
    // mantissa: (1 + 7/8) × 2^(15 - 7) = 1.875 × 256 = 480. The
    // formal spec caps at 448 (= 1.75 × 256) to leave a NaN slot;
    // we match that.
    const E4M3_MAX: f32 = 448.0;
    if mag >= E4M3_MAX {
        // Saturate to the largest finite code, 0x7E (for positive)
        // / 0xFE (for negative). 0x7F / 0xFF are NaN.
        return sign_bit | 0x7E;
    }
    if mag == 0.0 {
        return sign_bit; // ±0
    }
    // Decompose to (exp, mantissa). Pull the f32 bits then shift.
    let bits = mag.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let f32_mant = bits & 0x7F_FFFF;
    // E4M3 bias is 7; exponent field is 4 bits → values 0..15. The
    // "normal" range is exponents 1..15 (subnormal at 0). Round
    // ties-to-nearest-even on the mantissa.
    let e4m3_exp = f32_exp + 7;
    if e4m3_exp <= 0 {
        // Subnormal or zero. The E4M3 subnormal range stretches the
        // mantissa downward; for simplicity we flush-to-zero here
        // since the per-block scale rarely lands in this region for
        // typical weight / KV distributions.
        return sign_bit;
    }
    if e4m3_exp > 15 {
        // Overflow → saturate.
        return sign_bit | 0x7E;
    }
    // Round the 23-bit f32 mantissa down to a 3-bit E4M3 mantissa.
    // Bit 22 of the f32 mantissa is the MSB; we keep bits 22..=20
    // and round-half-to-even based on the dropped tail.
    let mantissa = (f32_mant >> 20) as u8 & 0x07;
    let dropped = f32_mant & ((1 << 20) - 1);
    let half_bit = 1u32 << 19;
    let rounded_mant = if dropped > half_bit
        || (dropped == half_bit && (mantissa & 1) == 1)
    {
        mantissa.wrapping_add(1)
    } else {
        mantissa
    };
    let (final_exp, final_mant) = if rounded_mant > 0x07 {
        // Mantissa carry → bump the exponent. If that overflows the
        // 4-bit exponent field, saturate.
        if e4m3_exp + 1 > 15 {
            return sign_bit | 0x7E;
        }
        ((e4m3_exp + 1) as u8, 0u8)
    } else {
        (e4m3_exp as u8, rounded_mant)
    };
    // E4M3 reserves exp=15, mantissa=7 (0x7F / 0xFF) as NaN. If we
    // landed on that slot via legitimate rounding (e.g., a value just
    // below 448 round-half-evens up), step down one mantissa code so
    // the bit pattern stays a finite number.
    let (final_exp, final_mant) = if final_exp == 15 && final_mant == 0x07 {
        (15, 0x06)
    } else {
        (final_exp, final_mant)
    };
    sign_bit | ((final_exp & 0x0F) << 3) | (final_mant & 0x07)
}

/// Decode an FP8 E4M3 byte to f32. `0x7F` / `0xFF` decode to f32 NaN.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = (b & 0x80) != 0;
    let exp = (b >> 3) & 0x0F;
    let mant = b & 0x07;
    // E4M3 NaN: the IEEE-ish spec defines NaN as exp=15 && mant=7
    // (both signs). Other (exp=15, mant) pairs are valid finite
    // numbers in the high range 256..448.
    if exp == 0x0F && mant == 0x07 {
        return f32::NAN;
    }
    let val = if exp == 0 {
        // Subnormal: value = mantissa / 8 × 2^(1 - 7) = mantissa × 2^-9.
        (mant as f32) * (1.0 / 512.0)
    } else {
        // Normal: value = (1 + mantissa/8) × 2^(exp - 7).
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

/// Find the codebook index whose decoded value is closest to `target`.
/// Linear search over 16 entries — branchless SIMD vectorization
/// lands when the kernel matvec is fused.
fn encode_e2m1(target: f32) -> u8 {
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &v) in NVFP4_CODEBOOK.iter().enumerate() {
        let err = (v - target).abs();
        if err < best_err {
            best_err = err;
            best = i as u8;
        }
    }
    best
}

/// Quantize one NVFP4 block (16 elements) into `out` (9 bytes).
/// Returns nothing — the per-block scale is encoded directly into
/// the last byte of `out`.
pub fn quantize_block(elems: &[f32], out: &mut [u8]) {
    assert_eq!(elems.len(), NVFP4_BLOCK_ELEMS);
    assert_eq!(out.len(), NVFP4_BLOCK_BYTES);

    // Pick a per-block scale so the largest-magnitude element lands
    // near the top of the codebook. `scale = max_abs / 6.0` puts the
    // peak at code ±6.0 (the top of the E2M1 range).
    let max_abs = elems.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let scale = if max_abs > 0.0 {
        max_abs / NVFP4_MAX_ABS
    } else {
        1.0
    };
    let inv_scale = 1.0 / scale;

    // Pack two 4-bit codes per output byte.
    for (i, byte) in out[..8].iter_mut().enumerate() {
        let lo = encode_e2m1(elems[i * 2] * inv_scale);
        let hi = encode_e2m1(elems[i * 2 + 1] * inv_scale);
        *byte = (lo & 0x0F) | ((hi & 0x0F) << 4);
    }
    out[8] = f32_to_e4m3(scale);
}

/// Dequantize one NVFP4 block to `out` (16 f32 elements).
pub fn dequantize_block(packed: &[u8], out: &mut [f32]) {
    assert_eq!(packed.len(), NVFP4_BLOCK_BYTES);
    assert_eq!(out.len(), NVFP4_BLOCK_ELEMS);
    let scale = e4m3_to_f32(packed[8]);
    for (i, byte) in packed[..8].iter().enumerate() {
        let lo = (*byte & 0x0F) as usize;
        let hi = ((*byte >> 4) & 0x0F) as usize;
        out[i * 2] = NVFP4_CODEBOOK[lo] * scale;
        out[i * 2 + 1] = NVFP4_CODEBOOK[hi] * scale;
    }
}

/// Convenience: encode an entire `[rows × n]` matrix. `n` must be a
/// multiple of [`NVFP4_BLOCK_ELEMS`]. Output `packed_out` is sized
/// `rows * (n / 16) * 9` bytes.
pub fn quantize_matrix(rows: usize, n: usize, input: &[f32], packed_out: &mut [u8]) {
    assert_eq!(n % NVFP4_BLOCK_ELEMS, 0, "n must be a multiple of 16");
    let blocks_per_row = n / NVFP4_BLOCK_ELEMS;
    assert_eq!(input.len(), rows * n);
    assert_eq!(packed_out.len(), rows * blocks_per_row * NVFP4_BLOCK_BYTES);
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let in_off = r * n + b * NVFP4_BLOCK_ELEMS;
            let out_off = (r * blocks_per_row + b) * NVFP4_BLOCK_BYTES;
            quantize_block(
                &input[in_off..in_off + NVFP4_BLOCK_ELEMS],
                &mut packed_out[out_off..out_off + NVFP4_BLOCK_BYTES],
            );
        }
    }
}

/// Convenience: dequantize the corresponding `[rows × n]` matrix.
pub fn dequantize_matrix(rows: usize, n: usize, packed: &[u8], out: &mut [f32]) {
    assert_eq!(n % NVFP4_BLOCK_ELEMS, 0);
    let blocks_per_row = n / NVFP4_BLOCK_ELEMS;
    assert_eq!(packed.len(), rows * blocks_per_row * NVFP4_BLOCK_BYTES);
    assert_eq!(out.len(), rows * n);
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let in_off = (r * blocks_per_row + b) * NVFP4_BLOCK_BYTES;
            let out_off = r * n + b * NVFP4_BLOCK_ELEMS;
            dequantize_block(
                &packed[in_off..in_off + NVFP4_BLOCK_BYTES],
                &mut out[out_off..out_off + NVFP4_BLOCK_ELEMS],
            );
        }
    }
}

/// `C = W · X` where:
///   - `W: [M, K]` is NVFP4-quantized, stored row-major as
///     `M × (K / 16) × 9` bytes (block format from [`quantize_matrix`]).
///   - `X: [K]` is F32 (single vector — matvec, not GEMM).
///   - `out: [M]` is F32.
///
/// Used by the engine's per-layer projections (q/k/v projections,
/// FFN gate/up/down) when weights ship as NVFP4 in the GGUF. Scalar
/// reference path — AVX2 / AVX-512 fast paths slot in alongside the
/// existing `matvec_iq4_nl_w_f32_a` family in a follow-up.
///
/// `k` must be a multiple of 16 (the NVFP4 block size). Asserts
/// catch misuse during prefill where the prompt's `d_model` lines
/// up with the weight matrix's K dimension.
pub fn matvec_nvfp4_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    assert_eq!(k % NVFP4_BLOCK_ELEMS, 0, "nvfp4 matvec requires k % 16 == 0");
    let blocks_per_row = k / NVFP4_BLOCK_ELEMS;
    assert_eq!(w_bytes.len(), m * blocks_per_row * NVFP4_BLOCK_BYTES);
    assert_eq!(x.len(), k);
    assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection. The kernel reads
            // 9-byte blocks for `m × blocks_per_row` blocks and writes
            // exactly `m` f32 outputs.
            unsafe { matvec_nvfp4_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    matvec_nvfp4_w_f32_a_scalar(w_bytes, x, out, m, k);
}

fn matvec_nvfp4_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    let blocks_per_row = k / NVFP4_BLOCK_ELEMS;
    for i in 0..m {
        let row_start = i * blocks_per_row * NVFP4_BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * NVFP4_BLOCK_BYTES;
            let scale = e4m3_to_f32(w_bytes[off + 8]);
            let x_chunk = &x[b * NVFP4_BLOCK_ELEMS..(b + 1) * NVFP4_BLOCK_ELEMS];
            for j in 0..8 {
                let byte = w_bytes[off + j];
                let lo = (byte & 0x0F) as usize;
                let hi = ((byte >> 4) & 0x0F) as usize;
                acc += scale * NVFP4_CODEBOOK[lo] * x_chunk[j * 2];
                acc += scale * NVFP4_CODEBOOK[hi] * x_chunk[j * 2 + 1];
            }
        }
        out[i] = acc;
    }
}

/// AVX2 fast path for the NVFP4 matvec. Per block (16 elements):
///   - Load 8 packed bytes; split into low + high nibbles → 16
///     codebook indices.
///   - Gather 16 codebook f32 values via two AVX2 gathers.
///   - Decode the per-block FP8 scale, broadcast it.
///   - Two FMAs against `x` accumulate into a pair of YMM lanes.
///
/// The horizontal sum at row end uses the standard hadd reduction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_nvfp4_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    let blocks_per_row = k / NVFP4_BLOCK_ELEMS;
    let codebook_base = NVFP4_CODEBOOK.as_ptr();
    let nibble_mask = _mm256_set1_epi32(0x0F);

    for i in 0..m {
        let row_start = i * blocks_per_row * NVFP4_BLOCK_BYTES;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * NVFP4_BLOCK_BYTES;
            // Load the 8 packed code bytes into a 64-bit XMM lane,
            // then widen each byte to a 32-bit lane in a YMM register.
            // After this `i32_ymm` holds 8 packed-byte values, each
            // containing two 4-bit codes.
            let bytes_xmm = _mm_loadl_epi64(w_bytes.as_ptr().add(off) as *const __m128i);
            let i32_ymm = _mm256_cvtepu8_epi32(bytes_xmm);
            // lo_idx[j] = byte_j & 0x0F (the first of two codes per byte).
            // hi_idx[j] = (byte_j >> 4) & 0x0F (the second).
            let lo_idx = _mm256_and_si256(i32_ymm, nibble_mask);
            let hi_idx = _mm256_and_si256(_mm256_srli_epi32(i32_ymm, 4), nibble_mask);
            // Gather 8 f32 codebook values for each nibble lane. The
            // codebook is 16 entries × 4 bytes = 64 bytes, well within
            // gather's reach.
            let lo_codes = _mm256_i32gather_ps::<4>(codebook_base, lo_idx);
            let hi_codes = _mm256_i32gather_ps::<4>(codebook_base, hi_idx);
            // Byte ordering: byte_j contains element 2j (lo) and
            // element 2j+1 (hi). We need to interleave the two YMM
            // streams into the contiguous element order
            // [0,1,2,3,…,15].
            //
            // AVX2 `unpacklo` / `unpackhi` interleave WITHIN each
            // 128-bit lane:
            //   unpacklo → [0,1,2,3, 8,9,10,11]   (lanes 0..4)
            //   unpackhi → [4,5,6,7, 12,13,14,15] (lanes 0..4)
            //
            // To recover contiguous [0..8] and [8..16] we then have
            // to permute across the 128-bit lane boundary. The
            // standard recipe: `_mm256_permute2f128_ps(unpacklo,
            // unpackhi, imm)` with imm=0x20 picks the low halves of
            // each, and imm=0x31 picks the high halves.
            let unpacked_lo = _mm256_unpacklo_ps(lo_codes, hi_codes);
            let unpacked_hi = _mm256_unpackhi_ps(lo_codes, hi_codes);
            let codes_low8 = _mm256_permute2f128_ps::<0x20>(unpacked_lo, unpacked_hi);
            let codes_high8 = _mm256_permute2f128_ps::<0x31>(unpacked_lo, unpacked_hi);
            // Apply per-block scale.
            let scale = _mm256_set1_ps(e4m3_to_f32(w_bytes[off + 8]));
            let scaled_low = _mm256_mul_ps(codes_low8, scale);
            let scaled_high = _mm256_mul_ps(codes_high8, scale);
            // Load 16 activations and FMA.
            let x_ptr = x.as_ptr().add(b * NVFP4_BLOCK_ELEMS);
            let x_low = _mm256_loadu_ps(x_ptr);
            let x_high = _mm256_loadu_ps(x_ptr.add(8));
            acc0 = _mm256_fmadd_ps(scaled_low, x_low, acc0);
            acc1 = _mm256_fmadd_ps(scaled_high, x_high, acc1);
        }

        // Horizontal sum: acc0 + acc1 → 8 lanes → scalar.
        let acc = _mm256_add_ps(acc0, acc1);
        let mut sum128 = _mm_add_ps(
            _mm256_castps256_ps128(acc),
            _mm256_extractf128_ps(acc, 1),
        );
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        out[i] = _mm_cvtss_f32(sum128);
    }
}

/// FlashAttention-decode for NVFP4 KV.
///
/// The current non-flash NVFP4 attention path (in
/// `rustllama-models::llama_arch`) dequantizes the full
/// `[n_kv_heads, max_ctx, head_dim]` slab to f32 before dispatching
/// to the standard F32 attention kernel — ~32 MB scratch at 7B
/// config. This variant fixes both: dequantizes one kv_h's slab at
/// a time (matching the TQ kernel's pattern) AND uses online
/// softmax over `t` per Q head (no `[kv_len]` scores scratch).
///
/// Peak scratch drops from `2 × n_kv_heads × max_ctx × head_dim ×
/// 4 bytes` to `2 × kv_len × head_dim × 4 bytes` per call. At 8K
/// context, 4 KV heads, head_dim 128 that's ~32 MB → ~8 MB; at
/// 32K context it's ~128 MB → ~32 MB. The bigger the context, the
/// bigger the win — exactly the regime flash is built for.
///
/// Greedy parity vs the slab-then-F32-attn path is asserted by the
/// `flash_decode_nvfp4_matches_dequant_then_f32_attn` test below.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_flash_decode_nvfp4(
    q: &[f32],
    k_packed: &[u8],
    v_packed: &[u8],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    assert_eq!(head_dim % NVFP4_BLOCK_ELEMS, 0, "head_dim must be a multiple of 16");
    let blocks_per_row = head_dim / NVFP4_BLOCK_ELEMS;
    let bytes_per_row = blocks_per_row * NVFP4_BLOCK_BYTES;
    assert_eq!(q.len(), n_heads * head_dim);
    assert_eq!(out.len(), n_heads * head_dim);
    assert_eq!(k_packed.len(), n_kv_heads * max_ctx * bytes_per_row);
    assert_eq!(v_packed.len(), n_kv_heads * max_ctx * bytes_per_row);
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

    // Per-kv-head dequant scratch — sized `kv_len × head_dim`
    // instead of the full `max_ctx × head_dim`. Re-used across the
    // `n_gqa` Q heads sharing this KV head.
    let mut k_h_scratch = vec![0.0f32; kv_len * head_dim];
    let mut v_h_scratch = vec![0.0f32; kv_len * head_dim];

    for kv_h in 0..n_kv_heads {
        // Phase 1: dequantize this kv head's K and V rows.
        for t in 0..kv_len {
            let row_idx = kv_h * max_ctx + t;
            let p_off = row_idx * bytes_per_row;
            let dst_off = t * head_dim;
            for b in 0..blocks_per_row {
                let blk_src = p_off + b * NVFP4_BLOCK_BYTES;
                let blk_dst_k = dst_off + b * NVFP4_BLOCK_ELEMS;
                let blk_dst_v = dst_off + b * NVFP4_BLOCK_ELEMS;
                dequantize_block(
                    &k_packed[blk_src..blk_src + NVFP4_BLOCK_BYTES],
                    &mut k_h_scratch[blk_dst_k..blk_dst_k + NVFP4_BLOCK_ELEMS],
                );
                dequantize_block(
                    &v_packed[blk_src..blk_src + NVFP4_BLOCK_BYTES],
                    &mut v_h_scratch[blk_dst_v..blk_dst_v + NVFP4_BLOCK_ELEMS],
                );
            }
        }
        // Phase 2: online softmax per Q head sharing this KV head.
        // Reuse the shared SIMD-dispatching helper from `turboquant`
        // — same inner-loop shape (Q·K dot over f32 scratch, online
        // softmax recurrence, V accumulation with rescale).
        for qh_off in 0..n_gqa {
            let h = kv_h * n_gqa + qh_off;
            let q_h = &q[h * head_dim..(h + 1) * head_dim];
            let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
            crate::turboquant::online_softmax_attn_f32_scratch(
                q_h, &k_h_scratch, &v_h_scratch, out_h, head_dim, kv_len, scale,
            );
        }
    }
}

/// Multi-query FlashAttention prefill for NVFP4 KV. Same outer
/// structure as [`gqa_attention_flash_decode_nvfp4`] (dequantize
/// each kv_h's K/V slab once into compact f32 scratch, reuse
/// across `n_gqa` Q heads) but processes `n_new` queries in one
/// call with causal masking — query `i` attends to
/// `[0, kv_len_base + i + 1)`.
///
/// Q layout: `[n_new, n_heads, head_dim]`. Out matches. Dequant
/// scratch is sized to the live `kv_len_base + n_new` window —
/// the full `max_ctx` slab is never materialized, so memory
/// scales with conversation length not the ceiling.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_flash_prefill_nvfp4(
    q: &[f32],
    k_packed: &[u8],
    v_packed: &[u8],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) {
    assert_eq!(head_dim % NVFP4_BLOCK_ELEMS, 0, "head_dim must be a multiple of 16");
    let blocks_per_row = head_dim / NVFP4_BLOCK_ELEMS;
    let bytes_per_row = blocks_per_row * NVFP4_BLOCK_BYTES;
    assert_eq!(q.len(), n_new * n_heads * head_dim);
    assert_eq!(out.len(), n_new * n_heads * head_dim);
    assert_eq!(k_packed.len(), n_kv_heads * max_ctx * bytes_per_row);
    assert_eq!(v_packed.len(), n_kv_heads * max_ctx * bytes_per_row);
    assert!(kv_len_base + n_new <= max_ctx);
    assert_eq!(n_heads % n_kv_heads, 0, "GQA requires n_heads divisible by n_kv_heads");
    if n_new == 0 {
        return;
    }

    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let total_positions = kv_len_base + n_new;

    // Per-kv-head dequant scratch: only the live window, not the
    // full max_ctx slab.
    let mut k_h_scratch = vec![0.0f32; total_positions * head_dim];
    let mut v_h_scratch = vec![0.0f32; total_positions * head_dim];

    for kv_h in 0..n_kv_heads {
        // Phase 1: dequantize this kv head's K and V rows.
        for t in 0..total_positions {
            let row_idx = kv_h * max_ctx + t;
            let p_off = row_idx * bytes_per_row;
            let dst_off = t * head_dim;
            for b in 0..blocks_per_row {
                let blk_src = p_off + b * NVFP4_BLOCK_BYTES;
                let blk_dst = dst_off + b * NVFP4_BLOCK_ELEMS;
                dequantize_block(
                    &k_packed[blk_src..blk_src + NVFP4_BLOCK_BYTES],
                    &mut k_h_scratch[blk_dst..blk_dst + NVFP4_BLOCK_ELEMS],
                );
                dequantize_block(
                    &v_packed[blk_src..blk_src + NVFP4_BLOCK_BYTES],
                    &mut v_h_scratch[blk_dst..blk_dst + NVFP4_BLOCK_ELEMS],
                );
            }
        }
        // Phase 2: per Q head in this group, per new query, online
        // softmax over [0, kv_len_base + i + 1). Reuse the shared
        // SIMD-dispatching helper from `turboquant`.
        for qh_off in 0..n_gqa {
            let h = kv_h * n_gqa + qh_off;
            for i in 0..n_new {
                let kv_len_i = kv_len_base + i + 1;
                let q_off = (i * n_heads + h) * head_dim;
                let q_h = &q[q_off..q_off + head_dim];
                let out_off = (i * n_heads + h) * head_dim;
                let out_h = &mut out[out_off..out_off + head_dim];
                crate::turboquant::online_softmax_attn_f32_scratch(
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

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn flash_prefill_nvfp4_matches_decode_loop() {
        // Multi-query NVFP4 prefill kernel must match calling the
        // NVFP4 flash-decode kernel `n_new` times with the right
        // per-query `kv_len`.
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 16;
        let max_ctx = 64;
        let kv_len_base = 8;
        let n_new = 6;
        let blocks_per_row = head_dim / NVFP4_BLOCK_ELEMS;
        let bytes_per_row = blocks_per_row * NVFP4_BLOCK_BYTES;
        let mut k_packed = vec![0u8; n_kv_heads * max_ctx * bytes_per_row];
        let mut v_packed = vec![0u8; n_kv_heads * max_ctx * bytes_per_row];
        // Quantize for every position the kernel will touch.
        for kv_h in 0..n_kv_heads {
            for t in 0..(kv_len_base + n_new) {
                let row_idx = kv_h * max_ctx + t;
                let p_off = row_idx * bytes_per_row;
                let k_row: Vec<f32> = (0..head_dim)
                    .map(|d| ((kv_h * 7 + t * 3 + d) % 13) as f32 * 0.1 - 0.6)
                    .collect();
                let v_row: Vec<f32> = (0..head_dim)
                    .map(|d| ((kv_h * 5 + t * 2 + d) % 11) as f32 * 0.15 - 0.7)
                    .collect();
                for b in 0..blocks_per_row {
                    let elem_off = b * NVFP4_BLOCK_ELEMS;
                    let blk_dst = p_off + b * NVFP4_BLOCK_BYTES;
                    quantize_block(
                        &k_row[elem_off..elem_off + NVFP4_BLOCK_ELEMS],
                        &mut k_packed[blk_dst..blk_dst + NVFP4_BLOCK_BYTES],
                    );
                    quantize_block(
                        &v_row[elem_off..elem_off + NVFP4_BLOCK_ELEMS],
                        &mut v_packed[blk_dst..blk_dst + NVFP4_BLOCK_BYTES],
                    );
                }
            }
        }
        let q: Vec<f32> = (0..n_new * n_heads * head_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.12)
            .collect();
        let mut out_prefill = vec![0f32; n_new * n_heads * head_dim];
        gqa_attention_flash_prefill_nvfp4(
            &q, &k_packed, &v_packed, &mut out_prefill,
            n_heads, n_kv_heads, head_dim, max_ctx,
            kv_len_base, n_new,
        );
        let mut out_reference = vec![0f32; n_new * n_heads * head_dim];
        for i in 0..n_new {
            let kv_len_i = kv_len_base + i + 1;
            let q_slice = &q[i * n_heads * head_dim..(i + 1) * n_heads * head_dim];
            let out_slice =
                &mut out_reference[i * n_heads * head_dim..(i + 1) * n_heads * head_dim];
            gqa_attention_flash_decode_nvfp4(
                q_slice, &k_packed, &v_packed, out_slice,
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
            "NVFP4 prefill vs decode-loop max abs error {max_err} > 1e-5",
        );
        let _ = approx_eq(0.0, 0.0, 0.0); // silence unused-fn warning
    }

    #[test]
    fn flash_decode_nvfp4_matches_dequant_then_f32_attn() {
        // The non-flash NVFP4 attention path dequantizes the whole
        // KV slab and dispatches to F32 attention. Flash-decode
        // should produce the same output (up to FP reduction order)
        // while only materializing one kv_h's slab at a time.
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 16;
        let max_ctx = 64;
        let kv_len = 32;
        let blocks_per_row = head_dim / NVFP4_BLOCK_ELEMS;
        let bytes_per_row = blocks_per_row * NVFP4_BLOCK_BYTES;
        // Quantize deterministic K/V rows.
        let mut k_packed = vec![0u8; n_kv_heads * max_ctx * bytes_per_row];
        let mut v_packed = vec![0u8; n_kv_heads * max_ctx * bytes_per_row];
        for kv_h in 0..n_kv_heads {
            for t in 0..kv_len {
                let row_idx = kv_h * max_ctx + t;
                let p_off = row_idx * bytes_per_row;
                let k_row: Vec<f32> = (0..head_dim)
                    .map(|d| ((kv_h * 7 + t * 3 + d) % 13) as f32 * 0.1 - 0.6)
                    .collect();
                let v_row: Vec<f32> = (0..head_dim)
                    .map(|d| ((kv_h * 5 + t * 2 + d) % 11) as f32 * 0.15 - 0.7)
                    .collect();
                for b in 0..blocks_per_row {
                    let elem_off = b * NVFP4_BLOCK_ELEMS;
                    let blk_dst = p_off + b * NVFP4_BLOCK_BYTES;
                    quantize_block(
                        &k_row[elem_off..elem_off + NVFP4_BLOCK_ELEMS],
                        &mut k_packed[blk_dst..blk_dst + NVFP4_BLOCK_BYTES],
                    );
                    quantize_block(
                        &v_row[elem_off..elem_off + NVFP4_BLOCK_ELEMS],
                        &mut v_packed[blk_dst..blk_dst + NVFP4_BLOCK_BYTES],
                    );
                }
            }
        }
        // Reference path: dequant full slab + standard F32 attention.
        let mut k_f32 = vec![0f32; n_kv_heads * max_ctx * head_dim];
        let mut v_f32 = vec![0f32; n_kv_heads * max_ctx * head_dim];
        for kv_h in 0..n_kv_heads {
            for t in 0..kv_len {
                let row_idx = kv_h * max_ctx + t;
                let p_off = row_idx * bytes_per_row;
                let out_off = row_idx * head_dim;
                for b in 0..blocks_per_row {
                    let blk_src = p_off + b * NVFP4_BLOCK_BYTES;
                    let blk_dst = out_off + b * NVFP4_BLOCK_ELEMS;
                    dequantize_block(
                        &k_packed[blk_src..blk_src + NVFP4_BLOCK_BYTES],
                        &mut k_f32[blk_dst..blk_dst + NVFP4_BLOCK_ELEMS],
                    );
                    dequantize_block(
                        &v_packed[blk_src..blk_src + NVFP4_BLOCK_BYTES],
                        &mut v_f32[blk_dst..blk_dst + NVFP4_BLOCK_ELEMS],
                    );
                }
            }
        }
        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.12)
            .collect();
        let mut out_standard = vec![0f32; n_heads * head_dim];
        let mut out_flash = vec![0f32; n_heads * head_dim];
        crate::gqa_attention_one_step(
            &q, &k_f32, &v_f32, &mut out_standard,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
        );
        gqa_attention_flash_decode_nvfp4(
            &q, &k_packed, &v_packed, &mut out_flash,
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
            "NVFP4 flash vs dequant+F32 attn max abs error {max_err} > 1e-5",
        );
    }

    #[test]
    fn codebook_has_16_entries() {
        // Pins the format constant — Blackwell tensor cores assume
        // exactly 16 codes; bumping this breaks every kernel.
        assert_eq!(NVFP4_CODEBOOK.len(), 16);
        assert_eq!(NVFP4_BLOCK_BYTES, 9);
    }

    #[test]
    fn e4m3_round_trips_common_values() {
        // Spot-check round-trip for values that should land exactly
        // on representable E4M3 codes.
        for &expected in &[
            1.0, 2.0, 4.0, 0.5, 0.25, 128.0, 256.0, -1.0, -2.0, -0.5,
        ] {
            let byte = f32_to_e4m3(expected);
            let decoded = e4m3_to_f32(byte);
            assert!(
                approx_eq(decoded, expected, expected.abs() * 0.06),
                "e4m3 round-trip {expected} → 0x{byte:02x} → {decoded}",
            );
        }
    }

    #[test]
    fn e4m3_zero_round_trips() {
        // +0 and -0 must encode to bytes whose decode is exactly 0.0
        // (no NaN, no Infinity).
        assert_eq!(e4m3_to_f32(f32_to_e4m3(0.0)), 0.0);
        // The negative-zero encoding may emit a 0x80 byte; decoding
        // should return -0 which is == 0 in numeric comparison.
        assert_eq!(e4m3_to_f32(f32_to_e4m3(-0.0)), -0.0);
    }

    #[test]
    fn e4m3_saturates_on_overflow() {
        // Values above E4M3's max (~448) must saturate to the
        // largest finite code (NOT NaN). A weight scale that
        // overflows should still produce usable output.
        let big = 10_000.0f32;
        let byte = f32_to_e4m3(big);
        let decoded = e4m3_to_f32(byte);
        assert!(decoded.is_finite(), "saturated decode must be finite");
        assert!(decoded > 100.0, "saturated decode should be near max: {decoded}");
        // Same for the negative side.
        let byte_neg = f32_to_e4m3(-big);
        let decoded_neg = e4m3_to_f32(byte_neg);
        assert!(decoded_neg < -100.0);
    }

    #[test]
    fn block_round_trip_within_4bit_resolution() {
        // 16 elements, peak ≈ 6 × scale. Quantization error should be
        // bounded by the gap between adjacent codebook entries × scale.
        // For E2M1 the worst gap is between 4.0 and 6.0 (= 2.0), so
        // worst-case error per element is scale × 1.0 (half the gap).
        let elems: [f32; 16] = [
            0.1, -0.7, 1.2, -1.5, 2.3, -2.7, 3.4, -3.9,
            4.2, -4.5, 5.0, -5.4, 5.9, 0.0, -0.05, 6.0,
        ];
        let mut packed = [0u8; NVFP4_BLOCK_BYTES];
        quantize_block(&elems, &mut packed);
        let mut decoded = [0f32; NVFP4_BLOCK_ELEMS];
        dequantize_block(&packed, &mut decoded);

        // The codebook's worst inter-code gap (scaled) is 1.0 — for a
        // peak-6 row, scale ≈ 1.0 so the bound is ~1.0. We use a
        // looser 1.5 to absorb the per-block scale's own E4M3
        // rounding loss.
        for (i, (&o, &d)) in elems.iter().zip(decoded.iter()).enumerate() {
            let err = (o - d).abs();
            assert!(
                err < 1.5,
                "nvfp4 block error at {i}: orig={o} decoded={d} err={err}",
            );
        }
    }

    #[test]
    fn block_preserves_zero_exactly() {
        // The 0.0 codebook slot is exact. A row of all zeros must
        // dequantize back to exact zeros — no rounding to a tiny
        // residual from the FP8 scale.
        let zeros = [0.0f32; NVFP4_BLOCK_ELEMS];
        let mut packed = [0u8; NVFP4_BLOCK_BYTES];
        quantize_block(&zeros, &mut packed);
        let mut decoded = [0f32; NVFP4_BLOCK_ELEMS];
        dequantize_block(&packed, &mut decoded);
        for (i, &d) in decoded.iter().enumerate() {
            assert_eq!(d, 0.0, "zero element {i} dequantized to {d}");
        }
    }

    #[test]
    fn matvec_matches_dequant_then_f32_matmul() {
        // Reference path: dequant the weight matrix to F32, then do
        // a plain F32 matvec. The fused kernel must match within the
        // codebook + FP8-scale quantization error budget — same
        // bound as the block round-trip test.
        let m = 4;
        let k = 32; // 2 blocks per row
        let blocks_per_row = k / NVFP4_BLOCK_ELEMS;
        // Construct a deterministic weight matrix with mixed signs +
        // magnitudes so each block exercises a different scale.
        let w_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32) * 0.13 - 5.0).sin() * (i as f32 / 7.0).cos())
            .collect();
        let mut w_packed = vec![0u8; m * blocks_per_row * NVFP4_BLOCK_BYTES];
        quantize_matrix(m, k, &w_f32, &mut w_packed);

        // Round-trip the weights so we have a "dequantized reference"
        // that's the exact view the kernel sees internally.
        let mut w_deq = vec![0f32; m * k];
        dequantize_matrix(m, k, &w_packed, &mut w_deq);

        let x: Vec<f32> = (0..k).map(|i| (i as f32 / 11.0).cos()).collect();

        // Reference: F32 matvec on the dequantized weights.
        let mut ref_out = vec![0f32; m];
        for i in 0..m {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += w_deq[i * k + j] * x[j];
            }
            ref_out[i] = acc;
        }

        // Fused kernel: NVFP4-packed weights × F32 activations.
        let mut kernel_out = vec![0f32; m];
        matvec_nvfp4_w_f32_a(&w_packed, &x, &mut kernel_out, m, k);

        for i in 0..m {
            assert!(
                (ref_out[i] - kernel_out[i]).abs() < 1e-4,
                "row {i}: kernel {} vs reference {}",
                kernel_out[i],
                ref_out[i],
            );
        }
    }

    #[test]
    fn matrix_round_trip_multiple_blocks() {
        // 2 rows × 32 columns = 4 blocks total. Pins that the per-row
        // stride lines up so we don't accidentally mix scales across
        // rows.
        let rows = 2;
        let n = 32;
        let input: Vec<f32> = (0..rows * n).map(|i| (i as f32 - 30.0) * 0.1).collect();
        let block = NVFP4_BLOCK_BYTES;
        let mut packed = vec![0u8; rows * (n / NVFP4_BLOCK_ELEMS) * block];
        quantize_matrix(rows, n, &input, &mut packed);
        let mut decoded = vec![0f32; rows * n];
        dequantize_matrix(rows, n, &packed, &mut decoded);
        // Per-row max-abs sets each row's scale independently; pin
        // average row error against that scale.
        for r in 0..rows {
            let row_in = &input[r * n..(r + 1) * n];
            let row_out = &decoded[r * n..(r + 1) * n];
            let max_abs = row_in.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-6);
            let avg_err: f32 = row_in
                .iter()
                .zip(row_out.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / n as f32;
            assert!(
                avg_err < max_abs * 0.25,
                "row {r} avg error {avg_err} too large vs max_abs {max_abs}",
            );
        }
    }
}
