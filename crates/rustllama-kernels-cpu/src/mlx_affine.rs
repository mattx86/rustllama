//! Apple **MLX affine** quantized-weight CPU reference: dequant + fused
//! quant matvec.
//!
//! MLX (`ml-explore/mlx`, `mlx-lm`) ships `mode="affine"` quantized
//! checkpoints as three sibling safetensors tensors per weight — packed
//! `bits`-bit codes (`uint32`), per-group `scales`, per-group `biases`.
//! The representation carrier is
//! [`rustllama_tensor::MlxAffineQuant`]; this module is the scalar,
//! correctness-first decode + matvec the CPU path runs. It mirrors the
//! MXFP / NVFP4 quant-matvec families in this crate (same
//! `out[m] = Σ_k W[m,k]·x[k]` shape and `_w_f32_a` naming), but scalar
//! only for now.
//!
//! # Packing — the one thing that's easy to get wrong
//!
//! MLX affine packing is a pure contiguous **little-endian bitstream**:
//! element `i` occupies bits `[i*bits, (i+1)*bits)` read LSB-first from
//! the packed bytes (the `uint32` tensor's raw little-endian bytes).
//!
//! - Power-of-two `bits` (2/4/8): `32/bits` elements per uint32 word,
//!   low element in the low bits — MLX CPU backend:
//!   `wi = *w++; for p: out = wi & mask; wi >>= bits;`.
//! - `bits` ∈ {3,5,6}: an element straddles byte **and** uint32
//!   boundaries. MLX `extract_bits<T,3>` (in `mlx/backend/cpu/quantized.cpp`)
//!   makes the layout explicit — element 2 is
//!   `(w[0] & 0xc0) >> 6 | (w[1] & 0x1) << 2`: 2 bits from byte 0 and the
//!   low bit of byte 1. That is exactly "bit offset `2*3 = 6`, read 3
//!   bits LSB-first across the byte boundary", which is what
//!   [`read_bits_le`] does for **every** width — so one unpacker is
//!   byte-exact with MLX for 2/3/4/5/6/8 bits.
//!
//! Because `group_size` is a multiple of 32 and 32 elements occupy
//! exactly `bits` uint32 words, there is no padding within a group, a
//! row, or between rows: the whole weight tensor is one bitstream, and
//! element `(row, col)` of an `[out_features, in_features]` weight sits
//! at bit `(row * in_features + col) * bits`.
//!
//! # Dequant formula
//!
//! Plain affine (asymmetric): `w[i] = scales[g] * q[i] + biases[g]`,
//! `g = i / group_size`, `q[i]` the unpacked code. The bias (the group
//! `min`) is **added**. MLX inner loop: `result += xi * (scale*wl + bias)`.
//!
//! # TODOs (later, build-env-gated phases)
//!
//! - **SIMD.** A SIMD fast path (mirroring `mxfp` / `nvfp4`) — the 3/5/6
//!   bit unpack is serial like MXFP6, so those widths would decode into a
//!   scratch then vectorize the FMA; 2/4/8 can nibble/byte-gather.
//! - **Apple Metal fast path.** On Apple Silicon this decode is a no-op:
//!   route to `mlx-c`'s `mlx_quantized_matmul` (native Metal
//!   `qmv`/`qmm`) instead of dequantizing on the CPU. This scalar
//!   reference stays as the cross-backend parity oracle (as the MXFP/CUDA
//!   parity harness uses the scalar kernels today).

/// Read `bits` bits starting at absolute bit offset `bit_pos` from a
/// little-endian bit buffer, LSB-first. `bits` must be ≤ 32.
///
/// This is the single unpacker for all MLX affine widths (2/3/4/5/6/8):
/// it walks byte by byte, taking the low `take` bits available in the
/// current byte and shifting them into place, so an element that
/// straddles byte/word boundaries (3/5/6-bit) reconstructs exactly as
/// MLX's `extract_bits` does. See the module docs for the bit-layout
/// derivation + source citation.
#[inline]
pub fn read_bits_le(packed: &[u8], bit_pos: usize, bits: u32) -> u32 {
    debug_assert!(bits <= 32);
    let mut val: u32 = 0;
    let mut got: u32 = 0;
    while got < bits {
        let abs = bit_pos + got as usize;
        let byte_idx = abs / 8;
        let bit_in_byte = (abs % 8) as u32;
        let avail = 8 - bit_in_byte; // bits left in this byte
        let take = avail.min(bits - got); // how many we consume here
        // Low `take` bits of the byte, starting at `bit_in_byte`.
        let mask = if take == 32 { u32::MAX } else { (1u32 << take) - 1 };
        let chunk = ((packed[byte_idx] as u32) >> bit_in_byte) & mask;
        val |= chunk << got;
        got += take;
    }
    val
}

/// Write `bits` bits for element `index` into `out`, LSB-first — the
/// exact inverse of [`read_bits_le`] (which reads element `i` from the
/// absolute bit offset `i*bits`). Here `index` is the *element* index,
/// so the bits land at `[index*bits, (index+1)*bits)`.
///
/// `value` must already be masked to `bits` bits; the target byte span
/// of `out` must be pre-zeroed, because we OR the code in (never clear)
/// — mirroring how MLX's packer accumulates successive elements into a
/// freshly-zeroed uint32 word. The straddle handling (3/5/6-bit codes
/// that cross a byte/word boundary) is the write-side twin of the read
/// loop: take the low `take` bits available in the current byte, OR
/// them in at `bit_in_byte`, advance, repeat.
#[inline]
pub fn write_bits_le(out: &mut [u8], index: usize, bits: u32, value: u32) {
    debug_assert!(bits <= 32);
    debug_assert!(bits == 32 || value < (1u32 << bits), "value {value} exceeds {bits} bits");
    let bit_pos = index * bits as usize;
    let mut got: u32 = 0;
    while got < bits {
        let abs = bit_pos + got as usize;
        let byte_idx = abs / 8;
        let bit_in_byte = (abs % 8) as u32;
        let avail = 8 - bit_in_byte; // bits free in this byte
        let take = avail.min(bits - got); // how many we deposit here
        let mask = if take == 32 { u32::MAX } else { (1u32 << take) - 1 };
        // Low `take` bits of the remaining value, shifted to sit at
        // `bit_in_byte`. `take <= 8` always (a byte holds <= 8 free
        // bits), so the cast to u8 never truncates real data.
        let chunk = ((value >> got) & mask) as u8;
        out[byte_idx] |= chunk << bit_in_byte;
        got += take;
    }
}

/// MLX group-affine (min/max) **encode**: quantize `weights` to the MLX
/// affine triple — packed `bits`-bit codes + per-group `scales` +
/// per-group `biases` — the exact inverse of [`dequantize_mlx_affine`].
///
/// This mirrors mlx-lm's `mx.quantize(..., mode="affine")`: process
/// `group_size` contiguous elements (along the flattened input dim) at
/// a time; per group take `lo = min`, `hi = max`, set
/// `scale = (hi - lo) / ((1<<bits) - 1)` and `bias = lo`, then
/// `q[i] = round((w[i] - lo) / scale)` clamped to `[0, (1<<bits)-1]`.
/// Dequant reconstructs `scale*q + bias`, so per-element error is
/// bounded by `scale/2` (plus f32 rounding).
///
/// Returns `(packed, scales, biases)` where `packed` is the contiguous
/// LSB-first bitstream **padded up to a whole number of uint32 words**
/// (multiple of 4 bytes) so that reinterpreting it as a `uint32-LE`
/// array yields exactly MLX's packed `weight` tensor. For a real
/// linear (`in_features` a multiple of `group_size`, itself a multiple
/// of 32) the natural byte length is already a multiple of 4, so the
/// padding is a no-op; it only ever matters for odd synthetic shapes.
///
/// # Zero-range groups
///
/// When a group is constant (`hi == lo`, or non-finite) `scale` would
/// be 0 and `round((w-lo)/scale)` is undefined. We fall back to
/// `scale = 1.0`, which forces every code to `round(0) = 0` and makes
/// dequant reproduce `bias = lo` exactly — the MLX behavior for a flat
/// group.
pub fn quantize_mlx_affine(
    weights: &[f32],
    group_size: usize,
    bits: u32,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    assert!(
        matches!(bits, 2 | 3 | 4 | 5 | 6 | 8),
        "unsupported bits {bits} (want one of 2,3,4,5,6,8)"
    );
    let n = weights.len();
    assert!(
        group_size > 0 && n % group_size == 0,
        "n {n} not a multiple of group_size {group_size}"
    );
    let n_groups = n / group_size;
    let qmax = ((1u32 << bits) - 1) as f32; // top code = levels - 1

    // Packed byte length, padded to a multiple of 4 so the byte view
    // maps to a whole-word uint32-LE array (see the doc note above).
    let packed_len = (n * bits as usize).div_ceil(8).next_multiple_of(4);
    let mut packed = vec![0u8; packed_len];
    let mut scales = Vec::with_capacity(n_groups);
    let mut biases = Vec::with_capacity(n_groups);

    for g in 0..n_groups {
        let base = g * group_size;
        let grp = &weights[base..base + group_size];
        let mut lo = grp[0];
        let mut hi = grp[0];
        for &w in &grp[1..] {
            if w < lo {
                lo = w;
            }
            if w > hi {
                hi = w;
            }
        }
        // `scale > 0` catches hi==lo (zero range) AND a NaN range, both
        // of which fall back to the flat-group encoding (all codes 0).
        let mut scale = (hi - lo) / qmax;
        if !(scale > 0.0) {
            scale = 1.0;
        }
        let inv = 1.0 / scale;
        for (j, &w) in grp.iter().enumerate() {
            // round-half-to-even via f32::round (half away from zero);
            // the offset (w-lo) is >= 0 so direction bias is moot.
            let q = (((w - lo) * inv).round()).clamp(0.0, qmax) as u32;
            write_bits_le(&mut packed, base + j, bits, q);
        }
        scales.push(scale);
        biases.push(lo);
    }

    (packed, scales, biases)
}

/// Decode `out.len()` MLX affine-quantized elements to f32.
///
/// `packed` is the contiguous little-endian bitstream (`out.len()*bits/8`
/// bytes); `scales`/`biases` each hold `out.len()/group_size` per-group
/// values in the same row-major order as the elements. Reference decode:
/// `out[i] = scales[i/group_size] * q[i] + biases[i/group_size]`.
pub fn dequantize_mlx_affine(
    packed: &[u8],
    scales: &[f32],
    biases: &[f32],
    group_size: usize,
    bits: u32,
    out: &mut [f32],
) {
    let n = out.len();
    assert!(group_size > 0 && n % group_size == 0, "n {n} not a multiple of group_size {group_size}");
    let n_groups = n / group_size;
    assert_eq!(scales.len(), n_groups, "scales len");
    assert_eq!(biases.len(), n_groups, "biases len");
    assert_eq!(packed.len(), n * bits as usize / 8, "packed len");

    // Walk groups so scale/bias are hoisted out of the inner loop; the
    // bit cursor advances `bits` per element (contiguous stream).
    let mut bit_pos = 0usize;
    for g in 0..n_groups {
        let s = scales[g];
        let b = biases[g];
        let base = g * group_size;
        for j in 0..group_size {
            let q = read_bits_le(packed, bit_pos, bits) as f32;
            out[base + j] = s * q + b;
            bit_pos += bits as usize;
        }
    }
}

/// Fused MLX affine quant matvec: `out[i] = Σ_{c<k} dequant(W[i,c]) · x[c]`.
///
/// `W` is `[m, k]` (`m` = out_features, `k` = in_features) stored as the
/// MLX triple: `packed` is the whole-tensor bitstream (`m*k*bits/8`
/// bytes), `scales`/`biases` are `m*(k/group_size)` per-group values in
/// row-major order. Scalar reference (correctness-first); see the module
/// TODOs for the SIMD + Apple-Metal fast paths.
#[allow(clippy::too_many_arguments)]
pub fn matvec_mlx_affine_w_f32_a(
    packed: &[u8],
    scales: &[f32],
    biases: &[f32],
    group_size: usize,
    bits: u32,
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    assert!(group_size > 0 && k % group_size == 0, "k {k} not a multiple of group_size {group_size}");
    let groups_per_row = k / group_size;
    assert_eq!(x.len(), k, "x len");
    assert_eq!(out.len(), m, "out len");
    assert_eq!(scales.len(), m * groups_per_row, "scales len");
    assert_eq!(biases.len(), m * groups_per_row, "biases len");
    assert_eq!(packed.len(), m * k * bits as usize / 8, "packed len");

    // TODO(mlx): Apple-Metal fast path. On Apple Silicon, skip this
    // scalar decode entirely and route to `mlx-c`'s
    // `mlx_quantized_matmul` (native Metal `qmv`/`qmm`) — this reference
    // then serves only as the cross-backend parity oracle.
    // TODO(mlx): SIMD (AVX2/AVX-512) decode+FMA, mirroring `mxfp`/`nvfp4`.
    let row_bits = k * bits as usize; // each row is word-aligned (see docs)
    for i in 0..m {
        let mut acc = 0.0f32;
        let mut bit_pos = i * row_bits;
        let sb_row = i * groups_per_row;
        for g in 0..groups_per_row {
            let s = scales[sb_row + g];
            let b = biases[sb_row + g];
            let xc = &x[g * group_size..(g + 1) * group_size];
            for &xj in xc {
                let q = read_bits_le(packed, bit_pos, bits) as f32;
                acc += (s * q + b) * xj;
                bit_pos += bits as usize;
            }
        }
        out[i] = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack `qs` (each already masked to `bits` bits) into a contiguous
    /// little-endian bitstream — the inverse of [`read_bits_le`], and the
    /// same layout MLX's `mx.quantize` emits. Test-only encoder.
    fn pack_bits_le(qs: &[u32], bits: u32) -> Vec<u8> {
        let total_bits = qs.len() * bits as usize;
        assert_eq!(total_bits % 8, 0, "test packs must be byte-aligned");
        let mut out = vec![0u8; total_bits / 8];
        let mut bit_pos = 0usize;
        for &q in qs {
            debug_assert!(bits == 32 || q < (1u32 << bits), "q {q} exceeds {bits} bits");
            let mut got = 0u32;
            while got < bits {
                let abs = bit_pos + got as usize;
                let byte_idx = abs / 8;
                let bit_in_byte = (abs % 8) as u32;
                let avail = 8 - bit_in_byte;
                let take = avail.min(bits - got);
                let mask = (1u32 << take) - 1;
                let chunk = ((q >> got) & mask) as u8;
                out[byte_idx] |= chunk << bit_in_byte;
                got += take;
            }
            bit_pos += bits as usize;
        }
        out
    }

    #[test]
    fn read_bits_matches_mlx_3bit_example() {
        // MLX `extract_bits<T,3>`: element 2 = (w0 & 0xc0)>>6 | (w1 & 0x1)<<2.
        // Build two bytes, confirm elements 0,1,2 decode as MLX derives them.
        let w = [0b1011_0101u8, 0b0000_0001u8];
        // element 0 = bits 0..3 = 0b101 = 5
        assert_eq!(read_bits_le(&w, 0, 3), 0b101);
        // element 1 = bits 3..6 = 0b110 = 6
        assert_eq!(read_bits_le(&w, 3, 3), 0b110);
        // element 2 = bits 6..9 = (w0>>6 = 0b10) | (w1&1)<<2 = 0b1_10 = 6
        let mlx = (((w[0] as u32) & 0xc0) >> 6) | (((w[1] as u32) & 0x1) << 2);
        assert_eq!(read_bits_le(&w, 6, 3), mlx);
        assert_eq!(read_bits_le(&w, 6, 3), 0b110);
    }

    #[test]
    fn pack_unpack_roundtrip_all_widths() {
        for &bits in &[2u32, 3, 4, 5, 6, 8] {
            // 64 codes → byte-aligned for every width (64*bits % 8 == 0).
            let maxv = 1u32 << bits;
            let qs: Vec<u32> = (0..64).map(|i| (i as u32 * 7 + 1) % maxv).collect();
            let packed = pack_bits_le(&qs, bits);
            for (i, &q) in qs.iter().enumerate() {
                let got = read_bits_le(&packed, i * bits as usize, bits);
                assert_eq!(got, q, "bits={bits} i={i}");
            }
        }
    }

    #[test]
    fn dequant_matches_affine_formula() {
        // group_size=32, bits=4: two groups (n=64). Known q + per-group s,b.
        let bits = 4u32;
        let group_size = 32usize;
        let n = 64usize;
        let qs: Vec<u32> = (0..n).map(|i| (i as u32) % 16).collect();
        let packed = pack_bits_le(&qs, bits);
        let scales = vec![0.5f32, 2.0f32];
        let biases = vec![-1.0f32, 3.0f32];
        let mut out = vec![0f32; n];
        dequantize_mlx_affine(&packed, &scales, &biases, group_size, bits, &mut out);
        for i in 0..n {
            let g = i / group_size;
            let want = scales[g] * qs[i] as f32 + biases[g];
            assert!((out[i] - want).abs() < 1e-6, "i={i} got {} want {want}", out[i]);
        }
    }

    /// Deterministic pseudo-random f32 in `[-2, 2)` — a fixed spread so
    /// the encode tests are reproducible without an `rand` dep.
    fn pseudo_f32(i: usize) -> f32 {
        let h = (i as u32).wrapping_mul(2654435761) ^ 0x9E37_79B9;
        (h % 10_007) as f32 / 10_007.0 * 4.0 - 2.0
    }

    #[test]
    fn write_then_read_bits_roundtrips_all_widths() {
        // write_bits_le ∘ read_bits_le must be the identity for every
        // supported width, including the straddling 3/5/6-bit codes.
        for &bits in &[2u32, 3, 4, 5, 6, 8] {
            let maxv = 1u32 << bits;
            // 64 elements → byte-aligned for every width (64*bits % 8 == 0).
            let vals: Vec<u32> = (0..64).map(|i| (i as u32).wrapping_mul(2246822519) % maxv).collect();
            let mut packed = vec![0u8; 64 * bits as usize / 8];
            for (i, &v) in vals.iter().enumerate() {
                write_bits_le(&mut packed, i, bits, v);
            }
            for (i, &v) in vals.iter().enumerate() {
                let got = read_bits_le(&packed, i * bits as usize, bits);
                assert_eq!(got, v, "bits={bits} i={i}");
            }
        }
    }

    #[test]
    fn quantize_then_dequantize_within_scale_half() {
        // Encode → decode must reconstruct each element within scale/2
        // (+ f32 rounding eps), across every group_size × bits combo.
        for &group_size in &[32usize, 64, 128] {
            for &bits in &[2u32, 3, 4, 5, 6, 8] {
                let n = group_size * 5; // 5 groups
                let weights: Vec<f32> = (0..n).map(pseudo_f32).collect();
                let (packed, scales, biases) =
                    quantize_mlx_affine(&weights, group_size, bits);

                // Packed length is padded to a multiple of 4 (u32 words).
                assert_eq!(packed.len() % 4, 0, "packed not word-padded");
                // For these shapes the natural length is already a
                // multiple of 4, so padding adds nothing.
                assert_eq!(packed.len(), n * bits as usize / 8);
                assert_eq!(scales.len(), n / group_size);
                assert_eq!(biases.len(), n / group_size);

                let mut recon = vec![0f32; n];
                dequantize_mlx_affine(
                    &packed, &scales, &biases, group_size, bits, &mut recon,
                );
                for i in 0..n {
                    let g = i / group_size;
                    let tol = scales[g] * 0.5 + 1e-4;
                    assert!(
                        (recon[i] - weights[i]).abs() <= tol,
                        "gs={group_size} bits={bits} i={i}: recon {} vs {} (tol {tol})",
                        recon[i],
                        weights[i],
                    );
                }
            }
        }
    }

    #[test]
    fn quantize_constant_group_is_exact() {
        // A flat group (hi==lo) must hit the scale=1 fallback and
        // reproduce the constant exactly (all codes 0, bias=value).
        let weights = vec![0.375f32; 64];
        let (packed, scales, biases) = quantize_mlx_affine(&weights, 32, 4);
        assert_eq!(scales, vec![1.0, 1.0]);
        assert_eq!(biases, vec![0.375, 0.375]);
        let mut recon = vec![0f32; 64];
        dequantize_mlx_affine(&packed, &scales, &biases, 32, 4, &mut recon);
        assert!(recon.iter().all(|&r| (r - 0.375).abs() < 1e-6));
    }

    #[test]
    fn quantized_matvec_approximates_naive_f32_matvec() {
        // Encode a full [m,k] weight, then confirm the fused quant
        // matvec over the encoded weights tracks a naive f32 matvec on
        // the ORIGINAL weights within a quantization-error bound.
        for &(group_size, bits) in &[(32usize, 8u32), (64, 4), (128, 6), (32, 3)] {
            let m = 4usize;
            let k = group_size * 3; // 3 groups per row
            let n = m * k;
            let weights: Vec<f32> = (0..n).map(pseudo_f32).collect();
            let x: Vec<f32> = (0..k).map(|c| (c as f32) * 0.003 - 0.4).collect();

            let (packed, scales, biases) =
                quantize_mlx_affine(&weights, group_size, bits);

            // Naive f32 reference on the ORIGINAL weights.
            let mut want = vec![0f32; m];
            for i in 0..m {
                let mut acc = 0.0f32;
                for c in 0..k {
                    acc += weights[i * k + c] * x[c];
                }
                want[i] = acc;
            }

            let mut out = vec![0f32; m];
            matvec_mlx_affine_w_f32_a(
                &packed, &scales, &biases, group_size, bits, &x, &mut out, m, k,
            );

            // Per-output error <= Σ_c (scale_c/2)*|x_c|. Bound it loosely
            // with the max group scale and the L1 norm of x.
            let max_scale = scales.iter().cloned().fold(0.0f32, f32::max);
            let l1_x: f32 = x.iter().map(|v| v.abs()).sum();
            let tol = max_scale * 0.5 * l1_x + 1e-3;
            for i in 0..m {
                assert!(
                    (out[i] - want[i]).abs() <= tol,
                    "gs={group_size} bits={bits} row {i}: {} vs {} (tol {tol})",
                    out[i],
                    want[i],
                );
            }
        }
    }

    #[test]
    fn matvec_matches_dequant_then_dot() {
        // Exercise several (group_size, bits) shapes, a couple rows each.
        for &(group_size, bits) in &[(32usize, 4u32), (64, 3), (32, 8), (32, 6), (128, 5), (32, 2)] {
            let m = 3usize;
            let k = group_size * 2; // 2 groups per row
            let groups_per_row = k / group_size;
            let n = m * k;
            let maxv = 1u32 << bits;
            // Deterministic pseudo-random-ish codes.
            let qs: Vec<u32> = (0..n).map(|i| (i as u32).wrapping_mul(2654435761) % maxv).collect();
            let packed = pack_bits_le(&qs, bits);
            let scales: Vec<f32> = (0..m * groups_per_row)
                .map(|g| 0.1 + (g as f32) * 0.05)
                .collect();
            let biases: Vec<f32> = (0..m * groups_per_row)
                .map(|g| -0.3 + (g as f32) * 0.02)
                .collect();
            let x: Vec<f32> = (0..k).map(|c| (c as f32) * 0.01 - 0.5).collect();

            // Reference: dequantize the full [m,k] then do a plain dot per row.
            let mut w = vec![0f32; n];
            dequantize_mlx_affine(&packed, &scales, &biases, group_size, bits, &mut w);
            let mut want = vec![0f32; m];
            for i in 0..m {
                let mut acc = 0.0f32;
                for c in 0..k {
                    acc += w[i * k + c] * x[c];
                }
                want[i] = acc;
            }

            let mut out = vec![0f32; m];
            matvec_mlx_affine_w_f32_a(
                &packed, &scales, &biases, group_size, bits, &x, &mut out, m, k,
            );
            for i in 0..m {
                assert!(
                    (out[i] - want[i]).abs() < 1e-4 * want[i].abs().max(1.0),
                    "group_size={group_size} bits={bits} row {i}: got {} want {}",
                    out[i],
                    want[i],
                );
            }
        }
    }
}
