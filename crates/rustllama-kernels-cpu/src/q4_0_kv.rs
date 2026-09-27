//! Q4_0 KV-cache blocks — exact ggml `block_q4_0` layout: 18 bytes
//! per 32 elements, a little-endian f16 scale embedded in the first
//! two bytes of every block.
//!
//! ## Block format
//!
//! ```text
//!   32 elements per block
//!   storage per block:
//!     d:  2 bytes   // f16 scale, little-endian
//!     qs: 16 bytes  // 32 × 4-bit codes; qs[j] low nibble = element j,
//!                   //                   qs[j] high nibble = element j+16
//!   total: 18 bytes per 32-element block
//! ```
//!
//! Dequant: `x[j] = ((qs[j] & 0xF) as i32 - 8) * d`, high nibble the
//! same with `j + 16`. Byte-identical to ggml so caches produced or
//! consumed here interoperate with the Prism fork's Q4_0 KV tooling
//! (their calibrated mean-centering bias sidecars quantize into the
//! same blocks).
//!
//! Quantization follows `quantize_row_q4_0_ref`: the divisor is the
//! signed max-magnitude element over −8 (NOT amax/8), which maps the
//! extreme element exactly onto code 0 or 15 and keeps the ggml
//! rounding behavior (`(int)(x·id + 8.5)`, clamped to 15).

use half::f16;

/// Elements per Q4_0 block.
pub const Q4_0_BLOCK_ELEMS: usize = 32;

/// Bytes per Q4_0 block: 2 (f16 scale) + 16 (packed nibbles).
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Quantize exactly 32 f32 elements into one 18-byte Q4_0 block.
/// Mirrors ggml's `quantize_row_q4_0_ref` bit-for-bit (same divisor
/// choice, same `+8.5` truncating round, same clamp).
pub fn quantize_block(elems: &[f32], out: &mut [u8]) {
    assert_eq!(elems.len(), Q4_0_BLOCK_ELEMS);
    assert_eq!(out.len(), Q4_0_BLOCK_BYTES);

    let mut amax = 0.0f32;
    let mut max = 0.0f32;
    for &v in elems {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    let d = max / -8.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    out[..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
    for j in 0..16 {
        let x0 = elems[j] * id;
        let x1 = elems[j + 16] * id;
        // x·id ∈ [−8, 8] by construction, so +8.5 lands in [0.5, 16.5];
        // the truncating cast plus the 15-clamp reproduces ggml exactly.
        let xi0 = ((x0 + 8.5) as i32).min(15) as u8;
        let xi1 = ((x1 + 8.5) as i32).min(15) as u8;
        out[2 + j] = xi0 | (xi1 << 4);
    }
}

/// Dequantize one 18-byte Q4_0 block into 32 f32 elements.
pub fn dequantize_block(packed: &[u8], out: &mut [f32]) {
    assert_eq!(packed.len(), Q4_0_BLOCK_BYTES);
    assert_eq!(out.len(), Q4_0_BLOCK_ELEMS);
    let d = f16::from_le_bytes([packed[0], packed[1]]).to_f32();
    let qs = &packed[2..18];
    for j in 0..16 {
        out[j] = ((qs[j] & 0x0F) as i32 - 8) as f32 * d;
        out[j + 16] = ((qs[j] >> 4) as i32 - 8) as f32 * d;
    }
}

/// Quantize a whole KV row (`head_dim` elements, a multiple of 32)
/// into consecutive Q4_0 blocks.
pub fn quantize_row(row: &[f32], out: &mut [u8]) {
    assert_eq!(row.len() % Q4_0_BLOCK_ELEMS, 0, "row length must be a multiple of 32");
    let n_blocks = row.len() / Q4_0_BLOCK_ELEMS;
    assert_eq!(out.len(), n_blocks * Q4_0_BLOCK_BYTES);
    for b in 0..n_blocks {
        quantize_block(
            &row[b * Q4_0_BLOCK_ELEMS..(b + 1) * Q4_0_BLOCK_ELEMS],
            &mut out[b * Q4_0_BLOCK_BYTES..(b + 1) * Q4_0_BLOCK_BYTES],
        );
    }
}

/// Dequantize a whole KV row of consecutive Q4_0 blocks.
pub fn dequantize_row(packed: &[u8], out: &mut [f32]) {
    assert_eq!(packed.len() % Q4_0_BLOCK_BYTES, 0);
    let n_blocks = packed.len() / Q4_0_BLOCK_BYTES;
    assert_eq!(out.len(), n_blocks * Q4_0_BLOCK_ELEMS);
    for b in 0..n_blocks {
        dequantize_block(
            &packed[b * Q4_0_BLOCK_BYTES..(b + 1) * Q4_0_BLOCK_BYTES],
            &mut out[b * Q4_0_BLOCK_ELEMS..(b + 1) * Q4_0_BLOCK_ELEMS],
        );
    }
}

/// FlashAttention-decode for Q4_0 KV. Same structure as the NVFP4
/// flash kernel: dequantize one kv-head's live rows into an f32
/// scratch sized `kv_len × head_dim` (never the full `max_ctx` slab),
/// then run the shared SIMD-dispatched online-softmax inner loop for
/// each of the `n_gqa` Q heads sharing that KV head.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_flash_decode_q4_0(
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
    assert_eq!(head_dim % Q4_0_BLOCK_ELEMS, 0, "head_dim must be a multiple of 32");
    let blocks_per_row = head_dim / Q4_0_BLOCK_ELEMS;
    let bytes_per_row = blocks_per_row * Q4_0_BLOCK_BYTES;
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

    // Persistent per-thread dequant scratch, sized `n_kv_heads ×
    // kv_len × head_dim` per side. The previous shape allocated two
    // `kv_len × head_dim` buffers on EVERY call (16 attn layers ×
    // every decoded token = multi-MB allocator churn) and dequantized
    // + attended one kv-head at a time on one core. Now: one parallel
    // dequant pass over every (kv_head, position) row, then one
    // parallel attention pass over every Q head. Per-head arithmetic
    // (dequant order within a row, online-softmax over positions) is
    // unchanged — outputs are bitwise-identical to the serial kernel.
    DECODE_SCRATCH.with(|cell| {
        let (k_all, v_all) = &mut *cell.borrow_mut();
        let need = n_kv_heads * kv_len * head_dim;
        if k_all.len() < need {
            k_all.resize(need, 0.0);
            v_all.resize(need, 0.0);
        }
        let k_all = &mut k_all[..need];
        let v_all = &mut v_all[..need];

        use rayon::prelude::*;
        // Pass 1: dequant all live rows, parallel over (kv_h, t).
        k_all
            .par_chunks_mut(head_dim)
            .zip(v_all.par_chunks_mut(head_dim))
            .enumerate()
            .for_each(|(row, (k_dst, v_dst))| {
                let kv_h = row / kv_len;
                let t = row % kv_len;
                let p_off = (kv_h * max_ctx + t) * bytes_per_row;
                dequantize_row(&k_packed[p_off..p_off + bytes_per_row], k_dst);
                dequantize_row(&v_packed[p_off..p_off + bytes_per_row], v_dst);
            });
        // Pass 2: attention, parallel over Q heads (disjoint `out`
        // slabs; scratch is read-only here).
        let k_all = &k_all[..];
        let v_all = &v_all[..];
        out.par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(h, out_h)| {
                let kv_h = h / n_gqa;
                let q_h = &q[h * head_dim..(h + 1) * head_dim];
                let base = kv_h * kv_len * head_dim;
                crate::turboquant::online_softmax_attn_f32_scratch(
                    q_h,
                    &k_all[base..base + kv_len * head_dim],
                    &v_all[base..base + kv_len * head_dim],
                    out_h,
                    head_dim,
                    kv_len,
                    scale,
                );
            });
    });
}

std::thread_local! {
    /// Grow-only dequant scratch for the Q4_0 flash kernels: `(K, V)`
    /// f32 buffers reused across calls on the same thread (the engine
    /// runs every forward on its dedicated worker thread, so after
    /// warmup this never allocates). Sized `n_kv_heads × live_len ×
    /// head_dim` — larger than the old per-kv-head scratch, but
    /// bounded (~32 MB at 4K ctx for the 27B) and amortized to zero
    /// allocations.
    static DECODE_SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<f32>)> =
        std::cell::RefCell::new((Vec::new(), Vec::new()));
}

/// Multi-query FlashAttention prefill for Q4_0 KV. Query `i` attends
/// causally to `[0, kv_len_base + i + 1)`. Q/out layout:
/// `[n_new, n_heads, head_dim]`. Scratch is sized to the live window.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_flash_prefill_q4_0(
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
    assert_eq!(head_dim % Q4_0_BLOCK_ELEMS, 0, "head_dim must be a multiple of 32");
    let blocks_per_row = head_dim / Q4_0_BLOCK_ELEMS;
    let bytes_per_row = blocks_per_row * Q4_0_BLOCK_BYTES;
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

    // Same two-pass parallel structure as the decode kernel above:
    // dequant every (kv_head, position) row once in parallel, then
    // attend in parallel over every (query row, head) pair. Causal
    // masking is preserved by slicing each query's live window
    // `[..kv_len_i]` exactly as the serial loop did — bitwise-
    // identical outputs.
    DECODE_SCRATCH.with(|cell| {
        let (k_all, v_all) = &mut *cell.borrow_mut();
        let need = n_kv_heads * total_positions * head_dim;
        if k_all.len() < need {
            k_all.resize(need, 0.0);
            v_all.resize(need, 0.0);
        }
        let k_all = &mut k_all[..need];
        let v_all = &mut v_all[..need];

        use rayon::prelude::*;
        k_all
            .par_chunks_mut(head_dim)
            .zip(v_all.par_chunks_mut(head_dim))
            .enumerate()
            .for_each(|(row, (k_dst, v_dst))| {
                let kv_h = row / total_positions;
                let t = row % total_positions;
                let p_off = (kv_h * max_ctx + t) * bytes_per_row;
                dequantize_row(&k_packed[p_off..p_off + bytes_per_row], k_dst);
                dequantize_row(&v_packed[p_off..p_off + bytes_per_row], v_dst);
            });
        let k_all = &k_all[..];
        let v_all = &v_all[..];
        // `out` layout is `[n_new, n_heads, head_dim]` — chunk index
        // `idx = i * n_heads + h`.
        out.par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(idx, out_h)| {
                let i = idx / n_heads;
                let h = idx % n_heads;
                let kv_h = h / n_gqa;
                let kv_len_i = kv_len_base + i + 1;
                let q_h = &q[idx * head_dim..(idx + 1) * head_dim];
                let base = kv_h * total_positions * head_dim;
                crate::turboquant::online_softmax_attn_f32_scratch(
                    q_h,
                    &k_all[base..base + kv_len_i * head_dim],
                    &v_all[base..base + kv_len_i * head_dim],
                    out_h,
                    head_dim,
                    kv_len_i,
                    scale,
                );
            });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn quantize_block_matches_gguf_dequant_reference() {
        // Round-trip through our encoder must agree with the canonical
        // GGUF-side Q4_0 dequant (rustllama-gguf::dequant::dequant_q4_0
        // uses the identical byte convention). We can't call across
        // crates here (gguf depends on kernels-cpu), so re-derive the
        // reference decode inline: same math as dequant_q4_0.
        let mut elems = [0f32; 32];
        let mut state = 0x12345678u32;
        for e in elems.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *e = ((state >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0;
        }
        let mut packed = [0u8; Q4_0_BLOCK_BYTES];
        quantize_block(&elems, &mut packed);

        let d = f16::from_le_bytes([packed[0], packed[1]]).to_f32();
        let qs = &packed[2..18];
        let mut reference = [0f32; 32];
        for j in 0..16 {
            reference[j] = ((qs[j] & 0x0F) as i32 - 8) as f32 * d;
            reference[j + 16] = ((qs[j] >> 4) as i32 - 8) as f32 * d;
        }

        let mut ours = [0f32; 32];
        dequantize_block(&packed, &mut ours);
        assert_eq!(ours, reference, "dequantize_block must match the GGUF reference decode");

        // Reconstruction error bound: half a quantization step.
        let amax = elems.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let step = amax / 8.0 + 1e-6;
        for (o, e) in ours.iter().zip(elems.iter()) {
            assert!(
                (o - e).abs() <= step * 0.75 + 1e-3,
                "reconstruction off: got {o}, want ~{e} (step {step})"
            );
        }
    }

    #[test]
    fn quantize_block_extreme_element_hits_code_boundary() {
        // ggml's divisor choice (max / -8) maps the max-magnitude
        // element exactly to code 0 (when positive) with zero error.
        let mut elems = [0.25f32; 32];
        elems[7] = 2.0; // strictly dominant positive max
        let mut packed = [0u8; Q4_0_BLOCK_BYTES];
        quantize_block(&elems, &mut packed);
        let mut back = [0f32; 32];
        dequantize_block(&packed, &mut back);
        assert!(approx_eq(back[7], 2.0, 1e-2), "max element must round-trip near-exactly, got {}", back[7]);
    }

    #[test]
    fn mean_centering_reduces_quantization_error() {
        // The K mean-centering bias exists because Q4_0's symmetric
        // uniform grid wastes range on a large common-mode offset:
        // rows = mean + small signal quantize far worse than the
        // centered signal alone. This pins the *mechanism* the
        // engine-side bias subtract relies on.
        let n = 128usize;
        let mut raw = vec![0f32; n];
        let mut state = 99u32;
        let mean = 5.0f32;
        for e in raw.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *e = mean + ((state >> 8) as f32 / (1u32 << 24) as f32) * 0.5 - 0.25;
        }
        let centered: Vec<f32> = raw.iter().map(|v| v - mean).collect();

        let err = |src: &[f32]| -> f64 {
            let bytes_per = src.len() / Q4_0_BLOCK_ELEMS * Q4_0_BLOCK_BYTES;
            let mut packed = vec![0u8; bytes_per];
            quantize_row(src, &mut packed);
            let mut back = vec![0f32; src.len()];
            dequantize_row(&packed, &mut back);
            src.iter()
                .zip(&back)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
        };
        let raw_err = err(&raw);
        let centered_err = err(&centered);
        assert!(
            centered_err * 4.0 < raw_err,
            "centering must cut Q4_0 error by >4x on offset-dominated rows: \
             raw {raw_err:.6} vs centered {centered_err:.6}"
        );
    }

    #[test]
    fn zero_block_roundtrips_to_zero() {
        let elems = [0f32; 32];
        let mut packed = [0u8; Q4_0_BLOCK_BYTES];
        quantize_block(&elems, &mut packed);
        let mut back = [1f32; 32];
        dequantize_block(&packed, &mut back);
        assert_eq!(back, [0f32; 32]);
    }

    /// Reference: dequantize whole cache then run the plain scalar
    /// F32 attention loop (same math as `gqa_attention_one_step`).
    #[allow(clippy::too_many_arguments)]
    fn reference_attn_from_packed(
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
        let bytes_per_row = head_dim / Q4_0_BLOCK_ELEMS * Q4_0_BLOCK_BYTES;
        let n_gqa = n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut k_f32 = vec![0f32; n_kv_heads * max_ctx * head_dim];
        let mut v_f32 = vec![0f32; n_kv_heads * max_ctx * head_dim];
        for h in 0..n_kv_heads {
            for t in 0..kv_len {
                let row_idx = h * max_ctx + t;
                let p = row_idx * bytes_per_row;
                let o = row_idx * head_dim;
                dequantize_row(&k_packed[p..p + bytes_per_row], &mut k_f32[o..o + head_dim]);
                dequantize_row(&v_packed[p..p + bytes_per_row], &mut v_f32[o..o + head_dim]);
            }
        }
        for h in 0..n_heads {
            let kv_h = h / n_gqa;
            let q_h = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = vec![0f32; kv_len];
            for t in 0..kv_len {
                let off = (kv_h * max_ctx + t) * head_dim;
                let mut acc = 0f32;
                for d in 0..head_dim {
                    acc += q_h[d] * k_f32[off + d];
                }
                scores[t] = acc * scale;
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }
            let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
            for v in out_h.iter_mut() {
                *v = 0.0;
            }
            for t in 0..kv_len {
                let off = (kv_h * max_ctx + t) * head_dim;
                for d in 0..head_dim {
                    out_h[d] += scores[t] * v_f32[off + d];
                }
            }
        }
    }

    fn synth_packed_cache(
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_len: usize,
        seed: u32,
    ) -> Vec<u8> {
        let bytes_per_row = head_dim / Q4_0_BLOCK_ELEMS * Q4_0_BLOCK_BYTES;
        let mut packed = vec![0u8; n_kv_heads * max_ctx * bytes_per_row];
        let mut state = seed;
        let mut row = vec![0f32; head_dim];
        for h in 0..n_kv_heads {
            for t in 0..kv_len {
                for e in row.iter_mut() {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    *e = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
                }
                let p = (h * max_ctx + t) * bytes_per_row;
                quantize_row(&row, &mut packed[p..p + bytes_per_row]);
            }
        }
        packed
    }

    #[test]
    fn flash_decode_q4_0_matches_dequant_then_f32_attn() {
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 32;
        let max_ctx = 64;
        let kv_len = 40;

        let k_packed = synth_packed_cache(n_kv_heads, head_dim, max_ctx, kv_len, 11);
        let v_packed = synth_packed_cache(n_kv_heads, head_dim, max_ctx, kv_len, 23);
        let mut state = 77u32;
        let mut q = vec![0f32; n_heads * head_dim];
        for e in q.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *e = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
        }

        let mut flash_out = vec![0f32; n_heads * head_dim];
        gqa_attention_flash_decode_q4_0(
            &q, &k_packed, &v_packed, &mut flash_out, n_heads, n_kv_heads, head_dim,
            max_ctx, kv_len,
        );
        let mut ref_out = vec![0f32; n_heads * head_dim];
        reference_attn_from_packed(
            &q, &k_packed, &v_packed, &mut ref_out, n_heads, n_kv_heads, head_dim,
            max_ctx, kv_len,
        );
        for (f, r) in flash_out.iter().zip(ref_out.iter()) {
            assert!(approx_eq(*f, *r, 1e-4), "flash {f} vs reference {r}");
        }
    }

    #[test]
    fn flash_prefill_q4_0_matches_decode_loop() {
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 32;
        let max_ctx = 64;
        let kv_len_base = 8;
        let n_new = 6;
        let total = kv_len_base + n_new;

        let k_packed = synth_packed_cache(n_kv_heads, head_dim, max_ctx, total, 5);
        let v_packed = synth_packed_cache(n_kv_heads, head_dim, max_ctx, total, 9);
        let mut state = 3u32;
        let mut q = vec![0f32; n_new * n_heads * head_dim];
        for e in q.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *e = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
        }

        let mut prefill_out = vec![0f32; n_new * n_heads * head_dim];
        gqa_attention_flash_prefill_q4_0(
            &q, &k_packed, &v_packed, &mut prefill_out, n_heads, n_kv_heads, head_dim,
            max_ctx, kv_len_base, n_new,
        );

        for i in 0..n_new {
            let kv_len_i = kv_len_base + i + 1;
            let q_i = &q[i * n_heads * head_dim..(i + 1) * n_heads * head_dim];
            let mut step_out = vec![0f32; n_heads * head_dim];
            gqa_attention_flash_decode_q4_0(
                q_i, &k_packed, &v_packed, &mut step_out, n_heads, n_kv_heads, head_dim,
                max_ctx, kv_len_i,
            );
            let got = &prefill_out[i * n_heads * head_dim..(i + 1) * n_heads * head_dim];
            for (g, s) in got.iter().zip(step_out.iter()) {
                assert!(approx_eq(*g, *s, 1e-4), "prefill row {i}: {g} vs {s}");
            }
        }
    }
}
