//! OCP Microscaling (MX) KV-cache blocks + flash attention.
//!
//! MXFP4/6/8 as KV-cache quant formats: 32 elements per block sharing
//! one trailing E8M0 (power-of-two) scale byte, byte-identical to the
//! weight-side blocks in [`crate::mxfp`] and the GGUF `dequant_mxfp*`.
//! Block bytes: MXFP4 17, MXFP6 25 (24-byte LE 6-bit E3M2 bitstream),
//! MXFP8 33. `head_dim % 32 == 0`.
//!
//! Structure mirrors [`crate::q4_0_kv`] (also a 32-elem KV block): a
//! per-block `quantize_block`, a `dequantize_row`, and the two-pass
//! rayon flash decode/prefill that dequant the live KV window into an
//! f32 scratch then run the shared online-softmax inner loop. Unlike
//! Q4_0 there is NO whitening / mean-centering — MXFP's E8M0 scale is
//! embedded per block, like NVFP4.

use crate::mxfp::{
    e3m2_to_f32, e4m3_to_f32, e8m0_to_f32, E2M1_CODEBOOK, MXFP4_BLOCK_BYTES, MXFP4_BLOCK_ELEMS,
    MXFP6_BLOCK_BYTES, MXFP6_CODE_BYTES, MXFP8_BLOCK_BYTES,
};

// ---- Element quantizers (f32 -> low-precision code) ------------------
// Byte-exact inverses of the mxfp decoders; ported from the GGUF
// encoder (gguf/src/encode_mx.rs) so KV blocks round-trip through the
// same layout the GPU flash kernels decode.

const E2M1_LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const E2M1_MAX_EXP: i32 = 2;
const E3M2_MAX_EXP: i32 = 4;
const E4M3_MAX_EXP: i32 = 8;

/// E8M0 scale byte for a block whose largest magnitude is `absmax`,
/// targeting element format max-exponent `elem_max_exp`.
fn block_scale_byte(absmax: f32, elem_max_exp: i32) -> u8 {
    if absmax == 0.0 || !absmax.is_finite() {
        return 127; // 2^0
    }
    let e = absmax.log2().floor() as i32;
    (e - elem_max_exp + 127).clamp(0, 254) as u8
}

fn quant_e2m1_nibble(v: f32) -> u8 {
    let a = v.abs();
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, &lv) in E2M1_LEVELS.iter().enumerate() {
        let d = (a - lv).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    let nib = best as u8;
    if v.is_sign_negative() && nib != 0 {
        nib | 0x08
    } else {
        nib
    }
}

fn quant_e3m2_code(v: f32) -> u8 {
    let mut best = 0u8;
    let mut best_d = f32::INFINITY;
    for c in 0u8..64 {
        let d = (v - e3m2_to_f32(c)).abs();
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    best
}

fn quant_e4m3_byte(v: f32) -> u8 {
    let mut best = 0u8;
    let mut best_d = f32::INFINITY;
    for c in 0u16..256 {
        let c = c as u8;
        if c == 0x7F || c == 0xFF {
            continue;
        }
        let d = (v - e4m3_to_f32(c)).abs();
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    best
}

// ---- Per-block quantize / per-row dequant ----------------------------

/// Quantize 32 f32 KV elements into one 17-byte MXFP4 block.
pub fn quantize_block_mxfp4(elems: &[f32], out: &mut [u8]) {
    assert_eq!(elems.len(), MXFP4_BLOCK_ELEMS);
    assert_eq!(out.len(), MXFP4_BLOCK_BYTES);
    let absmax = elems.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let sb = block_scale_byte(absmax, E2M1_MAX_EXP);
    let inv = { let s = e8m0_to_f32(sb); if s != 0.0 { 1.0 / s } else { 0.0 } };
    for j in 0..16 {
        let lo = quant_e2m1_nibble(elems[j * 2] * inv);
        let hi = quant_e2m1_nibble(elems[j * 2 + 1] * inv);
        out[j] = (lo & 0x0F) | (hi << 4);
    }
    out[16] = sb;
}

/// Quantize 32 f32 KV elements into one 25-byte MXFP6 block.
pub fn quantize_block_mxfp6(elems: &[f32], out: &mut [u8]) {
    assert_eq!(elems.len(), MXFP6_BLOCK_ELEMS_LOCAL);
    assert_eq!(out.len(), MXFP6_BLOCK_BYTES);
    let absmax = elems.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let sb = block_scale_byte(absmax, E3M2_MAX_EXP);
    let inv = { let s = e8m0_to_f32(sb); if s != 0.0 { 1.0 / s } else { 0.0 } };
    for b in out[..MXFP6_CODE_BYTES].iter_mut() {
        *b = 0;
    }
    for j in 0..32usize {
        let code = quant_e3m2_code(elems[j] * inv) as u32 & 0x3F;
        let bitpos = j * 6;
        let byte_idx = bitpos / 8;
        let bit_off = bitpos % 8;
        let shifted = code << bit_off;
        out[byte_idx] |= (shifted & 0xFF) as u8;
        if byte_idx + 1 < MXFP6_CODE_BYTES {
            out[byte_idx + 1] |= ((shifted >> 8) & 0xFF) as u8;
        }
    }
    out[MXFP6_CODE_BYTES] = sb;
}

const MXFP6_BLOCK_ELEMS_LOCAL: usize = 32;
const MXFP8_BLOCK_ELEMS_LOCAL: usize = 32;

/// Quantize 32 f32 KV elements into one 33-byte MXFP8 block.
pub fn quantize_block_mxfp8(elems: &[f32], out: &mut [u8]) {
    assert_eq!(elems.len(), MXFP8_BLOCK_ELEMS_LOCAL);
    assert_eq!(out.len(), MXFP8_BLOCK_BYTES);
    let absmax = elems.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let sb = block_scale_byte(absmax, E4M3_MAX_EXP);
    let inv = { let s = e8m0_to_f32(sb); if s != 0.0 { 1.0 / s } else { 0.0 } };
    for j in 0..32 {
        out[j] = quant_e4m3_byte(elems[j] * inv);
    }
    out[32] = sb;
}

/// Dequantize one MXFP4 packed row (`head_dim` elements) into f32.
pub fn dequantize_row_mxfp4(packed: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / MXFP4_BLOCK_ELEMS;
    debug_assert_eq!(packed.len(), n_blocks * MXFP4_BLOCK_BYTES);
    for b in 0..n_blocks {
        let off = b * MXFP4_BLOCK_BYTES;
        let scale = e8m0_to_f32(packed[off + 16]);
        let dst = b * MXFP4_BLOCK_ELEMS;
        for j in 0..16 {
            let byte = packed[off + j];
            out[dst + j * 2] = E2M1_CODEBOOK[(byte & 0x0F) as usize] * scale;
            out[dst + j * 2 + 1] = E2M1_CODEBOOK[((byte >> 4) & 0x0F) as usize] * scale;
        }
    }
}

/// Dequantize one MXFP6 packed row into f32.
pub fn dequantize_row_mxfp6(packed: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / 32;
    debug_assert_eq!(packed.len(), n_blocks * MXFP6_BLOCK_BYTES);
    for b in 0..n_blocks {
        let off = b * MXFP6_BLOCK_BYTES;
        let scale = e8m0_to_f32(packed[off + MXFP6_CODE_BYTES]);
        let codes = &packed[off..off + MXFP6_CODE_BYTES];
        let dst = b * 32;
        for j in 0..32 {
            let bitpos = j * 6;
            let byte_idx = bitpos / 8;
            let bit_off = bitpos % 8;
            let lo = codes[byte_idx] as u16;
            let hi = if byte_idx + 1 < MXFP6_CODE_BYTES {
                codes[byte_idx + 1] as u16
            } else {
                0
            };
            let code = (((lo | (hi << 8)) >> bit_off) & 0x3F) as u8;
            out[dst + j] = e3m2_to_f32(code) * scale;
        }
    }
}

/// Dequantize one MXFP8 packed row into f32.
pub fn dequantize_row_mxfp8(packed: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / 32;
    debug_assert_eq!(packed.len(), n_blocks * MXFP8_BLOCK_BYTES);
    for b in 0..n_blocks {
        let off = b * MXFP8_BLOCK_BYTES;
        let scale = e8m0_to_f32(packed[off + 32]);
        let dst = b * 32;
        for j in 0..32 {
            out[dst + j] = e4m3_to_f32(packed[off + j]) * scale;
        }
    }
}

// ---- Flash attention (generic over block bytes + row dequant) --------

std::thread_local! {
    static DECODE_SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<f32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

type DeqRow = fn(&[u8], &mut [f32]);

#[allow(clippy::too_many_arguments)]
fn flash_decode_generic(
    q: &[f32],
    k_packed: &[u8],
    v_packed: &[u8],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
    bytes_per_row: usize,
    deq: DeqRow,
    window: usize,
    softcap: f32,
) {
    assert_eq!(head_dim % 32, 0, "head_dim must be a multiple of 32");
    assert_eq!(q.len(), n_heads * head_dim);
    assert_eq!(out.len(), n_heads * head_dim);
    assert_eq!(k_packed.len(), n_kv_heads * max_ctx * bytes_per_row);
    assert_eq!(v_packed.len(), n_kv_heads * max_ctx * bytes_per_row);
    assert!(kv_len <= max_ctx);
    assert_eq!(n_heads % n_kv_heads, 0, "GQA requires n_heads divisible by n_kv_heads");
    if kv_len == 0 {
        out.iter_mut().for_each(|v| *v = 0.0);
        return;
    }
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    // Decode window lower bound (0 ⇒ full attention).
    let start = if window > 0 { kv_len.saturating_sub(window) } else { 0 };
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
        k_all
            .par_chunks_mut(head_dim)
            .zip(v_all.par_chunks_mut(head_dim))
            .enumerate()
            .for_each(|(row, (k_dst, v_dst))| {
                let kv_h = row / kv_len;
                let t = row % kv_len;
                let p_off = (kv_h * max_ctx + t) * bytes_per_row;
                deq(&k_packed[p_off..p_off + bytes_per_row], k_dst);
                deq(&v_packed[p_off..p_off + bytes_per_row], v_dst);
            });
        let k_all = &k_all[..];
        let v_all = &v_all[..];
        out.par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(h, out_h)| {
                let kv_h = h / n_gqa;
                let q_h = &q[h * head_dim..(h + 1) * head_dim];
                let base = kv_h * kv_len * head_dim;
                crate::turboquant::online_softmax_attn_f32_scratch_capped_windowed(
                    q_h,
                    &k_all[base..base + kv_len * head_dim],
                    &v_all[base..base + kv_len * head_dim],
                    out_h,
                    head_dim,
                    kv_len,
                    scale,
                    start,
                    softcap,
                );
            });
    });
}

#[allow(clippy::too_many_arguments)]
fn flash_prefill_generic(
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
    bytes_per_row: usize,
    deq: DeqRow,
    window: usize,
    softcap: f32,
) {
    assert_eq!(head_dim % 32, 0, "head_dim must be a multiple of 32");
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
                deq(&k_packed[p_off..p_off + bytes_per_row], k_dst);
                deq(&v_packed[p_off..p_off + bytes_per_row], v_dst);
            });
        let k_all = &k_all[..];
        let v_all = &v_all[..];
        out.par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(idx, out_h)| {
                let i = idx / n_heads;
                let h = idx % n_heads;
                let kv_h = h / n_gqa;
                let kv_len_i = kv_len_base + i + 1;
                let start_i = if window > 0 { kv_len_i.saturating_sub(window) } else { 0 };
                let q_h = &q[idx * head_dim..(idx + 1) * head_dim];
                let base = kv_h * total_positions * head_dim;
                crate::turboquant::online_softmax_attn_f32_scratch_capped_windowed(
                    q_h,
                    &k_all[base..base + kv_len_i * head_dim],
                    &v_all[base..base + kv_len_i * head_dim],
                    out_h,
                    head_dim,
                    kv_len_i,
                    scale,
                    start_i,
                    softcap,
                );
            });
    });
}

macro_rules! mxfp_kv_flash {
    ($decode:ident, $prefill:ident, $decode_cw:ident, $prefill_cw:ident, $deq:path, $bpb:expr) => {
        /// FlashAttention decode for this MXFP KV format.
        #[allow(clippy::too_many_arguments)]
        pub fn $decode(
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
            let bytes_per_row = (head_dim / 32) * $bpb;
            flash_decode_generic(
                q, k_packed, v_packed, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                bytes_per_row, $deq, 0, 0.0,
            );
        }
        /// Sliding-window + Gemma2 soft-cap FlashAttention decode for this MXFP
        /// KV format. `window == 0` attends `[0, kv_len)`; otherwise the decode
        /// window `[kv_len-window, kv_len)`. `softcap > 0.0` applies
        /// `s = c*tanh(s/c)` after the scale and before the online-softmax
        /// max/exp (Gemma2). `window == 0 && softcap == 0.0` is BIT-IDENTICAL to
        #[doc = concat!("[`", stringify!($decode), "`].")]
        #[allow(clippy::too_many_arguments)]
        pub fn $decode_cw(
            q: &[f32],
            k_packed: &[u8],
            v_packed: &[u8],
            out: &mut [f32],
            n_heads: usize,
            n_kv_heads: usize,
            head_dim: usize,
            max_ctx: usize,
            kv_len: usize,
            window: usize,
            softcap: f32,
        ) {
            let bytes_per_row = (head_dim / 32) * $bpb;
            flash_decode_generic(
                q, k_packed, v_packed, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                bytes_per_row, $deq, window, softcap,
            );
        }
        /// FlashAttention prefill for this MXFP KV format.
        #[allow(clippy::too_many_arguments)]
        pub fn $prefill(
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
            let bytes_per_row = (head_dim / 32) * $bpb;
            flash_prefill_generic(
                q, k_packed, v_packed, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base,
                n_new, bytes_per_row, $deq, 0, 0.0,
            );
        }
        /// Sliding-window + Gemma2 soft-cap FlashAttention prefill for this MXFP
        /// KV format. Query `i` (abs pos `kv_len_base + i`) attends
        /// `[start_i, kv_len_i)` with `kv_len_i = kv_len_base + i + 1` and
        /// `start_i = kv_len_i.saturating_sub(window)` (window 0 ⇒ full causal).
        /// `window == 0 && softcap == 0.0` is BIT-IDENTICAL to
        #[doc = concat!("[`", stringify!($prefill), "`].")]
        #[allow(clippy::too_many_arguments)]
        pub fn $prefill_cw(
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
            window: usize,
            softcap: f32,
        ) {
            let bytes_per_row = (head_dim / 32) * $bpb;
            flash_prefill_generic(
                q, k_packed, v_packed, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base,
                n_new, bytes_per_row, $deq, window, softcap,
            );
        }
    };
}

mxfp_kv_flash!(
    gqa_attention_flash_decode_mxfp4,
    gqa_attention_flash_prefill_mxfp4,
    gqa_attention_flash_decode_mxfp4_capped_windowed,
    gqa_attention_flash_prefill_mxfp4_capped_windowed,
    dequantize_row_mxfp4,
    MXFP4_BLOCK_BYTES
);
mxfp_kv_flash!(
    gqa_attention_flash_decode_mxfp6,
    gqa_attention_flash_prefill_mxfp6,
    gqa_attention_flash_decode_mxfp6_capped_windowed,
    gqa_attention_flash_prefill_mxfp6_capped_windowed,
    dequantize_row_mxfp6,
    MXFP6_BLOCK_BYTES
);
mxfp_kv_flash!(
    gqa_attention_flash_decode_mxfp8,
    gqa_attention_flash_prefill_mxfp8,
    gqa_attention_flash_decode_mxfp8_capped_windowed,
    gqa_attention_flash_prefill_mxfp8_capped_windowed,
    dequantize_row_mxfp8,
    MXFP8_BLOCK_BYTES
);

#[cfg(test)]
mod tests {
    use super::*;

    // Naive f32 attention reference for one query head over kv_len positions.
    fn naive_attn(q: &[f32], k: &[f32], v: &[f32], head_dim: usize, kv_len: usize) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = vec![0f32; kv_len];
        for (t, s) in scores.iter_mut().enumerate() {
            let mut dot = 0f32;
            for d in 0..head_dim {
                dot += q[d] * k[t * head_dim + d];
            }
            *s = dot * scale;
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            denom += *s;
        }
        let mut out = vec![0f32; head_dim];
        for (t, &s) in scores.iter().enumerate() {
            let w = s / denom;
            for d in 0..head_dim {
                out[d] += w * v[t * head_dim + d];
            }
        }
        out
    }

    #[test]
    fn mxfp4_kv_roundtrip_and_flash_decode() {
        let head_dim = 64usize;
        let kv_len = 5usize;
        let max_ctx = 8usize;
        // Build f32 K/V, quantize to MXFP4 blocks, run flash vs naive-on-dequant.
        let mk = |seed: usize| -> Vec<f32> {
            (0..kv_len * head_dim)
                .map(|i| ((i * 7 + seed) % 23) as f32 * 0.03 - 0.3)
                .collect()
        };
        let kf = mk(1);
        let vf = mk(2);
        let bpr = (head_dim / 32) * MXFP4_BLOCK_BYTES;
        let mut kp = vec![0u8; max_ctx * bpr];
        let mut vp = vec![0u8; max_ctx * bpr];
        for t in 0..kv_len {
            for blk in 0..head_dim / 32 {
                quantize_block_mxfp4(
                    &kf[t * head_dim + blk * 32..t * head_dim + blk * 32 + 32],
                    &mut kp[t * bpr + blk * MXFP4_BLOCK_BYTES
                        ..t * bpr + blk * MXFP4_BLOCK_BYTES + MXFP4_BLOCK_BYTES],
                );
                quantize_block_mxfp4(
                    &vf[t * head_dim + blk * 32..t * head_dim + blk * 32 + 32],
                    &mut vp[t * bpr + blk * MXFP4_BLOCK_BYTES
                        ..t * bpr + blk * MXFP4_BLOCK_BYTES + MXFP4_BLOCK_BYTES],
                );
            }
        }
        let q: Vec<f32> = (0..head_dim).map(|i| (i as f32).sin() * 0.2).collect();
        let mut out = vec![0f32; head_dim];
        gqa_attention_flash_decode_mxfp4(&q, &kp, &vp, &mut out, 1, 1, head_dim, max_ctx, kv_len);
        // Reference: dequantize the same blocks and run naive attention.
        let mut kd = vec![0f32; kv_len * head_dim];
        let mut vd = vec![0f32; kv_len * head_dim];
        for t in 0..kv_len {
            dequantize_row_mxfp4(&kp[t * bpr..t * bpr + bpr], &mut kd[t * head_dim..(t + 1) * head_dim]);
            dequantize_row_mxfp4(&vp[t * bpr..t * bpr + bpr], &mut vd[t * head_dim..(t + 1) * head_dim]);
        }
        let want = naive_attn(&q, &kd, &vd, head_dim, kv_len);
        let err = out.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 1e-4, "mxfp4 flash decode vs naive-on-dequant err={err}");
    }

    #[test]
    fn mxfp8_kv_block_roundtrips_tightly() {
        // MXFP8 (E4M3) should reconstruct a smooth block within a few %.
        let elems: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let mut blk = vec![0u8; MXFP8_BLOCK_BYTES];
        quantize_block_mxfp8(&elems, &mut blk);
        let mut dec = vec![0f32; 32];
        dequantize_row_mxfp8(&blk, &mut dec);
        let absmax = elems.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let err = elems.iter().zip(&dec).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < absmax * 0.05, "mxfp8 kv block err={err}");
    }

    #[test]
    fn mxfp6_kv_prefill_causal_matches_naive() {
        let head_dim = 32usize;
        let n_new = 3usize;
        let max_ctx = 8usize;
        let kv_len_base = 0usize;
        let total = kv_len_base + n_new;
        let bpr = (head_dim / 32) * MXFP6_BLOCK_BYTES;
        let kvf: Vec<f32> = (0..total * head_dim).map(|i| ((i % 17) as f32) * 0.05 - 0.4).collect();
        let mut kp = vec![0u8; max_ctx * bpr];
        let mut vp = vec![0u8; max_ctx * bpr];
        for t in 0..total {
            quantize_block_mxfp6(&kvf[t * head_dim..t * head_dim + 32], &mut kp[t * bpr..t * bpr + MXFP6_BLOCK_BYTES]);
            quantize_block_mxfp6(&kvf[t * head_dim..t * head_dim + 32], &mut vp[t * bpr..t * bpr + MXFP6_BLOCK_BYTES]);
        }
        let q: Vec<f32> = (0..n_new * head_dim).map(|i| (i as f32).cos() * 0.15).collect();
        let mut out = vec![0f32; n_new * head_dim];
        gqa_attention_flash_prefill_mxfp6(&q, &kp, &vp, &mut out, 1, 1, head_dim, max_ctx, kv_len_base, n_new);
        // Reference per query i attends to [0, i+1).
        let mut kd = vec![0f32; total * head_dim];
        for t in 0..total {
            dequantize_row_mxfp6(&kp[t * bpr..t * bpr + bpr], &mut kd[t * head_dim..(t + 1) * head_dim]);
        }
        for i in 0..n_new {
            let want = naive_attn(&q[i * head_dim..(i + 1) * head_dim], &kd, &kd, head_dim, i + 1);
            let err = out[i * head_dim..(i + 1) * head_dim]
                .iter()
                .zip(&want)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(err < 1e-4, "mxfp6 prefill row {i} err={err}");
        }
    }

    // ---- Sliding-window + Gemma2 soft-cap (`_capped_windowed`) tests ----

    /// Naive windowed + soft-capped single-head attention over f32 K/V. The
    /// query attends only `[start, kv_len)`; `softcap > 0.0` caps each logit.
    fn naive_attn_cw(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        head_dim: usize,
        kv_len: usize,
        start: usize,
        softcap: f32,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = Vec::new();
        for t in start..kv_len {
            let mut dot = 0f32;
            for d in 0..head_dim {
                dot += q[d] * k[t * head_dim + d];
            }
            let mut s = dot * scale;
            if softcap > 0.0 {
                s = softcap * (s / softcap).tanh();
            }
            scores.push(s);
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            denom += *s;
        }
        let mut out = vec![0f32; head_dim];
        for (i, t) in (start..kv_len).enumerate() {
            let w = if denom > 0.0 { scores[i] / denom } else { 0.0 };
            for d in 0..head_dim {
                out[d] += w * v[t * head_dim + d];
            }
        }
        out
    }

    /// Quantize deterministic single-kv-head MXFP4 K/V for `positions` slots.
    fn build_mxfp4_cache(head_dim: usize, max_ctx: usize, positions: usize) -> (Vec<u8>, Vec<u8>, Vec<f32>, Vec<f32>) {
        let bpr = (head_dim / 32) * MXFP4_BLOCK_BYTES;
        let kf: Vec<f32> = (0..positions * head_dim).map(|i| ((i * 7) % 23) as f32 * 0.03 - 0.3).collect();
        let vf: Vec<f32> = (0..positions * head_dim).map(|i| ((i * 5) % 19) as f32 * 0.04 - 0.35).collect();
        let mut kp = vec![0u8; max_ctx * bpr];
        let mut vp = vec![0u8; max_ctx * bpr];
        for t in 0..positions {
            for blk in 0..head_dim / 32 {
                let src = t * head_dim + blk * 32;
                let dst = t * bpr + blk * MXFP4_BLOCK_BYTES;
                quantize_block_mxfp4(&kf[src..src + 32], &mut kp[dst..dst + MXFP4_BLOCK_BYTES]);
                quantize_block_mxfp4(&vf[src..src + 32], &mut vp[dst..dst + MXFP4_BLOCK_BYTES]);
            }
        }
        // Dequantize back so the reference sees the exact values the kernel reads.
        let mut kd = vec![0f32; positions * head_dim];
        let mut vd = vec![0f32; positions * head_dim];
        for t in 0..positions {
            dequantize_row_mxfp4(&kp[t * bpr..t * bpr + bpr], &mut kd[t * head_dim..(t + 1) * head_dim]);
            dequantize_row_mxfp4(&vp[t * bpr..t * bpr + bpr], &mut vd[t * head_dim..(t + 1) * head_dim]);
        }
        (kp, vp, kd, vd)
    }

    #[test]
    fn mxfp_capped_windowed_passthrough_bit_identical() {
        // `_capped_windowed(0, 0.0)` must be BIT-IDENTICAL to the base kernel for
        // decode AND prefill, across all three MXFP widths (the macro expansion).
        let head_dim = 64;
        let max_ctx = 16;
        let kv_len = 9;
        let (kp, vp, _kd, _vd) = build_mxfp4_cache(head_dim, max_ctx, kv_len);
        let q: Vec<f32> = (0..head_dim).map(|i| (i as f32).sin() * 0.2).collect();

        let mut base = vec![0f32; head_dim];
        let mut cw = vec![0f32; head_dim];
        gqa_attention_flash_decode_mxfp4(&q, &kp, &vp, &mut base, 1, 1, head_dim, max_ctx, kv_len);
        gqa_attention_flash_decode_mxfp4_capped_windowed(&q, &kp, &vp, &mut cw, 1, 1, head_dim, max_ctx, kv_len, 0, 0.0);
        assert_eq!(base, cw, "mxfp4 decode passthrough not bit-identical");

        // Prefill passthrough.
        let kv_len_base = 2;
        let n_new = 5;
        let (kp2, vp2, _, _) = build_mxfp4_cache(head_dim, max_ctx, kv_len_base + n_new);
        let qp: Vec<f32> = (0..n_new * head_dim).map(|i| (i as f32).cos() * 0.15).collect();
        let mut base_pf = vec![0f32; n_new * head_dim];
        let mut cw_pf = vec![0f32; n_new * head_dim];
        gqa_attention_flash_prefill_mxfp4(&qp, &kp2, &vp2, &mut base_pf, 1, 1, head_dim, max_ctx, kv_len_base, n_new);
        gqa_attention_flash_prefill_mxfp4_capped_windowed(&qp, &kp2, &vp2, &mut cw_pf, 1, 1, head_dim, max_ctx, kv_len_base, n_new, 0, 0.0);
        assert_eq!(base_pf, cw_pf, "mxfp4 prefill passthrough not bit-identical");

        // mxfp8 decode passthrough (second macro arm).
        let bpr8 = (head_dim / 32) * MXFP8_BLOCK_BYTES;
        let kvf: Vec<f32> = (0..kv_len * head_dim).map(|i| ((i * 3) % 29) as f32 * 0.02 - 0.3).collect();
        let mut kp8 = vec![0u8; max_ctx * bpr8];
        let mut vp8 = vec![0u8; max_ctx * bpr8];
        for t in 0..kv_len {
            for blk in 0..head_dim / 32 {
                let src = t * head_dim + blk * 32;
                let dst = t * bpr8 + blk * MXFP8_BLOCK_BYTES;
                quantize_block_mxfp8(&kvf[src..src + 32], &mut kp8[dst..dst + MXFP8_BLOCK_BYTES]);
                quantize_block_mxfp8(&kvf[src..src + 32], &mut vp8[dst..dst + MXFP8_BLOCK_BYTES]);
            }
        }
        let mut base8 = vec![0f32; head_dim];
        let mut cw8 = vec![0f32; head_dim];
        gqa_attention_flash_decode_mxfp8(&q, &kp8, &vp8, &mut base8, 1, 1, head_dim, max_ctx, kv_len);
        gqa_attention_flash_decode_mxfp8_capped_windowed(&q, &kp8, &vp8, &mut cw8, 1, 1, head_dim, max_ctx, kv_len, 0, 0.0);
        assert_eq!(base8, cw8, "mxfp8 decode passthrough not bit-identical");
    }

    #[test]
    fn mxfp4_windowed_and_softcap_match_naive() {
        let head_dim = 64;
        let max_ctx = 16;
        let kv_len = 5;
        let (kp, vp, kd, vd) = build_mxfp4_cache(head_dim, max_ctx, kv_len);
        let q: Vec<f32> = (0..head_dim).map(|i| (i as f32).sin() * 0.2).collect();

        // (b) windowed: window=2, kv_len=5 ⇒ start = 3.
        let mut got = vec![0f32; head_dim];
        gqa_attention_flash_decode_mxfp4_capped_windowed(&q, &kp, &vp, &mut got, 1, 1, head_dim, max_ctx, kv_len, 2, 0.0);
        let want = naive_attn_cw(&q, &kd, &vd, head_dim, kv_len, 3, 0.0);
        let err = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 1e-4, "mxfp4 windowed err={err}");

        // (c) soft-cap: softcap=50.0, full window (start=0).
        let mut got_sc = vec![0f32; head_dim];
        gqa_attention_flash_decode_mxfp4_capped_windowed(&q, &kp, &vp, &mut got_sc, 1, 1, head_dim, max_ctx, kv_len, 0, 50.0);
        let want_sc = naive_attn_cw(&q, &kd, &vd, head_dim, kv_len, 0, 50.0);
        let err_sc = got_sc.iter().zip(&want_sc).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err_sc < 1e-4, "mxfp4 softcap err={err_sc}");
    }
}
