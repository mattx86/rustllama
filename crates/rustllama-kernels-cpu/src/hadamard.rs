//! Blockwise normalized Sylvester Walsh-Hadamard rotation — the
//! activation transform PrismML Bonsai models require (GGUF metadata
//! `prism.hadamard.*`, transform `"normalized-sylvester-walsh-hadamard"`).
//!
//! Ternary Bonsai weights are stored in a rotated basis:
//! `W' = W · Rᵀ` with `R = (1/√n) · H_n · S`, where `H_n` is the
//! Sylvester Hadamard matrix at the metadata block size (1024 for
//! Bonsai 2) and `S = diag(signs)` is a fixed ±1 vector shipped in
//! the GGUF per full input width. At runtime the matching transform
//! must hit the activations or the model produces garbage:
//!
//!   forward (before a matmul against a folded weight):
//!       x' = R x   =  (1/√n) · WHT( s ⊙ x )        per 1024-block
//!   inverse (after the `token_embd` row lookup — the table stores
//!   rotated rows):
//!       h  = Rᵀ z  =  s ⊙ ( (1/√n) · WHT(z) )      per 1024-block
//!
//! (`R` is orthonormal, `H` is symmetric, so `R⁻¹ = Rᵀ`.)
//!
//! The reference implementation applies `R` as a dense `[n, n]`
//! matmul (GPU-friendly); we use the O(n·log n) butterfly from
//! [`crate::turboquant::wht_inplace`] — mathematically identical,
//! ~50× less work at n = 1024. Sign vectors span the FULL activation
//! width (widths are multiples of the block size); each block uses
//! its own slice of the signs, so callers pass the full-width slice
//! and this module walks it blockwise.

use crate::turboquant::wht_inplace;

/// Forward rotation `out = R x` applied per `block`-sized chunk:
/// sign-flip first, then unnormalized WHT, then `1/√block`.
///
/// - `x.len() == signs.len() == out.len()` and all are a multiple of
///   `block`; `block` must be a power of two (the butterfly asserts).
/// - `signs` entries are ±1.0 (validated at metadata load).
pub fn hadamard_forward(x: &[f32], signs: &[f32], block: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    debug_assert_eq!(x.len(), signs.len());
    debug_assert_eq!(x.len() % block, 0);
    let inv_sqrt = 1.0f32 / (block as f32).sqrt();
    for b in 0..x.len() / block {
        let o = b * block;
        let xb = &x[o..o + block];
        let sb = &signs[o..o + block];
        let ob = &mut out[o..o + block];
        for i in 0..block {
            ob[i] = xb[i] * sb[i];
        }
        wht_inplace(ob);
        for v in ob.iter_mut() {
            *v *= inv_sqrt;
        }
    }
}

/// Inverse rotation `y ← Rᵀ y` in place, per `block`-sized chunk:
/// unnormalized WHT, `1/√block`, then sign-flip. Used on the result
/// of a `token_embd` lookup when the embedding table is registered
/// in `prism.hadamard.inverse_weight_names`.
pub fn hadamard_inverse_inplace(y: &mut [f32], signs: &[f32], block: usize) {
    debug_assert_eq!(y.len(), signs.len());
    debug_assert_eq!(y.len() % block, 0);
    let inv_sqrt = 1.0f32 / (block as f32).sqrt();
    for b in 0..y.len() / block {
        let o = b * block;
        let yb = &mut y[o..o + block];
        wht_inplace(yb);
        let sb = &signs[o..o + block];
        for i in 0..block {
            yb[i] = yb[i] * inv_sqrt * sb[i];
        }
    }
}

/// KV-cache whitening: self-inverse normalized WHT applied per
/// `chunk`-sized span, **no signs**. With `Ĥ = H/√chunk` Sylvester,
/// `Ĥ² = I`, so the same call both whitens and un-whitens.
///
/// Used by the quantized-KV quality path (fork parity, `attn_rot_*`):
/// applied to Q and K post-RoPE (per-head dot products are exactly
/// preserved because chunks never cross head boundaries), to V before
/// the cache write, and to the attention output afterwards (which
/// un-rotates the V basis). Only the *cached representation* changes —
/// flattening the per-channel distribution so uniform Q4_0 bins lose
/// less.
///
/// `x.len()` must be a multiple of `chunk`; `chunk` a power of two.
pub fn whiten_chunks_inplace(x: &mut [f32], chunk: usize) {
    debug_assert_eq!(x.len() % chunk, 0);
    let inv_sqrt = 1.0f32 / (chunk as f32).sqrt();
    for b in 0..x.len() / chunk {
        let o = b * chunk;
        let xb = &mut x[o..o + chunk];
        wht_inplace(xb);
        for v in xb.iter_mut() {
            *v *= inv_sqrt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signs_pattern(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| if (i * 2654435761usize) & 0x10 == 0 { 1.0 } else { -1.0 })
            .collect()
    }

    fn x_pattern(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * 37 + 11) % 61) as f32 * 0.05 - 1.4).collect()
    }

    /// Dense reference: `R[r][c] = parity(r & c) ? -s : +s` scaled by
    /// `1/√n`, times the sign diagonal — ported from the PrismML
    /// fork's parity-matrix construction (llama-model.cpp popcount
    /// build). Forward reference: `out[r] = Σ_c H[r][c]·s[c]·x[c]/√n`.
    fn dense_forward_ref(x: &[f32], signs: &[f32], block: usize) -> Vec<f32> {
        let inv_sqrt = 1.0f64 / (block as f64).sqrt();
        let mut out = vec![0f32; x.len()];
        for b in 0..x.len() / block {
            let o = b * block;
            for r in 0..block {
                let mut acc = 0f64;
                for c in 0..block {
                    let h = if (r & c).count_ones() % 2 == 1 { -1.0 } else { 1.0 };
                    acc += h * (signs[o + c] as f64) * (x[o + c] as f64);
                }
                out[o + r] = (acc * inv_sqrt) as f32;
            }
        }
        out
    }

    #[test]
    fn forward_matches_dense_reference() {
        for block in [16usize, 64, 1024] {
            let n = block * 2;
            let x = x_pattern(n);
            let s = signs_pattern(n);
            let mut fast = vec![0f32; n];
            hadamard_forward(&x, &s, block, &mut fast);
            let dense = dense_forward_ref(&x, &s, block);
            for i in 0..n {
                assert!(
                    (fast[i] - dense[i]).abs() < 1e-3,
                    "block {block} elem {i}: fast {} vs dense {}",
                    fast[i],
                    dense[i]
                );
            }
        }
    }

    #[test]
    fn forward_then_inverse_is_identity() {
        let block = 128usize;
        let n = block * 3;
        let x = x_pattern(n);
        let s = signs_pattern(n);
        let mut y = vec![0f32; n];
        hadamard_forward(&x, &s, block, &mut y);
        hadamard_inverse_inplace(&mut y, &s, block);
        for i in 0..n {
            assert!(
                (y[i] - x[i]).abs() < 1e-4,
                "elem {i}: round-trip {} vs original {}",
                y[i],
                x[i]
            );
        }
    }

    #[test]
    fn whiten_is_self_inverse() {
        for chunk in [64usize, 128] {
            let n = chunk * 4;
            let x = x_pattern(n);
            let mut y = x.clone();
            whiten_chunks_inplace(&mut y, chunk);
            // Whitened must differ from the input (transform is real).
            assert!(
                y.iter().zip(&x).any(|(a, b)| (a - b).abs() > 1e-3),
                "chunk {chunk}: whitening was a no-op"
            );
            whiten_chunks_inplace(&mut y, chunk);
            for i in 0..n {
                assert!(
                    (y[i] - x[i]).abs() < 1e-4,
                    "chunk {chunk} elem {i}: double-whiten {} vs original {}",
                    y[i],
                    x[i]
                );
            }
        }
    }

    #[test]
    fn whiten_preserves_dot_products() {
        // Orthonormality: <Ĥq, Ĥk> == <q, k> per chunk, hence for the
        // whole vector when both sides are whitened with equal chunks.
        let chunk = 64usize;
        let n = chunk * 4; // one 256-wide "head" = 4 chunks
        let q = x_pattern(n);
        let k: Vec<f32> = (0..n).map(|i| ((i * 53 + 7) % 97) as f32 * 0.03 - 1.2).collect();
        let dot_before: f64 = q.iter().zip(&k).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
        let mut qw = q.clone();
        let mut kw = k.clone();
        whiten_chunks_inplace(&mut qw, chunk);
        whiten_chunks_inplace(&mut kw, chunk);
        let dot_after: f64 = qw.iter().zip(&kw).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
        assert!(
            (dot_before - dot_after).abs() < 1e-2 * dot_before.abs().max(1.0),
            "dot changed: {dot_before} vs {dot_after}"
        );
    }

    /// The order matters: forward is signs-then-WHT; flipping it is
    /// a different (wrong) transform whenever signs vary inside a
    /// block. This is the test that catches a transposed port.
    #[test]
    fn sign_order_asymmetry_detected() {
        let block = 32usize;
        let x = x_pattern(block);
        let s = signs_pattern(block);
        assert!(s.iter().any(|&v| v > 0.0) && s.iter().any(|&v| v < 0.0));
        let mut correct = vec![0f32; block];
        hadamard_forward(&x, &s, block, &mut correct);
        // Wrong order: WHT first, then signs (that's the INVERSE
        // shape, not the forward).
        let mut wrong = x.clone();
        wht_inplace(&mut wrong);
        let inv_sqrt = 1.0f32 / (block as f32).sqrt();
        for i in 0..block {
            wrong[i] = wrong[i] * inv_sqrt * s[i];
        }
        let max_diff = correct
            .iter()
            .zip(&wrong)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff > 1e-2,
            "sign/WHT order flip was not detectable (max_diff {max_diff}) — \
             test fixture too symmetric"
        );
    }
}
