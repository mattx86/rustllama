// deliberate: needless_range_loop — explicit position/head indexing in the online-
// softmax merge; too_many_arguments — flat kernel/test entry-point signatures.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

//! Cascade / shared-prefix attention (P3).
//!
//! When `N` sequences share a common prompt PREFIX — the APC case: one system
//! prompt, `N` concurrent user turns — the non-cascade path has every sequence
//! attend the full prefix independently, so the prefix K/V is stored `N` times
//! and re-streamed `N` times per decode tick (`O(prefix)×N`).
//!
//! Cascade attention computes the attention contribution of the SHARED prefix in
//! a single batched pass over one prefix K/V buffer (`O(prefix)` streaming, read
//! once and reused across all `N` queries), then for each sequence merges that
//! shared-prefix partial with the sequence's own SUFFIX partial via the standard
//! online-softmax `(m, l, acc)` merge — the SAME running-softmax merge the CUDA
//! flash-decode does across warps (`rsl_cuda.cu` ~L2662) and the CPU flash
//! `online_softmax_attn_*` does across positions.
//!
//! The per-query dot against the prefix is still `q_i · K_prefix` (each query
//! dots the shared prefix), so the FLOPs of the prefix scores are unchanged; the
//! win is **memory** — the prefix K/V lives in one buffer and is streamed once
//! per tick instead of `N` times — plus the batched, cache-resident pass shape
//! (load each prefix K/V row ONCE, reuse it across every sequence's query, the
//! same locality the F32 `gqa_attention_flash_prefill` kernel gets over its
//! `n_new` queries).
//!
//! ## Correctness
//!
//! The online-softmax merge is mathematically exact: for a key/value range split
//! into two disjoint chunks A (positions `[0, p)`) and B (positions `[p, L)`),
//! `merge(partial(A), partial(B))` equals a single online-softmax sweep over the
//! concatenation `A ++ B` (both equal full soft-max attention over `[0, L)` in
//! exact arithmetic; they differ only by floating-point rounding, ~1e-6). The
//! unit tests below pin `merge ≡ full` directly and `cascade ≡ per-sequence`
//! end-to-end for 2–4 same-prefix sequences.
//!
//! Nothing in this module is wired into the forward path yet — it is an additive
//! kernel surface the paged continuous-batching driver can call once it detects
//! ≥2 same-prefix slots (the driver auto-engage is the optimization; this is the
//! validated correctness primitive). A single sequence / no-shared-prefix caller
//! that keeps using the existing `gqa_attention_flash_decode` path is therefore
//! byte-identical.

/// Un-normalized online-softmax partial for one `(query, head)` over a chunk of
/// K/V. Tracks the three quantities the flash-attention merge needs:
///   - `m`   — running max of the scaled (optionally soft-capped) scores,
///   - `l`   — `Σ exp(s_t − m)`           (un-normalized soft-max denominator),
///   - `acc` — `Σ exp(s_t − m) · v_t`     (un-normalized weighted-V, len `head_dim`).
///
/// The normalized attention output is `acc / l`. Partials over disjoint,
/// contiguous chunks combine by [`FlashPartial::merge`] (the standard rescale
/// rule). An empty partial (`m = −∞`, `l = 0`, `acc = 0`) is the identity for
/// `merge`, so merging in a zero-length suffix is a no-op.
#[derive(Clone, Debug)]
pub struct FlashPartial {
    /// Running max of the scaled scores seen so far (`−∞` when empty).
    pub m: f32,
    /// Un-normalized soft-max denominator `Σ exp(s_t − m)`.
    pub l: f32,
    /// Un-normalized weighted-V accumulator `Σ exp(s_t − m) · v_t`, len `head_dim`.
    pub acc: Vec<f32>,
}

/// Fold one key/value position's scaled score `s` and value row `v_row` into an
/// online-softmax `(m, l, acc)` state. This is the single arithmetic kernel
/// shared by the per-head primitive and the batched cascade pass; it is exactly
/// the body of `online_softmax_attn_f32_scratch` (and the CUDA flash-decode warp
/// loop) minus the final normalization.
#[inline]
fn accumulate(m: &mut f32, l: &mut f32, acc: &mut [f32], s: f32, v_row: &[f32]) {
    let m_new = m.max(s);
    // `exp(m_old − m_new)` rescales the accumulators onto the new max. When the
    // state is still empty (`m_old = −∞`) the rescale is 0 (nothing to carry).
    let rescale = if m.is_finite() {
        (*m - m_new).exp()
    } else {
        0.0
    };
    let p = (s - m_new).exp();
    *l = *l * rescale + p;
    for d in 0..acc.len() {
        acc[d] = acc[d] * rescale + p * v_row[d];
    }
    *m = m_new;
}

/// Combine two un-normalized partials `a` and `b` onto a shared max. Mirrors the
/// CUDA flash-decode cross-warp merge (`rsl_cuda.cu` ~L2662): global max, then
/// each partial's denominator and weighted-V accumulator are rescaled by
/// `exp(m_x − m_g)` and summed.
#[inline]
fn merge_into(
    m_a: f32,
    l_a: f32,
    acc_a: &[f32],
    m_b: f32,
    l_b: f32,
    acc_b: &[f32],
    out: &mut [f32],
) -> (f32, f32) {
    let m_g = m_a.max(m_b);
    let e_a = if m_a.is_finite() {
        (m_a - m_g).exp()
    } else {
        0.0
    };
    let e_b = if m_b.is_finite() {
        (m_b - m_g).exp()
    } else {
        0.0
    };
    let l_g = l_a * e_a + l_b * e_b;
    for d in 0..out.len() {
        out[d] = acc_a[d] * e_a + acc_b[d] * e_b;
    }
    (m_g, l_g)
}

impl FlashPartial {
    /// The identity partial for a `head_dim`-wide head: `m = −∞`, `l = 0`,
    /// `acc = 0`. Merging it into any other partial is a no-op.
    pub fn empty(head_dim: usize) -> Self {
        FlashPartial {
            m: f32::NEG_INFINITY,
            l: 0.0,
            acc: vec![0.0; head_dim],
        }
    }

    /// Merge `other` (a partial over a DISJOINT, contiguous chunk) into `self`.
    /// Associative + commutative in exact arithmetic; order affects only the last
    /// bits of rounding. After the merge `self` is the partial over the union of
    /// the two chunks.
    pub fn merge(&mut self, other: &FlashPartial) {
        debug_assert_eq!(self.acc.len(), other.acc.len());
        // Temp copy of `self.acc` so `merge_into` can write back in place.
        let a = self.acc.clone();
        let (m_g, l_g) = merge_into(
            self.m,
            self.l,
            &a,
            other.m,
            other.l,
            &other.acc,
            &mut self.acc,
        );
        self.m = m_g;
        self.l = l_g;
    }

    /// Write the normalized attention output `acc / l` into `out` (len
    /// `head_dim`). An all-empty partial (`l == 0`) yields zeros — matching
    /// `gqa_attention_flash_decode`'s `kv_len == 0` behaviour.
    pub fn finalize(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.acc.len());
        let inv_l = if self.l > 0.0 { 1.0 / self.l } else { 0.0 };
        for d in 0..out.len() {
            out[d] = self.acc[d] * inv_l;
        }
    }
}

/// Un-normalized online-softmax partial for ONE query head over a tightly packed
/// `[kv_len, head_dim]` K/V scratch. `scale` is `1/sqrt(head_dim)`; `softcap > 0`
/// applies the Gemma2 attention-logit cap `s = c·tanh(s/c)` after the scale and
/// before the max/exp (no-op at `softcap == 0`).
///
/// `flash_partial_f32(q, k, v, .., kv_len, scale, 0.0).finalize(out)` equals
/// `gqa_attention_flash_decode` for a single head within rounding.
pub fn flash_partial_f32(
    q_h: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    kv_len: usize,
    scale: f32,
    softcap: f32,
) -> FlashPartial {
    debug_assert_eq!(q_h.len(), head_dim);
    debug_assert!(k.len() >= kv_len * head_dim);
    debug_assert!(v.len() >= kv_len * head_dim);
    let mut p = FlashPartial::empty(head_dim);
    for t in 0..kv_len {
        let off = t * head_dim;
        let k_row = &k[off..off + head_dim];
        let v_row = &v[off..off + head_dim];
        let mut s = 0.0f32;
        for d in 0..head_dim {
            s += q_h[d] * k_row[d];
        }
        s *= scale;
        if softcap > 0.0 {
            s = softcap * (s / softcap).tanh();
        }
        accumulate(&mut p.m, &mut p.l, &mut p.acc, s, v_row);
    }
    p
}

/// One sequence's private SUFFIX K/V (the positions AFTER the shared prefix),
/// tightly packed `[n_kv_heads, len, head_dim]`.
pub struct CascadeSuffix<'a> {
    /// Suffix K, `[n_kv_heads, len, head_dim]` row-major (tight, stride `len`).
    pub k: &'a [f32],
    /// Suffix V, same layout.
    pub v: &'a [f32],
    /// Number of suffix KV positions for this sequence (may be 0).
    pub len: usize,
}

/// Cascade / shared-prefix flash-attention DECODE for `n_seq` sequences that
/// share the prefix K/V `[n_kv_heads, prefix_len, head_dim]`, each with its own
/// one-token query `[n_heads, head_dim]` and private suffix K/V.
///
/// Layouts (all tightly packed, row-major):
///   - `prefix_k` / `prefix_v`: `[n_kv_heads, prefix_len, head_dim]` — ONE buffer
///     shared by every sequence (this is the memory win; the non-cascade path
///     holds `n_seq` copies of it).
///   - `queries`:               `[n_seq, n_heads, head_dim]`.
///   - `suffixes[s]`:           `[n_kv_heads, suffixes[s].len, head_dim]`.
///   - `out`:                   `[n_seq, n_heads, head_dim]` (written).
///
/// GQA head mapping matches the F32 kernels: head `h = kv_h·n_gqa + g` reads
/// kv-head `kv_h = h / n_gqa` with `n_gqa = n_heads / n_kv_heads`. The scale is
/// `1/sqrt(head_dim)`; `softcap > 0` applies the Gemma2 cap to every score.
///
/// For each sequence `s` and head `h` the result is
/// `merge( attn(q_{s,h}, prefix_KV),  attn(q_{s,h}, suffix_s_KV) )`, which equals
/// full soft-max attention over the concatenation `prefix ++ suffix_s` (the
/// `cascade ≡ per-sequence` anchor).
///
/// The prefix contribution is computed in a single batched pass: each prefix K/V
/// row is loaded once and folded into every sequence's partial — streaming the
/// prefix `O(prefix)` once rather than `O(prefix)·n_seq`.
#[allow(clippy::too_many_arguments)]
pub fn cascade_attention_f32(
    prefix_k: &[f32],
    prefix_v: &[f32],
    prefix_len: usize,
    queries: &[f32],
    suffixes: &[CascadeSuffix],
    out: &mut [f32],
    n_seq: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    softcap: f32,
) {
    debug_assert_eq!(queries.len(), n_seq * n_heads * head_dim);
    debug_assert_eq!(out.len(), n_seq * n_heads * head_dim);
    debug_assert_eq!(suffixes.len(), n_seq);
    debug_assert_eq!(n_heads % n_kv_heads, 0);
    debug_assert_eq!(prefix_k.len(), n_kv_heads * prefix_len * head_dim);
    debug_assert_eq!(prefix_v.len(), n_kv_heads * prefix_len * head_dim);

    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_part = n_seq * n_heads;

    // Prefix partials, flat `[n_seq, n_heads]` × `head_dim`. Shared by the
    // batched prefix pass, then merged with the per-sequence suffix partials.
    let mut p_m = vec![f32::NEG_INFINITY; n_part];
    let mut p_l = vec![0.0f32; n_part];
    let mut p_acc = vec![0.0f32; n_part * head_dim];

    // --- Batched shared-prefix pass -----------------------------------------
    // Walk the prefix K/V ONCE (per kv-head). Each K/V row is loaded a single
    // time and folded into every sequence's query partial — the cache-locality
    // + single-stream win over `n_seq` independent prefix scans.
    for kv_h in 0..n_kv_heads {
        let cache_base = kv_h * prefix_len * head_dim;
        for t in 0..prefix_len {
            let off = cache_base + t * head_dim;
            let k_row = &prefix_k[off..off + head_dim];
            let v_row = &prefix_v[off..off + head_dim];
            for s in 0..n_seq {
                for g in 0..n_gqa {
                    let h = kv_h * n_gqa + g;
                    let part = s * n_heads + h;
                    let q_off = part * head_dim;
                    let q_row = &queries[q_off..q_off + head_dim];
                    let mut sc = 0.0f32;
                    for d in 0..head_dim {
                        sc += q_row[d] * k_row[d];
                    }
                    sc *= scale;
                    if softcap > 0.0 {
                        sc = softcap * (sc / softcap).tanh();
                    }
                    let acc = &mut p_acc[q_off..q_off + head_dim];
                    accumulate(&mut p_m[part], &mut p_l[part], acc, sc, v_row);
                }
            }
        }
    }

    // --- Per-sequence suffix pass + merge -----------------------------------
    // Each sequence's suffix is private, so this is an ordinary per-sequence
    // online-softmax pass; its partial is then merged with the shared-prefix
    // partial and normalized straight into `out`.
    let mut s_acc = vec![0.0f32; head_dim];
    for s in 0..n_seq {
        let suf = &suffixes[s];
        debug_assert_eq!(suf.k.len(), n_kv_heads * suf.len * head_dim);
        debug_assert_eq!(suf.v.len(), n_kv_heads * suf.len * head_dim);
        for kv_h in 0..n_kv_heads {
            let cache_base = kv_h * suf.len * head_dim;
            for g in 0..n_gqa {
                let h = kv_h * n_gqa + g;
                let part = s * n_heads + h;
                let q_off = part * head_dim;
                let q_row = &queries[q_off..q_off + head_dim];
                // Fresh suffix partial for this (seq, head).
                let mut sm = f32::NEG_INFINITY;
                let mut sl = 0.0f32;
                for x in s_acc.iter_mut() {
                    *x = 0.0;
                }
                for t in 0..suf.len {
                    let off = cache_base + t * head_dim;
                    let k_row = &suf.k[off..off + head_dim];
                    let v_row = &suf.v[off..off + head_dim];
                    let mut sc = 0.0f32;
                    for d in 0..head_dim {
                        sc += q_row[d] * k_row[d];
                    }
                    sc *= scale;
                    if softcap > 0.0 {
                        sc = softcap * (sc / softcap).tanh();
                    }
                    accumulate(&mut sm, &mut sl, &mut s_acc, sc, v_row);
                }
                // Merge the shared-prefix partial with this suffix partial and
                // write the normalized output.
                let out_row = &mut out[q_off..q_off + head_dim];
                let p_acc_row = &p_acc[q_off..q_off + head_dim];
                let (_m_g, l_g) =
                    merge_into(p_m[part], p_l[part], p_acc_row, sm, sl, &s_acc, out_row);
                let inv_l = if l_g > 0.0 { 1.0 / l_g } else { 0.0 };
                for x in out_row.iter_mut() {
                    *x *= inv_l;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gqa_attention_flash_decode;

    // Small deterministic PRNG (xorshift) so the tests are reproducible without a
    // dev-dependency. Returns values in roughly [-1, 1).
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next_f32()).collect()
        }
    }

    /// Full soft-max attention over a tight `[kv_len, head_dim]` K/V for one head
    /// — the independent, non-flash reference (a second opinion beside
    /// `gqa_attention_flash_decode`).
    fn naive_attn_one_head(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        head_dim: usize,
        kv_len: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut scores = vec![0.0f32; kv_len];
        for (t, sc) in scores.iter_mut().enumerate() {
            let off = t * head_dim;
            let mut s = 0.0f32;
            for d in 0..head_dim {
                s += q[d] * k[off + d];
            }
            *sc = s * scale;
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for sc in scores.iter_mut() {
            *sc = (*sc - m).exp();
            denom += *sc;
        }
        let mut out = vec![0.0f32; head_dim];
        for (t, w) in scores.iter().enumerate() {
            let w = if denom > 0.0 { w / denom } else { 0.0 };
            let off = t * head_dim;
            for d in 0..head_dim {
                out[d] += w * v[off + d];
            }
        }
        out
    }

    /// merge ≡ full: split a single head's K/V range into chunk A `[0, p)` and
    /// chunk B `[p, L)`, compute each partial, merge, finalize, and compare to
    /// full attention over `[0, L)`. Proves the online-softmax merge primitive.
    #[test]
    fn merge_partial_equals_full_attention() {
        let head_dim = 64;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut rng = Rng(0x1234_5678_9abc_def0);
        // Several random splits of several lengths, including degenerate splits.
        for &(kv_len, split) in &[
            (1usize, 0usize),
            (1, 1),
            (8, 0),
            (8, 8),
            (8, 3),
            (37, 11),
            (128, 64),
            (200, 1),
            (200, 199),
        ] {
            let q = rng.vec(head_dim);
            let k = rng.vec(kv_len * head_dim);
            let v = rng.vec(kv_len * head_dim);

            // Full reference (two independent references agree too).
            let full_naive = naive_attn_one_head(&q, &k, &v, head_dim, kv_len, scale);
            let mut full_flash = vec![0.0f32; head_dim];
            gqa_attention_flash_decode(
                &q,
                &k,
                &v,
                &mut full_flash,
                1,
                1,
                head_dim,
                kv_len.max(1),
                kv_len,
            );

            // Cascade: partial(A) merged with partial(B).
            let a = flash_partial_f32(
                &q,
                &k[..split * head_dim],
                &v[..split * head_dim],
                head_dim,
                split,
                scale,
                0.0,
            );
            let b = flash_partial_f32(
                &q,
                &k[split * head_dim..],
                &v[split * head_dim..],
                head_dim,
                kv_len - split,
                scale,
                0.0,
            );
            let mut merged = a.clone();
            merged.merge(&b);
            let mut got = vec![0.0f32; head_dim];
            merged.finalize(&mut got);

            for d in 0..head_dim {
                assert!(
                    (got[d] - full_naive[d]).abs() < 1e-4,
                    "merge≡full(naive) kv_len={kv_len} split={split} d={d}: {} vs {}",
                    got[d],
                    full_naive[d]
                );
                assert!(
                    (got[d] - full_flash[d]).abs() < 1e-4,
                    "merge≡full(flash) kv_len={kv_len} split={split} d={d}: {} vs {}",
                    got[d],
                    full_flash[d]
                );
            }
            // Merge is commutative within rounding.
            let mut merged_rev = b.clone();
            merged_rev.merge(&a);
            let mut got_rev = vec![0.0f32; head_dim];
            merged_rev.finalize(&mut got_rev);
            for d in 0..head_dim {
                assert!(
                    (got_rev[d] - got[d]).abs() < 1e-4,
                    "merge commutativity kv_len={kv_len} split={split} d={d}"
                );
            }
        }
    }

    /// merge ≡ full with the Gemma2 soft-cap engaged on every score.
    #[test]
    fn merge_partial_equals_full_attention_softcap() {
        let head_dim = 32;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let softcap = 50.0f32;
        let mut rng = Rng(0xdead_beef_0bad_f00d);
        for &(kv_len, split) in &[(8usize, 3usize), (64, 16), (100, 50)] {
            let q = rng.vec(head_dim);
            let k = rng.vec(kv_len * head_dim);
            let v = rng.vec(kv_len * head_dim);

            // Full soft-cap reference via the online-softmax scratch kernel.
            let mut full = vec![0.0f32; head_dim];
            crate::turboquant::online_softmax_attn_f32_scratch_capped_windowed(
                &q, &k, &v, &mut full, head_dim, kv_len, scale, 0, softcap,
            );

            let a = flash_partial_f32(
                &q,
                &k[..split * head_dim],
                &v[..split * head_dim],
                head_dim,
                split,
                scale,
                softcap,
            );
            let b = flash_partial_f32(
                &q,
                &k[split * head_dim..],
                &v[split * head_dim..],
                head_dim,
                kv_len - split,
                scale,
                softcap,
            );
            let mut merged = a.clone();
            merged.merge(&b);
            let mut got = vec![0.0f32; head_dim];
            merged.finalize(&mut got);
            for d in 0..head_dim {
                assert!(
                    (got[d] - full[d]).abs() < 1e-4,
                    "merge≡full softcap kv_len={kv_len} split={split} d={d}: {} vs {}",
                    got[d],
                    full[d]
                );
            }
        }
    }

    /// Merging in an empty (zero-length) partial is a no-op (identity element).
    #[test]
    fn merge_empty_is_identity() {
        let head_dim = 16;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut rng = Rng(42);
        let q = rng.vec(head_dim);
        let k = rng.vec(10 * head_dim);
        let v = rng.vec(10 * head_dim);
        let p = flash_partial_f32(&q, &k, &v, head_dim, 10, scale, 0.0);
        let mut got = vec![0.0f32; head_dim];
        let mut merged = p.clone();
        merged.merge(&FlashPartial::empty(head_dim));
        merged.finalize(&mut got);
        let mut want = vec![0.0f32; head_dim];
        p.finalize(&mut want);
        assert_eq!(got, want, "merge(p, empty) must equal p bit-for-bit");

        // empty ⊕ empty finalizes to zeros (kv_len==0 semantics).
        let mut z = vec![9.0f32; head_dim];
        let e = FlashPartial::empty(head_dim);
        e.finalize(&mut z);
        assert!(z.iter().all(|&x| x == 0.0));
    }

    /// Build a per-sequence full K/V `[n_kv_heads, prefix_len+suffix_len, head_dim]`
    /// = prefix (shared) ++ suffix_s, run full `gqa_attention_flash_decode`, and
    /// compare to the cascade output. Proves `cascade ≡ per-sequence`.
    fn check_cascade_equals_per_sequence(
        n_seq: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        prefix_len: usize,
        suffix_lens: &[usize],
        softcap: f32,
        seed: u64,
    ) {
        let mut rng = Rng(seed);
        let prefix_k = rng.vec(n_kv_heads * prefix_len * head_dim);
        let prefix_v = rng.vec(n_kv_heads * prefix_len * head_dim);
        let queries = rng.vec(n_seq * n_heads * head_dim);

        let mut suffix_k = Vec::with_capacity(n_seq);
        let mut suffix_v = Vec::with_capacity(n_seq);
        for &sl in suffix_lens {
            suffix_k.push(rng.vec(n_kv_heads * sl * head_dim));
            suffix_v.push(rng.vec(n_kv_heads * sl * head_dim));
        }
        let suffixes: Vec<CascadeSuffix> = (0..n_seq)
            .map(|s| CascadeSuffix {
                k: &suffix_k[s],
                v: &suffix_v[s],
                len: suffix_lens[s],
            })
            .collect();

        let mut cascade_out = vec![0.0f32; n_seq * n_heads * head_dim];
        cascade_attention_f32(
            &prefix_k,
            &prefix_v,
            prefix_len,
            &queries,
            &suffixes,
            &mut cascade_out,
            n_seq,
            n_heads,
            n_kv_heads,
            head_dim,
            softcap,
        );

        // Per-sequence reference: concat prefix ++ suffix_s into one tight cache.
        let scale = 1.0 / (head_dim as f32).sqrt();
        for s in 0..n_seq {
            let sl = suffix_lens[s];
            let kv_len = prefix_len + sl;
            let mut k_full = vec![0.0f32; n_kv_heads * kv_len * head_dim];
            let mut v_full = vec![0.0f32; n_kv_heads * kv_len * head_dim];
            for kv_h in 0..n_kv_heads {
                let dst = kv_h * kv_len * head_dim;
                let psrc = kv_h * prefix_len * head_dim;
                k_full[dst..dst + prefix_len * head_dim]
                    .copy_from_slice(&prefix_k[psrc..psrc + prefix_len * head_dim]);
                v_full[dst..dst + prefix_len * head_dim]
                    .copy_from_slice(&prefix_v[psrc..psrc + prefix_len * head_dim]);
                let ssrc = kv_h * sl * head_dim;
                k_full[dst + prefix_len * head_dim..dst + kv_len * head_dim]
                    .copy_from_slice(&suffix_k[s][ssrc..ssrc + sl * head_dim]);
                v_full[dst + prefix_len * head_dim..dst + kv_len * head_dim]
                    .copy_from_slice(&suffix_v[s][ssrc..ssrc + sl * head_dim]);
            }
            let q_s = &queries[s * n_heads * head_dim..(s + 1) * n_heads * head_dim];
            let mut ref_out = vec![0.0f32; n_heads * head_dim];
            if softcap > 0.0 {
                // Per-head soft-cap reference via the scratch kernel.
                for kv_h in 0..n_kv_heads {
                    let kb = kv_h * kv_len * head_dim;
                    for g in 0..(n_heads / n_kv_heads) {
                        let h = kv_h * (n_heads / n_kv_heads) + g;
                        let out_h = &mut ref_out[h * head_dim..(h + 1) * head_dim];
                        crate::turboquant::online_softmax_attn_f32_scratch_capped_windowed(
                            &q_s[h * head_dim..(h + 1) * head_dim],
                            &k_full[kb..kb + kv_len * head_dim],
                            &v_full[kb..kb + kv_len * head_dim],
                            out_h,
                            head_dim,
                            kv_len,
                            scale,
                            0,
                            softcap,
                        );
                    }
                }
            } else {
                gqa_attention_flash_decode(
                    q_s,
                    &k_full,
                    &v_full,
                    &mut ref_out,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_len,
                    kv_len,
                );
            }
            let cas_s = &cascade_out[s * n_heads * head_dim..(s + 1) * n_heads * head_dim];
            for i in 0..n_heads * head_dim {
                assert!(
                    (cas_s[i] - ref_out[i]).abs() < 1e-4,
                    "cascade≡per-seq seq={s} i={i}: {} vs {} (softcap={softcap})",
                    cas_s[i],
                    ref_out[i]
                );
            }
        }
    }

    #[test]
    fn cascade_equals_per_sequence_mha() {
        // Multi-head (n_kv_heads == n_heads), 4 same-prefix sequences, varied suffixes.
        check_cascade_equals_per_sequence(4, 4, 4, 32, 40, &[3, 7, 1, 12], 0.0, 0xa1);
    }

    #[test]
    fn cascade_equals_per_sequence_gqa() {
        // GQA (Qwen2.5-0.5B shape: 14 heads / 2 kv-heads, head_dim 64), 3 seqs.
        check_cascade_equals_per_sequence(3, 14, 2, 64, 96, &[5, 9, 2], 0.0, 0xb2);
    }

    #[test]
    fn cascade_equals_per_sequence_softcap() {
        // Gemma2-style soft-cap, 2 same-prefix sequences.
        check_cascade_equals_per_sequence(2, 4, 2, 32, 64, &[6, 10], 50.0, 0xc3);
    }

    #[test]
    fn cascade_equals_per_sequence_zero_suffix() {
        // A sequence whose suffix is empty (query attends only the shared prefix)
        // must still equal per-sequence attention over the prefix alone.
        check_cascade_equals_per_sequence(3, 8, 2, 16, 24, &[0, 4, 0], 0.0, 0xd4);
    }

    #[test]
    fn cascade_equals_per_sequence_no_shared_prefix() {
        // prefix_len == 0 ⇒ cascade degenerates to pure per-sequence suffix
        // attention (the "no shared prefix" case stays correct).
        check_cascade_equals_per_sequence(2, 4, 2, 16, 0, &[5, 8], 0.0, 0xe5);
    }

    /// Micro-benchmark of the shared-prefix pass: cascade (one shared prefix
    /// buffer, batched pass) vs the non-cascade baseline (every sequence holds +
    /// re-scans its own prefix copy). Reports wall time, speedup, and the prefix
    /// KV memory-footprint / streamed-bytes reduction.
    ///
    /// Ignored by default (timing is environment-dependent); run explicitly:
    /// `cargo test -p rustllama-kernels-cpu cascade_microbench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn cascade_microbench() {
        use std::time::Instant;
        // Qwen2.5-0.5B-ish attention shape.
        let n_heads = 14;
        let n_kv_heads = 2;
        let head_dim = 64;
        let prefix_len = 512; // long shared system prompt
        let suffix_len = 8; // short per-user decode tail
        let reps = 200;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let _ = scale;

        for &n_seq in &[4usize, 16] {
            let mut rng = Rng(0x5151_5151 ^ n_seq as u64);
            let prefix_k = rng.vec(n_kv_heads * prefix_len * head_dim);
            let prefix_v = rng.vec(n_kv_heads * prefix_len * head_dim);
            let queries = rng.vec(n_seq * n_heads * head_dim);
            let mut suffix_k = Vec::new();
            let mut suffix_v = Vec::new();
            for _ in 0..n_seq {
                suffix_k.push(rng.vec(n_kv_heads * suffix_len * head_dim));
                suffix_v.push(rng.vec(n_kv_heads * suffix_len * head_dim));
            }
            let suffixes: Vec<CascadeSuffix> = (0..n_seq)
                .map(|s| CascadeSuffix {
                    k: &suffix_k[s],
                    v: &suffix_v[s],
                    len: suffix_len,
                })
                .collect();
            let mut cas_out = vec![0.0f32; n_seq * n_heads * head_dim];

            // Cascade timing.
            let t0 = Instant::now();
            for _ in 0..reps {
                cascade_attention_f32(
                    &prefix_k,
                    &prefix_v,
                    prefix_len,
                    &queries,
                    &suffixes,
                    &mut cas_out,
                    n_seq,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    0.0,
                );
            }
            let cas_t = t0.elapsed().as_secs_f64() / reps as f64;

            // Non-cascade baseline: each sequence owns a full prefix++suffix copy
            // and runs an independent flash-decode over it.
            let mut full_k = Vec::with_capacity(n_seq);
            let mut full_v = Vec::with_capacity(n_seq);
            let kv_len = prefix_len + suffix_len;
            for s in 0..n_seq {
                let mut kf = vec![0.0f32; n_kv_heads * kv_len * head_dim];
                let mut vf = vec![0.0f32; n_kv_heads * kv_len * head_dim];
                for kv_h in 0..n_kv_heads {
                    let dst = kv_h * kv_len * head_dim;
                    let psrc = kv_h * prefix_len * head_dim;
                    kf[dst..dst + prefix_len * head_dim]
                        .copy_from_slice(&prefix_k[psrc..psrc + prefix_len * head_dim]);
                    vf[dst..dst + prefix_len * head_dim]
                        .copy_from_slice(&prefix_v[psrc..psrc + prefix_len * head_dim]);
                    let ssrc = kv_h * suffix_len * head_dim;
                    kf[dst + prefix_len * head_dim..dst + kv_len * head_dim]
                        .copy_from_slice(&suffix_k[s][ssrc..ssrc + suffix_len * head_dim]);
                    vf[dst + prefix_len * head_dim..dst + kv_len * head_dim]
                        .copy_from_slice(&suffix_v[s][ssrc..ssrc + suffix_len * head_dim]);
                }
                full_k.push(kf);
                full_v.push(vf);
            }
            let mut base_out = vec![0.0f32; n_seq * n_heads * head_dim];
            let t1 = Instant::now();
            for _ in 0..reps {
                for s in 0..n_seq {
                    let q_s = &queries[s * n_heads * head_dim..(s + 1) * n_heads * head_dim];
                    let o_s = &mut base_out[s * n_heads * head_dim..(s + 1) * n_heads * head_dim];
                    gqa_attention_flash_decode(
                        q_s, &full_k[s], &full_v[s], o_s, n_heads, n_kv_heads, head_dim, kv_len,
                        kv_len,
                    );
                }
            }
            let base_t = t1.elapsed().as_secs_f64() / reps as f64;

            // Scalar baseline — the apples-to-apples wall-time twin for this
            // (scalar) cascade reference, isolating the memory-locality effect
            // from the SIMD dispatch the SIMD baseline above enjoys.
            let mut base_scalar_out = vec![0.0f32; n_seq * n_heads * head_dim];
            let t2 = Instant::now();
            for _ in 0..reps {
                for s in 0..n_seq {
                    let q_s = &queries[s * n_heads * head_dim..(s + 1) * n_heads * head_dim];
                    let o_s =
                        &mut base_scalar_out[s * n_heads * head_dim..(s + 1) * n_heads * head_dim];
                    crate::gqa_attention_flash_decode_scalar(
                        q_s, &full_k[s], &full_v[s], o_s, n_heads, n_kv_heads, head_dim, kv_len,
                        kv_len, 0, 0.0,
                    );
                }
            }
            let base_scalar_t = t2.elapsed().as_secs_f64() / reps as f64;

            // Prefix KV footprint: cascade keeps 1 copy, baseline keeps n_seq.
            let prefix_bytes = (n_kv_heads * prefix_len * head_dim * 4 * 2) as f64; // K+V, f32
            let cas_prefix_store = prefix_bytes;
            let base_prefix_store = prefix_bytes * n_seq as f64;

            println!(
                "[cascade microbench] N={n_seq} prefix_len={prefix_len} suffix_len={suffix_len} \
                 (heads {n_heads}/{n_kv_heads}, head_dim {head_dim}):"
            );
            println!(
                "    cascade  tick = {:.3} us   ({:.0} ticks/s)",
                cas_t * 1e6,
                1.0 / cas_t
            );
            println!(
                "    baseline tick = {:.3} us   ({:.0} ticks/s)  [SIMD-dispatched flash-decode]",
                base_t * 1e6,
                1.0 / base_t
            );
            println!(
                "    baseline tick = {:.3} us   ({:.0} ticks/s)  [SCALAR flash-decode, apples-to-apples]",
                base_scalar_t * 1e6,
                1.0 / base_scalar_t
            );
            println!(
                "    speedup vs scalar baseline = {:.2}x   (vs SIMD baseline = {:.2}x)",
                base_scalar_t / cas_t,
                base_t / cas_t
            );
            println!(
                "    prefix KV store/stream = {:.2} MiB (cascade) vs {:.2} MiB (baseline) => {:.1}x reduction",
                cas_prefix_store / (1024.0 * 1024.0),
                base_prefix_store / (1024.0 * 1024.0),
                base_prefix_store / cas_prefix_store
            );

            // Sanity: both paths agree (cascade is correct at the bench shape too).
            for i in 0..n_seq * n_heads * head_dim {
                assert!(
                    (cas_out[i] - base_out[i]).abs() < 1e-4,
                    "microbench cascade≠baseline i={i}"
                );
            }
        }
    }
}
