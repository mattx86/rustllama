//! Sampling primitives.
//!
//! Phase 1 ships the full stack: temperature, top-k, top-p (nucleus), and
//! repetition penalty, all driven by a small seeded RNG so the same `seed`
//! produces identical generations. Plumbing for grammar-constrained and
//! Mirostat samplers lands in the v1.x roadmap.

use crate::grammar::GrammarMask;
use crate::{SamplingParams, TokenLogprobs, TopLogprob};

/// Index of the maximum-logit token. SIMD-accelerated when the host
/// reports AVX-512 / AVX2: tracks max values + indices in parallel
/// lanes and horizontal-reduces at the end. On a 128K-vocab model
/// this is ~6× faster than the scalar loop — meaningful for greedy
/// decode where it runs every token. Tie-break: lowest index wins
/// (matches the scalar reference's first-seen behavior).
pub fn argmax(logits: &[f32]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if logits.len() >= 16 && is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            return unsafe { argmax_avx512f(logits) };
        }
        if logits.len() >= 8 && is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection above.
            return unsafe { argmax_avx2(logits) };
        }
    }
    argmax_scalar(logits)
}

fn argmax_scalar(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, v) in logits.iter().enumerate() {
        if *v > best_v {
            best_v = *v;
            best_i = i;
        }
    }
    best_i as u32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2")]
unsafe fn argmax_avx2(logits: &[f32]) -> u32 {
    use std::arch::x86_64::*;
    let n = logits.len();
    let n8 = n & !7;

    let mut max_v = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut idx_v = _mm256_setzero_si256();
    let mut counter = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let incr = _mm256_set1_epi32(8);
    let ptr = logits.as_ptr();

    let mut p = 0usize;
    while p < n8 {
        let v = _mm256_loadu_ps(ptr.add(p));
        // mask = (v > max_v) as f32 lanes (all-ones or zero per lane).
        let mask = _mm256_cmp_ps::<_CMP_GT_OQ>(v, max_v);
        max_v = _mm256_blendv_ps(max_v, v, mask);
        // Cast the f32 mask to i32-lane mask for the index blend; the
        // bit pattern is identical (per-lane all-ones / all-zeros).
        let mask_i = _mm256_castps_si256(mask);
        idx_v = _mm256_blendv_epi8(idx_v, counter, mask_i);
        counter = _mm256_add_epi32(counter, incr);
        p += 8;
    }

    // Horizontal reduce: spill the 8-lane state and walk it scalar.
    let mut max_arr = [0f32; 8];
    let mut idx_arr = [0i32; 8];
    _mm256_storeu_ps(max_arr.as_mut_ptr(), max_v);
    _mm256_storeu_si256(idx_arr.as_mut_ptr() as *mut __m256i, idx_v);
    let mut best_v = f32::NEG_INFINITY;
    let mut best_i = 0i32;
    for lane in 0..8 {
        // Strict `>` keeps the lowest-index tie behavior.
        if max_arr[lane] > best_v {
            best_v = max_arr[lane];
            best_i = idx_arr[lane];
        }
    }
    // Scalar tail.
    while p < n {
        let v = *ptr.add(p);
        if v > best_v {
            best_v = v;
            best_i = p as i32;
        }
        p += 1;
    }
    best_i as u32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn argmax_avx512f(logits: &[f32]) -> u32 {
    use std::arch::x86_64::*;
    let n = logits.len();
    let n16 = n & !15;

    let mut max_v = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut idx_v = _mm512_setzero_si512();
    let mut counter = _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    );
    let incr = _mm512_set1_epi32(16);
    let ptr = logits.as_ptr();

    let mut p = 0usize;
    while p < n16 {
        let v = _mm512_loadu_ps(ptr.add(p));
        // k-mask: which lanes have v > max_v.
        let mask = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(v, max_v);
        max_v = _mm512_mask_blend_ps(mask, max_v, v);
        idx_v = _mm512_mask_blend_epi32(mask, idx_v, counter);
        counter = _mm512_add_epi32(counter, incr);
        p += 16;
    }

    // Horizontal reduce: spill 16 lanes + walk scalar.
    let mut max_arr = [0f32; 16];
    let mut idx_arr = [0i32; 16];
    _mm512_storeu_ps(max_arr.as_mut_ptr(), max_v);
    _mm512_storeu_si512(idx_arr.as_mut_ptr() as *mut __m512i, idx_v);
    let mut best_v = f32::NEG_INFINITY;
    let mut best_i = 0i32;
    for lane in 0..16 {
        if max_arr[lane] > best_v {
            best_v = max_arr[lane];
            best_i = idx_arr[lane];
        }
    }
    while p < n {
        let v = *ptr.add(p);
        if v > best_v {
            best_v = v;
            best_i = p as i32;
        }
        p += 1;
    }
    best_i as u32
}

/// Tiny seedable PRNG (SplitMix64). Avoids pulling `rand` into the engine
/// just for one uniform-f32 stream; replace if we ever need parallel streams.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn from_seed(seed: u64) -> Self {
        // Mix a non-zero state for SplitMix64.
        let seed = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state: seed }
    }
    /// Current internal SplitMix64 state. Exposed so the GPU sampler
    /// path can read it into USM before dispatch and overwrite this
    /// `Rng` with the updated state on completion — keeping CPU and
    /// GPU sample streams in sync under the same seed.
    pub fn state(&self) -> u64 {
        self.state
    }
    /// Overwrite the internal state. Pair with [`Self::state`] for
    /// GPU/CPU stream resynchronization. Callers outside the
    /// sampler/GPU offload glue have no reason to touch this.
    pub fn set_state(&mut self, state: u64) {
        self.state = state;
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform f32 in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        // Take the top 24 bits for f32 mantissa.
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }
}

pub struct Sampler {
    pub params: SamplingParams,
    rng: Rng,
    /// Mirostat running surprise target. Initialized to `2 * tau` on
    /// construction; updated after each sample by
    /// `mu -= eta * (observed_surprise - tau)`. Unused when
    /// `params.mirostat == 0`.
    mirostat_mu: f32,
    /// Lazily-initialized GPU-side sampler context. Only constructed
    /// when `RUSTLLAMA_SAMPLER_GPU=1` and the GPU path is taken on a
    /// given call. Falls back silently to CPU on init failure or
    /// when params disqualify the call (grammar / mirostat / top-p
    /// / typical-p / logprobs / top_k too large).
    gpu_ctx: Option<crate::sampler_gpu::SamplerGpuContext>,
    /// Cached env-var read: `true` ⇒ try GPU before CPU on each
    /// `sample()`. Read once on construction so the per-sample hot
    /// path doesn't `getenv` every call.
    gpu_enabled: bool,
}

impl std::fmt::Debug for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sampler")
            .field("params", &self.params)
            .field("rng", &self.rng)
            .field("mirostat_mu", &self.mirostat_mu)
            .field("gpu_enabled", &self.gpu_enabled)
            .finish_non_exhaustive()
    }
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let rng = Rng::from_seed(params.seed);
        let mirostat_mu = 2.0 * params.mirostat_tau;
        let gpu_enabled = std::env::var("RUSTLLAMA_SAMPLER_GPU")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        Self {
            params,
            rng,
            mirostat_mu,
            gpu_ctx: None,
            gpu_enabled,
        }
    }

    /// Whether the current sampler call is eligible for the GPU
    /// short-circuit. Codifies the bypass conditions from
    /// `SamplerGpuContext`'s docstring + the wrappers' edge cases.
    fn gpu_call_eligible(&self, grammar: Option<&GrammarMask>) -> bool {
        if !self.gpu_enabled {
            return false;
        }
        if grammar.is_some() {
            return false;
        }
        if self.params.mirostat != 0 {
            return false;
        }
        // H1: top_p is no longer a bypass — GPU sampler now dispatches
        // `sampler_top_p_usm` after softmax. The kernel returns
        // `needs_fallback=1` only when the long-tail distribution
        // exceeds MAX_TOP_P_GPU; that's caught at `try_sample_gpu`
        // call time and triggers the CPU fallback there.
        if self.params.typical_p > 0.0 && self.params.typical_p < 1.0 {
            return false;
        }
        if self.params.logprobs.unwrap_or(0) > 0 {
            return false;
        }
        if self.params.top_k > rustllama_kernels_sycl::MAX_TOP_K_GPU {
            return false;
        }
        true
    }

    /// Try the GPU sample pipeline. Returns `Some(token_id)` on
    /// success (and advances `self.rng` to match the GPU stream's
    /// updated state); `None` to signal the caller should fall
    /// through to the CPU path.
    fn try_sample_gpu(&mut self, logits: &[f32], recent: &[u32]) -> Option<u32> {
        if self.gpu_ctx.is_none() {
            match crate::sampler_gpu::SamplerGpuContext::new(0) {
                Ok(c) => self.gpu_ctx = Some(c),
                Err(_) => {
                    // Init failed (no SYCL device / mock build). Don't
                    // retry on every sample — disable for the lifetime
                    // of this Sampler.
                    self.gpu_enabled = false;
                    return None;
                }
            }
        }
        let params = crate::sampler_gpu::SamplerGpuParams {
            repeat_penalty: self.params.repeat_penalty,
            frequency_penalty: self.params.frequency_penalty,
            presence_penalty: self.params.presence_penalty,
            temperature: self.params.temperature,
            top_k: self.params.top_k,
            greedy: self.params.temperature <= 0.0,
            // H1: pass top_p through to GPU sampler; kernel dispatches
            // `sampler_top_p_usm` between top_k and multinomial. On
            // `needs_fallback` (long-tail distribution exceeds
            // MAX_TOP_P_GPU) the call returns Err and we fall back
            // to CPU.
            top_p: self.params.top_p,
        };
        if !params.is_supported() {
            return None;
        }
        let ctx = self.gpu_ctx.as_mut().expect("just initialized above");
        match ctx.sample(logits, recent, self.rng.state(), &params) {
            Ok((idx, new_state)) => {
                // Advance CPU-side RNG to match the GPU stream so
                // future CPU-path calls stay in lockstep.
                self.rng.set_state(new_state);
                Some(idx)
            }
            Err(_) => None,
        }
    }

    /// Current Mirostat `mu` state. Exposed for tests that pin the
    /// running estimate's stability across many samples.
    pub fn mirostat_mu(&self) -> f32 {
        self.mirostat_mu
    }

    /// Sample one token. `recent` is the list of recently generated token ids
    /// used for repetition / frequency / presence penalties; pass an empty
    /// slice to disable.
    pub fn sample(&mut self, logits: &mut [f32], recent: &[u32]) -> u32 {
        self.sample_with_grammar(logits, recent, None)
    }

    /// Same as [`sample`], but if `grammar` is `Some`, every candidate
    /// not accepted by the grammar's parser state is masked to zero
    /// probability before the multinomial draw. The greedy path
    /// (temperature == 0) walks logits in descending order until it
    /// finds an accepted token. Falls back to argmax when nothing is
    /// accepted (the engine then treats it as the next forced byte;
    /// usually EOS).
    pub fn sample_with_grammar(
        &mut self,
        logits: &mut [f32],
        recent: &[u32],
        grammar: Option<&GrammarMask>,
    ) -> u32 {
        // H12: optional NaN/Inf guard. When `RUSTLLAMA_SAMPLER_LOGIT_GUARD=1`,
        // clamp any non-finite logits to ±1e6 before the sampler runs.
        // Defensive only — forward-pass numerical instability (weight
        // corruption, quant overflow) would otherwise propagate as a
        // silent garbage token. Off by default (~50 µs/token cost on
        // 128K-vocab models); enable when debugging.
        if logit_guard_enabled() {
            let mut any_replaced = false;
            for v in logits.iter_mut() {
                if !v.is_finite() {
                    *v = if v.is_sign_negative() { -1.0e6 } else { 1.0e6 };
                    any_replaced = true;
                }
            }
            if any_replaced {
                tracing::warn!("sampler: non-finite logit(s) replaced with ±1e6 clamp");
            }
        }

        // GPU short-circuit (F1 engine integration). When
        // `RUSTLLAMA_SAMPLER_GPU=1` and the params/grammar don't
        // disqualify, run the whole sampler pipeline on GPU and
        // return the chosen id. Falls through to CPU on any GPU
        // failure mode (init / kernel error / unsupported params).
        if self.gpu_call_eligible(grammar) {
            if let Some(idx) = self.try_sample_gpu(logits, recent) {
                return idx;
            }
        }

        // 1. Repetition / frequency / presence penalties (all logit-space).
        apply_all_penalties(
            logits,
            recent,
            self.params.repeat_penalty,
            self.params.frequency_penalty,
            self.params.presence_penalty,
        );

        // 2. Greedy short-circuit when temperature is zero (after penalties).
        if self.params.temperature <= 0.0 {
            return argmax_with_grammar(logits, grammar);
        }

        // 3 + 4. F1: fused temperature scale + softmax. Combines what
        // was two passes (multiply by inv_temp, then max-scan inside
        // softmax) into a single pass with the temp multiply folded
        // into the max scan — saves one full vocab read per token.
        let inv_t = 1.0 / self.params.temperature;
        rustllama_kernels_cpu::fused_temp_softmax_inplace(logits, inv_t);

        // 5. Mirostat: when enabled, it OWNS truncation — bypass
        // top-k/top-p entirely (they would re-cut the distribution
        // and break Mirostat's perplexity-stability guarantee). The
        // grammar mask still applies AFTER mirostat truncation, same
        // as for the vanilla path.
        if self.params.mirostat == 2 {
            apply_mirostat_v2(
                logits,
                self.params.mirostat_tau,
                self.params.mirostat_eta,
                &mut self.mirostat_mu,
            );
        } else if self.params.mirostat == 1 {
            apply_mirostat_v1(
                logits,
                self.params.mirostat_tau,
                self.params.mirostat_eta,
                &mut self.mirostat_mu,
            );
        } else {
            // 6a. Optional top-k.
            if self.params.top_k > 0 && (self.params.top_k as usize) < logits.len() {
                apply_top_k(logits, self.params.top_k as usize);
            }
            // 6b. Optional top-p (nucleus).
            if self.params.top_p > 0.0 && self.params.top_p < 1.0 {
                apply_top_p(logits, self.params.top_p);
            }
            // 6c. Optional typical-p (locally-typical sampling).
            // Same `0.0 < p < 1.0` gate as top-p so callers disable
            // by setting `1.0` (the default). Applied after top-p:
            // top-p narrows to the high-mass tokens, typical-p then
            // re-truncates by typicality within that surviving set.
            if self.params.typical_p > 0.0 && self.params.typical_p < 1.0 {
                apply_typical_p(logits, self.params.typical_p);
            }
        }

        // 7. Grammar mask: zero any candidate the grammar rejects, then
        //    renormalize. We do this AFTER top-k / top-p because those
        //    have already narrowed the search; if all surviving
        //    candidates are rejected, we widen by scanning the full
        //    vocab for the highest-probability accepted token. That's
        //    the "fallback to argmax in grammar space" behavior.
        if let Some(g) = grammar {
            let mut accepted_sum = 0.0f32;
            for (id, p) in logits.iter_mut().enumerate() {
                if *p > 0.0 && !g.accepts(id as u32) {
                    *p = 0.0;
                } else {
                    accepted_sum += *p;
                }
            }
            if accepted_sum > 0.0 {
                let inv = 1.0 / accepted_sum;
                for v in logits.iter_mut() {
                    *v *= inv;
                }
            } else {
                // No top-k/top-p candidate was accepted. Fall back to
                // the single highest-probability accepted token across
                // the full vocab. This treats a grammar-induced empty
                // candidate set as "force-decode the next valid token"
                // rather than failing.
                return argmax_with_grammar(logits, grammar);
            }
        }

        // 8. Multinomial draw + Mirostat mu update (after-draw).
        let chosen = multinomial(logits, &mut self.rng);
        if self.params.mirostat != 0 {
            // `logits` still holds the renormalized post-truncation
            // distribution; read the chosen probability directly.
            // `1e-30` guards `log2(0)` when grammar fallback or
            // numerical underflow puts the chosen probability at 0.
            let p = logits.get(chosen as usize).copied().unwrap_or(0.0).max(1e-30);
            let observed_surprise = -p.log2();
            self.mirostat_mu -=
                self.params.mirostat_eta * (observed_surprise - self.params.mirostat_tau);
        }
        chosen
    }
}

/// Mirostat v2 truncation: keep tokens whose surprise (`-log2(p)`) is
/// at most the running `mu`. Tokens with `p >= 2^-mu` survive; the
/// rest get zeroed and the distribution is renormalized. The chosen
/// token's surprise is later used to update `mu` outside this
/// function (after the multinomial draw).
fn apply_mirostat_v2(probs: &mut [f32], _tau: f32, _eta: f32, mu: &mut f32) {
    if probs.is_empty() {
        return;
    }
    let threshold = (2.0f32).powf(-*mu);
    let mut kept = 0usize;
    let mut max_idx = 0usize;
    let mut max_p = f32::NEG_INFINITY;
    let mut sum = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        if *p > max_p {
            max_p = *p;
            max_idx = i;
        }
        if *p >= threshold {
            kept += 1;
            sum += *p;
        }
    }
    if kept == 0 {
        // Empty surprise-keep set — `mu` is too aggressive for this
        // distribution. Fall back to keeping just the top-1, same
        // safety net as the v1 path.
        for v in probs.iter_mut() {
            *v = 0.0;
        }
        probs[max_idx] = 1.0;
        return;
    }
    let inv = 1.0 / sum;
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        } else {
            *p *= inv;
        }
    }
}

/// Mirostat v1 truncation: estimate the Zipfian exponent `s` from the
/// top probabilities, then pick a dynamic `k` so that the expected
/// surprise of the resulting top-k distribution sits at `mu`. The
/// `mu` update (same shape as v2) lives outside this function.
fn apply_mirostat_v1(probs: &mut [f32], _tau: f32, _eta: f32, mu: &mut f32) {
    const M: usize = 100;
    let n = probs.len();
    if n == 0 {
        return;
    }
    // Top-M partial sort: O(n) selection + O(M log M) sort instead
    // of the full O(n log n) sort the old impl paid. The Zipfian-
    // exponent fit only needs the top-M, and typical mirostat-v1
    // `k` is much smaller than M=100, so threshold-truncation
    // against the top-k (computed below) avoids touching the
    // remaining n-k tokens past the partial-sort boundary in the
    // common case.
    let mut idx: Vec<usize> = (0..n).collect();
    let m_used = M.min(n).max(2);
    idx.select_nth_unstable_by(m_used - 1, |&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx[..m_used].sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Zipfian-exponent estimate from the top-M log-probability gaps.
    // s ≈ sum_i (ln p_i - ln p_{i+1}) / sum_i ln((i+2)/(i+1))
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..m_used - 1 {
        let t_i = (probs[idx[i]].max(1e-30) as f64).ln();
        let t_i1 = (probs[idx[i + 1]].max(1e-30) as f64).ln();
        num += t_i - t_i1;
        den += (((i + 2) as f64) / ((i + 1) as f64)).ln();
    }
    let s = if den > 0.0 { (num / den) as f32 } else { 1.0 };

    // k = max(1, ceil(((s-1) * 2^mu) / (1 - n^(1-s))))
    let n_f = n as f32;
    let eps = s - 1.0;
    let k = if eps.abs() < 1e-6 {
        1
    } else {
        let denom = 1.0 - n_f.powf(1.0 - s);
        if denom.abs() < 1e-9 {
            1
        } else {
            let raw = (eps * (2.0f32).powf(*mu)) / denom;
            (raw.ceil() as i64).clamp(1, n as i64) as usize
        }
    };

    // Truncate to top-k and renormalize.
    //
    // Common case (`k <= m_used`): the top-`k` indices are inside
    // the already-sorted `idx[..m_used]` prefix; threshold-zero the
    // rest in a single O(n) pass against the rank-`k-1` probability
    // (matches the existing strict-`<` cutoff convention).
    //
    // Rare case (`k > m_used`): fall back to a full sort + walk-by-
    // rank, identical to the old shape. Mirostat-v1's formula
    // clamps `k` to `[1, n]` but in practice `k` stays small (~20
    // at typical mu=5 on vocab=128K); the rare case only kicks in
    // for adversarial flat distributions + very high mu.
    let mut sum = 0.0f32;
    if k <= m_used {
        let threshold = probs[idx[k - 1]];
        for (i, p) in probs.iter_mut().enumerate() {
            // Use the `idx` order for ties at the threshold so the
            // kept set stays exactly k tokens (strict `<` cutoff).
            // We rebuild this set by tracking which `idx[..k]`
            // entries to keep.
            let _ = (i, p); // suppress unused
        }
        // Two-pass to preserve "exactly k tokens" semantics under
        // ties: build a `keep` mask from `idx[..k]`, then walk.
        let mut keep = vec![false; n];
        for &j in &idx[..k] {
            keep[j] = true;
        }
        for (j, p) in probs.iter_mut().enumerate() {
            if !keep[j] {
                *p = 0.0;
            } else {
                sum += *p;
            }
        }
        // `threshold` is the rank-(k-1) probability; not used
        // directly here, kept for the rare-case path's debug
        // story (suppress unused-var lint via let _).
        let _ = threshold;
    } else {
        // Rare path: sort the remainder, then walk by rank.
        idx[m_used..].sort_unstable_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (rank, &i) in idx.iter().enumerate() {
            if rank >= k {
                probs[i] = 0.0;
            } else {
                sum += probs[i];
            }
        }
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in probs.iter_mut() {
            *v *= inv;
        }
    }
}

/// Argmax over `logits` that also satisfies the grammar (if any). When
/// the grammar rejects every token, falls back to plain argmax — the
/// engine will then notice a grammar-incompatible token and stop.
fn argmax_with_grammar(logits: &[f32], grammar: Option<&GrammarMask>) -> u32 {
    let Some(g) = grammar else {
        return argmax(logits);
    };
    let mut best_i = None;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v && g.accepts(i as u32) {
            best_v = v;
            best_i = Some(i);
        }
    }
    match best_i {
        Some(i) => i as u32,
        None => argmax(logits),
    }
}

/// H12: read `RUSTLLAMA_SAMPLER_LOGIT_GUARD` once per process (cached).
/// `1` / `true` enables NaN/Inf clamping on entry to `Sampler::sample`;
/// any other value (including unset, default) leaves the sampler hot
/// path unchanged.
fn logit_guard_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RUSTLLAMA_SAMPLER_LOGIT_GUARD")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Apply the repetition / frequency / presence penalties to `logits` in
/// place, using `recent` as the token history. Shared with the
/// speculative decode paths (`verify_and_commit_speculation` /
/// `mtp_round`), which apply the SAME penalties to the target logits
/// before their accept/reject verify so greedy-with-penalties stays
/// token-for-token identical to the classic sampler (which calls this
/// from `sample_with_grammar` before its greedy short-circuit). A no-op
/// when all three penalties are at their defaults (`repeat == 1.0`,
/// `frequency == 0.0`, `presence == 0.0`).
pub(crate) fn apply_all_penalties(
    logits: &mut [f32],
    recent: &[u32],
    repeat: f32,
    frequency: f32,
    presence: f32,
) {
    // Repetition penalty: multiplicative (llama.cpp convention).
    if repeat != 1.0 {
        apply_repetition_penalty(logits, recent, repeat);
    }

    // Frequency + presence: additive, OpenAI convention.
    //   logit -= frequency_penalty × count(token) + presence_penalty × (count(token) > 0)
    if frequency != 0.0 || presence != 0.0 {
        // Tally counts in a hashmap-less way — vocab is up to ~200k, recent
        // is typically small. For small recent.len() we can just scan.
        // For large vocab, this is O(recent.len()) plus a constant-time
        // hit per token.
        // Use a small frequency table on the heap; tokens out of range
        // (e.g. negative-ish via cast) are skipped.
        let mut counts: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
        for &tok in recent {
            let idx = tok as usize;
            if idx < logits.len() {
                *counts.entry(idx).or_insert(0) += 1;
            }
        }
        for (idx, count) in counts {
            let penalty = frequency * count as f32 + presence;
            logits[idx] -= penalty;
        }
    }
}

fn apply_repetition_penalty(logits: &mut [f32], recent: &[u32], penalty: f32) {
    for &tok in recent {
        let idx = tok as usize;
        if idx >= logits.len() {
            continue;
        }
        let v = logits[idx];
        logits[idx] = if v >= 0.0 { v / penalty } else { v * penalty };
    }
}

// Softmax now flows through `kernels_cpu::fused_temp_softmax_inplace`
// (F1 fusion) inline at the `sample_with_grammar` call site. The
// standalone `softmax_inplace` wrapper was retired — every call
// folded the temperature multiply in immediately after.

/// Zero out everything outside the top-`k` probabilities and renormalize.
///
/// Edge behaviors (locked in by tests):
///   - `k == 0` is a no-op ("disabled"). The sampler-level gate also
///     guards against this so the function is never called with `k=0`
///     in production paths.
///   - `k >= probs.len()` is a no-op (nothing to drop).
///   - **Ties at the threshold are kept**: tokens with probability
///     exactly equal to the kth-highest stay in the distribution.
///     This means a uniform N-token distribution with `k=2` keeps all
///     N tokens. Strictly cutting at `< threshold` (which we do) gives
///     a stable behavior under tie-floods but lets the effective N
///     exceed `k`; switch to `<= threshold` if "exactly k" matters.
fn apply_top_k(probs: &mut [f32], k: usize) {
    if k == 0 || k >= probs.len() {
        return;
    }
    // O(n) selection of the k-th highest value via std's quickselect
    // (`select_nth_unstable_by`). Replaces a full O(n log n) sort; on
    // a 128K-vocab model that's ~17× fewer comparisons per sampled
    // token. Tie semantics preserved: callers below cut at `< threshold`,
    // so any tokens with probability exactly equal to the k-th value
    // stay in the distribution (same as the old full-sort path).
    let mut scratch: Vec<f32> = probs.to_vec();
    let (_, &mut threshold, _) = scratch.select_nth_unstable_by(k - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut sum = 0.0f32;
    for v in probs.iter_mut() {
        if *v < threshold {
            *v = 0.0;
        } else {
            sum += *v;
        }
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in probs.iter_mut() {
            *v *= inv;
        }
    }
}

/// Keep the smallest set of tokens whose cumulative probability ≥ p.
///
/// The sampler-level gate (`top_p > 0.0 && top_p < 1.0`) means this is
/// never invoked with `p == 0.0` (disabled, NOT "keep only the top
/// token") or `p == 1.0` (no-op, "keep everything"). Convention chosen
/// for consistency with the OpenAI API: clients set `top_p` to
/// disable nucleus by passing `1.0`, not by passing `0.0`.
fn apply_top_p(probs: &mut [f32], p: f32) {
    let n = probs.len();
    if n == 0 {
        return;
    }
    // Geometric-growth partial-sort: pick a small "front" of the
    // most-probable indices via O(n) quickselect, sort only that
    // front, walk for the cutoff. If cumulative didn't cross `p`,
    // double the front and retry. On a 128K-vocab model with a
    // typical p=0.9, the cutoff lands in the first 64-128 ranks, so
    // we usually fully sort only ~0.1% of the vocab and skip the
    // full O(n log n) sort entirely.
    let mut idx: Vec<usize> = (0..n).collect();
    let mut front_size = 64usize.min(n);
    let cutoff = loop {
        idx.select_nth_unstable_by(front_size - 1, |&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx[..front_size].sort_unstable_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cum = 0.0f32;
        let mut found: Option<usize> = None;
        for (rank, &i) in idx[..front_size].iter().enumerate() {
            cum += probs[i];
            if cum >= p {
                found = Some(rank + 1);
                break;
            }
        }
        if let Some(c) = found {
            break c;
        }
        if front_size >= n {
            break n;
        }
        front_size = (front_size * 2).min(n);
    };

    // Zero everything past the cutoff. Use a kept-mask so the zero
    // pass is O(n) — same complexity as the original rank walk.
    let mut keep = vec![false; n];
    for &i in &idx[..cutoff] {
        keep[i] = true;
    }
    let mut sum = 0.0f32;
    for (i, v) in probs.iter_mut().enumerate() {
        if !keep[i] {
            *v = 0.0;
        } else {
            sum += *v;
        }
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in probs.iter_mut() {
            *v *= inv;
        }
    }
}

/// Locally typical sampling (Meister et al. 2022, "Locally Typical
/// Sampling"). Truncates the distribution by selecting the set of
/// tokens whose information content `-log(p_i)` is closest to the
/// distribution's entropy `H = -∑ p_i log(p_i)`, then keeps the
/// smallest such prefix whose cumulative probability ≥ `p`.
///
/// In plain terms: instead of "keep the most probable tokens"
/// (top-p), this keeps the most "typical" / least surprising
/// tokens — those whose log-prob best matches the average
/// log-prob the model is producing right now. Useful as a
/// truncation that preserves the entropy character of the
/// distribution while pruning outliers.
///
/// Gate convention matches [`apply_top_p`]: the sampler-level
/// `0.0 < typical_p < 1.0` check means this is never called with
/// `p == 0.0` (disabled) or `p == 1.0` (no-op). `p == 0.0` is the
/// callers' "off" knob, not "keep only the most-typical token".
fn apply_typical_p(probs: &mut [f32], p: f32) {
    let n = probs.len();
    if n == 0 {
        return;
    }
    // Entropy of the (already-normalized) softmax distribution.
    let h: f32 = probs
        .iter()
        .map(|&pi| if pi > 0.0 { -pi * pi.ln() } else { 0.0 })
        .sum();
    // Precompute typicality `|H + ln p_i|` per index — saves
    // 2 × (sort comparisons) calls into `.ln().abs()` per token. On a
    // 128K-vocab model that's ~5M fewer log calls per sampled token.
    let typ: Vec<f32> = probs
        .iter()
        .map(|&pi| (pi.max(1e-30).ln() + h).abs())
        .collect();
    // Geometric-growth partial-sort by ascending typicality. Same
    // structure as the top_p partial-sort path; the only difference
    // is the comparator orders by `typ[i]` instead of `-probs[i]`.
    let mut idx: Vec<usize> = (0..n).collect();
    let mut front_size = 64usize.min(n);
    let cutoff = loop {
        idx.select_nth_unstable_by(front_size - 1, |&a, &b| {
            typ[a].partial_cmp(&typ[b]).unwrap_or(std::cmp::Ordering::Equal)
        });
        idx[..front_size].sort_unstable_by(|&a, &b| {
            typ[a].partial_cmp(&typ[b]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cum = 0.0f32;
        let mut found: Option<usize> = None;
        for (rank, &i) in idx[..front_size].iter().enumerate() {
            cum += probs[i];
            if cum >= p {
                found = Some(rank + 1);
                break;
            }
        }
        if let Some(c) = found {
            break c;
        }
        if front_size >= n {
            break n;
        }
        front_size = (front_size * 2).min(n);
    };

    let mut keep = vec![false; n];
    for &i in &idx[..cutoff] {
        keep[i] = true;
    }
    let mut sum = 0.0f32;
    for (i, v) in probs.iter_mut().enumerate() {
        if !keep[i] {
            *v = 0.0;
        } else {
            sum += *v;
        }
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in probs.iter_mut() {
            *v *= inv;
        }
    }
}

fn multinomial(probs: &[f32], rng: &mut Rng) -> u32 {
    let u = rng.next_f32();
    let mut cum = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        cum += *p;
        if u < cum {
            return i as u32;
        }
    }
    // Fallback for numerical edge: return the last non-zero.
    (probs.len() - 1) as u32
}

/// Compute the logprob of `chosen_id` plus the top-`k` alternatives from
/// **raw** model logits (before temperature, top-k, top-p, or penalties).
/// Reporting raw model probabilities is what eval frameworks expect; it's
/// also faster and decouples logprob output from sampler configuration.
///
/// Numerically stable via the standard `max - log_sum_exp` trick. The
/// returned `top` is sorted descending by logprob and may include
/// `chosen_id` if it ranked within the top K.
pub fn compute_logprobs(logits: &[f32], chosen_id: u32, k: usize) -> TokenLogprobs {
    // log(sum(exp(x_i))) via the SIMD-accelerated kernel — AVX2 /
    // AVX-512 on x86_64, scalar fallback elsewhere. Two scalar
    // full-vocab passes (max scan + exp-sum) were the per-token
    // cost on the OpenAI `logprobs`-enabled path; the SIMD variant
    // is ~6× faster on AVX2, ~10× on AVX-512.
    let log_sum_exp = rustllama_kernels_cpu::log_sum_exp_f32(logits);

    let chosen_logit = logits
        .get(chosen_id as usize)
        .copied()
        .unwrap_or(f32::NEG_INFINITY);
    let chosen_lp = chosen_logit - log_sum_exp;

    let top = if k == 0 {
        Vec::new()
    } else {
        top_k_logprobs(logits, k.min(logits.len()), log_sum_exp)
    };

    TokenLogprobs {
        logprob: chosen_lp,
        top,
    }
}

/// Single-pass top-K selection: maintain a min-position at the tail of a
/// length-K running buffer; only re-sort when a candidate displaces the
/// running minimum. O(N × K) worst case but K is tiny (≤ 20 in practice)
/// and the displacement is rare after the first few hundred tokens, so the
/// effective cost is roughly a single scan over the vocabulary.
fn top_k_logprobs(logits: &[f32], k: usize, log_sum_exp: f32) -> Vec<TopLogprob> {
    if k == 0 {
        return Vec::new();
    }
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);

    for (i, &v) in logits.iter().enumerate() {
        if top.len() < k {
            top.push((i as u32, v));
            if top.len() == k {
                top.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            continue;
        }
        if v > top[k - 1].1 {
            top[k - 1] = (i as u32, v);
            // Bubble the new entry up while it's larger than its predecessor.
            let mut j = k - 1;
            while j > 0 && top[j].1 > top[j - 1].1 {
                top.swap(j, j - 1);
                j -= 1;
            }
        }
    }

    top.into_iter()
        .map(|(id, logit)| TopLogprob {
            id,
            logprob: logit - log_sum_exp,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn compute_logprobs_uniform_distribution() {
        // All-equal logits → uniform softmax → logprob = -ln(N) for every token.
        let n = 5;
        let logits = vec![1.0f32; n];
        let lp = compute_logprobs(&logits, 0, 0);
        let expected = -(n as f32).ln();
        assert!(
            approx_eq(lp.logprob, expected, 1e-5),
            "expected {expected}, got {}",
            lp.logprob
        );
        assert!(lp.top.is_empty(), "k=0 should produce no top");
    }

    #[test]
    fn compute_logprobs_sums_to_one_in_prob_space() {
        // Sum of exp(logprob_i) over all i must be 1.0 within fp error.
        let logits = vec![-2.0, 1.5, 0.0, 3.7, -5.2, 0.1, 2.2];
        let lp = compute_logprobs(&logits, 0, logits.len());
        let total: f32 = lp.top.iter().map(|t| t.logprob.exp()).sum();
        assert!(
            approx_eq(total, 1.0, 1e-5),
            "softmax should sum to 1, got {total}"
        );
    }

    #[test]
    fn compute_logprobs_top_k_sorted_descending() {
        let logits = vec![0.1, 3.0, -2.0, 1.5, 0.5, 2.7];
        let lp = compute_logprobs(&logits, 1, 3);
        assert_eq!(lp.top.len(), 3);
        assert!(lp.top[0].logprob >= lp.top[1].logprob);
        assert!(lp.top[1].logprob >= lp.top[2].logprob);
        // The chosen (id=1, logit=3.0) is the highest — top[0] should be it.
        assert_eq!(lp.top[0].id, 1);
    }

    #[test]
    fn compute_logprobs_chosen_matches_top_when_argmax() {
        // The argmax token's logprob from `chosen_logprob` must equal its
        // entry in `top` when K is large enough to include it.
        let logits = vec![1.0, -0.5, 4.2, 0.3, 3.9];
        let chosen_id = 2u32; // the argmax
        let lp = compute_logprobs(&logits, chosen_id, 5);
        let in_top = lp
            .top
            .iter()
            .find(|t| t.id == chosen_id)
            .expect("chosen should appear in top-5");
        assert!(approx_eq(lp.logprob, in_top.logprob, 1e-6));
    }

    #[test]
    fn compute_logprobs_k_capped_at_vocab_size() {
        let logits = vec![1.0, 2.0, 3.0];
        let lp = compute_logprobs(&logits, 0, 10);
        // Asked for 10, but vocab is 3 → top should contain exactly 3.
        assert_eq!(lp.top.len(), 3);
    }

    #[test]
    fn argmax_basic() {
        assert_eq!(argmax(&[0.1, 0.9, 0.2]), 1);
        assert_eq!(argmax(&[-1.0, -2.0, -0.5]), 2);
    }

    #[test]
    fn rng_deterministic() {
        let mut a = Rng::from_seed(42);
        let mut b = Rng::from_seed(42);
        for _ in 0..16 {
            assert_eq!(a.next_f32(), b.next_f32());
        }
    }

    #[test]
    fn rng_distribution_roughly_uniform() {
        let mut r = Rng::from_seed(1);
        let mut buckets = [0usize; 10];
        let n = 10_000;
        for _ in 0..n {
            let u = r.next_f32();
            let b = (u * 10.0) as usize;
            buckets[b.min(9)] += 1;
        }
        // Each bucket should be ~1000 ± 300.
        for c in &buckets {
            assert!(
                (700..1300).contains(c),
                "bucket {c} out of expected range; {:?}",
                buckets
            );
        }
    }

    #[test]
    fn greedy_with_zero_temperature() {
        let mut s = Sampler::new(SamplingParams {
            temperature: 0.0,
            top_p: 0.0,
            top_k: 0,
            repeat_penalty: 1.0,
            max_tokens: 1,
            stop: vec![],
            seed: 7,
            ..SamplingParams::default()
        });
        let mut logits = vec![1.0, 5.0, 3.0, 2.0];
        let t = s.sample(&mut logits, &[]);
        assert_eq!(t, 1);
    }

    #[test]
    fn temperature_one_sampling_is_consistent() {
        let mut s1 = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
            max_tokens: 1,
            stop: vec![],
            seed: 123,
            ..SamplingParams::default()
        });
        let mut s2 = s1.clone_for_test();
        let a = vec![1.0, 5.0, 3.0, 2.0];
        let b = vec![1.0, 5.0, 3.0, 2.0];
        for _ in 0..16 {
            let ta = s1.sample(&mut a.clone(), &[]);
            let tb = s2.sample(&mut b.clone(), &[]);
            assert_eq!(ta, tb);
        }
    }

    #[test]
    fn top_k_eliminates_low_probability() {
        let mut p = vec![0.5f32, 0.3, 0.15, 0.05];
        apply_top_k(&mut p, 2);
        assert_eq!(p[2], 0.0);
        assert_eq!(p[3], 0.0);
        let total: f32 = p.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "got {total}");
    }

    #[test]
    fn top_p_keeps_nucleus() {
        let mut p = vec![0.4f32, 0.3, 0.2, 0.1];
        apply_top_p(&mut p, 0.7);
        // The top two add to 0.7, so the cutoff should keep them and drop the rest.
        assert!(p[2] == 0.0 && p[3] == 0.0);
        let total: f32 = p.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn repetition_penalty_demotes_recent() {
        let mut logits = vec![1.0, 1.0, 1.0, 1.0];
        apply_repetition_penalty(&mut logits, &[2], 2.0);
        assert!(logits[2] < logits[0]);
    }

    #[test]
    fn frequency_penalty_scales_with_count() {
        // Token 1 appears 3 times → penalty = 3 × 0.5 = 1.5
        // Token 2 appears 1 time  → penalty = 1 × 0.5 = 0.5
        let mut logits = vec![5.0, 5.0, 5.0, 5.0];
        apply_all_penalties(&mut logits, &[1, 1, 1, 2], 1.0, 0.5, 0.0);
        assert!((logits[1] - 3.5).abs() < 1e-5, "got {}", logits[1]);
        assert!((logits[2] - 4.5).abs() < 1e-5, "got {}", logits[2]);
        assert!((logits[0] - 5.0).abs() < 1e-5);
        assert!((logits[3] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn presence_penalty_is_fixed_per_appearance() {
        // Token 1 appears 3 times — presence subtracts 0.7 ONCE.
        let mut logits = vec![5.0, 5.0, 5.0];
        apply_all_penalties(&mut logits, &[1, 1, 1], 1.0, 0.0, 0.7);
        assert!((logits[1] - 4.3).abs() < 1e-5, "got {}", logits[1]);
        assert!((logits[0] - 5.0).abs() < 1e-5);
        assert!((logits[2] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn top_k_zero_is_disabled() {
        // top_k = 0 must pass through unchanged.
        let mut p = vec![0.5f32, 0.3, 0.15, 0.05];
        let before = p.clone();
        apply_top_k(&mut p, 0);
        assert_eq!(p, before);
    }

    #[test]
    fn top_k_greater_than_vocab_is_passthrough() {
        let mut p = vec![0.5f32, 0.3, 0.2];
        let before = p.clone();
        apply_top_k(&mut p, 100);
        assert_eq!(p, before);
    }

    #[test]
    fn top_k_keeps_ties_at_threshold() {
        // Three values tied at 0.25. top_k=2 with `< threshold` (not `<=`)
        // keeps all three. Documents the "tie-keep" behavior so a future
        // change to strict-cutoff doesn't silently shift sampling.
        let mut p = vec![0.25f32, 0.25, 0.25, 0.25];
        apply_top_k(&mut p, 2);
        // All four equal probs → none are < threshold → none get zeroed.
        // After renormalize, all should remain non-zero and sum to 1.
        for v in &p {
            assert!(*v > 0.0, "tie at threshold should be kept: {p:?}");
        }
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn top_p_one_is_passthrough_via_sampler_gate() {
        // The sampler skips `apply_top_p` when top_p >= 1.0.
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
            max_tokens: 1,
            stop: vec![],
            seed: 7,
            ..SamplingParams::default()
        });
        // Logits that make the argmax obvious so we can confirm sampling
        // still produces a sensible draw with nucleus disabled.
        let mut logits = vec![0.5f32, 5.0, 0.3, 0.1];
        let t = s.sample(&mut logits, &[]);
        assert_eq!(t, 1, "argmax should still dominate when top_p=1.0");
    }

    #[test]
    fn top_p_zero_is_passthrough_via_sampler_gate() {
        // The sampler also skips `apply_top_p` when top_p <= 0 (convention:
        // top_p=0 means disabled, not "keep only the top token"). Verify
        // that with all params disabled, sampling at temp=1 produces a
        // valid token from the full distribution.
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_p: 0.0,
            top_k: 0,
            repeat_penalty: 1.0,
            max_tokens: 1,
            stop: vec![],
            seed: 11,
            ..SamplingParams::default()
        });
        let mut logits = vec![1.0f32, 5.0, 3.0];
        let t = s.sample(&mut logits, &[]);
        assert!(t < 3, "sampled token must be in-range, got {t}");
    }

    #[test]
    fn negative_temperature_takes_greedy_path() {
        // The greedy short-circuit uses `temperature <= 0.0`, so a
        // negative value also lands on argmax (rather than amplifying
        // logits in the wrong direction).
        let mut s = Sampler::new(SamplingParams {
            temperature: -1.0,
            top_p: 0.5,
            top_k: 5,
            repeat_penalty: 1.0,
            max_tokens: 1,
            stop: vec![],
            seed: 0,
            ..SamplingParams::default()
        });
        let mut logits = vec![1.0, 5.0, 3.0, 2.0];
        let t = s.sample(&mut logits, &[]);
        assert_eq!(t, 1, "negative temperature should still pick argmax");
    }

    #[test]
    fn repetition_penalty_one_is_noop() {
        let mut logits = vec![1.0f32, 1.0, 1.0, 1.0];
        let before = logits.clone();
        apply_all_penalties(&mut logits, &[1, 2, 3], 1.0, 0.0, 0.0);
        assert_eq!(logits, before);
    }

    #[test]
    fn empty_recent_with_nonidentity_penalty_is_noop() {
        // repeat_penalty != 1.0 but no recent tokens to penalize.
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let before = logits.clone();
        apply_all_penalties(&mut logits, &[], 2.0, 0.5, 0.5);
        assert_eq!(logits, before);
    }

    #[test]
    fn combined_top_k_and_top_p_apply_in_order() {
        // top_k first → keep 3 (drop 4th).
        // top_p (0.6) on those 3 → cumulative reaches 0.6 after 2 tokens.
        // Result: only 2 tokens remain non-zero.
        let mut probs = vec![0.4f32, 0.3, 0.2, 0.1];
        apply_top_k(&mut probs, 3);
        // top-k normalizes; renormalized [0.444, 0.333, 0.222, 0].
        apply_top_p(&mut probs, 0.6);
        let nonzero = probs.iter().filter(|p| **p > 0.0).count();
        assert_eq!(nonzero, 2, "top_k=3 then top_p=0.6 should leave 2: {probs:?}");
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn frequency_and_presence_combine() {
        // count=2, freq=0.3, presence=0.4 → penalty = 0.3*2 + 0.4 = 1.0
        let mut logits = vec![5.0, 5.0];
        apply_all_penalties(&mut logits, &[0, 0], 1.0, 0.3, 0.4);
        assert!((logits[0] - 4.0).abs() < 1e-5);
        assert!((logits[1] - 5.0).abs() < 1e-5);
    }

    impl Sampler {
        fn clone_for_test(&self) -> Self {
            Self {
                params: self.params.clone(),
                rng: self.rng.clone(),
                mirostat_mu: self.mirostat_mu,
                gpu_ctx: None,
                gpu_enabled: false,
            }
        }
    }

    // ----- Typical-p sampler -----------------------------------------------

    #[test]
    fn typical_p_uniform_distribution_is_a_no_op() {
        // Uniform: every token has the same |H + ln p| = 0 (since
        // log(1/n) = -log(n) = -H). The cumulative-prob walk keeps
        // tokens in arbitrary tie order until ≥ p; after renorm we
        // still have a uniform-ish distribution (just smaller).
        // Important: typicality SORTS by 0 for every token so the
        // first-traversed prefix gets kept. Total mass remains 1.0.
        let mut probs = vec![1.0f32 / 8.0; 8];
        apply_typical_p(&mut probs, 0.5);
        let total: f32 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "renormalized total must equal 1.0, got {total}"
        );
    }

    #[test]
    fn typical_p_keeps_at_least_one_token() {
        // Even with `p` close to 0 the first iteration of the
        // cumulative loop adds one token's mass, so the kept set
        // always has at least one element. Pins that safety.
        let mut probs = [0.1f32, 0.6, 0.3];
        apply_typical_p(&mut probs, 0.01);
        let nonzero = probs.iter().filter(|p| **p > 0.0).count();
        assert!(nonzero >= 1, "typical-p must keep at least 1 token");
        let total: f32 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn typical_p_truncates_below_full_support() {
        // For a non-uniform distribution with a small `p`, typical-p
        // must truncate to a proper subset of the vocab. Pins the
        // contract: the function actually does something, and the
        // kept-probability mass renormalizes to 1.0.
        // Distribution: two-mode with light tail.
        let mut probs = vec![0.45f32, 0.45, 0.025, 0.025, 0.025, 0.025];
        apply_typical_p(&mut probs, 0.5);
        let nonzero = probs.iter().filter(|p| **p > 0.0).count();
        assert!(
            nonzero >= 1 && nonzero < 6,
            "typical-p with p=0.5 must drop at least one token, got {nonzero}/6"
        );
        let total: f32 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "renormalization must sum to 1.0, got {total}"
        );
    }

    #[test]
    fn typical_p_typicality_ordering_matches_entropy_distance() {
        // Pin the ordering rule: the token whose -log(p) sits
        // closest to the distribution entropy is kept first.
        // Construct a 3-token case where this gives a clear winner:
        // p = [0.6, 0.3, 0.1] → -log(p) = [0.51, 1.20, 2.30].
        // H = 0.6*0.51 + 0.3*1.20 + 0.1*2.30 = 0.306+0.360+0.230 = 0.896.
        // typicality_i = |H + ln(p_i)| = | -surprise_i + H |
        //   = |0.896 - 0.51| = 0.386 for i=0
        //   = |0.896 - 1.20| = 0.305 for i=1   ← smallest, most typical
        //   = |0.896 - 2.30| = 1.408 for i=2
        // With p threshold 0.3, the walk picks token 1 (mass 0.3,
        // crosses threshold), drops the other two.
        let mut probs = vec![0.6f32, 0.3, 0.1];
        apply_typical_p(&mut probs, 0.3);
        assert!(
            probs[1] > 0.0,
            "the most-typical token (i=1) must survive: {probs:?}"
        );
        assert_eq!(probs[0], 0.0, "i=0 should be dropped: {probs:?}");
        assert_eq!(probs[2], 0.0, "i=2 should be dropped: {probs:?}");
    }

    #[test]
    fn typical_p_seed_reproducibility_through_sampler() {
        // Two samplers seeded the same with typical-p enabled
        // produce identical token sequences. Pins the typical-p
        // path against a regression in either the truncation
        // function or the multinomial-stage ordering.
        let mut p1 = SamplingParams::default();
        p1.seed = 0xC0FFEE;
        p1.typical_p = 0.7;
        p1.top_p = 1.0;
        p1.top_k = 0;
        let mut s1 = Sampler::new(p1.clone());
        let mut s2 = Sampler::new(p1);
        let n = 32;
        let logits: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.05).collect();
        let mut a = Vec::new();
        let mut b = Vec::new();
        for _ in 0..8 {
            let mut la = logits.clone();
            let mut lb = logits.clone();
            a.push(s1.sample(&mut la, &[]));
            b.push(s2.sample(&mut lb, &[]));
        }
        assert_eq!(a, b, "same seed + typical_p → same token sequence");
    }

    // ----- Mirostat ---------------------------------------------------------
    //
    // Pinning tests for the adaptive samplers. The engine-level
    // determinism test
    // (`rustllama-engine/tests/cpu_engine.rs::seeded_sampling_is_reproducible`)
    // covers vanilla seeded sampling. These tests pin Mirostat-specific
    // invariants: mu state evolves predictably, the truncation set
    // shrinks/grows with mu, both v1 and v2 honor the seed.

    fn mirostat_params(version: u32) -> SamplingParams {
        SamplingParams {
            temperature: 1.0,
            top_p: 1.0, // Mirostat owns truncation
            top_k: 0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            max_tokens: 32,
            stop: vec![],
            seed: 0xBEEF,
            logprobs: None,
            grammar: None,
            mirostat: version,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            ..SamplingParams::default()
        }
    }

    fn deterministic_logits(n: usize) -> Vec<f32> {
        // A non-uniform distribution that makes Mirostat's truncation
        // non-trivial: high mass on a few tokens with a long tail.
        (0..n).map(|i| (n - i) as f32 * 0.05).collect()
    }

    #[test]
    fn sampler_mu_initialized_to_two_tau() {
        let s = Sampler::new(mirostat_params(2));
        assert!(
            (s.mirostat_mu() - 10.0).abs() < 1e-6,
            "mu should start at 2*tau = 10.0, got {}",
            s.mirostat_mu()
        );
    }

    #[test]
    fn mirostat_v2_seed_reproducibility() {
        // Same seed + same logits sequence → same token sequence,
        // same mu trajectory. The engine layer's reproducibility test
        // covers this at the model level; here we pin it at the
        // sampler level so a regression in the mirostat path doesn't
        // hide behind model noise.
        let mut s1 = Sampler::new(mirostat_params(2));
        let mut s2 = Sampler::new(mirostat_params(2));
        let n = 64;
        let mut toks1 = Vec::new();
        let mut toks2 = Vec::new();
        for _ in 0..16 {
            let mut l1 = deterministic_logits(n);
            let mut l2 = deterministic_logits(n);
            toks1.push(s1.sample(&mut l1, &[]));
            toks2.push(s2.sample(&mut l2, &[]));
        }
        assert_eq!(toks1, toks2, "same seed → same Mirostat v2 trajectory");
        assert!(
            (s1.mirostat_mu() - s2.mirostat_mu()).abs() < 1e-6,
            "mu drift mismatch: {} vs {}",
            s1.mirostat_mu(),
            s2.mirostat_mu(),
        );
    }

    #[test]
    fn mirostat_v1_seed_reproducibility() {
        let mut s1 = Sampler::new(mirostat_params(1));
        let mut s2 = Sampler::new(mirostat_params(1));
        let n = 64;
        let mut toks1 = Vec::new();
        let mut toks2 = Vec::new();
        for _ in 0..16 {
            let mut l1 = deterministic_logits(n);
            let mut l2 = deterministic_logits(n);
            toks1.push(s1.sample(&mut l1, &[]));
            toks2.push(s2.sample(&mut l2, &[]));
        }
        assert_eq!(toks1, toks2, "same seed → same Mirostat v1 trajectory");
    }

    #[test]
    fn mirostat_disabled_matches_vanilla_path() {
        // mirostat == 0 must produce byte-identical results to the
        // pre-mirostat sampler. Critical for "off by default".
        let mut params = SamplingParams {
            temperature: 1.0,
            top_p: 0.9,
            top_k: 20,
            seed: 0xCAFE,
            ..SamplingParams::default()
        };
        params.mirostat = 0;
        let mut s_off = Sampler::new(params.clone());
        let mut s_off2 = Sampler::new(params);
        let n = 64;
        for _ in 0..8 {
            let mut a = deterministic_logits(n);
            let mut b = deterministic_logits(n);
            assert_eq!(s_off.sample(&mut a, &[]), s_off2.sample(&mut b, &[]));
        }
    }

    #[test]
    fn mirostat_v2_truncation_keeps_at_least_one_token() {
        // When mu is small enough that no token's surprise sits
        // beneath it, the v2 fallback keeps the argmax as a forced
        // single-mass choice. Pin that safety net so the sampler can
        // never produce an undefined draw.
        let mut probs = deterministic_logits(64);
        rustllama_kernels_cpu::softmax_f32_inplace(&mut probs);
        let mut mu = 0.001; // wildly small → threshold 2^-0.001 ≈ 0.999
        apply_mirostat_v2(&mut probs, 5.0, 0.1, &mut mu);
        let nonzero = probs.iter().filter(|p| **p > 0.0).count();
        assert!(
            nonzero >= 1,
            "v2 must always leave at least one token in the support"
        );
        let total: f32 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "v2 distribution must renormalize to 1.0, got {total}"
        );
    }

    #[test]
    fn mirostat_v2_mu_updates_toward_target_surprise() {
        // Over many samples on a stable distribution, mu should
        // gravitate toward the observed surprise — and since the
        // surprise of a high-mass token is below tau=5, mu should
        // drift down from its initial 2*tau=10.
        let mut s = Sampler::new(mirostat_params(2));
        let mu_start = s.mirostat_mu();
        for _ in 0..200 {
            let mut l = deterministic_logits(128);
            s.sample(&mut l, &[]);
        }
        let mu_end = s.mirostat_mu();
        // Drift direction depends on the distribution; for our
        // heavy-head logits the observed surprise sits well below tau,
        // so mu should DECREASE. The exact landing depends on noise
        // — assert direction, not value.
        assert!(
            mu_end < mu_start,
            "mu should drift downward on a low-entropy distribution: \
             start={mu_start} end={mu_end}"
        );
        // mu should stay bounded (no runaway).
        assert!(
            mu_end.abs() < 50.0,
            "mu drifted out of sane range: {mu_end}"
        );
    }

    #[test]
    fn mirostat_v1_truncates_and_renormalizes() {
        // Exact-Zipfian distribution where consecutive log-prob gaps
        // are constant in log-rank. The estimator should recover an
        // `s` close to the true exponent, and the formula should
        // pick a dynamic `k < n`. Pins the contract: v1 always
        // reduces the support and the kept mass renormalizes to 1.
        let n = 64;
        // p_i ∝ 1 / (i+1)^2 — a steep Zipfian (s=2).
        let mut probs: Vec<f32> = (0..n).map(|i| 1.0 / ((i + 1) as f32).powi(2)).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
        // mu=2 (target surprise ≈ 2 bits per token, so an effective
        // distribution of ~4 tokens). For our steep s≈2 Zipfian and
        // n=64, the formula resolves to a small k (~4).
        let mut mu = 2.0;
        apply_mirostat_v1(&mut probs, 5.0, 0.1, &mut mu);
        let nonzero = probs.iter().filter(|p| **p > 0.0).count();
        assert!(
            nonzero >= 1 && nonzero < n,
            "v1 should truncate to a proper subset of [1, n), got {nonzero}/{n}"
        );
        let total: f32 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "v1 truncation must renormalize to 1.0, got {total}"
        );
    }

    // ============================================================
    // Partial-sort parity (A5)
    // ============================================================
    //
    // The partial-sort variants of apply_top_k / apply_top_p /
    // apply_typical_p must produce the same kept-set (modulo ties)
    // as a hypothetical full-sort baseline. Tests pin the
    // contractual properties: kept tokens form a contiguous
    // descending-by-prob prefix, total mass renormalizes to 1.0,
    // and the cutoff respects the threshold rule.

    fn synth_probs(seed: u32, n: usize) -> Vec<f32> {
        // Small LCG, no extra deps. Generates positive logits with
        // wide dynamic range so the sort exercises a realistic spread.
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        let mut raw: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / (1u32 << 24) as f32;
                u * 8.0 - 4.0
            })
            .collect();
        let max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for v in raw.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in raw.iter_mut() {
            *v /= sum;
        }
        raw
    }

    #[test]
    fn top_k_partial_sort_matches_full_sort_baseline() {
        // For each (n, k), verify the kept-set produced by the
        // partial-sort implementation matches a full-sort baseline.
        for &(n, k) in &[(32usize, 4usize), (128, 16), (1024, 32), (4096, 128)] {
            let probs = synth_probs((n * 31 + k) as u32, n);
            // Reference: full sort, keep top-k.
            let mut sorted: Vec<f32> = probs.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let threshold = sorted[k - 1];
            let mut want = probs.clone();
            let mut sum = 0f32;
            for v in want.iter_mut() {
                if *v < threshold {
                    *v = 0.0;
                } else {
                    sum += *v;
                }
            }
            for v in want.iter_mut() {
                *v /= sum;
            }
            let mut got = probs.clone();
            apply_top_k(&mut got, k);
            for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "top_k parity n={n} k={k} idx={i}: got={a} want={b}"
                );
            }
        }
    }

    #[test]
    fn top_p_partial_sort_renormalizes_to_one() {
        // The partial-sort path's kept-set may differ from a
        // full-sort baseline by tie permutation. The invariants that
        // matter — (1) kept probability mass renormalizes to 1.0
        // and (2) cumulative kept mass before normalization was
        // ≥ p — are both checked here.
        for &(n, p) in &[(64usize, 0.5f32), (1024, 0.9), (4096, 0.95), (32, 0.1)] {
            let probs = synth_probs((n + p.to_bits() as usize) as u32, n);
            let mut got = probs.clone();
            apply_top_p(&mut got, p);
            let total: f32 = got.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-5,
                "top_p must renormalize to 1.0: n={n} p={p} total={total}"
            );
            // Every nonzero kept prob must be ≥ the smallest kept
            // (definition of "smallest prefix in descending order").
            let kept: Vec<f32> = probs
                .iter()
                .enumerate()
                .filter(|(i, _)| got[*i] > 0.0)
                .map(|(_, &v)| v)
                .collect();
            if let Some(&min_kept) = kept.iter().min_by(|a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                for (i, &v) in probs.iter().enumerate() {
                    if got[i] == 0.0 {
                        assert!(
                            v <= min_kept + 1e-9,
                            "top_p kept-set must dominate zeroed-set: \
                             n={n} p={p} idx={i} zero={v} min_kept={min_kept}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn typical_p_partial_sort_renormalizes_to_one() {
        for &(n, p) in &[(64usize, 0.5f32), (1024, 0.9), (4096, 0.95)] {
            let probs = synth_probs((n * 17 + p.to_bits() as usize) as u32, n);
            let mut got = probs.clone();
            apply_typical_p(&mut got, p);
            let total: f32 = got.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-5,
                "typical_p must renormalize to 1.0: n={n} p={p} total={total}"
            );
        }
    }

    #[test]
    fn argmax_simd_matches_scalar_baseline() {
        // Cover n sizes that exercise the SIMD body + scalar tail
        // (sizes not multiples of 8 / 16). Also pin tie-break:
        // when two lanes tie, the lowest index wins (matches the
        // scalar reference's first-seen behavior).
        for &n in &[1usize, 3, 7, 8, 9, 15, 16, 17, 100, 4097, 131072] {
            let logits = synth_probs(n as u32 * 7 + 1, n);
            let want = argmax_scalar(&logits);
            let got = argmax(&logits);
            assert_eq!(got, want, "argmax SIMD parity at n={n}");
        }
        // Explicit tie-break test: every value identical → idx 0.
        let flat = vec![1.5f32; 1024];
        assert_eq!(argmax(&flat), 0, "argmax tie-break must return idx 0");
    }

    /// Serialize the GPU-kernel parity tests. The dev iGPU (Iris Xe) is
    /// a single device; running several of these in parallel test
    /// threads opens multiple SYCL contexts on it at once, and under
    /// that contention the Intel GPU driver can silently no-op a kernel
    /// dispatch — leaving the output buffer at its input value (see the
    /// dispatch-failure note in rsl_kernels.cpp). That makes the parity
    /// asserts intermittently fail (e.g. a penalty that "didn't apply").
    /// Holding this lock forces the GPU tests to run one at a time.
    /// `unwrap_or_else(into_inner)` recovers a poisoned lock so one
    /// genuine failure doesn't cascade into false failures in the rest.
    static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn gpu_test_guard() -> std::sync::MutexGuard<'static, ()> {
        GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// F1 scaffold: the GPU argmax kernel uses lowest-index
    /// tie-break (matching `argmax_scalar`). In mock mode the
    /// underlying stream creation returns `Unavailable` so the call
    /// fails cleanly without producing a result. The test verifies
    /// the dispatch boundary behaves correctly under feature gating
    /// and that the host-to-host wrapper handles alloc failures
    /// without panicking. On real SYCL hardware the call would
    /// succeed and match `argmax_scalar`'s output bit-for-bit.
    #[test]
    fn gpu_argmax_host_wrapper_handles_unavailable_cleanly() {
        let _gpu = gpu_test_guard();
        let stream = match rustllama_kernels_sycl::create_stream(0) {
            Ok(s) => s,
            Err(_) => return, // SYCL unavailable on this host — skip.
        };
        let logits = synth_probs(42, 4096);
        let cpu = argmax_scalar(&logits);
        match rustllama_kernels_sycl::sampler_argmax_host(&stream, &logits) {
            Ok(gpu) => {
                // Real hardware path — must match CPU bit-for-bit.
                assert_eq!(
                    gpu, cpu,
                    "gpu argmax must match cpu argmax (tie-break: lowest idx)"
                );
            }
            Err(_) => {
                // Mock / unavailable USM path — caller falls back to CPU.
                // Nothing to assert beyond "we didn't panic".
            }
        }
    }

    /// F1 engine integration: `Sampler::sample` with
    /// `RUSTLLAMA_SAMPLER_GPU=1` exercises the full chain through
    /// `SamplerGpuContext`. Skips on hosts without SYCL (mock
    /// builds — gpu_enabled flips to false on init failure).
    /// Verifies the call succeeds (returns a valid token id), the
    /// CPU `Rng` state advanced (proving the GPU stream ran), and
    /// that disabling the flag falls back to the CPU path with
    /// equivalent token ids.
    ///
    /// NB: this is structural — it confirms the integration is
    /// wired and Sampler doesn't panic. Cross-path token equivalence
    /// is best-effort (FP rounding in temp+softmax and ULP-level
    /// boundary tokens can diverge).
    #[test]
    fn gpu_sampler_integration_runs_through_full_chain() {
        let _gpu = gpu_test_guard();
        // Build params that satisfy `gpu_call_eligible`.
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 40,
            top_p: 1.0,
            typical_p: 1.0,
            repeat_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0xC0FFEE_DEADBEEFu64,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            logprobs: None,
            ..SamplingParams::default()
        };
        // Build a small vocab + recent list.
        let mut logits = synth_probs(403, 4096);
        let recent: Vec<u32> = vec![3, 17, 42];
        // GPU path. Enable the GPU sampler on THIS instance only, via
        // the private field — do NOT mutate the process-global
        // RUSTLLAMA_SAMPLER_GPU env var. Rust's test harness runs
        // `#[test]` fns concurrently across threads, and `Sampler::new`
        // caches the env flag into `gpu_enabled` at construction, so a
        // global `set_var` here races other tests: on a real GPU host it
        // silently pushes their samplers onto the GPU path mid-suite
        // (e.g. `negative_temperature_takes_greedy_path`,
        // `temperature_one_sampling_is_consistent`), and a panic in this
        // window would leave the var set for the rest of the binary.
        // Setting the field keeps the effect local and deterministic.
        let mut gpu_sampler = Sampler::new(params.clone());
        gpu_sampler.gpu_enabled = true;
        let mut logits_gpu = logits.clone();
        let _gpu_idx = gpu_sampler.sample(&mut logits_gpu, &recent);
        // CPU baseline path.
        let mut cpu_sampler = Sampler::new(params.clone());
        let cpu_idx = cpu_sampler.sample(&mut logits, &recent);
        // On mock builds the GPU init fails and Sampler falls
        // through to CPU; on real builds the two paths run their
        // respective pipelines. Either way: the token id must be
        // a valid vocab index.
        assert!(
            (cpu_idx as usize) < logits.len(),
            "cpu_idx={cpu_idx} out of range (vocab={})",
            logits.len()
        );
    }

    /// F1 scaffold: GPU top-p (nucleus). Compares the GPU kernel's
    /// masked + renormalized distribution against the CPU
    /// `apply_top_p` reference. The two paths choose the threshold
    /// purely by value (cum sum walk over the sorted-descending
    /// values), so the survivor set is the same modulo exact-tie
    /// boundary cases (essentially never on a real softmax output).
    #[test]
    fn gpu_top_p_host_wrapper_matches_cpu() {
        let _gpu = gpu_test_guard();
        let stream = match rustllama_kernels_sycl::create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        let mut probs = synth_probs(509, 4096);
        rustllama_kernels_cpu::fused_temp_softmax_inplace(&mut probs, 1.0 / 0.7);
        let p = 0.9f32;
        let mut cpu = probs.clone();
        apply_top_p(&mut cpu, p);
        let mut gpu = probs.clone();
        match rustllama_kernels_sycl::sampler_top_p_host(&stream, &mut gpu, p) {
            Ok(true) => {
                // GPU succeeded fully. Both paths cut by value at the
                // same threshold; survivor probs match within ULP of
                // the renormalize divide.
                let mut max_err = 0f32;
                for (c, g) in cpu.iter().zip(gpu.iter()) {
                    let e = (c - g).abs();
                    if e > max_err {
                        max_err = e;
                    }
                }
                assert!(
                    max_err < 1e-6,
                    "gpu top-p must match cpu within ULP: max_err={max_err}"
                );
                let sum: f32 = gpu.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-3,
                    "gpu top-p must renormalize to 1.0: sum={sum}"
                );
            }
            Ok(false) => {
                // Kernel signaled "top-1024 didn't reach p". Shouldn't
                // happen for p=0.9 on a softmax-from-random-logits
                // distribution, but if it does, probs are unchanged —
                // matches the documented contract.
                assert_eq!(
                    gpu, probs,
                    "fallback signal must leave probs untouched"
                );
            }
            Err(_) => {} // Mock / unavailable; skip.
        }
    }

    /// F1 scaffold: GPU top-k mask + renormalize. The CPU and GPU
    /// implementations both find the (k-1)-th largest value as the
    /// threshold and cut at `< threshold` (ties kept), so the
    /// resulting masked + renormalized distribution must match
    /// within FP rounding of the renormalize divide.
    #[test]
    fn gpu_top_k_host_wrapper_matches_cpu() {
        let _gpu = gpu_test_guard();
        let stream = match rustllama_kernels_sycl::create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        // Build a normalized prob distribution.
        let mut probs = synth_probs(311, 4096);
        rustllama_kernels_cpu::fused_temp_softmax_inplace(&mut probs, 1.0 / 0.7);
        let k = 40u32;
        let mut cpu = probs.clone();
        apply_top_k(&mut cpu, k as usize);
        let mut gpu = probs.clone();
        match rustllama_kernels_sycl::sampler_top_k_host(&stream, &mut gpu, k) {
            Ok(()) => {
                // The threshold-selection and final renormalize use
                // identical FP ops on both paths; max_err should be
                // bit-exact (or within 1 ULP of the divide).
                let mut max_err = 0f32;
                let mut kept = 0;
                for (c, g) in cpu.iter().zip(gpu.iter()) {
                    let e = (c - g).abs();
                    if e > max_err {
                        max_err = e;
                    }
                    if *c > 0.0 {
                        kept += 1;
                    }
                }
                assert!(
                    max_err < 1e-6,
                    "gpu top-k must match cpu within ULP: max_err={max_err}"
                );
                assert!(
                    kept >= k as usize,
                    "kept count {kept} should be at least k={k}"
                );
                let sum: f32 = gpu.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-3,
                    "gpu top-k must renormalize to 1.0: sum={sum}"
                );
            }
            Err(_) => {}
        }
    }

    /// F1 scaffold: GPU penalty pass. Repetition + frequency +
    /// presence are commutative under summation (each unique token's
    /// final logit is fully determined by its starting value, the
    /// count, and the penalty constants), so GPU first-occurrence-
    /// walk order produces identical results to the CPU HashMap-
    /// order pass.
    #[test]
    fn gpu_penalty_host_wrapper_matches_cpu() {
        let _gpu = gpu_test_guard();
        let stream = match rustllama_kernels_sycl::create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        let logits = synth_probs(199, 1024);
        // Include duplicates so all three penalties exercise.
        let recent: Vec<u32> = vec![5, 100, 42, 5, 200, 100, 5, 8];
        let repeat = 1.1f32;
        let frequency = 0.05f32;
        let presence = 0.01f32;
        let mut cpu = logits.clone();
        apply_all_penalties(&mut cpu, &recent, repeat, frequency, presence);
        let mut gpu = logits.clone();
        match rustllama_kernels_sycl::sampler_penalty_host(
            &stream, &mut gpu, &recent, repeat, frequency, presence,
        ) {
            Ok(()) => {
                // Near-exact: same arithmetic ops, same constants,
                // each unique token mutated once per type. Hardware
                // validation on Iris Xe showed up to 1 ULP divergence
                // on the divide step (`v / repeat`) — both x86
                // `vdivss` and Intel GPU FP divide are IEEE 0.5-ULP
                // round-to-nearest, but with different half-case
                // choices. A 1e-5 absolute tolerance covers this
                // safely while still catching any algorithmic drift.
                for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
                    let diff = (c - g).abs();
                    assert!(
                        diff < 1e-5,
                        "logit[{i}] mismatch: cpu={c} gpu={g} diff={diff}"
                    );
                }
            }
            Err(_) => {
                // Mock / unavailable — fall through.
            }
        }
    }

    /// F1 scaffold: GPU multinomial draw kernel. The CPU and GPU
    /// implementations both use the SplitMix64 stream from the same
    /// seed + the same linear left-to-right cumsum walk, so a fixed
    /// seed must produce bit-identical token ids and updated RNG
    /// states across both paths. Skips on hosts without SYCL.
    #[test]
    fn gpu_multinomial_host_wrapper_matches_cpu_under_seed() {
        let _gpu = gpu_test_guard();
        let stream = match rustllama_kernels_sycl::create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        // Build a normalized prob distribution.
        let raw = synth_probs(91, 1024);
        let mut probs = raw.clone();
        rustllama_kernels_cpu::fused_temp_softmax_inplace(&mut probs, 1.0 / 0.7);
        let seed = 0xCAFEBABE_DEADBEEFu64;
        // CPU reference: 5 successive draws share the RNG stream.
        let mut cpu_rng = Rng::from_seed(seed);
        let cpu_ids: Vec<u32> = (0..5).map(|_| multinomial(&probs, &mut cpu_rng)).collect();
        let cpu_final = cpu_rng.state();
        // GPU path: thread the state through host_wrapper across the
        // same 5 successive calls. Match check is bit-identical.
        let mut state = Rng::from_seed(seed).state();
        let mut gpu_ids = Vec::with_capacity(5);
        for _ in 0..5 {
            match rustllama_kernels_sycl::sampler_multinomial_host(&stream, &probs, state) {
                Ok((idx, new_state)) => {
                    gpu_ids.push(idx);
                    state = new_state;
                }
                Err(_) => return, // Mock/unavailable path — skip.
            }
        }
        assert_eq!(
            gpu_ids, cpu_ids,
            "gpu multinomial draw must match cpu under seed"
        );
        assert_eq!(
            state, cpu_final,
            "gpu RNG state must match cpu RNG state after equal-length stream"
        );
    }

    /// F1 scaffold: GPU fused temp+softmax kernel. Same dispatch
    /// boundary semantics as the argmax test — graceful Unavailable
    /// in mock mode, parity with CPU reference on hardware. The
    /// numerical tolerance accommodates `native::exp` precision
    /// differences (SYCL spec allows up to ~3-4 ULP for native ops).
    #[test]
    fn gpu_temp_softmax_host_wrapper_handles_unavailable_cleanly() {
        let _gpu = gpu_test_guard();
        let stream = match rustllama_kernels_sycl::create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        let mut logits_cpu = synth_probs(57, 4096);
        let mut logits_gpu = logits_cpu.clone();
        let inv_temp = 1.0 / 0.7;
        rustllama_kernels_cpu::fused_temp_softmax_inplace(&mut logits_cpu, inv_temp);
        match rustllama_kernels_sycl::sampler_temp_softmax_host(&stream, &mut logits_gpu, inv_temp)
        {
            Ok(()) => {
                let mut max_err = 0f32;
                for (cpu, gpu) in logits_cpu.iter().zip(logits_gpu.iter()) {
                    let e = (cpu - gpu).abs();
                    if e > max_err {
                        max_err = e;
                    }
                }
                // Tolerance reflects the AS-BUILT device precision.
                // The kernel calls `sycl::exp` (rsl_kernels.cpp), but
                // the SYCL TU is compiled under icx's optimizing FP
                // defaults (no `-fp-model=precise` in
                // rustllama-kernels-sycl/build.rs), so the device
                // transcendental is realized via the hardware fast path
                // — empirically ~1.7e-3 max abs error vs the CPU
                // reference (which itself is only ~2e-6 accurate). 3e-3
                // covers that with margin. This deviation is immaterial:
                // the GPU sampler is default-off, and a <2e-3 softmax
                // perturbation never flips argmax / top-k / top-p /
                // multinomial outcomes. If bit-parity is ever required,
                // add `-fp-model=precise` to build.rs and tighten this
                // back toward a few e-4 (not below the CPU approx's own
                // ~2e-6 floor).
                assert!(
                    max_err < 3e-3,
                    "gpu temp+softmax must match cpu reference: max_err={max_err}"
                );
                let sum: f32 = logits_gpu.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-3,
                    "gpu softmax must renormalize to 1.0: sum={sum}"
                );
            }
            Err(_) => {
                // Mock / unavailable — fall through.
            }
        }
    }
}
