//! P12b — beam search: deterministic, multi-hypothesis decoding.
//!
//! Unlike P12a's `n` / `best_of` (which samples `k` independent completions
//! and ranks them after the fact), beam search maintains `num_beams` partial
//! hypotheses *simultaneously*. At each step every live beam is expanded by
//! its top candidate tokens; the global top `num_beams` continuations (by
//! cumulative log-prob) survive to the next step. A beam that emits EOS is
//! retired to a finished pool with a length-penalty-adjusted score. The driver
//! returns the top `k` finished hypotheses.
//!
//! Beam search here is **fully deterministic** — there is no sampling. The
//! only ordering input is the model's logits, so the same prompt always yields
//! the same beams. With `num_beams == 1` the single surviving beam is the
//! argmax at every step, i.e. byte-identical to the existing greedy decode
//! (the server never even enters the beam path at `num_beams == 1`; the engine
//! driver reproduces it for the parity anchor).
//!
//! This module holds the *pure* beam bookkeeping — candidate construction,
//! the HuggingFace-faithful selection rule, length-penalty scoring, and the
//! finished-hypothesis pool. The engine driver
//! ([`crate::inference::CpuEngine::generate_beam_token_ids`]) owns the KV
//! fork/restore + forward passes and calls into these functions. Keeping the
//! logic pure makes the ranking hand-verifiable in unit tests without a model.

/// Request-level beam-search parameters.
///
/// `num_beams == 1` is greedy; the server routes that through the ordinary
/// single-sequence decode and never constructs a `BeamParams`. The engine
/// driver still accepts `num_beams == 1` so the parity test can prove the
/// machinery degenerates to greedy.
#[derive(Debug, Clone)]
pub struct BeamParams {
    /// Number of hypotheses kept alive at each step. `>= 1`.
    pub num_beams: usize,
    /// GNMT/HuggingFace length penalty exponent applied when a hypothesis is
    /// scored for final ranking: `score = cum_logprob / len^length_penalty`.
    /// `1.0` = plain per-token mean in log-space-ish (divide by length);
    /// `> 1.0` rewards longer sequences, `< 1.0` rewards shorter, `0.0`
    /// disables normalization (rank by raw cumulative log-prob).
    pub length_penalty: f32,
    /// When `true`, stop as soon as `num_beams` finished (EOS) hypotheses
    /// exist. When `false` (default), keep expanding until the best finished
    /// hypothesis provably can't be beaten by any live beam (the standard
    /// HuggingFace bound).
    pub early_stopping: bool,
    /// How many ranked hypotheses to return (`k`). Clamped to `num_beams`.
    pub num_return: usize,
}

impl BeamParams {
    /// Effective number of returned sequences — at least 1, at most
    /// `num_beams`.
    pub fn k(&self) -> usize {
        self.num_return.clamp(1, self.num_beams.max(1))
    }
}

/// One finished (or finalized-at-max-length) beam hypothesis.
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Generated tokens (excludes the prompt). A hypothesis that ended on EOS
    /// has EOS as its last token — matching the greedy decode, which also
    /// emits the EOS token before stopping.
    pub tokens: Vec<u32>,
    /// Raw sum of chosen-token log-probs over `tokens`.
    pub cum_logprob: f32,
    /// Length-penalty-adjusted score used for ranking. Higher is better.
    pub score: f32,
}

/// Length-penalty-adjusted score (GNMT / HuggingFace convention):
/// `score = cum_logprob / len^length_penalty`.
///
/// An empty hypothesis scores `-inf` so it never wins a ranking. With
/// `length_penalty == 0` the denominator is `1` and the score is the raw
/// cumulative log-prob (no length normalization).
pub fn length_adjusted_score(cum_logprob: f32, len: usize, length_penalty: f32) -> f32 {
    if len == 0 {
        return f32::NEG_INFINITY;
    }
    cum_logprob / (len as f32).powf(length_penalty)
}

/// A single beam-search step's candidate: the live beam it extends, the next
/// token, and the resulting RAW cumulative log-prob (parent cum + token
/// log-prob).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamCandidate {
    pub parent: usize,
    pub token: u32,
    pub cum_logprob: f32,
}

/// Build the sorted candidate list for one step.
///
/// `per_beam[i]` is `(parent_cum_logprob, &[(token, token_logprob), ...])` for
/// live beam `i` — the beam's running cumulative log-prob and its top
/// expansions (already selected by logit, highest first). The returned
/// candidates are sorted by `cum_logprob` DESCENDING, ties broken by `token`
/// ASCENDING then `parent` ASCENDING. That tie-break is what makes a
/// single-beam top pick equal the argmax token (lowest id on a logit tie),
/// so `num_beams == 1` matches greedy exactly.
pub fn build_sorted_candidates(per_beam: &[(f32, &[(u32, f32)])]) -> Vec<BeamCandidate> {
    let mut cands: Vec<BeamCandidate> = Vec::new();
    for (parent, (parent_cum, expansions)) in per_beam.iter().enumerate() {
        for (token, token_lp) in expansions.iter() {
            cands.push(BeamCandidate {
                parent,
                token: *token,
                cum_logprob: parent_cum + token_lp,
            });
        }
    }
    cands.sort_by(|a, b| {
        // Primary: cumulative log-prob, descending.
        b.cum_logprob
            .partial_cmp(&a.cum_logprob)
            .unwrap_or(std::cmp::Ordering::Equal)
            // Tie-break 1: lower token id first (matches argmax's lowest-index
            // tie-break so num_beams=1 == greedy).
            .then(a.token.cmp(&b.token))
            // Tie-break 2: lower parent beam first.
            .then(a.parent.cmp(&b.parent))
    });
    cands
}

/// One step's selection outcome.
pub struct StepSelection {
    /// Beams that survive to the next step (non-EOS), at most `num_beams`.
    pub continuing: Vec<BeamCandidate>,
    /// EOS-terminated candidates that should be retired to the finished pool.
    pub finished: Vec<BeamCandidate>,
}

/// HuggingFace-faithful per-step selection over the sorted candidates.
///
/// Scans candidates in rank order. A non-EOS candidate becomes a continuing
/// beam. An EOS candidate is retired to `finished` *only if its rank is within
/// the top `num_beams`* (an EOS ranked below that can't out-score the kept
/// continuations, so HF drops it). The scan stops once `num_beams` continuing
/// beams have been collected.
pub fn select_step(
    candidates: &[BeamCandidate],
    num_beams: usize,
    eos: Option<u32>,
) -> StepSelection {
    let mut continuing: Vec<BeamCandidate> = Vec::with_capacity(num_beams);
    let mut finished: Vec<BeamCandidate> = Vec::new();
    for (rank, c) in candidates.iter().enumerate() {
        if continuing.len() == num_beams {
            break;
        }
        if eos == Some(c.token) {
            if rank < num_beams {
                finished.push(*c);
            }
            // An EOS ranked below the top num_beams is discarded (HF rule).
        } else {
            continuing.push(*c);
        }
    }
    StepSelection {
        continuing,
        finished,
    }
}

/// The finished-hypothesis pool: keeps the best `num_beams` completed beams by
/// length-adjusted score and answers the early-stopping done check.
#[derive(Debug)]
pub struct BeamPool {
    num_beams: usize,
    length_penalty: f32,
    beams: Vec<BeamHypothesis>,
    /// Worst (lowest) score currently in the pool; `+inf` while not yet full.
    worst_score: f32,
}

impl BeamPool {
    pub fn new(num_beams: usize, length_penalty: f32) -> Self {
        Self {
            num_beams: num_beams.max(1),
            length_penalty,
            beams: Vec::with_capacity(num_beams.max(1)),
            worst_score: f32::INFINITY,
        }
    }

    /// Number of finished hypotheses currently retained.
    pub fn len(&self) -> usize {
        self.beams.len()
    }

    pub fn is_empty(&self) -> bool {
        self.beams.is_empty()
    }

    /// Add a completed hypothesis (`tokens`, raw `cum_logprob`). The pool
    /// keeps only the best `num_beams` by length-adjusted score.
    pub fn add(&mut self, tokens: Vec<u32>, cum_logprob: f32) {
        let score = length_adjusted_score(cum_logprob, tokens.len(), self.length_penalty);
        // Skip a hypothesis that can't make the cut once the pool is full.
        if self.beams.len() >= self.num_beams && score <= self.worst_score {
            return;
        }
        self.beams.push(BeamHypothesis {
            tokens,
            cum_logprob,
            score,
        });
        // Keep only the best num_beams.
        if self.beams.len() > self.num_beams {
            // Evict the single worst entry.
            let (worst_idx, _) = self
                .beams
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.score
                        .partial_cmp(&b.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("pool non-empty after push");
            self.beams.swap_remove(worst_idx);
        }
        self.recompute_worst();
    }

    fn recompute_worst(&mut self) {
        if self.beams.len() < self.num_beams {
            self.worst_score = f32::INFINITY;
        } else {
            self.worst_score = self
                .beams
                .iter()
                .map(|b| b.score)
                .fold(f32::INFINITY, f32::min);
        }
    }

    /// HuggingFace `BeamHypotheses.is_done`: are we finished?
    ///
    /// - Fewer than `num_beams` finished ⇒ never done.
    /// - `early_stopping` ⇒ done as soon as `num_beams` are finished.
    /// - otherwise ⇒ done when the worst finished score is already at least
    ///   the best score any live beam could still reach at the current length
    ///   (`best_running_cum / cur_len^length_penalty`).
    pub fn is_done(&self, early_stopping: bool, best_running_cum: f32, cur_len: usize) -> bool {
        if self.beams.len() < self.num_beams {
            return false;
        }
        if early_stopping {
            return true;
        }
        let highest_attainable =
            length_adjusted_score(best_running_cum, cur_len.max(1), self.length_penalty);
        self.worst_score >= highest_attainable
    }

    /// Consume the pool, returning hypotheses sorted by score (best first).
    pub fn into_ranked(mut self) -> Vec<BeamHypothesis> {
        self.beams.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                // Deterministic tie-break: shorter sequence, then lexicographic
                // token order, so ties are reproducible across runs.
                .then(a.tokens.len().cmp(&b.tokens.len()))
                .then(a.tokens.cmp(&b.tokens))
        });
        self.beams
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn length_adjusted_score_matches_formula() {
        // lp = 1.0 → per-token mean of the cumulative log-prob.
        assert!(approx(length_adjusted_score(-2.0, 2, 1.0), -1.0));
        assert!(approx(length_adjusted_score(-3.0, 4, 1.0), -0.75));
        // lp = 0.0 → raw cumulative log-prob (no normalization).
        assert!(approx(length_adjusted_score(-2.0, 2, 0.0), -2.0));
        assert!(approx(length_adjusted_score(-4.5, 5, 0.0), -4.5));
        // Empty hypothesis never wins.
        assert_eq!(length_adjusted_score(0.0, 0, 1.0), f32::NEG_INFINITY);
    }

    #[test]
    fn length_penalty_flips_short_vs_long_ranking() {
        // Beam A: short, higher raw log-prob. Beam B: longer, lower raw
        // log-prob but a better per-token average.
        //   A: cum = -2.0, len = 2  → per-token -1.0
        //   B: cum = -4.5, len = 5  → per-token -0.9
        let a = (-2.0f32, 2usize);
        let b = (-4.5f32, 5usize);

        // length_penalty = 0.0 → rank by raw cum log-prob → A (-2.0) beats B
        // (-4.5).
        let a0 = length_adjusted_score(a.0, a.1, 0.0);
        let b0 = length_adjusted_score(b.0, b.1, 0.0);
        assert!(a0 > b0, "lp=0 should favor the shorter higher-logprob beam");

        // length_penalty = 1.0 → per-token mean → B (-0.9) beats A (-1.0).
        let a1 = length_adjusted_score(a.0, a.1, 1.0);
        let b1 = length_adjusted_score(b.0, b.1, 1.0);
        assert!(b1 > a1, "lp=1 should flip to the longer beam with better mean");
    }

    #[test]
    fn single_beam_top_candidate_is_argmax() {
        // One beam; its expansions are the top tokens by logit, highest first.
        // The top candidate must be the first (argmax) token.
        let exp = [(7u32, -0.1f32), (3u32, -0.5f32)];
        let per_beam = [(0.0f32, exp.as_slice())];
        let cands = build_sorted_candidates(&per_beam);
        assert_eq!(cands[0].token, 7);
        assert_eq!(cands[0].parent, 0);
    }

    #[test]
    fn candidate_tie_break_prefers_lower_token_id() {
        // Two candidates with identical cumulative log-prob must order by the
        // lower token id (argmax's lowest-index tie-break).
        let exp_a = [(9u32, -1.0f32)];
        let exp_b = [(4u32, -1.0f32)];
        let per_beam = [(0.0f32, exp_a.as_slice()), (0.0f32, exp_b.as_slice())];
        let cands = build_sorted_candidates(&per_beam);
        assert_eq!(cands[0].token, 4, "tie must resolve to the lower token id");
    }

    #[test]
    fn select_step_keeps_true_top_two_by_score() {
        // Hand-verified 2-beam fixture. Two live beams, each expands to two
        // tokens. The kept continuing set must be the true global top-2 by
        // cumulative log-prob.
        //
        // Beam 0 (cum 0.0): token 1 @ -0.2, token 2 @ -1.0
        //   → candidates: (0,1,-0.2), (0,2,-1.0)
        // Beam 1 (cum -0.5): token 3 @ -0.1, token 4 @ -0.9
        //   → candidates: (1,3,-0.6), (1,4,-1.4)
        //
        // Sorted:  (0,1,-0.2), (1,3,-0.6), (0,2,-1.0), (1,4,-1.4)
        // Top-2 continuing: (0,1,-0.2) and (1,3,-0.6).
        let exp0 = [(1u32, -0.2f32), (2u32, -1.0f32)];
        let exp1 = [(3u32, -0.1f32), (4u32, -0.9f32)];
        let per_beam = [(0.0f32, exp0.as_slice()), (-0.5f32, exp1.as_slice())];
        let cands = build_sorted_candidates(&per_beam);

        // Verify the full sorted order first.
        assert_eq!(cands[0].token, 1);
        assert_eq!(cands[1].token, 3);
        assert_eq!(cands[2].token, 2);
        assert_eq!(cands[3].token, 4);

        let sel = select_step(&cands, 2, Some(99));
        assert_eq!(sel.continuing.len(), 2);
        assert_eq!(sel.continuing[0].token, 1);
        assert_eq!(sel.continuing[0].parent, 0);
        assert!(approx(sel.continuing[0].cum_logprob, -0.2));
        assert_eq!(sel.continuing[1].token, 3);
        assert_eq!(sel.continuing[1].parent, 1);
        assert!(approx(sel.continuing[1].cum_logprob, -0.6));
        assert!(sel.finished.is_empty());
    }

    #[test]
    fn select_step_retires_eos_within_top_num_beams() {
        // A top-ranked EOS retires to the finished set; the next non-EOS fills
        // the continuing slot. An EOS ranked below num_beams is dropped.
        //   (0, EOS=0, -0.1), (1, 5, -0.3), (1, EOS=0, -2.0)
        let cands = vec![
            BeamCandidate { parent: 0, token: 0, cum_logprob: -0.1 },
            BeamCandidate { parent: 1, token: 5, cum_logprob: -0.3 },
            BeamCandidate { parent: 1, token: 0, cum_logprob: -2.0 },
        ];
        let sel = select_step(&cands, 2, Some(0));
        // EOS at rank 0 retires; token 5 continues; we now have 1 continuing.
        // The rank-2 EOS is beyond num_beams and is discarded.
        assert_eq!(sel.finished.len(), 1);
        assert_eq!(sel.finished[0].parent, 0);
        assert_eq!(sel.continuing.len(), 1);
        assert_eq!(sel.continuing[0].token, 5);
    }

    #[test]
    fn beam_pool_keeps_best_and_ranks() {
        let mut pool = BeamPool::new(2, 1.0);
        pool.add(vec![1, 2], -2.0); // score -1.0
        pool.add(vec![3, 4, 5], -1.5); // score -0.5
        pool.add(vec![6], -3.0); // score -3.0 — worse than both, evicted
        assert_eq!(pool.len(), 2);
        let ranked = pool.into_ranked();
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].tokens, vec![3, 4, 5]); // best score -0.5
        assert_eq!(ranked[1].tokens, vec![1, 2]);
    }

    #[test]
    fn beam_pool_early_stopping_done_when_full() {
        let mut pool = BeamPool::new(2, 1.0);
        assert!(!pool.is_done(true, 0.0, 1));
        pool.add(vec![1], -1.0);
        assert!(!pool.is_done(true, 0.0, 1), "not full yet");
        pool.add(vec![2], -1.0);
        assert!(pool.is_done(true, 0.0, 1), "full ⇒ early_stopping done");
    }

    #[test]
    fn beam_pool_bounded_stopping_waits_for_unbeatable() {
        // With early_stopping=false, the pool is done only when the worst
        // finished score beats the best attainable running score.
        let mut pool = BeamPool::new(1, 1.0);
        pool.add(vec![1, 2], -1.0); // score -0.5
        // A live beam with cum -0.2 at len 1 could reach -0.2 → still better
        // than -0.5, so NOT done.
        assert!(!pool.is_done(false, -0.2, 1));
        // A live beam with cum -2.0 at len 1 attains only -2.0 < -0.5 → done.
        assert!(pool.is_done(false, -2.0, 1));
    }
}
