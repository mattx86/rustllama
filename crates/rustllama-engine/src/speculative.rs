//! Speculative decoding scaffold (Chen et al. 2023, "Accelerating
//! Large Language Model Decoding with Speculative Sampling").
//!
//! A *draft* model proposes K tokens. A *target* model verifies them
//! all in one forward pass. Accepted tokens commit immediately;
//! rejection at position j commits 0..j and resamples one token from
//! the residual distribution `max(0, p - q)`.
//!
//! Real-model integration (loading two models, building the joint
//! forward, plumbing through the `Engine` trait) is a multi-week task;
//! this module ships the **algorithm** in isolation so the math is
//! pinned and the calling shape is locked in. The accept/reject loop
//! here is pure: it operates on already-sampled draft tokens + their
//! probability vectors and produces a `SpeculationOutcome` that the
//! engine consumes.
//!
//! Why land the scaffold now?
//!   1. The math is finicky (probability ratios, residual sampling,
//!      "if rejected, sample from `max(0, p-q)`"). Getting this right
//!      once, in a test-driven module, beats discovering bugs after
//!      we've stood up two models.
//!   2. Callers can already wire mock draft/target distributions into
//!      tests so the engine-level integration has a known-good
//!      reference to validate against.
//!
//! Out of scope for this commit:
//!   - Loading and managing two models.
//!   - The fused draft+target forward pass.
//!   - KV-cache rewinding when a speculation rejects mid-batch.
//!   - SYCL-side acceptance kernel (for now we run on CPU).

use crate::sampling::Rng;

/// One token proposed by the draft model, with its probability under
/// the draft's distribution. The accept/reject step needs both halves
/// — the id to compare against the target's vocab and the probability
/// to form the `p / q` ratio.
#[derive(Debug, Clone, Copy)]
pub struct DraftToken {
    pub id: u32,
    /// Probability the draft assigned to this token under its
    /// (already-softmax'd, post-temperature) distribution at the
    /// position it was sampled.
    pub q: f32,
}

/// Outcome of one speculative-decode step.
///
/// `accepted` is the prefix of draft tokens that survived the
/// rejection test — possibly empty (early reject) or the full draft
/// (all-K accept).
///
/// `replacement` is the next token to commit AFTER the accepted
/// prefix:
///   - If the speculation was rejected at position j, this is sampled
///     from the residual distribution `max(0, p_j - q_j)` and replaces
///     what would have been `draft[j]`.
///   - If all K drafts were accepted, this is sampled fresh from
///     `target_probs[K]` (the position one past the last accepted
///     token's slot) — i.e. the "bonus" token that paid for itself.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculationOutcome {
    pub accepted: Vec<u32>,
    pub replacement: u32,
}

/// Run the accept/reject loop.
///
/// Inputs:
///   - `drafts[i]`: the i-th draft token + its `q` (probability under
///     the draft's distribution).
///   - `target_probs[i]`: the FULL target-distribution probability
///     vector at position i (over the entire vocab). Must be already-
///     softmax'd. `target_probs.len()` must be `drafts.len() + 1` —
///     the extra slot is the "bonus" position used when all drafts
///     are accepted (sample one fresh token).
///   - `rng`: the same seeded RNG the rest of the sampler uses so
///     determinism by-seed extends to the speculative path.
///
/// Algorithm (Chen et al., Algorithm 1 in the speculative-sampling
/// paper, with the standard residual-renormalize finishing step):
///
/// ```text
/// for i in 0..K:
///   p = target_probs[i][drafts[i].id]
///   q = drafts[i].q
///   if u < p/q:                   # u uniform in [0, 1)
///       accept drafts[i]
///   else:
///       # reject; sample one token from residual max(0, p - q) over the vocab,
///       # renormalized. Return accepted ++ residual_sample.
///       break
/// if all K accepted:
///   # bonus: sample from target_probs[K]
///   return accepted ++ multinomial(target_probs[K])
/// ```
///
/// `accept_reject` panics if `target_probs.len() != drafts.len() + 1`
/// — the bonus slot is mandatory.
pub fn accept_reject(
    drafts: &[DraftToken],
    target_probs: &[&[f32]],
    rng: &mut Rng,
) -> SpeculationOutcome {
    assert_eq!(
        target_probs.len(),
        drafts.len() + 1,
        "speculative: target_probs must have one extra position for the bonus token"
    );

    let mut accepted: Vec<u32> = Vec::with_capacity(drafts.len());

    for (i, draft) in drafts.iter().enumerate() {
        let id = draft.id as usize;
        let p = if id < target_probs[i].len() {
            target_probs[i][id]
        } else {
            0.0
        };
        let q = draft.q.max(1e-30);
        let ratio = (p / q).min(1.0);
        let u = rng.next_f32();
        if u <= ratio {
            accepted.push(draft.id);
        } else {
            // Reject. Sample one token from the residual
            // distribution `max(0, p - q_at_id)` renormalized.
            // The "q_at_id" piece is a vector that's q at the
            // draft token's id and 0 elsewhere — this is what the
            // paper's `q(x)` actually is for a single-sample draft.
            let mut residual: Vec<f32> = target_probs[i].iter().copied().collect();
            if id < residual.len() {
                residual[id] = (residual[id] - q).max(0.0);
                // The other positions just keep `p` since `q(x)` is
                // zero for x != drafts[i].id (a single sampled draft
                // assigns zero mass to anything else by definition).
            }
            let sum: f32 = residual.iter().sum();
            let chosen = if sum > 0.0 {
                let inv = 1.0 / sum;
                for v in residual.iter_mut() {
                    *v *= inv;
                }
                multinomial(&residual, rng)
            } else {
                // Degenerate case: target's distribution at this
                // position was a delta on the draft's token. Fall
                // back to argmax(target).
                argmax(target_probs[i])
            };
            return SpeculationOutcome {
                accepted,
                replacement: chosen,
            };
        }
    }

    // All K drafts accepted. Sample the bonus token from the next
    // target position.
    let bonus = multinomial(target_probs[drafts.len()], rng);
    SpeculationOutcome {
        accepted,
        replacement: bonus,
    }
}

/// Exact temperature→0 limit of [`accept_reject`]: accept a draft
/// iff it IS the target argmax; the replacement (on reject) and the
/// bonus (on full acceptance) are the argmax. Fully deterministic —
/// no RNG.
///
/// Callers route `temperature <= 0.0` requests here (same greedy
/// condition as the classic sampler). The stochastic path samples
/// the raw softmax regardless of the request temperature, which
/// flips near-tie tokens by seed — observed live on the 27B, where
/// a greedy request answered "red, yellow, and blue" (RYB painter's
/// primaries, ~coin-flip vs RGB) instead of the classic path's
/// "red, green, and blue". As temp→0 the target distribution
/// collapses to a delta on the argmax, so the acceptance ratio
/// `p/q` is 1 for the argmax and 0 for everything else — this
/// function is that limit computed exactly, making greedy
/// speculative output equal to classic greedy TOKEN FOR TOKEN by
/// construction rather than with high probability.
pub fn accept_reject_greedy(
    drafts: &[DraftToken],
    target_probs: &[&[f32]],
) -> SpeculationOutcome {
    assert_eq!(
        target_probs.len(),
        drafts.len() + 1,
        "speculative: target_probs must have one extra position for the bonus token"
    );
    let mut accepted: Vec<u32> = Vec::with_capacity(drafts.len());
    for (i, draft) in drafts.iter().enumerate() {
        let am = argmax(target_probs[i]);
        if draft.id == am {
            accepted.push(draft.id);
        } else {
            return SpeculationOutcome {
                accepted,
                replacement: am,
            };
        }
    }
    let bonus = argmax(target_probs[drafts.len()]);
    SpeculationOutcome {
        accepted,
        replacement: bonus,
    }
}

/// One node of a drafted token TREE (EAGLE / Medusa / SpecInfer style).
///
/// A tree generalizes the linear draft chain: instead of one proposed
/// continuation, the drafter proposes a branching set of candidates, and a
/// single target forward (with a tree attention mask, so each node attends only
/// its ancestors) scores them all at once. Verification then walks the single
/// best accepted root→leaf path.
#[derive(Debug, Clone, Copy)]
pub struct TreeNode {
    /// Candidate token id at this node.
    pub id: u32,
    /// Draft probability the drafter assigned to this token at this position
    /// (the `q` in the `p/q` acceptance ratio).
    pub q: f32,
    /// Index of the parent node in the same slice, or `None` for a depth-1
    /// candidate (a child of the already-committed context tip, i.e. the root
    /// level). Nodes MUST be topologically ordered: `parent < self index`.
    pub parent: Option<usize>,
}

/// Outcome of tree-structured verification: like [`SpeculationOutcome`] but also
/// carrying the flattened tree-node INDEX of each accepted token, so the engine
/// can compact the accepted path's (scattered) K/V cache slots into contiguous
/// committed positions without a second forward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeVerifyOutcome {
    /// Accepted token ids along the single best path (root → …).
    pub accepted: Vec<u32>,
    /// Flattened node index of each accepted token (parallel to `accepted`). The
    /// token at `accepted[j]` was tree node `accepted_nodes[j]`, living at cache
    /// slot `committed_len + accepted_nodes[j]` after the verify forward.
    pub accepted_nodes: Vec<usize>,
    /// Next token to commit after the accepted path (residual sample on reject /
    /// bonus on full-path acceptance).
    pub replacement: u32,
}

/// Tree-structured speculative verification — the branching generalization of
/// [`accept_reject`]. Walks the single best accepted path from the root level
/// down the tree, running the same recursive-rejection test at each frontier.
///
/// Inputs:
///   - `nodes`: the draft tree, topologically ordered (`parent < self`).
///   - `target_root`: the target distribution for the FIRST speculative token
///     (at the committed context tip); the depth-1 candidates test against it.
///   - `target_next[j]`: the target distribution for the token that FOLLOWS
///     node `j`'s path — consulted only once node `j` is accepted, to test its
///     children and (if it becomes a leaf of the accepted path) draw the bonus.
///     `target_next.len()` MUST equal `nodes.len()`.
///   - `rng`: the shared seeded RNG, so by-seed determinism extends here too.
///
/// At each frontier the siblings are tried in slice order with independent
/// uniform draws; the first to pass its `u <= min(1, p/q)` test is accepted and
/// the walk descends into it. A rejected sibling's mass is removed from the
/// residual pool (`max(0, residual[id] - q)`) before the next sibling is tried,
/// so when every sibling rejects the replacement is drawn from the renormalized
/// residual — exactly the recursive-rejection scheme that keeps tree
/// verification unbiased. With one child per frontier this is **bit-identical**
/// (same RNG draw sequence, same residual) to [`accept_reject`] over the
/// equivalent chain — the property `tree_accept_reject_matches_linear_for_chain`
/// pins.
pub fn tree_accept_reject(
    nodes: &[TreeNode],
    target_root: &[f32],
    target_next: &[&[f32]],
    rng: &mut Rng,
) -> TreeVerifyOutcome {
    assert_eq!(
        target_next.len(),
        nodes.len(),
        "speculative: target_next must have one distribution per tree node"
    );
    let mut accepted: Vec<u32> = Vec::with_capacity(nodes.len());
    let mut accepted_nodes: Vec<usize> = Vec::with_capacity(nodes.len());
    // The frontier: whose children we're choosing among, and the target
    // distribution that governs that choice. `None` = the root level (children
    // of the committed tip); `Some(j)` = children of accepted node `j`.
    let mut frontier: Option<usize> = None;
    let mut current_target: &[f32] = target_root;

    loop {
        // Residual pool for this frontier: starts as the governing target and
        // loses each rejected sibling's mass. Only materialized (cloned) if a
        // reject actually happens, to keep the all-accept path allocation-light
        // past the descent.
        let mut residual: Option<Vec<f32>> = None;
        let mut descended = false;

        for (j, node) in nodes.iter().enumerate() {
            if node.parent != frontier {
                continue;
            }
            let id = node.id as usize;
            let p = if id < current_target.len() {
                current_target[id]
            } else {
                0.0
            };
            let q = node.q.max(1e-30);
            let ratio = (p / q).min(1.0);
            let u = rng.next_f32();
            if u <= ratio {
                // Accept this child: commit it and descend into its subtree.
                accepted.push(node.id);
                accepted_nodes.push(j);
                frontier = Some(j);
                current_target = target_next[j];
                descended = true;
                break;
            }
            // Reject: strike this sibling's mass from the residual pool so a
            // later replacement sample can't re-pick it.
            let r = residual.get_or_insert_with(|| current_target.to_vec());
            if id < r.len() {
                r[id] = (r[id] - q).max(0.0);
            }
        }

        if descended {
            continue;
        }

        // No child accepted at this frontier (every sibling rejected, or this
        // is a leaf of the accepted path with no children at all). Draw the
        // replacement / bonus from the residual pool (== `current_target` when
        // no sibling was tried, i.e. the leaf-bonus case).
        let replacement = match residual {
            Some(mut r) => {
                let sum: f32 = r.iter().sum();
                if sum > 0.0 {
                    let inv = 1.0 / sum;
                    for v in r.iter_mut() {
                        *v *= inv;
                    }
                    multinomial(&r, rng)
                } else {
                    // The governing distribution was a delta on the rejected
                    // candidate(s); fall back to its argmax (mirrors the linear
                    // path's degenerate branch).
                    argmax(current_target)
                }
            }
            None => multinomial(current_target, rng),
        };
        return TreeVerifyOutcome {
            accepted,
            accepted_nodes,
            replacement,
        };
    }
}

/// Exact temperature→0 limit of [`tree_accept_reject`] (the tree analogue of
/// [`accept_reject_greedy`]): at each frontier accept the child that IS the
/// governing distribution's argmax, else reject and replace with that argmax.
/// Fully deterministic. Reduces to [`accept_reject_greedy`] token-for-token for
/// a chain-shaped tree.
pub fn tree_accept_reject_greedy(
    nodes: &[TreeNode],
    target_root: &[f32],
    target_next: &[&[f32]],
) -> TreeVerifyOutcome {
    assert_eq!(
        target_next.len(),
        nodes.len(),
        "speculative: target_next must have one distribution per tree node"
    );
    let mut accepted: Vec<u32> = Vec::with_capacity(nodes.len());
    let mut accepted_nodes: Vec<usize> = Vec::with_capacity(nodes.len());
    let mut frontier: Option<usize> = None;
    let mut current_target: &[f32] = target_root;
    loop {
        let am = argmax(current_target);
        // Find the child (if any) matching the argmax; accept + descend.
        let hit = nodes
            .iter()
            .enumerate()
            .find(|(_, n)| n.parent == frontier && n.id == am);
        match hit {
            Some((j, node)) => {
                accepted.push(node.id);
                accepted_nodes.push(j);
                frontier = Some(j);
                current_target = target_next[j];
            }
            None => {
                return TreeVerifyOutcome {
                    accepted,
                    accepted_nodes,
                    replacement: am,
                };
            }
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
    // Fall back: return the last non-zero index.
    (probs
        .iter()
        .rposition(|p| *p > 0.0)
        .unwrap_or(0)) as u32
}

fn argmax(probs: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, v) in probs.iter().enumerate() {
        if *v > best_v {
            best_v = *v;
            best_i = i;
        }
    }
    best_i as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::Rng;

    fn mk_rng(seed: u64) -> Rng {
        Rng::from_seed(seed)
    }

    /// Greedy gate: a near-tie distribution (0.51 / 0.49) must ALWAYS
    /// resolve to the argmax — this is the exact case where the
    /// stochastic path flips by seed ("yellow" vs "green" on the real
    /// 27B). No RNG involved, so no seed sweep needed; the companion
    /// check below pins the reject behavior.
    #[test]
    fn greedy_gate_takes_argmax_on_near_ties() {
        // Draft proposes token 1; target argmax at position 0 is
        // token 0 by a hair. Greedy must reject and replace with 0.
        let drafts = vec![DraftToken { id: 1, q: 0.9 }];
        let p0 = [0.51f32, 0.49, 0.0, 0.0];
        let p1 = [0.0f32, 0.0, 1.0, 0.0];
        let out = accept_reject_greedy(&drafts, &[&p0, &p1]);
        assert!(out.accepted.is_empty(), "non-argmax draft must reject");
        assert_eq!(out.replacement, 0, "replacement must be the argmax");

        // Same shapes, draft IS the argmax → accept + argmax bonus.
        let drafts = vec![DraftToken { id: 0, q: 0.1 }];
        let out = accept_reject_greedy(&drafts, &[&p0, &p1]);
        assert_eq!(out.accepted, vec![0]);
        assert_eq!(out.replacement, 2, "bonus must be argmax of the extra slot");
    }

    /// Greedy gate over a longer draft run: acceptance stops at the
    /// FIRST non-argmax draft and later positions are never consulted.
    #[test]
    fn greedy_gate_stops_at_first_divergence() {
        let drafts = vec![
            DraftToken { id: 2, q: 0.5 },
            DraftToken { id: 3, q: 0.5 },
            DraftToken { id: 1, q: 0.5 }, // argmax at this position is 0
        ];
        let p = [
            &[0.0f32, 0.1, 0.8, 0.1][..],
            &[0.1f32, 0.0, 0.1, 0.8][..],
            &[0.8f32, 0.1, 0.1, 0.0][..],
            &[0.0f32, 1.0, 0.0, 0.0][..],
        ];
        let out = accept_reject_greedy(&drafts, &p);
        assert_eq!(out.accepted, vec![2, 3]);
        assert_eq!(out.replacement, 0);
    }

    #[test]
    fn accept_all_when_target_strongly_agrees() {
        // Draft proposed tokens 1, 2, 3 with low confidence (q=0.1).
        // The target distribution puts ~all its mass on those exact
        // tokens at those positions. `p / q = ~10` → clamp to 1.0
        // → accept every draft → bonus token from target_probs[3].
        let drafts = vec![
            DraftToken { id: 1, q: 0.1 },
            DraftToken { id: 2, q: 0.1 },
            DraftToken { id: 3, q: 0.1 },
        ];
        // 5-token vocab. Target puts 1.0 mass on the drafted token
        // at each position; bonus position puts mass on token 4.
        let p0 = vec![0.0, 1.0, 0.0, 0.0, 0.0];
        let p1 = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let p2 = vec![0.0, 0.0, 0.0, 1.0, 0.0];
        let p_bonus = vec![0.0, 0.0, 0.0, 0.0, 1.0];
        let target = [p0.as_slice(), p1.as_slice(), p2.as_slice(), p_bonus.as_slice()];

        let mut rng = mk_rng(0xC0FFEE);
        let out = accept_reject(&drafts, &target, &mut rng);
        assert_eq!(out.accepted, vec![1, 2, 3]);
        assert_eq!(out.replacement, 4);
    }

    #[test]
    fn reject_immediately_when_target_disagrees() {
        // Draft confidently picked token 1 (q=0.9), but the target
        // assigns zero mass to it. Ratio = 0 → always reject. The
        // residual is `[0, 0.0-0.9 → 0, p2, p3, p4]` (negative-mass
        // entry truncated to 0) renormalized. Since target_probs[0]
        // assigns zero to token 1, the residual concentrates on
        // tokens 2..4.
        let drafts = vec![DraftToken { id: 1, q: 0.9 }];
        let p0 = vec![0.0, 0.0, 0.5, 0.3, 0.2];
        let p_bonus = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let target = [p0.as_slice(), p_bonus.as_slice()];

        let mut rng = mk_rng(42);
        let out = accept_reject(&drafts, &target, &mut rng);
        assert!(out.accepted.is_empty(), "must reject the disagreeing draft");
        assert!(out.replacement != 1, "residual must pick something other than the rejected draft");
        // The replacement must come from {2, 3, 4} since p0 puts
        // zero mass on indices 0 and 1.
        assert!(matches!(out.replacement, 2 | 3 | 4));
    }

    #[test]
    fn partial_accept_then_reject() {
        // Draft: [1, 2, 3] with q=0.5 each. Target agrees on token
        // 1 (p=0.9 there), disagrees on token 2 (p=0.0 there). So
        // we accept index 0, reject at index 1, and the residual
        // sample replaces draft[1].
        let drafts = vec![
            DraftToken { id: 1, q: 0.5 },
            DraftToken { id: 2, q: 0.5 },
            DraftToken { id: 3, q: 0.5 },
        ];
        let p0 = vec![0.05, 0.9, 0.05, 0.0, 0.0]; // accept slot
        let p1 = vec![0.5, 0.0, 0.0, 0.5, 0.0]; // reject at index 2
        let p2 = vec![0.25, 0.25, 0.25, 0.25, 0.0];
        let p_bonus = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let target = [
            p0.as_slice(),
            p1.as_slice(),
            p2.as_slice(),
            p_bonus.as_slice(),
        ];

        let mut rng = mk_rng(7);
        let out = accept_reject(&drafts, &target, &mut rng);
        assert_eq!(out.accepted, vec![1], "exactly draft[0] should survive");
        // Replacement must come from the residual at position 1,
        // which is `max(0, p1 - q_at_2)` = `[0.5, 0, 0, 0.5, 0]`
        // renormalized → 50/50 between tokens 0 and 3.
        assert!(
            matches!(out.replacement, 0 | 3),
            "replacement must come from the residual support {{0, 3}}, got {}",
            out.replacement,
        );
    }

    #[test]
    fn determinism_under_same_seed() {
        // Two runs with the same RNG seed + same inputs must produce
        // the same SpeculationOutcome. Pins the spec-decoding
        // path against seed-leak regressions.
        let drafts = vec![
            DraftToken { id: 1, q: 0.4 },
            DraftToken { id: 2, q: 0.6 },
        ];
        let p0 = vec![0.0, 0.5, 0.3, 0.2];
        let p1 = vec![0.1, 0.0, 0.4, 0.5];
        let p_bonus = vec![0.25, 0.25, 0.25, 0.25];
        let target = [p0.as_slice(), p1.as_slice(), p_bonus.as_slice()];

        let mut r1 = mk_rng(0xDEAD_BEEF);
        let mut r2 = mk_rng(0xDEAD_BEEF);
        let out1 = accept_reject(&drafts, &target, &mut r1);
        let out2 = accept_reject(&drafts, &target, &mut r2);
        assert_eq!(out1, out2);
    }

    #[test]
    fn empty_drafts_returns_bonus_only() {
        // K=0 means "no speculation, just sample one token from
        // the target at the current position." Should still work.
        let drafts: Vec<DraftToken> = vec![];
        let p_bonus = vec![0.0, 0.0, 1.0, 0.0];
        let target = [p_bonus.as_slice()];
        let mut rng = mk_rng(1);
        let out = accept_reject(&drafts, &target, &mut rng);
        assert!(out.accepted.is_empty());
        assert_eq!(out.replacement, 2);
    }

    #[test]
    #[should_panic(expected = "target_probs must have one extra position")]
    fn missing_bonus_slot_panics() {
        // Caller bug: target_probs.len() must be drafts.len() + 1.
        // Pin the assertion message so a future loosening of the
        // contract gets caught.
        let drafts = vec![DraftToken { id: 0, q: 1.0 }];
        let p = vec![1.0];
        let target = [p.as_slice()]; // missing the bonus slot
        let mut rng = mk_rng(0);
        let _ = accept_reject(&drafts, &target, &mut rng);
    }

    // ---- Tree-structured verification (Feature 4a) ----

    /// Build the chain-shaped tree equivalent of a linear draft: node `j`'s
    /// parent is `j-1` (root for `j==0`).
    fn chain_tree(drafts: &[DraftToken]) -> Vec<TreeNode> {
        drafts
            .iter()
            .enumerate()
            .map(|(j, d)| TreeNode {
                id: d.id,
                q: d.q,
                parent: if j == 0 { None } else { Some(j - 1) },
            })
            .collect()
    }

    /// THE parity guard: over a chain-shaped tree, `tree_accept_reject` must be
    /// bit-for-bit identical to the linear `accept_reject` for the same seed —
    /// same accepted prefix AND same replacement — proving the tree walk's RNG
    /// draw sequence + residual math reduce exactly to the pinned linear path.
    #[test]
    fn tree_accept_reject_matches_linear_for_chain() {
        let drafts = vec![
            DraftToken { id: 1, q: 0.4 },
            DraftToken { id: 2, q: 0.55 },
            DraftToken { id: 3, q: 0.2 },
        ];
        // 5-token vocab; a mix of agreement/disagreement so the sweep exercises
        // accept-all, early-reject, and mid-chain-reject outcomes.
        let p0 = [0.1f32, 0.5, 0.2, 0.1, 0.1];
        let p1 = [0.2f32, 0.1, 0.4, 0.2, 0.1];
        let p2 = [0.3f32, 0.1, 0.1, 0.4, 0.1];
        let p3 = [0.25f32, 0.25, 0.25, 0.25, 0.0];
        let target = [p0.as_slice(), p1.as_slice(), p2.as_slice(), p3.as_slice()];

        let nodes = chain_tree(&drafts);
        let target_root = target[0];
        let target_next: Vec<&[f32]> = target[1..].to_vec();

        for seed in 0u64..512 {
            let mut r_lin = mk_rng(seed);
            let mut r_tree = mk_rng(seed);
            let lin = accept_reject(&drafts, &target, &mut r_lin);
            let tree = tree_accept_reject(&nodes, target_root, &target_next, &mut r_tree);
            assert_eq!(lin.accepted, tree.accepted, "chain accepted at seed {seed}");
            assert_eq!(lin.replacement, tree.replacement, "chain replacement at seed {seed}");
            // For a chain, each accepted token IS its own node index.
            let expect_nodes: Vec<usize> = (0..tree.accepted.len()).collect();
            assert_eq!(tree.accepted_nodes, expect_nodes, "chain node indices at seed {seed}");
        }
    }

    /// Greedy chain parity: the temp→0 tree walk must match `accept_reject_greedy`
    /// token-for-token over the equivalent chain.
    #[test]
    fn tree_greedy_matches_linear_for_chain() {
        let drafts = vec![
            DraftToken { id: 2, q: 0.5 },
            DraftToken { id: 3, q: 0.5 },
            DraftToken { id: 1, q: 0.5 },
        ];
        let p = [
            &[0.0f32, 0.1, 0.8, 0.1][..],
            &[0.1f32, 0.0, 0.1, 0.8][..],
            &[0.8f32, 0.1, 0.1, 0.0][..],
            &[0.0f32, 1.0, 0.0, 0.0][..],
        ];
        let nodes = chain_tree(&drafts);
        let target_next: Vec<&[f32]> = p[1..].to_vec();
        let lin = accept_reject_greedy(&drafts, &p);
        let tree = tree_accept_reject_greedy(&nodes, p[0], &target_next);
        assert_eq!(lin.accepted, tree.accepted);
        assert_eq!(lin.replacement, tree.replacement);
    }

    /// Branching walk (greedy, fully deterministic): with two depth-1 candidates
    /// the walk must descend the argmax child — NOT just the first in slice
    /// order — then continue down its subtree to a leaf + bonus.
    #[test]
    fn tree_greedy_walks_best_branch() {
        // Root children: node0=id1 (NOT argmax), node1=id2 (argmax). Then
        // node1's child node2=id5 (argmax of target_next[1]). node2 is a leaf.
        let nodes = vec![
            TreeNode { id: 1, q: 0.5, parent: None },
            TreeNode { id: 2, q: 0.5, parent: None },
            TreeNode { id: 5, q: 0.5, parent: Some(1) },
        ];
        let target_root = [0.1f32, 0.2, 0.7, 0.0, 0.0, 0.0]; // argmax = 2 → node1
        // target_next indexed by node: [node0, node1, node2].
        let tn0 = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0]; // unused (node0 not taken)
        let tn1 = [0.0f32, 0.0, 0.0, 0.0, 0.0, 1.0]; // argmax = 5 → node2
        let tn2 = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0]; // leaf bonus → argmax = 3
        let target_next = [tn0.as_slice(), tn1.as_slice(), tn2.as_slice()];
        let out = tree_accept_reject_greedy(&nodes, &target_root, &target_next);
        assert_eq!(out.accepted, vec![2, 5], "must walk the argmax branch");
        assert_eq!(out.replacement, 3, "leaf bonus = argmax(target_next[leaf])");
    }

    /// A rejected sibling's mass is struck from the residual: when every depth-1
    /// candidate rejects, the replacement is drawn from the renormalized pool and
    /// can't be a rejected candidate's token.
    #[test]
    fn tree_stochastic_rejects_all_siblings_then_resamples() {
        // Two root candidates id1, id2; target_root puts ZERO mass on both, so
        // both always reject (ratio 0). Residual = target_root with 1,2 struck
        // (already zero) → support {0, 3} → replacement ∈ {0, 3}, never 1 or 2.
        let nodes = vec![
            TreeNode { id: 1, q: 0.9, parent: None },
            TreeNode { id: 2, q: 0.9, parent: None },
        ];
        let target_root = [0.5f32, 0.0, 0.0, 0.5];
        let tn = [1.0f32, 0.0, 0.0, 0.0];
        let target_next = [tn.as_slice(), tn.as_slice()];
        for seed in 0u64..64 {
            let mut rng = mk_rng(seed);
            let out = tree_accept_reject(&nodes, &target_root, &target_next, &mut rng);
            assert!(out.accepted.is_empty(), "both siblings must reject at seed {seed}");
            assert!(
                matches!(out.replacement, 0 | 3),
                "replacement from residual support {{0,3}}, got {} at seed {seed}",
                out.replacement
            );
        }
    }

    /// Empty tree ⇒ no speculation: draw one token from the root distribution.
    #[test]
    fn tree_empty_returns_root_sample() {
        let nodes: Vec<TreeNode> = vec![];
        let target_root = [0.0f32, 0.0, 1.0, 0.0];
        let target_next: Vec<&[f32]> = vec![];
        let mut rng = mk_rng(5);
        let out = tree_accept_reject(&nodes, &target_root, &target_next, &mut rng);
        assert!(out.accepted.is_empty());
        assert_eq!(out.replacement, 2);
        // Greedy twin: argmax of the root.
        let out_g = tree_accept_reject_greedy(&nodes, &target_root, &target_next);
        assert_eq!(out_g.replacement, 2);
    }
}

// ============================================================
// N-gram drafter (Prompt Lookup Decoding)
// ============================================================
//
// "Speculative" decoding without a second model. The drafter scans
// the existing token history for an n-gram match against the last
// `n - 1` tokens and proposes the K tokens that followed that
// n-gram. Mechanically free at inference time (just a hash lookup
// + clone) and surprisingly effective on coding workloads where
// recent context heavily repeats: variable names, function calls,
// language keywords, copy-and-modify patterns.
//
// This module ships the drafter in isolation. The full engine
// integration (multi-position model forward + accept/reject driver)
// is the next item — splitting the work this way lets us pin the
// drafter math via unit tests before the model surface gets
// touched. The drafter outputs `Vec<DraftToken>` which is exactly
// what [`accept_reject`] expects.

/// Configuration knobs for [`NgramDrafter`].
#[derive(Debug, Clone, Copy)]
pub struct NgramDrafterConfig {
    /// Match-window size. The drafter looks at the last `n_match`
    /// tokens of the history and searches for an identical
    /// sub-sequence earlier in the history. Default `3` —
    /// surprisingly good for code (matches "def name(", " return ",
    /// etc.) without ballooning lookup time.
    pub n_match: usize,
    /// Maximum number of speculative tokens to propose per step.
    /// Capped by both the rest-of-history available after the match
    /// and this value. Default `4` — Chen et al.'s standard.
    /// Raising it increases the per-step gain on a successful
    /// speculation at the cost of wasted compute on a rejection.
    pub n_draft: usize,
    /// Probability assigned to each drafted token under the
    /// drafter's "distribution". A real model draft gives a true
    /// posterior; the n-gram drafter has no probability semantics,
    /// so we pick a constant `q` that the accept/reject step
    /// interprets sensibly. Setting `q < 1.0` means the
    /// `p / q` ratio in [`accept_reject`] always clamps to 1.0 when
    /// the target agrees (target's `p` for a sane token is > q),
    /// so n-gram matches accept whenever the target is even
    /// mildly aligned. Default `0.5` — accepts confidently when
    /// the target agrees, rejects cleanly when it doesn't.
    pub q: f32,
}

impl Default for NgramDrafterConfig {
    fn default() -> Self {
        Self {
            n_match: 3,
            n_draft: 4,
            q: 0.5,
        }
    }
}

/// Prompt-lookup-decoding drafter.
///
/// Given a token history `hist`, scans backward (newest-match-wins)
/// for the longest occurrence of the last `n_match` tokens earlier
/// in the history. When found, proposes the up-to-`n_draft` tokens
/// that followed that earlier match.
///
/// The implementation is intentionally simple — a linear scan from
/// `len(hist) - n_match - 1` down to `0`. For typical prompt lengths
/// (≤ 32K tokens) this is microsecond-scale, dominated by L1 cache
/// behavior. A future enhancement would build a suffix-array or
/// rolling-hash index over the history; for v1 the linear scan is
/// fast enough that the lookup cost is invisible against decode.
#[derive(Debug, Clone)]
pub struct NgramDrafter {
    cfg: NgramDrafterConfig,
}

impl NgramDrafter {
    pub fn new(cfg: NgramDrafterConfig) -> Self {
        Self { cfg }
    }

    pub fn with_defaults() -> Self {
        Self::new(NgramDrafterConfig::default())
    }

    pub fn config(&self) -> NgramDrafterConfig {
        self.cfg
    }

    /// Propose up to `n_draft` speculative tokens from `hist`.
    /// Returns an empty `Vec` when no match exists (caller proceeds
    /// to the regular single-token decode path — the engine should
    /// degrade gracefully when the drafter finds nothing).
    ///
    /// The history must include at least `n_match + 1` tokens for a
    /// useful lookup. Shorter histories return empty.
    pub fn propose(&self, hist: &[u32]) -> Vec<DraftToken> {
        let n_match = self.cfg.n_match;
        let n_draft = self.cfg.n_draft;
        if n_match == 0 || n_draft == 0 {
            return Vec::new();
        }
        if hist.len() <= n_match {
            return Vec::new();
        }
        // The query is the trailing `n_match` tokens of `hist`. We
        // search for an earlier occurrence whose `n_match`-token
        // suffix matches the query — the tokens following that
        // earlier suffix are our draft proposals.
        let q_start = hist.len() - n_match;
        let query = &hist[q_start..];
        // Newest-match-wins: scan from just-before-query down to 0.
        // `start` is the index where the candidate `n_match`-token
        // window begins; we never check the query's own location
        // (`q_start`) since that would propose the query as its own
        // draft.
        let mut start = q_start;
        while start > 0 {
            start -= 1;
            if start + n_match > q_start {
                continue;
            }
            if &hist[start..start + n_match] == query {
                // Found a match. Drafts are the tokens after the
                // match, capped at n_draft AND the remaining history
                // length. Skip the query's own position by clamping
                // the end to `q_start` — drafting past the query
                // would just propose the query's own tokens.
                let draft_start = start + n_match;
                let draft_end = (draft_start + n_draft).min(q_start);
                if draft_end <= draft_start {
                    return Vec::new();
                }
                return hist[draft_start..draft_end]
                    .iter()
                    .map(|&id| DraftToken { id, q: self.cfg.q })
                    .collect();
            }
        }
        Vec::new()
    }
}

// ============================================================
// E4 phase 4c: MtpDrafter — DeepSeek-V3-style multi-token
// prediction as a drafter.
// ============================================================
//
// Unlike `NgramDrafter` (pure-history lookup, no model
// dependency), the MTP drafter consumes the per-head logits the
// main model produces from a single forward pass via
// [`crate::Engine::forward_one_with_mtp_logits`] (or the
// underlying [`rustllama_models::llama_arch::LlamaModel`] method
// of the same name). One head ⇒ one draft token.
//
// The bridge keeps this module pure-data: the engine layer runs
// the forward + passes the resulting `Vec<Vec<f32>>` of head
// logits into [`MtpDrafter::propose`], which returns the same
// `Vec<DraftToken>` shape `NgramDrafter` does. The downstream
// `accept_reject` is identical.

/// Sampling mode for MTP head logits → draft tokens.
#[derive(Debug, Clone, Copy)]
pub enum MtpSampleMode {
    /// Argmax: each head's top-1 token. Deterministic; pairs
    /// well with the existing `verify_speculation` path's
    /// argmax-based commit logic. Default.
    Argmax,
}

/// Stateless drafter that turns per-MTP-head logits into the
/// `Vec<DraftToken>` shape `accept_reject` consumes. Stateless
/// because the model's `forward_one_with_mtp_logits` does all the
/// per-call work — this struct only encapsulates the sampling
/// policy.
#[derive(Debug, Clone, Copy)]
pub struct MtpDrafter {
    pub mode: MtpSampleMode,
}

impl Default for MtpDrafter {
    fn default() -> Self {
        Self { mode: MtpSampleMode::Argmax }
    }
}

impl MtpDrafter {
    pub fn new(mode: MtpSampleMode) -> Self {
        Self { mode }
    }

    /// Convert per-head logits into draft tokens. `head_logits[k]`
    /// is the full-vocab logit row from the k-th MTP head; the
    /// drafter samples one token per head per the chosen mode and
    /// attaches the head's softmax probability as the `q` field
    /// (drives the `p / q` acceptance ratio in `accept_reject`).
    ///
    /// Returns an empty `Vec` when `head_logits` is empty — the
    /// engine driver should fall back to the single-token decode
    /// path in that case (no MTP signal).
    pub fn propose(&self, head_logits: &[Vec<f32>]) -> Vec<DraftToken> {
        if head_logits.is_empty() {
            return Vec::new();
        }
        let mut drafts = Vec::with_capacity(head_logits.len());
        for logits in head_logits {
            if logits.is_empty() {
                break;
            }
            match self.mode {
                MtpSampleMode::Argmax => {
                    let (idx, q) = argmax_and_softmax_p(logits);
                    drafts.push(DraftToken { id: idx as u32, q });
                }
            }
        }
        drafts
    }
}

/// Returns `(argmax_idx, softmax_prob_at_argmax)`. The softmax is
/// numerically-stable (subtract max first); the returned prob is
/// in `(0, 1]` and is exactly the value the accept/reject ratio
/// `p / q` divides by.
fn argmax_and_softmax_p(logits: &[f32]) -> (usize, f32) {
    debug_assert!(!logits.is_empty());
    let mut max_idx = 0usize;
    let mut max_val = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    let mut denom: f32 = 0.0;
    for &v in logits {
        denom += (v - max_val).exp();
    }
    // denom >= 1 (the max term contributes 1.0 exactly), so 1/denom ∈ (0, 1].
    let p = 1.0_f32 / denom.max(1e-30);
    (max_idx, p.clamp(1e-30, 1.0))
}

/// Build a speculative draft TREE from a model's MTP-head logits (the tree
/// analogue of [`MtpDrafter::propose`]). `head_logits[d]` is head `d`'s
/// full-vocab logit row — head `d` independently predicts the token `d+1`
/// positions ahead. The tree is a "caterpillar": a main chain down the per-head
/// argmax, with `widths[d]-1` alternative siblings branching off the main node
/// at each depth. Only the main (argmax) node of each level gets children, so
/// the node count is `sum(widths)` — compact, and `width == 1` at every depth
/// reduces EXACTLY to the linear [`MtpDrafter`] chain (pinned by
/// `mtp_tree_width_one_is_a_chain`). Each node's `q` is the head's softmax
/// probability at that candidate (the `p/q` acceptance denominator).
pub fn build_mtp_tree(head_logits: &[Vec<f32>], widths: &[usize]) -> Vec<TreeNode> {
    let mut nodes: Vec<TreeNode> = Vec::new();
    // Parent for the NEXT level's nodes = this level's main (argmax) node.
    let mut main_parent: Option<usize> = None;
    for (d, logits) in head_logits.iter().enumerate() {
        if logits.is_empty() {
            break;
        }
        let w = widths.get(d).copied().unwrap_or(1).max(1);
        let topk = top_k_softmax(logits, w);
        if topk.is_empty() {
            break;
        }
        let level_start = nodes.len();
        for (id, q) in topk {
            nodes.push(TreeNode { id, q, parent: main_parent });
        }
        // The argmax candidate is pushed first, so the main node is level_start.
        main_parent = Some(level_start);
    }
    nodes
}

/// Per-node tree depth (the RoPE-position offset): `0` for a root-level node,
/// else `depth(parent) + 1`. Topological order (`parent < self`) lets one pass
/// fill it. Feeds `forward_tree_verify_f32`'s `depths` argument.
pub fn tree_depths(nodes: &[TreeNode]) -> Vec<u32> {
    let mut depths = vec![0u32; nodes.len()];
    for (i, n) in nodes.iter().enumerate() {
        depths[i] = match n.parent {
            Some(p) => depths[p] + 1,
            None => 0,
        };
    }
    depths
}

/// Top-`k` `(token_id, softmax_prob)` by logit, highest first. Softmax is over
/// the FULL row (numerically stable), so `prob` matches [`argmax_and_softmax_p`]
/// exactly at `k == 1` (the chain-reduction relies on this). `k` is clamped to
/// the vocab size.
fn top_k_softmax(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    if logits.is_empty() || k == 0 {
        return Vec::new();
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denom: f32 = logits.iter().map(|&v| (v - max_val).exp()).sum::<f32>().max(1e-30);
    let inv = 1.0 / denom;
    let kk = k.min(logits.len());
    // Small k over a big vocab → a k-pass selection beats a full sort.
    let mut used = vec![false; logits.len()];
    let mut out: Vec<(u32, f32)> = Vec::with_capacity(kk);
    for _ in 0..kk {
        let mut best = usize::MAX;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if !used[i] && v > best_v {
                best_v = v;
                best = i;
            }
        }
        if best == usize::MAX {
            break;
        }
        used[best] = true;
        out.push((best as u32, ((best_v - max_val).exp() * inv).clamp(1e-30, 1.0)));
    }
    out
}

#[cfg(test)]
mod mtp_drafter_tests {
    use super::*;

    #[test]
    fn empty_head_logits_yield_no_drafts() {
        let d = MtpDrafter::default();
        assert!(d.propose(&[]).is_empty());
    }

    #[test]
    fn argmax_picks_max_logit_per_head() {
        let d = MtpDrafter::default();
        let heads = vec![
            vec![0.0, 1.0, 0.5, -1.0],          // argmax = 1
            vec![3.0, -2.0, 5.0, 4.9, 0.0],     // argmax = 2
            vec![-1.0, -2.0, -3.0, -4.0],       // argmax = 0
        ];
        let drafts = d.propose(&heads);
        assert_eq!(drafts.len(), 3);
        assert_eq!(drafts[0].id, 1);
        assert_eq!(drafts[1].id, 2);
        assert_eq!(drafts[2].id, 0);
        // softmax probabilities are positive and ≤ 1.
        for d in &drafts {
            assert!(d.q > 0.0 && d.q <= 1.0, "q out of range: {}", d.q);
        }
    }

    #[test]
    fn softmax_q_matches_hand_computed_value() {
        let d = MtpDrafter::default();
        // logits = [1.0, 2.0, 3.0]; argmax = 2.
        // softmax(2.0 - 3.0) + softmax(1.0 - 3.0) + softmax(0) over the
        // shifted denom: e^-2 + e^-1 + e^0 ≈ 0.1353 + 0.3679 + 1.0
        // = 1.5032. p = 1/1.5032 ≈ 0.6652.
        let drafts = d.propose(&[vec![1.0, 2.0, 3.0]]);
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].id, 2);
        let expected = 1.0_f32 / (((-2.0_f32).exp()) + ((-1.0_f32).exp()) + 1.0);
        assert!(
            (drafts[0].q - expected).abs() < 1e-5,
            "q = {}, expected ~{}",
            drafts[0].q,
            expected
        );
    }

    #[test]
    fn empty_inner_logit_row_stops_drafting() {
        // Defensive: if a head emits an empty logit row (corrupt
        // weights, mis-sized buffer), the drafter stops there
        // rather than panicking.
        let d = MtpDrafter::default();
        let heads = vec![vec![0.1, 0.2, 0.3], vec![], vec![1.0]];
        let drafts = d.propose(&heads);
        assert_eq!(drafts.len(), 1, "stops at the first empty head");
    }

    // ---- tree drafter (Feature 4b) ----

    /// Degenerate guard: `build_mtp_tree` with width 1 at every depth is the
    /// linear `MtpDrafter` chain — same ids, same `q`, chained parents, depths
    /// `0,1,2,…`. So tree spec at width 1 == the existing linear MTP spec.
    #[test]
    fn mtp_tree_width_one_is_a_chain() {
        let heads = vec![
            vec![0.0, 1.0, 0.5, -1.0],
            vec![3.0, -2.0, 5.0, 4.9, 0.0],
            vec![-1.0, -2.0, -3.0, -4.0],
        ];
        let chain = MtpDrafter::default().propose(&heads);
        let nodes = build_mtp_tree(&heads, &[1, 1, 1]);
        assert_eq!(nodes.len(), chain.len());
        for (d, (node, draft)) in nodes.iter().zip(chain.iter()).enumerate() {
            assert_eq!(node.id, draft.id, "level {d} id");
            assert!((node.q - draft.q).abs() < 1e-6, "level {d} q");
            assert_eq!(node.parent, if d == 0 { None } else { Some(d - 1) }, "level {d} parent");
        }
        assert_eq!(tree_depths(&nodes), vec![0, 1, 2]);
    }

    /// Branching: width 2 at the first two depths yields a caterpillar — 2
    /// root-level candidates, and the main (argmax) root node gets 2 children.
    #[test]
    fn mtp_tree_branches_with_width() {
        let heads = vec![
            vec![0.0, 1.0, 0.5, -1.0], // argmax 1, runner-up 2
            vec![5.0, 0.0, 4.0, 0.0],  // argmax 0, runner-up 2
        ];
        let nodes = build_mtp_tree(&heads, &[2, 2]);
        assert_eq!(nodes.len(), 4, "sum(widths) nodes");
        // Level 0: nodes 0,1 at root (parent None); node 0 is the argmax (id 1).
        assert_eq!(nodes[0].parent, None);
        assert_eq!(nodes[1].parent, None);
        assert_eq!(nodes[0].id, 1, "argmax first");
        assert_eq!(nodes[1].id, 2, "runner-up second");
        // Level 1: nodes 2,3 are children of the main level-0 node (index 0).
        assert_eq!(nodes[2].parent, Some(0));
        assert_eq!(nodes[3].parent, Some(0));
        assert_eq!(nodes[2].id, 0, "level-1 argmax");
        assert_eq!(tree_depths(&nodes), vec![0, 0, 1, 1]);
    }

    #[test]
    fn top_k_softmax_orders_desc_and_sums_sane() {
        let logits = vec![1.0, 3.0, 2.0, 0.0];
        let top = top_k_softmax(&logits, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 1, "highest logit first");
        assert_eq!(top[1].0, 2);
        assert_eq!(top[2].0, 0);
        // probs descending, each in (0,1].
        assert!(top[0].1 >= top[1].1 && top[1].1 >= top[2].1);
        for (_, q) in &top {
            assert!(*q > 0.0 && *q <= 1.0);
        }
        // k==1 matches argmax_and_softmax_p exactly (the chain-reduction basis).
        let (amax, aq) = argmax_and_softmax_p(&logits);
        let t1 = top_k_softmax(&logits, 1);
        assert_eq!(t1[0].0 as usize, amax);
        assert!((t1[0].1 - aq).abs() < 1e-7);
    }
}

#[cfg(test)]
mod ngram_tests {
    use super::*;

    #[test]
    fn empty_history_yields_no_draft() {
        let d = NgramDrafter::with_defaults();
        assert!(d.propose(&[]).is_empty());
    }

    #[test]
    fn history_shorter_than_n_match_yields_no_draft() {
        let d = NgramDrafter::new(NgramDrafterConfig {
            n_match: 3,
            ..Default::default()
        });
        assert!(d.propose(&[10, 20]).is_empty());
    }

    #[test]
    fn proposes_continuation_when_recent_ngram_repeats() {
        // History: "def foo ( x ) : return x def foo ( x ) :"
        // The query is the last `n_match=3` tokens "def foo (". An
        // earlier occurrence at index 0..3 was followed by "x ) :
        // return x". With n_draft=4 we expect those 4 follow-up
        // tokens.
        let hist = vec![
            10, // def
            20, // foo
            30, // (
            40, // x
            50, // )
            60, // :
            70, // return
            40, // x
            10, // def
            20, // foo
            30, // (
        ];
        let d = NgramDrafter::new(NgramDrafterConfig {
            n_match: 3,
            n_draft: 4,
            q: 0.5,
        });
        let drafts = d.propose(&hist);
        let ids: Vec<u32> = drafts.iter().map(|d| d.id).collect();
        assert_eq!(ids, vec![40, 50, 60, 70]);
        // All drafts carry the configured q.
        assert!(drafts.iter().all(|d| (d.q - 0.5).abs() < 1e-9));
    }

    #[test]
    fn drafts_clamp_to_remaining_history() {
        // History: "a b c X Y a b c". Query "a b c"; earlier match
        // is at 0..3, followed by "X Y" — only 2 tokens before the
        // query starts. n_draft=4 should clamp to those 2 tokens.
        let hist = vec![1, 2, 3, 10, 11, 1, 2, 3];
        let d = NgramDrafter::new(NgramDrafterConfig {
            n_match: 3,
            n_draft: 4,
            q: 0.5,
        });
        let ids: Vec<u32> = d.propose(&hist).iter().map(|d| d.id).collect();
        assert_eq!(ids, vec![10, 11]);
    }

    #[test]
    fn no_match_yields_empty() {
        // "a b c d e" — no repeat, no draft.
        let hist = vec![1, 2, 3, 4, 5];
        let d = NgramDrafter::new(NgramDrafterConfig {
            n_match: 2,
            n_draft: 3,
            q: 0.5,
        });
        assert!(d.propose(&hist).is_empty());
    }

    #[test]
    fn newest_match_wins() {
        // Two earlier matches of "X Y": at index 0..2 (followed by
        // 100) and at index 5..7 (followed by 200). The newest
        // earlier match wins, so we draft 200 first.
        let hist = vec![
            10, 20, // X Y    (earlier-match-A)
            100, // follow-A
            30, 40, //  filler
            10, 20, // X Y    (earlier-match-B, newer)
            200, // follow-B
            50, // filler
            10, 20, // X Y    (query)
        ];
        let d = NgramDrafter::new(NgramDrafterConfig {
            n_match: 2,
            n_draft: 1,
            q: 0.5,
        });
        let ids: Vec<u32> = d.propose(&hist).iter().map(|d| d.id).collect();
        assert_eq!(ids, vec![200], "newer match should win over older");
    }

    #[test]
    fn match_must_not_overlap_query_position() {
        // History: "a b c a b c". Query is the last "a b c"; the
        // earlier occurrence at index 0..3 IS the right answer.
        // The drafter must not "match" the query against itself.
        let hist = vec![1, 2, 3, 1, 2, 3];
        let d = NgramDrafter::new(NgramDrafterConfig {
            n_match: 3,
            n_draft: 2,
            q: 0.5,
        });
        // No tokens follow the query, and the only earlier match
        // (at 0..3) is followed by the query itself. The drafter
        // clamps to before the query.
        let ids: Vec<u32> = d.propose(&hist).iter().map(|d| d.id).collect();
        assert!(
            ids.is_empty(),
            "must not propose the query's own tokens as drafts: got {ids:?}"
        );
    }

    #[test]
    fn proposals_feed_accept_reject_cleanly() {
        // End-to-end: drafter produces tokens, accept_reject
        // consumes them with a synthetic target distribution that
        // strongly agrees with the drafter. Verifies the
        // `DraftToken` shape from the drafter is the same shape
        // `accept_reject` expects (no struct/field mismatch).
        let hist = vec![10, 20, 30, 40, 50, 10, 20, 30];
        let drafter = NgramDrafter::new(NgramDrafterConfig {
            n_match: 3,
            n_draft: 2,
            q: 0.5,
        });
        let drafts = drafter.propose(&hist);
        assert_eq!(drafts.len(), 2);
        // Target puts ~all mass on the drafter's proposed ids at
        // each position. With q=0.5, p/q clamps to 1.0 and every
        // draft accepts.
        let vocab = 64usize;
        let mut p_vecs: Vec<Vec<f32>> = Vec::with_capacity(drafts.len() + 1);
        for d in &drafts {
            let mut v = vec![0f32; vocab];
            v[d.id as usize] = 1.0;
            p_vecs.push(v);
        }
        // Bonus slot: any valid distribution.
        let mut bonus = vec![0f32; vocab];
        bonus[7] = 1.0;
        p_vecs.push(bonus);
        let target_refs: Vec<&[f32]> = p_vecs.iter().map(|v| v.as_slice()).collect();
        let mut rng = Rng::from_seed(42);
        let outcome = accept_reject(&drafts, &target_refs, &mut rng);
        // All accepted + bonus token = 3 commits.
        assert_eq!(outcome.accepted.len(), drafts.len());
        assert_eq!(outcome.accepted, vec![drafts[0].id, drafts[1].id]);
        assert_eq!(outcome.replacement, 7);
    }
}
