//! X1+D5: one place that applies the autotuned speculative-decoding +
//! CacheBlend winners to a loaded [`ContiguousEngine`].
//!
//! Before this, the `serve` CLI load path and the HTTP model-load path each
//! applied these winners with their own hand-rolled `cpu.set_*` calls — and
//! they had DRIFTED: `serve` applied draft / n-gram / CacheBlend but not
//! tree-speculation; the HTTP path applied tree-speculation but not n-gram or
//! CacheBlend. A model loaded over HTTP therefore behaved differently from the
//! same model served from the CLI. Collecting the winners into one struct and
//! one [`TunerWinners::apply_to`] makes both paths apply the identical set.
//!
//! The struct holds already-RESOLVED values: each caller reads the tuner cache
//! (keyed by `system_fingerprint()` + model) with its own helper and fills the
//! fields; `apply_to` only pushes them onto the engine. That keeps the
//! cache-reading where it already lives while unifying the application — the
//! part that had diverged.

use std::sync::Arc;

use crate::speculative::{LookaheadDrafterConfig, NgramDrafterConfig};
use crate::{ContiguousEngine, Engine};

/// The resolved autotuner (+ config) winners that get pushed onto a freshly
/// loaded engine. Every field is the final value to apply — `None` / `false`
/// means "leave the engine's default" (except `mtp`, which is always set, to
/// mirror both legacy paths' unconditional `set_mtp_speculative`).
#[derive(Default)]
pub struct TunerWinners {
    /// Two-model (draft) speculative decoding: the already-loaded, already
    /// compatibility-checked draft engine + its draft length `k`. `None` when
    /// no draft is configured (the HTTP load path never configures one).
    pub draft: Option<(Arc<dyn Engine>, u32)>,
    /// N-gram speculative drafter config. `None` ⇒ off.
    pub ngram: Option<NgramDrafterConfig>,
    /// F10 — lookahead (Jacobi n-gram-pool) drafter config. `None` ⇒ off.
    /// Takes dispatch precedence over `ngram` when both are set.
    pub lookahead: Option<LookaheadDrafterConfig>,
    /// MTP / NextN self-speculation on/off (a no-op on models without an MTP
    /// head, so always applied). Resolved from the `speculative_mtp` winner.
    pub mtp: bool,
    /// Tree-structured MTP speculation branching widths. `Some` ⇒ enable with
    /// these widths; `None` ⇒ leave off.
    pub tree_widths: Option<Vec<u32>>,
    /// Per-model best-drafter winner from the `tune --spec-strategy` sweep. `Some`
    /// ⇒ MUTUAL-EXCLUSION mode: enable EXACTLY the winning self-contained drafter
    /// (n-gram / lookahead / MTP / tree) and DISABLE the other three — the one
    /// thing a precedence-only dispatch can't do (a config that turns two drafters
    /// on would always run the higher-precedence one even if the other measured
    /// faster). `None` ⇒ legacy per-field behavior (each `Some`/`mtp` applied
    /// independently). Spec decode is lossless, so this only changes SPEED.
    ///
    /// The winner governs only the four SELF-CONTAINED drafters the sweep measures.
    /// `draft` and an attached EAGLE head are deliberate user choices the sweep
    /// can't load/measure and that take dispatch precedence, so they keep their own
    /// field (`draft`) / loaded state and are NOT cleared here.
    pub spec_strategy: Option<rustllama_tuner::SpecStrategy>,
    /// CacheBlend recompute ratio for RAG `documents` requests. `Some(0.0)`
    /// means "ran the stage, keep it off"; `None` ⇒ leave the engine default.
    pub cacheblend_ratio: Option<f32>,
}

impl TunerWinners {
    /// Push every resolved winner onto `cpu`. The ONE place the speculative /
    /// CacheBlend `set_*` toggles are called from — the anchor for X1+D5.
    pub fn apply_to(&self, cpu: &mut ContiguousEngine) {
        // Draft + EAGLE are governed by their own field / loaded state and take
        // dispatch precedence (`eagle > tree > MTP > lookahead > n-gram`): an
        // explicitly configured companion the spec-strategy sweep can't measure
        // stays as the user set it, in both modes.
        if let Some((engine, k)) = &self.draft {
            cpu.set_draft_speculative(Some((engine.clone(), *k)));
        }

        match self.spec_strategy {
            // Per-model best-drafter winner: enable EXACTLY the winning
            // self-contained drafter and disable the other three. Falls back to the
            // measurement's own default config when the field wasn't separately
            // populated (the winner names the strategy, not its knobs).
            Some(strategy) => {
                use rustllama_tuner::SpecStrategy;
                let (ngram, lookahead, mtp, tree) = match strategy {
                    SpecStrategy::None => (None, None, false, None),
                    SpecStrategy::Ngram => {
                        (Some(self.ngram.unwrap_or_default()), None, false, None)
                    }
                    SpecStrategy::Lookahead => {
                        (None, Some(self.lookahead.unwrap_or_default()), false, None)
                    }
                    SpecStrategy::Mtp => (None, None, true, None),
                    SpecStrategy::Tree => (
                        None,
                        None,
                        false,
                        Some(self.tree_widths.clone().unwrap_or_else(|| vec![2, 2, 2, 2])),
                    ),
                    // Never chosen by the sweep (Eagle/Draft are omitted
                    // candidates); keep the self-contained drafters off and let the
                    // attached companion (own field) + precedence decide.
                    SpecStrategy::Eagle | SpecStrategy::Draft => (None, None, false, None),
                };
                cpu.set_ngram_speculative(ngram);
                cpu.set_lookahead_speculative(lookahead);
                cpu.set_mtp_speculative(mtp);
                cpu.set_tree_speculative(tree);
            }
            // Legacy per-field behavior (no per-model winner cached / auto-apply
            // off): apply each drafter field independently, as before.
            None => {
                if let Some(ngram) = &self.ngram {
                    cpu.set_ngram_speculative(Some(ngram.clone()));
                }
                if let Some(lookahead) = &self.lookahead {
                    cpu.set_lookahead_speculative(Some(*lookahead));
                }
                cpu.set_mtp_speculative(self.mtp);
                if let Some(widths) = &self.tree_widths {
                    cpu.set_tree_speculative(Some(widths.clone()));
                }
            }
        }

        if let Some(ratio) = self.cacheblend_ratio {
            cpu.set_cacheblend(Some(ratio));
        }
    }
}
