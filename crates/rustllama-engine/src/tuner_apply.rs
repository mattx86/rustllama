//! X1+D5: one place that applies the autotuned speculative-decoding +
//! CacheBlend winners to a loaded [`CpuEngine`].
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

use crate::speculative::NgramDrafterConfig;
use crate::{CpuEngine, Engine};

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
    /// MTP / NextN self-speculation on/off (a no-op on models without an MTP
    /// head, so always applied). Resolved from the `speculative_mtp` winner.
    pub mtp: bool,
    /// Tree-structured MTP speculation branching widths. `Some` ⇒ enable with
    /// these widths; `None` ⇒ leave off.
    pub tree_widths: Option<Vec<u32>>,
    /// CacheBlend recompute ratio for RAG `documents` requests. `Some(0.0)`
    /// means "ran the stage, keep it off"; `None` ⇒ leave the engine default.
    pub cacheblend_ratio: Option<f32>,
}

impl TunerWinners {
    /// Push every resolved winner onto `cpu`. The ONE place the speculative /
    /// CacheBlend `set_*` toggles are called from — the anchor for X1+D5.
    pub fn apply_to(&self, cpu: &mut CpuEngine) {
        if let Some((engine, k)) = &self.draft {
            cpu.set_draft_speculative(Some((engine.clone(), *k)));
        }
        if let Some(ngram) = &self.ngram {
            cpu.set_ngram_speculative(Some(ngram.clone()));
        }
        cpu.set_mtp_speculative(self.mtp);
        if let Some(widths) = &self.tree_widths {
            cpu.set_tree_speculative(Some(widths.clone()));
        }
        if let Some(ratio) = self.cacheblend_ratio {
            cpu.set_cacheblend(Some(ratio));
        }
    }
}
