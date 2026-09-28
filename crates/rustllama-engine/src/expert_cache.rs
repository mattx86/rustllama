//! Learning-cache policy for the MoE expert-pin cache.
//!
//! The *mechanism* — VirtualLock residency, LRU eviction, per-expert
//! access counts, the prefetch registry — lives in
//! [`rustllama_models::accel`] (process-global, consulted by the MoE
//! forward pass on every routed expert). This module is the *policy*
//! side the engine drives around it:
//!
//! 1. **Usage sidecar** — routing frequencies are persisted to a
//!    `<gguf>.rlusage` file next to the model. Counts are merged with
//!    a per-session decay (never replaced), so the ranking learns
//!    across sessions while old habits fade.
//! 2. **Load-time pre-pin** — at engine load the hottest experts from
//!    the sidecar are pinned up to a fraction of the cache budget,
//!    *before* the first prefill, so even the first request of a
//!    session hits a warm cache. Colibrì measured a 28%→66% hit-rate
//!    improvement from exactly this policy.
//!
//! The sidecar is a perf hint, never a correctness input: a stale or
//! corrupt file is detected by fingerprint (size + mtime + head hash)
//! and ignored; pre-pinned experts are refcount-0 and evict normally
//! under LRU pressure if the learned ranking turns out wrong.
//!
//! 3. **Disk-spill store location** — when the opt-in disk-spill /
//!    lower-bit expert store is enabled (`RUSTLLAMA_MOE_DISK_SPILL`,
//!    mechanism in `accel`), [`init_moe_spill_store`] points its spill
//!    file at the runtime cache dir. Called automatically from
//!    [`UsageLearner::open`]. Unset ⇒ never touched.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub use rustllama_models::accel::ExpertKey;
use rustllama_models::accel::{expert_access_take, expert_prepin};
// Disk-spill + lower-bit secondary store (the mechanism lives in `accel`).
// Re-exported so the engine/integrator reaches the fault-in path through
// this policy module. Inert unless `RUSTLLAMA_MOE_DISK_SPILL` is set.
pub use rustllama_models::accel::{
    moe_spill_enabled, moe_spill_reconstruct, moe_spill_stats, MoeSpillStats, SpilledExpert,
    SpilledPart,
};

/// Sidecar magic + format version (bump on layout change).
const MAGIC: &[u8; 8] = b"RLUSAGE\x01";
/// Decay applied to prior sessions' counts at flush: the written
/// ranking is `prior * DECAY + this_session`. One session of fresh
/// routing outweighs one stale session; several sessions of stable
/// routing accumulate toward `count / (1 - DECAY)`.
const DECAY: f64 = 0.5;
/// Fraction of the expert-cache budget the load-time pre-pin may
/// fill. The remainder stays free for LRU churn on this session's
/// actual routing.
const PREPIN_BUDGET_FRACTION: f64 = 0.70;
/// Minimum interval between periodic sidecar flushes (the engine
/// also flushes on drop). Keeps request-start overhead negligible.
const FLUSH_MIN_INTERVAL: Duration = Duration::from_secs(60);

/// Identity of the GGUF a sidecar's counts belong to. mtime + size
/// catch re-quantized / replaced files; the FNV-1a hash of the first
/// 64 KiB catches same-size swaps without reading gigabytes. Shared
/// with the `.rlkv` KV-persistence sidecar (`kv_persist.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Fingerprint {
    pub(crate) size: u64,
    pub(crate) mtime_secs: u64,
    pub(crate) head_hash: u64,
}

pub(crate) fn fingerprint(path: &Path) -> Option<Fingerprint> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime_secs = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    let mut head = vec![0u8; 64 * 1024];
    let n = {
        use std::io::Read as _;
        let mut f = std::fs::File::open(path).ok()?;
        f.read(&mut head).ok()?
    };
    // FNV-1a — same prefilter hash family the prefix cache uses.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in &head[..n] {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Some(Fingerprint {
        size: meta.len(),
        mtime_secs,
        head_hash: h,
    })
}

/// Where a model's usage sidecar lives: `<gguf path>.rlusage`.
pub fn sidecar_path(model_path: &Path) -> PathBuf {
    let mut os = model_path.as_os_str().to_os_string();
    os.push(".rlusage");
    PathBuf::from(os)
}

/// Point the process-global MoE disk-spill store at the runtime cache dir
/// (`<runtime>/moe-spill/`). No-op — and no runtime-path resolution — unless
/// `RUSTLLAMA_MOE_DISK_SPILL` is set. Idempotent; called automatically from
/// [`UsageLearner::open`] at model load, and exposed so a headless caller
/// without a usage learner can wire the location explicitly. When it is never
/// called, the store falls back to a folder under the OS temp dir.
pub fn init_moe_spill_store() {
    if !moe_spill_enabled() {
        return;
    }
    let dir = rustllama_runtime::paths().runtime_dir.join("moe-spill");
    rustllama_models::accel::set_moe_spill_dir(dir);
}

/// Per-model learning-cache driver. Owned by the engine that loaded
/// the model (forks get `None` — one flusher per model, and the accel
/// counts are process-global anyway).
pub struct UsageLearner {
    path: PathBuf,
    fp: Fingerprint,
    /// Decayed counts carried over from previous sessions.
    baseline: HashMap<ExpertKey, f64>,
    /// Counts folded in from the accel side this session.
    session: HashMap<ExpertKey, u64>,
    last_flush: Instant,
}

impl UsageLearner {
    /// Open the learner for `model_path`. Returns `None` only when the
    /// model file itself can't be fingerprinted (deleted mid-load). A
    /// missing / stale / corrupt sidecar yields an empty baseline.
    pub fn open(model_path: &Path) -> Option<Self> {
        // Opt-in: locate the MoE disk-spill store under the runtime dir. No-op
        // (and no runtime-path resolution) unless the feature is enabled.
        init_moe_spill_store();
        let fp = fingerprint(model_path)?;
        let path = sidecar_path(model_path);
        let baseline = match read_sidecar(&path, fp) {
            Some(rows) => rows,
            None => HashMap::new(),
        };
        Some(Self {
            path,
            fp,
            baseline,
            session: HashMap::new(),
            last_flush: Instant::now(),
        })
    }

    /// Experts ranked hottest-first from the prior-session baseline.
    pub fn ranked_keys(&self) -> Vec<ExpertKey> {
        let mut rows: Vec<(ExpertKey, f64)> =
            self.baseline.iter().map(|(k, v)| (*k, *v)).collect();
        rows.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        rows.into_iter().map(|(k, _)| k).collect()
    }

    /// Pre-pin the learned-hottest experts up to
    /// [`PREPIN_BUDGET_FRACTION`] of `budget_bytes`. Call after the
    /// working set has been prepared (pagelock) and before the first
    /// prefill. Returns `(experts_pinned, bytes_pinned)`.
    pub fn prepin(&self, budget_bytes: u64) -> (usize, u64) {
        let ranked = self.ranked_keys();
        if ranked.is_empty() {
            return (0, 0);
        }
        let cap = (budget_bytes as f64 * PREPIN_BUDGET_FRACTION) as u64;
        expert_prepin(&ranked, cap)
    }

    /// Ranking over the prior baseline *and* this session's observed
    /// counts — the same decayed merge the sidecar write uses. This is
    /// what a mid-session re-pin (elastic budget change) should rank
    /// by: the live session usually knows more than last session's
    /// file. Call [`Self::flush`] first so the session tally is
    /// current.
    pub fn merged_ranked_keys(&self) -> Vec<ExpertKey> {
        let mut merged: HashMap<ExpertKey, f64> = self
            .baseline
            .iter()
            .map(|(k, v)| (*k, v * DECAY))
            .collect();
        for (k, n) in &self.session {
            *merged.entry(*k).or_insert(0.0) += *n as f64;
        }
        let mut rows: Vec<(ExpertKey, f64)> = merged.into_iter().collect();
        rows.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        rows.into_iter().map(|(k, _)| k).collect()
    }

    /// Re-pin after an elastic budget change: like [`Self::prepin`]
    /// but ranked over baseline + live session counts.
    pub fn prepin_merged(&self, budget_bytes: u64) -> (usize, u64) {
        let ranked = self.merged_ranked_keys();
        if ranked.is_empty() {
            return (0, 0);
        }
        let cap = (budget_bytes as f64 * PREPIN_BUDGET_FRACTION) as u64;
        expert_prepin(&ranked, cap)
    }

    /// Fold the accel-side access counts taken since the last call
    /// into this session's tally and rewrite the sidecar. `force`
    /// skips the rate limit (engine drop / explicit teardown).
    pub fn flush(&mut self, force: bool) {
        if !force && self.last_flush.elapsed() < FLUSH_MIN_INTERVAL {
            return;
        }
        for (k, n) in expert_access_take() {
            *self.session.entry(k).or_insert(0) += n;
        }
        self.last_flush = Instant::now();
        if self.session.is_empty() && self.baseline.is_empty() {
            return; // nothing observed yet — don't create an empty file
        }
        // Written ranking = decayed prior + this session so far. The
        // formula is stable under repeated flushes within one session
        // (baseline and session are never mutated by the write).
        let mut merged: HashMap<ExpertKey, f64> = self
            .baseline
            .iter()
            .map(|(k, v)| (*k, v * DECAY))
            .collect();
        for (k, n) in &self.session {
            *merged.entry(*k).or_insert(0.0) += *n as f64;
        }
        if let Err(e) = write_sidecar(&self.path, self.fp, &merged) {
            tracing::debug!(path = %self.path.display(), error = %e, "expert usage sidecar write failed (non-fatal)");
        }
    }
}

fn read_sidecar(path: &Path, expect: Fingerprint) -> Option<HashMap<ExpertKey, f64>> {
    let bytes = std::fs::read(path).ok()?;
    let mut off = 0usize;
    let take = |off: &mut usize, n: usize| -> Option<&[u8]> {
        let s = bytes.get(*off..*off + n)?;
        *off += n;
        Some(s)
    };
    if take(&mut off, 8)? != MAGIC {
        return None;
    }
    let u64_at = |s: &[u8]| u64::from_le_bytes(s.try_into().unwrap());
    let fp = Fingerprint {
        size: u64_at(take(&mut off, 8)?),
        mtime_secs: u64_at(take(&mut off, 8)?),
        head_hash: u64_at(take(&mut off, 8)?),
    };
    if fp != expect {
        tracing::info!(
            path = %path.display(),
            "expert usage sidecar is for a different model file (re-quantized?); starting fresh"
        );
        return None;
    }
    let n_rows = u32::from_le_bytes(take(&mut off, 4)?.try_into().unwrap()) as usize;
    // Row = layer u32 + expert u32 + weight f64.
    let mut rows = HashMap::with_capacity(n_rows.min(1 << 20));
    for _ in 0..n_rows {
        let s = take(&mut off, 16)?;
        let layer = u32::from_le_bytes(s[0..4].try_into().unwrap());
        let expert = u32::from_le_bytes(s[4..8].try_into().unwrap());
        let weight = f64::from_le_bytes(s[8..16].try_into().unwrap());
        if weight.is_finite() && weight > 0.0 {
            rows.insert(ExpertKey::new(layer, expert), weight);
        }
    }
    Some(rows)
}

fn write_sidecar(
    path: &Path,
    fp: Fingerprint,
    rows: &HashMap<ExpertKey, f64>,
) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(36 + rows.len() * 16);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&fp.size.to_le_bytes());
    buf.extend_from_slice(&fp.mtime_secs.to_le_bytes());
    buf.extend_from_slice(&fp.head_hash.to_le_bytes());
    buf.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    for (k, w) in rows {
        buf.extend_from_slice(&k.layer.to_le_bytes());
        buf.extend_from_slice(&k.expert.to_le_bytes());
        buf.extend_from_slice(&w.to_le_bytes());
    }
    // Crash-safe-enough for a perf hint: write a temp sibling, then
    // swap it in. Windows `rename` won't overwrite, so remove first —
    // a crash in the gap costs one sidecar, which regenerates.
    let tmp = path.with_extension("rlusage.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&buf)?;
        f.sync_all()?;
    }
    let _ = std::fs::remove_file(path);
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_model(name: &str, contents: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir().join("rustllama-expert-cache-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, contents).unwrap();
        p
    }

    #[test]
    fn sidecar_roundtrip_preserves_ranking() {
        let model = tmp_model("m1.gguf", b"model-bytes-1");
        let fp = fingerprint(&model).unwrap();
        let side = sidecar_path(&model);
        let mut rows = HashMap::new();
        rows.insert(ExpertKey::new(0, 7), 100.0);
        rows.insert(ExpertKey::new(3, 1), 25.5);
        write_sidecar(&side, fp, &rows).unwrap();
        let back = read_sidecar(&side, fp).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[&ExpertKey::new(0, 7)], 100.0);
        let _ = std::fs::remove_file(&side);
    }

    #[test]
    fn sidecar_rejected_on_model_change() {
        let model = tmp_model("m2.gguf", b"model-bytes-2");
        let fp = fingerprint(&model).unwrap();
        let side = sidecar_path(&model);
        let mut rows = HashMap::new();
        rows.insert(ExpertKey::new(1, 2), 5.0);
        write_sidecar(&side, fp, &rows).unwrap();
        // Same size, different content → head hash changes.
        std::fs::write(&model, b"model-bytes-3").unwrap();
        let fp2 = fingerprint(&model).unwrap();
        assert!(read_sidecar(&side, fp2).is_none());
        let _ = std::fs::remove_file(&side);
    }

    #[test]
    fn ranked_keys_sorted_hottest_first_and_stable() {
        let model = tmp_model("m3.gguf", b"model-bytes-4");
        let mut learner = UsageLearner::open(&model).unwrap();
        learner.baseline.insert(ExpertKey::new(0, 1), 10.0);
        learner.baseline.insert(ExpertKey::new(0, 2), 30.0);
        learner.baseline.insert(ExpertKey::new(2, 9), 30.0);
        let ranked = learner.ranked_keys();
        assert_eq!(ranked[0], ExpertKey::new(0, 2)); // ties break by key
        assert_eq!(ranked[1], ExpertKey::new(2, 9));
        assert_eq!(ranked[2], ExpertKey::new(0, 1));
    }

    #[test]
    fn corrupt_sidecar_is_ignored() {
        let model = tmp_model("m4.gguf", b"model-bytes-5");
        let fp = fingerprint(&model).unwrap();
        let side = sidecar_path(&model);
        std::fs::write(&side, b"garbage").unwrap();
        assert!(read_sidecar(&side, fp).is_none());
        let _ = std::fs::remove_file(&side);
    }
}
