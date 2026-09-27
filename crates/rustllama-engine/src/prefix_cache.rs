//! Cross-request prompt-prefix KV cache.
//!
//! `CpuEngine` already reuses the live KV cache when the next prompt
//! shares an LCP with the previous one — great for a single rolling
//! conversation. This module extends that to multiple conversations:
//! after generation finishes, the engine pushes a clipped snapshot of
//! the (prompt + output) KV state into a small pool. The next request
//! scans the pool plus the live state and restores the entry with the
//! longest matching token prefix.
//!
//! Eviction is LRU on `last_used`. The pool's size is configurable; a
//! pool of 0 entries disables the pool (the engine falls back to the
//! original single-live-state LCP behavior).
//!
//! Hashes (FNV-1a 64) are kept alongside each entry as a fast first-pass
//! filter on candidates whose token sequences obviously can't match, but
//! the actual selection always runs a token-by-token LCP comparison so
//! hash collisions can never cause a wrong restore.

use rustllama_models::llama_arch::{DeltaNetCache, DeltaNetSnapshot, KvCache, KvSnapshot};

/// One stored snapshot. The KV blob is sized to the token sequence's
/// length, not the live cache's full `max_ctx`, so memory cost scales
/// with how much was actually generated.
#[derive(Debug, Clone)]
pub struct PrefixSnapshot {
    /// The token sequence reflected by `kv`. For text-only entries
    /// `kv.prefix_len == ids.len()`. For VLM entries `kv.prefix_len`
    /// is the *expanded* length (each image-placeholder token in
    /// `ids` produced `num_image_tokens` KV positions during the
    /// original prefill).
    pub ids: Vec<u32>,
    /// FNV-1a 64 hash of `ids`. Used as a cheap "definitely doesn't
    /// match" filter; selection still runs LCP byte-for-byte.
    pub hash: u64,
    /// Image-bytes hash for the request that produced this snapshot.
    /// `None` for text-only requests. Used to gate cache reuse: a
    /// new request with a different image (or a text-only request)
    /// must NOT restore a VLM entry, since the KV state encodes
    /// image-specific projected patches at the placeholder positions.
    pub image_hash: Option<u64>,
    /// Number of KV positions the snapshot's K/V blob actually
    /// occupies. Equals `ids.len()` for text-only entries; for VLM
    /// it's `text_tokens + sum_of_image_token_counts`. Pinned here
    /// because the restoring code can't recover it from `ids` alone
    /// when image placeholders expand to multiple KV positions.
    pub kv_len: usize,
    /// The clipped K/V state.
    pub kv: KvSnapshot,
    /// Hybrid models only (roadmap Phase 5): the DeltaNet recurrent
    /// state captured at exactly `ids.len()` processed tokens. Its
    /// presence marks a **semantic anchor** — the entry can only be
    /// restored in full (recurrent state has no per-position
    /// addressing), so hybrid reuse restores the whole anchor and
    /// re-prefills the suffix. `None` for standard transformer /
    /// dense-MoE entries, which restore at any LCP.
    pub dn: Option<DeltaNetSnapshot>,
    /// Monotonic counter at the moment of last use (insert or restore).
    /// LRU eviction picks the entry with the smallest value.
    pub last_used: u64,
}

impl PrefixSnapshot {
    /// Bytes this entry holds (KV blob + recurrent state + ids).
    pub fn approx_bytes(&self) -> u64 {
        let kv: usize = self.kv.layers.iter().map(|l| l.approx_bytes()).sum();
        let dn = self.dn.as_ref().map(|d| d.approx_bytes()).unwrap_or(0);
        (kv + dn + self.ids.len() * 4) as u64
    }
}

/// A small bounded pool of prefix snapshots. Linear-scan on lookup —
/// the pool is meant to be tiny (≤16 entries) so a hash map isn't worth
/// the bookkeeping.
#[derive(Debug)]
pub struct PrefixCachePool {
    entries: Vec<PrefixSnapshot>,
    max_entries: usize,
    /// Byte cap across all entries (`0` = uncapped). Enforced at
    /// insert alongside `max_entries` — snapshots at long contexts
    /// are hundreds of MB each, and an entry-count-only bound let
    /// the pool grow past what the memory planner had granted away.
    max_bytes: u64,
    counter: u64,
    /// H5: hit-rate telemetry. `hits` counts entries returned by
    /// `find_best` / `find_full_match_with_image_hash` (≥ 1 token of
    /// shared prefix); `misses` counts queries that found no match.
    /// Together they let the engine surface `cache_hit_rate` so the
    /// `prefix_cache_max_snapshots` tuner knob can be tuned with data
    /// instead of guesses.
    hits: u64,
    misses: u64,
}

impl PrefixCachePool {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::with_capacity(max_entries.min(64)),
            max_entries,
            max_bytes: 0,
            counter: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Total bytes held across every entry.
    pub fn approx_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.approx_bytes()).sum()
    }

    /// Set the byte cap (`0` = uncapped), evicting LRU entries until
    /// the pool fits. Fed by the memory-budget planner at engine
    /// load; hot-appliable later.
    pub fn set_max_bytes(&mut self, max_bytes: u64) {
        self.max_bytes = max_bytes;
        if max_bytes > 0 {
            while self.approx_bytes() > max_bytes && self.entries.len() > 1 {
                self.evict_lru();
            }
        }
    }

    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// H5: total cache lookups that returned a non-empty match.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// H5: total cache lookups that found nothing.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// H5: `hits / (hits + misses)` as a fraction. Returns 0.0 when
    /// no lookups have happened yet (avoids NaN on cold engines).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// H5: reset telemetry counters. Useful between bench runs or
    /// when a request-level metric is captured separately.
    pub fn reset_telemetry(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }

    /// H5: record the outcome of a cache lookup. Called by the engine
    /// after each `find_best` / `find_full_match_with_image_hash`
    /// query — kept separate from the lookup methods themselves so
    /// the lookup signatures stay `&self` (read-only) and callers
    /// don't need to thread a mutable pool ref through analysis-only
    /// code paths.
    pub fn record_lookup(&mut self, hit: bool) {
        if hit {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
    }

    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    pub fn set_max_entries(&mut self, n: usize) {
        self.max_entries = n;
        while self.entries.len() > n {
            self.evict_lru();
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Bump the internal counter and return the new value. Used both
    /// for `last_used` stamps and for callers that want a stable
    /// monotonic clock tied to pool activity.
    fn tick(&mut self) -> u64 {
        self.counter = self.counter.wrapping_add(1);
        self.counter
    }

    /// Scan the pool for the entry whose token sequence shares the
    /// longest common prefix with `ids`. Returns `(index, lcp_len)` or
    /// `None` if no entry has any non-empty match.
    ///
    /// Text-only callers (the default) — use [`find_best`].
    ///
    /// `image_hash` filters entries by their stored image-bytes hash:
    /// only entries whose `image_hash` matches the query's are
    /// considered. A text-only query (`None`) cannot match a VLM
    /// entry (`Some(_)`), and a VLM query with a given hash cannot
    /// match a different-image entry. Prevents cross-contamination
    /// of KV state between text and image conversations.
    pub fn find_best_with_image_hash(
        &self,
        ids: &[u32],
        image_hash: Option<u64>,
    ) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.image_hash != image_hash {
                continue;
            }
            let lcp = compute_lcp(&entry.ids, ids);
            if lcp == 0 {
                continue;
            }
            match best {
                Some((_, b)) if lcp <= b => {}
                _ => best = Some((i, lcp)),
            }
        }
        best
    }

    /// Text-only variant of [`find_best_with_image_hash`] — shorthand
    /// for callers that aren't VLM-aware.
    pub fn find_best(&self, ids: &[u32]) -> Option<(usize, usize)> {
        self.find_best_with_image_hash(ids, None)
    }

    /// Look up an entry that the caller can restore **in full**
    /// (lcp == entry.ids.len()). Returns the index of the longest
    /// such fully-matching entry whose `image_hash` matches. Useful
    /// for the VLM cache path, which doesn't yet support partial-
    /// trim restoration (image-placeholder token → multi-KV-position
    /// expansion makes the mapping non-trivial; full-match avoids
    /// the bookkeeping entirely and covers the common multi-turn
    /// case where each new prompt extends the previous one).
    pub fn find_full_match_with_image_hash(
        &self,
        ids: &[u32],
        image_hash: Option<u64>,
    ) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.image_hash != image_hash {
                continue;
            }
            if entry.ids.len() > ids.len() {
                continue;
            }
            if !ids.starts_with(&entry.ids) {
                continue;
            }
            let prefix_len = entry.ids.len();
            if best.map(|(_, b)| prefix_len > b).unwrap_or(true) {
                best = Some((i, prefix_len));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Mark the entry at `idx` as just-used. Returns the underlying
    /// snapshot by reference so the caller can plumb its `kv` into the
    /// live cache via [`KvCache::restore_prefix`].
    pub fn touch(&mut self, idx: usize) -> &PrefixSnapshot {
        let stamp = self.tick();
        self.entries[idx].last_used = stamp;
        &self.entries[idx]
    }

    /// Insert (or refresh) a snapshot for `ids` + `kv`. If an existing
    /// entry's `ids` is a *prefix* of the new one, it gets replaced in
    /// place rather than coexisting — keeping a 4-token prefix entry
    /// next to a 200-token entry that starts with the same 4 tokens is
    /// just memory waste, since the longer entry dominates LCP matches.
    /// Likewise, if the new `ids` is a prefix of an existing entry,
    /// the new one is discarded (the existing entry already covers it).
    /// LRU eviction kicks in once `len() == max_entries`.
    pub fn insert(&mut self, ids: Vec<u32>, kv: KvSnapshot) {
        self.insert_with_image_hash(ids, kv, None);
    }

    /// VLM-aware insert. `image_hash` is the FNV-1a 64 hash of the
    /// concatenated image bytes; passing `None` matches
    /// [`Self::insert`] for the text-only path. Prefix-relation dedup
    /// is gated on matching `image_hash` — a VLM entry never dedupes
    /// against a text-only entry, even if their token ids overlap.
    pub fn insert_with_image_hash(
        &mut self,
        ids: Vec<u32>,
        kv: KvSnapshot,
        image_hash: Option<u64>,
    ) {
        self.insert_entry(ids, kv, image_hash, None);
    }

    /// Core insert with dedup + LRU. `dn` marks a hybrid **anchor**
    /// entry (restorable only in full — see [`PrefixSnapshot::dn`]).
    fn insert_entry(
        &mut self,
        ids: Vec<u32>,
        kv: KvSnapshot,
        image_hash: Option<u64>,
        dn: Option<DeltaNetSnapshot>,
    ) {
        if self.max_entries == 0 || ids.is_empty() {
            return;
        }
        let hash = fnv1a64(&ids);
        let kv_len = kv.prefix_len;
        // Prefix relations: dedup before eviction. Same image-class
        // only (text-vs-VLM never collapse into each other).
        //
        // Anchor entries (dn.is_some()) deliberately KEEP nested
        // prefixes: a longer anchor does NOT dominate a shorter one,
        // because anchors restore only in full — after an agent edit
        // truncates history back to the shorter anchor's boundary,
        // the shorter anchor is exactly the entry that saves the
        // re-prefill (FreeToken's prefix-tree checkpoints). Only an
        // identical-ids reinsert replaces an anchor.
        let mut i = 0;
        while i < self.entries.len() {
            let other = &self.entries[i];
            if other.image_hash != image_hash {
                i += 1;
                continue;
            }
            let either_anchor = other.dn.is_some() || dn.is_some();
            if other.ids.len() <= ids.len()
                && ids.starts_with(&other.ids)
                && (!either_anchor || other.ids.len() == ids.len())
            {
                // Existing entry is a (non-strict) prefix of the new one.
                // Remove it; the new one supersedes it. (Anchors: only
                // on identical ids.)
                self.entries.swap_remove(i);
                continue;
            }
            if ids.len() < other.ids.len() && other.ids.starts_with(&ids) && !either_anchor {
                // New entry is a strict prefix of an existing one. The
                // existing one already covers all LCP queries for the
                // new ids. Refresh that entry's `last_used` and bail.
                let stamp = self.tick();
                self.entries[i].last_used = stamp;
                return;
            }
            i += 1;
        }
        while self.entries.len() >= self.max_entries {
            self.evict_lru();
        }
        let stamp = self.tick();
        let entry = PrefixSnapshot {
            ids,
            hash,
            image_hash,
            kv_len,
            kv,
            dn,
            last_used: stamp,
        };
        // Byte cap: a single over-cap entry is refused outright;
        // otherwise evict LRU until the newcomer fits.
        if self.max_bytes > 0 {
            let new_bytes = entry.approx_bytes();
            if new_bytes > self.max_bytes {
                return;
            }
            while self.approx_bytes() + new_bytes > self.max_bytes && !self.entries.is_empty() {
                self.evict_lru();
            }
        }
        self.entries.push(entry);
    }

    /// Insert a fully-formed snapshot (the `.rlkv` persistence loader
    /// path — roadmap Phase 5 warm restarts). Runs the same dedup +
    /// LRU as a live insert; the entry's `last_used` is restamped so
    /// loaded entries start as most-recently-used in load order.
    pub fn insert_snapshot(&mut self, snap: PrefixSnapshot) {
        self.insert_entry(snap.ids, snap.kv, snap.image_hash, snap.dn);
    }

    /// Read-only view of the entries (persistence writer + tests).
    pub fn entries(&self) -> &[PrefixSnapshot] {
        &self.entries
    }

    /// Snapshot the prefix `ids` into the pool from `live_kv`. Convenience
    /// wrapper around [`KvCache::snapshot_prefix`] + [`Self::insert`].
    pub fn snapshot_and_insert(&mut self, ids: Vec<u32>, live_kv: &KvCache) {
        self.snapshot_and_insert_with_image_hash(ids, live_kv, None);
    }

    /// VLM-aware snapshot. `image_hash` discriminates VLM entries
    /// from text-only ones (and from each other across distinct
    /// images). For VLM the live `kv.seq_len` is the expanded
    /// (spliced) length, which becomes the snapshot's `kv_len`.
    ///
    /// `ids` is the token sequence (with placeholders, not the
    /// expanded sequence). The snapshot stores both: `ids` drives
    /// lookup-side LCP, `kv_len` (= live_kv.seq_len) drives
    /// restoration-side `seq_len` setting.
    pub fn snapshot_and_insert_with_image_hash(
        &mut self,
        ids: Vec<u32>,
        live_kv: &KvCache,
        image_hash: Option<u64>,
    ) {
        if self.max_entries == 0 || ids.is_empty() {
            return;
        }
        // For text-only, kv.seq_len == ids.len(). For VLM,
        // kv.seq_len is the expanded length and is what we want to
        // snapshot. Don't clip `ids` for VLM — placeholders are part
        // of the lookup key.
        let kv_len = match image_hash {
            None => ids.len().min(live_kv.seq_len),
            Some(_) => live_kv.seq_len,
        };
        if kv_len == 0 {
            return;
        }
        let ids_clipped = match image_hash {
            None if kv_len < ids.len() => ids[..kv_len].to_vec(),
            _ => ids,
        };
        let kv = live_kv.snapshot_prefix(kv_len);
        self.insert_with_image_hash(ids_clipped, kv, image_hash);
    }

    /// Hybrid-model anchor snapshot (roadmap Phase 5): capture the KV
    /// state (full-attention layers only — SSM layers' slabs are
    /// never read) AND the DeltaNet recurrent state, both at exactly
    /// `ids.len()` processed tokens. Skips silently when the live
    /// state doesn't correspond (`ids.len() != live_kv.seq_len`) — a
    /// recurrent snapshot at the wrong position would poison every
    /// later restore.
    pub fn snapshot_and_insert_hybrid(
        &mut self,
        ids: Vec<u32>,
        live_kv: &KvCache,
        dn: &DeltaNetCache,
    ) {
        if self.max_entries == 0 || ids.is_empty() {
            return;
        }
        if ids.len() != live_kv.seq_len {
            return;
        }
        // Attention layers are exactly the ones whose DeltaNet state
        // is the empty sentinel.
        let keep: Vec<bool> = dn.layers.iter().map(|l| l.is_empty()).collect();
        if keep.len() != live_kv.layers.len() {
            return;
        }
        let kv = live_kv.snapshot_prefix_selective(ids.len(), &keep);
        let dn_snap = dn.snapshot();
        self.insert_entry(ids, kv, None, Some(dn_snap));
    }

    /// Longest **anchor** entry (carries DeltaNet state) whose ids
    /// form a complete prefix of `ids`. Hybrid restore is
    /// all-or-nothing — the recurrent state is only valid at the
    /// anchor's exact position — so partial-LCP entries are useless
    /// here. Returns `(index, anchor_len)`.
    pub fn find_full_match_hybrid(&self, ids: &[u32]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.dn.is_none() || entry.image_hash.is_some() {
                continue;
            }
            if entry.ids.len() > ids.len() || !ids.starts_with(&entry.ids) {
                continue;
            }
            let len = entry.ids.len();
            if best.map(|(_, b)| len > b).unwrap_or(true) {
                best = Some((i, len));
            }
        }
        best
    }

    fn evict_lru(&mut self) {
        if let Some((idx, _)) = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.last_used)
        {
            self.entries.swap_remove(idx);
        }
    }
}

impl Default for PrefixCachePool {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Longest common prefix length between two slices.
pub fn compute_lcp<T: Eq>(a: &[T], b: &[T]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// FNV-1a 64-bit hash over a token sequence. Used only as a cheap
/// filter for the prefix pool; the authoritative comparison is still
/// a byte-for-byte LCP, so hash collisions are inert.
pub fn fnv1a64(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &id in ids {
        for &b in &id.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// FNV-1a 64-bit hash over a slab of raw bytes. Used by the VLM
/// prefix cache to discriminate entries by attached image content:
/// hash the concatenation of every attached image's bytes (in
/// message order) and pass the result as `image_hash` to the cache
/// APIs.
///
/// Collisions are inert here too — they'd cause two requests with
/// different images to share a cache bin where they currently can't
/// match (different bytes → different ids paths anyway).
pub fn fnv1a64_bytes_chained(slabs: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for slab in slabs {
        for &b in *slab {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        // Length terminator between slabs so `[a, b]` doesn't hash
        // the same as `[ab]`.
        for &b in &(slab.len() as u64).to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustllama_models::llama_arch::{KvCache, KvDtype};
    use rustllama_models::llama_config::LlamaConfig;

    fn synthetic_cfg() -> LlamaConfig {
        LlamaConfig {
            arch: "test".into(),
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            d_model: 8,
            d_ff: 16,
            head_dim: 4,
            rope_dim: 4,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            vocab_size: 32,
            ctx_train: 16,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: false,
            n_mtp_heads: 0,
            moe: None,
            hybrid: None,
            hadamard: None,
        }
    }

    fn fill_kv(kv: &mut KvCache, seq_len: usize, base: f32) {
        kv.seq_len = seq_len;
        for (li, layer) in kv.layers.iter_mut().enumerate() {
            if let rustllama_models::llama_arch::KvLayer::F32 { k, v } = layer {
                if k.is_empty() {
                    continue; // sparse placeholder layer
                }
                for h in 0..kv.n_kv_heads {
                    for pos in 0..seq_len {
                        for d in 0..kv.head_dim {
                            let idx = h * kv.max_ctx * kv.head_dim + pos * kv.head_dim + d;
                            // Pattern lets us trivially verify the snapshot
                            // copied the right (head, pos, dim) cells:
                            // value depends on all four indices.
                            k[idx] = base + (li as f32) + (h as f32) * 0.1
                                + (pos as f32) * 0.01
                                + (d as f32) * 0.001;
                            v[idx] = -k[idx];
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn snapshot_then_restore_round_trips_kv_data() {
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
        fill_kv(&mut live, 5, 1.0);

        let snap = live.snapshot_prefix(5);
        // Wipe live and restore: the (head, pos, dim) values must come back.
        let saved_layers: Vec<rustllama_models::llama_arch::KvLayer> =
            live.layers.iter().cloned().collect();
        live.reset();
        live.restore_prefix(&snap);
        assert_eq!(live.seq_len, 5);
        for (orig, restored) in saved_layers.iter().zip(live.layers.iter()) {
            if let (
                rustllama_models::llama_arch::KvLayer::F32 { k: ok, v: ov },
                rustllama_models::llama_arch::KvLayer::F32 { k: rk, v: rv },
            ) = (orig, restored)
            {
                for h in 0..live.n_kv_heads {
                    for pos in 0..5 {
                        for d in 0..live.head_dim {
                            let idx =
                                h * live.max_ctx * live.head_dim + pos * live.head_dim + d;
                            assert!((ok[idx] - rk[idx]).abs() < 1e-6);
                            assert!((ov[idx] - rv[idx]).abs() < 1e-6);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn snapshot_then_restore_round_trips_q4_0_kv() {
        // Q4_0 KV snapshot/restore: packed bytes must round-trip
        // exactly (the snapshot copies quantized blocks verbatim, no
        // requantization). head_dim must be a multiple of 32.
        let mut cfg = synthetic_cfg();
        cfg.head_dim = 32;
        let mut live = KvCache::new_with_dtype(&cfg, 16, KvDtype::Q4_0);
        // Write a recognizable pattern through the quantizer so block
        // structure (scales + nibbles) is realistic.
        live.seq_len = 5;
        let (n_h, max_ctx, hd) = (live.n_kv_heads, live.max_ctx, live.head_dim);
        let bytes_per_row = hd / 32 * 18;
        for layer in live.layers.iter_mut() {
            if let rustllama_models::llama_arch::KvLayer::Q4_0 { k_q, v_q } = layer {
                let mut row = vec![0f32; hd];
                for h in 0..n_h {
                    for pos in 0..5usize {
                        for (d, e) in row.iter_mut().enumerate() {
                            *e = (h as f32) - (pos as f32) * 0.25 + (d as f32) * 0.03;
                        }
                        let p = (h * max_ctx + pos) * bytes_per_row;
                        rustllama_kernels_cpu::q4_0_kv::quantize_row(
                            &row,
                            &mut k_q[p..p + bytes_per_row],
                        );
                        for e in row.iter_mut() {
                            *e = -*e;
                        }
                        rustllama_kernels_cpu::q4_0_kv::quantize_row(
                            &row,
                            &mut v_q[p..p + bytes_per_row],
                        );
                        for e in row.iter_mut() {
                            *e = -*e;
                        }
                    }
                }
            } else {
                panic!("expected Q4_0 layer");
            }
        }

        let snap = live.snapshot_prefix(5);
        let saved_layers: Vec<rustllama_models::llama_arch::KvLayer> =
            live.layers.iter().cloned().collect();
        live.reset();
        live.restore_prefix(&snap);
        assert_eq!(live.seq_len, 5);
        for (orig, restored) in saved_layers.iter().zip(live.layers.iter()) {
            if let (
                rustllama_models::llama_arch::KvLayer::Q4_0 { k_q: ok, v_q: ov },
                rustllama_models::llama_arch::KvLayer::Q4_0 { k_q: rk, v_q: rv },
            ) = (orig, restored)
            {
                for h in 0..live.n_kv_heads {
                    let base = h * live.max_ctx * bytes_per_row;
                    let prefix_bytes = 5 * bytes_per_row;
                    assert_eq!(
                        &ok[base..base + prefix_bytes],
                        &rk[base..base + prefix_bytes],
                        "K packed bytes must round-trip exactly"
                    );
                    assert_eq!(
                        &ov[base..base + prefix_bytes],
                        &rv[base..base + prefix_bytes],
                        "V packed bytes must round-trip exactly"
                    );
                }
            } else {
                panic!("expected Q4_0 layers after restore");
            }
        }
    }

    #[test]
    fn pool_picks_longest_lcp_match() {
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);

        // Snapshot 1: [1, 2, 3, 4]
        fill_kv(&mut live, 4, 10.0);
        pool.snapshot_and_insert(vec![1, 2, 3, 4], &live);
        // Snapshot 2: [1, 2, 3, 4, 5, 6, 7]
        fill_kv(&mut live, 7, 20.0);
        pool.snapshot_and_insert(vec![1, 2, 3, 4, 5, 6, 7], &live);
        // Snapshot 3: [1, 2, 99]
        fill_kv(&mut live, 3, 30.0);
        pool.snapshot_and_insert(vec![1, 2, 99], &live);

        // The prefix-dedup rule means [1,2,3,4] should have been removed
        // when [1,2,3,4,5,6,7] was inserted.
        assert_eq!(pool.len(), 2);

        // Query with [1, 2, 3, 4, 8, 9]: best LCP is 4 with entry 2.
        let q = vec![1u32, 2, 3, 4, 8, 9];
        let (idx, lcp) = pool.find_best(&q).expect("must match something");
        assert_eq!(lcp, 4);
        assert_eq!(pool.entries[idx].ids, vec![1, 2, 3, 4, 5, 6, 7]);

        // Query with [1, 2, 99, 100]: LCP=3 with entry whose ids=[1,2,99].
        let q2 = vec![1u32, 2, 99, 100];
        let (idx2, lcp2) = pool.find_best(&q2).expect("must match the other");
        assert_eq!(lcp2, 3);
        assert_eq!(pool.entries[idx2].ids, vec![1, 2, 99]);
    }

    #[test]
    fn pool_evicts_lru_when_full() {
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(2);

        fill_kv(&mut live, 3, 1.0);
        pool.snapshot_and_insert(vec![1, 1, 1], &live);
        fill_kv(&mut live, 3, 2.0);
        pool.snapshot_and_insert(vec![2, 2, 2], &live);
        assert_eq!(pool.len(), 2);

        // Touch entry [1,1,1] so [2,2,2] is now the LRU.
        let (idx, _) = pool.find_best(&[1, 1, 1, 9]).unwrap();
        pool.touch(idx);

        fill_kv(&mut live, 3, 3.0);
        pool.snapshot_and_insert(vec![3, 3, 3], &live);
        assert_eq!(pool.len(), 2);
        // [2,2,2] should have been evicted, [1,1,1] survives.
        assert!(pool.entries.iter().any(|e| e.ids == vec![1, 1, 1]));
        assert!(pool.entries.iter().any(|e| e.ids == vec![3, 3, 3]));
        assert!(!pool.entries.iter().any(|e| e.ids == vec![2, 2, 2]));
    }

    #[test]
    fn pool_zero_capacity_is_a_noop() {
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
        fill_kv(&mut live, 4, 5.0);
        let mut pool = PrefixCachePool::new(0);
        pool.snapshot_and_insert(vec![1, 2, 3, 4], &live);
        assert_eq!(pool.len(), 0);
        assert!(pool.find_best(&[1, 2, 3, 4]).is_none());
    }

    #[test]
    fn fnv1a_is_deterministic_and_distinguishes_short_sequences() {
        assert_eq!(fnv1a64(&[1, 2, 3]), fnv1a64(&[1, 2, 3]));
        assert_ne!(fnv1a64(&[1, 2, 3]), fnv1a64(&[1, 2, 4]));
        assert_ne!(fnv1a64(&[1, 2]), fnv1a64(&[1, 2, 0]));
    }

    // ----- Eviction-policy audit -------------------------------------------

    #[test]
    fn set_max_entries_shrinks_existing_pool_via_lru() {
        // The pool can be reconfigured at runtime via
        // `set_max_entries` — e.g. when the operator tweaks
        // `prefix_cache_max_snapshots` in config. Shrinking below the
        // current count must evict the least-recently-used entries,
        // not just refuse the change or wipe everything.
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        for tag in 1..=4u32 {
            fill_kv(&mut live, 3, tag as f32);
            pool.snapshot_and_insert(vec![tag, tag, tag], &live);
        }
        assert_eq!(pool.len(), 4);
        // Touch entry #3 so it becomes the most-recently-used.
        let (idx, _) = pool.find_best(&[3, 3, 3]).unwrap();
        pool.touch(idx);
        // Shrink to 2 — should evict the two oldest (entries 1 and 2)
        // while keeping the touched entry 3 and the most-recent entry 4.
        pool.set_max_entries(2);
        assert_eq!(pool.len(), 2);
        assert!(
            pool.entries.iter().any(|e| e.ids == vec![3, 3, 3]),
            "the touched entry should survive"
        );
        assert!(
            pool.entries.iter().any(|e| e.ids == vec![4, 4, 4]),
            "the most-recent insert should survive"
        );
        assert!(
            !pool.entries.iter().any(|e| e.ids == vec![1, 1, 1]),
            "[1,1,1] must have been evicted: {:?}",
            pool.entries.iter().map(|e| &e.ids).collect::<Vec<_>>()
        );
    }

    #[test]
    fn clear_empties_the_pool() {
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        fill_kv(&mut live, 3, 1.0);
        pool.snapshot_and_insert(vec![1, 1, 1], &live);
        assert_eq!(pool.len(), 1);
        pool.clear();
        assert!(pool.is_empty());
        assert!(pool.find_best(&[1, 1, 1]).is_none());
    }

    #[test]
    fn insert_of_strict_prefix_of_existing_does_not_grow_pool() {
        // The dedup rule: when the NEW ids is a strict prefix of an
        // existing entry, the new insert is discarded (the existing
        // longer entry already covers all LCP queries for the new
        // ids). Pin that the pool doesn't grow and the existing
        // entry's last_used gets refreshed.
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        fill_kv(&mut live, 7, 1.0);
        pool.snapshot_and_insert(vec![1, 2, 3, 4, 5, 6, 7], &live);
        assert_eq!(pool.len(), 1);
        let before = pool.entries[0].last_used;
        // Now try to insert a strict prefix of the same sequence.
        fill_kv(&mut live, 3, 1.0);
        pool.snapshot_and_insert(vec![1, 2, 3], &live);
        assert_eq!(pool.len(), 1, "prefix-of-existing must not grow the pool");
        assert_eq!(
            pool.entries[0].ids,
            vec![1, 2, 3, 4, 5, 6, 7],
            "the surviving entry is the LONGER one"
        );
        assert!(
            pool.entries[0].last_used > before,
            "the existing entry's last_used should be refreshed"
        );
    }

    #[test]
    fn insert_of_identical_ids_replaces_in_place() {
        // Re-inserting the same sequence — dedup runs first
        // (`other.ids.len() <= ids.len() && ids.starts_with(other)`
        // matches when equal), so the existing entry is removed
        // and a fresh one is added. Net result: pool size stays
        // 1, but the entry's KV blob may have been refreshed.
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        fill_kv(&mut live, 4, 1.0);
        pool.snapshot_and_insert(vec![1, 2, 3, 4], &live);
        assert_eq!(pool.len(), 1);
        // Refresh with a different KV pattern.
        fill_kv(&mut live, 4, 99.0);
        pool.snapshot_and_insert(vec![1, 2, 3, 4], &live);
        assert_eq!(pool.len(), 1, "identical-ids reinsert must not grow the pool");
    }

    #[test]
    fn find_best_returns_none_for_empty_query_or_empty_pool() {
        // Edge case: callers can pass `find_best(&[])` (e.g., when
        // the request hasn't been tokenized yet). The lookup
        // shouldn't panic or claim a match.
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        // Empty pool → None for any query.
        assert!(pool.find_best(&[1, 2, 3]).is_none());
        assert!(pool.find_best(&[]).is_none());
        // Populated pool, empty query → None (compute_lcp of any
        // entry vs `[]` is 0; the function skips zero-LCP matches).
        fill_kv(&mut live, 3, 1.0);
        pool.snapshot_and_insert(vec![1, 2, 3], &live);
        assert!(pool.find_best(&[]).is_none());
    }

    #[test]
    fn touch_advances_lru_so_touched_entry_survives_pressure() {
        // Strengthens `pool_evicts_lru_when_full`: explicitly verify
        // that the LAST-touched entry is the one that survives when
        // the pool overflows. Regression target if someone "fixes"
        // the LRU by tracking insert-time instead of last-use-time.
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(3);
        for tag in 1..=3u32 {
            fill_kv(&mut live, 3, tag as f32);
            pool.snapshot_and_insert(vec![tag, tag, tag], &live);
        }
        // Touch the OLDEST entry to flip the LRU order.
        let (idx, _) = pool.find_best(&[1, 1, 1]).unwrap();
        pool.touch(idx);
        // Insert a 4th entry — pool is full, must evict.
        fill_kv(&mut live, 3, 4.0);
        pool.snapshot_and_insert(vec![4, 4, 4], &live);
        assert_eq!(pool.len(), 3);
        assert!(
            pool.entries.iter().any(|e| e.ids == vec![1, 1, 1]),
            "touched entry [1,1,1] must survive"
        );
        // Either [2,2,2] or whatever LRU was at the moment of insert
        // got evicted; [4,4,4] is fresh; [1,1,1] is touched.
        assert!(pool.entries.iter().any(|e| e.ids == vec![4, 4, 4]));
    }

    #[test]
    fn hash_filter_does_not_cause_false_matches() {
        // The pool stores `hash` for each entry as a fast filter, but
        // `find_best` always runs an actual `compute_lcp`. Pin that
        // hash-equal-but-content-different entries (an artificially
        // collided fixture) don't cause a wrong match. We can't
        // easily produce a real fnv1a collision; verify the simpler
        // contract that `find_best` only considers entries with a
        // non-zero LCP against the query.
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        fill_kv(&mut live, 3, 1.0);
        pool.snapshot_and_insert(vec![10, 20, 30], &live);
        // Query with a token sequence that shares NO prefix.
        assert!(
            pool.find_best(&[99, 98, 97]).is_none(),
            "no shared prefix → no match, regardless of hash table contents"
        );
    }

    // ---- P-1: VLM image-hash gating ----------------------------------

    #[test]
    fn image_hash_filters_block_cross_class_matches() {
        // Insert a text-only entry and a VLM entry with overlapping
        // token ids. Each should only be visible to its own
        // image_hash class — preventing cross-contamination.
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        fill_kv(&mut live, 4, 1.0);
        pool.snapshot_and_insert(vec![1, 2, 3, 4], &live); // text-only
        fill_kv(&mut live, 4, 2.0);
        pool.snapshot_and_insert_with_image_hash(
            vec![1, 2, 3, 4],
            &live,
            Some(0xCAFEBABE),
        );
        assert_eq!(pool.len(), 2, "different image classes coexist");

        // Text-only query should match the text-only entry only.
        let (_, lcp) = pool.find_best(&[1, 2, 3, 4]).expect("text match");
        assert_eq!(lcp, 4);
        let entry = pool
            .find_best_with_image_hash(&[1, 2, 3, 4], None)
            .unwrap();
        assert_eq!(pool.entries[entry.0].image_hash, None);

        // Image query with same hash matches the VLM entry only.
        let entry = pool
            .find_best_with_image_hash(&[1, 2, 3, 4], Some(0xCAFEBABE))
            .unwrap();
        assert_eq!(pool.entries[entry.0].image_hash, Some(0xCAFEBABE));

        // Image query with a DIFFERENT hash misses entirely.
        assert!(
            pool.find_best_with_image_hash(&[1, 2, 3, 4], Some(0xDEADBEEF))
                .is_none(),
            "different image hash must NOT match"
        );
    }

    #[test]
    fn snapshot_with_image_hash_records_kv_len_and_hash() {
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 64, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        // Synthesize a "VLM" snapshot where ids.len() = 4 but the
        // expanded KV length is 20 (e.g. one image placeholder
        // contributed 17 patches). Set live.seq_len = 20 then
        // snapshot the 4-token ids; for VLM the snapshot keeps the
        // expanded kv_len.
        fill_kv(&mut live, 20, 1.0);
        pool.snapshot_and_insert_with_image_hash(
            vec![10, 20, 30, 40],
            &live,
            Some(0xFEEDFACE),
        );
        let snap = &pool.entries[0];
        assert_eq!(snap.ids, vec![10, 20, 30, 40]);
        assert_eq!(snap.kv_len, 20, "VLM entry holds expanded kv_len");
        assert_eq!(snap.image_hash, Some(0xFEEDFACE));
    }

    #[test]
    fn find_full_match_with_image_hash_returns_only_complete_prefix_entries() {
        // Pool has a 4-token entry + a 7-token entry, both with the
        // same image hash. A query of length 5 should pick the
        // 4-token entry (full match); the 7-token entry is too long.
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 32, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        fill_kv(&mut live, 4, 1.0);
        pool.snapshot_and_insert_with_image_hash(
            vec![1, 2, 3, 4],
            &live,
            Some(0xAA),
        );
        fill_kv(&mut live, 7, 2.0);
        pool.snapshot_and_insert_with_image_hash(
            vec![1, 2, 3, 4, 5, 6, 7],
            &live,
            Some(0xAA),
        );
        // dedup may have collapsed these — see how many remain.
        // The 4-token is a prefix of the 7-token, so the 4-token
        // entry should have been removed by the dedup pass during
        // the 7-token insert.
        assert_eq!(
            pool.len(),
            1,
            "4-token snapshot is a prefix of the 7-token one — dedup keeps the longer"
        );
        // Query [1,2,3,4,5] — too short to fully match the 7-token entry.
        assert!(
            pool.find_full_match_with_image_hash(&[1, 2, 3, 4, 5], Some(0xAA))
                .is_none(),
            "query shorter than entry → no full match"
        );
        // Query [1..=7,8] — long enough to fully match the 7-token entry.
        let idx = pool
            .find_full_match_with_image_hash(&[1, 2, 3, 4, 5, 6, 7, 8], Some(0xAA))
            .expect("full match");
        assert_eq!(pool.entries[idx].ids.len(), 7);
    }

    #[test]
    fn fnv1a64_bytes_chained_distinguishes_concatenations() {
        // [a, b] vs [ab] must NOT hash to the same value (the length
        // terminator between slabs ensures this).
        let a: &[u8] = &[1, 2, 3];
        let b: &[u8] = &[4, 5, 6];
        let h_split = fnv1a64_bytes_chained(&[a, b]);
        let ab: &[u8] = &[1, 2, 3, 4, 5, 6];
        let h_joined = fnv1a64_bytes_chained(&[ab]);
        assert_ne!(
            h_split, h_joined,
            "split vs joined slabs must hash differently"
        );
    }

    // ---- Phase 5: hybrid anchors -------------------------------------

    fn mk_anchor(pool_kv: &KvCache, ids: Vec<u32>) -> PrefixSnapshot {
        use rustllama_models::llama_arch::{DeltaNetLayerState, DeltaNetSnapshot};
        let kv = pool_kv.snapshot_prefix(ids.len().min(pool_kv.seq_len));
        let kv_len = kv.prefix_len;
        PrefixSnapshot {
            hash: fnv1a64(&ids),
            ids,
            image_hash: None,
            kv_len,
            kv,
            dn: Some(DeltaNetSnapshot {
                layers: vec![DeltaNetLayerState {
                    conv_state: vec![1.0; 4],
                    recurrent_state: vec![2.0; 4],
                }],
            }),
            last_used: 0,
        }
    }

    #[test]
    fn anchor_entries_keep_nested_prefixes() {
        // Dense dedup collapses a shorter prefix entry into the longer
        // one — correct when restore works at any LCP. Anchors restore
        // only in full, so a shorter anchor is NOT dominated: after an
        // agent edit truncates history to the shorter boundary, it's
        // the only entry that saves the re-prefill. Pin that nested
        // anchors coexist.
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        fill_kv(&mut live, 4, 1.0);
        pool.insert_snapshot(mk_anchor(&live, vec![1, 2, 3, 4]));
        fill_kv(&mut live, 7, 2.0);
        pool.insert_snapshot(mk_anchor(&live, vec![1, 2, 3, 4, 5, 6, 7]));
        assert_eq!(pool.len(), 2, "nested anchors must both survive");

        // A prompt containing only the shorter anchor restores it.
        let (idx, len) = pool
            .find_full_match_hybrid(&[1, 2, 3, 4, 99])
            .expect("short anchor matches");
        assert_eq!(len, 4);
        assert_eq!(pool.entries()[idx].ids, vec![1, 2, 3, 4]);
        // A prompt containing both picks the longer one.
        let (_, len) = pool
            .find_full_match_hybrid(&[1, 2, 3, 4, 5, 6, 7, 8])
            .expect("long anchor matches");
        assert_eq!(len, 7);
        // Partial overlap with the longer anchor only → no full
        // match beyond the short one (all-or-nothing restore).
        let (_, len) = pool
            .find_full_match_hybrid(&[1, 2, 3, 4, 5, 6, 99])
            .expect("short anchor still matches");
        assert_eq!(len, 4, "partial overlap of the 7-anchor must not match it");
    }

    #[test]
    fn identical_ids_anchor_reinsert_replaces_in_place() {
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        fill_kv(&mut live, 4, 1.0);
        pool.insert_snapshot(mk_anchor(&live, vec![1, 2, 3, 4]));
        fill_kv(&mut live, 4, 9.0);
        pool.insert_snapshot(mk_anchor(&live, vec![1, 2, 3, 4]));
        assert_eq!(pool.len(), 1, "identical-ids anchor reinsert must not grow the pool");
    }

    #[test]
    fn find_full_match_hybrid_ignores_dense_entries() {
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        fill_kv(&mut live, 4, 1.0);
        pool.snapshot_and_insert(vec![1, 2, 3, 4], &live); // dense, no dn
        assert!(
            pool.find_full_match_hybrid(&[1, 2, 3, 4, 5]).is_none(),
            "dense entries carry no recurrent state — never anchor-restorable"
        );
    }

    // ---- Tranche 2: sparse KV + pool byte cap ------------------------

    #[test]
    fn sparse_kv_allocates_only_kept_layers_and_round_trips() {
        let cfg = synthetic_cfg(); // 2 layers
        let keep = [true, false];
        let mut live = KvCache::new_with_dtype_sparse(&cfg, 16, KvDtype::F32, &keep);
        // Only layer 0 holds bytes.
        let dense = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
        assert_eq!(live.approx_bytes() * 2, dense.approx_bytes());
        // Fill layer 0 only (fill_kv writes every layer's F32 slab —
        // layer 1's is empty, iter_mut no-ops).
        fill_kv(&mut live, 5, 3.0);
        // Snapshot emits a placeholder for the empty layer instead of
        // panicking on the max_ctx-strided slice.
        let snap = live.snapshot_prefix(5);
        assert!(snap.layers[0].approx_bytes() > 0);
        assert_eq!(snap.layers[1].approx_bytes(), 0, "placeholder for skipped layer");
        // Restore round-trips and skips the placeholder.
        live.reset();
        live.restore_prefix(&snap);
        assert_eq!(live.seq_len, 5);
        assert_eq!(live.layers[1].approx_bytes(), 0);
    }

    #[test]
    fn selective_snapshot_tolerates_keep_true_on_sparse_empty_slab() {
        // The anchor keep-mask marks the MTP block keep=true (it IS a
        // full-attention layer) while the sparse cache skipped its
        // slab — the selective snapshot must emit a placeholder, not
        // panic (this fires on the very first hybrid anchor).
        let cfg = synthetic_cfg();
        let keep_alloc = [true, false];
        let mut live = KvCache::new_with_dtype_sparse(&cfg, 16, KvDtype::F32, &keep_alloc);
        fill_kv(&mut live, 4, 1.0);
        let keep_snapshot = [true, true]; // caller wants both
        let snap = live.snapshot_prefix_selective(4, &keep_snapshot);
        assert!(snap.layers[0].approx_bytes() > 0);
        assert_eq!(snap.layers[1].approx_bytes(), 0);
    }

    #[test]
    fn pool_byte_cap_evicts_lru_and_refuses_oversize() {
        let cfg = synthetic_cfg();
        let mut live = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(8);
        fill_kv(&mut live, 8, 1.0);
        pool.snapshot_and_insert(vec![1; 8], &live);
        let one = pool.approx_bytes();
        assert!(one > 0);
        // Cap at ~1.5 entries: the second insert must evict the first.
        pool.set_max_bytes(one + one / 2);
        fill_kv(&mut live, 8, 2.0);
        pool.snapshot_and_insert(vec![2; 8], &live);
        assert_eq!(pool.len(), 1, "byte cap must have evicted the LRU entry");
        assert!(pool.entries()[0].ids == vec![2; 8]);
        // An entry larger than the whole cap is refused outright.
        pool.set_max_bytes(one / 2);
        fill_kv(&mut live, 8, 3.0);
        pool.snapshot_and_insert(vec![3; 8], &live);
        assert!(
            !pool.entries().iter().any(|e| e.ids == vec![3; 8]),
            "over-cap entry must be refused"
        );
    }

    #[test]
    fn fnv1a64_bytes_chained_is_deterministic() {
        let a: &[u8] = &[1, 2, 3];
        let b: &[u8] = &[4, 5, 6];
        assert_eq!(
            fnv1a64_bytes_chained(&[a, b]),
            fnv1a64_bytes_chained(&[a, b])
        );
    }
}
