//! CacheBlend — vLLM-style **non-prefix** KV reuse for RAG.
//!
//! Automatic prefix caching ([`crate::apc`]) only reuses a KV block when it is
//! an exact *prefix* of the new request (every preceding token identical). RAG
//! breaks that: the retrieved chunks that make up a prompt are concatenated in
//! different orders across requests, and each one is almost never at the front.
//! CacheBlend lifts the prefix restriction — it reuses the KV of a text chunk
//! **wherever** it lands in the assembled context.
//!
//! The idea (Yao et al., "CacheBlend: Fast LLM Serving for RAG"):
//!   1. Precompute each chunk's KV **in isolation** (position 0, no surrounding
//!      context) once, and cache it content-addressed ([`ChunkKvCache`]).
//!   2. At serve time, assemble the chunks (+ the query suffix) into one context
//!      and load each chunk's cached KV at its real offset. A standalone prefill
//!      rotates each token's K by its within-chunk index, so placing the chunk
//!      at offset `off` needs one extra RoPE rotation by the constant `off`
//!      (RoPE is angle-additive: `rope-at-j` ∘ `rope-at-off` = `rope-at-(j+off)`,
//!      exactly a fresh prefill at that offset). V is position-independent and
//!      reused verbatim.
//!   3. Isolated KV misses the *cross-chunk* attention the real context would
//!      have. CacheBlend recovers it by **selectively recomputing** only the
//!      small fraction of tokens whose KV deviates most from the isolated value
//!      (the "high KV-deviation" / HKVD tokens), reusing the cached KV for the
//!      rest. At `recompute_ratio = 1.0` every token is recomputed, so the pass
//!      is numerically a full prefill — that identity is the parity anchor.
//!
//! This module holds the data types + the content-addressed chunk cache. The
//! selective-recompute forward pass itself lives in
//! [`crate::llama_arch::LlamaModel::forward_prefill_cacheblend`] because it needs
//! the model's private forward internals.

use std::collections::HashMap;
use std::sync::Arc;

use crate::llama_arch::KvSnapshot;

// FNV-1a 64 — the same hash family [`crate::apc`] and the expert cache use.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Content hash of a chunk's token ids (order-sensitive). A CacheBlend chunk is
/// computed in ISOLATION at position 0, so — unlike the APC prefix-chain, which
/// must fold in every preceding block — its KV depends only on its own tokens.
/// A plain content hash of the ids therefore fully identifies the cached KV
/// (for a fixed model + KV dtype).
pub fn chunk_content_hash(tokens: &[i32]) -> u64 {
    let mut h = FNV_OFFSET;
    for &t in tokens {
        for b in (t as u32).to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

/// One chunk placed into the assembled context: its isolated KV snapshot
/// (computed standalone at position 0) and the absolute offset where it begins.
/// Placements are contiguous and cover `[0, suffix_start)`; the trailing
/// `[suffix_start, n)` positions are the always-recomputed query suffix.
#[derive(Clone)]
pub struct ChunkPlacement {
    pub offset: usize,
    pub snapshot: Arc<KvSnapshot>,
}

/// Knobs for the selective-recompute pass.
#[derive(Clone, Copy, Debug)]
pub struct CacheBlendConfig {
    /// Fraction of chunk tokens to recompute, `r ∈ (0, 1]`. The HKVD set is the
    /// top-`ceil(r · n_chunk_tokens)` chunk positions by KV deviation; the query
    /// suffix is always recomputed on top. `r >= 1.0` recomputes everything →
    /// the pass equals a full prefill (the parity anchor).
    pub recompute_ratio: f32,
    /// The layer whose per-token K deviation (fresh vs isolated-cached) selects
    /// the HKVD set. Layer 0's K is `proj(norm(embed), pos)` — no attention yet,
    /// so the isolated and assembled K are identical and carry no signal; the
    /// deviation only emerges once a layer of attention has mixed cross-chunk
    /// information, i.e. layer ≥ 1. Default 1. Layers `0..=selection_layer` are
    /// recomputed in full; only layers above it run the HKVD-only subset.
    pub selection_layer: usize,
}

impl Default for CacheBlendConfig {
    fn default() -> Self {
        Self { recompute_ratio: 0.15, selection_layer: 1 }
    }
}

fn snapshot_bytes(snap: &KvSnapshot) -> usize {
    snap.layers.iter().map(|l| l.approx_bytes()).sum()
}

struct ChunkEntry {
    snapshot: Arc<KvSnapshot>,
    bytes: usize,
    last_used: u64,
}

/// Content-addressed chunk-KV cache: `content_hash → isolated KV snapshot`.
///
/// Byte-capped LRU. A chunk's KV is position-independent (computed at pos 0), so
/// the same chunk text reused across requests — the common RAG case where a hot
/// document is retrieved by many queries — hits here and skips its prefill
/// entirely. Disabled (stores nothing) at `cap_bytes == 0`.
pub struct ChunkKvCache {
    map: HashMap<u64, ChunkEntry>,
    cap_bytes: usize,
    cur_bytes: usize,
    clock: u64,
    hits: u64,
    misses: u64,
}

impl ChunkKvCache {
    pub fn new(cap_bytes: usize) -> Self {
        Self {
            map: HashMap::new(),
            cap_bytes,
            cur_bytes: 0,
            clock: 0,
            hits: 0,
            misses: 0,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.cap_bytes > 0
    }

    /// Fetch a chunk's cached KV by content hash, bumping its LRU stamp.
    pub fn get(&mut self, hash: u64) -> Option<Arc<KvSnapshot>> {
        self.clock += 1;
        let clock = self.clock;
        match self.map.get_mut(&hash) {
            Some(e) => {
                e.last_used = clock;
                self.hits += 1;
                Some(e.snapshot.clone())
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    pub fn contains(&self, hash: u64) -> bool {
        self.map.contains_key(&hash)
    }

    /// Insert a chunk's isolated KV snapshot. Evicts LRU entries until the new
    /// total fits the cap. A single snapshot larger than the whole cap is
    /// dropped (not stored) rather than evicting everything to no benefit.
    pub fn insert(&mut self, hash: u64, snapshot: Arc<KvSnapshot>) {
        if self.cap_bytes == 0 {
            return;
        }
        let bytes = snapshot_bytes(&snapshot);
        if bytes > self.cap_bytes {
            return;
        }
        // Replacing an existing entry: drop its bytes first.
        if let Some(old) = self.map.remove(&hash) {
            self.cur_bytes = self.cur_bytes.saturating_sub(old.bytes);
        }
        while self.cur_bytes + bytes > self.cap_bytes {
            match self.lru_key() {
                Some(victim) => {
                    if let Some(e) = self.map.remove(&victim) {
                        self.cur_bytes = self.cur_bytes.saturating_sub(e.bytes);
                    }
                }
                None => break,
            }
        }
        self.clock += 1;
        self.cur_bytes += bytes;
        self.map.insert(
            hash,
            ChunkEntry { snapshot, bytes, last_used: self.clock },
        );
    }

    fn lru_key(&self) -> Option<u64> {
        self.map.iter().min_by_key(|(_, e)| e.last_used).map(|(&k, _)| k)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    pub fn bytes(&self) -> usize {
        self.cur_bytes
    }
    pub fn hits(&self) -> u64 {
        self.hits
    }
    pub fn misses(&self) -> u64 {
        self.misses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama_arch::{KvBuf, KvDtype, KvLayer};

    fn mk_snap(n_layers: usize, n_kv_heads: usize, head_dim: usize, prefix_len: usize) -> Arc<KvSnapshot> {
        let per = n_kv_heads * prefix_len * head_dim;
        let layers = (0..n_layers)
            .map(|li| KvLayer::F32 {
                k: KvBuf::from_vec(vec![li as f32 + 1.0; per]),
                v: KvBuf::from_vec(vec![li as f32 + 2.0; per]),
            })
            .collect();
        Arc::new(KvSnapshot {
            layers,
            prefix_len,
            n_kv_heads,
            head_dim,
            dtype: KvDtype::F32,
        })
    }

    #[test]
    fn content_hash_is_order_sensitive() {
        assert_eq!(chunk_content_hash(&[1, 2, 3]), chunk_content_hash(&[1, 2, 3]));
        assert_ne!(chunk_content_hash(&[1, 2, 3]), chunk_content_hash(&[3, 2, 1]));
        assert_ne!(chunk_content_hash(&[1, 2, 3]), chunk_content_hash(&[1, 2]));
    }

    #[test]
    fn disabled_cache_stores_nothing() {
        let mut c = ChunkKvCache::new(0);
        assert!(!c.is_enabled());
        c.insert(7, mk_snap(2, 2, 4, 3));
        assert!(c.get(7).is_none());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn insert_get_roundtrips() {
        let mut c = ChunkKvCache::new(1 << 20);
        let s = mk_snap(2, 2, 4, 3);
        c.insert(7, s.clone());
        assert!(c.contains(7));
        let got = c.get(7).expect("hit");
        assert_eq!(got.prefix_len, 3);
        assert_eq!(c.hits(), 1);
        assert!(c.get(99).is_none());
        assert_eq!(c.misses(), 1);
    }

    #[test]
    fn lru_evicts_under_byte_cap() {
        // One snapshot = 2 layers * 2 heads * 2 len * 2 hd * 4 B * 2 (k+v) = 256 B.
        let one = snapshot_bytes(&mk_snap(2, 2, 2, 2));
        let mut c = ChunkKvCache::new(one * 2 + one / 2); // room for 2, not 3
        c.insert(1, mk_snap(2, 2, 2, 2));
        c.insert(2, mk_snap(2, 2, 2, 2));
        assert_eq!(c.len(), 2);
        let _ = c.get(1); // touch 1 → 2 is now LRU
        c.insert(3, mk_snap(2, 2, 2, 2)); // evicts 2
        assert_eq!(c.len(), 2);
        assert!(c.contains(1));
        assert!(!c.contains(2));
        assert!(c.contains(3));
    }

    #[test]
    fn oversized_snapshot_not_stored() {
        let one = snapshot_bytes(&mk_snap(2, 2, 2, 2));
        let mut c = ChunkKvCache::new(one / 2);
        c.insert(1, mk_snap(2, 2, 2, 2));
        assert_eq!(c.len(), 0);
        assert_eq!(c.bytes(), 0);
    }

    #[test]
    fn default_config_is_15pct_at_layer_1() {
        let cfg = CacheBlendConfig::default();
        assert!((cfg.recompute_ratio - 0.15).abs() < 1e-6);
        assert_eq!(cfg.selection_layer, 1);
    }
}
