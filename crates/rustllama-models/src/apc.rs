//! Content-addressed block index for vLLM-style **automatic prefix caching**
//! (APC) over the paged KV cache.
//!
//! The paged store already holds KV in fixed-size, ref-counted, relocatable
//! blocks ([`crate::page_table::PageTable`] + [`crate::paged_kv_store`]). What
//! it lacked was a way to find, across ALL requests (including completed ones),
//! a block whose KV was already computed for the same token content. This module
//! is that index: a map from a **block hash** to the [`PageId`] holding that
//! block's KV.
//!
//! The hash is the vLLM **prefix-hash chain**: a block's hash folds in the hash
//! of every block before it, so equal hashes ⇒ identical prefix content ⇒ the
//! cached KV is bit-for-bit reusable (attention at those positions saw exactly
//! the same preceding tokens). This is what lets a new prompt reuse a running
//! block-prefix of some earlier request without a token-by-token scan and
//! without being limited to currently-live slots.
//!
//! This module is pure bookkeeping — it never touches KV bytes or refcounts.
//! Liveness is owned by [`PageTable`](crate::page_table::PageTable)'s refcounts;
//! the index is only a lookup accelerator, so a stale entry (its page reclaimed)
//! is harmless until pruned ([`ApcIndex::remove`] / [`ApcIndex::remove_page`]).

use std::collections::HashMap;

use crate::page_table::PageId;

/// A content hash identifying a KV block together with every block before it —
/// the vLLM prefix-hash chain `h(block_i) = H(h(block_{i-1}) ‖ tokens_i)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHash(pub u64);

// FNV-1a 64 — the same hash family the prefix cache + expert cache use here.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[inline]
fn fnv1a_u64(mut h: u64, v: u64) -> u64 {
    for b in v.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Chain the next block's hash from the previous block's hash (`None` for the
/// first block → a fixed seed) and the block's token ids. Folding in the
/// predecessor is what makes the hash encode the full prefix, not just the
/// block's own tokens.
pub fn chain_block_hash(prev: Option<BlockHash>, tokens: &[u32]) -> BlockHash {
    let mut h = fnv1a_u64(FNV_OFFSET, prev.map(|b| b.0).unwrap_or(0));
    for &t in tokens {
        h = fnv1a_u64(h, t as u64);
    }
    BlockHash(h)
}

/// The full block-hash chain for a token sequence at `page_size` tokens/block.
/// Only **full** blocks are hashed — a trailing partial block's content isn't
/// finalized, so it has no stable hash and can't be shared. One hash per full
/// block, in order.
pub fn block_hash_chain(tokens: &[u32], page_size: usize) -> Vec<BlockHash> {
    if page_size == 0 {
        return Vec::new();
    }
    let full_blocks = tokens.len() / page_size;
    let mut out = Vec::with_capacity(full_blocks);
    let mut prev = None;
    for i in 0..full_blocks {
        let blk = &tokens[i * page_size..(i + 1) * page_size];
        let h = chain_block_hash(prev, blk);
        out.push(h);
        prev = Some(h);
    }
    out
}

#[derive(Debug, Clone, Copy)]
struct Entry {
    page: PageId,
    last_used: u64,
}

/// Global content-addressed block index: block-hash → the page holding that
/// block's KV. Shared across requests (so a completed request's blocks are
/// reusable, not just live ones). Eviction is LRU over entries; the page
/// refcount, not this index, governs whether the KV is actually still resident,
/// so a lookup that returns a since-reclaimed page is caught by the caller
/// (which holds the authoritative [`PageTable`]).
#[derive(Debug, Default)]
pub struct ApcIndex {
    map: HashMap<u64, Entry>,
    clock: u64,
    hits: u64,
    misses: u64,
}

impl ApcIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a finalized block's page under its hash. A re-registered hash
    /// repoints at the freshest page (the old page is freed by its owner).
    pub fn register(&mut self, hash: BlockHash, page: PageId) {
        self.clock += 1;
        self.map.insert(hash.0, Entry { page, last_used: self.clock });
    }

    /// Look up one block hash, bumping its LRU stamp. `None` = miss.
    pub fn get(&mut self, hash: BlockHash) -> Option<PageId> {
        self.clock += 1;
        let clock = self.clock;
        self.map.get_mut(&hash.0).map(|e| {
            e.last_used = clock;
            e.page
        })
    }

    /// Longest matching leading run: for a prompt's block-hash chain, return the
    /// pages for the longest PREFIX of blocks all present in the index (a gap
    /// stops the run — vLLM reuses a contiguous prefix). Records one hit/miss
    /// for the whole lookup and bumps LRU on the matched blocks.
    pub fn match_prefix(&mut self, chain: &[BlockHash]) -> Vec<PageId> {
        let mut pages = Vec::new();
        for &h in chain {
            match self.get(h) {
                Some(p) => pages.push(p),
                None => break,
            }
        }
        if pages.is_empty() {
            self.misses += 1;
        } else {
            self.hits += 1;
        }
        pages
    }

    /// Drop a hash→page mapping (on block eviction when the hash is known).
    pub fn remove(&mut self, hash: BlockHash) {
        self.map.remove(&hash.0);
    }

    /// Drop every mapping that points at `page` (on page reclaim when the hash
    /// isn't handy). O(n) — the eviction fallback.
    pub fn remove_page(&mut self, page: PageId) {
        self.map.retain(|_, e| e.page != page);
    }

    /// The LRU victim hash among current entries (for a size-capped index).
    pub fn lru_victim(&self) -> Option<BlockHash> {
        self.map
            .iter()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(&h, _)| BlockHash(h))
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
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

    #[test]
    fn chain_is_deterministic_and_prefix_sensitive() {
        // Two sequences sharing the first two blocks hash identically there,
        // and diverge at the third.
        let a: Vec<u32> = (0..48).collect(); // 3 blocks of 16
        let mut b = a.clone();
        b[32] = 9999; // perturb block 2 only
        let ca = block_hash_chain(&a, 16);
        let cb = block_hash_chain(&b, 16);
        assert_eq!(ca.len(), 3);
        assert_eq!(cb.len(), 3);
        assert_eq!(ca[0], cb[0], "block 0 identical");
        assert_eq!(ca[1], cb[1], "block 1 identical");
        assert_ne!(ca[2], cb[2], "block 2 diverges");
    }

    #[test]
    fn position_matters_via_chaining() {
        // The same block content at a different position (different predecessor)
        // must hash differently.
        let blk: Vec<u32> = (0..16).collect();
        let h_at_0 = chain_block_hash(None, &blk);
        let h_at_1 = chain_block_hash(Some(BlockHash(12345)), &blk);
        assert_ne!(h_at_0, h_at_1);
    }

    #[test]
    fn partial_trailing_block_is_not_hashed() {
        let toks: Vec<u32> = (0..40).collect(); // 2 full blocks + 8 leftover
        assert_eq!(block_hash_chain(&toks, 16).len(), 2);
    }

    #[test]
    fn match_prefix_stops_at_first_gap() {
        let mut idx = ApcIndex::new();
        let chain = block_hash_chain(&(0..64).collect::<Vec<u32>>(), 16); // 4 blocks
        // Register blocks 0,1,3 — leave block 2 absent.
        idx.register(chain[0], PageId(10));
        idx.register(chain[1], PageId(11));
        idx.register(chain[3], PageId(13));
        let pages = idx.match_prefix(&chain);
        // The run must stop at the gap (block 2), so only 0,1 match.
        assert_eq!(pages, vec![PageId(10), PageId(11)]);
        assert_eq!(idx.hits(), 1);
    }

    #[test]
    fn register_get_remove_roundtrip() {
        let mut idx = ApcIndex::new();
        let h = chain_block_hash(None, &[1, 2, 3]);
        assert!(idx.get(h).is_none());
        idx.register(h, PageId(7));
        assert_eq!(idx.get(h), Some(PageId(7)));
        idx.remove(h);
        assert!(idx.get(h).is_none());
    }

    #[test]
    fn remove_page_prunes_all_its_hashes() {
        let mut idx = ApcIndex::new();
        idx.register(BlockHash(1), PageId(5));
        idx.register(BlockHash(2), PageId(5));
        idx.register(BlockHash(3), PageId(6));
        idx.remove_page(PageId(5));
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.get(BlockHash(3)), Some(PageId(6)));
    }

    #[test]
    fn lru_victim_is_least_recently_used() {
        let mut idx = ApcIndex::new();
        idx.register(BlockHash(1), PageId(1));
        idx.register(BlockHash(2), PageId(2));
        idx.register(BlockHash(3), PageId(3));
        // Touch 1 and 3; 2 is now the LRU victim.
        let _ = idx.get(BlockHash(1));
        let _ = idx.get(BlockHash(3));
        assert_eq!(idx.lru_victim(), Some(BlockHash(2)));
    }
}
