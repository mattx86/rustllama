//! LMCache cold tier — CPU-RAM KV offload for the paged backend.
//!
//! When a KV block's hot page is about to be PHYSICALLY reallocated (its content
//! overwritten for a new block), vLLM APC alone loses it — a later request with
//! that same prefix would have to recompute it. This module keeps those evicted
//! blocks' page bytes in a byte-capped CPU-RAM store keyed by block hash, so an
//! APC lookup that misses the hot index can be served from here (PROMOTE back
//! into a fresh page) instead of recomputing. It is the CPU tier of an
//! LMCache-style hierarchy; a disk tier layers underneath it later.
//!
//! Pure bookkeeping over owned byte buffers — it never touches the paged store
//! or the page table. The engine copies a page's bytes in on demote
//! ([`crate::paged_kv_store::PagedKvStore::read_page`]) and back out on promote
//! ([`write_page`](crate::paged_kv_store::PagedKvStore::write_page)).

use std::collections::HashMap;

use crate::apc::BlockHash;
use crate::kv_disk_tier::KvDiskTier;

struct ColdEntry {
    bytes: Vec<f32>,
    last_used: u64,
}

/// CPU-RAM cold tier: block-hash → the block's serialized page bytes, LRU-evicted
/// under a byte budget. `cap_bytes == 0` disables it (every op is a no-op).
#[derive(Default)]
pub struct KvColdTier {
    map: HashMap<u64, ColdEntry>,
    clock: u64,
    bytes: usize,     // current total payload bytes held
    cap_bytes: usize, // 0 = disabled
    hits: u64,
    misses: u64,
    /// Optional disk tier UNDER the CPU tier: a block evicted from RAM is
    /// demoted here instead of lost, and a RAM miss falls through to it.
    disk: Option<KvDiskTier>,
}

impl KvColdTier {
    pub fn new(cap_bytes: usize) -> Self {
        Self {
            cap_bytes,
            ..Default::default()
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.cap_bytes > 0
    }

    /// Attach a disk tier UNDER this CPU tier (blocks evicted from RAM demote to
    /// it; RAM misses fall through to it). Requires the CPU tier enabled.
    pub fn attach_disk(&mut self, disk: KvDiskTier) {
        self.disk = Some(disk);
    }

    pub fn has_disk(&self) -> bool {
        self.disk.is_some()
    }

    /// Set the byte budget (e.g. from config at engine init), evicting LRU
    /// entries if the new cap is smaller. `0` disables + clears the tier.
    pub fn set_cap_bytes(&mut self, cap_bytes: usize) {
        self.cap_bytes = cap_bytes;
        if cap_bytes == 0 {
            self.map.clear();
            self.bytes = 0;
            return;
        }
        while self.bytes > cap_bytes {
            match self.lru_key() {
                Some(k) => {
                    if let Some(e) = self.map.remove(&k) {
                        self.bytes -= e.bytes.len() * std::mem::size_of::<f32>();
                    }
                }
                None => break,
            }
        }
    }

    /// Offload a block's page bytes under its hash. No-op when disabled or a
    /// single block exceeds the whole budget. Evicts LRU entries to make room.
    pub fn demote(&mut self, hash: BlockHash, bytes: &[f32]) {
        if self.cap_bytes == 0 {
            return;
        }
        let cost = bytes.len() * std::mem::size_of::<f32>();
        if cost > self.cap_bytes {
            return; // a single block can never fit → don't thrash the tier
        }
        // Drop any prior entry for this hash first (replace, not double-count).
        if let Some(old) = self.map.remove(&hash.0) {
            self.bytes -= old.bytes.len() * std::mem::size_of::<f32>();
        }
        while self.bytes + cost > self.cap_bytes {
            match self.lru_key() {
                Some(k) => {
                    if let Some(e) = self.map.remove(&k) {
                        self.bytes -= e.bytes.len() * std::mem::size_of::<f32>();
                        // Don't lose it — demote one tier down to disk.
                        if let Some(disk) = self.disk.as_mut() {
                            disk.demote(BlockHash(k), &e.bytes);
                        }
                    }
                }
                None => break,
            }
        }
        self.clock += 1;
        self.map.insert(
            hash.0,
            ColdEntry {
                bytes: bytes.to_vec(),
                last_used: self.clock,
            },
        );
        self.bytes += cost;
    }

    /// Fetch a copy of a block's bytes by hash, bumping its LRU stamp. The copy
    /// is KEPT in the tier (so another request can promote it too) — the byte
    /// budget bounds it. `None` = miss. Records hit/miss.
    pub fn promote(&mut self, hash: BlockHash) -> Option<Vec<f32>> {
        self.clock += 1;
        let clock = self.clock;
        if let Some(e) = self.map.get_mut(&hash.0) {
            e.last_used = clock;
            self.hits += 1;
            return Some(e.bytes.clone());
        }
        // RAM miss → fall through to the disk tier.
        if let Some(disk) = self.disk.as_mut() {
            if let Some(bytes) = disk.promote(hash) {
                self.hits += 1;
                return Some(bytes);
            }
        }
        self.misses += 1;
        None
    }

    pub fn contains(&self, hash: BlockHash) -> bool {
        self.map.contains_key(&hash.0)
    }

    fn lru_key(&self) -> Option<u64> {
        self.map
            .iter()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(&k, _)| k)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    pub fn bytes(&self) -> usize {
        self.bytes
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
    fn disabled_tier_is_a_noop() {
        let mut t = KvColdTier::new(0);
        assert!(!t.is_enabled());
        t.demote(BlockHash(1), &[1.0, 2.0, 3.0]);
        assert!(t.is_empty());
        assert!(t.promote(BlockHash(1)).is_none());
    }

    #[test]
    fn demote_promote_roundtrips() {
        let mut t = KvColdTier::new(1 << 20);
        let block = vec![1.5f32, 2.5, 3.5, 4.5];
        t.demote(BlockHash(42), &block);
        assert!(t.contains(BlockHash(42)));
        assert_eq!(t.promote(BlockHash(42)).as_deref(), Some(block.as_slice()));
        // Promote keeps the copy — a second promote still hits.
        assert!(t.promote(BlockHash(42)).is_some());
        assert_eq!(t.hits(), 2);
    }

    #[test]
    fn lru_evicts_under_byte_cap() {
        // Cap holds exactly two 2-element (8-byte) blocks = 16 bytes.
        let mut t = KvColdTier::new(16);
        t.demote(BlockHash(1), &[1.0, 1.0]);
        t.demote(BlockHash(2), &[2.0, 2.0]);
        assert_eq!(t.len(), 2);
        // Touch block 1 so block 2 is the LRU victim.
        let _ = t.promote(BlockHash(1));
        // Insert block 3 → evict block 2.
        t.demote(BlockHash(3), &[3.0, 3.0]);
        assert_eq!(t.len(), 2);
        assert!(t.contains(BlockHash(1)));
        assert!(!t.contains(BlockHash(2)), "LRU victim evicted");
        assert!(t.contains(BlockHash(3)));
        assert!(t.bytes() <= 16);
    }

    #[test]
    fn oversized_block_is_rejected() {
        let mut t = KvColdTier::new(8); // 2 f32
        t.demote(BlockHash(1), &[1.0, 2.0, 3.0, 4.0]); // 16 bytes > cap
        assert!(t.is_empty());
    }

    #[test]
    fn cpu_tier_evicts_to_disk_and_promotes_back() {
        use crate::kv_disk_tier::KvDiskTier;
        let p = std::env::temp_dir().join(format!(
            "rustllama-coldtier-disk-{}.bin",
            std::process::id()
        ));
        // CPU tier holds exactly one 2-elem (8-byte) block; disk holds more.
        let mut t = KvColdTier::new(8);
        t.attach_disk(KvDiskTier::new(&p, 2, 1 << 20).expect("disk"));
        t.demote(BlockHash(1), &[1.0, 1.0]);
        // Inserting block 2 evicts block 1 from RAM → it demotes to disk.
        t.demote(BlockHash(2), &[2.0, 2.0]);
        assert!(!t.contains(BlockHash(1)), "block 1 left the RAM tier");
        // Promoting block 1 → RAM miss → served from the disk tier.
        assert_eq!(
            t.promote(BlockHash(1)).as_deref(),
            Some([1.0f32, 1.0].as_slice())
        );
        let _ = std::fs::remove_file(&p);
    }
}
