//! LMCache disk tier — a file-backed KV block store that sits UNDER the CPU
//! cold tier ([`crate::kv_cold_tier::KvColdTier`]).
//!
//! Every KV block is the same size (one page: `page_stride` f32s), so the file
//! is a flat array of fixed-size slots — no fragmentation, no compaction. A
//! block is demoted by writing its bytes to a slot (`hash → slot`), promoted by
//! reading that slot back, and LRU-evicted by reusing the least-recently-used
//! slot. Everything FAILS SOFT: an I/O error degrades to a miss (recompute),
//! never a crash or a wrong answer.
//!
//! v1 is EPHEMERAL (the file is truncated at startup) — it absorbs CPU-tier
//! overflow within a run. Cross-restart persistence (fingerprint-gated, like the
//! `.rlkv` sidecar) is a follow-up.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::apc::BlockHash;

struct DiskEntry {
    slot: u32,
    last_used: u64,
}

/// File-backed fixed-slot KV block store. Created disabled-returning-`None` when
/// the budget or geometry is zero or the file can't be opened.
pub struct KvDiskTier {
    file: File,
    slot_elems: usize, // f32 per block (= page_stride)
    slot_bytes: usize, // slot_elems * 4
    map: HashMap<u64, DiskEntry>,
    free: Vec<u32>,
    clock: u64,
    hits: u64,
    misses: u64,
}

impl KvDiskTier {
    /// Open a disk tier at `path` holding blocks of `slot_elems` f32s, capped at
    /// `cap_bytes` of payload (→ `cap_bytes / (slot_elems*4)` slots). `None` if
    /// disabled (cap 0), geometry is zero, or the file can't be opened.
    pub fn new(path: &Path, slot_elems: usize, cap_bytes: usize) -> Option<Self> {
        if slot_elems == 0 || cap_bytes == 0 {
            return None;
        }
        let slot_bytes = slot_elems.checked_mul(std::mem::size_of::<f32>())?;
        let total_slots = (cap_bytes / slot_bytes).max(1) as u32;
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(path)
            .ok()?;
        Some(Self {
            file,
            slot_elems,
            slot_bytes,
            map: HashMap::new(),
            free: (0..total_slots).collect(),
            clock: 0,
            hits: 0,
            misses: 0,
        })
    }

    /// Offload one block's bytes to a slot (reusing its existing slot, a free
    /// slot, or the LRU victim's slot). No-op on a size mismatch or I/O error.
    pub fn demote(&mut self, hash: BlockHash, bytes: &[f32]) {
        if bytes.len() != self.slot_elems {
            return;
        }
        let slot = if let Some(e) = self.map.get(&hash.0) {
            e.slot
        } else if let Some(s) = self.free.pop() {
            s
        } else if let Some(victim) = self.lru_key() {
            self.map.remove(&victim).map(|e| e.slot).unwrap_or(0)
        } else {
            return;
        };
        let off = (slot as u64) * (self.slot_bytes as u64);
        if self.file.seek(SeekFrom::Start(off)).is_err() {
            return;
        }
        // Serialize f32 → little-endian bytes (no bytemuck dep needed).
        let mut raw = Vec::with_capacity(self.slot_bytes);
        for &x in bytes {
            raw.extend_from_slice(&x.to_le_bytes());
        }
        if self.file.write_all(&raw).is_err() {
            return;
        }
        self.clock += 1;
        self.map.insert(
            hash.0,
            DiskEntry {
                slot,
                last_used: self.clock,
            },
        );
    }

    /// Read a block's bytes back by hash, bumping its LRU stamp. `None` = miss
    /// (counts) or I/O error.
    pub fn promote(&mut self, hash: BlockHash) -> Option<Vec<f32>> {
        self.clock += 1;
        let clock = self.clock;
        let slot = match self.map.get_mut(&hash.0) {
            Some(e) => {
                e.last_used = clock;
                e.slot
            }
            None => {
                self.misses += 1;
                return None;
            }
        };
        let off = (slot as u64) * (self.slot_bytes as u64);
        if self.file.seek(SeekFrom::Start(off)).is_err() {
            self.misses += 1;
            return None;
        }
        let mut buf = vec![0u8; self.slot_bytes];
        if self.file.read_exact(&mut buf).is_err() {
            self.misses += 1;
            return None;
        }
        self.hits += 1;
        // Deserialize little-endian bytes → f32.
        let out: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Some(out)
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

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "rustllama-kvdisk-{name}-{}.bin",
            std::process::id()
        ))
    }

    #[test]
    fn disabled_when_cap_or_geometry_zero() {
        assert!(KvDiskTier::new(&tmp("z0"), 0, 1 << 20).is_none());
        assert!(KvDiskTier::new(&tmp("z1"), 16, 0).is_none());
    }

    #[test]
    fn demote_promote_roundtrips_through_file() {
        let p = tmp("rt");
        let slot_elems = 8;
        let mut t = KvDiskTier::new(&p, slot_elems, 1 << 20).expect("tier");
        let block: Vec<f32> = (0..slot_elems).map(|i| i as f32 * 1.25 - 3.0).collect();
        t.demote(BlockHash(7), &block);
        assert!(t.contains(BlockHash(7)));
        assert_eq!(t.promote(BlockHash(7)).as_deref(), Some(block.as_slice()));
        assert!(t.promote(BlockHash(999)).is_none());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn lru_evicts_slot_under_cap() {
        let p = tmp("lru");
        let slot_elems = 2; // 8 bytes/slot
                            // Cap = 16 bytes = 2 slots.
        let mut t = KvDiskTier::new(&p, slot_elems, 16).expect("tier");
        t.demote(BlockHash(1), &[1.0, 1.0]);
        t.demote(BlockHash(2), &[2.0, 2.0]);
        assert_eq!(t.len(), 2);
        let _ = t.promote(BlockHash(1)); // touch 1 → 2 is LRU
        t.demote(BlockHash(3), &[3.0, 3.0]); // evicts 2's slot
        assert_eq!(t.len(), 2);
        assert!(t.contains(BlockHash(1)));
        assert!(!t.contains(BlockHash(2)));
        assert_eq!(
            t.promote(BlockHash(3)).as_deref(),
            Some([3.0f32, 3.0].as_slice())
        );
        let _ = std::fs::remove_file(&p);
    }
}
