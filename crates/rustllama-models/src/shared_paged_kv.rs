//! Shared (multi-owner) wrapper around `PagedKvStore` + `PageTable`.
//!
//! Single-flight engines (and one-slot CB engines) own their store
//! + table directly and pass `&mut` references to `PagedKvCache`
//! methods — no synchronization needed because at most one cache
//! lives in scope at a time.
//!
//! Continuous-batching engines that serve M concurrent slots from
//! one shared paged pool need a different ownership model: M
//! `PagedKvCache` instances must coexist, each writing its own
//! pages without interfering with the others. This module's
//! [`SharedPagedKv`] is the wrapper that makes that legal — wraps
//! the store and table in `Arc<Mutex<...>>` so per-slot caches can
//! hold a clone of the handle and acquire short-lived locks per
//! call.
//!
//! Lock granularity is whole-store / whole-table for v1. The
//! engine's fused-decode forward pass calls write/gather under
//! the lock for each (layer, slot) pair — for M=4 slots and 28
//! layers that's ~224 lock acquires per decode tick (~10 µs
//! aggregate, negligible vs decode time). A per-page-shard lock
//! is a v1.x perf knob if profiling demands it.
//!
//! Lock ordering policy: **table BEFORE store** to prevent
//! deadlock between admit (acquires table for alloc) and
//! decode (acquires store for write). All public methods on the
//! cache that touch both follow this order.

use std::sync::{Arc, Mutex};

use crate::apc::BlockHash;
use crate::kv_cold_tier::KvColdTier;
use crate::page_table::{PageId, PageTable};
use crate::paged_kv_store::PagedKvStore;

/// Multi-owner handle to a paged KV store + its page table.
/// Cheap to clone (`Arc` increments). Pass clones into per-slot
/// `PagedKvCache` instances so they can all reach the same byte
/// storage and free list.
///
/// The vLLM APC block index lives inside the `PageTable` (so `alloc` auto-prunes
/// reused pages); this wrapper exposes it + the LMCache cold tier. Lock order:
/// **table → store → cold**.
#[derive(Clone)]
pub struct SharedPagedKv {
    store: Arc<Mutex<PagedKvStore>>,
    /// The page table OWNS the APC block index (so `alloc` auto-prunes reused
    /// pages — see page_table.rs).
    table: Arc<Mutex<PageTable>>,
    /// LMCache CPU cold tier — evicted blocks' bytes keyed by block hash.
    /// Disabled (cap 0) until `set_cold_cap_bytes`. Lock order (whole file):
    /// **table → store → cold**.
    cold: Arc<Mutex<KvColdTier>>,
}

impl std::fmt::Debug for SharedPagedKv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Avoid taking the locks during a Debug print — that would
        // deadlock if the print happened from a code path that already
        // holds one. The store + table contents are huge anyway; the
        // useful field is the strong-count.
        f.debug_struct("SharedPagedKv")
            .field("store_refs", &Arc::strong_count(&self.store))
            .field("table_refs", &Arc::strong_count(&self.table))
            .finish()
    }
}

impl SharedPagedKv {
    /// Build a shared handle around a freshly-constructed store +
    /// table. Takes ownership of both — caller relinquishes direct
    /// access in exchange for the `Clone`-able handle.
    pub fn new(store: PagedKvStore, table: PageTable) -> Self {
        Self {
            store: Arc::new(Mutex::new(store)),
            table: Arc::new(Mutex::new(table)),
            cold: Arc::new(Mutex::new(KvColdTier::new(0))),
        }
    }

    /// Set the LMCache cold-tier byte budget (from config at engine init).
    /// `0` disables it.
    pub fn set_cold_cap_bytes(&self, bytes: usize) {
        self.cold.lock().expect("cold lock").set_cap_bytes(bytes);
    }

    /// Snapshot of the free-page count. Cheap (single mutex tap).
    /// Used by the engine's scheduler to decide whether to admit
    /// another slot — if free_pages drops below the admit threshold
    /// new requests queue instead of starting prefill.
    pub fn free_pages(&self) -> usize {
        self.table.lock().expect("table lock").free_count()
    }

    /// Returns `(total_pages, n_layers, n_kv_heads, page_size,
    /// head_dim)` — the immutable geometry shared by every cache
    /// pointing at this store. Lets per-cache code size scratch
    /// buffers without re-locking the store.
    pub fn geometry(&self) -> (u32, u32, u32, u32, u32) {
        let s = self.store.lock().expect("store lock");
        (
            s.total_pages(),
            s.n_layers(),
            s.n_kv_heads(),
            s.page_size(),
            s.head_dim(),
        )
    }

    /// Lock-and-execute helper for store reads/writes. Used by
    /// [`crate::paged_kv_cache::PagedKvCache`]'s shared-access
    /// methods. Keep the closure short — the lock blocks every
    /// other slot.
    pub fn with_store_mut<R>(&self, f: impl FnOnce(&mut PagedKvStore) -> R) -> R {
        let mut s = self.store.lock().expect("store lock");
        f(&mut *s)
    }

    pub fn with_store<R>(&self, f: impl FnOnce(&PagedKvStore) -> R) -> R {
        let s = self.store.lock().expect("store lock");
        f(&*s)
    }

    /// Lock-and-execute helper for table writes (alloc / free).
    /// Always acquired BEFORE `with_store_mut` to maintain the
    /// `table → store` lock order policy documented at the module
    /// level.
    pub fn with_table_mut<R>(&self, f: impl FnOnce(&mut PageTable) -> R) -> R {
        let mut t = self.table.lock().expect("table lock");
        f(&mut *t)
    }

    /// vLLM APC acquire (LMCache-aware): the longest leading run of `chain`
    /// blocks reusable as a prefix. Each block is tried in the HOT index first
    /// (reacquire a freed page / add_ref a live one); on a hot miss it is tried
    /// in the COLD tier — PROMOTED back into a freshly-allocated page (copy bytes
    /// in + re-register). Stops at the first block missing in BOTH tiers. With
    /// the cold tier disabled this is byte-identical to the hot-only acquire.
    /// The returned pages are ref-held (like a fresh `alloc`). Lock order
    /// table → store → cold.
    pub fn apc_acquire_prefix(&self, chain: &[BlockHash]) -> Vec<PageId> {
        let mut table = self.table.lock().expect("table lock");
        let mut out = Vec::new();
        for &h in chain {
            // 1. Hot index.
            if let Some(page) = table.apc_try_acquire(h) {
                out.push(page);
                continue;
            }
            // 2. Cold tier: promote into a fresh page (`alloc` auto-prunes any
            // stale hash on it).
            let mut store = self.store.lock().expect("store lock");
            let mut cold = self.cold.lock().expect("cold lock");
            let Some(bytes) = cold.promote(h) else {
                break;
            };
            let Some(page) = table.alloc(1).map(|v| v[0]) else {
                break; // pool full — can't promote
            };
            if store.write_page(page, &bytes) {
                table.apc_register(h, page);
                out.push(page);
            } else {
                table.free(&[page]);
                break;
            }
        }
        out
    }

    /// Register finalized blocks: each `(block_hash, page)` becomes reusable by
    /// a later request with the same prefix content.
    pub fn apc_register(&self, entries: &[(BlockHash, PageId)]) {
        if entries.is_empty() {
            return;
        }
        let mut table = self.table.lock().expect("table lock");
        for (h, p) in entries {
            table.apc_register(*h, *p);
        }
    }

    /// LMCache demote: before a reuse `alloc` auto-prunes their hashes, offload
    /// the blocks on the `n` pages `alloc` is about to reuse to the cold tier
    /// (keyed by hash — their bytes are still intact at this point). No-op when
    /// the cold tier is disabled. Call right BEFORE growing the cache. Lock
    /// order table → store → cold.
    pub fn apc_demote_reuse(&self, n: usize) {
        if n == 0 {
            return;
        }
        let table = self.table.lock().expect("table lock");
        let reuse = table.peek_reuse(n);
        let evicting = table.apc_hashes_for_pages(&reuse);
        if evicting.is_empty() {
            return;
        }
        let store = self.store.lock().expect("store lock");
        let mut cold = self.cold.lock().expect("cold lock");
        if cold.is_enabled() {
            for (hash, page) in evicting {
                if let Some(bytes) = store.read_page(page) {
                    cold.demote(hash, bytes);
                }
            }
        }
    }

    /// `(hits, misses, entries)` of the APC index, for stats/telemetry.
    pub fn apc_stats(&self) -> (u64, u64, usize) {
        self.table.lock().expect("table lock").apc_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_shared() -> SharedPagedKv {
        // 4 pages, 2 layers, 2 heads, page_size 4, head_dim 8.
        let store = PagedKvStore::new(4, 2, 2, 4, 8).unwrap();
        let table = PageTable::new(4, 4);
        SharedPagedKv::new(store, table)
    }

    #[test]
    fn clone_is_arc_share_not_data_copy() {
        let a = small_shared();
        let b = a.clone();
        // Both clones see the same free-page count; allocating from
        // one shows up in the other's `free_pages()`.
        assert_eq!(a.free_pages(), 4);
        assert_eq!(b.free_pages(), 4);
        a.with_table_mut(|t| {
            t.alloc(2).expect("alloc");
        });
        assert_eq!(a.free_pages(), 2);
        assert_eq!(b.free_pages(), 2, "b sees a's allocation through shared Arc");
    }

    #[test]
    fn geometry_matches_store_construction() {
        let s = small_shared();
        assert_eq!(s.geometry(), (4, 2, 2, 4, 8));
    }

    #[test]
    fn debug_does_not_deadlock_with_locks_held() {
        // Regression test for the easy-to-write mistake of
        // formatting `SharedPagedKv` while holding one of its
        // locks (e.g. inside a `tracing::info!(?shared, ...)`
        // call from a method that's already inside `with_store`).
        // The custom Debug impl deliberately skips the inner locks.
        let s = small_shared();
        s.with_store(|_inner_store| {
            let dbg = format!("{s:?}");
            assert!(dbg.contains("SharedPagedKv"));
            assert!(dbg.contains("store_refs"));
        });
    }

    #[test]
    fn alloc_visible_across_clones() {
        // Three handles to the same shared pool. Handle A allocates
        // 2 pages; handles B and C see the reduced free count; then
        // C frees them and A's view recovers.
        let a = small_shared();
        let b = a.clone();
        let c = a.clone();
        let pages = a.with_table_mut(|t| t.alloc(2).expect("alloc"));
        assert_eq!(b.free_pages(), 2);
        assert_eq!(c.free_pages(), 2);
        c.with_table_mut(|t| t.free(&pages));
        assert_eq!(a.free_pages(), 4);
    }

    #[test]
    fn apc_register_reacquire_and_prune() {
        use crate::apc::chain_block_hash;
        let shared = small_shared(); // 4 pages, page_size 4
        // Slot A: alloc 2 pages, register their block hashes (2 blocks of 4).
        let pages = shared.with_table_mut(|t| t.alloc(2).expect("alloc 2"));
        let h0 = chain_block_hash(None, &[1, 2, 3, 4]);
        let h1 = chain_block_hash(Some(h0), &[5, 6, 7, 8]);
        shared.apc_register(&[(h0, pages[0]), (h1, pages[1])]);

        // Slot A completes → free. Pages drop to refcount 0 but stay in the
        // index (their KV content is still valid + reusable).
        shared.with_table_mut(|t| t.free(&pages));
        assert_eq!(shared.free_pages(), 4);

        // Slot B re-asks the same prefix → reacquires both blocks by content
        // hash, pulling them back off the free list without recompute.
        let got = shared.apc_acquire_prefix(&[h0, h1]);
        assert_eq!(got, pages, "reacquired the same physical pages");
        assert_eq!(shared.free_pages(), 2, "reacquired pages left the free list");

        // A gap in the chain stops the run: an unknown third block isn't shared.
        let h_other = chain_block_hash(Some(h1), &[99, 99, 99, 99]);
        assert_eq!(shared.apc_acquire_prefix(&[h0, h1, h_other]).len(), 2);
        // (that last acquire add_ref'd h0/h1 again — release those extra refs)
        shared.with_table_mut(|t| t.free(&pages));

        // Slot B frees, then those pages are PHYSICALLY realloc'd for new
        // content → `alloc` AUTO-prunes the stale hashes.
        shared.with_table_mut(|t| t.free(&got));
        let _fresh = shared.with_table_mut(|t| t.alloc(2).expect("realloc 2"));
        assert!(
            shared.apc_acquire_prefix(&[h0, h1]).is_empty(),
            "pruned hashes no longer resolve after physical realloc"
        );
    }

    #[test]
    fn lmcache_demote_then_promote_recovers_block_bytes() {
        use crate::apc::chain_block_hash;
        let shared = small_shared(); // 4 pages, 2 layers, 2 heads, page_size 4, head_dim 8
        shared.set_cold_cap_bytes(1 << 20); // enable the cold tier
        let stride = shared.with_store(|s| s.page_stride());
        let h = chain_block_hash(None, &[1, 2, 3, 4]);

        // Request 1: alloc a page, write a distinctive block, register it.
        let page = shared.with_table_mut(|t| t.alloc(1).expect("alloc 1")[0]);
        let payload: Vec<f32> = (0..stride).map(|i| i as f32 + 0.25).collect();
        shared.with_store_mut(|s| assert!(s.write_page(page, &payload)));
        shared.apc_register(&[(h, page)]);

        // Request 1 completes: free the page (content retained, still indexed).
        shared.with_table_mut(|t| t.free(&[page]));
        // A new slot is about to reuse it → DEMOTE its block to the cold tier
        // (peeks the free list before the grow), then `alloc` reuses the page +
        // AUTO-prunes its hot hash, and the slot overwrites it with new content.
        shared.apc_demote_reuse(1);
        let reused = shared.with_table_mut(|t| t.alloc(1).expect("reuse")[0]);
        assert_eq!(reused, page, "reused the just-freed MRU page");
        shared.with_store_mut(|s| assert!(s.write_page(page, &vec![0f32; stride])));

        // Acquire by hash: hot MISS (auto-pruned) → promote from the cold tier
        // into a fresh page holding the original bytes.
        let got = shared.apc_acquire_prefix(&[h]);
        assert_eq!(got.len(), 1, "promoted the block from the cold tier");
        let promoted = got[0];
        assert_ne!(promoted, page, "promoted into a different physical page");
        let read = shared.with_store(|s| s.read_page(promoted).expect("read").to_vec());
        assert_eq!(read, payload, "promoted bytes equal the demoted block");
    }

    #[test]
    fn concurrent_table_alloc_from_multiple_threads_serializes_safely() {
        // 100 threads each try to alloc 1 page from a pool of 20.
        // After all threads finish, exactly 20 should have succeeded
        // and the free count should be 0. The Mutex serializes them
        // — this just verifies no panics / data races / deadlocks
        // and the accounting is exact.
        let store = PagedKvStore::new(20, 1, 1, 1, 1).unwrap();
        let table = PageTable::new(20, 1);
        let shared = SharedPagedKv::new(store, table);
        let mut handles = Vec::new();
        let success_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for _ in 0..100 {
            let s = shared.clone();
            let cnt = Arc::clone(&success_count);
            handles.push(std::thread::spawn(move || {
                let got = s.with_table_mut(|t| t.alloc(1));
                if got.is_some() {
                    cnt.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(
            success_count.load(std::sync::atomic::Ordering::SeqCst),
            20,
            "exactly pool-size allocs should have succeeded"
        );
        assert_eq!(shared.free_pages(), 0);
    }
}
