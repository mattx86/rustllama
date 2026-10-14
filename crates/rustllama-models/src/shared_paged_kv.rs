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

use crate::apc::{ApcIndex, BlockHash};
use crate::page_table::{PageId, PageTable};
use crate::paged_kv_store::PagedKvStore;

/// Multi-owner handle to a paged KV store + its page table.
/// Cheap to clone (`Arc` increments). Pass clones into per-slot
/// `PagedKvCache` instances so they can all reach the same byte
/// storage and free list.
///
/// Also owns the process-wide [`ApcIndex`] (vLLM automatic prefix caching):
/// the content-addressed block index that lets a new request reacquire the KV
/// blocks of an earlier (even completed) request by content hash. Lock order:
/// **table → apc → store** — the acquire path holds table + apc together.
#[derive(Clone)]
pub struct SharedPagedKv {
    store: Arc<Mutex<PagedKvStore>>,
    table: Arc<Mutex<PageTable>>,
    apc: Arc<Mutex<ApcIndex>>,
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
            apc: Arc::new(Mutex::new(ApcIndex::new())),
        }
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

    /// Lock-and-execute helper for the APC block index (register / prune /
    /// stats). Acquired AFTER the table when both are needed.
    pub fn with_apc_mut<R>(&self, f: impl FnOnce(&mut ApcIndex) -> R) -> R {
        let mut a = self.apc.lock().expect("apc lock");
        f(&mut *a)
    }

    /// vLLM APC acquire: for a prompt's block-hash `chain`, find the longest
    /// leading run of blocks present in the index whose pages are still valid,
    /// acquire one reference to each (REACQUIRE a freed page — reviving its
    /// retained KV — or ADD_REF a live one), and return them in page order. The
    /// returned pages are ref-held by the caller, exactly like a fresh `alloc`,
    /// so [`crate::paged_kv_cache::PagedKvCache::pin_shared_prefix`] +
    /// `release` balance them. A stale entry (its page since reused) stops the
    /// run and is pruned. Holds table → apc.
    pub fn apc_acquire_prefix(&self, chain: &[BlockHash]) -> Vec<PageId> {
        let mut table = self.table.lock().expect("table lock");
        let mut apc = self.apc.lock().expect("apc lock");
        let mut out = Vec::new();
        for &h in chain {
            let Some(page) = apc.get(h) else {
                break; // gap — vLLM reuses only a contiguous prefix
            };
            let acquired = if table.refs_of(page) > 0 {
                table.add_ref(&[page]);
                true
            } else {
                table.reacquire(page)
            };
            if acquired {
                out.push(page);
            } else {
                // Page vanished (physically reused) — drop the stale entry.
                apc.remove(h);
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
        let mut apc = self.apc.lock().expect("apc lock");
        for (h, p) in entries {
            apc.register(*h, *p);
        }
    }

    /// Prune stale index entries for pages that were just PHYSICALLY allocated
    /// for new content (their old block hashes no longer describe them). Must
    /// be called right after any fresh `alloc` of pages for a slot.
    pub fn apc_prune_realloc(&self, pages: &[PageId]) {
        if pages.is_empty() {
            return;
        }
        self.with_apc_mut(|apc| apc.remove_pages(pages));
    }

    /// `(hits, misses, entries)` of the APC index, for stats/telemetry.
    pub fn apc_stats(&self) -> (u64, u64, usize) {
        let a = self.apc.lock().expect("apc lock");
        (a.hits(), a.misses(), a.len())
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
        // content → prune drops the stale hashes.
        shared.with_table_mut(|t| t.free(&got));
        let fresh = shared.with_table_mut(|t| t.alloc(2).expect("realloc 2"));
        shared.apc_prune_realloc(&fresh);
        assert!(
            shared.apc_acquire_prefix(&[h0, h1]).is_empty(),
            "pruned hashes no longer resolve after physical realloc"
        );
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
