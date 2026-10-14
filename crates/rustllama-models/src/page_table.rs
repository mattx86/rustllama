//! `PageId` + `PageTable` — opaque page-id type and the free-list
//! allocator that hands ids out, plus the content-addressed APC block
//! index (`ApcIndex`) it owns.
//!
//! The index lives INSIDE the allocator so that `alloc` — the single
//! point where a page is physically re-purposed for new content —
//! automatically prunes that page's stale block hash. This makes the
//! "a reused page can never be reacquired under its old hash" invariant
//! hold by construction for every caller (both the single-request
//! `CpuEngine` and the continuous-batching pool), with no per-call-site
//! pruning to forget.
//!
//! These were originally in `rustllama-engine::batch_scheduler` and
//! moved here so the paged KV-cache types (`PagedKvStore`,
//! `PagedKvCache`) — which live alongside the model's KV cache —
//! can depend on them without forcing a `rustllama-models ➜
//! rustllama-engine` dep cycle. The engine crate re-exports both
//! via `rustllama_engine::batch_scheduler::{PageId, PageTable}` so
//! existing call sites (`rustllama-server`, `batch_scheduler` tests)
//! keep working unchanged.

/// Identifier for one fixed-size KV page. Indexes into the
/// [`PageTable`]'s flat backing storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub u32);

/// Fixed-page allocator. `free` is a stack of `PageId`s — most-
/// recently-freed are reused first, which keeps repeated short
/// sessions touching the same physical pages (warmer L2 / VRAM
/// residency).
///
/// E3.2: pages carry a per-page reference count so two slots can
/// share the K/V data of a common prompt prefix. `alloc` stamps
/// the page with refcount=1; `add_ref` bumps it (the second slot
/// gains a "view" into the same page without copying); `free`
/// decrements and only returns the page to the free list when
/// refcount drops to zero. Single-owner usage (the common path)
/// is bit-identical to the pre-E3.2 alloc/free behavior — every
/// alloc + free pair stays on refcount=1, refcount=0.
#[derive(Debug)]
pub struct PageTable {
    free: Vec<PageId>,
    /// `refs[i]` = number of slots currently holding `PageId(i)`.
    /// `refs[i] == 0` ⇒ the page is on the free list.
    refs: Vec<u32>,
    total: u32,
    page_size: u32,
    /// Content-addressed vLLM APC block index (block-hash → the page holding
    /// that block's KV). Owned here so [`alloc`](Self::alloc) auto-prunes a
    /// reused page's stale hash.
    apc: crate::apc::ApcIndex,
}

impl PageTable {
    pub fn new(total: u32, page_size: u32) -> Self {
        Self {
            free: (0..total).rev().map(PageId).collect(),
            refs: vec![0; total as usize],
            total,
            page_size,
            apc: crate::apc::ApcIndex::new(),
        }
    }

    pub fn total(&self) -> u32 {
        self.total
    }

    pub fn page_size(&self) -> u32 {
        self.page_size
    }

    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// Current reference count of a page. `0` ⇒ on the free list (its content
    /// is still valid until re-allocated — see [`reacquire`]); `> 0` ⇒ live and
    /// held by that many slots. Used by the APC acquire path to choose between
    /// [`add_ref`] (live) and [`reacquire`] (free).
    ///
    /// [`reacquire`]: Self::reacquire
    /// [`add_ref`]: Self::add_ref
    pub fn refs_of(&self, page: PageId) -> u32 {
        self.refs.get(page.0 as usize).copied().unwrap_or(0)
    }

    /// Allocate `n` pages, returning them in the order they were
    /// popped (most-recently-freed first). Returns `None` if the
    /// pool is short of `n` pages — the scheduler should keep the
    /// requesting slot in `Pending` until capacity opens up.
    pub fn alloc(&mut self, n: usize) -> Option<Vec<PageId>> {
        if self.free.len() < n {
            return None;
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let p = self.free.pop().unwrap();
            self.refs[p.0 as usize] = 1;
            out.push(p);
        }
        // Auto-prune: these pages are now physically re-purposed for new
        // content, so any stale APC block hash pointing at them is dropped —
        // a content-hash hit can never resolve to an overwritten page.
        self.apc.remove_pages(&out);
        Some(out)
    }

    /// The pages `alloc(n)` WOULD return next, without consuming them (the
    /// free list's LIFO top `n`, in pop order). Used by the LMCache demote
    /// path to offload each about-to-be-overwritten block BEFORE `alloc`
    /// auto-prunes its hash. Fewer than `n` if the pool is short.
    pub fn peek_reuse(&self, n: usize) -> Vec<PageId> {
        self.free.iter().rev().take(n).copied().collect()
    }

    /// Bump the reference count on each page by 1. Used by E3.2's
    /// cross-request prefix sharing: when slot B admits with a
    /// prompt whose first M pages match slot A's, slot B calls
    /// `add_ref` on those M pages so they survive slot A's
    /// eventual `free`. Each page must currently be allocated
    /// (refcount > 0) — otherwise the caller is fishing pages off
    /// the free list, which is a logic bug. Debug builds assert.
    pub fn add_ref(&mut self, pages: &[PageId]) {
        for p in pages {
            debug_assert!(
                self.refs[p.0 as usize] > 0,
                "add_ref on page {p:?} which is on the free list — caller \
                 must hold an allocation handle before sharing it"
            );
            self.refs[p.0 as usize] += 1;
        }
    }

    /// APC un-evict: revive a page that is currently on the free list
    /// (refcount 0) back to refcount 1 **without clearing its contents**.
    /// A freed page still physically holds its old block's KV until the
    /// page is re-allocated, so a content-hash (prefix-cache) hit can bring
    /// the block back instead of recomputing it — this is what makes
    /// completed requests' KV reusable. Removes the page from the free list
    /// and returns `true`; returns `false` when the page is NOT free
    /// (refcount > 0 — already live, so the caller should [`add_ref`] it
    /// instead; or the page was since physically reused and its old content
    /// is gone). The caller (the APC index owner) must drop the page's stale
    /// hash entries whenever `alloc` hands that page out for new content.
    ///
    /// [`add_ref`]: Self::add_ref
    pub fn reacquire(&mut self, page: PageId) -> bool {
        if self.refs[page.0 as usize] != 0 {
            return false; // live (or out-of-range handled by index) — not a revival
        }
        match self.free.iter().position(|p| *p == page) {
            Some(pos) => {
                self.free.swap_remove(pos);
                self.refs[page.0 as usize] = 1;
                true
            }
            // refcount 0 but not on the free list should never happen; be safe.
            None => false,
        }
    }

    /// Decrement reference counts. Pages whose count reaches 0 go
    /// back on the free list. Idempotent on the assumption that the
    /// caller is the sole owner of `pages` for this drop event (no
    /// double-frees of the same handle).
    pub fn free(&mut self, pages: &[PageId]) {
        for p in pages {
            let r = &mut self.refs[p.0 as usize];
            if *r == 0 {
                // Defensive: double-free would underflow. Skip — the
                // page is already on the free list.
                continue;
            }
            *r -= 1;
            if *r == 0 {
                self.free.push(*p);
            }
        }
    }

    // ---- APC block index (owned here so `alloc` auto-prunes) ----------------

    /// Register a finalized block's page under its content hash.
    pub fn apc_register(&mut self, hash: crate::apc::BlockHash, page: PageId) {
        self.apc.register(hash, page);
    }

    /// vLLM APC acquire: the longest leading run of `chain` blocks present in
    /// the index whose pages are still valid, taking one reference on each
    /// (REACQUIRE a freed page — reviving its retained KV — or ADD_REF a live
    /// one). A stale entry stops the run and is pruned. Returns ref-held pages.
    pub fn acquire_prefix(&mut self, chain: &[crate::apc::BlockHash]) -> Vec<PageId> {
        let mut out = Vec::new();
        for &h in chain {
            match self.apc_try_acquire(h) {
                Some(page) => out.push(page),
                None => break, // gap — reuse only a contiguous prefix
            }
        }
        out
    }

    /// Try to acquire ONE block from the hot index: reacquire a freed page /
    /// add_ref a live one. `None` on miss (or a stale entry, which is pruned).
    /// The per-block primitive the LMCache tiered acquire interleaves with the
    /// cold tier.
    pub fn apc_try_acquire(&mut self, hash: crate::apc::BlockHash) -> Option<PageId> {
        let page = self.apc.get(hash)?;
        let acquired = if self.refs_of(page) > 0 {
            self.add_ref(&[page]);
            true
        } else {
            self.reacquire(page)
        };
        if acquired {
            Some(page)
        } else {
            self.apc.remove(hash);
            None
        }
    }

    /// Drop a hash→page mapping (used by the tiered acquire when a hot miss is
    /// about to be served from the cold tier under a fresh page).
    pub fn apc_remove(&mut self, hash: crate::apc::BlockHash) {
        self.apc.remove(hash);
    }

    /// The `(hash, page)` entries pointing at any of `pages` — the LMCache
    /// demote path reads this (before a reuse `alloc` auto-prunes them) to
    /// offload each block's bytes keyed by its hash.
    pub fn apc_hashes_for_pages(&self, pages: &[PageId]) -> Vec<(crate::apc::BlockHash, PageId)> {
        self.apc.hashes_for_pages(pages)
    }

    /// `(hits, misses, entries)` of the APC index.
    pub fn apc_stats(&self) -> (u64, u64, usize) {
        (self.apc.hits(), self.apc.misses(), self.apc.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_table_alloc_and_free_round_trips() {
        let mut t = PageTable::new(4, 16);
        assert_eq!(t.free_count(), 4);
        let pages = t.alloc(3).expect("alloc 3");
        assert_eq!(pages.len(), 3);
        assert_eq!(t.free_count(), 1);
        t.free(&pages);
        assert_eq!(t.free_count(), 4);
    }

    #[test]
    fn page_table_alloc_returns_none_when_short() {
        let mut t = PageTable::new(2, 16);
        let r = t.alloc(3);
        assert!(r.is_none());
        assert_eq!(t.free_count(), 2);
    }

    /// E3.2: a shared page (refcount=2) is only returned to the
    /// free list once both holders call `free` on it. The first
    /// `free` is a refcount decrement; the second is the actual
    /// reclaim.
    #[test]
    fn add_ref_holds_page_until_all_refs_freed() {
        let mut t = PageTable::new(4, 16);
        let pages = t.alloc(2).expect("alloc 2");
        assert_eq!(t.free_count(), 2);
        // Slot B shares both pages.
        t.add_ref(&pages);
        // Slot A finishes — decrement only, page still alive.
        t.free(&pages);
        assert_eq!(t.free_count(), 2);
        // Slot B finishes — pages return to free list.
        t.free(&pages);
        assert_eq!(t.free_count(), 4);
    }

    /// APC un-evict: a freed page can be reacquired off the free list
    /// (refcount 0 → 1) and later re-freed, round-tripping the free count.
    /// Reacquiring a LIVE page fails (the caller must `add_ref` instead).
    #[test]
    fn reacquire_revives_a_freed_page() {
        let mut t = PageTable::new(3, 16);
        let pages = t.alloc(1).expect("alloc 1");
        let p = pages[0];
        t.free(&pages);
        assert_eq!(t.free_count(), 3);
        // Revive it (content-hash hit).
        assert!(t.reacquire(p), "a free page should reacquire");
        assert_eq!(t.free_count(), 2);
        // Reacquiring a now-live page fails.
        assert!(!t.reacquire(p), "a live page must not reacquire");
        // Freeing it returns it to the list.
        t.free(&[p]);
        assert_eq!(t.free_count(), 3);
    }

    /// `add_ref` followed by N+1 `free` calls is a defensive no-op
    /// (the extra `free` after refcount hits zero must NOT push the
    /// page back onto the free list).
    #[test]
    fn double_free_after_refcount_zero_is_noop() {
        let mut t = PageTable::new(2, 16);
        let pages = t.alloc(1).expect("alloc 1");
        t.free(&pages);
        assert_eq!(t.free_count(), 2);
        // Defensive second free — silently dropped, no double-add to free list.
        t.free(&pages);
        assert_eq!(t.free_count(), 2);
    }
}
