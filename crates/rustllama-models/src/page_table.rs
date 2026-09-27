//! `PageId` + `PageTable` — opaque page-id type and the free-list
//! allocator that hands ids out.
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
}

impl PageTable {
    pub fn new(total: u32, page_size: u32) -> Self {
        Self {
            free: (0..total).rev().map(PageId).collect(),
            refs: vec![0; total as usize],
            total,
            page_size,
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
        Some(out)
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
