//! Per-request view onto a paged KV cache.
//!
//! [`crate::paged_kv_store::PagedKvStore`] owns the F32 K/V byte
//! storage that pages index into. [`crate::batch_scheduler::PageTable`]
//! owns the [`PageId`] lifecycle (free list, alloc/free). This
//! module's [`PagedKvCache`] is the per-request glue: holds the
//! slot's page assignment + `seq_len` watermark and exposes the
//! `write_token` / `gather_layer` / `release` surface the engine's
//! forward pass will call.
//!
//! Ownership shape:
//!
//! - The engine OWNS one [`PagedKvStore`] + one [`PageTable`] sized
//!   to the device's KV-cache budget. (In the multi-slot future
//!   they'll be shared across all in-flight requests; today the
//!   single-flight path also uses this — one request still gets the
//!   whole store.)
//! - Each in-flight request OWNS one `PagedKvCache`. The cache
//!   takes `&mut PagedKvStore` + `&mut PageTable` as call
//!   parameters rather than `Arc<Mutex<_>>`-borrowing them, because
//!   forward passes run sequentially per request — no concurrent
//!   access to worry about. When fused multi-slot decode lands the
//!   store/table will be wrapped at the call site instead.
//! - On request completion the cache calls [`PagedKvCache::release`]
//!   to return its pages to the table's free list. Drop is a no-op
//!   on the page side (the cache doesn't borrow the table) so
//!   forgetting to call `release` leaks pages — caught by the
//!   engine's per-request RAII guard, not here.
//!
//! Hot-path layout assumptions: the engine's prefill loop writes all
//! N new tokens to one layer's pages, then gathers + attends, then
//! moves to the next layer. `write_token` bumps `seq_len` to
//! `max(seq_len, pos + 1)` on each call so consecutive layers see
//! the right total — they're written in lockstep across all layers
//! at every position, so the lazy bump is safe (a future paged
//! variant that interleaves layers per-token would still work).

use crate::page_table::{PageId, PageTable};
use crate::paged_kv_store::PagedKvStore;

/// One request's KV-cache state. Cheap to construct (just a Vec
/// of PageIds + counters); the actual byte storage lives in the
/// shared [`PagedKvStore`] passed into every write/gather call.
#[derive(Debug)]
pub struct PagedKvCache {
    /// Pages currently assigned to this request, in
    /// position-ascending order. Page `i` holds positions
    /// `[i * page_size, (i + 1) * page_size)` for every layer.
    pages: Vec<PageId>,
    /// Total token positions written so far. Promoted by
    /// `write_token` (lazy `max(seq_len, pos + 1)` semantics) and
    /// readable via `seq_len()` for the attention call's `kv_len`
    /// argument.
    seq_len: u32,
    /// Mirrored from the store to keep `capacity_tokens` /
    /// `ensure_capacity` free of an indirect call. Immutable for the
    /// lifetime of a request — if `PagedKvStore::resize` ever lands,
    /// the cache will need an invalidate hook.
    page_size: u32,
}

impl PagedKvCache {
    /// Build an empty cache shaped to the given store. The cache
    /// starts with zero pages; the engine grows it via
    /// [`Self::ensure_capacity`] before each prefill / decode call.
    pub fn new_for(store: &PagedKvStore) -> Self {
        Self {
            pages: Vec::new(),
            seq_len: 0,
            page_size: store.page_size(),
        }
    }

    /// G2.4: Q8_0 paged variant. The cache itself only holds page IDs
    /// + seq_len + page_size — independent of how the store quantizes
    /// the cells, so the only difference vs `new_for` is the source
    /// `page_size()` accessor lives on a different struct.
    pub fn new_for_q8_0(store: &crate::paged_kv_store::PagedKvStoreQ8_0) -> Self {
        Self {
            pages: Vec::new(),
            seq_len: 0,
            page_size: store.page_size(),
        }
    }

    /// H9b: generic constructor accepting a raw page_size for TQ /
    /// NVFP4 paged variants (and any future store types).
    pub fn new_with_page_size(page_size: u32) -> Self {
        Self {
            pages: Vec::new(),
            seq_len: 0,
            page_size,
        }
    }

    /// Total token positions written so far (the attention call's
    /// `kv_len`).
    pub fn seq_len(&self) -> u32 {
        self.seq_len
    }

    /// Pages currently assigned to this request, in position order.
    /// Exposed for the engine's status surfaces + tests; mutating is
    /// owned by the cache (callers should go through
    /// `ensure_capacity` / `release`).
    pub fn pages(&self) -> &[PageId] {
        &self.pages
    }

    /// Total positions the cache can currently hold across its
    /// allocated pages. Drives the "do I need to grow?" check at
    /// each forward pass entry.
    pub fn capacity_tokens(&self) -> u32 {
        (self.pages.len() as u32).saturating_mul(self.page_size)
    }

    /// E3.2: install a read-only view onto another slot's already-
    /// written prefix pages. The caller must have already bumped
    /// each page's refcount via [`crate::page_table::PageTable::add_ref`]
    /// so the donor slot's eventual `free` doesn't reclaim them while
    /// this cache still references them.
    ///
    /// `seq_len` must equal `pages.len() * page_size` — partial-page
    /// sharing isn't supported (each page covers a fixed token
    /// window for every layer, so a "half-shared" page would force
    /// a copy). Callers (the engine's prefix-share admission path)
    /// align the LCP down to a page boundary before calling this.
    ///
    /// After this call, the cache behaves as though the shared
    /// positions had been prefilled in place — subsequent
    /// `write_token` at `pos >= seq_len` allocates suffix pages via
    /// the normal `ensure_capacity` path; subsequent `gather_layer`
    /// reads the shared pages alongside any new ones.
    pub fn pin_shared_prefix(
        &mut self,
        pages: Vec<crate::page_table::PageId>,
        seq_len: u32,
    ) {
        debug_assert!(
            self.pages.is_empty() && self.seq_len == 0,
            "pin_shared_prefix expects a freshly-built cache; call it before \
             ensure_capacity / any write_token"
        );
        debug_assert_eq!(
            seq_len,
            (pages.len() as u32).saturating_mul(self.page_size),
            "shared prefix must end on a page boundary — partial-page \
             sharing isn't supported"
        );
        self.pages = pages;
        self.seq_len = seq_len;
    }

    /// Ensure the cache has capacity for at least `n_total` token
    /// positions. Pulls new pages from `table` as needed. Returns
    /// `Ok(())` on success or `Err(shortfall)` if the page pool is
    /// short — the engine should defer the request in that case
    /// (CB scheduler queues it; single-flight currently 503s).
    ///
    /// Failure is atomic: on shortfall, no pages are consumed
    /// (same contract as [`PageTable::alloc`]).
    pub fn ensure_capacity(
        &mut self,
        table: &mut PageTable,
        n_total: u32,
    ) -> Result<(), usize> {
        if n_total <= self.capacity_tokens() {
            return Ok(());
        }
        let pages_needed = n_total.div_ceil(self.page_size) as usize;
        let have = self.pages.len();
        let new_pages = pages_needed - have;
        match table.alloc(new_pages) {
            Some(new) => {
                self.pages.extend(new);
                Ok(())
            }
            None => Err(new_pages.saturating_sub(table.free_count())),
        }
    }

    /// Append one token's K/V rows (one row per kv-head) at
    /// absolute position `pos` for layer `layer`. `k_row` / `v_row`
    /// must each be `n_kv_heads * head_dim` floats long, laid out
    /// head-major (i.e. `k_row[h * head_dim + d]`) — matches the
    /// existing `forward_*` code's per-token K/V scratch.
    ///
    /// Returns `Some(())` on success, `None` on shape mismatch or
    /// when `pos` exceeds the cache's current `capacity_tokens()`
    /// (caller must call `ensure_capacity` first). Bumps `seq_len`
    /// to `max(seq_len, pos + 1)`.
    pub fn write_token(
        &mut self,
        store: &mut PagedKvStore,
        layer: u32,
        pos: u32,
        k_row: &[f32],
        v_row: &[f32],
    ) -> Option<()> {
        if pos >= self.capacity_tokens() {
            return None;
        }
        store.write_token(&self.pages, layer, pos, k_row, v_row)?;
        if pos + 1 > self.seq_len {
            self.seq_len = pos + 1;
        }
        Some(())
    }

    /// Gather all positions `[0, seq_len)` of one layer's K and V
    /// into the contiguous `[n_kv_heads, seq_len, head_dim]` slabs
    /// the existing attention kernels (CPU `gqa_attention_flash_*`,
    /// SYCL `rsl_flash_attn_prefill_usm`) consume. Buffers must each
    /// be exactly `n_kv_heads * seq_len * head_dim` long.
    ///
    /// Returns `Some(())` on success, `None` on shape mismatch or
    /// when the page list under-covers `seq_len` (a logic bug —
    /// would only happen if `ensure_capacity` was bypassed).
    pub fn gather_layer(
        &self,
        store: &PagedKvStore,
        layer: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Option<()> {
        store.gather_layer(&self.pages, layer, self.seq_len, k_out, v_out)
    }

    /// H9a: Q8_0 paged gather. Same shape as [`Self::gather_layer`]
    /// but with the Q8_0 paged storage backend — quantized rows are
    /// dequantized back to f32 into the caller's slabs by
    /// `PagedKvStoreQ8_0::gather_layer`.
    pub fn gather_layer_q8_0(
        &self,
        store: &crate::paged_kv_store::PagedKvStoreQ8_0,
        layer: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Option<()> {
        store.gather_layer(&self.pages, layer, self.seq_len, k_out, v_out)
    }

    /// H9a: Q8_0 paged write. Same shape as [`Self::write_token`]
    /// — quantizes the f32 K/V rows to Q8_0 + scale on write.
    pub fn write_token_q8_0(
        &mut self,
        store: &mut crate::paged_kv_store::PagedKvStoreQ8_0,
        layer: u32,
        pos: u32,
        k_row: &[f32],
        v_row: &[f32],
    ) -> Option<()> {
        if pos >= self.capacity_tokens() {
            return None;
        }
        store.write_token(&self.pages, layer, pos, k_row, v_row)?;
        if pos + 1 > self.seq_len {
            self.seq_len = pos + 1;
        }
        Some(())
    }

    /// H9a: dyn-trait variants — let `forward_one_paged_*` route both
    /// F32 and Q8_0 (and future TQ/NVFP4) stores through the same code
    /// path. The trait object's `write_token` / `gather_layer` call
    /// the right backend internally; this cache wrapper just stages
    /// the page list + seq_len bookkeeping.
    pub fn write_token_dyn(
        &mut self,
        store: &mut dyn crate::paged_kv_store::PagedKvStoreOps,
        layer: u32,
        pos: u32,
        k_row: &[f32],
        v_row: &[f32],
    ) -> Option<()> {
        if pos >= self.capacity_tokens() {
            return None;
        }
        store.write_token(&self.pages, layer, pos, k_row, v_row)?;
        if pos + 1 > self.seq_len {
            self.seq_len = pos + 1;
        }
        Some(())
    }

    pub fn gather_layer_dyn(
        &self,
        store: &dyn crate::paged_kv_store::PagedKvStoreOps,
        layer: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Option<()> {
        store.gather_layer(&self.pages, layer, self.seq_len, k_out, v_out)
    }

    /// Override the seq_len watermark to a smaller value — used by
    /// speculative decoding's reject-and-rewind path. The pages
    /// stay allocated (they'll be overwritten on the next
    /// `write_token` past the watermark, or released at request
    /// end). Refuses to grow `seq_len` here on purpose — `write_token`
    /// is the only path that should advance it.
    pub fn set_seq_len_for_rollback(&mut self, new_seq_len: u32) {
        if new_seq_len < self.seq_len {
            self.seq_len = new_seq_len;
        }
    }

    /// Return every page held by this cache to `table`'s free list.
    /// Idempotent (second call is a no-op). Called by the engine at
    /// request completion (success, cancel, error) so the next
    /// admission can reuse the pages.
    pub fn release(&mut self, table: &mut PageTable) {
        if self.pages.is_empty() {
            return;
        }
        table.free(&self.pages);
        self.pages.clear();
        self.seq_len = 0;
    }

    // ============================================================
    // Shared-store variants (multi-slot continuous batching)
    // ============================================================
    //
    // These mirror the single-owner methods above but take a
    // `&SharedPagedKv` (Arc<Mutex<store>> + Arc<Mutex<table>>) so
    // M concurrent per-slot caches can coexist around one shared
    // paged pool. The locks are short — each method acquires,
    // does the underlying single-owner call, and releases.
    //
    // Lock order policy (matches `SharedPagedKv`'s module doc):
    // table BEFORE store when both are needed.

    /// Shared-store variant of [`Self::ensure_capacity`]. Allocates
    /// pages from the shared table only.
    pub fn ensure_capacity_shared(
        &mut self,
        shared: &crate::shared_paged_kv::SharedPagedKv,
        n_total: u32,
    ) -> Result<(), usize> {
        shared.with_table_mut(|table| self.ensure_capacity(table, n_total))
    }

    /// Shared-store variant of [`Self::write_token`]. Writes a
    /// single token's K/V rows under the store lock.
    pub fn write_token_shared(
        &mut self,
        shared: &crate::shared_paged_kv::SharedPagedKv,
        layer: u32,
        pos: u32,
        k_row: &[f32],
        v_row: &[f32],
    ) -> Option<()> {
        shared.with_store_mut(|store| self.write_token(store, layer, pos, k_row, v_row))
    }

    /// Shared-store variant of [`Self::gather_layer`]. Reads the
    /// layer's K/V slab into caller-provided scratch under the
    /// store lock.
    pub fn gather_layer_shared(
        &self,
        shared: &crate::shared_paged_kv::SharedPagedKv,
        layer: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Option<()> {
        shared.with_store(|store| self.gather_layer(store, layer, k_out, v_out))
    }

    /// Shared-store variant of [`Self::release`]. Returns pages to
    /// the shared table; other slots can immediately reclaim them.
    pub fn release_shared(&mut self, shared: &crate::shared_paged_kv::SharedPagedKv) {
        shared.with_table_mut(|table| self.release(table));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_store() -> PagedKvStore {
        // Match paged_kv_store's small_store(): 4 pages, 2 layers,
        // 3 heads, page_size 4, head_dim 8.
        PagedKvStore::new(4, 2, 3, 4, 8).unwrap()
    }

    fn token_row(seed: f32, n_heads: u32, head_dim: u32) -> Vec<f32> {
        (0..n_heads * head_dim)
            .map(|i| seed + i as f32 * 0.01)
            .collect()
    }

    #[test]
    fn new_for_picks_up_store_geometry() {
        let store = small_store();
        let cache = PagedKvCache::new_for(&store);
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.pages().len(), 0);
        assert_eq!(cache.capacity_tokens(), 0);
    }

    #[test]
    fn ensure_capacity_allocates_only_what_is_missing() {
        let store = small_store();
        let mut table = PageTable::new(4, 4);
        let mut cache = PagedKvCache::new_for(&store);
        // Request 5 tokens → ceil(5/4) = 2 pages.
        cache.ensure_capacity(&mut table, 5).expect("alloc 2");
        assert_eq!(cache.pages().len(), 2);
        assert_eq!(cache.capacity_tokens(), 8);
        assert_eq!(table.free_count(), 2);
        // Request 6 → still fits in current 2 pages, no new alloc.
        cache.ensure_capacity(&mut table, 6).expect("noop");
        assert_eq!(cache.pages().len(), 2);
        assert_eq!(table.free_count(), 2);
        // Request 9 → needs 3 pages, one more alloc.
        cache.ensure_capacity(&mut table, 9).expect("alloc 1 more");
        assert_eq!(cache.pages().len(), 3);
        assert_eq!(table.free_count(), 1);
    }

    #[test]
    fn ensure_capacity_returns_err_on_oom_without_partial_consumption() {
        let store = small_store();
        let mut table = PageTable::new(2, 4);
        let mut cache = PagedKvCache::new_for(&store);
        // 2 pages = 8 tokens. Ask for 100 (= 25 pages) — short by 23.
        let err = cache.ensure_capacity(&mut table, 100).unwrap_err();
        assert!(err > 0, "shortfall must be reported");
        // Atomic failure: pool untouched, cache pages still empty.
        assert_eq!(table.free_count(), 2);
        assert_eq!(cache.pages().len(), 0);
    }

    #[test]
    fn write_token_rejects_pos_beyond_capacity() {
        let store = small_store();
        let mut store = store;
        let mut table = PageTable::new(4, 4);
        let mut cache = PagedKvCache::new_for(&store);
        cache.ensure_capacity(&mut table, 4).unwrap(); // 1 page = 4 tokens
        let n = (store.n_kv_heads() * store.head_dim()) as usize;
        let row = vec![0.0; n];
        // pos 0..4 are valid.
        cache.write_token(&mut store, 0, 0, &row, &row).expect("pos 0");
        cache.write_token(&mut store, 0, 3, &row, &row).expect("pos 3");
        // pos 4 needs a second page — fail until ensure_capacity grows.
        assert!(cache.write_token(&mut store, 0, 4, &row, &row).is_none());
    }

    #[test]
    fn write_token_advances_seq_len_to_max_pos_plus_one() {
        let mut store = small_store();
        let mut table = PageTable::new(4, 4);
        let mut cache = PagedKvCache::new_for(&store);
        cache.ensure_capacity(&mut table, 8).unwrap();
        let n_heads = store.n_kv_heads();
        let hd = store.head_dim();
        // Write at pos 5 first (out-of-order) → seq_len = 6.
        let row = token_row(0.5, n_heads, hd);
        cache.write_token(&mut store, 0, 5, &row, &row).expect("pos 5");
        assert_eq!(cache.seq_len(), 6);
        // Write at pos 2 → seq_len stays at 6 (lazy max).
        cache.write_token(&mut store, 0, 2, &row, &row).expect("pos 2");
        assert_eq!(cache.seq_len(), 6);
        // Write at pos 7 → seq_len = 8.
        cache.write_token(&mut store, 0, 7, &row, &row).expect("pos 7");
        assert_eq!(cache.seq_len(), 8);
    }

    #[test]
    fn write_then_gather_round_trips_through_pages() {
        // The store's own tests cover gather correctness in detail.
        // Here we just confirm the cache's write_token + gather_layer
        // chain reproduces the input pattern for one full prefill of
        // N tokens spanning multiple pages.
        let mut store = small_store();
        let mut table = PageTable::new(4, 4);
        let mut cache = PagedKvCache::new_for(&store);
        let n_heads = store.n_kv_heads() as usize;
        let hd = store.head_dim() as usize;
        let n_tokens = 7u32; // 2 pages (8 capacity), 1 partial
        cache.ensure_capacity(&mut table, n_tokens).unwrap();
        let layer = 1u32;
        // Write a unique-per-cell pattern.
        for pos in 0..n_tokens {
            let mut k = vec![0f32; n_heads * hd];
            let mut v = vec![0f32; n_heads * hd];
            for h in 0..n_heads {
                for d in 0..hd {
                    let val = (pos as f32) * 1000.0 + (h as f32) * 10.0 + d as f32;
                    k[h * hd + d] = val;
                    v[h * hd + d] = -val;
                }
            }
            cache.write_token(&mut store, layer, pos, &k, &v).expect("write");
        }
        assert_eq!(cache.seq_len(), n_tokens);
        let need = n_heads * (n_tokens as usize) * hd;
        let mut k_out = vec![0f32; need];
        let mut v_out = vec![0f32; need];
        cache.gather_layer(&store, layer, &mut k_out, &mut v_out).expect("gather");
        // Cells follow the same pos*1000 + h*10 + d pattern.
        for h in 0..n_heads {
            for pos in 0..n_tokens as usize {
                for d in 0..hd {
                    let off = h * (n_tokens as usize) * hd + pos * hd + d;
                    let want = (pos as f32) * 1000.0 + (h as f32) * 10.0 + d as f32;
                    assert!((k_out[off] - want).abs() < 1e-6,
                            "k mismatch h={h} pos={pos} d={d}");
                    assert!((v_out[off] + want).abs() < 1e-6,
                            "v mismatch h={h} pos={pos} d={d}");
                }
            }
        }
    }

    #[test]
    fn release_returns_pages_and_resets_seq_len() {
        let store = small_store();
        let mut store = store;
        let mut table = PageTable::new(4, 4);
        let mut cache = PagedKvCache::new_for(&store);
        cache.ensure_capacity(&mut table, 8).unwrap();
        let n = (store.n_kv_heads() * store.head_dim()) as usize;
        let row = vec![0.0; n];
        cache.write_token(&mut store, 0, 3, &row, &row).unwrap();
        assert_eq!(cache.seq_len(), 4);
        assert_eq!(table.free_count(), 2);
        cache.release(&mut table);
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.pages().len(), 0);
        assert_eq!(table.free_count(), 4);
        // Second release is a no-op.
        cache.release(&mut table);
        assert_eq!(table.free_count(), 4);
    }

    /// Two `PagedKvCache` instances pointing at one shared store +
    /// table can write distinct token streams to non-overlapping
    /// pages without corrupting each other. The shared methods
    /// must be the moral equivalent of the single-owner methods —
    /// gather on cache A only reads pages owned by cache A, never
    /// pages allocated to cache B.
    #[test]
    fn shared_caches_write_to_disjoint_pages_without_corruption() {
        use crate::shared_paged_kv::SharedPagedKv;
        let store = small_store();
        let table = PageTable::new(4, 4);
        let shared = SharedPagedKv::new(store, table);

        let mut cache_a = PagedKvCache::new_for(&shared.with_store(|s| {
            // Build a throwaway cache geometry borrow — we just
            // need the store's geometry, not a write borrow.
            // PagedKvCache::new_for only reads page_size, so this
            // is cheap.
            PagedKvStore::new(
                s.total_pages(),
                s.n_layers(),
                s.n_kv_heads(),
                s.page_size(),
                s.head_dim(),
            )
            .unwrap()
        }));
        let mut cache_b = PagedKvCache::new_for(&shared.with_store(|s| {
            PagedKvStore::new(
                s.total_pages(),
                s.n_layers(),
                s.n_kv_heads(),
                s.page_size(),
                s.head_dim(),
            )
            .unwrap()
        }));

        // Each cache asks for 4 tokens (1 page each). Shared table
        // hands them disjoint pages.
        cache_a.ensure_capacity_shared(&shared, 4).expect("alloc A");
        cache_b.ensure_capacity_shared(&shared, 4).expect("alloc B");
        assert_eq!(shared.free_pages(), 2, "2 of 4 pages allocated");
        assert_ne!(cache_a.pages(), cache_b.pages(), "disjoint page assignments");

        // Each cache writes a unique pattern at pos 0, layer 0.
        let geom = shared.geometry();
        let n_heads = geom.2 as usize;
        let hd = geom.4 as usize;
        let row_a: Vec<f32> = (0..n_heads * hd).map(|i| i as f32 * 0.1).collect();
        let row_b: Vec<f32> = (0..n_heads * hd).map(|i| i as f32 * -0.1).collect();
        cache_a
            .write_token_shared(&shared, 0, 0, &row_a, &row_a)
            .expect("write A");
        cache_b
            .write_token_shared(&shared, 0, 0, &row_b, &row_b)
            .expect("write B");
        assert_eq!(cache_a.seq_len(), 1);
        assert_eq!(cache_b.seq_len(), 1);

        // Gather: A reads its own row back, B reads its own —
        // neither sees the other's data.
        let need = n_heads * 1 * hd;
        let mut a_k = vec![0f32; need];
        let mut a_v = vec![0f32; need];
        let mut b_k = vec![0f32; need];
        let mut b_v = vec![0f32; need];
        cache_a
            .gather_layer_shared(&shared, 0, &mut a_k, &mut a_v)
            .expect("gather A");
        cache_b
            .gather_layer_shared(&shared, 0, &mut b_k, &mut b_v)
            .expect("gather B");
        // Cache A's gather reproduces `row_a` (interleaved by head
        // — gather emits `[h, pos=0, d]` flat, write_token wrote
        // `[h, d]` flat at pos 0, so the bytes are reachable).
        for h in 0..n_heads {
            for d in 0..hd {
                let off = h * hd + d;
                assert_eq!(a_k[off], row_a[h * hd + d], "A k[{h},{d}]");
                assert_eq!(a_v[off], row_a[h * hd + d], "A v[{h},{d}]");
                assert_eq!(b_k[off], row_b[h * hd + d], "B k[{h},{d}]");
                assert_eq!(b_v[off], row_b[h * hd + d], "B v[{h},{d}]");
            }
        }
    }

    /// Release on one shared cache returns its pages to the shared
    /// table so a different cache can reclaim them on the next
    /// `ensure_capacity_shared`. Regression: a missing
    /// `release_shared` implementation would leak pages and
    /// eventually the second cache would fail with "pool short".
    #[test]
    fn release_shared_makes_pages_reclaimable_by_other_caches() {
        use crate::shared_paged_kv::SharedPagedKv;
        let store = small_store();
        let table = PageTable::new(4, 4);
        let shared = SharedPagedKv::new(store, table);

        let mut cache_a = PagedKvCache::new_for(&PagedKvStore::new(4, 2, 3, 4, 8).unwrap());
        cache_a.ensure_capacity_shared(&shared, 16).expect("alloc 4 pages");
        assert_eq!(shared.free_pages(), 0, "pool exhausted");

        cache_a.release_shared(&shared);
        assert_eq!(shared.free_pages(), 4, "release returned all 4 pages");
        assert_eq!(cache_a.pages().len(), 0);
        assert_eq!(cache_a.seq_len(), 0);

        // A different cache can now grab those pages.
        let mut cache_b = PagedKvCache::new_for(&PagedKvStore::new(4, 2, 3, 4, 8).unwrap());
        cache_b
            .ensure_capacity_shared(&shared, 16)
            .expect("alloc reclaimed pages");
        assert_eq!(shared.free_pages(), 0);
    }

    /// Parity probe: same K/V data written through both a contiguous
    /// `[n_kv_heads, max_ctx, head_dim]` slab AND a paged cache
    /// produces bit-identical buffers after the paged cache's
    /// `gather_layer` reconstructs its slab. This is the v1
    /// "paged is a drop-in for contiguous" guarantee — any divergence
    /// here would surface in attention output before we even call the
    /// kernel.
    #[test]
    fn paged_gather_matches_contiguous_write_for_same_inputs() {
        let mut store = small_store();
        let mut table = PageTable::new(4, 4);
        let mut cache = PagedKvCache::new_for(&store);
        let n_heads = store.n_kv_heads() as usize;
        let hd = store.head_dim() as usize;
        let max_ctx = 8u32; // 2 pages
        let n_tokens = 6u32;
        let layer = 0u32;
        cache.ensure_capacity(&mut table, n_tokens).unwrap();
        // Contiguous reference slab — same layout the existing F32
        // KV path uses: `[h, pos, d]` flat at `(h * max_ctx + pos) * head_dim`.
        let mut contig_k = vec![0f32; n_heads * (max_ctx as usize) * hd];
        let mut contig_v = vec![0f32; n_heads * (max_ctx as usize) * hd];
        for pos in 0..n_tokens {
            let mut k_row = vec![0f32; n_heads * hd];
            let mut v_row = vec![0f32; n_heads * hd];
            for h in 0..n_heads {
                for d in 0..hd {
                    let val = ((pos * 7 + h as u32 * 3 + d as u32) % 19) as f32 * 0.13;
                    k_row[h * hd + d] = val;
                    v_row[h * hd + d] = val + 0.5;
                }
            }
            // Write to paged cache.
            cache.write_token(&mut store, layer, pos, &k_row, &v_row).unwrap();
            // Mirror write into contiguous slab — same indexing the
            // current F32 forward pass uses.
            for h in 0..n_heads {
                let dst = (h * max_ctx as usize + pos as usize) * hd;
                let src = h * hd;
                contig_k[dst..dst + hd].copy_from_slice(&k_row[src..src + hd]);
                contig_v[dst..dst + hd].copy_from_slice(&v_row[src..src + hd]);
            }
        }
        // Gather paged → contiguous-shaped slab sized to seq_len
        // (NOT max_ctx — that's the attention kernel call's
        // `max_ctx` argument under the gather path, see the design
        // note in this module's doc comment).
        let need = n_heads * (n_tokens as usize) * hd;
        let mut gathered_k = vec![0f32; need];
        let mut gathered_v = vec![0f32; need];
        cache.gather_layer(&store, layer, &mut gathered_k, &mut gathered_v).unwrap();
        // Compare: for each (h, pos) the gathered slab's row must
        // equal the contiguous slab's same (h, pos) row.
        for h in 0..n_heads {
            for pos in 0..n_tokens as usize {
                let g_off = h * (n_tokens as usize) * hd + pos * hd;
                let c_off = (h * max_ctx as usize + pos) * hd;
                assert_eq!(
                    &gathered_k[g_off..g_off + hd],
                    &contig_k[c_off..c_off + hd],
                    "k row mismatch at h={h} pos={pos}"
                );
                assert_eq!(
                    &gathered_v[g_off..g_off + hd],
                    &contig_v[c_off..c_off + hd],
                    "v row mismatch at h={h} pos={pos}"
                );
            }
        }
    }
}
