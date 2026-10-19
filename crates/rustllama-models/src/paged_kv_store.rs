//! Paged KV-cache byte storage + gather helper.
//!
//! [`crate::batch_scheduler::PageTable`] owns the `PageId` lifecycles
//! (free list, alloc/free). [`PagedKvStore`] owns the actual K/V
//! float bytes those page IDs index into. Splitting the two keeps
//! the scheduler-level bookkeeping testable without an MB-scale
//! allocation, and lets a later turn swap in a different cell type
//! (quantized KV) without churn at the scheduler layer.
//!
//! Layout (chosen so writes are contiguous and gather reads only
//! one head-sized strip per (head, page) pair):
//!
//! ```text
//! data[(page * page_stride)
//!      + (layer * 2 * n_kv_heads * page_size * head_dim)
//!      + (kv_axis * n_kv_heads * page_size * head_dim)   // 0 = K, 1 = V
//!      + (h * page_size * head_dim)
//!      + (pos_in_page * head_dim)
//!      + d]
//! ```
//!
//! - Writing one new token's K row across all heads: one strided
//!   `head_dim`-wide write per head; cheap on shared LPDDR.
//! - Gather to the existing kernel's `[n_kv_heads, kv_len, head_dim]`
//!   layout: one `n_kv_heads * pages * page_size * head_dim` outer
//!   loop with each per-(head, page) iteration copying a contiguous
//!   `valid_pos_in_page * head_dim` strip. Last page may be partial.
//!
//! This is v1's "feed paged storage into the existing attention
//! kernel" path. A future indirect-addressing kernel will read this
//! layout directly without the gather copy.
//!
//! Not in this module:
//!   - Per-slot page assignment lives in each request's
//!     [`crate::paged_kv_cache::PagedKvCache`] (its `pages` list).
//!   - Quantized KV (Q8_0, TurboQuant, NVFP4) variants — once the
//!     F32 path is end-to-end we'll add `PagedKvStoreQ8_0` etc.
//!     mirroring the existing `KvLayer` variants.

use crate::page_table::PageId;

/// Single-cell F32 paged KV storage. Holds the flat backing buffer
/// for `total_pages` pages, each covering `page_size` token positions
/// across `n_layers` layers, `n_kv_heads` heads, `head_dim` channels,
/// for both K and V (the `* 2` in the page stride).
///
/// Memory footprint: `total_pages * n_layers * 2 * n_kv_heads *
/// page_size * head_dim * 4` bytes. For a Qwen-2.5-7B-class config
/// (`n_layers = 28`, `n_kv_heads = 4`, `head_dim = 128`, `page_size
/// = 16`, `total_pages = 1024`): `1024 * 28 * 2 * 4 * 16 * 128 * 4 =
/// ~470 MiB`. Sized once at engine load from the device's free VRAM
/// budget; the scheduler hands out / reclaims pages from there.
#[derive(Debug)]
pub struct PagedKvStore {
    data: Vec<f32>,
    total_pages: u32,
    n_layers: u32,
    n_kv_heads: u32,
    page_size: u32,
    head_dim: u32,
    // Precomputed strides — recomputing per-index hurts the gather
    // hot path more than the 24 bytes saves.
    page_stride: usize,
    layer_stride: usize,
    kv_stride: usize,
    head_stride: usize,
}

impl PagedKvStore {
    /// Bytes of pool storage (memory-budget planner input).
    pub fn approx_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// Build a fresh paged store zero-initialized to
    /// `total_pages * n_layers * 2 * n_kv_heads * page_size * head_dim`
    /// f32 elements. Allocates one contiguous `Vec<f32>` — `total_pages`
    /// is expected to be sized to fit comfortably in the device's
    /// VRAM/RAM budget. Returns `None` if any dimension is zero (a
    /// degenerate config that would silently misbehave further down).
    pub fn new(
        total_pages: u32,
        n_layers: u32,
        n_kv_heads: u32,
        page_size: u32,
        head_dim: u32,
    ) -> Option<Self> {
        if total_pages == 0 || n_layers == 0 || n_kv_heads == 0 || page_size == 0 || head_dim == 0 {
            return None;
        }
        let head_stride = (page_size as usize) * (head_dim as usize);
        let kv_stride = (n_kv_heads as usize) * head_stride;
        let layer_stride = 2usize * kv_stride;
        let page_stride = (n_layers as usize) * layer_stride;
        let total_elems = (total_pages as usize).checked_mul(page_stride)?;
        Some(Self {
            data: vec![0.0f32; total_elems],
            total_pages,
            n_layers,
            n_kv_heads,
            page_size,
            head_dim,
            page_stride,
            layer_stride,
            kv_stride,
            head_stride,
        })
    }

    pub fn total_pages(&self) -> u32 {
        self.total_pages
    }
    pub fn n_layers(&self) -> u32 {
        self.n_layers
    }
    pub fn n_kv_heads(&self) -> u32 {
        self.n_kv_heads
    }
    pub fn page_size(&self) -> u32 {
        self.page_size
    }
    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// Total f32 elements per page across all layers and K/V. Useful
    /// for callers that want to bulk-zero a freshly-allocated page
    /// without touching the strided indexing.
    pub fn page_stride(&self) -> usize {
        self.page_stride
    }

    /// Borrow one page's full contiguous KV slab (every layer, both K and V,
    /// all heads × positions). A page is the outermost axis, so this is one
    /// contiguous slice — the natural unit for LMCache cold-tier offload.
    /// `None` if `page` is out of range.
    pub fn read_page(&self, page: crate::page_table::PageId) -> Option<&[f32]> {
        let off = (page.0 as usize).checked_mul(self.page_stride)?;
        self.data.get(off..off + self.page_stride)
    }

    /// Overwrite one page's full contiguous KV slab from `src` (exactly
    /// `page_stride` elements) — used to PROMOTE a block back from the cold tier
    /// into a freshly-allocated page. Returns false on a size/range mismatch.
    pub fn write_page(&mut self, page: crate::page_table::PageId, src: &[f32]) -> bool {
        if src.len() != self.page_stride {
            return false;
        }
        let Some(off) = (page.0 as usize).checked_mul(self.page_stride) else {
            return false;
        };
        match self.data.get_mut(off..off + self.page_stride) {
            Some(dst) => {
                dst.copy_from_slice(src);
                // Phase A (GPU paged attention): sync the promoted page into the
                // device KV mirror (no-op off-CUDA / when no mirror is present).
                crate::accel::paged_mirror_write_page(page.0, src);
                true
            }
            None => false,
        }
    }

    /// Internal: offset within `data` for one (page, layer, kv_axis,
    /// head, pos) cell. Returns `None` on any out-of-range axis so
    /// upstream bugs fail loudly rather than silently address into
    /// the wrong head.
    #[inline]
    fn cell_offset(
        &self,
        page: PageId,
        layer: u32,
        kv_axis: u32,
        h: u32,
        pos_in_page: u32,
    ) -> Option<usize> {
        if page.0 >= self.total_pages
            || layer >= self.n_layers
            || kv_axis >= 2
            || h >= self.n_kv_heads
            || pos_in_page >= self.page_size
        {
            return None;
        }
        let off = (page.0 as usize) * self.page_stride
            + (layer as usize) * self.layer_stride
            + (kv_axis as usize) * self.kv_stride
            + (h as usize) * self.head_stride
            + (pos_in_page as usize) * (self.head_dim as usize);
        Some(off)
    }

    /// Mutable slice for the K row at `(page, layer, h, pos_in_page)`.
    /// Length always equals `head_dim` on success. Returns `None` if
    /// any axis is out of range.
    pub fn k_row_mut(
        &mut self,
        page: PageId,
        layer: u32,
        h: u32,
        pos_in_page: u32,
    ) -> Option<&mut [f32]> {
        let off = self.cell_offset(page, layer, 0, h, pos_in_page)?;
        let hd = self.head_dim as usize;
        Some(&mut self.data[off..off + hd])
    }

    pub fn v_row_mut(
        &mut self,
        page: PageId,
        layer: u32,
        h: u32,
        pos_in_page: u32,
    ) -> Option<&mut [f32]> {
        let off = self.cell_offset(page, layer, 1, h, pos_in_page)?;
        let hd = self.head_dim as usize;
        Some(&mut self.data[off..off + hd])
    }

    pub fn k_row(&self, page: PageId, layer: u32, h: u32, pos_in_page: u32) -> Option<&[f32]> {
        let off = self.cell_offset(page, layer, 0, h, pos_in_page)?;
        let hd = self.head_dim as usize;
        Some(&self.data[off..off + hd])
    }

    pub fn v_row(&self, page: PageId, layer: u32, h: u32, pos_in_page: u32) -> Option<&[f32]> {
        let off = self.cell_offset(page, layer, 1, h, pos_in_page)?;
        let hd = self.head_dim as usize;
        Some(&self.data[off..off + hd])
    }

    /// Write one new (K, V) token row at `(slot's pages list, layer,
    /// absolute pos within slot)`. The absolute pos is decomposed
    /// into `(page_index = pos / page_size, pos_in_page = pos %
    /// page_size)` so the caller doesn't have to thread page indexing
    /// through every layer-loop write site. `k`/`v` slices must each
    /// be `n_kv_heads * head_dim` long (one row per head).
    ///
    /// Returns `Some(())` on success, `None` if `pos` exceeds the
    /// slot's allocated capacity or any input length is wrong. The
    /// caller is expected to have already called
    /// [`crate::paged_kv_cache::PagedKvCache::ensure_capacity`] to ensure the
    /// page exists before invoking.
    pub fn write_token(
        &mut self,
        pages: &[PageId],
        layer: u32,
        pos: u32,
        k: &[f32],
        v: &[f32],
    ) -> Option<()> {
        let need = (self.n_kv_heads as usize) * (self.head_dim as usize);
        if k.len() != need || v.len() != need {
            return None;
        }
        let page_idx = (pos / self.page_size) as usize;
        let pos_in_page = pos % self.page_size;
        let page = *pages.get(page_idx)?;
        let hd = self.head_dim as usize;
        for h in 0..self.n_kv_heads {
            let h_off_src = (h as usize) * hd;
            let dst_k = self.k_row_mut(page, layer, h, pos_in_page)?;
            dst_k.copy_from_slice(&k[h_off_src..h_off_src + hd]);
            let dst_v = self.v_row_mut(page, layer, h, pos_in_page)?;
            dst_v.copy_from_slice(&v[h_off_src..h_off_src + hd]);
        }
        // Phase A (GPU paged attention): sync the just-written (K, V) row into
        // the device KV mirror. No-op off-CUDA / when no mirror is present;
        // byte-identical geometry means a plain offset H2D copy.
        crate::accel::paged_mirror_write_token(layer, page.0, pos_in_page, k, v);
        Some(())
    }

    /// Gather all positions `[0, kv_len)` of a slot's K and V at
    /// `layer` into contiguous `[n_kv_heads, kv_len, head_dim]` slabs
    /// matching the layout the existing attention kernels (and the
    /// SYCL `rsl_flash_attn_prefill_usm`) expect. The caller provides
    /// the destination buffers; both must be exactly
    /// `n_kv_heads * kv_len * head_dim` long.
    ///
    /// Per-head copy pattern: walk pages in order, copy each page's
    /// contribution (`valid_pos_in_page * head_dim` floats for the
    /// last page, full `page_size * head_dim` otherwise) as one
    /// memcpy. Total copies: `n_kv_heads * ceil(kv_len / page_size)`.
    ///
    /// Returns `Some(())` on success, `None` on shape mismatch or
    /// when the page list doesn't cover `kv_len` positions.
    pub fn gather_layer(
        &self,
        pages: &[PageId],
        layer: u32,
        kv_len: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Option<()> {
        if layer >= self.n_layers {
            return None;
        }
        let need = (self.n_kv_heads as usize) * (kv_len as usize) * (self.head_dim as usize);
        if k_out.len() != need || v_out.len() != need {
            return None;
        }
        // Verify the page list covers kv_len positions.
        let pages_needed = (kv_len as usize).div_ceil(self.page_size as usize);
        if pages.len() < pages_needed {
            return None;
        }
        let hd = self.head_dim as usize;
        let page_size = self.page_size as usize;
        let kv_len_us = kv_len as usize;
        for h in 0..self.n_kv_heads {
            // Output stride per-head in the gather slab: kv_len * head_dim.
            let out_head_base = (h as usize) * kv_len_us * hd;
            let mut pos_written = 0usize;
            for (page_i, &page) in pages.iter().enumerate() {
                if pos_written >= kv_len_us {
                    break;
                }
                // How many positions of this page contribute? Full
                // page unless we're on the last page and it's partial.
                let remaining = kv_len_us - pos_written;
                let pos_in_this_page = remaining.min(page_size);
                let copy_len = pos_in_this_page * hd;
                // Source span: one contiguous `pos_in_this_page *
                // head_dim` strip starting at (page, layer, h, 0).
                let src_off = self
                    .cell_offset(page, layer, 0, h, 0)
                    .expect("page id, layer, head in range");
                let v_src_off = self
                    .cell_offset(page, layer, 1, h, 0)
                    .expect("page id, layer, head in range");
                let dst_off = out_head_base + page_i * page_size * hd;
                k_out[dst_off..dst_off + copy_len]
                    .copy_from_slice(&self.data[src_off..src_off + copy_len]);
                v_out[dst_off..dst_off + copy_len]
                    .copy_from_slice(&self.data[v_src_off..v_src_off + copy_len]);
                pos_written += pos_in_this_page;
            }
        }
        Some(())
    }

    /// P3 cascade: like [`Self::gather_layer`] but gathers only the ABSOLUTE
    /// position range `[start, end)` into a TIGHT `[n_kv_heads, end-start,
    /// head_dim]` slab. The shared-prefix (cascade) decode path uses it to
    /// gather a shared prompt prefix ONCE (`[0, prefix_len)`) and each
    /// sequence's private suffix (`[prefix_len, kv_len)`) separately, so the
    /// prefix K/V lives in one buffer instead of `n_seq` copies.
    ///
    /// Assumes NO eviction (the paged continuous-batching pool never evicts —
    /// `ensure_capacity_shared` routes through the plain allocator), so absolute
    /// position `p` lives in `pages[p / page_size]` at in-page offset `p %
    /// page_size`. Returns `None` on a geometry / length mismatch.
    pub fn gather_layer_range(
        &self,
        pages: &[PageId],
        layer: u32,
        start: u32,
        end: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Option<()> {
        if layer >= self.n_layers || end < start {
            return None;
        }
        let len = (end - start) as usize;
        let hd = self.head_dim as usize;
        let need = (self.n_kv_heads as usize) * len * hd;
        if k_out.len() != need || v_out.len() != need {
            return None;
        }
        if len == 0 {
            return Some(());
        }
        let ps = self.page_size as usize;
        let pages_needed = (end as usize).div_ceil(ps);
        if pages.len() < pages_needed {
            return None;
        }
        for h in 0..self.n_kv_heads {
            let out_head_base = (h as usize) * len * hd;
            let mut pos = start as usize;
            let mut out_pos = 0usize;
            while pos < end as usize {
                let page = pages[pos / ps];
                let in_page = pos % ps;
                // Positions [in_page, in_page + strip) are contiguous for this
                // head within the page; copy them in one strip.
                let strip = (ps - in_page).min(end as usize - pos);
                let k_src = self.cell_offset(page, layer, 0, h, in_page as u32)?;
                let v_src = self.cell_offset(page, layer, 1, h, in_page as u32)?;
                let copy = strip * hd;
                let dst = out_head_base + out_pos * hd;
                k_out[dst..dst + copy].copy_from_slice(&self.data[k_src..k_src + copy]);
                v_out[dst..dst + copy].copy_from_slice(&self.data[v_src..v_src + copy]);
                pos += strip;
                out_pos += strip;
            }
        }
        Some(())
    }

    /// P12b paged beam: copy KV rows for ABSOLUTE positions `[start, end)`
    /// across every layer and head from `src_pages` to `dst_pages` (both laid
    /// out with this store's geometry). The beam fork uses it to clone a parent
    /// beam's divergent suffix into a child's freshly-allocated pages while the
    /// shared prompt prefix stays SHARED (never copied). Assumes no eviction
    /// (absolute pos → `pages[pos / page_size]`). Reads each row into a short
    /// temp first so a `src == dst` page aliasing move stays safe.
    pub fn copy_positions_between(
        &mut self,
        src_pages: &[PageId],
        dst_pages: &[PageId],
        start: usize,
        end: usize,
    ) {
        let ps = self.page_size as usize;
        let hd = self.head_dim as usize;
        let mut tmp = vec![0f32; hd];
        for layer in 0..self.n_layers {
            for pos in start..end {
                let (Some(&sp), Some(&dp)) = (src_pages.get(pos / ps), dst_pages.get(pos / ps))
                else {
                    debug_assert!(false, "copy_positions_between: pos {pos} out of page range");
                    continue;
                };
                let sin = (pos % ps) as u32;
                let din = (pos % ps) as u32;
                for h in 0..self.n_kv_heads {
                    for kv in 0..2u32 {
                        let so = self
                            .cell_offset(sp, layer, kv, h, sin)
                            .expect("copy_positions_between: src cell in range");
                        tmp.copy_from_slice(&self.data[so..so + hd]);
                        let do_ = self
                            .cell_offset(dp, layer, kv, h, din)
                            .expect("copy_positions_between: dst cell in range");
                        self.data[do_..do_ + hd].copy_from_slice(&tmp);
                    }
                }
            }
        }
    }

    /// F7 tree-spec commit: move a speculative tree's ACCEPTED-path K/V rows
    /// into contiguous committed positions. After a paged tree verify wrote
    /// every node's K/V at absolute position `base + node_index`, the accepted
    /// root→leaf path's nodes sit at scattered positions `base + src_slots[j]`;
    /// this copies each to its committed position `base + j`, across every layer
    /// and head. The caller then shrinks the cache's `seq_len` to `base +
    /// src_slots.len()` (reclaiming the rejected drafts' positions — the paged
    /// analog of the contiguous [`crate::llama_arch::KvCache::compact_tokens`] +
    /// truncate). `pages` is the slot's page list (absolute position → page via
    /// `page_size`). F32 store only (the paged tree-spec KV path).
    ///
    /// A depth-`j` node's flat index is always `>= j`, so `src_slots[j] >= j`
    /// and the ascending copy never clobbers a not-yet-moved source; `src == j`
    /// slots (the common all-accept / chain case) are skipped. Reading each row
    /// into a short temp first keeps `src == dst` page aliasing safe.
    pub fn compact_positions(&mut self, pages: &[PageId], base: usize, src_slots: &[usize]) {
        let ps = self.page_size as usize;
        for (j, &src) in src_slots.iter().enumerate() {
            if src == j {
                continue;
            }
            debug_assert!(src >= j, "compact source slot {src} must be >= dest {j}");
            let src_pos = base + src;
            let dst_pos = base + j;
            let (Some(&src_page), Some(&dst_page)) =
                (pages.get(src_pos / ps), pages.get(dst_pos / ps))
            else {
                debug_assert!(false, "compact_positions: pos out of the slot's page range");
                continue;
            };
            let src_in = (src_pos % ps) as u32;
            let dst_in = (dst_pos % ps) as u32;
            for layer in 0..self.n_layers {
                for h in 0..self.n_kv_heads {
                    // Read src rows into temps first so a same-page src/dst move
                    // (common when both positions share one page) is overlap-safe.
                    let k = match self.k_row(src_page, layer, h, src_in) {
                        Some(r) => r.to_vec(),
                        None => continue,
                    };
                    let v = match self.v_row(src_page, layer, h, src_in) {
                        Some(r) => r.to_vec(),
                        None => continue,
                    };
                    if let Some(dst) = self.k_row_mut(dst_page, layer, h, dst_in) {
                        dst.copy_from_slice(&k);
                    }
                    if let Some(dst) = self.v_row_mut(dst_page, layer, h, dst_in) {
                        dst.copy_from_slice(&v);
                    }
                }
            }
        }
    }

    /// Zero out every cell of one page. Used when a freed page is
    /// re-allocated to a new slot — leftover bytes from the previous
    /// owner are never read by attention (the kv_len bound guards
    /// it), but zeroing makes debug dumps + tests deterministic.
    pub fn zero_page(&mut self, page: PageId) -> Option<()> {
        if page.0 >= self.total_pages {
            return None;
        }
        let start = (page.0 as usize) * self.page_stride;
        let end = start + self.page_stride;
        for v in &mut self.data[start..end] {
            *v = 0.0;
        }
        Some(())
    }
}

// ============================================================
// H9a: trait abstraction over paged-KV storage variants.
// ============================================================

/// H9a: object-safe abstraction over the paged-KV storage variants.
/// `LlamaModel::forward_one_paged` takes a `&mut dyn PagedKvStoreOps`
/// so the F32 (`PagedKvStore`) and Q8_0 (`PagedKvStoreQ8_0`) storage
/// types route through the same code path. The write_token /
/// gather_layer signatures are byte-for-byte the same modulo
/// quantize-on-write + dequantize-on-gather inside the Q8_0 impl.
pub trait PagedKvStoreOps: Send + Sync {
    fn write_token(
        &mut self,
        pages: &[PageId],
        layer: u32,
        pos: u32,
        k: &[f32],
        v: &[f32],
    ) -> Option<()>;
    fn gather_layer(
        &self,
        pages: &[PageId],
        layer: u32,
        kv_len: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Option<()>;
    fn zero_page(&mut self, page: PageId) -> Option<()>;
    fn page_size(&self) -> u32;
    fn n_layers(&self) -> u32;
    fn n_kv_heads(&self) -> u32;
    fn head_dim(&self) -> u32;
    fn total_pages(&self) -> u32;
}

impl PagedKvStoreOps for PagedKvStore {
    fn write_token(
        &mut self,
        pages: &[PageId],
        layer: u32,
        pos: u32,
        k: &[f32],
        v: &[f32],
    ) -> Option<()> {
        PagedKvStore::write_token(self, pages, layer, pos, k, v)
    }
    fn gather_layer(
        &self,
        pages: &[PageId],
        layer: u32,
        kv_len: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Option<()> {
        PagedKvStore::gather_layer(self, pages, layer, kv_len, k_out, v_out)
    }
    fn zero_page(&mut self, page: PageId) -> Option<()> {
        PagedKvStore::zero_page(self, page)
    }
    fn page_size(&self) -> u32 {
        PagedKvStore::page_size(self)
    }
    fn n_layers(&self) -> u32 {
        PagedKvStore::n_layers(self)
    }
    fn n_kv_heads(&self) -> u32 {
        PagedKvStore::n_kv_heads(self)
    }
    fn head_dim(&self) -> u32 {
        PagedKvStore::head_dim(self)
    }
    fn total_pages(&self) -> u32 {
        PagedKvStore::total_pages(self)
    }
}

impl PagedKvStoreOps for PagedKvStoreQ8_0 {
    fn write_token(
        &mut self,
        pages: &[PageId],
        layer: u32,
        pos: u32,
        k: &[f32],
        v: &[f32],
    ) -> Option<()> {
        PagedKvStoreQ8_0::write_token(self, pages, layer, pos, k, v)
    }
    fn gather_layer(
        &self,
        pages: &[PageId],
        layer: u32,
        kv_len: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Option<()> {
        PagedKvStoreQ8_0::gather_layer(self, pages, layer, kv_len, k_out, v_out)
    }
    fn zero_page(&mut self, page: PageId) -> Option<()> {
        PagedKvStoreQ8_0::zero_page(self, page)
    }
    fn page_size(&self) -> u32 {
        PagedKvStoreQ8_0::page_size(self)
    }
    fn n_layers(&self) -> u32 {
        PagedKvStoreQ8_0::n_layers(self)
    }
    fn n_kv_heads(&self) -> u32 {
        PagedKvStoreQ8_0::n_kv_heads(self)
    }
    fn head_dim(&self) -> u32 {
        PagedKvStoreQ8_0::head_dim(self)
    }
    fn total_pages(&self) -> u32 {
        PagedKvStoreQ8_0::total_pages(self)
    }
}

impl PagedKvStoreOps for PagedKvStoreF16 {
    fn write_token(
        &mut self,
        pages: &[PageId],
        layer: u32,
        pos: u32,
        k: &[f32],
        v: &[f32],
    ) -> Option<()> {
        PagedKvStoreF16::write_token(self, pages, layer, pos, k, v)
    }
    fn gather_layer(
        &self,
        pages: &[PageId],
        layer: u32,
        kv_len: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Option<()> {
        PagedKvStoreF16::gather_layer(self, pages, layer, kv_len, k_out, v_out)
    }
    fn zero_page(&mut self, page: PageId) -> Option<()> {
        PagedKvStoreF16::zero_page(self, page)
    }
    fn page_size(&self) -> u32 {
        PagedKvStoreF16::page_size(self)
    }
    fn n_layers(&self) -> u32 {
        PagedKvStoreF16::n_layers(self)
    }
    fn n_kv_heads(&self) -> u32 {
        PagedKvStoreF16::n_kv_heads(self)
    }
    fn head_dim(&self) -> u32 {
        PagedKvStoreF16::head_dim(self)
    }
    fn total_pages(&self) -> u32 {
        PagedKvStoreF16::total_pages(self)
    }
}

// ============================================================
// G2.4: Q8_0 paged KV storage.
// ============================================================

/// Q8_0 paged KV storage. Same layout shape as [`PagedKvStore`] but
/// each `(page, layer, kv_axis, head, pos_in_page)` cell holds:
///   - `head_dim` i8 quantized values
///   - 1 f32 per-row scale
///
/// Per-row byte size = `head_dim + 4`. Lifts the historic
/// "paged requires F32" constraint for Q8_0 KV-dtype.
///
/// `write_token` takes f32 K/V slices and quantizes per-head row
/// before storing; `gather_layer` dequantizes back to f32 into the
/// caller's contiguous `[n_kv_heads, kv_len, head_dim]` slabs so
/// existing attention kernels can consume the gather output
/// unchanged.
#[derive(Debug)]
pub struct PagedKvStoreQ8_0 {
    /// Byte storage. Length = `total_pages × page_byte_stride`.
    data: Vec<u8>,
    total_pages: u32,
    n_layers: u32,
    n_kv_heads: u32,
    page_size: u32,
    head_dim: u32,
    /// Precomputed strides (in bytes).
    bytes_per_row: usize, // head_dim + 4
    head_stride: usize,  // page_size * bytes_per_row
    kv_stride: usize,    // n_kv_heads * head_stride
    layer_stride: usize, // 2 * kv_stride
    page_stride: usize,  // n_layers * layer_stride
}

impl PagedKvStoreQ8_0 {
    /// Bytes of pool storage (memory-budget planner input).
    pub fn approx_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn new(
        total_pages: u32,
        n_layers: u32,
        n_kv_heads: u32,
        page_size: u32,
        head_dim: u32,
    ) -> Option<Self> {
        if total_pages == 0 || n_layers == 0 || n_kv_heads == 0 || page_size == 0 || head_dim == 0 {
            return None;
        }
        let bytes_per_row = (head_dim as usize).checked_add(4)?;
        let head_stride = (page_size as usize).checked_mul(bytes_per_row)?;
        let kv_stride = (n_kv_heads as usize).checked_mul(head_stride)?;
        let layer_stride = kv_stride.checked_mul(2)?;
        let page_stride = (n_layers as usize).checked_mul(layer_stride)?;
        let total_bytes = (total_pages as usize).checked_mul(page_stride)?;
        Some(Self {
            data: vec![0u8; total_bytes],
            total_pages,
            n_layers,
            n_kv_heads,
            page_size,
            head_dim,
            bytes_per_row,
            head_stride,
            kv_stride,
            layer_stride,
            page_stride,
        })
    }

    pub fn total_pages(&self) -> u32 {
        self.total_pages
    }
    pub fn n_layers(&self) -> u32 {
        self.n_layers
    }
    pub fn n_kv_heads(&self) -> u32 {
        self.n_kv_heads
    }
    pub fn page_size(&self) -> u32 {
        self.page_size
    }
    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }
    pub fn page_stride(&self) -> usize {
        self.page_stride
    }

    #[inline]
    fn cell_offset(
        &self,
        page: PageId,
        layer: u32,
        kv_axis: u32,
        h: u32,
        pos_in_page: u32,
    ) -> Option<usize> {
        if page.0 >= self.total_pages
            || layer >= self.n_layers
            || kv_axis >= 2
            || h >= self.n_kv_heads
            || pos_in_page >= self.page_size
        {
            return None;
        }
        let off = (page.0 as usize) * self.page_stride
            + (layer as usize) * self.layer_stride
            + (kv_axis as usize) * self.kv_stride
            + (h as usize) * self.head_stride
            + (pos_in_page as usize) * self.bytes_per_row;
        Some(off)
    }

    /// Quantize one f32 K or V row to Q8_0 (i8 + f32 scale) and write
    /// at `(page, layer, kv_axis, h, pos_in_page)`. Algorithm matches
    /// the contiguous Q8_0 path: `scale = max|x|/127`, `q = round(x/scale)`
    /// clamped to `[-128, 127]`. Scale is `1.0` when all zeros.
    #[inline]
    fn quantize_row_into(&mut self, off: usize, src_row: &[f32]) {
        debug_assert_eq!(src_row.len(), self.head_dim as usize);
        let mut max_abs = 0f32;
        for &x in src_row {
            let a = x.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let hd = self.head_dim as usize;
        let q_bytes = &mut self.data[off..off + hd];
        let (scale, inv) = if max_abs == 0.0 {
            (1.0f32, 0.0f32)
        } else {
            let s = max_abs / 127.0;
            (s, 1.0 / s)
        };
        if max_abs == 0.0 {
            for q in q_bytes.iter_mut() {
                *q = 0;
            }
        } else {
            for (q, &x) in q_bytes.iter_mut().zip(src_row.iter()) {
                let r = (x * inv).round().clamp(-128.0, 127.0) as i8;
                *q = r as u8;
            }
        }
        // Scale bytes immediately follow the head_dim i8 values.
        let scale_bytes = scale.to_le_bytes();
        self.data[off + hd..off + hd + 4].copy_from_slice(&scale_bytes);
    }

    /// Dequantize one Q8_0 stored row into a destination f32 slice.
    #[inline]
    fn dequantize_row_into(&self, off: usize, dst: &mut [f32]) {
        let hd = self.head_dim as usize;
        debug_assert_eq!(dst.len(), hd);
        let scale_bytes: [u8; 4] = self.data[off + hd..off + hd + 4]
            .try_into()
            .expect("4-byte slice");
        let scale = f32::from_le_bytes(scale_bytes);
        for (i, d) in dst.iter_mut().enumerate() {
            let q = self.data[off + i] as i8;
            *d = (q as f32) * scale;
        }
    }

    /// Same shape as [`PagedKvStore::write_token`] — quantize each
    /// f32 head row to Q8_0 and write into the paged storage.
    pub fn write_token(
        &mut self,
        pages: &[PageId],
        layer: u32,
        pos: u32,
        k: &[f32],
        v: &[f32],
    ) -> Option<()> {
        let hd = self.head_dim as usize;
        let need = (self.n_kv_heads as usize) * hd;
        if k.len() != need || v.len() != need {
            return None;
        }
        let page_idx = (pos / self.page_size) as usize;
        let pos_in_page = pos % self.page_size;
        let page = *pages.get(page_idx)?;
        for h in 0..self.n_kv_heads {
            let h_off_src = (h as usize) * hd;
            let k_off = self.cell_offset(page, layer, 0, h, pos_in_page)?;
            self.quantize_row_into(k_off, &k[h_off_src..h_off_src + hd]);
            let v_off = self.cell_offset(page, layer, 1, h, pos_in_page)?;
            self.quantize_row_into(v_off, &v[h_off_src..h_off_src + hd]);
        }
        Some(())
    }

    /// Same shape as [`PagedKvStore::gather_layer`] — dequantize each
    /// Q8_0 row back to f32 into the caller's contiguous slabs.
    /// Output layout matches the existing attention kernels'
    /// `[n_kv_heads, kv_len, head_dim]` expectation.
    pub fn gather_layer(
        &self,
        pages: &[PageId],
        layer: u32,
        kv_len: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Option<()> {
        if layer >= self.n_layers {
            return None;
        }
        let hd = self.head_dim as usize;
        let need = (self.n_kv_heads as usize) * (kv_len as usize) * hd;
        if k_out.len() != need || v_out.len() != need {
            return None;
        }
        let pages_needed = (kv_len as usize).div_ceil(self.page_size as usize);
        if pages.len() < pages_needed {
            return None;
        }
        let page_size = self.page_size as usize;
        let kv_len_us = kv_len as usize;
        for h in 0..self.n_kv_heads {
            let out_head_base = (h as usize) * kv_len_us * hd;
            let mut pos_written = 0usize;
            for (page_i, &page) in pages.iter().enumerate() {
                if pos_written >= kv_len_us {
                    break;
                }
                let remaining = kv_len_us - pos_written;
                let pos_in_this_page = remaining.min(page_size);
                for p in 0..pos_in_this_page {
                    let k_off = self
                        .cell_offset(page, layer, 0, h, p as u32)
                        .expect("page/layer/head/pos in range");
                    let v_off = self
                        .cell_offset(page, layer, 1, h, p as u32)
                        .expect("page/layer/head/pos in range");
                    let dst_pos = out_head_base + (page_i * page_size + p) * hd;
                    self.dequantize_row_into(k_off, &mut k_out[dst_pos..dst_pos + hd]);
                    self.dequantize_row_into(v_off, &mut v_out[dst_pos..dst_pos + hd]);
                }
                pos_written += pos_in_this_page;
            }
        }
        Some(())
    }

    /// Zero out one page (debug determinism — attention's kv_len bound
    /// already guards against stale reads).
    pub fn zero_page(&mut self, page: PageId) -> Option<()> {
        if page.0 >= self.total_pages {
            return None;
        }
        let start = (page.0 as usize) * self.page_stride;
        let end = start + self.page_stride;
        for v in &mut self.data[start..end] {
            *v = 0;
        }
        Some(())
    }
}

// ============================================================
// M6: half-precision (f16 / bf16) paged KV storage.
// ============================================================

/// Half-precision paged KV storage. Same page geometry as
/// [`PagedKvStore`] but each element is two bytes (an f16 or bf16 bit
/// pattern, selected by `bf16`), with no block scale. `write_token`
/// rounds the f32 K/V rows to half on write; `gather_layer` widens them
/// back to f32 into the caller's contiguous `[n_kv_heads, kv_len,
/// head_dim]` slabs, so the existing f32 attention kernels consume the
/// gather output unchanged. Covers BOTH f16 and bf16 (the `bf16` flag is
/// the only difference) — the mid-range paged KV option between f32 and
/// q8_0.
#[derive(Debug)]
pub struct PagedKvStoreF16 {
    /// True ⇒ bfloat16 bit patterns; false ⇒ IEEE f16.
    bf16: bool,
    /// Byte storage. Length = `total_pages × page_byte_stride`.
    data: Vec<u8>,
    total_pages: u32,
    n_layers: u32,
    n_kv_heads: u32,
    page_size: u32,
    head_dim: u32,
    /// Precomputed strides (in bytes). `bytes_per_row = head_dim * 2`.
    bytes_per_row: usize,
    head_stride: usize,
    kv_stride: usize,
    layer_stride: usize,
    page_stride: usize,
}

impl PagedKvStoreF16 {
    /// Bytes of pool storage (memory-budget planner input).
    pub fn approx_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn new(
        total_pages: u32,
        n_layers: u32,
        n_kv_heads: u32,
        page_size: u32,
        head_dim: u32,
        bf16: bool,
    ) -> Option<Self> {
        if total_pages == 0 || n_layers == 0 || n_kv_heads == 0 || page_size == 0 || head_dim == 0 {
            return None;
        }
        let bytes_per_row = (head_dim as usize).checked_mul(2)?;
        let head_stride = (page_size as usize).checked_mul(bytes_per_row)?;
        let kv_stride = (n_kv_heads as usize).checked_mul(head_stride)?;
        let layer_stride = kv_stride.checked_mul(2)?;
        let page_stride = (n_layers as usize).checked_mul(layer_stride)?;
        let total_bytes = (total_pages as usize).checked_mul(page_stride)?;
        Some(Self {
            bf16,
            data: vec![0u8; total_bytes],
            total_pages,
            n_layers,
            n_kv_heads,
            page_size,
            head_dim,
            bytes_per_row,
            head_stride,
            kv_stride,
            layer_stride,
            page_stride,
        })
    }

    pub fn total_pages(&self) -> u32 {
        self.total_pages
    }
    pub fn n_layers(&self) -> u32 {
        self.n_layers
    }
    pub fn n_kv_heads(&self) -> u32 {
        self.n_kv_heads
    }
    pub fn page_size(&self) -> u32 {
        self.page_size
    }
    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }
    pub fn page_stride(&self) -> usize {
        self.page_stride
    }
    pub fn is_bf16(&self) -> bool {
        self.bf16
    }

    #[inline]
    fn cell_offset(
        &self,
        page: PageId,
        layer: u32,
        kv_axis: u32,
        h: u32,
        pos_in_page: u32,
    ) -> Option<usize> {
        if page.0 >= self.total_pages
            || layer >= self.n_layers
            || kv_axis >= 2
            || h >= self.n_kv_heads
            || pos_in_page >= self.page_size
        {
            return None;
        }
        let off = (page.0 as usize) * self.page_stride
            + (layer as usize) * self.layer_stride
            + (kv_axis as usize) * self.kv_stride
            + (h as usize) * self.head_stride
            + (pos_in_page as usize) * self.bytes_per_row;
        Some(off)
    }

    /// Round one f32 row to half and store its little-endian bit
    /// patterns at `off` (`head_dim` values, 2 bytes each).
    #[inline]
    fn encode_row_into(&mut self, off: usize, src_row: &[f32]) {
        debug_assert_eq!(src_row.len(), self.head_dim as usize);
        let dst = &mut self.data[off..off + src_row.len() * 2];
        if self.bf16 {
            for (c, &x) in dst.chunks_exact_mut(2).zip(src_row.iter()) {
                c.copy_from_slice(&half::bf16::from_f32(x).to_bits().to_le_bytes());
            }
        } else {
            for (c, &x) in dst.chunks_exact_mut(2).zip(src_row.iter()) {
                c.copy_from_slice(&half::f16::from_f32(x).to_bits().to_le_bytes());
            }
        }
    }

    /// Widen one stored half row back to f32.
    #[inline]
    fn decode_row_into(&self, off: usize, dst: &mut [f32]) {
        let hd = self.head_dim as usize;
        debug_assert_eq!(dst.len(), hd);
        let src = &self.data[off..off + hd * 2];
        if self.bf16 {
            for (d, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                *d = half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
            }
        } else {
            for (d, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                *d = half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
            }
        }
    }

    /// Same shape as [`PagedKvStore::write_token`] — round each f32 head
    /// row to half and write into the paged storage.
    pub fn write_token(
        &mut self,
        pages: &[PageId],
        layer: u32,
        pos: u32,
        k: &[f32],
        v: &[f32],
    ) -> Option<()> {
        let hd = self.head_dim as usize;
        let need = (self.n_kv_heads as usize) * hd;
        if k.len() != need || v.len() != need {
            return None;
        }
        let page_idx = (pos / self.page_size) as usize;
        let pos_in_page = pos % self.page_size;
        let page = *pages.get(page_idx)?;
        for h in 0..self.n_kv_heads {
            let h_off_src = (h as usize) * hd;
            let k_off = self.cell_offset(page, layer, 0, h, pos_in_page)?;
            self.encode_row_into(k_off, &k[h_off_src..h_off_src + hd]);
            let v_off = self.cell_offset(page, layer, 1, h, pos_in_page)?;
            self.encode_row_into(v_off, &v[h_off_src..h_off_src + hd]);
        }
        Some(())
    }

    /// Same shape as [`PagedKvStore::gather_layer`] — widen each half row
    /// back to f32 into the caller's contiguous `[n_kv_heads, kv_len,
    /// head_dim]` slabs.
    pub fn gather_layer(
        &self,
        pages: &[PageId],
        layer: u32,
        kv_len: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Option<()> {
        if layer >= self.n_layers {
            return None;
        }
        let hd = self.head_dim as usize;
        let need = (self.n_kv_heads as usize) * (kv_len as usize) * hd;
        if k_out.len() != need || v_out.len() != need {
            return None;
        }
        let pages_needed = (kv_len as usize).div_ceil(self.page_size as usize);
        if pages.len() < pages_needed {
            return None;
        }
        let page_size = self.page_size as usize;
        let kv_len_us = kv_len as usize;
        for h in 0..self.n_kv_heads {
            let out_head_base = (h as usize) * kv_len_us * hd;
            let mut pos_written = 0usize;
            for (page_i, &page) in pages.iter().enumerate() {
                if pos_written >= kv_len_us {
                    break;
                }
                let remaining = kv_len_us - pos_written;
                let pos_in_this_page = remaining.min(page_size);
                for p in 0..pos_in_this_page {
                    let k_off = self
                        .cell_offset(page, layer, 0, h, p as u32)
                        .expect("page/layer/head/pos in range");
                    let v_off = self
                        .cell_offset(page, layer, 1, h, p as u32)
                        .expect("page/layer/head/pos in range");
                    let dst_pos = out_head_base + (page_i * page_size + p) * hd;
                    self.decode_row_into(k_off, &mut k_out[dst_pos..dst_pos + hd]);
                    self.decode_row_into(v_off, &mut v_out[dst_pos..dst_pos + hd]);
                }
                pos_written += pos_in_this_page;
            }
        }
        Some(())
    }

    /// Zero out one page (debug determinism — attention's kv_len bound
    /// already guards against stale reads).
    pub fn zero_page(&mut self, page: PageId) -> Option<()> {
        if page.0 >= self.total_pages {
            return None;
        }
        let start = (page.0 as usize) * self.page_stride;
        let end = start + self.page_stride;
        for v in &mut self.data[start..end] {
            *v = 0;
        }
        Some(())
    }
}

// ============================================================
// H9b: TurboQuant paged KV storage.
// ============================================================

/// TurboQuant paged KV storage. Each `(page, layer, kv_axis, head,
/// pos_in_page)` cell holds:
///   - `bytes_per_block(head_dim, bits)` packed bytes
///   - 1 f32 per-row scale
///
/// Per-row byte size = `bytes_per_block(head_dim, bits) + 4`.
/// Quantize-on-write uses [`rustllama_kernels_cpu::turboquant::quantize_row`];
/// dequant-on-gather uses [`rustllama_kernels_cpu::turboquant::dequantize_row`].
/// The bit width is fixed at construction.
#[derive(Debug)]
#[allow(non_camel_case_types)]
pub struct PagedKvStoreTQ {
    data: Vec<u8>,
    total_pages: u32,
    n_layers: u32,
    n_kv_heads: u32,
    page_size: u32,
    head_dim: u32,
    bits: u8,
    bytes_per_row: usize,
    packed_bytes: usize, // bytes_per_block(head_dim, bits)
    head_stride: usize,
    kv_stride: usize,
    layer_stride: usize,
    page_stride: usize,
}

impl PagedKvStoreTQ {
    /// Bytes of pool storage (memory-budget planner input).
    pub fn approx_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn new(
        total_pages: u32,
        n_layers: u32,
        n_kv_heads: u32,
        page_size: u32,
        head_dim: u32,
        bits: u8,
    ) -> Option<Self> {
        if total_pages == 0
            || n_layers == 0
            || n_kv_heads == 0
            || page_size == 0
            || head_dim == 0
            || !matches!(bits, 1 | 2 | 4 | 8)
        {
            return None;
        }
        let packed_bytes =
            rustllama_kernels_cpu::turboquant::bytes_per_block(head_dim as usize, bits);
        let bytes_per_row = packed_bytes.checked_add(4)?;
        let head_stride = (page_size as usize).checked_mul(bytes_per_row)?;
        let kv_stride = (n_kv_heads as usize).checked_mul(head_stride)?;
        let layer_stride = kv_stride.checked_mul(2)?;
        let page_stride = (n_layers as usize).checked_mul(layer_stride)?;
        let total_bytes = (total_pages as usize).checked_mul(page_stride)?;
        Some(Self {
            data: vec![0u8; total_bytes],
            total_pages,
            n_layers,
            n_kv_heads,
            page_size,
            head_dim,
            bits,
            bytes_per_row,
            packed_bytes,
            head_stride,
            kv_stride,
            layer_stride,
            page_stride,
        })
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }
    pub fn total_pages(&self) -> u32 {
        self.total_pages
    }
    pub fn n_layers(&self) -> u32 {
        self.n_layers
    }
    pub fn n_kv_heads(&self) -> u32 {
        self.n_kv_heads
    }
    pub fn page_size(&self) -> u32 {
        self.page_size
    }
    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }

    #[inline]
    fn cell_offset(
        &self,
        page: PageId,
        layer: u32,
        kv_axis: u32,
        h: u32,
        pos_in_page: u32,
    ) -> Option<usize> {
        if page.0 >= self.total_pages
            || layer >= self.n_layers
            || kv_axis >= 2
            || h >= self.n_kv_heads
            || pos_in_page >= self.page_size
        {
            return None;
        }
        let off = (page.0 as usize) * self.page_stride
            + (layer as usize) * self.layer_stride
            + (kv_axis as usize) * self.kv_stride
            + (h as usize) * self.head_stride
            + (pos_in_page as usize) * self.bytes_per_row;
        Some(off)
    }

    pub fn write_token(
        &mut self,
        pages: &[PageId],
        layer: u32,
        pos: u32,
        k: &[f32],
        v: &[f32],
    ) -> Option<()> {
        let hd = self.head_dim as usize;
        let need = (self.n_kv_heads as usize) * hd;
        if k.len() != need || v.len() != need {
            return None;
        }
        let page_idx = (pos / self.page_size) as usize;
        let pos_in_page = pos % self.page_size;
        let page = *pages.get(page_idx)?;
        // `quantize_row` mutates its input (in-place WHT) — clone into scratch.
        let mut row_scratch = vec![0f32; hd];
        for h in 0..self.n_kv_heads {
            let h_off_src = (h as usize) * hd;
            let k_off = self.cell_offset(page, layer, 0, h, pos_in_page)?;
            row_scratch.copy_from_slice(&k[h_off_src..h_off_src + hd]);
            let scale = rustllama_kernels_cpu::turboquant::quantize_row(
                &mut row_scratch,
                self.bits,
                &mut self.data[k_off..k_off + self.packed_bytes],
            );
            self.data[k_off + self.packed_bytes..k_off + self.packed_bytes + 4]
                .copy_from_slice(&scale.to_le_bytes());

            let v_off = self.cell_offset(page, layer, 1, h, pos_in_page)?;
            row_scratch.copy_from_slice(&v[h_off_src..h_off_src + hd]);
            let scale = rustllama_kernels_cpu::turboquant::quantize_row(
                &mut row_scratch,
                self.bits,
                &mut self.data[v_off..v_off + self.packed_bytes],
            );
            self.data[v_off + self.packed_bytes..v_off + self.packed_bytes + 4]
                .copy_from_slice(&scale.to_le_bytes());
        }
        Some(())
    }

    pub fn gather_layer(
        &self,
        pages: &[PageId],
        layer: u32,
        kv_len: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Option<()> {
        if layer >= self.n_layers {
            return None;
        }
        let hd = self.head_dim as usize;
        let need = (self.n_kv_heads as usize) * (kv_len as usize) * hd;
        if k_out.len() != need || v_out.len() != need {
            return None;
        }
        let pages_needed = (kv_len as usize).div_ceil(self.page_size as usize);
        if pages.len() < pages_needed {
            return None;
        }
        let page_size = self.page_size as usize;
        let kv_len_us = kv_len as usize;
        for h in 0..self.n_kv_heads {
            let out_head_base = (h as usize) * kv_len_us * hd;
            let mut pos_written = 0usize;
            for (page_i, &page) in pages.iter().enumerate() {
                if pos_written >= kv_len_us {
                    break;
                }
                let remaining = kv_len_us - pos_written;
                let pos_in_this_page = remaining.min(page_size);
                for p in 0..pos_in_this_page {
                    let k_off = self
                        .cell_offset(page, layer, 0, h, p as u32)
                        .expect("page/layer/head/pos in range");
                    let v_off = self
                        .cell_offset(page, layer, 1, h, p as u32)
                        .expect("page/layer/head/pos in range");
                    let dst_pos = out_head_base + (page_i * page_size + p) * hd;
                    let k_scale_bytes: [u8; 4] = self.data
                        [k_off + self.packed_bytes..k_off + self.packed_bytes + 4]
                        .try_into()
                        .expect("4-byte slice");
                    let v_scale_bytes: [u8; 4] = self.data
                        [v_off + self.packed_bytes..v_off + self.packed_bytes + 4]
                        .try_into()
                        .expect("4-byte slice");
                    rustllama_kernels_cpu::turboquant::dequantize_row(
                        &self.data[k_off..k_off + self.packed_bytes],
                        f32::from_le_bytes(k_scale_bytes),
                        self.bits,
                        &mut k_out[dst_pos..dst_pos + hd],
                    );
                    rustllama_kernels_cpu::turboquant::dequantize_row(
                        &self.data[v_off..v_off + self.packed_bytes],
                        f32::from_le_bytes(v_scale_bytes),
                        self.bits,
                        &mut v_out[dst_pos..dst_pos + hd],
                    );
                }
                pos_written += pos_in_this_page;
            }
        }
        Some(())
    }

    pub fn zero_page(&mut self, page: PageId) -> Option<()> {
        if page.0 >= self.total_pages {
            return None;
        }
        let start = (page.0 as usize) * self.page_stride;
        let end = start + self.page_stride;
        for v in &mut self.data[start..end] {
            *v = 0;
        }
        Some(())
    }
}

impl PagedKvStoreOps for PagedKvStoreTQ {
    fn write_token(
        &mut self,
        pages: &[PageId],
        layer: u32,
        pos: u32,
        k: &[f32],
        v: &[f32],
    ) -> Option<()> {
        PagedKvStoreTQ::write_token(self, pages, layer, pos, k, v)
    }
    fn gather_layer(
        &self,
        pages: &[PageId],
        layer: u32,
        kv_len: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Option<()> {
        PagedKvStoreTQ::gather_layer(self, pages, layer, kv_len, k_out, v_out)
    }
    fn zero_page(&mut self, page: PageId) -> Option<()> {
        PagedKvStoreTQ::zero_page(self, page)
    }
    fn page_size(&self) -> u32 {
        PagedKvStoreTQ::page_size(self)
    }
    fn n_layers(&self) -> u32 {
        PagedKvStoreTQ::n_layers(self)
    }
    fn n_kv_heads(&self) -> u32 {
        PagedKvStoreTQ::n_kv_heads(self)
    }
    fn head_dim(&self) -> u32 {
        PagedKvStoreTQ::head_dim(self)
    }
    fn total_pages(&self) -> u32 {
        PagedKvStoreTQ::total_pages(self)
    }
}

// ============================================================
// H9b: NVFP4 paged KV storage.
// ============================================================

/// NVFP4 paged KV storage. Each `(page, layer, kv_axis, head,
/// pos_in_page)` cell holds `head_dim / 16` blocks of 9 bytes
/// each (8 i4 codes + 1 FP8 E4M3 scale per block). No separate
/// per-row scale — scales are embedded per-block.
///
/// Per-row byte size = `head_dim / 16 × NVFP4_BLOCK_BYTES`.
/// `head_dim` must be a multiple of 16.
#[derive(Debug)]
#[allow(non_camel_case_types)]
pub struct PagedKvStoreNvfp4 {
    data: Vec<u8>,
    total_pages: u32,
    n_layers: u32,
    n_kv_heads: u32,
    page_size: u32,
    head_dim: u32,
    bytes_per_row: usize,
    head_stride: usize,
    kv_stride: usize,
    layer_stride: usize,
    page_stride: usize,
}

impl PagedKvStoreNvfp4 {
    /// Bytes of pool storage (memory-budget planner input).
    pub fn approx_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn new(
        total_pages: u32,
        n_layers: u32,
        n_kv_heads: u32,
        page_size: u32,
        head_dim: u32,
    ) -> Option<Self> {
        if total_pages == 0
            || n_layers == 0
            || n_kv_heads == 0
            || page_size == 0
            || head_dim == 0
            || (head_dim as usize) % rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS != 0
        {
            return None;
        }
        let blocks_per_row = (head_dim as usize) / rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS;
        let bytes_per_row =
            blocks_per_row.checked_mul(rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES)?;
        let head_stride = (page_size as usize).checked_mul(bytes_per_row)?;
        let kv_stride = (n_kv_heads as usize).checked_mul(head_stride)?;
        let layer_stride = kv_stride.checked_mul(2)?;
        let page_stride = (n_layers as usize).checked_mul(layer_stride)?;
        let total_bytes = (total_pages as usize).checked_mul(page_stride)?;
        Some(Self {
            data: vec![0u8; total_bytes],
            total_pages,
            n_layers,
            n_kv_heads,
            page_size,
            head_dim,
            bytes_per_row,
            head_stride,
            kv_stride,
            layer_stride,
            page_stride,
        })
    }

    pub fn total_pages(&self) -> u32 {
        self.total_pages
    }
    pub fn n_layers(&self) -> u32 {
        self.n_layers
    }
    pub fn n_kv_heads(&self) -> u32 {
        self.n_kv_heads
    }
    pub fn page_size(&self) -> u32 {
        self.page_size
    }
    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }

    #[inline]
    fn cell_offset(
        &self,
        page: PageId,
        layer: u32,
        kv_axis: u32,
        h: u32,
        pos_in_page: u32,
    ) -> Option<usize> {
        if page.0 >= self.total_pages
            || layer >= self.n_layers
            || kv_axis >= 2
            || h >= self.n_kv_heads
            || pos_in_page >= self.page_size
        {
            return None;
        }
        let off = (page.0 as usize) * self.page_stride
            + (layer as usize) * self.layer_stride
            + (kv_axis as usize) * self.kv_stride
            + (h as usize) * self.head_stride
            + (pos_in_page as usize) * self.bytes_per_row;
        Some(off)
    }

    fn quantize_row_into(&mut self, off: usize, src_row: &[f32]) {
        let hd = self.head_dim as usize;
        debug_assert_eq!(src_row.len(), hd);
        let blocks = hd / rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS;
        let bb = rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES;
        let be = rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS;
        for b in 0..blocks {
            rustllama_kernels_cpu::nvfp4::quantize_block(
                &src_row[b * be..(b + 1) * be],
                &mut self.data[off + b * bb..off + (b + 1) * bb],
            );
        }
    }

    fn dequantize_row_into(&self, off: usize, dst: &mut [f32]) {
        let hd = self.head_dim as usize;
        debug_assert_eq!(dst.len(), hd);
        let blocks = hd / rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS;
        let bb = rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_BYTES;
        let be = rustllama_kernels_cpu::nvfp4::NVFP4_BLOCK_ELEMS;
        for b in 0..blocks {
            rustllama_kernels_cpu::nvfp4::dequantize_block(
                &self.data[off + b * bb..off + (b + 1) * bb],
                &mut dst[b * be..(b + 1) * be],
            );
        }
    }

    pub fn write_token(
        &mut self,
        pages: &[PageId],
        layer: u32,
        pos: u32,
        k: &[f32],
        v: &[f32],
    ) -> Option<()> {
        let hd = self.head_dim as usize;
        let need = (self.n_kv_heads as usize) * hd;
        if k.len() != need || v.len() != need {
            return None;
        }
        let page_idx = (pos / self.page_size) as usize;
        let pos_in_page = pos % self.page_size;
        let page = *pages.get(page_idx)?;
        for h in 0..self.n_kv_heads {
            let h_off_src = (h as usize) * hd;
            let k_off = self.cell_offset(page, layer, 0, h, pos_in_page)?;
            self.quantize_row_into(k_off, &k[h_off_src..h_off_src + hd]);
            let v_off = self.cell_offset(page, layer, 1, h, pos_in_page)?;
            self.quantize_row_into(v_off, &v[h_off_src..h_off_src + hd]);
        }
        Some(())
    }

    pub fn gather_layer(
        &self,
        pages: &[PageId],
        layer: u32,
        kv_len: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Option<()> {
        if layer >= self.n_layers {
            return None;
        }
        let hd = self.head_dim as usize;
        let need = (self.n_kv_heads as usize) * (kv_len as usize) * hd;
        if k_out.len() != need || v_out.len() != need {
            return None;
        }
        let pages_needed = (kv_len as usize).div_ceil(self.page_size as usize);
        if pages.len() < pages_needed {
            return None;
        }
        let page_size = self.page_size as usize;
        let kv_len_us = kv_len as usize;
        for h in 0..self.n_kv_heads {
            let out_head_base = (h as usize) * kv_len_us * hd;
            let mut pos_written = 0usize;
            for (page_i, &page) in pages.iter().enumerate() {
                if pos_written >= kv_len_us {
                    break;
                }
                let remaining = kv_len_us - pos_written;
                let pos_in_this_page = remaining.min(page_size);
                for p in 0..pos_in_this_page {
                    let k_off = self
                        .cell_offset(page, layer, 0, h, p as u32)
                        .expect("page/layer/head/pos in range");
                    let v_off = self
                        .cell_offset(page, layer, 1, h, p as u32)
                        .expect("page/layer/head/pos in range");
                    let dst_pos = out_head_base + (page_i * page_size + p) * hd;
                    self.dequantize_row_into(k_off, &mut k_out[dst_pos..dst_pos + hd]);
                    self.dequantize_row_into(v_off, &mut v_out[dst_pos..dst_pos + hd]);
                }
                pos_written += pos_in_this_page;
            }
        }
        Some(())
    }

    pub fn zero_page(&mut self, page: PageId) -> Option<()> {
        if page.0 >= self.total_pages {
            return None;
        }
        let start = (page.0 as usize) * self.page_stride;
        let end = start + self.page_stride;
        for v in &mut self.data[start..end] {
            *v = 0;
        }
        Some(())
    }
}

impl PagedKvStoreOps for PagedKvStoreNvfp4 {
    fn write_token(
        &mut self,
        pages: &[PageId],
        layer: u32,
        pos: u32,
        k: &[f32],
        v: &[f32],
    ) -> Option<()> {
        PagedKvStoreNvfp4::write_token(self, pages, layer, pos, k, v)
    }
    fn gather_layer(
        &self,
        pages: &[PageId],
        layer: u32,
        kv_len: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Option<()> {
        PagedKvStoreNvfp4::gather_layer(self, pages, layer, kv_len, k_out, v_out)
    }
    fn zero_page(&mut self, page: PageId) -> Option<()> {
        PagedKvStoreNvfp4::zero_page(self, page)
    }
    fn page_size(&self) -> u32 {
        PagedKvStoreNvfp4::page_size(self)
    }
    fn n_layers(&self) -> u32 {
        PagedKvStoreNvfp4::n_layers(self)
    }
    fn n_kv_heads(&self) -> u32 {
        PagedKvStoreNvfp4::n_kv_heads(self)
    }
    fn head_dim(&self) -> u32 {
        PagedKvStoreNvfp4::head_dim(self)
    }
    fn total_pages(&self) -> u32 {
        PagedKvStoreNvfp4::total_pages(self)
    }
}

// ============================================================
// Wave 2: MXFP4 / MXFP6 / MXFP8 paged KV storage.
// ============================================================

/// Generate an OCP Microscaling (MX) paged KV store — one struct per
/// element format (MXFP4/6/8). Each `(page, layer, kv_axis, head,
/// pos_in_page)` cell holds `head_dim / 32` blocks; a block is 32
/// elements sharing one trailing E8M0 (power-of-two) scale byte,
/// byte-identical to the contiguous MXFP KV blocks in
/// [`rustllama_kernels_cpu::mxfp_kv`] and the weight-side MXFP blocks.
/// Block bytes: MXFP4 17, MXFP6 25, MXFP8 33. There is NO separate
/// per-row scale — the E8M0 scale is embedded per block, like NVFP4
/// (contrast Q8_0 / TQ, which append an f32 row scale).
///
/// Per-row byte size = `head_dim / 32 × <block bytes>`. `head_dim` must
/// be a multiple of 32 (the MX block size), so `new` rejects any other
/// geometry up front rather than mis-striding later.
///
/// `quantize_row_into` runs the per-block encoder (`$qfn`) over each
/// 32-elem block; `dequantize_row_into` uses the row-level MXFP decoder
/// (`$deqrow`), which walks the same block stride the encoder wrote,
/// to reconstruct the whole `head_dim` row into the caller's f32 slab.
/// This mirrors [`PagedKvStoreNvfp4`] exactly modulo the 32-elem (vs
/// NVFP4's 16) block geometry and the MX element codec. The three
/// formats differ only in block bytes + codec, so a macro keeps the
/// offset math single-sourced — the same DRY pattern the flash kernels
/// use in `rustllama_kernels_cpu::mxfp_kv`.
macro_rules! paged_kv_store_mxfp {
    ($name:ident, $blk_bytes:expr, $blk_elems:expr, $qfn:path, $deqrow:path) => {
        #[derive(Debug)]
        #[allow(non_camel_case_types)]
        pub struct $name {
            data: Vec<u8>,
            total_pages: u32,
            n_layers: u32,
            n_kv_heads: u32,
            page_size: u32,
            head_dim: u32,
            bytes_per_row: usize,
            head_stride: usize,
            kv_stride: usize,
            layer_stride: usize,
            page_stride: usize,
        }

        impl $name {
            /// Bytes of pool storage (memory-budget planner input).
            pub fn approx_bytes(&self) -> usize {
                self.data.len()
            }

            pub fn new(
                total_pages: u32,
                n_layers: u32,
                n_kv_heads: u32,
                page_size: u32,
                head_dim: u32,
            ) -> Option<Self> {
                let blk_bytes: usize = $blk_bytes;
                let blk_elems: usize = $blk_elems;
                if total_pages == 0
                    || n_layers == 0
                    || n_kv_heads == 0
                    || page_size == 0
                    || head_dim == 0
                    || (head_dim as usize) % blk_elems != 0
                {
                    return None;
                }
                let blocks_per_row = (head_dim as usize) / blk_elems;
                let bytes_per_row = blocks_per_row.checked_mul(blk_bytes)?;
                let head_stride = (page_size as usize).checked_mul(bytes_per_row)?;
                let kv_stride = (n_kv_heads as usize).checked_mul(head_stride)?;
                let layer_stride = kv_stride.checked_mul(2)?;
                let page_stride = (n_layers as usize).checked_mul(layer_stride)?;
                let total_bytes = (total_pages as usize).checked_mul(page_stride)?;
                Some(Self {
                    data: vec![0u8; total_bytes],
                    total_pages,
                    n_layers,
                    n_kv_heads,
                    page_size,
                    head_dim,
                    bytes_per_row,
                    head_stride,
                    kv_stride,
                    layer_stride,
                    page_stride,
                })
            }

            pub fn total_pages(&self) -> u32 {
                self.total_pages
            }
            pub fn n_layers(&self) -> u32 {
                self.n_layers
            }
            pub fn n_kv_heads(&self) -> u32 {
                self.n_kv_heads
            }
            pub fn page_size(&self) -> u32 {
                self.page_size
            }
            pub fn head_dim(&self) -> u32 {
                self.head_dim
            }

            #[inline]
            fn cell_offset(
                &self,
                page: PageId,
                layer: u32,
                kv_axis: u32,
                h: u32,
                pos_in_page: u32,
            ) -> Option<usize> {
                if page.0 >= self.total_pages
                    || layer >= self.n_layers
                    || kv_axis >= 2
                    || h >= self.n_kv_heads
                    || pos_in_page >= self.page_size
                {
                    return None;
                }
                let off = (page.0 as usize) * self.page_stride
                    + (layer as usize) * self.layer_stride
                    + (kv_axis as usize) * self.kv_stride
                    + (h as usize) * self.head_stride
                    + (pos_in_page as usize) * self.bytes_per_row;
                Some(off)
            }

            /// Quantize one f32 head row (`head_dim` elems) into the packed
            /// cell at byte offset `off`. One `$qfn` call per 32-elem block;
            /// each block writes `blk_bytes` bytes (per-element codes +
            /// the trailing E8M0 scale byte).
            fn quantize_row_into(&mut self, off: usize, src_row: &[f32]) {
                let hd = self.head_dim as usize;
                debug_assert_eq!(src_row.len(), hd);
                let blk_bytes: usize = $blk_bytes;
                let blk_elems: usize = $blk_elems;
                let blocks = hd / blk_elems;
                for b in 0..blocks {
                    $qfn(
                        &src_row[b * blk_elems..(b + 1) * blk_elems],
                        &mut self.data[off + b * blk_bytes..off + (b + 1) * blk_bytes],
                    );
                }
            }

            /// Dequantize one packed cell (`bytes_per_row` bytes) back into
            /// a `head_dim`-long f32 slice. The row-level MX decoder walks
            /// the same per-block stride the encoder wrote, so this is a
            /// single call over the whole row (unlike NVFP4's per-block
            /// `dequantize_block`).
            fn dequantize_row_into(&self, off: usize, dst: &mut [f32]) {
                debug_assert_eq!(dst.len(), self.head_dim as usize);
                $deqrow(&self.data[off..off + self.bytes_per_row], dst);
            }

            pub fn write_token(
                &mut self,
                pages: &[PageId],
                layer: u32,
                pos: u32,
                k: &[f32],
                v: &[f32],
            ) -> Option<()> {
                let hd = self.head_dim as usize;
                let need = (self.n_kv_heads as usize) * hd;
                if k.len() != need || v.len() != need {
                    return None;
                }
                let page_idx = (pos / self.page_size) as usize;
                let pos_in_page = pos % self.page_size;
                let page = *pages.get(page_idx)?;
                for h in 0..self.n_kv_heads {
                    let h_off_src = (h as usize) * hd;
                    let k_off = self.cell_offset(page, layer, 0, h, pos_in_page)?;
                    self.quantize_row_into(k_off, &k[h_off_src..h_off_src + hd]);
                    let v_off = self.cell_offset(page, layer, 1, h, pos_in_page)?;
                    self.quantize_row_into(v_off, &v[h_off_src..h_off_src + hd]);
                }
                Some(())
            }

            pub fn gather_layer(
                &self,
                pages: &[PageId],
                layer: u32,
                kv_len: u32,
                k_out: &mut [f32],
                v_out: &mut [f32],
            ) -> Option<()> {
                if layer >= self.n_layers {
                    return None;
                }
                let hd = self.head_dim as usize;
                let need = (self.n_kv_heads as usize) * (kv_len as usize) * hd;
                if k_out.len() != need || v_out.len() != need {
                    return None;
                }
                let pages_needed = (kv_len as usize).div_ceil(self.page_size as usize);
                if pages.len() < pages_needed {
                    return None;
                }
                let page_size = self.page_size as usize;
                let kv_len_us = kv_len as usize;
                for h in 0..self.n_kv_heads {
                    let out_head_base = (h as usize) * kv_len_us * hd;
                    let mut pos_written = 0usize;
                    for (page_i, &page) in pages.iter().enumerate() {
                        if pos_written >= kv_len_us {
                            break;
                        }
                        let remaining = kv_len_us - pos_written;
                        let pos_in_this_page = remaining.min(page_size);
                        for p in 0..pos_in_this_page {
                            let k_off = self
                                .cell_offset(page, layer, 0, h, p as u32)
                                .expect("page/layer/head/pos in range");
                            let v_off = self
                                .cell_offset(page, layer, 1, h, p as u32)
                                .expect("page/layer/head/pos in range");
                            let dst_pos = out_head_base + (page_i * page_size + p) * hd;
                            self.dequantize_row_into(k_off, &mut k_out[dst_pos..dst_pos + hd]);
                            self.dequantize_row_into(v_off, &mut v_out[dst_pos..dst_pos + hd]);
                        }
                        pos_written += pos_in_this_page;
                    }
                }
                Some(())
            }

            pub fn zero_page(&mut self, page: PageId) -> Option<()> {
                if page.0 >= self.total_pages {
                    return None;
                }
                let start = (page.0 as usize) * self.page_stride;
                let end = start + self.page_stride;
                for v in &mut self.data[start..end] {
                    *v = 0;
                }
                Some(())
            }
        }

        impl PagedKvStoreOps for $name {
            fn write_token(
                &mut self,
                pages: &[PageId],
                layer: u32,
                pos: u32,
                k: &[f32],
                v: &[f32],
            ) -> Option<()> {
                $name::write_token(self, pages, layer, pos, k, v)
            }
            fn gather_layer(
                &self,
                pages: &[PageId],
                layer: u32,
                kv_len: u32,
                k_out: &mut [f32],
                v_out: &mut [f32],
            ) -> Option<()> {
                $name::gather_layer(self, pages, layer, kv_len, k_out, v_out)
            }
            fn zero_page(&mut self, page: PageId) -> Option<()> {
                $name::zero_page(self, page)
            }
            fn page_size(&self) -> u32 {
                $name::page_size(self)
            }
            fn n_layers(&self) -> u32 {
                $name::n_layers(self)
            }
            fn n_kv_heads(&self) -> u32 {
                $name::n_kv_heads(self)
            }
            fn head_dim(&self) -> u32 {
                $name::head_dim(self)
            }
            fn total_pages(&self) -> u32 {
                $name::total_pages(self)
            }
        }
    };
}

paged_kv_store_mxfp!(
    PagedKvStoreMxfp4,
    rustllama_kernels_cpu::mxfp::MXFP4_BLOCK_BYTES,
    rustllama_kernels_cpu::mxfp::MXFP4_BLOCK_ELEMS,
    rustllama_kernels_cpu::mxfp_kv::quantize_block_mxfp4,
    rustllama_kernels_cpu::mxfp_kv::dequantize_row_mxfp4
);
paged_kv_store_mxfp!(
    PagedKvStoreMxfp6,
    rustllama_kernels_cpu::mxfp::MXFP6_BLOCK_BYTES,
    rustllama_kernels_cpu::mxfp::MXFP6_BLOCK_ELEMS,
    rustllama_kernels_cpu::mxfp_kv::quantize_block_mxfp6,
    rustllama_kernels_cpu::mxfp_kv::dequantize_row_mxfp6
);
paged_kv_store_mxfp!(
    PagedKvStoreMxfp8,
    rustllama_kernels_cpu::mxfp::MXFP8_BLOCK_BYTES,
    rustllama_kernels_cpu::mxfp::MXFP8_BLOCK_ELEMS,
    rustllama_kernels_cpu::mxfp_kv::quantize_block_mxfp8,
    rustllama_kernels_cpu::mxfp_kv::dequantize_row_mxfp8
);

#[cfg(test)]
mod tests {
    use super::*;

    fn small_store() -> PagedKvStore {
        // 4 pages, 2 layers, 3 kv_heads, page_size 4, head_dim 8.
        // total_elems = 4 * 2 * 2 * 3 * 4 * 8 = 1536 f32 (~6 KiB).
        PagedKvStore::new(4, 2, 3, 4, 8).expect("small store")
    }

    #[test]
    fn new_rejects_zero_dims() {
        assert!(PagedKvStore::new(0, 1, 1, 1, 1).is_none());
        assert!(PagedKvStore::new(1, 0, 1, 1, 1).is_none());
        assert!(PagedKvStore::new(1, 1, 0, 1, 1).is_none());
        assert!(PagedKvStore::new(1, 1, 1, 0, 1).is_none());
        assert!(PagedKvStore::new(1, 1, 1, 1, 0).is_none());
    }

    #[test]
    fn page_stride_matches_axis_product() {
        let s = small_store();
        // 2 (layers) * 2 (K|V) * 3 (heads) * 4 (page_size) * 8 (head_dim) = 384
        assert_eq!(s.page_stride(), 384);
    }

    #[test]
    fn read_write_page_round_trips() {
        let mut s = small_store();
        let stride = s.page_stride(); // 384
        let payload: Vec<f32> = (0..stride).map(|i| i as f32 * 0.5 - 7.0).collect();
        // Write a whole page (LMCache promote) and read it back (demote).
        assert!(s.write_page(PageId(2), &payload));
        assert_eq!(s.read_page(PageId(2)).unwrap(), payload.as_slice());
        // A different page is untouched (still zero).
        assert!(s.read_page(PageId(0)).unwrap().iter().all(|&x| x == 0.0));
        // Wrong-size write and out-of-range page are rejected.
        assert!(!s.write_page(PageId(0), &payload[..10]));
        assert!(s.read_page(PageId(99)).is_none());
        assert!(!s.write_page(PageId(99), &payload));
    }

    /// M6: the paged half store round-trips f32 K/V rows through its
    /// 2-byte cells within the format's precision (f16 and bf16), and
    /// `gather_layer` widens them back to f32 for the attention kernels.
    #[test]
    fn paged_f16_store_write_gather_round_trips() {
        for bf16 in [false, true] {
            // 2 pages, 2 layers, 2 kv_heads, page_size 4, head_dim 8.
            let mut s = PagedKvStoreF16::new(2, 2, 2, 4, 8, bf16).expect("f16 store");
            assert_eq!(s.is_bf16(), bf16);
            let n_heads = s.n_kv_heads() as usize;
            let hd = s.head_dim() as usize;
            let kv_len = 3u32;
            let pages = [PageId(0)];
            let mut k_in = vec![vec![0f32; n_heads * hd]; kv_len as usize];
            let mut v_in = vec![vec![0f32; n_heads * hd]; kv_len as usize];
            for (p, (kr, vr)) in k_in.iter_mut().zip(v_in.iter_mut()).enumerate() {
                for i in 0..n_heads * hd {
                    kr[i] = (p as f32) * 0.5 + i as f32 * 0.013 - 0.7;
                    vr[i] = -(p as f32) * 0.25 - i as f32 * 0.011 + 0.4;
                }
                s.write_token(&pages, /*layer=*/ 1, p as u32, kr, vr)
                    .expect("write token");
            }
            let mut k_out = vec![0f32; n_heads * kv_len as usize * hd];
            let mut v_out = vec![0f32; n_heads * kv_len as usize * hd];
            s.gather_layer(&pages, 1, kv_len, &mut k_out, &mut v_out)
                .expect("gather");
            // Layout: [n_kv_heads, kv_len, head_dim]. Compare vs input,
            // allowing the half format's rounding.
            let tol = if bf16 { 1e-2 } else { 2e-3 };
            for h in 0..n_heads {
                for p in 0..kv_len as usize {
                    for d in 0..hd {
                        let got_k = k_out[(h * kv_len as usize + p) * hd + d];
                        let want_k = k_in[p][h * hd + d];
                        assert!(
                            (got_k - want_k).abs() <= want_k.abs() * tol + tol,
                            "bf16={bf16} K[h{h} p{p} d{d}] {got_k} vs {want_k}"
                        );
                        let got_v = v_out[(h * kv_len as usize + p) * hd + d];
                        let want_v = v_in[p][h * hd + d];
                        assert!(
                            (got_v - want_v).abs() <= want_v.abs() * tol + tol,
                            "bf16={bf16} V[h{h} p{p} d{d}] {got_v} vs {want_v}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cell_offset_rejects_out_of_range_axes() {
        let s = small_store();
        // Valid: (0,0,0,0,0) → 0
        assert_eq!(s.cell_offset(PageId(0), 0, 0, 0, 0), Some(0));
        // Out-of-range checks:
        assert!(s.cell_offset(PageId(4), 0, 0, 0, 0).is_none()); // page
        assert!(s.cell_offset(PageId(0), 2, 0, 0, 0).is_none()); // layer
        assert!(s.cell_offset(PageId(0), 0, 2, 0, 0).is_none()); // kv_axis
        assert!(s.cell_offset(PageId(0), 0, 0, 3, 0).is_none()); // head
        assert!(s.cell_offset(PageId(0), 0, 0, 0, 4).is_none()); // pos_in_page
    }

    #[test]
    fn write_token_round_trips_through_k_row_v_row() {
        let mut s = small_store();
        let pages = [PageId(0), PageId(1)]; // 2 pages = 8 positions
                                            // Write three tokens at positions 0, 1, 4 (last one straddles into page 1).
        let n_heads = s.n_kv_heads() as usize;
        let hd = s.head_dim() as usize;
        let k0: Vec<f32> = (0..n_heads * hd).map(|i| 0.1 + i as f32 * 0.01).collect();
        let v0: Vec<f32> = (0..n_heads * hd).map(|i| -0.1 - i as f32 * 0.01).collect();
        let k4: Vec<f32> = (0..n_heads * hd).map(|i| 1.0 + i as f32 * 0.1).collect();
        let v4: Vec<f32> = (0..n_heads * hd).map(|i| -1.0 - i as f32 * 0.1).collect();
        s.write_token(&pages, /*layer=*/ 1, /*pos=*/ 0, &k0, &v0)
            .expect("write pos 0");
        s.write_token(&pages, /*layer=*/ 1, /*pos=*/ 4, &k4, &v4)
            .expect("write pos 4");
        // Pos 0 → page 0, pos_in_page 0.
        for h in 0..n_heads as u32 {
            let h_us = h as usize;
            let got_k = s.k_row(PageId(0), 1, h, 0).unwrap();
            let got_v = s.v_row(PageId(0), 1, h, 0).unwrap();
            assert_eq!(got_k, &k0[h_us * hd..(h_us + 1) * hd]);
            assert_eq!(got_v, &v0[h_us * hd..(h_us + 1) * hd]);
        }
        // Pos 4 → page 1, pos_in_page 0.
        for h in 0..n_heads as u32 {
            let h_us = h as usize;
            let got_k = s.k_row(PageId(1), 1, h, 0).unwrap();
            let got_v = s.v_row(PageId(1), 1, h, 0).unwrap();
            assert_eq!(got_k, &k4[h_us * hd..(h_us + 1) * hd]);
            assert_eq!(got_v, &v4[h_us * hd..(h_us + 1) * hd]);
        }
    }

    #[test]
    fn write_token_rejects_wrong_input_length() {
        let mut s = small_store();
        let pages = [PageId(0)];
        let bad = vec![0.0; 5]; // need n_kv_heads * head_dim = 24
        assert!(s.write_token(&pages, 0, 0, &bad, &bad).is_none());
    }

    #[test]
    fn write_token_rejects_pos_beyond_page_list() {
        let mut s = small_store();
        let pages = [PageId(0)]; // one page = 4 positions; pos 4 is OOB
        let n = (s.n_kv_heads() * s.head_dim()) as usize;
        let row = vec![0.0; n];
        assert!(s.write_token(&pages, 0, 4, &row, &row).is_none());
    }

    #[test]
    fn gather_layer_reconstructs_contiguous_kv_slab() {
        // Write a deterministic pattern, then gather and assert each
        // gathered position matches what write_token put in.
        let mut s = small_store();
        let pages = [PageId(2), PageId(0)]; // pages out-of-order on purpose
        let n_heads = s.n_kv_heads() as usize;
        let hd = s.head_dim() as usize;
        let kv_len = 7u32; // 1 full page + 3 positions of next
        let layer = 0u32;

        // Write `kv_len` tokens whose k[h,d] = pos*1000 + h*10 + d,
        // v = -k. Distinct per (pos, h, d) so any gather mis-strides
        // would surface as a wrong-value mismatch.
        for pos in 0..kv_len {
            let mut k_row = vec![0f32; n_heads * hd];
            let mut v_row = vec![0f32; n_heads * hd];
            for h in 0..n_heads {
                for d in 0..hd {
                    let val = (pos as f32) * 1000.0 + (h as f32) * 10.0 + d as f32;
                    k_row[h * hd + d] = val;
                    v_row[h * hd + d] = -val;
                }
            }
            s.write_token(&pages, layer, pos, &k_row, &v_row)
                .expect("write");
        }

        let need = n_heads * (kv_len as usize) * hd;
        let mut k_out = vec![0f32; need];
        let mut v_out = vec![0f32; need];
        s.gather_layer(&pages, layer, kv_len, &mut k_out, &mut v_out)
            .expect("gather");

        // Per-head, per-pos: assert gathered cell matches the input
        // formula. The gather slab's layout is [h, pos, d].
        for h in 0..n_heads {
            for pos in 0..kv_len as usize {
                for d in 0..hd {
                    let expected = (pos as f32) * 1000.0 + (h as f32) * 10.0 + d as f32;
                    let off = h * (kv_len as usize) * hd + pos * hd + d;
                    assert!(
                        (k_out[off] - expected).abs() < 1e-6,
                        "k mismatch at h={h} pos={pos} d={d}: got {} want {}",
                        k_out[off],
                        expected,
                    );
                    assert!(
                        (v_out[off] + expected).abs() < 1e-6,
                        "v mismatch at h={h} pos={pos} d={d}: got {} want {}",
                        v_out[off],
                        -expected,
                    );
                }
            }
        }
    }

    #[test]
    fn gather_layer_rejects_short_page_list() {
        let s = small_store();
        let pages = [PageId(0)]; // 1 page = 4 positions; kv_len 5 needs 2
        let n_heads = s.n_kv_heads() as usize;
        let hd = s.head_dim() as usize;
        let need = n_heads * 5 * hd;
        let mut k_out = vec![0f32; need];
        let mut v_out = vec![0f32; need];
        assert!(s
            .gather_layer(&pages, 0, 5, &mut k_out, &mut v_out)
            .is_none());
    }

    #[test]
    fn gather_layer_rejects_wrong_output_length() {
        let s = small_store();
        let pages = [PageId(0)];
        let mut k_out = vec![0f32; 3]; // grossly wrong
        let mut v_out = vec![0f32; 3];
        assert!(s
            .gather_layer(&pages, 0, 1, &mut k_out, &mut v_out)
            .is_none());
    }

    #[test]
    fn gather_layer_handles_partial_last_page() {
        // Write a single token at pos 0, then gather kv_len=1 and
        // verify the gather doesn't copy from page positions 1..4
        // (uninitialized garbage zeros).
        let mut s = small_store();
        let pages = [PageId(3)]; // arbitrary page
        let n_heads = s.n_kv_heads() as usize;
        let hd = s.head_dim() as usize;
        let k_row: Vec<f32> = (1..=n_heads * hd).map(|i| i as f32).collect();
        let v_row: Vec<f32> = (1..=n_heads * hd).map(|i| -(i as f32)).collect();
        s.write_token(&pages, 0, 0, &k_row, &v_row).expect("write");
        let need = n_heads * 1 * hd;
        let mut k_out = vec![999.0; need];
        let mut v_out = vec![999.0; need];
        s.gather_layer(&pages, 0, 1, &mut k_out, &mut v_out)
            .expect("gather");
        for h in 0..n_heads {
            for d in 0..hd {
                let off = h * hd + d;
                let expected = (h * hd + d + 1) as f32;
                assert_eq!(k_out[off], expected, "k[{h},{d}] partial-page");
                assert_eq!(v_out[off], -expected, "v[{h},{d}] partial-page");
            }
        }
    }

    #[test]
    fn zero_page_clears_every_cell_of_one_page() {
        let mut s = small_store();
        // Write non-zero values into page 1, then zero only page 1.
        // Page 0 stays untouched (regression: zero_page must not
        // bleed into adjacent pages).
        let pages = [PageId(0), PageId(1)];
        let n_heads = s.n_kv_heads() as usize;
        let hd = s.head_dim() as usize;
        let row: Vec<f32> = (1..=n_heads * hd).map(|i| i as f32).collect();
        s.write_token(&pages, 0, 0, &row, &row).unwrap();
        s.write_token(&pages, 0, 4, &row, &row).unwrap(); // page 1, pos 0
        s.zero_page(PageId(1)).expect("zero page 1");
        // Page 0's data preserved.
        assert_eq!(s.k_row(PageId(0), 0, 0, 0).unwrap(), &row[..hd]);
        // Page 1's data cleared.
        for h in 0..n_heads as u32 {
            for pos in 0..s.page_size() {
                assert!(s
                    .k_row(PageId(1), 0, h, pos)
                    .unwrap()
                    .iter()
                    .all(|&v| v == 0.0));
                assert!(s
                    .v_row(PageId(1), 0, h, pos)
                    .unwrap()
                    .iter()
                    .all(|&v| v == 0.0));
            }
        }
    }

    #[test]
    fn zero_page_rejects_out_of_range_id() {
        let mut s = small_store();
        assert!(s.zero_page(PageId(4)).is_none());
    }
}
