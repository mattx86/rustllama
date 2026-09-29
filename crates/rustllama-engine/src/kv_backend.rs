//! `KvBackend` — the engine's choice of KV-cache storage layout.
//!
//! At engine load, [`KvBackend::from_inference_config`] reads the
//! `[inference].kv_cache_layout` config field and builds either:
//!
//! - `Contiguous(KvCache)` — the historical per-request
//!   `[n_kv_heads, max_ctx, head_dim]` slab per layer. Used by
//!   every existing forward path (`forward_one`, prefix cache,
//!   speculative decoding, `fork_for_concurrent_use`).
//! - `Paged { cache, store, table }` — fixed-size pages allocated
//!   from a shared pool. The engine's forward dispatch routes
//!   through the model's `forward_*_paged_f32` mirrors, which
//!   gather paged data back into the slab shape the existing
//!   attention kernels expect. Single-slot per backend instance
//!   today; multi-slot CB lands when the scheduler-driven generate
//!   loop replaces the current per-engine state (item 3.7).
//!
//! What this module does today (3.6d):
//!   - Type definition + constructor + dtype/seq_len/reset surface.
//!   - Validated at engine load: a malformed `kv_cache_layout`
//!     value fails fast with a clear error.
//!   - Paged-incompatible config combos (`kv_dtype = "q8_0"` +
//!     `kv_cache_layout = "paged"`) are detected and rejected here
//!     so the engine never enters a partial state.
//!
//! What lands in 3.6e:
//!   - `EngineState.kv: KvCache` → `EngineState.kv_backend: KvBackend`
//!   - Forward-path dispatch helpers that match on the backend
//!     and call the right `forward_*` / `forward_*_paged_f32`
//!     pair.
//!   - Prefix cache + speculative + fork: skip / refuse cleanly
//!     when backend is `Paged` (V1 trade-off; paged variants of
//!     those features land later).

use rustllama_models::llama_arch::{KvCache, KvDtype};
use rustllama_models::llama_config::LlamaConfig;
use rustllama_models::page_table::PageTable;
use rustllama_models::paged_kv_cache::PagedKvCache;
use rustllama_models::paged_kv_store::PagedKvStore;

/// V1 page size for paged KV. 16 tokens is the vLLM-convention
/// default — small enough to keep fragmentation low for short
/// chats, large enough that the per-page bookkeeping cost is
/// amortized. Exposed as a constant so the engine load + tests
/// agree, and so a future config knob slots in as a one-line
/// override.
pub const DEFAULT_PAGE_SIZE: u32 = 16;

#[derive(Debug, thiserror::Error)]
pub enum KvBackendError {
    #[error("unknown kv_cache_layout: '{0}' (expected \"contiguous\" or \"paged\")")]
    UnknownLayout(String),
    #[error(
        "kv_cache_layout = \"paged\" + kv_dtype = \"{kv_dtype}\" not yet supported. \
         Paged backend now supports F32, Q8_0, TQ, NVFP4 (H9b) and MXFP4/6/8 \
         (Wave 2); this error remains only for the documented Q4_0 non-goal and \
         as a safety net for future unknown dtypes."
    )]
    PagedRequiresF32 { kv_dtype: String },
    #[error("paged KV store alloc failed (geometry would zero out)")]
    PagedStoreShape,
}

/// Engine-level KV-cache storage. One variant per `kv_cache_layout`
/// config value. Held by `EngineState.kv_backend` (after 3.6e
/// wire-up) and consumed by the forward dispatch helpers.
///
/// G2.4: `PagedQ8_0` lifts the historic "paged requires F32"
/// constraint for Q8_0 KV-dtype. The Q8_0 paged store
/// quantizes-on-write and dequantizes-on-gather; existing
/// attention kernels consume the gathered f32 output unchanged.
/// TQ and NVFP4 paged variants need their own `PagedKvStoreTQ` /
/// `PagedKvStoreNvfp4` to land (deferred).
#[derive(Debug)]
pub enum KvBackend {
    Contiguous(KvCache),
    Paged {
        cache: PagedKvCache,
        store: PagedKvStore,
        table: PageTable,
    },
    PagedQ8_0 {
        cache: PagedKvCache,
        store: rustllama_models::paged_kv_store::PagedKvStoreQ8_0,
        table: PageTable,
    },
    /// H9b: TurboQuant paged variant (bits ∈ {1, 2, 4, 8}). The bit
    /// width is fixed at construction (matches KvDtype::Tq's u8).
    PagedTQ {
        cache: PagedKvCache,
        store: rustllama_models::paged_kv_store::PagedKvStoreTQ,
        table: PageTable,
    },
    /// H9b: NVFP4 paged variant. `head_dim` must be a multiple of 16.
    PagedNvfp4 {
        cache: PagedKvCache,
        store: rustllama_models::paged_kv_store::PagedKvStoreNvfp4,
        table: PageTable,
    },
    /// Wave 2: MXFP4 paged variant (OCP Microscaling E2M1 + embedded
    /// E8M0 scale, 17 B / 32-elem block). `head_dim` must be a multiple
    /// of 32.
    PagedMxfp4 {
        cache: PagedKvCache,
        store: rustllama_models::paged_kv_store::PagedKvStoreMxfp4,
        table: PageTable,
    },
    /// Wave 2: MXFP6 paged variant (E3M2 + E8M0, 25 B / 32-elem block).
    PagedMxfp6 {
        cache: PagedKvCache,
        store: rustllama_models::paged_kv_store::PagedKvStoreMxfp6,
        table: PageTable,
    },
    /// Wave 2: MXFP8 paged variant (E4M3 + E8M0, 33 B / 32-elem block).
    PagedMxfp8 {
        cache: PagedKvCache,
        store: rustllama_models::paged_kv_store::PagedKvStoreMxfp8,
        table: PageTable,
    },
}

impl KvBackend {
    /// Bytes of KV buffer storage this backend holds (contiguous
    /// slabs or the paged pool). Feeds the memory-budget planner's
    /// peak projection — measured from the live allocation, so hybrid
    /// models and every dtype report their true footprint.
    pub fn approx_host_bytes(&self) -> u64 {
        match self {
            KvBackend::Contiguous(kv) => kv.approx_bytes() as u64,
            KvBackend::Paged { store, .. } => store.approx_bytes() as u64,
            KvBackend::PagedQ8_0 { store, .. } => store.approx_bytes() as u64,
            KvBackend::PagedTQ { store, .. } => store.approx_bytes() as u64,
            KvBackend::PagedNvfp4 { store, .. } => store.approx_bytes() as u64,
            KvBackend::PagedMxfp4 { store, .. } => store.approx_bytes() as u64,
            KvBackend::PagedMxfp6 { store, .. } => store.approx_bytes() as u64,
            KvBackend::PagedMxfp8 { store, .. } => store.approx_bytes() as u64,
        }
    }

    /// Build a backend from the resolved inference config + model
    /// shape. `ctx_size` is the engine's effective context cap
    /// (post-clamp; see `LlamaConfig::ctx_train` for the model's
    /// native ceiling). `kv_dtype` is the resolved K/V cell dtype.
    /// Both layouts honor all `KvDtype` variants: contiguous quantizes
    /// in-place via `KvCache::new_with_dtype`, and paged selects the
    /// matching `PagedKvStore*` backend (F32 / Q8_0 / TurboQuant /
    /// NVFP4) below.
    ///
    /// `layout` is the raw config string ("contiguous" / "paged");
    /// invalid values surface as [`KvBackendError::UnknownLayout`].
    pub fn from_inference_config(
        layout: &str,
        model_cfg: &LlamaConfig,
        ctx_size: u32,
        kv_dtype: KvDtype,
    ) -> Result<Self, KvBackendError> {
        Self::from_inference_config_with_page_size(
            layout, model_cfg, ctx_size, kv_dtype, DEFAULT_PAGE_SIZE,
        )
    }

    /// Build a backend with a caller-chosen paged-KV page size.
    /// `page_size` is ignored for the `"contiguous"` layout. For
    /// `"paged"` it controls the per-page token count — affects
    /// paging overhead (smaller = more bookkeeping, less waste on
    /// short conversations) vs. SLM-tile efficiency on the GPU
    /// (larger = better attention-kernel utilization). The E2 tuner
    /// sweep (`rustllama tune --kv-page-size`) measures decode tok/s
    /// across a candidate set and persists the winner.
    ///
    /// `page_size == 0` falls back to [`DEFAULT_PAGE_SIZE`] to keep
    /// configs that explicitly set 0 from crashing the store
    /// constructor.
    pub fn from_inference_config_with_page_size(
        layout: &str,
        model_cfg: &LlamaConfig,
        ctx_size: u32,
        kv_dtype: KvDtype,
        page_size: u32,
    ) -> Result<Self, KvBackendError> {
        Self::from_inference_config_with_page_size_and_keep(
            layout, model_cfg, ctx_size, kv_dtype, page_size, None,
        )
    }

    /// Like [`Self::from_inference_config_with_page_size`] plus an
    /// optional per-layer keep-mask for **sparse contiguous KV**
    /// (roadmap: memory reclaim). Hybrid models pass the
    /// full-attention mask so DeltaNet/MTP layers get zero-length
    /// placeholder slabs (~992 MiB reclaimed on qwen35moe-35B).
    /// `None` (dense models) allocates every layer as before. Paged
    /// layouts ignore the mask — hybrid models force contiguous.
    pub fn from_inference_config_with_page_size_and_keep(
        layout: &str,
        model_cfg: &LlamaConfig,
        ctx_size: u32,
        kv_dtype: KvDtype,
        page_size: u32,
        keep: Option<&[bool]>,
    ) -> Result<Self, KvBackendError> {
        let page_size = if page_size == 0 { DEFAULT_PAGE_SIZE } else { page_size };
        match layout {
            "contiguous" => {
                let kv = match keep {
                    Some(mask) => KvCache::new_with_dtype_sparse(
                        model_cfg, ctx_size as usize, kv_dtype, mask,
                    ),
                    None => KvCache::new_with_dtype(model_cfg, ctx_size as usize, kv_dtype),
                };
                Ok(KvBackend::Contiguous(kv))
            }
            "paged" => {
                let pages_for_one_request = ctx_size.div_ceil(page_size);
                let table = PageTable::new(pages_for_one_request, page_size);
                match kv_dtype {
                    KvDtype::F32 => {
                        let store = PagedKvStore::new(
                            pages_for_one_request,
                            model_cfg.n_layers as u32,
                            model_cfg.n_kv_heads as u32,
                            page_size,
                            model_cfg.head_dim as u32,
                        )
                        .ok_or(KvBackendError::PagedStoreShape)?;
                        let cache = PagedKvCache::new_for(&store);
                        Ok(KvBackend::Paged { cache, store, table })
                    }
                    KvDtype::Q8_0 => {
                        let store = rustllama_models::paged_kv_store::PagedKvStoreQ8_0::new(
                            pages_for_one_request,
                            model_cfg.n_layers as u32,
                            model_cfg.n_kv_heads as u32,
                            page_size,
                            model_cfg.head_dim as u32,
                        )
                        .ok_or(KvBackendError::PagedStoreShape)?;
                        let cache = PagedKvCache::new_for_q8_0(&store);
                        Ok(KvBackend::PagedQ8_0 { cache, store, table })
                    }
                    KvDtype::Tq(bits) => {
                        let store = rustllama_models::paged_kv_store::PagedKvStoreTQ::new(
                            pages_for_one_request,
                            model_cfg.n_layers as u32,
                            model_cfg.n_kv_heads as u32,
                            page_size,
                            model_cfg.head_dim as u32,
                            bits,
                        )
                        .ok_or(KvBackendError::PagedStoreShape)?;
                        let cache = PagedKvCache::new_with_page_size(store.page_size());
                        Ok(KvBackend::PagedTQ { cache, store, table })
                    }
                    KvDtype::Nvfp4 => {
                        let store = rustllama_models::paged_kv_store::PagedKvStoreNvfp4::new(
                            pages_for_one_request,
                            model_cfg.n_layers as u32,
                            model_cfg.n_kv_heads as u32,
                            page_size,
                            model_cfg.head_dim as u32,
                        )
                        .ok_or(KvBackendError::PagedStoreShape)?;
                        let cache = PagedKvCache::new_with_page_size(store.page_size());
                        Ok(KvBackend::PagedNvfp4 { cache, store, table })
                    }
                    // Paged Q4_0 is a documented non-goal (v1): the
                    // Q4_0 KV path is contiguous-only, same staging
                    // as TQ/NVFP4 had historically. Explicit error
                    // instead of a silent fallthrough so a config
                    // combining `paged` + `q4_0` fails loudly at load.
                    KvDtype::Q4_0 => Err(KvBackendError::PagedRequiresF32 {
                        kv_dtype: "q4_0".to_string(),
                    }),
                    // MXFP KV paged stores (Wave 2): quantize-on-write /
                    // dequantize-on-gather, same page geometry as NVFP4
                    // but 32-elem blocks (17/25/33 B). `head_dim % 32`
                    // must be 0 or `new` returns None → PagedStoreShape.
                    KvDtype::Mxfp4 => {
                        let store = rustllama_models::paged_kv_store::PagedKvStoreMxfp4::new(
                            pages_for_one_request,
                            model_cfg.n_layers as u32,
                            model_cfg.n_kv_heads as u32,
                            page_size,
                            model_cfg.head_dim as u32,
                        )
                        .ok_or(KvBackendError::PagedStoreShape)?;
                        let cache = PagedKvCache::new_with_page_size(store.page_size());
                        Ok(KvBackend::PagedMxfp4 { cache, store, table })
                    }
                    KvDtype::Mxfp6 => {
                        let store = rustllama_models::paged_kv_store::PagedKvStoreMxfp6::new(
                            pages_for_one_request,
                            model_cfg.n_layers as u32,
                            model_cfg.n_kv_heads as u32,
                            page_size,
                            model_cfg.head_dim as u32,
                        )
                        .ok_or(KvBackendError::PagedStoreShape)?;
                        let cache = PagedKvCache::new_with_page_size(store.page_size());
                        Ok(KvBackend::PagedMxfp6 { cache, store, table })
                    }
                    KvDtype::Mxfp8 => {
                        let store = rustllama_models::paged_kv_store::PagedKvStoreMxfp8::new(
                            pages_for_one_request,
                            model_cfg.n_layers as u32,
                            model_cfg.n_kv_heads as u32,
                            page_size,
                            model_cfg.head_dim as u32,
                        )
                        .ok_or(KvBackendError::PagedStoreShape)?;
                        let cache = PagedKvCache::new_with_page_size(store.page_size());
                        Ok(KvBackend::PagedMxfp8 { cache, store, table })
                    }
                }
            }
            other => Err(KvBackendError::UnknownLayout(other.to_string())),
        }
    }

    /// Reset to empty. For paged, releases all pages back to the
    /// pool; for contiguous, calls `KvCache::reset`.
    pub fn reset(&mut self) {
        match self {
            KvBackend::Contiguous(kv) => kv.reset(),
            KvBackend::Paged { cache, table, .. } => cache.release(table),
            KvBackend::PagedQ8_0 { cache, table, .. } => cache.release(table),
            KvBackend::PagedTQ { cache, table, .. } => cache.release(table),
            KvBackend::PagedNvfp4 { cache, table, .. } => cache.release(table),
            KvBackend::PagedMxfp4 { cache, table, .. } => cache.release(table),
            KvBackend::PagedMxfp6 { cache, table, .. } => cache.release(table),
            KvBackend::PagedMxfp8 { cache, table, .. } => cache.release(table),
        }
    }

    /// Tokens currently in the cache.
    pub fn seq_len(&self) -> usize {
        match self {
            KvBackend::Contiguous(kv) => kv.seq_len,
            KvBackend::Paged { cache, .. }
            | KvBackend::PagedQ8_0 { cache, .. }
            | KvBackend::PagedTQ { cache, .. }
            | KvBackend::PagedNvfp4 { cache, .. }
            | KvBackend::PagedMxfp4 { cache, .. }
            | KvBackend::PagedMxfp6 { cache, .. }
            | KvBackend::PagedMxfp8 { cache, .. } => cache.seq_len() as usize,
        }
    }

    /// Set the `seq_len` watermark. Used by speculative decoding's
    /// rollback (draft sequence rejected; rewind to a prior position
    /// without clearing the cache bytes). For paged, this is a
    /// best-effort: positions past the new watermark are still in
    /// pages but get logically masked out — the next attention call
    /// won't read them because the kernel walks `kv_len` positions
    /// (= `seq_len`). The pages remain allocated (release happens
    /// at request end via `reset`).
    pub fn set_seq_len(&mut self, n: usize) {
        match self {
            KvBackend::Contiguous(kv) => kv.seq_len = n,
            KvBackend::Paged { cache, .. }
            | KvBackend::PagedQ8_0 { cache, .. }
            | KvBackend::PagedTQ { cache, .. }
            | KvBackend::PagedNvfp4 { cache, .. }
            | KvBackend::PagedMxfp4 { cache, .. }
            | KvBackend::PagedMxfp6 { cache, .. }
            | KvBackend::PagedMxfp8 { cache, .. } => cache.set_seq_len_for_rollback(n as u32),
        }
    }

    /// Cell dtype. Reports the actual KvDtype the backend is storing.
    pub fn kv_dtype(&self) -> KvDtype {
        match self {
            KvBackend::Contiguous(kv) => kv.dtype,
            KvBackend::Paged { .. } => KvDtype::F32,
            KvBackend::PagedQ8_0 { .. } => KvDtype::Q8_0,
            KvBackend::PagedTQ { store, .. } => KvDtype::Tq(store.bits()),
            KvBackend::PagedNvfp4 { .. } => KvDtype::Nvfp4,
            KvBackend::PagedMxfp4 { .. } => KvDtype::Mxfp4,
            KvBackend::PagedMxfp6 { .. } => KvDtype::Mxfp6,
            KvBackend::PagedMxfp8 { .. } => KvDtype::Mxfp8,
        }
    }

    pub fn is_paged(&self) -> bool {
        matches!(
            self,
            KvBackend::Paged { .. }
                | KvBackend::PagedQ8_0 { .. }
                | KvBackend::PagedTQ { .. }
                | KvBackend::PagedNvfp4 { .. }
                | KvBackend::PagedMxfp4 { .. }
                | KvBackend::PagedMxfp6 { .. }
                | KvBackend::PagedMxfp8 { .. }
        )
    }

    pub fn is_contiguous(&self) -> bool {
        matches!(self, KvBackend::Contiguous(_))
    }

    /// `(total_pages, free_pages)` for paged backends. Returns
    /// `None` on contiguous so callers can short-circuit
    /// paged-only telemetry surfaces (the GUI Status page hides
    /// the paged-pool panel when this is None).
    pub fn paged_pool_stats(&self) -> Option<(u32, u32)> {
        match self {
            KvBackend::Contiguous(_) => None,
            KvBackend::Paged { table, .. }
            | KvBackend::PagedQ8_0 { table, .. }
            | KvBackend::PagedTQ { table, .. }
            | KvBackend::PagedNvfp4 { table, .. }
            | KvBackend::PagedMxfp4 { table, .. }
            | KvBackend::PagedMxfp6 { table, .. }
            | KvBackend::PagedMxfp8 { table, .. } => {
                Some((table.total(), table.free_count() as u32))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_cfg() -> LlamaConfig {
        LlamaConfig {
            arch: "llama".into(),
            n_layers: 4,
            n_heads: 2,
            n_kv_heads: 2,
            d_model: 16,
            d_ff: 32,
            head_dim: 8,
            rope_dim: 8,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            vocab_size: 256,
            ctx_train: 64,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: false,
            n_mtp_heads: 0,
            moe: None,
            hybrid: None,
            hadamard: None,
        }
    }

    #[test]
    fn from_config_contiguous_builds_kvcache() {
        let cfg = synth_cfg();
        let b = KvBackend::from_inference_config("contiguous", &cfg, 32, KvDtype::F32)
            .expect("build contiguous");
        assert!(b.is_contiguous());
        assert!(!b.is_paged());
        assert_eq!(b.seq_len(), 0);
        assert_eq!(b.kv_dtype(), KvDtype::F32);
    }

    #[test]
    fn from_config_paged_builds_paged_backend() {
        let cfg = synth_cfg();
        let b = KvBackend::from_inference_config("paged", &cfg, 32, KvDtype::F32)
            .expect("build paged");
        assert!(b.is_paged());
        assert!(!b.is_contiguous());
        assert_eq!(b.seq_len(), 0);
        assert_eq!(b.kv_dtype(), KvDtype::F32);
    }

    #[test]
    fn from_config_paged_builds_q8_0_backend() {
        // H9b lifted the historic "paged requires F32" constraint —
        // all four KV dtypes build a paged backend now (the
        // `PagedRequiresF32` error survives only as a safety net for
        // future unknown dtypes). Pin the Q8_0 success path.
        let cfg = synth_cfg();
        let b = KvBackend::from_inference_config("paged", &cfg, 32, KvDtype::Q8_0)
            .expect("Q8_0 paged backend builds (H9b)");
        assert!(b.is_paged());
        assert_eq!(b.kv_dtype(), KvDtype::Q8_0);
    }

    #[test]
    fn from_config_rejects_unknown_layout_with_clear_message() {
        let cfg = synth_cfg();
        let err = KvBackend::from_inference_config("memory-mapped", &cfg, 32, KvDtype::F32)
            .expect_err("unknown layout must fail");
        let msg = err.to_string();
        assert!(msg.contains("memory-mapped"), "msg names the bad value: {msg}");
        assert!(msg.contains("contiguous"), "msg lists the valid options: {msg}");
        assert!(msg.contains("paged"), "msg lists paged option: {msg}");
    }

    #[test]
    fn reset_returns_paged_pages_to_pool() {
        // A fresh paged backend has all pages free; growing the
        // cache + then reset must return them. Catches a leak
        // bug in the reset/release wire-up.
        let cfg = synth_cfg();
        let mut b = KvBackend::from_inference_config("paged", &cfg, 32, KvDtype::F32)
            .expect("paged");
        // Capacity at ctx_size=32, page_size=16 → 2 pages.
        if let KvBackend::Paged { cache, table, .. } = &mut b {
            cache.ensure_capacity(table, 24).expect("alloc 2 pages");
            assert_eq!(table.free_count(), 0, "2 pages used up the 2-page pool");
        } else {
            unreachable!();
        }
        b.reset();
        if let KvBackend::Paged { table, .. } = &b {
            assert_eq!(table.free_count(), 2, "reset returns pages");
        }
    }

    #[test]
    fn reset_on_contiguous_zeros_seq_len() {
        let cfg = synth_cfg();
        let mut b = KvBackend::from_inference_config("contiguous", &cfg, 32, KvDtype::F32)
            .expect("contiguous");
        if let KvBackend::Contiguous(kv) = &mut b {
            kv.seq_len = 5;
        }
        b.reset();
        assert_eq!(b.seq_len(), 0);
    }
}
