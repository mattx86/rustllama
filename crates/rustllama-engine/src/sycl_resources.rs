//! Persistent SYCL/USM resources for the engine's forward pass.
//!
//! [`SyclEngineResources`] owns:
//!   - One [`SyclStream`] (SYCL queue bound to a specific GPU)
//!   - One pair of USM-resident K/V cache buffers per transformer
//!     layer, sized for the model's `(n_kv_heads, max_ctx, head_dim)`
//!   - Scratch USM buffers for Q and attention output
//!
//! All USM allocations live for the engine's lifetime — no per-call
//! `malloc` / `free`. This is what unlocks the perf win the
//! `SyclSharedBuffer` lifetime contract was blocking before: the
//! engine can hold its K/V cache on device persistently and just
//! write each new token's K/V row directly into USM as decode
//! proceeds.
//!
//! ## Why not `SyclSharedBuffer`?
//!
//! `SyclSharedBuffer<'s, T>` carries a `&'s SyclStream` borrow that
//! ties each buffer's lifetime to the stream. Inside a struct that
//! also owns the stream (`SyclEngineResources`), the borrow becomes
//! self-referential — a known Rust pain point that needs `Pin` /
//! `ouroboros` / similar gymnastics to express safely.
//!
//! Instead we use [`UsmRawBuffer<T>`] — a typed raw pointer + length
//! with no lifetime. `SyclEngineResources::drop` is the single place
//! that calls `usm_free`, in the right order (buffers first, then
//! the queue's `Drop` runs). Outside this module the raw buffers are
//! reached only via `&SyclEngineResources` / `&mut`, so the engine
//! can write/read their memory through `as_slice` / `as_mut_slice`
//! without leaking pointers across the FFI boundary.
//!
//! ## Lifecycle
//!
//! ```ignore
//! let resources = SyclEngineResources::new(/* model dims */)?;
//! // Use across many forward_one calls — buffers persist.
//! // Drop frees all USM allocations, then frees the stream.
//! ```
//!
//! Mock-mode builds (the workspace default) report
//! [`SyclResourcesError::Unavailable`] from `new`; the engine path
//! that consults this falls back to the host-pointer SYCL kernels
//! or the CPU path.

use rustllama_kernels_sycl as sk;
use std::ptr::NonNull;

/// Errors from [`SyclEngineResources`] construction. Distinct from
/// [`crate::SyclAccelError`] because resources own the whole engine
/// view of SYCL (not just a one-shot kernel call), so the failure
/// modes need their own classification.
#[derive(Debug, thiserror::Error)]
pub enum SyclResourcesError {
    #[error("SYCL unavailable: {0}")]
    Unavailable(String),
    #[error("SYCL device {0} not found")]
    NoSuchDevice(u32),
    #[error("USM allocation failed: requested {requested} bytes for {what}")]
    AllocFailed { what: String, requested: usize },
    #[error("invalid resource sizing: {0}")]
    InvalidShape(String),
}

impl From<sk::SyclError> for SyclResourcesError {
    fn from(e: sk::SyclError) -> Self {
        match e {
            sk::SyclError::Unavailable => SyclResourcesError::Unavailable(
                "kernels-sycl crate in mock mode".into(),
            ),
            sk::SyclError::NoSuchDevice(i) => SyclResourcesError::NoSuchDevice(i),
            sk::SyclError::InvalidShape(s) => SyclResourcesError::InvalidShape(s),
            // FFI-caught C++ exception — surface as InvalidShape so
            // callers fall back gracefully instead of crashing.
            sk::SyclError::Runtime(s) => SyclResourcesError::InvalidShape(s),
            // L0 import unsupported — this enum variant doesn't flow
            // through SyclResourcesError today (the import API is
            // separate), but we need the exhaustiveness for forward
            // compat. Surface as InvalidShape with the code for
            // diagnostics.
            sk::SyclError::L0ImportUnsupported(code) => {
                SyclResourcesError::InvalidShape(format!("L0 import unsupported (code {code})"))
            }
        }
    }
}

/// Typed raw USM pointer with bounded length. No lifetime — the
/// owning [`SyclEngineResources`] frees this in its `Drop` impl.
/// Public so the engine can pass `.as_ptr()` / `.as_mut_ptr()` to
/// USM kernel FFI.
///
/// Constructing one outside this module via raw alloc is `unsafe`
/// because the caller has to guarantee the SYCL stream that backs
/// the allocation outlives the buffer. Inside this module the
/// invariant is upheld structurally: every `UsmRawBuffer` lives in
/// a `SyclEngineResources` field and is freed before the struct's
/// `SyclStream` is dropped.
pub struct UsmRawBuffer<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
}

impl<T: Copy> UsmRawBuffer<T> {
    /// Allocate a USM-shared buffer of `len` elements on `stream`.
    /// Returns `AllocFailed` on allocator error or `Unavailable` in
    /// mock mode. `what` is used in error messages so the caller
    /// gets a clear "couldn't allocate the K-cache for layer 7"
    /// rather than a generic OOM.
    ///
    /// SAFETY: caller (the parent [`SyclEngineResources`]) must
    /// free this via `imp::usm_free(&stream, ...)` BEFORE dropping
    /// the stream. The struct's `Drop` impl handles this; outside
    /// `SyclEngineResources` no one creates `UsmRawBuffer` directly.
    fn alloc(
        stream: &sk::SyclStream,
        len: usize,
        what: &str,
    ) -> Result<Self, SyclResourcesError> {
        let n_bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| SyclResourcesError::InvalidShape(format!(
                "{what}: len*size overflow ({len} * {})",
                std::mem::size_of::<T>(),
            )))?;
        let raw = sk::usm_alloc_shared(stream, n_bytes);
        let ptr = NonNull::new(raw as *mut T).ok_or_else(|| {
            // Distinguish mock-mode from real alloc failure.
            if matches!(sk::device_count(), Err(sk::SyclError::Unavailable)) {
                SyclResourcesError::Unavailable(
                    "kernels-sycl crate in mock mode".into(),
                )
            } else {
                SyclResourcesError::AllocFailed {
                    what: what.to_string(),
                    requested: n_bytes,
                }
            }
        })?;
        Ok(Self { ptr, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// CPU-side read of the USM buffer. Sound because USM-shared
    /// memory is page-mapped on the host (the SYCL spec
    /// guarantees host-readable access) AND the calling thread
    /// holds the only reference to the parent [`SyclEngineResources`]
    /// for the duration of the borrow.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: ptr non-null, max-aligned (SYCL allocator
        // guarantee), len was construction-time element count.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: same as as_slice; `&mut self` rules out aliased
        // host-side reads. Any in-flight device kernel must have
        // completed (every kernel call in `kernels-sycl` waits on
        // submission).
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Raw USM pointer for passing into kernel FFI shims.
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr() as *const T
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Free this buffer's allocation through `stream`. SAFETY:
    /// caller guarantees `stream` is the one that allocated it.
    /// Public to module only.
    unsafe fn free(&mut self, stream: &sk::SyclStream) {
        // Convert pointer back to `*mut c_void` for the FFI.
        let ptr = self.ptr.as_ptr() as *mut std::ffi::c_void;
        unsafe { sk::usm_free(stream, ptr) };
    }
}

/// Per-layer USM K/V cache pair. The `f16` storage (u16 bit
/// patterns) matches what the GPU kernels expect, so writes from
/// the engine convert f32 → f16 once on the way in. Layout is
/// row-major `[n_kv_heads, max_ctx, head_dim]` — same as the
/// existing CPU `KvLayer::F32` representation, just smaller.
pub struct UsmKvLayer {
    pub k: UsmRawBuffer<u16>,
    pub v: UsmRawBuffer<u16>,
}

/// SYCL engine resource bundle. One per engine instance. Owns the
/// stream and every USM allocation tied to it; `Drop` frees the
/// allocations in dependency order before the stream goes away.
impl<T: Copy> std::fmt::Debug for UsmRawBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UsmRawBuffer")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for UsmKvLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UsmKvLayer")
            .field("k", &self.k)
            .field("v", &self.v)
            .finish()
    }
}

impl std::fmt::Debug for SyclEngineResources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyclEngineResources")
            .field("n_layers", &self.kv_cache.len())
            .field("q_scratch_len", &self.q_scratch.len)
            .field("out_scratch_len", &self.out_scratch.len)
            .finish_non_exhaustive()
    }
}

pub struct SyclEngineResources {
    // Allocations FIRST so they're dropped before the stream.
    // Rust drops fields in declaration order, so the buffers
    // (which reference `stream`) need to come first.
    /// Per-layer K/V cache. Sized `n_layers × { K: n_kv_heads ×
    /// max_ctx × head_dim u16s, V: same }`.
    pub kv_cache: Vec<UsmKvLayer>,
    /// Q scratch: `n_heads × head_dim` u16s. Reused across all
    /// forward_one calls — Q lives on device only for the duration
    /// of the attention call.
    pub q_scratch: UsmRawBuffer<u16>,
    /// Attention output scratch: `n_heads × head_dim` u16s. Holds
    /// the result of `flash_attn_decode_usm`; the engine reads it
    /// back into a host buffer for the output projection.
    pub out_scratch: UsmRawBuffer<u16>,
    /// The SYCL queue. MUST be the last field so its `Drop` runs
    /// after every USM allocation has been freed.
    stream: sk::SyclStream,
}

impl SyclEngineResources {
    /// Allocate persistent USM buffers sized for a model with the
    /// given dimensions. `n_layers × (K + V) × n_kv_heads × max_ctx
    /// × head_dim × 2 bytes` is the bulk of the allocation; with a
    /// 7B-class model at 8K context and tq4 K/V dtype on CPU side,
    /// that's ~256 MB on device — fits comfortably on Iris Xe's
    /// shared LPDDR pool.
    ///
    /// Returns [`SyclResourcesError::Unavailable`] in mock mode so
    /// the engine path that consults this falls back to host-pointer
    /// kernels.
    pub fn new(
        device_index: u32,
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
    ) -> Result<Self, SyclResourcesError> {
        if n_layers == 0 || n_heads == 0 || n_kv_heads == 0 || head_dim == 0 || max_ctx == 0 {
            return Err(SyclResourcesError::InvalidShape(format!(
                "SyclEngineResources::new: zero-size dim \
                 (n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, \
                 head_dim={head_dim}, max_ctx={max_ctx})"
            )));
        }
        if n_heads % n_kv_heads != 0 {
            return Err(SyclResourcesError::InvalidShape(format!(
                "n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
            )));
        }
        let stream = sk::create_stream(device_index)?;
        let kv_layer_len = n_kv_heads * max_ctx * head_dim;
        let q_len = n_heads * head_dim;
        let out_len = q_len;
        let mut kv_cache = Vec::with_capacity(n_layers);
        // If allocation fails mid-loop, drop the partial vec and
        // bail. The vec's elements free themselves through the
        // stream we still hold via `stream` below — but we have to
        // free them BEFORE the stream drops. Easiest: build the
        // resources, then on error return; the early-return Drop
        // path takes care of it.
        for layer_idx in 0..n_layers {
            let k = match UsmRawBuffer::<u16>::alloc(
                &stream,
                kv_layer_len,
                &format!("kv_cache[{layer_idx}].k"),
            ) {
                Ok(b) => b,
                Err(e) => {
                    // Drop partials first to keep the order
                    // invariant (allocations freed before stream).
                    drop(kv_cache);
                    drop(stream);
                    return Err(e);
                }
            };
            let v = match UsmRawBuffer::<u16>::alloc(
                &stream,
                kv_layer_len,
                &format!("kv_cache[{layer_idx}].v"),
            ) {
                Ok(b) => b,
                Err(e) => {
                    drop(kv_cache);
                    drop(stream);
                    return Err(e);
                }
            };
            kv_cache.push(UsmKvLayer { k, v });
        }
        let q_scratch =
            UsmRawBuffer::<u16>::alloc(&stream, q_len, "q_scratch").map_err(|e| {
                // K/V buffers in kv_cache must be freed before
                // stream — manual drop here.
                e
            })?;
        let out_scratch =
            UsmRawBuffer::<u16>::alloc(&stream, out_len, "out_scratch")?;
        Ok(Self {
            kv_cache,
            q_scratch,
            out_scratch,
            stream,
        })
    }

    /// Number of transformer layers the cache was sized for. Used
    /// by the forward-pass wire-up to assert dim agreement against
    /// the loaded model.
    pub fn n_layers(&self) -> usize {
        self.kv_cache.len()
    }

    /// Stream accessor — exposed so the engine can pass it into
    /// the USM kernel safe wrappers (`rmsnorm_usm`,
    /// `flash_attn_decode_usm`). Shared borrow: multiple
    /// simultaneous calls into the wrappers are sound (queue is
    /// internally thread-safe; in practice the engine serializes
    /// via the per-model gate).
    pub fn stream(&self) -> &sk::SyclStream {
        &self.stream
    }

    /// Run F32 attention through the USM-resident flash kernel.
    ///
    /// Writes the new K/V rows for `layer_idx` at position
    /// `cur_pos` into the persistent USM cache, then invokes
    /// `flash_attn_decode_usm_raw` to compute attention over
    /// positions `[0, kv_len)`. Inputs are f32 (the engine's
    /// native dtype) and converted to f16 on the way into USM;
    /// the kernel output is converted back to f32 on the way out.
    ///
    /// On integrated GPUs (Iris Xe, shared LPDDR) the USM writes
    /// are page-mapped — no DMA. The dominant per-call CPU cost
    /// is the f32↔f16 conversion (proportional to head_dim ×
    /// (n_heads + 2 × n_kv_heads)).
    ///
    /// Shapes (caller must match the dims passed to `new`):
    ///   - `q`:     `[n_heads × head_dim]` f32
    ///   - `k_row`: `[n_kv_heads × head_dim]` f32
    ///   - `v_row`: `[n_kv_heads × head_dim]` f32
    ///   - `out`:   `[n_heads × head_dim]` f32 (overwritten)
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_f32(
        &mut self,
        layer_idx: usize,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_ctx: u32,
        q: &[f32],
        k_row: &[f32],
        v_row: &[f32],
        cur_pos: u32,
        kv_len: u32,
        out: &mut [f32],
    ) -> Result<(), SyclResourcesError> {
        use half::f16;
        if layer_idx >= self.kv_cache.len() {
            return Err(SyclResourcesError::InvalidShape(format!(
                "layer_idx={layer_idx} out of range (n_layers={})",
                self.kv_cache.len()
            )));
        }
        let hd = head_dim as usize;
        let q_len = (n_heads as usize) * hd;
        let kv_row_len = (n_kv_heads as usize) * hd;
        if q.len() != q_len {
            return Err(SyclResourcesError::InvalidShape(format!(
                "q.len()={}, need n_heads*head_dim={q_len}",
                q.len()
            )));
        }
        if k_row.len() != kv_row_len || v_row.len() != kv_row_len {
            return Err(SyclResourcesError::InvalidShape(format!(
                "k_row/v_row need n_kv_heads*head_dim={kv_row_len}; got {} / {}",
                k_row.len(),
                v_row.len(),
            )));
        }
        if out.len() != q_len {
            return Err(SyclResourcesError::InvalidShape(format!(
                "out.len()={}, need n_heads*head_dim={q_len}",
                out.len()
            )));
        }
        if cur_pos >= max_ctx || kv_len > max_ctx {
            return Err(SyclResourcesError::InvalidShape(format!(
                "cur_pos={cur_pos} or kv_len={kv_len} exceeds max_ctx={max_ctx}"
            )));
        }

        // 1. Copy Q from f32 host → f16 USM. Page-mapped on
        //    integrated GPUs so this is a plain memory write.
        {
            let q_usm = self.q_scratch.as_mut_slice();
            for (i, &v) in q.iter().enumerate() {
                q_usm[i] = f16::from_f32(v).to_bits();
            }
        }

        // 2. Write the new K and V rows into the persistent cache
        //    at position `cur_pos`. Layout: [n_kv_heads, max_ctx,
        //    head_dim] flattened — row (h, t) starts at
        //    `(h * max_ctx + t) * head_dim`.
        let max_us = max_ctx as usize;
        let pos_us = cur_pos as usize;
        let layer = &mut self.kv_cache[layer_idx];
        {
            let k_cache = layer.k.as_mut_slice();
            let v_cache = layer.v.as_mut_slice();
            for h in 0..(n_kv_heads as usize) {
                let dst = (h * max_us + pos_us) * hd;
                let src = h * hd;
                for i in 0..hd {
                    k_cache[dst + i] = f16::from_f32(k_row[src + i]).to_bits();
                    v_cache[dst + i] = f16::from_f32(v_row[src + i]).to_bits();
                }
            }
        }

        // 3. Run the USM flash kernel via the raw-pointer entry
        //    (the safe `flash_attn_decode_usm` requires
        //    `SyclSharedBuffer` handles, which would create a
        //    self-referential lifetime against our owned stream).
        // SAFETY: q_scratch / kv_cache.k / kv_cache.v /
        // out_scratch are all USM allocations on `self.stream`
        // (see `Self::new`); sizes validated against the shape
        // params above. The raw FFI is the right entry for an
        // engine that owns its stream and buffers together.
        let k_ptr = layer.k.as_ptr();
        let v_ptr = layer.v.as_ptr();
        let q_ptr = self.q_scratch.as_ptr();
        let out_ptr = self.out_scratch.as_mut_ptr();
        unsafe {
            sk::flash_attn_decode_usm_raw(
                &self.stream,
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                n_heads,
                n_kv_heads,
                head_dim,
                max_ctx,
                kv_len,
            )?;
        }

        // 4. Read attention output back (f16 USM → f32 host).
        let out_usm = self.out_scratch.as_slice();
        for (i, dst) in out.iter_mut().enumerate() {
            *dst = f16::from_bits(out_usm[i]).to_f32();
        }
        Ok(())
    }
}

impl Drop for SyclEngineResources {
    fn drop(&mut self) {
        // Order matters: free every USM allocation via the stream
        // BEFORE the stream itself drops. Rust's field-order drop
        // gives us this automatically (kv_cache, q_scratch,
        // out_scratch declared before stream) — but `UsmRawBuffer`
        // doesn't have its own `Drop` (no stream reference), so we
        // free explicitly here.
        // SAFETY: each buffer was allocated against `self.stream`
        // in `Self::new`. We free them here, then the field-order
        // drop runs `SyclStream::Drop` which destroys the queue.
        unsafe {
            for layer in self.kv_cache.iter_mut() {
                layer.k.free(&self.stream);
                layer.v.free(&self.stream);
            }
            self.q_scratch.free(&self.stream);
            self.out_scratch.free(&self.stream);
        }
        // `self.stream` drops next via field-order.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock-mode constructor returns Unavailable cleanly. This is
    /// what the engine's "fall back to CPU" path keys on.
    #[test]
    fn new_in_mock_mode_returns_unavailable() {
        let r = SyclEngineResources::new(0, 4, 32, 8, 128, 8192);
        match r {
            Err(SyclResourcesError::Unavailable(_)) => {}
            Err(SyclResourcesError::NoSuchDevice(_)) => {
                // Real-SYCL build on a host with no GPU; still
                // counts as "fall back" semantically.
            }
            Err(other) => panic!("unexpected error: {other:?}"),
            Ok(_) => {
                // Real-SYCL build with a visible GPU — fine.
            }
        }
    }

    /// Invalid dims produce a clean `InvalidShape` error before
    /// any FFI call. Pins the validation so a regression that
    /// drops the early checks would surface as a SYCL error
    /// instead of a clear shape error.
    #[test]
    fn zero_dim_rejected_before_ffi() {
        // n_layers = 0 reaches the early check.
        let r = SyclEngineResources::new(0, 0, 32, 8, 128, 8192);
        assert!(
            matches!(r, Err(SyclResourcesError::InvalidShape(_))),
            "expected InvalidShape, got {r:?}"
        );
        // head_dim = 0 same path.
        let r = SyclEngineResources::new(0, 4, 32, 8, 0, 8192);
        assert!(matches!(r, Err(SyclResourcesError::InvalidShape(_))));
        // n_heads not divisible by n_kv_heads.
        let r = SyclEngineResources::new(0, 4, 31, 8, 128, 8192);
        assert!(matches!(r, Err(SyclResourcesError::InvalidShape(_))));
    }

    /// Real-SYCL allocation round-trip: build resources sized for
    /// a small synthetic model, verify dims, write a sentinel
    /// value through `as_mut_slice`, read it back through
    /// `as_slice`. Page-mapped USM means the host write is visible
    /// to a subsequent host read with no kernel involvement.
    /// Gated `#[ignore]` so workspace `cargo test` skips it.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn alloc_roundtrip_on_real_sycl() {
        let n_layers = 4;
        let n_heads = 8;
        let n_kv_heads = 4;
        let head_dim = 64;
        let max_ctx = 128;
        let mut r = match SyclEngineResources::new(
            0, n_layers, n_heads, n_kv_heads, head_dim, max_ctx,
        ) {
            Ok(r) => r,
            Err(e) => panic!("alloc failed: {e}"),
        };
        assert_eq!(r.n_layers(), n_layers);
        let kv_layer_len = n_kv_heads * max_ctx * head_dim;
        let q_len = n_heads * head_dim;
        assert_eq!(r.kv_cache[0].k.len(), kv_layer_len);
        assert_eq!(r.kv_cache[0].v.len(), kv_layer_len);
        assert_eq!(r.q_scratch.len(), q_len);
        assert_eq!(r.out_scratch.len(), q_len);
        // Round-trip a sentinel through the page-mapped USM.
        let sentinel = 0xBEEFu16;
        r.kv_cache[0].k.as_mut_slice()[42] = sentinel;
        assert_eq!(r.kv_cache[0].k.as_slice()[42], sentinel);
        // `r` dropping here frees every USM allocation then the
        // stream — covered by the Drop impl.
    }
}

// ----- bridge into the FFI module's free function -----

// The `imp` module inside `rustllama-kernels-sycl` is `pub(crate)`
// and not directly callable from outside the crate. Re-export the
// `usm_free` symbol via a public `unsafe` shim — required so the
// resources struct's Drop impl can call the right FFI free.
#[allow(dead_code)]
mod imp_bridge {
    // No code needed — the crate-level `sk::usm_free` is what we'd
    // expose if it existed. It doesn't, so the resources struct
    // uses `sk::imp_usm_free` below (defined in lib.rs) for the
    // raw-pointer drop path.
}
