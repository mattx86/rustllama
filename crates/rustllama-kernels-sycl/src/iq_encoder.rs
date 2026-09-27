//! GPU SYCL implementation of `rustllama_gguf::iq_gpu::IqGpuEncoder`.
//!
//! Owns:
//!   - one `SyclStream` (thread-bound `sycl::queue`)
//!   - a USM-shared upload of the IQ1_S grid table (16384 f32, ~64 KB),
//!     uploaded once on construction and reused for every batched call
//!   - a USM-shared input staging buffer (resized lazily to fit the
//!     largest `n_chunks * 8` f32 batch seen)
//!   - three USM-shared output buffers for `grid_idx` (u16),
//!     `signed_score` (f32), `norm_sq` (f32), each resized lazily
//!
//! Drop order: output bufs → input buf → grid buf → stream. All
//! USM frees are routed through the parent stream's queue.
//!
//! **Status**: IQ1_S (`iq_8elt_delta_batched`) wired end-to-end.
//! IQ2_* and IQ3_* methods return `Unavailable` pending matching
//! C++ kernels. The CPU fallback in `rustllama_gguf::iq_gpu` is
//! always available; the pipeline can fall through cleanly on the
//! unwired formats.
//!
//! **Hardware validation**: building always compiles the real SYCL
//! kernel (Intel oneAPI 2025.0+ icx/icpx required); executing the GPU
//! path additionally needs an Intel iGPU/dGPU. Mark experimental until
//! the parity gate is hit on hardware.

use std::cell::RefCell;

use rustllama_gguf::iq_gpu::{
    Iq4EltGridFormat, Iq4EltPairedPick, Iq8EltDeltaPick, Iq8EltGridFormat, Iq8EltSignedPick,
    IqGpuEncoder, IqGpuError,
};
use rustllama_gguf::encode_iq_vec::{
    iq1s_grid_f32_table, iq2s_grid_f32_table, iq2s_grid_norm_sq_table, iq2xs_grid_f32_table,
    iq2xs_grid_norm_sq_table, iq2xxs_grid_f32_table, iq2xxs_grid_norm_sq_table,
    iq3s_grid_f32_table, iq3s_grid_norm_sq_table, iq3xxs_grid_f32_table,
    iq3xxs_grid_norm_sq_table, kmask_iq2xs_table, ksigns_iq2xs_reverse_table,
};

use crate::{
    create_stream, iq_search_4elt_paired_signed_raw, iq_search_8elt_delta_iq1s_all3_raw,
    iq_search_8elt_delta_iq1s_all3_w_raw, iq_search_8elt_delta_iq1s_raw, iq_search_8elt_signed_raw,
    usm_alloc_shared, usm_free,
    SyclStream,
};

const IQ1S_GRID_F32_COUNT: usize = 2048 * 8;
const KSIGNS_REV_COUNT: usize = 256;
const KMASK_IQ2XS_COUNT: usize = 8;


/// Per-IQ2/IQ3-format USM resources: grid + norm-sq table,
/// uploaded lazily on first use of that format. The encoder
/// caches both once built since the tables are static per-format
/// constants. IQ2 grids are 8 floats/entry; IQ3 grids are 4
/// floats/entry — same struct, the kernel knows the shape.
#[derive(Default)]
struct Iq2FormatUsm {
    grid_usm: *mut f32,
    grid_norm_usm: *mut f32,
    n_grid: u32,
}

/// Resizable USM buffer state for the encoder. Wrapped in
/// `RefCell` so the `IqGpuEncoder` trait's `&self` methods can
/// grow the pools without requiring `&mut self`.
struct PoolState {
    targets_usm: *mut f32,
    targets_capacity: usize,
    /// Imatrix weights staging buffer ([n_chunks × 8] f32). Lazily
    /// grown on the first weighted all3 call; freed in Drop.
    weights_usm: *mut f32,
    weights_capacity: usize,
    out_grid_idx_usm: *mut u16,
    out_signed_score_usm: *mut f32,
    out_norm_sq_usm: *mut f32,
    out_capacity: usize,
    /// IQ2 extra outputs (sign_idx).
    out_sign_idx_usm: *mut u8,
    /// IQ3 extra output (second grid index per chunk).
    out_grid2_idx_usm: *mut u16,
    /// #3 extra outputs: max-positive picks (parallel arrays sized to
    /// out_capacity_all3). Lazily-grown on the first all3 call.
    out_grid_idx_pos_usm: *mut u16,
    out_signed_score_pos_usm: *mut f32,
    out_norm_sq_pos_usm: *mut f32,
    /// #3 extra outputs: max-negative picks.
    out_grid_idx_neg_usm: *mut u16,
    out_signed_score_neg_usm: *mut f32,
    out_norm_sq_neg_usm: *mut f32,
    out_capacity_all3: usize,
    /// Lazily-uploaded per-format IQ2 USM tables. Allocated on
    /// first call for that format; never freed until Drop.
    iq2xxs: Iq2FormatUsm,
    iq2xs: Iq2FormatUsm,
    iq2s: Iq2FormatUsm,
    /// Same shape for the two IQ3 formats (4-element grid entries).
    iq3xxs: Iq2FormatUsm,
    iq3s: Iq2FormatUsm,
}

/// GPU-backed implementation of `IqGpuEncoder`. Construct via
/// [`SyclIqEncoder::new`]; drop to release all USM buffers + the
/// underlying stream.
///
/// Not `Send + Sync` — the contained `SyclStream` is thread-bound.
pub struct SyclIqEncoder {
    stream: SyclStream,
    /// USM-shared pointer to the IQ1_S grid (`2048 × 8` f32 entries).
    /// Allocated + populated in `new`, freed in `Drop`. Never resized.
    iq1s_grid_usm: *mut f32,
    /// USM-shared pointer to the 256-entry `KSIGNS_IQ2XS` reverse
    /// lookup table (one byte per mask, 0xFF for unmapped). Uploaded
    /// once in `new`; needed by every IQ2/IQ3 batched call.
    ksigns_rev_usm: *mut u8,
    /// USM-shared pointer to the 8-byte `KMASK_IQ2XS` constant.
    /// Needed only by the IQ3 paired-grid kernel (for combining
    /// per-half greedy sign bits into the global 8-bit mask).
    kmask_iq2xs_usm: *mut u8,
    /// Lazily-grown USM staging + output buffers, behind a RefCell
    /// so trait methods (which take `&self`) can resize without
    /// requiring `&mut self`.
    pool: RefCell<PoolState>,
}

impl SyclIqEncoder {
    /// Create a GPU encoder on the given SYCL device (typically 0).
    /// Uploads the IQ1_S grid table to USM up-front.
    pub fn new(device_index: u32) -> Result<Self, IqGpuError> {
        let stream = create_stream(device_index).map_err(|e| {
            IqGpuError::KernelFailed(format!("create_stream({device_index}) failed: {e}"))
        })?;

        let grid_bytes = IQ1S_GRID_F32_COUNT * std::mem::size_of::<f32>();
        let grid_usm = usm_alloc_shared(&stream, grid_bytes) as *mut f32;
        if grid_usm.is_null() {
            return Err(IqGpuError::Unavailable);
        }
        let cpu_grid = iq1s_grid_f32_table();
        debug_assert_eq!(cpu_grid.len(), IQ1S_GRID_F32_COUNT);
        unsafe {
            std::ptr::copy_nonoverlapping(cpu_grid.as_ptr(), grid_usm, IQ1S_GRID_F32_COUNT);
        }

        let ksigns_usm = usm_alloc_shared(&stream, KSIGNS_REV_COUNT) as *mut u8;
        if ksigns_usm.is_null() {
            unsafe { usm_free(&stream, grid_usm as *mut std::ffi::c_void) };
            return Err(IqGpuError::Unavailable);
        }
        let cpu_ksigns = ksigns_iq2xs_reverse_table();
        unsafe {
            std::ptr::copy_nonoverlapping(cpu_ksigns.as_ptr(), ksigns_usm, KSIGNS_REV_COUNT);
        }

        let kmask_usm = usm_alloc_shared(&stream, KMASK_IQ2XS_COUNT) as *mut u8;
        if kmask_usm.is_null() {
            unsafe {
                usm_free(&stream, grid_usm as *mut std::ffi::c_void);
                usm_free(&stream, ksigns_usm as *mut std::ffi::c_void);
            }
            return Err(IqGpuError::Unavailable);
        }
        let cpu_kmask = kmask_iq2xs_table();
        unsafe {
            std::ptr::copy_nonoverlapping(cpu_kmask.as_ptr(), kmask_usm, KMASK_IQ2XS_COUNT);
        }

        Ok(SyclIqEncoder {
            stream,
            iq1s_grid_usm: grid_usm,
            ksigns_rev_usm: ksigns_usm,
            kmask_iq2xs_usm: kmask_usm,
            pool: RefCell::new(PoolState {
                targets_usm: std::ptr::null_mut(),
                targets_capacity: 0,
                weights_usm: std::ptr::null_mut(),
                weights_capacity: 0,
                out_grid_idx_usm: std::ptr::null_mut(),
                out_signed_score_usm: std::ptr::null_mut(),
                out_norm_sq_usm: std::ptr::null_mut(),
                out_capacity: 0,
                out_sign_idx_usm: std::ptr::null_mut(),
                out_grid2_idx_usm: std::ptr::null_mut(),
                out_grid_idx_pos_usm: std::ptr::null_mut(),
                out_signed_score_pos_usm: std::ptr::null_mut(),
                out_norm_sq_pos_usm: std::ptr::null_mut(),
                out_grid_idx_neg_usm: std::ptr::null_mut(),
                out_signed_score_neg_usm: std::ptr::null_mut(),
                out_norm_sq_neg_usm: std::ptr::null_mut(),
                out_capacity_all3: 0,
                iq2xxs: Iq2FormatUsm::default(),
                iq2xs: Iq2FormatUsm::default(),
                iq2s: Iq2FormatUsm::default(),
                iq3xxs: Iq2FormatUsm::default(),
                iq3s: Iq2FormatUsm::default(),
            }),
        })
    }

    /// Lazy upload of an IQ-format's grid + norm-sq tables.
    /// `entries_per_grid` is 8 for IQ2_*, 4 for IQ3_*. Idempotent.
    fn ensure_iq_format(
        &self,
        slot: &mut Iq2FormatUsm,
        grid: &[f32],
        norm_sq: &[f32],
        entries_per_grid: usize,
    ) -> Result<(), IqGpuError> {
        if !slot.grid_usm.is_null() {
            return Ok(());
        }
        debug_assert_eq!(grid.len() % entries_per_grid, 0);
        debug_assert_eq!(norm_sq.len(), grid.len() / entries_per_grid);
        let grid_bytes = grid.len() * std::mem::size_of::<f32>();
        let norm_bytes = norm_sq.len() * std::mem::size_of::<f32>();
        let g = usm_alloc_shared(&self.stream, grid_bytes) as *mut f32;
        let n = usm_alloc_shared(&self.stream, norm_bytes) as *mut f32;
        if g.is_null() || n.is_null() {
            if !g.is_null() {
                unsafe { usm_free(&self.stream, g as *mut std::ffi::c_void) };
            }
            if !n.is_null() {
                unsafe { usm_free(&self.stream, n as *mut std::ffi::c_void) };
            }
            return Err(IqGpuError::Unavailable);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(grid.as_ptr(), g, grid.len());
            std::ptr::copy_nonoverlapping(norm_sq.as_ptr(), n, norm_sq.len());
        }
        slot.grid_usm = g;
        slot.grid_norm_usm = n;
        slot.n_grid = (grid.len() / entries_per_grid) as u32;
        Ok(())
    }

    /// Wrapper for IQ2 callers (8 entries/grid).
    fn ensure_iq2_format(
        &self,
        slot: &mut Iq2FormatUsm,
        grid: &[f32],
        norm_sq: &[f32],
    ) -> Result<(), IqGpuError> {
        self.ensure_iq_format(slot, grid, norm_sq, 8)
    }

    /// Wrapper for IQ3 callers (4 entries/grid).
    fn ensure_iq3_format(
        &self,
        slot: &mut Iq2FormatUsm,
        grid: &[f32],
        norm_sq: &[f32],
    ) -> Result<(), IqGpuError> {
        self.ensure_iq_format(slot, grid, norm_sq, 4)
    }

    fn ensure_grid2_idx_capacity(
        &self,
        pool: &mut PoolState,
        n_chunks: usize,
    ) -> Result<(), IqGpuError> {
        if !pool.out_grid2_idx_usm.is_null() && pool.out_capacity >= n_chunks {
            return Ok(());
        }
        if !pool.out_grid2_idx_usm.is_null() {
            unsafe { usm_free(&self.stream, pool.out_grid2_idx_usm as *mut std::ffi::c_void) };
            pool.out_grid2_idx_usm = std::ptr::null_mut();
        }
        let bytes = n_chunks * std::mem::size_of::<u16>();
        let p = usm_alloc_shared(&self.stream, bytes) as *mut u16;
        if p.is_null() {
            return Err(IqGpuError::Unavailable);
        }
        pool.out_grid2_idx_usm = p;
        Ok(())
    }

    fn ensure_sign_idx_capacity(
        &self,
        pool: &mut PoolState,
        n_chunks: usize,
    ) -> Result<(), IqGpuError> {
        // Sign-idx buffer rides on the same `out_capacity` budget as
        // the other output buffers; ensure_output_capacity bumps the
        // shared float/u16 buffers, and we maintain the u8 sign-idx
        // buffer here in sync.
        if !pool.out_sign_idx_usm.is_null() && pool.out_capacity >= n_chunks {
            return Ok(());
        }
        if !pool.out_sign_idx_usm.is_null() {
            unsafe { usm_free(&self.stream, pool.out_sign_idx_usm as *mut std::ffi::c_void) };
            pool.out_sign_idx_usm = std::ptr::null_mut();
        }
        let bytes = n_chunks * std::mem::size_of::<u8>();
        let p = usm_alloc_shared(&self.stream, bytes) as *mut u8;
        if p.is_null() {
            return Err(IqGpuError::Unavailable);
        }
        pool.out_sign_idx_usm = p;
        Ok(())
    }

    fn ensure_targets_capacity(&self, pool: &mut PoolState, need: usize) -> Result<(), IqGpuError> {
        if pool.targets_capacity >= need {
            return Ok(());
        }
        if !pool.targets_usm.is_null() {
            unsafe { usm_free(&self.stream, pool.targets_usm as *mut std::ffi::c_void) };
        }
        let bytes = need * std::mem::size_of::<f32>();
        let p = usm_alloc_shared(&self.stream, bytes) as *mut f32;
        if p.is_null() {
            pool.targets_usm = std::ptr::null_mut();
            pool.targets_capacity = 0;
            return Err(IqGpuError::Unavailable);
        }
        pool.targets_usm = p;
        pool.targets_capacity = need;
        Ok(())
    }

    fn ensure_weights_capacity(&self, pool: &mut PoolState, need: usize) -> Result<(), IqGpuError> {
        if pool.weights_capacity >= need {
            return Ok(());
        }
        if !pool.weights_usm.is_null() {
            unsafe { usm_free(&self.stream, pool.weights_usm as *mut std::ffi::c_void) };
        }
        let bytes = need * std::mem::size_of::<f32>();
        let p = usm_alloc_shared(&self.stream, bytes) as *mut f32;
        if p.is_null() {
            pool.weights_usm = std::ptr::null_mut();
            pool.weights_capacity = 0;
            return Err(IqGpuError::Unavailable);
        }
        pool.weights_usm = p;
        pool.weights_capacity = need;
        Ok(())
    }

    fn ensure_output_capacity(
        &self,
        pool: &mut PoolState,
        n_chunks: usize,
    ) -> Result<(), IqGpuError> {
        if pool.out_capacity >= n_chunks {
            return Ok(());
        }
        if !pool.out_grid_idx_usm.is_null() {
            unsafe {
                usm_free(&self.stream, pool.out_grid_idx_usm as *mut std::ffi::c_void);
                usm_free(&self.stream, pool.out_signed_score_usm as *mut std::ffi::c_void);
                usm_free(&self.stream, pool.out_norm_sq_usm as *mut std::ffi::c_void);
            }
        }
        let idx_bytes = n_chunks * std::mem::size_of::<u16>();
        let f32_bytes = n_chunks * std::mem::size_of::<f32>();
        let p_idx = usm_alloc_shared(&self.stream, idx_bytes) as *mut u16;
        let p_score = usm_alloc_shared(&self.stream, f32_bytes) as *mut f32;
        let p_norm = usm_alloc_shared(&self.stream, f32_bytes) as *mut f32;
        if p_idx.is_null() || p_score.is_null() || p_norm.is_null() {
            if !p_idx.is_null() {
                unsafe { usm_free(&self.stream, p_idx as *mut std::ffi::c_void) };
            }
            if !p_score.is_null() {
                unsafe { usm_free(&self.stream, p_score as *mut std::ffi::c_void) };
            }
            if !p_norm.is_null() {
                unsafe { usm_free(&self.stream, p_norm as *mut std::ffi::c_void) };
            }
            pool.out_grid_idx_usm = std::ptr::null_mut();
            pool.out_signed_score_usm = std::ptr::null_mut();
            pool.out_norm_sq_usm = std::ptr::null_mut();
            pool.out_capacity = 0;
            return Err(IqGpuError::Unavailable);
        }
        pool.out_grid_idx_usm = p_idx;
        pool.out_signed_score_usm = p_score;
        pool.out_norm_sq_usm = p_norm;
        pool.out_capacity = n_chunks;
        Ok(())
    }

    /// #3: allocate USM for the max-positive and max-negative output
    /// arrays (the max-|score| output reuses the existing
    /// out_grid_idx/score/norm buffers from ensure_output_capacity).
    fn ensure_all3_output_capacity(
        &self,
        pool: &mut PoolState,
        n_chunks: usize,
    ) -> Result<(), IqGpuError> {
        if pool.out_capacity_all3 >= n_chunks {
            return Ok(());
        }
        if !pool.out_grid_idx_pos_usm.is_null() {
            unsafe {
                usm_free(&self.stream, pool.out_grid_idx_pos_usm as *mut std::ffi::c_void);
                usm_free(&self.stream, pool.out_signed_score_pos_usm as *mut std::ffi::c_void);
                usm_free(&self.stream, pool.out_norm_sq_pos_usm as *mut std::ffi::c_void);
                usm_free(&self.stream, pool.out_grid_idx_neg_usm as *mut std::ffi::c_void);
                usm_free(&self.stream, pool.out_signed_score_neg_usm as *mut std::ffi::c_void);
                usm_free(&self.stream, pool.out_norm_sq_neg_usm as *mut std::ffi::c_void);
            }
        }
        let idx_bytes = n_chunks * std::mem::size_of::<u16>();
        let f32_bytes = n_chunks * std::mem::size_of::<f32>();
        let p_idx_p = usm_alloc_shared(&self.stream, idx_bytes) as *mut u16;
        let p_score_p = usm_alloc_shared(&self.stream, f32_bytes) as *mut f32;
        let p_norm_p = usm_alloc_shared(&self.stream, f32_bytes) as *mut f32;
        let p_idx_n = usm_alloc_shared(&self.stream, idx_bytes) as *mut u16;
        let p_score_n = usm_alloc_shared(&self.stream, f32_bytes) as *mut f32;
        let p_norm_n = usm_alloc_shared(&self.stream, f32_bytes) as *mut f32;
        if p_idx_p.is_null() || p_score_p.is_null() || p_norm_p.is_null()
            || p_idx_n.is_null() || p_score_n.is_null() || p_norm_n.is_null()
        {
            // Free any that succeeded.
            for p in [p_idx_p as *mut std::ffi::c_void, p_score_p as *mut std::ffi::c_void,
                      p_norm_p as *mut std::ffi::c_void, p_idx_n as *mut std::ffi::c_void,
                      p_score_n as *mut std::ffi::c_void, p_norm_n as *mut std::ffi::c_void] {
                if !p.is_null() { unsafe { usm_free(&self.stream, p) }; }
            }
            pool.out_grid_idx_pos_usm = std::ptr::null_mut();
            pool.out_signed_score_pos_usm = std::ptr::null_mut();
            pool.out_norm_sq_pos_usm = std::ptr::null_mut();
            pool.out_grid_idx_neg_usm = std::ptr::null_mut();
            pool.out_signed_score_neg_usm = std::ptr::null_mut();
            pool.out_norm_sq_neg_usm = std::ptr::null_mut();
            pool.out_capacity_all3 = 0;
            return Err(IqGpuError::Unavailable);
        }
        pool.out_grid_idx_pos_usm = p_idx_p;
        pool.out_signed_score_pos_usm = p_score_p;
        pool.out_norm_sq_pos_usm = p_norm_p;
        pool.out_grid_idx_neg_usm = p_idx_n;
        pool.out_signed_score_neg_usm = p_score_n;
        pool.out_norm_sq_neg_usm = p_norm_n;
        pool.out_capacity_all3 = n_chunks;
        Ok(())
    }
}

impl IqGpuEncoder for SyclIqEncoder {
    fn iq_8elt_delta_batched(
        &self,
        targets: &[f32],
        delta: f32,
        out: &mut [Iq8EltDeltaPick],
    ) -> Result<(), IqGpuError> {
        let n = out.len();
        let need_targets = n * 8;
        if targets.len() != need_targets {
            return Err(IqGpuError::BadInputSize {
                got: targets.len(),
                expected: need_targets,
                n_chunks: n,
            });
        }
        if n == 0 {
            return Ok(());
        }

        let mut pool = self.pool.borrow_mut();
        self.ensure_targets_capacity(&mut pool, need_targets)?;
        self.ensure_output_capacity(&mut pool, n)?;

        // Stage input. USM-shared is host-writable directly.
        unsafe {
            std::ptr::copy_nonoverlapping(targets.as_ptr(), pool.targets_usm, need_targets);
        }

        // Dispatch. SAFETY: all pointers come from the same SyclStream's
        // USM allocator; sizes match the kernel's declared expectations;
        // kernel waits before returning.
        let rc = unsafe {
            iq_search_8elt_delta_iq1s_raw(
                &self.stream,
                pool.targets_usm,
                delta,
                self.iq1s_grid_usm,
                pool.out_grid_idx_usm,
                pool.out_signed_score_usm,
                pool.out_norm_sq_usm,
                n as u32,
            )
        };
        if let Err(e) = rc {
            return Err(IqGpuError::KernelFailed(format!(
                "iq_search_8elt_delta_iq1s_raw failed: {e}"
            )));
        }

        // Gather outputs. USM-shared so the host pointer reads the
        // kernel's writes after .wait() (which the kernel performs
        // internally before returning).
        for (i, slot) in out.iter_mut().enumerate() {
            unsafe {
                slot.grid_idx = *pool.out_grid_idx_usm.add(i);
                slot.signed_score = *pool.out_signed_score_usm.add(i);
                slot.norm_sq = *pool.out_norm_sq_usm.add(i);
            }
        }
        Ok(())
    }

    fn iq_8elt_delta_batched_all3(
        &self,
        targets: &[f32],
        delta: f32,
        out_abs: &mut [Iq8EltDeltaPick],
        out_pos: &mut [Iq8EltDeltaPick],
        out_neg: &mut [Iq8EltDeltaPick],
    ) -> Result<(), IqGpuError> {
        let n = out_abs.len();
        let need_targets = n * 8;
        if targets.len() != need_targets || out_pos.len() != n || out_neg.len() != n {
            return Err(IqGpuError::BadInputSize {
                got: targets.len(),
                expected: need_targets,
                n_chunks: n,
            });
        }
        if n == 0 { return Ok(()); }
        let mut pool = self.pool.borrow_mut();
        self.ensure_targets_capacity(&mut pool, need_targets)?;
        self.ensure_output_capacity(&mut pool, n)?;
        self.ensure_all3_output_capacity(&mut pool, n)?;
        unsafe {
            std::ptr::copy_nonoverlapping(targets.as_ptr(), pool.targets_usm, need_targets);
        }
        let rc = unsafe {
            iq_search_8elt_delta_iq1s_all3_raw(
                &self.stream,
                pool.targets_usm,
                delta,
                self.iq1s_grid_usm,
                pool.out_grid_idx_usm, pool.out_signed_score_usm, pool.out_norm_sq_usm,
                pool.out_grid_idx_pos_usm, pool.out_signed_score_pos_usm, pool.out_norm_sq_pos_usm,
                pool.out_grid_idx_neg_usm, pool.out_signed_score_neg_usm, pool.out_norm_sq_neg_usm,
                n as u32,
            )
        };
        if let Err(e) = rc {
            return Err(IqGpuError::KernelFailed(format!(
                "iq_search_8elt_delta_iq1s_all3_raw failed: {e}"
            )));
        }
        for i in 0..n {
            unsafe {
                out_abs[i].grid_idx = *pool.out_grid_idx_usm.add(i);
                out_abs[i].signed_score = *pool.out_signed_score_usm.add(i);
                out_abs[i].norm_sq = *pool.out_norm_sq_usm.add(i);
                out_pos[i].grid_idx = *pool.out_grid_idx_pos_usm.add(i);
                out_pos[i].signed_score = *pool.out_signed_score_pos_usm.add(i);
                out_pos[i].norm_sq = *pool.out_norm_sq_pos_usm.add(i);
                out_neg[i].grid_idx = *pool.out_grid_idx_neg_usm.add(i);
                out_neg[i].signed_score = *pool.out_signed_score_neg_usm.add(i);
                out_neg[i].norm_sq = *pool.out_norm_sq_neg_usm.add(i);
            }
        }
        Ok(())
    }

    fn iq_8elt_delta_batched_all3_w(
        &self,
        targets: &[f32],
        weights: &[f32],
        delta: f32,
        out_abs: &mut [Iq8EltDeltaPick],
        out_pos: &mut [Iq8EltDeltaPick],
        out_neg: &mut [Iq8EltDeltaPick],
    ) -> Result<(), IqGpuError> {
        let n = out_abs.len();
        let need_targets = n * 8;
        if targets.len() != need_targets
            || weights.len() != need_targets
            || out_pos.len() != n
            || out_neg.len() != n
        {
            return Err(IqGpuError::BadInputSize {
                got: targets.len(),
                expected: need_targets,
                n_chunks: n,
            });
        }
        if n == 0 { return Ok(()); }
        let mut pool = self.pool.borrow_mut();
        self.ensure_targets_capacity(&mut pool, need_targets)?;
        self.ensure_weights_capacity(&mut pool, need_targets)?;
        self.ensure_output_capacity(&mut pool, n)?;
        self.ensure_all3_output_capacity(&mut pool, n)?;
        unsafe {
            std::ptr::copy_nonoverlapping(targets.as_ptr(), pool.targets_usm, need_targets);
            std::ptr::copy_nonoverlapping(weights.as_ptr(), pool.weights_usm, need_targets);
        }
        let rc = unsafe {
            iq_search_8elt_delta_iq1s_all3_w_raw(
                &self.stream,
                pool.targets_usm,
                pool.weights_usm,
                delta,
                self.iq1s_grid_usm,
                pool.out_grid_idx_usm, pool.out_signed_score_usm, pool.out_norm_sq_usm,
                pool.out_grid_idx_pos_usm, pool.out_signed_score_pos_usm, pool.out_norm_sq_pos_usm,
                pool.out_grid_idx_neg_usm, pool.out_signed_score_neg_usm, pool.out_norm_sq_neg_usm,
                n as u32,
            )
        };
        if let Err(e) = rc {
            return Err(IqGpuError::KernelFailed(format!(
                "iq_search_8elt_delta_iq1s_all3_w_raw failed: {e}"
            )));
        }
        for i in 0..n {
            unsafe {
                out_abs[i].grid_idx = *pool.out_grid_idx_usm.add(i);
                out_abs[i].signed_score = *pool.out_signed_score_usm.add(i);
                out_abs[i].norm_sq = *pool.out_norm_sq_usm.add(i);
                out_pos[i].grid_idx = *pool.out_grid_idx_pos_usm.add(i);
                out_pos[i].signed_score = *pool.out_signed_score_pos_usm.add(i);
                out_pos[i].norm_sq = *pool.out_norm_sq_pos_usm.add(i);
                out_neg[i].grid_idx = *pool.out_grid_idx_neg_usm.add(i);
                out_neg[i].signed_score = *pool.out_signed_score_neg_usm.add(i);
                out_neg[i].norm_sq = *pool.out_norm_sq_neg_usm.add(i);
            }
        }
        Ok(())
    }

    fn iq_8elt_signed_batched(
        &self,
        targets: &[f32],
        format: Iq8EltGridFormat,
        out: &mut [Iq8EltSignedPick],
    ) -> Result<(), IqGpuError> {
        let n = out.len();
        let need_targets = n * 8;
        if targets.len() != need_targets {
            return Err(IqGpuError::BadInputSize {
                got: targets.len(),
                expected: need_targets,
                n_chunks: n,
            });
        }
        if n == 0 {
            return Ok(());
        }
        // IQ1_S routes via the delta-batched path; defensive Unavailable
        // here so callers don't accidentally double-dispatch.
        if matches!(format, Iq8EltGridFormat::Iq1s) {
            return Err(IqGpuError::Unavailable);
        }
        let mut pool = self.pool.borrow_mut();
        self.ensure_targets_capacity(&mut pool, need_targets)?;
        self.ensure_output_capacity(&mut pool, n)?;
        self.ensure_sign_idx_capacity(&mut pool, n)?;

        // Lazy-upload the per-format grid + norm tables. We have to
        // pull the slot out by-value (borrow), get the pointers, then
        // dispatch — borrow checker dance to satisfy "two &mut into
        // pool fields" since the ensure_iq2_format method takes
        // &mut Iq2FormatUsm.
        let (grid_ptr, norm_ptr, n_grid) = match format {
            Iq8EltGridFormat::Iq1s => unreachable!("guarded above"),
            Iq8EltGridFormat::Iq2Xxs => {
                let grid = iq2xxs_grid_f32_table();
                let norm = iq2xxs_grid_norm_sq_table();
                self.ensure_iq2_format(&mut pool.iq2xxs, grid, &norm)?;
                (pool.iq2xxs.grid_usm, pool.iq2xxs.grid_norm_usm, pool.iq2xxs.n_grid)
            }
            Iq8EltGridFormat::Iq2Xs => {
                let grid = iq2xs_grid_f32_table();
                let norm = iq2xs_grid_norm_sq_table();
                self.ensure_iq2_format(&mut pool.iq2xs, grid, &norm)?;
                (pool.iq2xs.grid_usm, pool.iq2xs.grid_norm_usm, pool.iq2xs.n_grid)
            }
            Iq8EltGridFormat::Iq2S => {
                let grid = iq2s_grid_f32_table();
                let norm = iq2s_grid_norm_sq_table();
                self.ensure_iq2_format(&mut pool.iq2s, grid, &norm)?;
                (pool.iq2s.grid_usm, pool.iq2s.grid_norm_usm, pool.iq2s.n_grid)
            }
        };

        // Stage input. USM-shared is host-writable directly.
        unsafe {
            std::ptr::copy_nonoverlapping(targets.as_ptr(), pool.targets_usm, need_targets);
        }

        let rc = unsafe {
            iq_search_8elt_signed_raw(
                &self.stream,
                pool.targets_usm,
                grid_ptr,
                norm_ptr,
                self.ksigns_rev_usm,
                n_grid,
                pool.out_grid_idx_usm,
                pool.out_sign_idx_usm,
                pool.out_signed_score_usm,
                pool.out_norm_sq_usm,
                n as u32,
            )
        };
        if let Err(e) = rc {
            return Err(IqGpuError::KernelFailed(format!(
                "iq_search_8elt_signed_raw failed: {e}"
            )));
        }

        for (i, slot) in out.iter_mut().enumerate() {
            unsafe {
                slot.grid_idx = *pool.out_grid_idx_usm.add(i);
                slot.sign_idx = *pool.out_sign_idx_usm.add(i);
                slot.signed_score = *pool.out_signed_score_usm.add(i);
                slot.grid_norm_sq = *pool.out_norm_sq_usm.add(i);
            }
        }
        Ok(())
    }

    fn iq_4elt_paired_signed_batched(
        &self,
        targets: &[f32],
        format: Iq4EltGridFormat,
        out: &mut [Iq4EltPairedPick],
    ) -> Result<(), IqGpuError> {
        let n = out.len();
        let need_targets = n * 8;
        if targets.len() != need_targets {
            return Err(IqGpuError::BadInputSize {
                got: targets.len(),
                expected: need_targets,
                n_chunks: n,
            });
        }
        if n == 0 {
            return Ok(());
        }
        let mut pool = self.pool.borrow_mut();
        self.ensure_targets_capacity(&mut pool, need_targets)?;
        self.ensure_output_capacity(&mut pool, n)?;
        self.ensure_sign_idx_capacity(&mut pool, n)?;
        self.ensure_grid2_idx_capacity(&mut pool, n)?;

        let (grid_ptr, norm_ptr, n_grid) = match format {
            Iq4EltGridFormat::Iq3Xxs => {
                let grid = iq3xxs_grid_f32_table();
                let norm = iq3xxs_grid_norm_sq_table();
                self.ensure_iq3_format(&mut pool.iq3xxs, grid, &norm)?;
                (pool.iq3xxs.grid_usm, pool.iq3xxs.grid_norm_usm, pool.iq3xxs.n_grid)
            }
            Iq4EltGridFormat::Iq3S => {
                let grid = iq3s_grid_f32_table();
                let norm = iq3s_grid_norm_sq_table();
                self.ensure_iq3_format(&mut pool.iq3s, grid, &norm)?;
                (pool.iq3s.grid_usm, pool.iq3s.grid_norm_usm, pool.iq3s.n_grid)
            }
        };

        unsafe {
            std::ptr::copy_nonoverlapping(targets.as_ptr(), pool.targets_usm, need_targets);
        }

        let rc = unsafe {
            iq_search_4elt_paired_signed_raw(
                &self.stream,
                pool.targets_usm,
                grid_ptr,
                norm_ptr,
                self.kmask_iq2xs_usm,
                self.ksigns_rev_usm,
                n_grid,
                pool.out_grid_idx_usm,
                pool.out_grid2_idx_usm,
                pool.out_sign_idx_usm,
                pool.out_signed_score_usm,
                pool.out_norm_sq_usm,
                n as u32,
            )
        };
        if let Err(e) = rc {
            return Err(IqGpuError::KernelFailed(format!(
                "iq_search_4elt_paired_signed_raw failed: {e}"
            )));
        }

        for (i, slot) in out.iter_mut().enumerate() {
            unsafe {
                slot.grid1_idx = *pool.out_grid_idx_usm.add(i);
                slot.grid2_idx = *pool.out_grid2_idx_usm.add(i);
                slot.sign_idx = *pool.out_sign_idx_usm.add(i);
                slot.signed_score = *pool.out_signed_score_usm.add(i);
                slot.grid_norm_sq = *pool.out_norm_sq_usm.add(i);
            }
        }
        Ok(())
    }

    /// G5: GPU K-quant dequant. Routes Q3_K/Q4_K/Q5_K/Q6_K source
    /// bytes through the matching `rsl_dequant_*_to_f32_usm` SYCL
    /// kernel. Returns `Unavailable` for any other dtype so the
    /// pipeline can fall through to its CPU dequant.
    fn try_dequant_kquant_to_f32(
        &self,
        src_dtype: rustllama_gguf::GgmlType,
        src_bytes: &[u8],
        out_f32: &mut [f32],
    ) -> Result<(), IqGpuError> {
        use rustllama_gguf::GgmlType;
        let format = match src_dtype {
            GgmlType::Q3_K => crate::KQuantFormat::Q3K,
            GgmlType::Q4_K => crate::KQuantFormat::Q4K,
            GgmlType::Q5_K => crate::KQuantFormat::Q5K,
            GgmlType::Q6_K => crate::KQuantFormat::Q6K,
            _ => return Err(IqGpuError::Unavailable),
        };
        crate::dequant_kquant_via_gpu(&self.stream, format, src_bytes, out_f32)
            .map_err(|_e| IqGpuError::Unavailable)
    }

    /// G6: Q6_K block encoder via SYCL. Allocates USM-shared input
    /// and output buffers, runs the kernel, copies bytes back to host.
    /// Falls back to `Unavailable` on any USM alloc / kernel failure
    /// so the caller drops to the CPU `encode_q6_k` path cleanly.
    fn try_encode_q6_k_blocks(
        &self,
        src_f32: &[f32],
        dst_bytes: &mut [u8],
    ) -> Result<(), IqGpuError> {
        use crate::SyclSharedBuffer;
        const QK_K: usize = 256;
        const BLOCK_BYTES: usize = 210;
        if src_f32.len() % QK_K != 0 {
            return Err(IqGpuError::Unavailable);
        }
        let n_blocks = src_f32.len() / QK_K;
        if dst_bytes.len() != n_blocks * BLOCK_BYTES {
            return Err(IqGpuError::Unavailable);
        }
        let mut src_usm = SyclSharedBuffer::<f32>::alloc(&self.stream, src_f32.len())
            .map_err(|_| IqGpuError::Unavailable)?;
        let mut dst_usm = SyclSharedBuffer::<u8>::alloc(&self.stream, dst_bytes.len())
            .map_err(|_| IqGpuError::Unavailable)?;
        src_usm.as_mut_slice().copy_from_slice(src_f32);
        unsafe {
            crate::encode_q6_k_blocks_usm_raw(
                &self.stream,
                src_usm.as_ptr(),
                dst_usm.as_mut_ptr(),
                n_blocks as u32,
            ).map_err(|_| IqGpuError::Unavailable)?;
        }
        dst_bytes.copy_from_slice(dst_usm.as_slice());
        Ok(())
    }

    /// G6: Q3_K block encoder via SYCL. Same shape as Q6_K but 110
    /// bytes per super-block instead of 210.
    fn try_encode_q3_k_blocks(
        &self,
        src_f32: &[f32],
        dst_bytes: &mut [u8],
    ) -> Result<(), IqGpuError> {
        use crate::SyclSharedBuffer;
        const QK_K: usize = 256;
        const BLOCK_BYTES: usize = 110;
        if src_f32.len() % QK_K != 0 {
            return Err(IqGpuError::Unavailable);
        }
        let n_blocks = src_f32.len() / QK_K;
        if dst_bytes.len() != n_blocks * BLOCK_BYTES {
            return Err(IqGpuError::Unavailable);
        }
        let mut src_usm = SyclSharedBuffer::<f32>::alloc(&self.stream, src_f32.len())
            .map_err(|_| IqGpuError::Unavailable)?;
        let mut dst_usm = SyclSharedBuffer::<u8>::alloc(&self.stream, dst_bytes.len())
            .map_err(|_| IqGpuError::Unavailable)?;
        src_usm.as_mut_slice().copy_from_slice(src_f32);
        unsafe {
            crate::encode_q3_k_blocks_usm_raw(
                &self.stream,
                src_usm.as_ptr(),
                dst_usm.as_mut_ptr(),
                n_blocks as u32,
            ).map_err(|_| IqGpuError::Unavailable)?;
        }
        dst_bytes.copy_from_slice(dst_usm.as_slice());
        Ok(())
    }

    /// G6: Q4_K block encoder via SYCL. 144 bytes per super-block.
    fn try_encode_q4_k_blocks(
        &self,
        src_f32: &[f32],
        dst_bytes: &mut [u8],
    ) -> Result<(), IqGpuError> {
        use crate::SyclSharedBuffer;
        const QK_K: usize = 256;
        const BLOCK_BYTES: usize = 144;
        if src_f32.len() % QK_K != 0 {
            return Err(IqGpuError::Unavailable);
        }
        let n_blocks = src_f32.len() / QK_K;
        if dst_bytes.len() != n_blocks * BLOCK_BYTES {
            return Err(IqGpuError::Unavailable);
        }
        let mut src_usm = SyclSharedBuffer::<f32>::alloc(&self.stream, src_f32.len())
            .map_err(|_| IqGpuError::Unavailable)?;
        let mut dst_usm = SyclSharedBuffer::<u8>::alloc(&self.stream, dst_bytes.len())
            .map_err(|_| IqGpuError::Unavailable)?;
        src_usm.as_mut_slice().copy_from_slice(src_f32);
        unsafe {
            crate::encode_q4_k_blocks_usm_raw(
                &self.stream,
                src_usm.as_ptr(),
                dst_usm.as_mut_ptr(),
                n_blocks as u32,
            ).map_err(|_| IqGpuError::Unavailable)?;
        }
        dst_bytes.copy_from_slice(dst_usm.as_slice());
        Ok(())
    }

    /// G6: Q5_K block encoder via SYCL. 176 bytes per super-block.
    fn try_encode_q5_k_blocks(
        &self,
        src_f32: &[f32],
        dst_bytes: &mut [u8],
    ) -> Result<(), IqGpuError> {
        use crate::SyclSharedBuffer;
        const QK_K: usize = 256;
        const BLOCK_BYTES: usize = 176;
        if src_f32.len() % QK_K != 0 {
            return Err(IqGpuError::Unavailable);
        }
        let n_blocks = src_f32.len() / QK_K;
        if dst_bytes.len() != n_blocks * BLOCK_BYTES {
            return Err(IqGpuError::Unavailable);
        }
        let mut src_usm = SyclSharedBuffer::<f32>::alloc(&self.stream, src_f32.len())
            .map_err(|_| IqGpuError::Unavailable)?;
        let mut dst_usm = SyclSharedBuffer::<u8>::alloc(&self.stream, dst_bytes.len())
            .map_err(|_| IqGpuError::Unavailable)?;
        src_usm.as_mut_slice().copy_from_slice(src_f32);
        unsafe {
            crate::encode_q5_k_blocks_usm_raw(
                &self.stream,
                src_usm.as_ptr(),
                dst_usm.as_mut_ptr(),
                n_blocks as u32,
            ).map_err(|_| IqGpuError::Unavailable)?;
        }
        dst_bytes.copy_from_slice(dst_usm.as_slice());
        Ok(())
    }
}

impl Drop for SyclIqEncoder {
    fn drop(&mut self) {
        let pool = self.pool.borrow();
        unsafe {
            if !pool.out_grid_idx_usm.is_null() {
                usm_free(&self.stream, pool.out_grid_idx_usm as *mut std::ffi::c_void);
            }
            if !pool.out_signed_score_usm.is_null() {
                usm_free(&self.stream, pool.out_signed_score_usm as *mut std::ffi::c_void);
            }
            if !pool.out_norm_sq_usm.is_null() {
                usm_free(&self.stream, pool.out_norm_sq_usm as *mut std::ffi::c_void);
            }
            if !pool.out_sign_idx_usm.is_null() {
                usm_free(&self.stream, pool.out_sign_idx_usm as *mut std::ffi::c_void);
            }
            if !pool.out_grid2_idx_usm.is_null() {
                usm_free(&self.stream, pool.out_grid2_idx_usm as *mut std::ffi::c_void);
            }
            for p in [
                pool.out_grid_idx_pos_usm as *mut std::ffi::c_void,
                pool.out_signed_score_pos_usm as *mut std::ffi::c_void,
                pool.out_norm_sq_pos_usm as *mut std::ffi::c_void,
                pool.out_grid_idx_neg_usm as *mut std::ffi::c_void,
                pool.out_signed_score_neg_usm as *mut std::ffi::c_void,
                pool.out_norm_sq_neg_usm as *mut std::ffi::c_void,
            ] {
                if !p.is_null() { usm_free(&self.stream, p); }
            }
            if !pool.targets_usm.is_null() {
                usm_free(&self.stream, pool.targets_usm as *mut std::ffi::c_void);
            }
            if !pool.weights_usm.is_null() {
                usm_free(&self.stream, pool.weights_usm as *mut std::ffi::c_void);
            }
            for slot in [
                &pool.iq2xxs,
                &pool.iq2xs,
                &pool.iq2s,
                &pool.iq3xxs,
                &pool.iq3s,
            ] {
                if !slot.grid_usm.is_null() {
                    usm_free(&self.stream, slot.grid_usm as *mut std::ffi::c_void);
                }
                if !slot.grid_norm_usm.is_null() {
                    usm_free(&self.stream, slot.grid_norm_usm as *mut std::ffi::c_void);
                }
            }
            if !self.kmask_iq2xs_usm.is_null() {
                usm_free(&self.stream, self.kmask_iq2xs_usm as *mut std::ffi::c_void);
            }
            if !self.ksigns_rev_usm.is_null() {
                usm_free(&self.stream, self.ksigns_rev_usm as *mut std::ffi::c_void);
            }
            if !self.iq1s_grid_usm.is_null() {
                usm_free(&self.stream, self.iq1s_grid_usm as *mut std::ffi::c_void);
            }
        }
        // `self.stream` Drop runs after this returns, calling
        // rsl_stream_destroy via its own Drop impl.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustllama_gguf::iq_gpu::{
        CpuFallbackEncoder, Iq4EltPairedPick, Iq8EltDeltaPick, Iq8EltSignedPick,
    };

    /// Deterministic test input — sparse-ish floats, the kind real
    /// activations look like post-norm.
    fn make_targets(n_chunks: usize, seed: u32) -> Vec<f32> {
        let n = n_chunks * 8;
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = (s >> 8) as f32 / (1u32 << 24) as f32;
            out.push(u * 2.0 - 1.0);
        }
        out
    }

    /// Perf bench: time `encode_iq1_s_with_encoder` (the real GPU+CPU
    /// split path used by `rustllama quantize`) against a realistic
    /// tensor size. CPU-only `encode_iq1_s` perf doesn't reflect the
    /// quant pipeline cost because the GPU handles the dominant
    /// max-|score| search.
    ///
    /// Run with: `cargo test --release -p rustllama-kernels-sycl
    ///           sycl_iq_encoder_iq1s_perf -- --nocapture --ignored`
    #[test]
    #[ignore = "perf timing; requires SYCL hardware"]
    fn sycl_iq_encoder_iq1s_perf() {
        use rustllama_gguf::encode_iq_vec::encode_iq1_s_with_encoder;
        use std::time::Instant;
        let encoder = match SyclIqEncoder::new(0) {
            Ok(e) => e,
            Err(e) => { println!("no SYCL device: {:?}", e); return; }
        };
        // 4096 super-blocks = 1 MiB elements — same scale as one
        // matrix of one expert (256 experts × this = full expert
        // weight tensor in production).
        let n_blocks = 4096;
        let n_elements = n_blocks * 256;
        let targets = make_targets(n_blocks * 32, 17);
        // BLOCK_IQ1_S_BYTES = 50 per super-block.
        let mut dst = vec![0u8; n_blocks * 50];
        // Warm up: first call includes GPU kernel JIT / lazy USM alloc.
        encode_iq1_s_with_encoder(&targets, &mut dst, &encoder);
        // Timed.
        let t = Instant::now();
        encode_iq1_s_with_encoder(&targets, &mut dst, &encoder);
        let dt = t.elapsed();
        println!(
            "encode_iq1_s_with_encoder({} blocks = {} elements): {:.2} ms = {:.0} ns/element = {:.0} blocks/sec",
            n_blocks, n_elements, dt.as_secs_f64() * 1000.0,
            dt.as_nanos() as f64 / n_elements as f64,
            n_blocks as f64 / dt.as_secs_f64()
        );
        // Project full v8: 31.5e9 elements / blocks_per_sec * 256 / 60 = minutes.
        let blocks_per_sec = n_blocks as f64 / dt.as_secs_f64();
        let v8_total_blocks = 31.5e9 / 256.0; // ~123M super-blocks
        let est_min = v8_total_blocks / blocks_per_sec / 60.0;
        println!("Projected v8 IQ1_S encode time: {:.1} min", est_min);
    }

    /// F4 hardware-validation: SyclIqEncoder's IQ1_S delta search
    /// must match the CPU reference output within rounding. Verifies
    /// the SPIR-V codegen + USM ABI for the IQ1_S kernel against
    /// real Intel GPU.
    #[test]
    fn sycl_iq_encoder_iq1s_matches_cpu() {
        let encoder = match SyclIqEncoder::new(0) {
            Ok(e) => e,
            Err(_) => return, // No SYCL device — skip.
        };
        let cpu = CpuFallbackEncoder;
        let n_chunks = 64;
        let targets = make_targets(n_chunks, 17);
        let delta = -1.0_f32 + 0.125; // IQ1S delta value
        let mut out_gpu = vec![
            Iq8EltDeltaPick {
                grid_idx: 0,
                signed_score: 0.0,
                norm_sq: 1.0,
            };
            n_chunks
        ];
        let mut out_cpu = out_gpu.clone();
        encoder
            .iq_8elt_delta_batched(&targets, delta, &mut out_gpu)
            .expect("sycl iq1s delta search");
        cpu.iq_8elt_delta_batched(&targets, delta, &mut out_cpu)
            .expect("cpu iq1s delta search");
        for (i, (g, c)) in out_gpu.iter().zip(out_cpu.iter()).enumerate() {
            // grid_idx is integer — must match exactly. signed_score
            // and norm_sq can drift by FP rounding (sum of 8 FMAs).
            assert_eq!(g.grid_idx, c.grid_idx, "iq1s chunk {i}: grid_idx mismatch");
            assert!(
                (g.signed_score - c.signed_score).abs() < 1e-4,
                "iq1s chunk {i}: signed_score gpu={} cpu={}",
                g.signed_score,
                c.signed_score
            );
            assert!(
                (g.norm_sq - c.norm_sq).abs() < 1e-4,
                "iq1s chunk {i}: norm_sq gpu={} cpu={}",
                g.norm_sq,
                c.norm_sq
            );
        }
    }

    /// F4 hardware-validation: IQ2_XXS / IQ2_XS / IQ2_S 8-elt signed
    /// search must match CPU reference.
    ///
    /// F4 hardware-validation: IQ2_XXS / IQ2_XS / IQ2_S 8-elt signed
    /// search must match the **CPU scalar reference** bit-for-bit.
    ///
    /// **Important parity note**: the CPU's `search_chunk_8` function
    /// dispatches to AVX2+FMA on x86, which uses a higher-precision
    /// FMA intermediate and can break Cauchy-Schwarz comparison ties
    /// differently from a pure scalar mul+add sequence. The GPU
    /// kernel uses scalar-equivalent FP ops on the Iris Xe FP unit
    /// and matches `search_chunk_8_scalar` exactly. Both AVX2 and
    /// scalar/GPU paths produce valid quantizations — they only
    /// differ at FP-tie-break boundaries. Hardware diagnosis
    /// confirmed: across multiple seeds, GPU disagreed with the
    /// AVX2 path on 0-1 chunks per 64, and agreed with the scalar
    /// path on ALL chunks.
    #[test]
    fn sycl_iq_encoder_iq2_family_matches_cpu() {
        use rustllama_gguf::iq_gpu::Iq8EltGridFormat;

        let encoder = match SyclIqEncoder::new(0) {
            Ok(e) => e,
            Err(_) => return,
        };
        let n_chunks = 32;
        for format in [
            Iq8EltGridFormat::Iq2Xxs,
            Iq8EltGridFormat::Iq2Xs,
            Iq8EltGridFormat::Iq2S,
        ] {
            let targets = make_targets(n_chunks, 23);
            let mut out_gpu = vec![
                Iq8EltSignedPick {
                    grid_idx: 0,
                    sign_idx: 0,
                    signed_score: -1.0,
                    grid_norm_sq: 1.0,
                };
                n_chunks
            ];
            encoder
                .iq_8elt_signed_batched(&targets, format, &mut out_gpu)
                .expect("sycl iq2 signed search");
            // Reference: CPU scalar path (matches GPU FP semantics).
            // The AVX2+FMA path can disagree at FP-tie boundaries
            // — see docstring above.
            let mut out_cpu_scalar = vec![out_gpu[0]; out_gpu.len()];
            for i in 0..n_chunks {
                out_cpu_scalar[i] =
                    rustllama_gguf::encode_iq_vec::search_chunk_8_scalar_for_format(
                        &targets[i * 8..(i + 1) * 8],
                        format,
                    );
            }
            for (i, (g, c)) in out_gpu.iter().zip(out_cpu_scalar.iter()).enumerate() {
                assert_eq!(
                    g.grid_idx, c.grid_idx,
                    "{format:?} chunk {i}: grid_idx mismatch (gpu={} cpu_scalar={})",
                    g.grid_idx, c.grid_idx
                );
                assert_eq!(
                    g.sign_idx, c.sign_idx,
                    "{format:?} chunk {i}: sign_idx mismatch (gpu={} cpu_scalar={})",
                    g.sign_idx, c.sign_idx
                );
                assert!(
                    (g.signed_score - c.signed_score).abs() < 1e-4,
                    "{format:?} chunk {i}: signed_score gpu={} cpu_scalar={}",
                    g.signed_score,
                    c.signed_score
                );
            }
        }
    }

    /// F4 hardware-validation: IQ3_XXS / IQ3_S 4-elt paired-signed
    /// search must match the **CPU scalar reference**. See the IQ2
    /// test's docstring above for the AVX2-vs-scalar FP-parity
    /// context — same applies here (CPU uses SSE4.1 for the 4-elt
    /// search by default).
    #[test]
    fn sycl_iq_encoder_iq3_family_matches_cpu() {
        use rustllama_gguf::iq_gpu::Iq4EltGridFormat;

        let encoder = match SyclIqEncoder::new(0) {
            Ok(e) => e,
            Err(_) => return,
        };
        let n_chunks = 32;
        let targets = make_targets(n_chunks, 29);
        for format in [Iq4EltGridFormat::Iq3Xxs, Iq4EltGridFormat::Iq3S] {
            let mut out_gpu = vec![
                Iq4EltPairedPick {
                    grid1_idx: 0,
                    grid2_idx: 0,
                    sign_idx: 0,
                    signed_score: -1.0,
                    grid_norm_sq: 1.0,
                };
                n_chunks
            ];
            encoder
                .iq_4elt_paired_signed_batched(&targets, format, &mut out_gpu)
                .expect("sycl iq3 paired-signed search");
            // Reference: CPU scalar path. (The default best_grid_4
            // dispatches to SSE4.1; this scalar path matches the
            // GPU's FP semantics.)
            let mut out_cpu = vec![out_gpu[0]; out_gpu.len()];
            for i in 0..n_chunks {
                out_cpu[i] =
                    rustllama_gguf::encode_iq_vec::search_chunk_iq3_scalar_for_format(
                        &targets[i * 8..(i + 1) * 8],
                        format,
                    );
            }
            for (i, (g, c)) in out_gpu.iter().zip(out_cpu.iter()).enumerate() {
                assert_eq!(
                    g.grid1_idx, c.grid1_idx,
                    "{format:?} chunk {i}: grid1_idx mismatch"
                );
                assert_eq!(
                    g.grid2_idx, c.grid2_idx,
                    "{format:?} chunk {i}: grid2_idx mismatch"
                );
                assert_eq!(
                    g.sign_idx, c.sign_idx,
                    "{format:?} chunk {i}: sign_idx mismatch"
                );
                assert!(
                    (g.signed_score - c.signed_score).abs() < 1e-4,
                    "{format:?} chunk {i}: signed_score gpu={} cpu={}",
                    g.signed_score,
                    c.signed_score
                );
            }
        }
    }
}
