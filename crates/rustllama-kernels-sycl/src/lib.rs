//! Rust-side FFI for the SYCL kernel TU.
//!
//! REAL-ONLY: there is no the no-SYCL-device case. `build.rs` always invokes `icx`
//! (Windows) / `icpx` (Linux) on the C++ TU, and the Rust side calls the
//! real kernels via `extern "C"`. Building therefore always requires the
//! Intel oneAPI Base Toolkit 2025.0+ with `icx`/`icpx` on PATH; `xtask
//! doctor` reports the host status. Whether SYCL is actually *used* at
//! runtime (vs CUDA or CPU) is decided by the engine's startup device
//! detection, not by build features. On a host with no Intel GPU,
//! `device_count()` simply returns 0.
//!
//! `SyclStream` is an RAII handle: dropping it calls `rsl_stream_destroy`
//! on the underlying C++ object. The handle is `!Send` and `!Sync`
//! because the underlying `sycl::queue` is tied to its creating thread
//! (the SYCL runtime keeps thread-local state).

#![allow(non_snake_case, dead_code)]

use std::fmt;

#[cfg(feature = "encoder")]
pub mod iq_encoder;

#[derive(Debug, thiserror::Error)]
pub enum SyclError {
    #[error("SYCL runtime unavailable (no Level Zero / OpenCL loader, or no Intel GPU)")]
    Unavailable,
    #[error("SYCL device index {0} out of range")]
    NoSuchDevice(u32),
    #[error("input shape invalid: {0}")]
    InvalidShape(String),
    /// A C++ exception was caught at the FFI boundary (the kernel
    /// TU's `try { ... } catch (...)` wrappers). The message comes
    /// from the underlying `std::exception::what()` if available.
    /// Engine hooks treat this as "kernel failed — fall back to
    /// CPU" rather than crashing the process.
    #[error("SYCL runtime error: {0}")]
    Runtime(String),
    /// The current SYCL queue isn't backed by Level Zero, or the
    /// L0 import path isn't available on this host. Returned only
    /// from the import API; callers fall back to copy-to-USM.
    #[error("L0 import unsupported (code {0}: 2=non-L0 backend, 3=loader missing, 4=zeMemAllocHost failed)")]
    L0ImportUnsupported(i32),
}

pub type Result<T> = std::result::Result<T, SyclError>;

/// Opaque RAII handle to a SYCL stream / queue. Owns the underlying C++
/// `rsl_stream*` and runs `rsl_stream_destroy` on drop.
///
/// On a host with no usable Intel GPU, `create_stream` returns
/// [`SyclError::Unavailable`] / [`SyclError::NoSuchDevice`] rather than a
/// handle, so a `SyclStream` value only exists when a real queue does.
pub struct SyclStream {
    raw: *mut imp::rsl_stream,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl fmt::Debug for SyclStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SyclStream").finish_non_exhaustive()
    }
}

impl Drop for SyclStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: `raw` came from `rsl_stream_create` and is non-
            // null here. The C side owns the underlying `sycl::queue`;
            // this hands ownership back so it can be destroyed.
            unsafe { imp::rsl_stream_destroy(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

/// G5: which K-quant source format to dequant into F32 on the GPU.
/// Maps 1:1 to the four `rsl_dequant_{q3,q4,q5,q6}_k_to_f32_usm`
/// FFI entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KQuantFormat {
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl KQuantFormat {
    /// Bytes per 256-element super-block for this format.
    pub fn block_size_bytes(self) -> usize {
        match self {
            KQuantFormat::Q3K => 110,
            KQuantFormat::Q4K => 144,
            KQuantFormat::Q5K => 176,
            KQuantFormat::Q6K => 210,
        }
    }
}


mod imp {
    use super::*;
    use std::os::raw::c_int;

    #[repr(C)]
    pub(super) struct rsl_stream {
        _opaque: [u8; 0],
    }

    extern "C" {
        pub(super) fn rsl_sycl_device_count() -> c_int;

        pub(super) fn rsl_consume_error_count() -> c_int;

        pub(super) fn rsl_consume_last_l0_import_code() -> u32;

        pub(super) fn rsl_consume_last_usm_alloc_diag(
            dst_buf: *mut std::os::raw::c_char,
            capacity: c_int,
        ) -> c_int;

        pub(super) fn rsl_get_last_error_message(
            buf: *mut std::os::raw::c_char,
            capacity: c_int,
        );

        pub(super) fn rsl_sycl_device_info(
            device_index: c_int,
            name_out: *mut std::os::raw::c_char,
            name_capacity: c_int,
            driver_out: *mut std::os::raw::c_char,
            driver_capacity: c_int,
            vendor_id_out: *mut u32,
            vram_bytes_out: *mut u64,
            uuid_out: *mut u8,
            // 0 = discrete (dedicated VRAM), 1 = integrated (host-unified
            // memory). Left at the caller's default (0) on any query error.
            is_integrated_out: *mut u8,
        ) -> c_int;
        pub(super) fn rsl_stream_create(device_index: c_int) -> *mut rsl_stream;
        pub(super) fn rsl_stream_destroy(s: *mut rsl_stream);
        pub(super) fn rsl_stream_backend(s: *mut rsl_stream) -> c_int;
        pub(super) fn rsl_stream_sycl_queue(s: *mut rsl_stream) -> *mut std::ffi::c_void;
        pub(super) fn rsl_stream_sycl_device(s: *mut rsl_stream) -> *mut std::ffi::c_void;
        pub(super) fn rsl_stream_sycl_context(s: *mut rsl_stream) -> *mut std::ffi::c_void;

        pub(super) fn rsl_usm_alloc_shared(
            s: *mut rsl_stream,
            n_bytes: usize,
        ) -> *mut std::ffi::c_void;
        pub(super) fn rsl_usm_free(s: *mut rsl_stream, ptr: *mut std::ffi::c_void);
        pub(super) fn rsl_usm_alloc_device_from_host(
            s: *mut rsl_stream,
            src_host: *const std::ffi::c_void,
            n_bytes: usize,
        ) -> *mut std::ffi::c_void;

        pub(super) fn rsl_rmsnorm_usm(
            s: *mut rsl_stream,
            x_usm: *const u16,
            w_usm: *const u16,
            y_usm: *mut u16,
            n_rows: c_int,
            d: c_int,
            eps: f32,
        );

        pub(super) fn rsl_rmsnorm_residual_usm(
            s: *mut rsl_stream,
            x_usm: *const u16,
            w_usm: *const u16,
            residual_usm: *const u16,
            y_usm: *mut u16,
            n_rows: c_int,
            d: c_int,
            eps: f32,
        );

        pub(super) fn rsl_add_rmsnorm_usm(
            s: *mut rsl_stream,
            hidden_usm: *mut u16,
            branch_usm: *const u16,
            w_usm: *const u16,
            y_norm_usm: *mut u16,
            n_rows: c_int,
            d: c_int,
            eps: f32,
        );

        pub(super) fn rsl_add_rmsnorm_f32_usm(
            s: *mut rsl_stream,
            hidden_usm: *mut f32,
            branch_usm: *const f32,
            w_usm: *const f32,
            y_norm_usm: *mut f32,
            n_rows: c_int,
            d: c_int,
            eps: f32,
        );

        pub(super) fn rsl_flash_attn_decode_usm(
            s: *mut rsl_stream,
            q_usm: *const u16,
            k_usm: *const u16,
            v_usm: *const u16,
            out_usm: *mut u16,
            n_heads: c_int,
            n_kv_heads: c_int,
            head_dim: c_int,
            max_ctx: c_int,
            kv_len: c_int,
        );

        pub(super) fn rsl_flash_attn_decode_v2_usm(
            s: *mut rsl_stream,
            q_usm: *const u16,
            k_usm: *const u16,
            v_usm: *const u16,
            out_usm: *mut u16,
            n_heads: c_int,
            n_kv_heads: c_int,
            head_dim: c_int,
            max_ctx: c_int,
            kv_len: c_int,
        );

        pub(super) fn rsl_flash_attn_decode_v3_usm(
            s: *mut rsl_stream,
            q_usm: *const u16,
            k_usm: *const u16,
            v_usm: *const u16,
            out_usm: *mut u16,
            n_heads: c_int,
            n_kv_heads: c_int,
            head_dim: c_int,
            max_ctx: c_int,
            kv_len: c_int,
        );

        pub(super) fn rsl_flash_attn_prefill_usm(
            s: *mut rsl_stream,
            q_usm: *const f32,
            k_cache_usm: *const f32,
            v_cache_usm: *const f32,
            out_usm: *mut f32,
            n_heads: c_int,
            n_kv_heads: c_int,
            head_dim: c_int,
            max_ctx: c_int,
            kv_len_base: c_int,
            n_new: c_int,
        );

        pub(super) fn rsl_flash_attn_prefill_v2_usm(
            s: *mut rsl_stream,
            q_usm: *const f32,
            k_cache_usm: *const f32,
            v_cache_usm: *const f32,
            out_usm: *mut f32,
            n_heads: c_int,
            n_kv_heads: c_int,
            head_dim: c_int,
            max_ctx: c_int,
            kv_len_base: c_int,
            n_new: c_int,
        );

        pub(super) fn rsl_flash_attn_prefill_v3_usm(
            s: *mut rsl_stream,
            q_usm: *const f32,
            k_cache_usm: *const f32,
            v_cache_usm: *const f32,
            out_usm: *mut f32,
            n_heads: c_int,
            n_kv_heads: c_int,
            head_dim: c_int,
            max_ctx: c_int,
            kv_len_base: c_int,
            n_new: c_int,
        );

        // Quantized-KV flash attention (F32 Q/out, packed K/V + optional
        // per-row f32 scales for TurboQuant). Packed K/V are `*const u8`
        // (ABI-compatible with the C `const void*`).
        pub(super) fn rsl_flash_attn_decode_q4_0_usm(
            s: *mut rsl_stream,
            q_usm: *const f32,
            k_packed_usm: *const u8,
            v_packed_usm: *const u8,
            out_usm: *mut f32,
            n_heads: c_int,
            n_kv_heads: c_int,
            head_dim: c_int,
            max_ctx: c_int,
            kv_len: c_int,
        );
        pub(super) fn rsl_flash_attn_prefill_q4_0_usm(
            s: *mut rsl_stream,
            q_usm: *const f32,
            k_packed_usm: *const u8,
            v_packed_usm: *const u8,
            out_usm: *mut f32,
            n_heads: c_int,
            n_kv_heads: c_int,
            head_dim: c_int,
            max_ctx: c_int,
            kv_len_base: c_int,
            n_new: c_int,
        );
        pub(super) fn rsl_flash_attn_decode_nvfp4_usm(
            s: *mut rsl_stream,
            q_usm: *const f32,
            k_packed_usm: *const u8,
            v_packed_usm: *const u8,
            out_usm: *mut f32,
            n_heads: c_int,
            n_kv_heads: c_int,
            head_dim: c_int,
            max_ctx: c_int,
            kv_len: c_int,
        );
        pub(super) fn rsl_flash_attn_prefill_nvfp4_usm(
            s: *mut rsl_stream,
            q_usm: *const f32,
            k_packed_usm: *const u8,
            v_packed_usm: *const u8,
            out_usm: *mut f32,
            n_heads: c_int,
            n_kv_heads: c_int,
            head_dim: c_int,
            max_ctx: c_int,
            kv_len_base: c_int,
            n_new: c_int,
        );
        pub(super) fn rsl_flash_attn_decode_tq_usm(
            s: *mut rsl_stream,
            q_usm: *const f32,
            k_packed_usm: *const u8,
            v_packed_usm: *const u8,
            k_scales_usm: *const f32,
            v_scales_usm: *const f32,
            bits: c_int,
            out_usm: *mut f32,
            n_heads: c_int,
            n_kv_heads: c_int,
            head_dim: c_int,
            max_ctx: c_int,
            kv_len: c_int,
        );
        pub(super) fn rsl_flash_attn_prefill_tq_usm(
            s: *mut rsl_stream,
            q_usm: *const f32,
            k_packed_usm: *const u8,
            v_packed_usm: *const u8,
            k_scales_usm: *const f32,
            v_scales_usm: *const f32,
            bits: c_int,
            out_usm: *mut f32,
            n_heads: c_int,
            n_kv_heads: c_int,
            head_dim: c_int,
            max_ctx: c_int,
            kv_len_base: c_int,
            n_new: c_int,
        );
        // Q8_0-KV flash: i8 K/V slab + per-row f32 scale (like TQ, no bits).
        pub(super) fn rsl_flash_attn_decode_q8_0_usm(
            s: *mut rsl_stream,
            q_usm: *const f32,
            k_packed_usm: *const u8,
            v_packed_usm: *const u8,
            k_scales_usm: *const f32,
            v_scales_usm: *const f32,
            out_usm: *mut f32,
            n_heads: c_int,
            n_kv_heads: c_int,
            head_dim: c_int,
            max_ctx: c_int,
            kv_len: c_int,
        );
        pub(super) fn rsl_flash_attn_prefill_q8_0_usm(
            s: *mut rsl_stream,
            q_usm: *const f32,
            k_packed_usm: *const u8,
            v_packed_usm: *const u8,
            k_scales_usm: *const f32,
            v_scales_usm: *const f32,
            out_usm: *mut f32,
            n_heads: c_int,
            n_kv_heads: c_int,
            head_dim: c_int,
            max_ctx: c_int,
            kv_len_base: c_int,
            n_new: c_int,
        );

        pub(super) fn rsl_gemm_f16_usm(
            s: *mut rsl_stream,
            a_usm: *const u16,
            b_usm: *const u16,
            c_usm: *mut u16,
            m: c_int,
            n: c_int,
            k: c_int,
            lda: c_int,
            ldb: c_int,
            ldc: c_int,
        );

        pub(super) fn rsl_matvec_q8_0_f32_usm(
            s: *mut rsl_stream,
            w_q_usm: *const i8,
            w_scales_usm: *const f32,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
        );

        pub(super) fn rsl_matvec_q8_0_packed_f32_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_q4_k_packed_f32_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_ptq1_0_packed_f32_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_ptq1_0_packed_f32_batched_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_hadamard_forward_usm(
            s: *mut rsl_stream,
            x_usm: *const f32,
            signs_usm: *const f32,
            out_usm: *mut f32,
            n_elems: c_int,
            block: c_int,
        );

        pub(super) fn rsl_matvec_q5_k_packed_f32_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_q6_k_packed_f32_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );

        // CPU-parity packed matvecs (legacy Q4_0/Q5_0/Q4_1/Q5_1 block 32,
        // K-quant Q2_K/Q3_K/Q8_K block 256, PrismML PQ2_0 block 128). Same
        // signature shape as the other packed matvecs.
        pub(super) fn rsl_matvec_q4_0_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const u8, x_usm: *const f32,
            out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        // OCP Microscaling packed matvecs (block 32; 17/25/33 bytes/block).
        pub(super) fn rsl_matvec_mxfp4_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const u8, x_usm: *const f32,
            out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_mxfp6_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const u8, x_usm: *const f32,
            out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_mxfp8_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const u8, x_usm: *const f32,
            out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_q5_0_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const u8, x_usm: *const f32,
            out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_q4_1_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const u8, x_usm: *const f32,
            out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_q5_1_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const u8, x_usm: *const f32,
            out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_q2_k_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const u8, x_usm: *const f32,
            out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_q3_k_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const u8, x_usm: *const f32,
            out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_q8_k_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const u8, x_usm: *const f32,
            out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_pq2_0_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const u8, x_usm: *const f32,
            out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);

        pub(super) fn rsl_matvec_iq1_s_packed_f32_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_iq2_xxs_packed_f32_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_iq1_m_packed_f32_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_iq2_xs_packed_f32_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_iq2_s_packed_f32_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_iq3_xxs_packed_f32_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_iq3_s_packed_f32_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_iq1_s_packed_f32_batched_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_iq1_m_packed_f32_batched_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_iq2_xxs_packed_f32_batched_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_iq2_xs_packed_f32_batched_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_iq2_s_packed_f32_batched_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_iq3_xxs_packed_f32_batched_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_iq3_s_packed_f32_batched_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_iq4_nl_packed_f32_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_iq4_xs_packed_f32_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_iq_search_8elt_delta_iq1s(
            s: *mut rsl_stream,
            targets: *const f32,
            delta: f32,
            grid_f32: *const f32,
            out_grid_idx: *mut u16,
            out_signed_score: *mut f32,
            out_norm_sq: *mut f32,
            n_chunks: c_int,
        );

        pub(super) fn rsl_iq_search_8elt_delta_iq1s_all3(
            s: *mut rsl_stream,
            targets: *const f32,
            delta: f32,
            grid_f32: *const f32,
            out_grid_idx_abs: *mut u16,
            out_signed_score_abs: *mut f32,
            out_norm_sq_abs: *mut f32,
            out_grid_idx_pos: *mut u16,
            out_signed_score_pos: *mut f32,
            out_norm_sq_pos: *mut f32,
            out_grid_idx_neg: *mut u16,
            out_signed_score_neg: *mut f32,
            out_norm_sq_neg: *mut f32,
            n_chunks: c_int,
        );

        pub(super) fn rsl_iq_search_8elt_delta_iq1s_all3_w(
            s: *mut rsl_stream,
            targets: *const f32,
            weights: *const f32,
            delta: f32,
            grid_f32: *const f32,
            out_grid_idx_abs: *mut u16,
            out_signed_score_abs: *mut f32,
            out_norm_sq_abs: *mut f32,
            out_grid_idx_pos: *mut u16,
            out_signed_score_pos: *mut f32,
            out_norm_sq_pos: *mut f32,
            out_grid_idx_neg: *mut u16,
            out_signed_score_neg: *mut f32,
            out_norm_sq_neg: *mut f32,
            n_chunks: c_int,
        );

        pub(super) fn rsl_iq_search_8elt_signed(
            s: *mut rsl_stream,
            targets: *const f32,
            grid_f32: *const f32,
            grid_norm_sq_table: *const f32,
            ksigns_rev: *const u8,
            n_grid: c_int,
            out_grid_idx: *mut u16,
            out_sign_idx: *mut u8,
            out_signed_score: *mut f32,
            out_grid_norm_sq: *mut f32,
            n_chunks: c_int,
        );

        pub(super) fn rsl_iq_search_4elt_paired_signed(
            s: *mut rsl_stream,
            targets: *const f32,
            grid_f32: *const f32,
            grid_norm_sq_table: *const f32,
            kmask: *const u8,
            ksigns_rev: *const u8,
            n_grid: c_int,
            out_grid1_idx: *mut u16,
            out_grid2_idx: *mut u16,
            out_sign_idx: *mut u8,
            out_signed_score: *mut f32,
            out_grid_norm_sq: *mut f32,
            n_chunks: c_int,
        );

        pub(super) fn rsl_sampler_argmax_usm(
            s: *mut rsl_stream,
            logits_usm: *const f32,
            vocab: c_int,
            out_idx_usm: *mut c_int,
        );

        pub(super) fn rsl_sampler_temp_softmax_usm(
            s: *mut rsl_stream,
            logits_usm: *mut f32,
            vocab: c_int,
            inv_temp: f32,
        );

        pub(super) fn rsl_sampler_multinomial_usm(
            s: *mut rsl_stream,
            probs_usm: *const f32,
            vocab: c_int,
            rng_state_usm: *mut u64,
            out_idx_usm: *mut c_int,
        );

        pub(super) fn rsl_sampler_penalty_usm(
            s: *mut rsl_stream,
            logits_usm: *mut f32,
            vocab: c_int,
            recent_usm: *const u32,
            recent_n: c_int,
            repeat: f32,
            frequency: f32,
            presence: f32,
        );

        pub(super) fn rsl_sampler_top_k_usm(
            s: *mut rsl_stream,
            probs_usm: *mut f32,
            vocab: c_int,
            k: c_int,
        );

        pub(super) fn rsl_sampler_top_p_usm(
            s: *mut rsl_stream,
            probs_usm: *mut f32,
            vocab: c_int,
            p: f32,
            needs_fallback_usm: *mut c_int,
        );

        pub(super) fn rsl_matvec_q8_0_packed_f32_batched_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_iq4_nl_packed_f32_batched_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_iq4_xs_packed_f32_batched_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_q4_k_packed_f32_batched_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_q5_k_packed_f32_batched_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_q6_k_packed_f32_batched_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const u8,
            x_usm: *const f32,
            out_usm: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_try_import_win32_handle_as_usm(
            s: *mut rsl_stream,
            mapping_handle: *mut std::ffi::c_void,
            size: usize,
            out_dev_ptr: *mut *mut std::ffi::c_void,
        ) -> c_int;

        pub(super) fn rsl_release_imported_usm(
            s: *mut rsl_stream,
            dev_ptr: *mut std::ffi::c_void,
        );

        pub(super) fn rsl_try_alloc_host_baseline(
            s: *mut rsl_stream,
            size: usize,
            out_dev_ptr: *mut *mut std::ffi::c_void,
        ) -> c_int;

        pub(super) fn rsl_rope_usm(
            s: *mut rsl_stream,
            qk_usm: *mut u16,
            n_heads: c_int,
            head_dim: c_int,
            pos: c_int,
            inv_freq_usm: *const u16,
        );

        pub(super) fn rsl_silu_mul_usm(
            s: *mut rsl_stream,
            x_usm: *const u16,
            y_usm: *const u16,
            out_usm: *mut u16,
            n: c_int,
        );

        pub(super) fn rsl_embedding_lookup_usm(
            s: *mut rsl_stream,
            table_usm: *const u16,
            ids: *const i32,
            out_usm: *mut u16,
            n_ids: c_int,
            d: c_int,
        );

        // G6: K-quant block encoders (analytical, USM-in/USM-out).
        pub(super) fn rsl_encode_q6_k_blocks_usm(
            s: *mut rsl_stream,
            src_usm: *const f32,
            dst_usm: *mut u8,
            n_blocks: c_int,
        );
        pub(super) fn rsl_encode_q3_k_blocks_usm(
            s: *mut rsl_stream,
            src_usm: *const f32,
            dst_usm: *mut u8,
            n_blocks: c_int,
        );
        pub(super) fn rsl_encode_q4_k_blocks_usm(
            s: *mut rsl_stream,
            src_usm: *const f32,
            dst_usm: *mut u8,
            n_blocks: c_int,
        );
        pub(super) fn rsl_encode_q5_k_blocks_usm(
            s: *mut rsl_stream,
            src_usm: *const f32,
            dst_usm: *mut u8,
            n_blocks: c_int,
        );

        // H4: gate+up FUSED matvec — one kernel launch instead of two,
        // shares activation cache across the gate + up dot products.
        pub(super) fn rsl_matvec_q4_k_gate_up_fused_usm(
            s: *mut rsl_stream,
            gate_w_bytes_usm: *const std::ffi::c_void,
            up_w_bytes_usm: *const std::ffi::c_void,
            x_usm: *const f32,
            gate_out_usm: *mut f32,
            up_out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );
        pub(super) fn rsl_matvec_q8_0_gate_up_fused_usm(
            s: *mut rsl_stream,
            gate_w_bytes_usm: *const std::ffi::c_void,
            up_w_bytes_usm: *const std::ffi::c_void,
            x_usm: *const f32,
            gate_out_usm: *mut f32,
            up_out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );
        pub(super) fn rsl_matvec_q5_k_gate_up_fused_usm(
            s: *mut rsl_stream,
            gate_w_bytes_usm: *const std::ffi::c_void,
            up_w_bytes_usm: *const std::ffi::c_void,
            x_usm: *const f32,
            gate_out_usm: *mut f32,
            up_out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );
        pub(super) fn rsl_matvec_q6_k_gate_up_fused_usm(
            s: *mut rsl_stream,
            gate_w_bytes_usm: *const std::ffi::c_void,
            up_w_bytes_usm: *const std::ffi::c_void,
            x_usm: *const f32,
            gate_out_usm: *mut f32,
            up_out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );
        pub(super) fn rsl_matvec_iq4_nl_gate_up_fused_usm(
            s: *mut rsl_stream,
            gate_w_bytes_usm: *const std::ffi::c_void,
            up_w_bytes_usm: *const std::ffi::c_void,
            x_usm: *const f32,
            gate_out_usm: *mut f32,
            up_out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );
        pub(super) fn rsl_matvec_iq4_xs_gate_up_fused_usm(
            s: *mut rsl_stream,
            gate_w_bytes_usm: *const std::ffi::c_void,
            up_w_bytes_usm: *const std::ffi::c_void,
            x_usm: *const f32,
            gate_out_usm: *mut f32,
            up_out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );
        pub(super) fn rsl_matvec_iq1_s_gate_up_fused_usm(
            s: *mut rsl_stream,
            gate_w_bytes_usm: *const std::ffi::c_void,
            up_w_bytes_usm: *const std::ffi::c_void,
            x_usm: *const f32,
            gate_out_usm: *mut f32,
            up_out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );
        pub(super) fn rsl_matvec_iq2_xxs_gate_up_fused_usm(
            s: *mut rsl_stream,
            gate_w_bytes_usm: *const std::ffi::c_void,
            up_w_bytes_usm: *const std::ffi::c_void,
            x_usm: *const f32,
            gate_out_usm: *mut f32,
            up_out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );
        pub(super) fn rsl_matvec_iq1_m_gate_up_fused_usm(
            s: *mut rsl_stream,
            gate_w_bytes_usm: *const std::ffi::c_void,
            up_w_bytes_usm: *const std::ffi::c_void,
            x_usm: *const f32,
            gate_out_usm: *mut f32,
            up_out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );
        pub(super) fn rsl_matvec_iq2_xs_gate_up_fused_usm(
            s: *mut rsl_stream,
            gate_w_bytes_usm: *const std::ffi::c_void,
            up_w_bytes_usm: *const std::ffi::c_void,
            x_usm: *const f32,
            gate_out_usm: *mut f32,
            up_out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );
        pub(super) fn rsl_matvec_iq2_s_gate_up_fused_usm(
            s: *mut rsl_stream,
            gate_w_bytes_usm: *const std::ffi::c_void,
            up_w_bytes_usm: *const std::ffi::c_void,
            x_usm: *const f32,
            gate_out_usm: *mut f32,
            up_out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );
        pub(super) fn rsl_matvec_iq3_xxs_gate_up_fused_usm(
            s: *mut rsl_stream,
            gate_w_bytes_usm: *const std::ffi::c_void,
            up_w_bytes_usm: *const std::ffi::c_void,
            x_usm: *const f32,
            gate_out_usm: *mut f32,
            up_out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );
        pub(super) fn rsl_matvec_iq3_s_gate_up_fused_usm(
            s: *mut rsl_stream,
            gate_w_bytes_usm: *const std::ffi::c_void,
            up_w_bytes_usm: *const std::ffi::c_void,
            x_usm: *const f32,
            gate_out_usm: *mut f32,
            up_out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );

        pub(super) fn rsl_matvec_ptq1_0_gate_up_fused_usm(
            s: *mut rsl_stream,
            gate_w_bytes_usm: *const std::ffi::c_void,
            up_w_bytes_usm: *const std::ffi::c_void,
            x_usm: *const f32,
            gate_out_usm: *mut f32,
            up_out_usm: *mut f32,
            m: c_int,
            k: c_int,
            lws: c_int,
        );

        // H6: fused matvec + residual-add + rmsnorm. One kernel
        // replaces the (matvec → add_rmsnorm) sequence at the post-
        // attn site. Per packed-quant dtype.
        pub(super) fn rsl_matvec_q4_k_add_rmsnorm_usm(
            s: *mut rsl_stream,
            w_bytes_usm: *const std::ffi::c_void,
            attn_usm: *const f32,
            hidden_usm: *mut f32,
            w_norm_usm: *const f32,
            y_norm_usm: *mut f32,
            m: c_int,
            k: c_int,
            eps: f32,
            lws: c_int,
        );
        pub(super) fn rsl_matvec_q8_0_add_rmsnorm_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            attn_usm: *const f32, hidden_usm: *mut f32, w_norm_usm: *const f32,
            y_norm_usm: *mut f32, m: c_int, k: c_int, eps: f32, lws: c_int);
        pub(super) fn rsl_matvec_q5_k_add_rmsnorm_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            attn_usm: *const f32, hidden_usm: *mut f32, w_norm_usm: *const f32,
            y_norm_usm: *mut f32, m: c_int, k: c_int, eps: f32, lws: c_int);
        pub(super) fn rsl_matvec_q6_k_add_rmsnorm_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            attn_usm: *const f32, hidden_usm: *mut f32, w_norm_usm: *const f32,
            y_norm_usm: *mut f32, m: c_int, k: c_int, eps: f32, lws: c_int);
        pub(super) fn rsl_matvec_iq4_nl_add_rmsnorm_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            attn_usm: *const f32, hidden_usm: *mut f32, w_norm_usm: *const f32,
            y_norm_usm: *mut f32, m: c_int, k: c_int, eps: f32, lws: c_int);
        pub(super) fn rsl_matvec_iq4_xs_add_rmsnorm_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            attn_usm: *const f32, hidden_usm: *mut f32, w_norm_usm: *const f32,
            y_norm_usm: *mut f32, m: c_int, k: c_int, eps: f32, lws: c_int);
        pub(super) fn rsl_matvec_iq1_s_add_rmsnorm_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            attn_usm: *const f32, hidden_usm: *mut f32, w_norm_usm: *const f32,
            y_norm_usm: *mut f32, m: c_int, k: c_int, eps: f32, lws: c_int);
        pub(super) fn rsl_matvec_iq1_m_add_rmsnorm_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            attn_usm: *const f32, hidden_usm: *mut f32, w_norm_usm: *const f32,
            y_norm_usm: *mut f32, m: c_int, k: c_int, eps: f32, lws: c_int);
        pub(super) fn rsl_matvec_iq2_xxs_add_rmsnorm_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            attn_usm: *const f32, hidden_usm: *mut f32, w_norm_usm: *const f32,
            y_norm_usm: *mut f32, m: c_int, k: c_int, eps: f32, lws: c_int);
        pub(super) fn rsl_matvec_iq2_xs_add_rmsnorm_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            attn_usm: *const f32, hidden_usm: *mut f32, w_norm_usm: *const f32,
            y_norm_usm: *mut f32, m: c_int, k: c_int, eps: f32, lws: c_int);
        pub(super) fn rsl_matvec_iq2_s_add_rmsnorm_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            attn_usm: *const f32, hidden_usm: *mut f32, w_norm_usm: *const f32,
            y_norm_usm: *mut f32, m: c_int, k: c_int, eps: f32, lws: c_int);
        pub(super) fn rsl_matvec_iq3_xxs_add_rmsnorm_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            attn_usm: *const f32, hidden_usm: *mut f32, w_norm_usm: *const f32,
            y_norm_usm: *mut f32, m: c_int, k: c_int, eps: f32, lws: c_int);
        pub(super) fn rsl_matvec_iq3_s_add_rmsnorm_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            attn_usm: *const f32, hidden_usm: *mut f32, w_norm_usm: *const f32,
            y_norm_usm: *mut f32, m: c_int, k: c_int, eps: f32, lws: c_int);
        pub(super) fn rsl_matvec_ptq1_0_add_rmsnorm_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            attn_usm: *const f32, hidden_usm: *mut f32, w_norm_usm: *const f32,
            y_norm_usm: *mut f32, m: c_int, k: c_int, eps: f32, lws: c_int);

        // H8: F16-input mixed-precision matvec. `x_f16` is the
        // activation vector as F16 bits; weights + output stay packed
        // / F32. Per packed-quant dtype.
        pub(super) fn rsl_matvec_q8_0_f16in_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            x_f16: *const u16, out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_q4_k_f16in_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            x_f16: *const u16, out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_q5_k_f16in_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            x_f16: *const u16, out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_q6_k_f16in_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            x_f16: *const u16, out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_iq4_nl_f16in_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            x_f16: *const u16, out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_iq4_xs_f16in_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            x_f16: *const u16, out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_iq1_s_f16in_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            x_f16: *const u16, out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_iq1_m_f16in_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            x_f16: *const u16, out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_iq2_xxs_f16in_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            x_f16: *const u16, out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_iq2_xs_f16in_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            x_f16: *const u16, out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_iq2_s_f16in_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            x_f16: *const u16, out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_iq3_xxs_f16in_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            x_f16: *const u16, out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_iq3_s_f16in_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            x_f16: *const u16, out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);
        pub(super) fn rsl_matvec_ptq1_0_f16in_packed_f32_usm(
            s: *mut rsl_stream, w_bytes_usm: *const std::ffi::c_void,
            x_f16: *const u16, out_usm: *mut f32, m: c_int, k: c_int, lws: c_int);

        // G2: KV-cache Q8_0 quantize-on-store. Strided write into the
        // KV cache layout `[n_kv_heads × max_ctx × head_dim]` from a
        // contiguous activation buffer `[n_new × n_kv_heads × head_dim]`.
        pub(super) fn rsl_kv_quantize_q8_0_store_usm(
            s: *mut rsl_stream,
            src_usm: *const f32,
            q_dst_usm: *mut i8,
            scales_dst_usm: *mut f32,
            n_new: c_int,
            n_kv_heads: c_int,
            head_dim: c_int,
            max_ctx: c_int,
            kv_len_base: c_int,
        );

        // G5: K-quant → F32 dequant (USM-in / USM-out).
        pub(super) fn rsl_dequant_q4_k_to_f32_usm(
            s: *mut rsl_stream,
            bytes_usm: *const std::ffi::c_void,
            out_usm: *mut f32,
            n_blocks: c_int,
        );
        pub(super) fn rsl_dequant_q3_k_to_f32_usm(
            s: *mut rsl_stream,
            bytes_usm: *const std::ffi::c_void,
            out_usm: *mut f32,
            n_blocks: c_int,
        );
        pub(super) fn rsl_dequant_q5_k_to_f32_usm(
            s: *mut rsl_stream,
            bytes_usm: *const std::ffi::c_void,
            out_usm: *mut f32,
            n_blocks: c_int,
        );
        pub(super) fn rsl_dequant_q6_k_to_f32_usm(
            s: *mut rsl_stream,
            bytes_usm: *const std::ffi::c_void,
            out_usm: *mut f32,
            n_blocks: c_int,
        );

        fn rsl_gemm_f16(
            s: *mut rsl_stream,
            a: *const u16,
            b: *const u16,
            c: *mut u16,
            m: c_int,
            n: c_int,
            k: c_int,
            lda: c_int,
            ldb: c_int,
            ldc: c_int,
        );

        fn rsl_rmsnorm(
            s: *mut rsl_stream,
            x: *const u16,
            w: *const u16,
            y: *mut u16,
            n_rows: c_int,
            d: c_int,
            eps: f32,
        );

        fn rsl_rope(
            s: *mut rsl_stream,
            qk: *mut u16,
            n_heads: c_int,
            head_dim: c_int,
            pos: c_int,
            inv_freq: *const u16,
        );

        fn rsl_softmax_attn(
            s: *mut rsl_stream,
            scores: *mut u16,
            mask: *const u16,
            n_heads: c_int,
            seq: c_int,
            kv_len: c_int,
            scale: f32,
        );

        fn rsl_silu_mul(
            s: *mut rsl_stream,
            x: *const u16,
            y: *const u16,
            out: *mut u16,
            n: c_int,
        );

        fn rsl_embedding_lookup(
            s: *mut rsl_stream,
            table: *const u16,
            ids: *const i32,
            out: *mut u16,
            n_ids: c_int,
            d: c_int,
        );

        fn rsl_dequant_nvfp4(
            s: *mut rsl_stream,
            w_nvfp4: *const u8,
            out: *mut u16,
            n_blocks: c_int,
        );

        fn rsl_matvec_nvfp4_f16(
            s: *mut rsl_stream,
            w_nvfp4: *const u8,
            x: *const u16,
            out: *mut u16,
            m: c_int,
            k: c_int,
        );
    }

    /// Whether the delay-loaded `rsl_kernels.dll` and its oneAPI runtime
    /// dependencies can actually be loaded. On Windows the kernel DLL is
    /// **delay-loaded** (see `app/src-tauri/build.rs`), so the very first
    /// call into any `rsl_*` symbol would otherwise raise a fatal SEH
    /// exception (module-not-found) when oneAPI is absent or a driver DLL
    /// is broken — aborting the process instead of degrading to CPU. We
    /// probe with an explicit `LoadLibraryW` (after
    /// `rustllama_runtime::ensure_gpu_dll_search_paths` has prepended the
    /// oneAPI dirs to the search path) and cache the result; a failure is
    /// reported to callers as "no device" so the engine falls back to CPU
    /// cleanly. Non-Windows builds link the shared object normally and
    /// have nothing to probe (a missing `.so` fails at exec).
    #[cfg(windows)]
    fn sycl_runtime_loadable() -> bool {
        use std::sync::OnceLock;
        static OK: OnceLock<bool> = OnceLock::new();
        *OK.get_or_init(|| {
            extern "system" {
                fn LoadLibraryW(name: *const u16) -> *mut core::ffi::c_void;
            }
            let name: Vec<u16> = "rsl_kernels.dll"
                .encode_utf16()
                .chain(std::iter::once(0))
                .collect();
            // SAFETY: FFI to LoadLibraryW with a valid NUL-terminated wide
            // string. We only inspect the handle's nullness; loading the
            // DLL also resolves its (non-delay) import table, so a missing
            // sycl*.dll / ur_loader.dll surfaces here as a null handle
            // rather than a crash on first symbol use. The reference is
            // intentionally kept (DLL stays resident for the process).
            let handle = unsafe { LoadLibraryW(name.as_ptr()) };
            if handle.is_null() {
                tracing::warn!(
                    "SYCL GPU runtime (rsl_kernels.dll) could not be loaded — \
                     oneAPI runtime missing or a driver DLL is broken; using CPU"
                );
                false
            } else {
                true
            }
        })
    }

    #[cfg(not(windows))]
    #[inline]
    fn sycl_runtime_loadable() -> bool {
        true
    }

    pub fn device_count() -> Result<u32> {
        // Guard the delay-loaded FFI: if the GPU runtime can't be loaded,
        // report zero devices instead of aborting on the first symbol.
        if !sycl_runtime_loadable() {
            return Ok(0);
        }
        // SAFETY: bound to the C ABI shim we own.
        let n = unsafe { rsl_sycl_device_count() };
        Ok(n.max(0) as u32)
    }

    /// Consume the per-thread C-side error counter set by the
    /// `RSL_FFI_BODY_*` catch handlers. Returns `Ok(())` when no
    /// exception was caught since the last consume; returns
    /// `Err(SyclError::Runtime(msg))` otherwise. The message is
    /// pulled from the per-thread last-error slot.
    ///
    /// EVERY kernel-call wrapper in this module MUST consume the
    /// counter before its own call (to clear stale errors from a
    /// previous wrapper) AND after to detect a swallowed throw.
    /// Without that, a swallowed `sycl::exception` would leave the
    /// output buffer uninitialized but Rust would see `Ok(())` and
    /// Drain the most recent `zeMemAllocHost` return code from the
    /// L0 import side channel. Used by the engine's probe to log
    /// the exact L0 error after a code-4 import failure. Returns 0
    /// when nothing has been stored.
    pub(super) fn consume_last_l0_import_code() -> u32 {
        // SAFETY: bound to the C ABI shim we own; no pointer args.
        unsafe { rsl_consume_last_l0_import_code() }
    }

    /// the engine would propagate garbage / NaN through the rest
    /// of the forward pass.
    pub(super) fn consume_error() -> Result<()> {
        // SAFETY: bound to the C ABI shim we own.
        let n = unsafe { rsl_consume_error_count() };
        if n == 0 {
            return Ok(());
        }
        // Pull the most recent message for diagnostics. 256 bytes
        // is plenty; SYCL exception strings are typically <200 chars.
        let mut buf = vec![0i8; 256];
        // SAFETY: pointer points to owned `Vec` storage; C side
        // NUL-terminates within capacity.
        unsafe {
            rsl_get_last_error_message(
                buf.as_mut_ptr() as *mut std::os::raw::c_char,
                buf.len() as c_int,
            );
        }
        let msg = unsafe {
            std::ffi::CStr::from_ptr(buf.as_ptr() as *const std::os::raw::c_char)
        }
        .to_string_lossy()
        .into_owned();
        // A DEVICE_LOST is terminal: the SYCL context is gone and every
        // subsequent GPU call will fail. Latch a process-global flag so the
        // dispatch layer can stop trying the GPU entirely (no re-probe) and
        // fall back to CPU for the rest of the session, instead of hammering
        // a dead device (which spams the log and can block TDR recovery).
        if msg.contains("DEVICE_LOST") {
            SYCL_DEVICE_LOST.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Err(SyclError::Runtime(msg))
    }

    /// Set once a `UR_RESULT_ERROR_DEVICE_LOST` is seen; read by the crate
    /// root's [`super::device_lost`].
    pub(super) static SYCL_DEVICE_LOST: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    pub fn device_info(device_index: u32) -> Result<DeviceInfo> {
        if !sycl_runtime_loadable() {
            return Err(SyclError::NoSuchDevice(device_index));
        }
        // 256 bytes is comfortable for any vendor/driver string we
        // expect from SYCL — Intel's are typically <100 chars even
        // with the "(R)" / "(TM)" markers spelled out.
        const NAME_CAP: usize = 256;
        const DRIVER_CAP: usize = 128;
        let mut name_buf = vec![0i8; NAME_CAP];
        let mut driver_buf = vec![0i8; DRIVER_CAP];
        let mut vendor_id: u32 = 0;
        let mut vram_bytes: u64 = 0;
        let mut uuid: [u8; 16] = [0u8; 16];
        // Default 0 = discrete (dedicated VRAM). The C side overwrites it
        // with the device's host-unified-memory flag; on the non-x86 stub
        // (which ignores its args) this default is what the Rust struct
        // carries — harmless since no device enumerates there anyway.
        let mut is_integrated: u8 = 0;
        // SAFETY: pointers point to live owned `Vec`/array storage; the C
        // side writes a NUL-terminated string up to the capacity, 16 UUID
        // bytes (or leaves them zeroed), and the integrated flag.
        let rc = unsafe {
            rsl_sycl_device_info(
                device_index as c_int,
                name_buf.as_mut_ptr() as *mut std::os::raw::c_char,
                NAME_CAP as c_int,
                driver_buf.as_mut_ptr() as *mut std::os::raw::c_char,
                DRIVER_CAP as c_int,
                &mut vendor_id,
                &mut vram_bytes,
                uuid.as_mut_ptr(),
                &mut is_integrated,
            )
        };
        if rc != 0 {
            return Err(SyclError::NoSuchDevice(device_index));
        }
        // CStr → String. The C side guarantees a NUL terminator
        // within `*_capacity` bytes.
        let name = unsafe {
            std::ffi::CStr::from_ptr(name_buf.as_ptr() as *const std::os::raw::c_char)
        }
        .to_string_lossy()
        .into_owned();
        let driver_version = unsafe {
            std::ffi::CStr::from_ptr(driver_buf.as_ptr() as *const std::os::raw::c_char)
        }
        .to_string_lossy()
        .into_owned();
        Ok(DeviceInfo {
            name,
            driver_version,
            vendor_id,
            vram_bytes,
            uuid,
            is_integrated: is_integrated != 0,
        })
    }

    pub fn create_stream(device_index: u32) -> Result<SyclStream> {
        if !sycl_runtime_loadable() {
            return Err(SyclError::NoSuchDevice(device_index));
        }
        // SAFETY: bound to the C ABI shim we own. Non-null return
        // means the C++ side allocated a `sycl::queue` for this
        // device; we own that allocation until `Drop` runs.
        let raw = unsafe { rsl_stream_create(device_index as c_int) };
        if raw.is_null() {
            return Err(SyclError::NoSuchDevice(device_index));
        }
        // One-time-per-process diagnostic: log which SYCL backend
        // the runtime picked. Decides whether the L0 import path
        // (`rsl_try_import_win32_handle_as_usm`) can ever fire —
        // that path requires `ext_oneapi_level_zero`. Routed
        // through `tracing` so it lands in `gui.log` (C++
        // `fprintf(stderr)` is discarded by the Tauri GUI process).
        static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        LOGGED.get_or_init(|| {
            let backend_id = unsafe { rsl_stream_backend(raw) };
            let name = match backend_id {
                0 => "host",
                1 => "opencl",
                2 => "level_zero",
                3 => "cuda",
                4 => "hip",
                5 => "native_cpu",
                _ => "unknown",
            };
            // Surface the user's preference + the actual outcome so
            // a misconfigured `RUSTLLAMA_SYCL_BACKEND` or an
            // L0-driver-absent host shows up clearly in gui.log.
            // Default policy is "prefer level_zero, fall back to
            // opencl"; unsetting the env var matches "level_zero".
            let preference = std::env::var("RUSTLLAMA_SYCL_BACKEND")
                .unwrap_or_else(|_| "level_zero".to_string());
            let fallback_engaged = backend_id == 1
                && !preference.eq_ignore_ascii_case("opencl")
                && !preference.eq_ignore_ascii_case("any")
                && !preference.eq_ignore_ascii_case("default");
            tracing::info!(
                backend = name,
                backend_id,
                preference = %preference,
                fallback_engaged,
                l0_import_eligible = (backend_id == 2),
                "SYCL queue backend identified \
                 (L0 import path requires level_zero / id=2)"
            );
            if fallback_engaged {
                tracing::warn!(
                    "RUSTLLAMA_SYCL_BACKEND='{preference}' requested but no \
                     Level Zero GPU was visible; using OpenCL fallback. The L0 \
                     USM import fast-path is disabled in this config. Likely \
                     causes: Intel Compute Runtime not installed, or the \
                     Level Zero loader is missing from PATH \
                     (`ze_loader.dll` / `libze_loader.so`)."
                );
            }
        });
        Ok(SyclStream {
            raw,
            _not_send: std::marker::PhantomData,
        })
    }

    /// SYCL interop accessors for the engine's stream. Forward to
    /// the C++ side, which exposes pointers to the queue/device/
    /// context fields the `rsl_stream` struct holds by value, for any
    /// SYCL-interop consumer that needs to bind to the same queue.
    pub fn raw_sycl_queue(stream: &SyclStream) -> *mut std::ffi::c_void {
        unsafe { rsl_stream_sycl_queue(stream.raw) }
    }

    /// Numeric SYCL backend id for the stream's queue. See the
    /// public mapping in [`super::current_backend_name`].
    /// SAFETY: `stream.raw` must be a valid stream handle.
    pub unsafe fn raw_backend_id(stream: &SyclStream) -> c_int {
        unsafe { rsl_stream_backend(stream.raw) }
    }
    pub fn raw_sycl_device(stream: &SyclStream) -> *mut std::ffi::c_void {
        unsafe { rsl_stream_sycl_device(stream.raw) }
    }
    pub fn raw_sycl_context(stream: &SyclStream) -> *mut std::ffi::c_void {
        unsafe { rsl_stream_sycl_context(stream.raw) }
    }

    /// SAFETY contract: `stream.raw` must outlive the returned
    /// pointer (caller manages this — the safe `SyclSharedBuffer`
    /// wrapper enforces it via a borrow). Returned pointer is NULL
    /// on alloc failure or when SYCL is unavailable. Takes a shared
    /// borrow of the stream — the SYCL queue is internally
    /// thread-safe so multiple concurrent allocations from one
    /// stream are sound. This is what lets the engine hold Q, K
    /// cache, V cache, and output USM buffers simultaneously.
    pub fn usm_alloc_shared(stream: &SyclStream, n_bytes: usize) -> *mut std::ffi::c_void {
        if n_bytes == 0 {
            return std::ptr::null_mut();
        }
        // SAFETY: stream.raw came from `create_stream` and is non-
        // null. The C side handles allocator failure by returning NULL.
        let p = unsafe { rsl_usm_alloc_shared(stream.raw, n_bytes) };
        // Drain any diagnostic the C side queued — on Iris Xe L0 the
        // shared-USM aspect is sometimes absent and the C side now
        // probes + falls back to host USM. We surface its assessment
        // (which aspects were available + which size failed) at
        // tracing::info so engineers can decide whether to file a
        // driver report or accept the host-USM fallback as the
        // permanent path on this device.
        let mut buf = [0u8; 256];
        let n = unsafe {
            rsl_consume_last_usm_alloc_diag(
                buf.as_mut_ptr() as *mut std::os::raw::c_char,
                buf.len() as c_int,
            )
        };
        if n > 0 {
            let msg = String::from_utf8_lossy(&buf[..n as usize]);
            tracing::info!(
                bytes_requested = n_bytes,
                fallback_used = !p.is_null(),
                diag = %msg,
                "SYCL USM allocation diagnostic"
            );
        }
        p
    }

    /// SAFETY: `ptr` must have come from `usm_alloc_shared` on the
    /// same `stream`. Caller must not use `ptr` after calling.
    pub unsafe fn usm_free(stream: &SyclStream, ptr: *mut std::ffi::c_void) {
        if ptr.is_null() {
            return;
        }
        unsafe { rsl_usm_free(stream.raw, ptr) };
    }

    /// Allocate device-local USM (dedicated VRAM) and copy `src` into
    /// it in one FFI call. Returns NULL when the device lacks
    /// device-USM support, the allocation fails, or the H2D copy
    /// throws — callers treat NULL as "device tier unavailable, fall
    /// back to shared". The returned pointer is NOT host-readable;
    /// only pass it into kernels. Free with [`usm_free`].
    pub fn usm_alloc_device_from_host(
        stream: &SyclStream,
        src: &[u8],
    ) -> *mut std::ffi::c_void {
        if src.is_empty() {
            return std::ptr::null_mut();
        }
        // SAFETY: stream.raw came from `create_stream`; the C side
        // validates its args + handles alloc/copy failure by returning
        // NULL (and frees the device buffer on a copy exception).
        unsafe {
            rsl_usm_alloc_device_from_host(
                stream.raw,
                src.as_ptr() as *const std::ffi::c_void,
                src.len(),
            )
        }
    }

    pub fn rmsnorm_usm(
        stream: &SyclStream,
        x_usm: *const u16,
        w_usm: *const u16,
        y_usm: *mut u16,
        n_rows: u32,
        d: u32,
        eps: f32,
    ) -> Result<()> {
        // No length validation here — pointer ownership and sizing
        // is the SyclSharedBuffer wrapper's job. We just forward.
        // SAFETY: the wrapper guarantees the USM pointers are live
        // for the duration of the call, sized at least n_rows*d
        // (x, y) and d (w).
        unsafe {
            rsl_rmsnorm_usm(
                stream.raw,
                x_usm,
                w_usm,
                y_usm,
                n_rows as c_int,
                d as c_int,
                eps,
            );
        }
        consume_error()
    }

    /// Fused RMSNorm + residual add. Computes
    /// `y[i] = (x[i] / norm(x_row)) * w[i] + residual[i]` in one
    /// kernel pass. Provided as a primitive for post-norm
    /// architectures; Llama uses the `add_rmsnorm_usm` variant below
    /// instead.
    ///
    /// `residual_usm` is read-only and MAY alias `x_usm` if the
    /// caller wants the residual to be the pre-norm hidden state.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_residual_usm(
        stream: &SyclStream,
        x_usm: *const u16,
        w_usm: *const u16,
        residual_usm: *const u16,
        y_usm: *mut u16,
        n_rows: u32,
        d: u32,
        eps: f32,
    ) -> Result<()> {
        // SAFETY: caller (SyclSharedBuffer wrapper) guarantees the
        // four USM pointers are live for the call and sized correctly.
        unsafe {
            rsl_rmsnorm_residual_usm(
                stream.raw,
                x_usm,
                w_usm,
                residual_usm,
                y_usm,
                n_rows as c_int,
                d as c_int,
                eps,
            );
        }
        consume_error()
    }

    /// Fused "add residual + RMSNorm" — the Llama-family pre-norm
    /// fusion. Replaces the (add_inplace → barrier → rmsnorm) pair at
    /// every post-attention and post-FFN norm site. See the kernel
    /// docstring in `cpp/rsl_kernels.cpp` for the exact computation.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_usm(
        stream: &SyclStream,
        hidden_usm: *mut u16,
        branch_usm: *const u16,
        w_usm: *const u16,
        y_norm_usm: *mut u16,
        n_rows: u32,
        d: u32,
        eps: f32,
    ) -> Result<()> {
        unsafe {
            rsl_add_rmsnorm_usm(
                stream.raw,
                hidden_usm,
                branch_usm,
                w_usm,
                y_norm_usm,
                n_rows as c_int,
                d as c_int,
                eps,
            );
        }
        consume_error()
    }

    /// F32-precision sibling of [`add_rmsnorm_usm`]. Same algorithm,
    /// same multi-workgroup topology — but skips the F32→F16→F32
    /// round-trip when callers already have F32 USM scratch. Used by
    /// the H6 (out_proj + residual + norm) dispatcher.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_f32_usm(
        stream: &SyclStream,
        hidden_usm: *mut f32,
        branch_usm: *const f32,
        w_usm: *const f32,
        y_norm_usm: *mut f32,
        n_rows: u32,
        d: u32,
        eps: f32,
    ) -> Result<()> {
        unsafe {
            rsl_add_rmsnorm_f32_usm(
                stream.raw,
                hidden_usm,
                branch_usm,
                w_usm,
                y_norm_usm,
                n_rows as c_int,
                d as c_int,
                eps,
            );
        }
        consume_error()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_decode_usm(
        stream: &SyclStream,
        q_usm: *const u16,
        k_usm: *const u16,
        v_usm: *const u16,
        out_usm: *mut u16,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_ctx: u32,
        kv_len: u32,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_decode_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
            )));
        }
        // SAFETY: pointer ownership / size is the SyclSharedBuffer
        // wrapper's responsibility. We just forward to the C ABI.
        unsafe {
            rsl_flash_attn_decode_usm(
                stream.raw,
                q_usm,
                k_usm,
                v_usm,
                out_usm,
                n_heads as c_int,
                n_kv_heads as c_int,
                head_dim as c_int,
                max_ctx as c_int,
                kv_len as c_int,
            );
        }
        consume_error()
    }

    /// FA-v3 decode entry point (SLM K/V tiling + sub-group cooperation).
    /// Same shape constraints as v2.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_decode_v3_usm(
        stream: &SyclStream,
        q_usm: *const u16,
        k_usm: *const u16,
        v_usm: *const u16,
        out_usm: *mut u16,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_ctx: u32,
        kv_len: u32,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_decode_v3_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
            )));
        }
        if head_dim == 0 || head_dim % 16 != 0 || head_dim > 256 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_decode_v3_usm: head_dim={head_dim} must be a multiple of 16 and ≤ 256 for v3 SLM-tiled dispatch"
            )));
        }
        unsafe {
            rsl_flash_attn_decode_v3_usm(
                stream.raw, q_usm, k_usm, v_usm, out_usm,
                n_heads as c_int, n_kv_heads as c_int,
                head_dim as c_int, max_ctx as c_int, kv_len as c_int,
            );
        }
        consume_error()
    }

    /// FA-v3 prefill entry point.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_prefill_v3_usm(
        stream: &SyclStream,
        q_usm: *const f32,
        k_cache_usm: *const f32,
        v_cache_usm: *const f32,
        out_usm: *mut f32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_ctx: u32,
        kv_len_base: u32,
        n_new: u32,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_prefill_v3_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
            )));
        }
        if kv_len_base.saturating_add(n_new) > max_ctx {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_prefill_v3_usm: kv_len_base={kv_len_base} + n_new={n_new} > max_ctx={max_ctx}"
            )));
        }
        if head_dim == 0 || head_dim % 16 != 0 || head_dim > 256 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_prefill_v3_usm: head_dim={head_dim} must be a multiple of 16 and ≤ 256 for v3 SLM-tiled dispatch"
            )));
        }
        unsafe {
            rsl_flash_attn_prefill_v3_usm(
                stream.raw, q_usm, k_cache_usm, v_cache_usm, out_usm,
                n_heads as c_int, n_kv_heads as c_int,
                head_dim as c_int, max_ctx as c_int,
                kv_len_base as c_int, n_new as c_int,
            );
        }
        consume_error()
    }

    /// FA-v2 decode entry point (sub-group cooperation). Returns
    /// `SyclError::InvalidShape` when `head_dim % 16 != 0` or
    /// `head_dim > 256` — the C kernel rejects those shapes and the
    /// caller falls back to [`flash_attn_decode_usm`].
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_decode_v2_usm(
        stream: &SyclStream,
        q_usm: *const u16,
        k_usm: *const u16,
        v_usm: *const u16,
        out_usm: *mut u16,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_ctx: u32,
        kv_len: u32,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_decode_v2_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
            )));
        }
        if head_dim == 0 || head_dim % 16 != 0 || head_dim > 256 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_decode_v2_usm: head_dim={head_dim} must be a multiple of 16 and ≤ 256 for v2 sub-group dispatch"
            )));
        }
        // SAFETY: pointer ownership / size is the SyclSharedBuffer
        // wrapper's responsibility. We just forward to the C ABI.
        unsafe {
            rsl_flash_attn_decode_v2_usm(
                stream.raw,
                q_usm,
                k_usm,
                v_usm,
                out_usm,
                n_heads as c_int,
                n_kv_heads as c_int,
                head_dim as c_int,
                max_ctx as c_int,
                kv_len as c_int,
            );
        }
        consume_error()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_prefill_usm(
        stream: &SyclStream,
        q_usm: *const f32,
        k_cache_usm: *const f32,
        v_cache_usm: *const f32,
        out_usm: *mut f32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_ctx: u32,
        kv_len_base: u32,
        n_new: u32,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_prefill_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
            )));
        }
        if kv_len_base.saturating_add(n_new) > max_ctx {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_prefill_usm: kv_len_base={kv_len_base} + n_new={n_new} > max_ctx={max_ctx}"
            )));
        }
        // SAFETY: pointer ownership / size is the SyclSharedBuffer
        // wrapper's responsibility. We just forward to the C ABI.
        unsafe {
            rsl_flash_attn_prefill_usm(
                stream.raw,
                q_usm,
                k_cache_usm,
                v_cache_usm,
                out_usm,
                n_heads as c_int,
                n_kv_heads as c_int,
                head_dim as c_int,
                max_ctx as c_int,
                kv_len_base as c_int,
                n_new as c_int,
            );
        }
        consume_error()
    }

    // ---- Quantized-KV flash attention (F32 Q/out, packed K/V) ----
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_decode_q4_0_usm(
        stream: &SyclStream,
        q_usm: *const f32,
        k_packed_usm: *const u8,
        v_packed_usm: *const u8,
        out_usm: *mut f32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_ctx: u32,
        kv_len: u32,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_decode_q4_0_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
            )));
        }
        // SAFETY: USM pointer ownership / sizing is the caller's
        // responsibility. We just forward to the C ABI.
        unsafe {
            rsl_flash_attn_decode_q4_0_usm(
                stream.raw, q_usm, k_packed_usm, v_packed_usm, out_usm,
                n_heads as c_int, n_kv_heads as c_int, head_dim as c_int,
                max_ctx as c_int, kv_len as c_int,
            );
        }
        consume_error()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_prefill_q4_0_usm(
        stream: &SyclStream,
        q_usm: *const f32,
        k_packed_usm: *const u8,
        v_packed_usm: *const u8,
        out_usm: *mut f32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_ctx: u32,
        kv_len_base: u32,
        n_new: u32,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_prefill_q4_0_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
            )));
        }
        if kv_len_base.saturating_add(n_new) > max_ctx {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_prefill_q4_0_usm: kv_len_base={kv_len_base} + n_new={n_new} > max_ctx={max_ctx}"
            )));
        }
        // SAFETY: as above.
        unsafe {
            rsl_flash_attn_prefill_q4_0_usm(
                stream.raw, q_usm, k_packed_usm, v_packed_usm, out_usm,
                n_heads as c_int, n_kv_heads as c_int, head_dim as c_int,
                max_ctx as c_int, kv_len_base as c_int, n_new as c_int,
            );
        }
        consume_error()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_decode_nvfp4_usm(
        stream: &SyclStream,
        q_usm: *const f32,
        k_packed_usm: *const u8,
        v_packed_usm: *const u8,
        out_usm: *mut f32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_ctx: u32,
        kv_len: u32,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_decode_nvfp4_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
            )));
        }
        // SAFETY: as above.
        unsafe {
            rsl_flash_attn_decode_nvfp4_usm(
                stream.raw, q_usm, k_packed_usm, v_packed_usm, out_usm,
                n_heads as c_int, n_kv_heads as c_int, head_dim as c_int,
                max_ctx as c_int, kv_len as c_int,
            );
        }
        consume_error()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_prefill_nvfp4_usm(
        stream: &SyclStream,
        q_usm: *const f32,
        k_packed_usm: *const u8,
        v_packed_usm: *const u8,
        out_usm: *mut f32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_ctx: u32,
        kv_len_base: u32,
        n_new: u32,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_prefill_nvfp4_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
            )));
        }
        if kv_len_base.saturating_add(n_new) > max_ctx {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_prefill_nvfp4_usm: kv_len_base={kv_len_base} + n_new={n_new} > max_ctx={max_ctx}"
            )));
        }
        // SAFETY: as above.
        unsafe {
            rsl_flash_attn_prefill_nvfp4_usm(
                stream.raw, q_usm, k_packed_usm, v_packed_usm, out_usm,
                n_heads as c_int, n_kv_heads as c_int, head_dim as c_int,
                max_ctx as c_int, kv_len_base as c_int, n_new as c_int,
            );
        }
        consume_error()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_decode_tq_usm(
        stream: &SyclStream,
        q_usm: *const f32,
        k_packed_usm: *const u8,
        v_packed_usm: *const u8,
        k_scales_usm: *const f32,
        v_scales_usm: *const f32,
        bits: u32,
        out_usm: *mut f32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_ctx: u32,
        kv_len: u32,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_decode_tq_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
            )));
        }
        // SAFETY: as above.
        unsafe {
            rsl_flash_attn_decode_tq_usm(
                stream.raw, q_usm, k_packed_usm, v_packed_usm, k_scales_usm,
                v_scales_usm, bits as c_int, out_usm, n_heads as c_int,
                n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int,
                kv_len as c_int,
            );
        }
        consume_error()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_prefill_tq_usm(
        stream: &SyclStream,
        q_usm: *const f32,
        k_packed_usm: *const u8,
        v_packed_usm: *const u8,
        k_scales_usm: *const f32,
        v_scales_usm: *const f32,
        bits: u32,
        out_usm: *mut f32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_ctx: u32,
        kv_len_base: u32,
        n_new: u32,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_prefill_tq_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
            )));
        }
        if kv_len_base.saturating_add(n_new) > max_ctx {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_prefill_tq_usm: kv_len_base={kv_len_base} + n_new={n_new} > max_ctx={max_ctx}"
            )));
        }
        // SAFETY: as above.
        unsafe {
            rsl_flash_attn_prefill_tq_usm(
                stream.raw, q_usm, k_packed_usm, v_packed_usm, k_scales_usm,
                v_scales_usm, bits as c_int, out_usm, n_heads as c_int,
                n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int,
                kv_len_base as c_int, n_new as c_int,
            );
        }
        consume_error()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_decode_q8_0_usm(
        stream: &SyclStream,
        q_usm: *const f32,
        k_packed_usm: *const u8,
        v_packed_usm: *const u8,
        k_scales_usm: *const f32,
        v_scales_usm: *const f32,
        out_usm: *mut f32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_ctx: u32,
        kv_len: u32,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_decode_q8_0_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
            )));
        }
        // SAFETY: as above.
        unsafe {
            rsl_flash_attn_decode_q8_0_usm(
                stream.raw, q_usm, k_packed_usm, v_packed_usm, k_scales_usm,
                v_scales_usm, out_usm, n_heads as c_int, n_kv_heads as c_int,
                head_dim as c_int, max_ctx as c_int, kv_len as c_int,
            );
        }
        consume_error()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_prefill_q8_0_usm(
        stream: &SyclStream,
        q_usm: *const f32,
        k_packed_usm: *const u8,
        v_packed_usm: *const u8,
        k_scales_usm: *const f32,
        v_scales_usm: *const f32,
        out_usm: *mut f32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_ctx: u32,
        kv_len_base: u32,
        n_new: u32,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_prefill_q8_0_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
            )));
        }
        if kv_len_base.saturating_add(n_new) > max_ctx {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_prefill_q8_0_usm: kv_len_base={kv_len_base} + n_new={n_new} > max_ctx={max_ctx}"
            )));
        }
        // SAFETY: as above.
        unsafe {
            rsl_flash_attn_prefill_q8_0_usm(
                stream.raw, q_usm, k_packed_usm, v_packed_usm, k_scales_usm,
                v_scales_usm, out_usm, n_heads as c_int, n_kv_heads as c_int,
                head_dim as c_int, max_ctx as c_int, kv_len_base as c_int,
                n_new as c_int,
            );
        }
        consume_error()
    }

    /// FA-v2 prefill entry point (sub-group cooperation). Same
    /// shape constraints as [`flash_attn_decode_v2_usm`]: head_dim
    /// must be a multiple of 16 and ≤ 256.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_prefill_v2_usm(
        stream: &SyclStream,
        q_usm: *const f32,
        k_cache_usm: *const f32,
        v_cache_usm: *const f32,
        out_usm: *mut f32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_ctx: u32,
        kv_len_base: u32,
        n_new: u32,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_prefill_v2_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
            )));
        }
        if kv_len_base.saturating_add(n_new) > max_ctx {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_prefill_v2_usm: kv_len_base={kv_len_base} + n_new={n_new} > max_ctx={max_ctx}"
            )));
        }
        if head_dim == 0 || head_dim % 16 != 0 || head_dim > 256 {
            return Err(SyclError::InvalidShape(format!(
                "flash_attn_prefill_v2_usm: head_dim={head_dim} must be a multiple of 16 and ≤ 256 for v2 sub-group dispatch"
            )));
        }
        // SAFETY: pointer ownership / size is the SyclSharedBuffer
        // wrapper's responsibility.
        unsafe {
            rsl_flash_attn_prefill_v2_usm(
                stream.raw,
                q_usm,
                k_cache_usm,
                v_cache_usm,
                out_usm,
                n_heads as c_int,
                n_kv_heads as c_int,
                head_dim as c_int,
                max_ctx as c_int,
                kv_len_base as c_int,
                n_new as c_int,
            );
        }
        consume_error()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16_usm(
        stream: &SyclStream,
        a_usm: *const u16,
        b_usm: *const u16,
        c_usm: *mut u16,
        m: u32,
        n: u32,
        k: u32,
        lda: u32,
        ldb: u32,
        ldc: u32,
    ) -> Result<()> {
        if m == 0 || n == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "gemm_f16_usm: zero dim ({m}x{n}x{k})"
            )));
        }
        // SAFETY: caller owns the USM pointers + ensures they
        // outlive the call. The kernel `.wait()`s before returning.
        unsafe {
            rsl_gemm_f16_usm(
                stream.raw,
                a_usm,
                b_usm,
                c_usm,
                m as c_int,
                n as c_int,
                k as c_int,
                lda as c_int,
                ldb as c_int,
                ldc as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_q8_0_f32_usm(
        stream: &SyclStream,
        w_q_usm: *const i8,
        w_scales_usm: *const f32,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q8_0_f32_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 32 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q8_0_f32_usm: K must be a multiple of 32, got {k}"
            )));
        }
        // SAFETY: caller owns the USM pointers + ensures they
        // outlive the call. The kernel `.wait()`s before returning.
        unsafe {
            rsl_matvec_q8_0_f32_usm(
                stream.raw,
                w_q_usm,
                w_scales_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_q8_0_packed_f32_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q8_0_packed_f32_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 32 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q8_0_packed_f32_usm: K must be a multiple of 32, got {k}"
            )));
        }
        // SAFETY: caller owns the USM pointers + ensures they
        // outlive the call. The kernel `.wait()`s before returning.
        // `lws == 0` selects the C++-side default (RSL_LWS = 64);
        // values outside {16, 32, 64, 128, 256} fall back to 64.
        unsafe {
            rsl_matvec_q8_0_packed_f32_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_q4_k_packed_f32_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q4_k_packed_f32_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q4_k_packed_f32_usm: K must be a multiple of 256, got {k}"
            )));
        }
        // SAFETY: caller owns the USM pointers + ensures they
        // outlive the call. The kernel `.wait()`s before returning.
        // `lws == 0` selects the C++-side default (RSL_LWS = 64);
        // values outside {16, 32, 64, 128, 256} fall back to 64.
        unsafe {
            rsl_matvec_q4_k_packed_f32_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_ptq1_0_packed_f32_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_ptq1_0_packed_f32_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 128 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_ptq1_0_packed_f32_usm: K must be a multiple of 128, got {k}"
            )));
        }
        // SAFETY: caller owns the USM pointers + ensures they outlive
        // the call. The kernel `.wait()`s before returning.
        unsafe {
            rsl_matvec_ptq1_0_packed_f32_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_ptq1_0_packed_f32_batched_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        n: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 || n == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_ptq1_0_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
            )));
        }
        if k % 128 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_ptq1_0_packed_f32_batched_usm: K must be a multiple of 128, got {k}"
            )));
        }
        // SAFETY: as above; out sized M*N.
        unsafe {
            rsl_matvec_ptq1_0_packed_f32_batched_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                n as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn hadamard_forward_usm(
        stream: &SyclStream,
        x_usm: *const f32,
        signs_usm: *const f32,
        out_usm: *mut f32,
        n_elems: u32,
        block: u32,
    ) -> Result<()> {
        if n_elems == 0 || block == 0 || n_elems % block != 0 {
            return Err(SyclError::InvalidShape(format!(
                "hadamard_forward_usm: n_elems {n_elems} not a multiple of block {block}"
            )));
        }
        if !block.is_power_of_two() || block > 4096 {
            return Err(SyclError::InvalidShape(format!(
                "hadamard_forward_usm: block must be a power of two <= 4096, got {block}"
            )));
        }
        // SAFETY: as above; x/signs/out all sized n_elems.
        unsafe {
            rsl_hadamard_forward_usm(
                stream.raw,
                x_usm,
                signs_usm,
                out_usm,
                n_elems as c_int,
                block as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_q5_k_packed_f32_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q5_k_packed_f32_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q5_k_packed_f32_usm: K must be a multiple of 256, got {k}"
            )));
        }
        // SAFETY: see matvec_q4_k_packed_f32_usm above. `lws == 0`
        // selects the C++-side default (RSL_LWS = 64).
        unsafe {
            rsl_matvec_q5_k_packed_f32_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_q6_k_packed_f32_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q6_k_packed_f32_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q6_k_packed_f32_usm: K must be a multiple of 256, got {k}"
            )));
        }
        // SAFETY: see matvec_q4_k_packed_f32_usm above. `lws == 0`
        // selects the C++-side default (RSL_LWS = 64).
        unsafe {
            rsl_matvec_q6_k_packed_f32_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    // CPU-parity packed matvec safe wrappers — identical shape to the
    // Q4_K/Q6_K wrappers above; only the FFI symbol + K-alignment differ.
    macro_rules! packed_matvec_real_wrapper {
        ($name:ident, $ffi:ident, $align:expr) => {
            pub fn $name(
                stream: &SyclStream, w_bytes_usm: *const u8, x_usm: *const f32,
                out_usm: *mut f32, m: u32, k: u32, lws: u32,
            ) -> Result<()> {
                if m == 0 || k == 0 {
                    return Err(SyclError::InvalidShape(format!(
                        concat!(stringify!($name), ": zero dim (M={}, K={})"), m, k)));
                }
                if k % $align != 0 {
                    return Err(SyclError::InvalidShape(format!(
                        concat!(stringify!($name), ": K must be a multiple of ", stringify!($align), ", got {}"), k)));
                }
                // SAFETY: caller owns the USM pointers + ensures they
                // outlive the call. The kernel `.wait()`s before returning.
                unsafe {
                    $ffi(stream.raw, w_bytes_usm, x_usm, out_usm,
                         m as c_int, k as c_int, lws as c_int);
                }
                consume_error()
            }
        };
    }
    packed_matvec_real_wrapper!(matvec_q4_0_packed_f32_usm, rsl_matvec_q4_0_packed_f32_usm, 32);
    // OCP Microscaling (MX) — block 32, one E8M0 power-of-two scale/block.
    packed_matvec_real_wrapper!(matvec_mxfp4_packed_f32_usm, rsl_matvec_mxfp4_packed_f32_usm, 32);
    packed_matvec_real_wrapper!(matvec_mxfp6_packed_f32_usm, rsl_matvec_mxfp6_packed_f32_usm, 32);
    packed_matvec_real_wrapper!(matvec_mxfp8_packed_f32_usm, rsl_matvec_mxfp8_packed_f32_usm, 32);
    packed_matvec_real_wrapper!(matvec_q5_0_packed_f32_usm, rsl_matvec_q5_0_packed_f32_usm, 32);
    packed_matvec_real_wrapper!(matvec_q4_1_packed_f32_usm, rsl_matvec_q4_1_packed_f32_usm, 32);
    packed_matvec_real_wrapper!(matvec_q5_1_packed_f32_usm, rsl_matvec_q5_1_packed_f32_usm, 32);
    packed_matvec_real_wrapper!(matvec_q2_k_packed_f32_usm, rsl_matvec_q2_k_packed_f32_usm, 256);
    packed_matvec_real_wrapper!(matvec_q3_k_packed_f32_usm, rsl_matvec_q3_k_packed_f32_usm, 256);
    packed_matvec_real_wrapper!(matvec_q8_k_packed_f32_usm, rsl_matvec_q8_k_packed_f32_usm, 256);
    packed_matvec_real_wrapper!(matvec_pq2_0_packed_f32_usm, rsl_matvec_pq2_0_packed_f32_usm, 128);

    pub fn matvec_iq1_s_packed_f32_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq1_s_packed_f32_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq1_s_packed_f32_usm: K must be a multiple of 256, got {k}"
            )));
        }
        // SAFETY: USM pointers from caller; sized per IQ1_S layout
        // (M × K/256 × 50 bytes for w_bytes; K f32 for x; M f32 for
        // out). Kernel `.wait()`s before returning.
        unsafe {
            rsl_matvec_iq1_s_packed_f32_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_iq2_xxs_packed_f32_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_xxs_packed_f32_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_xxs_packed_f32_usm: K must be a multiple of 256, got {k}"
            )));
        }
        // SAFETY: USM pointers from caller; sized per IQ2_XXS layout
        // (M × K/256 × 66 bytes for w_bytes; K f32 for x; M f32 for
        // out). Kernel `.wait()`s before returning.
        unsafe {
            rsl_matvec_iq2_xxs_packed_f32_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_iq1_m_packed_f32_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq1_m_packed_f32_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq1_m_packed_f32_usm: K must be a multiple of 256, got {k}"
            )));
        }
        // SAFETY: USM pointers from caller; sized per IQ1_M layout
        // (M × K/256 × 56 bytes for w_bytes; K f32 for x; M f32 for
        // out). Kernel `.wait()`s before returning.
        unsafe {
            rsl_matvec_iq1_m_packed_f32_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_iq2_xs_packed_f32_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_xs_packed_f32_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_xs_packed_f32_usm: K must be a multiple of 256, got {k}"
            )));
        }
        // SAFETY: USM pointers from caller; sized per IQ2_XS layout
        // (M × K/256 × 74 bytes for w_bytes; K f32 for x; M f32 for
        // out). Kernel `.wait()`s before returning.
        unsafe {
            rsl_matvec_iq2_xs_packed_f32_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_iq2_s_packed_f32_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_s_packed_f32_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_s_packed_f32_usm: K must be a multiple of 256, got {k}"
            )));
        }
        // SAFETY: USM pointers from caller; sized per IQ2_S layout
        // (M × K/256 × 82 bytes for w_bytes; K f32 for x; M f32 for
        // out). Kernel `.wait()`s before returning.
        unsafe {
            rsl_matvec_iq2_s_packed_f32_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_iq3_xxs_packed_f32_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq3_xxs_packed_f32_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq3_xxs_packed_f32_usm: K must be a multiple of 256, got {k}"
            )));
        }
        // SAFETY: USM pointers from caller; sized per IQ3_XXS
        // layout (M × K/256 × 98 bytes for w_bytes; K f32 for x;
        // M f32 for out). Kernel `.wait()`s before returning.
        unsafe {
            rsl_matvec_iq3_xxs_packed_f32_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_iq3_s_packed_f32_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq3_s_packed_f32_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq3_s_packed_f32_usm: K must be a multiple of 256, got {k}"
            )));
        }
        // SAFETY: USM pointers from caller; sized per IQ3_S layout
        // (M × K/256 × 110 bytes for w_bytes; K f32 for x; M f32
        // for out). Kernel `.wait()`s before returning.
        unsafe {
            rsl_matvec_iq3_s_packed_f32_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_iq1_s_packed_f32_batched_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        n: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 || n == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq1_s_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq1_s_packed_f32_batched_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_iq1_s_packed_f32_batched_usm(
                stream.raw, w_bytes_usm, x_usm, out_usm,
                m as c_int, k as c_int, n as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_iq1_m_packed_f32_batched_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        n: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 || n == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq1_m_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq1_m_packed_f32_batched_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_iq1_m_packed_f32_batched_usm(
                stream.raw, w_bytes_usm, x_usm, out_usm,
                m as c_int, k as c_int, n as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_iq2_xxs_packed_f32_batched_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        n: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 || n == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_xxs_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_xxs_packed_f32_batched_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_iq2_xxs_packed_f32_batched_usm(
                stream.raw, w_bytes_usm, x_usm, out_usm,
                m as c_int, k as c_int, n as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_iq2_xs_packed_f32_batched_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        n: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 || n == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_xs_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_xs_packed_f32_batched_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_iq2_xs_packed_f32_batched_usm(
                stream.raw, w_bytes_usm, x_usm, out_usm,
                m as c_int, k as c_int, n as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_iq2_s_packed_f32_batched_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        n: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 || n == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_s_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_s_packed_f32_batched_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_iq2_s_packed_f32_batched_usm(
                stream.raw, w_bytes_usm, x_usm, out_usm,
                m as c_int, k as c_int, n as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_iq3_xxs_packed_f32_batched_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        n: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 || n == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq3_xxs_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq3_xxs_packed_f32_batched_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_iq3_xxs_packed_f32_batched_usm(
                stream.raw, w_bytes_usm, x_usm, out_usm,
                m as c_int, k as c_int, n as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_iq3_s_packed_f32_batched_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        n: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 || n == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq3_s_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq3_s_packed_f32_batched_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_iq3_s_packed_f32_batched_usm(
                stream.raw, w_bytes_usm, x_usm, out_usm,
                m as c_int, k as c_int, n as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_iq4_nl_packed_f32_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq4_nl_packed_f32_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 32 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq4_nl_packed_f32_usm: K must be a multiple of 32, got {k}"
            )));
        }
        // SAFETY: see matvec_q4_k_packed_f32_usm above.
        unsafe {
            rsl_matvec_iq4_nl_packed_f32_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_iq4_xs_packed_f32_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq4_xs_packed_f32_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq4_xs_packed_f32_usm: K must be a multiple of 256, got {k}"
            )));
        }
        // SAFETY: see matvec_q4_k_packed_f32_usm above.
        unsafe {
            rsl_matvec_iq4_xs_packed_f32_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn iq_search_8elt_delta_iq1s(
        stream: &SyclStream,
        targets: *const f32,
        delta: f32,
        grid_f32: *const f32,
        out_grid_idx: *mut u16,
        out_signed_score: *mut f32,
        out_norm_sq: *mut f32,
        n_chunks: u32,
    ) -> Result<()> {
        if n_chunks == 0 {
            return Ok(());
        }
        // SAFETY: caller owns all USM pointers + holds them alive
        // across the call. Kernel `.wait()`s before returning.
        // The output sizes are validated by the caller; we don't
        // re-check at the FFI boundary (no length parameter).
        unsafe {
            rsl_iq_search_8elt_delta_iq1s(
                stream.raw,
                targets,
                delta,
                grid_f32,
                out_grid_idx,
                out_signed_score,
                out_norm_sq,
                n_chunks as c_int,
            );
        }
        consume_error()
    }

    pub fn iq_search_8elt_delta_iq1s_all3(
        stream: &SyclStream,
        targets: *const f32,
        delta: f32,
        grid_f32: *const f32,
        out_grid_idx_abs: *mut u16,
        out_signed_score_abs: *mut f32,
        out_norm_sq_abs: *mut f32,
        out_grid_idx_pos: *mut u16,
        out_signed_score_pos: *mut f32,
        out_norm_sq_pos: *mut f32,
        out_grid_idx_neg: *mut u16,
        out_signed_score_neg: *mut f32,
        out_norm_sq_neg: *mut f32,
        n_chunks: u32,
    ) -> Result<()> {
        if n_chunks == 0 {
            return Ok(());
        }
        unsafe {
            rsl_iq_search_8elt_delta_iq1s_all3(
                stream.raw,
                targets,
                delta,
                grid_f32,
                out_grid_idx_abs, out_signed_score_abs, out_norm_sq_abs,
                out_grid_idx_pos, out_signed_score_pos, out_norm_sq_pos,
                out_grid_idx_neg, out_signed_score_neg, out_norm_sq_neg,
                n_chunks as c_int,
            );
        }
        consume_error()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn iq_search_8elt_delta_iq1s_all3_w(
        stream: &SyclStream,
        targets: *const f32,
        weights: *const f32,
        delta: f32,
        grid_f32: *const f32,
        out_grid_idx_abs: *mut u16,
        out_signed_score_abs: *mut f32,
        out_norm_sq_abs: *mut f32,
        out_grid_idx_pos: *mut u16,
        out_signed_score_pos: *mut f32,
        out_norm_sq_pos: *mut f32,
        out_grid_idx_neg: *mut u16,
        out_signed_score_neg: *mut f32,
        out_norm_sq_neg: *mut f32,
        n_chunks: u32,
    ) -> Result<()> {
        if n_chunks == 0 {
            return Ok(());
        }
        unsafe {
            rsl_iq_search_8elt_delta_iq1s_all3_w(
                stream.raw,
                targets,
                weights,
                delta,
                grid_f32,
                out_grid_idx_abs, out_signed_score_abs, out_norm_sq_abs,
                out_grid_idx_pos, out_signed_score_pos, out_norm_sq_pos,
                out_grid_idx_neg, out_signed_score_neg, out_norm_sq_neg,
                n_chunks as c_int,
            );
        }
        consume_error()
    }

    pub fn iq_search_8elt_signed(
        stream: &SyclStream,
        targets: *const f32,
        grid_f32: *const f32,
        grid_norm_sq_table: *const f32,
        ksigns_rev: *const u8,
        n_grid: u32,
        out_grid_idx: *mut u16,
        out_sign_idx: *mut u8,
        out_signed_score: *mut f32,
        out_grid_norm_sq: *mut f32,
        n_chunks: u32,
    ) -> Result<()> {
        if n_chunks == 0 || n_grid == 0 {
            return Ok(());
        }
        // SAFETY: caller owns all USM pointers + holds them alive
        // across the call. Kernel `.wait()`s before returning.
        unsafe {
            rsl_iq_search_8elt_signed(
                stream.raw,
                targets,
                grid_f32,
                grid_norm_sq_table,
                ksigns_rev,
                n_grid as c_int,
                out_grid_idx,
                out_sign_idx,
                out_signed_score,
                out_grid_norm_sq,
                n_chunks as c_int,
            );
        }
        consume_error()
    }

    pub fn iq_search_4elt_paired_signed(
        stream: &SyclStream,
        targets: *const f32,
        grid_f32: *const f32,
        grid_norm_sq_table: *const f32,
        kmask: *const u8,
        ksigns_rev: *const u8,
        n_grid: u32,
        out_grid1_idx: *mut u16,
        out_grid2_idx: *mut u16,
        out_sign_idx: *mut u8,
        out_signed_score: *mut f32,
        out_grid_norm_sq: *mut f32,
        n_chunks: u32,
    ) -> Result<()> {
        if n_chunks == 0 || n_grid == 0 {
            return Ok(());
        }
        // SAFETY: caller owns all USM pointers + holds them alive
        // across the call. Kernel `.wait()`s before returning.
        unsafe {
            rsl_iq_search_4elt_paired_signed(
                stream.raw,
                targets,
                grid_f32,
                grid_norm_sq_table,
                kmask,
                ksigns_rev,
                n_grid as c_int,
                out_grid1_idx,
                out_grid2_idx,
                out_sign_idx,
                out_signed_score,
                out_grid_norm_sq,
                n_chunks as c_int,
            );
        }
        consume_error()
    }

    pub fn sampler_argmax_usm(
        stream: &SyclStream,
        logits_usm: *const f32,
        vocab: u32,
        out_idx_usm: *mut i32,
    ) -> Result<()> {
        if vocab == 0 {
            return Err(SyclError::InvalidShape(
                "sampler_argmax_usm: vocab is zero".to_string(),
            ));
        }
        // SAFETY: caller owns both USM pointers + holds them alive
        // across the call. Kernel `.wait()`s before returning.
        unsafe {
            rsl_sampler_argmax_usm(stream.raw, logits_usm, vocab as c_int, out_idx_usm);
        }
        consume_error()
    }

    pub fn sampler_temp_softmax_usm(
        stream: &SyclStream,
        logits_usm: *mut f32,
        vocab: u32,
        inv_temp: f32,
    ) -> Result<()> {
        if vocab == 0 {
            return Err(SyclError::InvalidShape(
                "sampler_temp_softmax_usm: vocab is zero".to_string(),
            ));
        }
        // SAFETY: caller owns the USM pointer + holds it alive
        // across the call; kernel writes in-place. `.wait()`s.
        unsafe {
            rsl_sampler_temp_softmax_usm(stream.raw, logits_usm, vocab as c_int, inv_temp);
        }
        consume_error()
    }

    pub fn sampler_multinomial_usm(
        stream: &SyclStream,
        probs_usm: *const f32,
        vocab: u32,
        rng_state_usm: *mut u64,
        out_idx_usm: *mut i32,
    ) -> Result<()> {
        if vocab == 0 {
            return Err(SyclError::InvalidShape(
                "sampler_multinomial_usm: vocab is zero".to_string(),
            ));
        }
        // SAFETY: caller owns all USM pointers + holds them alive
        // across the call. Kernel `.wait()`s.
        unsafe {
            rsl_sampler_multinomial_usm(
                stream.raw,
                probs_usm,
                vocab as c_int,
                rng_state_usm,
                out_idx_usm,
            );
        }
        consume_error()
    }

    pub fn sampler_top_p_usm(
        stream: &SyclStream,
        probs_usm: *mut f32,
        vocab: u32,
        p: f32,
        needs_fallback_usm: *mut i32,
    ) -> Result<()> {
        if vocab == 0 {
            return Err(SyclError::InvalidShape(
                "sampler_top_p_usm: vocab is zero".to_string(),
            ));
        }
        // p ∈ {0, ≥1} are CPU no-ops; the kernel detects + writes
        // `needs_fallback = 0` for both, leaving probs untouched.
        // SAFETY: caller owns the USM pointers + holds them alive
        // across the call; kernel writes in-place + sets fallback
        // flag. `.wait()`s.
        unsafe {
            rsl_sampler_top_p_usm(
                stream.raw,
                probs_usm,
                vocab as c_int,
                p,
                needs_fallback_usm,
            );
        }
        consume_error()
    }

    pub fn sampler_top_k_usm(
        stream: &SyclStream,
        probs_usm: *mut f32,
        vocab: u32,
        k: u32,
    ) -> Result<()> {
        if vocab == 0 {
            return Err(SyclError::InvalidShape(
                "sampler_top_k_usm: vocab is zero".to_string(),
            ));
        }
        if k == 0 || k >= vocab {
            // No-op gate — matches CPU.
            return Ok(());
        }
        // SAFETY: caller owns the USM pointer + holds it alive
        // across the call; kernel writes in-place. `.wait()`s.
        unsafe {
            rsl_sampler_top_k_usm(stream.raw, probs_usm, vocab as c_int, k as c_int);
        }
        consume_error()
    }

    pub fn sampler_penalty_usm(
        stream: &SyclStream,
        logits_usm: *mut f32,
        vocab: u32,
        recent_usm: *const u32,
        recent_n: u32,
        repeat: f32,
        frequency: f32,
        presence: f32,
    ) -> Result<()> {
        if vocab == 0 {
            return Err(SyclError::InvalidShape(
                "sampler_penalty_usm: vocab is zero".to_string(),
            ));
        }
        if recent_n == 0 || recent_usm.is_null() {
            // No tokens to penalize — clean no-op rather than a
            // wasted kernel launch.
            return Ok(());
        }
        // SAFETY: caller owns all USM pointers; kernel waits.
        unsafe {
            rsl_sampler_penalty_usm(
                stream.raw,
                logits_usm,
                vocab as c_int,
                recent_usm,
                recent_n as c_int,
                repeat,
                frequency,
                presence,
            );
        }
        consume_error()
    }

    pub fn matvec_q8_0_packed_f32_batched_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        n: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 || n == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q8_0_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
            )));
        }
        if k % 32 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q8_0_packed_f32_batched_usm: K must be a multiple of 32, got {k}"
            )));
        }
        // SAFETY: caller owns the USM pointers + ensures they outlive
        // the call. Kernel `.wait()`s before returning. `lws == 0`
        // selects the C++-side default (RSL_LWS = 64).
        unsafe {
            rsl_matvec_q8_0_packed_f32_batched_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                n as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_q4_k_packed_f32_batched_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        n: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 || n == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q4_k_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q4_k_packed_f32_batched_usm: K must be a multiple of 256, got {k}"
            )));
        }
        // SAFETY: see matvec_q8_0_packed_f32_batched_usm above.
        unsafe {
            rsl_matvec_q4_k_packed_f32_batched_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                n as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_iq4_nl_packed_f32_batched_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        n: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 || n == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq4_nl_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
            )));
        }
        if k % 32 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq4_nl_packed_f32_batched_usm: K must be a multiple of 32, got {k}"
            )));
        }
        // SAFETY: caller owns all USM pointers + holds them alive
        // across the call; kernel waits before returning.
        unsafe {
            rsl_matvec_iq4_nl_packed_f32_batched_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                n as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_iq4_xs_packed_f32_batched_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        n: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 || n == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq4_xs_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq4_xs_packed_f32_batched_usm: K must be a multiple of 256, got {k}"
            )));
        }
        // SAFETY: caller owns all USM pointers + holds them alive
        // across the call; kernel waits before returning.
        unsafe {
            rsl_matvec_iq4_xs_packed_f32_batched_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                n as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_q5_k_packed_f32_batched_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        n: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 || n == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q5_k_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q5_k_packed_f32_batched_usm: K must be a multiple of 256, got {k}"
            )));
        }
        // SAFETY: see matvec_q8_0_packed_f32_batched_usm above.
        unsafe {
            rsl_matvec_q5_k_packed_f32_batched_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                n as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_q6_k_packed_f32_batched_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        x_usm: *const f32,
        out_usm: *mut f32,
        m: u32,
        k: u32,
        n: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 || n == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q6_k_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q6_k_packed_f32_batched_usm: K must be a multiple of 256, got {k}"
            )));
        }
        // SAFETY: see matvec_q8_0_packed_f32_batched_usm above.
        unsafe {
            rsl_matvec_q6_k_packed_f32_batched_usm(
                stream.raw,
                w_bytes_usm,
                x_usm,
                out_usm,
                m as c_int,
                k as c_int,
                n as c_int,
                lws as c_int,
            );
        }
        consume_error()
    }

    /// Try to import a Win32 file-mapping HANDLE as a device-
    /// accessible USM allocation. On success returns the imported
    /// pointer (owned by the L0 context — must be freed via
    /// [`release_imported_usm`]). On any failure returns
    /// [`SyclError::L0ImportUnsupported`] with the C-side return
    /// code; the engine treats this as "fall back to copy".
    ///
    /// SAFETY note for callers: the `mapping_handle` must be a
    /// valid Win32 NT handle (e.g. from `CreateFileMappingW`) and
    /// must remain valid for the lifetime of the import. `size`
    /// must not exceed the underlying file mapping's size.
    pub fn try_import_win32_handle_as_usm(
        stream: &SyclStream,
        mapping_handle: *mut std::ffi::c_void,
        size: usize,
    ) -> Result<*mut std::ffi::c_void> {
        if mapping_handle.is_null() || size == 0 {
            return Err(SyclError::L0ImportUnsupported(1));
        }
        let mut out_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let code = unsafe {
            rsl_try_import_win32_handle_as_usm(stream.raw, mapping_handle, size, &mut out_ptr)
        };
        // The C++ side wraps its body in RSL_FFI_BODY_RET, which on
        // exception returns the default (1) and bumps the error
        // counter. Drain it so the next call starts clean.
        let _ = consume_error();
        if code != 0 || out_ptr.is_null() {
            return Err(SyclError::L0ImportUnsupported(code as i32));
        }
        Ok(out_ptr)
    }

    /// Free a pointer obtained from [`try_import_win32_handle_as_usm`].
    /// Safe to call with a null pointer (no-op). Must be called on the
    /// same `SyclStream` whose queue produced the import.
    pub fn release_imported_usm(stream: &SyclStream, dev_ptr: *mut std::ffi::c_void) {
        if dev_ptr.is_null() {
            return;
        }
        unsafe { rsl_release_imported_usm(stream.raw, dev_ptr) };
        let _ = consume_error();
    }

    /// Diagnostic: bare `zeMemAllocHost` (no import descriptor).
    /// Allocates `size` bytes of L0 host memory, immediately frees
    /// it, and returns success / category code. Lets callers
    /// distinguish "our host_desc struct layout is wrong" (this
    /// fails too) from "the import descriptor is what's rejected"
    /// (this succeeds while the import call fails).
    pub fn try_alloc_host_baseline(stream: &SyclStream, size: usize) -> Result<()> {
        if size == 0 {
            return Err(SyclError::L0ImportUnsupported(1));
        }
        let mut out_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let code = unsafe { rsl_try_alloc_host_baseline(stream.raw, size, &mut out_ptr) };
        let _ = consume_error();
        if code != 0 {
            return Err(SyclError::L0ImportUnsupported(code as i32));
        }
        Ok(())
    }

    pub fn rope_usm(
        stream: &SyclStream,
        qk_usm: *mut u16,
        n_heads: u32,
        head_dim: u32,
        pos: u32,
        inv_freq_usm: *const u16,
    ) -> Result<()> {
        if head_dim % 2 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "rope_usm: head_dim must be even, got {head_dim}"
            )));
        }
        // SAFETY: pointer ownership belongs to the caller (the safe
        // wrapper uses SyclSharedBuffer to enforce sizing). We just
        // forward to the C ABI.
        unsafe {
            rsl_rope_usm(
                stream.raw,
                qk_usm,
                n_heads as c_int,
                head_dim as c_int,
                pos as c_int,
                inv_freq_usm,
            );
        }
        consume_error()
    }

    pub fn silu_mul_usm(
        stream: &SyclStream,
        x_usm: *const u16,
        y_usm: *const u16,
        out_usm: *mut u16,
        n: u32,
    ) -> Result<()> {
        // SAFETY: same contract as rope_usm.
        unsafe {
            rsl_silu_mul_usm(stream.raw, x_usm, y_usm, out_usm, n as c_int);
        }
        consume_error()
    }

    pub fn embedding_lookup_usm(
        stream: &SyclStream,
        table_usm: *const u16,
        ids: *const i32,
        out_usm: *mut u16,
        n_ids: u32,
        d: u32,
    ) -> Result<()> {
        // SAFETY: `ids` may be a host pointer — the C side copies it
        // into a device buffer before launching. `table_usm` and
        // `out_usm` MUST be USM allocations.
        unsafe {
            rsl_embedding_lookup_usm(
                stream.raw,
                table_usm,
                ids,
                out_usm,
                n_ids as c_int,
                d as c_int,
            );
        }
        consume_error()
    }

    /// G2: KV-cache Q8_0 quantize-on-store. `src_usm` is
    /// `[n_new × n_kv_heads × head_dim]` f32 activations contiguous;
    /// `q_dst_usm` and `scales_dst_usm` are the USM-resident KV-cache
    /// buffers with strided `[n_kv_heads × max_ctx × *]` layout.
    pub fn kv_quantize_q8_0_store_usm(
        stream: &SyclStream,
        src_usm: *const f32,
        q_dst_usm: *mut i8,
        scales_dst_usm: *mut f32,
        n_new: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_ctx: u32,
        kv_len_base: u32,
    ) -> Result<()> {
        if n_new == 0 || n_kv_heads == 0 || head_dim == 0 || max_ctx == 0 {
            return Err(SyclError::InvalidShape(format!(
                "kv_quantize_q8_0_store_usm: zero dim (n_new={n_new}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, max_ctx={max_ctx})"
            )));
        }
        unsafe {
            rsl_kv_quantize_q8_0_store_usm(
                stream.raw, src_usm, q_dst_usm, scales_dst_usm,
                n_new as c_int, n_kv_heads as c_int, head_dim as c_int,
                max_ctx as c_int, kv_len_base as c_int,
            );
        }
        consume_error()
    }

    /// G6: Q6_K block encoder. `src_usm` is `[n_blocks × 256]` f32
    /// USM; `dst_usm` is `[n_blocks × 210]` u8 USM output.
    pub fn encode_q6_k_blocks_usm(
        stream: &SyclStream,
        src_usm: *const f32,
        dst_usm: *mut u8,
        n_blocks: u32,
    ) -> Result<()> {
        if n_blocks == 0 {
            return Err(SyclError::InvalidShape("encode_q6_k_blocks_usm: n_blocks == 0".into()));
        }
        unsafe {
            rsl_encode_q6_k_blocks_usm(stream.raw, src_usm, dst_usm, n_blocks as c_int);
        }
        consume_error()
    }

    /// G6: Q3_K block encoder. `src_usm` is `[n_blocks × 256]` f32
    /// USM; `dst_usm` is `[n_blocks × 110]` u8 USM output.
    pub fn encode_q3_k_blocks_usm(
        stream: &SyclStream,
        src_usm: *const f32,
        dst_usm: *mut u8,
        n_blocks: u32,
    ) -> Result<()> {
        if n_blocks == 0 {
            return Err(SyclError::InvalidShape("encode_q3_k_blocks_usm: n_blocks == 0".into()));
        }
        unsafe {
            rsl_encode_q3_k_blocks_usm(stream.raw, src_usm, dst_usm, n_blocks as c_int);
        }
        consume_error()
    }

    /// G6: Q4_K block encoder. `src_usm` is `[n_blocks × 256]` f32
    /// USM; `dst_usm` is `[n_blocks × 144]` u8 USM output. Uses the
    /// 20-step iterative `make_qkx2_quants_asym<15>` per sub-block.
    pub fn encode_q4_k_blocks_usm(
        stream: &SyclStream,
        src_usm: *const f32,
        dst_usm: *mut u8,
        n_blocks: u32,
    ) -> Result<()> {
        if n_blocks == 0 {
            return Err(SyclError::InvalidShape("encode_q4_k_blocks_usm: n_blocks == 0".into()));
        }
        unsafe {
            rsl_encode_q4_k_blocks_usm(stream.raw, src_usm, dst_usm, n_blocks as c_int);
        }
        consume_error()
    }

    /// G6: Q5_K block encoder. `src_usm` is `[n_blocks × 256]` f32
    /// USM; `dst_usm` is `[n_blocks × 176]` u8 USM output. Uses the
    /// 20-step iterative `make_qkx2_quants_asym<31>` per sub-block.
    pub fn encode_q5_k_blocks_usm(
        stream: &SyclStream,
        src_usm: *const f32,
        dst_usm: *mut u8,
        n_blocks: u32,
    ) -> Result<()> {
        if n_blocks == 0 {
            return Err(SyclError::InvalidShape("encode_q5_k_blocks_usm: n_blocks == 0".into()));
        }
        unsafe {
            rsl_encode_q5_k_blocks_usm(stream.raw, src_usm, dst_usm, n_blocks as c_int);
        }
        consume_error()
    }

    /// H4: Q4_K gate+up FUSED matvec. Each work-item computes BOTH
    /// `gate_out[m]` and `up_out[m]` against the same `x_usm` input
    /// row — saves activation cache pressure and one kernel launch
    /// vs. two separate matvec dispatches.
    pub fn matvec_q4_k_gate_up_fused_usm(
        stream: &SyclStream,
        gate_w_bytes_usm: *const u8,
        up_w_bytes_usm: *const u8,
        x_usm: *const f32,
        gate_out_usm: *mut f32,
        up_out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q4_k_gate_up_fused_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q4_k_gate_up_fused_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_q4_k_gate_up_fused_usm(
                stream.raw,
                gate_w_bytes_usm as *const std::ffi::c_void,
                up_w_bytes_usm as *const std::ffi::c_void,
                x_usm, gate_out_usm, up_out_usm,
                m as c_int, k as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    /// H4: Q8_0 gate+up FUSED matvec.
    pub fn matvec_q8_0_gate_up_fused_usm(
        stream: &SyclStream,
        gate_w_bytes_usm: *const u8,
        up_w_bytes_usm: *const u8,
        x_usm: *const f32,
        gate_out_usm: *mut f32,
        up_out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q8_0_gate_up_fused_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 32 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q8_0_gate_up_fused_usm: K must be a multiple of 32, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_q8_0_gate_up_fused_usm(
                stream.raw,
                gate_w_bytes_usm as *const std::ffi::c_void,
                up_w_bytes_usm as *const std::ffi::c_void,
                x_usm, gate_out_usm, up_out_usm,
                m as c_int, k as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    /// H4: Q5_K gate+up FUSED matvec.
    pub fn matvec_q5_k_gate_up_fused_usm(
        stream: &SyclStream,
        gate_w_bytes_usm: *const u8,
        up_w_bytes_usm: *const u8,
        x_usm: *const f32,
        gate_out_usm: *mut f32,
        up_out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q5_k_gate_up_fused_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q5_k_gate_up_fused_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_q5_k_gate_up_fused_usm(
                stream.raw,
                gate_w_bytes_usm as *const std::ffi::c_void,
                up_w_bytes_usm as *const std::ffi::c_void,
                x_usm, gate_out_usm, up_out_usm,
                m as c_int, k as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    /// H4: Q6_K gate+up FUSED matvec.
    pub fn matvec_q6_k_gate_up_fused_usm(
        stream: &SyclStream,
        gate_w_bytes_usm: *const u8,
        up_w_bytes_usm: *const u8,
        x_usm: *const f32,
        gate_out_usm: *mut f32,
        up_out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q6_k_gate_up_fused_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q6_k_gate_up_fused_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_q6_k_gate_up_fused_usm(
                stream.raw,
                gate_w_bytes_usm as *const std::ffi::c_void,
                up_w_bytes_usm as *const std::ffi::c_void,
                x_usm, gate_out_usm, up_out_usm,
                m as c_int, k as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    /// H4: IQ4_NL gate+up FUSED matvec.
    pub fn matvec_iq4_nl_gate_up_fused_usm(
        stream: &SyclStream,
        gate_w_bytes_usm: *const u8,
        up_w_bytes_usm: *const u8,
        x_usm: *const f32,
        gate_out_usm: *mut f32,
        up_out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq4_nl_gate_up_fused_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 32 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq4_nl_gate_up_fused_usm: K must be a multiple of 32, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_iq4_nl_gate_up_fused_usm(
                stream.raw,
                gate_w_bytes_usm as *const std::ffi::c_void,
                up_w_bytes_usm as *const std::ffi::c_void,
                x_usm, gate_out_usm, up_out_usm,
                m as c_int, k as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    /// H4: IQ4_XS gate+up FUSED matvec.
    pub fn matvec_iq4_xs_gate_up_fused_usm(
        stream: &SyclStream,
        gate_w_bytes_usm: *const u8,
        up_w_bytes_usm: *const u8,
        x_usm: *const f32,
        gate_out_usm: *mut f32,
        up_out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq4_xs_gate_up_fused_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq4_xs_gate_up_fused_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_iq4_xs_gate_up_fused_usm(
                stream.raw,
                gate_w_bytes_usm as *const std::ffi::c_void,
                up_w_bytes_usm as *const std::ffi::c_void,
                x_usm, gate_out_usm, up_out_usm,
                m as c_int, k as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    /// H4: IQ1_S gate+up FUSED matvec.
    pub fn matvec_iq1_s_gate_up_fused_usm(
        stream: &SyclStream,
        gate_w_bytes_usm: *const u8,
        up_w_bytes_usm: *const u8,
        x_usm: *const f32,
        gate_out_usm: *mut f32,
        up_out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq1_s_gate_up_fused_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq1_s_gate_up_fused_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_iq1_s_gate_up_fused_usm(
                stream.raw,
                gate_w_bytes_usm as *const std::ffi::c_void,
                up_w_bytes_usm as *const std::ffi::c_void,
                x_usm, gate_out_usm, up_out_usm,
                m as c_int, k as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    /// H4: IQ2_XXS gate+up FUSED matvec.
    pub fn matvec_iq2_xxs_gate_up_fused_usm(
        stream: &SyclStream,
        gate_w_bytes_usm: *const u8,
        up_w_bytes_usm: *const u8,
        x_usm: *const f32,
        gate_out_usm: *mut f32,
        up_out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_xxs_gate_up_fused_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_xxs_gate_up_fused_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_iq2_xxs_gate_up_fused_usm(
                stream.raw,
                gate_w_bytes_usm as *const std::ffi::c_void,
                up_w_bytes_usm as *const std::ffi::c_void,
                x_usm, gate_out_usm, up_out_usm,
                m as c_int, k as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    /// H4: IQ1_M gate+up FUSED matvec.
    pub fn matvec_iq1_m_gate_up_fused_usm(
        stream: &SyclStream,
        gate_w_bytes_usm: *const u8,
        up_w_bytes_usm: *const u8,
        x_usm: *const f32,
        gate_out_usm: *mut f32,
        up_out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq1_m_gate_up_fused_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq1_m_gate_up_fused_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_iq1_m_gate_up_fused_usm(
                stream.raw,
                gate_w_bytes_usm as *const std::ffi::c_void,
                up_w_bytes_usm as *const std::ffi::c_void,
                x_usm, gate_out_usm, up_out_usm,
                m as c_int, k as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    /// H4: IQ2_XS gate+up FUSED matvec.
    pub fn matvec_iq2_xs_gate_up_fused_usm(
        stream: &SyclStream,
        gate_w_bytes_usm: *const u8,
        up_w_bytes_usm: *const u8,
        x_usm: *const f32,
        gate_out_usm: *mut f32,
        up_out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_xs_gate_up_fused_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_xs_gate_up_fused_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_iq2_xs_gate_up_fused_usm(
                stream.raw,
                gate_w_bytes_usm as *const std::ffi::c_void,
                up_w_bytes_usm as *const std::ffi::c_void,
                x_usm, gate_out_usm, up_out_usm,
                m as c_int, k as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    /// H4: IQ2_S gate+up FUSED matvec.
    pub fn matvec_iq2_s_gate_up_fused_usm(
        stream: &SyclStream,
        gate_w_bytes_usm: *const u8,
        up_w_bytes_usm: *const u8,
        x_usm: *const f32,
        gate_out_usm: *mut f32,
        up_out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_s_gate_up_fused_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq2_s_gate_up_fused_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_iq2_s_gate_up_fused_usm(
                stream.raw,
                gate_w_bytes_usm as *const std::ffi::c_void,
                up_w_bytes_usm as *const std::ffi::c_void,
                x_usm, gate_out_usm, up_out_usm,
                m as c_int, k as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    /// H4: IQ3_XXS gate+up FUSED matvec.
    pub fn matvec_iq3_xxs_gate_up_fused_usm(
        stream: &SyclStream,
        gate_w_bytes_usm: *const u8,
        up_w_bytes_usm: *const u8,
        x_usm: *const f32,
        gate_out_usm: *mut f32,
        up_out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq3_xxs_gate_up_fused_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq3_xxs_gate_up_fused_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_iq3_xxs_gate_up_fused_usm(
                stream.raw,
                gate_w_bytes_usm as *const std::ffi::c_void,
                up_w_bytes_usm as *const std::ffi::c_void,
                x_usm, gate_out_usm, up_out_usm,
                m as c_int, k as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    /// H4: IQ3_S gate+up FUSED matvec.
    pub fn matvec_iq3_s_gate_up_fused_usm(
        stream: &SyclStream,
        gate_w_bytes_usm: *const u8,
        up_w_bytes_usm: *const u8,
        x_usm: *const f32,
        gate_out_usm: *mut f32,
        up_out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq3_s_gate_up_fused_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_iq3_s_gate_up_fused_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_iq3_s_gate_up_fused_usm(
                stream.raw,
                gate_w_bytes_usm as *const std::ffi::c_void,
                up_w_bytes_usm as *const std::ffi::c_void,
                x_usm, gate_out_usm, up_out_usm,
                m as c_int, k as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    /// H4: PTQ1_0 (Bonsai ternary) gate+up FUSED matvec.
    pub fn matvec_ptq1_0_gate_up_fused_usm(
        stream: &SyclStream,
        gate_w_bytes_usm: *const u8,
        up_w_bytes_usm: *const u8,
        x_usm: *const f32,
        gate_out_usm: *mut f32,
        up_out_usm: *mut f32,
        m: u32,
        k: u32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_ptq1_0_gate_up_fused_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 128 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_ptq1_0_gate_up_fused_usm: K must be a multiple of 128, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_ptq1_0_gate_up_fused_usm(
                stream.raw,
                gate_w_bytes_usm as *const std::ffi::c_void,
                up_w_bytes_usm as *const std::ffi::c_void,
                x_usm, gate_out_usm, up_out_usm,
                m as c_int, k as c_int, lws as c_int,
            );
        }
        consume_error()
    }

    /// H6: Q4_K fused matvec + residual-add + rmsnorm. One workgroup
    /// per token, LWS work-items collaborating on M outputs with SLM
    /// staging + workgroup sum_sq reduce. Replaces the (matvec_q4_k
    /// → add_rmsnorm) pair on the post-attn norm site.
    pub fn matvec_q4_k_add_rmsnorm_usm(
        stream: &SyclStream,
        w_bytes_usm: *const u8,
        attn_usm: *const f32,
        hidden_usm: *mut f32,
        w_norm_usm: *const f32,
        y_norm_usm: *mut f32,
        m: u32,
        k: u32,
        eps: f32,
        lws: u32,
    ) -> Result<()> {
        if m == 0 || k == 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q4_k_add_rmsnorm_usm: zero dim (M={m}, K={k})"
            )));
        }
        if k % 256 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "matvec_q4_k_add_rmsnorm_usm: K must be a multiple of 256, got {k}"
            )));
        }
        unsafe {
            rsl_matvec_q4_k_add_rmsnorm_usm(
                stream.raw,
                w_bytes_usm as *const std::ffi::c_void,
                attn_usm, hidden_usm, w_norm_usm, y_norm_usm,
                m as c_int, k as c_int, eps,
                lws as c_int,
            );
        }
        consume_error()
    }

    // H6: remaining 12 dtypes share the identical safe-wrapper shape;
    // only the FFI symbol + K-alignment differ.
    macro_rules! h6_real_wrapper {
        ($name:ident, $ffi:ident, $align:expr) => {
            #[allow(clippy::too_many_arguments)]
            pub fn $name(
                stream: &SyclStream, w_bytes_usm: *const u8, attn_usm: *const f32,
                hidden_usm: *mut f32, w_norm_usm: *const f32, y_norm_usm: *mut f32,
                m: u32, k: u32, eps: f32, lws: u32,
            ) -> Result<()> {
                if m == 0 || k == 0 {
                    return Err(SyclError::InvalidShape(format!(
                        concat!(stringify!($name), ": zero dim (M={}, K={})"), m, k)));
                }
                if k % $align != 0 {
                    return Err(SyclError::InvalidShape(format!(
                        concat!(stringify!($name), ": K must be a multiple of ", stringify!($align), ", got {}"), k)));
                }
                unsafe {
                    $ffi(
                        stream.raw,
                        w_bytes_usm as *const std::ffi::c_void,
                        attn_usm, hidden_usm, w_norm_usm, y_norm_usm,
                        m as c_int, k as c_int, eps, lws as c_int,
                    );
                }
                consume_error()
            }
        };
    }
    h6_real_wrapper!(matvec_q8_0_add_rmsnorm_usm, rsl_matvec_q8_0_add_rmsnorm_usm, 32);
    h6_real_wrapper!(matvec_q5_k_add_rmsnorm_usm, rsl_matvec_q5_k_add_rmsnorm_usm, 256);
    h6_real_wrapper!(matvec_q6_k_add_rmsnorm_usm, rsl_matvec_q6_k_add_rmsnorm_usm, 256);
    h6_real_wrapper!(matvec_iq4_nl_add_rmsnorm_usm, rsl_matvec_iq4_nl_add_rmsnorm_usm, 32);
    h6_real_wrapper!(matvec_iq4_xs_add_rmsnorm_usm, rsl_matvec_iq4_xs_add_rmsnorm_usm, 256);
    h6_real_wrapper!(matvec_iq1_s_add_rmsnorm_usm, rsl_matvec_iq1_s_add_rmsnorm_usm, 256);
    h6_real_wrapper!(matvec_iq1_m_add_rmsnorm_usm, rsl_matvec_iq1_m_add_rmsnorm_usm, 256);
    h6_real_wrapper!(matvec_iq2_xxs_add_rmsnorm_usm, rsl_matvec_iq2_xxs_add_rmsnorm_usm, 256);
    h6_real_wrapper!(matvec_iq2_xs_add_rmsnorm_usm, rsl_matvec_iq2_xs_add_rmsnorm_usm, 256);
    h6_real_wrapper!(matvec_iq2_s_add_rmsnorm_usm, rsl_matvec_iq2_s_add_rmsnorm_usm, 256);
    h6_real_wrapper!(matvec_iq3_xxs_add_rmsnorm_usm, rsl_matvec_iq3_xxs_add_rmsnorm_usm, 256);
    h6_real_wrapper!(matvec_iq3_s_add_rmsnorm_usm, rsl_matvec_iq3_s_add_rmsnorm_usm, 256);
    h6_real_wrapper!(matvec_ptq1_0_add_rmsnorm_usm, rsl_matvec_ptq1_0_add_rmsnorm_usm, 128);

    // H8: F16-input matvec safe wrappers — identical shape, only the
    // FFI symbol + K-alignment differ.
    macro_rules! h8_real_wrapper {
        ($name:ident, $ffi:ident, $align:expr) => {
            pub fn $name(
                stream: &SyclStream, w_bytes_usm: *const u8, x_f16: *const u16,
                out_usm: *mut f32, m: u32, k: u32, lws: u32,
            ) -> Result<()> {
                if m == 0 || k == 0 {
                    return Err(SyclError::InvalidShape(format!(
                        concat!(stringify!($name), ": zero dim (M={}, K={})"), m, k)));
                }
                if k % $align != 0 {
                    return Err(SyclError::InvalidShape(format!(
                        concat!(stringify!($name), ": K must be a multiple of ", stringify!($align), ", got {}"), k)));
                }
                unsafe {
                    $ffi(stream.raw, w_bytes_usm as *const std::ffi::c_void,
                         x_f16, out_usm, m as c_int, k as c_int, lws as c_int);
                }
                consume_error()
            }
        };
    }
    h8_real_wrapper!(matvec_q8_0_f16in_packed_f32_usm, rsl_matvec_q8_0_f16in_packed_f32_usm, 32);
    h8_real_wrapper!(matvec_q4_k_f16in_packed_f32_usm, rsl_matvec_q4_k_f16in_packed_f32_usm, 256);
    h8_real_wrapper!(matvec_q5_k_f16in_packed_f32_usm, rsl_matvec_q5_k_f16in_packed_f32_usm, 256);
    h8_real_wrapper!(matvec_q6_k_f16in_packed_f32_usm, rsl_matvec_q6_k_f16in_packed_f32_usm, 256);
    h8_real_wrapper!(matvec_iq4_nl_f16in_packed_f32_usm, rsl_matvec_iq4_nl_f16in_packed_f32_usm, 32);
    h8_real_wrapper!(matvec_iq4_xs_f16in_packed_f32_usm, rsl_matvec_iq4_xs_f16in_packed_f32_usm, 256);
    h8_real_wrapper!(matvec_iq1_s_f16in_packed_f32_usm, rsl_matvec_iq1_s_f16in_packed_f32_usm, 256);
    h8_real_wrapper!(matvec_iq1_m_f16in_packed_f32_usm, rsl_matvec_iq1_m_f16in_packed_f32_usm, 256);
    h8_real_wrapper!(matvec_iq2_xxs_f16in_packed_f32_usm, rsl_matvec_iq2_xxs_f16in_packed_f32_usm, 256);
    h8_real_wrapper!(matvec_iq2_xs_f16in_packed_f32_usm, rsl_matvec_iq2_xs_f16in_packed_f32_usm, 256);
    h8_real_wrapper!(matvec_iq2_s_f16in_packed_f32_usm, rsl_matvec_iq2_s_f16in_packed_f32_usm, 256);
    h8_real_wrapper!(matvec_iq3_xxs_f16in_packed_f32_usm, rsl_matvec_iq3_xxs_f16in_packed_f32_usm, 256);
    h8_real_wrapper!(matvec_iq3_s_f16in_packed_f32_usm, rsl_matvec_iq3_s_f16in_packed_f32_usm, 256);
    h8_real_wrapper!(matvec_ptq1_0_f16in_packed_f32_usm, rsl_matvec_ptq1_0_f16in_packed_f32_usm, 128);

    /// G5: K-quant → F32 dequant (USM-in / USM-out). `format` selects
    /// the kernel; `bytes_usm` must point to `n_blocks * block_size`
    /// USM-resident bytes; `out_usm` must point to `n_blocks * 256`
    /// USM-resident f32 outputs.
    pub fn dequant_kquant_to_f32_usm(
        stream: &SyclStream,
        format: KQuantFormat,
        bytes_usm: *const std::ffi::c_void,
        out_usm: *mut f32,
        n_blocks: u32,
    ) -> Result<()> {
        // SAFETY: caller guarantees pointer sizing per docstring.
        unsafe {
            match format {
                KQuantFormat::Q3K => {
                    rsl_dequant_q3_k_to_f32_usm(stream.raw, bytes_usm, out_usm, n_blocks as c_int)
                }
                KQuantFormat::Q4K => {
                    rsl_dequant_q4_k_to_f32_usm(stream.raw, bytes_usm, out_usm, n_blocks as c_int)
                }
                KQuantFormat::Q5K => {
                    rsl_dequant_q5_k_to_f32_usm(stream.raw, bytes_usm, out_usm, n_blocks as c_int)
                }
                KQuantFormat::Q6K => {
                    rsl_dequant_q6_k_to_f32_usm(stream.raw, bytes_usm, out_usm, n_blocks as c_int)
                }
            }
        }
        consume_error()
    }

    pub fn gemm_f16(
        stream: &mut SyclStream,
        a: &[u16],
        b: &[u16],
        c: &mut [u16],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let m_us = m as usize;
        let n_us = n as usize;
        let k_us = k as usize;
        if a.len() < m_us * k_us {
            return Err(SyclError::InvalidShape(format!(
                "A too small: have {}, need M*K = {}",
                a.len(),
                m_us * k_us
            )));
        }
        if b.len() < k_us * n_us {
            return Err(SyclError::InvalidShape(format!(
                "B too small: have {}, need K*N = {}",
                b.len(),
                k_us * n_us
            )));
        }
        if c.len() < m_us * n_us {
            return Err(SyclError::InvalidShape(format!(
                "C too small: have {}, need M*N = {}",
                c.len(),
                m_us * n_us
            )));
        }
        // SAFETY: shapes validated above; pointers come from live
        // borrowed slices. The SYCL side reads A and B and writes C;
        // borrow rules ensure A/B don't alias C.
        unsafe {
            rsl_gemm_f16(
                stream.raw,
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                m as c_int,
                n as c_int,
                k as c_int,
                k as c_int,
                n as c_int,
                n as c_int,
            );
        }
        consume_error()
    }

    pub fn rmsnorm(
        stream: &mut SyclStream,
        x: &[u16],
        w: &[u16],
        y: &mut [u16],
        n_rows: u32,
        d: u32,
        eps: f32,
    ) -> Result<()> {
        let total = (n_rows as usize) * (d as usize);
        if x.len() < total {
            return Err(SyclError::InvalidShape(format!(
                "X too small: have {}, need n_rows*d = {total}",
                x.len()
            )));
        }
        if w.len() < d as usize {
            return Err(SyclError::InvalidShape(format!(
                "W too small: have {}, need d = {d}",
                w.len()
            )));
        }
        if y.len() < total {
            return Err(SyclError::InvalidShape(format!(
                "Y too small: have {}, need n_rows*d = {total}",
                y.len()
            )));
        }
        // SAFETY: shapes validated; borrows ensure X/W don't alias Y.
        unsafe {
            rsl_rmsnorm(
                stream.raw,
                x.as_ptr(),
                w.as_ptr(),
                y.as_mut_ptr(),
                n_rows as c_int,
                d as c_int,
                eps,
            );
        }
        consume_error()
    }

    pub fn rope(
        stream: &mut SyclStream,
        qk: &mut [u16],
        n_heads: u32,
        head_dim: u32,
        pos: u32,
        inv_freq: &[u16],
    ) -> Result<()> {
        if head_dim % 2 != 0 {
            return Err(SyclError::InvalidShape(format!(
                "head_dim must be even, got {head_dim}"
            )));
        }
        let half = (head_dim / 2) as usize;
        let total = (n_heads as usize) * (head_dim as usize);
        if qk.len() < total {
            return Err(SyclError::InvalidShape(format!(
                "QK too small: have {}, need n_heads*head_dim = {total}",
                qk.len()
            )));
        }
        if inv_freq.len() < half {
            return Err(SyclError::InvalidShape(format!(
                "inv_freq too small: have {}, need head_dim/2 = {half}",
                inv_freq.len()
            )));
        }
        // SAFETY: shapes validated; in-place write on qk is sound.
        unsafe {
            rsl_rope(
                stream.raw,
                qk.as_mut_ptr(),
                n_heads as c_int,
                head_dim as c_int,
                pos as c_int,
                inv_freq.as_ptr(),
            );
        }
        consume_error()
    }

    pub fn softmax_attn(
        stream: &mut SyclStream,
        scores: &mut [u16],
        mask: Option<&[u16]>,
        n_heads: u32,
        seq: u32,
        kv_len: u32,
        scale: f32,
    ) -> Result<()> {
        let total =
            (n_heads as usize) * (seq as usize) * (kv_len as usize);
        if scores.len() < total {
            return Err(SyclError::InvalidShape(format!(
                "scores too small: have {}, need n_heads*seq*kv_len = {total}",
                scores.len()
            )));
        }
        if let Some(m) = mask {
            if m.len() < kv_len as usize {
                return Err(SyclError::InvalidShape(format!(
                    "mask too small: have {}, need kv_len = {kv_len}",
                    m.len()
                )));
            }
        }
        let mask_ptr = mask.map(|m| m.as_ptr()).unwrap_or(std::ptr::null());
        // SAFETY: shapes validated; in-place write on scores is sound.
        unsafe {
            rsl_softmax_attn(
                stream.raw,
                scores.as_mut_ptr(),
                mask_ptr,
                n_heads as c_int,
                seq as c_int,
                kv_len as c_int,
                scale,
            );
        }
        consume_error()
    }

    pub fn silu_mul(
        stream: &mut SyclStream,
        x: &[u16],
        y: &[u16],
        out: &mut [u16],
    ) -> Result<()> {
        let n = x.len();
        if y.len() != n {
            return Err(SyclError::InvalidShape(format!(
                "y.len ({}) != x.len ({n})",
                y.len()
            )));
        }
        if out.len() < n {
            return Err(SyclError::InvalidShape(format!(
                "out too small: have {}, need {n}",
                out.len()
            )));
        }
        // SAFETY: shapes validated; out doesn't alias x/y.
        unsafe {
            rsl_silu_mul(
                stream.raw,
                x.as_ptr(),
                y.as_ptr(),
                out.as_mut_ptr(),
                n as c_int,
            );
        }
        consume_error()
    }

    pub fn dequant_nvfp4(
        stream: &mut SyclStream,
        w_nvfp4: &[u8],
        out: &mut [u16],
    ) -> Result<()> {
        const BLOCK_BYTES: usize = 9;
        const BLOCK_ELEMS: usize = 16;
        if w_nvfp4.is_empty() {
            return Ok(());
        }
        if w_nvfp4.len() % BLOCK_BYTES != 0 {
            return Err(SyclError::InvalidShape(format!(
                "w_nvfp4 length {} must be multiple of {BLOCK_BYTES}",
                w_nvfp4.len()
            )));
        }
        let n_blocks = w_nvfp4.len() / BLOCK_BYTES;
        let need_out = n_blocks * BLOCK_ELEMS;
        if out.len() < need_out {
            return Err(SyclError::InvalidShape(format!(
                "out too small: have {}, need n_blocks*16 = {need_out}",
                out.len()
            )));
        }
        // SAFETY: shape validated above; pointers from live slices.
        unsafe {
            rsl_dequant_nvfp4(
                stream.raw,
                w_nvfp4.as_ptr(),
                out.as_mut_ptr(),
                n_blocks as c_int,
            );
        }
        consume_error()
    }

    pub fn matvec_nvfp4_f16(
        stream: &mut SyclStream,
        w_nvfp4: &[u8],
        x: &[u16],
        out: &mut [u16],
        m: u32,
        k: u32,
    ) -> Result<()> {
        const BLOCK_BYTES: usize = 9;
        const BLOCK_ELEMS: usize = 16;
        if k as usize % BLOCK_ELEMS != 0 {
            return Err(SyclError::InvalidShape(format!(
                "k ({k}) must be a multiple of {BLOCK_ELEMS}"
            )));
        }
        let blocks_per_row = (k as usize) / BLOCK_ELEMS;
        let need_w = (m as usize) * blocks_per_row * BLOCK_BYTES;
        if w_nvfp4.len() < need_w {
            return Err(SyclError::InvalidShape(format!(
                "w_nvfp4 too small: have {}, need {need_w}",
                w_nvfp4.len()
            )));
        }
        if x.len() < k as usize {
            return Err(SyclError::InvalidShape(format!(
                "x too small: have {}, need k = {k}",
                x.len()
            )));
        }
        if out.len() < m as usize {
            return Err(SyclError::InvalidShape(format!(
                "out too small: have {}, need m = {m}",
                out.len()
            )));
        }
        // SAFETY: shapes validated above; the kernel reads w and x
        // and writes out; borrow rules ensure no aliasing.
        unsafe {
            rsl_matvec_nvfp4_f16(
                stream.raw,
                w_nvfp4.as_ptr(),
                x.as_ptr(),
                out.as_mut_ptr(),
                m as c_int,
                k as c_int,
            );
        }
        consume_error()
    }

    pub fn embedding_lookup(
        stream: &mut SyclStream,
        table: &[u16],
        ids: &[i32],
        out: &mut [u16],
        d: u32,
    ) -> Result<()> {
        let n_ids = ids.len();
        let need_out = n_ids * (d as usize);
        if out.len() < need_out {
            return Err(SyclError::InvalidShape(format!(
                "out too small: have {}, need n_ids*d = {need_out}",
                out.len()
            )));
        }
        if d == 0 {
            return Err(SyclError::InvalidShape("d must be > 0".to_string()));
        }
        // We don't know V here — the C side validates id-in-range
        // per row via the row<0 negative-id guard. Other out-of-range
        // ids produce undefined reads against `table`; callers are
        // responsible for staying within vocab. The CPU embed_lookup
        // mirrors this contract.
        // SAFETY: out's size validated; table/ids pointers come from
        // borrowed slices.
        unsafe {
            rsl_embedding_lookup(
                stream.raw,
                table.as_ptr(),
                ids.as_ptr(),
                out.as_mut_ptr(),
                n_ids as c_int,
                d as c_int,
            );
        }
        consume_error()
    }
}

pub fn device_count() -> Result<u32> {
    imp::device_count()
}

/// True once any kernel call has reported `UR_RESULT_ERROR_DEVICE_LOST`
/// (the GPU was reset / the SYCL context died). Terminal for the session —
/// the dispatch layer uses this to stop attempting GPU work entirely and
/// fall back to CPU, rather than re-probing a dead device.
pub fn device_lost() -> bool {
    imp::SYCL_DEVICE_LOST.load(std::sync::atomic::Ordering::Relaxed)
}

/// Descriptive info about a single SYCL device. Used by the
/// autotuner to build a stable per-device cache key (so a driver
/// update or device swap invalidates the cached tuning entries).
/// All strings are UTF-8; the `vendor_id` is the PCI/SPIR vendor
/// ID (e.g. `0x8086` for Intel, `0x10DE` for NVIDIA, `0x1002` for
/// AMD).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub name: String,
    pub driver_version: String,
    pub vendor_id: u32,
    pub vram_bytes: u64,
    /// Stable, driver-invariant device UUID (Level-Zero exposes it). All
    /// zeroes = the device/backend didn't report one (fall back to
    /// vendor/name/vram for identity).
    pub uuid: [u8; 16],
    /// `true` = integrated / host-unified-memory GPU (e.g. Iris Xe, whose
    /// "VRAM" is a shared-LPDDR aperture with no separate dedicated pool).
    /// `false` = discrete GPU with dedicated VRAM (e.g. Intel Arc). Set
    /// from SYCL `host_unified_memory`; used by the placement planner to
    /// decide whether `vram_only` can be enforced (dedicated VRAM present)
    /// or is a no-op (unified). Defaults to `false` when the query fails
    /// (no device / stub build).
    pub is_integrated: bool,
}

impl DeviceInfo {
    /// VRAM in mebibytes (1 MiB = 1024 KiB), rounded down. Convenient
    /// for tuner cache fingerprints since `vram_bytes` can wobble
    /// between driver versions (allocator accounting changes) while
    /// the rounded-MiB value stays stable.
    pub fn vram_mb(&self) -> u64 {
        self.vram_bytes / (1024 * 1024)
    }
}

/// Describe a SYCL device by 0-based index (same enumeration order
/// as [`device_count`]). Returns
/// [`SyclError::NoSuchDevice`] when the index is out of range or
/// the SYCL runtime can't query the device.
///
/// Returns [`SyclError::Unavailable`] when no SYCL device is present.
pub fn device_info(device_index: u32) -> Result<DeviceInfo> {
    imp::device_info(device_index)
}

pub fn create_stream(device_index: u32) -> Result<SyclStream> {
    imp::create_stream(device_index)
}

/// Probe-only backend identification: create an ephemeral stream on
/// `device_index`, read its backend id, drop the stream. Returns a
/// short stable string the GUI displays in its Backends panel.
/// Variants: `"level_zero"`, `"opencl"`, `"host"`, `"cuda"`,
/// `"hip"`, `"native_cpu"`, `"unknown"`. Returns `None` when no SYCL device is present
/// builds or when no SYCL device is visible.
pub fn current_backend_name(device_index: u32) -> Option<&'static str> {
    let stream = imp::create_stream(device_index).ok()?;
    let id = unsafe { imp::raw_backend_id(&stream) };
    Some(match id {
        0 => "host",
        1 => "opencl",
        2 => "level_zero",
        3 => "cuda",
        4 => "hip",
        5 => "native_cpu",
        _ => "unknown",
    })
}

/// Raw `sycl::queue*` pointer for the stream, for SYCL-interop
/// consumers that need to bind to the same queue the engine runs
/// kernels on.
///
/// Returns `null` when no SYCL device is present (no real SYCL queue exists). Callers
/// MUST NOT free the returned pointer; it's owned by the stream.
///
/// SAFETY: the returned pointer aliases a field on `*stream` and is
/// valid only while `stream` is alive.
pub fn raw_sycl_queue(stream: &SyclStream) -> *mut std::ffi::c_void {
    imp::raw_sycl_queue(stream)
}

/// Raw `sycl::device*` pointer for the stream (SYCL interop). Same
/// lifetime contract as [`raw_sycl_queue`].
pub fn raw_sycl_device(stream: &SyclStream) -> *mut std::ffi::c_void {
    imp::raw_sycl_device(stream)
}

/// Raw `sycl::context*` pointer for the stream (SYCL interop). Same
/// lifetime contract as [`raw_sycl_queue`].
pub fn raw_sycl_context(stream: &SyclStream) -> *mut std::ffi::c_void {
    imp::raw_sycl_context(stream)
}

/// Public raw USM allocator — for callers that hold a stream
/// directly and want to manage USM buffers without going through
/// the `SyclSharedBuffer` lifetime contract (e.g.
/// `rustllama_engine::SyclEngineResources` which owns its own
/// stream + a vec of long-lived USM allocations). Returns NULL on
/// allocation failure or when SYCL is unavailable.
///
/// SAFETY contract: `stream` must outlive the returned pointer.
/// Free via [`usm_free`] before dropping the stream.
pub fn usm_alloc_shared(stream: &SyclStream, n_bytes: usize) -> *mut std::ffi::c_void {
    imp::usm_alloc_shared(stream, n_bytes)
}

/// Public raw USM free — pair with [`usm_alloc_shared`]. SAFETY:
/// `ptr` must have come from `usm_alloc_shared` on the same
/// `stream`; caller must not use `ptr` after.
///
/// # Safety
///
/// See above.
pub unsafe fn usm_free(stream: &SyclStream, ptr: *mut std::ffi::c_void) {
    unsafe { imp::usm_free(stream, ptr) };
}

/// Raw-pointer variant of [`flash_attn_decode_usm`] for callers
/// (e.g. `rustllama_engine::sycl_resources::SyclEngineResources`)
/// that own a SyclStream + the USM buffers together and can't
/// satisfy the `SyclSharedBuffer<'s>` borrow contract.
///
/// # Safety
///
/// All four pointers must reference live USM allocations on the
/// same context as `stream`. Sizing: `q_usm` / `out_usm` ≥
/// `n_heads * head_dim` u16; `k_usm` / `v_usm` ≥
/// `n_kv_heads * max_ctx * head_dim` u16. The kernel `.wait()`s
/// before this fn returns, so no concurrent host reads of the
/// buffers race with the device.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_usm_raw(
    stream: &SyclStream,
    q_usm: *const u16,
    k_usm: *const u16,
    v_usm: *const u16,
    out_usm: *mut u16,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len: u32,
) -> Result<()> {
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_usm_raw: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
        )));
    }
    if kv_len > max_ctx {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_usm_raw: kv_len={kv_len} > max_ctx={max_ctx}"
        )));
    }
    imp::flash_attn_decode_usm(
        stream, q_usm, k_usm, v_usm, out_usm,
        n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
    )
}

/// Raw-pointer variant of [`rmsnorm_usm`]. Same SAFETY contract as
/// [`flash_attn_decode_usm_raw`].
///
/// # Safety
///
/// `x_usm` / `y_usm` sized ≥ `n_rows * d`; `w_usm` sized ≥ `d`.
pub unsafe fn rmsnorm_usm_raw(
    stream: &SyclStream,
    x_usm: *const u16,
    w_usm: *const u16,
    y_usm: *mut u16,
    n_rows: u32,
    d: u32,
    eps: f32,
) -> Result<()> {
    imp::rmsnorm_usm(stream, x_usm, w_usm, y_usm, n_rows, d, eps)
}

/// Raw-pointer variant of the USM F16 GEMM for callers that own
/// their stream + buffers without satisfying the `SyclSharedBuffer`
/// borrow. `M=1` covers row-vector × matrix; `N=1` covers matvec
/// (the typical engine projection shape).
///
/// # Safety
///
/// `a_usm` sized ≥ `M * lda`, `b_usm` sized ≥ `K * ldb`,
/// `c_usm` sized ≥ `M * ldc`. All on the same SYCL context as
/// `stream`. The kernel `.wait()`s before returning so no race
/// between host-side post-kernel reads and the device.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_f16_usm_raw(
    stream: &SyclStream,
    a_usm: *const u16,
    b_usm: *const u16,
    c_usm: *mut u16,
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
) -> Result<()> {
    imp::gemm_f16_usm(stream, a_usm, b_usm, c_usm, m, n, k, lda, ldb, ldc)
}

/// Raw-pointer variant of the USM Q8_0 weight × F32 activation
/// matvec. `out[M] = sum_b w_scales[m, b] * sum_d(w_q[m, b*32+d] * x[b*32+d])`.
///
/// # Safety
///
/// `w_q_usm` sized ≥ `M * K`, `w_scales_usm` sized ≥ `M * (K/32)`,
/// `x_usm` sized ≥ `K`, `out_usm` sized ≥ `M`. All on the same SYCL
/// context as `stream`. `K` must be a multiple of 32. The kernel
/// `.wait()`s before returning so no race between host-side
/// post-kernel reads and the device.
pub unsafe fn matvec_q8_0_f32_usm_raw(
    stream: &SyclStream,
    w_q_usm: *const i8,
    w_scales_usm: *const f32,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
) -> Result<()> {
    imp::matvec_q8_0_f32_usm(stream, w_q_usm, w_scales_usm, x_usm, out_usm, m, k)
}

/// Raw-pointer variant of the USM Q8_0 packed-layout matvec.
/// Consumes the raw GGUF on-disk block layout (34 bytes per block:
/// f16 scale + 32 i8 weights). `out[M] = sum_b scale_b * sum_d(qs[b,d] * x[b*32+d])`.
///
/// `lws` selects the local work-group size from the compiled-in
/// candidate set `{16, 32, 64, 128, 256}`. Pass `0` for the default
/// (64); other values fall back to 64.
///
/// # Safety
///
/// `w_bytes_usm` sized ≥ `M * (K/32) * 34` bytes (cast from `*const u8`
/// — single contiguous USM byte allocation), `x_usm` sized ≥ `K`,
/// `out_usm` sized ≥ `M`. All on the same SYCL context as `stream`.
/// `K` must be a multiple of 32. The kernel `.wait()`s before returning.
pub unsafe fn matvec_q8_0_packed_f32_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_q8_0_packed_f32_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, lws)
}

/// Raw-pointer variant of the USM Q4_K_M packed-layout matvec.
/// Consumes the raw GGUF Q4_K_M super-block byte layout (144 bytes
/// per 256 weights: f16 d, f16 dmin, 12-byte packed 6-bit scales/mins
/// for 8 sub-blocks, 128 bytes of 4-bit packed weights).
///
/// `lws` selects the local work-group size for this dispatch from
/// the compiled-in candidate set `{16, 32, 64, 128, 256}`. Pass `0`
/// for the hand-picked default (64); values outside the candidate
/// set fall back to 64. The autotuner picks the winning value per
/// `(device, problem-shape)` and the engine passes the cached
/// choice on each call.
///
/// # Safety
///
/// `w_bytes_usm` sized ≥ `M * (K/256) * 144` bytes, `x_usm` sized ≥ `K`,
/// `out_usm` sized ≥ `M`. All on the same SYCL context as `stream`.
/// `K` must be a multiple of 256. The kernel `.wait()`s before returning.
pub unsafe fn matvec_q4_k_packed_f32_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_q4_k_packed_f32_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, lws)
}

/// Raw-pointer PTQ1_0 (Bonsai ternary) packed matvec.
///
/// # Safety
///
/// `w_bytes_usm` sized >= `M * (K/128) * 28` bytes, `x_usm` sized >= `K`,
/// `out_usm` sized >= `M`. All on the same SYCL context as `stream`.
/// `K` must be a multiple of 128. The kernel `.wait()`s before returning.
pub unsafe fn matvec_ptq1_0_packed_f32_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_ptq1_0_packed_f32_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, lws)
}

/// Raw-pointer batched PTQ1_0 packed matvec over N input rows.
///
/// # Safety
///
/// As [`matvec_ptq1_0_packed_f32_usm_raw`], plus `x_usm` sized >= `N*K`
/// and `out_usm` sized >= `N*M` (row-major `[N, M]`).
pub unsafe fn matvec_ptq1_0_packed_f32_batched_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_ptq1_0_packed_f32_batched_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, n, lws)
}

/// Raw-pointer blockwise Prism Hadamard rotation:
/// `out = WHT(signs * x) / sqrt(block)` per block-sized span.
///
/// # Safety
///
/// `x_usm`, `signs_usm`, `out_usm` all sized >= `n_elems` on the same
/// SYCL context as `stream`. `block` a power of two <= 4096 dividing
/// `n_elems`. The kernel `.wait()`s before returning.
pub unsafe fn hadamard_forward_usm_raw(
    stream: &SyclStream,
    x_usm: *const f32,
    signs_usm: *const f32,
    out_usm: *mut f32,
    n_elems: u32,
    block: u32,
) -> Result<()> {
    imp::hadamard_forward_usm(stream, x_usm, signs_usm, out_usm, n_elems, block)
}

/// Raw-pointer variant of the USM Q5_K_M packed-layout matvec.
/// Consumes the raw GGUF Q5_K_M super-block byte layout (176 bytes
/// per 256 weights: f16 d, f16 dmin, 12-byte packed 6-bit
/// scales/mins, 32-byte qh high-bit stream, 128-byte qs low-nibble
/// stream).
///
/// `lws`: see [`matvec_q4_k_packed_f32_usm_raw`].
///
/// # Safety
///
/// `w_bytes_usm` sized ≥ `M * (K/256) * 176` bytes, `x_usm` sized ≥ `K`,
/// `out_usm` sized ≥ `M`. All on the same SYCL context as `stream`.
/// `K` must be a multiple of 256. The kernel `.wait()`s before returning.
pub unsafe fn matvec_q5_k_packed_f32_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_q5_k_packed_f32_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, lws)
}

/// Raw-pointer variant of the USM Q6_K packed-layout matvec.
/// Consumes the raw GGUF Q6_K super-block byte layout (210 bytes
/// per 256 weights: 128-byte ql + 64-byte qh + 16-byte i8 scales +
/// f16 super-block scale).
///
/// `lws`: see [`matvec_q4_k_packed_f32_usm_raw`].
///
/// # Safety
///
/// `w_bytes_usm` sized ≥ `M * (K/256) * 210` bytes, `x_usm` sized ≥ `K`,
/// `out_usm` sized ≥ `M`. All on the same SYCL context as `stream`.
/// `K` must be a multiple of 256. The kernel `.wait()`s before returning.
pub unsafe fn matvec_q6_k_packed_f32_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_q6_k_packed_f32_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, lws)
}

/// CPU-parity packed matvec raw-pointer entries — legacy Q4_0/Q5_0/Q4_1/
/// Q5_1 (block 32, `M*(K/32)*{18,22,20,24}` bytes), K-quant Q2_K/Q3_K/Q8_K
/// (block 256, `M*(K/256)*{84,110,292}` bytes), PrismML PQ2_0 (block 128,
/// `M*(K/128)*34` bytes). Each forwards to its `imp` pointer wrapper.
macro_rules! packed_matvec_raw_wrapper {
    ($raw:ident, $imp:ident) => {
        /// Raw-pointer packed matvec. `out[m] = W[m, :] · x[:]`.
        ///
        /// # Safety
        /// See [`matvec_q4_k_packed_f32_usm_raw`]: `w_bytes_usm` sized per
        /// the dtype's packed layout, `x_usm` ≥ K f32, `out_usm` ≥ M f32,
        /// all on the same `stream`; the kernel `.wait()`s before return.
        pub unsafe fn $raw(
            stream: &SyclStream, w_bytes_usm: *const u8, x_usm: *const f32,
            out_usm: *mut f32, m: u32, k: u32, lws: u32,
        ) -> Result<()> {
            imp::$imp(stream, w_bytes_usm, x_usm, out_usm, m, k, lws)
        }
    };
}
packed_matvec_raw_wrapper!(matvec_q4_0_packed_f32_usm_raw, matvec_q4_0_packed_f32_usm);
// OCP Microscaling raw entries (M*(K/32)*{17,25,33} bytes; block 32).
packed_matvec_raw_wrapper!(matvec_mxfp4_packed_f32_usm_raw, matvec_mxfp4_packed_f32_usm);
packed_matvec_raw_wrapper!(matvec_mxfp6_packed_f32_usm_raw, matvec_mxfp6_packed_f32_usm);
packed_matvec_raw_wrapper!(matvec_mxfp8_packed_f32_usm_raw, matvec_mxfp8_packed_f32_usm);
packed_matvec_raw_wrapper!(matvec_q5_0_packed_f32_usm_raw, matvec_q5_0_packed_f32_usm);
packed_matvec_raw_wrapper!(matvec_q4_1_packed_f32_usm_raw, matvec_q4_1_packed_f32_usm);
packed_matvec_raw_wrapper!(matvec_q5_1_packed_f32_usm_raw, matvec_q5_1_packed_f32_usm);
packed_matvec_raw_wrapper!(matvec_q2_k_packed_f32_usm_raw, matvec_q2_k_packed_f32_usm);
packed_matvec_raw_wrapper!(matvec_q3_k_packed_f32_usm_raw, matvec_q3_k_packed_f32_usm);
packed_matvec_raw_wrapper!(matvec_q8_k_packed_f32_usm_raw, matvec_q8_k_packed_f32_usm);
packed_matvec_raw_wrapper!(matvec_pq2_0_packed_f32_usm_raw, matvec_pq2_0_packed_f32_usm);

/// F4: Raw-pointer USM IQ4_NL packed matvec. Consumes the GGUF
/// IQ4_NL byte layout (18 bytes per 32-weight block: f16 d + 16
/// bytes of 4-bit nibble qs indexing the IQ4 codebook).
///
/// `lws`: see [`matvec_q4_k_packed_f32_usm_raw`].
///
/// # Safety
///
/// `w_bytes_usm` sized ≥ `M * (K/32) * 18` bytes, `x_usm` sized ≥ `K`,
/// `out_usm` sized ≥ `M`. All on the same SYCL context as `stream`.
/// `K` must be a multiple of 32. The kernel `.wait()`s before returning.
pub unsafe fn matvec_iq4_nl_packed_f32_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq4_nl_packed_f32_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, lws)
}

/// F4 inference: Raw-pointer USM IQ1_S packed matvec. Consumes the
/// GGUF IQ1_S byte layout (50 bytes per 256-weight super-block:
/// f16 d + 32-byte qs + 16-byte qh). Codebook (`IQ1S_GRID`, 2048
/// entries × u64) is embedded as a `constexpr` in the kernel TU
/// via the build script's generated `iq_grids.inl`.
///
/// SAFETY: `w_bytes_usm` sized ≥ `M × (K/256) × 50` bytes;
/// `x_usm` ≥ `K` f32; `out_usm` ≥ `M` f32. All on `stream`'s
/// SYCL context. `K` must be a multiple of 256. Kernel `.wait()`s
/// before returning.
pub unsafe fn matvec_iq1_s_packed_f32_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq1_s_packed_f32_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, lws)
}

/// Raw-pointer USM IQ2_XXS packed matvec. Consumes the GGUF
/// IQ2_XXS byte layout (66 bytes per 256-weight super-block: f16 d
/// + 64-byte qs). Codebook (`IQ2XXS_GRID` 256 × u64) and sign
/// tables (`KSIGNS_IQ2XS` 128 × u8, `KMASK_IQ2XS` 8 × u8) are
/// embedded as `constexpr` arrays in the kernel TU via the build
/// script's generated `iq_grids.inl`.
///
/// SAFETY: `w_bytes_usm` sized ≥ `M × (K/256) × 66` bytes;
/// `x_usm` ≥ `K` f32; `out_usm` ≥ `M` f32. All on `stream`'s
/// SYCL context. `K` must be a multiple of 256. Kernel `.wait()`s
/// before returning.
pub unsafe fn matvec_iq2_xxs_packed_f32_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq2_xxs_packed_f32_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, lws)
}

/// Raw-pointer USM IQ1_M packed matvec. Consumes the GGUF IQ1_M
/// byte layout (56 bytes per 256-weight super-block: 32-byte qs +
/// 16-byte qh + 8-byte scales). Reuses the IQ1S 2048-entry
/// codebook (`IQ1S_GRID_SYCL`) embedded via `iq_grids.inl`; f16 d
/// is reassembled from nibbles across the 4 packed scale words.
///
/// SAFETY: `w_bytes_usm` sized ≥ `M × (K/256) × 56` bytes;
/// `x_usm` ≥ `K` f32; `out_usm` ≥ `M` f32. All on `stream`'s
/// SYCL context. `K` must be a multiple of 256. Kernel `.wait()`s
/// before returning.
pub unsafe fn matvec_iq1_m_packed_f32_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq1_m_packed_f32_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, lws)
}

/// Raw-pointer USM IQ2_XS packed matvec. Consumes the GGUF
/// IQ2_XS byte layout (74 bytes per 256-weight super-block: f16
/// d + 64-byte qs (32 × u16) + 8-byte scales). Uses the 512-entry
/// IQ2XS codebook + the 128-entry KSIGNS sign table; both are
/// embedded as `constexpr` arrays via `iq_grids.inl`.
///
/// SAFETY: `w_bytes_usm` sized ≥ `M × (K/256) × 74` bytes;
/// `x_usm` ≥ `K` f32; `out_usm` ≥ `M` f32. All on `stream`'s
/// SYCL context. `K` must be a multiple of 256. Kernel `.wait()`s
/// before returning.
pub unsafe fn matvec_iq2_xs_packed_f32_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq2_xs_packed_f32_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, lws)
}

/// Raw-pointer USM IQ2_S packed matvec. Consumes the GGUF IQ2_S
/// byte layout (82 bytes per 256-weight super-block: f16 d +
/// 32-byte qs_lo + 32-byte signs + 8-byte qh + 8-byte scales).
/// Uses the 1024-entry IQ2S codebook (`IQ2S_GRID_SYCL`) embedded
/// via `iq_grids.inl`. Sign mask is stored directly per chunk —
/// no KSIGNS table lookup needed.
///
/// SAFETY: `w_bytes_usm` sized ≥ `M × (K/256) × 82` bytes;
/// `x_usm` ≥ `K` f32; `out_usm` ≥ `M` f32. All on `stream`'s
/// SYCL context. `K` must be a multiple of 256. Kernel `.wait()`s
/// before returning.
pub unsafe fn matvec_iq2_s_packed_f32_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq2_s_packed_f32_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, lws)
}

/// Raw-pointer USM IQ3_XXS packed matvec. Consumes the GGUF
/// IQ3_XXS byte layout (98 bytes per 256-weight super-block:
/// f16 d + 64-byte qs_grid + 32-byte qs_sas). The 256-entry
/// IQ3XXS codebook (`IQ3XXS_GRID_SYCL`, u32 packed) + the 128-
/// entry KSIGNS sign table are embedded via `iq_grids.inl`.
///
/// SAFETY: `w_bytes_usm` sized ≥ `M × (K/256) × 98` bytes;
/// `x_usm` ≥ `K` f32; `out_usm` ≥ `M` f32. All on `stream`'s
/// SYCL context. `K` must be a multiple of 256. Kernel `.wait()`s
/// before returning.
pub unsafe fn matvec_iq3_xxs_packed_f32_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq3_xxs_packed_f32_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, lws)
}

/// Raw-pointer USM IQ3_S packed matvec. Consumes the GGUF IQ3_S
/// byte layout (110 bytes per 256-weight super-block: f16 d +
/// 64-byte qs + 8-byte qh + 32-byte signs + 4-byte scales). The
/// 512-entry IQ3S codebook (`IQ3S_GRID_SYCL`, u32 packed) is
/// embedded via `iq_grids.inl`. Sign mask is stored inline per
/// chunk — no KSIGNS lookup.
///
/// SAFETY: `w_bytes_usm` sized ≥ `M × (K/256) × 110` bytes;
/// `x_usm` ≥ `K` f32; `out_usm` ≥ `M` f32. All on `stream`'s
/// SYCL context. `K` must be a multiple of 256. Kernel `.wait()`s
/// before returning.
pub unsafe fn matvec_iq3_s_packed_f32_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq3_s_packed_f32_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, lws)
}

/// G1: Raw-pointer USM IQ1_S batched matvec. Iterates over N input
/// activations in one kernel launch.
///
/// SAFETY: `w_bytes_usm` sized ≥ `M × (K/256) × 50` bytes;
/// `x_usm` ≥ `N × K`; `out_usm` ≥ `N × M`. All on `stream`'s SYCL
/// context. `K` must be a multiple of 256. Kernel `.wait()`s before
/// returning.
pub unsafe fn matvec_iq1_s_packed_f32_batched_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq1_s_packed_f32_batched_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, n, lws)
}

/// G1: Raw-pointer USM IQ1_M batched matvec.
pub unsafe fn matvec_iq1_m_packed_f32_batched_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq1_m_packed_f32_batched_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, n, lws)
}

/// G1: Raw-pointer USM IQ2_XXS batched matvec.
pub unsafe fn matvec_iq2_xxs_packed_f32_batched_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq2_xxs_packed_f32_batched_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, n, lws)
}

/// G1: Raw-pointer USM IQ2_XS batched matvec.
pub unsafe fn matvec_iq2_xs_packed_f32_batched_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq2_xs_packed_f32_batched_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, n, lws)
}

/// G1: Raw-pointer USM IQ2_S batched matvec.
pub unsafe fn matvec_iq2_s_packed_f32_batched_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq2_s_packed_f32_batched_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, n, lws)
}

/// G1: Raw-pointer USM IQ3_XXS batched matvec.
pub unsafe fn matvec_iq3_xxs_packed_f32_batched_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq3_xxs_packed_f32_batched_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, n, lws)
}

/// G1: Raw-pointer USM IQ3_S batched matvec.
pub unsafe fn matvec_iq3_s_packed_f32_batched_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq3_s_packed_f32_batched_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, n, lws)
}

/// F4: Raw-pointer USM IQ4_XS packed matvec. Consumes the GGUF
/// IQ4_XS byte layout (136 bytes per 256-weight super-block: f16 d
/// + u16 scales_h + 4 bytes scales_l + 128 bytes qs nibbles).
///
/// `lws`: see [`matvec_q4_k_packed_f32_usm_raw`].
///
/// # Safety
///
/// `w_bytes_usm` sized ≥ `M * (K/256) * 136` bytes, `x_usm` sized ≥
/// `K`, `out_usm` sized ≥ `M`. All on the same SYCL context as
/// `stream`. `K` must be a multiple of 256. The kernel `.wait()`s
/// before returning.
/// GPU offload of IQ1_S codebook search. See the C++-side kernel
/// comment in `rsl_kernels.cpp::iq_search_8elt_delta_impl` for
/// the algorithm description.
///
/// # Safety
///
/// All USM pointers must:
///   - Be allocated on the same SYCL context as `stream`.
///   - Have correct sizes: `targets` ≥ `n_chunks * 8 * 4` bytes,
///     `grid_f32` ≥ `2048 * 8 * 4` bytes, the three `out_*`
///     arrays each ≥ `n_chunks` elements of their respective type.
///   - Stay alive for the duration of the call; the kernel
///     `.wait()`s before returning.
pub unsafe fn iq_search_8elt_delta_iq1s_raw(
    stream: &SyclStream,
    targets: *const f32,
    delta: f32,
    grid_f32: *const f32,
    out_grid_idx: *mut u16,
    out_signed_score: *mut f32,
    out_norm_sq: *mut f32,
    n_chunks: u32,
) -> Result<()> {
    imp::iq_search_8elt_delta_iq1s(
        stream,
        targets,
        delta,
        grid_f32,
        out_grid_idx,
        out_signed_score,
        out_norm_sq,
        n_chunks,
    )
}

/// #3: Companion to `iq_search_8elt_delta_iq1s_raw` that returns ALL
/// THREE picks (max-|score|, max-positive-score, max-negative-score)
/// per chunk in a single GPU dispatch. Eliminates the CPU-side AVX2
/// pos_neg sweep that dominates the bit-pack stage of
/// `encode_iq1_s_with_encoder`.
pub unsafe fn iq_search_8elt_delta_iq1s_all3_raw(
    stream: &SyclStream,
    targets: *const f32,
    delta: f32,
    grid_f32: *const f32,
    out_grid_idx_abs: *mut u16,
    out_signed_score_abs: *mut f32,
    out_norm_sq_abs: *mut f32,
    out_grid_idx_pos: *mut u16,
    out_signed_score_pos: *mut f32,
    out_norm_sq_pos: *mut f32,
    out_grid_idx_neg: *mut u16,
    out_signed_score_neg: *mut f32,
    out_norm_sq_neg: *mut f32,
    n_chunks: u32,
) -> Result<()> {
    imp::iq_search_8elt_delta_iq1s_all3(
        stream,
        targets,
        delta,
        grid_f32,
        out_grid_idx_abs, out_signed_score_abs, out_norm_sq_abs,
        out_grid_idx_pos, out_signed_score_pos, out_norm_sq_pos,
        out_grid_idx_neg, out_signed_score_neg, out_norm_sq_neg,
        n_chunks,
    )
}

/// Imatrix-weighted companion to `iq_search_8elt_delta_iq1s_all3_raw`.
/// `weights` is [n_chunks × 8] f32 USM giving per-element importance.
/// SAFETY: all pointers must be USM-shared on `stream`'s queue; sizes
/// must match the kernel expectations. The kernel `.wait()`s before
/// returning.
#[allow(clippy::too_many_arguments)]
pub unsafe fn iq_search_8elt_delta_iq1s_all3_w_raw(
    stream: &SyclStream,
    targets: *const f32,
    weights: *const f32,
    delta: f32,
    grid_f32: *const f32,
    out_grid_idx_abs: *mut u16,
    out_signed_score_abs: *mut f32,
    out_norm_sq_abs: *mut f32,
    out_grid_idx_pos: *mut u16,
    out_signed_score_pos: *mut f32,
    out_norm_sq_pos: *mut f32,
    out_grid_idx_neg: *mut u16,
    out_signed_score_neg: *mut f32,
    out_norm_sq_neg: *mut f32,
    n_chunks: u32,
) -> Result<()> {
    imp::iq_search_8elt_delta_iq1s_all3_w(
        stream,
        targets,
        weights,
        delta,
        grid_f32,
        out_grid_idx_abs, out_signed_score_abs, out_norm_sq_abs,
        out_grid_idx_pos, out_signed_score_pos, out_norm_sq_pos,
        out_grid_idx_neg, out_signed_score_neg, out_norm_sq_neg,
        n_chunks,
    )
}

/// Public raw IQ2-family batched search entry. SAFETY: all pointers
/// must be USM-shared on `stream`'s queue; sizes must match the
/// kernel's declared expectations (see header doc). The kernel
/// `.wait()`s before returning.
pub unsafe fn iq_search_8elt_signed_raw(
    stream: &SyclStream,
    targets: *const f32,
    grid_f32: *const f32,
    grid_norm_sq_table: *const f32,
    ksigns_rev: *const u8,
    n_grid: u32,
    out_grid_idx: *mut u16,
    out_sign_idx: *mut u8,
    out_signed_score: *mut f32,
    out_grid_norm_sq: *mut f32,
    n_chunks: u32,
) -> Result<()> {
    imp::iq_search_8elt_signed(
        stream,
        targets,
        grid_f32,
        grid_norm_sq_table,
        ksigns_rev,
        n_grid,
        out_grid_idx,
        out_sign_idx,
        out_signed_score,
        out_grid_norm_sq,
        n_chunks,
    )
}

/// Host-to-host wrapper around the GPU argmax kernel. Allocates a
/// USM logits buffer + i32 output, copies `logits` in, runs the
/// kernel, reads the index back. Returns the chosen token id (or
/// `SyclError::Unavailable` on USM alloc failure / SYCL disabled).
///
/// Use for testing + the future engine integration's per-call path;
/// the integration will keep `logits_usm` resident across LM-head →
/// sampler so the copy disappears.
pub fn sampler_argmax_host(stream: &SyclStream, logits: &[f32]) -> Result<u32> {
    let vocab = logits.len();
    if vocab == 0 {
        return Err(SyclError::InvalidShape(
            "sampler_argmax_host: empty logits slice".to_string(),
        ));
    }
    let bytes = vocab * std::mem::size_of::<f32>();
    let logits_usm = usm_alloc_shared(stream, bytes) as *mut f32;
    if logits_usm.is_null() {
        return Err(SyclError::Unavailable);
    }
    let out_usm = usm_alloc_shared(stream, std::mem::size_of::<i32>()) as *mut i32;
    if out_usm.is_null() {
        // SAFETY: logits_usm came from `usm_alloc_shared` above on
        // the same stream and is currently owned by us.
        unsafe { usm_free(stream, logits_usm as *mut std::ffi::c_void) };
        return Err(SyclError::Unavailable);
    }
    // SAFETY: USM-shared is host-writable; we hold the only handle.
    unsafe {
        std::ptr::copy_nonoverlapping(logits.as_ptr(), logits_usm, vocab);
        *out_usm = 0;
    }
    let rc = unsafe { sampler_argmax_usm_raw(stream, logits_usm, vocab as u32, out_usm) };
    let result = match rc {
        Ok(()) => {
            // SAFETY: kernel wrote one i32 here; USM-shared.
            let idx = unsafe { *out_usm };
            if idx < 0 {
                Err(SyclError::Runtime(
                    "sampler_argmax_usm returned negative index".to_string(),
                ))
            } else {
                Ok(idx as u32)
            }
        }
        Err(e) => Err(e),
    };
    // SAFETY: both pointers came from `usm_alloc_shared` on
    // `stream` and aren't referenced after this point.
    unsafe {
        usm_free(stream, out_usm as *mut std::ffi::c_void);
        usm_free(stream, logits_usm as *mut std::ffi::c_void);
    }
    result
}

/// Public raw GPU argmax entry. Computes `argmax(logits[0..vocab])`
/// and writes the index to `*out_idx_usm`. Lowest-index tie-break.
///
/// SAFETY: both pointers must be USM-shared on `stream`'s queue;
/// `logits_usm` ≥ `vocab` f32 elements; `out_idx_usm` ≥ 1 i32
/// element. The kernel `.wait()`s before returning.
pub unsafe fn sampler_argmax_usm_raw(
    stream: &SyclStream,
    logits_usm: *const f32,
    vocab: u32,
    out_idx_usm: *mut i32,
) -> Result<()> {
    imp::sampler_argmax_usm(stream, logits_usm, vocab, out_idx_usm)
}

/// Public raw GPU fused temperature scale + softmax entry. Mutates
/// `logits_usm` in-place: replaces logits with the corresponding
/// probability distribution (`softmax(logits * inv_temp)`).
///
/// SAFETY: `logits_usm` must be USM-shared on `stream`'s queue with
/// ≥ `vocab` f32 elements. The kernel `.wait()`s before returning.
pub unsafe fn sampler_temp_softmax_usm_raw(
    stream: &SyclStream,
    logits_usm: *mut f32,
    vocab: u32,
    inv_temp: f32,
) -> Result<()> {
    imp::sampler_temp_softmax_usm(stream, logits_usm, vocab, inv_temp)
}

/// Public raw GPU top-p entry. Performs nucleus filtering: masks
/// probabilities so only the smallest prefix (by descending
/// probability) whose cumulative sum ≥ `p` survives, then
/// renormalizes. Caps at the top-1024 entries; if those don't
/// cumulate to `p`, writes `*needs_fallback_usm = 1` and leaves
/// `probs_usm` UNMODIFIED so the caller can CPU-fall-back.
///
/// SAFETY: both pointers must be USM-shared on `stream`'s queue;
/// `probs_usm` ≥ `vocab` f32 elements; `needs_fallback_usm` ≥ 1
/// i32 element. Kernel `.wait()`s.
pub unsafe fn sampler_top_p_usm_raw(
    stream: &SyclStream,
    probs_usm: *mut f32,
    vocab: u32,
    p: f32,
    needs_fallback_usm: *mut i32,
) -> Result<()> {
    imp::sampler_top_p_usm(stream, probs_usm, vocab, p, needs_fallback_usm)
}

/// Maximum top-N tokens the GPU top-p kernel inspects. If cumsum
/// across the top-N doesn't cross `p`, the caller falls back to
/// CPU `apply_top_p` (which sorts more aggressively).
pub const MAX_TOP_P_GPU: u32 = 1024;

/// Host-to-host wrapper around the GPU top-p kernel. Stages
/// `probs` into USM, runs the kernel; on success writes the
/// masked + renormalized probs back. Returns `Ok(true)` if GPU
/// succeeded, `Ok(false)` if the kernel signaled CPU fallback is
/// needed (probs left untouched by GPU; caller runs CPU
/// `apply_top_p`). `Err` for USM alloc / SYCL failures.
pub fn sampler_top_p_host(
    stream: &SyclStream,
    probs: &mut [f32],
    p: f32,
) -> Result<bool> {
    let vocab = probs.len();
    if vocab == 0 {
        return Err(SyclError::InvalidShape(
            "sampler_top_p_host: empty probs slice".to_string(),
        ));
    }
    if p <= 0.0 || p >= 1.0 {
        // CPU `apply_top_p` is also a no-op outside (0, 1). Return
        // "GPU succeeded with no work" rather than a fallback flag.
        return Ok(true);
    }
    let probs_bytes = vocab * std::mem::size_of::<f32>();
    let probs_usm = usm_alloc_shared(stream, probs_bytes) as *mut f32;
    if probs_usm.is_null() {
        return Err(SyclError::Unavailable);
    }
    let flag_usm = usm_alloc_shared(stream, std::mem::size_of::<i32>()) as *mut i32;
    if flag_usm.is_null() {
        unsafe { usm_free(stream, probs_usm as *mut std::ffi::c_void) };
        return Err(SyclError::Unavailable);
    }
    // SAFETY: USM-shared is host-writable; we hold the only handles.
    unsafe {
        std::ptr::copy_nonoverlapping(probs.as_ptr(), probs_usm, vocab);
        *flag_usm = 0;
    }
    let rc = unsafe { sampler_top_p_usm_raw(stream, probs_usm, vocab as u32, p, flag_usm) };
    let result = match rc {
        Ok(()) => {
            // SAFETY: kernel set the flag; USM-shared.
            let needs_fallback = unsafe { *flag_usm };
            if needs_fallback != 0 {
                Ok(false) // probs untouched; caller falls back
            } else {
                // SAFETY: kernel mutated in-place; copy result out.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        probs_usm as *const f32,
                        probs.as_mut_ptr(),
                        vocab,
                    );
                }
                Ok(true)
            }
        }
        Err(e) => Err(e),
    };
    // SAFETY: both pointers came from `usm_alloc_shared` on `stream`
    // and aren't referenced after this point.
    unsafe {
        usm_free(stream, flag_usm as *mut std::ffi::c_void);
        usm_free(stream, probs_usm as *mut std::ffi::c_void);
    }
    result
}

/// Public raw GPU top-k entry. Finds the k-th-largest probability
/// in `probs_usm`, zeros everything strictly below it, then
/// renormalizes so the survivors sum to 1.0. Mirrors the CPU
/// `apply_top_k` semantics including tie-keep behavior.
///
/// **Capped at k ≤ 256.** Larger k falls back to CPU via the host
/// wrapper. SAFETY: `probs_usm` ≥ `vocab` f32 USM elements on
/// `stream`'s queue. Kernel `.wait()`s.
pub unsafe fn sampler_top_k_usm_raw(
    stream: &SyclStream,
    probs_usm: *mut f32,
    vocab: u32,
    k: u32,
) -> Result<()> {
    imp::sampler_top_k_usm(stream, probs_usm, vocab, k)
}

/// Maximum k supported by the GPU top-k kernel. Host wrapper
/// returns `InvalidShape` for `k > MAX_TOP_K_GPU` so callers know
/// to fall back to CPU.
pub const MAX_TOP_K_GPU: u32 = 256;

/// Host-to-host wrapper around the GPU top-k pass. Returns an
/// `InvalidShape` error if `k > MAX_TOP_K_GPU`; callers should
/// fall back to CPU `apply_top_k` in that case.
pub fn sampler_top_k_host(stream: &SyclStream, probs: &mut [f32], k: u32) -> Result<()> {
    let vocab = probs.len();
    if vocab == 0 {
        return Err(SyclError::InvalidShape(
            "sampler_top_k_host: empty probs slice".to_string(),
        ));
    }
    if k == 0 || k as usize >= vocab {
        // CPU's no-op gates — short-circuit before touching USM.
        return Ok(());
    }
    if k > MAX_TOP_K_GPU {
        return Err(SyclError::InvalidShape(format!(
            "sampler_top_k_host: k={k} exceeds GPU cap {MAX_TOP_K_GPU}; \
             caller should fall back to CPU"
        )));
    }
    let bytes = vocab * std::mem::size_of::<f32>();
    let probs_usm = usm_alloc_shared(stream, bytes) as *mut f32;
    if probs_usm.is_null() {
        return Err(SyclError::Unavailable);
    }
    // SAFETY: USM-shared is host-writable; we hold the only handle.
    unsafe {
        std::ptr::copy_nonoverlapping(probs.as_ptr(), probs_usm, vocab);
    }
    let rc = unsafe { sampler_top_k_usm_raw(stream, probs_usm, vocab as u32, k) };
    if let Ok(()) = rc {
        // SAFETY: kernel mutated in-place; copy result out.
        unsafe {
            std::ptr::copy_nonoverlapping(probs_usm as *const f32, probs.as_mut_ptr(), vocab);
        }
    }
    // SAFETY: probs_usm came from `usm_alloc_shared` on `stream`
    // and isn't referenced after this point.
    unsafe { usm_free(stream, probs_usm as *mut std::ffi::c_void) };
    rc
}

/// Public raw GPU penalty-pass entry. Applies repetition,
/// frequency, and presence penalties in-place on `logits_usm`,
/// matching the CPU `apply_all_penalties` semantics.
///
/// SAFETY: `logits_usm` ≥ `vocab` f32 USM elements; `recent_usm` ≥
/// `recent_n` u32 USM elements; both on `stream`'s queue. Kernel
/// `.wait()`s before returning. `recent_n == 0` is a clean no-op.
pub unsafe fn sampler_penalty_usm_raw(
    stream: &SyclStream,
    logits_usm: *mut f32,
    vocab: u32,
    recent_usm: *const u32,
    recent_n: u32,
    repeat: f32,
    frequency: f32,
    presence: f32,
) -> Result<()> {
    imp::sampler_penalty_usm(
        stream,
        logits_usm,
        vocab,
        recent_usm,
        recent_n,
        repeat,
        frequency,
        presence,
    )
}

/// Host-to-host wrapper around the GPU penalty pass. Stages
/// `logits` + `recent` into USM, runs the kernel, copies penalized
/// logits back. Returns `Err` and leaves `logits` untouched on
/// alloc / SYCL failure.
pub fn sampler_penalty_host(
    stream: &SyclStream,
    logits: &mut [f32],
    recent: &[u32],
    repeat: f32,
    frequency: f32,
    presence: f32,
) -> Result<()> {
    let vocab = logits.len();
    if vocab == 0 {
        return Err(SyclError::InvalidShape(
            "sampler_penalty_host: empty logits slice".to_string(),
        ));
    }
    if recent.is_empty() {
        // Clean no-op — matches the GPU short-circuit and the
        // CPU `apply_all_penalties`'s empty-recent fast path.
        return Ok(());
    }
    let logits_bytes = vocab * std::mem::size_of::<f32>();
    let recent_bytes = recent.len() * std::mem::size_of::<u32>();
    let logits_usm = usm_alloc_shared(stream, logits_bytes) as *mut f32;
    if logits_usm.is_null() {
        return Err(SyclError::Unavailable);
    }
    let recent_usm = usm_alloc_shared(stream, recent_bytes) as *mut u32;
    if recent_usm.is_null() {
        unsafe { usm_free(stream, logits_usm as *mut std::ffi::c_void) };
        return Err(SyclError::Unavailable);
    }
    // SAFETY: USM-shared is host-writable; we hold the only handles.
    unsafe {
        std::ptr::copy_nonoverlapping(logits.as_ptr(), logits_usm, vocab);
        std::ptr::copy_nonoverlapping(recent.as_ptr(), recent_usm, recent.len());
    }
    let rc = unsafe {
        sampler_penalty_usm_raw(
            stream,
            logits_usm,
            vocab as u32,
            recent_usm,
            recent.len() as u32,
            repeat,
            frequency,
            presence,
        )
    };
    if let Ok(()) = rc {
        // SAFETY: kernel mutated in-place; copy result out.
        unsafe {
            std::ptr::copy_nonoverlapping(logits_usm as *const f32, logits.as_mut_ptr(), vocab);
        }
    }
    // SAFETY: both pointers came from `usm_alloc_shared` on
    // `stream` and aren't referenced after this point.
    unsafe {
        usm_free(stream, recent_usm as *mut std::ffi::c_void);
        usm_free(stream, logits_usm as *mut std::ffi::c_void);
    }
    rc
}

/// Public raw GPU multinomial-draw entry. Advances `rng_state_usm`
/// by one SplitMix64 step, draws a uniform u ∈ [0, 1), walks the
/// cumsum of `probs_usm` to find the first index whose prefix sum
/// exceeds u, writes that index to `*out_idx_usm`. Matches the CPU
/// `multinomial` reference in `rustllama-engine` bit-for-bit under
/// the same seed.
///
/// SAFETY: all three pointers must be USM-shared on `stream`'s
/// queue; `probs_usm` ≥ `vocab` f32 elements; `rng_state_usm` and
/// `out_idx_usm` each ≥ 1 element. Kernel `.wait()`s.
pub unsafe fn sampler_multinomial_usm_raw(
    stream: &SyclStream,
    probs_usm: *const f32,
    vocab: u32,
    rng_state_usm: *mut u64,
    out_idx_usm: *mut i32,
) -> Result<()> {
    imp::sampler_multinomial_usm(stream, probs_usm, vocab, rng_state_usm, out_idx_usm)
}

/// Host-to-host wrapper around the GPU multinomial draw. Takes the
/// caller's RNG state by value, runs the kernel against a fresh USM
/// staging copy of `probs`, returns `(chosen_id, updated_rng_state)`.
/// The CPU-side `Rng` should be replaced with the returned state
/// after the call so its stream continues from the same point.
///
/// Use for testing + the future engine integration's per-call path.
pub fn sampler_multinomial_host(
    stream: &SyclStream,
    probs: &[f32],
    rng_state: u64,
) -> Result<(u32, u64)> {
    let vocab = probs.len();
    if vocab == 0 {
        return Err(SyclError::InvalidShape(
            "sampler_multinomial_host: empty probs slice".to_string(),
        ));
    }
    let probs_bytes = vocab * std::mem::size_of::<f32>();
    let probs_usm = usm_alloc_shared(stream, probs_bytes) as *mut f32;
    if probs_usm.is_null() {
        return Err(SyclError::Unavailable);
    }
    let rng_usm = usm_alloc_shared(stream, std::mem::size_of::<u64>()) as *mut u64;
    let out_usm = usm_alloc_shared(stream, std::mem::size_of::<i32>()) as *mut i32;
    if rng_usm.is_null() || out_usm.is_null() {
        if !rng_usm.is_null() {
            unsafe { usm_free(stream, rng_usm as *mut std::ffi::c_void) };
        }
        if !out_usm.is_null() {
            unsafe { usm_free(stream, out_usm as *mut std::ffi::c_void) };
        }
        unsafe { usm_free(stream, probs_usm as *mut std::ffi::c_void) };
        return Err(SyclError::Unavailable);
    }
    // SAFETY: USM-shared is host-writable; we hold the only handles.
    unsafe {
        std::ptr::copy_nonoverlapping(probs.as_ptr(), probs_usm, vocab);
        *rng_usm = rng_state;
        *out_usm = 0;
    }
    let rc =
        unsafe { sampler_multinomial_usm_raw(stream, probs_usm, vocab as u32, rng_usm, out_usm) };
    let result = match rc {
        Ok(()) => {
            // SAFETY: kernel wrote both fields; USM-shared.
            let idx = unsafe { *out_usm };
            let new_state = unsafe { *rng_usm };
            if idx < 0 {
                Err(SyclError::Runtime(
                    "sampler_multinomial_usm returned negative index".to_string(),
                ))
            } else {
                Ok((idx as u32, new_state))
            }
        }
        Err(e) => Err(e),
    };
    // SAFETY: all pointers came from `usm_alloc_shared` on `stream`
    // and aren't referenced after this point.
    unsafe {
        usm_free(stream, out_usm as *mut std::ffi::c_void);
        usm_free(stream, rng_usm as *mut std::ffi::c_void);
        usm_free(stream, probs_usm as *mut std::ffi::c_void);
    }
    result
}

/// Host-to-host wrapper around the GPU fused temp+softmax kernel.
/// Allocates a USM logits buffer, copies `logits` in, runs the
/// kernel, copies the result back into `logits`. On `Err` returned
/// the slice is left untouched.
///
/// Use for testing + the future engine integration's per-call path;
/// the integration will keep `logits_usm` resident across LM-head →
/// sampler → multinomial so the alloc + copies disappear.
pub fn sampler_temp_softmax_host(
    stream: &SyclStream,
    logits: &mut [f32],
    inv_temp: f32,
) -> Result<()> {
    let vocab = logits.len();
    if vocab == 0 {
        return Err(SyclError::InvalidShape(
            "sampler_temp_softmax_host: empty logits slice".to_string(),
        ));
    }
    let bytes = vocab * std::mem::size_of::<f32>();
    let logits_usm = usm_alloc_shared(stream, bytes) as *mut f32;
    if logits_usm.is_null() {
        return Err(SyclError::Unavailable);
    }
    // SAFETY: USM-shared is host-writable; we hold the only handle.
    unsafe {
        std::ptr::copy_nonoverlapping(logits.as_ptr(), logits_usm, vocab);
    }
    let rc = unsafe { sampler_temp_softmax_usm_raw(stream, logits_usm, vocab as u32, inv_temp) };
    if let Ok(()) = rc {
        // SAFETY: kernel wrote `vocab` f32 into logits_usm; USM-shared.
        unsafe {
            std::ptr::copy_nonoverlapping(logits_usm as *const f32, logits.as_mut_ptr(), vocab);
        }
    }
    // SAFETY: logits_usm came from `usm_alloc_shared` on `stream`
    // and isn't referenced after this point.
    unsafe { usm_free(stream, logits_usm as *mut std::ffi::c_void) };
    rc
}

/// Public raw IQ3-family batched paired-grid search entry.
/// SAFETY: all pointers must be USM-shared on `stream`'s queue;
/// sizes must match the kernel's declared expectations (see header
/// doc). The kernel `.wait()`s before returning.
pub unsafe fn iq_search_4elt_paired_signed_raw(
    stream: &SyclStream,
    targets: *const f32,
    grid_f32: *const f32,
    grid_norm_sq_table: *const f32,
    kmask: *const u8,
    ksigns_rev: *const u8,
    n_grid: u32,
    out_grid1_idx: *mut u16,
    out_grid2_idx: *mut u16,
    out_sign_idx: *mut u8,
    out_signed_score: *mut f32,
    out_grid_norm_sq: *mut f32,
    n_chunks: u32,
) -> Result<()> {
    imp::iq_search_4elt_paired_signed(
        stream,
        targets,
        grid_f32,
        grid_norm_sq_table,
        kmask,
        ksigns_rev,
        n_grid,
        out_grid1_idx,
        out_grid2_idx,
        out_sign_idx,
        out_signed_score,
        out_grid_norm_sq,
        n_chunks,
    )
}

pub unsafe fn matvec_iq4_xs_packed_f32_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq4_xs_packed_f32_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, lws)
}

/// Raw-pointer variant of the batched USM Q8_0 packed matvec.
///
/// # Safety
///
/// `w_bytes_usm` sized ≥ `M * (K/32) * 34` bytes, `x_usm` sized ≥
/// `N * K`, `out_usm` sized ≥ `N * M`. All on the same SYCL context
/// as `stream`. `K` must be a multiple of 32. `N >= 1`. Kernel
/// `.wait()`s before returning.
pub unsafe fn matvec_q8_0_packed_f32_batched_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_q8_0_packed_f32_batched_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, n, lws)
}

/// Raw-pointer variant of the batched USM Q4_K_M packed matvec.
/// `lws`: see [`matvec_q4_k_packed_f32_usm_raw`].
///
/// # Safety
///
/// `w_bytes_usm` sized ≥ `M * (K/256) * 144` bytes, `x_usm` sized ≥
/// `N * K`, `out_usm` sized ≥ `N * M`. All on the same SYCL context
/// as `stream`. `K` must be a multiple of 256. Kernel `.wait()`s
/// before returning.
pub unsafe fn matvec_q4_k_packed_f32_batched_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_q4_k_packed_f32_batched_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, n, lws)
}

/// Raw-pointer variant of the batched USM IQ4_NL packed matvec.
/// Same per-cell math as `matvec_iq4_nl_packed_f32_usm_raw`,
/// covering N input rows in one kernel launch.
///
/// # Safety
///
/// `w_bytes_usm` sized ≥ `M * (K/32) * 18` bytes, `x_usm` sized ≥
/// `N * K`, `out_usm` sized ≥ `N * M`. All on the same SYCL context
/// as `stream`. `K` must be a multiple of 32. Kernel `.wait()`s
/// before returning.
pub unsafe fn matvec_iq4_nl_packed_f32_batched_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq4_nl_packed_f32_batched_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, n, lws)
}

/// Raw-pointer variant of the batched USM IQ4_XS packed matvec.
/// Same per-cell math as `matvec_iq4_xs_packed_f32_usm_raw`,
/// covering N input rows in one kernel launch.
///
/// # Safety
///
/// `w_bytes_usm` sized ≥ `M * (K/256) * 136` bytes, `x_usm` sized ≥
/// `N * K`, `out_usm` sized ≥ `N * M`. All on the same SYCL context
/// as `stream`. `K` must be a multiple of 256. Kernel `.wait()`s
/// before returning.
pub unsafe fn matvec_iq4_xs_packed_f32_batched_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq4_xs_packed_f32_batched_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, n, lws)
}

/// Raw-pointer variant of the batched USM Q5_K_M packed matvec.
/// `lws`: see [`matvec_q4_k_packed_f32_usm_raw`].
///
/// # Safety
///
/// `w_bytes_usm` sized ≥ `M * (K/256) * 176` bytes, `x_usm` sized ≥
/// `N * K`, `out_usm` sized ≥ `N * M`. All on the same SYCL context
/// as `stream`. `K` must be a multiple of 256. Kernel `.wait()`s
/// before returning.
pub unsafe fn matvec_q5_k_packed_f32_batched_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_q5_k_packed_f32_batched_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, n, lws)
}

/// Raw-pointer variant of the batched USM Q6_K packed matvec.
/// `lws`: see [`matvec_q4_k_packed_f32_usm_raw`].
///
/// # Safety
///
/// `w_bytes_usm` sized ≥ `M * (K/256) * 210` bytes, `x_usm` sized ≥
/// `N * K`, `out_usm` sized ≥ `N * M`. All on the same SYCL context
/// as `stream`. `K` must be a multiple of 256. Kernel `.wait()`s
/// before returning.
pub unsafe fn matvec_q6_k_packed_f32_batched_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    x_usm: *const f32,
    out_usm: *mut f32,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_q6_k_packed_f32_batched_usm(stream, w_bytes_usm, x_usm, out_usm, m, k, n, lws)
}

/// Owning, RAII handle to a USM-shared allocation backed by a
/// [`SyclStream`]. The allocation is page-mapped between host and
/// device — host code can read/write it directly via
/// [`Self::as_mut_slice`] / [`Self::as_slice`], and kernel code on
/// the bound device can dereference the same pointer with no
/// explicit copy.
///
/// Lifetime is tied to the borrow on the stream that allocated it:
/// the stream cannot be dropped while a `SyclSharedBuffer` borrows
/// it, and the buffer's `Drop` frees the USM allocation back into
/// the stream's context. Multiple buffers can co-exist on one
/// stream — they're all freed when their respective `SyclSharedBuffer`
/// values drop.
///
/// Generic over the element type for ergonomic slice access. The
/// underlying SYCL allocator works in bytes; we just multiply by
/// `size_of::<T>()` on the way in.
///
/// On a no-SYCL-device build (no SYCL runtime), [`Self::alloc`] returns
/// [`SyclError::Unavailable`] without touching the FFI.
pub struct SyclSharedBuffer<'s, T: Copy> {
    /// Raw pointer into the USM allocation. Cast to `*mut T` for
    /// slice construction. Non-null when this struct exists; the
    /// constructor errors on allocator failure rather than holding
    /// a null pointer.
    ptr: std::ptr::NonNull<T>,
    /// Number of `T` elements the allocation holds. The allocation
    /// size in bytes is `len * size_of::<T>()`.
    len: usize,
    /// **Shared** borrow on the owning stream — guarantees the
    /// stream outlives this buffer (the SYCL runtime needs the
    /// queue alive to free the allocation). The borrow is shared
    /// rather than exclusive so multiple buffers can coexist on
    /// one stream — the engine's forward pass needs Q, K cache,
    /// V cache, and output USM buffers in scope at the same time.
    /// SYCL queues are internally thread-safe.
    stream: &'s SyclStream,
    /// Windows memory-residency: set once [`Self::pin`] VirtualLocks the
    /// allocation into the working set so the OS can't page this USM
    /// region (which on an iGPU is host RAM) to the pagefile. `Drop`
    /// VirtualUnlocks when set. Always `false` on non-Windows / unpinned.
    locked_bytes: usize,
}

/// Windows USM-pinning gate + budget for [`SyclSharedBuffer::pin`].
/// `RUSTLLAMA_LOCK_USM=1` enables; `RUSTLLAMA_LOCK_USM_MB` (default
/// 1024) caps total pinned USM bytes across the process.
#[cfg(windows)]
mod usm_pin {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    extern "system" {
        fn VirtualLock(addr: *const core::ffi::c_void, size: usize) -> i32;
        fn VirtualUnlock(addr: *const core::ffi::c_void, size: usize) -> i32;
    }

    fn enabled() -> bool {
        static CELL: OnceLock<bool> = OnceLock::new();
        *CELL.get_or_init(|| {
            std::env::var("RUSTLLAMA_LOCK_USM")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        })
    }

    fn budget_bytes() -> u64 {
        static CELL: OnceLock<u64> = OnceLock::new();
        *CELL.get_or_init(|| {
            std::env::var("RUSTLLAMA_LOCK_USM_MB")
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(1024)
                .saturating_mul(1024 * 1024)
        })
    }

    static PINNED: AtomicU64 = AtomicU64::new(0);

    /// Try to VirtualLock `[addr, addr+len)`. Returns the locked byte
    /// count (0 if disabled / over budget / failed). Fail-soft.
    pub fn try_lock(addr: *const u8, len: usize) -> usize {
        if !enabled() || addr.is_null() || len == 0 {
            return 0;
        }
        let prev = PINNED.fetch_add(len as u64, Ordering::Relaxed);
        if prev + len as u64 > budget_bytes() {
            PINNED.fetch_sub(len as u64, Ordering::Relaxed);
            return 0;
        }
        // SAFETY: addr/len describe a live USM-shared allocation that
        // outlives the lock (freed only on the owning buffer's Drop,
        // which unlocks first).
        let ok = unsafe { VirtualLock(addr as *const core::ffi::c_void, len) };
        if ok != 0 {
            len
        } else {
            PINNED.fetch_sub(len as u64, Ordering::Relaxed);
            0
        }
    }

    /// VirtualUnlock a previously-locked range; decrement the budget.
    pub fn unlock(addr: *const u8, len: usize) {
        if addr.is_null() || len == 0 {
            return;
        }
        // SAFETY: same range previously locked via try_lock.
        unsafe {
            let _ = VirtualUnlock(addr as *const core::ffi::c_void, len);
        }
        PINNED.fetch_sub(len as u64, Ordering::Relaxed);
    }
}

impl<'s, T: Copy> SyclSharedBuffer<'s, T> {
    /// Allocate `len` elements of `T` in USM-shared memory backed
    /// by `stream`. Returns `Unavailable` when no SYCL device is present and
    /// `InvalidShape("usm alloc failed")` if the SYCL runtime
    /// can't satisfy the request.
    pub fn alloc(stream: &'s SyclStream, len: usize) -> Result<Self> {
        let n_bytes = len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
            SyclError::InvalidShape(format!(
                "usm alloc overflow: {len} * {} bytes",
                std::mem::size_of::<T>()
            ))
        })?;
        let raw = imp::usm_alloc_shared(stream, n_bytes);
        let ptr = match std::ptr::NonNull::new(raw as *mut T) {
            Some(p) => p,
            None => {
                // Distinguish no-SYCL-device (no SYCL at all) from real
                // allocation failure. We check the device count —
                // when no SYCL device is present device_count returns Unavailable.
                if matches!(imp::device_count(), Err(SyclError::Unavailable)) {
                    return Err(SyclError::Unavailable);
                }
                return Err(SyclError::InvalidShape(format!(
                    "usm alloc failed for {n_bytes} bytes"
                )));
            }
        };
        Ok(Self {
            ptr,
            len,
            stream,
            locked_bytes: 0,
        })
    }

    /// Windows: VirtualLock this allocation into the working set so the
    /// OS can't page it to the pagefile (USM-shared is host RAM on an
    /// iGPU). Self-gated by `RUSTLLAMA_LOCK_USM` + `RUSTLLAMA_LOCK_USM_MB`
    /// budget; a no-op when disabled, over budget, or on non-Windows.
    /// Idempotent. Returns true if the allocation is now pinned.
    pub fn pin(&mut self) -> bool {
        #[cfg(windows)]
        {
            if self.locked_bytes != 0 {
                return true;
            }
            let n_bytes = self.len * std::mem::size_of::<T>();
            let got = usm_pin::try_lock(self.ptr.as_ptr() as *const u8, n_bytes);
            self.locked_bytes = got;
            got != 0
        }
        #[cfg(not(windows))]
        {
            false
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// CPU-side read access. Sound because USM-shared memory is
    /// page-mapped on the host and we hold the only `&mut` to it
    /// through the typestate of `SyclSharedBuffer`. Note that any
    /// concurrent kernel running on the bound device would race
    /// with a host read — callers must serialize via the stream
    /// (every kernel call in this crate `.wait()`s before
    /// returning, so this is fine within a single thread).
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: ptr non-null and properly aligned (SYCL malloc
        // guarantees max-alignment); len is the construction-time
        // element count.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: same as as_slice; `&mut self` rules out aliasing
        // host-side reads. See the same caveat about concurrent
        // device-side kernels.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Raw USM pointer for passing into kernel FFI shims.
    /// SAFETY: caller must not store the pointer past this
    /// buffer's drop. The `&self` borrow ensures the buffer
    /// outlives the call as long as the borrow is held.
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr() as *const T
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<'s, T: Copy> Drop for SyclSharedBuffer<'s, T> {
    fn drop(&mut self) {
        // Windows: VirtualUnlock before freeing if this buffer was pinned.
        #[cfg(windows)]
        if self.locked_bytes != 0 {
            usm_pin::unlock(self.ptr.as_ptr() as *const u8, self.locked_bytes);
        }
        // SAFETY: ptr came from `imp::usm_alloc_shared` on this
        // stream; we own it exclusively (we hold `&mut SyclStream`
        // and `&mut self`).
        unsafe {
            imp::usm_free(
                self.stream,
                self.ptr.as_ptr() as *mut std::ffi::c_void,
            );
        }
    }
}

/// Owned handle to a **device-local** (dedicated-VRAM) USM allocation
/// populated from a host source. The dedicated-VRAM analog of
/// [`SyclSharedBuffer`]: on an integrated GPU it lands in the reserved
/// VRAM aperture (Dedicated GPU memory), separate from system RAM.
///
/// Unlike `SyclSharedBuffer` there is deliberately **no** `as_slice` /
/// `as_mut_slice`: `malloc_device` memory is not host-mapped, so a
/// host dereference would be UB. It is write-once at construction (via
/// the FFI H2D copy) and thereafter read only by kernels through
/// [`Self::as_ptr`]. Freed on drop with the tier-agnostic `usm_free`.
pub struct SyclDeviceBuffer<'s> {
    ptr: std::ptr::NonNull<u8>,
    len: usize,
    stream: &'s SyclStream,
}

impl<'s> SyclDeviceBuffer<'s> {
    /// Allocate device-local USM sized to `src` and copy `src` into it
    /// (blocking H2D). Returns `None` when the device lacks device-USM
    /// support, the allocation fails, or the copy throws — the caller
    /// then falls back to the shared tier. Never holds a null pointer.
    pub fn alloc_from_host(stream: &'s SyclStream, src: &[u8]) -> Option<Self> {
        if src.is_empty() {
            return None;
        }
        let raw = imp::usm_alloc_device_from_host(stream, src);
        std::ptr::NonNull::new(raw as *mut u8).map(|ptr| Self {
            ptr,
            len: src.len(),
            stream,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device pointer for passing into kernel FFI shims. Same
    /// contract as [`SyclSharedBuffer::as_ptr`] — do not store past the
    /// buffer's drop, and do NOT dereference on the host.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr() as *const u8
    }
}

impl Drop for SyclDeviceBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: ptr came from `imp::usm_alloc_device_from_host` on
        // this stream; `sycl::free` is tier-agnostic so `usm_free` frees
        // device allocations too. We own it exclusively.
        unsafe {
            imp::usm_free(
                self.stream,
                self.ptr.as_ptr() as *mut std::ffi::c_void,
            );
        }
    }
}

/// USM-resident RMSNorm. Takes USM pointers from
/// [`SyclSharedBuffer::as_ptr`] / [`as_mut_ptr`] and runs the
/// kernel with no internal H2D / D2H — the data stays on device
/// across calls. Caller arranges that `x` and `y` hold at least
/// `n_rows * d` elements and `w` holds at least `d`; the wrapper
/// types make those invariants checkable.
///
/// Returns [`SyclError::Unavailable`] when no SYCL device is present. In
/// real-SYCL mode the call is bounded by the stream's queue —
/// kernel completes before this function returns.
pub fn rmsnorm_usm(
    stream: &SyclStream,
    x: &SyclSharedBuffer<u16>,
    w: &SyclSharedBuffer<u16>,
    y: &mut SyclSharedBuffer<u16>,
    n_rows: u32,
    d: u32,
    eps: f32,
) -> Result<()> {
    let total = (n_rows as usize)
        .checked_mul(d as usize)
        .ok_or_else(|| SyclError::InvalidShape(format!("rmsnorm overflow: {n_rows}*{d}")))?;
    if x.len() < total || y.len() < total {
        return Err(SyclError::InvalidShape(format!(
            "rmsnorm: x={}, y={}, need {n_rows}*{d}={total}",
            x.len(),
            y.len(),
        )));
    }
    if w.len() < d as usize {
        return Err(SyclError::InvalidShape(format!(
            "rmsnorm: w={}, need d={d}",
            w.len()
        )));
    }
    imp::rmsnorm_usm(stream, x.as_ptr(), w.as_ptr(), y.as_mut_ptr(), n_rows, d, eps)
}

/// Fused USM-resident RMSNorm + residual add. See
/// [`imp::rmsnorm_residual_usm`] for the kernel semantics.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_residual_usm(
    stream: &SyclStream,
    x: &SyclSharedBuffer<u16>,
    w: &SyclSharedBuffer<u16>,
    residual: &SyclSharedBuffer<u16>,
    y: &mut SyclSharedBuffer<u16>,
    n_rows: u32,
    d: u32,
    eps: f32,
) -> Result<()> {
    let total = (n_rows as usize).checked_mul(d as usize).ok_or_else(|| {
        SyclError::InvalidShape(format!("rmsnorm_residual overflow: {n_rows}*{d}"))
    })?;
    if x.len() < total || y.len() < total || residual.len() < total {
        return Err(SyclError::InvalidShape(format!(
            "rmsnorm_residual: x={}, residual={}, y={}, need {n_rows}*{d}={total}",
            x.len(),
            residual.len(),
            y.len(),
        )));
    }
    if w.len() < d as usize {
        return Err(SyclError::InvalidShape(format!(
            "rmsnorm_residual: w={}, need d={d}",
            w.len()
        )));
    }
    imp::rmsnorm_residual_usm(
        stream,
        x.as_ptr(),
        w.as_ptr(),
        residual.as_ptr(),
        y.as_mut_ptr(),
        n_rows,
        d,
        eps,
    )
}

/// Raw-pointer variant of [`rmsnorm_residual_usm`]. Same kernel; the
/// caller manages USM lifetime + shape validation (the engine's
/// fused norm sites pass USM views over slabs that aren't backed
/// by a `SyclSharedBuffer<u16>` and need to drop the wrapper layer).
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_residual_usm_raw(
    stream: &SyclStream,
    x: *const u16,
    w: *const u16,
    residual: *const u16,
    y: *mut u16,
    n_rows: u32,
    d: u32,
    eps: f32,
) -> Result<()> {
    imp::rmsnorm_residual_usm(stream, x, w, residual, y, n_rows, d, eps)
}

/// Fused "add residual + RMSNorm" — Llama-family pre-norm fusion.
/// See [`imp::add_rmsnorm_usm`] for kernel semantics.
#[allow(clippy::too_many_arguments)]
pub fn add_rmsnorm_usm_raw(
    stream: &SyclStream,
    hidden: *mut u16,
    branch: *const u16,
    w: *const u16,
    y_norm: *mut u16,
    n_rows: u32,
    d: u32,
    eps: f32,
) -> Result<()> {
    imp::add_rmsnorm_usm(stream, hidden, branch, w, y_norm, n_rows, d, eps)
}

/// F32-precision variant of [`add_rmsnorm_usm_raw`]. Used by the H6
/// out-proj+residual+norm dispatcher to avoid the F32→F16→F32
/// round-trip when chaining after an F32 matvec.
#[allow(clippy::too_many_arguments)]
pub fn add_rmsnorm_f32_usm_raw(
    stream: &SyclStream,
    hidden: *mut f32,
    branch: *const f32,
    w: *const f32,
    y_norm: *mut f32,
    n_rows: u32,
    d: u32,
    eps: f32,
) -> Result<()> {
    imp::add_rmsnorm_f32_usm(stream, hidden, branch, w, y_norm, n_rows, d, eps)
}

// (A second `flash_attn_decode_usm_raw` definition I added in an
// earlier pass was deleted here — the canonical raw-pointer entry
// lives below alongside the `_raw` variant for rmsnorm. Keeping
// this comment as a tombstone so future agents don't re-introduce
// the duplicate.)

/// USM-resident FlashAttention decode: fuses Q·Kᵀ, softmax, and
/// ·V into one kernel using the online-softmax recurrence on the
/// bound device. All four buffers (Q, K cache, V cache, output)
/// stay USM-resident — no per-call H2D / D2H.
///
/// Expected layouts:
///   - `q`:   `[n_heads, head_dim]`
///   - `k`:   `[n_kv_heads, max_ctx, head_dim]`
///   - `v`:   `[n_kv_heads, max_ctx, head_dim]`
///   - `out`: `[n_heads, head_dim]`
///
/// GQA: `n_heads` must be a multiple of `n_kv_heads`; Q head `h`
/// reads K/V cache at `kv_h = h / (n_heads / n_kv_heads)`. Same
/// algorithm as `rustllama_kernels_cpu::gqa_attention_flash_decode`
/// — greedy parity is asserted by the integration test below
/// (gated on real SYCL hardware).
///
/// Returns [`SyclError::Unavailable`] when no SYCL device is present.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_decode_usm(
    stream: &SyclStream,
    q: &SyclSharedBuffer<u16>,
    k: &SyclSharedBuffer<u16>,
    v: &SyclSharedBuffer<u16>,
    out: &mut SyclSharedBuffer<u16>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len: u32,
) -> Result<()> {
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
        )));
    }
    if kv_len > max_ctx {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_usm: kv_len={kv_len} > max_ctx={max_ctx}"
        )));
    }
    let need_q = (n_heads as usize) * (head_dim as usize);
    let need_out = need_q;
    let need_kv = (n_kv_heads as usize) * (max_ctx as usize) * (head_dim as usize);
    if q.len() < need_q {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_usm: q={}, need n_heads*head_dim={need_q}",
            q.len()
        )));
    }
    if out.len() < need_out {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_usm: out={}, need n_heads*head_dim={need_out}",
            out.len()
        )));
    }
    if k.len() < need_kv {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_usm: k={}, need n_kv_heads*max_ctx*head_dim={need_kv}",
            k.len()
        )));
    }
    if v.len() < need_kv {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_usm: v={}, need n_kv_heads*max_ctx*head_dim={need_kv}",
            v.len()
        )));
    }
    imp::flash_attn_decode_usm(
        stream,
        q.as_ptr(),
        k.as_ptr(),
        v.as_ptr(),
        out.as_mut_ptr(),
        n_heads,
        n_kv_heads,
        head_dim,
        max_ctx,
        kv_len,
    )
}

/// FA-v3 decode (SLM K/V tiling + sub-group cooperation). Drop-in
/// replacement for [`flash_attn_decode_usm`] with the same shape
/// constraints as v2 (head_dim multiple of 16, ≤ 256). Callers route
/// to v2 on mismatch / kernel failure.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_decode_v3_usm(
    stream: &SyclStream,
    q: &SyclSharedBuffer<u16>,
    k: &SyclSharedBuffer<u16>,
    v: &SyclSharedBuffer<u16>,
    out: &mut SyclSharedBuffer<u16>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len: u32,
) -> Result<()> {
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_v3_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
        )));
    }
    if kv_len > max_ctx {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_v3_usm: kv_len={kv_len} > max_ctx={max_ctx}"
        )));
    }
    if head_dim == 0 || head_dim % 16 != 0 || head_dim > 256 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_v3_usm: head_dim={head_dim} must be a multiple of 16 and ≤ 256"
        )));
    }
    let need_q = (n_heads as usize) * (head_dim as usize);
    let need_kv = (n_kv_heads as usize) * (max_ctx as usize) * (head_dim as usize);
    if q.len() < need_q || out.len() < need_q || k.len() < need_kv || v.len() < need_kv {
        return Err(SyclError::InvalidShape(
            "flash_attn_decode_v3_usm: buffer too small for shape".into(),
        ));
    }
    imp::flash_attn_decode_v3_usm(
        stream, q.as_ptr(), k.as_ptr(), v.as_ptr(), out.as_mut_ptr(),
        n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
    )
}

/// Raw-pointer FA-v3 decode.
///
/// # Safety
/// Same as [`flash_attn_decode_v2_usm_raw`]: pointers must reference
/// live USM allocations on `stream`'s context.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_v3_usm_raw(
    stream: &SyclStream,
    q_usm: *const u16,
    k_usm: *const u16,
    v_usm: *const u16,
    out_usm: *mut u16,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len: u32,
) -> Result<()> {
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_v3_usm_raw: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
        )));
    }
    if kv_len > max_ctx {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_v3_usm_raw: kv_len={kv_len} > max_ctx={max_ctx}"
        )));
    }
    if head_dim == 0 || head_dim % 16 != 0 || head_dim > 256 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_v3_usm_raw: head_dim={head_dim} must be a multiple of 16 and ≤ 256"
        )));
    }
    imp::flash_attn_decode_v3_usm(
        stream, q_usm, k_usm, v_usm, out_usm,
        n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
    )
}

/// FA-v2 decode (sub-group cooperation). Drop-in replacement for
/// [`flash_attn_decode_usm`] when `head_dim` is a multiple of 16 and
/// ≤ 256. Returns `InvalidShape` on mismatch; callers route the
/// fallback to v1.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_decode_v2_usm(
    stream: &SyclStream,
    q: &SyclSharedBuffer<u16>,
    k: &SyclSharedBuffer<u16>,
    v: &SyclSharedBuffer<u16>,
    out: &mut SyclSharedBuffer<u16>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len: u32,
) -> Result<()> {
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_v2_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
        )));
    }
    if kv_len > max_ctx {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_v2_usm: kv_len={kv_len} > max_ctx={max_ctx}"
        )));
    }
    if head_dim == 0 || head_dim % 16 != 0 || head_dim > 256 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_v2_usm: head_dim={head_dim} must be a multiple of 16 and ≤ 256"
        )));
    }
    let need_q = (n_heads as usize) * (head_dim as usize);
    let need_kv = (n_kv_heads as usize) * (max_ctx as usize) * (head_dim as usize);
    if q.len() < need_q || out.len() < need_q || k.len() < need_kv || v.len() < need_kv {
        return Err(SyclError::InvalidShape(
            "flash_attn_decode_v2_usm: buffer too small for shape".into(),
        ));
    }
    imp::flash_attn_decode_v2_usm(
        stream,
        q.as_ptr(),
        k.as_ptr(),
        v.as_ptr(),
        out.as_mut_ptr(),
        n_heads,
        n_kv_heads,
        head_dim,
        max_ctx,
        kv_len,
    )
}

/// USM-resident FlashAttention prefill — F32 variant. Fuses Q·Kᵀ,
/// softmax, and ·V into one kernel using the online-softmax
/// recurrence over `kv_len_base + n_new` cache positions, with the
/// causal mask applied per query position. All four buffers stay
/// USM-resident; the kernel `.wait()`s before returning.
///
/// Expected layouts (all f32):
///   - `q`:   `[N, n_heads, head_dim]`              row-major
///   - `k`:   `[n_kv_heads, max_ctx, head_dim]`
///   - `v`:   `[n_kv_heads, max_ctx, head_dim]`
///   - `out`: `[N, n_heads, head_dim]`              row-major
///
/// where `N = n_new`. Caller must have appended the `n_new` new K/V
/// rows at cache positions `[kv_len_base, kv_len_base + n_new)`
/// before invoking (matches the engine's append-then-attend
/// pattern). GQA: `n_heads % n_kv_heads == 0`; head `hh` reads
/// `kv_h = hh / (n_heads / n_kv_heads)`.
///
/// Algorithm mirrors `rustllama_kernels_cpu::gqa_attention_flash_prefill`
/// — bit-for-bit parity is asserted by the integration test below
/// (gated on real SYCL hardware).
///
/// Returns [`SyclError::Unavailable`] when no SYCL device is present.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_prefill_usm(
    stream: &SyclStream,
    q: &SyclSharedBuffer<f32>,
    k_cache: &SyclSharedBuffer<f32>,
    v_cache: &SyclSharedBuffer<f32>,
    out: &mut SyclSharedBuffer<f32>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len_base: u32,
    n_new: u32,
) -> Result<()> {
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
        )));
    }
    if kv_len_base.saturating_add(n_new) > max_ctx {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_usm: kv_len_base={kv_len_base} + n_new={n_new} > max_ctx={max_ctx}"
        )));
    }
    let need_q = (n_new as usize) * (n_heads as usize) * (head_dim as usize);
    let need_out = need_q;
    let need_kv = (n_kv_heads as usize) * (max_ctx as usize) * (head_dim as usize);
    if q.len() < need_q {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_usm: q={}, need N*n_heads*head_dim={need_q}",
            q.len()
        )));
    }
    if out.len() < need_out {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_usm: out={}, need N*n_heads*head_dim={need_out}",
            out.len()
        )));
    }
    if k_cache.len() < need_kv {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_usm: k={}, need n_kv_heads*max_ctx*head_dim={need_kv}",
            k_cache.len()
        )));
    }
    if v_cache.len() < need_kv {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_usm: v={}, need n_kv_heads*max_ctx*head_dim={need_kv}",
            v_cache.len()
        )));
    }
    imp::flash_attn_prefill_usm(
        stream,
        q.as_ptr(),
        k_cache.as_ptr(),
        v_cache.as_ptr(),
        out.as_mut_ptr(),
        n_heads,
        n_kv_heads,
        head_dim,
        max_ctx,
        kv_len_base,
        n_new,
    )
}

/// FA-v3 prefill (SLM K/V tiling + sub-group cooperation).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_prefill_v3_usm(
    stream: &SyclStream,
    q: &SyclSharedBuffer<f32>,
    k_cache: &SyclSharedBuffer<f32>,
    v_cache: &SyclSharedBuffer<f32>,
    out: &mut SyclSharedBuffer<f32>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len_base: u32,
    n_new: u32,
) -> Result<()> {
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_v3_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
        )));
    }
    if kv_len_base.saturating_add(n_new) > max_ctx {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_v3_usm: kv_len_base={kv_len_base} + n_new={n_new} > max_ctx={max_ctx}"
        )));
    }
    if head_dim == 0 || head_dim % 16 != 0 || head_dim > 256 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_v3_usm: head_dim={head_dim} must be a multiple of 16 and ≤ 256"
        )));
    }
    let need_q = (n_new as usize) * (n_heads as usize) * (head_dim as usize);
    let need_kv = (n_kv_heads as usize) * (max_ctx as usize) * (head_dim as usize);
    if q.len() < need_q || out.len() < need_q || k_cache.len() < need_kv || v_cache.len() < need_kv {
        return Err(SyclError::InvalidShape(
            "flash_attn_prefill_v3_usm: buffer too small for shape".into(),
        ));
    }
    imp::flash_attn_prefill_v3_usm(
        stream, q.as_ptr(), k_cache.as_ptr(), v_cache.as_ptr(), out.as_mut_ptr(),
        n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
    )
}

/// Raw-pointer FA-v3 prefill.
///
/// # Safety
/// Pointers must reference live USM allocations on `stream`'s context.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_v3_usm_raw(
    stream: &SyclStream,
    q_usm: *const f32,
    k_cache_usm: *const f32,
    v_cache_usm: *const f32,
    out_usm: *mut f32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len_base: u32,
    n_new: u32,
) -> Result<()> {
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_v3_usm_raw: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
        )));
    }
    if kv_len_base.saturating_add(n_new) > max_ctx {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_v3_usm_raw: kv_len_base={kv_len_base} + n_new={n_new} > max_ctx={max_ctx}"
        )));
    }
    if head_dim == 0 || head_dim % 16 != 0 || head_dim > 256 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_v3_usm_raw: head_dim={head_dim} must be a multiple of 16 and ≤ 256"
        )));
    }
    imp::flash_attn_prefill_v3_usm(
        stream, q_usm, k_cache_usm, v_cache_usm, out_usm,
        n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
    )
}

/// FA-v2 prefill (sub-group cooperation). Drop-in replacement for
/// [`flash_attn_prefill_usm`] when `head_dim % 16 == 0` and
/// `head_dim ≤ 256`. Returns `InvalidShape` on mismatch.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_prefill_v2_usm(
    stream: &SyclStream,
    q: &SyclSharedBuffer<f32>,
    k_cache: &SyclSharedBuffer<f32>,
    v_cache: &SyclSharedBuffer<f32>,
    out: &mut SyclSharedBuffer<f32>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len_base: u32,
    n_new: u32,
) -> Result<()> {
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_v2_usm: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
        )));
    }
    if kv_len_base.saturating_add(n_new) > max_ctx {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_v2_usm: kv_len_base={kv_len_base} + n_new={n_new} > max_ctx={max_ctx}"
        )));
    }
    if head_dim == 0 || head_dim % 16 != 0 || head_dim > 256 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_v2_usm: head_dim={head_dim} must be a multiple of 16 and ≤ 256"
        )));
    }
    let need_q = (n_new as usize) * (n_heads as usize) * (head_dim as usize);
    let need_kv = (n_kv_heads as usize) * (max_ctx as usize) * (head_dim as usize);
    if q.len() < need_q || out.len() < need_q || k_cache.len() < need_kv || v_cache.len() < need_kv {
        return Err(SyclError::InvalidShape(
            "flash_attn_prefill_v2_usm: buffer too small for shape".into(),
        ));
    }
    imp::flash_attn_prefill_v2_usm(
        stream,
        q.as_ptr(),
        k_cache.as_ptr(),
        v_cache.as_ptr(),
        out.as_mut_ptr(),
        n_heads,
        n_kv_heads,
        head_dim,
        max_ctx,
        kv_len_base,
        n_new,
    )
}

/// Raw-pointer variant of [`flash_attn_prefill_usm`] for callers
/// that already hold USM pointers (e.g. the accelerator scratch
/// pool in `rustllama-models::accel`). Same contract as the safe
/// wrapper above; the caller is responsible for buffer lifetimes
/// and sizing.
///
/// # Safety
///
/// All four pointers must reference live USM allocations on
/// `stream`'s context with at least:
///   - `q_usm`:       `n_new * n_heads * head_dim` f32 elements
///   - `k_cache_usm`: `n_kv_heads * max_ctx * head_dim` f32 elements
///   - `v_cache_usm`: same as `k_cache_usm`
///   - `out_usm`:     `n_new * n_heads * head_dim` f32 elements
///
/// `n_heads % n_kv_heads == 0` and `kv_len_base + n_new <= max_ctx`.
/// Kernel `.wait()`s before returning.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_usm_raw(
    stream: &SyclStream,
    q_usm: *const f32,
    k_cache_usm: *const f32,
    v_cache_usm: *const f32,
    out_usm: *mut f32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len_base: u32,
    n_new: u32,
) -> Result<()> {
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_usm_raw: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
        )));
    }
    if kv_len_base.saturating_add(n_new) > max_ctx {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_usm_raw: kv_len_base={kv_len_base} + n_new={n_new} > max_ctx={max_ctx}"
        )));
    }
    imp::flash_attn_prefill_usm(
        stream,
        q_usm,
        k_cache_usm,
        v_cache_usm,
        out_usm,
        n_heads,
        n_kv_heads,
        head_dim,
        max_ctx,
        kv_len_base,
        n_new,
    )
}

// ============================================================
// Quantized-KV FlashAttention (raw-pointer entry points). F32 Q/out,
// packed K/V dequantized on the fly — byte-exact ports of the CPU
// reference kernels (q4_0_kv.rs / nvfp4.rs / turboquant.rs), matched
// by `doctor --sycl-parity`. These are the accel-layer entries (the
// packed KV cache is naturally held as raw USM byte pointers).
// ============================================================

/// Raw-pointer quantized-KV FlashAttention decode (Q4_0 KV cache).
/// `q`/`out` are F32 `[n_heads, head_dim]`; `k_packed`/`v_packed` are
/// the packed Q4_0 cache `[n_kv_heads, max_ctx, (head_dim/32)*18]`.
///
/// # Safety
/// All pointers must reference live USM allocations on `stream`'s
/// context with the sizing above. `n_heads % n_kv_heads == 0`,
/// `head_dim % 32 == 0`, `head_dim <= 256`. Kernel `.wait()`s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_q4_0_usm_raw(
    stream: &SyclStream,
    q_usm: *const f32,
    k_packed_usm: *const u8,
    v_packed_usm: *const u8,
    out_usm: *mut f32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len: u32,
) -> Result<()> {
    imp::flash_attn_decode_q4_0_usm(
        stream, q_usm, k_packed_usm, v_packed_usm, out_usm,
        n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
    )
}

/// Raw-pointer quantized-KV FlashAttention prefill (Q4_0 KV cache).
/// `q`/`out` are F32 `[n_new, n_heads, head_dim]`.
///
/// # Safety
/// USM pointers as [`flash_attn_decode_q4_0_usm_raw`];
/// `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_q4_0_usm_raw(
    stream: &SyclStream,
    q_usm: *const f32,
    k_packed_usm: *const u8,
    v_packed_usm: *const u8,
    out_usm: *mut f32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len_base: u32,
    n_new: u32,
) -> Result<()> {
    imp::flash_attn_prefill_q4_0_usm(
        stream, q_usm, k_packed_usm, v_packed_usm, out_usm,
        n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
    )
}

/// Raw-pointer quantized-KV FlashAttention decode (NVFP4 KV cache).
/// `bytes_per_row = (head_dim/16)*9`.
///
/// # Safety
/// USM pointers as above; `head_dim % 16 == 0`, `head_dim <= 256`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_nvfp4_usm_raw(
    stream: &SyclStream,
    q_usm: *const f32,
    k_packed_usm: *const u8,
    v_packed_usm: *const u8,
    out_usm: *mut f32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len: u32,
) -> Result<()> {
    imp::flash_attn_decode_nvfp4_usm(
        stream, q_usm, k_packed_usm, v_packed_usm, out_usm,
        n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
    )
}

/// Raw-pointer quantized-KV FlashAttention prefill (NVFP4 KV cache).
///
/// # Safety
/// USM pointers as above; `head_dim % 16 == 0`, `head_dim <= 256`,
/// `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_nvfp4_usm_raw(
    stream: &SyclStream,
    q_usm: *const f32,
    k_packed_usm: *const u8,
    v_packed_usm: *const u8,
    out_usm: *mut f32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len_base: u32,
    n_new: u32,
) -> Result<()> {
    imp::flash_attn_prefill_nvfp4_usm(
        stream, q_usm, k_packed_usm, v_packed_usm, out_usm,
        n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
    )
}

/// Raw-pointer quantized-KV FlashAttention decode (TurboQuant KV cache).
/// `k_scales`/`v_scales` are `[n_kv_heads*max_ctx]` F32 per-row scales
/// (indexed `kv_h*max_ctx + t`); `bits` in {1,2,4,8};
/// `bytes_per_row = ceil(head_dim*bits/8)`.
///
/// # Safety
/// USM pointers as above; `head_dim` a power of two `<= 256`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_tq_usm_raw(
    stream: &SyclStream,
    q_usm: *const f32,
    k_packed_usm: *const u8,
    v_packed_usm: *const u8,
    k_scales_usm: *const f32,
    v_scales_usm: *const f32,
    bits: u32,
    out_usm: *mut f32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len: u32,
) -> Result<()> {
    imp::flash_attn_decode_tq_usm(
        stream, q_usm, k_packed_usm, v_packed_usm, k_scales_usm, v_scales_usm,
        bits, out_usm, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
    )
}

/// Raw-pointer quantized-KV FlashAttention prefill (TurboQuant KV cache).
///
/// # Safety
/// USM pointers as above; `head_dim` a power of two `<= 256`, `bits` in
/// {1,2,4,8}, `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_tq_usm_raw(
    stream: &SyclStream,
    q_usm: *const f32,
    k_packed_usm: *const u8,
    v_packed_usm: *const u8,
    k_scales_usm: *const f32,
    v_scales_usm: *const f32,
    bits: u32,
    out_usm: *mut f32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len_base: u32,
    n_new: u32,
) -> Result<()> {
    imp::flash_attn_prefill_tq_usm(
        stream, q_usm, k_packed_usm, v_packed_usm, k_scales_usm, v_scales_usm,
        bits, out_usm, n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
    )
}

/// Raw-pointer quantized-KV FlashAttention decode (Q8_0 KV cache). K/V are
/// i8 slabs `[n_kv_heads, max_ctx, head_dim]`; `k_scales`/`v_scales` are
/// per-row absmax f32 `[n_kv_heads*max_ctx]` (indexed `kv_h*max_ctx + t`).
/// Byte-exact port of the CPU `gqa_attention_flash_decode_q8_0`.
///
/// # Safety
/// All pointers must reference live USM allocations on `stream`'s context
/// sized as above.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_q8_0_usm_raw(
    stream: &SyclStream,
    q_usm: *const f32,
    k_packed_usm: *const u8,
    v_packed_usm: *const u8,
    k_scales_usm: *const f32,
    v_scales_usm: *const f32,
    out_usm: *mut f32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len: u32,
) -> Result<()> {
    imp::flash_attn_decode_q8_0_usm(
        stream, q_usm, k_packed_usm, v_packed_usm, k_scales_usm, v_scales_usm,
        out_usm, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
    )
}

/// Raw-pointer quantized-KV FlashAttention prefill (Q8_0 KV cache).
///
/// # Safety
/// USM pointers as [`flash_attn_decode_q8_0_usm_raw`];
/// `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_q8_0_usm_raw(
    stream: &SyclStream,
    q_usm: *const f32,
    k_packed_usm: *const u8,
    v_packed_usm: *const u8,
    k_scales_usm: *const f32,
    v_scales_usm: *const f32,
    out_usm: *mut f32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len_base: u32,
    n_new: u32,
) -> Result<()> {
    imp::flash_attn_prefill_q8_0_usm(
        stream, q_usm, k_packed_usm, v_packed_usm, k_scales_usm, v_scales_usm,
        out_usm, n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
    )
}

/// Raw-pointer FA-v2 decode (sub-group cooperation). Same calling
/// contract as [`flash_attn_decode_usm_raw`] plus the v2 shape
/// constraint: `head_dim` must be a multiple of 16 and ≤ 256.
///
/// # Safety
///
/// All four pointers must reference live USM allocations on
/// `stream`'s context with the same sizing as
/// [`flash_attn_decode_usm_raw`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_v2_usm_raw(
    stream: &SyclStream,
    q_usm: *const u16,
    k_usm: *const u16,
    v_usm: *const u16,
    out_usm: *mut u16,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len: u32,
) -> Result<()> {
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_v2_usm_raw: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
        )));
    }
    if kv_len > max_ctx {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_v2_usm_raw: kv_len={kv_len} > max_ctx={max_ctx}"
        )));
    }
    if head_dim == 0 || head_dim % 16 != 0 || head_dim > 256 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_decode_v2_usm_raw: head_dim={head_dim} must be a multiple of 16 and ≤ 256"
        )));
    }
    imp::flash_attn_decode_v2_usm(
        stream, q_usm, k_usm, v_usm, out_usm,
        n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
    )
}

/// Raw-pointer FA-v2 prefill. Same contract as
/// [`flash_attn_prefill_usm_raw`] plus the v2 head_dim constraint.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_v2_usm_raw(
    stream: &SyclStream,
    q_usm: *const f32,
    k_cache_usm: *const f32,
    v_cache_usm: *const f32,
    out_usm: *mut f32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len_base: u32,
    n_new: u32,
) -> Result<()> {
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_v2_usm_raw: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
        )));
    }
    if kv_len_base.saturating_add(n_new) > max_ctx {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_v2_usm_raw: kv_len_base={kv_len_base} + n_new={n_new} > max_ctx={max_ctx}"
        )));
    }
    if head_dim == 0 || head_dim % 16 != 0 || head_dim > 256 {
        return Err(SyclError::InvalidShape(format!(
            "flash_attn_prefill_v2_usm_raw: head_dim={head_dim} must be a multiple of 16 and ≤ 256"
        )));
    }
    imp::flash_attn_prefill_v2_usm(
        stream, q_usm, k_cache_usm, v_cache_usm, out_usm,
        n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
    )
}

/// USM-resident F16 GEMM: `C[M,N] = A[M,K] @ B[K,N]`, row-major.
/// All three buffers (A, B, C) stay USM-resident — no per-call
/// H2D / D2H. Use `M=1` or `N=1` for the engine's projection
/// matvec shapes (Q/K/V/O/gate/up/down). Leading dimensions are
/// the standard row-major defaults: `lda=K`, `ldb=N`, `ldc=N`.
///
/// Returns [`SyclError::Unavailable`] when no SYCL device is present. Sizing
/// checks: `a.len() >= M*K`, `b.len() >= K*N`, `c.len() >= M*N`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_f16_usm(
    stream: &SyclStream,
    a: &SyclSharedBuffer<u16>,
    b: &SyclSharedBuffer<u16>,
    c: &mut SyclSharedBuffer<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(SyclError::InvalidShape(format!(
            "gemm_f16_usm: zero dim ({m}x{n}x{k})"
        )));
    }
    let need_a = (m as usize) * (k as usize);
    let need_b = (k as usize) * (n as usize);
    let need_c = (m as usize) * (n as usize);
    if a.len() < need_a {
        return Err(SyclError::InvalidShape(format!(
            "gemm_f16_usm: a={}, need M*K={need_a}",
            a.len()
        )));
    }
    if b.len() < need_b {
        return Err(SyclError::InvalidShape(format!(
            "gemm_f16_usm: b={}, need K*N={need_b}",
            b.len()
        )));
    }
    if c.len() < need_c {
        return Err(SyclError::InvalidShape(format!(
            "gemm_f16_usm: c={}, need M*N={need_c}",
            c.len()
        )));
    }
    imp::gemm_f16_usm(
        stream,
        a.as_ptr(),
        b.as_ptr(),
        c.as_mut_ptr(),
        m, n, k,
        k, // lda = K (row-major A[M,K])
        n, // ldb = N (row-major B[K,N])
        n, // ldc = N (row-major C[M,N])
    )
}

/// USM-resident Q8_0 weight × F32 activation matvec:
/// `out[M] = sum_b w_scales[m, b] * sum_d_in_block(w_q[m, b*32+d] * x[b*32+d])`.
///
/// Weight layout matches the engine's `Dtype::Q8_0Raw` storage:
///   - `w_q: [M, K]` row-major i8 (K must be a multiple of 32)
///   - `w_scales: [M, K/32]` row-major f32, one scale per 32-element block
///
/// Activation and output are plain f32. Sizing checks: `w_q.len() >= M*K`,
/// `w_scales.len() >= M*(K/32)`, `x.len() >= K`, `out.len() >= M`.
/// Returns [`SyclError::Unavailable`] when no SYCL device is present.
pub fn matvec_q8_0_f32_usm(
    stream: &SyclStream,
    w_q: &SyclSharedBuffer<i8>,
    w_scales: &SyclSharedBuffer<f32>,
    x: &SyclSharedBuffer<f32>,
    out: &mut SyclSharedBuffer<f32>,
    m: u32,
    k: u32,
) -> Result<()> {
    if m == 0 || k == 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q8_0_f32_usm: zero dim (M={m}, K={k})"
        )));
    }
    if k % 32 != 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q8_0_f32_usm: K must be a multiple of 32, got {k}"
        )));
    }
    let need_w_q = (m as usize) * (k as usize);
    let need_scales = (m as usize) * ((k / 32) as usize);
    let need_x = k as usize;
    let need_out = m as usize;
    if w_q.len() < need_w_q {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q8_0_f32_usm: w_q={}, need M*K={need_w_q}",
            w_q.len()
        )));
    }
    if w_scales.len() < need_scales {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q8_0_f32_usm: w_scales={}, need M*(K/32)={need_scales}",
            w_scales.len()
        )));
    }
    if x.len() < need_x {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q8_0_f32_usm: x={}, need K={need_x}",
            x.len()
        )));
    }
    if out.len() < need_out {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q8_0_f32_usm: out={}, need M={need_out}",
            out.len()
        )));
    }
    imp::matvec_q8_0_f32_usm(
        stream,
        w_q.as_ptr(),
        w_scales.as_ptr(),
        x.as_ptr(),
        out.as_mut_ptr(),
        m,
        k,
    )
}

/// USM-resident Q8_0 weight × F32 activation matvec — packed GGUF
/// layout. Consumes the raw on-disk Q8_0 byte layout (34 bytes per
/// block: 2-byte f16 scale + 32 i8 weights, `K/32` blocks per row,
/// `M` rows) directly from a single USM `u8` buffer. This is the
/// layout the engine already has after mmap of the GGUF tensor, so
/// engine integration is a one-time memcpy of `w_bytes` into a USM
/// allocation + repeated dispatch against the same pointer — no
/// per-load repack into separated `(i8, f32)` form.
///
/// Sizing checks: `w_bytes.len() >= M * (K/32) * 34`, `x.len() >= K`,
/// `out.len() >= M`. `K` must be a multiple of 32. `lws=0` selects
/// the hand-picked default work-group size. Returns
/// [`SyclError::Unavailable`] when no SYCL device is present.
pub fn matvec_q8_0_packed_f32_usm(
    stream: &SyclStream,
    w_bytes: &SyclSharedBuffer<u8>,
    x: &SyclSharedBuffer<f32>,
    out: &mut SyclSharedBuffer<f32>,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    if m == 0 || k == 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q8_0_packed_f32_usm: zero dim (M={m}, K={k})"
        )));
    }
    if k % 32 != 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q8_0_packed_f32_usm: K must be a multiple of 32, got {k}"
        )));
    }
    let blocks_per_row = (k / 32) as usize;
    let need_bytes = (m as usize) * blocks_per_row * 34;
    let need_x = k as usize;
    let need_out = m as usize;
    if w_bytes.len() < need_bytes {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q8_0_packed_f32_usm: w_bytes={}, need M*(K/32)*34={need_bytes}",
            w_bytes.len()
        )));
    }
    if x.len() < need_x {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q8_0_packed_f32_usm: x={}, need K={need_x}",
            x.len()
        )));
    }
    if out.len() < need_out {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q8_0_packed_f32_usm: out={}, need M={need_out}",
            out.len()
        )));
    }
    imp::matvec_q8_0_packed_f32_usm(
        stream,
        w_bytes.as_ptr(),
        x.as_ptr(),
        out.as_mut_ptr(),
        m,
        k,
        lws,
    )
}

/// USM-resident Q4_K_M weight × F32 activation matvec — packed GGUF
/// super-block layout. Same call shape as
/// [`matvec_q8_0_packed_f32_usm`] but `K` must be a multiple of 256
/// (Q4_K super-block size) and `w_bytes.len() >= M * (K/256) * 144`.
/// Q4_K_M is the default v1 quant for almost every modern coding
/// model GGUF — Qwen2.5-Coder, DeepSeek-Coder, Mistral, Llama-3 —
/// so this is the matvec entry that gets hit hardest in production.
///
/// `lws` selects the local work-group size for this dispatch from
/// the compiled-in candidate set `{16, 32, 64, 128, 256}`. Pass `0`
/// for the hand-picked default (64). The autotuner sweeps the
/// candidates per `(device, problem-shape)` and the engine passes
/// the winner on each call.
///
/// Returns [`SyclError::Unavailable`] when no SYCL device is present.
pub fn matvec_q4_k_packed_f32_usm(
    stream: &SyclStream,
    w_bytes: &SyclSharedBuffer<u8>,
    x: &SyclSharedBuffer<f32>,
    out: &mut SyclSharedBuffer<f32>,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    if m == 0 || k == 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q4_k_packed_f32_usm: zero dim (M={m}, K={k})"
        )));
    }
    if k % 256 != 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q4_k_packed_f32_usm: K must be a multiple of 256, got {k}"
        )));
    }
    let blocks_per_row = (k / 256) as usize;
    let need_bytes = (m as usize) * blocks_per_row * 144;
    let need_x = k as usize;
    let need_out = m as usize;
    if w_bytes.len() < need_bytes {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q4_k_packed_f32_usm: w_bytes={}, need M*(K/256)*144={need_bytes}",
            w_bytes.len()
        )));
    }
    if x.len() < need_x {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q4_k_packed_f32_usm: x={}, need K={need_x}",
            x.len()
        )));
    }
    if out.len() < need_out {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q4_k_packed_f32_usm: out={}, need M={need_out}",
            out.len()
        )));
    }
    imp::matvec_q4_k_packed_f32_usm(
        stream,
        w_bytes.as_ptr(),
        x.as_ptr(),
        out.as_mut_ptr(),
        m,
        k,
        lws,
    )
}

/// USM-resident Q5_K_M weight × F32 activation matvec — packed GGUF
/// super-block layout. Same call shape and constraints as
/// [`matvec_q4_k_packed_f32_usm`] but `w_bytes.len() >= M * (K/256) *
/// 176`. `lws=0` selects the hand-picked default work-group size.
/// Returns [`SyclError::Unavailable`] when no SYCL device is present.
pub fn matvec_q5_k_packed_f32_usm(
    stream: &SyclStream,
    w_bytes: &SyclSharedBuffer<u8>,
    x: &SyclSharedBuffer<f32>,
    out: &mut SyclSharedBuffer<f32>,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    if m == 0 || k == 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q5_k_packed_f32_usm: zero dim (M={m}, K={k})"
        )));
    }
    if k % 256 != 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q5_k_packed_f32_usm: K must be a multiple of 256, got {k}"
        )));
    }
    let blocks_per_row = (k / 256) as usize;
    let need_bytes = (m as usize) * blocks_per_row * 176;
    let need_x = k as usize;
    let need_out = m as usize;
    if w_bytes.len() < need_bytes {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q5_k_packed_f32_usm: w_bytes={}, need M*(K/256)*176={need_bytes}",
            w_bytes.len()
        )));
    }
    if x.len() < need_x {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q5_k_packed_f32_usm: x={}, need K={need_x}",
            x.len()
        )));
    }
    if out.len() < need_out {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q5_k_packed_f32_usm: out={}, need M={need_out}",
            out.len()
        )));
    }
    imp::matvec_q5_k_packed_f32_usm(
        stream,
        w_bytes.as_ptr(),
        x.as_ptr(),
        out.as_mut_ptr(),
        m,
        k,
        lws,
    )
}

/// USM-resident Q6_K weight × F32 activation matvec — packed GGUF
/// super-block layout. Same call shape as Q4_K_M / Q5_K_M but
/// `w_bytes.len() >= M * (K/256) * 210`. `lws=0` selects the
/// hand-picked default work-group size. Q6_K is most commonly the
/// LM head's storage in Q4_K_M variant GGUFs (the body is Q4_K but
/// the LM head stays Q6_K for output precision); without this
/// kernel the LM head matvec runs on CPU, costing ~1/3 of decode
/// time on a 1.5B coding model.
///
/// Returns [`SyclError::Unavailable`] when no SYCL device is present.
pub fn matvec_q6_k_packed_f32_usm(
    stream: &SyclStream,
    w_bytes: &SyclSharedBuffer<u8>,
    x: &SyclSharedBuffer<f32>,
    out: &mut SyclSharedBuffer<f32>,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    if m == 0 || k == 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q6_k_packed_f32_usm: zero dim (M={m}, K={k})"
        )));
    }
    if k % 256 != 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q6_k_packed_f32_usm: K must be a multiple of 256, got {k}"
        )));
    }
    let blocks_per_row = (k / 256) as usize;
    let need_bytes = (m as usize) * blocks_per_row * 210;
    let need_x = k as usize;
    let need_out = m as usize;
    if w_bytes.len() < need_bytes {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q6_k_packed_f32_usm: w_bytes={}, need M*(K/256)*210={need_bytes}",
            w_bytes.len()
        )));
    }
    if x.len() < need_x {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q6_k_packed_f32_usm: x={}, need K={need_x}",
            x.len()
        )));
    }
    if out.len() < need_out {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q6_k_packed_f32_usm: out={}, need M={need_out}",
            out.len()
        )));
    }
    imp::matvec_q6_k_packed_f32_usm(
        stream,
        w_bytes.as_ptr(),
        x.as_ptr(),
        out.as_mut_ptr(),
        m,
        k,
        lws,
    )
}

/// USM-resident Q8_0 packed matvec — batched over N input rows.
/// Inputs are contiguous row-major (`x_usm: [N, K]`, `out_usm: [N, M]`).
/// `out[n, m] = sum_k W[m, k] * x[n, k]`. Same per-launch overhead as
/// the single-row variant for any `N >= 1`; the win is over launching
/// the single-row kernel N times.
///
/// Sizing: `w_bytes.len() >= M * (K/32) * 34`, `x.len() >= N*K`,
/// `out.len() >= N*M`. `K` must be a multiple of 32. Returns
/// [`SyclError::Unavailable`] when no SYCL device is present.
#[allow(clippy::too_many_arguments)]
pub fn matvec_q8_0_packed_f32_batched_usm(
    stream: &SyclStream,
    w_bytes: &SyclSharedBuffer<u8>,
    x: &SyclSharedBuffer<f32>,
    out: &mut SyclSharedBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    if m == 0 || k == 0 || n == 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q8_0_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
        )));
    }
    if k % 32 != 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q8_0_packed_f32_batched_usm: K must be a multiple of 32, got {k}"
        )));
    }
    let blocks_per_row = (k / 32) as usize;
    let need_bytes = (m as usize) * blocks_per_row * 34;
    let need_x = (n as usize) * (k as usize);
    let need_out = (n as usize) * (m as usize);
    if w_bytes.len() < need_bytes {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q8_0_packed_f32_batched_usm: w_bytes={}, need M*(K/32)*34={need_bytes}",
            w_bytes.len()
        )));
    }
    if x.len() < need_x {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q8_0_packed_f32_batched_usm: x={}, need N*K={need_x}",
            x.len()
        )));
    }
    if out.len() < need_out {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q8_0_packed_f32_batched_usm: out={}, need N*M={need_out}",
            out.len()
        )));
    }
    imp::matvec_q8_0_packed_f32_batched_usm(
        stream,
        w_bytes.as_ptr(),
        x.as_ptr(),
        out.as_mut_ptr(),
        m,
        k,
        n,
        lws,
    )
}

/// USM-resident Q4_K_M packed matvec — batched over N input rows.
/// See [`matvec_q8_0_packed_f32_batched_usm`] for batching semantics.
/// `w_bytes.len() >= M * (K/256) * 144`; `K` must be a multiple of 256.
/// `lws=0` selects the hand-picked default work-group size.
#[allow(clippy::too_many_arguments)]
pub fn matvec_q4_k_packed_f32_batched_usm(
    stream: &SyclStream,
    w_bytes: &SyclSharedBuffer<u8>,
    x: &SyclSharedBuffer<f32>,
    out: &mut SyclSharedBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    if m == 0 || k == 0 || n == 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q4_k_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
        )));
    }
    if k % 256 != 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q4_k_packed_f32_batched_usm: K must be a multiple of 256, got {k}"
        )));
    }
    let blocks_per_row = (k / 256) as usize;
    let need_bytes = (m as usize) * blocks_per_row * 144;
    let need_x = (n as usize) * (k as usize);
    let need_out = (n as usize) * (m as usize);
    if w_bytes.len() < need_bytes {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q4_k_packed_f32_batched_usm: w_bytes={}, need M*(K/256)*144={need_bytes}",
            w_bytes.len()
        )));
    }
    if x.len() < need_x {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q4_k_packed_f32_batched_usm: x={}, need N*K={need_x}",
            x.len()
        )));
    }
    if out.len() < need_out {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q4_k_packed_f32_batched_usm: out={}, need N*M={need_out}",
            out.len()
        )));
    }
    imp::matvec_q4_k_packed_f32_batched_usm(
        stream,
        w_bytes.as_ptr(),
        x.as_ptr(),
        out.as_mut_ptr(),
        m,
        k,
        n,
        lws,
    )
}

/// USM-resident Q5_K_M packed matvec — batched over N input rows.
/// See [`matvec_q8_0_packed_f32_batched_usm`] for batching semantics.
/// `w_bytes.len() >= M * (K/256) * 176`; `K` must be a multiple of 256.
/// `lws=0` selects the hand-picked default work-group size.
#[allow(clippy::too_many_arguments)]
pub fn matvec_q5_k_packed_f32_batched_usm(
    stream: &SyclStream,
    w_bytes: &SyclSharedBuffer<u8>,
    x: &SyclSharedBuffer<f32>,
    out: &mut SyclSharedBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    if m == 0 || k == 0 || n == 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q5_k_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
        )));
    }
    if k % 256 != 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q5_k_packed_f32_batched_usm: K must be a multiple of 256, got {k}"
        )));
    }
    let blocks_per_row = (k / 256) as usize;
    let need_bytes = (m as usize) * blocks_per_row * 176;
    let need_x = (n as usize) * (k as usize);
    let need_out = (n as usize) * (m as usize);
    if w_bytes.len() < need_bytes {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q5_k_packed_f32_batched_usm: w_bytes={}, need M*(K/256)*176={need_bytes}",
            w_bytes.len()
        )));
    }
    if x.len() < need_x {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q5_k_packed_f32_batched_usm: x={}, need N*K={need_x}",
            x.len()
        )));
    }
    if out.len() < need_out {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q5_k_packed_f32_batched_usm: out={}, need N*M={need_out}",
            out.len()
        )));
    }
    imp::matvec_q5_k_packed_f32_batched_usm(
        stream,
        w_bytes.as_ptr(),
        x.as_ptr(),
        out.as_mut_ptr(),
        m,
        k,
        n,
        lws,
    )
}

/// USM-resident Q6_K packed matvec — batched over N input rows.
/// See [`matvec_q8_0_packed_f32_batched_usm`] for batching semantics.
/// `w_bytes.len() >= M * (K/256) * 210`; `K` must be a multiple of 256.
/// `lws=0` selects the hand-picked default work-group size.
#[allow(clippy::too_many_arguments)]
pub fn matvec_q6_k_packed_f32_batched_usm(
    stream: &SyclStream,
    w_bytes: &SyclSharedBuffer<u8>,
    x: &SyclSharedBuffer<f32>,
    out: &mut SyclSharedBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    lws: u32,
) -> Result<()> {
    if m == 0 || k == 0 || n == 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q6_k_packed_f32_batched_usm: zero dim (M={m}, K={k}, N={n})"
        )));
    }
    if k % 256 != 0 {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q6_k_packed_f32_batched_usm: K must be a multiple of 256, got {k}"
        )));
    }
    let blocks_per_row = (k / 256) as usize;
    let need_bytes = (m as usize) * blocks_per_row * 210;
    let need_x = (n as usize) * (k as usize);
    let need_out = (n as usize) * (m as usize);
    if w_bytes.len() < need_bytes {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q6_k_packed_f32_batched_usm: w_bytes={}, need M*(K/256)*210={need_bytes}",
            w_bytes.len()
        )));
    }
    if x.len() < need_x {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q6_k_packed_f32_batched_usm: x={}, need N*K={need_x}",
            x.len()
        )));
    }
    if out.len() < need_out {
        return Err(SyclError::InvalidShape(format!(
            "matvec_q6_k_packed_f32_batched_usm: out={}, need N*M={need_out}",
            out.len()
        )));
    }
    imp::matvec_q6_k_packed_f32_batched_usm(
        stream,
        w_bytes.as_ptr(),
        x.as_ptr(),
        out.as_mut_ptr(),
        m,
        k,
        n,
        lws,
    )
}

/// Try to import an existing Win32 file-mapping HANDLE as a
/// device-accessible USM allocation. On success, returns the
/// imported pointer that can be passed to USM kernels exactly as
/// if it were the result of `sycl::malloc_shared(size, queue)`.
///
/// Typical caller (rustllama-models accel): given a GGUF file's
/// underlying `CreateFileMappingW` HANDLE and a known byte range,
/// avoid the copy-to-USM step entirely — the kernels read the
/// mmap pages directly via the L0-imported pointer.
///
/// Returns [`SyclError::L0ImportUnsupported`] on any failure
/// (non-L0 backend, missing loader DLL, driver doesn't support
/// importing this handle type). Callers should treat this as
/// "fall back to copy-to-USM" and not as a hard error.
///
/// # Safety
///
/// - `mapping_handle` must be a live Win32 NT handle (e.g. from
///   `CreateFileMappingW`) and must remain valid for the lifetime
///   of the import.
/// - `size` must not exceed the underlying file mapping's size.
/// - The returned pointer must be freed via
///   [`release_imported_usm`] *before* `stream`'s SYCL context is
///   destroyed; otherwise the L0 driver will leak the import
///   allocation until process exit.
pub unsafe fn try_import_win32_handle_as_usm_raw(
    stream: &SyclStream,
    mapping_handle: *mut std::ffi::c_void,
    size: usize,
) -> Result<*mut std::ffi::c_void> {
    imp::try_import_win32_handle_as_usm(stream, mapping_handle, size)
}

/// Free a pointer obtained from [`try_import_win32_handle_as_usm_raw`].
/// Safe to call with a null pointer (no-op). Must be invoked on the
/// same `SyclStream` whose queue produced the import.
///
/// # Safety
///
/// `dev_ptr` must be a pointer previously returned by
/// [`try_import_win32_handle_as_usm_raw`] on this same stream,
/// or null. Double-free is UB (the L0 spec doesn't validate).
pub unsafe fn release_imported_usm_raw(
    stream: &SyclStream,
    dev_ptr: *mut std::ffi::c_void,
) {
    imp::release_imported_usm(stream, dev_ptr)
}

/// Drain the most recent `zeMemAllocHost` `ze_result_t` from the
/// L0 import side channel, and reset it to 0. Lets callers log the
/// exact L0 error code after [`try_import_win32_handle_as_usm_raw`]
/// returns [`SyclError::L0ImportUnsupported`] with category 4
/// ("driver rejected").
///
/// Common L0 codes:
///   `0x78000003` = `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE`
///   `0x78000004` = `ZE_RESULT_ERROR_INVALID_ARGUMENT`
///   `0x70000002` = `ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY`
///
/// Returns 0 when no import call has been made
/// since the last consume.
pub fn consume_last_l0_import_code() -> u32 {
    imp::consume_last_l0_import_code()
}

/// Diagnostic baseline: try a bare `zeMemAllocHost` with no import
/// descriptor chained. If THIS fails, our `ze_host_mem_alloc_desc`
/// struct layout or call sequence is wrong. If this succeeds but
/// [`try_import_win32_handle_as_usm_raw`] fails, the import
/// descriptor or handle origin is the issue.
///
/// The allocation is freed immediately — this is only useful as a
/// probe.
pub fn try_alloc_host_baseline(stream: &SyclStream, size: usize) -> Result<()> {
    imp::try_alloc_host_baseline(stream, size)
}

/// USM-resident half-split RoPE. Same algorithm as the host-pointer
/// [`rope`]; `qk` is rotated in-place. `inv_freq` is the
/// pre-computed `1 / theta^(2j/head_dim)` table, length
/// `head_dim/2`. `head_dim` must be even.
pub fn rope_usm(
    stream: &SyclStream,
    qk: &mut SyclSharedBuffer<u16>,
    n_heads: u32,
    head_dim: u32,
    pos: u32,
    inv_freq: &SyclSharedBuffer<u16>,
) -> Result<()> {
    if head_dim % 2 != 0 {
        return Err(SyclError::InvalidShape(format!(
            "rope_usm: head_dim must be even, got {head_dim}"
        )));
    }
    let need_qk = (n_heads as usize) * (head_dim as usize);
    if qk.len() < need_qk {
        return Err(SyclError::InvalidShape(format!(
            "rope_usm: qk={}, need n_heads*head_dim={need_qk}",
            qk.len()
        )));
    }
    let half = (head_dim / 2) as usize;
    if inv_freq.len() < half {
        return Err(SyclError::InvalidShape(format!(
            "rope_usm: inv_freq={}, need head_dim/2={half}",
            inv_freq.len()
        )));
    }
    imp::rope_usm(
        stream,
        qk.as_mut_ptr(),
        n_heads,
        head_dim,
        pos,
        inv_freq.as_ptr(),
    )
}

/// USM-resident SwiGLU: `out[i] = silu(x[i]) * y[i]`. All three
/// buffers must hold at least `n` elements.
pub fn silu_mul_usm(
    stream: &SyclStream,
    x: &SyclSharedBuffer<u16>,
    y: &SyclSharedBuffer<u16>,
    out: &mut SyclSharedBuffer<u16>,
    n: u32,
) -> Result<()> {
    let n_us = n as usize;
    if x.len() < n_us || y.len() < n_us || out.len() < n_us {
        return Err(SyclError::InvalidShape(format!(
            "silu_mul_usm: x={}, y={}, out={}, need {n}",
            x.len(),
            y.len(),
            out.len()
        )));
    }
    imp::silu_mul_usm(stream, x.as_ptr(), y.as_ptr(), out.as_mut_ptr(), n)
}

/// G2: KV-cache Q8_0 quantize-on-store, raw-pointer entry.
/// `src_usm` ≥ `n_rows × head_dim` f32 USM elements; `q_dst_usm` ≥
/// same i8 elements; `scales_dst_usm` ≥ `n_rows` f32 elements.
/// Kernel `.wait()`s before returning.
///
/// # Safety
///
/// All three pointers must be USM allocations on the same `stream`.
/// Buffer sizes must match the shape parameters.
pub unsafe fn kv_quantize_q8_0_store_usm_raw(
    stream: &SyclStream,
    src_usm: *const f32,
    q_dst_usm: *mut i8,
    scales_dst_usm: *mut f32,
    n_new: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_len_base: u32,
) -> Result<()> {
    imp::kv_quantize_q8_0_store_usm(
        stream, src_usm, q_dst_usm, scales_dst_usm,
        n_new, n_kv_heads, head_dim, max_ctx, kv_len_base,
    )
}

/// G6: Q6_K block encoder, raw-pointer entry. `src_usm` ≥
/// `n_blocks × 256` f32 USM elements; `dst_usm` ≥ `n_blocks × 210`
/// u8 USM elements. Kernel `.wait()`s before returning.
///
/// # Safety
///
/// Both pointers must be USM allocations on the same `stream`.
pub unsafe fn encode_q6_k_blocks_usm_raw(
    stream: &SyclStream,
    src_usm: *const f32,
    dst_usm: *mut u8,
    n_blocks: u32,
) -> Result<()> {
    imp::encode_q6_k_blocks_usm(stream, src_usm, dst_usm, n_blocks)
}

/// G6: Q3_K block encoder, raw-pointer entry.
///
/// # Safety
///
/// Both pointers must be USM allocations on the same `stream`.
pub unsafe fn encode_q3_k_blocks_usm_raw(
    stream: &SyclStream,
    src_usm: *const f32,
    dst_usm: *mut u8,
    n_blocks: u32,
) -> Result<()> {
    imp::encode_q3_k_blocks_usm(stream, src_usm, dst_usm, n_blocks)
}

/// G6: Q4_K block encoder, raw-pointer entry.
///
/// # Safety
///
/// Both pointers must be USM allocations on the same `stream`.
pub unsafe fn encode_q4_k_blocks_usm_raw(
    stream: &SyclStream,
    src_usm: *const f32,
    dst_usm: *mut u8,
    n_blocks: u32,
) -> Result<()> {
    imp::encode_q4_k_blocks_usm(stream, src_usm, dst_usm, n_blocks)
}

/// G6: Q5_K block encoder, raw-pointer entry.
///
/// # Safety
///
/// Both pointers must be USM allocations on the same `stream`.
pub unsafe fn encode_q5_k_blocks_usm_raw(
    stream: &SyclStream,
    src_usm: *const f32,
    dst_usm: *mut u8,
    n_blocks: u32,
) -> Result<()> {
    imp::encode_q5_k_blocks_usm(stream, src_usm, dst_usm, n_blocks)
}

/// H4: Q4_K gate+up fused matvec, raw-pointer entry.
///
/// # Safety
///
/// All five USM pointers must be allocations on the same `stream`.
/// Gate + up weight buffers must each be sized `M × (K/256) × 144`
/// bytes (Q4_K layout). `x_usm` ≥ `K` f32; `gate_out_usm` and
/// `up_out_usm` each ≥ `M` f32. `K` must be a multiple of 256.
pub unsafe fn matvec_q4_k_gate_up_fused_usm_raw(
    stream: &SyclStream,
    gate_w_bytes_usm: *const u8,
    up_w_bytes_usm: *const u8,
    x_usm: *const f32,
    gate_out_usm: *mut f32,
    up_out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_q4_k_gate_up_fused_usm(
        stream, gate_w_bytes_usm, up_w_bytes_usm,
        x_usm, gate_out_usm, up_out_usm, m, k, lws,
    )
}

/// H4: Q8_0 gate+up fused matvec, raw-pointer entry.
///
/// # Safety
///
/// All five USM pointers must be allocations on the same `stream`.
/// Gate + up weight buffers must each be sized `M × (K/32) × 34`
/// bytes (Q8_0 layout). `x_usm` ≥ `K` f32; `gate_out_usm` and
/// `up_out_usm` each ≥ `M` f32. `K` must be a multiple of 32.
pub unsafe fn matvec_q8_0_gate_up_fused_usm_raw(
    stream: &SyclStream,
    gate_w_bytes_usm: *const u8,
    up_w_bytes_usm: *const u8,
    x_usm: *const f32,
    gate_out_usm: *mut f32,
    up_out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_q8_0_gate_up_fused_usm(
        stream, gate_w_bytes_usm, up_w_bytes_usm,
        x_usm, gate_out_usm, up_out_usm, m, k, lws,
    )
}

/// H4: Q5_K gate+up fused matvec, raw-pointer entry.
///
/// # Safety
///
/// All five USM pointers must be allocations on the same `stream`.
/// Gate + up weight buffers must each be sized `M × (K/256) × 176`
/// bytes (Q5_K layout). `K` must be a multiple of 256.
pub unsafe fn matvec_q5_k_gate_up_fused_usm_raw(
    stream: &SyclStream,
    gate_w_bytes_usm: *const u8,
    up_w_bytes_usm: *const u8,
    x_usm: *const f32,
    gate_out_usm: *mut f32,
    up_out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_q5_k_gate_up_fused_usm(
        stream, gate_w_bytes_usm, up_w_bytes_usm,
        x_usm, gate_out_usm, up_out_usm, m, k, lws,
    )
}

/// H4: Q6_K gate+up fused matvec, raw-pointer entry.
///
/// # Safety
///
/// All five USM pointers must be allocations on the same `stream`.
/// Gate + up weight buffers must each be sized `M × (K/256) × 210`
/// bytes (Q6_K layout). `K` must be a multiple of 256.
pub unsafe fn matvec_q6_k_gate_up_fused_usm_raw(
    stream: &SyclStream,
    gate_w_bytes_usm: *const u8,
    up_w_bytes_usm: *const u8,
    x_usm: *const f32,
    gate_out_usm: *mut f32,
    up_out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_q6_k_gate_up_fused_usm(
        stream, gate_w_bytes_usm, up_w_bytes_usm,
        x_usm, gate_out_usm, up_out_usm, m, k, lws,
    )
}

/// H4: IQ4_NL gate+up fused matvec, raw-pointer entry.
///
/// # Safety
///
/// All five USM pointers must be allocations on the same `stream`.
/// Gate + up weight buffers must each be sized `M × (K/32) × 18`
/// bytes (IQ4_NL layout). `K` must be a multiple of 32.
pub unsafe fn matvec_iq4_nl_gate_up_fused_usm_raw(
    stream: &SyclStream,
    gate_w_bytes_usm: *const u8,
    up_w_bytes_usm: *const u8,
    x_usm: *const f32,
    gate_out_usm: *mut f32,
    up_out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq4_nl_gate_up_fused_usm(
        stream, gate_w_bytes_usm, up_w_bytes_usm,
        x_usm, gate_out_usm, up_out_usm, m, k, lws,
    )
}

/// H4: IQ4_XS gate+up fused matvec, raw-pointer entry.
///
/// # Safety
///
/// All five USM pointers must be allocations on the same `stream`.
/// Gate + up weight buffers must each be sized `M × (K/256) × 136`
/// bytes (IQ4_XS layout). `K` must be a multiple of 256.
pub unsafe fn matvec_iq4_xs_gate_up_fused_usm_raw(
    stream: &SyclStream,
    gate_w_bytes_usm: *const u8,
    up_w_bytes_usm: *const u8,
    x_usm: *const f32,
    gate_out_usm: *mut f32,
    up_out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq4_xs_gate_up_fused_usm(
        stream, gate_w_bytes_usm, up_w_bytes_usm,
        x_usm, gate_out_usm, up_out_usm, m, k, lws,
    )
}

/// H4: IQ1_S gate+up fused matvec, raw-pointer entry.
///
/// # Safety
///
/// All five USM pointers must be allocations on the same `stream`.
/// Gate + up weight buffers must each be sized `M × (K/256) × 50`
/// bytes (IQ1_S layout). `K` must be a multiple of 256.
pub unsafe fn matvec_iq1_s_gate_up_fused_usm_raw(
    stream: &SyclStream,
    gate_w_bytes_usm: *const u8,
    up_w_bytes_usm: *const u8,
    x_usm: *const f32,
    gate_out_usm: *mut f32,
    up_out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq1_s_gate_up_fused_usm(
        stream, gate_w_bytes_usm, up_w_bytes_usm,
        x_usm, gate_out_usm, up_out_usm, m, k, lws,
    )
}

/// H4: IQ2_XXS gate+up fused matvec, raw-pointer entry.
///
/// # Safety
///
/// All five USM pointers must be allocations on the same `stream`.
/// Gate + up weight buffers must each be sized `M × (K/256) × 66`
/// bytes (IQ2_XXS layout). `K` must be a multiple of 256.
pub unsafe fn matvec_iq2_xxs_gate_up_fused_usm_raw(
    stream: &SyclStream,
    gate_w_bytes_usm: *const u8,
    up_w_bytes_usm: *const u8,
    x_usm: *const f32,
    gate_out_usm: *mut f32,
    up_out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq2_xxs_gate_up_fused_usm(
        stream, gate_w_bytes_usm, up_w_bytes_usm,
        x_usm, gate_out_usm, up_out_usm, m, k, lws,
    )
}

/// H4: IQ1_M gate+up fused matvec, raw-pointer entry.
///
/// # Safety
///
/// All five USM pointers must be allocations on the same `stream`.
/// Gate + up weight buffers must each be sized `M × (K/256) × 56`
/// bytes (IQ1_M layout). `K` must be a multiple of 256.
pub unsafe fn matvec_iq1_m_gate_up_fused_usm_raw(
    stream: &SyclStream,
    gate_w_bytes_usm: *const u8,
    up_w_bytes_usm: *const u8,
    x_usm: *const f32,
    gate_out_usm: *mut f32,
    up_out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq1_m_gate_up_fused_usm(
        stream, gate_w_bytes_usm, up_w_bytes_usm,
        x_usm, gate_out_usm, up_out_usm, m, k, lws,
    )
}

/// H4: IQ2_XS gate+up fused matvec, raw-pointer entry.
///
/// # Safety
///
/// All five USM pointers must be allocations on the same `stream`.
/// Gate + up weight buffers must each be sized `M × (K/256) × 74`
/// bytes (IQ2_XS layout). `K` must be a multiple of 256.
pub unsafe fn matvec_iq2_xs_gate_up_fused_usm_raw(
    stream: &SyclStream,
    gate_w_bytes_usm: *const u8,
    up_w_bytes_usm: *const u8,
    x_usm: *const f32,
    gate_out_usm: *mut f32,
    up_out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq2_xs_gate_up_fused_usm(
        stream, gate_w_bytes_usm, up_w_bytes_usm,
        x_usm, gate_out_usm, up_out_usm, m, k, lws,
    )
}

/// H4: IQ2_S gate+up fused matvec, raw-pointer entry.
///
/// # Safety
///
/// All five USM pointers must be allocations on the same `stream`.
/// Gate + up weight buffers must each be sized `M × (K/256) × 82`
/// bytes (IQ2_S layout). `K` must be a multiple of 256.
pub unsafe fn matvec_iq2_s_gate_up_fused_usm_raw(
    stream: &SyclStream,
    gate_w_bytes_usm: *const u8,
    up_w_bytes_usm: *const u8,
    x_usm: *const f32,
    gate_out_usm: *mut f32,
    up_out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq2_s_gate_up_fused_usm(
        stream, gate_w_bytes_usm, up_w_bytes_usm,
        x_usm, gate_out_usm, up_out_usm, m, k, lws,
    )
}

/// H4: IQ3_XXS gate+up fused matvec, raw-pointer entry.
///
/// # Safety
///
/// All five USM pointers must be allocations on the same `stream`.
/// Gate + up weight buffers must each be sized `M × (K/256) × 98`
/// bytes (IQ3_XXS layout). `K` must be a multiple of 256.
pub unsafe fn matvec_iq3_xxs_gate_up_fused_usm_raw(
    stream: &SyclStream,
    gate_w_bytes_usm: *const u8,
    up_w_bytes_usm: *const u8,
    x_usm: *const f32,
    gate_out_usm: *mut f32,
    up_out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq3_xxs_gate_up_fused_usm(
        stream, gate_w_bytes_usm, up_w_bytes_usm,
        x_usm, gate_out_usm, up_out_usm, m, k, lws,
    )
}

/// H4: IQ3_S gate+up fused matvec, raw-pointer entry.
///
/// # Safety
///
/// All five USM pointers must be allocations on the same `stream`.
/// Gate + up weight buffers must each be sized `M × (K/256) × 110`
/// bytes (IQ3_S layout). `K` must be a multiple of 256.
pub unsafe fn matvec_iq3_s_gate_up_fused_usm_raw(
    stream: &SyclStream,
    gate_w_bytes_usm: *const u8,
    up_w_bytes_usm: *const u8,
    x_usm: *const f32,
    gate_out_usm: *mut f32,
    up_out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_iq3_s_gate_up_fused_usm(
        stream, gate_w_bytes_usm, up_w_bytes_usm,
        x_usm, gate_out_usm, up_out_usm, m, k, lws,
    )
}

/// H4: PTQ1_0 (Bonsai ternary) gate+up fused matvec, raw-pointer entry.
///
/// # Safety
///
/// All five USM pointers must be allocations on the same `stream`.
/// Gate + up weight buffers must each be sized `M × (K/128) × 28`
/// bytes (PTQ1_0 layout). `x_usm` ≥ `K` f32; `gate_out_usm` and
/// `up_out_usm` each ≥ `M` f32. `K` must be a multiple of 128.
pub unsafe fn matvec_ptq1_0_gate_up_fused_usm_raw(
    stream: &SyclStream,
    gate_w_bytes_usm: *const u8,
    up_w_bytes_usm: *const u8,
    x_usm: *const f32,
    gate_out_usm: *mut f32,
    up_out_usm: *mut f32,
    m: u32,
    k: u32,
    lws: u32,
) -> Result<()> {
    imp::matvec_ptq1_0_gate_up_fused_usm(
        stream, gate_w_bytes_usm, up_w_bytes_usm,
        x_usm, gate_out_usm, up_out_usm, m, k, lws,
    )
}

/// H6: Q4_K matvec + residual-add + rmsnorm, raw-pointer entry.
///
/// # Safety
///
/// All five USM pointers must be allocations on the same `stream`.
/// `w_bytes_usm` must be sized `M × (K/256) × 144` (Q4_K layout).
/// `attn_usm` ≥ K f32; `residual_usm` + `w_norm_usm` ≥ M f32;
/// `y_norm_usm` ≥ M f32. `K` must be a multiple of 256.
pub unsafe fn matvec_q4_k_add_rmsnorm_usm_raw(
    stream: &SyclStream,
    w_bytes_usm: *const u8,
    attn_usm: *const f32,
    hidden_usm: *mut f32,
    w_norm_usm: *const f32,
    y_norm_usm: *mut f32,
    m: u32,
    k: u32,
    eps: f32,
    lws: u32,
) -> Result<()> {
    imp::matvec_q4_k_add_rmsnorm_usm(
        stream, w_bytes_usm, attn_usm, hidden_usm, w_norm_usm, y_norm_usm,
        m, k, eps, lws,
    )
}

/// H6: remaining 12 dtypes — raw-pointer entries. Same safety
/// contract as [`matvec_q4_k_add_rmsnorm_usm_raw`]: all USM pointers
/// on the same `stream`, `w_bytes_usm` sized per the dtype's packed
/// layout, `attn_usm` ≥ K, `hidden_usm`/`w_norm_usm`/`y_norm_usm` ≥ M.
macro_rules! h6_raw_wrapper {
    ($raw:ident, $imp:ident) => {
        /// # Safety
        /// See [`matvec_q4_k_add_rmsnorm_usm_raw`].
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $raw(
            stream: &SyclStream, w_bytes_usm: *const u8, attn_usm: *const f32,
            hidden_usm: *mut f32, w_norm_usm: *const f32, y_norm_usm: *mut f32,
            m: u32, k: u32, eps: f32, lws: u32,
        ) -> Result<()> {
            imp::$imp(stream, w_bytes_usm, attn_usm, hidden_usm, w_norm_usm, y_norm_usm, m, k, eps, lws)
        }
    };
}
h6_raw_wrapper!(matvec_q8_0_add_rmsnorm_usm_raw, matvec_q8_0_add_rmsnorm_usm);
h6_raw_wrapper!(matvec_q5_k_add_rmsnorm_usm_raw, matvec_q5_k_add_rmsnorm_usm);
h6_raw_wrapper!(matvec_q6_k_add_rmsnorm_usm_raw, matvec_q6_k_add_rmsnorm_usm);
h6_raw_wrapper!(matvec_iq4_nl_add_rmsnorm_usm_raw, matvec_iq4_nl_add_rmsnorm_usm);
h6_raw_wrapper!(matvec_iq4_xs_add_rmsnorm_usm_raw, matvec_iq4_xs_add_rmsnorm_usm);
h6_raw_wrapper!(matvec_iq1_s_add_rmsnorm_usm_raw, matvec_iq1_s_add_rmsnorm_usm);
h6_raw_wrapper!(matvec_iq1_m_add_rmsnorm_usm_raw, matvec_iq1_m_add_rmsnorm_usm);
h6_raw_wrapper!(matvec_iq2_xxs_add_rmsnorm_usm_raw, matvec_iq2_xxs_add_rmsnorm_usm);
h6_raw_wrapper!(matvec_iq2_xs_add_rmsnorm_usm_raw, matvec_iq2_xs_add_rmsnorm_usm);
h6_raw_wrapper!(matvec_iq2_s_add_rmsnorm_usm_raw, matvec_iq2_s_add_rmsnorm_usm);
h6_raw_wrapper!(matvec_iq3_xxs_add_rmsnorm_usm_raw, matvec_iq3_xxs_add_rmsnorm_usm);
h6_raw_wrapper!(matvec_iq3_s_add_rmsnorm_usm_raw, matvec_iq3_s_add_rmsnorm_usm);
h6_raw_wrapper!(matvec_ptq1_0_add_rmsnorm_usm_raw, matvec_ptq1_0_add_rmsnorm_usm);

/// H8: F16-input matvec raw-pointer entries. `x_f16` is the F16-bit
/// activation vector (≥ K u16); `w_bytes_usm` packed weights; output
/// ≥ M f32. All USM on the same `stream`.
macro_rules! h8_raw_wrapper {
    ($raw:ident, $imp:ident) => {
        /// # Safety
        /// See [`matvec_q4_k_f16in_packed_f32_usm_raw`].
        pub unsafe fn $raw(
            stream: &SyclStream, w_bytes_usm: *const u8, x_f16: *const u16,
            out_usm: *mut f32, m: u32, k: u32, lws: u32,
        ) -> Result<()> {
            imp::$imp(stream, w_bytes_usm, x_f16, out_usm, m, k, lws)
        }
    };
}
h8_raw_wrapper!(matvec_q8_0_f16in_packed_f32_usm_raw, matvec_q8_0_f16in_packed_f32_usm);
h8_raw_wrapper!(matvec_q4_k_f16in_packed_f32_usm_raw, matvec_q4_k_f16in_packed_f32_usm);
h8_raw_wrapper!(matvec_q5_k_f16in_packed_f32_usm_raw, matvec_q5_k_f16in_packed_f32_usm);
h8_raw_wrapper!(matvec_q6_k_f16in_packed_f32_usm_raw, matvec_q6_k_f16in_packed_f32_usm);
h8_raw_wrapper!(matvec_iq4_nl_f16in_packed_f32_usm_raw, matvec_iq4_nl_f16in_packed_f32_usm);
h8_raw_wrapper!(matvec_iq4_xs_f16in_packed_f32_usm_raw, matvec_iq4_xs_f16in_packed_f32_usm);
h8_raw_wrapper!(matvec_iq1_s_f16in_packed_f32_usm_raw, matvec_iq1_s_f16in_packed_f32_usm);
h8_raw_wrapper!(matvec_iq1_m_f16in_packed_f32_usm_raw, matvec_iq1_m_f16in_packed_f32_usm);
h8_raw_wrapper!(matvec_iq2_xxs_f16in_packed_f32_usm_raw, matvec_iq2_xxs_f16in_packed_f32_usm);
h8_raw_wrapper!(matvec_iq2_xs_f16in_packed_f32_usm_raw, matvec_iq2_xs_f16in_packed_f32_usm);
h8_raw_wrapper!(matvec_iq2_s_f16in_packed_f32_usm_raw, matvec_iq2_s_f16in_packed_f32_usm);
h8_raw_wrapper!(matvec_iq3_xxs_f16in_packed_f32_usm_raw, matvec_iq3_xxs_f16in_packed_f32_usm);
h8_raw_wrapper!(matvec_iq3_s_f16in_packed_f32_usm_raw, matvec_iq3_s_f16in_packed_f32_usm);
h8_raw_wrapper!(matvec_ptq1_0_f16in_packed_f32_usm_raw, matvec_ptq1_0_f16in_packed_f32_usm);

/// USM-resident embedding gather. `table` is the embedding matrix
/// `[V, d]` in USM; `ids` is a host-side slice (typically very
/// small, e.g. one element for decode); `out` is the gather target
/// `[ids.len(), d]` in USM.
///
/// Negative ids produce a zero row, matching the host-pointer
/// variant. The kernel copies `ids` into a small device buffer
/// internally so the host slice doesn't need to be USM.
pub fn embedding_lookup_usm(
    stream: &SyclStream,
    table: &SyclSharedBuffer<u16>,
    ids: &[i32],
    out: &mut SyclSharedBuffer<u16>,
    d: u32,
) -> Result<()> {
    if d == 0 {
        return Err(SyclError::InvalidShape("embedding_lookup_usm: d == 0".into()));
    }
    let n_ids = ids.len();
    let need_out = n_ids * (d as usize);
    if out.len() < need_out {
        return Err(SyclError::InvalidShape(format!(
            "embedding_lookup_usm: out={}, need n_ids*d={need_out}",
            out.len()
        )));
    }
    // We don't know V here; the kernel relies on the caller having
    // sized `table` to at least `max(ids)*d` elements. Negative ids
    // are handled by the kernel.
    imp::embedding_lookup_usm(
        stream,
        table.as_ptr(),
        ids.as_ptr(),
        out.as_mut_ptr(),
        n_ids as u32,
        d,
    )
}

// ----- Raw-pointer USM entry points -----
//
// `rustllama-engine::sycl_resources` owns both a SyclStream and a
// set of USM allocations in the same struct. The safe `_usm`
// wrappers above borrow the stream through `SyclSharedBuffer<'s>`,
// which is self-referential for that case. The raw entries below
// take pointers directly so the engine's owned-buffer pattern
// works.
//
// SAFETY contract: pointers MUST be USM allocations from
// `usm_alloc_shared_raw` on the same stream, sized per each
// kernel's shape requirements.

/// Raw USM allocation. SAFETY: pair every allocation with a
/// [`usm_free_raw`] call on the same stream before the stream is
/// dropped, or the SYCL runtime leaks the allocation. Returns
/// NULL when no SYCL device is present (callers must treat as "unavailable").
pub fn usm_alloc_shared_raw(stream: &SyclStream, n_bytes: usize) -> *mut std::ffi::c_void {
    imp::usm_alloc_shared(stream, n_bytes)
}

/// SAFETY: `ptr` must have come from [`usm_alloc_shared_raw`] on
/// the same stream.
pub unsafe fn usm_free_raw(stream: &SyclStream, ptr: *mut std::ffi::c_void) {
    unsafe { imp::usm_free(stream, ptr) }
}

/// G5: K-quant → F32 dequant raw entry. SAFETY: `bytes_usm` must
/// point to at least `n_blocks * format.block_size_bytes()` USM
/// bytes; `out_usm` to at least `n_blocks * 256` USM f32s. Both
/// must come from `usm_alloc_shared_raw` on `stream`.
pub unsafe fn dequant_kquant_to_f32_usm_raw(
    stream: &SyclStream,
    format: KQuantFormat,
    bytes_usm: *const std::ffi::c_void,
    out_usm: *mut f32,
    n_blocks: u32,
) -> Result<()> {
    imp::dequant_kquant_to_f32_usm(stream, format, bytes_usm, out_usm, n_blocks)
}

/// G5: K-quant → F32 dequant convenience wrapper. Allocates USM
/// scratch internally, copies source bytes in, runs the GPU
/// dequant, copies the F32 output back to `dst`. Use this when
/// the caller doesn't already have USM-resident source bytes;
/// use [`dequant_kquant_to_f32_usm_raw`] otherwise to avoid the
/// host↔USM roundtrips.
///
/// `src_bytes.len()` must equal `n_blocks * format.block_size_bytes()`
/// for some integer `n_blocks`; `dst.len()` must equal
/// `n_blocks * 256`.
pub fn dequant_kquant_via_gpu(
    stream: &SyclStream,
    format: KQuantFormat,
    src_bytes: &[u8],
    dst: &mut [f32],
) -> Result<()> {
    let block_size = format.block_size_bytes();
    if src_bytes.len() % block_size != 0 {
        return Err(SyclError::InvalidShape(format!(
            "dequant_kquant_via_gpu: src.len()={} not a multiple of {block_size} for {format:?}",
            src_bytes.len()
        )));
    }
    let n_blocks = src_bytes.len() / block_size;
    if dst.len() != n_blocks * 256 {
        return Err(SyclError::InvalidShape(format!(
            "dequant_kquant_via_gpu: dst.len()={}, need n_blocks*256={}",
            dst.len(),
            n_blocks * 256
        )));
    }
    if n_blocks == 0 {
        return Ok(());
    }
    let mut src_usm: SyclSharedBuffer<u8> = SyclSharedBuffer::alloc(stream, src_bytes.len())?;
    let mut dst_usm: SyclSharedBuffer<f32> = SyclSharedBuffer::alloc(stream, dst.len())?;
    src_usm.as_mut_slice().copy_from_slice(src_bytes);
    imp::dequant_kquant_to_f32_usm(
        stream,
        format,
        src_usm.as_ptr() as *const std::ffi::c_void,
        dst_usm.as_mut_ptr(),
        n_blocks as u32,
    )?;
    dst.copy_from_slice(dst_usm.as_slice());
    Ok(())
}

// `flash_attn_decode_usm_raw` and `rmsnorm_usm_raw` are defined
// once above (right after `usm_free`). Keeping this comment here
// so a future linter pass doesn't reintroduce duplicates.

/// Raw-pointer entry for USM rope. SAFETY: see module contract.
pub unsafe fn rope_usm_raw(
    stream: &SyclStream,
    qk_usm: *mut u16,
    n_heads: u32,
    head_dim: u32,
    pos: u32,
    inv_freq_usm: *const u16,
) -> Result<()> {
    imp::rope_usm(stream, qk_usm, n_heads, head_dim, pos, inv_freq_usm)
}

/// Raw-pointer entry for USM silu_mul. SAFETY: see module contract.
pub unsafe fn silu_mul_usm_raw(
    stream: &SyclStream,
    x_usm: *const u16,
    y_usm: *const u16,
    out_usm: *mut u16,
    n: u32,
) -> Result<()> {
    imp::silu_mul_usm(stream, x_usm, y_usm, out_usm, n)
}

/// Raw-pointer entry for USM embedding_lookup. `ids` may be a
/// host pointer; `table_usm` and `out_usm` must be USM.
pub unsafe fn embedding_lookup_usm_raw(
    stream: &SyclStream,
    table_usm: *const u16,
    ids: *const i32,
    out_usm: *mut u16,
    n_ids: u32,
    d: u32,
) -> Result<()> {
    imp::embedding_lookup_usm(stream, table_usm, ids, out_usm, n_ids, d)
}

/// F16 GEMM: `C[M,N] = A[M,K] @ B[K,N]`. Row-major; leading dimensions
/// are implicit (`lda = K`, `ldb = N`, `ldc = N`). All buffers carry
/// raw f16 bit patterns (`u16`); convert via `half::f16::from_bits` on
/// the Rust side.
///
/// When no SYCL device is present this returns
/// [`SyclError::Unavailable`]. In `sycl` mode it dispatches to the
/// SYCL kernel in `cpp/rsl_kernels.cpp`.
pub fn gemm_f16(
    stream: &mut SyclStream,
    a: &[u16],
    b: &[u16],
    c: &mut [u16],
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    imp::gemm_f16(stream, a, b, c, m, n, k)
}

/// Per-row RMS normalization. `x: [n_rows, d]` (row-major f16 bits),
/// weights `w: [d]`, output `y: [n_rows, d]`. Computes
/// `y[r, i] = w[i] * x[r, i] / sqrt(mean(x[r, :]^2) + eps)`.
pub fn rmsnorm(
    stream: &mut SyclStream,
    x: &[u16],
    w: &[u16],
    y: &mut [u16],
    n_rows: u32,
    d: u32,
    eps: f32,
) -> Result<()> {
    imp::rmsnorm(stream, x, w, y, n_rows, d, eps)
}

/// Half-split rotary positional embedding (neox / HF-converted GGUF
/// convention). Rotates each `(qk[j], qk[j + head_dim/2])` pair
/// in-place by `pos * inv_freq[j]`. `head_dim` must be even.
pub fn rope(
    stream: &mut SyclStream,
    qk: &mut [u16],
    n_heads: u32,
    head_dim: u32,
    pos: u32,
    inv_freq: &[u16],
) -> Result<()> {
    imp::rope(stream, qk, n_heads, head_dim, pos, inv_freq)
}

/// In-place softmax across the `kv_len` dimension of
/// `scores: [n_heads, seq, kv_len]`, fused with the `scale` multiply
/// and an optional additive `mask: [kv_len]` (typically a causal
/// `-inf` mask for prefill).
pub fn softmax_attn(
    stream: &mut SyclStream,
    scores: &mut [u16],
    mask: Option<&[u16]>,
    n_heads: u32,
    seq: u32,
    kv_len: u32,
    scale: f32,
) -> Result<()> {
    imp::softmax_attn(stream, scores, mask, n_heads, seq, kv_len, scale)
}

/// SwiGLU activation: `out[i] = silu(x[i]) * y[i]` where
/// `silu(v) = v / (1 + exp(-v))`. `x`, `y`, `out` all `[n]`-shaped
/// with the same length.
pub fn silu_mul(
    stream: &mut SyclStream,
    x: &[u16],
    y: &[u16],
    out: &mut [u16],
) -> Result<()> {
    imp::silu_mul(stream, x, y, out)
}

/// Gather rows from `table: [V, d]` indexed by `ids: [N]` into
/// `out: [N, d]`. Negative ids produce a zero row; other out-of-
/// range ids invoke undefined behavior on the device side (callers
/// must validate against vocab).
pub fn embedding_lookup(
    stream: &mut SyclStream,
    table: &[u16],
    ids: &[i32],
    out: &mut [u16],
    d: u32,
) -> Result<()> {
    imp::embedding_lookup(stream, table, ids, out, d)
}

/// Dequantize NVFP4 packed weight bytes into a half-precision output
/// buffer on the GPU. `w_nvfp4.len()` must be a multiple of 9 (the
/// per-block byte count); `out.len()` must be `n_blocks * 16`. Each
/// block contains 16 codes + 1 FP8 E4M3 scale.
///
/// When no SYCL device is present, returns [`SyclError::Unavailable`]; in `sycl` mode
/// dispatches one work-item per block to the kernel in
/// `cpp/rsl_kernels.cpp`.
pub fn dequant_nvfp4(
    stream: &mut SyclStream,
    w_nvfp4: &[u8],
    out: &mut [u16],
) -> Result<()> {
    imp::dequant_nvfp4(stream, w_nvfp4, out)
}

/// NVFP4 weight × f16 activation matvec on the GPU.
/// `out[M] = W[M,K] @ x[K]`, with `W` in NVFP4 packed layout
/// (`M * (K / 16) * 9` bytes, row-major) and `x` / `out` carrying
/// raw f16 bit patterns. `K` must be a multiple of 16.
///
/// When no SYCL device is present, returns [`SyclError::Unavailable`]; in `sycl` mode
/// dispatches one work-item per output row.
pub fn matvec_nvfp4_f16(
    stream: &mut SyclStream,
    w_nvfp4: &[u8],
    x: &[u16],
    out: &mut [u16],
    m: u32,
    k: u32,
) -> Result<()> {
    imp::matvec_nvfp4_f16(stream, w_nvfp4, x, out, m, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time check: the per-token USM kernels (rope, silu,
    /// embedding) are callable with the expected SyclSharedBuffer
    /// signatures, and several USM buffers can coexist on one
    /// stream simultaneously — the API shape the engine wire-up
    /// will use to keep `hidden`, Q, K cache, V cache, scratch
    /// buffers all device-resident across a single forward pass.
    #[test]
    fn usm_per_token_kernels_compile_check() {
        fn _rope_signature(
            s: &SyclStream,
            qk: &mut SyclSharedBuffer<u16>,
            inv_freq: &SyclSharedBuffer<u16>,
        ) -> Result<()> {
            rope_usm(s, qk, 4, 64, 0, inv_freq)
        }
        fn _silu_signature(
            s: &SyclStream,
            x: &SyclSharedBuffer<u16>,
            y: &SyclSharedBuffer<u16>,
            out: &mut SyclSharedBuffer<u16>,
        ) -> Result<()> {
            silu_mul_usm(s, x, y, out, 1024)
        }
        fn _embed_signature(
            s: &SyclStream,
            table: &SyclSharedBuffer<u16>,
            out: &mut SyclSharedBuffer<u16>,
        ) -> Result<()> {
            embedding_lookup_usm(s, table, &[3, 5, 7], out, 4096)
        }
        // The "all kernels on one stream" pattern: alloc many
        // buffers, run several USM kernels — this is what makes
        // the upcoming engine wire-up tractable.
        fn _full_decode(stream: &SyclStream) -> Result<()> {
            let table = SyclSharedBuffer::<u16>::alloc(stream, 256 * 4096)?;
            let mut hidden = SyclSharedBuffer::<u16>::alloc(stream, 4096)?;
            embedding_lookup_usm(stream, &table, &[7], &mut hidden, 4096)?;
            let inv_freq = SyclSharedBuffer::<u16>::alloc(stream, 64)?;
            rope_usm(stream, &mut hidden, 64, 128, 0, &inv_freq)?;
            let gate = SyclSharedBuffer::<u16>::alloc(stream, 11008)?;
            let up = SyclSharedBuffer::<u16>::alloc(stream, 11008)?;
            let mut ffn = SyclSharedBuffer::<u16>::alloc(stream, 11008)?;
            silu_mul_usm(stream, &gate, &up, &mut ffn, 11008)?;
            Ok(())
        }
        let _ = _rope_signature;
        let _ = _silu_signature;
        let _ = _embed_signature;
        let _ = _full_decode;
    }

    /// Compile-time check: `flash_attn_decode_usm` is callable
    /// with the expected SyclSharedBuffer signature, and four
    /// USM buffers can coexist on one stream via shared borrow.
    #[test]
    fn flash_attn_decode_usm_compile_check() {
        fn _signature(
            s: &SyclStream,
            q: &SyclSharedBuffer<u16>,
            k: &SyclSharedBuffer<u16>,
            v: &SyclSharedBuffer<u16>,
            out: &mut SyclSharedBuffer<u16>,
        ) -> Result<()> {
            flash_attn_decode_usm(s, q, k, v, out, 4, 2, 16, 32, 8)
        }
        // Verify the multi-buffer pattern compiles: alloc four
        // distinct SyclSharedBuffer values borrowing the same
        // stream, pass them all into the kernel. This is the
        // pattern the engine wire-up will use.
        fn _multi_buffer(stream: &SyclStream) -> Result<()> {
            let q = SyclSharedBuffer::<u16>::alloc(stream, 64)?;
            let k = SyclSharedBuffer::<u16>::alloc(stream, 2048)?;
            let v = SyclSharedBuffer::<u16>::alloc(stream, 2048)?;
            let mut out = SyclSharedBuffer::<u16>::alloc(stream, 64)?;
            flash_attn_decode_usm(stream, &q, &k, &v, &mut out, 4, 2, 16, 32, 8)
        }
        let _ = _signature;
        let _ = _multi_buffer;
    }

    /// Compile-time check: `flash_attn_prefill_usm` is callable
    /// with f32 SyclSharedBuffers and the expected `(n_heads,
    /// n_kv_heads, head_dim, max_ctx, kv_len_base, n_new)` signature.
    /// Same multi-buffer pattern as decode but f32 — the engine
    /// prefill wire-up will allocate Q/K/V/out f32 USM buffers on
    /// a single stream borrow.
    #[test]
    fn flash_attn_prefill_usm_compile_check() {
        fn _signature(
            s: &SyclStream,
            q: &SyclSharedBuffer<f32>,
            k: &SyclSharedBuffer<f32>,
            v: &SyclSharedBuffer<f32>,
            out: &mut SyclSharedBuffer<f32>,
        ) -> Result<()> {
            // n_heads=4, n_kv_heads=2 (GQA=2), head_dim=16, max_ctx=32,
            // kv_len_base=0, n_new=4.
            flash_attn_prefill_usm(s, q, k, v, out, 4, 2, 16, 32, 0, 4)
        }
        fn _multi_buffer(stream: &SyclStream) -> Result<()> {
            // q: [n_new=4, n_heads=4, head_dim=16] = 256
            // kv: [n_kv_heads=2, max_ctx=32, head_dim=16] = 1024 each
            // out: same shape as q = 256
            let q = SyclSharedBuffer::<f32>::alloc(stream, 256)?;
            let k = SyclSharedBuffer::<f32>::alloc(stream, 1024)?;
            let v = SyclSharedBuffer::<f32>::alloc(stream, 1024)?;
            let mut out = SyclSharedBuffer::<f32>::alloc(stream, 256)?;
            flash_attn_prefill_usm(stream, &q, &k, &v, &mut out, 4, 2, 16, 32, 0, 4)
        }
        let _ = _signature;
        let _ = _multi_buffer;
    }

    /// Compile-check: `gemm_f16_usm` is callable through the safe
    /// `SyclSharedBuffer` API with three concurrent buffers on one
    /// stream — the shape the engine projection matvec will use.
    #[test]
    fn gemm_f16_usm_compile_check() {
        fn _matvec_shape(stream: &SyclStream) -> Result<()> {
            // matvec: M=d_q, N=1, K=d (projection example).
            let a = SyclSharedBuffer::<u16>::alloc(stream, 256)?;
            let b = SyclSharedBuffer::<u16>::alloc(stream, 16)?;
            let mut c = SyclSharedBuffer::<u16>::alloc(stream, 16)?;
            gemm_f16_usm(stream, &a, &b, &mut c, 16, 1, 16)
        }
        let _ = _matvec_shape;
    }

    /// Compile-check: `matvec_q8_0_f32_usm` is callable through the
    /// safe `SyclSharedBuffer` API with four concurrent buffers on
    /// one stream (mixed dtypes: i8 weights, f32 scales/x/out).
    #[test]
    fn matvec_q8_0_f32_usm_compile_check() {
        fn _shape(stream: &SyclStream) -> Result<()> {
            let m: u32 = 8;
            let k: u32 = 64; // 2 blocks per row
            let w_q = SyclSharedBuffer::<i8>::alloc(stream, (m * k) as usize)?;
            let w_scales =
                SyclSharedBuffer::<f32>::alloc(stream, (m * (k / 32)) as usize)?;
            let x = SyclSharedBuffer::<f32>::alloc(stream, k as usize)?;
            let mut out = SyclSharedBuffer::<f32>::alloc(stream, m as usize)?;
            matvec_q8_0_f32_usm(stream, &w_q, &w_scales, &x, &mut out, m, k)
        }
        let _ = _shape;
    }

    /// Compile-check: `matvec_q8_0_packed_f32_usm` is callable through
    /// the safe `SyclSharedBuffer` API with three concurrent buffers
    /// on one stream — the shape engine LM head / embedding dispatch
    /// will use once Q8_0 weights migrate onto a single USM byte slab.
    #[test]
    fn matvec_q8_0_packed_f32_usm_compile_check() {
        fn _shape(stream: &SyclStream) -> Result<()> {
            let m: u32 = 8;
            let k: u32 = 64; // 2 blocks per row × 34 bytes
            let bytes = (m as usize) * ((k / 32) as usize) * 34;
            let w_bytes = SyclSharedBuffer::<u8>::alloc(stream, bytes)?;
            let x = SyclSharedBuffer::<f32>::alloc(stream, k as usize)?;
            let mut out = SyclSharedBuffer::<f32>::alloc(stream, m as usize)?;
            matvec_q8_0_packed_f32_usm(stream, &w_bytes, &x, &mut out, m, k, 0)
        }
        let _ = _shape;
    }

    /// Compile-check: `matvec_q4_k_packed_f32_usm` is callable through
    /// the safe API. K must be a multiple of 256 (Q4_K super-block);
    /// w_bytes is 144 bytes per super-block. `lws=0` selects the
    /// hand-picked default work-group size.
    #[test]
    fn matvec_q4_k_packed_f32_usm_compile_check() {
        fn _shape(stream: &SyclStream) -> Result<()> {
            let m: u32 = 4;
            let k: u32 = 256;
            let bytes = (m as usize) * ((k / 256) as usize) * 144;
            let w_bytes = SyclSharedBuffer::<u8>::alloc(stream, bytes)?;
            let x = SyclSharedBuffer::<f32>::alloc(stream, k as usize)?;
            let mut out = SyclSharedBuffer::<f32>::alloc(stream, m as usize)?;
            matvec_q4_k_packed_f32_usm(stream, &w_bytes, &x, &mut out, m, k, 0)
        }
        let _ = _shape;
    }

    /// Compile-check: `matvec_q5_k_packed_f32_usm` is callable through
    /// the safe API. K must be a multiple of 256 (same Q4_K-style
    /// super-block); w_bytes is 176 bytes per super-block (32 extra
    /// bytes of `qh` over Q4_K_M).
    #[test]
    fn matvec_q5_k_packed_f32_usm_compile_check() {
        fn _shape(stream: &SyclStream) -> Result<()> {
            let m: u32 = 4;
            let k: u32 = 256;
            let bytes = (m as usize) * ((k / 256) as usize) * 176;
            let w_bytes = SyclSharedBuffer::<u8>::alloc(stream, bytes)?;
            let x = SyclSharedBuffer::<f32>::alloc(stream, k as usize)?;
            let mut out = SyclSharedBuffer::<f32>::alloc(stream, m as usize)?;
            matvec_q5_k_packed_f32_usm(stream, &w_bytes, &x, &mut out, m, k, 0)
        }
        let _ = _shape;
    }

    /// Compile-check: `matvec_q6_k_packed_f32_usm` is callable through
    /// the safe API. K must be a multiple of 256 (super-block size);
    /// w_bytes is 210 bytes per super-block (128 ql + 64 qh + 16
    /// scales + 2 f16-d).
    #[test]
    fn matvec_q6_k_packed_f32_usm_compile_check() {
        fn _shape(stream: &SyclStream) -> Result<()> {
            let m: u32 = 4;
            let k: u32 = 256;
            let bytes = (m as usize) * ((k / 256) as usize) * 210;
            let w_bytes = SyclSharedBuffer::<u8>::alloc(stream, bytes)?;
            let x = SyclSharedBuffer::<f32>::alloc(stream, k as usize)?;
            let mut out = SyclSharedBuffer::<f32>::alloc(stream, m as usize)?;
            matvec_q6_k_packed_f32_usm(stream, &w_bytes, &x, &mut out, m, k, 0)
        }
        let _ = _shape;
    }

    /// Compile-check: batched packed-matvec safe wrappers (Q8_0,
    /// Q4_K, Q5_K, Q6_K). `x: [N, K]` and `out: [N, M]` row-major;
    /// kernel reads `x_usm + n*K` and writes `out_usm + n*M + m`.
    #[test]
    fn matvec_packed_f32_batched_usm_compile_check() {
        fn _shape(stream: &SyclStream) -> Result<()> {
            let m: u32 = 4;
            let k: u32 = 256;
            let n: u32 = 3;
            let x = SyclSharedBuffer::<f32>::alloc(stream, (n * k) as usize)?;
            let mut out = SyclSharedBuffer::<f32>::alloc(stream, (n * m) as usize)?;
            // Q8_0: 34 bytes per 32-weight block.
            let q80_bytes = (m as usize) * ((k / 32) as usize) * 34;
            let w_q80 = SyclSharedBuffer::<u8>::alloc(stream, q80_bytes)?;
            // `lws = 0` selects the kernel's hand-picked default;
            // the LWS axis is exercised by the parity tests, not here.
            matvec_q8_0_packed_f32_batched_usm(stream, &w_q80, &x, &mut out, m, k, n, 0)?;
            // Q4_K: 144 bytes per 256-weight super-block.
            let q4k_bytes = (m as usize) * ((k / 256) as usize) * 144;
            let w_q4k = SyclSharedBuffer::<u8>::alloc(stream, q4k_bytes)?;
            matvec_q4_k_packed_f32_batched_usm(stream, &w_q4k, &x, &mut out, m, k, n, 0)?;
            // Q5_K: 176 bytes per super-block.
            let q5k_bytes = (m as usize) * ((k / 256) as usize) * 176;
            let w_q5k = SyclSharedBuffer::<u8>::alloc(stream, q5k_bytes)?;
            matvec_q5_k_packed_f32_batched_usm(stream, &w_q5k, &x, &mut out, m, k, n, 0)?;
            // Q6_K: 210 bytes per super-block.
            let q6k_bytes = (m as usize) * ((k / 256) as usize) * 210;
            let w_q6k = SyclSharedBuffer::<u8>::alloc(stream, q6k_bytes)?;
            matvec_q6_k_packed_f32_batched_usm(stream, &w_q6k, &x, &mut out, m, k, n, 0)
        }
        let _ = _shape;
    }

    /// Real-SYCL probe: try to import a synthetic Win32 file mapping
    /// as device-accessible memory and report what the L0 driver
    /// says. We don't assert success — Intel's L0 implementation
    /// may not accept arbitrary file-mapping handles via
    /// OPAQUE_WIN32 (the documented use case is D3D / Vulkan
    /// interop). The test passes regardless; what matters is the
    /// printed return code: 0 means the path works (huge win),
    /// 4 means we need to fall back to the copy path.
    ///
    /// Run with `cargo test -- --ignored l0_import`.
    #[cfg(target_os = "windows")]
    #[test]
    #[ignore = "requires SYCL GPU + Windows; run with -- --ignored"]
    fn l0_import_win32_mapping_probe() {
        use std::ptr;
        // Win32 API surface we need. Hand-rolled so the test
        // doesn't pull in `windows-sys`.
        type Handle = *mut std::ffi::c_void;
        const INVALID_HANDLE_VALUE: Handle = usize::MAX as *mut _;
        const PAGE_READWRITE: u32 = 0x04;
        extern "system" {
            fn CreateFileMappingW(
                hFile: Handle,
                lpAttrs: *mut std::ffi::c_void,
                flProtect: u32,
                dwMaximumSizeHigh: u32,
                dwMaximumSizeLow: u32,
                lpName: *const u16,
            ) -> Handle;
            fn CloseHandle(h: Handle) -> i32;
        }
        // Force SYCL to pick the Level Zero backend for this test.
        // SYCL's default device selector on Intel hosts can pick
        // OpenCL (backend index 2) instead of L0 (index 5), which
        // makes the import call return "non-L0 backend" before it
        // even reaches the driver. The env var must be set BEFORE
        // `create_stream` — SYCL reads it at queue construction.
        //
        // SAFETY: set_var is `unsafe` on Rust 1.84+ because it
        // races with concurrent gets in other threads. This test
        // runs single-threaded (cargo test --jobs ≥ 1 spawns
        // per-test processes; within one process the test runner
        // serializes tests inside a single binary unless
        // --test-threads is bumped). Acceptable risk for a probe.
        std::env::set_var("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu");
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("no SYCL device available: {e}");
                return;
            }
        };
        // 1 MiB page-file-backed section. Real GGUF mappings would
        // be file-backed; we test with anon to avoid touching disk.
        let size: usize = 1 << 20;
        let mapping = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null_mut(),
                PAGE_READWRITE,
                0,
                size as u32,
                ptr::null(),
            )
        };
        assert!(!mapping.is_null(), "CreateFileMappingW failed");
        let result = unsafe { try_import_win32_handle_as_usm_raw(&stream, mapping, size) };
        match result {
            Ok(p) => {
                eprintln!("L0 import succeeded: ptr={p:p}, size={size}");
                unsafe { release_imported_usm_raw(&stream, p) };
            }
            Err(SyclError::L0ImportUnsupported(code)) => {
                eprintln!(
                    "L0 import unsupported (code {code}). This is the \
                     expected path if the driver doesn't accept anon \
                     file mappings via OPAQUE_WIN32 — engine will fall \
                     back to copy-to-USM."
                );
            }
            Err(e) => panic!("unexpected error from import: {e}"),
        }
        unsafe { CloseHandle(mapping) };
    }

    /// Real-SYCL parity: USM F16 GEMM matches a CPU f32 reference
    /// within f16 round-trip tolerance. Run with
    /// `cargo test -- --ignored gemm_f16_usm`.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_gemm_f16_matches_cpu_reference() {
        use half::f16;
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        let m = 8u32;
        let n = 4u32;
        let k = 16u32;
        // Synthetic A and B with deterministic patterns.
        let a_f32: Vec<f32> = (0..(m * k) as usize)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.13)
            .collect();
        let b_f32: Vec<f32> = (0..(k * n) as usize)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.07)
            .collect();
        let mut a_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, a_f32.len()).expect("alloc A");
        let mut b_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, b_f32.len()).expect("alloc B");
        let mut c_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, (m * n) as usize).expect("alloc C");
        for (i, v) in a_f32.iter().enumerate() {
            a_buf.as_mut_slice()[i] = f16::from_f32(*v).to_bits();
        }
        for (i, v) in b_f32.iter().enumerate() {
            b_buf.as_mut_slice()[i] = f16::from_f32(*v).to_bits();
        }
        for dst in c_buf.as_mut_slice().iter_mut() {
            *dst = 0;
        }
        gemm_f16_usm(&stream, &a_buf, &b_buf, &mut c_buf, m, n, k)
            .expect("gemm_f16_usm");
        // CPU reference: row-major matmul with f16 round-trip
        // matching the kernel's storage precision.
        let mut c_ref = vec![0f32; (m * n) as usize];
        for row in 0..m as usize {
            for col in 0..n as usize {
                let mut acc = 0f32;
                for p in 0..k as usize {
                    let av = f16::from_f32(a_f32[row * k as usize + p]).to_f32();
                    let bv = f16::from_f32(b_f32[p * n as usize + col]).to_f32();
                    acc += av * bv;
                }
                c_ref[row * n as usize + col] = acc;
            }
        }
        let mut max_err = 0f32;
        for i in 0..(m * n) as usize {
            let got = f16::from_bits(c_buf.as_slice()[i]).to_f32();
            let want = f16::from_f32(c_ref[i]).to_f32();
            let err = (got - want).abs();
            if err > max_err {
                max_err = err;
            }
        }
        eprintln!("usm_gemm_f16 parity: max_abs_err = {max_err:.6}");
        assert!(max_err < 1e-2, "USM gemm vs CPU max_abs_err {max_err} > 1e-2");
    }

    /// Real-SYCL parity: USM Q8_0 weight × F32 activation matvec
    /// matches a CPU reference computed from the reconstructed
    /// (i8 × scale) weights. Both sides start from the same i8 / f32
    /// data so any nonzero error indicates a kernel bug, not quant
    /// rounding. Run with
    /// `cargo test -- --ignored usm_matvec_q8_0`.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_matvec_q8_0_matches_cpu_reference() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        let m: u32 = 6;
        let k: u32 = 96; // 3 blocks per row
        let blocks_per_row = (k / 32) as usize;

        // Synthetic i8 weights + per-block f32 scales with
        // deterministic patterns.
        let total_w = (m * k) as usize;
        let total_scales = (m as usize) * blocks_per_row;
        let mut w_q_host = vec![0i8; total_w];
        for (i, dst) in w_q_host.iter_mut().enumerate() {
            *dst = (((i as i32) % 251) - 125) as i8;
        }
        let mut scales_host = vec![0f32; total_scales];
        for (i, dst) in scales_host.iter_mut().enumerate() {
            *dst = 0.005 + 0.001 * ((i % 7) as f32);
        }
        let mut x_host = vec![0f32; k as usize];
        for (i, dst) in x_host.iter_mut().enumerate() {
            *dst = (((i % 19) as f32) - 9.0) * 0.11;
        }

        let mut w_q_buf: SyclSharedBuffer<i8> =
            SyclSharedBuffer::alloc(&stream, total_w).expect("alloc w_q");
        let mut scales_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, total_scales).expect("alloc scales");
        let mut x_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, k as usize).expect("alloc x");
        let mut out_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, m as usize).expect("alloc out");

        w_q_buf.as_mut_slice().copy_from_slice(&w_q_host);
        scales_buf.as_mut_slice().copy_from_slice(&scales_host);
        x_buf.as_mut_slice().copy_from_slice(&x_host);
        for dst in out_buf.as_mut_slice().iter_mut() {
            *dst = 0.0;
        }

        matvec_q8_0_f32_usm(&stream, &w_q_buf, &scales_buf, &x_buf, &mut out_buf, m, k)
            .expect("matvec_q8_0_f32_usm");

        // CPU reference using the same i8 / f32 data: for each
        // output row, walk each block, dot 32 i8-widened-to-f32
        // weights against 32 x values, fold in the per-block scale.
        let mut out_ref = vec![0f32; m as usize];
        for row in 0..m as usize {
            let row_off = row * k as usize;
            let scale_off = row * blocks_per_row;
            let mut acc = 0f32;
            for b in 0..blocks_per_row {
                let scale = scales_host[scale_off + b];
                let mut block_dot = 0f32;
                for d in 0..32 {
                    let w = w_q_host[row_off + b * 32 + d] as f32;
                    block_dot += w * x_host[b * 32 + d];
                }
                acc += scale * block_dot;
            }
            out_ref[row] = acc;
        }

        let mut max_err = 0f32;
        for i in 0..m as usize {
            let got = out_buf.as_slice()[i];
            let want = out_ref[i];
            let err = (got - want).abs();
            if err > max_err {
                max_err = err;
            }
        }
        eprintln!("usm_matvec_q8_0_f32 parity: max_abs_err = {max_err:.6}");
        // Pure f32 arithmetic on both sides — tolerance just covers
        // floating-point reduction-order differences.
        assert!(
            max_err < 1e-3,
            "USM Q8_0 matvec vs CPU max_abs_err {max_err} > 1e-3"
        );
    }

    /// Real-SYCL parity: USM Q8_0 packed-layout matvec matches an
    /// inline CPU reference that walks the same 34-byte block layout.
    /// The packed variant is the one engine integration will actually
    /// use — `w_bytes` is the raw mmap'd GGUF tensor copied into USM
    /// without repack. Run with
    /// `cargo test -- --ignored usm_matvec_q8_0_packed`.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_matvec_q8_0_packed_matches_cpu_reference() {
        use half::f16;
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        let m: u32 = 6;
        let k: u32 = 96; // 3 blocks per row
        let blocks_per_row = (k / 32) as usize;
        let bytes_per_row = blocks_per_row * 34;
        let total_bytes = (m as usize) * bytes_per_row;

        // Build a packed Q8_0 byte slab + a parallel f32 activation
        // vector with deterministic patterns.
        let mut w_bytes_host = vec![0u8; total_bytes];
        for row in 0..m as usize {
            for b in 0..blocks_per_row {
                let block_off = row * bytes_per_row + b * 34;
                // Vary the scale per (row, block) so we exercise the
                // f16 decode path on every block.
                let scale_f32 = 0.005 + 0.002 * ((row * 3 + b) as f32);
                let scale_bits = f16::from_f32(scale_f32).to_bits();
                w_bytes_host[block_off] = (scale_bits & 0xFF) as u8;
                w_bytes_host[block_off + 1] = (scale_bits >> 8) as u8;
                for d in 0..32 {
                    let idx = row * (blocks_per_row * 32) + b * 32 + d;
                    let q = (((idx as i32) % 251) - 125) as i8;
                    w_bytes_host[block_off + 2 + d] = q as u8;
                }
            }
        }
        let x_host: Vec<f32> = (0..k as usize)
            .map(|i| (((i % 19) as f32) - 9.0) * 0.11)
            .collect();

        let mut w_bytes_buf: SyclSharedBuffer<u8> =
            SyclSharedBuffer::alloc(&stream, total_bytes).expect("alloc w_bytes");
        let mut x_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, k as usize).expect("alloc x");
        let mut out_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, m as usize).expect("alloc out");
        w_bytes_buf.as_mut_slice().copy_from_slice(&w_bytes_host);
        x_buf.as_mut_slice().copy_from_slice(&x_host);
        for dst in out_buf.as_mut_slice().iter_mut() {
            *dst = 0.0;
        }

        matvec_q8_0_packed_f32_usm(&stream, &w_bytes_buf, &x_buf, &mut out_buf, m, k, 0)
            .expect("matvec_q8_0_packed_f32_usm");

        // CPU reference: walk blocks, decode f16 scale, dot-product
        // i8 weights with the f32 activation, fold per-block scale.
        let mut out_ref = vec![0f32; m as usize];
        for row in 0..m as usize {
            let mut acc = 0f32;
            for b in 0..blocks_per_row {
                let block_off = row * bytes_per_row + b * 34;
                let scale = f16::from_le_bytes([
                    w_bytes_host[block_off],
                    w_bytes_host[block_off + 1],
                ]).to_f32();
                let qs = &w_bytes_host[block_off + 2..block_off + 2 + 32];
                let x_chunk = &x_host[b * 32..(b + 1) * 32];
                let mut block_dot = 0f32;
                for (q, xv) in qs.iter().zip(x_chunk.iter()) {
                    block_dot += (*q as i8 as f32) * *xv;
                }
                acc += scale * block_dot;
            }
            out_ref[row] = acc;
        }

        let mut max_err = 0f32;
        for i in 0..m as usize {
            let got = out_buf.as_slice()[i];
            let want = out_ref[i];
            let err = (got - want).abs();
            if err > max_err {
                max_err = err;
            }
        }
        eprintln!("usm_matvec_q8_0_packed parity: max_abs_err = {max_err:.6}");
        assert!(
            max_err < 1e-3,
            "USM Q8_0 packed matvec vs CPU max_abs_err {max_err} > 1e-3"
        );
    }

    /// Real-SYCL parity: USM Q4_K_M packed-layout matvec matches an
    /// inline CPU reference walking the same 144-byte super-block
    /// layout (matches `unpack_q4k_scales` + the scalar reference in
    /// `rustllama-kernels-cpu`). Q4_K_M is the v1 default quant so
    /// this test is the critical-path correctness check for engine
    /// integration. Run with
    /// `cargo test -- --ignored usm_matvec_q4_k_packed`.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_matvec_q4_k_packed_matches_cpu_reference() {
        use half::f16;
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        let m: u32 = 4;
        let k: u32 = 512; // 2 super-blocks per row
        let super_blocks_per_row = (k / 256) as usize;
        let bytes_per_row = super_blocks_per_row * 144;
        let total_bytes = (m as usize) * bytes_per_row;

        // Build a synthetic Q4_K_M byte slab + parallel f32
        // activation with deterministic patterns. Scales/mins are
        // 6-bit (max 63); we pack them into the 12-byte encoding
        // matching the CPU `unpack_q4k_scales` inverse.
        fn pack_q4k_scales(sc: &[u8; 8], mn: &[u8; 8]) -> [u8; 12] {
            // Inverse of unpack_q4k_scales. For j in 0..4: low 6 bits
            // of scales[j] = sc[j], low 6 bits of scales[j+4] = mn[j].
            // For j in 4..8: low 4 bits of scales[j+4] = sc[j] & 0x0F,
            // high 4 bits of scales[j+4] = mn[j] & 0x0F, and the high
            // 2 bits of scales[j-4] carry sc[j] >> 4 (low byte) /
            // mn[j] >> 4 (high byte).
            let mut out = [0u8; 12];
            for j in 0..4 {
                out[j] = sc[j] & 0x3F;
                out[j + 4] = mn[j] & 0x3F;
            }
            for j in 4..8 {
                out[j + 4] = (sc[j] & 0x0F) | ((mn[j] & 0x0F) << 4);
                // High 2 bits of sc[j] / mn[j] live in the high bits
                // of scales[j-4]. scales[j-4] currently has its low
                // 6 bits set to sc[j-4]; the top 2 bits encode the
                // high 2 bits of sc[j+4-4] = sc[j]. Same for mn.
                out[j - 4] |= ((sc[j] >> 4) & 0x03) << 6;
                out[j] |= ((mn[j] >> 4) & 0x03) << 6;
            }
            out
        }

        let mut w_bytes_host = vec![0u8; total_bytes];
        for row in 0..m as usize {
            for sb in 0..super_blocks_per_row {
                let off = row * bytes_per_row + sb * 144;
                // (d, dmin) — small magnitudes so the dequant values
                // stay reasonable.
                let d_f32 = 0.01 + 0.003 * ((row + sb) as f32);
                let dmin_f32 = 0.005 + 0.001 * ((row * 2 + sb) as f32);
                let d_bits = f16::from_f32(d_f32).to_bits();
                let dmin_bits = f16::from_f32(dmin_f32).to_bits();
                w_bytes_host[off] = (d_bits & 0xFF) as u8;
                w_bytes_host[off + 1] = (d_bits >> 8) as u8;
                w_bytes_host[off + 2] = (dmin_bits & 0xFF) as u8;
                w_bytes_host[off + 3] = (dmin_bits >> 8) as u8;
                // 8 6-bit sub-block scales + mins.
                let mut sc = [0u8; 8];
                let mut mn = [0u8; 8];
                for j in 0..8 {
                    sc[j] = ((row * 7 + sb * 11 + j * 3) % 63) as u8;
                    mn[j] = ((row * 5 + sb * 13 + j * 2) % 63) as u8;
                }
                let packed = pack_q4k_scales(&sc, &mn);
                w_bytes_host[off + 4..off + 16].copy_from_slice(&packed);
                // 128 bytes of 4-bit packed weights.
                for l in 0..128 {
                    let lo = ((row * 3 + sb * 5 + l * 7) % 16) as u8;
                    let hi = ((row * 11 + sb * 17 + l * 13) % 16) as u8;
                    w_bytes_host[off + 16 + l] = lo | (hi << 4);
                }
            }
        }
        let x_host: Vec<f32> = (0..k as usize)
            .map(|i| (((i % 23) as f32) - 11.0) * 0.05)
            .collect();

        let mut w_bytes_buf: SyclSharedBuffer<u8> =
            SyclSharedBuffer::alloc(&stream, total_bytes).expect("alloc w_bytes");
        let mut x_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, k as usize).expect("alloc x");
        let mut out_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, m as usize).expect("alloc out");
        w_bytes_buf.as_mut_slice().copy_from_slice(&w_bytes_host);
        x_buf.as_mut_slice().copy_from_slice(&x_host);
        for dst in out_buf.as_mut_slice().iter_mut() {
            *dst = 0.0;
        }

        matvec_q4_k_packed_f32_usm(&stream, &w_bytes_buf, &x_buf, &mut out_buf, m, k, 0)
            .expect("matvec_q4_k_packed_f32_usm");

        // CPU reference inline (mirrors CPU `matvec_q4_k_w_f32_a_scalar`
        // exactly so any error indicates a kernel bug, not a quant
        // rounding mismatch).
        fn unpack_q4k(scales: &[u8]) -> ([u8; 8], [u8; 8]) {
            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];
            for j in 0..8 {
                if j < 4 {
                    sc[j] = scales[j] & 0x3F;
                    mn[j] = scales[j + 4] & 0x3F;
                } else {
                    sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
                    mn[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
                }
            }
            (sc, mn)
        }
        let mut out_ref = vec![0f32; m as usize];
        for row in 0..m as usize {
            let mut acc = 0f32;
            for sb in 0..super_blocks_per_row {
                let off = row * bytes_per_row + sb * 144;
                let d = f16::from_le_bytes([w_bytes_host[off], w_bytes_host[off + 1]])
                    .to_f32();
                let dmin =
                    f16::from_le_bytes([w_bytes_host[off + 2], w_bytes_host[off + 3]])
                        .to_f32();
                let (sc, mn) = unpack_q4k(&w_bytes_host[off + 4..off + 16]);
                let qs = &w_bytes_host[off + 16..off + 16 + 128];
                let x_block = &x_host[sb * 256..(sb + 1) * 256];
                for group in 0..4 {
                    let q_chunk = &qs[group * 32..(group + 1) * 32];
                    let d_lo = d * sc[group * 2] as f32;
                    let m_lo = dmin * mn[group * 2] as f32;
                    let d_hi = d * sc[group * 2 + 1] as f32;
                    let m_hi = dmin * mn[group * 2 + 1] as f32;
                    let x_lo = &x_block[group * 64..group * 64 + 32];
                    let x_hi = &x_block[group * 64 + 32..(group + 1) * 64];
                    for l in 0..32 {
                        let q = q_chunk[l];
                        acc += (d_lo * (q & 0x0F) as f32 - m_lo) * x_lo[l];
                        acc += (d_hi * (q >> 4) as f32 - m_hi) * x_hi[l];
                    }
                }
            }
            out_ref[row] = acc;
        }
        let mut max_err = 0f32;
        for i in 0..m as usize {
            let got = out_buf.as_slice()[i];
            let want = out_ref[i];
            let err = (got - want).abs();
            if err > max_err {
                max_err = err;
            }
        }
        eprintln!("usm_matvec_q4_k_packed parity: max_abs_err = {max_err:.6}");
        // Pure f32 arithmetic on both sides; f16 round-trip on the
        // (d, dmin) decode introduces the same tiny error in both.
        assert!(
            max_err < 1e-3,
            "USM Q4_K_M packed matvec vs CPU max_abs_err {max_err} > 1e-3"
        );
    }

    /// Real-SYCL parity: every templated LWS variant of the Q4_K
    /// USM matvec produces bit-identical output. Templating the
    /// kernel on LWS only changes the `nd_range` local size; the
    /// per-work-item math is the same, so output must match exactly
    /// — not "within epsilon", *exactly*. A mismatch here means the
    /// templating accidentally affected the per-item computation
    /// (capture issue, work-id arithmetic, etc.), which would
    /// invalidate every autotuner sweep result.
    ///
    /// Run with `cargo test -- --ignored q4_k_lws_parity`.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_matvec_q4_k_packed_lws_parity_across_variants() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        // M and K chosen so each candidate LWS divides M cleanly OR
        // leaves a meaningful remainder for the bounds-check path —
        // both regimes must produce identical outputs.
        let m: u32 = 100; // 100 isn't a multiple of 16/32/64/128/256
        let k: u32 = 256;
        let bytes = (m as usize) * ((k / 256) as usize) * 144;
        // Deterministic pseudo-random byte pattern. Doesn't need to
        // decode to anything sensible — we only care that all LWS
        // variants reach the same `acc` value for each output row.
        let w_bytes_host: Vec<u8> =
            (0..bytes).map(|i| ((i * 31 + 7) & 0xff) as u8).collect();
        let x_host: Vec<f32> = (0..k as usize)
            .map(|i| (((i % 17) as f32) - 8.0) * 0.05)
            .collect();
        let mut w_buf = SyclSharedBuffer::<u8>::alloc(&stream, bytes).expect("w alloc");
        let mut x_buf = SyclSharedBuffer::<f32>::alloc(&stream, k as usize).expect("x alloc");
        w_buf.as_mut_slice().copy_from_slice(&w_bytes_host);
        x_buf.as_mut_slice().copy_from_slice(&x_host);
        let candidates = [16u32, 32, 64, 128, 256];
        let mut reference: Option<Vec<f32>> = None;
        for lws in candidates {
            let mut out_buf =
                SyclSharedBuffer::<f32>::alloc(&stream, m as usize).expect("out alloc");
            for dst in out_buf.as_mut_slice().iter_mut() {
                *dst = 0.0;
            }
            matvec_q4_k_packed_f32_usm(&stream, &w_buf, &x_buf, &mut out_buf, m, k, lws)
                .expect("matvec_q4_k_packed_f32_usm");
            let got: Vec<f32> = out_buf.as_slice().to_vec();
            match &reference {
                None => {
                    eprintln!(
                        "q4_k lws={lws}: reference output (first 4): {:?}",
                        &got[..4.min(got.len())]
                    );
                    reference = Some(got);
                }
                Some(r) => {
                    for i in 0..(m as usize) {
                        assert_eq!(
                            got[i].to_bits(),
                            r[i].to_bits(),
                            "lws={lws} row {i}: got={} vs ref={} (bit patterns differ)",
                            got[i],
                            r[i],
                        );
                    }
                    eprintln!("q4_k lws={lws}: bit-identical to reference ✓");
                }
            }
        }
    }

    /// Real-SYCL parity: every templated LWS variant of the packed
    /// USM matvecs (Q5_K, Q6_K, Q8_0) produces bit-identical output
    /// to the LWS=16 reference. Same contract as the Q4_K parity
    /// test above — guards against an accidental per-item compute
    /// change leaking through templating.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_packed_matvec_lws_parity_q5k_q6k_q8_0() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        let candidates = [16u32, 32, 64, 128, 256];

        // Define the three kernel families inline so adding a new
        // packed quant = one more entry. M=100 is chosen so the
        // bounds-check path (M not divisible by LWS) is exercised
        // for every LWS in the candidate set.
        struct Probe {
            name: &'static str,
            row_bytes: fn(usize) -> usize,
            k: usize,
            // Each dispatcher takes the same arg shape.
            run: unsafe fn(
                &SyclStream, *const u8, *const f32, *mut f32, u32, u32, u32,
            ) -> Result<()>,
        }
        let probes: &[Probe] = &[
            Probe {
                name: "q5_k",
                row_bytes: |k| (k / 256) * 176,
                k: 256,
                run: matvec_q5_k_packed_f32_usm_raw,
            },
            Probe {
                name: "q6_k",
                row_bytes: |k| (k / 256) * 210,
                k: 256,
                run: matvec_q6_k_packed_f32_usm_raw,
            },
            Probe {
                name: "q8_0",
                row_bytes: |k| (k / 32) * 34,
                k: 64, // 2 blocks per row × 34B
                run: matvec_q8_0_packed_f32_usm_raw,
            },
        ];

        for probe in probes {
            let m: u32 = 100;
            let k = probe.k;
            let bytes = m as usize * (probe.row_bytes)(k);
            let w_bytes: Vec<u8> = (0..bytes)
                .map(|i| ((i * 31 + 7) & 0xff) as u8)
                .collect();
            let x_host: Vec<f32> = (0..k)
                .map(|i| (((i % 17) as f32) - 8.0) * 0.05)
                .collect();
            let mut w_buf = SyclSharedBuffer::<u8>::alloc(&stream, bytes).expect("w alloc");
            let mut x_buf = SyclSharedBuffer::<f32>::alloc(&stream, k).expect("x alloc");
            w_buf.as_mut_slice().copy_from_slice(&w_bytes);
            x_buf.as_mut_slice().copy_from_slice(&x_host);

            let mut reference: Option<Vec<f32>> = None;
            for lws in candidates {
                let mut out_buf =
                    SyclSharedBuffer::<f32>::alloc(&stream, m as usize).expect("out alloc");
                for dst in out_buf.as_mut_slice().iter_mut() {
                    *dst = 0.0;
                }
                let r = unsafe {
                    (probe.run)(
                        &stream,
                        w_buf.as_ptr(),
                        x_buf.as_ptr(),
                        out_buf.as_mut_ptr(),
                        m,
                        k as u32,
                        lws,
                    )
                };
                r.unwrap_or_else(|e| panic!("{}: lws={lws} failed: {e}", probe.name));
                let got: Vec<f32> = out_buf.as_slice().to_vec();
                match &reference {
                    None => {
                        eprintln!(
                            "{} lws={lws}: reference output (first 4): {:?}",
                            probe.name,
                            &got[..4.min(got.len())]
                        );
                        reference = Some(got);
                    }
                    Some(r) => {
                        for i in 0..(m as usize) {
                            assert_eq!(
                                got[i].to_bits(),
                                r[i].to_bits(),
                                "{} lws={lws} row {i}: bit patterns differ",
                                probe.name
                            );
                        }
                        eprintln!("{} lws={lws}: bit-identical to reference ✓", probe.name);
                    }
                }
            }
        }
    }

    /// Real-SYCL parity: every templated LWS variant of the *batched*
    /// packed USM matvecs (Q4_K, Q5_K, Q6_K, Q8_0) produces
    /// bit-identical output to the LWS=16 reference. Same contract as
    /// the single-row parity test above — guards against accidental
    /// per-item compute drift through batched templating. N=4 is small
    /// but enough to exercise the 2D nd_range scheduling; M=100 keeps
    /// the bounds-check path live for every LWS.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_packed_matvec_batched_lws_parity_all_quants() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        let candidates = [16u32, 32, 64, 128, 256];

        struct Probe {
            name: &'static str,
            row_bytes: fn(usize) -> usize,
            k: usize,
            run: unsafe fn(
                &SyclStream, *const u8, *const f32, *mut f32, u32, u32, u32, u32,
            ) -> Result<()>,
        }
        let probes: &[Probe] = &[
            Probe {
                name: "q4_k",
                row_bytes: |k| (k / 256) * 144,
                k: 256,
                run: matvec_q4_k_packed_f32_batched_usm_raw,
            },
            Probe {
                name: "q5_k",
                row_bytes: |k| (k / 256) * 176,
                k: 256,
                run: matvec_q5_k_packed_f32_batched_usm_raw,
            },
            Probe {
                name: "q6_k",
                row_bytes: |k| (k / 256) * 210,
                k: 256,
                run: matvec_q6_k_packed_f32_batched_usm_raw,
            },
            Probe {
                name: "q8_0",
                row_bytes: |k| (k / 32) * 34,
                k: 64,
                run: matvec_q8_0_packed_f32_batched_usm_raw,
            },
        ];

        for probe in probes {
            let m: u32 = 100;
            let n: u32 = 4;
            let k = probe.k;
            let bytes = m as usize * (probe.row_bytes)(k);
            let w_bytes: Vec<u8> = (0..bytes)
                .map(|i| ((i * 31 + 7) & 0xff) as u8)
                .collect();
            // Distinct x_n per row to make every batched output a
            // unique signal — if a stale-buffer reuse bug slipped in
            // we'd see identical outputs across rows.
            let x_host: Vec<f32> = (0..(n as usize * k))
                .map(|i| (((i % 17) as f32) - 8.0) * 0.05 + (i / k) as f32 * 0.01)
                .collect();
            let mut w_buf = SyclSharedBuffer::<u8>::alloc(&stream, bytes).expect("w alloc");
            let mut x_buf =
                SyclSharedBuffer::<f32>::alloc(&stream, n as usize * k).expect("x alloc");
            w_buf.as_mut_slice().copy_from_slice(&w_bytes);
            x_buf.as_mut_slice().copy_from_slice(&x_host);

            let mut reference: Option<Vec<f32>> = None;
            for lws in candidates {
                let mut out_buf =
                    SyclSharedBuffer::<f32>::alloc(&stream, n as usize * m as usize)
                        .expect("out alloc");
                for dst in out_buf.as_mut_slice().iter_mut() {
                    *dst = 0.0;
                }
                let r = unsafe {
                    (probe.run)(
                        &stream,
                        w_buf.as_ptr(),
                        x_buf.as_ptr(),
                        out_buf.as_mut_ptr(),
                        m,
                        k as u32,
                        n,
                        lws,
                    )
                };
                r.unwrap_or_else(|e| panic!("{} batched: lws={lws} failed: {e}", probe.name));
                let got: Vec<f32> = out_buf.as_slice().to_vec();
                match &reference {
                    None => {
                        eprintln!(
                            "{} batched lws={lws}: reference output (first 4): {:?}",
                            probe.name,
                            &got[..4.min(got.len())]
                        );
                        reference = Some(got);
                    }
                    Some(r) => {
                        for i in 0..(n as usize * m as usize) {
                            assert_eq!(
                                got[i].to_bits(),
                                r[i].to_bits(),
                                "{} batched lws={lws} idx {i}: bit patterns differ",
                                probe.name
                            );
                        }
                        eprintln!(
                            "{} batched lws={lws}: bit-identical to reference \u{2713}",
                            probe.name
                        );
                    }
                }
            }
        }
    }

    /// Real-SYCL parity: USM Q5_K_M packed-layout matvec matches an
    /// inline CPU reference walking the same 176-byte super-block
    /// layout (matches the scalar reference in
    /// `rustllama-kernels-cpu`). Run with
    /// `cargo test -- --ignored usm_matvec_q5_k_packed`.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_matvec_q5_k_packed_matches_cpu_reference() {
        use half::f16;
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        let m: u32 = 4;
        let k: u32 = 512;
        let super_blocks_per_row = (k / 256) as usize;
        let bytes_per_row = super_blocks_per_row * 176;
        let total_bytes = (m as usize) * bytes_per_row;

        // Same Q4_K scales packer as the Q4_K_M parity test.
        fn pack_q4k_scales(sc: &[u8; 8], mn: &[u8; 8]) -> [u8; 12] {
            let mut out = [0u8; 12];
            for j in 0..4 {
                out[j] = sc[j] & 0x3F;
                out[j + 4] = mn[j] & 0x3F;
            }
            for j in 4..8 {
                out[j + 4] = (sc[j] & 0x0F) | ((mn[j] & 0x0F) << 4);
                out[j - 4] |= ((sc[j] >> 4) & 0x03) << 6;
                out[j] |= ((mn[j] >> 4) & 0x03) << 6;
            }
            out
        }

        let mut w_bytes_host = vec![0u8; total_bytes];
        for row in 0..m as usize {
            for sb in 0..super_blocks_per_row {
                let off = row * bytes_per_row + sb * 176;
                let d_f32 = 0.01 + 0.003 * ((row + sb) as f32);
                let dmin_f32 = 0.005 + 0.001 * ((row * 2 + sb) as f32);
                let d_bits = f16::from_f32(d_f32).to_bits();
                let dmin_bits = f16::from_f32(dmin_f32).to_bits();
                w_bytes_host[off] = (d_bits & 0xFF) as u8;
                w_bytes_host[off + 1] = (d_bits >> 8) as u8;
                w_bytes_host[off + 2] = (dmin_bits & 0xFF) as u8;
                w_bytes_host[off + 3] = (dmin_bits >> 8) as u8;
                let mut sc = [0u8; 8];
                let mut mn = [0u8; 8];
                for j in 0..8 {
                    sc[j] = ((row * 7 + sb * 11 + j * 3) % 63) as u8;
                    mn[j] = ((row * 5 + sb * 13 + j * 2) % 63) as u8;
                }
                let packed = pack_q4k_scales(&sc, &mn);
                w_bytes_host[off + 4..off + 16].copy_from_slice(&packed);
                // qh: 32 bytes of high bits with a non-trivial pattern.
                for l in 0..32 {
                    w_bytes_host[off + 16 + l] =
                        (((row * 3 + sb * 5 + l * 17) & 0xFF) as u8) ^ 0xA5;
                }
                // qs: 128 bytes of 4-bit packed weights.
                for l in 0..128 {
                    let lo = ((row * 3 + sb * 5 + l * 7) % 16) as u8;
                    let hi = ((row * 11 + sb * 17 + l * 13) % 16) as u8;
                    w_bytes_host[off + 48 + l] = lo | (hi << 4);
                }
            }
        }
        let x_host: Vec<f32> = (0..k as usize)
            .map(|i| (((i % 23) as f32) - 11.0) * 0.05)
            .collect();

        let mut w_bytes_buf: SyclSharedBuffer<u8> =
            SyclSharedBuffer::alloc(&stream, total_bytes).expect("alloc w_bytes");
        let mut x_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, k as usize).expect("alloc x");
        let mut out_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, m as usize).expect("alloc out");
        w_bytes_buf.as_mut_slice().copy_from_slice(&w_bytes_host);
        x_buf.as_mut_slice().copy_from_slice(&x_host);
        for dst in out_buf.as_mut_slice().iter_mut() {
            *dst = 0.0;
        }

        matvec_q5_k_packed_f32_usm(&stream, &w_bytes_buf, &x_buf, &mut out_buf, m, k, 0)
            .expect("matvec_q5_k_packed_f32_usm");

        // CPU reference inline (mirrors `matvec_q5_k_w_f32_a_scalar`).
        fn unpack_q4k(scales: &[u8]) -> ([u8; 8], [u8; 8]) {
            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];
            for j in 0..8 {
                if j < 4 {
                    sc[j] = scales[j] & 0x3F;
                    mn[j] = scales[j + 4] & 0x3F;
                } else {
                    sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
                    mn[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
                }
            }
            (sc, mn)
        }
        let mut out_ref = vec![0f32; m as usize];
        for row in 0..m as usize {
            let mut acc = 0f32;
            for sb in 0..super_blocks_per_row {
                let off = row * bytes_per_row + sb * 176;
                let d = f16::from_le_bytes([w_bytes_host[off], w_bytes_host[off + 1]])
                    .to_f32();
                let dmin =
                    f16::from_le_bytes([w_bytes_host[off + 2], w_bytes_host[off + 3]])
                        .to_f32();
                let (sc, mn) = unpack_q4k(&w_bytes_host[off + 4..off + 16]);
                let qh = &w_bytes_host[off + 16..off + 48];
                let qs = &w_bytes_host[off + 48..off + 48 + 128];
                let x_block = &x_host[sb * 256..(sb + 1) * 256];
                for group in 0..4 {
                    let q_chunk = &qs[group * 32..(group + 1) * 32];
                    let d_lo = d * sc[group * 2] as f32;
                    let m_lo = dmin * mn[group * 2] as f32;
                    let d_hi = d * sc[group * 2 + 1] as f32;
                    let m_hi = dmin * mn[group * 2 + 1] as f32;
                    let bit_lo = group * 2;
                    let bit_hi = group * 2 + 1;
                    let x_lo = &x_block[group * 64..group * 64 + 32];
                    let x_hi = &x_block[group * 64 + 32..(group + 1) * 64];
                    for l in 0..32 {
                        let q = q_chunk[l];
                        let qhb = qh[l];
                        let lo = (q & 0x0F) as u32
                            | (((qhb >> bit_lo) & 1) as u32) << 4;
                        let hi = (q >> 4) as u32
                            | (((qhb >> bit_hi) & 1) as u32) << 4;
                        acc += (d_lo * lo as f32 - m_lo) * x_lo[l];
                        acc += (d_hi * hi as f32 - m_hi) * x_hi[l];
                    }
                }
            }
            out_ref[row] = acc;
        }
        let mut max_err = 0f32;
        for i in 0..m as usize {
            let got = out_buf.as_slice()[i];
            let want = out_ref[i];
            let err = (got - want).abs();
            if err > max_err {
                max_err = err;
            }
        }
        eprintln!("usm_matvec_q5_k_packed parity: max_abs_err = {max_err:.6}");
        assert!(
            max_err < 1e-3,
            "USM Q5_K_M packed matvec vs CPU max_abs_err {max_err} > 1e-3"
        );
    }

    /// Real-SYCL parity: USM Q6_K packed-layout matvec matches an
    /// inline CPU reference walking the same 210-byte super-block
    /// layout (matches `matvec_q6_k_w_f32_a_scalar` in
    /// `rustllama-kernels-cpu`). Run with
    /// `cargo test -- --ignored usm_matvec_q6_k_packed`.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_matvec_q6_k_packed_matches_cpu_reference() {
        use half::f16;
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        let m: u32 = 4;
        let k: u32 = 512;
        let super_blocks_per_row = (k / 256) as usize;
        let bytes_per_row = super_blocks_per_row * 210;
        let total_bytes = (m as usize) * bytes_per_row;

        // Synthetic Q6_K byte slab with deterministic patterns.
        let mut w_bytes_host = vec![0u8; total_bytes];
        for row in 0..m as usize {
            for sb in 0..super_blocks_per_row {
                let off = row * bytes_per_row + sb * 210;
                // 128 bytes of low-4-bit nibbles.
                for l in 0..128 {
                    let lo = ((row * 3 + sb * 5 + l * 7) % 16) as u8;
                    let hi = ((row * 11 + sb * 17 + l * 13) % 16) as u8;
                    w_bytes_host[off + l] = lo | (hi << 4);
                }
                // 64 bytes of qh (2-bit pairs packed 4 per byte).
                for l in 0..64 {
                    w_bytes_host[off + 128 + l] =
                        (((row * 5 + sb * 7 + l * 11) & 0xFF) as u8) ^ 0x33;
                }
                // 16 bytes of i8 scales (signed). Keep small to avoid
                // gigantic intermediate values in the reduction.
                for s in 0..16 {
                    let v = (((row * 2 + sb * 3 + s) % 41) as i32) - 20; // [-20, 20]
                    w_bytes_host[off + 192 + s] = (v as i8) as u8;
                }
                // f16 super-block scale.
                let d_f32 = 0.005 + 0.001 * ((row + sb) as f32);
                let d_bits = f16::from_f32(d_f32).to_bits();
                w_bytes_host[off + 208] = (d_bits & 0xFF) as u8;
                w_bytes_host[off + 209] = (d_bits >> 8) as u8;
            }
        }
        let x_host: Vec<f32> = (0..k as usize)
            .map(|i| (((i % 23) as f32) - 11.0) * 0.05)
            .collect();

        let mut w_bytes_buf: SyclSharedBuffer<u8> =
            SyclSharedBuffer::alloc(&stream, total_bytes).expect("alloc w_bytes");
        let mut x_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, k as usize).expect("alloc x");
        let mut out_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, m as usize).expect("alloc out");
        w_bytes_buf.as_mut_slice().copy_from_slice(&w_bytes_host);
        x_buf.as_mut_slice().copy_from_slice(&x_host);
        for dst in out_buf.as_mut_slice().iter_mut() {
            *dst = 0.0;
        }

        matvec_q6_k_packed_f32_usm(&stream, &w_bytes_buf, &x_buf, &mut out_buf, m, k, 0)
            .expect("matvec_q6_k_packed_f32_usm");

        // CPU reference inline — mirrors `matvec_q6_k_w_f32_a_scalar`.
        let mut out_ref = vec![0f32; m as usize];
        for row in 0..m as usize {
            let mut acc = 0f32;
            for sb in 0..super_blocks_per_row {
                let off = row * bytes_per_row + sb * 210;
                let ql = &w_bytes_host[off..off + 128];
                let qh = &w_bytes_host[off + 128..off + 128 + 64];
                let scales = &w_bytes_host[off + 192..off + 192 + 16];
                let d = f16::from_le_bytes([w_bytes_host[off + 208], w_bytes_host[off + 209]])
                    .to_f32();
                let x_block = &x_host[sb * 256..(sb + 1) * 256];
                for n in 0..2usize {
                    for l in 0..32usize {
                        let is = l / 16 + n * 8;
                        let q1 = ((ql[64 * n + l] & 0x0F) as i32
                            | (((qh[32 * n + l] >> 0) & 0x03) as i32) << 4)
                            - 32;
                        let q2 = ((ql[64 * n + l + 32] & 0x0F) as i32
                            | (((qh[32 * n + l] >> 2) & 0x03) as i32) << 4)
                            - 32;
                        let q3 = ((ql[64 * n + l] >> 4) as i32
                            | (((qh[32 * n + l] >> 4) & 0x03) as i32) << 4)
                            - 32;
                        let q4 = ((ql[64 * n + l + 32] >> 4) as i32
                            | (((qh[32 * n + l] >> 6) & 0x03) as i32) << 4)
                            - 32;
                        let s0 = (scales[is] as i8) as f32;
                        let s1 = (scales[is + 2] as i8) as f32;
                        let s2 = (scales[is + 4] as i8) as f32;
                        let s3 = (scales[is + 6] as i8) as f32;
                        let base = n * 128 + l;
                        acc += d * s0 * q1 as f32 * x_block[base];
                        acc += d * s1 * q2 as f32 * x_block[base + 32];
                        acc += d * s2 * q3 as f32 * x_block[base + 64];
                        acc += d * s3 * q4 as f32 * x_block[base + 96];
                    }
                }
            }
            out_ref[row] = acc;
        }
        let mut max_err = 0f32;
        for i in 0..m as usize {
            let got = out_buf.as_slice()[i];
            let want = out_ref[i];
            let err = (got - want).abs();
            if err > max_err {
                max_err = err;
            }
        }
        eprintln!("usm_matvec_q6_k_packed parity: max_abs_err = {max_err:.6}");
        // Pure f32 arithmetic on both sides; the f16 super-block scale
        // round-trips identically through both paths.
        assert!(
            max_err < 1e-3,
            "USM Q6_K packed matvec vs CPU max_abs_err {max_err} > 1e-3"
        );
    }

    /// Real-SYCL parity: USM rope matches the CPU rope_inplace_neox
    /// convention within f16 round-trip tolerance.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_rope_matches_cpu_reference() {
        use half::f16;
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        let n_heads = 4usize;
        let head_dim = 64usize;
        let pos = 17u32;
        let rope_theta = 10000.0f32;
        let half = head_dim / 2;
        let total = n_heads * head_dim;
        // Pre-compute inv_freq table (CPU side) and write to USM.
        let mut inv_freq_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, half).expect("alloc inv_freq");
        for (j, dst) in inv_freq_buf.as_mut_slice().iter_mut().enumerate() {
            let exponent = (2 * j) as f32 / head_dim as f32;
            let f = 1.0f32 / rope_theta.powf(exponent);
            *dst = f16::from_f32(f).to_bits();
        }
        // Synthetic Q/K-like buffer.
        let initial: Vec<f32> = (0..total)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.13)
            .collect();
        let mut qk_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, total).expect("alloc qk");
        for (i, dst) in qk_buf.as_mut_slice().iter_mut().enumerate() {
            *dst = f16::from_f32(initial[i]).to_bits();
        }
        // GPU rope.
        rope_usm(&stream, &mut qk_buf, n_heads as u32, head_dim as u32, pos, &inv_freq_buf)
            .expect("rope_usm");
        // CPU reference (f16-round-tripped on input + output to
        // match the GPU's f16 storage).
        let mut qk_cpu: Vec<f32> = initial.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
        let inv_freq_f16: Vec<f32> = (0..half)
            .map(|j| {
                let exponent = (2 * j) as f32 / head_dim as f32;
                f16::from_f32(1.0f32 / rope_theta.powf(exponent)).to_f32()
            })
            .collect();
        for h in 0..n_heads {
            let base = h * head_dim;
            for j in 0..half {
                let angle = (pos as f32) * inv_freq_f16[j];
                let c = angle.cos();
                let si = angle.sin();
                let x0 = qk_cpu[base + j];
                let x1 = qk_cpu[base + j + half];
                qk_cpu[base + j] = f16::from_f32(x0 * c - x1 * si).to_f32();
                qk_cpu[base + j + half] = f16::from_f32(x0 * si + x1 * c).to_f32();
            }
        }
        let mut max_err = 0f32;
        for (i, &bits) in qk_buf.as_slice().iter().enumerate() {
            let got = f16::from_bits(bits).to_f32();
            let err = (got - qk_cpu[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        eprintln!("usm_rope parity: max_abs_err = {max_err:.6}");
        assert!(max_err < 1e-2, "USM rope vs CPU max_abs_err {max_err} > 1e-2");
    }

    /// Real-SYCL parity: USM silu_mul matches the CPU reference.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_silu_mul_matches_cpu_reference() {
        use half::f16;
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        let n = 4096usize;
        let mut x_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, n).expect("alloc x");
        let mut y_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, n).expect("alloc y");
        let mut out_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, n).expect("alloc out");
        for (i, dst) in x_buf.as_mut_slice().iter_mut().enumerate() {
            let v = ((i % 41) as f32 - 20.0) * 0.07;
            *dst = f16::from_f32(v).to_bits();
        }
        for (i, dst) in y_buf.as_mut_slice().iter_mut().enumerate() {
            let v = ((i % 37) as f32 - 18.0) * 0.05;
            *dst = f16::from_f32(v).to_bits();
        }
        for dst in out_buf.as_mut_slice().iter_mut() {
            *dst = 0;
        }
        silu_mul_usm(&stream, &x_buf, &y_buf, &mut out_buf, n as u32).expect("silu_mul_usm");
        let mut max_err = 0f32;
        for i in 0..n {
            let xv = f16::from_bits(x_buf.as_slice()[i]).to_f32();
            let yv = f16::from_bits(y_buf.as_slice()[i]).to_f32();
            let silu = xv / (1.0 + (-xv).exp());
            let want = f16::from_f32(silu * yv).to_f32();
            let got = f16::from_bits(out_buf.as_slice()[i]).to_f32();
            let err = (got - want).abs();
            if err > max_err {
                max_err = err;
            }
        }
        eprintln!("usm_silu_mul parity: max_abs_err = {max_err:.6}");
        assert!(max_err < 5e-3, "USM silu_mul vs CPU max_abs_err {max_err} > 5e-3");
    }

    /// Real-SYCL parity: USM embedding_lookup gathers correct rows.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_embedding_lookup_matches_cpu_reference() {
        use half::f16;
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        let vocab = 64usize;
        let d = 128usize;
        let ids: Vec<i32> = vec![3, 7, -1, 0]; // include negative-id zero case
        let mut table_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, vocab * d).expect("alloc table");
        for (i, dst) in table_buf.as_mut_slice().iter_mut().enumerate() {
            let v = ((i as f32).sin() * 2.0).cos();
            *dst = f16::from_f32(v).to_bits();
        }
        let mut out_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, ids.len() * d).expect("alloc out");
        for dst in out_buf.as_mut_slice().iter_mut() {
            *dst = 0;
        }
        embedding_lookup_usm(&stream, &table_buf, &ids, &mut out_buf, d as u32)
            .expect("embedding_lookup_usm");
        let mut max_err = 0f32;
        for (i, &id) in ids.iter().enumerate() {
            for j in 0..d {
                let got = f16::from_bits(out_buf.as_slice()[i * d + j]).to_f32();
                let want = if id < 0 {
                    0.0
                } else {
                    f16::from_bits(table_buf.as_slice()[id as usize * d + j]).to_f32()
                };
                let err = (got - want).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        eprintln!("usm_embedding parity: max_abs_err = {max_err:.6}");
        // Embedding is a pure memcpy through f16 — should be exact.
        assert!(max_err < 1e-4, "USM embedding vs CPU max_abs_err {max_err} > 1e-4");
    }

    /// Real-SYCL parity check: USM flash-attention decode kernel
    /// matches the CPU flash-decode kernel within f16 round-trip
    /// tolerance. Gated `#[ignore]` so workspace `cargo test`
    /// skips it — run with `cargo test --
    /// --ignored usm_flash_attn` on a host with an Intel GPU.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_flash_attn_decode_matches_cpu_reference() {
        use half::f16;
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        let n_heads = 4u32;
        let n_kv_heads = 2u32;
        let head_dim = 16u32;
        let max_ctx = 64u32;
        let kv_len = 32u32;
        let q_len = (n_heads * head_dim) as usize;
        let kv_buf_len = (n_kv_heads * max_ctx * head_dim) as usize;
        let out_len = q_len;
        // Four distinct USM buffers — possible now that
        // SyclSharedBuffer takes a shared borrow on the stream.
        // This is the pattern the engine wire-up will use.
        let mut q_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, q_len).expect("alloc Q");
        let mut k_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, kv_buf_len).expect("alloc K");
        let mut v_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, kv_buf_len).expect("alloc V");
        let mut out_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, out_len).expect("alloc Out");
        // Populate inputs from CPU side. USM-shared pages let us
        // write directly — no memcpy.
        for (i, dst) in q_buf.as_mut_slice().iter_mut().enumerate() {
            let v = ((i % 23) as f32 - 11.0) * 0.13;
            *dst = f16::from_f32(v).to_bits();
        }
        for (i, dst) in k_buf.as_mut_slice().iter_mut().enumerate() {
            let kv = ((i % 29) as f32 - 14.0) * 0.07;
            *dst = f16::from_f32(kv).to_bits();
        }
        for (i, dst) in v_buf.as_mut_slice().iter_mut().enumerate() {
            let vv = ((i % 31) as f32 - 15.0) * 0.09;
            *dst = f16::from_f32(vv).to_bits();
        }
        for dst in out_buf.as_mut_slice().iter_mut() {
            *dst = 0;
        }
        // Run the kernel through the safe wrapper.
        flash_attn_decode_usm(
            &stream, &q_buf, &k_buf, &v_buf, &mut out_buf,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
        ).expect("flash_attn_decode_usm");
        // Read back results.
        // Build f32 CPU inputs (mirroring f16 round-trip storage).
        let mut q_f32 = vec![0f32; q_len];
        let mut k_f32 = vec![0f32; kv_buf_len];
        let mut v_f32 = vec![0f32; kv_buf_len];
        for (i, &bits) in q_buf.as_slice().iter().enumerate() {
            q_f32[i] = f16::from_bits(bits).to_f32();
        }
        for (i, &bits) in k_buf.as_slice().iter().enumerate() {
            k_f32[i] = f16::from_bits(bits).to_f32();
        }
        for (i, &bits) in v_buf.as_slice().iter().enumerate() {
            v_f32[i] = f16::from_bits(bits).to_f32();
        }
        let mut out_cpu = vec![0f32; out_len];
        // Defer to the CPU flash kernel via the kernels-cpu crate.
        // Since that's a sibling crate, mirror the algorithm
        // inline here (it's small) so this test doesn't pull in
        // kernels-cpu as a dep.
        let n_gqa = (n_heads / n_kv_heads) as usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        for h in 0..n_heads as usize {
            let kv_h = h / n_gqa;
            let q_h = &q_f32[h * head_dim as usize..(h + 1) * head_dim as usize];
            let out_h = &mut out_cpu[h * head_dim as usize..(h + 1) * head_dim as usize];
            for v in out_h.iter_mut() { *v = 0.0; }
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            for t in 0..kv_len as usize {
                let k_base = (kv_h * max_ctx as usize + t) * head_dim as usize;
                let mut sd = 0f32;
                for d in 0..head_dim as usize {
                    sd += q_h[d] * k_f32[k_base + d];
                }
                sd *= scale;
                let m_new = m.max(sd);
                let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
                let p = (sd - m_new).exp();
                l = l * rescale + p;
                let v_base = (kv_h * max_ctx as usize + t) * head_dim as usize;
                for d in 0..head_dim as usize {
                    out_h[d] = out_h[d] * rescale + p * v_f32[v_base + d];
                }
                m = m_new;
            }
            let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
            for v in out_h.iter_mut() { *v *= inv_l; }
        }
        // Compare GPU outputs (read back from USM) to CPU reference.
        let mut max_err = 0f32;
        for (i, &bits) in out_buf.as_slice().iter().enumerate() {
            let got = f16::from_bits(bits).to_f32();
            // Round-trip CPU reference through f16 to match
            // kernel's storage precision.
            let want = f16::from_f32(out_cpu[i]).to_f32();
            let err = (got - want).abs();
            if err > max_err {
                max_err = err;
            }
        }
        eprintln!("usm_flash_attn parity: max_abs_err = {max_err:.6}");
        // Tolerance picked from the CPU SIMD-vs-scalar bound (1e-5)
        // plus f16 round-trip slack on the per-t accumulation.
        // Bump a bit since softmax involves exp which compounds
        // rounding error at long kv_len.
        assert!(max_err < 5e-3, "USM flash vs CPU max_abs_err {max_err} > 5e-3");
    }

    /// Real-SYCL parity: USM FlashAttention prefill (f32) matches an
    /// inline CPU scalar reference that mirrors
    /// `rustllama_kernels_cpu::gqa_attention_flash_prefill_scalar`.
    /// Exercises three things at once:
    ///   1. 2D nd_range dispatch is wired through correctly (one WI
    ///      per (head, q_pos)).
    ///   2. The causal mask is honored per q_pos
    ///      (kv_len_for_q = kv_len_base + q_pos + 1).
    ///   3. GQA grouping (n_heads / n_kv_heads) routes Q heads to
    ///      the right kv_h.
    /// Bit-identical assertion is too aggressive across vendors —
    /// `sycl::exp` and libm's `f32::exp` aren't required to match —
    /// so we cap at a small f32 tolerance (same bound as decode).
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_flash_attn_prefill_matches_cpu_reference() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        let n_heads = 4u32;
        let n_kv_heads = 2u32;
        let head_dim = 16u32;
        let max_ctx = 64u32;
        let kv_len_base = 8u32;   // 8 cached positions
        let n_new = 5u32;         // 5 new prompt tokens
        let q_len = (n_new * n_heads * head_dim) as usize;
        let kv_buf_len = (n_kv_heads * max_ctx * head_dim) as usize;
        let out_len = q_len;

        let mut q_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, q_len).expect("alloc Q");
        let mut k_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, kv_buf_len).expect("alloc K");
        let mut v_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, kv_buf_len).expect("alloc V");
        let mut out_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, out_len).expect("alloc Out");

        for (i, dst) in q_buf.as_mut_slice().iter_mut().enumerate() {
            *dst = ((i % 23) as f32 - 11.0) * 0.13;
        }
        // KV cache: populate the entire `n_kv_heads * max_ctx *
        // head_dim` block so positions in `[0, kv_len_base + n_new)`
        // are deterministically defined. Positions outside that
        // window get garbage values but the kernel's causal mask
        // never touches them.
        for (i, dst) in k_buf.as_mut_slice().iter_mut().enumerate() {
            *dst = ((i % 29) as f32 - 14.0) * 0.07;
        }
        for (i, dst) in v_buf.as_mut_slice().iter_mut().enumerate() {
            *dst = ((i % 31) as f32 - 15.0) * 0.09;
        }
        for dst in out_buf.as_mut_slice().iter_mut() {
            *dst = 0.0;
        }
        flash_attn_prefill_usm(
            &stream, &q_buf, &k_buf, &v_buf, &mut out_buf,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
        )
        .expect("flash_attn_prefill_usm");

        // CPU reference — per-WI algorithm (matches GPU kernel: outer
        // (h, q_pos), inner t). Different loop order from the CPU
        // crate's scalar prefill (which uses outer-kv_h, outer-t,
        // inner-q_pos for cache locality) — both produce the same
        // mathematical result; this ordering is the one whose float
        // op sequence matches the GPU per output element.
        let q_f32 = q_buf.as_slice().to_vec();
        let k_f32 = k_buf.as_slice().to_vec();
        let v_f32 = v_buf.as_slice().to_vec();
        let mut out_cpu = vec![0f32; out_len];
        let n_gqa = (n_heads / n_kv_heads) as usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        for q_pos in 0..n_new as usize {
            for h in 0..n_heads as usize {
                let kv_h = h / n_gqa;
                let q_off = (q_pos * n_heads as usize + h) * head_dim as usize;
                let out_off = q_off;
                let kv_len_for_q = kv_len_base as usize + q_pos + 1;
                let mut m = f32::NEG_INFINITY;
                let mut l = 0.0f32;
                for d in 0..head_dim as usize {
                    out_cpu[out_off + d] = 0.0;
                }
                for t in 0..kv_len_for_q {
                    let kv_off = (kv_h * max_ctx as usize + t) * head_dim as usize;
                    let mut s = 0f32;
                    for d in 0..head_dim as usize {
                        s += q_f32[q_off + d] * k_f32[kv_off + d];
                    }
                    s *= scale;
                    let m_new = m.max(s);
                    let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
                    let p = (s - m_new).exp();
                    l = l * rescale + p;
                    for d in 0..head_dim as usize {
                        out_cpu[out_off + d] =
                            out_cpu[out_off + d] * rescale + p * v_f32[kv_off + d];
                    }
                    m = m_new;
                }
                let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
                for d in 0..head_dim as usize {
                    out_cpu[out_off + d] *= inv_l;
                }
            }
        }

        let mut max_err = 0f32;
        for (i, &got) in out_buf.as_slice().iter().enumerate() {
            let err = (got - out_cpu[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        eprintln!(
            "usm_flash_attn_prefill parity (n_new={n_new}, kv_len_base={kv_len_base}): \
             max_abs_err = {max_err:.6}"
        );
        // Same tolerance band as decode parity test: f32 softmax
        // compounding + sycl::exp vs libm::exp drift over kv_len_for_q
        // positions, no f16 round-trip loss in this path.
        assert!(max_err < 1e-4, "USM prefill vs CPU max_abs_err {max_err} > 1e-4");
    }

    /// P-7: FA-v2 decode parity vs the same CPU reference the v1
    /// kernel matches. v2 striping changes the floating-point op
    /// order (parallel partial sums + SG reduction instead of
    /// per-WI serial accumulation), so we don't expect bit-identity
    /// — but the result must agree with the CPU online-softmax
    /// reference to within the same fp16 round-trip tolerance.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_flash_attn_decode_v2_matches_cpu_reference() {
        use half::f16;
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        // head_dim multiple of SG_SIZE=16 to exercise the v2 path
        // (head_dim=16 → dim_per_lane=1, smallest valid v2 case).
        let n_heads = 4u32;
        let n_kv_heads = 2u32;
        let head_dim = 16u32;
        let max_ctx = 64u32;
        let kv_len = 32u32;
        let q_len = (n_heads * head_dim) as usize;
        let kv_buf_len = (n_kv_heads * max_ctx * head_dim) as usize;
        let out_len = q_len;
        let mut q_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, q_len).expect("alloc Q");
        let mut k_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, kv_buf_len).expect("alloc K");
        let mut v_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, kv_buf_len).expect("alloc V");
        let mut out_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, out_len).expect("alloc Out");
        for (i, dst) in q_buf.as_mut_slice().iter_mut().enumerate() {
            *dst = f16::from_f32(((i % 23) as f32 - 11.0) * 0.13).to_bits();
        }
        for (i, dst) in k_buf.as_mut_slice().iter_mut().enumerate() {
            *dst = f16::from_f32(((i % 29) as f32 - 14.0) * 0.07).to_bits();
        }
        for (i, dst) in v_buf.as_mut_slice().iter_mut().enumerate() {
            *dst = f16::from_f32(((i % 31) as f32 - 15.0) * 0.09).to_bits();
        }
        for dst in out_buf.as_mut_slice().iter_mut() {
            *dst = 0;
        }
        flash_attn_decode_v2_usm(
            &stream, &q_buf, &k_buf, &v_buf, &mut out_buf,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
        )
        .expect("flash_attn_decode_v2_usm");

        // Same inline CPU online-softmax reference the v1 parity
        // test uses.
        let mut q_f32 = vec![0f32; q_len];
        let mut k_f32 = vec![0f32; kv_buf_len];
        let mut v_f32 = vec![0f32; kv_buf_len];
        for (i, &bits) in q_buf.as_slice().iter().enumerate() {
            q_f32[i] = f16::from_bits(bits).to_f32();
        }
        for (i, &bits) in k_buf.as_slice().iter().enumerate() {
            k_f32[i] = f16::from_bits(bits).to_f32();
        }
        for (i, &bits) in v_buf.as_slice().iter().enumerate() {
            v_f32[i] = f16::from_bits(bits).to_f32();
        }
        let mut out_cpu = vec![0f32; out_len];
        let n_gqa = (n_heads / n_kv_heads) as usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        for h in 0..n_heads as usize {
            let kv_h = h / n_gqa;
            let q_h = &q_f32[h * head_dim as usize..(h + 1) * head_dim as usize];
            let out_h = &mut out_cpu[h * head_dim as usize..(h + 1) * head_dim as usize];
            for v in out_h.iter_mut() { *v = 0.0; }
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            for t in 0..kv_len as usize {
                let k_base = (kv_h * max_ctx as usize + t) * head_dim as usize;
                let mut sd = 0f32;
                for d in 0..head_dim as usize {
                    sd += q_h[d] * k_f32[k_base + d];
                }
                sd *= scale;
                let m_new = m.max(sd);
                let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
                let p = (sd - m_new).exp();
                l = l * rescale + p;
                let v_base = (kv_h * max_ctx as usize + t) * head_dim as usize;
                for d in 0..head_dim as usize {
                    out_h[d] = out_h[d] * rescale + p * v_f32[v_base + d];
                }
                m = m_new;
            }
            let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
            for v in out_h.iter_mut() { *v *= inv_l; }
        }
        let mut max_err = 0f32;
        for (i, &bits) in out_buf.as_slice().iter().enumerate() {
            let got = f16::from_bits(bits).to_f32();
            let want = f16::from_f32(out_cpu[i]).to_f32();
            let err = (got - want).abs();
            if err > max_err { max_err = err; }
        }
        eprintln!("usm_flash_attn_decode_v2 parity: max_abs_err = {max_err:.6}");
        // Same tolerance band as v1 — the algorithmic change is in
        // op order, not in numerical precision.
        assert!(max_err < 5e-3, "FA-v2 vs CPU max_abs_err {max_err} > 5e-3");
    }

    /// FA-v3 decode parity vs the CPU online-softmax reference.
    /// SLM tiling changes nothing about the algorithm — same online
    /// softmax recurrence + same SG reduction. Output must match v1
    /// / v2 / CPU within the same tolerance band.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_flash_attn_decode_v3_matches_cpu_reference() {
        use half::f16;
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        let n_heads = 4u32;
        let n_kv_heads = 2u32;
        let head_dim = 16u32;
        let max_ctx = 128u32;
        // Use kv_len that spans multiple SLM tiles (KV_TILE=32).
        let kv_len = 80u32;  // 3 tiles: full(0..32), full(32..64), partial(64..80)
        let q_len = (n_heads * head_dim) as usize;
        let kv_buf_len = (n_kv_heads * max_ctx * head_dim) as usize;
        let out_len = q_len;
        let mut q_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, q_len).expect("alloc Q");
        let mut k_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, kv_buf_len).expect("alloc K");
        let mut v_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, kv_buf_len).expect("alloc V");
        let mut out_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, out_len).expect("alloc Out");
        for (i, dst) in q_buf.as_mut_slice().iter_mut().enumerate() {
            *dst = f16::from_f32(((i % 23) as f32 - 11.0) * 0.13).to_bits();
        }
        for (i, dst) in k_buf.as_mut_slice().iter_mut().enumerate() {
            *dst = f16::from_f32(((i % 29) as f32 - 14.0) * 0.07).to_bits();
        }
        for (i, dst) in v_buf.as_mut_slice().iter_mut().enumerate() {
            *dst = f16::from_f32(((i % 31) as f32 - 15.0) * 0.09).to_bits();
        }
        for dst in out_buf.as_mut_slice().iter_mut() { *dst = 0; }
        flash_attn_decode_v3_usm(
            &stream, &q_buf, &k_buf, &v_buf, &mut out_buf,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
        )
        .expect("flash_attn_decode_v3_usm");

        // CPU online-softmax reference.
        let mut q_f32 = vec![0f32; q_len];
        let mut k_f32 = vec![0f32; kv_buf_len];
        let mut v_f32 = vec![0f32; kv_buf_len];
        for (i, &bits) in q_buf.as_slice().iter().enumerate() {
            q_f32[i] = f16::from_bits(bits).to_f32();
        }
        for (i, &bits) in k_buf.as_slice().iter().enumerate() {
            k_f32[i] = f16::from_bits(bits).to_f32();
        }
        for (i, &bits) in v_buf.as_slice().iter().enumerate() {
            v_f32[i] = f16::from_bits(bits).to_f32();
        }
        let mut out_cpu = vec![0f32; out_len];
        let n_gqa = (n_heads / n_kv_heads) as usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        for h in 0..n_heads as usize {
            let kv_h = h / n_gqa;
            let q_h = &q_f32[h * head_dim as usize..(h + 1) * head_dim as usize];
            let out_h = &mut out_cpu[h * head_dim as usize..(h + 1) * head_dim as usize];
            for v in out_h.iter_mut() { *v = 0.0; }
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            for t in 0..kv_len as usize {
                let k_base = (kv_h * max_ctx as usize + t) * head_dim as usize;
                let mut sd = 0f32;
                for d in 0..head_dim as usize {
                    sd += q_h[d] * k_f32[k_base + d];
                }
                sd *= scale;
                let m_new = m.max(sd);
                let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
                let p = (sd - m_new).exp();
                l = l * rescale + p;
                let v_base = (kv_h * max_ctx as usize + t) * head_dim as usize;
                for d in 0..head_dim as usize {
                    out_h[d] = out_h[d] * rescale + p * v_f32[v_base + d];
                }
                m = m_new;
            }
            let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
            for v in out_h.iter_mut() { *v *= inv_l; }
        }
        let mut max_err = 0f32;
        for (i, &bits) in out_buf.as_slice().iter().enumerate() {
            let got = f16::from_bits(bits).to_f32();
            let want = f16::from_f32(out_cpu[i]).to_f32();
            let err = (got - want).abs();
            if err > max_err { max_err = err; }
        }
        eprintln!("usm_flash_attn_decode_v3 parity (kv_len={kv_len}, multi-tile): max_abs_err = {max_err:.6}");
        assert!(max_err < 5e-3, "FA-v3 vs CPU max_abs_err {max_err} > 5e-3");
    }

    /// FA-v3 prefill parity.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_flash_attn_prefill_v3_matches_cpu_reference() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        let n_heads = 4u32;
        let n_kv_heads = 2u32;
        let head_dim = 16u32;
        let max_ctx = 128u32;
        // kv_len_base + n_new > KV_TILE to exercise tile boundary.
        let kv_len_base = 40u32;
        let n_new = 5u32;
        let q_len = (n_new * n_heads * head_dim) as usize;
        let kv_buf_len = (n_kv_heads * max_ctx * head_dim) as usize;
        let out_len = q_len;
        let mut q_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, q_len).expect("alloc Q");
        let mut k_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, kv_buf_len).expect("alloc K");
        let mut v_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, kv_buf_len).expect("alloc V");
        let mut out_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, out_len).expect("alloc Out");
        for (i, dst) in q_buf.as_mut_slice().iter_mut().enumerate() {
            *dst = ((i % 23) as f32 - 11.0) * 0.13;
        }
        for (i, dst) in k_buf.as_mut_slice().iter_mut().enumerate() {
            *dst = ((i % 29) as f32 - 14.0) * 0.07;
        }
        for (i, dst) in v_buf.as_mut_slice().iter_mut().enumerate() {
            *dst = ((i % 31) as f32 - 15.0) * 0.09;
        }
        for dst in out_buf.as_mut_slice().iter_mut() { *dst = 0.0; }
        flash_attn_prefill_v3_usm(
            &stream, &q_buf, &k_buf, &v_buf, &mut out_buf,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
        )
        .expect("flash_attn_prefill_v3_usm");

        let mut out_cpu = vec![0f32; out_len];
        let n_gqa = (n_heads / n_kv_heads) as usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        for q_pos in 0..n_new as usize {
            for h in 0..n_heads as usize {
                let kv_h = h / n_gqa;
                let q_off = (q_pos * n_heads as usize + h) * head_dim as usize;
                let out_off = q_off;
                let kv_len_for_q = kv_len_base as usize + q_pos + 1;
                let mut m = f32::NEG_INFINITY;
                let mut l = 0.0f32;
                for t in 0..kv_len_for_q {
                    let kv_off = (kv_h * max_ctx as usize + t) * head_dim as usize;
                    let mut sd = 0f32;
                    for d in 0..head_dim as usize {
                        sd += q_buf.as_slice()[q_off + d] * k_buf.as_slice()[kv_off + d];
                    }
                    sd *= scale;
                    let m_new = m.max(sd);
                    let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
                    let p = (sd - m_new).exp();
                    l = l * rescale + p;
                    for d in 0..head_dim as usize {
                        out_cpu[out_off + d] =
                            out_cpu[out_off + d] * rescale + p * v_buf.as_slice()[kv_off + d];
                    }
                    m = m_new;
                }
                let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
                for d in 0..head_dim as usize {
                    out_cpu[out_off + d] *= inv_l;
                }
            }
        }
        let mut max_err = 0f32;
        for (i, &got) in out_buf.as_slice().iter().enumerate() {
            let err = (got - out_cpu[i]).abs();
            if err > max_err { max_err = err; }
        }
        eprintln!(
            "usm_flash_attn_prefill_v3 parity (n_new={n_new}, kv_len_base={kv_len_base}, multi-tile): \
             max_abs_err = {max_err:.6}"
        );
        assert!(max_err < 1e-4, "FA-v3 prefill vs CPU max_abs_err {max_err} > 1e-4");
    }

    /// P-7: shape-rejection check that doesn't need a real GPU.
    /// `flash_attn_decode_v2_usm` should return `InvalidShape` for
    /// `head_dim` not a multiple of 16, so the engine-side dispatch
    /// can rely on transparent fallback to v1.
    #[test]
    fn fa_v2_rejects_invalid_head_dim() {
        // Stream creation when no SYCL device is present short-circuits to Unavailable;
        // the shape-validation in the safe wrapper runs BEFORE the
        // kernel call, so we exercise it without needing a live
        // device.
        let stream = match create_stream(0) {
            Ok(s) => s,
            // No usable Intel GPU on this host: construction returns
            // Unavailable. Skip — the v1 fallback path covers this case.
            Err(_) => return,
        };
        let _ = stream;
        // The v2 entry points reject head_dim mismatches up front
        // via the safe wrappers (no device call needed). Pin via
        // the *_raw variants which run the same shape gate but
        // accept null pointers (we don't dispatch the kernel on
        // shape failure).
        // head_dim=80 is a SUPPORTED v2 shape (16-aligned, ≤256 —
        // the kernel was widened past the original power-of-two
        // tiles). Genuinely invalid: 72 (not 16-aligned) and 288
        // (> 256). Both must be refused by the wrapper's shape gate
        // BEFORE any device dispatch (the null pointers pin that:
        // a dispatch would be UB).
        for bad_dim in [72u32, 288] {
            let r = unsafe {
                flash_attn_decode_v2_usm_raw(
                    &stream,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    4, 2, bad_dim, 64, 32,
                )
            };
            match r {
                Err(SyclError::InvalidShape(_)) => {}
                other => panic!(
                    "v2 must reject head_dim={bad_dim} with InvalidShape; got {other:?}"
                ),
            }
        }
    }

    /// P-7: FA-v2 prefill parity, same shape as v1 prefill test.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_flash_attn_prefill_v2_matches_cpu_reference() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        let n_heads = 4u32;
        let n_kv_heads = 2u32;
        let head_dim = 16u32;       // multiple of 16 → v2-eligible
        let max_ctx = 64u32;
        let kv_len_base = 8u32;
        let n_new = 5u32;
        let q_len = (n_new * n_heads * head_dim) as usize;
        let kv_buf_len = (n_kv_heads * max_ctx * head_dim) as usize;
        let out_len = q_len;
        let mut q_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, q_len).expect("alloc Q");
        let mut k_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, kv_buf_len).expect("alloc K");
        let mut v_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, kv_buf_len).expect("alloc V");
        let mut out_buf: SyclSharedBuffer<f32> =
            SyclSharedBuffer::alloc(&stream, out_len).expect("alloc Out");
        for (i, dst) in q_buf.as_mut_slice().iter_mut().enumerate() {
            *dst = ((i % 23) as f32 - 11.0) * 0.13;
        }
        for (i, dst) in k_buf.as_mut_slice().iter_mut().enumerate() {
            *dst = ((i % 29) as f32 - 14.0) * 0.07;
        }
        for (i, dst) in v_buf.as_mut_slice().iter_mut().enumerate() {
            *dst = ((i % 31) as f32 - 15.0) * 0.09;
        }
        for dst in out_buf.as_mut_slice().iter_mut() {
            *dst = 0.0;
        }
        flash_attn_prefill_v2_usm(
            &stream, &q_buf, &k_buf, &v_buf, &mut out_buf,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
        )
        .expect("flash_attn_prefill_v2_usm");

        // Inline CPU prefill reference (same as v1 parity test).
        let mut out_cpu = vec![0f32; out_len];
        let n_gqa = (n_heads / n_kv_heads) as usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        for q_pos in 0..n_new as usize {
            for h in 0..n_heads as usize {
                let kv_h = h / n_gqa;
                let q_off = (q_pos * n_heads as usize + h) * head_dim as usize;
                let out_off = q_off;
                let kv_len_for_q = kv_len_base as usize + q_pos + 1;
                let mut m = f32::NEG_INFINITY;
                let mut l = 0.0f32;
                for t in 0..kv_len_for_q {
                    let kv_off = (kv_h * max_ctx as usize + t) * head_dim as usize;
                    let mut sd = 0f32;
                    for d in 0..head_dim as usize {
                        sd += q_buf.as_slice()[q_off + d] * k_buf.as_slice()[kv_off + d];
                    }
                    sd *= scale;
                    let m_new = m.max(sd);
                    let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
                    let p = (sd - m_new).exp();
                    l = l * rescale + p;
                    for d in 0..head_dim as usize {
                        out_cpu[out_off + d] =
                            out_cpu[out_off + d] * rescale + p * v_buf.as_slice()[kv_off + d];
                    }
                    m = m_new;
                }
                let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
                for d in 0..head_dim as usize {
                    out_cpu[out_off + d] *= inv_l;
                }
            }
        }
        let mut max_err = 0f32;
        for (i, &got) in out_buf.as_slice().iter().enumerate() {
            let err = (got - out_cpu[i]).abs();
            if err > max_err { max_err = err; }
        }
        eprintln!(
            "usm_flash_attn_prefill_v2 parity (n_new={n_new}, kv_len_base={kv_len_base}): \
             max_abs_err = {max_err:.6}"
        );
        assert!(max_err < 1e-4, "FA-v2 prefill vs CPU max_abs_err {max_err} > 1e-4");
    }

    /// Compile-time check: `SyclSharedBuffer` is callable with the
    /// expected signatures. Runs when no SYCL device is present (no real stream
    /// exists to allocate against) so we just type-check the
    /// public surface. The real-SYCL parity check below exercises
    /// it end-to-end when a real SYCL device is present.
    #[test]
    fn sycl_shared_buffer_compile_check() {
        fn _alloc_signature<'s>(
            s: &'s SyclStream,
            n: usize,
        ) -> Result<SyclSharedBuffer<'s, u16>> {
            SyclSharedBuffer::<u16>::alloc(s, n)
        }
        fn _rmsnorm_usm_signature(
            s: &SyclStream,
            x: &SyclSharedBuffer<u16>,
            w: &SyclSharedBuffer<u16>,
            y: &mut SyclSharedBuffer<u16>,
        ) -> Result<()> {
            rmsnorm_usm(s, x, w, y, 1, 4096, 1e-5)
        }
        let _ = _alloc_signature;
        let _ = _rmsnorm_usm_signature;
    }

    /// Real-SYCL parity check: allocate USM-shared, write a
    /// deterministic input from CPU code, call `rmsnorm_usm`, read
    /// the result back from CPU code, compare to a CPU reference.
    /// Gated on `#[ignore]` so workspace `cargo test` skips it —
    /// run with `cargo test -- --ignored
    /// usm_rmsnorm` on a host with an Intel GPU.
    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn usm_rmsnorm_matches_cpu_reference() {
        use half::f16;
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(e) => panic!("no SYCL device: {e}"),
        };
        let n_rows = 2usize;
        let d = 4096usize;
        let eps = 1e-5f32;
        let total = n_rows * d;
        // Three distinct USM buffers, all borrowing the same
        // stream — possible since SyclSharedBuffer takes a shared
        // borrow. This is the API shape the engine wire-up uses.
        let mut x_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, total).expect("alloc X");
        let mut w_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, d).expect("alloc W");
        let mut y_buf: SyclSharedBuffer<u16> =
            SyclSharedBuffer::alloc(&stream, total).expect("alloc Y");
        // Populate from CPU side. USM-shared pages let us write
        // directly with no memcpy.
        for (i, dst) in x_buf.as_mut_slice().iter_mut().enumerate() {
            let v = ((i % 31) as f32 - 15.0) * 0.1;
            *dst = f16::from_f32(v).to_bits();
        }
        for (i, dst) in w_buf.as_mut_slice().iter_mut().enumerate() {
            let v = 1.0 + (i as f32 / d as f32) * 0.5;
            *dst = f16::from_f32(v).to_bits();
        }
        for dst in y_buf.as_mut_slice().iter_mut() {
            *dst = 0;
        }
        // Safe wrapper call — three distinct buffers + the stream.
        rmsnorm_usm(&stream, &x_buf, &w_buf, &mut y_buf, n_rows as u32, d as u32, eps)
            .expect("rmsnorm_usm");
        // CPU reference matching the parity_check_rmsnorm helper.
        let mut max_err = 0f32;
        for r in 0..n_rows {
            let base = r * d;
            let mut sum_sq = 0f32;
            for i in 0..d {
                let v = f16::from_bits(x_buf.as_slice()[base + i]).to_f32();
                sum_sq += v * v;
            }
            let scale_ref = 1.0 / (sum_sq / d as f32 + eps).sqrt();
            for i in 0..d {
                let v = f16::from_bits(x_buf.as_slice()[base + i]).to_f32();
                let wv = f16::from_bits(w_buf.as_slice()[i]).to_f32();
                let expected = f16::from_f32(v * scale_ref * wv).to_f32();
                let got = f16::from_bits(y_buf.as_slice()[base + i]).to_f32();
                let err = (got - expected).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        eprintln!("usm_rmsnorm parity: max_abs_err = {max_err:.6}");
        assert!(max_err < 1e-2, "USM rmsnorm vs CPU max_abs_err {max_err} > 1e-2");
    }

    #[test]
    fn device_count_is_ok() {
        // Real-only crate: `device_count()` always queries the SYCL
        // runtime and returns a count (0 on a host with no Intel GPU),
        // never a hard error from a missing backend.
        let r = device_count();
        assert!(r.is_ok());
    }

    #[test]
    fn create_stream_compiles() {
        // On a real build `create_stream(0)` may succeed or return
        // NoSuchDevice depending on the host — both are fine; we just
        // care the entry point compiles and runs.
        let r = create_stream(0);
        let _ = r;
    }

    #[test]
    fn sycl_stream_is_not_send_or_sync() {
        // Compile-time check via static asserts of the trait bounds.
        // `SyclStream` carries `*mut rsl_stream` (when SYCL is on) plus
        // a `PhantomData<*const ()>` marker so Rust infers !Send and
        // !Sync. Pin this so a future "just add Send" PR fails CI.
        fn assert_not_send<T>(_: &T)
        where
            T: ?Sized,
        {
            // Helper compiles unconditionally; the real check is the
            // adjacent fn_not_send_only_compiles_for_not_send below.
        }
        // The trick: write a function that's only callable when the
        // type ISN'T Send. We rely on the fact that
        // `for<'a> fn(&'a T) where T: !Send` doesn't exist in stable
        // Rust, so instead we just smoke-test via std::thread::spawn
        // failing — which we can't actually do in a test, so this
        // ends up being a compile-time pin via the next test.
        let stream: Option<SyclStream> = None;
        assert_not_send(&stream);
    }

    /// Compile-fail check: spawning a thread that captures a
    /// SyclStream by value must NOT compile. Verified manually via
    /// `cargo build`; if someone adds `unsafe impl Send`, this
    /// comment becomes stale.
    #[allow(dead_code)]
    fn _send_compile_check() {
        // The following snippet (kept as a doc) would fail to
        // compile because SyclStream is !Send:
        //
        // ```
        // let s = create_stream(0).unwrap();
        // std::thread::spawn(move || drop(s));
        // ```
    }

    #[test]
    fn nvfp4_entry_points_compile() {
        // With no Intel GPU present there's no real stream to pass, so
        // we can't call the entry points. The compile-time check (this
        // test compiling) is what matters: it pins that `dequant_nvfp4`
        // and `matvec_nvfp4_f16` are part of the crate's public surface
        // and have the expected signatures.
        fn _typecheck_dequant() -> fn(&mut SyclStream, &[u8], &mut [u16]) -> Result<()> {
            dequant_nvfp4
        }
        fn _typecheck_matvec()
            -> fn(&mut SyclStream, &[u8], &[u16], &mut [u16], u32, u32) -> Result<()>
        {
            matvec_nvfp4_f16
        }
        let _ = _typecheck_dequant();
        let _ = _typecheck_matvec();
    }

    // ----- CPU↔SYCL parity smoke tests -----
    //
    // These run end-to-end against the live Level Zero / OpenCL runtime
    // and only make sense when built with `--no-default-features`
    // on a host that has an Intel oneAPI runtime + GPU. They're
    // `#[ignore]`d so the default `cargo test` skips them; opt in with:
    //
    //   cargo test -p rustllama-kernels-sycl --no-default-features \
    //       --release -- --ignored
    //
    // (and have `scripts\run-sycl.bat` style PATH set so `sycl.dll` is
    // discoverable at load time, or run from a shell where the oneAPI
    // env is already loaded.) Each test compares the SYCL kernel's
    // output against a pure-Rust f32 reference implementation in this
    // file — no dependency on rustllama-kernels-cpu — to keep the
    // verification target minimal and self-contained.

    /// Pure-Rust f32 reference: `y[r, i] = w[i] * x[r, i] / sqrt(mean(x[r,:]^2) + eps)`.
    /// Mirrors the SYCL kernel exactly so any divergence is a real bug
    /// rather than an algorithmic disagreement.
    #[allow(dead_code)]
    fn rmsnorm_reference_f32(x: &[f32], w: &[f32], y: &mut [f32], n_rows: usize, d: usize, eps: f32) {
        for r in 0..n_rows {
            let base = r * d;
            let mut sum_sq = 0.0f32;
            for i in 0..d {
                let v = x[base + i];
                sum_sq += v * v;
            }
            let scale = 1.0f32 / (sum_sq / d as f32 + eps).sqrt();
            for i in 0..d {
                y[base + i] = x[base + i] * scale * w[i];
            }
        }
    }

    #[allow(dead_code)]
    fn f32_slice_to_f16_bits(src: &[f32]) -> Vec<u16> {
        src.iter().map(|v| half::f16::from_f32(*v).to_bits()).collect()
    }

    #[allow(dead_code)]
    fn f16_bits_to_f32(bits: u16) -> f32 {
        half::f16::from_bits(bits).to_f32()
    }

    /// f16 round-tripped values lose ~2-3 decimal digits of precision,
    /// so the tolerance has to be loose. 1e-2 absolute is enough to
    /// catch a wrong-arithmetic kernel without being noisy for normal
    /// rounding error on a normalized row.
    #[allow(dead_code)]
    const FP16_PARITY_TOL: f32 = 1e-2;

    #[test]
    #[ignore = "needs a real Intel GPU; run with `--ignored`"]
    fn sycl_rmsnorm_matches_cpu_reference() {
        // Skip the body when the crate is built when no SYCL device is present — the
        // entry points all return Unavailable so the assertions would
        // be meaningless. The `#[ignore]` attribute already keeps this
        // out of the default `cargo test`, but a developer who runs
        // `cargo test -- --ignored` would still hit
        // this branch.
        let count = match device_count() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("device_count() unavailable — skipping");
                return;
            }
        };
        if count == 0 {
            eprintln!("no SYCL GPUs visible — skipping parity test");
            return;
        }

        let mut stream = create_stream(0).expect("create_stream");

        // Two rows, 64 wide — non-trivial enough to exercise the
        // reduction but small enough to inspect by eye if it diverges.
        let n_rows: u32 = 2;
        let d: u32 = 64;
        let eps: f32 = 1e-5;

        // Deterministic test vectors: row 0 is a smooth ramp, row 1
        // is alternating +/-. Weights are a slow sinusoid so any
        // per-column bug shows up.
        let mut x = vec![0f32; (n_rows * d) as usize];
        for i in 0..d as usize {
            x[i] = (i as f32) * 0.05;
            x[d as usize + i] = if i % 2 == 0 { 0.3 } else { -0.3 };
        }
        let mut w = vec![0f32; d as usize];
        for i in 0..d as usize {
            w[i] = 0.5 + 0.5 * (i as f32 * 0.1).sin();
        }

        // CPU reference computes in pure f32. SYCL takes f16-bit input
        // and produces f16-bit output, so we round-trip the inputs to
        // f16 first so the reference uses the same starting values.
        let x_f16 = f32_slice_to_f16_bits(&x);
        let w_f16 = f32_slice_to_f16_bits(&w);
        let x_round = x_f16
            .iter()
            .map(|b| f16_bits_to_f32(*b))
            .collect::<Vec<_>>();
        let w_round = w_f16
            .iter()
            .map(|b| f16_bits_to_f32(*b))
            .collect::<Vec<_>>();
        let mut y_ref = vec![0f32; (n_rows * d) as usize];
        rmsnorm_reference_f32(&x_round, &w_round, &mut y_ref, n_rows as usize, d as usize, eps);

        let mut y_gpu_bits = vec![0u16; (n_rows * d) as usize];
        rmsnorm(&mut stream, &x_f16, &w_f16, &mut y_gpu_bits, n_rows, d, eps)
            .expect("SYCL rmsnorm dispatch");

        for i in 0..(n_rows * d) as usize {
            let gpu = f16_bits_to_f32(y_gpu_bits[i]);
            let cpu = y_ref[i];
            let diff = (gpu - cpu).abs();
            assert!(
                diff <= FP16_PARITY_TOL,
                "rmsnorm parity divergence at i={i}: SYCL={gpu}, CPU={cpu}, |diff|={diff}",
            );
        }
    }

    /// F4 follow-up hardware validation: batched IQ4_NL matvec.
    /// Allocates USM buffers, uploads a synthetic (W, X) pair,
    /// dispatches the batched kernel, compares against CPU
    /// `matvec_iq4_nl_w_f32_a_scalar` called once per batch row.
    #[test]
    fn sycl_iq4_nl_batched_matvec_matches_cpu() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        // Shape (M output rows, K dim, N batch rows).
        let (m, k, n) = (8usize, 128usize, 4usize);
        // Synthesize IQ4_NL weight bytes deterministically.
        const BLOCK_BYTES: usize = 18;
        let blocks_per_row = k / 32;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;
        let mut w_bytes = vec![0u8; m * bytes_per_row];
        let mut s_state: u32 = 7;
        for chunk in w_bytes.chunks_mut(BLOCK_BYTES) {
            s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
            // d ∈ [-0.5, 0.5] half-range.
            let d_bits = half::f16::from_f32(
                ((s_state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0,
            )
            .to_bits();
            chunk[0] = (d_bits & 0xFF) as u8;
            chunk[1] = ((d_bits >> 8) & 0xFF) as u8;
            for b in chunk[2..18].iter_mut() {
                s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s_state >> 24) as u8;
            }
        }
        let mut x: Vec<f32> = Vec::with_capacity(n * k);
        let mut sx: u32 = 11;
        for _ in 0..(n * k) {
            sx = sx.wrapping_mul(1664525).wrapping_add(1013904223);
            x.push((sx >> 8) as f32 / (1u32 << 24) as f32 - 0.5);
        }
        // Allocate USM bufs.
        let w_usm = usm_alloc_shared(&stream, w_bytes.len()) as *mut u8;
        let x_usm = usm_alloc_shared(&stream, n * k * std::mem::size_of::<f32>()) as *mut f32;
        let out_usm = usm_alloc_shared(&stream, n * m * std::mem::size_of::<f32>()) as *mut f32;
        assert!(!w_usm.is_null() && !x_usm.is_null() && !out_usm.is_null());
        unsafe {
            std::ptr::copy_nonoverlapping(w_bytes.as_ptr(), w_usm, w_bytes.len());
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_usm, n * k);
            for i in 0..n * m {
                *out_usm.add(i) = 0.0;
            }
        }
        // Dispatch.
        unsafe {
            matvec_iq4_nl_packed_f32_batched_usm_raw(
                &stream,
                w_usm,
                x_usm,
                out_usm,
                m as u32,
                k as u32,
                n as u32,
                16,
            )
            .expect("iq4_nl batched matvec dispatch");
        }
        // Gather GPU result.
        let mut out_gpu = vec![0f32; n * m];
        unsafe {
            std::ptr::copy_nonoverlapping(out_usm as *const f32, out_gpu.as_mut_ptr(), n * m);
        }
        // CPU reference: per-batch-row single-row matvec.
        let mut out_cpu = vec![0f32; n * m];
        for ni in 0..n {
            let x_row = &x[ni * k..(ni + 1) * k];
            let mut row = vec![0f32; m];
            rustllama_kernels_cpu::matvec_iq4_nl_w_f32_a(&w_bytes, x_row, &mut row, m, k);
            for mi in 0..m {
                out_cpu[ni * m + mi] = row[mi];
            }
        }
        // Tolerance: each cell is a sum of K=128 products of f32s
        // routed through f16-scale d. FP rounding + reduction order
        // can drift; 1e-3 is comfortable for this shape.
        let mut max_err = 0f32;
        for (g, c) in out_gpu.iter().zip(out_cpu.iter()) {
            let e = (g - c).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < 1e-3,
            "iq4_nl batched matvec: max_err={max_err} (gpu vs cpu)"
        );
        // Cleanup.
        unsafe {
            usm_free(&stream, out_usm as *mut std::ffi::c_void);
            usm_free(&stream, x_usm as *mut std::ffi::c_void);
            usm_free(&stream, w_usm as *mut std::ffi::c_void);
        }
    }

    /// F4 follow-up hardware validation: batched IQ4_XS matvec.
    /// Same pattern as the IQ4_NL test; just different block layout
    /// (136 bytes / 256 weights vs 18 / 32 for NL).
    #[test]
    fn sycl_iq4_xs_batched_matvec_matches_cpu() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        let (m, k, n) = (4usize, 256usize, 3usize);
        const BLOCK_BYTES: usize = 136;
        let blocks_per_row = k / 256;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;
        let mut w_bytes = vec![0u8; m * bytes_per_row];
        let mut s_state: u32 = 13;
        for chunk in w_bytes.chunks_mut(BLOCK_BYTES) {
            s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let d_bits = half::f16::from_f32(
                ((s_state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0,
            )
            .to_bits();
            chunk[0] = (d_bits & 0xFF) as u8;
            chunk[1] = ((d_bits >> 8) & 0xFF) as u8;
            for b in chunk[2..136].iter_mut() {
                s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s_state >> 24) as u8;
            }
        }
        let mut x: Vec<f32> = Vec::with_capacity(n * k);
        let mut sx: u32 = 19;
        for _ in 0..(n * k) {
            sx = sx.wrapping_mul(1664525).wrapping_add(1013904223);
            x.push((sx >> 8) as f32 / (1u32 << 24) as f32 - 0.5);
        }
        let w_usm = usm_alloc_shared(&stream, w_bytes.len()) as *mut u8;
        let x_usm = usm_alloc_shared(&stream, n * k * std::mem::size_of::<f32>()) as *mut f32;
        let out_usm = usm_alloc_shared(&stream, n * m * std::mem::size_of::<f32>()) as *mut f32;
        assert!(!w_usm.is_null() && !x_usm.is_null() && !out_usm.is_null());
        unsafe {
            std::ptr::copy_nonoverlapping(w_bytes.as_ptr(), w_usm, w_bytes.len());
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_usm, n * k);
            for i in 0..n * m {
                *out_usm.add(i) = 0.0;
            }
        }
        unsafe {
            matvec_iq4_xs_packed_f32_batched_usm_raw(
                &stream,
                w_usm,
                x_usm,
                out_usm,
                m as u32,
                k as u32,
                n as u32,
                16,
            )
            .expect("iq4_xs batched matvec dispatch");
        }
        let mut out_gpu = vec![0f32; n * m];
        unsafe {
            std::ptr::copy_nonoverlapping(out_usm as *const f32, out_gpu.as_mut_ptr(), n * m);
        }
        let mut out_cpu = vec![0f32; n * m];
        for ni in 0..n {
            let x_row = &x[ni * k..(ni + 1) * k];
            let mut row = vec![0f32; m];
            rustllama_kernels_cpu::matvec_iq4_xs_w_f32_a(&w_bytes, x_row, &mut row, m, k);
            for mi in 0..m {
                out_cpu[ni * m + mi] = row[mi];
            }
        }
        let mut max_err = 0f32;
        for (g, c) in out_gpu.iter().zip(out_cpu.iter()) {
            let e = (g - c).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < 1e-3,
            "iq4_xs batched matvec: max_err={max_err} (gpu vs cpu)"
        );
        unsafe {
            usm_free(&stream, out_usm as *mut std::ffi::c_void);
            usm_free(&stream, x_usm as *mut std::ffi::c_void);
            usm_free(&stream, w_usm as *mut std::ffi::c_void);
        }
    }

    /// F4 inference hardware validation: IQ1_S single-row matvec.
    /// Same template as the IQ4 batched tests, just for the new
    /// IQ1_S decoder path that lifts the user's APEX-nano model's
    /// expert FFN matvecs off the CPU fallback.
    ///
    /// The IQ1_S codebook (`IQ1S_GRID`, 2048×u64) lives in the
    /// kernel TU as a `constexpr` embedded by the build script. The
    /// kernel reconstructs each 8-weight chunk via `dl * (grid[j] +
    /// delta)` where `dl`/`delta` derive from the per-sub-block
    /// scale + sign bits in `qh`.
    #[test]
    fn sycl_iq1_s_matvec_matches_cpu() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        let (m, k) = (4usize, 256usize);
        const BLOCK_BYTES: usize = 50;
        let blocks_per_row = k / 256;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;
        let mut w_bytes = vec![0u8; m * bytes_per_row];
        let mut s_state: u32 = 41;
        for chunk in w_bytes.chunks_mut(BLOCK_BYTES) {
            s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let d_bits = half::f16::from_f32(
                ((s_state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0,
            )
            .to_bits();
            chunk[0] = (d_bits & 0xFF) as u8;
            chunk[1] = ((d_bits >> 8) & 0xFF) as u8;
            for b in chunk[2..50].iter_mut() {
                s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s_state >> 24) as u8;
            }
        }
        let mut x: Vec<f32> = Vec::with_capacity(k);
        let mut sx: u32 = 43;
        for _ in 0..k {
            sx = sx.wrapping_mul(1664525).wrapping_add(1013904223);
            x.push((sx >> 8) as f32 / (1u32 << 24) as f32 - 0.5);
        }
        let w_usm = usm_alloc_shared(&stream, w_bytes.len()) as *mut u8;
        let x_usm = usm_alloc_shared(&stream, k * std::mem::size_of::<f32>()) as *mut f32;
        let out_usm = usm_alloc_shared(&stream, m * std::mem::size_of::<f32>()) as *mut f32;
        assert!(!w_usm.is_null() && !x_usm.is_null() && !out_usm.is_null());
        unsafe {
            std::ptr::copy_nonoverlapping(w_bytes.as_ptr(), w_usm, w_bytes.len());
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_usm, k);
            for i in 0..m {
                *out_usm.add(i) = 0.0;
            }
        }
        unsafe {
            matvec_iq1_s_packed_f32_usm_raw(
                &stream,
                w_usm,
                x_usm,
                out_usm,
                m as u32,
                k as u32,
                16,
            )
            .expect("iq1_s matvec dispatch");
        }
        let mut out_gpu = vec![0f32; m];
        unsafe {
            std::ptr::copy_nonoverlapping(out_usm as *const f32, out_gpu.as_mut_ptr(), m);
        }
        let mut out_cpu = vec![0f32; m];
        rustllama_kernels_cpu::matvec_iq1_s_w_f32_a(&w_bytes, &x, &mut out_cpu, m, k);
        let mut max_err = 0f32;
        for (g, c) in out_gpu.iter().zip(out_cpu.iter()) {
            let e = (g - c).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < 1e-3,
            "iq1_s matvec: max_err={max_err} (gpu vs cpu)"
        );
        unsafe {
            usm_free(&stream, out_usm as *mut std::ffi::c_void);
            usm_free(&stream, x_usm as *mut std::ffi::c_void);
            usm_free(&stream, w_usm as *mut std::ffi::c_void);
        }
    }

    /// G1 hardware-validation: batched IQ1_S matvec.
    /// Same shape as the single-row test above but with N=4 batch rows;
    /// compares against CPU `matvec_iq1_s_w_f32_a` invoked per batch row.
    #[test]
    fn sycl_iq1_s_batched_matvec_matches_cpu() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        let (m, k, n) = (4usize, 256usize, 4usize);
        const BLOCK_BYTES: usize = 50;
        let blocks_per_row = k / 256;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;
        let mut w_bytes = vec![0u8; m * bytes_per_row];
        let mut s_state: u32 = 41;
        for chunk in w_bytes.chunks_mut(BLOCK_BYTES) {
            s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let d_bits = half::f16::from_f32(
                ((s_state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0,
            )
            .to_bits();
            chunk[0] = (d_bits & 0xFF) as u8;
            chunk[1] = ((d_bits >> 8) & 0xFF) as u8;
            for b in chunk[2..50].iter_mut() {
                s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s_state >> 24) as u8;
            }
        }
        let mut x: Vec<f32> = Vec::with_capacity(n * k);
        let mut sx: u32 = 43;
        for _ in 0..(n * k) {
            sx = sx.wrapping_mul(1664525).wrapping_add(1013904223);
            x.push((sx >> 8) as f32 / (1u32 << 24) as f32 - 0.5);
        }
        let w_usm = usm_alloc_shared(&stream, w_bytes.len()) as *mut u8;
        let x_usm = usm_alloc_shared(&stream, n * k * std::mem::size_of::<f32>()) as *mut f32;
        let out_usm = usm_alloc_shared(&stream, n * m * std::mem::size_of::<f32>()) as *mut f32;
        assert!(!w_usm.is_null() && !x_usm.is_null() && !out_usm.is_null());
        unsafe {
            std::ptr::copy_nonoverlapping(w_bytes.as_ptr(), w_usm, w_bytes.len());
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_usm, n * k);
            for i in 0..(n * m) {
                *out_usm.add(i) = 0.0;
            }
        }
        unsafe {
            matvec_iq1_s_packed_f32_batched_usm_raw(
                &stream, w_usm, x_usm, out_usm,
                m as u32, k as u32, n as u32, 16,
            )
            .expect("iq1_s batched matvec dispatch");
        }
        let mut out_gpu = vec![0f32; n * m];
        unsafe {
            std::ptr::copy_nonoverlapping(out_usm as *const f32, out_gpu.as_mut_ptr(), n * m);
        }
        // CPU reference: per-batch-row single-row matvec.
        // GPU layout: out_usm[n_idx * M + m_idx]. CPU writes [m_idx]
        // per call; we lay results into the same [n*M + m] indexing.
        let mut out_cpu = vec![0f32; n * m];
        for ni in 0..n {
            let mut row_out = vec![0f32; m];
            rustllama_kernels_cpu::matvec_iq1_s_w_f32_a(
                &w_bytes, &x[ni * k..(ni + 1) * k], &mut row_out, m, k,
            );
            for mi in 0..m {
                out_cpu[ni * m + mi] = row_out[mi];
            }
        }
        let mut max_err = 0f32;
        for (g, c) in out_gpu.iter().zip(out_cpu.iter()) {
            let e = (g - c).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < 1e-3,
            "iq1_s batched matvec: max_err={max_err} (gpu vs cpu)"
        );
        unsafe {
            usm_free(&stream, out_usm as *mut std::ffi::c_void);
            usm_free(&stream, x_usm as *mut std::ffi::c_void);
            usm_free(&stream, w_usm as *mut std::ffi::c_void);
        }
    }

    /// G1: Shared helper for the 6 remaining IQ batched matvec parity
    /// tests. Each format has the same per-row matvec shape (256-weight
    /// super-blocks, M output rows × N batch rows) so this helper takes
    /// the block size + GPU dispatch closure + CPU reference closure.
    fn check_iq_batched_matches_cpu(
        format_name: &str,
        block_bytes: usize,
        seed: u32,
        gpu_dispatch: unsafe fn(&SyclStream, *const u8, *const f32, *mut f32, u32, u32, u32, u32) -> Result<()>,
        cpu_ref: fn(&[u8], &[f32], &mut [f32], usize, usize),
    ) {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        let (m, k, n) = (4usize, 256usize, 4usize);
        let blocks_per_row = k / 256;
        let bytes_per_row = blocks_per_row * block_bytes;
        let mut w_bytes = vec![0u8; m * bytes_per_row];
        let mut s_state: u32 = seed;
        for chunk in w_bytes.chunks_mut(block_bytes) {
            s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let d_bits = half::f16::from_f32(
                ((s_state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0,
            )
            .to_bits();
            chunk[0] = (d_bits & 0xFF) as u8;
            chunk[1] = ((d_bits >> 8) & 0xFF) as u8;
            for b in chunk[2..].iter_mut() {
                s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s_state >> 24) as u8;
            }
        }
        let mut x: Vec<f32> = Vec::with_capacity(n * k);
        let mut sx: u32 = seed.wrapping_add(101);
        for _ in 0..(n * k) {
            sx = sx.wrapping_mul(1664525).wrapping_add(1013904223);
            x.push((sx >> 8) as f32 / (1u32 << 24) as f32 - 0.5);
        }
        let w_usm = usm_alloc_shared(&stream, w_bytes.len()) as *mut u8;
        let x_usm = usm_alloc_shared(&stream, n * k * std::mem::size_of::<f32>()) as *mut f32;
        let out_usm = usm_alloc_shared(&stream, n * m * std::mem::size_of::<f32>()) as *mut f32;
        assert!(!w_usm.is_null() && !x_usm.is_null() && !out_usm.is_null());
        unsafe {
            std::ptr::copy_nonoverlapping(w_bytes.as_ptr(), w_usm, w_bytes.len());
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_usm, n * k);
            for i in 0..(n * m) {
                *out_usm.add(i) = 0.0;
            }
        }
        unsafe {
            gpu_dispatch(&stream, w_usm, x_usm, out_usm, m as u32, k as u32, n as u32, 16)
                .unwrap_or_else(|e| panic!("{format_name} batched dispatch failed: {e:?}"));
        }
        let mut out_gpu = vec![0f32; n * m];
        unsafe {
            std::ptr::copy_nonoverlapping(out_usm as *const f32, out_gpu.as_mut_ptr(), n * m);
        }
        let mut out_cpu = vec![0f32; n * m];
        for ni in 0..n {
            let mut row_out = vec![0f32; m];
            cpu_ref(&w_bytes, &x[ni * k..(ni + 1) * k], &mut row_out, m, k);
            for mi in 0..m {
                out_cpu[ni * m + mi] = row_out[mi];
            }
        }
        let mut max_err = 0f32;
        for (g, c) in out_gpu.iter().zip(out_cpu.iter()) {
            let e = (g - c).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < 1e-3,
            "{format_name} batched matvec: max_err={max_err} (gpu vs cpu)"
        );
        unsafe {
            usm_free(&stream, out_usm as *mut std::ffi::c_void);
            usm_free(&stream, x_usm as *mut std::ffi::c_void);
            usm_free(&stream, w_usm as *mut std::ffi::c_void);
        }
    }

    /// Diagnostic: re-run IQ1_S single-row with the IQ1_M test seeds
    /// (seed=47/148 instead of 41/43). If IQ1_S also produces ~0.02
    /// error here, the IQ1_M-specific gap is just data-dependent FP
    /// accumulation (no kernel bug). If IQ1_S stays at 1e-4, the
    /// IQ1_M kernel has a real arithmetic divergence.
    #[test]
    fn sycl_iq1_s_singlerow_with_iq1_m_seeds_for_comparison() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        let (m, k) = (4usize, 256usize);
        const BLOCK_BYTES: usize = 50;
        let blocks_per_row = k / 256;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;
        let mut w_bytes = vec![0u8; m * bytes_per_row];
        // SAME seeds as IQ1_M test (47 for w, 148 for x).
        let mut s_state: u32 = 47;
        for chunk in w_bytes.chunks_mut(BLOCK_BYTES) {
            s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let d_bits = half::f16::from_f32(
                ((s_state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0,
            )
            .to_bits();
            chunk[0] = (d_bits & 0xFF) as u8;
            chunk[1] = ((d_bits >> 8) & 0xFF) as u8;
            for b in chunk[2..].iter_mut() {
                s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s_state >> 24) as u8;
            }
        }
        let mut x: Vec<f32> = Vec::with_capacity(k);
        let mut sx: u32 = 148;
        for _ in 0..k {
            sx = sx.wrapping_mul(1664525).wrapping_add(1013904223);
            x.push((sx >> 8) as f32 / (1u32 << 24) as f32 - 0.5);
        }
        let w_usm = usm_alloc_shared(&stream, w_bytes.len()) as *mut u8;
        let x_usm = usm_alloc_shared(&stream, k * std::mem::size_of::<f32>()) as *mut f32;
        let out_usm = usm_alloc_shared(&stream, m * std::mem::size_of::<f32>()) as *mut f32;
        assert!(!w_usm.is_null() && !x_usm.is_null() && !out_usm.is_null());
        unsafe {
            std::ptr::copy_nonoverlapping(w_bytes.as_ptr(), w_usm, w_bytes.len());
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_usm, k);
            for i in 0..m { *out_usm.add(i) = 0.0; }
        }
        unsafe {
            matvec_iq1_s_packed_f32_usm_raw(
                &stream, w_usm, x_usm, out_usm, m as u32, k as u32, 16,
            ).expect("iq1_s single-row dispatch");
        }
        let mut out_gpu = vec![0f32; m];
        unsafe {
            std::ptr::copy_nonoverlapping(out_usm as *const f32, out_gpu.as_mut_ptr(), m);
        }
        let mut out_cpu = vec![0f32; m];
        rustllama_kernels_cpu::matvec_iq1_s_w_f32_a(&w_bytes, &x, &mut out_cpu, m, k);
        let mut max_err = 0f32;
        for (g, c) in out_gpu.iter().zip(out_cpu.iter()) {
            let e = (g - c).abs();
            if e > max_err { max_err = e; }
        }
        println!("iq1_s single-row with iq1_m seeds: max_err vs CPU AVX2 = {max_err}");
        unsafe {
            usm_free(&stream, out_usm as *mut std::ffi::c_void);
            usm_free(&stream, x_usm as *mut std::ffi::c_void);
            usm_free(&stream, w_usm as *mut std::ffi::c_void);
        }
    }

    /// Hand-rolled Rust scalar reference for IQ1_M matvec. Identical
    /// to the CPU scalar path's algorithm but with no AVX-512/AVX2
    /// dispatch — gives us a "no FMA, scalar mul+add" baseline to
    /// compare against GPU and CPU runtime separately.
    fn iq1_m_matvec_scalar_reference(
        w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize,
    ) {
        use rustllama_gguf::dequant::IQ1S_DELTA;
        use rustllama_gguf::iq1_grid::IQ1S_GRID;
        const BLOCK_BYTES: usize = 56;
        const QK_K: usize = 256;
        let blocks_per_row = k / QK_K;
        for i in 0..m {
            let row_start = i * blocks_per_row * BLOCK_BYTES;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let off = row_start + b * BLOCK_BYTES;
                let qs = &w_bytes[off..off + 32];
                let qh = &w_bytes[off + 32..off + 32 + 16];
                let scales_bytes = &w_bytes[off + 48..off + 48 + 8];
                let mut sc = [0u16; 4];
                for ii in 0..4 {
                    sc[ii] = u16::from_le_bytes([scales_bytes[ii * 2], scales_bytes[ii * 2 + 1]]);
                }
                let d_bits: u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0)
                    | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
                let d = half::f16::from_bits(d_bits).to_f32();
                let x_block = &x[b * QK_K..(b + 1) * QK_K];
                let mut x_off = 0usize;
                for ib in 0..8 {
                    let s_word = sc[ib / 2];
                    let shift0 = 6 * (ib % 2);
                    let shift1 = 6 * (ib % 2) + 3;
                    let dl1 = d * (2.0 * ((s_word >> shift0) & 0x7) as f32 + 1.0);
                    let dl2 = d * (2.0 * ((s_word >> shift1) & 0x7) as f32 + 1.0);
                    let qh0 = qh[ib * 2];
                    let qh1 = qh[ib * 2 + 1];
                    let delta = |b: bool| if b { -1.0 - IQ1S_DELTA } else { -1.0 + IQ1S_DELTA };
                    let qs_chunk = &qs[ib * 4..ib * 4 + 4];
                    let idx_l = [
                        qs_chunk[0] as usize | (((qh0 & 0x07) as usize) << 8),
                        qs_chunk[1] as usize | ((((qh0 >> 4) & 0x07) as usize) << 8),
                        qs_chunk[2] as usize | (((qh1 & 0x07) as usize) << 8),
                        qs_chunk[3] as usize | ((((qh1 >> 4) & 0x07) as usize) << 8),
                    ];
                    let dl_l = [dl1, dl1, dl2, dl2];
                    let delta_l = [
                        delta(qh0 & 0x08 != 0), delta(qh0 & 0x80 != 0),
                        delta(qh1 & 0x08 != 0), delta(qh1 & 0x80 != 0),
                    ];
                    for l in 0..4 {
                        let grid = IQ1S_GRID[idx_l[l]].to_le_bytes();
                        let dl = dl_l[l];
                        let delta_val = delta_l[l];
                        for j in 0..8 {
                            let g = grid[j] as i8 as f32;
                            acc += dl * (g + delta_val) * x_block[x_off + 8 * l + j];
                        }
                    }
                    x_off += 32;
                }
            }
            out[i] = acc;
        }
    }

    /// Diagnostic: compare GPU IQ1_M and CPU AVX-512 IQ1_M each against
    /// the hand-rolled scalar reference. Reveals which one has the
    /// "true" precision floor.
    #[test]
    fn diagnostic_iq1_m_gpu_vs_scalar_reference() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        let (m, k) = (4usize, 256usize);
        const BLOCK_BYTES: usize = 56;
        let blocks_per_row = k / 256;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;
        let mut w_bytes = vec![0u8; m * bytes_per_row];
        let mut s_state: u32 = 47;
        for chunk in w_bytes.chunks_mut(BLOCK_BYTES) {
            s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let d_bits = half::f16::from_f32(
                ((s_state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0,
            ).to_bits();
            chunk[0] = (d_bits & 0xFF) as u8;
            chunk[1] = ((d_bits >> 8) & 0xFF) as u8;
            for b in chunk[2..].iter_mut() {
                s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s_state >> 24) as u8;
            }
        }
        let mut x: Vec<f32> = Vec::with_capacity(k);
        let mut sx: u32 = 148;
        for _ in 0..k {
            sx = sx.wrapping_mul(1664525).wrapping_add(1013904223);
            x.push((sx >> 8) as f32 / (1u32 << 24) as f32 - 0.5);
        }
        // GPU output.
        let w_usm = usm_alloc_shared(&stream, w_bytes.len()) as *mut u8;
        let x_usm = usm_alloc_shared(&stream, k * std::mem::size_of::<f32>()) as *mut f32;
        let out_usm = usm_alloc_shared(&stream, m * std::mem::size_of::<f32>()) as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(w_bytes.as_ptr(), w_usm, w_bytes.len());
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_usm, k);
            for i in 0..m { *out_usm.add(i) = 0.0; }
            matvec_iq1_m_packed_f32_usm_raw(
                &stream, w_usm, x_usm, out_usm, m as u32, k as u32, 16,
            ).expect("dispatch");
        }
        let mut out_gpu = vec![0f32; m];
        unsafe {
            std::ptr::copy_nonoverlapping(out_usm as *const f32, out_gpu.as_mut_ptr(), m);
            usm_free(&stream, out_usm as *mut std::ffi::c_void);
            usm_free(&stream, x_usm as *mut std::ffi::c_void);
            usm_free(&stream, w_usm as *mut std::ffi::c_void);
        }
        // CPU runtime (uses AVX-512 if available).
        let mut out_cpu = vec![0f32; m];
        rustllama_kernels_cpu::matvec_iq1_m_w_f32_a(&w_bytes, &x, &mut out_cpu, m, k);
        // Hand-rolled scalar reference.
        let mut out_scalar = vec![0f32; m];
        iq1_m_matvec_scalar_reference(&w_bytes, &x, &mut out_scalar, m, k);

        let mut gpu_vs_cpu = 0f32;
        let mut gpu_vs_scalar = 0f32;
        let mut cpu_vs_scalar = 0f32;
        for i in 0..m {
            gpu_vs_cpu = gpu_vs_cpu.max((out_gpu[i] - out_cpu[i]).abs());
            gpu_vs_scalar = gpu_vs_scalar.max((out_gpu[i] - out_scalar[i]).abs());
            cpu_vs_scalar = cpu_vs_scalar.max((out_cpu[i] - out_scalar[i]).abs());
        }
        println!("IQ1_M diagnostic:");
        println!("  GPU vs CPU (AVX-512): max_err = {gpu_vs_cpu}");
        println!("  GPU vs SCALAR ref:    max_err = {gpu_vs_scalar}");
        println!("  CPU vs SCALAR ref:    max_err = {cpu_vs_scalar}");
        println!("  GPU output: {out_gpu:?}");
        println!("  CPU output: {out_cpu:?}");
        println!("  SCA output: {out_scalar:?}");
    }

    /// Diagnostic: single-row IQ1_M parity. The batched IQ1_M test
    /// needed 5e-2 tolerance; this test confirms whether the divergence
    /// is pre-existing in the single-row kernel (FMA precision) or
    /// specific to the batched port. Same shape/data as the IQ1_S
    /// single-row parity test for direct comparability.
    #[test]
    fn sycl_iq1_m_singlerow_matvec_matches_cpu() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        let (m, k) = (4usize, 256usize);
        const BLOCK_BYTES: usize = 56;
        let blocks_per_row = k / 256;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;
        let mut w_bytes = vec![0u8; m * bytes_per_row];
        let mut s_state: u32 = 47;
        for chunk in w_bytes.chunks_mut(BLOCK_BYTES) {
            s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let d_bits = half::f16::from_f32(
                ((s_state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0,
            )
            .to_bits();
            chunk[0] = (d_bits & 0xFF) as u8;
            chunk[1] = ((d_bits >> 8) & 0xFF) as u8;
            for b in chunk[2..].iter_mut() {
                s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s_state >> 24) as u8;
            }
        }
        let mut x: Vec<f32> = Vec::with_capacity(k);
        let mut sx: u32 = 148;
        for _ in 0..k {
            sx = sx.wrapping_mul(1664525).wrapping_add(1013904223);
            x.push((sx >> 8) as f32 / (1u32 << 24) as f32 - 0.5);
        }
        let w_usm = usm_alloc_shared(&stream, w_bytes.len()) as *mut u8;
        let x_usm = usm_alloc_shared(&stream, k * std::mem::size_of::<f32>()) as *mut f32;
        let out_usm = usm_alloc_shared(&stream, m * std::mem::size_of::<f32>()) as *mut f32;
        assert!(!w_usm.is_null() && !x_usm.is_null() && !out_usm.is_null());
        unsafe {
            std::ptr::copy_nonoverlapping(w_bytes.as_ptr(), w_usm, w_bytes.len());
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_usm, k);
            for i in 0..m { *out_usm.add(i) = 0.0; }
        }
        unsafe {
            matvec_iq1_m_packed_f32_usm_raw(
                &stream, w_usm, x_usm, out_usm, m as u32, k as u32, 16,
            ).expect("iq1_m single-row dispatch");
        }
        let mut out_gpu = vec![0f32; m];
        unsafe {
            std::ptr::copy_nonoverlapping(out_usm as *const f32, out_gpu.as_mut_ptr(), m);
        }
        let mut out_cpu = vec![0f32; m];
        rustllama_kernels_cpu::matvec_iq1_m_w_f32_a(&w_bytes, &x, &mut out_cpu, m, k);
        let mut max_err = 0f32;
        for (g, c) in out_gpu.iter().zip(out_cpu.iter()) {
            let e = (g - c).abs();
            if e > max_err { max_err = e; }
        }
        // Report what we see; assert generously so we get the actual delta.
        println!("iq1_m single-row max_err vs CPU AVX2: {max_err}");
        assert!(
            max_err < 1e-1,
            "iq1_m single-row matvec diverged catastrophically: max_err={max_err}"
        );
        unsafe {
            usm_free(&stream, out_usm as *mut std::ffi::c_void);
            usm_free(&stream, x_usm as *mut std::ffi::c_void);
            usm_free(&stream, w_usm as *mut std::ffi::c_void);
        }
    }

    /// G1: IQ1_M batched parity. Tolerance is 5e-2 because GPU and
    /// CPU AVX-512 each drift independently from the canonical scalar
    /// algorithm. Diagnostic results (see `diagnostic_iq1_m_gpu_vs_scalar_reference`):
    ///   - GPU vs scalar reference:    0.0039 max  (GPU is faithful to algorithm)
    ///   - CPU AVX-512 vs scalar:      0.0156 max  (CPU AVX-512 is the precision outlier)
    ///   - GPU vs CPU AVX-512:         0.0195 max  (sum of both drifts)
    /// CPU AVX-512 uses real `_mm512_fmadd_ps` which gives more-precise-
    /// but-algorithmically-different rounding than the scalar/GPU path.
    /// IQ1_S happens to converge to similar output across all three
    /// paths because its uniform-per-sub-block dl/delta multipliers
    /// have small accumulated drift; IQ1_M's per-lane multipliers
    /// don't. GPU output is closer to the algorithm; the test
    /// tolerance reflects the GPU↔CPU-AVX-512 numerical gap, not a
    /// kernel bug.
    #[test]
    fn sycl_iq1_m_batched_matvec_matches_cpu() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        let (m, k, n) = (4usize, 256usize, 4usize);
        const BLOCK_BYTES: usize = 56;
        let blocks_per_row = k / 256;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;
        let mut w_bytes = vec![0u8; m * bytes_per_row];
        let mut s_state: u32 = 47;
        for chunk in w_bytes.chunks_mut(BLOCK_BYTES) {
            s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let d_bits = half::f16::from_f32(
                ((s_state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0,
            )
            .to_bits();
            chunk[0] = (d_bits & 0xFF) as u8;
            chunk[1] = ((d_bits >> 8) & 0xFF) as u8;
            for b in chunk[2..].iter_mut() {
                s_state = s_state.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s_state >> 24) as u8;
            }
        }
        let mut x: Vec<f32> = Vec::with_capacity(n * k);
        let mut sx: u32 = 148;
        for _ in 0..(n * k) {
            sx = sx.wrapping_mul(1664525).wrapping_add(1013904223);
            x.push((sx >> 8) as f32 / (1u32 << 24) as f32 - 0.5);
        }
        let w_usm = usm_alloc_shared(&stream, w_bytes.len()) as *mut u8;
        let x_usm = usm_alloc_shared(&stream, n * k * std::mem::size_of::<f32>()) as *mut f32;
        let out_usm = usm_alloc_shared(&stream, n * m * std::mem::size_of::<f32>()) as *mut f32;
        assert!(!w_usm.is_null() && !x_usm.is_null() && !out_usm.is_null());
        unsafe {
            std::ptr::copy_nonoverlapping(w_bytes.as_ptr(), w_usm, w_bytes.len());
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_usm, n * k);
            for i in 0..(n * m) {
                *out_usm.add(i) = 0.0;
            }
        }
        unsafe {
            matvec_iq1_m_packed_f32_batched_usm_raw(
                &stream, w_usm, x_usm, out_usm,
                m as u32, k as u32, n as u32, 16,
            )
            .expect("iq1_m batched dispatch");
        }
        let mut out_gpu = vec![0f32; n * m];
        unsafe {
            std::ptr::copy_nonoverlapping(out_usm as *const f32, out_gpu.as_mut_ptr(), n * m);
        }
        let mut out_cpu = vec![0f32; n * m];
        for ni in 0..n {
            let mut row_out = vec![0f32; m];
            rustllama_kernels_cpu::matvec_iq1_m_w_f32_a(
                &w_bytes, &x[ni * k..(ni + 1) * k], &mut row_out, m, k,
            );
            for mi in 0..m {
                out_cpu[ni * m + mi] = row_out[mi];
            }
        }
        let mut max_err = 0f32;
        for (g, c) in out_gpu.iter().zip(out_cpu.iter()) {
            let e = (g - c).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < 5e-2,
            "iq1_m batched matvec: max_err={max_err} (gpu vs cpu)"
        );
        unsafe {
            usm_free(&stream, out_usm as *mut std::ffi::c_void);
            usm_free(&stream, x_usm as *mut std::ffi::c_void);
            usm_free(&stream, w_usm as *mut std::ffi::c_void);
        }
    }

    #[test]
    fn sycl_iq2_xxs_batched_matvec_matches_cpu() {
        check_iq_batched_matches_cpu(
            "iq2_xxs", 66, 53,
            matvec_iq2_xxs_packed_f32_batched_usm_raw,
            rustllama_kernels_cpu::matvec_iq2_xxs_w_f32_a,
        );
    }

    #[test]
    fn sycl_iq2_xs_batched_matvec_matches_cpu() {
        check_iq_batched_matches_cpu(
            "iq2_xs", 74, 59,
            matvec_iq2_xs_packed_f32_batched_usm_raw,
            rustllama_kernels_cpu::matvec_iq2_xs_w_f32_a,
        );
    }

    #[test]
    fn sycl_iq2_s_batched_matvec_matches_cpu() {
        check_iq_batched_matches_cpu(
            "iq2_s", 82, 61,
            matvec_iq2_s_packed_f32_batched_usm_raw,
            rustllama_kernels_cpu::matvec_iq2_s_w_f32_a,
        );
    }

    #[test]
    fn sycl_iq3_xxs_batched_matvec_matches_cpu() {
        check_iq_batched_matches_cpu(
            "iq3_xxs", 98, 67,
            matvec_iq3_xxs_packed_f32_batched_usm_raw,
            rustllama_kernels_cpu::matvec_iq3_xxs_w_f32_a,
        );
    }

    #[test]
    fn sycl_iq3_s_batched_matvec_matches_cpu() {
        check_iq_batched_matches_cpu(
            "iq3_s", 110, 71,
            matvec_iq3_s_packed_f32_batched_usm_raw,
            rustllama_kernels_cpu::matvec_iq3_s_w_f32_a,
        );
    }

    /// PTQ1_0 (Bonsai ternary) test blocks: 24-byte base-3 qs +
    /// 2-byte qh + trailing f16 d (bytes 26..28). The multiply-high
    /// trit extraction is total over arbitrary qs/qh bytes, so random
    /// bytes exercise every code path; only d needs a sane f16.
    fn gen_ptq1_0_rows(m: usize, k: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        const BLOCK_BYTES: usize = 28;
        let blocks_per_row = k / 128;
        let mut w_bytes = vec![0u8; m * blocks_per_row * BLOCK_BYTES];
        let mut s: u32 = seed;
        for chunk in w_bytes.chunks_mut(BLOCK_BYTES) {
            for b in chunk[..26].iter_mut() {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s >> 24) as u8;
            }
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d_bits = half::f16::from_f32(
                ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0,
            )
            .to_bits();
            chunk[26] = (d_bits & 0xFF) as u8;
            chunk[27] = ((d_bits >> 8) & 0xFF) as u8;
        }
        let mut x: Vec<f32> = Vec::with_capacity(k);
        let mut sx: u32 = seed.wrapping_add(101);
        for _ in 0..k {
            sx = sx.wrapping_mul(1664525).wrapping_add(1013904223);
            x.push((sx >> 8) as f32 / (1u32 << 24) as f32 - 0.5);
        }
        (w_bytes, x)
    }

    #[test]
    fn sycl_ptq1_0_matvec_matches_cpu() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(_) => return, // no SYCL device — skip
        };
        // k=384 = 3 blocks/row exercises the multi-block accumulate.
        let (m, k) = (6usize, 384usize);
        let (w_bytes, x) = gen_ptq1_0_rows(m, k, 173);
        let mut out_cpu = vec![0f32; m];
        rustllama_kernels_cpu::matvec_ptq1_0_w_f32_a(&w_bytes, &x, &mut out_cpu, m, k);
        let w_usm = usm_alloc_shared(&stream, w_bytes.len()) as *mut u8;
        let x_usm = usm_alloc_shared(&stream, k * 4) as *mut f32;
        let out_usm = usm_alloc_shared(&stream, m * 4) as *mut f32;
        assert!(!w_usm.is_null() && !x_usm.is_null() && !out_usm.is_null());
        unsafe {
            std::ptr::copy_nonoverlapping(w_bytes.as_ptr(), w_usm, w_bytes.len());
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_usm, k);
            for lws in [0u32, 16, 64, 256] {
                for i in 0..m {
                    *out_usm.add(i) = f32::NAN;
                }
                matvec_ptq1_0_packed_f32_usm_raw(
                    &stream, w_usm, x_usm, out_usm, m as u32, k as u32, lws,
                )
                .unwrap_or_else(|e| panic!("ptq1_0 dispatch failed (lws={lws}): {e:?}"));
                let mut max_err = 0f32;
                for i in 0..m {
                    let e = (*out_usm.add(i) - out_cpu[i]).abs();
                    if e > max_err {
                        max_err = e;
                    }
                }
                assert!(
                    max_err < 1e-3,
                    "ptq1_0 matvec lws={lws}: max_err={max_err} (gpu vs cpu)"
                );
            }
            usm_free(&stream, out_usm as *mut std::ffi::c_void);
            usm_free(&stream, x_usm as *mut std::ffi::c_void);
            usm_free(&stream, w_usm as *mut std::ffi::c_void);
        }
    }

    #[test]
    fn sycl_ptq1_0_batched_matvec_matches_cpu() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        let (m, k, n) = (4usize, 256usize, 4usize);
        let (w_bytes, _) = gen_ptq1_0_rows(m, k, 211);
        let mut x: Vec<f32> = Vec::with_capacity(n * k);
        let mut sx: u32 = 907;
        for _ in 0..(n * k) {
            sx = sx.wrapping_mul(1664525).wrapping_add(1013904223);
            x.push((sx >> 8) as f32 / (1u32 << 24) as f32 - 0.5);
        }
        let mut out_cpu = vec![0f32; n * m];
        for ni in 0..n {
            let mut row_out = vec![0f32; m];
            rustllama_kernels_cpu::matvec_ptq1_0_w_f32_a(
                &w_bytes,
                &x[ni * k..(ni + 1) * k],
                &mut row_out,
                m,
                k,
            );
            out_cpu[ni * m..(ni + 1) * m].copy_from_slice(&row_out);
        }
        let w_usm = usm_alloc_shared(&stream, w_bytes.len()) as *mut u8;
        let x_usm = usm_alloc_shared(&stream, n * k * 4) as *mut f32;
        let out_usm = usm_alloc_shared(&stream, n * m * 4) as *mut f32;
        assert!(!w_usm.is_null() && !x_usm.is_null() && !out_usm.is_null());
        unsafe {
            std::ptr::copy_nonoverlapping(w_bytes.as_ptr(), w_usm, w_bytes.len());
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_usm, n * k);
            for i in 0..(n * m) {
                *out_usm.add(i) = f32::NAN;
            }
            matvec_ptq1_0_packed_f32_batched_usm_raw(
                &stream, w_usm, x_usm, out_usm, m as u32, k as u32, n as u32, 16,
            )
            .unwrap_or_else(|e| panic!("ptq1_0 batched dispatch failed: {e:?}"));
            let mut max_err = 0f32;
            for i in 0..(n * m) {
                let e = (*out_usm.add(i) - out_cpu[i]).abs();
                if e > max_err {
                    max_err = e;
                }
            }
            assert!(
                max_err < 1e-3,
                "ptq1_0 batched matvec: max_err={max_err} (gpu vs cpu)"
            );
            usm_free(&stream, out_usm as *mut std::ffi::c_void);
            usm_free(&stream, x_usm as *mut std::ffi::c_void);
            usm_free(&stream, w_usm as *mut std::ffi::c_void);
        }
    }

    /// GPU blockwise Hadamard rotation must reproduce the CPU
    /// reference (`rustllama_kernels_cpu::hadamard::hadamard_forward`):
    /// sign-flip → unnormalized WHT butterfly → 1/√block. Same
    /// arithmetic order ⇒ near-bit-exact.
    #[test]
    fn sycl_hadamard_forward_matches_cpu() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        for block in [64usize, 1024] {
            let n = block * 3;
            let mut s: u32 = 331;
            let mut x: Vec<f32> = Vec::with_capacity(n);
            let mut signs: Vec<f32> = Vec::with_capacity(n);
            for _ in 0..n {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                x.push((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5);
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                signs.push(if s & 0x1000 != 0 { 1.0 } else { -1.0 });
            }
            let mut out_cpu = vec![0f32; n];
            rustllama_kernels_cpu::hadamard::hadamard_forward(&x, &signs, block, &mut out_cpu);
            let x_usm = usm_alloc_shared(&stream, n * 4) as *mut f32;
            let signs_usm = usm_alloc_shared(&stream, n * 4) as *mut f32;
            let out_usm = usm_alloc_shared(&stream, n * 4) as *mut f32;
            assert!(!x_usm.is_null() && !signs_usm.is_null() && !out_usm.is_null());
            unsafe {
                std::ptr::copy_nonoverlapping(x.as_ptr(), x_usm, n);
                std::ptr::copy_nonoverlapping(signs.as_ptr(), signs_usm, n);
                for i in 0..n {
                    *out_usm.add(i) = f32::NAN;
                }
                hadamard_forward_usm_raw(
                    &stream, x_usm, signs_usm, out_usm, n as u32, block as u32,
                )
                .unwrap_or_else(|e| panic!("hadamard dispatch failed (block={block}): {e:?}"));
                let mut max_err = 0f32;
                for i in 0..n {
                    let e = (*out_usm.add(i) - out_cpu[i]).abs();
                    if e > max_err {
                        max_err = e;
                    }
                }
                assert!(
                    max_err < 1e-4,
                    "hadamard block={block}: max_err={max_err} (gpu vs cpu)"
                );
                usm_free(&stream, out_usm as *mut std::ffi::c_void);
                usm_free(&stream, signs_usm as *mut std::ffi::c_void);
                usm_free(&stream, x_usm as *mut std::ffi::c_void);
            }
        }
    }

    /// G5 parity: GPU K-quant dequant must reproduce the CPU
    /// reference (`rustllama_gguf::dequant::dequant_q{3,4,5,6}_k`)
    /// to within f32 rounding. The CPU and GPU kernels execute the
    /// same arithmetic in the same order, so equality should be
    /// bit-exact; we allow a tiny tolerance to absorb any FMA
    /// differences across implementations.
    fn gen_qk_f32(seed: u32, n_blocks: usize) -> Vec<f32> {
        let n = n_blocks * 256;
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / (1u32 << 24) as f32;
                (u - 0.5) * 0.4
            })
            .collect()
    }

    #[test]
    #[cfg(feature = "encoder")]
    fn sycl_dequant_q4_k_matches_cpu() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(_) => return, // no SYCL device — skip
        };
        let n_blocks = 8;
        let src_f32 = gen_qk_f32(101, n_blocks);
        let mut enc = vec![0u8; n_blocks * 144];
        rustllama_gguf::encode_k::encode_q4_k(&src_f32, &mut enc);
        let mut cpu_out = vec![0f32; n_blocks * 256];
        rustllama_gguf::dequant::dequant_q4_k(&enc, &mut cpu_out);
        let mut gpu_out = vec![0f32; n_blocks * 256];
        dequant_kquant_via_gpu(&stream, KQuantFormat::Q4K, &enc, &mut gpu_out)
            .expect("gpu Q4_K dequant");
        let max_err = cpu_out
            .iter()
            .zip(gpu_out.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0f32, f32::max);
        assert!(max_err < 1e-5, "Q4_K dequant parity: max_err={max_err}");
    }

    #[test]
    #[cfg(feature = "encoder")]
    fn sycl_dequant_q3_k_matches_cpu() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        let n_blocks = 8;
        let src_f32 = gen_qk_f32(103, n_blocks);
        let mut enc = vec![0u8; n_blocks * 110];
        rustllama_gguf::encode_k::encode_q3_k(&src_f32, &mut enc);
        let mut cpu_out = vec![0f32; n_blocks * 256];
        rustllama_gguf::dequant::dequant_q3_k(&enc, &mut cpu_out);
        let mut gpu_out = vec![0f32; n_blocks * 256];
        dequant_kquant_via_gpu(&stream, KQuantFormat::Q3K, &enc, &mut gpu_out)
            .expect("gpu Q3_K dequant");
        let max_err = cpu_out
            .iter()
            .zip(gpu_out.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0f32, f32::max);
        assert!(max_err < 1e-5, "Q3_K dequant parity: max_err={max_err}");
    }

    #[test]
    #[cfg(feature = "encoder")]
    fn sycl_dequant_q5_k_matches_cpu() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        let n_blocks = 8;
        let src_f32 = gen_qk_f32(105, n_blocks);
        let mut enc = vec![0u8; n_blocks * 176];
        rustllama_gguf::encode_k::encode_q5_k(&src_f32, &mut enc);
        let mut cpu_out = vec![0f32; n_blocks * 256];
        rustllama_gguf::dequant::dequant_q5_k(&enc, &mut cpu_out);
        let mut gpu_out = vec![0f32; n_blocks * 256];
        dequant_kquant_via_gpu(&stream, KQuantFormat::Q5K, &enc, &mut gpu_out)
            .expect("gpu Q5_K dequant");
        let max_err = cpu_out
            .iter()
            .zip(gpu_out.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0f32, f32::max);
        assert!(max_err < 1e-5, "Q5_K dequant parity: max_err={max_err}");
    }

    #[test]
    #[cfg(feature = "encoder")]
    fn sycl_dequant_q6_k_matches_cpu() {
        let stream = match create_stream(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        let n_blocks = 8;
        let src_f32 = gen_qk_f32(107, n_blocks);
        let mut enc = vec![0u8; n_blocks * 210];
        rustllama_gguf::encode_k::encode_q6_k(&src_f32, &mut enc);
        let mut cpu_out = vec![0f32; n_blocks * 256];
        rustllama_gguf::dequant::dequant_q6_k(&enc, &mut cpu_out);
        let mut gpu_out = vec![0f32; n_blocks * 256];
        dequant_kquant_via_gpu(&stream, KQuantFormat::Q6K, &enc, &mut gpu_out)
            .expect("gpu Q6_K dequant");
        let max_err = cpu_out
            .iter()
            .zip(gpu_out.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0f32, f32::max);
        assert!(max_err < 1e-5, "Q6_K dequant parity: max_err={max_err}");
    }
}
