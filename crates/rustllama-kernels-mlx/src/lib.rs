//! Apple-Metal / MLX compute kernels for rustllama (the macOS GPU backend).
//!
//! This crate is real-only (no mock/feature gates) and is a first-class
//! peer of the SYCL and CUDA kernel crates: the GPU kernel layer is ALWAYS
//! compiled in, and the backend is selected at engine STARTUP by device
//! detection — never by a cargo feature. The MLX backend is the 4th device
//! tier (beside SYCL, CUDA and CPU); the tuner/placement layer decides
//! actual use, weighted by measured performance and (unified) memory.
//!
//! REAL vs INERT, by target:
//!   * `aarch64-apple-darwin` (Apple Silicon) — the REAL path: a genuine
//!     Metal GPU driven through Apple's MLX (linked via the official mlx-c
//!     C API + hand-written Metal shaders in `mlx/rsl_mlx.mm` / `.metal`).
//!   * Everything else — Windows, Linux, AND Intel macOS (`x86_64-apple-
//!     darwin`, which has no MLX Metal GPU path) — an INERT no-op stub so
//!     the crate links everywhere. `device_count()` returns 0 there → the
//!     engine sees no MLX device and runs on CPU / SYCL / CUDA. build.rs
//!     synthesizes the stub from `mlx/rsl_mlx.def`.
//!
//! The Rust surface below is COMPLETE and compiles unchanged on every
//! target — it is pure FFI (extern decls + safe wrappers). On the stub
//! build the `rsl_mlx_*` symbols resolve to no-ops (return 0), so
//! `device_count()` reports zero devices and the whole backend is inert.
//! The real kernel math lives behind the same FFI names and lands in
//! Phase 1 (`mlx/rsl_mlx.mm` / `.metal`), validated against the CPU
//! reference by the same parity discipline as SYCL/CUDA.
//!
//! APPLE UNIFIED MEMORY (the key architectural difference from CUDA): on
//! Apple Silicon the CPU and GPU share one physical pool, so a Metal
//! `MTLBuffer` with `StorageModeShared` is host-addressable — much closer
//! to SYCL shared USM than to CUDA's separate device memory. The device-
//! resident weight cache below keeps CUDA's shape (upload-once, keyed by
//! host pointer) so dispatch is identical; Phase 1 may collapse the H2D
//! "upload" into a zero-copy wrap of the existing GGUF weight bytes. That
//! optimization is localized to `MlxDeviceBuffer` / the `.mm` allocator.

use std::collections::HashMap;
use std::os::raw::{c_char, c_int, c_void};

/// T4: FNV-1a-64 hash (hex) of this crate's kernel sources, computed by
/// `build.rs` and injected via `cargo:rustc-env=RSL_MLX_KERNEL_HASH`. Folded
/// (with the SYCL/CUDA twins) into `rustllama_tuner::kernel_build_id()` so a
/// Metal/MLX kernel edit re-triggers the first-load verdict self-heal.
pub const KERNEL_HASH: &str = env!("RSL_MLX_KERNEL_HASH");

/// Opaque handle to the C `rsl_mlx_stream` (device + Metal command queue /
/// MLX stream).
#[repr(C)]
struct RslMlxStreamRaw {
    _private: [u8; 0],
}

#[derive(Debug, thiserror::Error)]
pub enum MlxError {
    #[error("MLX device {0} not found")]
    NoSuchDevice(u32),
    #[error("MLX kernel error (code {0})")]
    Kernel(i32),
}

/// One Metal/MLX device's properties.
#[derive(Debug, Clone)]
pub struct MlxDeviceInfo {
    pub name: String,
    pub total_mem_bytes: u64,
    /// Stable, driver-invariant identifier: Metal's `MTLDevice.registryID`
    /// (a `uint64_t` from the IO registry). Metal has no CUDA-style 16-byte
    /// device UUID, so this is the primary fingerprint key on Apple.
    pub registry_id: u64,
    /// 16-byte device UUID for `system_fingerprint()` parity with the
    /// SYCL/CUDA backends. Phase 1 synthesizes it (e.g. `registry_id`
    /// spread over 8 bytes || a name hash); zeroed until then.
    pub uuid: [u8; 16],
}

extern "C" {
    fn rsl_mlx_device_count() -> c_int;
    fn rsl_mlx_device_info(
        idx: c_int,
        name: *mut c_char,
        name_cap: c_int,
        total_mem: *mut u64,
        registry_id: *mut u64,
        uuid: *mut u8,
    ) -> c_int;
    fn rsl_mlx_rmsnorm_f32(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        n_rows: c_int,
        d: c_int,
        eps: f32,
    ) -> c_int;
    fn rsl_mlx_matvec_f32(
        w: *const f32,
        x: *const f32,
        out: *mut f32,
        m_rows: c_int,
        k_dim: c_int,
    ) -> c_int;
    // P17: Gated-DeltaNet chunked-prefill recurrence over all V-heads (host-copy).
    fn rsl_mlx_delta_net_chunk_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        out: *mut f32,
        n_heads: c_int,
        seq_len: c_int,
        head_qk_dim: c_int,
        head_v_dim: c_int,
        chunk_len: c_int,
    ) -> c_int;

    // Device-resident path.
    fn rsl_mlx_stream_create(device_index: c_int) -> *mut RslMlxStreamRaw;
    fn rsl_mlx_stream_destroy(s: *mut RslMlxStreamRaw);
    fn rsl_mlx_malloc_from_host(
        s: *mut RslMlxStreamRaw,
        src: *const c_void,
        n_bytes: u64,
    ) -> *mut c_void;
    fn rsl_mlx_malloc_device(s: *mut RslMlxStreamRaw, n_bytes: u64) -> *mut c_void;
    fn rsl_mlx_free(s: *mut RslMlxStreamRaw, dev_ptr: *mut c_void);
    fn rsl_mlx_memcpy_h2d(
        s: *mut RslMlxStreamRaw,
        dst_dev: *mut c_void,
        src_host: *const c_void,
        n_bytes: u64,
    ) -> c_int;
    fn rsl_mlx_memcpy_d2h(
        s: *mut RslMlxStreamRaw,
        dst_host: *mut c_void,
        src_dev: *const c_void,
        n_bytes: u64,
    ) -> c_int;
    fn rsl_mlx_matvec_ptq1_0_packed_f32(
        s: *mut RslMlxStreamRaw,
        w_bytes_dev: *const c_void,
        x_dev: *const f32,
        out_dev: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_ptq1_0_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w_bytes_dev: *const c_void,
        x_dev: *const f32,
        out_dev: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_hadamard_forward(
        s: *mut RslMlxStreamRaw,
        x_dev: *const f32,
        signs_dev: *const f32,
        out_dev: *mut f32,
        n_elems: c_int,
        block: c_int,
    ) -> c_int;
    // B2 (Kindi rmsfold): device-resident RMSNorm on device pointers (the
    // host-slice `rsl_mlx_rmsnorm_f32` can't feed the next GPU stage). Dispatches
    // the SAME `rsl_mlx_rmsnorm_f32_kernel` on the caller's device buffers — no
    // internal alloc/copy — so the rmsnorm output stays resident for the
    // Hadamard + matvec that follow.
    fn rsl_mlx_rmsnorm_dev(
        s: *mut RslMlxStreamRaw,
        x_dev: *const f32,
        w_dev: *const f32,
        y_dev: *mut f32,
        n_rows: c_int,
        d: c_int,
        eps: f32,
    ) -> c_int;
    fn rsl_mlx_matvec_q8_0_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q8_0_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q4_k_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q4_k_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q6_k_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q6_k_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;

    // Additional packed matvecs (K-quants Q5_K/Q2_K/Q8_K/Q3_K, legacy quants
    // Q4_0/Q5_0/Q4_1/Q5_1, IQ family, NVFP4/MXFP4/6/8, PQ2_0). One GPU
    // thread(group) per output row.
    fn rsl_mlx_matvec_q5_k_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q5_k_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q2_k_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q2_k_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q8_k_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q8_k_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q4_0_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q4_0_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q5_0_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q5_0_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q4_1_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q4_1_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q5_1_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q5_1_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq4_nl_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq4_nl_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq4_xs_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq4_xs_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq2_xxs_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq2_xxs_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq2_xs_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq2_xs_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq2_s_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq2_s_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq3_xxs_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq3_xxs_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq3_s_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq3_s_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq1_s_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq1_s_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq1_m_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq1_m_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_nvfp4_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_nvfp4_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    // OCP Microscaling FP4/FP6/FP8. 32-elem blocks + trailing E8M0 scale
    // byte; K%32==0. MXFP4: 17 B, MXFP6: 25 B, MXFP8: 33 B per block.
    fn rsl_mlx_matvec_mxfp4_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_mxfp4_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_mxfp6_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_mxfp6_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_mxfp8_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_mxfp8_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    // Q3_K + PQ2_0. Q3_K: 110 B/256, K%256==0. PQ2_0: 34 B/128, K%128==0.
    fn rsl_mlx_matvec_q3_k_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q3_k_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_pq2_0_packed_f32(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_pq2_0_packed_f32_batched(
        s: *mut RslMlxStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;

    // Fused gate+up matvec (decode): one dispatch computes gate_out + up_out.
    // gw/uw = packed gate/up weights [M,K]; gout/uout = [M] device ptrs.
    fn rsl_mlx_matvec_q8_0_packed_f32_gate_up_fused(
        s: *mut RslMlxStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q4_k_packed_f32_gate_up_fused(
        s: *mut RslMlxStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q6_k_packed_f32_gate_up_fused(
        s: *mut RslMlxStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_q5_k_packed_f32_gate_up_fused(
        s: *mut RslMlxStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq4_nl_packed_f32_gate_up_fused(
        s: *mut RslMlxStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq4_xs_packed_f32_gate_up_fused(
        s: *mut RslMlxStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq1_s_packed_f32_gate_up_fused(
        s: *mut RslMlxStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq1_m_packed_f32_gate_up_fused(
        s: *mut RslMlxStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq2_xxs_packed_f32_gate_up_fused(
        s: *mut RslMlxStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq2_xs_packed_f32_gate_up_fused(
        s: *mut RslMlxStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq2_s_packed_f32_gate_up_fused(
        s: *mut RslMlxStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq3_xxs_packed_f32_gate_up_fused(
        s: *mut RslMlxStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_iq3_s_packed_f32_gate_up_fused(
        s: *mut RslMlxStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_mlx_matvec_ptq1_0_packed_f32_gate_up_fused(
        s: *mut RslMlxStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;

    // Forward-pass primitives (device-resident, f32).
    fn rsl_mlx_add_rmsnorm_f32(
        s: *mut RslMlxStreamRaw,
        hidden: *mut f32,
        branch: *const f32,
        w: *const f32,
        y_norm: *mut f32,
        n_rows: c_int,
        d: c_int,
        eps: f32,
    ) -> c_int;
    fn rsl_mlx_rope_f32(
        s: *mut RslMlxStreamRaw,
        qk: *mut f32,
        n_heads: c_int,
        head_dim: c_int,
        pos: c_int,
        inv_freq: *const f32,
    ) -> c_int;
    fn rsl_mlx_silu_mul_f32(
        s: *mut RslMlxStreamRaw,
        x: *const f32,
        y: *const f32,
        out: *mut f32,
        n: c_int,
    ) -> c_int;
    // P6-fusions (write-blind mirrors of the SYCL fused f32 kernels).
    fn rsl_mlx_rmsnorm_matvec_f32(
        s: *mut RslMlxStreamRaw,
        x: *const f32,
        wnorm: *const f32,
        wmat: *const f32,
        y: *mut f32,
        n_rows: c_int,
        k: c_int,
        m: c_int,
        eps: f32,
    ) -> c_int;
    fn rsl_mlx_down_residual_f32(
        s: *mut RslMlxStreamRaw,
        ff: *const f32,
        wmat: *const f32,
        resid: *const f32,
        y: *mut f32,
        n_rows: c_int,
        k: c_int,
        m: c_int,
    ) -> c_int;
    fn rsl_mlx_embedding_lookup_f32(
        s: *mut RslMlxStreamRaw,
        table: *const f32,
        ids: *const c_int,
        out: *mut f32,
        n_ids: c_int,
        d: c_int,
    ) -> c_int;
    fn rsl_mlx_flash_attn_decode_f32(
        s: *mut RslMlxStreamRaw,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        max_ctx: c_int,
        kv_len: c_int,
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;
    // GPU paged attention (Phase F): block-table flash-decode against the
    // whole-pool F32 device mirror. `block_table` = [ceil(kv_len/page_size)] i32.
    fn rsl_mlx_flash_attn_decode_paged_f32(
        s: *mut RslMlxStreamRaw,
        q: *const f32,
        kv_pool: *const f32,
        block_table: *const c_int,
        out: *mut f32,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        layer_idx: c_int,
        n_layers: c_int,
        page_size: c_int,
        kv_len: c_int,
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;
    fn rsl_mlx_flash_attn_prefill_f32(
        s: *mut RslMlxStreamRaw,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        max_ctx: c_int,
        kv_len_base: c_int,
        n_new: c_int,
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;

    // Quantized-KV flash attention (F32 Q/out, packed K/V dequantized on the
    // fly). K/V are packed-byte device pointers.
    fn rsl_mlx_flash_attn_decode_q4_0(
        s: *mut RslMlxStreamRaw,
        q: *const f32,
        k_packed: *const c_void,
        v_packed: *const c_void,
        out: *mut f32,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        max_ctx: c_int,
        kv_len: c_int,
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;
    fn rsl_mlx_flash_attn_prefill_q4_0(
        s: *mut RslMlxStreamRaw,
        q: *const f32,
        k_packed: *const c_void,
        v_packed: *const c_void,
        out: *mut f32,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        max_ctx: c_int,
        kv_len_base: c_int,
        n_new: c_int,
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;
    fn rsl_mlx_flash_attn_decode_nvfp4(
        s: *mut RslMlxStreamRaw,
        q: *const f32,
        k_packed: *const c_void,
        v_packed: *const c_void,
        out: *mut f32,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        max_ctx: c_int,
        kv_len: c_int,
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;
    fn rsl_mlx_flash_attn_prefill_nvfp4(
        s: *mut RslMlxStreamRaw,
        q: *const f32,
        k_packed: *const c_void,
        v_packed: *const c_void,
        out: *mut f32,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        max_ctx: c_int,
        kv_len_base: c_int,
        n_new: c_int,
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;
    // MXFP4/6/8-KV flash: 32-elem blocks + trailing E8M0 scale byte
    // (bytes_per_row = (head_dim/32)*{17,25,33}); mirror the NVFP4 entries.
    fn rsl_mlx_flash_attn_decode_mxfp4(
        s: *mut RslMlxStreamRaw,
        q: *const f32,
        k_packed: *const c_void,
        v_packed: *const c_void,
        out: *mut f32,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        max_ctx: c_int,
        kv_len: c_int,
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;
    fn rsl_mlx_flash_attn_prefill_mxfp4(
        s: *mut RslMlxStreamRaw,
        q: *const f32,
        k_packed: *const c_void,
        v_packed: *const c_void,
        out: *mut f32,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        max_ctx: c_int,
        kv_len_base: c_int,
        n_new: c_int,
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;
    fn rsl_mlx_flash_attn_decode_mxfp6(
        s: *mut RslMlxStreamRaw,
        q: *const f32,
        k_packed: *const c_void,
        v_packed: *const c_void,
        out: *mut f32,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        max_ctx: c_int,
        kv_len: c_int,
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;
    fn rsl_mlx_flash_attn_prefill_mxfp6(
        s: *mut RslMlxStreamRaw,
        q: *const f32,
        k_packed: *const c_void,
        v_packed: *const c_void,
        out: *mut f32,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        max_ctx: c_int,
        kv_len_base: c_int,
        n_new: c_int,
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;
    fn rsl_mlx_flash_attn_decode_mxfp8(
        s: *mut RslMlxStreamRaw,
        q: *const f32,
        k_packed: *const c_void,
        v_packed: *const c_void,
        out: *mut f32,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        max_ctx: c_int,
        kv_len: c_int,
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;
    fn rsl_mlx_flash_attn_prefill_mxfp8(
        s: *mut RslMlxStreamRaw,
        q: *const f32,
        k_packed: *const c_void,
        v_packed: *const c_void,
        out: *mut f32,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        max_ctx: c_int,
        kv_len_base: c_int,
        n_new: c_int,
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;
    fn rsl_mlx_flash_attn_decode_tq(
        s: *mut RslMlxStreamRaw,
        q: *const f32,
        k_packed: *const c_void,
        v_packed: *const c_void,
        k_scales: *const f32,
        v_scales: *const f32,
        bits: c_int,
        out: *mut f32,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        max_ctx: c_int,
        kv_len: c_int,
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;
    fn rsl_mlx_flash_attn_prefill_tq(
        s: *mut RslMlxStreamRaw,
        q: *const f32,
        k_packed: *const c_void,
        v_packed: *const c_void,
        k_scales: *const f32,
        v_scales: *const f32,
        bits: c_int,
        out: *mut f32,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        max_ctx: c_int,
        kv_len_base: c_int,
        n_new: c_int,
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;
    // Q8_0-KV flash: i8 K/V slab + per-row f32 scale (see safe wrappers).
    fn rsl_mlx_flash_attn_decode_q8_0(
        s: *mut RslMlxStreamRaw,
        q: *const f32,
        k_packed: *const c_void,
        v_packed: *const c_void,
        k_scales: *const f32,
        v_scales: *const f32,
        out: *mut f32,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        max_ctx: c_int,
        kv_len: c_int,
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;
    fn rsl_mlx_flash_attn_prefill_q8_0(
        s: *mut RslMlxStreamRaw,
        q: *const f32,
        k_packed: *const c_void,
        v_packed: *const c_void,
        k_scales: *const f32,
        v_scales: *const f32,
        out: *mut f32,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        max_ctx: c_int,
        kv_len_base: c_int,
        n_new: c_int,
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;
    fn rsl_mlx_argmax_f32(
        s: *mut RslMlxStreamRaw,
        logits: *const f32,
        vocab: c_int,
        out_idx: *mut c_int,
    ) -> c_int;

    // Feature 1c: device-resident DECODE attention block (one command buffer).
    // wq/wk/wv/wo + wg/wu/wd + k_mirror/v_mirror are DEVICE pointers; norms/
    // inv_freq + hidden/h_norm2/k_row/v_row are HOST f32. kq/kk/kv_kind/ko +
    // kg/ku/kd are weight format codes (0=Q4_K, 1=Q8_0). The gate/up/down weights
    // + d_ff are the OPTIONAL fused dense SwiGLU FFN (null + d_ff<=0 ⇒
    // attention-only); `ffn_ran` (nullable) receives 1 when the device FFN ran.
    // Returns 1 on success, 0 on decline.
    #[allow(clippy::too_many_arguments)]
    fn rsl_mlx_decode_attn_block_f32(
        s: *mut RslMlxStreamRaw,
        wq_dev: *const c_void,
        kq: c_int,
        wk_dev: *const c_void,
        kk: c_int,
        wv_dev: *const c_void,
        kv_kind: c_int,
        wo_dev: *const c_void,
        ko: c_int,
        wg_dev: *const c_void,
        kg: c_int,
        wu_dev: *const c_void,
        ku: c_int,
        wd_dev: *const c_void,
        kd: c_int,
        d_ff: c_int,
        ffn_ran: *mut c_int,
        attn_norm: *const f32,
        ffn_norm: *const f32,
        inv_freq: *const f32,
        hidden: *mut f32,
        h_norm2: *mut f32,
        k_row: *mut f32,
        v_row: *mut f32,
        k_mirror: *mut c_void,
        v_mirror: *mut c_void,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        d_model: c_int,
        pos: c_int,
        max_ctx: c_int,
        kv_len: c_int,
        rms_eps: f32,
        // Per-layer SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;

    fn rsl_mlx_consume_error_count() -> c_int;
}

/// Number of usable Metal GPU devices exposed through MLX (0 when none /
/// off Apple Silicon → the inert stub build always reports 0).
pub fn device_count() -> u32 {
    // SAFETY: no pointer args; the C shim returns a plain count.
    unsafe { rsl_mlx_device_count().max(0) as u32 }
}

/// Properties of MLX device `idx`.
pub fn device_info(idx: u32) -> Result<MlxDeviceInfo, MlxError> {
    let mut name = [0 as c_char; 256];
    let mut mem: u64 = 0;
    let mut registry_id: u64 = 0;
    let mut uuid: [u8; 16] = [0u8; 16];
    // SAFETY: all out-pointers reference live stack storage; the shim
    // NUL-terminates `name` within `name_cap` and writes 16 UUID bytes.
    let rc = unsafe {
        rsl_mlx_device_info(
            idx as c_int,
            name.as_mut_ptr(),
            256,
            &mut mem,
            &mut registry_id,
            uuid.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(MlxError::NoSuchDevice(idx));
    }
    let name = unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Ok(MlxDeviceInfo {
        name,
        total_mem_bytes: mem,
        registry_id,
        uuid,
    })
}

/// RMSNorm over `n_rows` rows of length `d`: `y = x·rsqrt(mean(x²)+eps)·w`.
/// `x`/`y` are `n_rows*d`; `w` is `d`.
pub fn rmsnorm_f32(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    n_rows: usize,
    d: usize,
    eps: f32,
) -> Result<(), MlxError> {
    assert_eq!(x.len(), n_rows * d);
    assert_eq!(y.len(), n_rows * d);
    assert!(w.len() >= d);
    // SAFETY: slices outlive the call; lengths are checked above.
    let rc = unsafe {
        rsl_mlx_rmsnorm_f32(
            x.as_ptr(),
            w.as_ptr(),
            y.as_mut_ptr(),
            n_rows as c_int,
            d as c_int,
            eps,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Dense f32 mat-vec: `out[m] = Σ_k W[m*k_dim+k]·x[k]` (row-major W).
pub fn matvec_f32(
    weights: &[f32],
    x: &[f32],
    out: &mut [f32],
    m_rows: usize,
    k_dim: usize,
) -> Result<(), MlxError> {
    assert_eq!(weights.len(), m_rows * k_dim);
    assert!(x.len() >= k_dim);
    assert!(out.len() >= m_rows);
    // SAFETY: slices outlive the call; lengths are checked above.
    let rc = unsafe {
        rsl_mlx_matvec_f32(
            weights.as_ptr(),
            x.as_ptr(),
            out.as_mut_ptr(),
            m_rows as c_int,
            k_dim as c_int,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// P17: Gated-DeltaNet chunked-prefill recurrence over ALL V-heads (host-copy).
/// WRITE-BLIND Metal mirror of the SYCL `delta_net_chunk_f32_usm_raw` + the CPU
/// chunked reference. Head-major: `q`/`k` `[n_heads*seq_len*head_qk_dim]`,
/// `v`/`out` `[n_heads*seq_len*head_v_dim]`, `g`/`beta` `[n_heads*seq_len]`,
/// `state` `[n_heads*head_qk_dim*head_v_dim]` read as incoming and overwritten
/// with the post-sequence recurrent state. `chunk_len == 0` ⇒ 64.
#[allow(clippy::too_many_arguments)]
pub fn delta_net_chunk_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    n_heads: usize,
    seq_len: usize,
    head_qk_dim: usize,
    head_v_dim: usize,
    chunk_len: usize,
) -> Result<(), MlxError> {
    assert_eq!(q.len(), n_heads * seq_len * head_qk_dim);
    assert_eq!(k.len(), n_heads * seq_len * head_qk_dim);
    assert_eq!(v.len(), n_heads * seq_len * head_v_dim);
    assert_eq!(g.len(), n_heads * seq_len);
    assert_eq!(beta.len(), n_heads * seq_len);
    assert_eq!(state.len(), n_heads * head_qk_dim * head_v_dim);
    assert_eq!(out.len(), n_heads * seq_len * head_v_dim);
    // SAFETY: slices outlive the call; lengths are checked above.
    let rc = unsafe {
        rsl_mlx_delta_net_chunk_f32(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            g.as_ptr(),
            beta.as_ptr(),
            state.as_mut_ptr(),
            out.as_mut_ptr(),
            n_heads as c_int,
            seq_len as c_int,
            head_qk_dim as c_int,
            head_v_dim as c_int,
            chunk_len as c_int,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

// ============================================================
// Device-resident path (streams + device buffers + packed kernels)
// ============================================================

/// A stream bound to a device — owns a Metal command queue (and/or an MLX
/// stream), the handle weights/activations are allocated against. Mirrors
/// the CUDA/SYCL crates' stream model; used from a single owning worker
/// thread.
pub struct MlxStream {
    raw: *mut RslMlxStreamRaw,
    device: u32,
}

// SAFETY: the stream is driven from one owning thread (the MLX worker),
// like the CUDA/SYCL stream; the raw handle is never shared concurrently.
unsafe impl Send for MlxStream {}

impl MlxStream {
    /// Create a stream on `device_index`; `None` when no such device.
    pub fn create(device_index: u32) -> Option<Self> {
        // SAFETY: FFI; returns null on failure, checked below.
        let raw = unsafe { rsl_mlx_stream_create(device_index as c_int) };
        if raw.is_null() {
            None
        } else {
            Some(Self {
                raw,
                device: device_index,
            })
        }
    }
    pub fn device(&self) -> u32 {
        self.device
    }
    fn raw(&self) -> *mut RslMlxStreamRaw {
        self.raw
    }
}

impl Drop for MlxStream {
    fn drop(&mut self) {
        // SAFETY: `raw` came from rsl_mlx_stream_create and is freed once.
        unsafe { rsl_mlx_stream_destroy(self.raw) };
    }
}

/// A device-memory buffer allocated on an [`MlxStream`]; freed on drop.
///
/// On Apple Silicon this wraps a `MTLBuffer` (StorageModeShared) whose
/// contents pointer is host-addressable — so `copy_from_host` /
/// `copy_to_host` are plain memcpys (and Phase 1 may skip the copy entirely
/// for GGUF weights that already live in a shared allocation).
pub struct MlxDeviceBuffer<'s> {
    ptr: *mut c_void,
    len_bytes: usize,
    stream: &'s MlxStream,
}

impl<'s> MlxDeviceBuffer<'s> {
    /// Allocate device memory and copy `src` bytes into it (e.g. a weight).
    pub fn from_host(stream: &'s MlxStream, src: &[u8]) -> Option<Self> {
        // SAFETY: FFI; null on failure.
        let ptr = unsafe {
            rsl_mlx_malloc_from_host(
                stream.raw(),
                src.as_ptr() as *const c_void,
                src.len() as u64,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self {
                ptr,
                len_bytes: src.len(),
                stream,
            })
        }
    }
    /// Allocate `n_bytes` of uninitialized device memory.
    pub fn alloc(stream: &'s MlxStream, n_bytes: usize) -> Option<Self> {
        if n_bytes == 0 {
            return None;
        }
        // SAFETY: FFI; null on failure.
        let ptr = unsafe { rsl_mlx_malloc_device(stream.raw(), n_bytes as u64) };
        if ptr.is_null() {
            None
        } else {
            Some(Self {
                ptr,
                len_bytes: n_bytes,
                stream,
            })
        }
    }
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr
    }
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }
    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }
    /// Copy `src` host bytes into this buffer.
    pub fn copy_from_host(&mut self, src: &[u8]) -> Result<(), MlxError> {
        // SAFETY: FFI; ptr is a live device allocation on `stream`.
        let rc = unsafe {
            rsl_mlx_memcpy_h2d(
                self.stream.raw(),
                self.ptr,
                src.as_ptr() as *const c_void,
                src.len() as u64,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(MlxError::Kernel(rc))
        }
    }
    /// Copy this buffer's bytes into `dst` (host).
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<(), MlxError> {
        // SAFETY: FFI; ptr is a live device allocation on `stream`.
        let rc = unsafe {
            rsl_mlx_memcpy_d2h(
                self.stream.raw(),
                dst.as_mut_ptr() as *mut c_void,
                self.ptr,
                dst.len() as u64,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(MlxError::Kernel(rc))
        }
    }
    /// Copy `src` host bytes into this buffer starting at `byte_offset`.
    /// Enables incremental in-place updates — e.g. appending one KV row to
    /// a persistent device-resident cache without re-uploading the whole
    /// slab. Errors (→ caller falls back to CPU) if `[offset, offset+len)`
    /// exceeds the allocation.
    pub fn copy_from_host_at(&mut self, byte_offset: usize, src: &[u8]) -> Result<(), MlxError> {
        if byte_offset
            .checked_add(src.len())
            .map(|end| end > self.len_bytes)
            .unwrap_or(true)
        {
            return Err(MlxError::Kernel(-1));
        }
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: FFI; the bounds check above keeps the offset write inside
        // this live device allocation on `stream`.
        let dst = unsafe { (self.ptr as *mut u8).add(byte_offset) as *mut c_void };
        let rc = unsafe {
            rsl_mlx_memcpy_h2d(
                self.stream.raw(),
                dst,
                src.as_ptr() as *const c_void,
                src.len() as u64,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(MlxError::Kernel(rc))
        }
    }
    /// Copy `dst.len()` bytes starting at `byte_offset` into `dst` (host).
    /// Symmetric with [`Self::copy_from_host_at`].
    pub fn copy_to_host_at(&self, byte_offset: usize, dst: &mut [u8]) -> Result<(), MlxError> {
        if byte_offset
            .checked_add(dst.len())
            .map(|end| end > self.len_bytes)
            .unwrap_or(true)
        {
            return Err(MlxError::Kernel(-1));
        }
        if dst.is_empty() {
            return Ok(());
        }
        // SAFETY: FFI; the bounds check above keeps the offset read inside
        // this live device allocation on `stream`.
        let src = unsafe { (self.ptr as *const u8).add(byte_offset) as *const c_void };
        let rc = unsafe {
            rsl_mlx_memcpy_d2h(
                self.stream.raw(),
                dst.as_mut_ptr() as *mut c_void,
                src,
                dst.len() as u64,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(MlxError::Kernel(rc))
        }
    }
}

impl<'s> Drop for MlxDeviceBuffer<'s> {
    fn drop(&mut self) {
        // SAFETY: ptr came from this stream's allocator; freed once.
        unsafe { rsl_mlx_free(self.stream.raw(), self.ptr) };
    }
}

/// Drain + reset the per-thread kernel error latch. Non-zero ⇒ a kernel
/// call failed (the caller falls back to the CPU kernel).
pub fn consume_error_count() -> i32 {
    // SAFETY: FFI; no arguments.
    unsafe { rsl_mlx_consume_error_count() }
}

/// PTQ1_0 packed matvec on device pointers: `out[M] = W(M,K) @ x[K]`.
///
/// SAFETY: `w_bytes` (M·(K/128)·28 bytes), `x` (K f32) and `out` (M f32)
/// must be device pointers valid on `stream`'s device; `k % 128 == 0`.
pub unsafe fn matvec_ptq1_0_packed_f32(
    stream: &MlxStream,
    w_bytes: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
) -> Result<(), MlxError> {
    let rc =
        rsl_mlx_matvec_ptq1_0_packed_f32(stream.raw(), w_bytes, x, out, m as c_int, k as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Batched PTQ1_0 matvec over N rows. `x` = [N,K], `out` = [N,M] device ptrs.
///
/// SAFETY: as [`matvec_ptq1_0_packed_f32`], with x/out sized for N rows.
pub unsafe fn matvec_ptq1_0_packed_f32_batched(
    stream: &MlxStream,
    w_bytes: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_matvec_ptq1_0_packed_f32_batched(
        stream.raw(),
        w_bytes,
        x,
        out,
        m as c_int,
        k as c_int,
        n as c_int,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Blockwise Prism Hadamard on device pointers.
///
/// SAFETY: `x`, `signs`, `out` are device pointers of `n_elems` f32 on
/// `stream`'s device; `block` a power of two ≤ 4096 dividing `n_elems`.
pub unsafe fn hadamard_forward(
    stream: &MlxStream,
    x: *const f32,
    signs: *const f32,
    out: *mut f32,
    n_elems: usize,
    block: usize,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_hadamard_forward(
        stream.raw(),
        x,
        signs,
        out,
        n_elems as c_int,
        block as c_int,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// B2 (Kindi rmsfold): device-resident RMSNorm `y[r,i] = x[r,i] *
/// rsqrt(mean(x_row²)+eps) * w[i]` on device pointers (reuses the
/// `rsl_mlx_rmsnorm_f32_kernel`). Unlike [`rmsnorm_f32`] (host-slice reference),
/// this keeps the result resident so the Hadamard + matvec can chain without a
/// host round-trip.
///
/// SAFETY: `x`, `y` are device pointers of `n_rows*d` f32 and `w` of `d` f32 on
/// `stream`'s device; the wrapper synchronizes before returning.
pub unsafe fn rmsnorm_dev_f32(
    stream: &MlxStream,
    x: *const f32,
    w: *const f32,
    y: *mut f32,
    n_rows: usize,
    d: usize,
    eps: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_rmsnorm_dev(stream.raw(), x, w, y, n_rows as c_int, d as c_int, eps);
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Generate the single + batched safe wrappers for a packed-quant matvec.
/// SAFETY (all): `w`, `x`, `out` are device pointers on `stream`'s device
/// sized for the quant's block layout and the given M/K(/N).
macro_rules! mlx_packed_matvec {
    ($single:ident, $batched:ident, $raw_s:ident, $raw_b:ident) => {
        #[allow(clippy::missing_safety_doc)]
        pub unsafe fn $single(
            stream: &MlxStream,
            w: *const c_void,
            x: *const f32,
            out: *mut f32,
            m: usize,
            k: usize,
        ) -> Result<(), MlxError> {
            let rc = $raw_s(stream.raw(), w, x, out, m as c_int, k as c_int);
            if rc == 0 {
                Ok(())
            } else {
                Err(MlxError::Kernel(rc))
            }
        }
        #[allow(clippy::missing_safety_doc)]
        pub unsafe fn $batched(
            stream: &MlxStream,
            w: *const c_void,
            x: *const f32,
            out: *mut f32,
            m: usize,
            k: usize,
            n: usize,
        ) -> Result<(), MlxError> {
            let rc = $raw_b(stream.raw(), w, x, out, m as c_int, k as c_int, n as c_int);
            if rc == 0 {
                Ok(())
            } else {
                Err(MlxError::Kernel(rc))
            }
        }
    };
}

/// Generate the safe wrapper for a fused gate+up matvec (decode). One dispatch
/// computes gate_out + up_out, reusing the format's single-matvec dequant.
/// SAFETY (all): `gw`/`uw`/`x`/`gout`/`uout` are device pointers on `stream`'s
/// device sized for the quant's block layout and the given M/K.
macro_rules! mlx_gate_up_fused {
    ($name:ident, $raw:ident) => {
        #[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
        pub unsafe fn $name(
            stream: &MlxStream,
            gw: *const c_void,
            uw: *const c_void,
            x: *const f32,
            gout: *mut f32,
            uout: *mut f32,
            m: usize,
            k: usize,
        ) -> Result<(), MlxError> {
            let rc = $raw(stream.raw(), gw, uw, x, gout, uout, m as c_int, k as c_int);
            if rc == 0 {
                Ok(())
            } else {
                Err(MlxError::Kernel(rc))
            }
        }
    };
}
mlx_gate_up_fused!(
    matvec_q8_0_gate_up_fused,
    rsl_mlx_matvec_q8_0_packed_f32_gate_up_fused
);
mlx_gate_up_fused!(
    matvec_q4_k_gate_up_fused,
    rsl_mlx_matvec_q4_k_packed_f32_gate_up_fused
);
mlx_gate_up_fused!(
    matvec_q6_k_gate_up_fused,
    rsl_mlx_matvec_q6_k_packed_f32_gate_up_fused
);
mlx_gate_up_fused!(
    matvec_q5_k_gate_up_fused,
    rsl_mlx_matvec_q5_k_packed_f32_gate_up_fused
);
mlx_gate_up_fused!(
    matvec_iq4_nl_gate_up_fused,
    rsl_mlx_matvec_iq4_nl_packed_f32_gate_up_fused
);
mlx_gate_up_fused!(
    matvec_iq4_xs_gate_up_fused,
    rsl_mlx_matvec_iq4_xs_packed_f32_gate_up_fused
);
mlx_gate_up_fused!(
    matvec_iq1_s_gate_up_fused,
    rsl_mlx_matvec_iq1_s_packed_f32_gate_up_fused
);
mlx_gate_up_fused!(
    matvec_iq1_m_gate_up_fused,
    rsl_mlx_matvec_iq1_m_packed_f32_gate_up_fused
);
mlx_gate_up_fused!(
    matvec_iq2_xxs_gate_up_fused,
    rsl_mlx_matvec_iq2_xxs_packed_f32_gate_up_fused
);
mlx_gate_up_fused!(
    matvec_iq2_xs_gate_up_fused,
    rsl_mlx_matvec_iq2_xs_packed_f32_gate_up_fused
);
mlx_gate_up_fused!(
    matvec_iq2_s_gate_up_fused,
    rsl_mlx_matvec_iq2_s_packed_f32_gate_up_fused
);
mlx_gate_up_fused!(
    matvec_iq3_xxs_gate_up_fused,
    rsl_mlx_matvec_iq3_xxs_packed_f32_gate_up_fused
);
mlx_gate_up_fused!(
    matvec_iq3_s_gate_up_fused,
    rsl_mlx_matvec_iq3_s_packed_f32_gate_up_fused
);
mlx_gate_up_fused!(
    matvec_ptq1_0_gate_up_fused,
    rsl_mlx_matvec_ptq1_0_packed_f32_gate_up_fused
);

mlx_packed_matvec!(
    matvec_q8_0_packed_f32,
    matvec_q8_0_packed_f32_batched,
    rsl_mlx_matvec_q8_0_packed_f32,
    rsl_mlx_matvec_q8_0_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_q4_k_packed_f32,
    matvec_q4_k_packed_f32_batched,
    rsl_mlx_matvec_q4_k_packed_f32,
    rsl_mlx_matvec_q4_k_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_q6_k_packed_f32,
    matvec_q6_k_packed_f32_batched,
    rsl_mlx_matvec_q6_k_packed_f32,
    rsl_mlx_matvec_q6_k_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_q5_k_packed_f32,
    matvec_q5_k_packed_f32_batched,
    rsl_mlx_matvec_q5_k_packed_f32,
    rsl_mlx_matvec_q5_k_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_q2_k_packed_f32,
    matvec_q2_k_packed_f32_batched,
    rsl_mlx_matvec_q2_k_packed_f32,
    rsl_mlx_matvec_q2_k_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_q8_k_packed_f32,
    matvec_q8_k_packed_f32_batched,
    rsl_mlx_matvec_q8_k_packed_f32,
    rsl_mlx_matvec_q8_k_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_q4_0_packed_f32,
    matvec_q4_0_packed_f32_batched,
    rsl_mlx_matvec_q4_0_packed_f32,
    rsl_mlx_matvec_q4_0_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_q5_0_packed_f32,
    matvec_q5_0_packed_f32_batched,
    rsl_mlx_matvec_q5_0_packed_f32,
    rsl_mlx_matvec_q5_0_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_q4_1_packed_f32,
    matvec_q4_1_packed_f32_batched,
    rsl_mlx_matvec_q4_1_packed_f32,
    rsl_mlx_matvec_q4_1_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_q5_1_packed_f32,
    matvec_q5_1_packed_f32_batched,
    rsl_mlx_matvec_q5_1_packed_f32,
    rsl_mlx_matvec_q5_1_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_iq4_nl_packed_f32,
    matvec_iq4_nl_packed_f32_batched,
    rsl_mlx_matvec_iq4_nl_packed_f32,
    rsl_mlx_matvec_iq4_nl_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_iq4_xs_packed_f32,
    matvec_iq4_xs_packed_f32_batched,
    rsl_mlx_matvec_iq4_xs_packed_f32,
    rsl_mlx_matvec_iq4_xs_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_iq2_xxs_packed_f32,
    matvec_iq2_xxs_packed_f32_batched,
    rsl_mlx_matvec_iq2_xxs_packed_f32,
    rsl_mlx_matvec_iq2_xxs_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_iq2_xs_packed_f32,
    matvec_iq2_xs_packed_f32_batched,
    rsl_mlx_matvec_iq2_xs_packed_f32,
    rsl_mlx_matvec_iq2_xs_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_iq2_s_packed_f32,
    matvec_iq2_s_packed_f32_batched,
    rsl_mlx_matvec_iq2_s_packed_f32,
    rsl_mlx_matvec_iq2_s_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_iq3_xxs_packed_f32,
    matvec_iq3_xxs_packed_f32_batched,
    rsl_mlx_matvec_iq3_xxs_packed_f32,
    rsl_mlx_matvec_iq3_xxs_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_iq3_s_packed_f32,
    matvec_iq3_s_packed_f32_batched,
    rsl_mlx_matvec_iq3_s_packed_f32,
    rsl_mlx_matvec_iq3_s_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_iq1_s_packed_f32,
    matvec_iq1_s_packed_f32_batched,
    rsl_mlx_matvec_iq1_s_packed_f32,
    rsl_mlx_matvec_iq1_s_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_iq1_m_packed_f32,
    matvec_iq1_m_packed_f32_batched,
    rsl_mlx_matvec_iq1_m_packed_f32,
    rsl_mlx_matvec_iq1_m_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_nvfp4_packed_f32,
    matvec_nvfp4_packed_f32_batched,
    rsl_mlx_matvec_nvfp4_packed_f32,
    rsl_mlx_matvec_nvfp4_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_mxfp4_packed_f32,
    matvec_mxfp4_packed_f32_batched,
    rsl_mlx_matvec_mxfp4_packed_f32,
    rsl_mlx_matvec_mxfp4_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_mxfp6_packed_f32,
    matvec_mxfp6_packed_f32_batched,
    rsl_mlx_matvec_mxfp6_packed_f32,
    rsl_mlx_matvec_mxfp6_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_mxfp8_packed_f32,
    matvec_mxfp8_packed_f32_batched,
    rsl_mlx_matvec_mxfp8_packed_f32,
    rsl_mlx_matvec_mxfp8_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_q3_k_packed_f32,
    matvec_q3_k_packed_f32_batched,
    rsl_mlx_matvec_q3_k_packed_f32,
    rsl_mlx_matvec_q3_k_packed_f32_batched
);
mlx_packed_matvec!(
    matvec_pq2_0_packed_f32,
    matvec_pq2_0_packed_f32_batched,
    rsl_mlx_matvec_pq2_0_packed_f32,
    rsl_mlx_matvec_pq2_0_packed_f32_batched
);

/// Device-ptr folded-matvec picker for the rmsfold/smfold chain: PTQ1_0 or its
/// 2-bit PQ2_0 sibling (the only two folded Prism formats). SAFETY: as the two
/// `matvec_*_packed_f32` it forwards to.
pub unsafe fn matvec_folded_dev(
    stream: &MlxStream,
    kind: MlxPackedKind,
    w: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
) -> Result<(), MlxError> {
    match kind {
        MlxPackedKind::Pq2_0 => matvec_pq2_0_packed_f32(stream, w, x, out, m, k),
        _ => matvec_ptq1_0_packed_f32(stream, w, x, out, m, k),
    }
}

// ============================================================
// Forward-pass primitive wrappers (device pointers, f32)
// ============================================================

/// Fused add-residual + RMSNorm: `hidden += branch` (in place), then
/// `y_norm = rmsnorm(hidden_row) * w`. Rows of length `d`.
///
/// SAFETY: all pointers are device pointers on `stream`'s device;
/// `hidden`/`y_norm` are `n_rows*d`, `branch` is `n_rows*d`, `w` is `d`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn add_rmsnorm_f32(
    stream: &MlxStream,
    hidden: *mut f32,
    branch: *const f32,
    w: *const f32,
    y_norm: *mut f32,
    n_rows: usize,
    d: usize,
    eps: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_add_rmsnorm_f32(
        stream.raw(),
        hidden,
        branch,
        w,
        y_norm,
        n_rows as c_int,
        d as c_int,
        eps,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// RoPE in place over `qk` = [n_heads, head_dim] with precomputed
/// `inv_freq` = [head_dim/2], at absolute position `pos`.
///
/// SAFETY: `qk` and `inv_freq` are device pointers; `head_dim` even.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn rope_f32(
    stream: &MlxStream,
    qk: *mut f32,
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    inv_freq: *const f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_rope_f32(
        stream.raw(),
        qk,
        n_heads as c_int,
        head_dim as c_int,
        pos as c_int,
        inv_freq,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// SwiGLU: `out[i] = silu(x[i]) * y[i]`, `n` elements.
///
/// SAFETY: `x`, `y`, `out` are device pointers of `n` f32.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn silu_mul_f32(
    stream: &MlxStream,
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: usize,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_silu_mul_f32(stream.raw(), x, y, out, n as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// P6-fusions (write-blind mirror): RMSNorm-of-input fused into the following
/// matvec. `y[r,o] = Σ_i (x[r,i]·rms_scale(x_row)·wnorm[i])·wmat[o,i]`, `wmat`
/// `[m,k]` out-major.
///
/// SAFETY: `x` (≥`n_rows*k`), `wnorm` (≥`k`), `wmat` (≥`m*k`), `y` (≥`n_rows*m`)
/// are device pointers on `stream`; the wrapper synchronizes before returning.
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
pub unsafe fn rmsnorm_matvec_f32(
    stream: &MlxStream,
    x: *const f32,
    wnorm: *const f32,
    wmat: *const f32,
    y: *mut f32,
    n_rows: usize,
    k: usize,
    m: usize,
    eps: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_rmsnorm_matvec_f32(
        stream.raw(),
        x,
        wnorm,
        wmat,
        y,
        n_rows as c_int,
        k as c_int,
        m as c_int,
        eps,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// P6-fusions (write-blind mirror): FFN down-proj fused with residual add.
/// `y[r,o] = resid[r,o] + Σ_i ff[r,i]·wmat[o,i]`, `wmat` `[m,k]` out-major.
///
/// SAFETY: `ff` (≥`n_rows*k`), `wmat` (≥`m*k`), `resid`/`y` (≥`n_rows*m`) are
/// device pointers on `stream`; the wrapper synchronizes before returning.
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
pub unsafe fn down_residual_f32(
    stream: &MlxStream,
    ff: *const f32,
    wmat: *const f32,
    resid: *const f32,
    y: *mut f32,
    n_rows: usize,
    k: usize,
    m: usize,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_down_residual_f32(
        stream.raw(),
        ff,
        wmat,
        resid,
        y,
        n_rows as c_int,
        k as c_int,
        m as c_int,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Embedding lookup: `out[i,:] = table[ids[i],:]` (row < 0 ⇒ zeros).
///
/// SAFETY: `table` = [V,d], `out` = [n_ids,d], `ids` = [n_ids] i32 are all
/// device pointers on `stream`'s device.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn embedding_lookup_f32(
    stream: &MlxStream,
    table: *const f32,
    ids: *const c_int,
    out: *mut f32,
    n_ids: usize,
    d: usize,
) -> Result<(), MlxError> {
    let rc =
        rsl_mlx_embedding_lookup_f32(stream.raw(), table, ids, out, n_ids as c_int, d as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// FlashAttention decode (one query per head).
///
/// SAFETY: `q`/`out` = [n_heads,head_dim], `k`/`v` = [n_kv_heads,max_ctx,
/// head_dim] device pointers; `n_heads % n_kv_heads == 0`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_f32(
    stream: &MlxStream,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the CPU
    // `gqa_attention_flash_decode_capped_windowed`. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_decode_f32(
        stream.raw(),
        q,
        k,
        v,
        out,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        max_ctx as c_int,
        kv_len as c_int,
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// GPU paged attention (Phase F — MLX twin): block-table FlashAttention decode.
/// Reads K/V from the whole-pool F32 device mirror `kv_pool` via `block_table`
/// (logical block -> physical page id) for ONE sequence at `layer_idx`. Matches
/// the contiguous [`flash_attn_decode_f32`] numerics (BIT-EXACT to the CPU
/// reference `gqa_attention_flash_decode_capped_windowed`, F32 pool — no f16
/// round-trip), differing only in how a logical KV position maps to a device
/// address. Mirror of the CUDA `flash_attn_decode_paged_f32` /
/// SYCL `flash_attn_decode_paged_usm`.
///
/// SAFETY: `q`/`out` = `[n_heads, head_dim]` f32; `kv_pool` = the pool mirror
/// slab; `block_table` = `[ceil(kv_len / page_size)]` i32 — all device pointers
/// on `stream`'s device; `n_heads % n_kv_heads == 0`; `layer_idx < n_layers`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_paged_f32(
    stream: &MlxStream,
    q: *const f32,
    kv_pool: *const f32,
    block_table: *const i32,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    layer_idx: usize,
    n_layers: usize,
    page_size: usize,
    kv_len: usize,
    window: i32,
    softcap: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_decode_paged_f32(
        stream.raw(),
        q,
        kv_pool,
        block_table as *const c_int,
        out,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        layer_idx as c_int,
        n_layers as c_int,
        page_size as c_int,
        kv_len as c_int,
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// FlashAttention prefill (causal, `n_new` queries).
///
/// SAFETY: `q`/`out` = [n_new,n_heads,head_dim], `k`/`v` = [n_kv_heads,
/// max_ctx,head_dim] device pointers; `n_heads % n_kv_heads == 0` and
/// `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_f32(
    stream: &MlxStream,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the CPU
    // `gqa_attention_flash_prefill_capped_windowed`. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_prefill_f32(
        stream.raw(),
        q,
        k,
        v,
        out,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        max_ctx as c_int,
        kv_len_base as c_int,
        n_new as c_int,
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Quantized-KV FlashAttention decode for a Q4_0 KV cache. `q`/`out` are
/// F32 device pointers `[n_heads, head_dim]`; `k_packed`/`v_packed` are the
/// packed Q4_0 KV cache `[n_kv_heads, max_ctx, (head_dim/32)*18]` bytes.
///
/// SAFETY: all pointers are device pointers on `stream`'s device sized as
/// above; `n_heads % n_kv_heads == 0`, `head_dim % 32 == 0`, `head_dim<=256`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_q4_0(
    stream: &MlxStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the
    // CPU `*_capped_windowed` reference. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_decode_q4_0(
        stream.raw(),
        q,
        k_packed,
        v_packed,
        out,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        max_ctx as c_int,
        kv_len as c_int,
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Quantized-KV FlashAttention prefill for a Q4_0 KV cache. `q`/`out` are
/// F32 `[n_new, n_heads, head_dim]`; K/V packed as in [`flash_attn_decode_q4_0`].
///
/// SAFETY: device pointers as above; `n_heads % n_kv_heads == 0`,
/// `head_dim % 32 == 0`, `head_dim <= 256`, `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_q4_0(
    stream: &MlxStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the
    // CPU `*_capped_windowed` reference. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_prefill_q4_0(
        stream.raw(),
        q,
        k_packed,
        v_packed,
        out,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        max_ctx as c_int,
        kv_len_base as c_int,
        n_new as c_int,
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Quantized-KV FlashAttention decode for an NVFP4 KV cache. Layout as
/// [`flash_attn_decode_q4_0`] but `bytes_per_row = (head_dim/16)*9`.
///
/// SAFETY: device pointers as above; `head_dim % 16 == 0`, `head_dim <= 256`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_nvfp4(
    stream: &MlxStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the
    // CPU `*_capped_windowed` reference. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_decode_nvfp4(
        stream.raw(),
        q,
        k_packed,
        v_packed,
        out,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        max_ctx as c_int,
        kv_len as c_int,
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Quantized-KV FlashAttention prefill for an NVFP4 KV cache.
///
/// SAFETY: device pointers as above; `head_dim % 16 == 0`, `head_dim <= 256`,
/// `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_nvfp4(
    stream: &MlxStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the
    // CPU `*_capped_windowed` reference. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_prefill_nvfp4(
        stream.raw(),
        q,
        k_packed,
        v_packed,
        out,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        max_ctx as c_int,
        kv_len_base as c_int,
        n_new as c_int,
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Quantized-KV FlashAttention decode for an MXFP4 KV cache. Layout as
/// [`flash_attn_decode_nvfp4`] but `bytes_per_row = (head_dim/32)*17`
/// (32 elems/block: 16 E2M1-nibble bytes + trailing E8M0 scale).
///
/// SAFETY: device pointers as above; `head_dim % 32 == 0`, `head_dim <= 256`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_mxfp4(
    stream: &MlxStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the
    // CPU `*_capped_windowed` reference. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_decode_mxfp4(
        stream.raw(),
        q,
        k_packed,
        v_packed,
        out,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        max_ctx as c_int,
        kv_len as c_int,
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Quantized-KV FlashAttention prefill for an MXFP4 KV cache.
///
/// SAFETY: device pointers as above; `head_dim % 32 == 0`, `head_dim <= 256`,
/// `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_mxfp4(
    stream: &MlxStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the
    // CPU `*_capped_windowed` reference. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_prefill_mxfp4(
        stream.raw(),
        q,
        k_packed,
        v_packed,
        out,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        max_ctx as c_int,
        kv_len_base as c_int,
        n_new as c_int,
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Quantized-KV FlashAttention decode for an MXFP6 KV cache. Layout as
/// [`flash_attn_decode_nvfp4`] but `bytes_per_row = (head_dim/32)*25`
/// (32 elems/block: 24-byte LE bitstream of 6-bit E3M2 codes + E8M0 scale).
///
/// SAFETY: device pointers as above; `head_dim % 32 == 0`, `head_dim <= 256`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_mxfp6(
    stream: &MlxStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the
    // CPU `*_capped_windowed` reference. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_decode_mxfp6(
        stream.raw(),
        q,
        k_packed,
        v_packed,
        out,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        max_ctx as c_int,
        kv_len as c_int,
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Quantized-KV FlashAttention prefill for an MXFP6 KV cache.
///
/// SAFETY: device pointers as above; `head_dim % 32 == 0`, `head_dim <= 256`,
/// `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_mxfp6(
    stream: &MlxStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the
    // CPU `*_capped_windowed` reference. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_prefill_mxfp6(
        stream.raw(),
        q,
        k_packed,
        v_packed,
        out,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        max_ctx as c_int,
        kv_len_base as c_int,
        n_new as c_int,
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Quantized-KV FlashAttention decode for an MXFP8 KV cache. Layout as
/// [`flash_attn_decode_nvfp4`] but `bytes_per_row = (head_dim/32)*33`
/// (32 elems/block: 32 E4M3 bytes + trailing E8M0 scale).
///
/// SAFETY: device pointers as above; `head_dim % 32 == 0`, `head_dim <= 256`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_mxfp8(
    stream: &MlxStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the
    // CPU `*_capped_windowed` reference. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_decode_mxfp8(
        stream.raw(),
        q,
        k_packed,
        v_packed,
        out,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        max_ctx as c_int,
        kv_len as c_int,
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Quantized-KV FlashAttention prefill for an MXFP8 KV cache.
///
/// SAFETY: device pointers as above; `head_dim % 32 == 0`, `head_dim <= 256`,
/// `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_mxfp8(
    stream: &MlxStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the
    // CPU `*_capped_windowed` reference. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_prefill_mxfp8(
        stream.raw(),
        q,
        k_packed,
        v_packed,
        out,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        max_ctx as c_int,
        kv_len_base as c_int,
        n_new as c_int,
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Quantized-KV FlashAttention decode for a TurboQuant KV cache.
/// `k_packed`/`v_packed` are `[n_kv_heads, max_ctx, ceil(head_dim*bits/8)]`;
/// `k_scales`/`v_scales` are `[n_kv_heads*max_ctx]` F32 per-row scales
/// (indexed `kv_h*max_ctx + t`). `bits` in {1,2,4,8}.
///
/// SAFETY: device pointers as above; `head_dim` a power of two `<= 256`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_tq(
    stream: &MlxStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    k_scales: *const f32,
    v_scales: *const f32,
    bits: u32,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the
    // CPU `*_capped_windowed` reference. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_decode_tq(
        stream.raw(),
        q,
        k_packed,
        v_packed,
        k_scales,
        v_scales,
        bits as c_int,
        out,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        max_ctx as c_int,
        kv_len as c_int,
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Quantized-KV FlashAttention prefill for a TurboQuant KV cache.
///
/// SAFETY: device pointers as above; `head_dim` a power of two `<= 256`,
/// `bits` in {1,2,4,8}, `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_tq(
    stream: &MlxStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    k_scales: *const f32,
    v_scales: *const f32,
    bits: u32,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the
    // CPU `*_capped_windowed` reference. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_prefill_tq(
        stream.raw(),
        q,
        k_packed,
        v_packed,
        k_scales,
        v_scales,
        bits as c_int,
        out,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        max_ctx as c_int,
        kv_len_base as c_int,
        n_new as c_int,
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Quantized-KV FlashAttention decode for a Q8_0 KV cache. `q`/`out` are
/// F32 device pointers `[n_heads, head_dim]`; `k_packed`/`v_packed` are the
/// i8 KV slabs `[n_kv_heads, max_ctx, head_dim]` (one byte per element, NOT
/// GGUF 34B/32 blocks); `k_scales`/`v_scales` are per-row absmax f32
/// `[n_kv_heads*max_ctx]` (indexed `kv_h*max_ctx + t`). Byte-exact port of
/// the CPU `gqa_attention_flash_decode_q8_0` (raw-i8 dot, factored scale).
///
/// SAFETY: all pointers are device pointers on `stream`'s device sized as
/// above; `n_heads % n_kv_heads == 0`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_q8_0(
    stream: &MlxStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    k_scales: *const f32,
    v_scales: *const f32,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the
    // CPU `*_capped_windowed` reference. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_decode_q8_0(
        stream.raw(),
        q,
        k_packed,
        v_packed,
        k_scales,
        v_scales,
        out,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        max_ctx as c_int,
        kv_len as c_int,
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Quantized-KV FlashAttention prefill for a Q8_0 KV cache. `q`/`out` are
/// F32 `[n_new, n_heads, head_dim]`; K/V + scales as in
/// [`flash_attn_decode_q8_0`].
///
/// SAFETY: device pointers as above; `n_heads % n_kv_heads == 0`,
/// `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_q8_0(
    stream: &MlxStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    k_scales: *const f32,
    v_scales: *const f32,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the
    // CPU `*_capped_windowed` reference. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_prefill_q8_0(
        stream.raw(),
        q,
        k_packed,
        v_packed,
        k_scales,
        v_scales,
        out,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        max_ctx as c_int,
        kv_len_base as c_int,
        n_new as c_int,
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Greedy argmax over `vocab` f32 logits; writes the chosen index to
/// `out_idx[0]` (lowest index on ties).
///
/// SAFETY: `logits` = [vocab] f32 and `out_idx` = [1] i32 device pointers.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn argmax_f32(
    stream: &MlxStream,
    logits: *const f32,
    vocab: usize,
    out_idx: *mut c_int,
) -> Result<(), MlxError> {
    let rc = rsl_mlx_argmax_f32(stream.raw(), logits, vocab as c_int, out_idx);
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Kernel(rc))
    }
}

/// Feature 1c: run the WHOLE per-layer DECODE attention block on the Metal GPU
/// as ONE command buffer with ONE compute encoder (the C ABI
/// `rsl_mlx_decode_attn_block_f32`) — rmsnorm → q/k/v matvec → RoPE → KV-mirror
/// write → flash-decode → o-proj → residual + rmsnorm [→ optional dense SwiGLU
/// FFN] — committed with a SINGLE `waitUntilCompleted` instead of the per-op
/// commit+wait flood. `Ok((true, ffn_ran))` = the block ran and wrote
/// `hidden`/`h_norm2`/`k_row`/`v_row` (and when `ffn_ran`, also the fused dense
/// FFN residual into `hidden`); `Ok((false, _))` = the Metal side declined (bad
/// shape / alloc / unsupported matvec code / GPU error) — the caller must fall
/// back to the per-op MLX/CPU ladder, as the host buffers are then unchanged
/// (results are copied out only on success). `kq/kk/kv_kind/ko` are weight format
/// codes (0 = Q4_K, 1 = Q8_0). `k_mirror`/`v_mirror` are the layer's persistent
/// F32 KV device buffers, pre-seeded with rows `[0, pos)`. The MLX KV mirror is
/// F32 (not f16 like SYCL), so the chain is bit-exact f32 end to end.
///
/// Optional dense SwiGLU FFN: pass `dense_ffn = Some((wg, kg, wu, ku, wd, kd,
/// d_ff))` (each weight a device pointer + format code) to run
/// gate/up→silu→down→`hidden += ffn_out` in the SAME command buffer — the whole
/// dense layer. `None` (or `d_ff == 0`) ⇒ attention-only: the second tuple
/// element of the result is then `false` and the caller runs the host FFN. When
/// the FFN ran, `h_norm2` is an unread intermediate and is NOT written back.
///
/// # Safety
///
/// Every pointer must be live for the call and sized to the shape arguments:
/// `wX_dev` the packed weight bytes (device); `attn_norm`/`ffn_norm` ≥ `d_model`,
/// `inv_freq` ≥ `head_dim/2`, `hidden`/`h_norm2` ≥ `d_model`, `k_row`/`v_row` ≥
/// `n_kv_heads*head_dim` (host); `k_mirror`/`v_mirror` ≥
/// `n_kv_heads*max_ctx*head_dim` f32 (device). The dense-FFN weights (when
/// `Some`) are device packed bytes sized to `[d_ff×d_model]` (gate/up) and
/// `[d_model×d_ff]` (down). The block syncs before returning, so there is no
/// host/device overlap on the staged buffers.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub unsafe fn decode_attn_block_f32(
    stream: &MlxStream,
    wq_dev: *const c_void,
    kq: i32,
    wk_dev: *const c_void,
    kk: i32,
    wv_dev: *const c_void,
    kv_kind: i32,
    wo_dev: *const c_void,
    ko: i32,
    // Optional dense FFN: (w_gate, k_gate, w_up, k_up, w_down, k_down, d_ff).
    dense_ffn: Option<(
        *const c_void,
        i32,
        *const c_void,
        i32,
        *const c_void,
        i32,
        u32,
    )>,
    attn_norm: *const f32,
    ffn_norm: *const f32,
    inv_freq: *const f32,
    hidden: *mut f32,
    h_norm2: *mut f32,
    k_row: *mut f32,
    v_row: *mut f32,
    k_mirror: *mut c_void,
    v_mirror: *mut c_void,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    d_model: u32,
    pos: u32,
    max_ctx: u32,
    kv_len: u32,
    rms_eps: f32,
    // Per-layer SWA window (0=full causal) + Gemma2 attention-logit soft-cap
    // (0=off), threaded into the block's internal flash-decode. Passthrough
    // no-ops at 0 so a non-SWA / non-Gemma2 layer is byte-identical.
    window: i32,
    softcap: f32,
) -> Result<(bool, bool), MlxError> {
    // Unpack the optional dense-FFN tuple into the raw C args (null pointers +
    // d_ff=0 when absent ⇒ attention-only, fail-closed on the Metal side).
    let null = core::ptr::null::<c_void>();
    let (wg, kg, wu, ku, wd, kd, d_ff) = match dense_ffn {
        Some((wg, kg, wu, ku, wd, kd, d_ff)) => (wg, kg, wu, ku, wd, kd, d_ff as c_int),
        None => (null, 0, null, 0, null, 0, 0),
    };
    let mut ffn_ran: c_int = 0;
    let ran = rsl_mlx_decode_attn_block_f32(
        stream.raw(),
        wq_dev,
        kq as c_int,
        wk_dev,
        kk as c_int,
        wv_dev,
        kv_kind as c_int,
        wo_dev,
        ko as c_int,
        wg,
        kg as c_int,
        wu,
        ku as c_int,
        wd,
        kd as c_int,
        d_ff,
        &mut ffn_ran as *mut c_int,
        attn_norm,
        ffn_norm,
        inv_freq,
        hidden,
        h_norm2,
        k_row,
        v_row,
        k_mirror,
        v_mirror,
        n_heads as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        d_model as c_int,
        pos as c_int,
        max_ctx as c_int,
        kv_len as c_int,
        rms_eps,
        window as c_int,
        softcap,
    );
    // A non-zero kernel-error latch means an encode/launch failure even if the
    // return code didn't capture it — treat it as a decline (fall back to CPU).
    if consume_error_count() != 0 {
        return Ok((false, false));
    }
    Ok((ran == 1, ran == 1 && ffn_ran == 1))
}

// ============================================================
// Device-resident packed-matvec cache (host-orchestrated dispatch)
// ============================================================
//
// The MLX analogue of the CUDA/SYCL packed-matvec dispatch. The shape is
// copied verbatim from CUDA (upload each weight ONCE, keyed by the host
// pointer of the GGUF weight bytes; reuse for every subsequent matvec)
// even though Apple Silicon is UNIFIED-MEMORY: keeping the identical shape
// makes the accel-layer dispatch backend-agnostic. On Apple the "upload"
// (H2D memcpy into a StorageModeShared MTLBuffer) is cheap, and Phase 1 may
// replace it with a zero-copy MTLBuffer wrap of the existing weight bytes
// (`newBufferWithBytesNoCopy:`), collapsing the cache to a pointer table.
//
// This cache owns a stream and is used from a single lock holder at a time
// (the accel layer wraps it in a mutex): Metal command submissions here are
// serialized, so concurrent forward passes are correct if slower. The
// dominant compute (packed matvecs) routes here; attention / norms stay on
// the proven CPU path in v1.

/// Which packed quant the MLX matvec cache can dispatch. Mirrors the
/// `rsl_mlx_matvec_*_packed_f32` kernels and `CudaPackedKind` /
/// `PackedMatvecKind` (rustllama-models). Variant names match the GGUF
/// quant names. Every arm gets a byte-exact CPU-parity port in Phase 1
/// (`mlx/rsl_mlx.mm` / `.metal`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum MlxPackedKind {
    Ptq1_0,
    Q8_0,
    Q4_K,
    Q6_K,
    // K-quants
    Q5_K,
    Q2_K,
    Q8_K,
    // Legacy (per-block scale) quants
    Q4_0,
    Q5_0,
    Q4_1,
    Q5_1,
    // IQ codebook / grid quants
    Iq4_Nl,
    Iq4_Xs,
    Iq2_Xxs,
    Iq2_Xs,
    Iq2_S,
    Iq3_Xxs,
    Iq3_S,
    Iq1_S,
    Iq1_M,
    // NVIDIA FP4 (E2M1 + FP8 E4M3 per-16 scale)
    Nvfp4,
    // OCP Microscaling FP4/FP6/FP8 (E8M0 shared scale, 32-elem blocks)
    Mxfp4,
    Mxfp6,
    Mxfp8,
    // 3-bit K-quant
    Q3_K,
    // PrismML Bonsai 2-bit
    Pq2_0,
}

impl MlxPackedKind {
    /// K must be a multiple of this for the kernel's block layout.
    pub fn k_alignment(self) -> usize {
        match self {
            MlxPackedKind::Nvfp4 => 16,
            MlxPackedKind::Q8_0
            | MlxPackedKind::Q4_0
            | MlxPackedKind::Q5_0
            | MlxPackedKind::Q4_1
            | MlxPackedKind::Q5_1
            | MlxPackedKind::Iq4_Nl
            | MlxPackedKind::Mxfp4
            | MlxPackedKind::Mxfp6
            | MlxPackedKind::Mxfp8 => 32,
            MlxPackedKind::Ptq1_0 | MlxPackedKind::Pq2_0 => 128,
            MlxPackedKind::Q4_K
            | MlxPackedKind::Q6_K
            | MlxPackedKind::Q5_K
            | MlxPackedKind::Q2_K
            | MlxPackedKind::Q8_K
            | MlxPackedKind::Q3_K
            | MlxPackedKind::Iq4_Xs
            | MlxPackedKind::Iq2_Xxs
            | MlxPackedKind::Iq2_Xs
            | MlxPackedKind::Iq2_S
            | MlxPackedKind::Iq3_Xxs
            | MlxPackedKind::Iq3_S
            | MlxPackedKind::Iq1_S
            | MlxPackedKind::Iq1_M => 256,
        }
    }
    /// Bytes per row for a K-wide weight row in this quant's layout.
    pub fn row_bytes(self, k: usize) -> usize {
        match self {
            MlxPackedKind::Ptq1_0 => (k / 128) * 28,
            MlxPackedKind::Q8_0 => (k / 32) * 34,
            MlxPackedKind::Q4_K => (k / 256) * 144,
            MlxPackedKind::Q6_K => (k / 256) * 210,
            MlxPackedKind::Q5_K => (k / 256) * 176,
            MlxPackedKind::Q2_K => (k / 256) * 84,
            MlxPackedKind::Q8_K => (k / 256) * 292,
            MlxPackedKind::Q4_0 => (k / 32) * 18,
            MlxPackedKind::Q5_0 => (k / 32) * 22,
            MlxPackedKind::Q4_1 => (k / 32) * 20,
            MlxPackedKind::Q5_1 => (k / 32) * 24,
            MlxPackedKind::Iq4_Nl => (k / 32) * 18,
            MlxPackedKind::Iq4_Xs => (k / 256) * 136,
            MlxPackedKind::Iq2_Xxs => (k / 256) * 66,
            MlxPackedKind::Iq2_Xs => (k / 256) * 74,
            MlxPackedKind::Iq2_S => (k / 256) * 82,
            MlxPackedKind::Iq3_Xxs => (k / 256) * 98,
            MlxPackedKind::Iq3_S => (k / 256) * 110,
            MlxPackedKind::Iq1_S => (k / 256) * 50,
            MlxPackedKind::Iq1_M => (k / 256) * 56,
            MlxPackedKind::Nvfp4 => (k / 16) * 9,
            MlxPackedKind::Mxfp4 => (k / 32) * 17,
            MlxPackedKind::Mxfp6 => (k / 32) * 25,
            MlxPackedKind::Mxfp8 => (k / 32) * 33,
            MlxPackedKind::Q3_K => (k / 256) * 110,
            MlxPackedKind::Pq2_0 => (k / 128) * 34,
        }
    }
}

/// An owned device allocation (raw pointer + capacity), freed on drop via
/// its owning stream. Not tied to the stream by lifetime so it can be
/// cached alongside the stream inside [`MlxMatvecCache`]; the cache
/// declares its buffers BEFORE the stream so they drop first.
struct RawDevBuf {
    ptr: *mut c_void,
    cap_bytes: usize,
    raw_stream: *mut RslMlxStreamRaw,
}

impl RawDevBuf {
    fn alloc(stream: &MlxStream, n_bytes: usize) -> Option<Self> {
        if n_bytes == 0 {
            return None;
        }
        // SAFETY: FFI; null on failure.
        let ptr = unsafe { rsl_mlx_malloc_device(stream.raw(), n_bytes as u64) };
        if ptr.is_null() {
            None
        } else {
            Some(Self {
                ptr,
                cap_bytes: n_bytes,
                raw_stream: stream.raw(),
            })
        }
    }
    /// Upload `src` bytes (must be <= cap). Returns false on failure.
    fn upload(&mut self, src: &[u8]) -> bool {
        if src.len() > self.cap_bytes {
            return false;
        }
        // SAFETY: FFI; ptr is a live device allocation of cap_bytes.
        let rc = unsafe {
            rsl_mlx_memcpy_h2d(
                self.raw_stream,
                self.ptr,
                src.as_ptr() as *const c_void,
                src.len() as u64,
            )
        };
        rc == 0
    }
    /// Download `dst.len()` bytes into `dst` (must be <= cap).
    fn download(&self, dst: &mut [u8]) -> bool {
        if dst.len() > self.cap_bytes {
            return false;
        }
        // SAFETY: FFI; ptr is a live device allocation of cap_bytes.
        let rc = unsafe {
            rsl_mlx_memcpy_d2h(
                self.raw_stream,
                dst.as_mut_ptr() as *mut c_void,
                self.ptr,
                dst.len() as u64,
            )
        };
        rc == 0
    }
}

impl Drop for RawDevBuf {
    fn drop(&mut self) {
        // SAFETY: ptr came from this stream's allocator; freed once.
        unsafe { rsl_mlx_free(self.raw_stream, self.ptr) };
    }
}

/// Process-/model-lifetime device-resident weight cache + scratch for the
/// packed-matvec MLX dispatch. See the module comment above.
pub struct MlxMatvecCache {
    // Declared before `stream` so buffers drop (and free) first.
    weights: HashMap<usize, RawDevBuf>,
    x_scratch: Option<RawDevBuf>,
    out_scratch: Option<RawDevBuf>,
    // B2 (Kindi rmsfold): device-resident folded-projection scratch — GPU
    // RMSNorm → GPU Hadamard → matvec with the activation kept resident (mirror
    // of the CUDA cache's rmsfold fields). `_hidden`/`_hnorm`/`_xrot` are the
    // per-stage activations (k), `_out` the matvec result (max m); `_wnorm` /
    // `_signs` cache the per-layer norm weight + sign vector, keyed by
    // (host_ptr, len) so they upload once per layer-constant.
    rmsfold_hidden: Option<RawDevBuf>,
    rmsfold_hnorm: Option<RawDevBuf>,
    rmsfold_xrot: Option<RawDevBuf>,
    rmsfold_out: Option<RawDevBuf>,
    rmsfold_wnorm: Option<RawDevBuf>,
    rmsfold_wnorm_key: Option<(usize, usize)>,
    rmsfold_signs: Option<RawDevBuf>,
    rmsfold_signs_key: Option<(usize, usize)>,
    // B2 (Kindi rmsfold+smfold): device-resident dense-FFN scratch (mirror of the
    // CUDA cache's ffn fields). `_gate`/`_up`/`_act`/`_arot` are the width-`d_ff`
    // stages (gate/up matvec outputs, the SiLU-gate product, and its Hadamard-
    // rotated down-proj input); the width-`d` stages reuse the `rmsfold_*` buffers
    // (qkv + FFN never run in the same layer). `_wnorm` is a SEPARATE norm cache
    // (post-attention norm, distinct from the attn-norm `rmsfold_wnorm`);
    // `_signs_ff` caches the width-`d_ff` down-input signs. Both host-addr-keyed.
    ffn_gate: Option<RawDevBuf>,
    ffn_up: Option<RawDevBuf>,
    ffn_act: Option<RawDevBuf>,
    ffn_arot: Option<RawDevBuf>,
    ffn_wnorm: Option<RawDevBuf>,
    ffn_wnorm_key: Option<(usize, usize)>,
    ffn_signs_ff: Option<RawDevBuf>,
    ffn_signs_ff_key: Option<(usize, usize)>,
    used_bytes: usize,
    budget_bytes: usize,
    // AirLLM-style layer streaming (mirror of CudaMatvecCache). `stream_window_layers
    // == 0` (the default) keeps the legacy grow-until-budget behavior —
    // BYTE-IDENTICAL. On Apple unified memory this caps duplicated weight bytes
    // (the "upload" is an intra-RAM copy) rather than fitting a bigger-than-VRAM
    // model, but it still bounds residency for a model larger than the pool.
    stream_window_layers: u32,
    current_layer_tag: u32,
    tick: u64,
    layer_keys: HashMap<u32, Vec<usize>>,
    layer_last_used: HashMap<u32, u64>,
    stream: MlxStream,
}

impl MlxMatvecCache {
    /// Create a cache on `device_index` with a device-memory budget for
    /// cached weights (bytes). `None` when no such device / stream fails.
    pub fn new(device_index: u32, budget_bytes: usize) -> Option<Self> {
        let stream = MlxStream::create(device_index)?;
        Some(Self {
            weights: HashMap::new(),
            x_scratch: None,
            out_scratch: None,
            rmsfold_hidden: None,
            rmsfold_hnorm: None,
            rmsfold_xrot: None,
            rmsfold_out: None,
            rmsfold_wnorm: None,
            rmsfold_wnorm_key: None,
            rmsfold_signs: None,
            rmsfold_signs_key: None,
            ffn_gate: None,
            ffn_up: None,
            ffn_act: None,
            ffn_arot: None,
            ffn_wnorm: None,
            ffn_wnorm_key: None,
            ffn_signs_ff: None,
            ffn_signs_ff_key: None,
            used_bytes: 0,
            budget_bytes,
            stream_window_layers: 0,
            current_layer_tag: 0,
            tick: 0,
            layer_keys: HashMap::new(),
            layer_last_used: HashMap::new(),
            stream,
        })
    }

    /// The device this cache's stream is bound to.
    pub fn device(&self) -> u32 {
        self.stream.device()
    }

    /// Bytes of cached weights currently resident on the device.
    pub fn resident_bytes(&self) -> usize {
        self.used_bytes
    }

    /// Drop all cached weights (freeing their device buffers) and reset the
    /// byte counter. Call on model reload: the weight-cache key is the host byte
    /// address, which the allocator reuses across model loads, so a stale entry
    /// would otherwise serve a wrong-sized/`wrong` buffer to the next model or
    /// accumulate until OOM. (Mirrors `CudaMatvecCache::clear`.)
    pub fn clear(&mut self) {
        self.weights.clear();
        self.used_bytes = 0;
        self.layer_keys.clear();
        self.layer_last_used.clear();
        self.tick = 0;
        self.stream_window_layers = 0;
        // B2 rmsfold scratch + host-addr cache keys (null the keys so a reload's
        // reused host addresses never serve a stale norm/sign upload).
        self.rmsfold_hidden = None;
        self.rmsfold_hnorm = None;
        self.rmsfold_xrot = None;
        self.rmsfold_out = None;
        self.rmsfold_wnorm = None;
        self.rmsfold_wnorm_key = None;
        self.rmsfold_signs = None;
        self.rmsfold_signs_key = None;
        // B2 FFN scratch + host-addr cache keys.
        self.ffn_gate = None;
        self.ffn_up = None;
        self.ffn_act = None;
        self.ffn_arot = None;
        self.ffn_wnorm = None;
        self.ffn_wnorm_key = None;
        self.ffn_signs_ff = None;
        self.ffn_signs_ff_key = None;
    }

    /// Enable AirLLM layer streaming (`k > 0`) — `ensure_weight` then cycles whole
    /// layers through the cache by evicting the coldest instead of declining.
    /// `0` (default) = legacy grow-until-budget.
    pub fn set_stream_window_layers(&mut self, k: u32) {
        self.stream_window_layers = k;
    }

    /// Tag the layer — or `encode(layer, expert)` — whose weights the next
    /// `ensure_weight` calls belong to (set by the dispatch bridge under the
    /// cache lock), for whole-layer eviction + LRU aging.
    pub fn set_layer_tag(&mut self, tag: u32) {
        self.current_layer_tag = tag;
    }

    fn untrack_key(&mut self, key: usize) {
        for keys in self.layer_keys.values_mut() {
            if let Some(pos) = keys.iter().position(|&k| k == key) {
                keys.swap_remove(pos);
                return;
            }
        }
    }

    fn evict_until_fits(&mut self, need: usize, keep_a: u32, keep_b: u32) -> bool {
        while self.used_bytes + need > self.budget_bytes {
            let mut victim: Option<u32> = None;
            let mut victim_tick: u64 = u64::MAX;
            for (&t, &tick) in self.layer_last_used.iter() {
                if t != keep_a && t != keep_b && tick < victim_tick {
                    victim_tick = tick;
                    victim = Some(t);
                }
            }
            let Some(victim) = victim else {
                return false;
            };
            if let Some(keys) = self.layer_keys.remove(&victim) {
                for k in keys {
                    if let Some(buf) = self.weights.remove(&k) {
                        self.used_bytes = self.used_bytes.saturating_sub(buf.cap_bytes);
                    }
                }
            }
            self.layer_last_used.remove(&victim);
        }
        true
    }

    fn ensure_scratch(slot: &mut Option<RawDevBuf>, stream: &MlxStream, need_bytes: usize) -> bool {
        let big_enough = matches!(slot, Some(b) if b.cap_bytes >= need_bytes);
        if big_enough {
            return true;
        }
        match RawDevBuf::alloc(stream, need_bytes) {
            Some(b) => {
                *slot = Some(b);
                true
            }
            None => false,
        }
    }

    /// Ensure `weight_key`'s bytes are resident; upload on first use.
    /// Returns false if over budget or an alloc/upload fails.
    fn ensure_weight(&mut self, weight_key: usize, w_bytes: &[u8]) -> bool {
        let tag = self.current_layer_tag;
        if self.stream_window_layers > 0 {
            self.tick += 1;
            self.layer_last_used.insert(tag, self.tick);
        }
        if let Some(existing) = self.weights.get(&weight_key) {
            if existing.cap_bytes == w_bytes.len() {
                return true;
            }
            // Stale entry from a prior model (host-address key reused across
            // reloads for a different-sized weight) — evict + re-upload so the
            // kernel never reads a wrong-sized buffer. (Clean reload also calls
            // `clear`; this is the per-entry safety net.)
            self.used_bytes = self.used_bytes.saturating_sub(existing.cap_bytes);
            self.weights.remove(&weight_key);
            if self.stream_window_layers > 0 {
                self.untrack_key(weight_key);
            }
        }
        if self.used_bytes + w_bytes.len() > self.budget_bytes {
            // Over budget. Legacy: decline → caller falls back to CPU. Streaming:
            // evict the coldest whole layer(s) to cycle the window.
            if self.stream_window_layers == 0 || !self.evict_until_fits(w_bytes.len(), tag, tag) {
                return false;
            }
        }
        let Some(mut buf) = RawDevBuf::alloc(&self.stream, w_bytes.len()) else {
            return false;
        };
        if !buf.upload(w_bytes) {
            return false;
        }
        self.used_bytes += w_bytes.len();
        self.weights.insert(weight_key, buf);
        if self.stream_window_layers > 0 {
            self.layer_keys.entry(tag).or_default().push(weight_key);
        }
        true
    }

    /// B2 (Kindi rmsfold): device-resident folded Q/K/V projection for a hybrid
    /// attention layer (Bonsai PTQ1_0). The pre-attention RMSNorm + the Prism
    /// block-Hadamard rotation run ON the GPU and the rotated activation STAYS
    /// resident while all three projections (`wq`/`wk`/`wv`) read it — mirror of
    /// `CudaMatvecCache::matvec_ptq1_0_qkv_rmsfold`. q/k/v share one rmsfold (the
    /// WHT depends only on the input + its width). Returns false (⇒ caller runs
    /// the byte-identical CPU rmsnorm+rotate+matvec path) on any bad shape /
    /// budget / kernel failure.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_ptq1_0_qkv_rmsfold(
        &mut self,
        // `kind` is the folded weight format — `Ptq1_0` or its 2-bit `Pq2_0`
        // sibling (both Prism, k%128). rmsnorm/hadamard are dtype-agnostic; only
        // the per-projection matvec + row-bytes differ.
        kind: MlxPackedKind,
        hidden: &[f32],
        norm_weight: &[f32],
        signs: &[f32],
        block: usize,
        eps: f32,
        wq_key: usize,
        wq_bytes: &[u8],
        wk_key: usize,
        wk_bytes: &[u8],
        wv_key: usize,
        wv_bytes: &[u8],
        out_q: &mut [f32],
        out_k: &mut [f32],
        out_v: &mut [f32],
        mq: usize,
        mkv: usize,
        k: usize,
    ) -> bool {
        if !matches!(kind, MlxPackedKind::Ptq1_0 | MlxPackedKind::Pq2_0) {
            return false;
        }
        let align = kind.k_alignment();
        if mq == 0
            || mkv == 0
            || k == 0
            || block == 0
            || k % block != 0
            || k % align != 0
            || hidden.len() != k
            || norm_weight.len() < k
            || signs.len() < k
            || out_q.len() != mq
            || out_k.len() != mkv
            || out_v.len() != mkv
        {
            return false;
        }
        let rb = kind.row_bytes(k);
        if wq_bytes.len() < mq * rb || wk_bytes.len() < mkv * rb || wv_bytes.len() < mkv * rb {
            return false;
        }
        // Weights resident (upload + cache on first use, keyed by host addr).
        if !self.ensure_weight(wq_key, wq_bytes)
            || !self.ensure_weight(wk_key, wk_bytes)
            || !self.ensure_weight(wv_key, wv_bytes)
        {
            return false;
        }
        // Activation stages (k) + output (max m) + the cached norm/signs (k).
        let out_cap = mq.max(mkv) * 4;
        if !Self::ensure_scratch(&mut self.rmsfold_hidden, &self.stream, k * 4)
            || !Self::ensure_scratch(&mut self.rmsfold_hnorm, &self.stream, k * 4)
            || !Self::ensure_scratch(&mut self.rmsfold_xrot, &self.stream, k * 4)
            || !Self::ensure_scratch(&mut self.rmsfold_out, &self.stream, out_cap)
            || !Self::ensure_scratch(&mut self.rmsfold_wnorm, &self.stream, k * 4)
            || !Self::ensure_scratch(&mut self.rmsfold_signs, &self.stream, k * 4)
        {
            return false;
        }
        // Cache the per-layer norm weight + sign vector — upload only when the
        // (host_ptr, len) key changes (constant across a model's tokens). A grow
        // only happens on a wider layer, itself a key change, so the key check
        // alone suffices.
        let wn_key = (norm_weight.as_ptr() as usize, k);
        if self.rmsfold_wnorm_key != Some(wn_key) {
            let b = unsafe { std::slice::from_raw_parts(norm_weight.as_ptr() as *const u8, k * 4) };
            if !self.rmsfold_wnorm.as_mut().unwrap().upload(b) {
                return false;
            }
            self.rmsfold_wnorm_key = Some(wn_key);
        }
        let sg_key = (signs.as_ptr() as usize, k);
        if self.rmsfold_signs_key != Some(sg_key) {
            let b = unsafe { std::slice::from_raw_parts(signs.as_ptr() as *const u8, k * 4) };
            if !self.rmsfold_signs.as_mut().unwrap().upload(b) {
                return false;
            }
            self.rmsfold_signs_key = Some(sg_key);
        }
        // Upload the raw hidden row (the ONE H2D per layer; the CPU path uploads a
        // rotated activation per projection, ×3).
        let hb = unsafe { std::slice::from_raw_parts(hidden.as_ptr() as *const u8, k * 4) };
        if !self.rmsfold_hidden.as_mut().unwrap().upload(hb) {
            return false;
        }
        // Stage 1 — GPU RMSNorm: hidden → hnorm.
        let hidden_ptr = self.rmsfold_hidden.as_ref().unwrap().ptr as *const f32;
        let wnorm_ptr = self.rmsfold_wnorm.as_ref().unwrap().ptr as *const f32;
        let hnorm_ptr = self.rmsfold_hnorm.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: three live device buffers on `self.stream` sized k; the wrapper
        // synchronizes before returning.
        if unsafe { rmsnorm_dev_f32(&self.stream, hidden_ptr, wnorm_ptr, hnorm_ptr, 1, k, eps) }
            .is_err()
        {
            return false;
        }
        // Stage 2 — GPU Hadamard: hnorm → xrot (resident; no host copy).
        let hnorm_cptr = self.rmsfold_hnorm.as_ref().unwrap().ptr as *const f32;
        let signs_ptr = self.rmsfold_signs.as_ref().unwrap().ptr as *const f32;
        let xrot_ptr = self.rmsfold_xrot.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: live device buffers on `self.stream` sized k; wrapper syncs.
        if unsafe { hadamard_forward(&self.stream, hnorm_cptr, signs_ptr, xrot_ptr, k, block) }
            .is_err()
        {
            return false;
        }
        // Stage 3 — three PTQ1_0 matvecs read the SAME resident rotated activation
        // (q/k/v), each writing the shared `rmsfold_out` then downloading to its
        // host output before the next overwrites it (ordered on one stream).
        let xrot_cptr = self.rmsfold_xrot.as_ref().unwrap().ptr as *const f32;
        for (w_key, out, m) in [
            (wq_key, &mut *out_q, mq),
            (wk_key, &mut *out_k, mkv),
            (wv_key, &mut *out_v, mkv),
        ] {
            let w_ptr = self.weights[&w_key].ptr;
            let out_ptr = self.rmsfold_out.as_ref().unwrap().ptr as *mut f32;
            // SAFETY: w (cached), xrot (resident), out (scratch ≥ m) live on
            // `self.stream`; the wrapper synchronizes before returning. The picker
            // routes by `kind` (PTQ1_0 vs its 2-bit PQ2_0 sibling).
            if unsafe { matvec_folded_dev(&self.stream, kind, w_ptr, xrot_cptr, out_ptr, m, k) }
                .is_err()
                || consume_error_count() != 0
            {
                return false;
            }
            let ob = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, m * 4) };
            if !self.rmsfold_out.as_ref().unwrap().download(ob) {
                return false;
            }
        }
        true
    }

    /// B2 (Kindi rmsfold+smfold): device-resident dense FFN for a hybrid layer
    /// (Bonsai PTQ1_0) — the MLX mirror of `CudaMatvecCache::matvec_ptq1_0_ffn_
    /// rmsfold_smfold`. The whole FFN runs on this cache's stream over its
    /// resident weights with the activation kept device-resident: GPU RMSNorm →
    /// Hadamard (gate/up input) → gate + up matvec → SiLU-gate → Hadamard (down
    /// input, "smfold") → down matvec → one D2H. Eliminates the per-projection CPU
    /// rotation + host round-trips the CPU `dense_ffn_one_into` pays.
    ///
    /// `hidden` is the raw d_model input (the device path re-norms it with
    /// `norm_weight` = the post-attention norm, matching the CPU path's `tmp_d`);
    /// `signs_in` is the width-`d` sign vector (gate/up input), `signs_ff` the
    /// width-`d_ff` one (down input). Writes `out` = `w_down(silu(w_gate(x))·
    /// w_up(x))` (ffn_out only — the caller adds the residual). Returns `true`
    /// iff the GPU produced `out`; any miss ⇒ `false` ⇒ caller's CPU path.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_ptq1_0_ffn_rmsfold_smfold(
        &mut self,
        // Folded weight format — `Ptq1_0` or the 2-bit `Pq2_0` (see the qkv twin).
        kind: MlxPackedKind,
        hidden: &[f32],
        norm_weight: &[f32],
        signs_in: &[f32],
        signs_ff: &[f32],
        block: usize,
        eps: f32,
        wg_key: usize,
        wg_bytes: &[u8],
        wu_key: usize,
        wu_bytes: &[u8],
        wd_key: usize,
        wd_bytes: &[u8],
        out: &mut [f32],
        d: usize,
        d_ff: usize,
    ) -> bool {
        if !matches!(kind, MlxPackedKind::Ptq1_0 | MlxPackedKind::Pq2_0) {
            return false;
        }
        let align = kind.k_alignment();
        if d == 0
            || d_ff == 0
            || block == 0
            || d % block != 0
            || d_ff % block != 0
            || d % align != 0
            || d_ff % align != 0
            || hidden.len() != d
            || norm_weight.len() < d
            || signs_in.len() < d
            || signs_ff.len() < d_ff
            || out.len() != d
        {
            return false;
        }
        let rb = kind.row_bytes(d);
        let rb_ff = kind.row_bytes(d_ff);
        // gate/up: [d_ff × d]; down: [d × d_ff].
        if wg_bytes.len() < d_ff * rb || wu_bytes.len() < d_ff * rb || wd_bytes.len() < d * rb_ff {
            return false;
        }
        if !self.ensure_weight(wg_key, wg_bytes)
            || !self.ensure_weight(wu_key, wu_bytes)
            || !self.ensure_weight(wd_key, wd_bytes)
        {
            return false;
        }
        // Width-`d` stages reuse the rmsfold scratch (qkv + FFN never run in the
        // same layer); width-`d_ff` legs are own.
        if !Self::ensure_scratch(&mut self.rmsfold_hidden, &self.stream, d * 4)
            || !Self::ensure_scratch(&mut self.rmsfold_hnorm, &self.stream, d * 4)
            || !Self::ensure_scratch(&mut self.rmsfold_xrot, &self.stream, d * 4)
            || !Self::ensure_scratch(&mut self.rmsfold_out, &self.stream, d * 4)
            || !Self::ensure_scratch(&mut self.ffn_gate, &self.stream, d_ff * 4)
            || !Self::ensure_scratch(&mut self.ffn_up, &self.stream, d_ff * 4)
            || !Self::ensure_scratch(&mut self.ffn_act, &self.stream, d_ff * 4)
            || !Self::ensure_scratch(&mut self.ffn_arot, &self.stream, d_ff * 4)
            || !Self::ensure_scratch(&mut self.ffn_wnorm, &self.stream, d * 4)
            || !Self::ensure_scratch(&mut self.rmsfold_signs, &self.stream, d * 4)
            || !Self::ensure_scratch(&mut self.ffn_signs_ff, &self.stream, d_ff * 4)
        {
            return false;
        }
        // FFN norm weight — SEPARATE key from `rmsfold_wnorm` (post-attention norm
        // vs attn-norm would otherwise thrash across a layer).
        let wn_key = (norm_weight.as_ptr() as usize, d);
        if self.ffn_wnorm_key != Some(wn_key) {
            let b = unsafe { std::slice::from_raw_parts(norm_weight.as_ptr() as *const u8, d * 4) };
            if !self.ffn_wnorm.as_mut().unwrap().upload(b) {
                return false;
            }
            self.ffn_wnorm_key = Some(wn_key);
        }
        // Gate/up-input signs (width `d`) reuse the `rmsfold_signs` cache — same
        // width-`d` key space as the qkv path (they don't coexist in a layer).
        let sg_key = (signs_in.as_ptr() as usize, d);
        if self.rmsfold_signs_key != Some(sg_key) {
            let b = unsafe { std::slice::from_raw_parts(signs_in.as_ptr() as *const u8, d * 4) };
            if !self.rmsfold_signs.as_mut().unwrap().upload(b) {
                return false;
            }
            self.rmsfold_signs_key = Some(sg_key);
        }
        // Down-input signs (width `d_ff`).
        let sf_key = (signs_ff.as_ptr() as usize, d_ff);
        if self.ffn_signs_ff_key != Some(sf_key) {
            let b = unsafe { std::slice::from_raw_parts(signs_ff.as_ptr() as *const u8, d_ff * 4) };
            if !self.ffn_signs_ff.as_mut().unwrap().upload(b) {
                return false;
            }
            self.ffn_signs_ff_key = Some(sf_key);
        }
        // Upload the raw hidden row once (the CPU path uploads a rotated activation
        // per projection instead).
        let hb = unsafe { std::slice::from_raw_parts(hidden.as_ptr() as *const u8, d * 4) };
        if !self.rmsfold_hidden.as_mut().unwrap().upload(hb) {
            return false;
        }
        // Stage 1 — RMSNorm: hidden → hnorm (width d).
        let hidden_ptr = self.rmsfold_hidden.as_ref().unwrap().ptr as *const f32;
        let wnorm_ptr = self.ffn_wnorm.as_ref().unwrap().ptr as *const f32;
        let hnorm_ptr = self.rmsfold_hnorm.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: three live device buffers on `self.stream` sized d; wrapper syncs.
        if unsafe { rmsnorm_dev_f32(&self.stream, hidden_ptr, wnorm_ptr, hnorm_ptr, 1, d, eps) }
            .is_err()
        {
            return false;
        }
        // Stage 2 — Hadamard (gate/up input): hnorm → xrot (resident).
        let hnorm_cptr = self.rmsfold_hnorm.as_ref().unwrap().ptr as *const f32;
        let signs_ptr = self.rmsfold_signs.as_ref().unwrap().ptr as *const f32;
        let xrot_ptr = self.rmsfold_xrot.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: live device buffers on `self.stream` sized d; wrapper syncs.
        if unsafe { hadamard_forward(&self.stream, hnorm_cptr, signs_ptr, xrot_ptr, d, block) }
            .is_err()
        {
            return false;
        }
        // Stage 3 — gate + up matvecs read the SAME resident xrot (d → d_ff).
        let xrot_cptr = self.rmsfold_xrot.as_ref().unwrap().ptr as *const f32;
        let wg_ptr = self.weights[&wg_key].ptr;
        let wu_ptr = self.weights[&wu_key].ptr;
        let gate_ptr = self.ffn_gate.as_ref().unwrap().ptr as *mut f32;
        let up_ptr = self.ffn_up.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: cached weights + resident xrot + ffn scratch (≥ d_ff) on the
        // stream. The picker routes by `kind` (PTQ1_0 / PQ2_0).
        if unsafe { matvec_folded_dev(&self.stream, kind, wg_ptr, xrot_cptr, gate_ptr, d_ff, d) }
            .is_err()
            || unsafe { matvec_folded_dev(&self.stream, kind, wu_ptr, xrot_cptr, up_ptr, d_ff, d) }
                .is_err()
            || consume_error_count() != 0
        {
            return false;
        }
        // Stage 4 — SiLU-gate: act = silu(gate)·up (d_ff; resident).
        let gate_cptr = self.ffn_gate.as_ref().unwrap().ptr as *const f32;
        let up_cptr = self.ffn_up.as_ref().unwrap().ptr as *const f32;
        let act_ptr = self.ffn_act.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: live d_ff-sized device buffers on `self.stream`; wrapper syncs.
        if unsafe { silu_mul_f32(&self.stream, gate_cptr, up_cptr, act_ptr, d_ff) }.is_err() {
            return false;
        }
        // Stage 5 — smfold: act → arot (d_ff; down input, resident).
        let act_cptr = self.ffn_act.as_ref().unwrap().ptr as *const f32;
        let signs_ff_ptr = self.ffn_signs_ff.as_ref().unwrap().ptr as *const f32;
        let arot_ptr = self.ffn_arot.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: live d_ff-sized device buffers on `self.stream`; wrapper syncs.
        if unsafe { hadamard_forward(&self.stream, act_cptr, signs_ff_ptr, arot_ptr, d_ff, block) }
            .is_err()
        {
            return false;
        }
        // Stage 6 — down matvec: arot → out (d_ff → d).
        let arot_cptr = self.ffn_arot.as_ref().unwrap().ptr as *const f32;
        let wd_ptr = self.weights[&wd_key].ptr;
        let out_ptr = self.rmsfold_out.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: cached weight + resident arot + rmsfold_out (≥ d) on the stream.
        // Picker routes by `kind` (PTQ1_0 / PQ2_0).
        if unsafe { matvec_folded_dev(&self.stream, kind, wd_ptr, arot_cptr, out_ptr, d, d_ff) }
            .is_err()
            || consume_error_count() != 0
        {
            return false;
        }
        let ob = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, d * 4) };
        if !self.rmsfold_out.as_ref().unwrap().download(ob) {
            return false;
        }
        true
    }

    /// Single-row packed matvec `out[M] = W(M,K) @ x[K]` on the device.
    /// Uploads the weight once (keyed by `weight_key`), copies `x` H2D,
    /// launches the kernel, copies `out` D2H. Returns false (caller falls
    /// back to CPU) on any bad shape / budget / kernel failure — leaving
    /// `out` untouched on failure.
    pub fn matvec_packed(
        &mut self,
        kind: MlxPackedKind,
        weight_key: usize,
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
    ) -> bool {
        if m == 0 || k == 0 || x.len() != k || out.len() != m {
            return false;
        }
        if k % kind.k_alignment() != 0 || w_bytes.len() < m * kind.row_bytes(k) {
            return false;
        }
        if !self.ensure_weight(weight_key, w_bytes) {
            return false;
        }
        if !Self::ensure_scratch(&mut self.x_scratch, &self.stream, k * 4)
            || !Self::ensure_scratch(&mut self.out_scratch, &self.stream, m * 4)
        {
            return false;
        }
        let x_bytes: &[u8] = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, k * 4) };
        if !self.x_scratch.as_mut().unwrap().upload(x_bytes) {
            return false;
        }
        let w_ptr = self.weights[&weight_key].ptr;
        let x_ptr = self.x_scratch.as_ref().unwrap().ptr as *const f32;
        let out_ptr = self.out_scratch.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: w/x/out are live device buffers on `self.stream` sized for
        // (M,K); the wrapper synchronizes before returning.
        let res = unsafe {
            match kind {
                MlxPackedKind::Ptq1_0 => {
                    matvec_ptq1_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Q8_0 => {
                    matvec_q8_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Q4_K => {
                    matvec_q4_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Q6_K => {
                    matvec_q6_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Q5_K => {
                    matvec_q5_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Q2_K => {
                    matvec_q2_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Q8_K => {
                    matvec_q8_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Q4_0 => {
                    matvec_q4_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Q5_0 => {
                    matvec_q5_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Q4_1 => {
                    matvec_q4_1_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Q5_1 => {
                    matvec_q5_1_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Iq4_Nl => {
                    matvec_iq4_nl_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Iq4_Xs => {
                    matvec_iq4_xs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Iq2_Xxs => {
                    matvec_iq2_xxs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Iq2_Xs => {
                    matvec_iq2_xs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Iq2_S => {
                    matvec_iq2_s_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Iq3_Xxs => {
                    matvec_iq3_xxs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Iq3_S => {
                    matvec_iq3_s_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Iq1_S => {
                    matvec_iq1_s_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Iq1_M => {
                    matvec_iq1_m_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Nvfp4 => {
                    matvec_nvfp4_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Mxfp4 => {
                    matvec_mxfp4_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Mxfp6 => {
                    matvec_mxfp6_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Mxfp8 => {
                    matvec_mxfp8_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Q3_K => {
                    matvec_q3_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                MlxPackedKind::Pq2_0 => {
                    matvec_pq2_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
            }
        };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// Fused gate+up matvec (decode): one dispatch computes `gate_out = gate_w
    /// @ x` and `up_out = up_w @ x`, reusing the format's single-matvec dequant
    /// (so bit-exact to two separate [`Self::matvec_packed`] calls). Mirror of
    /// the CUDA `matvec_gate_up_fused`. Uploads both weights (keyed separately),
    /// stages `x` + a 2*M output slab, downloads both halves. Returns false
    /// (caller runs the two-matvec fallback) on any bad shape / budget /
    /// unsupported kind / kernel failure.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_gate_up_fused(
        &mut self,
        kind: MlxPackedKind,
        gate_key: usize,
        gate_bytes: &[u8],
        up_key: usize,
        up_bytes: &[u8],
        x: &[f32],
        gate_out: &mut [f32],
        up_out: &mut [f32],
        m: usize,
        k: usize,
    ) -> bool {
        if m == 0 || k == 0 || x.len() != k || gate_out.len() != m || up_out.len() != m {
            return false;
        }
        let row_bytes = m * kind.row_bytes(k);
        if k % kind.k_alignment() != 0 || gate_bytes.len() < row_bytes || up_bytes.len() < row_bytes
        {
            return false;
        }
        if !self.ensure_weight(gate_key, gate_bytes) || !self.ensure_weight(up_key, up_bytes) {
            return false;
        }
        if !Self::ensure_scratch(&mut self.x_scratch, &self.stream, k * 4)
            || !Self::ensure_scratch(&mut self.out_scratch, &self.stream, 2 * m * 4)
        {
            return false;
        }
        let x_bytes: &[u8] = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, k * 4) };
        if !self.x_scratch.as_mut().unwrap().upload(x_bytes) {
            return false;
        }
        let gw_ptr = self.weights[&gate_key].ptr;
        let uw_ptr = self.weights[&up_key].ptr;
        let x_ptr = self.x_scratch.as_ref().unwrap().ptr as *const f32;
        // out_scratch holds [gate_out(M) | up_out(M)]; up starts at element M.
        let gout_ptr = self.out_scratch.as_ref().unwrap().ptr as *mut f32;
        let uout_ptr = unsafe { gout_ptr.add(m) };
        // SAFETY: gw/uw/x/gout/uout are live device buffers on `self.stream`
        // sized for (M,K) / 2*M; the wrapper synchronizes before returning.
        let res = unsafe {
            match kind {
                MlxPackedKind::Q8_0 => matvec_q8_0_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                MlxPackedKind::Q4_K => matvec_q4_k_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                MlxPackedKind::Q6_K => matvec_q6_k_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                MlxPackedKind::Q5_K => matvec_q5_k_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                MlxPackedKind::Iq4_Nl => matvec_iq4_nl_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                MlxPackedKind::Iq4_Xs => matvec_iq4_xs_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                MlxPackedKind::Iq1_S => matvec_iq1_s_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                MlxPackedKind::Iq1_M => matvec_iq1_m_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                MlxPackedKind::Iq2_Xxs => matvec_iq2_xxs_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                MlxPackedKind::Iq2_Xs => matvec_iq2_xs_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                MlxPackedKind::Iq2_S => matvec_iq2_s_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                MlxPackedKind::Iq3_Xxs => matvec_iq3_xxs_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                MlxPackedKind::Iq3_S => matvec_iq3_s_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                MlxPackedKind::Ptq1_0 => matvec_ptq1_0_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                // No fused kernel for the remaining kinds — caller falls back.
                _ => return false,
            }
        };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        // Download the combined [gate(M) | up(M)] slab and split.
        let mut both = vec![0f32; 2 * m];
        {
            let both_bytes: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(both.as_mut_ptr() as *mut u8, 2 * m * 4) };
            if !self.out_scratch.as_ref().unwrap().download(both_bytes) {
                return false;
            }
        }
        gate_out.copy_from_slice(&both[..m]);
        up_out.copy_from_slice(&both[m..2 * m]);
        true
    }

    /// Batched packed matvec over `n` input rows. `x` = [N,K] row-major,
    /// `out` = [N,M] row-major. Same semantics as [`Self::matvec_packed`].
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_packed_batched(
        &mut self,
        kind: MlxPackedKind,
        weight_key: usize,
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        if m == 0 || k == 0 || n == 0 || x.len() != n * k || out.len() != n * m {
            return false;
        }
        if k % kind.k_alignment() != 0 || w_bytes.len() < m * kind.row_bytes(k) {
            return false;
        }
        if !self.ensure_weight(weight_key, w_bytes) {
            return false;
        }
        if !Self::ensure_scratch(&mut self.x_scratch, &self.stream, n * k * 4)
            || !Self::ensure_scratch(&mut self.out_scratch, &self.stream, n * m * 4)
        {
            return false;
        }
        let x_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, n * k * 4) };
        if !self.x_scratch.as_mut().unwrap().upload(x_bytes) {
            return false;
        }
        let w_ptr = self.weights[&weight_key].ptr;
        let x_ptr = self.x_scratch.as_ref().unwrap().ptr as *const f32;
        let out_ptr = self.out_scratch.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: as `matvec_packed`, with x/out sized for N rows.
        let res = unsafe {
            match kind {
                MlxPackedKind::Ptq1_0 => {
                    matvec_ptq1_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Q8_0 => {
                    matvec_q8_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Q4_K => {
                    matvec_q4_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Q6_K => {
                    matvec_q6_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Q5_K => {
                    matvec_q5_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Q2_K => {
                    matvec_q2_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Q8_K => {
                    matvec_q8_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Q4_0 => {
                    matvec_q4_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Q5_0 => {
                    matvec_q5_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Q4_1 => {
                    matvec_q4_1_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Q5_1 => {
                    matvec_q5_1_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Iq4_Nl => {
                    matvec_iq4_nl_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Iq4_Xs => {
                    matvec_iq4_xs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Iq2_Xxs => {
                    matvec_iq2_xxs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Iq2_Xs => {
                    matvec_iq2_xs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Iq2_S => {
                    matvec_iq2_s_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Iq3_Xxs => {
                    matvec_iq3_xxs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Iq3_S => {
                    matvec_iq3_s_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Iq1_S => {
                    matvec_iq1_s_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Iq1_M => {
                    matvec_iq1_m_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Nvfp4 => {
                    matvec_nvfp4_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Mxfp4 => {
                    matvec_mxfp4_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Mxfp6 => {
                    matvec_mxfp6_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Mxfp8 => {
                    matvec_mxfp8_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Q3_K => {
                    matvec_q3_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                MlxPackedKind::Pq2_0 => {
                    matvec_pq2_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
            }
        };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }
}

// SAFETY: the cache owns its stream + device buffers and is only ever
// touched by one thread at a time (the accel layer holds it behind a
// mutex). The raw device pointers never alias host memory unsafely.
unsafe impl Send for MlxMatvecCache {}
