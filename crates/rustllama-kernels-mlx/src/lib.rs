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
    fn rsl_mlx_matvec_q8_0_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_q8_0_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_q4_k_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_q4_k_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_q6_k_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_q6_k_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;

    // Additional packed matvecs (K-quants Q5_K/Q2_K/Q8_K/Q3_K, legacy quants
    // Q4_0/Q5_0/Q4_1/Q5_1, IQ family, NVFP4/MXFP4/6/8, PQ2_0). One GPU
    // thread(group) per output row.
    fn rsl_mlx_matvec_q5_k_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_q5_k_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_q2_k_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_q2_k_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_q8_k_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_q8_k_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_q4_0_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_q4_0_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_q5_0_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_q5_0_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_q4_1_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_q4_1_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_q5_1_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_q5_1_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_iq4_nl_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq4_nl_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_iq4_xs_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq4_xs_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_iq2_xxs_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq2_xxs_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_iq2_xs_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq2_xs_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_iq2_s_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq2_s_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_iq3_xxs_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq3_xxs_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_iq3_s_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq3_s_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_iq1_s_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq1_s_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_iq1_m_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq1_m_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_nvfp4_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_nvfp4_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    // OCP Microscaling FP4/FP6/FP8. 32-elem blocks + trailing E8M0 scale
    // byte; K%32==0. MXFP4: 17 B, MXFP6: 25 B, MXFP8: 33 B per block.
    fn rsl_mlx_matvec_mxfp4_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_mxfp4_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_mxfp6_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_mxfp6_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_mxfp8_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_mxfp8_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    // Q3_K + PQ2_0. Q3_K: 110 B/256, K%256==0. PQ2_0: 34 B/128, K%128==0.
    fn rsl_mlx_matvec_q3_k_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_q3_k_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_mlx_matvec_pq2_0_packed_f32(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_pq2_0_packed_f32_batched(s: *mut RslMlxStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;

    // Fused gate+up matvec (decode): one dispatch computes gate_out + up_out.
    // gw/uw = packed gate/up weights [M,K]; gout/uout = [M] device ptrs.
    fn rsl_mlx_matvec_q8_0_packed_f32_gate_up_fused(s: *mut RslMlxStreamRaw, gw: *const c_void, uw: *const c_void, x: *const f32, gout: *mut f32, uout: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_q4_k_packed_f32_gate_up_fused(s: *mut RslMlxStreamRaw, gw: *const c_void, uw: *const c_void, x: *const f32, gout: *mut f32, uout: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_q6_k_packed_f32_gate_up_fused(s: *mut RslMlxStreamRaw, gw: *const c_void, uw: *const c_void, x: *const f32, gout: *mut f32, uout: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_q5_k_packed_f32_gate_up_fused(s: *mut RslMlxStreamRaw, gw: *const c_void, uw: *const c_void, x: *const f32, gout: *mut f32, uout: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq4_nl_packed_f32_gate_up_fused(s: *mut RslMlxStreamRaw, gw: *const c_void, uw: *const c_void, x: *const f32, gout: *mut f32, uout: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq4_xs_packed_f32_gate_up_fused(s: *mut RslMlxStreamRaw, gw: *const c_void, uw: *const c_void, x: *const f32, gout: *mut f32, uout: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq1_s_packed_f32_gate_up_fused(s: *mut RslMlxStreamRaw, gw: *const c_void, uw: *const c_void, x: *const f32, gout: *mut f32, uout: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq1_m_packed_f32_gate_up_fused(s: *mut RslMlxStreamRaw, gw: *const c_void, uw: *const c_void, x: *const f32, gout: *mut f32, uout: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq2_xxs_packed_f32_gate_up_fused(s: *mut RslMlxStreamRaw, gw: *const c_void, uw: *const c_void, x: *const f32, gout: *mut f32, uout: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq2_xs_packed_f32_gate_up_fused(s: *mut RslMlxStreamRaw, gw: *const c_void, uw: *const c_void, x: *const f32, gout: *mut f32, uout: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq2_s_packed_f32_gate_up_fused(s: *mut RslMlxStreamRaw, gw: *const c_void, uw: *const c_void, x: *const f32, gout: *mut f32, uout: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq3_xxs_packed_f32_gate_up_fused(s: *mut RslMlxStreamRaw, gw: *const c_void, uw: *const c_void, x: *const f32, gout: *mut f32, uout: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_iq3_s_packed_f32_gate_up_fused(s: *mut RslMlxStreamRaw, gw: *const c_void, uw: *const c_void, x: *const f32, gout: *mut f32, uout: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_mlx_matvec_ptq1_0_packed_f32_gate_up_fused(s: *mut RslMlxStreamRaw, gw: *const c_void, uw: *const c_void, x: *const f32, gout: *mut f32, uout: *mut f32, m: c_int, k: c_int) -> c_int;

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
    ) -> c_int;
    fn rsl_mlx_argmax_f32(
        s: *mut RslMlxStreamRaw,
        logits: *const f32,
        vocab: c_int,
        out_idx: *mut c_int,
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
        rsl_mlx_rmsnorm_f32(x.as_ptr(), w.as_ptr(), y.as_mut_ptr(), n_rows as c_int, d as c_int, eps)
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
        rsl_mlx_matvec_f32(weights.as_ptr(), x.as_ptr(), out.as_mut_ptr(), m_rows as c_int, k_dim as c_int)
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
            Some(Self { raw, device: device_index })
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
            Some(Self { ptr, len_bytes: src.len(), stream })
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
            Some(Self { ptr, len_bytes: n_bytes, stream })
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
    let rc =
        rsl_mlx_hadamard_forward(stream.raw(), x, signs, out, n_elems as c_int, block as c_int);
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
mlx_gate_up_fused!(matvec_q8_0_gate_up_fused, rsl_mlx_matvec_q8_0_packed_f32_gate_up_fused);
mlx_gate_up_fused!(matvec_q4_k_gate_up_fused, rsl_mlx_matvec_q4_k_packed_f32_gate_up_fused);
mlx_gate_up_fused!(matvec_q6_k_gate_up_fused, rsl_mlx_matvec_q6_k_packed_f32_gate_up_fused);
mlx_gate_up_fused!(matvec_q5_k_gate_up_fused, rsl_mlx_matvec_q5_k_packed_f32_gate_up_fused);
mlx_gate_up_fused!(matvec_iq4_nl_gate_up_fused, rsl_mlx_matvec_iq4_nl_packed_f32_gate_up_fused);
mlx_gate_up_fused!(matvec_iq4_xs_gate_up_fused, rsl_mlx_matvec_iq4_xs_packed_f32_gate_up_fused);
mlx_gate_up_fused!(matvec_iq1_s_gate_up_fused, rsl_mlx_matvec_iq1_s_packed_f32_gate_up_fused);
mlx_gate_up_fused!(matvec_iq1_m_gate_up_fused, rsl_mlx_matvec_iq1_m_packed_f32_gate_up_fused);
mlx_gate_up_fused!(matvec_iq2_xxs_gate_up_fused, rsl_mlx_matvec_iq2_xxs_packed_f32_gate_up_fused);
mlx_gate_up_fused!(matvec_iq2_xs_gate_up_fused, rsl_mlx_matvec_iq2_xs_packed_f32_gate_up_fused);
mlx_gate_up_fused!(matvec_iq2_s_gate_up_fused, rsl_mlx_matvec_iq2_s_packed_f32_gate_up_fused);
mlx_gate_up_fused!(matvec_iq3_xxs_gate_up_fused, rsl_mlx_matvec_iq3_xxs_packed_f32_gate_up_fused);
mlx_gate_up_fused!(matvec_iq3_s_gate_up_fused, rsl_mlx_matvec_iq3_s_packed_f32_gate_up_fused);
mlx_gate_up_fused!(matvec_ptq1_0_gate_up_fused, rsl_mlx_matvec_ptq1_0_packed_f32_gate_up_fused);

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
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_decode_q4_0(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int, kv_len as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(MlxError::Kernel(rc)) }
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
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_prefill_q4_0(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int,
        kv_len_base as c_int, n_new as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(MlxError::Kernel(rc)) }
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
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_decode_nvfp4(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int, kv_len as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(MlxError::Kernel(rc)) }
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
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_prefill_nvfp4(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int,
        kv_len_base as c_int, n_new as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(MlxError::Kernel(rc)) }
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
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_decode_mxfp4(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int, kv_len as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(MlxError::Kernel(rc)) }
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
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_prefill_mxfp4(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int,
        kv_len_base as c_int, n_new as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(MlxError::Kernel(rc)) }
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
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_decode_mxfp6(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int, kv_len as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(MlxError::Kernel(rc)) }
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
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_prefill_mxfp6(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int,
        kv_len_base as c_int, n_new as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(MlxError::Kernel(rc)) }
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
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_decode_mxfp8(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int, kv_len as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(MlxError::Kernel(rc)) }
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
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_prefill_mxfp8(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int,
        kv_len_base as c_int, n_new as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(MlxError::Kernel(rc)) }
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
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_decode_tq(
        stream.raw(), q, k_packed, v_packed, k_scales, v_scales, bits as c_int,
        out, n_heads as c_int, n_kv_heads as c_int, head_dim as c_int,
        max_ctx as c_int, kv_len as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(MlxError::Kernel(rc)) }
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
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_prefill_tq(
        stream.raw(), q, k_packed, v_packed, k_scales, v_scales, bits as c_int,
        out, n_heads as c_int, n_kv_heads as c_int, head_dim as c_int,
        max_ctx as c_int, kv_len_base as c_int, n_new as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(MlxError::Kernel(rc)) }
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
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_decode_q8_0(
        stream.raw(), q, k_packed, v_packed, k_scales, v_scales, out,
        n_heads as c_int, n_kv_heads as c_int, head_dim as c_int,
        max_ctx as c_int, kv_len as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(MlxError::Kernel(rc)) }
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
) -> Result<(), MlxError> {
    let rc = rsl_mlx_flash_attn_prefill_q8_0(
        stream.raw(), q, k_packed, v_packed, k_scales, v_scales, out,
        n_heads as c_int, n_kv_heads as c_int, head_dim as c_int,
        max_ctx as c_int, kv_len_base as c_int, n_new as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(MlxError::Kernel(rc)) }
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
            Some(Self { ptr, cap_bytes: n_bytes, raw_stream: stream.raw() })
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
    used_bytes: usize,
    budget_bytes: usize,
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
            used_bytes: 0,
            budget_bytes,
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
        if self.weights.contains_key(&weight_key) {
            return true;
        }
        if self.used_bytes + w_bytes.len() > self.budget_bytes {
            return false;
        }
        let Some(mut buf) = RawDevBuf::alloc(&self.stream, w_bytes.len()) else {
            return false;
        };
        if !buf.upload(w_bytes) {
            return false;
        }
        self.used_bytes += w_bytes.len();
        self.weights.insert(weight_key, buf);
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
        let x_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, k * 4) };
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
                MlxPackedKind::Ptq1_0 => matvec_ptq1_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Q8_0 => matvec_q8_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Q4_K => matvec_q4_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Q6_K => matvec_q6_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Q5_K => matvec_q5_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Q2_K => matvec_q2_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Q8_K => matvec_q8_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Q4_0 => matvec_q4_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Q5_0 => matvec_q5_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Q4_1 => matvec_q4_1_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Q5_1 => matvec_q5_1_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Iq4_Nl => matvec_iq4_nl_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Iq4_Xs => matvec_iq4_xs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Iq2_Xxs => matvec_iq2_xxs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Iq2_Xs => matvec_iq2_xs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Iq2_S => matvec_iq2_s_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Iq3_Xxs => matvec_iq3_xxs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Iq3_S => matvec_iq3_s_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Iq1_S => matvec_iq1_s_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Iq1_M => matvec_iq1_m_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Nvfp4 => matvec_nvfp4_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Mxfp4 => matvec_mxfp4_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Mxfp6 => matvec_mxfp6_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Mxfp8 => matvec_mxfp8_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Q3_K => matvec_q3_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                MlxPackedKind::Pq2_0 => matvec_pq2_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
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
        if k % kind.k_alignment() != 0
            || gate_bytes.len() < row_bytes
            || up_bytes.len() < row_bytes
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
        let x_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, k * 4) };
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
                MlxPackedKind::Q8_0 => matvec_q8_0_gate_up_fused(&self.stream, gw_ptr, uw_ptr, x_ptr, gout_ptr, uout_ptr, m, k),
                MlxPackedKind::Q4_K => matvec_q4_k_gate_up_fused(&self.stream, gw_ptr, uw_ptr, x_ptr, gout_ptr, uout_ptr, m, k),
                MlxPackedKind::Q6_K => matvec_q6_k_gate_up_fused(&self.stream, gw_ptr, uw_ptr, x_ptr, gout_ptr, uout_ptr, m, k),
                MlxPackedKind::Q5_K => matvec_q5_k_gate_up_fused(&self.stream, gw_ptr, uw_ptr, x_ptr, gout_ptr, uout_ptr, m, k),
                MlxPackedKind::Iq4_Nl => matvec_iq4_nl_gate_up_fused(&self.stream, gw_ptr, uw_ptr, x_ptr, gout_ptr, uout_ptr, m, k),
                MlxPackedKind::Iq4_Xs => matvec_iq4_xs_gate_up_fused(&self.stream, gw_ptr, uw_ptr, x_ptr, gout_ptr, uout_ptr, m, k),
                MlxPackedKind::Iq1_S => matvec_iq1_s_gate_up_fused(&self.stream, gw_ptr, uw_ptr, x_ptr, gout_ptr, uout_ptr, m, k),
                MlxPackedKind::Iq1_M => matvec_iq1_m_gate_up_fused(&self.stream, gw_ptr, uw_ptr, x_ptr, gout_ptr, uout_ptr, m, k),
                MlxPackedKind::Iq2_Xxs => matvec_iq2_xxs_gate_up_fused(&self.stream, gw_ptr, uw_ptr, x_ptr, gout_ptr, uout_ptr, m, k),
                MlxPackedKind::Iq2_Xs => matvec_iq2_xs_gate_up_fused(&self.stream, gw_ptr, uw_ptr, x_ptr, gout_ptr, uout_ptr, m, k),
                MlxPackedKind::Iq2_S => matvec_iq2_s_gate_up_fused(&self.stream, gw_ptr, uw_ptr, x_ptr, gout_ptr, uout_ptr, m, k),
                MlxPackedKind::Iq3_Xxs => matvec_iq3_xxs_gate_up_fused(&self.stream, gw_ptr, uw_ptr, x_ptr, gout_ptr, uout_ptr, m, k),
                MlxPackedKind::Iq3_S => matvec_iq3_s_gate_up_fused(&self.stream, gw_ptr, uw_ptr, x_ptr, gout_ptr, uout_ptr, m, k),
                MlxPackedKind::Ptq1_0 => matvec_ptq1_0_gate_up_fused(&self.stream, gw_ptr, uw_ptr, x_ptr, gout_ptr, uout_ptr, m, k),
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
                MlxPackedKind::Ptq1_0 => matvec_ptq1_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Q8_0 => matvec_q8_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Q4_K => matvec_q4_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Q6_K => matvec_q6_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Q5_K => matvec_q5_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Q2_K => matvec_q2_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Q8_K => matvec_q8_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Q4_0 => matvec_q4_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Q5_0 => matvec_q5_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Q4_1 => matvec_q4_1_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Q5_1 => matvec_q5_1_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Iq4_Nl => matvec_iq4_nl_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Iq4_Xs => matvec_iq4_xs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Iq2_Xxs => matvec_iq2_xxs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Iq2_Xs => matvec_iq2_xs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Iq2_S => matvec_iq2_s_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Iq3_Xxs => matvec_iq3_xxs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Iq3_S => matvec_iq3_s_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Iq1_S => matvec_iq1_s_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Iq1_M => matvec_iq1_m_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Nvfp4 => matvec_nvfp4_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Mxfp4 => matvec_mxfp4_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Mxfp6 => matvec_mxfp6_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Mxfp8 => matvec_mxfp8_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Q3_K => matvec_q3_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                MlxPackedKind::Pq2_0 => matvec_pq2_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
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
