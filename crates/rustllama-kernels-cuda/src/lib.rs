//! Native NVIDIA CUDA compute kernels for rustllama.
//!
//! This crate is real-only (no mock) and is pulled in solely when the
//! `cuda` backend is enabled, so CPU / Intel-only builds never compile it
//! and never need the CUDA toolkit. It links the nvcc-compiled kernels
//! statically (see build.rs), so a CUDA-enabled binary carries its device
//! code and the static CUDA runtime; only the NVIDIA driver is a runtime
//! dependency.
//!
//! The surface is intentionally small for now — device query + reference
//! rmsnorm/matvec — establishing the FFI + build pattern. The quant-packed
//! matvecs (PTQ1_0 / Q4_K / Q6_K / Q8_0) and flash attention port from the
//! SYCL kernels, kept behind the same parity discipline against the CPU
//! reference.

use std::collections::HashMap;
use std::os::raw::{c_char, c_int, c_void};

/// Opaque handle to the C `rsl_cuda_stream` (device + cudaStream_t).
#[repr(C)]
struct RslCudaStreamRaw {
    _private: [u8; 0],
}

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("CUDA device {0} not found")]
    NoSuchDevice(u32),
    #[error("CUDA kernel error (code {0})")]
    Kernel(i32),
}

/// One CUDA device's properties.
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub name: String,
    pub total_mem_bytes: u64,
    /// Compute capability, e.g. `(9, 0)` for Hopper.
    pub compute_capability: (i32, i32),
    /// Stable, driver-invariant device UUID (`cudaDeviceProp.uuid`).
    pub uuid: [u8; 16],
}

extern "C" {
    fn rsl_cuda_device_count() -> c_int;
    fn rsl_cuda_device_info(
        idx: c_int,
        name: *mut c_char,
        name_cap: c_int,
        total_mem: *mut u64,
        cc_major: *mut c_int,
        cc_minor: *mut c_int,
        uuid: *mut u8,
    ) -> c_int;
    fn rsl_cuda_rmsnorm_f32(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        n_rows: c_int,
        d: c_int,
        eps: f32,
    ) -> c_int;
    fn rsl_cuda_matvec_f32(
        w: *const f32,
        x: *const f32,
        out: *mut f32,
        m_rows: c_int,
        k_dim: c_int,
    ) -> c_int;

    // Device-resident path.
    fn rsl_cuda_stream_create(device_index: c_int) -> *mut RslCudaStreamRaw;
    fn rsl_cuda_stream_destroy(s: *mut RslCudaStreamRaw);
    fn rsl_cuda_malloc_from_host(
        s: *mut RslCudaStreamRaw,
        src: *const c_void,
        n_bytes: u64,
    ) -> *mut c_void;
    fn rsl_cuda_malloc_device(s: *mut RslCudaStreamRaw, n_bytes: u64) -> *mut c_void;
    fn rsl_cuda_free(s: *mut RslCudaStreamRaw, dev_ptr: *mut c_void);
    fn rsl_cuda_memcpy_h2d(
        s: *mut RslCudaStreamRaw,
        dst_dev: *mut c_void,
        src_host: *const c_void,
        n_bytes: u64,
    ) -> c_int;
    fn rsl_cuda_memcpy_d2h(
        s: *mut RslCudaStreamRaw,
        dst_host: *mut c_void,
        src_dev: *const c_void,
        n_bytes: u64,
    ) -> c_int;
    fn rsl_cuda_matvec_ptq1_0_packed_f32(
        s: *mut RslCudaStreamRaw,
        w_bytes_dev: *const c_void,
        x_dev: *const f32,
        out_dev: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_ptq1_0_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w_bytes_dev: *const c_void,
        x_dev: *const f32,
        out_dev: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_hadamard_forward(
        s: *mut RslCudaStreamRaw,
        x_dev: *const f32,
        signs_dev: *const f32,
        out_dev: *mut f32,
        n_elems: c_int,
        block: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q8_0_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_q8_0_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_q4_k_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_q4_k_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_q6_k_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_q6_k_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;

    // Additional packed matvecs (K-quants Q5_K/Q2_K/Q8_K, legacy quants
    // Q4_0/Q5_0/Q4_1/Q5_1, IQ family, NVFP4). One thread per output row.
    fn rsl_cuda_matvec_q5_k_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_q5_k_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_q2_k_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_q2_k_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_q8_k_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_q8_k_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_q4_0_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_q4_0_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_q5_0_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_q5_0_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_q4_1_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_q4_1_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_q5_1_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_q5_1_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_iq4_nl_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_iq4_nl_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_iq4_xs_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_iq4_xs_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_iq2_xxs_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_iq2_xxs_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_iq2_xs_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_iq2_xs_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_iq2_s_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_iq2_s_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_iq3_xxs_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_iq3_xxs_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_iq3_s_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_iq3_s_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_iq1_s_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_iq1_s_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_iq1_m_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_iq1_m_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_nvfp4_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_nvfp4_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    // OCP Microscaling FP4/FP6/FP8. 32-elem blocks + trailing E8M0 scale
    // byte; K%32==0. MXFP4: 17 B, MXFP6: 25 B, MXFP8: 33 B per block.
    fn rsl_cuda_matvec_mxfp4_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_mxfp4_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_mxfp6_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_mxfp6_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_mxfp8_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_mxfp8_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    // Q3_K + PQ2_0 (parity-gap close). Q3_K: 110 B/256, K%256==0.
    // PQ2_0: 34 B/128, K%128==0.
    fn rsl_cuda_matvec_q3_k_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_q3_k_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;
    fn rsl_cuda_matvec_pq2_0_packed_f32(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int) -> c_int;
    fn rsl_cuda_matvec_pq2_0_packed_f32_batched(s: *mut RslCudaStreamRaw, w: *const c_void, x: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int) -> c_int;

    // Forward-pass primitives (device-resident, f32).
    fn rsl_cuda_add_rmsnorm_f32(
        s: *mut RslCudaStreamRaw,
        hidden: *mut f32,
        branch: *const f32,
        w: *const f32,
        y_norm: *mut f32,
        n_rows: c_int,
        d: c_int,
        eps: f32,
    ) -> c_int;
    fn rsl_cuda_rope_f32(
        s: *mut RslCudaStreamRaw,
        qk: *mut f32,
        n_heads: c_int,
        head_dim: c_int,
        pos: c_int,
        inv_freq: *const f32,
    ) -> c_int;
    fn rsl_cuda_silu_mul_f32(
        s: *mut RslCudaStreamRaw,
        x: *const f32,
        y: *const f32,
        out: *mut f32,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_embedding_lookup_f32(
        s: *mut RslCudaStreamRaw,
        table: *const f32,
        ids: *const c_int,
        out: *mut f32,
        n_ids: c_int,
        d: c_int,
    ) -> c_int;
    fn rsl_cuda_flash_attn_decode_f32(
        s: *mut RslCudaStreamRaw,
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
    fn rsl_cuda_flash_attn_prefill_f32(
        s: *mut RslCudaStreamRaw,
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

    // Quantized-KV flash attention (F32 Q/out, packed K/V dequantized
    // on the fly). K/V are packed-byte device pointers.
    fn rsl_cuda_flash_attn_decode_q4_0(
        s: *mut RslCudaStreamRaw,
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
    fn rsl_cuda_flash_attn_prefill_q4_0(
        s: *mut RslCudaStreamRaw,
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
    fn rsl_cuda_flash_attn_decode_nvfp4(
        s: *mut RslCudaStreamRaw,
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
    fn rsl_cuda_flash_attn_prefill_nvfp4(
        s: *mut RslCudaStreamRaw,
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
    fn rsl_cuda_flash_attn_decode_mxfp4(
        s: *mut RslCudaStreamRaw,
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
    fn rsl_cuda_flash_attn_prefill_mxfp4(
        s: *mut RslCudaStreamRaw,
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
    fn rsl_cuda_flash_attn_decode_mxfp6(
        s: *mut RslCudaStreamRaw,
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
    fn rsl_cuda_flash_attn_prefill_mxfp6(
        s: *mut RslCudaStreamRaw,
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
    fn rsl_cuda_flash_attn_decode_mxfp8(
        s: *mut RslCudaStreamRaw,
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
    fn rsl_cuda_flash_attn_prefill_mxfp8(
        s: *mut RslCudaStreamRaw,
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
    fn rsl_cuda_flash_attn_decode_tq(
        s: *mut RslCudaStreamRaw,
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
    fn rsl_cuda_flash_attn_prefill_tq(
        s: *mut RslCudaStreamRaw,
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
    fn rsl_cuda_flash_attn_decode_q8_0(
        s: *mut RslCudaStreamRaw,
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
    fn rsl_cuda_flash_attn_prefill_q8_0(
        s: *mut RslCudaStreamRaw,
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
    fn rsl_cuda_argmax_f32(
        s: *mut RslCudaStreamRaw,
        logits: *const f32,
        vocab: c_int,
        out_idx: *mut c_int,
    ) -> c_int;

    // ---- Blackwell SM12x tensor-core GEMMs (cuda/rsl_blackwell.cuh) ----
    // `*_available` is 1 only when this binary compiled the real `mma.sync`
    // path (an SM12x `a`/`f` arch was in RUSTLLAMA_CUDA_ARCHS) AND device
    // `dev` is SM12x Blackwell. The GEMM entry points take W (M×K, packed
    // FP4), X (N×K f32, quantized to FP4 in-kernel => W4A4), OUT (N×M f32),
    // and return -2 when the TC path is unavailable so the caller falls back.
    fn rsl_cuda_blackwell_tc_available(dev: c_int) -> c_int;
    fn rsl_cuda_gemm_nvfp4_tc_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_gemm_mxfp4_tc_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> c_int;
    // FP8 (MXFP8, W8A8) and FP6 (MXFP6, W6A6) block-scaled TC GEMMs
    // (kind::mxf8f6f4, m16n8k32, K % 32 == 0). Same (M,N,K)/return contract.
    fn rsl_cuda_gemm_mxfp8_tc_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_gemm_mxfp6_tc_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> c_int;
    // Phase 4: TMA-staged NVFP4 (cp.async.bulk activation staging). Same
    // (M,N,K)/return contract as rsl_cuda_gemm_nvfp4_tc_f32.
    fn rsl_cuda_gemm_nvfp4_tc_tma_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> c_int;

    fn rsl_cuda_consume_error_count() -> c_int;
}

/// Whether this binary compiled the Blackwell SM12x tensor-core GEMM path
/// (an `a`/`f` SM12x arch was in `RUSTLLAMA_CUDA_ARCHS`) AND device `dev` is a
/// real SM12x Blackwell GPU. `false` on every non-Blackwell host and on a
/// default (Ampere→Hopper) build — callers keep the scalar packed path there.
pub fn blackwell_tc_available(dev: u32) -> bool {
    // SAFETY: no pointer args; the shim queries cudaDeviceProp internally.
    unsafe { rsl_cuda_blackwell_tc_available(dev as c_int) == 1 }
}

/// The Blackwell FP4 tensor-core GEMMs. `kind` selects NVFP4 vs MXFP4. See the
/// extern block's note for the operand contract. Returns `Err(Kernel(-2))`
/// when the TC path is unavailable (the caller then uses the scalar path);
/// any other non-zero rc is a launch error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaFp4TcKind {
    Nvfp4,
    Mxfp4,
}

/// Run a Blackwell FP4 tensor-core GEMM: `out[N×M] = X[N×K] · W[M×K]ᵀ`, W in
/// FP4 (NVFP4/MXFP4) blocks, X f32 (quantized to FP4 on the fly). `K % 64 == 0`.
///
/// # Safety
/// `w`/`x`/`out` must be live device buffers on `stream` sized for (M,N,K):
/// `w` = M rows of FP4 blocks, `x` = N·K f32, `out` = N·M f32.
pub unsafe fn gemm_fp4_tc_f32(
    kind: CudaFp4TcKind,
    stream: &CudaStream,
    w: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), CudaError> {
    let rc = match kind {
        CudaFp4TcKind::Nvfp4 => {
            rsl_cuda_gemm_nvfp4_tc_f32(stream.raw(), w, x, out, m as c_int, n as c_int, k as c_int)
        }
        CudaFp4TcKind::Mxfp4 => {
            rsl_cuda_gemm_mxfp4_tc_f32(stream.raw(), w, x, out, m as c_int, n as c_int, k as c_int)
        }
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Blackwell MXFP8 (W8A8) tensor-core GEMM: `out[N×M] = X[N×K] · W[M×K]ᵀ`, W in
/// MXFP8 (per-32 E8M0, E4M3 elements), X f32 (quantized to E4M3 on the fly).
/// `K % 32 == 0`. `Err(Kernel(-2))` when the TC path is unavailable.
///
/// # Safety
/// `w` (M rows of MXFP8 blocks), `x` (N·K f32), `out` (N·M f32) must be live
/// device buffers on `stream`.
pub unsafe fn gemm_mxfp8_tc_f32(
    stream: &CudaStream,
    w: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_gemm_mxfp8_tc_f32(stream.raw(), w, x, out, m as c_int, n as c_int, k as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Blackwell MXFP6 (W6A6) tensor-core GEMM, as [`gemm_mxfp8_tc_f32`] but W in
/// MXFP6 (per-32 E8M0, E3M2 elements), activations quantized to E3M2.
///
/// # Safety
/// As [`gemm_mxfp8_tc_f32`], with `w` holding MXFP6 (25B/32) blocks.
pub unsafe fn gemm_mxfp6_tc_f32(
    stream: &CudaStream,
    w: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_gemm_mxfp6_tc_f32(stream.raw(), w, x, out, m as c_int, n as c_int, k as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Blackwell TMA-staged NVFP4 tensor-core GEMM — numerically identical to
/// [`gemm_fp4_tc_f32`] with `Nvfp4`, but stages the activation tile via
/// `cp.async.bulk` + an mbarrier (Phase 4). `K % 64 == 0`.
///
/// # Safety
/// As [`gemm_fp4_tc_f32`].
pub unsafe fn gemm_nvfp4_tc_tma_f32(
    stream: &CudaStream,
    w: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), CudaError> {
    let rc =
        rsl_cuda_gemm_nvfp4_tc_tma_f32(stream.raw(), w, x, out, m as c_int, n as c_int, k as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Number of visible NVIDIA CUDA devices (0 when none / driver absent).
pub fn device_count() -> u32 {
    // SAFETY: no pointer args; the C shim returns a plain count.
    unsafe { rsl_cuda_device_count().max(0) as u32 }
}

/// Properties of CUDA device `idx`.
pub fn device_info(idx: u32) -> Result<CudaDeviceInfo, CudaError> {
    let mut name = [0 as c_char; 256];
    let mut mem: u64 = 0;
    let mut maj: c_int = 0;
    let mut min: c_int = 0;
    let mut uuid: [u8; 16] = [0u8; 16];
    // SAFETY: all out-pointers reference live stack storage; the shim
    // NUL-terminates `name` within `name_cap` and writes 16 UUID bytes.
    let rc = unsafe {
        rsl_cuda_device_info(
            idx as c_int,
            name.as_mut_ptr(),
            256,
            &mut mem,
            &mut maj,
            &mut min,
            uuid.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(CudaError::NoSuchDevice(idx));
    }
    let name = unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Ok(CudaDeviceInfo {
        name,
        total_mem_bytes: mem,
        compute_capability: (maj, min),
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
) -> Result<(), CudaError> {
    assert_eq!(x.len(), n_rows * d);
    assert_eq!(y.len(), n_rows * d);
    assert!(w.len() >= d);
    // SAFETY: slices outlive the call; lengths are checked above.
    let rc = unsafe {
        rsl_cuda_rmsnorm_f32(x.as_ptr(), w.as_ptr(), y.as_mut_ptr(), n_rows as c_int, d as c_int, eps)
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Dense f32 mat-vec: `out[m] = Σ_k W[m*k_dim+k]·x[k]` (row-major W).
pub fn matvec_f32(
    weights: &[f32],
    x: &[f32],
    out: &mut [f32],
    m_rows: usize,
    k_dim: usize,
) -> Result<(), CudaError> {
    assert_eq!(weights.len(), m_rows * k_dim);
    assert!(x.len() >= k_dim);
    assert!(out.len() >= m_rows);
    // SAFETY: slices outlive the call; lengths are checked above.
    let rc = unsafe {
        rsl_cuda_matvec_f32(weights.as_ptr(), x.as_ptr(), out.as_mut_ptr(), m_rows as c_int, k_dim as c_int)
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

// ============================================================
// Device-resident path (streams + device buffers + packed kernels)
// ============================================================

/// A CUDA stream bound to a device — owns a `cudaStream_t`, the handle
/// weights/activations are allocated against. Mirrors the SYCL crate's
/// stream model; used from a single owning worker thread.
pub struct CudaStream {
    raw: *mut RslCudaStreamRaw,
    device: u32,
}

// SAFETY: the stream is driven from one owning thread (the CUDA worker),
// like the SYCL stream; the raw handle is never shared concurrently.
unsafe impl Send for CudaStream {}

impl CudaStream {
    /// Create a stream on `device_index`; `None` when no such device.
    pub fn create(device_index: u32) -> Option<Self> {
        // SAFETY: FFI; returns null on failure, checked below.
        let raw = unsafe { rsl_cuda_stream_create(device_index as c_int) };
        if raw.is_null() {
            None
        } else {
            Some(Self { raw, device: device_index })
        }
    }
    pub fn device(&self) -> u32 {
        self.device
    }
    fn raw(&self) -> *mut RslCudaStreamRaw {
        self.raw
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        // SAFETY: `raw` came from rsl_cuda_stream_create and is freed once.
        unsafe { rsl_cuda_stream_destroy(self.raw) };
    }
}

/// A device-memory buffer allocated on a [`CudaStream`]; freed on drop.
pub struct CudaDeviceBuffer<'s> {
    ptr: *mut c_void,
    len_bytes: usize,
    stream: &'s CudaStream,
}

impl<'s> CudaDeviceBuffer<'s> {
    /// Allocate device memory and copy `src` bytes into it (e.g. a weight).
    pub fn from_host(stream: &'s CudaStream, src: &[u8]) -> Option<Self> {
        // SAFETY: FFI; null on failure.
        let ptr = unsafe {
            rsl_cuda_malloc_from_host(
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
    pub fn alloc(stream: &'s CudaStream, n_bytes: usize) -> Option<Self> {
        if n_bytes == 0 {
            return None;
        }
        // SAFETY: FFI; null on failure.
        let ptr = unsafe { rsl_cuda_malloc_device(stream.raw(), n_bytes as u64) };
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
    pub fn copy_from_host(&mut self, src: &[u8]) -> Result<(), CudaError> {
        // SAFETY: FFI; ptr is a live device allocation on `stream`.
        let rc = unsafe {
            rsl_cuda_memcpy_h2d(
                self.stream.raw(),
                self.ptr,
                src.as_ptr() as *const c_void,
                src.len() as u64,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(CudaError::Kernel(rc))
        }
    }
    /// Copy this buffer's bytes into `dst` (host).
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<(), CudaError> {
        // SAFETY: FFI; ptr is a live device allocation on `stream`.
        let rc = unsafe {
            rsl_cuda_memcpy_d2h(
                self.stream.raw(),
                dst.as_mut_ptr() as *mut c_void,
                self.ptr,
                dst.len() as u64,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(CudaError::Kernel(rc))
        }
    }
    /// Copy `src` host bytes into this buffer starting at `byte_offset`.
    /// Enables incremental in-place updates — e.g. appending one KV row to
    /// a persistent device-resident cache without re-uploading the whole
    /// slab. Errors (→ caller falls back to CPU) if `[offset, offset+len)`
    /// exceeds the allocation.
    pub fn copy_from_host_at(&mut self, byte_offset: usize, src: &[u8]) -> Result<(), CudaError> {
        if byte_offset
            .checked_add(src.len())
            .map(|end| end > self.len_bytes)
            .unwrap_or(true)
        {
            return Err(CudaError::Kernel(-1));
        }
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: FFI; the bounds check above keeps the offset write inside
        // this live device allocation on `stream`.
        let dst = unsafe { (self.ptr as *mut u8).add(byte_offset) as *mut c_void };
        let rc = unsafe {
            rsl_cuda_memcpy_h2d(
                self.stream.raw(),
                dst,
                src.as_ptr() as *const c_void,
                src.len() as u64,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(CudaError::Kernel(rc))
        }
    }
    /// Copy `dst.len()` bytes starting at `byte_offset` into `dst` (host).
    /// Symmetric with [`Self::copy_from_host_at`].
    pub fn copy_to_host_at(&self, byte_offset: usize, dst: &mut [u8]) -> Result<(), CudaError> {
        if byte_offset
            .checked_add(dst.len())
            .map(|end| end > self.len_bytes)
            .unwrap_or(true)
        {
            return Err(CudaError::Kernel(-1));
        }
        if dst.is_empty() {
            return Ok(());
        }
        // SAFETY: FFI; the bounds check above keeps the offset read inside
        // this live device allocation on `stream`.
        let src = unsafe { (self.ptr as *const u8).add(byte_offset) as *const c_void };
        let rc = unsafe {
            rsl_cuda_memcpy_d2h(
                self.stream.raw(),
                dst.as_mut_ptr() as *mut c_void,
                src,
                dst.len() as u64,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(CudaError::Kernel(rc))
        }
    }
}

impl<'s> Drop for CudaDeviceBuffer<'s> {
    fn drop(&mut self) {
        // SAFETY: ptr came from this stream's allocator; freed once.
        unsafe { rsl_cuda_free(self.stream.raw(), self.ptr) };
    }
}

/// Drain + reset the per-thread kernel error latch. Non-zero ⇒ a kernel
/// call failed (the caller falls back to the CPU kernel).
pub fn consume_error_count() -> i32 {
    // SAFETY: FFI; no arguments.
    unsafe { rsl_cuda_consume_error_count() }
}

/// PTQ1_0 packed matvec on device pointers: `out[M] = W(M,K) @ x[K]`.
///
/// SAFETY: `w_bytes` (M·(K/128)·28 bytes), `x` (K f32) and `out` (M f32)
/// must be device pointers valid on `stream`'s device; `k % 128 == 0`.
pub unsafe fn matvec_ptq1_0_packed_f32(
    stream: &CudaStream,
    w_bytes: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
) -> Result<(), CudaError> {
    let rc =
        rsl_cuda_matvec_ptq1_0_packed_f32(stream.raw(), w_bytes, x, out, m as c_int, k as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Batched PTQ1_0 matvec over N rows. `x` = [N,K], `out` = [N,M] device ptrs.
///
/// SAFETY: as [`matvec_ptq1_0_packed_f32`], with x/out sized for N rows.
pub unsafe fn matvec_ptq1_0_packed_f32_batched(
    stream: &CudaStream,
    w_bytes: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_matvec_ptq1_0_packed_f32_batched(
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
        Err(CudaError::Kernel(rc))
    }
}

/// Blockwise Prism Hadamard on device pointers.
///
/// SAFETY: `x`, `signs`, `out` are device pointers of `n_elems` f32 on
/// `stream`'s device; `block` a power of two ≤ 4096 dividing `n_elems`.
pub unsafe fn hadamard_forward(
    stream: &CudaStream,
    x: *const f32,
    signs: *const f32,
    out: *mut f32,
    n_elems: usize,
    block: usize,
) -> Result<(), CudaError> {
    let rc =
        rsl_cuda_hadamard_forward(stream.raw(), x, signs, out, n_elems as c_int, block as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Generate the single + batched safe wrappers for a packed-quant matvec.
/// SAFETY (all): `w`, `x`, `out` are device pointers on `stream`'s device
/// sized for the quant's block layout and the given M/K(/N).
macro_rules! cuda_packed_matvec {
    ($single:ident, $batched:ident, $raw_s:ident, $raw_b:ident) => {
        #[allow(clippy::missing_safety_doc)]
        pub unsafe fn $single(
            stream: &CudaStream,
            w: *const c_void,
            x: *const f32,
            out: *mut f32,
            m: usize,
            k: usize,
        ) -> Result<(), CudaError> {
            let rc = $raw_s(stream.raw(), w, x, out, m as c_int, k as c_int);
            if rc == 0 {
                Ok(())
            } else {
                Err(CudaError::Kernel(rc))
            }
        }
        #[allow(clippy::missing_safety_doc)]
        pub unsafe fn $batched(
            stream: &CudaStream,
            w: *const c_void,
            x: *const f32,
            out: *mut f32,
            m: usize,
            k: usize,
            n: usize,
        ) -> Result<(), CudaError> {
            let rc = $raw_b(stream.raw(), w, x, out, m as c_int, k as c_int, n as c_int);
            if rc == 0 {
                Ok(())
            } else {
                Err(CudaError::Kernel(rc))
            }
        }
    };
}

cuda_packed_matvec!(
    matvec_q8_0_packed_f32,
    matvec_q8_0_packed_f32_batched,
    rsl_cuda_matvec_q8_0_packed_f32,
    rsl_cuda_matvec_q8_0_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_q4_k_packed_f32,
    matvec_q4_k_packed_f32_batched,
    rsl_cuda_matvec_q4_k_packed_f32,
    rsl_cuda_matvec_q4_k_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_q6_k_packed_f32,
    matvec_q6_k_packed_f32_batched,
    rsl_cuda_matvec_q6_k_packed_f32,
    rsl_cuda_matvec_q6_k_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_q5_k_packed_f32,
    matvec_q5_k_packed_f32_batched,
    rsl_cuda_matvec_q5_k_packed_f32,
    rsl_cuda_matvec_q5_k_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_q2_k_packed_f32,
    matvec_q2_k_packed_f32_batched,
    rsl_cuda_matvec_q2_k_packed_f32,
    rsl_cuda_matvec_q2_k_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_q8_k_packed_f32,
    matvec_q8_k_packed_f32_batched,
    rsl_cuda_matvec_q8_k_packed_f32,
    rsl_cuda_matvec_q8_k_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_q4_0_packed_f32,
    matvec_q4_0_packed_f32_batched,
    rsl_cuda_matvec_q4_0_packed_f32,
    rsl_cuda_matvec_q4_0_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_q5_0_packed_f32,
    matvec_q5_0_packed_f32_batched,
    rsl_cuda_matvec_q5_0_packed_f32,
    rsl_cuda_matvec_q5_0_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_q4_1_packed_f32,
    matvec_q4_1_packed_f32_batched,
    rsl_cuda_matvec_q4_1_packed_f32,
    rsl_cuda_matvec_q4_1_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_q5_1_packed_f32,
    matvec_q5_1_packed_f32_batched,
    rsl_cuda_matvec_q5_1_packed_f32,
    rsl_cuda_matvec_q5_1_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_iq4_nl_packed_f32,
    matvec_iq4_nl_packed_f32_batched,
    rsl_cuda_matvec_iq4_nl_packed_f32,
    rsl_cuda_matvec_iq4_nl_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_iq4_xs_packed_f32,
    matvec_iq4_xs_packed_f32_batched,
    rsl_cuda_matvec_iq4_xs_packed_f32,
    rsl_cuda_matvec_iq4_xs_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_iq2_xxs_packed_f32,
    matvec_iq2_xxs_packed_f32_batched,
    rsl_cuda_matvec_iq2_xxs_packed_f32,
    rsl_cuda_matvec_iq2_xxs_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_iq2_xs_packed_f32,
    matvec_iq2_xs_packed_f32_batched,
    rsl_cuda_matvec_iq2_xs_packed_f32,
    rsl_cuda_matvec_iq2_xs_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_iq2_s_packed_f32,
    matvec_iq2_s_packed_f32_batched,
    rsl_cuda_matvec_iq2_s_packed_f32,
    rsl_cuda_matvec_iq2_s_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_iq3_xxs_packed_f32,
    matvec_iq3_xxs_packed_f32_batched,
    rsl_cuda_matvec_iq3_xxs_packed_f32,
    rsl_cuda_matvec_iq3_xxs_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_iq3_s_packed_f32,
    matvec_iq3_s_packed_f32_batched,
    rsl_cuda_matvec_iq3_s_packed_f32,
    rsl_cuda_matvec_iq3_s_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_iq1_s_packed_f32,
    matvec_iq1_s_packed_f32_batched,
    rsl_cuda_matvec_iq1_s_packed_f32,
    rsl_cuda_matvec_iq1_s_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_iq1_m_packed_f32,
    matvec_iq1_m_packed_f32_batched,
    rsl_cuda_matvec_iq1_m_packed_f32,
    rsl_cuda_matvec_iq1_m_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_nvfp4_packed_f32,
    matvec_nvfp4_packed_f32_batched,
    rsl_cuda_matvec_nvfp4_packed_f32,
    rsl_cuda_matvec_nvfp4_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_mxfp4_packed_f32,
    matvec_mxfp4_packed_f32_batched,
    rsl_cuda_matvec_mxfp4_packed_f32,
    rsl_cuda_matvec_mxfp4_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_mxfp6_packed_f32,
    matvec_mxfp6_packed_f32_batched,
    rsl_cuda_matvec_mxfp6_packed_f32,
    rsl_cuda_matvec_mxfp6_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_mxfp8_packed_f32,
    matvec_mxfp8_packed_f32_batched,
    rsl_cuda_matvec_mxfp8_packed_f32,
    rsl_cuda_matvec_mxfp8_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_q3_k_packed_f32,
    matvec_q3_k_packed_f32_batched,
    rsl_cuda_matvec_q3_k_packed_f32,
    rsl_cuda_matvec_q3_k_packed_f32_batched
);
cuda_packed_matvec!(
    matvec_pq2_0_packed_f32,
    matvec_pq2_0_packed_f32_batched,
    rsl_cuda_matvec_pq2_0_packed_f32,
    rsl_cuda_matvec_pq2_0_packed_f32_batched
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
    stream: &CudaStream,
    hidden: *mut f32,
    branch: *const f32,
    w: *const f32,
    y_norm: *mut f32,
    n_rows: usize,
    d: usize,
    eps: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_add_rmsnorm_f32(
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
        Err(CudaError::Kernel(rc))
    }
}

/// RoPE in place over `qk` = [n_heads, head_dim] with precomputed
/// `inv_freq` = [head_dim/2], at absolute position `pos`.
///
/// SAFETY: `qk` and `inv_freq` are device pointers; `head_dim` even.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn rope_f32(
    stream: &CudaStream,
    qk: *mut f32,
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    inv_freq: *const f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_rope_f32(
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
        Err(CudaError::Kernel(rc))
    }
}

/// SwiGLU: `out[i] = silu(x[i]) * y[i]`, `n` elements.
///
/// SAFETY: `x`, `y`, `out` are device pointers of `n` f32.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn silu_mul_f32(
    stream: &CudaStream,
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_silu_mul_f32(stream.raw(), x, y, out, n as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Embedding lookup: `out[i,:] = table[ids[i],:]` (row < 0 ⇒ zeros).
///
/// SAFETY: `table` = [V,d], `out` = [n_ids,d], `ids` = [n_ids] i32 are all
/// device pointers on `stream`'s device.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn embedding_lookup_f32(
    stream: &CudaStream,
    table: *const f32,
    ids: *const c_int,
    out: *mut f32,
    n_ids: usize,
    d: usize,
) -> Result<(), CudaError> {
    let rc =
        rsl_cuda_embedding_lookup_f32(stream.raw(), table, ids, out, n_ids as c_int, d as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// FlashAttention decode (one query per head).
///
/// SAFETY: `q`/`out` = [n_heads,head_dim], `k`/`v` = [n_kv_heads,max_ctx,
/// head_dim] device pointers; `n_heads % n_kv_heads == 0`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_f32(
    stream: &CudaStream,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_decode_f32(
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
        Err(CudaError::Kernel(rc))
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
    stream: &CudaStream,
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
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_prefill_f32(
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
        Err(CudaError::Kernel(rc))
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
    stream: &CudaStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_decode_q4_0(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int, kv_len as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(CudaError::Kernel(rc)) }
}

/// Quantized-KV FlashAttention prefill for a Q4_0 KV cache. `q`/`out` are
/// F32 `[n_new, n_heads, head_dim]`; K/V packed as in [`flash_attn_decode_q4_0`].
///
/// SAFETY: device pointers as above; `n_heads % n_kv_heads == 0`,
/// `head_dim % 32 == 0`, `head_dim <= 256`, `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_q4_0(
    stream: &CudaStream,
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
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_prefill_q4_0(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int,
        kv_len_base as c_int, n_new as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(CudaError::Kernel(rc)) }
}

/// Quantized-KV FlashAttention decode for an NVFP4 KV cache. Layout as
/// [`flash_attn_decode_q4_0`] but `bytes_per_row = (head_dim/16)*9`.
///
/// SAFETY: device pointers as above; `head_dim % 16 == 0`, `head_dim <= 256`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_nvfp4(
    stream: &CudaStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_decode_nvfp4(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int, kv_len as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(CudaError::Kernel(rc)) }
}

/// Quantized-KV FlashAttention prefill for an NVFP4 KV cache.
///
/// SAFETY: device pointers as above; `head_dim % 16 == 0`, `head_dim <= 256`,
/// `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_nvfp4(
    stream: &CudaStream,
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
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_prefill_nvfp4(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int,
        kv_len_base as c_int, n_new as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(CudaError::Kernel(rc)) }
}

/// Quantized-KV FlashAttention decode for an MXFP4 KV cache. Layout as
/// [`flash_attn_decode_nvfp4`] but `bytes_per_row = (head_dim/32)*17`
/// (32 elems/block: 16 E2M1-nibble bytes + trailing E8M0 scale).
///
/// SAFETY: device pointers as above; `head_dim % 32 == 0`, `head_dim <= 256`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_mxfp4(
    stream: &CudaStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_decode_mxfp4(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int, kv_len as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(CudaError::Kernel(rc)) }
}

/// Quantized-KV FlashAttention prefill for an MXFP4 KV cache.
///
/// SAFETY: device pointers as above; `head_dim % 32 == 0`, `head_dim <= 256`,
/// `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_mxfp4(
    stream: &CudaStream,
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
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_prefill_mxfp4(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int,
        kv_len_base as c_int, n_new as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(CudaError::Kernel(rc)) }
}

/// Quantized-KV FlashAttention decode for an MXFP6 KV cache. Layout as
/// [`flash_attn_decode_nvfp4`] but `bytes_per_row = (head_dim/32)*25`
/// (32 elems/block: 24-byte LE bitstream of 6-bit E3M2 codes + E8M0 scale).
///
/// SAFETY: device pointers as above; `head_dim % 32 == 0`, `head_dim <= 256`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_mxfp6(
    stream: &CudaStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_decode_mxfp6(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int, kv_len as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(CudaError::Kernel(rc)) }
}

/// Quantized-KV FlashAttention prefill for an MXFP6 KV cache.
///
/// SAFETY: device pointers as above; `head_dim % 32 == 0`, `head_dim <= 256`,
/// `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_mxfp6(
    stream: &CudaStream,
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
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_prefill_mxfp6(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int,
        kv_len_base as c_int, n_new as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(CudaError::Kernel(rc)) }
}

/// Quantized-KV FlashAttention decode for an MXFP8 KV cache. Layout as
/// [`flash_attn_decode_nvfp4`] but `bytes_per_row = (head_dim/32)*33`
/// (32 elems/block: 32 E4M3 bytes + trailing E8M0 scale).
///
/// SAFETY: device pointers as above; `head_dim % 32 == 0`, `head_dim <= 256`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_mxfp8(
    stream: &CudaStream,
    q: *const f32,
    k_packed: *const c_void,
    v_packed: *const c_void,
    out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_decode_mxfp8(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int, kv_len as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(CudaError::Kernel(rc)) }
}

/// Quantized-KV FlashAttention prefill for an MXFP8 KV cache.
///
/// SAFETY: device pointers as above; `head_dim % 32 == 0`, `head_dim <= 256`,
/// `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_mxfp8(
    stream: &CudaStream,
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
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_prefill_mxfp8(
        stream.raw(), q, k_packed, v_packed, out, n_heads as c_int,
        n_kv_heads as c_int, head_dim as c_int, max_ctx as c_int,
        kv_len_base as c_int, n_new as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(CudaError::Kernel(rc)) }
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
    stream: &CudaStream,
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
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_decode_tq(
        stream.raw(), q, k_packed, v_packed, k_scales, v_scales, bits as c_int,
        out, n_heads as c_int, n_kv_heads as c_int, head_dim as c_int,
        max_ctx as c_int, kv_len as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(CudaError::Kernel(rc)) }
}

/// Quantized-KV FlashAttention prefill for a TurboQuant KV cache.
///
/// SAFETY: device pointers as above; `head_dim` a power of two `<= 256`,
/// `bits` in {1,2,4,8}, `kv_len_base + n_new <= max_ctx`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_prefill_tq(
    stream: &CudaStream,
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
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_prefill_tq(
        stream.raw(), q, k_packed, v_packed, k_scales, v_scales, bits as c_int,
        out, n_heads as c_int, n_kv_heads as c_int, head_dim as c_int,
        max_ctx as c_int, kv_len_base as c_int, n_new as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(CudaError::Kernel(rc)) }
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
    stream: &CudaStream,
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
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_decode_q8_0(
        stream.raw(), q, k_packed, v_packed, k_scales, v_scales, out,
        n_heads as c_int, n_kv_heads as c_int, head_dim as c_int,
        max_ctx as c_int, kv_len as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(CudaError::Kernel(rc)) }
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
    stream: &CudaStream,
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
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_prefill_q8_0(
        stream.raw(), q, k_packed, v_packed, k_scales, v_scales, out,
        n_heads as c_int, n_kv_heads as c_int, head_dim as c_int,
        max_ctx as c_int, kv_len_base as c_int, n_new as c_int,
    );
    if rc == 0 { Ok(()) } else { Err(CudaError::Kernel(rc)) }
}

/// Greedy argmax over `vocab` f32 logits; writes the chosen index to
/// `out_idx[0]` (lowest index on ties).
///
/// SAFETY: `logits` = [vocab] f32 and `out_idx` = [1] i32 device pointers.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn argmax_f32(
    stream: &CudaStream,
    logits: *const f32,
    vocab: usize,
    out_idx: *mut c_int,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_argmax_f32(stream.raw(), logits, vocab as c_int, out_idx);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

// ============================================================
// Device-resident packed-matvec cache (host-orchestrated dispatch)
// ============================================================
//
// The CUDA analogue of the SYCL layer's USM matvec dispatch. Because
// CUDA device memory is NOT host-accessible (unlike SYCL shared USM),
// weights cannot live in `rustllama-tensor`'s `Storage` and be read by
// both host and device — so the CUDA path keeps its OWN device-resident
// weight table here, keyed by the host pointer of the GGUF weight bytes
// (stable for the model's lifetime). Each weight is uploaded ONCE on
// first use and reused for every subsequent matvec (never re-uploaded
// per request — the model's weights stay resident for the process).
//
// This cache owns a stream and is used from a single lock holder at a
// time (the accel layer wraps it in a mutex): CUDA stream ops here are
// serialized, so concurrent forward passes are correct if slower. The
// dominant compute (packed matvecs) routes here; attention / norms stay
// on the proven CPU path in v1.

/// Which packed quant the CUDA matvec cache can dispatch. Mirrors the
/// `rsl_cuda_matvec_*_packed_f32` kernels. Variant names match the GGUF
/// quant names (and `PackedMatvecKind` in rustllama-models). Every arm
/// has a byte-exact CPU-parity port in `cuda/rsl_cuda.cu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum CudaPackedKind {
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
    // 3-bit K-quant (parity-gap close)
    Q3_K,
    // PrismML Bonsai 2-bit (parity-gap close)
    Pq2_0,
}

impl CudaPackedKind {
    /// K must be a multiple of this for the kernel's block layout.
    pub fn k_alignment(self) -> usize {
        match self {
            CudaPackedKind::Nvfp4 => 16,
            CudaPackedKind::Q8_0
            | CudaPackedKind::Q4_0
            | CudaPackedKind::Q5_0
            | CudaPackedKind::Q4_1
            | CudaPackedKind::Q5_1
            | CudaPackedKind::Iq4_Nl
            | CudaPackedKind::Mxfp4
            | CudaPackedKind::Mxfp6
            | CudaPackedKind::Mxfp8 => 32,
            CudaPackedKind::Ptq1_0 | CudaPackedKind::Pq2_0 => 128,
            CudaPackedKind::Q4_K
            | CudaPackedKind::Q6_K
            | CudaPackedKind::Q5_K
            | CudaPackedKind::Q2_K
            | CudaPackedKind::Q8_K
            | CudaPackedKind::Q3_K
            | CudaPackedKind::Iq4_Xs
            | CudaPackedKind::Iq2_Xxs
            | CudaPackedKind::Iq2_Xs
            | CudaPackedKind::Iq2_S
            | CudaPackedKind::Iq3_Xxs
            | CudaPackedKind::Iq3_S
            | CudaPackedKind::Iq1_S
            | CudaPackedKind::Iq1_M => 256,
        }
    }
    /// Bytes per row for a K-wide weight row in this quant's layout.
    pub fn row_bytes(self, k: usize) -> usize {
        match self {
            CudaPackedKind::Ptq1_0 => (k / 128) * 28,
            CudaPackedKind::Q8_0 => (k / 32) * 34,
            CudaPackedKind::Q4_K => (k / 256) * 144,
            CudaPackedKind::Q6_K => (k / 256) * 210,
            CudaPackedKind::Q5_K => (k / 256) * 176,
            CudaPackedKind::Q2_K => (k / 256) * 84,
            CudaPackedKind::Q8_K => (k / 256) * 292,
            CudaPackedKind::Q4_0 => (k / 32) * 18,
            CudaPackedKind::Q5_0 => (k / 32) * 22,
            CudaPackedKind::Q4_1 => (k / 32) * 20,
            CudaPackedKind::Q5_1 => (k / 32) * 24,
            CudaPackedKind::Iq4_Nl => (k / 32) * 18,
            CudaPackedKind::Iq4_Xs => (k / 256) * 136,
            CudaPackedKind::Iq2_Xxs => (k / 256) * 66,
            CudaPackedKind::Iq2_Xs => (k / 256) * 74,
            CudaPackedKind::Iq2_S => (k / 256) * 82,
            CudaPackedKind::Iq3_Xxs => (k / 256) * 98,
            CudaPackedKind::Iq3_S => (k / 256) * 110,
            CudaPackedKind::Iq1_S => (k / 256) * 50,
            CudaPackedKind::Iq1_M => (k / 256) * 56,
            CudaPackedKind::Nvfp4 => (k / 16) * 9,
            CudaPackedKind::Mxfp4 => (k / 32) * 17,
            CudaPackedKind::Mxfp6 => (k / 32) * 25,
            CudaPackedKind::Mxfp8 => (k / 32) * 33,
            CudaPackedKind::Q3_K => (k / 256) * 110,
            CudaPackedKind::Pq2_0 => (k / 128) * 34,
        }
    }
}

/// An owned device allocation (raw pointer + capacity), freed on drop
/// via its owning stream. Not tied to the stream by lifetime so it can
/// be cached alongside the stream inside [`CudaMatvecCache`]; the cache
/// declares its buffers BEFORE the stream so they drop first.
struct RawDevBuf {
    ptr: *mut c_void,
    cap_bytes: usize,
    raw_stream: *mut RslCudaStreamRaw,
}

impl RawDevBuf {
    fn alloc(stream: &CudaStream, n_bytes: usize) -> Option<Self> {
        if n_bytes == 0 {
            return None;
        }
        // SAFETY: FFI; null on failure.
        let ptr = unsafe { rsl_cuda_malloc_device(stream.raw(), n_bytes as u64) };
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
            rsl_cuda_memcpy_h2d(
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
            rsl_cuda_memcpy_d2h(
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
        unsafe { rsl_cuda_free(self.raw_stream, self.ptr) };
    }
}

/// Process-/model-lifetime device-resident weight cache + scratch for
/// the packed-matvec CUDA dispatch. See the module comment above.
pub struct CudaMatvecCache {
    // Declared before `stream` so buffers drop (and free) first.
    weights: HashMap<usize, RawDevBuf>,
    x_scratch: Option<RawDevBuf>,
    out_scratch: Option<RawDevBuf>,
    used_bytes: usize,
    budget_bytes: usize,
    stream: CudaStream,
}

impl CudaMatvecCache {
    /// Create a cache on `device_index` with a device-memory budget for
    /// cached weights (bytes). `None` when no such device / stream fails.
    pub fn new(device_index: u32, budget_bytes: usize) -> Option<Self> {
        let stream = CudaStream::create(device_index)?;
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

    fn ensure_scratch(slot: &mut Option<RawDevBuf>, stream: &CudaStream, need_bytes: usize) -> bool {
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
    /// launches the kernel, copies `out` D2H. Returns false (caller
    /// falls back to CPU) on any bad shape / budget / kernel failure —
    /// leaving `out` untouched on failure.
    pub fn matvec_packed(
        &mut self,
        kind: CudaPackedKind,
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
        // SAFETY: w/x/out are live device buffers on `self.stream` sized
        // for (M,K); the wrapper synchronizes before returning.
        let res = unsafe {
            match kind {
                CudaPackedKind::Ptq1_0 => matvec_ptq1_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Q8_0 => matvec_q8_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Q4_K => matvec_q4_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Q6_K => matvec_q6_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Q5_K => matvec_q5_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Q2_K => matvec_q2_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Q8_K => matvec_q8_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Q4_0 => matvec_q4_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Q5_0 => matvec_q5_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Q4_1 => matvec_q4_1_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Q5_1 => matvec_q5_1_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Iq4_Nl => matvec_iq4_nl_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Iq4_Xs => matvec_iq4_xs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Iq2_Xxs => matvec_iq2_xxs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Iq2_Xs => matvec_iq2_xs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Iq2_S => matvec_iq2_s_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Iq3_Xxs => matvec_iq3_xxs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Iq3_S => matvec_iq3_s_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Iq1_S => matvec_iq1_s_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Iq1_M => matvec_iq1_m_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Nvfp4 => matvec_nvfp4_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Mxfp4 => matvec_mxfp4_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Mxfp6 => matvec_mxfp6_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Mxfp8 => matvec_mxfp8_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Q3_K => matvec_q3_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
                CudaPackedKind::Pq2_0 => matvec_pq2_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k),
            }
        };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// Batched packed matvec over `n` input rows. `x` = [N,K] row-major,
    /// `out` = [N,M] row-major. Same semantics as [`Self::matvec_packed`].
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_packed_batched(
        &mut self,
        kind: CudaPackedKind,
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
                CudaPackedKind::Ptq1_0 => matvec_ptq1_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Q8_0 => matvec_q8_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Q4_K => matvec_q4_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Q6_K => matvec_q6_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Q5_K => matvec_q5_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Q2_K => matvec_q2_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Q8_K => matvec_q8_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Q4_0 => matvec_q4_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Q5_0 => matvec_q5_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Q4_1 => matvec_q4_1_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Q5_1 => matvec_q5_1_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Iq4_Nl => matvec_iq4_nl_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Iq4_Xs => matvec_iq4_xs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Iq2_Xxs => matvec_iq2_xxs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Iq2_Xs => matvec_iq2_xs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Iq2_S => matvec_iq2_s_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Iq3_Xxs => matvec_iq3_xxs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Iq3_S => matvec_iq3_s_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Iq1_S => matvec_iq1_s_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Iq1_M => matvec_iq1_m_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Nvfp4 => matvec_nvfp4_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Mxfp4 => matvec_mxfp4_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Mxfp6 => matvec_mxfp6_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Mxfp8 => matvec_mxfp8_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Q3_K => matvec_q3_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
                CudaPackedKind::Pq2_0 => matvec_pq2_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n),
            }
        };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// Blackwell FP4 tensor-core batched GEMM (W4A4) — the TC analogue of
    /// [`matvec_packed_batched`](Self::matvec_packed_batched) for NVFP4/MXFP4
    /// weights, routed to the block-scaled `mma.sync` path. Same device-buffer
    /// management (weight uploaded once keyed by `weight_key`, `x` H2D, `out`
    /// D2H) and the SAME `out[n*M + m]` layout, so it is a drop-in for the
    /// scalar batched path when `blackwell_tc_available()`. Returns `false`
    /// (caller falls back to the scalar path) on bad shape / `K % 64 != 0` /
    /// budget / kernel failure, leaving `out` untouched on failure.
    pub fn gemm_fp4_tc(
        &mut self,
        kind: CudaFp4TcKind,
        weight_key: usize,
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        // FP4 block geometry: NVFP4 = K/16 × 9B (per-16 E4M3); MXFP4 = K/32 ×
        // 17B (per-32 E8M0). The TC atom also needs K % 64 == 0.
        let (kalign, row_bytes) = match kind {
            CudaFp4TcKind::Nvfp4 => (16usize, (k / 16) * 9),
            CudaFp4TcKind::Mxfp4 => (32usize, (k / 32) * 17),
        };
        if m == 0 || k == 0 || n == 0 || x.len() != n * k || out.len() != n * m {
            return false;
        }
        if k % 64 != 0 || k % kalign != 0 || w_bytes.len() < m * row_bytes {
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
        // SAFETY: w (M×K FP4 blocks), x (N·K f32), out (N·M f32) are live
        // device buffers on `self.stream`; the wrapper synchronizes.
        let res = unsafe { gemm_fp4_tc_f32(kind, &self.stream, w_ptr, x_ptr, out_ptr, m, n, k) };
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
// mutex). The raw device pointers never alias host memory.
unsafe impl Send for CudaMatvecCache {}
