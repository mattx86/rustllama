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

/// T4: FNV-1a-64 hash (hex) of this crate's kernel sources, computed by
/// `build.rs` and injected via `cargo:rustc-env=RSL_CUDA_KERNEL_HASH`. Folded
/// (with the SYCL/MLX twins) into `rustllama_tuner::kernel_build_id()` so a CUDA
/// kernel edit re-triggers the first-load verdict self-heal.
pub const KERNEL_HASH: &str = env!("RSL_CUDA_KERNEL_HASH");

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
    // P17: Gated-DeltaNet chunked-prefill recurrence over all V-heads (host-copy).
    fn rsl_cuda_delta_net_chunk_f32(
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
    fn rsl_cuda_stream_create(device_index: c_int) -> *mut RslCudaStreamRaw;
    fn rsl_cuda_stream_destroy(s: *mut RslCudaStreamRaw);
    // Free the stream's cached Q8_0 prefill-GEMM weight repacks (model-reload hook).
    fn rsl_cuda_q8_0_repack_reset(s: *mut RslCudaStreamRaw);
    fn rsl_cuda_malloc_from_host(
        s: *mut RslCudaStreamRaw,
        src: *const c_void,
        n_bytes: u64,
    ) -> *mut c_void;
    fn rsl_cuda_malloc_device(s: *mut RslCudaStreamRaw, n_bytes: u64) -> *mut c_void;
    fn rsl_cuda_free(s: *mut RslCudaStreamRaw, dev_ptr: *mut c_void);
    // Device-indexed managed-memory alloc/free (MoE tiered tier; stream-free).
    fn rsl_cuda_malloc_managed(device: c_int, n_bytes: u64) -> *mut c_void;
    fn rsl_cuda_free_managed(device: c_int, ptr: *mut c_void);
    // Prefetch a managed allocation onto the GPU (avoids lazy fault-migration
    // during decode) + a device-wide sync to drain the async prefetches.
    fn rsl_cuda_prefetch_managed(device: c_int, ptr: *const c_void, n_bytes: u64) -> c_int;
    fn rsl_cuda_device_sync(device: c_int) -> c_int;
    // Grouped routed-expert FFN (MoE tiered-expert decode fast path): the whole
    // top-k expert FFN for one token in one on-device pass. One entry per
    // warp-cooperative quant (Q4_K / Q8_0 / Q6_K); identical signature.
    #[allow(clippy::too_many_arguments)]
    fn rsl_cuda_moe_ffn_grouped_q4k(
        s: *mut RslCudaStreamRaw,
        gate_ptrs: *const *const c_void,
        up_ptrs: *const *const c_void,
        down_ptrs: *const *const c_void,
        weights: *const f32,
        x: *const f32,
        ff: *mut f32,
        out: *mut f32,
        n_pick: c_int,
        d_model: c_int,
        d_ff: c_int,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn rsl_cuda_moe_ffn_grouped_q8_0(
        s: *mut RslCudaStreamRaw,
        gate_ptrs: *const *const c_void,
        up_ptrs: *const *const c_void,
        down_ptrs: *const *const c_void,
        weights: *const f32,
        x: *const f32,
        ff: *mut f32,
        out: *mut f32,
        n_pick: c_int,
        d_model: c_int,
        d_ff: c_int,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn rsl_cuda_moe_ffn_grouped_q6_k(
        s: *mut RslCudaStreamRaw,
        gate_ptrs: *const *const c_void,
        up_ptrs: *const *const c_void,
        down_ptrs: *const *const c_void,
        weights: *const f32,
        x: *const f32,
        ff: *mut f32,
        out: *mut f32,
        n_pick: c_int,
        d_model: c_int,
        d_ff: c_int,
    ) -> c_int;
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
    // CUDA graph capture for the device-resident DECODE attention block
    // (Feature 1b). Async memcpy variants + stream sync + graph lifecycle; every
    // entry returns a status and the Rust caller falls back to the eager block on
    // any non-zero return. Graph/exec handles are opaque `*mut c_void`.
    fn rsl_cuda_memcpy_h2d_async(
        s: *mut RslCudaStreamRaw,
        dst_dev: *mut c_void,
        src_host: *const c_void,
        n_bytes: u64,
    ) -> c_int;
    fn rsl_cuda_memcpy_d2h_async(
        s: *mut RslCudaStreamRaw,
        dst_host: *mut c_void,
        src_dev: *const c_void,
        n_bytes: u64,
    ) -> c_int;
    fn rsl_cuda_memcpy_d2d_async(
        s: *mut RslCudaStreamRaw,
        dst_dev: *mut c_void,
        src_dev: *const c_void,
        n_bytes: u64,
    ) -> c_int;
    // PP-0 multi-GPU transport: enable P2P + cross-device peer copy.
    fn rsl_cuda_enable_peer_access(accessing_dev: c_int, peer_dev: c_int) -> c_int;
    fn rsl_cuda_memcpy_peer(
        s: *mut RslCudaStreamRaw,
        dst: *mut c_void,
        dst_dev: c_int,
        src: *const c_void,
        src_dev: c_int,
        n_bytes: u64,
    ) -> c_int;
    fn rsl_cuda_stream_sync(s: *mut RslCudaStreamRaw) -> c_int;
    // C1 — resident-state single-token Gated-DeltaNet recurrence step. All pointers
    // are DEVICE pointers; `state` is the RESIDENT per-layer S (caller-owned), the
    // rest are the uploaded per-token I/O. Declines (nonzero) when qk != vd.
    fn rsl_cuda_delta_net_step_f32(
        s: *mut RslCudaStreamRaw,
        qkv_conv: *const f32,
        gate: *const f32,
        decay: *const f32,
        beta: *const f32,
        ssm_norm: *const f32,
        state: *mut f32,
        out: *mut f32,
        n_v_heads: c_int,
        n_qk_heads: c_int,
        qk: c_int,
        vd: c_int,
        q_block: c_int,
        k_block: c_int,
        rms_eps: f32,
    ) -> c_int;
    fn rsl_cuda_graph_begin_capture(s: *mut RslCudaStreamRaw) -> c_int;
    fn rsl_cuda_graph_end_capture(s: *mut RslCudaStreamRaw, out_graph: *mut *mut c_void) -> c_int;
    fn rsl_cuda_graph_instantiate(graph: *mut c_void, out_exec: *mut *mut c_void) -> c_int;
    fn rsl_cuda_graph_exec_update(exec: *mut c_void, graph: *mut c_void) -> c_int;
    fn rsl_cuda_graph_launch(exec: *mut c_void, s: *mut RslCudaStreamRaw) -> c_int;
    fn rsl_cuda_graph_destroy(graph: *mut c_void);
    fn rsl_cuda_graph_exec_destroy(exec: *mut c_void);
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
    fn rsl_cuda_matvec_q8_0_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q8_0_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q4_k_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q4_k_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    // Q4_K prefill GEMM (Phase 1): tiled, shared-mem weight REUSE, f32 accumulate
    // (bit-exact). out[N,M] = x[N,K] . W[M,K]^T. Selected by accel.rs on the
    // prefill path via the `gemm:q4_k_f32` verdict.
    fn rsl_cuda_gemm_q4_k_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    // Q8_0 prefill GEMM: tiled, shared-mem weight REUSE, f32 accumulate (bit-exact,
    // LOSSLESS — the Q8_0 analogue of `rsl_cuda_gemm_q4_k_f32`). out[N,M] = x[N,K] .
    // W[M,K]^T. Selected by accel.rs on the prefill path via `gemm:q8_0_f32`.
    fn rsl_cuda_gemm_q8_0_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    // Q4_K prefill GEMM (Phase 2): int8 tensor-core W8A8 (lossy). Same shape;
    // selected via the `gemm:q4_k_w8a8_tc` verdict.
    fn rsl_cuda_gemm_q4_k_w8a8_tc(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    // Q8_0 prefill GEMM (T3c): int8 tensor-core W8A8 (lossy). Same shape; selected
    // via the `gemm:q8_0_w8a8_tc` verdict. K % 32 == 0.
    fn rsl_cuda_gemm_q8_0_w8a8_tc(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    // Q4_K W4A8 DECODE matvec (single-row, lossy int8-activation). Selected via
    // the `matvec:q4_k_w4a8` verdict (correct + perf-gated vs the f32 matvec).
    fn rsl_cuda_matvec_q4_k_w4a8(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    // Q8_0 W8A8 DECODE matvec (single-row, lossy int8-activation; T3a). Selected
    // via the `matvec:q8_0_w8a8` verdict (correct + perf-gated vs the f32 matvec).
    fn rsl_cuda_matvec_q8_0_w8a8(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    // Aligned DECODE matvec over the cached Q8_0 repack (lookup-only, -1 if the
    // repack isn't cached → caller runs the byte-packed matvec).
    fn rsl_cuda_matvec_q8_0_repacked(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    // Q4_0 / Q5_0 W8A8 DECODE matvecs (single-row, lossy int8-activation; T3b).
    // Selected via `matvec:q4_0_w8a8` / `matvec:q5_0_w8a8` (correct + perf-gated).
    fn rsl_cuda_matvec_q4_0_w8a8(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q5_0_w8a8(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q6_k_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q6_k_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;

    // Additional packed matvecs (K-quants Q5_K/Q2_K/Q8_K, legacy quants
    // Q4_0/Q5_0/Q4_1/Q5_1, IQ family, NVFP4). One thread per output row.
    fn rsl_cuda_matvec_q5_k_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q5_k_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q2_k_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q2_k_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q8_k_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q8_k_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q4_0_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q4_0_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q5_0_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q5_0_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q4_1_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q4_1_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q5_1_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q5_1_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq4_nl_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq4_nl_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq4_xs_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq4_xs_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq2_xxs_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq2_xxs_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq2_xs_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq2_xs_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq2_s_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq2_s_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq3_xxs_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq3_xxs_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq3_s_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq3_s_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq1_s_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq1_s_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq1_m_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq1_m_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_nvfp4_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_nvfp4_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    // OCP Microscaling FP4/FP6/FP8. 32-elem blocks + trailing E8M0 scale
    // byte; K%32==0. MXFP4: 17 B, MXFP6: 25 B, MXFP8: 33 B per block.
    fn rsl_cuda_matvec_mxfp4_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_mxfp4_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_mxfp6_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_mxfp6_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_mxfp8_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_mxfp8_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    // Q3_K + PQ2_0 (parity-gap close). Q3_K: 110 B/256, K%256==0.
    // PQ2_0: 34 B/128, K%128==0.
    fn rsl_cuda_matvec_q3_k_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q3_k_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_pq2_0_packed_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_pq2_0_packed_f32_batched(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    ) -> c_int;

    // Fused gate+up matvec (decode): one launch computes gate_out + up_out for
    // the FFN. gw/uw = packed gate/up weights [M,K]; gout/uout = [M] device ptrs.
    fn rsl_cuda_matvec_q8_0_packed_f32_gate_up_fused(
        s: *mut RslCudaStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q4_k_packed_f32_gate_up_fused(
        s: *mut RslCudaStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q6_k_packed_f32_gate_up_fused(
        s: *mut RslCudaStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_q5_k_packed_f32_gate_up_fused(
        s: *mut RslCudaStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq4_nl_packed_f32_gate_up_fused(
        s: *mut RslCudaStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq4_xs_packed_f32_gate_up_fused(
        s: *mut RslCudaStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq1_s_packed_f32_gate_up_fused(
        s: *mut RslCudaStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq1_m_packed_f32_gate_up_fused(
        s: *mut RslCudaStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq2_xxs_packed_f32_gate_up_fused(
        s: *mut RslCudaStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq2_xs_packed_f32_gate_up_fused(
        s: *mut RslCudaStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq2_s_packed_f32_gate_up_fused(
        s: *mut RslCudaStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq3_xxs_packed_f32_gate_up_fused(
        s: *mut RslCudaStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_iq3_s_packed_f32_gate_up_fused(
        s: *mut RslCudaStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_matvec_ptq1_0_packed_f32_gate_up_fused(
        s: *mut RslCudaStreamRaw,
        gw: *const c_void,
        uw: *const c_void,
        x: *const f32,
        gout: *mut f32,
        uout: *mut f32,
        m: c_int,
        k: c_int,
    ) -> c_int;

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
    // Device-resident PREFILL block: batched RoPE (per-row position base_pos+row)
    // + head-major KV mirror scatter of n_new token-major rows.
    fn rsl_cuda_rope_rows_f32(
        s: *mut RslCudaStreamRaw,
        qk: *mut f32,
        n_heads: c_int,
        head_dim: c_int,
        base_pos: c_int,
        n_rows: c_int,
        inv_freq: *const f32,
    ) -> c_int;
    fn rsl_cuda_kv_scatter_rows_f32(
        s: *mut RslCudaStreamRaw,
        mirror: *mut f32,
        src: *const f32,
        n_new: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        max_ctx: c_int,
        col_base: c_int,
    ) -> c_int;
    // Stream-based device-resident RMSNorm + Qwen3 per-head q/k RMSNorm (#2
    // activation residency): device pointers, no internal round-trip.
    fn rsl_cuda_rmsnorm_dev_f32(
        s: *mut RslCudaStreamRaw,
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        n_rows: c_int,
        d: c_int,
        eps: f32,
    ) -> c_int;
    fn rsl_cuda_qk_head_norm_f32(
        s: *mut RslCudaStreamRaw,
        buf: *mut f32,
        w: *const f32,
        n_heads: c_int,
        head_dim: c_int,
        eps: f32,
    ) -> c_int;
    fn rsl_cuda_silu_mul_f32(
        s: *mut RslCudaStreamRaw,
        x: *const f32,
        y: *const f32,
        out: *mut f32,
        n: c_int,
    ) -> c_int;
    // P6-fusions (write-blind mirrors of the SYCL fused f32 kernels).
    fn rsl_cuda_rmsnorm_matvec_f32(
        s: *mut RslCudaStreamRaw,
        x: *const f32,
        wnorm: *const f32,
        wmat: *const f32,
        y: *mut f32,
        n_rows: c_int,
        k: c_int,
        m: c_int,
        eps: f32,
    ) -> c_int;
    fn rsl_cuda_down_residual_f32(
        s: *mut RslCudaStreamRaw,
        ff: *const f32,
        wmat: *const f32,
        resid: *const f32,
        y: *mut f32,
        n_rows: c_int,
        k: c_int,
        m: c_int,
    ) -> c_int;
    fn rsl_cuda_add_f32(
        s: *mut RslCudaStreamRaw,
        hidden: *mut f32,
        branch: *const f32,
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
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
    ) -> c_int;
    // GPU paged attention (Phase B): block-table flash-decode against the
    // whole-pool device mirror. `block_table` = [ceil(kv_len/page_size)] i32.
    fn rsl_cuda_flash_attn_decode_paged_f32(
        s: *mut RslCudaStreamRaw,
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
        window: c_int,
        softcap: f32,
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
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
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
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
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
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
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
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
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
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
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
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
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
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
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
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
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
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
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
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
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
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
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
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
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
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
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
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
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
        // SWA window (0=full causal) + Gemma2 attention-logit soft-cap (0=off).
        window: c_int,
        softcap: f32,
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
    // Phase 5: 2:4 structured-sparse FP8 GEMM. W = dense E4M3 (1 B/elem),
    // magnitude-pruned to 2:4 on-device + mma.sp. OPT-IN (quality caveat).
    fn rsl_cuda_gemm_fp8_sp24_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> c_int;

    // Hopper (sm_90a) FP8 `wgmma.mma_async` tensor-core GEMM (MXFP8 W8A8,
    // m64n16k32, K % 32 == 0) + a TMA-staged variant. Same (M,N,K)/return
    // contract as the Blackwell GEMMs; return -2 when the Hopper TC path is
    // unavailable so the caller falls back to the scalar path. WRITE-BLIND:
    // compile-only — the project has no GH200 to runtime-validate these.
    fn rsl_cuda_hopper_tc_available(dev: c_int) -> c_int;
    fn rsl_cuda_gemm_mxfp8_wgmma_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> c_int;
    fn rsl_cuda_gemm_mxfp8_wgmma_tma_f32(
        s: *mut RslCudaStreamRaw,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> c_int;

    fn rsl_cuda_consume_error_count() -> c_int;
    fn rsl_cuda_clear_error();
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
    let rc =
        rsl_cuda_gemm_mxfp8_tc_f32(stream.raw(), w, x, out, m as c_int, n as c_int, k as c_int);
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
    let rc =
        rsl_cuda_gemm_mxfp6_tc_f32(stream.raw(), w, x, out, m as c_int, n as c_int, k as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Whether this binary compiled the Hopper sm_90a FP8 `wgmma` tensor-core GEMM
/// path (an sm_90x `a`/`f` arch was in `RUSTLLAMA_CUDA_ARCHS`) AND device `dev`
/// is a real Hopper (sm_90) GPU. `false` on every non-Hopper host and on a
/// default (Ampere→Hopper, non-accelerated) build. WRITE-BLIND: never validated
/// at runtime by this project (no GH200) — a GH200 owner confirms via
/// `doctor --cuda-parity`.
pub fn hopper_tc_available(dev: u32) -> bool {
    // SAFETY: no pointer args; the shim queries cudaDeviceProp internally.
    unsafe { rsl_cuda_hopper_tc_available(dev as c_int) == 1 }
}

/// Hopper MXFP8 (W8A8) FP8 `wgmma` tensor-core GEMM: `out[N×M] = X[N×K] · W[M×K]ᵀ`,
/// W in MXFP8 (per-32 E8M0, E4M3 elements), X f32 (quantized to E4M3 on the fly).
/// `K % 32 == 0`. `tma` selects the `cp.async.bulk`-staged variant.
/// `Err(Kernel(-2))` when the Hopper TC path is unavailable (caller falls back
/// to the scalar path). WRITE-BLIND: compile-only, no GH200 to validate.
///
/// # Safety
/// `w` (M rows of MXFP8 blocks), `x` (N·K f32), `out` (N·M f32) must be live
/// device buffers on `stream`.
pub unsafe fn gemm_mxfp8_wgmma_f32(
    stream: &CudaStream,
    w: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    tma: bool,
) -> Result<(), CudaError> {
    let rc = if tma {
        rsl_cuda_gemm_mxfp8_wgmma_tma_f32(
            stream.raw(),
            w,
            x,
            out,
            m as c_int,
            n as c_int,
            k as c_int,
        )
    } else {
        rsl_cuda_gemm_mxfp8_wgmma_f32(stream.raw(), w, x, out, m as c_int, n as c_int, k as c_int)
    };
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

/// Blackwell 2:4 structured-sparse FP8 tensor-core GEMM. `w` is a DENSE E4M3
/// weight (1 byte/elem, M×K, row-major); the kernel magnitude-prunes each
/// 4-group to 2:4 and runs `mma.sp`. `K % 64 == 0`.
///
/// QUALITY CAVEAT: 2:4 magnitude pruning drops 50% of weights per 4-group — a
/// no-op only for 2:4-trained models, otherwise a silent accuracy loss. Use
/// only for sparsity-aware models or accepted-loss scenarios.
///
/// # Safety
/// `w` (M·K E4M3 bytes), `x` (N·K f32), `out` (N·M f32) must be live device
/// buffers on `stream`.
pub unsafe fn gemm_fp8_sp24_f32(
    stream: &CudaStream,
    w: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), CudaError> {
    let rc =
        rsl_cuda_gemm_fp8_sp24_f32(stream.raw(), w, x, out, m as c_int, n as c_int, k as c_int);
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
        rsl_cuda_rmsnorm_f32(
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
        rsl_cuda_matvec_f32(
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
        Err(CudaError::Kernel(rc))
    }
}

/// P17: Gated-DeltaNet chunked-prefill recurrence over ALL V-heads (host-copy).
/// WRITE-BLIND CUDA mirror of the SYCL `delta_net_chunk_f32_usm_raw` + the CPU
/// chunked reference. Head-major: `q`/`k` are `[n_heads*seq_len*head_qk_dim]`,
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
) -> Result<(), CudaError> {
    assert_eq!(q.len(), n_heads * seq_len * head_qk_dim);
    assert_eq!(k.len(), n_heads * seq_len * head_qk_dim);
    assert_eq!(v.len(), n_heads * seq_len * head_v_dim);
    assert_eq!(g.len(), n_heads * seq_len);
    assert_eq!(beta.len(), n_heads * seq_len);
    assert_eq!(state.len(), n_heads * head_qk_dim * head_v_dim);
    assert_eq!(out.len(), n_heads * seq_len * head_v_dim);
    // SAFETY: slices outlive the call; lengths are checked above.
    let rc = unsafe {
        rsl_cuda_delta_net_chunk_f32(
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
        Err(CudaError::Kernel(rc))
    }
}

// ============================================================
// Phase profiler (env `RUSTLLAMA_PHASE_PROFILE=1`): host-side wall-time split of
// the matvec ladder into H2D(x) / kernel / D2H(out), keyed by `single` (decode
// single-row matvec) vs `batched` (prefill GEMM). This localizes the host-in/
// host-out overhead — whether forward time is transfers vs compute. Disabled =
// one relaxed-atomic bool load, no timing. Reported as us/call by `model bench`.
// ============================================================
pub mod phase_prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    pub fn enabled() -> bool {
        *EN.get_or_init(|| std::env::var("RUSTLLAMA_PHASE_PROFILE").is_ok())
    }
    static S_H2D: AtomicU64 = AtomicU64::new(0);
    static S_KER: AtomicU64 = AtomicU64::new(0);
    static S_D2H: AtomicU64 = AtomicU64::new(0);
    static S_CALLS: AtomicU64 = AtomicU64::new(0);
    static B_H2D: AtomicU64 = AtomicU64::new(0);
    static B_KER: AtomicU64 = AtomicU64::new(0);
    static B_D2H: AtomicU64 = AtomicU64::new(0);
    static B_CALLS: AtomicU64 = AtomicU64::new(0);
    // Prefill GEMM total (the USM/TC batched path `try_matvec_tensor_batched_usm_f32`,
    // which the batched matvec_packed buckets above do NOT see): whole-call wall
    // time incl. its internal transfers+kernel.
    static PG_NS: AtomicU64 = AtomicU64::new(0);
    static PG_CALLS: AtomicU64 = AtomicU64::new(0);
    /// Record one prefill USM/TC GEMM call's total nanoseconds.
    pub fn add_prefill_gemm(ns: u64) {
        PG_NS.fetch_add(ns, Ordering::Relaxed);
        PG_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    // Device-resident PREFILL block (`prefill_attn_block_f32`) per-LAYER phase
    // split: hidden/norm H2D, on-device matmul+glue compute (rmsnorm/QKV/qk-norm/
    // RoPE/O/residual/FFN), attention (KV-scatter + flash-prefill), K/V + hidden
    // D2H. Isolated with a stream sync per phase (perturbs absolute time, but the
    // RATIO is the signal) — the device block bypasses the matvec buckets above.
    static PB_H2D: AtomicU64 = AtomicU64::new(0);
    static PB_COMPUTE: AtomicU64 = AtomicU64::new(0);
    static PB_ATTN: AtomicU64 = AtomicU64::new(0);
    static PB_D2H: AtomicU64 = AtomicU64::new(0);
    static PB_CALLS: AtomicU64 = AtomicU64::new(0);
    // Sub-split of PB_COMPUTE (which lumps the 7 batched GEMMs + the glue kernels:
    // rmsnorm / qk-norm / rope / silu_mul / add). PB_GEMM = all 7 projection+FFN
    // GEMMs (qkv/o/gate/up/down); PB_FFN = the 3 FFN GEMMs alone (gate/up/down).
    // glue = compute - gemm. Lets the ncu-free profiler say whether the prefill
    // compute is GEMM-bound (→ the ncu-gated GEMM wall) or glue-bound (→ kernel
    // fusion) and whether the FFN or the attn projections dominate the GEMM time.
    static PB_GEMM: AtomicU64 = AtomicU64::new(0);
    static PB_FFN: AtomicU64 = AtomicU64::new(0);
    /// Record one device-resident prefill-block layer's H2D/compute/attn/D2H ns,
    /// plus the GEMM (all 7) and FFN-GEMM (gate/up/down) sub-slices of compute.
    pub fn add_prefill_block(h2d: u64, compute: u64, attn: u64, d2h: u64, gemm: u64, ffn: u64) {
        PB_H2D.fetch_add(h2d, Ordering::Relaxed);
        PB_COMPUTE.fetch_add(compute, Ordering::Relaxed);
        PB_ATTN.fetch_add(attn, Ordering::Relaxed);
        PB_D2H.fetch_add(d2h, Ordering::Relaxed);
        PB_GEMM.fetch_add(gemm, Ordering::Relaxed);
        PB_FFN.fetch_add(ffn, Ordering::Relaxed);
        PB_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    // Device-resident DECODE block (`decode_attn_block_f32`, eager path —
    // RUSTLLAMA_DEV_STEP_GRAPH=0 + dev-attn on) per-LAYER phase split: the 7
    // matvecs (qkv/o/gate/up/down), the flash-decode attention (+ mirror write),
    // the glue (rmsnorm/qk-norm/rope/silu/residual), and H2D/D2H. Localizes the
    // decode bottleneck (matvec bandwidth vs attn vs the many-small-kernels glue).
    static DB_MATVEC: AtomicU64 = AtomicU64::new(0);
    static DB_ATTN: AtomicU64 = AtomicU64::new(0);
    static DB_GLUE: AtomicU64 = AtomicU64::new(0);
    static DB_XFER: AtomicU64 = AtomicU64::new(0);
    static DB_CALLS: AtomicU64 = AtomicU64::new(0);
    /// Record one eager decode-block layer's matvec/attn/glue/transfer ns.
    pub fn add_decode_block(matvec: u64, attn: u64, glue: u64, xfer: u64) {
        DB_MATVEC.fetch_add(matvec, Ordering::Relaxed);
        DB_ATTN.fetch_add(attn, Ordering::Relaxed);
        DB_GLUE.fetch_add(glue, Ordering::Relaxed);
        DB_XFER.fetch_add(xfer, Ordering::Relaxed);
        DB_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    /// Record one decode (single-row) matvec's H2D/kernel/D2H nanoseconds.
    pub fn add_single(h2d: u64, ker: u64, d2h: u64) {
        S_H2D.fetch_add(h2d, Ordering::Relaxed);
        S_KER.fetch_add(ker, Ordering::Relaxed);
        S_D2H.fetch_add(d2h, Ordering::Relaxed);
        S_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    /// Record one prefill (batched) GEMM's H2D/kernel/D2H nanoseconds.
    pub fn add_batched(h2d: u64, ker: u64, d2h: u64) {
        B_H2D.fetch_add(h2d, Ordering::Relaxed);
        B_KER.fetch_add(ker, Ordering::Relaxed);
        B_D2H.fetch_add(d2h, Ordering::Relaxed);
        B_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    pub fn reset() {
        for c in [
            &S_H2D,
            &S_KER,
            &S_D2H,
            &S_CALLS,
            &B_H2D,
            &B_KER,
            &B_D2H,
            &B_CALLS,
            &PG_NS,
            &PG_CALLS,
            &PB_H2D,
            &PB_COMPUTE,
            &PB_ATTN,
            &PB_D2H,
            &PB_GEMM,
            &PB_FFN,
            &PB_CALLS,
            &DB_MATVEC,
            &DB_ATTN,
            &DB_GLUE,
            &DB_XFER,
            &DB_CALLS,
        ] {
            c.store(0, Ordering::Relaxed);
        }
    }
    /// Formatted us-per-call breakdown (empty string if nothing recorded).
    pub fn dump() -> String {
        let ld = |c: &AtomicU64| c.load(Ordering::Relaxed);
        let (sc, bc, pg, pb) = (ld(&S_CALLS), ld(&B_CALLS), ld(&PG_CALLS), ld(&PB_CALLS));
        let db = ld(&DB_CALLS);
        if sc == 0 && bc == 0 && pg == 0 && pb == 0 && db == 0 {
            return String::new();
        }
        let per = |n: u64, calls: u64| {
            if calls > 0 {
                n as f64 / 1e3 / calls as f64
            } else {
                0.0
            }
        };
        let mut s = String::from("[phase] matvec H2D / kernel / D2H, us per call:\n");
        if db > 0 {
            s += &format!(
                "  dev-decode block (per layer): calls={db} matvec={:.1} attn={:.1} glue={:.1} xfer={:.1} total={:.1}\n",
                per(ld(&DB_MATVEC), db),
                per(ld(&DB_ATTN), db),
                per(ld(&DB_GLUE), db),
                per(ld(&DB_XFER), db),
                per(
                    ld(&DB_MATVEC) + ld(&DB_ATTN) + ld(&DB_GLUE) + ld(&DB_XFER),
                    db
                )
            );
        }
        if pb > 0 {
            let gemm = ld(&PB_GEMM);
            let ffn = ld(&PB_FFN);
            let glue = ld(&PB_COMPUTE).saturating_sub(gemm);
            s += &format!(
                "  dev-prefill block (per layer): calls={pb} h2d={:.1} compute={:.1} attn={:.1} d2h={:.1} total={:.1}\n",
                per(ld(&PB_H2D), pb),
                per(ld(&PB_COMPUTE), pb),
                per(ld(&PB_ATTN), pb),
                per(ld(&PB_D2H), pb),
                per(ld(&PB_H2D) + ld(&PB_COMPUTE) + ld(&PB_ATTN) + ld(&PB_D2H), pb)
            );
            s += &format!(
                "    compute split: gemm={:.1} (ffn={:.1} qkvo={:.1}) glue={:.1}  [gemm {:.0}% of compute]\n",
                per(gemm, pb),
                per(ffn, pb),
                per(gemm.saturating_sub(ffn), pb),
                per(glue, pb),
                if ld(&PB_COMPUTE) > 0 {
                    gemm as f64 * 100.0 / ld(&PB_COMPUTE) as f64
                } else {
                    0.0
                }
            );
        }
        if pg > 0 {
            s += &format!(
                "  prefill GEMM (usm/TC, whole call): calls={pg} total={:.1} us/call ({:.1} ms all)\n",
                per(ld(&PG_NS), pg),
                ld(&PG_NS) as f64 / 1e6
            );
        }
        if bc > 0 {
            s += &format!(
                "  prefill(batched): calls={bc} h2d={:.1} kernel={:.1} d2h={:.1} total={:.1}\n",
                per(ld(&B_H2D), bc),
                per(ld(&B_KER), bc),
                per(ld(&B_D2H), bc),
                per(ld(&B_H2D) + ld(&B_KER) + ld(&B_D2H), bc)
            );
        }
        if sc > 0 {
            s += &format!(
                "  decode(single):   calls={sc} h2d={:.1} kernel={:.1} d2h={:.1} total={:.1}",
                per(ld(&S_H2D), sc),
                per(ld(&S_KER), sc),
                per(ld(&S_D2H), sc),
                per(ld(&S_H2D) + ld(&S_KER) + ld(&S_D2H), sc)
            );
        }
        s
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
            Some(Self {
                raw,
                device: device_index,
            })
        }
    }
    pub fn device(&self) -> u32 {
        self.device
    }
    fn raw(&self) -> *mut RslCudaStreamRaw {
        self.raw
    }
    /// Free this stream's cached Q8_0 prefill-GEMM weight repacks. Call on model
    /// reload: the repack cache keys on the weight device pointer, which the
    /// allocator reuses across model loads, so a stale entry must not survive.
    /// Idempotent (empty cache → no-op).
    pub fn q8_0_repack_reset(&self) {
        // SAFETY: FFI; frees + clears the stream's repack cache.
        unsafe { rsl_cuda_q8_0_repack_reset(self.raw) };
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
            Some(Self {
                ptr,
                len_bytes: src.len(),
                stream,
            })
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

/// Drain + discard the per-thread CUDA runtime last-error WITHOUT touching the
/// kernel-error latch. Call at the entry of a device path that must not inherit
/// a stale `cudaErrorInvalidDevice` left latched by an out-of-range enumeration
/// probe on the same thread (see `rsl_cuda_clear_error`), so that path's own
/// post-launch check reports only its own result.
pub fn clear_last_error() {
    // SAFETY: FFI; no arguments.
    unsafe { rsl_cuda_clear_error() };
}

// ---- CUDA graph capture helpers (Feature 1b) -----------------------------
// Thin safe wrappers over the graph FFI, used by
// `CudaMatvecCache::decode_attn_block_f32_graph`. Each returns `true`/a handle
// on success; the caller treats any failure as "fall back to the eager block".

/// Begin capturing `stream` into a graph. `true` on success.
fn graph_begin_capture(stream: &CudaStream) -> bool {
    // SAFETY: FFI; the stream handle is live for the call.
    unsafe { rsl_cuda_graph_begin_capture(stream.raw()) == 0 }
}

/// End capture, returning the captured graph handle (`None` on failure — the
/// stream is returned to the non-capturing state either way).
fn graph_end_capture(stream: &CudaStream) -> Option<*mut c_void> {
    let mut graph: *mut c_void = std::ptr::null_mut();
    // SAFETY: FFI; `graph` is written on success.
    let rc = unsafe { rsl_cuda_graph_end_capture(stream.raw(), &mut graph as *mut *mut c_void) };
    if rc == 0 && !graph.is_null() {
        Some(graph)
    } else {
        None
    }
}

/// Instantiate a captured `graph` into an executable (`None` on failure).
///
/// SAFETY: `graph` is a live graph handle from [`graph_end_capture`].
unsafe fn graph_instantiate(graph: *mut c_void) -> Option<CudaGraphExecHandle> {
    let mut exec: *mut c_void = std::ptr::null_mut();
    let rc = rsl_cuda_graph_instantiate(graph, &mut exec as *mut *mut c_void);
    if rc == 0 && !exec.is_null() {
        Some(CudaGraphExecHandle { exec })
    } else {
        None
    }
}

/// Patch `exec` in place from a freshly captured `graph` of identical topology.
/// `true` only when the update fully succeeded (else the caller re-instantiates).
///
/// SAFETY: `exec`/`graph` are live handles on this cache's device.
unsafe fn graph_exec_update(exec: &CudaGraphExecHandle, graph: *mut c_void) -> bool {
    rsl_cuda_graph_exec_update(exec.exec, graph) == 0
}

/// Launch `exec` on `stream` (async; pair with [`stream_sync`]). `true` on success.
///
/// SAFETY: `exec` is a live executable graph; `stream` is its device's stream.
unsafe fn graph_launch(exec: &CudaGraphExecHandle, stream: &CudaStream) -> bool {
    rsl_cuda_graph_launch(exec.exec, stream.raw()) == 0
}

/// Block until all queued stream work (incl. a graph launch) completes.
fn stream_sync(stream: &CudaStream) -> bool {
    // SAFETY: FFI; the stream handle is live for the call.
    unsafe { rsl_cuda_stream_sync(stream.raw()) == 0 }
}

/// Free a captured graph handle (the exec is independent once instantiated /
/// updated). Null-safe.
///
/// SAFETY: `graph` is a live graph handle or null.
unsafe fn graph_destroy(graph: *mut c_void) {
    rsl_cuda_graph_destroy(graph);
}

/// Async device→device copy of `n_bytes` from `src` to `dst` on `stream` (the
/// graph path's KV-mirror scatter, issued inside the capture region). `true` on
/// success.
///
/// SAFETY: `dst`/`src` are live device pointers on `stream` with ≥ `n_bytes`.
unsafe fn memcpy_d2d_async(
    stream: &CudaStream,
    dst: *mut c_void,
    src: *const c_void,
    n_bytes: usize,
) -> bool {
    rsl_cuda_memcpy_d2d_async(stream.raw(), dst, src, n_bytes as u64) == 0
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

/// Device-ptr folded-matvec picker for the rmsfold/smfold chain: PTQ1_0 or its
/// 2-bit PQ2_0 sibling (the only two folded Prism formats). SAFETY: as the two
/// `matvec_*_packed_f32` it forwards to.
pub unsafe fn matvec_folded_dev(
    stream: &CudaStream,
    kind: CudaPackedKind,
    w_bytes: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
) -> Result<(), CudaError> {
    match kind {
        CudaPackedKind::Pq2_0 => matvec_pq2_0_packed_f32(stream, w_bytes, x, out, m, k),
        _ => matvec_ptq1_0_packed_f32(stream, w_bytes, x, out, m, k),
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
    let rc = rsl_cuda_hadamard_forward(
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
        Err(CudaError::Kernel(rc))
    }
}

/// Allocate `n_bytes` of CUDA **managed** memory on `device` (CPU-mappable,
/// migrates on access) for the MoE tiered-expert tier. `None` on failure.
/// Device-global (no stream) — free with [`free_managed`] (the two together
/// back `Storage::device_from_raw(DeviceBackend::Cuda, …)` in the engine).
pub fn malloc_managed(device: u32, n_bytes: usize) -> Option<*mut u8> {
    if n_bytes == 0 {
        return None;
    }
    // SAFETY: FFI; returns null on failure, checked below.
    let p = unsafe { rsl_cuda_malloc_managed(device as c_int, n_bytes as u64) };
    if p.is_null() {
        None
    } else {
        Some(p as *mut u8)
    }
}

/// Free a [`malloc_managed`] allocation on `device`.
///
/// # Safety
/// `ptr` came from [`malloc_managed`] on this `device` and is freed exactly once.
pub unsafe fn free_managed(device: u32, ptr: *mut u8) {
    rsl_cuda_free_managed(device as c_int, ptr as *mut c_void);
}

/// Prefetch a [`malloc_managed`] allocation onto `device` so its pages are
/// device-resident before the first kernel touches them (skips lazy CPU→GPU
/// fault-migration during decode). Best-effort: `true` on success, `false` if
/// the device/OS doesn't support it (pages then migrate lazily, as before).
///
/// # Safety
/// `ptr` is a live managed allocation of at least `n_bytes` on `device`.
pub unsafe fn prefetch_managed(device: u32, ptr: *const u8, n_bytes: usize) -> bool {
    if ptr.is_null() || n_bytes == 0 {
        return false;
    }
    rsl_cuda_prefetch_managed(device as c_int, ptr as *const c_void, n_bytes as u64) == 0
}

/// Block until all prior work on `device` finishes — including the async
/// [`prefetch_managed`] migrations. Call once after a batch of promotions.
pub fn device_sync(device: u32) -> bool {
    // SAFETY: FFI; sets the device + synchronizes. No pointers involved.
    unsafe { rsl_cuda_device_sync(device as c_int) == 0 }
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

/// Generate the safe wrapper for a fused gate+up matvec (decode). One launch
/// computes gate_out + up_out, reusing the parity-validated per-format row_dot.
/// SAFETY (all): `gw`/`uw`/`x`/`gout`/`uout` are device pointers on `stream`'s
/// device sized for the quant's block layout and the given M/K.
macro_rules! cuda_gate_up_fused {
    ($name:ident, $raw:ident) => {
        #[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
        pub unsafe fn $name(
            stream: &CudaStream,
            gw: *const c_void,
            uw: *const c_void,
            x: *const f32,
            gout: *mut f32,
            uout: *mut f32,
            m: usize,
            k: usize,
        ) -> Result<(), CudaError> {
            let rc = $raw(stream.raw(), gw, uw, x, gout, uout, m as c_int, k as c_int);
            if rc == 0 {
                Ok(())
            } else {
                Err(CudaError::Kernel(rc))
            }
        }
    };
}
cuda_gate_up_fused!(
    matvec_q8_0_gate_up_fused,
    rsl_cuda_matvec_q8_0_packed_f32_gate_up_fused
);
cuda_gate_up_fused!(
    matvec_q4_k_gate_up_fused,
    rsl_cuda_matvec_q4_k_packed_f32_gate_up_fused
);
cuda_gate_up_fused!(
    matvec_q6_k_gate_up_fused,
    rsl_cuda_matvec_q6_k_packed_f32_gate_up_fused
);
cuda_gate_up_fused!(
    matvec_q5_k_gate_up_fused,
    rsl_cuda_matvec_q5_k_packed_f32_gate_up_fused
);
cuda_gate_up_fused!(
    matvec_iq4_nl_gate_up_fused,
    rsl_cuda_matvec_iq4_nl_packed_f32_gate_up_fused
);
cuda_gate_up_fused!(
    matvec_iq4_xs_gate_up_fused,
    rsl_cuda_matvec_iq4_xs_packed_f32_gate_up_fused
);
cuda_gate_up_fused!(
    matvec_iq1_s_gate_up_fused,
    rsl_cuda_matvec_iq1_s_packed_f32_gate_up_fused
);
cuda_gate_up_fused!(
    matvec_iq1_m_gate_up_fused,
    rsl_cuda_matvec_iq1_m_packed_f32_gate_up_fused
);
cuda_gate_up_fused!(
    matvec_iq2_xxs_gate_up_fused,
    rsl_cuda_matvec_iq2_xxs_packed_f32_gate_up_fused
);
cuda_gate_up_fused!(
    matvec_iq2_xs_gate_up_fused,
    rsl_cuda_matvec_iq2_xs_packed_f32_gate_up_fused
);
cuda_gate_up_fused!(
    matvec_iq2_s_gate_up_fused,
    rsl_cuda_matvec_iq2_s_packed_f32_gate_up_fused
);
cuda_gate_up_fused!(
    matvec_iq3_xxs_gate_up_fused,
    rsl_cuda_matvec_iq3_xxs_packed_f32_gate_up_fused
);
cuda_gate_up_fused!(
    matvec_iq3_s_gate_up_fused,
    rsl_cuda_matvec_iq3_s_packed_f32_gate_up_fused
);
cuda_gate_up_fused!(
    matvec_ptq1_0_gate_up_fused,
    rsl_cuda_matvec_ptq1_0_packed_f32_gate_up_fused
);

/// Safe wrapper for the Q4_K prefill GEMM (tiled, shared-mem weight reuse, f32
/// accumulate — bit-exact). `out` = [N,M] row-major, `x` = [N,K] row-major, `w`
/// = Q4_K packed [M,K]. This is the weight-reuse replacement for the batched
/// matvec on the prefill path.
///
/// SAFETY: `w`/`x`/`out` are device pointers on `stream`'s device sized for the
/// Q4_K block layout and the given M/K/N (K % 256 == 0).
#[allow(clippy::missing_safety_doc)]
pub unsafe fn gemm_q4_k_f32(
    stream: &CudaStream,
    w: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_gemm_q4_k_f32(stream.raw(), w, x, out, m as c_int, k as c_int, n as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Safe wrapper for the Q8_0 f32-reuse prefill GEMM (bit-exact, LOSSLESS — the
/// Q8_0 analogue of [`gemm_q4_k_f32`]). Selected only when the `gemm:q8_0_f32`
/// verdict passed.
///
/// SAFETY: `w`/`x`/`out` are device pointers on `stream`'s device sized for the
/// Q8_0 block layout and the given M/K/N (K % 32 == 0).
#[allow(clippy::missing_safety_doc)]
pub unsafe fn gemm_q8_0_f32(
    stream: &CudaStream,
    w: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_gemm_q8_0_f32(stream.raw(), w, x, out, m as c_int, k as c_int, n as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Safe wrapper for the Q4_K int8 tensor-core prefill GEMM (W8A8, lossy). Same
/// device-pointer contract as `gemm_q4_k_f32`; the kernel quantizes activations
/// to int8 internally. Selected only when the `gemm:q4_k_w8a8_tc` verdict passed.
///
/// SAFETY: `w`/`x`/`out` are device pointers on `stream`'s device sized for the
/// Q4_K block layout and the given M/K/N (K % 256 == 0).
#[allow(clippy::missing_safety_doc)]
pub unsafe fn gemm_q4_k_w8a8_tc(
    stream: &CudaStream,
    w: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), CudaError> {
    let rc =
        rsl_cuda_gemm_q4_k_w8a8_tc(stream.raw(), w, x, out, m as c_int, k as c_int, n as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Q8_0 W8A8 int8 tensor-core PREFILL GEMM (T3c). Same device-pointer contract as
/// [`gemm_q4_k_w8a8_tc`]; Q8_0 weights are already int8 (symmetric), the kernel
/// quantizes activations to int8 internally. Selected only when the
/// `gemm:q8_0_w8a8_tc` verdict passed. K % 32 == 0.
///
/// SAFETY: `w`/`x`/`out` are device pointers on `stream`'s device sized for the
/// Q8_0 block layout and the given M/K/N.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn gemm_q8_0_w8a8_tc(
    stream: &CudaStream,
    w: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), CudaError> {
    let rc =
        rsl_cuda_gemm_q8_0_w8a8_tc(stream.raw(), w, x, out, m as c_int, k as c_int, n as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// W4A8 Q4_K DECODE matvec (single row): int8-activation dot, the lossy +
/// perf-gated decode twin of [`gemm_q4_k_w8a8_tc`]. `out` = [M], `x` = [K], `w`
/// = Q4_K packed [M,K].
///
/// # Safety
/// `w`/`x`/`out` are device pointers on `stream`'s device sized for (M,K)/K/M;
/// `K % 256 == 0`. The wrapper quantizes `x` to int8 internally + synchronizes.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn matvec_q4_k_w4a8(
    stream: &CudaStream,
    w: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_matvec_q4_k_w4a8(stream.raw(), w, x, out, m as c_int, k as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Aligned Q8_0 DECODE matvec over the cached repack (lookup-only). Reuses the
/// per-stream Q8_0 repack the prefill GEMM builds (keyed by the SAME weight device
/// ptr) so the decode's coalesced weight reads are 32-ALIGNED instead of straddling
/// the 34-byte block boundary. Returns `Err` when the repack isn't cached (caller
/// falls back to the byte-packed [`matvec_q8_0_packed_f32`]). Bit-equal math.
///
/// # Safety
/// `w`/`x`/`out` are device pointers on `stream`'s device sized for (M,K)/K/M;
/// `K % 32 == 0`.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn matvec_q8_0_repacked(
    stream: &CudaStream,
    w: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_matvec_q8_0_repacked(stream.raw(), w, x, out, m as c_int, k as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Whether the aligned repacked Q8_0 DECODE matvec is enabled (A/B lever
/// `RUSTLLAMA_Q8_REPACK_DECODE=1`). Default OFF → the byte-packed matvec runs
/// (byte-identical to today). Lookup-only + fail-closed, so even ON it only
/// engages where the prefill already built the repack.
fn q8_repack_decode_enabled() -> bool {
    static EN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // Default ON: the on-device A/B measured a consistent ~+10% decode and it is
    // bit-equal + lookup-only + fallback-safe (no repack cached ⇒ byte-packed
    // matvec). RUSTLLAMA_Q8_REPACK_DECODE=0 disables it for an A/B.
    *EN.get_or_init(|| {
        std::env::var("RUSTLLAMA_Q8_REPACK_DECODE")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Single-row Q8_0 W8A8 int-dot DECODE matvec (int8 activations, LOSSY) — the
/// MMVQ-style decode analog of [`matvec_q4_k_w4a8`] for Q8_0 (symmetric: no
/// min term). `out` = [M], `x` = [K], `w` = Q8_0 packed [M,K].
///
/// # Safety
/// `w`/`x`/`out` are device pointers on `stream`'s device sized for (M,K)/K/M;
/// `K % 32 == 0`. The wrapper quantizes `x` to int8 internally + synchronizes.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn matvec_q8_0_w8a8(
    stream: &CudaStream,
    w: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_matvec_q8_0_w8a8(stream.raw(), w, x, out, m as c_int, k as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Single-row Q4_0 W8A8 int-dot DECODE matvec (int8 activations, LOSSY; T3b).
/// Symmetric (weight = nibble-8). `K % 32 == 0`.
///
/// # Safety
/// Device pointers on `stream`'s device sized for (M,K)/K/M; wrapper quantizes +
/// synchronizes.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn matvec_q4_0_w8a8(
    stream: &CudaStream,
    w: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_matvec_q4_0_w8a8(stream.raw(), w, x, out, m as c_int, k as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Single-row Q5_0 W8A8 int-dot DECODE matvec (int8 activations, LOSSY; T3b).
/// Symmetric (weight = (nibble|5th_bit<<4)-16). `K % 32 == 0`.
///
/// # Safety
/// Device pointers on `stream`'s device sized for (M,K)/K/M; wrapper quantizes +
/// synchronizes.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn matvec_q5_0_w8a8(
    stream: &CudaStream,
    w: *const c_void,
    x: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_matvec_q5_0_w8a8(stream.raw(), w, x, out, m as c_int, k as c_int);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}
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

/// Batched RoPE for the device-resident PREFILL block: rotate `n_rows` rows of
/// `qk` = [n_rows, n_heads, head_dim], each row `r` at absolute position
/// `base_pos + r`. Matches [`rope_f32`] numerics at `n_rows == 1`.
///
/// SAFETY: `qk` (≥ `n_rows*n_heads*head_dim`) + `inv_freq` (≥ `head_dim/2`) are
/// device pointers on `stream`; `head_dim` even.
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
pub unsafe fn rope_rows_f32(
    stream: &CudaStream,
    qk: *mut f32,
    n_rows: usize,
    n_heads: usize,
    head_dim: usize,
    base_pos: usize,
    inv_freq: *const f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_rope_rows_f32(
        stream.raw(),
        qk,
        n_heads as c_int,
        head_dim as c_int,
        base_pos as c_int,
        n_rows as c_int,
        inv_freq,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Scatter `n_new` post-RoPE K/V rows (token-major `[n_new, n_kv_heads,
/// head_dim]`) into the head-major KV mirror `[n_kv_heads, max_ctx, head_dim]`
/// at column `col_base` (`mirror[h, col_base+i, :] = src[i, h, :]`). The
/// device-resident PREFILL analogue of the decode block's per-column d2d write.
///
/// SAFETY: `mirror` (≥ `n_kv_heads*max_ctx*head_dim`) + `src` (≥
/// `n_new*n_kv_heads*head_dim`) are device pointers on `stream`; `col_base +
/// n_new <= max_ctx`.
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
pub unsafe fn kv_scatter_rows_f32(
    stream: &CudaStream,
    mirror: *mut f32,
    src: *const f32,
    n_new: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    col_base: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_kv_scatter_rows_f32(
        stream.raw(),
        mirror,
        src,
        n_new as c_int,
        n_kv_heads as c_int,
        head_dim as c_int,
        max_ctx as c_int,
        col_base as c_int,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Device-resident RMSNorm: `y[r] = rmsnorm(x[r]) * w`, `n_rows` rows of `d`.
///
/// SAFETY: `x`/`w`/`y` are device pointers on `stream` (`x`,`y` ≥ `n_rows*d`;
/// `w` ≥ `d`); the wrapper synchronizes before returning.
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
pub unsafe fn rmsnorm_dev_f32(
    stream: &CudaStream,
    x: *const f32,
    w: *const f32,
    y: *mut f32,
    n_rows: usize,
    d: usize,
    eps: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_rmsnorm_dev_f32(stream.raw(), x, w, y, n_rows as c_int, d as c_int, eps);
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// Qwen3 per-head q/k RMSNorm in place: RMSNorm each of `n_heads` rows of
/// `head_dim` in `buf` with the shared weight `w`.
///
/// SAFETY: `buf` (≥ `n_heads*head_dim`) and `w` (≥ `head_dim`) are device
/// pointers on `stream`; the wrapper synchronizes before returning.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn qk_head_norm_f32(
    stream: &CudaStream,
    buf: *mut f32,
    w: *const f32,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_qk_head_norm_f32(
        stream.raw(),
        buf,
        w,
        n_heads as c_int,
        head_dim as c_int,
        eps,
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

/// P6-fusions (write-blind mirror): RMSNorm-of-input fused into the following
/// matvec. `y[r,o] = Σ_i (x[r,i]·rms_scale(x_row)·wnorm[i])·wmat[o,i]`, `wmat`
/// `[m,k]` out-major.
///
/// SAFETY: `x` (≥`n_rows*k`), `wnorm` (≥`k`), `wmat` (≥`m*k`), `y` (≥`n_rows*m`)
/// are device pointers on `stream`; the wrapper synchronizes before returning.
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
pub unsafe fn rmsnorm_matvec_f32(
    stream: &CudaStream,
    x: *const f32,
    wnorm: *const f32,
    wmat: *const f32,
    y: *mut f32,
    n_rows: usize,
    k: usize,
    m: usize,
    eps: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_rmsnorm_matvec_f32(
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
        Err(CudaError::Kernel(rc))
    }
}

/// P6-fusions (write-blind mirror): FFN down-proj fused with residual add.
/// `y[r,o] = resid[r,o] + Σ_i ff[r,i]·wmat[o,i]`, `wmat` `[m,k]` out-major.
///
/// SAFETY: `ff` (≥`n_rows*k`), `wmat` (≥`m*k`), `resid`/`y` (≥`n_rows*m`) are
/// device pointers on `stream`; the wrapper synchronizes before returning.
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
pub unsafe fn down_residual_f32(
    stream: &CudaStream,
    ff: *const f32,
    wmat: *const f32,
    resid: *const f32,
    y: *mut f32,
    n_rows: usize,
    k: usize,
    m: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_down_residual_f32(
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
        Err(CudaError::Kernel(rc))
    }
}

/// Plain element-wise residual add: `hidden[i] += branch[i]` (device f32, in
/// place). Used for the device-resident dense FFN residual (`hidden += ffn_out`)
/// where there's no following RMSNorm to fuse (the next op is the next layer's
/// attn-norm) — so [`add_rmsnorm_f32`] would be wrong.
///
/// SAFETY: `hidden` (mut) and `branch` are device pointers of `n` f32.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn add_f32(
    stream: &CudaStream,
    hidden: *mut f32,
    branch: *const f32,
    n: usize,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_add_f32(stream.raw(), hidden, branch, n as c_int);
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
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the CPU
    // `gqa_attention_flash_decode_capped_windowed`. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
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
        window as c_int,
        softcap,
    );
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::Kernel(rc))
    }
}

/// GPU paged attention (Phase B): block-table FlashAttention decode. Reads K/V
/// from the whole-pool device mirror `kv_pool` via `block_table` (logical block
/// -> physical page id) for ONE sequence at `layer_idx`. Matches the contiguous
/// [`flash_attn_decode_f32`] numerics (same CPU reference), differing only in
/// how a logical KV position maps to a device address.
///
/// SAFETY: `q`/`out` = `[n_heads, head_dim]`; `kv_pool` = the pool mirror slab;
/// `block_table` = `[ceil(kv_len / page_size)]` i32 device ptr; `n_heads %
/// n_kv_heads == 0`; `layer_idx < n_layers`.
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_decode_paged_f32(
    stream: &CudaStream,
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
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_decode_paged_f32(
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
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); see the CPU
    // `gqa_attention_flash_prefill_capped_windowed`. Byte-identical at 0/0.
    window: i32,
    softcap: f32,
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
        window as c_int,
        softcap,
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
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); the last
    // two params by contract (see `flash_attn_decode_f32`). Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_decode_q4_0(
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
        Err(CudaError::Kernel(rc))
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
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); the last
    // two params by contract (see `flash_attn_prefill_f32`). Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_prefill_q4_0(
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
        Err(CudaError::Kernel(rc))
    }
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
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); the last
    // two params by contract (see `flash_attn_decode_f32`). Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_decode_nvfp4(
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
        Err(CudaError::Kernel(rc))
    }
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
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); the last
    // two params by contract (see `flash_attn_prefill_f32`). Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_prefill_nvfp4(
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
        Err(CudaError::Kernel(rc))
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
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); the last
    // two params by contract (see `flash_attn_decode_f32`). Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_decode_mxfp4(
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
        Err(CudaError::Kernel(rc))
    }
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
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); the last
    // two params by contract (see `flash_attn_prefill_f32`). Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_prefill_mxfp4(
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
        Err(CudaError::Kernel(rc))
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
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); the last
    // two params by contract (see `flash_attn_decode_f32`). Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_decode_mxfp6(
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
        Err(CudaError::Kernel(rc))
    }
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
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); the last
    // two params by contract (see `flash_attn_prefill_f32`). Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_prefill_mxfp6(
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
        Err(CudaError::Kernel(rc))
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
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); the last
    // two params by contract (see `flash_attn_decode_f32`). Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_decode_mxfp8(
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
        Err(CudaError::Kernel(rc))
    }
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
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); the last
    // two params by contract (see `flash_attn_prefill_f32`). Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_prefill_mxfp8(
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
        Err(CudaError::Kernel(rc))
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
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); the last
    // two params by contract (see `flash_attn_decode_f32`). Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_decode_tq(
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
        Err(CudaError::Kernel(rc))
    }
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
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); the last
    // two params by contract (see `flash_attn_prefill_f32`). Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_prefill_tq(
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
        Err(CudaError::Kernel(rc))
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
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); the last
    // two params by contract (see `flash_attn_decode_f32`). Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_decode_q8_0(
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
        Err(CudaError::Kernel(rc))
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
    // SWA window (0=full) + Gemma2 attention-logit soft-cap (0=off); the last
    // two params by contract (see `flash_attn_prefill_f32`). Byte-identical at 0/0.
    window: i32,
    softcap: f32,
) -> Result<(), CudaError> {
    let rc = rsl_cuda_flash_attn_prefill_q8_0(
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
        Err(CudaError::Kernel(rc))
    }
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
    /// H2D copy of `src` into this buffer at `byte_offset` (for partial writes:
    /// the activation slab's sub-regions + a KV-mirror row). `byte_offset +
    /// src.len()` must be <= cap.
    fn upload_at(&mut self, byte_offset: usize, src: &[u8]) -> bool {
        if byte_offset + src.len() > self.cap_bytes {
            return false;
        }
        // SAFETY: dst is within the live cap_bytes allocation.
        let dst = unsafe { (self.ptr as *mut u8).add(byte_offset) as *mut c_void };
        let rc = unsafe {
            rsl_cuda_memcpy_h2d(
                self.raw_stream,
                dst,
                src.as_ptr() as *const c_void,
                src.len() as u64,
            )
        };
        rc == 0
    }
    /// D2H copy of `dst.len()` bytes from `byte_offset` into `dst`.
    fn download_at(&self, byte_offset: usize, dst: &mut [u8]) -> bool {
        if byte_offset + dst.len() > self.cap_bytes {
            return false;
        }
        // SAFETY: src is within the live cap_bytes allocation.
        let src = unsafe { (self.ptr as *const u8).add(byte_offset) as *const c_void };
        let rc = unsafe {
            rsl_cuda_memcpy_d2h(
                self.raw_stream,
                dst.as_mut_ptr() as *mut c_void,
                src,
                dst.len() as u64,
            )
        };
        rc == 0
    }
    /// Raw device pointer at `f32_offset` f32-elements into the buffer.
    fn f32_ptr_at(&self, f32_offset: usize) -> *mut f32 {
        // SAFETY: caller keeps the offset within cap_bytes.
        unsafe { (self.ptr as *mut f32).add(f32_offset) }
    }
    /// ASYNC H2D copy of `src` into this buffer at `byte_offset` (the graph
    /// path's non-synchronizing upload; pair with a stream sync). Returns false
    /// on a bad range / launch failure. (Feature 1b.)
    fn upload_at_async(&mut self, byte_offset: usize, src: &[u8]) -> bool {
        if byte_offset + src.len() > self.cap_bytes {
            return false;
        }
        // SAFETY: dst is within the live cap_bytes allocation.
        let dst = unsafe { (self.ptr as *mut u8).add(byte_offset) as *mut c_void };
        let rc = unsafe {
            rsl_cuda_memcpy_h2d_async(
                self.raw_stream,
                dst,
                src.as_ptr() as *const c_void,
                src.len() as u64,
            )
        };
        rc == 0
    }
    /// ASYNC D2H copy of `dst.len()` bytes from `byte_offset` into `dst` (the
    /// graph path's deferred drain, issued AFTER the post-launch stream sync, so
    /// the graph has already completed). Returns false on a bad range / launch
    /// failure. (Feature 1b.)
    fn download_at_async(&self, byte_offset: usize, dst: &mut [u8]) -> bool {
        if byte_offset + dst.len() > self.cap_bytes {
            return false;
        }
        // SAFETY: src is within the live cap_bytes allocation.
        let src = unsafe { (self.ptr as *const u8).add(byte_offset) as *const c_void };
        let rc = unsafe {
            rsl_cuda_memcpy_d2h_async(
                self.raw_stream,
                dst.as_mut_ptr() as *mut c_void,
                src,
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

#[inline]
fn dn_f32_bytes(s: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding/invalid bit-patterns as bytes; len is exact.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
#[inline]
fn dn_f32_bytes_mut(s: &mut [f32]) -> &mut [u8] {
    // SAFETY: as above; exclusive borrow preserved.
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, std::mem::size_of_val(s)) }
}

// ============================================================
// PP-0 — multi-GPU peer transport (pipeline-stage handoff + TP collectives).
// ============================================================

/// Best-effort enable of direct P2P so `accessing_dev` can read `peer_dev`'s
/// memory (NVLink/PCIe). Returns true if direct peer access is now enabled, false
/// if the pair doesn't support it (a peer copy still works, staged through host) or
/// on any error. Idempotent. Same device → true.
pub fn enable_peer_access(accessing_dev: u32, peer_dev: u32) -> bool {
    // SAFETY: FFI; the C side validates the device indices.
    unsafe { rsl_cuda_enable_peer_access(accessing_dev as c_int, peer_dev as c_int) == 0 }
}

/// Async peer copy of `n_bytes` from `src` (resident on `src_dev`) to `dst`
/// (resident on `dst_dev`) on `stream`. Correct whether or not peer access was
/// enabled (CUDA stages through host when it wasn't). Non-synchronizing — pair with
/// a stream sync. Returns true on success.
///
/// SAFETY: `dst`/`src` must be live device allocations of ≥ `n_bytes` on the named
/// devices for the duration of the (async) copy.
pub unsafe fn peer_memcpy(
    stream: &CudaStream,
    dst: *mut c_void,
    dst_dev: u32,
    src: *const c_void,
    src_dev: u32,
    n_bytes: usize,
) -> bool {
    rsl_cuda_memcpy_peer(
        stream.raw(),
        dst,
        dst_dev as c_int,
        src,
        src_dev as c_int,
        n_bytes as u64,
    ) == 0
}

/// PP-0 parity self-check: round-trip a buffer through `peer_memcpy`. On a
/// single-GPU box `src_dev == dst_dev == device_index`, so this validates the API +
/// exact byte fidelity (the only part verifiable without a 2nd GPU); on real
/// multi-GPU the identical call exercises P2P transport. Returns
/// `(passed, worst_abs_diff)`, or `None` if the device/stream/alloc fails.
pub fn peer_copy_self_check(device_index: u32) -> Option<(bool, f32)> {
    let stream = CudaStream::create(device_index)?;
    let _ = enable_peer_access(device_index, device_index); // no-op (same dev); exercises the path
    let n = 4096usize;
    let src: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 7.0).collect();
    let nbytes = n * 4;
    let mut src_dev = RawDevBuf::alloc(&stream, nbytes)?;
    let dst_dev = RawDevBuf::alloc(&stream, nbytes)?;
    if !src_dev.upload(dn_f32_bytes(&src)) {
        return None;
    }
    // SAFETY: both buffers are live, sized nbytes, on device_index.
    let ok = unsafe {
        peer_memcpy(
            &stream,
            dst_dev.ptr,
            device_index,
            src_dev.ptr as *const c_void,
            device_index,
            nbytes,
        )
    };
    if !ok {
        return Some((false, f32::INFINITY));
    }
    // SAFETY: FFI; blocks until the async peer copy completes.
    unsafe {
        rsl_cuda_stream_sync(stream.raw());
    }
    let mut out = vec![0f32; n];
    if !dst_dev.download(dn_f32_bytes_mut(&mut out)) {
        return None;
    }
    let worst = src
        .iter()
        .zip(&out)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    Some((worst == 0.0, worst))
}

// ============================================================
// TP-1 — multi-GPU collectives (all-reduce / all-gather), hand-rolled over the
// PP-0 peer transport + the on-device add_f32 kernel. No NCCL/oneCCL dependency
// (clean-room, owns its whole compute stack). These are the building blocks of
// tensor parallelism: a row-parallel matmul produces a per-device PARTIAL that an
// all-reduce SUMS to the final output; head-parallel attention produces per-device
// head outputs that an all-gather CONCATENATES before the O-projection.
// ============================================================

/// In-place all-reduce (SUM) across `bufs` — one device buffer per participating
/// GPU, each `n` f32s. STAR schedule: every non-root buffer is peer-copied into a
/// scratch on `bufs[0]`'s device and added into `bufs[0]` (the running sum), then
/// the sum is broadcast back to every buffer. After the call EVERY buffer holds the
/// element-wise sum. All work is enqueued on the ONE root stream, so stream
/// ordering (not explicit syncs) sequences each copy before its add + reuse of the
/// scratch. `stream` MUST be on `bufs[0]`'s device. Returns false on any
/// peer/add/alloc/sync failure.
///
/// SAFETY: each `(dev, ptr)` must be a live device allocation of >= `n` f32s on its
/// named device for the duration of the (async) collective.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn all_reduce_sum_f32(stream: &CudaStream, bufs: &[(u32, *mut f32)], n: usize) -> bool {
    if bufs.is_empty() || n == 0 {
        return false;
    }
    if bufs.len() == 1 {
        return true; // a single participant is already "reduced"
    }
    let root_dev = bufs[0].0;
    let root = bufs[0].1;
    let nbytes = n * 4;
    let Some(scratch) = RawDevBuf::alloc(stream, nbytes) else {
        return false;
    };
    // Reduce every other buffer into the root (copy → add on one stream = ordered).
    for &(dev, ptr) in &bufs[1..] {
        if !peer_memcpy(
            stream,
            scratch.ptr,
            root_dev,
            ptr as *const c_void,
            dev,
            nbytes,
        ) {
            return false;
        }
        if add_f32(stream, root, scratch.ptr as *const f32, n).is_err() {
            return false;
        }
    }
    // Broadcast the sum back to every other buffer.
    for &(dev, ptr) in &bufs[1..] {
        if !peer_memcpy(
            stream,
            ptr as *mut c_void,
            dev,
            root as *const c_void,
            root_dev,
            nbytes,
        ) {
            return false;
        }
    }
    rsl_cuda_stream_sync(stream.raw()) == 0
}

/// All-gather: concatenate each input chunk (`ins[i]`, `n` f32s) into every
/// participant's output slab (`outs[d]`, `ins.len()*n` f32s), chunk `i` at f32
/// offset `i*n`. Pure peer-copy (no reduce) → bit-exact. `ins.len()` must equal
/// `outs.len()` (one in-chunk + one out-slab per participant). The single `stream`
/// drives every copy (`cudaMemcpyPeerAsync` is correct cross-device from one
/// stream). Returns false on any failure.
///
/// SAFETY: each `ins[i]` is a live alloc of >= `n` f32s and each `outs[d]` of >=
/// `ins.len()*n` f32s on their named devices for the call's duration.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn all_gather_f32(
    stream: &CudaStream,
    ins: &[(u32, *const f32)],
    outs: &[(u32, *mut f32)],
    n: usize,
) -> bool {
    let k = ins.len();
    if k == 0 || k != outs.len() || n == 0 {
        return false;
    }
    let nbytes = n * 4;
    for &(ddev, dptr) in outs {
        for (i, &(sdev, sptr)) in ins.iter().enumerate() {
            let dst = (dptr as *mut u8).add(i * nbytes) as *mut c_void;
            if !peer_memcpy(stream, dst, ddev, sptr as *const c_void, sdev, nbytes) {
                return false;
            }
        }
    }
    rsl_cuda_stream_sync(stream.raw()) == 0
}

/// TP-1 parity self-check: all-reduce (SUM). Builds K=3 partials of N f32 on
/// `device_index` (so a single-GPU box validates the reduction ALGEBRA + the
/// peer/add plumbing via self-copies; the real cross-device sum is confirmed on
/// 2+ GPU HW), runs [`all_reduce_sum_f32`], and checks EVERY participant's result
/// (first + last, covering the reduce and the broadcast legs) against the CPU
/// reference sum. Returns `(passed, worst_abs_diff)`, or `None` on setup failure.
pub fn collective_all_reduce_self_check(device_index: u32) -> Option<(bool, f32)> {
    // Drain any stale cudaGetLastError latch (e.g. cudaErrorInvalidDevice from an
    // enumeration probe on this thread) so it isn't misattributed to the add/sync
    // inside the reduce → a false MISCOMPUTE (worst_abs=inf) on a 1-GPU box.
    clear_last_error();
    let stream = CudaStream::create(device_index)?;
    let _ = enable_peer_access(device_index, device_index); // no-op same-dev; exercises the path
    let k = 3usize;
    let n = 2048usize;
    let nbytes = n * 4;
    let parts: Vec<Vec<f32>> = (0..k)
        .map(|p| {
            (0..n)
                .map(|i| ((i + p * 7) as f32) * 0.25 - p as f32)
                .collect()
        })
        .collect();
    let mut reference = vec![0f32; n];
    for p in &parts {
        for (r, &v) in reference.iter_mut().zip(p) {
            *r += v;
        }
    }
    let mut bufs_raw: Vec<RawDevBuf> = Vec::with_capacity(k);
    for p in &parts {
        let mut b = RawDevBuf::alloc(&stream, nbytes)?;
        if !b.upload(dn_f32_bytes(p)) {
            return None;
        }
        bufs_raw.push(b);
    }
    let bufs: Vec<(u32, *mut f32)> = bufs_raw
        .iter()
        .map(|b| (device_index, b.ptr as *mut f32))
        .collect();
    // SAFETY: all buffers are live, N f32, on device_index for this call.
    if !unsafe { all_reduce_sum_f32(&stream, &bufs, n) } {
        return Some((false, f32::INFINITY));
    }
    let mut worst = 0f32;
    for idx in [0usize, k - 1] {
        let mut out = vec![0f32; n];
        if !bufs_raw[idx].download(dn_f32_bytes_mut(&mut out)) {
            return None;
        }
        for (a, b) in out.iter().zip(&reference) {
            worst = worst.max((a - b).abs());
        }
    }
    Some((worst <= 1e-3, worst))
}

/// TP-1 parity self-check: all-gather. Builds K=3 input chunks of N f32 + K output
/// slabs (K*N each) on `device_index`, runs [`all_gather_f32`], and checks the
/// gathered slabs bit-exact against the host concatenation. Single-GPU validates
/// the concat-placement algebra + plumbing; real P2P gather on 2+ GPU HW. Returns
/// `(passed, worst_abs_diff)`, or `None` on setup failure.
pub fn collective_all_gather_self_check(device_index: u32) -> Option<(bool, f32)> {
    clear_last_error(); // drop a stale latch (see all_reduce self-check) → no false inf
    let stream = CudaStream::create(device_index)?;
    let _ = enable_peer_access(device_index, device_index);
    let k = 3usize;
    let n = 1024usize;
    let nbytes = n * 4;
    let ins_host: Vec<Vec<f32>> = (0..k)
        .map(|p| (0..n).map(|i| ((i + p * 11) as f32) * 0.1).collect())
        .collect();
    let mut reference: Vec<f32> = Vec::with_capacity(k * n);
    for p in &ins_host {
        reference.extend_from_slice(p);
    }
    let mut in_bufs: Vec<RawDevBuf> = Vec::with_capacity(k);
    for p in &ins_host {
        let mut b = RawDevBuf::alloc(&stream, nbytes)?;
        if !b.upload(dn_f32_bytes(p)) {
            return None;
        }
        in_bufs.push(b);
    }
    let mut out_bufs: Vec<RawDevBuf> = Vec::with_capacity(k);
    for _ in 0..k {
        out_bufs.push(RawDevBuf::alloc(&stream, k * nbytes)?);
    }
    let ins: Vec<(u32, *const f32)> = in_bufs
        .iter()
        .map(|b| (device_index, b.ptr as *const f32))
        .collect();
    let outs: Vec<(u32, *mut f32)> = out_bufs
        .iter()
        .map(|b| (device_index, b.ptr as *mut f32))
        .collect();
    // SAFETY: ins are N-f32 live allocs, outs are K*N-f32 live allocs, on device.
    if !unsafe { all_gather_f32(&stream, &ins, &outs, n) } {
        return Some((false, f32::INFINITY));
    }
    let mut worst = 0f32;
    for idx in [0usize, k - 1] {
        let mut out = vec![0f32; k * n];
        if !out_bufs[idx].download(dn_f32_bytes_mut(&mut out)) {
            return None;
        }
        for (a, b) in out.iter().zip(&reference) {
            worst = worst.max((a - b).abs());
        }
    }
    Some((worst == 0.0, worst))
}

/// C1 — resident-state Gated-DeltaNet DECODE cache. Holds each SSM layer's
/// recurrent state S (`n_v_heads * head_qk_dim * head_v_dim` f32) RESIDENT in
/// device memory across decoded tokens + reusable per-token I/O scratch. The
/// recurrence is memory-bound on S (~2 MiB/head), so keeping S resident and
/// running the step at device bandwidth beats the CPU per-head path — but ONLY
/// because S never leaves the device on the hot path. Host-authoritative
/// coherence: the caller passes `upload_state = Some(host S)` on the FIRST step
/// after any host write (prefill / restore / reset, tracked by the accel seam),
/// and calls [`download_state`](Self::download_state) to refresh host S at the
/// (rare) snapshot / serialize points. Reset on model load (drops all buffers).
pub struct CudaDeltaNetCache {
    stream: CudaStream,
    states: Vec<Option<RawDevBuf>>, // per layer: [n_v_heads*qk*vd] f32, RESIDENT
    d_qkv: Option<RawDevBuf>,
    d_gate: Option<RawDevBuf>,
    d_decay: Option<RawDevBuf>,
    d_beta: Option<RawDevBuf>,
    d_norm: Option<RawDevBuf>,
    d_out: Option<RawDevBuf>,
}

// SAFETY: the raw device/stream pointers are only ever used behind the global
// `Mutex<CudaDeltaNetCache>` (one thread at a time), exactly as `CudaMatvecCache`.
unsafe impl Send for CudaDeltaNetCache {}

impl CudaDeltaNetCache {
    pub fn new(device_index: u32) -> Option<Self> {
        Some(Self {
            stream: CudaStream::create(device_index)?,
            states: Vec::new(),
            d_qkv: None,
            d_gate: None,
            d_decay: None,
            d_beta: None,
            d_norm: None,
            d_out: None,
        })
    }

    fn ensure_scratch(buf: &mut Option<RawDevBuf>, stream: &CudaStream, n_bytes: usize) -> bool {
        if buf.as_ref().map_or(true, |b| b.cap_bytes < n_bytes) {
            *buf = RawDevBuf::alloc(stream, n_bytes);
        }
        buf.is_some()
    }

    /// Ensure layer `li`'s resident state buffer exists (zeroed on first alloc).
    fn ensure_state(&mut self, li: usize, elems: usize) -> bool {
        if self.states.len() <= li {
            self.states.resize_with(li + 1, || None);
        }
        let need = elems * 4;
        if self.states[li]
            .as_ref()
            .map_or(true, |b| b.cap_bytes < need)
        {
            let mut b = match RawDevBuf::alloc(&self.stream, need) {
                Some(b) => b,
                None => return false,
            };
            if !b.upload(&vec![0u8; need]) {
                return false;
            }
            self.states[li] = Some(b);
        }
        self.states[li].is_some()
    }

    /// One resident-state decode step for SSM layer `li`. `upload_state` is
    /// `Some(host S)` when the device copy is stale and must be refreshed before
    /// the step (None on the hot path). On success `out` holds v_out for all heads
    /// and the device state is updated in place. Returns false (caller falls back
    /// to the CPU recurrence) on any alloc/transfer/kernel miss or `qk != vd`.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        li: usize,
        qkv_conv: &[f32],
        gate: &[f32],
        decay: &[f32],
        beta: &[f32],
        ssm_norm: &[f32],
        out: &mut [f32],
        upload_state: Option<&[f32]>,
        n_v_heads: usize,
        n_qk_heads: usize,
        qk: usize,
        vd: usize,
        q_block: usize,
        k_block: usize,
        rms_eps: f32,
    ) -> bool {
        if qk != vd || n_qk_heads == 0 || n_v_heads == 0 || n_v_heads % n_qk_heads != 0 {
            return false;
        }
        let state_elems = n_v_heads * qk * vd;
        if out.len() != n_v_heads * vd {
            return false;
        }
        if !self.ensure_state(li, state_elems) {
            return false;
        }
        if let Some(hs) = upload_state {
            if hs.len() != state_elems
                || !self.states[li].as_mut().unwrap().upload(dn_f32_bytes(hs))
            {
                return false;
            }
        }
        if !Self::ensure_scratch(
            &mut self.d_qkv,
            &self.stream,
            std::mem::size_of_val(qkv_conv),
        ) || !Self::ensure_scratch(&mut self.d_gate, &self.stream, std::mem::size_of_val(gate))
            || !Self::ensure_scratch(
                &mut self.d_decay,
                &self.stream,
                std::mem::size_of_val(decay),
            )
            || !Self::ensure_scratch(&mut self.d_beta, &self.stream, std::mem::size_of_val(beta))
            || !Self::ensure_scratch(
                &mut self.d_norm,
                &self.stream,
                std::mem::size_of_val(ssm_norm),
            )
            || !Self::ensure_scratch(&mut self.d_out, &self.stream, std::mem::size_of_val(out))
        {
            return false;
        }
        if !self.d_qkv.as_mut().unwrap().upload(dn_f32_bytes(qkv_conv))
            || !self.d_gate.as_mut().unwrap().upload(dn_f32_bytes(gate))
            || !self.d_decay.as_mut().unwrap().upload(dn_f32_bytes(decay))
            || !self.d_beta.as_mut().unwrap().upload(dn_f32_bytes(beta))
            || !self.d_norm.as_mut().unwrap().upload(dn_f32_bytes(ssm_norm))
        {
            return false;
        }
        // SAFETY: all pointers are live device allocations sized to the dims; the
        // wrapper syncs (rsl_cuda_check) before returning, so the download below
        // reads completed results.
        let rc = unsafe {
            rsl_cuda_delta_net_step_f32(
                self.stream.raw(),
                self.d_qkv.as_ref().unwrap().f32_ptr_at(0) as *const f32,
                self.d_gate.as_ref().unwrap().f32_ptr_at(0) as *const f32,
                self.d_decay.as_ref().unwrap().f32_ptr_at(0) as *const f32,
                self.d_beta.as_ref().unwrap().f32_ptr_at(0) as *const f32,
                self.d_norm.as_ref().unwrap().f32_ptr_at(0) as *const f32,
                self.states[li].as_ref().unwrap().f32_ptr_at(0),
                self.d_out.as_ref().unwrap().f32_ptr_at(0),
                n_v_heads as c_int,
                n_qk_heads as c_int,
                qk as c_int,
                vd as c_int,
                q_block as c_int,
                k_block as c_int,
                rms_eps,
            )
        };
        if rc != 0 {
            return false;
        }
        self.d_out.as_ref().unwrap().download(dn_f32_bytes_mut(out))
    }

    /// D2H layer `li`'s resident state into `host` (snapshot / serialize). Returns
    /// false when the layer has no resident buffer (nothing to sync).
    pub fn download_state(&self, li: usize, host: &mut [f32]) -> bool {
        match self.states.get(li).and_then(|o| o.as_ref()) {
            Some(b) if b.cap_bytes >= std::mem::size_of_val(host) => {
                b.download(dn_f32_bytes_mut(host))
            }
            _ => false,
        }
    }

    /// Whether layer `li` currently holds a resident state buffer.
    pub fn has_state(&self, li: usize) -> bool {
        self.states.get(li).map(|o| o.is_some()).unwrap_or(false)
    }

    /// Drop all resident state + scratch (model load / device reset).
    pub fn reset(&mut self) {
        self.states.clear();
        self.d_qkv = None;
        self.d_gate = None;
        self.d_decay = None;
        self.d_beta = None;
        self.d_norm = None;
        self.d_out = None;
    }
}

/// An instantiated CUDA graph executable (`cudaGraphExec_t`) for the Feature-1b
/// device-resident attention block, cached per layer in [`CudaMatvecCache`].
/// Opaque `*mut c_void` handle; freed once on drop via the FFI. The exec is
/// refreshed every decode step (re-captured graph → `exec_update`, or
/// re-instantiate) so it never holds a stale device pointer — but it is still
/// destroyed on model reload (`clear`) and on an attention-shape change.
struct CudaGraphExecHandle {
    exec: *mut c_void,
}

impl Drop for CudaGraphExecHandle {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            // SAFETY: `exec` came from rsl_cuda_graph_instantiate; freed once.
            unsafe { rsl_cuda_graph_exec_destroy(self.exec) };
        }
    }
}

// SAFETY: driven from the single owning worker thread, like CudaStream — the
// handle is never shared concurrently.
unsafe impl Send for CudaGraphExecHandle {}

/// Process-/model-lifetime device-resident weight cache + scratch for
/// the packed-matvec CUDA dispatch. See the module comment above.
pub struct CudaMatvecCache {
    // Declared before `stream` so buffers drop (and free) first.
    weights: HashMap<usize, RawDevBuf>,
    x_scratch: Option<RawDevBuf>,
    out_scratch: Option<RawDevBuf>,
    // Grouped Q4_K expert-FFN scratch (decode fast path): intermediate
    // ff [n_pick*d_ff], the three expert device-ptr arrays packed back-to-back
    // [3*n_pick ptrs], and the routing weights [n_pick].
    ff_scratch: Option<RawDevBuf>,
    ptr_scratch: Option<RawDevBuf>,
    wt_scratch: Option<RawDevBuf>,
    // #2 on-device activation-residency decode (decode_attn_block_f32): a
    // persistent per-layer F32 KV mirror (same role as CudaAttnContext's, but on
    // THIS cache's stream so the whole attention block runs ordered on one
    // stream) + reusable activation/norm scratch. Allocated lazily on first use.
    attn_k_mirror: Vec<Option<RawDevBuf>>,
    attn_v_mirror: Vec<Option<RawDevBuf>>,
    attn_valid_len: Vec<u32>,
    attn_mirror_ctx: (u32, u32, u32), // (n_layers, n_kv_heads*head_dim, max_ctx) — rebuild on change
    attn_act: Option<RawDevBuf>,      // activation slab: hidden|h_norm|q|k|v|attn_out|o|h_norm2
    attn_norms: Option<RawDevBuf>, // norm-weight slab: attn_norm|post_norm|q_norm|k_norm|inv_freq
    // Feature 1b: one cached CUDA-graph executable per layer for the graph-mode
    // attention block (`decode_attn_block_f32_graph`). `None` until the first
    // graph capture for that layer instantiates it; refreshed every step via
    // exec_update. Sized to n_layers (rebuilt on a layer-count change); dropped
    // (execs destroyed) on model reload via `clear`.
    attn_graph_exec: Vec<Option<CudaGraphExecHandle>>,
    // T2 (B1): the ONE cached executable graph for the WHOLE-DECODE-STEP graph —
    // every dense layer's chain captured into a single graph and replayed with one
    // launch + one `stream_sync` per token (vs the per-layer graph's one of each
    // PER layer). `None` until the first step-graph launch; invalidated alongside
    // `attn_graph_exec` on a mirror-ctx change and on `clear`.
    step_graph_exec: Option<CudaGraphExecHandle>,
    // A1: when set, the EAGER decode block routes its Q4_K matvecs through the
    // lossy, verdict-gated W4A8 int8-activation kernel instead of the bit-exact
    // f32 warp matvec. Set per-call at the top of `decode_attn_block_f32` from
    // the caller's `q4k_w4a8_enabled()` decision; forced OFF in the graph twin
    // (the W4A8 wrapper synchronizes, so it is not CUDA-graph-capturable).
    q4k_w4a8: bool,
    // Device-resident PREFILL block (the batched prefill twin of the decode #2
    // block). The caller sets these per-model from the prefill-GEMM verdicts
    // before `prefill_attn_block_f32`, so the batched device-I/O GEMM selects the
    // SAME fast kernel the host-in/host-out prefill path would (weight-reusing
    // Q8_0/Q4_K GEMM). All false ⇒ the bit-exact per-kind batched matvec (still
    // device-resident, just slower) — fail-safe default.
    prefill_q8_0_w8a8_tc: bool,
    prefill_q8_0_f32: bool,
    prefill_q4_k_w8a8_tc: bool,
    prefill_q4_k_f32: bool,
    // Device-resident PREFILL activation slab (the batched analogue of
    // `attn_act`): hidden|h_norm|q|k|v|attn_out|o|h_norm2[|gate|up|act], each
    // ×n_new rows. Separate from the single-row `attn_act` so prefill and decode
    // don't contend for one slab's size.
    prefill_act: Option<RawDevBuf>,
    // AirLLM-style layer streaming: a bounded VRAM window that cycles whole
    // layers' weights through the cache (admit-by-evicting the COLDEST layer)
    // instead of declining once full. `stream_window_layers == 0` (the default)
    // keeps the legacy grow-until-budget behavior — BYTE-IDENTICAL. The dispatch
    // bridge calls `set_layer_tag` before a layer's (or `encode(layer,expert)`'s)
    // matvecs so `ensure_weight` groups + ages weights by layer; these maps are
    // keyed PARALLEL to `weights` so `RawDevBuf` is untouched.
    stream_window_layers: u32,
    current_layer_tag: u32,
    tick: u64,
    layer_keys: HashMap<u32, Vec<usize>>, // layer tag -> its resident weight keys
    layer_last_used: HashMap<u32, u64>,   // layer tag -> last-touched tick
    // B2 (Kindi rmsfold): device-resident folded-qkv scratch — GPU RMSNorm → GPU
    // Hadamard → 3 PTQ1_0 matvecs with the rotated activation kept resident on
    // THIS stream (ordered; no cross-stream race) reading the cached weights.
    // `_hidden`/`_hnorm`/`_xrot` are the k-sized activation stages, `_out` the
    // m-sized projection output; `_wnorm`/`_signs` cache the per-layer norm
    // weight + sign vector keyed by (host_ptr, len) so they upload once (constant
    // across a model's tokens). Declared before `stream` so they free first.
    rmsfold_hidden: Option<RawDevBuf>,
    rmsfold_hnorm: Option<RawDevBuf>,
    rmsfold_xrot: Option<RawDevBuf>,
    rmsfold_out: Option<RawDevBuf>,
    rmsfold_wnorm: Option<RawDevBuf>,
    rmsfold_wnorm_key: Option<(usize, usize)>,
    rmsfold_signs: Option<RawDevBuf>,
    rmsfold_signs_key: Option<(usize, usize)>,
    // B2 FFN (Kindi rmsfold+smfold): the width-`d_ff` legs of the device-resident
    // dense FFN (gate/up projections, the SiLU-gate `act`, and its Hadamard-rotated
    // `arot` down-proj input). The width-`d` stages (rmsnorm/hadamard of the gate/up
    // INPUT) reuse the `rmsfold_*` buffers — qkv and FFN never run in the same layer.
    // `ffn_wnorm` caches the post-attn norm weight SEPARATELY from `rmsfold_wnorm`
    // (attn-norm vs post-attn-norm alternate every layer; a shared key would thrash);
    // `ffn_signs_ff` caches the width-`d_ff` down-input signs. Both host-addr-keyed.
    ffn_gate: Option<RawDevBuf>,
    ffn_up: Option<RawDevBuf>,
    ffn_act: Option<RawDevBuf>,
    ffn_arot: Option<RawDevBuf>,
    ffn_wnorm: Option<RawDevBuf>,
    ffn_wnorm_key: Option<(usize, usize)>,
    ffn_signs_ff: Option<RawDevBuf>,
    ffn_signs_ff_key: Option<(usize, usize)>,
    // Phase 2: a second CUDA stream for PREFETCH — the next layer's weights
    // upload here (async) while the current layer computes on `stream`. Declared
    // before `stream` but after `weights` so prefetch buffers (in `weights`,
    // allocated on this stream) free before it drops. `None` if a second stream
    // couldn't be created (prefetch then degrades to the sync compute-path upload).
    copy_stream: Option<CudaStream>,
    used_bytes: usize,
    budget_bytes: usize,
    stream: CudaStream,
}

/// T2 (B1) — one dense layer's inputs for the whole-decode-step graph
/// ([`CudaMatvecCache::decode_step_graph_f32`]). The caller gathers one of these
/// per layer (weights + norm-weight slices + per-layer SWA window / soft-cap)
/// before the capture; every weight byte-slice + norm slice must outlive the
/// driver call. Mirrors the per-layer `decode_attn_block_f32_graph` args.
#[derive(Clone, Copy)]
pub struct StepLayer<'a> {
    pub wq: (CudaPackedKind, usize, &'a [u8]),
    pub wk: (CudaPackedKind, usize, &'a [u8]),
    pub wv: (CudaPackedKind, usize, &'a [u8]),
    pub wo: (CudaPackedKind, usize, &'a [u8]),
    /// Dense SwiGLU FFN (gate, up, down, d_ff). The step graph requires EVERY
    /// layer dense (FFN in-graph) — a `None` here makes the driver decline.
    pub dense_ffn: Option<(
        (CudaPackedKind, usize, &'a [u8]),
        (CudaPackedKind, usize, &'a [u8]),
        (CudaPackedKind, usize, &'a [u8]),
        usize,
    )>,
    pub attn_norm: &'a [f32],
    pub post_norm: &'a [f32],
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    pub inv_freq: &'a [f32],
    /// Per-layer SWA window (0 = full) + Gemma2 attention-logit soft-cap (0 = off).
    pub window: i32,
    pub softcap: f32,
}

impl CudaMatvecCache {
    /// Create a cache on `device_index` with a device-memory budget for
    /// cached weights (bytes). `None` when no such device / stream fails.
    pub fn new(device_index: u32, budget_bytes: usize) -> Option<Self> {
        let stream = CudaStream::create(device_index)?;
        // Best-effort second stream for prefetch; None just disables overlap.
        let copy_stream = CudaStream::create(device_index);
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
            ff_scratch: None,
            ptr_scratch: None,
            wt_scratch: None,
            attn_k_mirror: Vec::new(),
            attn_v_mirror: Vec::new(),
            attn_valid_len: Vec::new(),
            attn_mirror_ctx: (0, 0, 0),
            attn_act: None,
            attn_norms: None,
            attn_graph_exec: Vec::new(),
            step_graph_exec: None,
            q4k_w4a8: false,
            prefill_q8_0_w8a8_tc: false,
            prefill_q8_0_f32: false,
            prefill_q4_k_w8a8_tc: false,
            prefill_q4_k_f32: false,
            prefill_act: None,
            stream_window_layers: 0,
            current_layer_tag: 0,
            tick: 0,
            layer_keys: HashMap::new(),
            layer_last_used: HashMap::new(),
            copy_stream,
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

    /// Enable AirLLM-style layer streaming: `k > 0` lets `ensure_weight` cycle
    /// whole layers through the cache by evicting the coldest layer to admit a
    /// new one (rather than declining once the budget is full). `0` (default)
    /// keeps the legacy grow-until-budget behavior. `k` is advisory (the actual
    /// window is byte-bounded by `budget_bytes`); it is recorded for diagnostics
    /// and to flip the admission policy on.
    pub fn set_stream_window_layers(&mut self, k: u32) {
        self.stream_window_layers = k;
    }

    /// Tag the layer — or `encode(layer, expert)` for a MoE expert — whose
    /// weights the subsequent `ensure_weight` calls belong to. The dispatch
    /// bridge sets this (under the cache lock) before a layer's matvecs so
    /// streaming can group weights per layer for whole-layer eviction and age
    /// them for the LRU. A no-op effect when streaming is off.
    pub fn set_layer_tag(&mut self, tag: u32) {
        self.current_layer_tag = tag;
    }

    /// Remove `key` from whichever layer group currently holds it (used on the
    /// rare stale-key re-upload path, where a host address now maps to a
    /// different-sized weight).
    fn untrack_key(&mut self, key: usize) {
        for keys in self.layer_keys.values_mut() {
            if let Some(pos) = keys.iter().position(|&k| k == key) {
                keys.swap_remove(pos);
                return;
            }
        }
    }

    /// Evict whole COLDEST layers (never `keep_tag`, the layer being admitted)
    /// until `need` more bytes fit under `budget_bytes`. Whole-layer eviction
    /// guarantees a layer is never left half-resident. Returns false when no
    /// evictable layer remains (everything is the current layer) — the caller
    /// then declines and falls back to CPU.
    fn evict_until_fits(&mut self, need: usize, keep_a: u32, keep_b: u32) -> bool {
        while self.used_bytes + need > self.budget_bytes {
            // Coldest evictable layer: lowest tick, excluding the layer being
            // admitted (`keep_a`) and, for prefetch, the one in flight on the
            // copy stream (`keep_b`; pass `keep_a` again when there is none).
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

    /// AirLLM Phase 2: asynchronously upload `weights` (the NEXT layer's
    /// `(host_key, bytes)` set) into the streaming window on the COPY stream,
    /// overlapping the H2D transfer with the CURRENT layer's compute on the main
    /// stream. Pair with [`Self::wait_prefetch`] at the top of the next layer
    /// before any compute reads these weights. Eviction protects BOTH the current
    /// compute layer and this prefetch target. No-op when streaming is off or no
    /// copy stream exists; any failure abandons the prefetch (the compute path's
    /// synchronous `ensure_weight` fills the gap, so correctness is unaffected).
    pub fn prefetch_layer(&mut self, prefetch_tag: u32, weights: &[(usize, &[u8])]) {
        if self.stream_window_layers == 0 || self.copy_stream.is_none() {
            return;
        }
        self.tick += 1;
        self.layer_last_used.insert(prefetch_tag, self.tick);
        let current = self.current_layer_tag;
        for &(key, bytes) in weights {
            if bytes.is_empty() {
                continue;
            }
            if matches!(self.weights.get(&key), Some(b) if b.cap_bytes == bytes.len()) {
                continue; // already resident
            }
            if let Some(b) = self.weights.remove(&key) {
                self.used_bytes = self.used_bytes.saturating_sub(b.cap_bytes);
                self.untrack_key(key);
            }
            if self.used_bytes + bytes.len() > self.budget_bytes
                && !self.evict_until_fits(bytes.len(), current, prefetch_tag)
            {
                return; // window can't hold current + prefetch; compute will sync-upload
            }
            // Scope the copy-stream borrow to just the alloc so the subsequent
            // `self.weights` mutation doesn't conflict (disjoint fields, but the
            // alloc call borrows one and insert the other).
            let buf = {
                let copy = self.copy_stream.as_ref().unwrap();
                RawDevBuf::alloc(copy, bytes.len())
            };
            let Some(mut buf) = buf else {
                return;
            };
            if !buf.upload_at_async(0, bytes) {
                return;
            }
            self.used_bytes += bytes.len();
            self.weights.insert(key, buf);
            self.layer_keys.entry(prefetch_tag).or_default().push(key);
        }
    }

    /// AirLLM Phase 2: block the host until the copy stream's prefetch uploads
    /// have completed, so the compute issued next (on the main stream) can only
    /// observe fully-uploaded weights. Cheap no-op when there is no copy stream.
    pub fn wait_prefetch(&self) {
        if let Some(copy) = self.copy_stream.as_ref() {
            stream_sync(copy);
        }
    }

    /// Drop all cached weights (freeing their device buffers) and reset the
    /// byte counter; the reusable scratch buffers are kept (model-agnostic,
    /// resized on demand). Call on model reload: the weight-cache key is the
    /// host byte address (`w_bytes.as_ptr()`), which the allocator reuses across
    /// model loads, so a stale entry from the previous model would otherwise
    /// serve a wrong-sized device buffer (out-of-bounds kernel read → illegal
    /// memory access) or wrong data to the next model, and the never-freed
    /// entries accumulate until VRAM OOMs.
    pub fn clear(&mut self) {
        self.weights.clear();
        self.used_bytes = 0;
        // Free the per-stream Q8_0 prefill-GEMM weight repacks: like `weights`,
        // they are keyed on a device address the allocator reuses across model
        // loads, so a stale entry would serve the wrong weight to the next model.
        self.stream.q8_0_repack_reset();
        // B2 rmsfold: the device scratch + the host-addr cache keys are per-model
        // (keyed like `weights`), so a reload must drop them — otherwise a reused
        // host address could skip the re-upload of a DIFFERENT model's norm/signs.
        self.rmsfold_hidden = None;
        self.rmsfold_hnorm = None;
        self.rmsfold_xrot = None;
        self.rmsfold_out = None;
        self.rmsfold_wnorm = None;
        self.rmsfold_wnorm_key = None;
        self.rmsfold_signs = None;
        self.rmsfold_signs_key = None;
        // B2 FFN: same per-model (host-addr-keyed) lifetime as the rmsfold scratch.
        self.ffn_gate = None;
        self.ffn_up = None;
        self.ffn_act = None;
        self.ffn_arot = None;
        self.ffn_wnorm = None;
        self.ffn_wnorm_key = None;
        self.ffn_signs_ff = None;
        self.ffn_signs_ff_key = None;
        // Streaming residency bookkeeping is per-model (keyed by host addr like
        // `weights`), so it must reset on reload too.
        self.layer_keys.clear();
        self.layer_last_used.clear();
        self.tick = 0;
        // The streaming admission policy is per-model (placement re-sets it); a
        // new model must not inherit the previous one's window.
        self.stream_window_layers = 0;
        // Feature 1b: destroy any cached graph executables. Their captured nodes
        // reference this model's device buffers (weights/mirrors/activation slab),
        // which are now evicted; the next model re-captures + re-instantiates from
        // scratch. Empties the Vec (Drop frees each cudaGraphExec_t).
        self.attn_graph_exec = Vec::new();
        // T2: drop the cached whole-step graph exec for the same reason.
        self.step_graph_exec = None;
    }

    fn ensure_scratch(
        slot: &mut Option<RawDevBuf>,
        stream: &CudaStream,
        need_bytes: usize,
    ) -> bool {
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
    /// Returns false if over budget or an alloc/upload fails. Under streaming
    /// (`stream_window_layers > 0`) an over-budget admission EVICTS the coldest
    /// whole layer to make room instead of declining.
    fn ensure_weight(&mut self, weight_key: usize, w_bytes: &[u8]) -> bool {
        // Streaming: age the current layer so the LRU keeps it hot. Untouched
        // (and the maps stay empty) when streaming is off ⇒ byte-identical.
        let tag = self.current_layer_tag;
        if self.stream_window_layers > 0 {
            self.tick += 1;
            self.layer_last_used.insert(tag, self.tick);
        }
        if let Some(existing) = self.weights.get(&weight_key) {
            if existing.cap_bytes == w_bytes.len() {
                return true;
            }
            // Stale entry: the key is the host byte address, which the allocator
            // reuses across model reloads. A reused address now maps to a
            // DIFFERENT-sized weight — serving the old (possibly smaller) device
            // buffer would read out of bounds (illegal memory access). Evict +
            // re-upload the correct bytes. (A clean reload also calls `clear`;
            // this is the per-entry safety net.)
            self.used_bytes = self.used_bytes.saturating_sub(existing.cap_bytes);
            self.weights.remove(&weight_key);
            if self.stream_window_layers > 0 {
                self.untrack_key(weight_key);
            }
        }
        if self.used_bytes + w_bytes.len() > self.budget_bytes {
            // Over budget. Legacy (window 0): decline → the caller falls back to
            // CPU. Streaming: evict the coldest whole layer(s) — never the one
            // being admitted — to cycle the window; decline only if nothing is
            // evictable (everything is the current layer ⇒ window too small).
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
        let prof = phase_prof::enabled();
        let x_bytes: &[u8] = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, k * 4) };
        let t_h2d = prof.then(std::time::Instant::now);
        if !self.x_scratch.as_mut().unwrap().upload(x_bytes) {
            return false;
        }
        let h2d_ns = t_h2d.map_or(0, |t| t.elapsed().as_nanos() as u64);
        let w_ptr = self.weights[&weight_key].ptr;
        let x_ptr = self.x_scratch.as_ref().unwrap().ptr as *const f32;
        let out_ptr = self.out_scratch.as_ref().unwrap().ptr as *mut f32;
        let t_ker = prof.then(std::time::Instant::now);
        // SAFETY: w/x/out are live device buffers on `self.stream` sized
        // for (M,K); the wrapper synchronizes before returning.
        let res = unsafe {
            match kind {
                CudaPackedKind::Ptq1_0 => {
                    matvec_ptq1_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q8_0 => {
                    matvec_q8_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q4_K => {
                    matvec_q4_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q6_K => {
                    matvec_q6_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q5_K => {
                    matvec_q5_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q2_K => {
                    matvec_q2_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q8_K => {
                    matvec_q8_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q4_0 => {
                    matvec_q4_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q5_0 => {
                    matvec_q5_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q4_1 => {
                    matvec_q4_1_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q5_1 => {
                    matvec_q5_1_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq4_Nl => {
                    matvec_iq4_nl_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq4_Xs => {
                    matvec_iq4_xs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq2_Xxs => {
                    matvec_iq2_xxs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq2_Xs => {
                    matvec_iq2_xs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq2_S => {
                    matvec_iq2_s_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq3_Xxs => {
                    matvec_iq3_xxs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq3_S => {
                    matvec_iq3_s_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq1_S => {
                    matvec_iq1_s_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq1_M => {
                    matvec_iq1_m_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Nvfp4 => {
                    matvec_nvfp4_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Mxfp4 => {
                    matvec_mxfp4_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Mxfp6 => {
                    matvec_mxfp6_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Mxfp8 => {
                    matvec_mxfp8_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q3_K => {
                    matvec_q3_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Pq2_0 => {
                    matvec_pq2_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
            }
        };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
        }
        let ker_ns = t_ker.map_or(0, |t| t.elapsed().as_nanos() as u64);
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, m * 4) };
        let t_d2h = prof.then(std::time::Instant::now);
        let ok = self.out_scratch.as_ref().unwrap().download(out_bytes);
        if prof {
            phase_prof::add_single(
                h2d_ns,
                ker_ns,
                t_d2h.map_or(0, |t| t.elapsed().as_nanos() as u64),
            );
        }
        ok
    }

    /// B2 (Kindi rmsfold): device-resident folded Q/K/V projection — the
    /// pre-attention RMSNorm + the Prism block-Hadamard rotation run ON the
    /// device and the rotated activation STAYS resident while all THREE PTQ1_0
    /// projections (`wq`/`wk`/`wv`) read it, ALL on this cache's stream (ordered
    /// — no cross-stream race) with the weights served from the resident weight
    /// cache. Mirrors the SYCL `try_folded_qkv_rmsfold_usm_f32` chain: it
    /// eliminates the per-projection CPU rotation AND the host round-trip the CPU
    /// path pays (CPU rmsnorm → CPU hadamard → upload → GPU matvec, ×3). q/k/v
    /// share one input + one rotation (the WHT depends only on the input + its
    /// width). `norm_weight` / `signs` cache device-side keyed by (host_ptr, len)
    /// so a layer's constants upload once, not per token.
    ///
    /// Returns false (caller runs the BYTE-IDENTICAL CPU path) on any bad shape /
    /// budget / kernel failure, leaving the outputs untouched. PTQ1_0 only.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_ptq1_0_qkv_rmsfold(
        &mut self,
        // `kind` is the folded weight format — `Ptq1_0` or its 2-bit sibling
        // `Pq2_0` (both Prism, k%128, one f16 scale per 128). rmsnorm/hadamard
        // are dtype-agnostic; only the per-projection matvec + row-bytes differ.
        kind: CudaPackedKind,
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
        if !matches!(kind, CudaPackedKind::Ptq1_0 | CudaPackedKind::Pq2_0) {
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
        // alone suffices (no realloc signal needed).
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

    /// B2 (Kindi rmsfold+smfold): device-resident dense FFN for a folded hybrid
    /// layer (PTQ1_0). The whole FFN stays resident on THIS stream: GPU RMSNorm →
    /// Hadamard (gate/up input, "rmsfold") → gate + up matvec → SiLU-gate →
    /// Hadamard (down input, "smfold") → down matvec → one D2H of the result.
    /// Eliminates the two CPU Hadamard rotations + the per-projection host
    /// round-trips the CPU `dense_ffn_one_into` pays. `out` is `w_down(silu(
    /// w_gate(x))·w_up(x))` ONLY (ffn_out) — the caller adds the residual. Width-`d`
    /// stages reuse the `rmsfold_*` scratch (qkv + FFN never run in the same layer);
    /// the width-`d_ff` legs use the `ffn_*` buffers. Returns false ⇒ CPU path.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_ptq1_0_ffn_rmsfold_smfold(
        &mut self,
        // Folded weight format — `Ptq1_0` or the 2-bit `Pq2_0` (see the qkv twin).
        kind: CudaPackedKind,
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
        if !matches!(kind, CudaPackedKind::Ptq1_0 | CudaPackedKind::Pq2_0) {
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
        // Width-`d` stages reuse the rmsfold scratch; width-`d_ff` legs are own.
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
        // FFN norm weight — SEPARATE key from `rmsfold_wnorm` (attn-norm vs
        // post-attn-norm alternate every layer; a shared key would thrash).
        let wn_key = (norm_weight.as_ptr() as usize, d);
        if self.ffn_wnorm_key != Some(wn_key) {
            let b = unsafe { std::slice::from_raw_parts(norm_weight.as_ptr() as *const u8, d * 4) };
            if !self.ffn_wnorm.as_mut().unwrap().upload(b) {
                return false;
            }
            self.ffn_wnorm_key = Some(wn_key);
        }
        // Gate/up-input signs (width `d`) reuse the rmsfold_signs cache — the same
        // width-`d` sign vector qkv uses, so the key never thrashes between them.
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
        // Upload the raw hidden row once.
        let hb = unsafe { std::slice::from_raw_parts(hidden.as_ptr() as *const u8, d * 4) };
        if !self.rmsfold_hidden.as_mut().unwrap().upload(hb) {
            return false;
        }
        let hidden_ptr = self.rmsfold_hidden.as_ref().unwrap().ptr as *const f32;
        let wnorm_ptr = self.ffn_wnorm.as_ref().unwrap().ptr as *const f32;
        let hnorm_ptr = self.rmsfold_hnorm.as_ref().unwrap().ptr as *mut f32;
        // Stage 1 — RMSNorm: hidden → hnorm (d).
        // SAFETY: three live device buffers on `self.stream` sized d; wrapper syncs.
        if unsafe { rmsnorm_dev_f32(&self.stream, hidden_ptr, wnorm_ptr, hnorm_ptr, 1, d, eps) }
            .is_err()
        {
            return false;
        }
        // Stage 2 — rmsfold: hnorm → xrot (d; gate/up input, resident).
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

    /// Device-resident single-row packed matvec — the MoE tiered-expert fast
    /// path. Identical math to [`Self::matvec_packed`], but the weight is
    /// ALREADY on the device (`w_ptr_dev`, e.g. a `cudaMallocManaged` buffer
    /// backing a promoted expert): no `weight_key` lookup and **no host→device
    /// weight upload** — that per-call re-streaming is exactly what the device
    /// tier exists to skip. `w_nbytes` is the allocation length, checked against
    /// the format's row stride exactly as the uploading path does. Only `x`
    /// (H2D) and `out` (D2H) still cross the bus. Returns false (caller falls
    /// through to the upload path / CPU) on bad shape / scratch / kernel
    /// failure, leaving `out` untouched. Weight-buffer ownership stays with the
    /// caller's `Tensor` — this cache never frees `w_ptr_dev`.
    ///
    /// # Safety
    /// `w_ptr_dev` must be a live device-accessible pointer (CUDA device or
    /// managed/unified memory) of at least `w_nbytes` bytes on this cache's
    /// device, valid for the duration of the call.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matvec_packed_dev_resident(
        &mut self,
        kind: CudaPackedKind,
        w_ptr_dev: *const c_void,
        w_nbytes: usize,
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
    ) -> bool {
        if m == 0 || k == 0 || x.len() != k || out.len() != m {
            return false;
        }
        if w_ptr_dev.is_null() {
            return false;
        }
        if k % kind.k_alignment() != 0 || w_nbytes < m * kind.row_bytes(k) {
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
        let w_ptr = w_ptr_dev;
        let x_ptr = self.x_scratch.as_ref().unwrap().ptr as *const f32;
        let out_ptr = self.out_scratch.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: w is a caller-guaranteed live device buffer; x/out are live
        // device scratch on `self.stream` sized for (M,K); the wrapper
        // synchronizes before returning. Kernel dispatch mirrors
        // `matvec_packed` exactly (same per-format row_dot → bit-exact).
        let res = unsafe {
            match kind {
                CudaPackedKind::Ptq1_0 => {
                    matvec_ptq1_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q8_0 => {
                    matvec_q8_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q4_K => {
                    matvec_q4_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q6_K => {
                    matvec_q6_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q5_K => {
                    matvec_q5_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q2_K => {
                    matvec_q2_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q8_K => {
                    matvec_q8_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q4_0 => {
                    matvec_q4_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q5_0 => {
                    matvec_q5_0_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q4_1 => {
                    matvec_q4_1_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q5_1 => {
                    matvec_q5_1_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq4_Nl => {
                    matvec_iq4_nl_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq4_Xs => {
                    matvec_iq4_xs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq2_Xxs => {
                    matvec_iq2_xxs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq2_Xs => {
                    matvec_iq2_xs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq2_S => {
                    matvec_iq2_s_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq3_Xxs => {
                    matvec_iq3_xxs_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq3_S => {
                    matvec_iq3_s_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq1_S => {
                    matvec_iq1_s_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Iq1_M => {
                    matvec_iq1_m_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Nvfp4 => {
                    matvec_nvfp4_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Mxfp4 => {
                    matvec_mxfp4_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Mxfp6 => {
                    matvec_mxfp6_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Mxfp8 => {
                    matvec_mxfp8_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Q3_K => {
                    matvec_q3_k_packed_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k)
                }
                CudaPackedKind::Pq2_0 => {
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

    /// Device-I/O packed matvec — the on-device activation-residency primitive
    /// (#2). Identical math to [`Self::matvec_packed`], but BOTH `x` and `out`
    /// are ALREADY device pointers on this cache's stream, so there is **no
    /// H2D upload of x and no D2H download of out** — the activation stays
    /// resident across ops. The weight is still host-keyed (`weight_key` =
    /// `w_bytes.as_ptr()`), uploaded once + cached via [`Self::ensure_weight`]
    /// (attention/dense layer weights aren't promoted, unlike MoE experts).
    /// Bit-exact to `matvec_packed` (same per-format row_dot). Returns false
    /// (caller falls back to the host-I/O path) on bad shape / budget / kernel
    /// failure; `out_dev` is then undefined and the caller must not consume it.
    ///
    /// # Safety
    /// `x_dev` (≥ `k` f32) and `out_dev` (≥ `m` f32) are live device pointers on
    /// this cache's stream for the call; the wrapper synchronizes before return.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matvec_packed_dev_io(
        &mut self,
        kind: CudaPackedKind,
        weight_key: usize,
        w_bytes: &[u8],
        x_dev: *const f32,
        out_dev: *mut f32,
        m: usize,
        k: usize,
    ) -> bool {
        if m == 0 || k == 0 || x_dev.is_null() || out_dev.is_null() {
            return false;
        }
        if k % kind.k_alignment() != 0 || w_bytes.len() < m * kind.row_bytes(k) {
            return false;
        }
        if !self.ensure_weight(weight_key, w_bytes) {
            return false;
        }
        let w_ptr = self.weights[&weight_key].ptr;
        // SAFETY: w is cached device memory of the right size; x_dev/out_dev are
        // caller-guaranteed live device pointers on this stream; the wrapper
        // synchronizes. Same per-format row_dot as `matvec_packed` → bit-exact.
        let res = unsafe {
            match kind {
                CudaPackedKind::Ptq1_0 => {
                    matvec_ptq1_0_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Q8_0 => {
                    // Aligned repacked decode matvec (A/B via RUSTLLAMA_Q8_REPACK_DECODE):
                    // reuses the prefill-built repack for 32-aligned weight reads. Falls
                    // back to the byte-packed matvec when the repack isn't cached.
                    if q8_repack_decode_enabled() {
                        match matvec_q8_0_repacked(&self.stream, w_ptr, x_dev, out_dev, m, k) {
                            Ok(()) => Ok(()),
                            Err(_) => {
                                matvec_q8_0_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                            }
                        }
                    } else {
                        matvec_q8_0_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                    }
                }
                CudaPackedKind::Q4_K => {
                    // A1: lossy W4A8 int8-activation matvec where its verdict
                    // enabled it (the kernel requires K % 256 == 0); else the
                    // bit-exact f32 warp matvec. Only the eager block sets the
                    // flag — the graph twin forces it off (W4A8 syncs internally,
                    // so it is not capturable).
                    if self.q4k_w4a8 && k % 256 == 0 {
                        matvec_q4_k_w4a8(&self.stream, w_ptr as *const c_void, x_dev, out_dev, m, k)
                    } else {
                        matvec_q4_k_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                    }
                }
                CudaPackedKind::Q6_K => {
                    matvec_q6_k_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Q5_K => {
                    matvec_q5_k_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Q2_K => {
                    matvec_q2_k_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Q8_K => {
                    matvec_q8_k_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Q4_0 => {
                    matvec_q4_0_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Q5_0 => {
                    matvec_q5_0_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Q4_1 => {
                    matvec_q4_1_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Q5_1 => {
                    matvec_q5_1_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Iq4_Nl => {
                    matvec_iq4_nl_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Iq4_Xs => {
                    matvec_iq4_xs_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Iq2_Xxs => {
                    matvec_iq2_xxs_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Iq2_Xs => {
                    matvec_iq2_xs_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Iq2_S => {
                    matvec_iq2_s_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Iq3_Xxs => {
                    matvec_iq3_xxs_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Iq3_S => {
                    matvec_iq3_s_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Iq1_S => {
                    matvec_iq1_s_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Iq1_M => {
                    matvec_iq1_m_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Nvfp4 => {
                    matvec_nvfp4_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Mxfp4 => {
                    matvec_mxfp4_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Mxfp6 => {
                    matvec_mxfp6_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Mxfp8 => {
                    matvec_mxfp8_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Q3_K => {
                    matvec_q3_k_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
                CudaPackedKind::Pq2_0 => {
                    matvec_pq2_0_packed_f32(&self.stream, w_ptr, x_dev, out_dev, m, k)
                }
            }
        };
        res.is_ok() && consume_error_count() == 0
    }

    /// Set the device-resident PREFILL block's GEMM selection from the caller's
    /// per-model verdicts (prefill-GEMM paths), so the batched device-I/O GEMM
    /// picks the SAME fast kernel the host-in/host-out prefill path would. All
    /// false ⇒ the bit-exact per-kind batched matvec.
    pub fn set_prefill_gemm(
        &mut self,
        q8_0_w8a8_tc: bool,
        q8_0_f32: bool,
        q4_k_w8a8_tc: bool,
        q4_k_f32: bool,
    ) {
        self.prefill_q8_0_w8a8_tc = q8_0_w8a8_tc;
        self.prefill_q8_0_f32 = q8_0_f32;
        self.prefill_q4_k_w8a8_tc = q4_k_w8a8_tc;
        self.prefill_q4_k_f32 = q4_k_f32;
    }

    /// Batched device-I/O GEMM — the N-row analogue of
    /// [`Self::matvec_packed_dev_io`] for the device-resident PREFILL block.
    /// `x_dev` = `[n, k]`, `out_dev` = `[n, m]` are activation-slab device
    /// pointers; nothing is uploaded or downloaded (that's the whole point — the
    /// per-op H2D(x)/D2H(out) the host prefill path pays is what this removes).
    /// Q8_0 / Q4_K route to the weight-reusing prefill GEMM when the caller
    /// enabled its verdict via [`Self::set_prefill_gemm`]; every other format (and
    /// the un-verdicted Q8_0/Q4_K) uses the bit-exact per-kind batched matvec.
    /// Returns false on bad shape / unaligned K / a kernel failure.
    ///
    /// # Safety
    /// `w_bytes` outlives the call; `x_dev` (≥ `n*k`) / `out_dev` (≥ `n*m`) are
    /// live device pointers on this cache's stream.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matvec_packed_batched_dev_io(
        &mut self,
        kind: CudaPackedKind,
        weight_key: usize,
        w_bytes: &[u8],
        x_dev: *const f32,
        out_dev: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        if m == 0 || k == 0 || n == 0 || x_dev.is_null() || out_dev.is_null() {
            return false;
        }
        if k % kind.k_alignment() != 0 || w_bytes.len() < m * kind.row_bytes(k) {
            return false;
        }
        if !self.ensure_weight(weight_key, w_bytes) {
            return false;
        }
        let w_ptr = self.weights[&weight_key].ptr;
        // SAFETY: w is cached device memory of the right size; x_dev/out_dev are
        // caller-guaranteed live device pointers on this stream. Same per-format
        // kernels as the host-in/host-out `matvec_packed_batched` /
        // `gemm_packed_batched_q8_0_*`, so bit-identical where they overlap.
        let res = unsafe {
            match kind {
                CudaPackedKind::Q8_0 if self.prefill_q8_0_w8a8_tc && k % 32 == 0 => {
                    gemm_q8_0_w8a8_tc(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Q8_0 if self.prefill_q8_0_f32 && k % 32 == 0 => {
                    gemm_q8_0_f32(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Q4_K if self.prefill_q4_k_w8a8_tc && k % 256 == 0 => {
                    gemm_q4_k_w8a8_tc(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Q4_K if self.prefill_q4_k_f32 && k % 256 == 0 => {
                    gemm_q4_k_f32(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Ptq1_0 => {
                    matvec_ptq1_0_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Q8_0 => {
                    matvec_q8_0_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Q4_K => {
                    matvec_q4_k_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Q6_K => {
                    matvec_q6_k_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Q5_K => {
                    matvec_q5_k_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Q2_K => {
                    matvec_q2_k_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Q8_K => {
                    matvec_q8_k_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Q4_0 => {
                    matvec_q4_0_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Q5_0 => {
                    matvec_q5_0_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Q4_1 => {
                    matvec_q4_1_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Q5_1 => {
                    matvec_q5_1_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Iq4_Nl => {
                    matvec_iq4_nl_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Iq4_Xs => {
                    matvec_iq4_xs_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Iq2_Xxs => {
                    matvec_iq2_xxs_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Iq2_Xs => {
                    matvec_iq2_xs_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Iq2_S => {
                    matvec_iq2_s_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Iq3_Xxs => {
                    matvec_iq3_xxs_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Iq3_S => {
                    matvec_iq3_s_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Iq1_S => {
                    matvec_iq1_s_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Iq1_M => {
                    matvec_iq1_m_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Nvfp4 => {
                    matvec_nvfp4_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Mxfp4 => {
                    matvec_mxfp4_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Mxfp6 => {
                    matvec_mxfp6_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Mxfp8 => {
                    matvec_mxfp8_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Q3_K => {
                    matvec_q3_k_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
                CudaPackedKind::Pq2_0 => {
                    matvec_pq2_0_packed_f32_batched(&self.stream, w_ptr, x_dev, out_dev, m, k, n)
                }
            }
        };
        res.is_ok() && consume_error_count() == 0
    }

    /// On-device activation-residency DECODE attention block (#2). Runs the whole
    /// attention block for one token on THIS cache's stream, keeping the
    /// activation resident on device across every op — so instead of ~5 H2D + ~5
    /// D2H per layer, there is one hidden upload + one small K/V download (for the
    /// host KV cache) + hidden/h_norm2 download. Sequence:
    /// `rmsnorm(hidden)→q/k/v=matvec_dev_io→qk_head_norm→rope→[dl k/v; mirror
    /// write]→flash(q,mirror)→attn_out→o=matvec_dev_io→add_rmsnorm(hidden+=o)`.
    /// Weights are `(kind, host_key, bytes)` tuples (uploaded once via the normal
    /// weight cache). `k_row_out`/`v_row_out` receive the post-RoPE K/V so the
    /// caller appends them to the host KV cache. Returns false (caller runs the
    /// host path, `hidden` untouched since it's only written back on full success)
    /// on bad shape / a mirror gap (`pos > valid` — caller must seed) / any
    /// kernel failure. NOT bit-exact (device rmsnorm/attn parallel reduce) —
    /// verdict + fallback gated by the caller. F32 KV only (quant-KV residency is
    /// a follow-up). No qkv bias (caller falls back if biases are present).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn decode_attn_block_f32(
        &mut self,
        wq: (CudaPackedKind, usize, &[u8]),
        wk: (CudaPackedKind, usize, &[u8]),
        wv: (CudaPackedKind, usize, &[u8]),
        wo: (CudaPackedKind, usize, &[u8]),
        attn_norm: &[f32],
        post_norm: &[f32],
        q_norm: Option<&[f32]>,
        k_norm: Option<&[f32]>,
        inv_freq: &[f32],
        hidden: &mut [f32],
        h_norm2: &mut [f32],
        k_row_out: &mut [f32],
        v_row_out: &mut [f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        d_model: usize,
        pos: usize,
        max_ctx: usize,
        n_layers: usize,
        layer_idx: usize,
        rms_eps: f32,
        // Per-layer SWA window (0 = full causal) + Gemma2 attention-logit
        // soft-cap (0 = off) for THIS layer's internal flash-decode. Passthrough
        // no-ops at 0 so a non-SWA / non-Gemma2 layer is byte-identical to the
        // pre-feature block (see `flash_attn_decode_f32`).
        window: i32,
        softcap: f32,
        // Optional dense SwiGLU FFN run device-resident right after the attention
        // block (gate/up → silu → down → residual), so a whole dense layer's
        // decode stays on device. `Some((w_gate, w_up, w_down, d_ff))`, each
        // weight (packed kind, host key, bytes). `None` ⇒ attention-only (the
        // caller runs the FFN on the host path, e.g. for a MoE layer).
        dense_ffn: Option<(
            (CudaPackedKind, usize, &[u8]),
            (CudaPackedKind, usize, &[u8]),
            (CudaPackedKind, usize, &[u8]),
            usize,
        )>,
        // Cross-layer hidden residency (#2 Phase 2). `upload_hidden=false` ⇒
        // `hidden` is ALREADY resident in the activation slab from the previous
        // layer (skip the H2D upload); `download_hidden=false` ⇒ leave the
        // updated hidden in the slab for the next layer (skip the D2H). The
        // per-layer default is `(true, true)` (Phase-1 behavior). The slab's
        // offset-0 `hidden` survives across calls because same-dim layers don't
        // realloc it and nothing else overwrites offset 0.
        upload_hidden: bool,
        download_hidden: bool,
        // A1: route this block's Q4_K matvecs through the lossy, verdict-gated
        // W4A8 int8-activation kernel (the caller passes `q4k_w4a8_enabled()`).
        // Inert on non-Q4_K weights and on K not divisible by 256.
        w4a8_q4k: bool,
    ) -> bool {
        // Record the per-call W4A8 preference the matvecs below read. Eager path
        // only — the graph twin forces this OFF (W4A8 is not capturable).
        self.q4k_w4a8 = w4a8_q4k;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = dense_ffn.as_ref().map(|f| f.3).unwrap_or(0);
        if hidden.len() != d_model
            || h_norm2.len() != d_model
            || k_row_out.len() != d_kv
            || v_row_out.len() != d_kv
            || attn_norm.len() != d_model
            || post_norm.len() != d_model
            || inv_freq.len() != head_dim / 2
            || head_dim % 2 != 0
            || layer_idx >= n_layers
            || pos >= max_ctx
            || n_kv_heads == 0
            || n_heads % n_kv_heads != 0
        {
            return false;
        }
        // Per-layer F32 KV mirror on this stream; rebuilt on a shape change.
        let mctx = (n_layers as u32, d_kv as u32, max_ctx as u32);
        if self.attn_mirror_ctx != mctx {
            self.attn_k_mirror = (0..n_layers).map(|_| None).collect();
            self.attn_v_mirror = (0..n_layers).map(|_| None).collect();
            self.attn_valid_len = vec![0; n_layers];
            self.attn_mirror_ctx = mctx;
        }
        let valid = self.attn_valid_len[layer_idx];
        if pos as u32 > valid {
            return false; // mirror gap: caller seeds the prefill history / falls back
        }
        // +3·d_ff when a dense FFN follows (gate|up|act appended after h_norm2;
        // ffn_out reuses the now-free `o` region). d_ff = 0 ⇒ attention-only size.
        let act_bytes = (4 * d_model + 2 * d_q + 2 * d_kv + 3 * d_ff) * 4;
        let norm_bytes = (2 * d_model + 2 * head_dim + head_dim / 2) * 4;
        let mir_bytes = d_kv * max_ctx * 4;
        if !Self::ensure_scratch(&mut self.attn_act, &self.stream, act_bytes)
            || !Self::ensure_scratch(&mut self.attn_norms, &self.stream, norm_bytes)
            || !Self::ensure_scratch(&mut self.attn_k_mirror[layer_idx], &self.stream, mir_bytes)
            || !Self::ensure_scratch(&mut self.attn_v_mirror[layer_idx], &self.stream, mir_bytes)
        {
            return false;
        }
        // Activation-slab f32 offsets: hidden|h_norm|q|k|v|attn_out|o|h_norm2.
        let (o_hidden, o_hnorm, o_q, o_k, o_v, o_attn, o_o, o_hn2) = (
            0,
            d_model,
            2 * d_model,
            2 * d_model + d_q,
            2 * d_model + d_q + d_kv,
            2 * d_model + d_q + 2 * d_kv,
            2 * d_model + 2 * d_q + 2 * d_kv,
            3 * d_model + 2 * d_q + 2 * d_kv,
        );
        // Dense-FFN slab offsets (gate|up|act), appended after h_norm2; ffn_out
        // reuses `o_o` (free once step-8's add_rmsnorm consumes the o-projection).
        let ffn_base = 4 * d_model + 2 * d_q + 2 * d_kv;
        let (o_gate, o_up, o_act) = (ffn_base, ffn_base + d_ff, ffn_base + 2 * d_ff);
        // Norm-slab f32 offsets: attn_norm|post_norm|q_norm|k_norm|inv_freq.
        let (no_attn, no_post, no_qn, no_kn, no_if) = (
            0,
            d_model,
            2 * d_model,
            2 * d_model + head_dim,
            2 * d_model + 2 * head_dim,
        );
        let fb = |s: &[f32]| -> &[u8] {
            unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
        };
        // Upload hidden (unless it's already resident from the previous layer —
        // cross-layer residency) + the norm weights.
        if upload_hidden
            && !self
                .attn_act
                .as_mut()
                .unwrap()
                .upload_at(o_hidden * 4, fb(hidden))
        {
            return false;
        }
        {
            let nrm = self.attn_norms.as_mut().unwrap();
            if !nrm.upload_at(no_attn * 4, fb(attn_norm))
                || !nrm.upload_at(no_post * 4, fb(post_norm))
                || !nrm.upload_at(no_if * 4, fb(inv_freq))
            {
                return false;
            }
            if let Some(w) = q_norm {
                if !nrm.upload_at(no_qn * 4, fb(w)) {
                    return false;
                }
            }
            if let Some(w) = k_norm {
                if !nrm.upload_at(no_kn * 4, fb(w)) {
                    return false;
                }
            }
        }
        // Raw device pointers into the two slabs (copied out → no borrow held).
        let (ph, phn, pq, pk, pv, pat, po, phn2, p_gate, p_up, p_act) = {
            let a = self.attn_act.as_ref().unwrap();
            (
                a.f32_ptr_at(o_hidden),
                a.f32_ptr_at(o_hnorm),
                a.f32_ptr_at(o_q),
                a.f32_ptr_at(o_k),
                a.f32_ptr_at(o_v),
                a.f32_ptr_at(o_attn),
                a.f32_ptr_at(o_o),
                a.f32_ptr_at(o_hn2),
                // Valid only when d_ff > 0 (dense FFN); unused otherwise.
                a.f32_ptr_at(o_gate),
                a.f32_ptr_at(o_up),
                a.f32_ptr_at(o_act),
            )
        };
        let (pan, ppn, pqn, pkn, pif) = {
            let n = self.attn_norms.as_ref().unwrap();
            (
                n.f32_ptr_at(no_attn),
                n.f32_ptr_at(no_post),
                n.f32_ptr_at(no_qn),
                n.f32_ptr_at(no_kn),
                n.f32_ptr_at(no_if),
            )
        };
        // Decode-block phase profiler (RUSTLLAMA_PHASE_PROFILE): per-layer
        // matvec / attn / glue / xfer split, sync-isolated (abs is perturbed; the
        // RATIO is the signal). db = [matvec, attn, glue, xfer] ns.
        let prof = phase_prof::enabled();
        let mut db = [0u64; 4];
        let mut tmark = std::time::Instant::now();
        // 1. attn RMSNorm; 2. q/k/v projections (device-I/O, no copy).
        unsafe {
            if rmsnorm_dev_f32(&self.stream, ph, pan, phn, 1, d_model, rms_eps).is_err() {
                return false;
            }
        }
        if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            db[2] += tmark.elapsed().as_nanos() as u64; // glue (rmsnorm)
            tmark = std::time::Instant::now();
        }
        if !unsafe { self.matvec_packed_dev_io(wq.0, wq.1, wq.2, phn, pq, d_q, d_model) }
            || !unsafe { self.matvec_packed_dev_io(wk.0, wk.1, wk.2, phn, pk, d_kv, d_model) }
            || !unsafe { self.matvec_packed_dev_io(wv.0, wv.1, wv.2, phn, pv, d_kv, d_model) }
        {
            return false;
        }
        if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            db[0] += tmark.elapsed().as_nanos() as u64; // matvec (qkv)
            tmark = std::time::Instant::now();
        }
        // 3. Qwen3 qk-head-norm; 4. RoPE (both device, in place).
        unsafe {
            if q_norm.is_some()
                && qk_head_norm_f32(&self.stream, pq, pqn, n_heads, head_dim, rms_eps).is_err()
            {
                return false;
            }
            if k_norm.is_some()
                && qk_head_norm_f32(&self.stream, pk, pkn, n_kv_heads, head_dim, rms_eps).is_err()
            {
                return false;
            }
            if rope_f32(&self.stream, pq, n_heads, head_dim, pos, pif).is_err()
                || rope_f32(&self.stream, pk, n_kv_heads, head_dim, pos, pif).is_err()
            {
                return false;
            }
        }
        if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            db[2] += tmark.elapsed().as_nanos() as u64; // glue (qk-norm + rope)
            tmark = std::time::Instant::now();
        }
        // 5. Download post-RoPE K/V into the caller's host KV-cache row, then
        //    scatter the SAME post-RoPE device K/V into the F32 mirror at `pos`,
        //    one kv-head at a time (head-major). A2: the mirror write is now an
        //    on-chip DEVICE→DEVICE copy straight from the activation slab
        //    (pk/pv) instead of re-uploading the just-downloaded host rows —
        //    byte-identical to the old host round-trip (same source, same mirror
        //    offsets) but without the redundant H2D re-upload. This mirrors the
        //    graph twin's step 5. The host download stays: `k_row_out`/`v_row_out`
        //    are the caller's output (its host KV cache).
        {
            let a = self.attn_act.as_ref().unwrap();
            let kb: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(k_row_out.as_mut_ptr() as *mut u8, d_kv * 4)
            };
            let vb: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(v_row_out.as_mut_ptr() as *mut u8, d_kv * 4)
            };
            if !a.download_at(o_k * 4, kb) || !a.download_at(o_v * 4, vb) {
                return false;
            }
        }
        {
            let km_base = self.attn_k_mirror[layer_idx].as_ref().unwrap().ptr as *mut f32;
            let vm_base = self.attn_v_mirror[layer_idx].as_ref().unwrap().ptr as *mut f32;
            let row_bytes = head_dim * 4;
            for h in 0..n_kv_heads {
                let kd = unsafe { km_base.add((h * max_ctx + pos) * head_dim) } as *mut c_void;
                let ks = unsafe { pk.add(h * head_dim) } as *const c_void;
                let vd = unsafe { vm_base.add((h * max_ctx + pos) * head_dim) } as *mut c_void;
                let vs = unsafe { pv.add(h * head_dim) } as *const c_void;
                if !unsafe { memcpy_d2d_async(&self.stream, kd, ks, row_bytes) }
                    || !unsafe { memcpy_d2d_async(&self.stream, vd, vs, row_bytes) }
                {
                    return false;
                }
            }
        }
        if pos as u32 == valid {
            self.attn_valid_len[layer_idx] = pos as u32 + 1;
        }
        if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            db[3] += tmark.elapsed().as_nanos() as u64; // xfer (K/V download + mirror)
            tmark = std::time::Instant::now();
        }
        // 6. Flash decode over [0, pos+1) against the mirror → attn_out (device).
        let km_ptr = self.attn_k_mirror[layer_idx].as_ref().unwrap().ptr as *const f32;
        let vm_ptr = self.attn_v_mirror[layer_idx].as_ref().unwrap().ptr as *const f32;
        unsafe {
            if flash_attn_decode_f32(
                &self.stream,
                pq,
                km_ptr,
                vm_ptr,
                pat,
                n_heads,
                n_kv_heads,
                head_dim,
                max_ctx,
                pos + 1,
                // Per-layer SWA window / Gemma2 soft-cap for this decode step.
                window,
                softcap,
            )
            .is_err()
            {
                return false;
            }
        }
        if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            db[1] += tmark.elapsed().as_nanos() as u64; // attn (flash-decode)
            tmark = std::time::Instant::now();
        }
        // 7. o projection (device-I/O); 8. fused residual + post-attn RMSNorm.
        if !unsafe { self.matvec_packed_dev_io(wo.0, wo.1, wo.2, pat, po, d_model, d_q) } {
            return false;
        }
        if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            db[0] += tmark.elapsed().as_nanos() as u64; // matvec (o-proj)
            tmark = std::time::Instant::now();
        }
        unsafe {
            if add_rmsnorm_f32(&self.stream, ph, po, ppn, phn2, 1, d_model, rms_eps).is_err() {
                return false;
            }
        }
        if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            db[2] += tmark.elapsed().as_nanos() as u64; // glue (residual + post-attn rmsnorm)
            tmark = std::time::Instant::now();
        }
        // 8b. Optional dense SwiGLU FFN, device-resident: gate/up = W·h_norm2,
        // act = silu(gate)⊙up, ffn_out = W_down·act, hidden += ffn_out. Reads
        // h_norm2 (phn2), writes hidden (ph) in place; ffn_out reuses `po` (the
        // o-projection region, free after step 8). On success the whole dense
        // layer is done on device and the caller skips the host FFN.
        if let Some((wg, wu, wd, _)) = dense_ffn {
            unsafe {
                if !self.matvec_packed_dev_io(wg.0, wg.1, wg.2, phn2, p_gate, d_ff, d_model)
                    || !self.matvec_packed_dev_io(wu.0, wu.1, wu.2, phn2, p_up, d_ff, d_model)
                {
                    return false;
                }
                if silu_mul_f32(&self.stream, p_gate, p_up, p_act, d_ff).is_err() {
                    return false;
                }
                if !self.matvec_packed_dev_io(wd.0, wd.1, wd.2, p_act, po, d_model, d_ff) {
                    return false;
                }
                if add_f32(&self.stream, ph, po, d_model).is_err() {
                    return false;
                }
            }
        }
        if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            db[0] += tmark.elapsed().as_nanos() as u64; // matvec (FFN gate/up/down; silu+add negligible)
            tmark = std::time::Instant::now();
        }
        // 9. Download the updated hidden (hidden += o [+ ffn_out]). The post-attn
        // norm (h_norm2) is downloaded only when the FFN ran on the HOST — when
        // the device FFN above consumed it, the caller skips the host FFN so it's
        // unused.
        {
            let a = self.attn_act.as_ref().unwrap();
            // Download hidden unless it stays resident for the next layer
            // (cross-layer residency); the caller drains it once at loop end (or
            // on a fallback to the host path) via `download_resident_hidden_f32`.
            if download_hidden {
                let hb: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(hidden.as_mut_ptr() as *mut u8, d_model * 4)
                };
                if !a.download_at(o_hidden * 4, hb) {
                    return false;
                }
            }
            if dense_ffn.is_none() {
                let h2b: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(h_norm2.as_mut_ptr() as *mut u8, d_model * 4)
                };
                if !a.download_at(o_hn2 * 4, h2b) {
                    return false;
                }
            }
        }
        if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            db[3] += tmark.elapsed().as_nanos() as u64; // xfer (hidden/h_norm2 download)
            phase_prof::add_decode_block(db[0], db[1], db[2], db[3]);
        }
        consume_error_count() == 0
    }

    /// On-device activation-residency PREFILL block — the batched (n_new-row)
    /// twin of [`Self::decode_attn_block_f32`]. Runs a whole dense layer
    /// (attention + optional SwiGLU FFN) for `n_new` prefill tokens on THIS
    /// cache's stream, keeping the activation slab resident on device across every
    /// op. Instead of ~7 batched GEMMs each paying an H2D(x)+D2H(out) (the prefill
    /// transfer bottleneck — measured ~63% of the batched GEMM time), there is one
    /// hidden upload (layer 0) + one hidden download (last layer) + a small K/V
    /// download for the host KV cache. Sequence per layer:
    /// `rmsnorm→q/k/v=batched_dev_io→qk_head_norm→rope_rows→[dl k/v; mirror
    /// scatter]→flash_prefill(q,mirror)→o=batched_dev_io→add_rmsnorm(hidden+=o)
    /// →[gate/up→silu→down→hidden+=ffn_out]`.
    ///
    /// `hidden` is `[n_new, d_model]` (token-major). `k_rows_out`/`v_rows_out`
    /// receive the post-RoPE K/V `[n_new, d_kv]` (token-major) so the caller
    /// scatters them into its host head-major KV cache. The device KV mirror for
    /// this layer holds `[0, kv_len_base + n_new)` after the call (the prefix is
    /// seeded from the host cache when `kv_len_base > 0`), so a following decode
    /// continues from it. F32 KV only; no qkv bias; same `(upload_hidden,
    /// download_hidden)` cross-layer-residency contract as the decode block.
    /// Returns false (caller runs the host prefill path, `hidden` untouched since
    /// it is only written back on full success) on bad shape / a mirror gap / any
    /// kernel failure. NOT bit-exact (device rmsnorm/attn parallel reduce + the
    /// optional lossy W8A8 GEMM) — verdict + fallback gated by the caller.
    ///
    /// # Safety
    /// Same contract as [`Self::decode_attn_block_f32`]: weight byte-slices live
    /// for the call, host buffers are sized to the dims, the cache owns the
    /// stream + mirrors.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prefill_attn_block_f32(
        &mut self,
        wq: (CudaPackedKind, usize, &[u8]),
        wk: (CudaPackedKind, usize, &[u8]),
        wv: (CudaPackedKind, usize, &[u8]),
        wo: (CudaPackedKind, usize, &[u8]),
        attn_norm: &[f32],
        post_norm: &[f32],
        q_norm: Option<&[f32]>,
        k_norm: Option<&[f32]>,
        inv_freq: &[f32],
        hidden: &mut [f32],     // [n_new * d_model]
        h_norm2: &mut [f32],    // [n_new * d_model] (FFN input when host FFN)
        k_rows_out: &mut [f32], // [n_new * d_kv] post-RoPE K (token-major)
        v_rows_out: &mut [f32], // [n_new * d_kv] post-RoPE V (token-major)
        k_cache: &[f32],        // host head-major KV cache (prefix seed; unused at base 0)
        v_cache: &[f32],
        dense_ffn: Option<(
            (CudaPackedKind, usize, &[u8]),
            (CudaPackedKind, usize, &[u8]),
            (CudaPackedKind, usize, &[u8]),
            usize,
        )>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        d_model: usize,
        kv_len_base: usize,
        n_new: usize,
        max_ctx: usize,
        n_layers: usize,
        layer_idx: usize,
        rms_eps: f32,
        window: i32,
        softcap: f32,
        upload_hidden: bool,
        download_hidden: bool,
    ) -> bool {
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = dense_ffn.as_ref().map(|f| f.3).unwrap_or(0);
        if n_new == 0
            || hidden.len() != n_new * d_model
            || h_norm2.len() != n_new * d_model
            || k_rows_out.len() != n_new * d_kv
            || v_rows_out.len() != n_new * d_kv
            || attn_norm.len() != d_model
            || post_norm.len() != d_model
            || inv_freq.len() != head_dim / 2
            || head_dim % 2 != 0
            || layer_idx >= n_layers
            || kv_len_base + n_new > max_ctx
            || n_kv_heads == 0
            || n_heads % n_kv_heads != 0
        {
            return false;
        }
        // Per-layer F32 KV mirror on this stream (shared with the decode block);
        // rebuilt on a shape change.
        let mctx = (n_layers as u32, d_kv as u32, max_ctx as u32);
        if self.attn_mirror_ctx != mctx {
            self.attn_k_mirror = (0..n_layers).map(|_| None).collect();
            self.attn_v_mirror = (0..n_layers).map(|_| None).collect();
            self.attn_valid_len = vec![0; n_layers];
            self.attn_mirror_ctx = mctx;
        }
        // Seed the mirror prefix [valid, kv_len_base) from the host cache (a no-op
        // for a fresh single-chunk prefill where kv_len_base == 0).
        if (self.attn_valid_len[layer_idx] as usize) < kv_len_base
            && !unsafe {
                self.seed_attn_mirror_f32(
                    layer_idx,
                    k_cache,
                    v_cache,
                    kv_len_base,
                    n_kv_heads,
                    head_dim,
                    max_ctx,
                    n_layers,
                )
            }
        {
            return false;
        }
        if (self.attn_valid_len[layer_idx] as usize) < kv_len_base {
            return false; // mirror gap: caller falls back / seeds the history
        }
        // Prefill activation slab (×n_new): hidden|h_norm|q|k|v|attn_out|o|h_norm2
        // [|gate|up|act]. +3·n·d_ff when a dense FFN follows.
        let act_bytes = (4 * d_model + 2 * d_q + 2 * d_kv + 3 * d_ff) * n_new * 4;
        let norm_bytes = (2 * d_model + 2 * head_dim + head_dim / 2) * 4;
        let mir_bytes = d_kv * max_ctx * 4;
        if !Self::ensure_scratch(&mut self.prefill_act, &self.stream, act_bytes)
            || !Self::ensure_scratch(&mut self.attn_norms, &self.stream, norm_bytes)
            || !Self::ensure_scratch(&mut self.attn_k_mirror[layer_idx], &self.stream, mir_bytes)
            || !Self::ensure_scratch(&mut self.attn_v_mirror[layer_idx], &self.stream, mir_bytes)
        {
            return false;
        }
        // Activation-slab f32 offsets (each region is n_new rows wide).
        let (o_hidden, o_hnorm, o_q, o_k, o_v, o_attn, o_o, o_hn2) = (
            0,
            n_new * d_model,
            2 * n_new * d_model,
            n_new * (2 * d_model + d_q),
            n_new * (2 * d_model + d_q + d_kv),
            n_new * (2 * d_model + d_q + 2 * d_kv),
            n_new * (2 * d_model + 2 * d_q + 2 * d_kv),
            n_new * (3 * d_model + 2 * d_q + 2 * d_kv),
        );
        let ffn_base = n_new * (4 * d_model + 2 * d_q + 2 * d_kv);
        let (o_gate, o_up, o_act) = (
            ffn_base,
            ffn_base + n_new * d_ff,
            ffn_base + 2 * n_new * d_ff,
        );
        // Norm-slab offsets (per-element, shared with decode).
        let (no_attn, no_post, no_qn, no_kn, no_if) = (
            0,
            d_model,
            2 * d_model,
            2 * d_model + head_dim,
            2 * d_model + 2 * head_dim,
        );
        let fb = |s: &[f32]| -> &[u8] {
            unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
        };
        // Per-layer phase profiling (RUSTLLAMA_PHASE_PROFILE): H2D / compute / attn
        // / D2H, each isolated by a stream sync (perturbs absolute time; the RATIO
        // is the signal). pb = [h2d, compute, attn, d2h] ns.
        let prof = phase_prof::enabled();
        let mut pb = [0u64; 4];
        // Sub-split of the compute phase (pb[1]): GEMM-only vs glue. gemm_ns sums
        // the 7 batched GEMMs (qkv/o/gate/up/down); ffn_ns is the FFN subset
        // (gate/up/down, with the tiny silu/add lumped in). Only measured under
        // RUSTLLAMA_PHASE_PROFILE; each group is bracketed with its own stream sync.
        let mut gemm_ns = 0u64;
        let mut ffn_ns = 0u64;
        let mut tmark = std::time::Instant::now();
        if upload_hidden
            && !self
                .prefill_act
                .as_mut()
                .unwrap()
                .upload_at(o_hidden * 4, fb(hidden))
        {
            return false;
        }
        {
            let nrm = self.attn_norms.as_mut().unwrap();
            if !nrm.upload_at(no_attn * 4, fb(attn_norm))
                || !nrm.upload_at(no_post * 4, fb(post_norm))
                || !nrm.upload_at(no_if * 4, fb(inv_freq))
            {
                return false;
            }
            if let Some(w) = q_norm {
                if !nrm.upload_at(no_qn * 4, fb(w)) {
                    return false;
                }
            }
            if let Some(w) = k_norm {
                if !nrm.upload_at(no_kn * 4, fb(w)) {
                    return false;
                }
            }
        }
        if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            pb[0] += tmark.elapsed().as_nanos() as u64;
            tmark = std::time::Instant::now();
        }
        let (ph, phn, pq, pk, pv, pat, po, phn2, p_gate, p_up, p_act) = {
            let a = self.prefill_act.as_ref().unwrap();
            (
                a.f32_ptr_at(o_hidden),
                a.f32_ptr_at(o_hnorm),
                a.f32_ptr_at(o_q),
                a.f32_ptr_at(o_k),
                a.f32_ptr_at(o_v),
                a.f32_ptr_at(o_attn),
                a.f32_ptr_at(o_o),
                a.f32_ptr_at(o_hn2),
                a.f32_ptr_at(o_gate),
                a.f32_ptr_at(o_up),
                a.f32_ptr_at(o_act),
            )
        };
        let (pan, ppn, pqn, pkn, pif) = {
            let n = self.attn_norms.as_ref().unwrap();
            (
                n.f32_ptr_at(no_attn),
                n.f32_ptr_at(no_post),
                n.f32_ptr_at(no_qn),
                n.f32_ptr_at(no_kn),
                n.f32_ptr_at(no_if),
            )
        };
        // 1. attn RMSNorm over n_new rows; 2. q/k/v batched device-I/O GEMMs.
        unsafe {
            if rmsnorm_dev_f32(&self.stream, ph, pan, phn, n_new, d_model, rms_eps).is_err() {
                return false;
            }
        }
        let g_qkv = if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            Some(std::time::Instant::now())
        } else {
            None
        };
        if !unsafe {
            self.matvec_packed_batched_dev_io(wq.0, wq.1, wq.2, phn, pq, d_q, d_model, n_new)
        } || !unsafe {
            self.matvec_packed_batched_dev_io(wk.0, wk.1, wk.2, phn, pk, d_kv, d_model, n_new)
        } || !unsafe {
            self.matvec_packed_batched_dev_io(wv.0, wv.1, wv.2, phn, pv, d_kv, d_model, n_new)
        } {
            return false;
        }
        if let Some(t) = g_qkv {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            gemm_ns += t.elapsed().as_nanos() as u64;
        }
        // 3. Qwen3 per-head q/k RMSNorm over all n_new*heads rows (shared weight);
        // 4. batched RoPE (each row at kv_len_base+row).
        unsafe {
            if q_norm.is_some()
                && qk_head_norm_f32(&self.stream, pq, pqn, n_new * n_heads, head_dim, rms_eps)
                    .is_err()
            {
                return false;
            }
            if k_norm.is_some()
                && qk_head_norm_f32(&self.stream, pk, pkn, n_new * n_kv_heads, head_dim, rms_eps)
                    .is_err()
            {
                return false;
            }
            if rope_rows_f32(&self.stream, pq, n_new, n_heads, head_dim, kv_len_base, pif).is_err()
                || rope_rows_f32(
                    &self.stream,
                    pk,
                    n_new,
                    n_kv_heads,
                    head_dim,
                    kv_len_base,
                    pif,
                )
                .is_err()
            {
                return false;
            }
        }
        if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            pb[1] += tmark.elapsed().as_nanos() as u64; // compute (pre-attn)
            tmark = std::time::Instant::now();
        }
        // 5. Scatter the post-RoPE device K/V into the head-major mirror at col
        //    kv_len_base (device→device, one launch) and advance valid_len. P1c:
        //    the host-cache K/V download is DEFERRED to step 9 (after the FFN) so
        //    the whole attention+FFN chain runs as one async stream segment with no
        //    mid-layer D2H sync; the o_k/o_v slab regions are untouched by the
        //    o-proj/residual/FFN below, so the deferred read is still the post-RoPE
        //    K/V.
        {
            let km = self.attn_k_mirror[layer_idx].as_ref().unwrap().ptr as *mut f32;
            let vm = self.attn_v_mirror[layer_idx].as_ref().unwrap().ptr as *mut f32;
            unsafe {
                if kv_scatter_rows_f32(
                    &self.stream,
                    km,
                    pk,
                    n_new,
                    n_kv_heads,
                    head_dim,
                    max_ctx,
                    kv_len_base,
                )
                .is_err()
                    || kv_scatter_rows_f32(
                        &self.stream,
                        vm,
                        pv,
                        n_new,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_len_base,
                    )
                    .is_err()
                {
                    return false;
                }
            }
        }
        if (self.attn_valid_len[layer_idx] as usize) < kv_len_base + n_new {
            self.attn_valid_len[layer_idx] = (kv_len_base + n_new) as u32;
        }
        // 6. Flash-prefill: n_new causal queries over the mirror [0, kv_len_base+n_new).
        let km_ptr = self.attn_k_mirror[layer_idx].as_ref().unwrap().ptr as *const f32;
        let vm_ptr = self.attn_v_mirror[layer_idx].as_ref().unwrap().ptr as *const f32;
        unsafe {
            if flash_attn_prefill_f32(
                &self.stream,
                pq,
                km_ptr,
                vm_ptr,
                pat,
                n_heads,
                n_kv_heads,
                head_dim,
                max_ctx,
                kv_len_base,
                n_new,
                window,
                softcap,
            )
            .is_err()
            {
                return false;
            }
        }
        if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            pb[2] += tmark.elapsed().as_nanos() as u64; // attention (scatter + flash)
            tmark = std::time::Instant::now();
        }
        // 7. o projection (batched device-I/O); 8. fused residual + post-attn RMSNorm.
        let g_o = if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            Some(std::time::Instant::now())
        } else {
            None
        };
        if !unsafe {
            self.matvec_packed_batched_dev_io(wo.0, wo.1, wo.2, pat, po, d_model, d_q, n_new)
        } {
            return false;
        }
        if let Some(t) = g_o {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            gemm_ns += t.elapsed().as_nanos() as u64;
        }
        unsafe {
            if add_rmsnorm_f32(&self.stream, ph, po, ppn, phn2, n_new, d_model, rms_eps).is_err() {
                return false;
            }
        }
        // 8b. Optional dense SwiGLU FFN, device-resident.
        if let Some((wg, wu, wd, _)) = dense_ffn {
            let g_ffn = if prof {
                unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
                Some(std::time::Instant::now())
            } else {
                None
            };
            unsafe {
                if !self.matvec_packed_batched_dev_io(
                    wg.0, wg.1, wg.2, phn2, p_gate, d_ff, d_model, n_new,
                ) || !self.matvec_packed_batched_dev_io(
                    wu.0, wu.1, wu.2, phn2, p_up, d_ff, d_model, n_new,
                ) {
                    return false;
                }
                if silu_mul_f32(&self.stream, p_gate, p_up, p_act, n_new * d_ff).is_err() {
                    return false;
                }
                if !self
                    .matvec_packed_batched_dev_io(wd.0, wd.1, wd.2, p_act, po, d_model, d_ff, n_new)
                {
                    return false;
                }
                if add_f32(&self.stream, ph, po, n_new * d_model).is_err() {
                    return false;
                }
            }
            // FFN GEMM time (gate/up/down; the tiny silu/add are lumped in). Counts
            // toward both the FFN-specific slice and the whole-GEMM slice.
            if let Some(t) = g_ffn {
                unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
                let e = t.elapsed().as_nanos() as u64;
                ffn_ns += e;
                gemm_ns += e;
            }
        }
        if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            pb[1] += tmark.elapsed().as_nanos() as u64; // compute (post-attn: o + FFN)
            tmark = std::time::Instant::now();
        }
        // 9. Deferred K/V host-cache download (P1c, post-RoPE from the preserved
        //    o_k/o_v slab regions) + the updated hidden (unless it stays resident
        //    for the next layer). h_norm2 is downloaded only when the FFN runs on
        //    the HOST. Batching the K/V download here (one sync at layer end)
        //    instead of mid-layer lets the attention+FFN chain run async.
        {
            let a = self.prefill_act.as_ref().unwrap();
            let kb: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(k_rows_out.as_mut_ptr() as *mut u8, n_new * d_kv * 4)
            };
            let vb: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(v_rows_out.as_mut_ptr() as *mut u8, n_new * d_kv * 4)
            };
            if !a.download_at(o_k * 4, kb) || !a.download_at(o_v * 4, vb) {
                return false;
            }
            if download_hidden {
                let hb: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(
                        hidden.as_mut_ptr() as *mut u8,
                        n_new * d_model * 4,
                    )
                };
                if !a.download_at(o_hidden * 4, hb) {
                    return false;
                }
            }
            if dense_ffn.is_none() {
                let h2b: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(
                        h_norm2.as_mut_ptr() as *mut u8,
                        n_new * d_model * 4,
                    )
                };
                if !a.download_at(o_hn2 * 4, h2b) {
                    return false;
                }
            }
        }
        if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
            pb[3] += tmark.elapsed().as_nanos() as u64; // D2H (K/V + hidden)
            phase_prof::add_prefill_block(pb[0], pb[1], pb[2], pb[3], gemm_ns, ffn_ns);
        }
        consume_error_count() == 0
    }

    /// CUDA-graph-captured twin of [`Self::decode_attn_block_f32`] (Feature 1b).
    /// Runs the IDENTICAL device chain + math, but captures the whole chain into
    /// a `cudaGraph` and replays it as ONE launch, collapsing the per-op CPU
    /// launch/sync cost the eager block pays for every kernel. The result is the
    /// same (same kernels, same inputs) — this only changes HOW the ops are
    /// submitted, so correctness tracks the eager block (verdict-gated by the
    /// caller; graded by `doctor --cuda-parity attn:dev_resident_block_cuda_graph`).
    ///
    /// Capture rules handled here (a sync / a synchronous memcpy inside a capture
    /// region is illegal): the hidden/norm H2D uploads + the weight pre-warm run
    /// EAGER *before* `begin_capture`; the post-RoPE K/V write into the device KV
    /// mirror is an async device→device copy issued INSIDE the capture; and the
    /// post-RoPE K/V + hidden/h_norm2 device→host copies (for the host KV cache /
    /// the FFN input) are DEFERRED to AFTER the single post-launch stream sync.
    /// Per-layer the executable graph is instantiated once then patched in place
    /// (`exec_update`) with the grown kv_len/pos each step; a topology change
    /// re-instantiates.
    ///
    /// FAIL-SOFT: ANY failure at any step (capture begin/end, a kernel launch, an
    /// instantiate/update, the launch or sync) ends the capture, destroys the
    /// partial graph, drains the error latch, and returns `false` — leaving the
    /// host buffers (`hidden`/`h_norm2`/`k_row_out`/`v_row_out`) UNTOUCHED and the
    /// mirror `valid_len` un-advanced, so the caller re-runs the proven eager
    /// block (which re-does the mirror write at `pos`) with no corruption.
    ///
    /// # Safety
    /// Same contract as [`Self::decode_attn_block_f32`]: weight byte-slices live
    /// for the call, host buffers are sized to the dims, and the cache owns the
    /// stream + mirrors.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn decode_attn_block_f32_graph(
        &mut self,
        wq: (CudaPackedKind, usize, &[u8]),
        wk: (CudaPackedKind, usize, &[u8]),
        wv: (CudaPackedKind, usize, &[u8]),
        wo: (CudaPackedKind, usize, &[u8]),
        attn_norm: &[f32],
        post_norm: &[f32],
        q_norm: Option<&[f32]>,
        k_norm: Option<&[f32]>,
        inv_freq: &[f32],
        hidden: &mut [f32],
        h_norm2: &mut [f32],
        k_row_out: &mut [f32],
        v_row_out: &mut [f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        d_model: usize,
        pos: usize,
        max_ctx: usize,
        n_layers: usize,
        layer_idx: usize,
        rms_eps: f32,
        // Per-layer SWA window (0 = full) + Gemma2 attention-logit soft-cap
        // (0 = off); threaded into the captured flash-decode. Passthrough at 0.
        window: i32,
        softcap: f32,
        dense_ffn: Option<(
            (CudaPackedKind, usize, &[u8]),
            (CudaPackedKind, usize, &[u8]),
            (CudaPackedKind, usize, &[u8]),
            usize,
        )>,
        upload_hidden: bool,
        download_hidden: bool,
    ) -> bool {
        let ok = unsafe {
            self.decode_attn_block_f32_graph_impl(
                wq,
                wk,
                wv,
                wo,
                attn_norm,
                post_norm,
                q_norm,
                k_norm,
                inv_freq,
                hidden,
                h_norm2,
                k_row_out,
                v_row_out,
                n_heads,
                n_kv_heads,
                head_dim,
                d_model,
                pos,
                max_ctx,
                n_layers,
                layer_idx,
                rms_eps,
                window,
                softcap,
                dense_ffn,
                upload_hidden,
                download_hidden,
            )
        };
        if !ok {
            // Drain any latched capture/launch error so the caller's eager
            // fallback (same stream) starts from a clean latch — otherwise its
            // first `consume_error_count` would see this failure and decline too.
            let _ = consume_error_count();
        }
        ok
    }

    /// Inner body of [`Self::decode_attn_block_f32_graph`]; returns `false` on any
    /// failure (the public wrapper then drains the error latch). Kept separate so
    /// the drain happens on every failure path without duplicating it.
    #[allow(clippy::too_many_arguments)]
    unsafe fn decode_attn_block_f32_graph_impl(
        &mut self,
        wq: (CudaPackedKind, usize, &[u8]),
        wk: (CudaPackedKind, usize, &[u8]),
        wv: (CudaPackedKind, usize, &[u8]),
        wo: (CudaPackedKind, usize, &[u8]),
        attn_norm: &[f32],
        post_norm: &[f32],
        q_norm: Option<&[f32]>,
        k_norm: Option<&[f32]>,
        inv_freq: &[f32],
        hidden: &mut [f32],
        h_norm2: &mut [f32],
        k_row_out: &mut [f32],
        v_row_out: &mut [f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        d_model: usize,
        pos: usize,
        max_ctx: usize,
        n_layers: usize,
        layer_idx: usize,
        rms_eps: f32,
        // Per-layer SWA window (0 = full) + Gemma2 soft-cap (0 = off); see the
        // eager block. Passthrough no-ops at 0.
        window: i32,
        softcap: f32,
        dense_ffn: Option<(
            (CudaPackedKind, usize, &[u8]),
            (CudaPackedKind, usize, &[u8]),
            (CudaPackedKind, usize, &[u8]),
            usize,
        )>,
        upload_hidden: bool,
        download_hidden: bool,
    ) -> bool {
        // A1: the W4A8 matvec synchronizes internally, so it is NOT CUDA-graph
        // capturable — force the flag off for the whole captured chain so the
        // Q4_K matvecs stay on the bit-exact f32 warp kernel.
        self.q4k_w4a8 = false;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let d_ff = dense_ffn.as_ref().map(|f| f.3).unwrap_or(0);
        // Shape validation — identical to the eager block.
        if hidden.len() != d_model
            || h_norm2.len() != d_model
            || k_row_out.len() != d_kv
            || v_row_out.len() != d_kv
            || attn_norm.len() != d_model
            || post_norm.len() != d_model
            || inv_freq.len() != head_dim / 2
            || head_dim % 2 != 0
            || layer_idx >= n_layers
            || pos >= max_ctx
            || n_kv_heads == 0
            || n_heads % n_kv_heads != 0
        {
            return false;
        }
        // Per-layer F32 KV mirror on this stream; rebuilt on a shape change. On a
        // shape change ALSO drop every cached graph exec (its nodes reference the
        // old-shape mirror/slab pointers, and a different n_kv_heads changes the
        // mirror-scatter topology → an exec_update would fail anyway).
        let mctx = (n_layers as u32, d_kv as u32, max_ctx as u32);
        if self.attn_mirror_ctx != mctx {
            self.attn_k_mirror = (0..n_layers).map(|_| None).collect();
            self.attn_v_mirror = (0..n_layers).map(|_| None).collect();
            self.attn_valid_len = vec![0; n_layers];
            self.attn_mirror_ctx = mctx;
            self.attn_graph_exec = (0..n_layers).map(|_| None).collect();
        }
        // First use (or a layer-count change not caught above): size the exec Vec.
        if self.attn_graph_exec.len() != n_layers {
            self.attn_graph_exec = (0..n_layers).map(|_| None).collect();
        }
        let valid = self.attn_valid_len[layer_idx];
        if pos as u32 > valid {
            return false; // mirror gap: caller seeds the prefill history / falls back
        }
        let act_bytes = (4 * d_model + 2 * d_q + 2 * d_kv + 3 * d_ff) * 4;
        let norm_bytes = (2 * d_model + 2 * head_dim + head_dim / 2) * 4;
        let mir_bytes = d_kv * max_ctx * 4;
        if !Self::ensure_scratch(&mut self.attn_act, &self.stream, act_bytes)
            || !Self::ensure_scratch(&mut self.attn_norms, &self.stream, norm_bytes)
            || !Self::ensure_scratch(&mut self.attn_k_mirror[layer_idx], &self.stream, mir_bytes)
            || !Self::ensure_scratch(&mut self.attn_v_mirror[layer_idx], &self.stream, mir_bytes)
        {
            return false;
        }
        // Activation-slab f32 offsets (same as the eager block).
        let (o_hidden, o_hnorm, o_q, o_k, o_v, o_attn, o_o, o_hn2) = (
            0,
            d_model,
            2 * d_model,
            2 * d_model + d_q,
            2 * d_model + d_q + d_kv,
            2 * d_model + d_q + 2 * d_kv,
            2 * d_model + 2 * d_q + 2 * d_kv,
            3 * d_model + 2 * d_q + 2 * d_kv,
        );
        let ffn_base = 4 * d_model + 2 * d_q + 2 * d_kv;
        let (o_gate, o_up, o_act) = (ffn_base, ffn_base + d_ff, ffn_base + 2 * d_ff);
        let (no_attn, no_post, no_qn, no_kn, no_if) = (
            0,
            d_model,
            2 * d_model,
            2 * d_model + head_dim,
            2 * d_model + 2 * head_dim,
        );
        let fb = |s: &[f32]| -> &[u8] {
            unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
        };

        // ---- Weight pre-warm (BEFORE capture): a weight's first-use upload in
        // `ensure_weight` is a synchronous H2D (illegal inside a capture region),
        // so force all weights resident now. A cached weight is a cheap no-op.
        if !self.ensure_weight(wq.1, wq.2)
            || !self.ensure_weight(wk.1, wk.2)
            || !self.ensure_weight(wv.1, wv.2)
            || !self.ensure_weight(wo.1, wo.2)
        {
            return false;
        }
        if let Some((wg, wu, wd, _)) = dense_ffn {
            if !self.ensure_weight(wg.1, wg.2)
                || !self.ensure_weight(wu.1, wu.2)
                || !self.ensure_weight(wd.1, wd.2)
            {
                return false;
            }
        }

        // ---- Host→device uploads (BEFORE capture): hidden (unless resident from
        // the previous layer) + the per-layer norm weights. Async-issued on the
        // stream, then a stream sync leaves the device idle so capture starts
        // clean. (On pageable host memory these complete synchronously anyway.)
        if upload_hidden
            && !self
                .attn_act
                .as_mut()
                .unwrap()
                .upload_at_async(o_hidden * 4, fb(hidden))
        {
            return false;
        }
        {
            let nrm = self.attn_norms.as_mut().unwrap();
            if !nrm.upload_at_async(no_attn * 4, fb(attn_norm))
                || !nrm.upload_at_async(no_post * 4, fb(post_norm))
                || !nrm.upload_at_async(no_if * 4, fb(inv_freq))
            {
                return false;
            }
            if let Some(w) = q_norm {
                if !nrm.upload_at_async(no_qn * 4, fb(w)) {
                    return false;
                }
            }
            if let Some(w) = k_norm {
                if !nrm.upload_at_async(no_kn * 4, fb(w)) {
                    return false;
                }
            }
        }
        if !stream_sync(&self.stream) {
            return false;
        }

        // Raw device pointers into the two slabs + the mirror (copied out → no
        // borrow held across the capture chain's `&mut self` matvec calls).
        let (ph, phn, pq, pk, pv, pat, po, phn2, p_gate, p_up, p_act) = {
            let a = self.attn_act.as_ref().unwrap();
            (
                a.f32_ptr_at(o_hidden),
                a.f32_ptr_at(o_hnorm),
                a.f32_ptr_at(o_q),
                a.f32_ptr_at(o_k),
                a.f32_ptr_at(o_v),
                a.f32_ptr_at(o_attn),
                a.f32_ptr_at(o_o),
                a.f32_ptr_at(o_hn2),
                a.f32_ptr_at(o_gate),
                a.f32_ptr_at(o_up),
                a.f32_ptr_at(o_act),
            )
        };
        let (pan, ppn, pqn, pkn, pif) = {
            let n = self.attn_norms.as_ref().unwrap();
            (
                n.f32_ptr_at(no_attn),
                n.f32_ptr_at(no_post),
                n.f32_ptr_at(no_qn),
                n.f32_ptr_at(no_kn),
                n.f32_ptr_at(no_if),
            )
        };
        let km_base = self.attn_k_mirror[layer_idx].as_ref().unwrap().ptr as *mut f32;
        let vm_base = self.attn_v_mirror[layer_idx].as_ref().unwrap().ptr as *mut f32;

        // ================= BEGIN CAPTURE =================
        if !graph_begin_capture(&self.stream) {
            return false;
        }
        // Run the whole device chain; accumulate success WITHOUT early-returning
        // (we must always reach `end_capture` to restore the stream). `&&`
        // short-circuits so once a step fails the rest are skipped.
        // Issue the whole device chain (T2: factored into `issue_layer_chain` so a
        // whole-decode-step graph can replay the SAME chain for N layers inside one
        // capture; here it is the N=1 case — byte-for-byte the former inline body).
        let chain_ok = unsafe {
            self.issue_layer_chain(
                ph,
                phn,
                pq,
                pk,
                pv,
                pat,
                po,
                phn2,
                p_gate,
                p_up,
                p_act,
                pan,
                ppn,
                pqn,
                pkn,
                pif,
                km_base,
                vm_base,
                wq,
                wk,
                wv,
                wo,
                dense_ffn,
                q_norm.is_some(),
                k_norm.is_some(),
                n_heads,
                n_kv_heads,
                head_dim,
                d_model,
                d_q,
                d_kv,
                d_ff,
                pos,
                max_ctx,
                window,
                softcap,
                rms_eps,
            )
        };
        // ================= END CAPTURE (always) =================
        let graph = graph_end_capture(&self.stream);
        let Some(graph) = graph else {
            // Capture failed/invalidated; the stream was reset by end_capture.
            return false;
        };
        if !chain_ok {
            unsafe { graph_destroy(graph) };
            return false;
        }

        // Instantiate (first time for this layer) or patch the cached exec in
        // place (subsequent steps — cheap; updates the grown kv_len/pos + the
        // mirror-scatter offset). A failed update re-instantiates. The captured
        // graph is freed once the exec reflects it (the exec is independent).
        let have_exec = self.attn_graph_exec[layer_idx].is_some();
        let mut launch_exec_ok = true;
        if have_exec {
            let updated = {
                let exec = self.attn_graph_exec[layer_idx].as_ref().unwrap();
                unsafe { graph_exec_update(exec, graph) }
            };
            if !updated {
                // Topology changed (e.g. FFN toggled): drop + re-instantiate.
                self.attn_graph_exec[layer_idx] = None;
                match unsafe { graph_instantiate(graph) } {
                    Some(e) => self.attn_graph_exec[layer_idx] = Some(e),
                    None => launch_exec_ok = false,
                }
            }
        } else {
            match unsafe { graph_instantiate(graph) } {
                Some(e) => self.attn_graph_exec[layer_idx] = Some(e),
                None => launch_exec_ok = false,
            }
        }
        unsafe { graph_destroy(graph) };
        if !launch_exec_ok {
            return false;
        }

        // Launch the executable graph, then ONE stream sync.
        {
            let exec = self.attn_graph_exec[layer_idx].as_ref().unwrap();
            if !unsafe { graph_launch(exec, &self.stream) } {
                return false;
            }
        }
        if !stream_sync(&self.stream) {
            return false;
        }
        if consume_error_count() != 0 {
            return false;
        }

        // Mirror row [pos] is now written by the graph; advance the valid range
        // exactly like the eager block (only when filling the next contiguous pos).
        if pos as u32 == valid {
            self.attn_valid_len[layer_idx] = pos as u32 + 1;
        }

        // ---- DEFERRED device→host copies (AFTER the sync, so the graph has
        // completed): the post-RoPE K/V row into the host KV cache, and the
        // updated hidden / post-attn norm. Async-issued on the now-idle stream
        // (on pageable host memory these complete synchronously).
        {
            let a = self.attn_act.as_ref().unwrap();
            let kb: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(k_row_out.as_mut_ptr() as *mut u8, d_kv * 4)
            };
            let vb: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(v_row_out.as_mut_ptr() as *mut u8, d_kv * 4)
            };
            if !a.download_at_async(o_k * 4, kb) || !a.download_at_async(o_v * 4, vb) {
                return false;
            }
            if download_hidden {
                let hb: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(hidden.as_mut_ptr() as *mut u8, d_model * 4)
                };
                if !a.download_at_async(o_hidden * 4, hb) {
                    return false;
                }
            }
            if dense_ffn.is_none() {
                let h2b: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(h_norm2.as_mut_ptr() as *mut u8, d_model * 4)
                };
                if !a.download_at_async(o_hn2 * 4, h2b) {
                    return false;
                }
            }
        }
        consume_error_count() == 0
    }

    /// T2 (B1) — WHOLE-DECODE-STEP CUDA graph: capture EVERY dense layer's chain
    /// (`issue_layer_chain`) into ONE graph and replay it with ONE `graph_launch`
    /// + ONE `stream_sync` per token — vs the per-layer graph's one launch + one
    /// sync PER layer. The captured kernels are the SAME as the per-layer graph,
    /// so the result is byte-identical to (and correctness-anchored by) the
    /// validated per-layer path; this removes `(n_layers-1)` syncs + the per-layer
    /// host round-trips per token. PURE-DENSE only: every `StepLayer` must carry
    /// `dense_ffn` (FFN in-graph) — the caller gates on that + the
    /// `step:dev_resident_cuda_graph` verdict + F32-KV / no-bias / no-ring / no-YaRN
    /// / no-streaming preconditions.
    ///
    /// `hidden` is the d_model residual stream (in: layer-0 input; out: post-stack,
    /// pre-final-norm). `k_caches[i]`/`v_caches[i]` are layer i's host KV cache
    /// (`d_kv*max_ctx`, head-major): they SEED each layer's device mirror with the
    /// prefill history before capture, and RECEIVE this token's K/V at `[pos]`
    /// after the single sync — read back from each layer's PERSISTENT mirror (the
    /// shared activation slab holds only the LAST layer's, so it can't be used).
    ///
    /// SAFETY: every weight byte-slice + norm slice must outlive the call and be
    /// sized to the dims; host caches sized `d_kv*max_ctx`; `pos < max_ctx`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn decode_step_graph_f32(
        &mut self,
        layers: &[StepLayer],
        hidden: &mut [f32],
        k_caches: &mut [&mut [f32]],
        v_caches: &mut [&mut [f32]],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        d_model: usize,
        pos: usize,
        max_ctx: usize,
        rms_eps: f32,
    ) -> bool {
        // W4A8 self-syncs ⇒ uncapturable; force the bit-exact f32 warp matvec for
        // the whole step (same invariant as the per-layer graph).
        self.q4k_w4a8 = false;
        let n_layers = layers.len();
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        if n_layers == 0
            || hidden.len() != d_model
            || head_dim % 2 != 0
            || n_kv_heads == 0
            || n_heads % n_kv_heads != 0
            || pos >= max_ctx
            || k_caches.len() != n_layers
            || v_caches.len() != n_layers
        {
            return false;
        }
        // Every layer must be dense with a uniform d_ff + matching norm shapes.
        let Some((_, _, _, d_ff)) = layers[0].dense_ffn else {
            return false;
        };
        for l in layers {
            if l.dense_ffn.map(|f| f.3) != Some(d_ff)
                || l.attn_norm.len() != d_model
                || l.post_norm.len() != d_model
                || l.inv_freq.len() != head_dim / 2
            {
                return false;
            }
        }
        // Mirror ctx: rebuild mirrors + BOTH exec caches on a shape change.
        let mctx = (n_layers as u32, d_kv as u32, max_ctx as u32);
        if self.attn_mirror_ctx != mctx {
            self.attn_k_mirror = (0..n_layers).map(|_| None).collect();
            self.attn_v_mirror = (0..n_layers).map(|_| None).collect();
            self.attn_valid_len = vec![0; n_layers];
            self.attn_mirror_ctx = mctx;
            self.attn_graph_exec = (0..n_layers).map(|_| None).collect();
            self.step_graph_exec = None;
        }
        if self.attn_valid_len.len() != n_layers {
            self.attn_valid_len = vec![0; n_layers];
        }

        // Slab sizing: ONE activation slab (shared across layers — offset-0 hidden
        // carries the residual stream from layer to layer) + an N-LAYER norm slab
        // (each layer's norms must be resident before capture; token-invariant but
        // re-uploaded here — tiny vs the removed per-layer syncs).
        let act_bytes = (4 * d_model + 2 * d_q + 2 * d_kv + 3 * d_ff) * 4;
        let norm_floats = 2 * d_model + 2 * head_dim + head_dim / 2;
        let mir_bytes = d_kv * max_ctx * 4;
        if !Self::ensure_scratch(&mut self.attn_act, &self.stream, act_bytes)
            || !Self::ensure_scratch(
                &mut self.attn_norms,
                &self.stream,
                norm_floats * n_layers * 4,
            )
        {
            return false;
        }
        for i in 0..n_layers {
            if !Self::ensure_scratch(&mut self.attn_k_mirror[i], &self.stream, mir_bytes)
                || !Self::ensure_scratch(&mut self.attn_v_mirror[i], &self.stream, mir_bytes)
            {
                return false;
            }
            // Fill any prefill gap for this layer from its host cache (pre-capture
            // H2D), then require contiguity to `pos` — else decline the step graph.
            if (pos as u32) > self.attn_valid_len[i]
                && !self.seed_attn_mirror_f32(
                    i,
                    &k_caches[i][..],
                    &v_caches[i][..],
                    pos,
                    n_kv_heads,
                    head_dim,
                    max_ctx,
                    n_layers,
                )
            {
                return false;
            }
            if (pos as u32) > self.attn_valid_len[i] {
                return false;
            }
        }

        let fb = |s: &[f32]| -> &[u8] {
            unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
        };
        // Weight pre-warm (BEFORE capture): all 7 weights for every layer.
        for l in layers {
            if !self.ensure_weight(l.wq.1, l.wq.2)
                || !self.ensure_weight(l.wk.1, l.wk.2)
                || !self.ensure_weight(l.wv.1, l.wv.2)
                || !self.ensure_weight(l.wo.1, l.wo.2)
            {
                return false;
            }
            if let Some((wg, wu, wd, _)) = l.dense_ffn {
                if !self.ensure_weight(wg.1, wg.2)
                    || !self.ensure_weight(wu.1, wu.2)
                    || !self.ensure_weight(wd.1, wd.2)
                {
                    return false;
                }
            }
        }
        // H2D uploads (BEFORE capture): hidden once + every layer's norms into its
        // own `norm_floats`-sized block of the slab.
        let (no_attn, no_post, no_qn, no_kn, no_if) = (
            0,
            d_model,
            2 * d_model,
            2 * d_model + head_dim,
            2 * d_model + 2 * head_dim,
        );
        if !self
            .attn_act
            .as_mut()
            .unwrap()
            .upload_at_async(0, fb(hidden))
        {
            return false;
        }
        {
            let nrm = self.attn_norms.as_mut().unwrap();
            for (i, l) in layers.iter().enumerate() {
                let base = i * norm_floats;
                if !nrm.upload_at_async((base + no_attn) * 4, fb(l.attn_norm))
                    || !nrm.upload_at_async((base + no_post) * 4, fb(l.post_norm))
                    || !nrm.upload_at_async((base + no_if) * 4, fb(l.inv_freq))
                {
                    return false;
                }
                if let Some(w) = l.q_norm {
                    if !nrm.upload_at_async((base + no_qn) * 4, fb(w)) {
                        return false;
                    }
                }
                if let Some(w) = l.k_norm {
                    if !nrm.upload_at_async((base + no_kn) * 4, fb(w)) {
                        return false;
                    }
                }
            }
        }
        if !stream_sync(&self.stream) {
            return false;
        }

        // Shared activation-slab ptrs (same offsets as the per-layer block; reused
        // by every layer — the chain overwrites them per layer but offset-0 hidden
        // carries the residual forward).
        let (o_hidden, o_hnorm, o_q, o_k, o_v, o_attn, o_o, o_hn2) = (
            0,
            d_model,
            2 * d_model,
            2 * d_model + d_q,
            2 * d_model + d_q + d_kv,
            2 * d_model + d_q + 2 * d_kv,
            2 * d_model + 2 * d_q + 2 * d_kv,
            3 * d_model + 2 * d_q + 2 * d_kv,
        );
        let ffn_base = 4 * d_model + 2 * d_q + 2 * d_kv;
        let (o_gate, o_up, o_act) = (ffn_base, ffn_base + d_ff, ffn_base + 2 * d_ff);
        let (ph, phn, pq, pk, pv, pat, po, phn2, p_gate, p_up, p_act) = {
            let a = self.attn_act.as_ref().unwrap();
            (
                a.f32_ptr_at(o_hidden),
                a.f32_ptr_at(o_hnorm),
                a.f32_ptr_at(o_q),
                a.f32_ptr_at(o_k),
                a.f32_ptr_at(o_v),
                a.f32_ptr_at(o_attn),
                a.f32_ptr_at(o_o),
                a.f32_ptr_at(o_hn2),
                a.f32_ptr_at(o_gate),
                a.f32_ptr_at(o_up),
                a.f32_ptr_at(o_act),
            )
        };
        // Per-layer norm + mirror ptrs (copied out as raw ptrs → no borrow held
        // across the capture loop's `&mut self` calls).
        type LayerPtrs = (
            *mut f32,
            *mut f32,
            *mut f32,
            *mut f32,
            *mut f32,
            *mut f32,
            *mut f32,
        );
        let layer_ptrs: Vec<LayerPtrs> = (0..n_layers)
            .map(|i| {
                let base = i * norm_floats;
                let n = self.attn_norms.as_ref().unwrap();
                (
                    n.f32_ptr_at(base + no_attn),
                    n.f32_ptr_at(base + no_post),
                    n.f32_ptr_at(base + no_qn),
                    n.f32_ptr_at(base + no_kn),
                    n.f32_ptr_at(base + no_if),
                    self.attn_k_mirror[i].as_ref().unwrap().ptr as *mut f32,
                    self.attn_v_mirror[i].as_ref().unwrap().ptr as *mut f32,
                )
            })
            .collect();

        // ================= BEGIN CAPTURE (one for the whole step) =================
        if !graph_begin_capture(&self.stream) {
            return false;
        }
        let mut chain_ok = true;
        for (i, l) in layers.iter().enumerate() {
            let (pan, ppn, pqn, pkn, pif, km_base, vm_base) = layer_ptrs[i];
            chain_ok = chain_ok
                && unsafe {
                    self.issue_layer_chain(
                        ph,
                        phn,
                        pq,
                        pk,
                        pv,
                        pat,
                        po,
                        phn2,
                        p_gate,
                        p_up,
                        p_act,
                        pan,
                        ppn,
                        pqn,
                        pkn,
                        pif,
                        km_base,
                        vm_base,
                        l.wq,
                        l.wk,
                        l.wv,
                        l.wo,
                        l.dense_ffn,
                        l.q_norm.is_some(),
                        l.k_norm.is_some(),
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        d_model,
                        d_q,
                        d_kv,
                        d_ff,
                        pos,
                        max_ctx,
                        l.window,
                        l.softcap,
                        rms_eps,
                    )
                };
        }
        // ================= END CAPTURE (always) =================
        let graph = graph_end_capture(&self.stream);
        let Some(graph) = graph else {
            return false;
        };
        if !chain_ok {
            unsafe { graph_destroy(graph) };
            return false;
        }
        // Patch the cached step-exec in place (subsequent tokens — cheap) or
        // instantiate first time / on topology change.
        let mut launch_ok = true;
        if self.step_graph_exec.is_some() {
            let updated = {
                let e = self.step_graph_exec.as_ref().unwrap();
                unsafe { graph_exec_update(e, graph) }
            };
            if !updated {
                self.step_graph_exec = None;
                match unsafe { graph_instantiate(graph) } {
                    Some(e) => self.step_graph_exec = Some(e),
                    None => launch_ok = false,
                }
            }
        } else {
            match unsafe { graph_instantiate(graph) } {
                Some(e) => self.step_graph_exec = Some(e),
                None => launch_ok = false,
            }
        }
        unsafe { graph_destroy(graph) };
        if !launch_ok {
            return false;
        }
        {
            let e = self.step_graph_exec.as_ref().unwrap();
            if !unsafe { graph_launch(e, &self.stream) } {
                return false;
            }
        }
        if !stream_sync(&self.stream) {
            return false;
        }
        if consume_error_count() != 0 {
            return false;
        }

        // Advance each layer's valid range (mirror row [pos] is now written).
        for i in 0..n_layers {
            if pos as u32 == self.attn_valid_len[i] {
                self.attn_valid_len[i] = pos as u32 + 1;
            }
        }
        // DEFERRED D2H (after the single sync): this token's K/V for each layer
        // read FROM ITS MIRROR (the shared slab holds only the last layer's) into
        // the host KV cache at `[pos]`, head-major; plus the final hidden once.
        for i in 0..n_layers {
            for h in 0..n_kv_heads {
                let off = (h * max_ctx + pos) * head_dim;
                let kb: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(
                        k_caches[i][off..off + head_dim].as_mut_ptr() as *mut u8,
                        head_dim * 4,
                    )
                };
                let vb: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(
                        v_caches[i][off..off + head_dim].as_mut_ptr() as *mut u8,
                        head_dim * 4,
                    )
                };
                let km_ok = self.attn_k_mirror[i]
                    .as_ref()
                    .unwrap()
                    .download_at_async(off * 4, kb);
                let vm_ok = self.attn_v_mirror[i]
                    .as_ref()
                    .unwrap()
                    .download_at_async(off * 4, vb);
                if !km_ok || !vm_ok {
                    return false;
                }
            }
        }
        {
            let a = self.attn_act.as_ref().unwrap();
            let hb: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(hidden.as_mut_ptr() as *mut u8, d_model * 4)
            };
            if !a.download_at_async(0, hb) {
                return false;
            }
        }
        consume_error_count() == 0
    }

    /// T2 — issue ONE layer's captured device chain (rmsnorm → q/k/v → optional
    /// qk-head-norm → RoPE → KV-mirror scatter → flash-decode → o-proj → fused
    /// residual+post-norm → optional dense SwiGLU FFN) onto `self.stream`, using
    /// device pointers the caller already copied out of the slabs/mirror (so no
    /// slab borrow is held across the `&mut self` matvec calls). Issues NO
    /// begin/end-capture and NO sync — so `decode_attn_block_f32_graph_impl` wraps
    /// a single call in one capture (the N=1 case, behavior-identical to the
    /// former inline body), and a whole-decode-step graph can issue it for every
    /// dense layer inside ONE capture (T2 B1). Accumulates success with `&&` so a
    /// failed op skips the rest without early-returning (the caller must still
    /// reach `end_capture`). `has_q_norm`/`has_k_norm` replace the former
    /// `q_norm.is_some()`/`k_norm.is_some()` checks (the caller owns the slices).
    ///
    /// SAFETY: every pointer must be a live device buffer on `self.stream` sized
    /// for the (n_heads,n_kv_heads,head_dim,d_model,d_ff) dims, and `pos < max_ctx`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn issue_layer_chain(
        &mut self,
        ph: *mut f32,
        phn: *mut f32,
        pq: *mut f32,
        pk: *mut f32,
        pv: *mut f32,
        pat: *mut f32,
        po: *mut f32,
        phn2: *mut f32,
        p_gate: *mut f32,
        p_up: *mut f32,
        p_act: *mut f32,
        pan: *mut f32,
        ppn: *mut f32,
        pqn: *mut f32,
        pkn: *mut f32,
        pif: *mut f32,
        km_base: *mut f32,
        vm_base: *mut f32,
        wq: (CudaPackedKind, usize, &[u8]),
        wk: (CudaPackedKind, usize, &[u8]),
        wv: (CudaPackedKind, usize, &[u8]),
        wo: (CudaPackedKind, usize, &[u8]),
        dense_ffn: Option<(
            (CudaPackedKind, usize, &[u8]),
            (CudaPackedKind, usize, &[u8]),
            (CudaPackedKind, usize, &[u8]),
            usize,
        )>,
        has_q_norm: bool,
        has_k_norm: bool,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        d_model: usize,
        d_q: usize,
        d_kv: usize,
        d_ff: usize,
        pos: usize,
        max_ctx: usize,
        window: i32,
        softcap: f32,
        rms_eps: f32,
    ) -> bool {
        let mut chain_ok = true;
        // 1. attn RMSNorm.
        chain_ok = chain_ok
            && unsafe { rmsnorm_dev_f32(&self.stream, ph, pan, phn, 1, d_model, rms_eps).is_ok() };
        // 2. q/k/v projections (device-I/O; weights pre-warmed so no upload).
        chain_ok = chain_ok
            && unsafe { self.matvec_packed_dev_io(wq.0, wq.1, wq.2, phn, pq, d_q, d_model) };
        chain_ok = chain_ok
            && unsafe { self.matvec_packed_dev_io(wk.0, wk.1, wk.2, phn, pk, d_kv, d_model) };
        chain_ok = chain_ok
            && unsafe { self.matvec_packed_dev_io(wv.0, wv.1, wv.2, phn, pv, d_kv, d_model) };
        // 3. Qwen3 qk-head-norm (in place, device).
        if has_q_norm {
            chain_ok = chain_ok
                && unsafe {
                    qk_head_norm_f32(&self.stream, pq, pqn, n_heads, head_dim, rms_eps).is_ok()
                };
        }
        if has_k_norm {
            chain_ok = chain_ok
                && unsafe {
                    qk_head_norm_f32(&self.stream, pk, pkn, n_kv_heads, head_dim, rms_eps).is_ok()
                };
        }
        // 4. RoPE (both device, in place) at absolute `pos`.
        chain_ok =
            chain_ok && unsafe { rope_f32(&self.stream, pq, n_heads, head_dim, pos, pif).is_ok() };
        chain_ok = chain_ok
            && unsafe { rope_f32(&self.stream, pk, n_kv_heads, head_dim, pos, pif).is_ok() };
        // 5. Scatter the post-RoPE K / plain V into the F32 KV mirror at `pos`,
        //    one kv-head at a time (head-major), via async DEVICE→DEVICE copies
        //    (NO host round-trip — that is what makes the chain capturable). The
        //    device→host copy for the host KV cache is deferred past the sync.
        let row_bytes = head_dim * 4;
        for h in 0..n_kv_heads {
            let kd = unsafe { km_base.add((h * max_ctx + pos) * head_dim) } as *mut c_void;
            let ks = unsafe { pk.add(h * head_dim) } as *const c_void;
            chain_ok = chain_ok && unsafe { memcpy_d2d_async(&self.stream, kd, ks, row_bytes) };
            let vd = unsafe { vm_base.add((h * max_ctx + pos) * head_dim) } as *mut c_void;
            let vs = unsafe { pv.add(h * head_dim) } as *const c_void;
            chain_ok = chain_ok && unsafe { memcpy_d2d_async(&self.stream, vd, vs, row_bytes) };
        }
        // 6. Flash decode over [0, pos+1) against the mirror → attn_out (device).
        chain_ok = chain_ok
            && unsafe {
                flash_attn_decode_f32(
                    &self.stream,
                    pq,
                    km_base as *const f32,
                    vm_base as *const f32,
                    pat,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_ctx,
                    pos + 1,
                    // Per-layer SWA window / Gemma2 soft-cap for this decode step.
                    window,
                    softcap,
                )
                .is_ok()
            };
        // 7. o projection (device-I/O); 8. fused residual + post-attn RMSNorm.
        chain_ok = chain_ok
            && unsafe { self.matvec_packed_dev_io(wo.0, wo.1, wo.2, pat, po, d_model, d_q) };
        chain_ok = chain_ok
            && unsafe {
                add_rmsnorm_f32(&self.stream, ph, po, ppn, phn2, 1, d_model, rms_eps).is_ok()
            };
        // 8b. Optional dense SwiGLU FFN, device-resident (ffn_out reuses `po`).
        if let Some((wg, wu, wd, _)) = dense_ffn {
            chain_ok = chain_ok
                && unsafe {
                    self.matvec_packed_dev_io(wg.0, wg.1, wg.2, phn2, p_gate, d_ff, d_model)
                };
            chain_ok = chain_ok
                && unsafe {
                    self.matvec_packed_dev_io(wu.0, wu.1, wu.2, phn2, p_up, d_ff, d_model)
                };
            chain_ok = chain_ok
                && unsafe { silu_mul_f32(&self.stream, p_gate, p_up, p_act, d_ff).is_ok() };
            chain_ok = chain_ok
                && unsafe { self.matvec_packed_dev_io(wd.0, wd.1, wd.2, p_act, po, d_model, d_ff) };
            chain_ok = chain_ok && unsafe { add_f32(&self.stream, ph, po, d_model).is_ok() };
        }
        chain_ok
    }

    /// Drain the cross-layer-resident `hidden` (activation-slab offset 0) to the
    /// host — used when a dense model finishes its device-resident layer run
    /// (loop end) or must fall back to the host path mid-stream (so the stale
    /// host `hidden` is refreshed first). Returns false if the slab isn't present
    /// or the copy failed. `d_model` must match the dims the block ran with.
    pub fn download_resident_hidden_f32(&self, hidden: &mut [f32], d_model: usize) -> bool {
        if hidden.len() != d_model {
            return false;
        }
        let Some(a) = self.attn_act.as_ref() else {
            return false;
        };
        let hb: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(hidden.as_mut_ptr() as *mut u8, d_model * 4) };
        // offset 0 = hidden (see the activation-slab offset tuple).
        a.download_at(0, hb) && consume_error_count() == 0
    }

    /// PREFILL twin of [`Self::download_resident_hidden_f32`] — drain the
    /// cross-layer-resident prefill `hidden` (`[n_new, d_model]`) from the prefill
    /// activation slab (offset 0) to the host. The caller uses it when the
    /// device-resident prefill run ends (before the final norm / LM head) or must
    /// fall back to the host layer body (so the stale host `hidden` is refreshed).
    pub fn download_prefill_resident_hidden_f32(
        &self,
        hidden: &mut [f32],
        n_new: usize,
        d_model: usize,
    ) -> bool {
        if hidden.len() != n_new * d_model || n_new == 0 {
            return false;
        }
        let Some(a) = self.prefill_act.as_ref() else {
            return false;
        };
        let hb: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(hidden.as_mut_ptr() as *mut u8, n_new * d_model * 4)
        };
        a.download_at(0, hb) && consume_error_count() == 0
    }

    /// Seed the on-device F32 KV mirror for `layer_idx` with the host cache's
    /// prefill history `[valid, pos)`, so `decode_attn_block_f32`'s `pos > valid`
    /// gate passes once decode starts at `pos = prompt_len` (the mirror begins
    /// empty). `k_cache`/`v_cache` are the layer's host KV slabs, head-major
    /// `[n_kv_heads, max_ctx, head_dim]` (same layout as the mirror), so each
    /// head's missing rows upload as ONE contiguous copy. No-op once current.
    /// Call once per layer at the prefill→decode boundary, before the block.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn seed_attn_mirror_f32(
        &mut self,
        layer_idx: usize,
        k_cache: &[f32],
        v_cache: &[f32],
        pos: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        n_layers: usize,
    ) -> bool {
        let d_kv = n_kv_heads * head_dim;
        if layer_idx >= n_layers
            || pos > max_ctx
            || k_cache.len() < d_kv * max_ctx
            || v_cache.len() < d_kv * max_ctx
        {
            return false;
        }
        let mctx = (n_layers as u32, d_kv as u32, max_ctx as u32);
        if self.attn_mirror_ctx != mctx {
            self.attn_k_mirror = (0..n_layers).map(|_| None).collect();
            self.attn_v_mirror = (0..n_layers).map(|_| None).collect();
            self.attn_valid_len = vec![0; n_layers];
            self.attn_mirror_ctx = mctx;
        }
        let mir_bytes = d_kv * max_ctx * 4;
        if !Self::ensure_scratch(&mut self.attn_k_mirror[layer_idx], &self.stream, mir_bytes)
            || !Self::ensure_scratch(&mut self.attn_v_mirror[layer_idx], &self.stream, mir_bytes)
        {
            return false;
        }
        let valid = self.attn_valid_len[layer_idx] as usize;
        if valid >= pos {
            return true; // already current
        }
        let kcb: &[u8] = unsafe {
            std::slice::from_raw_parts(
                k_cache.as_ptr() as *const u8,
                std::mem::size_of_val(k_cache),
            )
        };
        let vcb: &[u8] = unsafe {
            std::slice::from_raw_parts(
                v_cache.as_ptr() as *const u8,
                std::mem::size_of_val(v_cache),
            )
        };
        let n = pos - valid; // rows to upload per head
        {
            let km = self.attn_k_mirror[layer_idx].as_mut().unwrap();
            for h in 0..n_kv_heads {
                let src = (h * max_ctx + valid) * head_dim * 4;
                let dst = (h * max_ctx + valid) * head_dim * 4;
                if !km.upload_at(dst, &kcb[src..src + n * head_dim * 4]) {
                    return false;
                }
            }
        }
        {
            let vm = self.attn_v_mirror[layer_idx].as_mut().unwrap();
            for h in 0..n_kv_heads {
                let src = (h * max_ctx + valid) * head_dim * 4;
                let dst = (h * max_ctx + valid) * head_dim * 4;
                if !vm.upload_at(dst, &vcb[src..src + n * head_dim * 4]) {
                    return false;
                }
            }
        }
        self.attn_valid_len[layer_idx] = pos as u32;
        consume_error_count() == 0
    }

    /// Grouped Q4_K routed-expert FFN — the MoE tiered-expert DECODE fast path.
    /// Computes `out = Σ_e weight[e] · (Wdown_e · silu(Wgate_e·x) ⊙ (Wup_e·x))`
    /// for all `n_pick` promoted (device-resident) experts in ONE on-device pass
    /// (two batched kernels), instead of the per-expert dispatch's ~17
    /// host↔device round-trips per layer. `x` is uploaded once + `out`
    /// downloaded once; the silu stays on-device, so no per-expert managed-page
    /// migration. The `*_ptrs` are the experts' live device (managed) weight
    /// pointers — Q4_K packed, `gate`/`up` = `[d_ff, d_model]`, `down` =
    /// `[d_model, d_ff]`. `out` is OVERWRITTEN with the routed-expert sum (the
    /// caller adds any shared-expert contribution afterward). Returns false
    /// (caller runs the per-expert path) on bad shape / scratch / kernel
    /// failure, leaving `out` untouched. NOT bit-exact vs the sequential
    /// accumulate (atomicAdd reorders the weighted sum) — verdict + perf gated
    /// by the caller.
    ///
    /// # Safety
    /// Each pointer in `gate_ptrs`/`up_ptrs`/`down_ptrs` is a live
    /// device-accessible (managed/unified) Q4_K weight buffer on this cache's
    /// device, valid for the call, sized for the stated (M,K).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn moe_ffn_grouped_dev_resident(
        &mut self,
        kind: CudaPackedKind,
        x: &[f32],
        gate_ptrs: &[*const c_void],
        up_ptrs: &[*const c_void],
        down_ptrs: &[*const c_void],
        weights: &[f32],
        out: &mut [f32],
        d_model: usize,
        d_ff: usize,
    ) -> bool {
        // Only the warp-cooperative quants have a grouped FFN kernel.
        if !matches!(
            kind,
            CudaPackedKind::Q4_K | CudaPackedKind::Q8_0 | CudaPackedKind::Q6_K
        ) {
            return false;
        }
        let align = kind.k_alignment();
        let n = gate_ptrs.len();
        if n == 0
            || n != up_ptrs.len()
            || n != down_ptrs.len()
            || n != weights.len()
            || x.len() != d_model
            || out.len() != d_model
            || d_model == 0
            || d_ff == 0
            || d_model % align != 0
            || d_ff % align != 0
        {
            return false;
        }
        if gate_ptrs
            .iter()
            .chain(up_ptrs)
            .chain(down_ptrs)
            .any(|p| p.is_null())
        {
            return false;
        }
        let ptr_bytes = std::mem::size_of::<*const c_void>();
        if !Self::ensure_scratch(&mut self.x_scratch, &self.stream, d_model * 4)
            || !Self::ensure_scratch(&mut self.out_scratch, &self.stream, d_model * 4)
            || !Self::ensure_scratch(&mut self.ff_scratch, &self.stream, n * d_ff * 4)
            || !Self::ensure_scratch(&mut self.ptr_scratch, &self.stream, 3 * n * ptr_bytes)
            || !Self::ensure_scratch(&mut self.wt_scratch, &self.stream, n * 4)
        {
            return false;
        }
        // Upload x.
        let x_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, d_model * 4) };
        if !self.x_scratch.as_mut().unwrap().upload(x_bytes) {
            return false;
        }
        // Upload the three expert device-ptr arrays packed [gate | up | down].
        // Managed pointers are unified-VA, so their host-side value is the same
        // address the kernel dereferences on-device.
        let mut ptrs: Vec<*const c_void> = Vec::with_capacity(3 * n);
        ptrs.extend_from_slice(gate_ptrs);
        ptrs.extend_from_slice(up_ptrs);
        ptrs.extend_from_slice(down_ptrs);
        let ptr_src: &[u8] =
            unsafe { std::slice::from_raw_parts(ptrs.as_ptr() as *const u8, 3 * n * ptr_bytes) };
        if !self.ptr_scratch.as_mut().unwrap().upload(ptr_src) {
            return false;
        }
        // Upload routing weights.
        let wt_src: &[u8] =
            unsafe { std::slice::from_raw_parts(weights.as_ptr() as *const u8, n * 4) };
        if !self.wt_scratch.as_mut().unwrap().upload(wt_src) {
            return false;
        }
        let base = self.ptr_scratch.as_ref().unwrap().ptr as *const *const c_void;
        let gate_dev = base;
        let up_dev = unsafe { base.add(n) };
        let down_dev = unsafe { base.add(2 * n) };
        let wt_dev = self.wt_scratch.as_ref().unwrap().ptr as *const f32;
        let x_ptr = self.x_scratch.as_ref().unwrap().ptr as *const f32;
        let ff_ptr = self.ff_scratch.as_ref().unwrap().ptr as *mut f32;
        let out_ptr = self.out_scratch.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: all buffers are live device scratch on this stream sized above;
        // the entry memsets `out`, runs both batched kernels in stream order
        // (kernel 2 reads what kernel 1 wrote), and synchronizes before return.
        // One C entry per warp-cooperative quant; call directly per arm (each
        // returns c_int → the match unifies without a fn-pointer coercion).
        let s = self.stream.raw();
        let (n_c, dm_c, df_c) = (n as c_int, d_model as c_int, d_ff as c_int);
        let rc = unsafe {
            match kind {
                CudaPackedKind::Q4_K => rsl_cuda_moe_ffn_grouped_q4k(
                    s, gate_dev, up_dev, down_dev, wt_dev, x_ptr, ff_ptr, out_ptr, n_c, dm_c, df_c,
                ),
                CudaPackedKind::Q8_0 => rsl_cuda_moe_ffn_grouped_q8_0(
                    s, gate_dev, up_dev, down_dev, wt_dev, x_ptr, ff_ptr, out_ptr, n_c, dm_c, df_c,
                ),
                CudaPackedKind::Q6_K => rsl_cuda_moe_ffn_grouped_q6_k(
                    s, gate_dev, up_dev, down_dev, wt_dev, x_ptr, ff_ptr, out_ptr, n_c, dm_c, df_c,
                ),
                _ => return false, // guarded at entry; unreachable
            }
        };
        if rc != 0 || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, d_model * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// Fused gate+up matvec (decode): one launch computes `gate_out = gate_w @ x`
    /// and `up_out = up_w @ x`, reusing the parity-validated per-format row_dot
    /// (so bit-exact to two separate [`Self::matvec_packed`] calls). Uploads
    /// both weights (keyed separately), stages `x` + a 2*M output slab, downloads
    /// both halves. Returns false (caller runs the two-matvec fallback) on any
    /// bad shape / budget / unsupported kind / kernel failure. Only the formats
    /// with a fused kernel are handled; other kinds return false.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_gate_up_fused(
        &mut self,
        kind: CudaPackedKind,
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
                CudaPackedKind::Q8_0 => matvec_q8_0_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                CudaPackedKind::Q4_K => matvec_q4_k_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                CudaPackedKind::Q6_K => matvec_q6_k_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                CudaPackedKind::Q5_K => matvec_q5_k_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                CudaPackedKind::Iq4_Nl => matvec_iq4_nl_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                CudaPackedKind::Iq4_Xs => matvec_iq4_xs_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                CudaPackedKind::Iq1_S => matvec_iq1_s_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                CudaPackedKind::Iq1_M => matvec_iq1_m_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                CudaPackedKind::Iq2_Xxs => matvec_iq2_xxs_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                CudaPackedKind::Iq2_Xs => matvec_iq2_xs_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                CudaPackedKind::Iq2_S => matvec_iq2_s_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                CudaPackedKind::Iq3_Xxs => matvec_iq3_xxs_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                CudaPackedKind::Iq3_S => matvec_iq3_s_gate_up_fused(
                    &self.stream,
                    gw_ptr,
                    uw_ptr,
                    x_ptr,
                    gout_ptr,
                    uout_ptr,
                    m,
                    k,
                ),
                CudaPackedKind::Ptq1_0 => matvec_ptq1_0_gate_up_fused(
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
        let prof = phase_prof::enabled();
        let x_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, n * k * 4) };
        let t_h2d = prof.then(std::time::Instant::now);
        if !self.x_scratch.as_mut().unwrap().upload(x_bytes) {
            return false;
        }
        let h2d_ns = t_h2d.map_or(0, |t| t.elapsed().as_nanos() as u64);
        let w_ptr = self.weights[&weight_key].ptr;
        let x_ptr = self.x_scratch.as_ref().unwrap().ptr as *const f32;
        let out_ptr = self.out_scratch.as_ref().unwrap().ptr as *mut f32;
        let t_ker = prof.then(std::time::Instant::now);
        // SAFETY: as `matvec_packed`, with x/out sized for N rows.
        let res = unsafe {
            match kind {
                CudaPackedKind::Ptq1_0 => {
                    matvec_ptq1_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q8_0 => {
                    matvec_q8_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q4_K => {
                    matvec_q4_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q6_K => {
                    matvec_q6_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q5_K => {
                    matvec_q5_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q2_K => {
                    matvec_q2_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q8_K => {
                    matvec_q8_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q4_0 => {
                    matvec_q4_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q5_0 => {
                    matvec_q5_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q4_1 => {
                    matvec_q4_1_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q5_1 => {
                    matvec_q5_1_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq4_Nl => {
                    matvec_iq4_nl_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq4_Xs => {
                    matvec_iq4_xs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq2_Xxs => {
                    matvec_iq2_xxs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq2_Xs => {
                    matvec_iq2_xs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq2_S => {
                    matvec_iq2_s_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq3_Xxs => {
                    matvec_iq3_xxs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq3_S => {
                    matvec_iq3_s_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq1_S => {
                    matvec_iq1_s_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq1_M => {
                    matvec_iq1_m_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Nvfp4 => {
                    matvec_nvfp4_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Mxfp4 => {
                    matvec_mxfp4_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Mxfp6 => {
                    matvec_mxfp6_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Mxfp8 => {
                    matvec_mxfp8_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q3_K => {
                    matvec_q3_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Pq2_0 => {
                    matvec_pq2_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
            }
        };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
        }
        let ker_ns = t_ker.map_or(0, |t| t.elapsed().as_nanos() as u64);
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * m * 4) };
        let t_d2h = prof.then(std::time::Instant::now);
        let ok = self.out_scratch.as_ref().unwrap().download(out_bytes);
        if prof {
            phase_prof::add_batched(
                h2d_ns,
                ker_ns,
                t_d2h.map_or(0, |t| t.elapsed().as_nanos() as u64),
            );
        }
        ok
    }

    /// Device-resident batched packed matvec — the prefill-path analogue of
    /// [`Self::matvec_packed_dev_resident`] (which see for the rationale). The
    /// weight already lives on the device (`w_ptr_dev`, a promoted expert in
    /// managed memory): no `weight_key` lookup, no host→device weight upload.
    /// `w_nbytes` is the allocation length, checked exactly as the uploading
    /// path does. `x` = [N,K] row-major, `out` = [N,M] row-major. Returns false
    /// (caller falls through) on bad shape / scratch / kernel failure, leaving
    /// `out` untouched. Weight ownership stays with the caller's `Tensor`.
    ///
    /// # Safety
    /// `w_ptr_dev` must be a live device-accessible pointer of at least
    /// `w_nbytes` bytes on this cache's device, valid for the call.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matvec_packed_batched_dev_resident(
        &mut self,
        kind: CudaPackedKind,
        w_ptr_dev: *const c_void,
        w_nbytes: usize,
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        if m == 0 || k == 0 || n == 0 || x.len() != n * k || out.len() != n * m {
            return false;
        }
        if w_ptr_dev.is_null() {
            return false;
        }
        if k % kind.k_alignment() != 0 || w_nbytes < m * kind.row_bytes(k) {
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
        let w_ptr = w_ptr_dev;
        let x_ptr = self.x_scratch.as_ref().unwrap().ptr as *const f32;
        let out_ptr = self.out_scratch.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: w is a caller-guaranteed live device buffer; x/out are live
        // device scratch on `self.stream` sized for N rows; dispatch mirrors
        // `matvec_packed_batched` exactly (same per-format kernels → bit-exact).
        let res = unsafe {
            match kind {
                CudaPackedKind::Ptq1_0 => {
                    matvec_ptq1_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q8_0 => {
                    matvec_q8_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q4_K => {
                    matvec_q4_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q6_K => {
                    matvec_q6_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q5_K => {
                    matvec_q5_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q2_K => {
                    matvec_q2_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q8_K => {
                    matvec_q8_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q4_0 => {
                    matvec_q4_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q5_0 => {
                    matvec_q5_0_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q4_1 => {
                    matvec_q4_1_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q5_1 => {
                    matvec_q5_1_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq4_Nl => {
                    matvec_iq4_nl_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq4_Xs => {
                    matvec_iq4_xs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq2_Xxs => {
                    matvec_iq2_xxs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq2_Xs => {
                    matvec_iq2_xs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq2_S => {
                    matvec_iq2_s_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq3_Xxs => {
                    matvec_iq3_xxs_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq3_S => {
                    matvec_iq3_s_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq1_S => {
                    matvec_iq1_s_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Iq1_M => {
                    matvec_iq1_m_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Nvfp4 => {
                    matvec_nvfp4_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Mxfp4 => {
                    matvec_mxfp4_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Mxfp6 => {
                    matvec_mxfp6_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Mxfp8 => {
                    matvec_mxfp8_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Q3_K => {
                    matvec_q3_k_packed_f32_batched(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n)
                }
                CudaPackedKind::Pq2_0 => {
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

    /// Q4_K prefill GEMM (Phase 1): tiled, shared-mem weight reuse, f32 (bit-
    /// exact). Same device-buffer lifecycle as `matvec_packed_batched` — the
    /// weight is the SAME packed Q4_K bytes — but the kernel stages a dequantized
    /// weight super-block in shared memory and reuses it across the token tile,
    /// so each weight is read from global once (vs once per token). Selected by
    /// accel.rs on the prefill (batched) path when the `gemm:q4_k_f32` verdict
    /// passed.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_packed_batched_q4k_f32(
        &mut self,
        weight_key: usize,
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        let kind = CudaPackedKind::Q4_K;
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
        // SAFETY: device buffers sized for N rows, as `matvec_packed_batched`.
        let res = unsafe { gemm_q4_k_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// Q8_0 f32-reuse prefill GEMM — the LOSSLESS Q8_0 analogue of
    /// [`Self::gemm_packed_batched_q4k_f32`] (same device-buffer lifecycle, same
    /// shared-mem weight-reuse tiled GEMM; only the packed dtype differs). Selected
    /// by accel.rs on the prefill (batched) path when the `gemm:q8_0_f32` verdict
    /// passed.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_packed_batched_q8_0_f32(
        &mut self,
        weight_key: usize,
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        let kind = CudaPackedKind::Q8_0;
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
        // SAFETY: device buffers sized for N rows, as `matvec_packed_batched`.
        let res = unsafe { gemm_q8_0_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// Device-resident Q4_K prefill GEMM — the promoted-expert analogue of
    /// [`Self::gemm_packed_batched_q4k_f32`]. The weight already lives on the
    /// device (`w_ptr_dev`, managed memory): no `weight_key` / no upload. The
    /// SAME bit-exact tiled GEMM (shared-mem weight reuse), so a promoted Q4_K
    /// expert keeps the prefill-GEMM win AND skips the re-upload — no prefill
    /// regression vs the uploading GEMM. Returns false (caller falls through) on
    /// bad shape / scratch / kernel failure; leaves `out` untouched.
    ///
    /// # Safety
    /// `w_ptr_dev` must be a live device-accessible pointer of at least
    /// `w_nbytes` bytes on this cache's device, valid for the call.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_packed_batched_q4k_f32_dev_resident(
        &mut self,
        w_ptr_dev: *const c_void,
        w_nbytes: usize,
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        let kind = CudaPackedKind::Q4_K;
        if m == 0 || k == 0 || n == 0 || x.len() != n * k || out.len() != n * m {
            return false;
        }
        if w_ptr_dev.is_null() {
            return false;
        }
        if k % kind.k_alignment() != 0 || w_nbytes < m * kind.row_bytes(k) {
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
        let w_ptr = w_ptr_dev;
        let x_ptr = self.x_scratch.as_ref().unwrap().ptr as *const f32;
        let out_ptr = self.out_scratch.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: w is a caller-guaranteed live device buffer; x/out are device
        // scratch sized for N rows; same kernel as the uploading GEMM → bit-exact.
        let res = unsafe { gemm_q4_k_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// Device-resident Q8_0 f32-reuse prefill GEMM — the promoted-expert analogue
    /// of [`Self::gemm_packed_batched_q8_0_f32`] and the Q8_0 twin of
    /// [`Self::gemm_packed_batched_q4k_f32_dev_resident`]. The weight already lives
    /// on the device (`w_ptr_dev`, managed memory): no `weight_key` / no upload.
    /// LOSSLESS (bit-exact) and keeps the prefill-GEMM win for a promoted Q8_0
    /// expert. Returns false (caller falls through) on bad shape / scratch / kernel
    /// failure; leaves `out` untouched.
    ///
    /// # Safety
    /// `w_ptr_dev` must be a live device-accessible pointer of at least
    /// `w_nbytes` bytes on this cache's device, valid for the call.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_packed_batched_q8_0_f32_dev_resident(
        &mut self,
        w_ptr_dev: *const c_void,
        w_nbytes: usize,
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        let kind = CudaPackedKind::Q8_0;
        if m == 0 || k == 0 || n == 0 || x.len() != n * k || out.len() != n * m {
            return false;
        }
        if w_ptr_dev.is_null() {
            return false;
        }
        if k % kind.k_alignment() != 0 || w_nbytes < m * kind.row_bytes(k) {
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
        let w_ptr = w_ptr_dev;
        let x_ptr = self.x_scratch.as_ref().unwrap().ptr as *const f32;
        let out_ptr = self.out_scratch.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: w is a caller-guaranteed live device buffer; x/out are device
        // scratch sized for N rows; same kernel as the uploading GEMM → bit-exact.
        let res = unsafe { gemm_q8_0_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// Q4_K int8 tensor-core prefill GEMM (W8A8, lossy). Identical device-buffer
    /// lifecycle to [`gemm_packed_batched_q4k_f32`](Self::gemm_packed_batched_q4k_f32)
    /// — the weight is the SAME packed Q4_K bytes — but the kernel int8-quantizes
    /// the activations and runs the s8 tensor-core MMA. Selected by accel.rs on
    /// the prefill path when the `gemm:q4_k_w8a8_tc` verdict passed.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_packed_batched_q4k_w8a8_tc(
        &mut self,
        weight_key: usize,
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        let kind = CudaPackedKind::Q4_K;
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
        // SAFETY: device buffers sized for N rows, as `gemm_packed_batched_q4k_f32`.
        let res = unsafe { gemm_q4_k_w8a8_tc(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// T3c — batched Q8_0 prefill GEMM via the int8 tensor-core kernel (verdict
    /// `gemm:q8_0_w8a8_tc`, W8A8/lossy). Weights are the SAME packed Q8_0 bytes;
    /// the kernel int8-quantizes activations internally. Mirrors
    /// [`Self::gemm_packed_batched_q4k_w8a8_tc`].
    pub fn gemm_packed_batched_q8_0_w8a8_tc(
        &mut self,
        weight_key: usize,
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        let kind = CudaPackedKind::Q8_0;
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
        let prof = phase_prof::enabled();
        let x_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, n * k * 4) };
        let t_h2d = prof.then(std::time::Instant::now);
        if !self.x_scratch.as_mut().unwrap().upload(x_bytes) {
            return false;
        }
        let h2d_ns = t_h2d.map_or(0, |t| t.elapsed().as_nanos() as u64);
        let w_ptr = self.weights[&weight_key].ptr;
        let x_ptr = self.x_scratch.as_ref().unwrap().ptr as *const f32;
        let out_ptr = self.out_scratch.as_ref().unwrap().ptr as *mut f32;
        let t_ker = prof.then(std::time::Instant::now);
        // SAFETY: device buffers sized for N rows, as `gemm_packed_batched_q4k_w8a8_tc`.
        let res = unsafe { gemm_q8_0_w8a8_tc(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        if prof {
            unsafe { rsl_cuda_stream_sync(self.stream.raw()) };
        }
        let ker_ns = t_ker.map_or(0, |t| t.elapsed().as_nanos() as u64);
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * m * 4) };
        let t_d2h = prof.then(std::time::Instant::now);
        let ok = self.out_scratch.as_ref().unwrap().download(out_bytes);
        if prof {
            phase_prof::add_batched(
                h2d_ns,
                ker_ns,
                t_d2h.map_or(0, |t| t.elapsed().as_nanos() as u64),
            );
        }
        ok
    }

    /// Device-resident Q4_K int8 tensor-core prefill GEMM (W8A8, lossy) — the
    /// promoted-expert analogue of [`Self::gemm_packed_batched_q4k_w8a8_tc`].
    /// Weight already on the device (`w_ptr_dev`, managed memory): no
    /// `weight_key` / no upload. Same lossy W8A8 TC kernel, so a promoted Q4_K
    /// expert gets the faster prefill path (where its verdict/perf gate passed)
    /// without re-upload. Returns false (caller falls through to the f32 GEMM /
    /// matvec) on any miss.
    ///
    /// # Safety
    /// `w_ptr_dev` must be a live device-accessible pointer of at least
    /// `w_nbytes` bytes on this cache's device, valid for the call.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_packed_batched_q4k_w8a8_tc_dev_resident(
        &mut self,
        w_ptr_dev: *const c_void,
        w_nbytes: usize,
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        let kind = CudaPackedKind::Q4_K;
        if m == 0 || k == 0 || n == 0 || x.len() != n * k || out.len() != n * m {
            return false;
        }
        if w_ptr_dev.is_null() {
            return false;
        }
        if k % kind.k_alignment() != 0 || w_nbytes < m * kind.row_bytes(k) {
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
        let w_ptr = w_ptr_dev;
        let x_ptr = self.x_scratch.as_ref().unwrap().ptr as *const f32;
        let out_ptr = self.out_scratch.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: w caller-guaranteed live device buffer; x/out device scratch
        // sized for N rows; same kernel as the uploading W8A8 TC GEMM.
        let res = unsafe { gemm_q4_k_w8a8_tc(&self.stream, w_ptr, x_ptr, out_ptr, m, k, n) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// W4A8 Q4_K DECODE matvec (single row): the lossy, perf-gated int8-
    /// activation decode twin of [`Self::gemm_packed_batched_q4k_w8a8_tc`].
    /// `x` = [K], `out` = [M]. Returns false (caller runs the bit-exact f32
    /// warp matvec) on bad shape / budget / kernel failure.
    pub fn matvec_q4k_w4a8(
        &mut self,
        weight_key: usize,
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
    ) -> bool {
        let kind = CudaPackedKind::Q4_K;
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
        // SAFETY: device buffers sized (M,K)/K/M on self.stream; wrapper syncs.
        let res = unsafe { matvec_q4_k_w4a8(&self.stream, w_ptr, x_ptr, out_ptr, m, k) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// T3a — W8A8 Q8_0 DECODE matvec (single row): the lossy, perf-gated int8-
    /// activation MMVQ-style decode path for Q8_0 (symmetric — no min term).
    /// `x` = [K], `out` = [M]. Returns false (caller runs the bit-exact f32 warp
    /// matvec) on bad shape / budget / kernel failure. Mirrors
    /// [`Self::matvec_q4k_w4a8`]; `K % 32 == 0`.
    pub fn matvec_q8_0_w8a8(
        &mut self,
        weight_key: usize,
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
    ) -> bool {
        let kind = CudaPackedKind::Q8_0;
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
        // SAFETY: device buffers sized (M,K)/K/M on self.stream; wrapper syncs.
        let res = unsafe { matvec_q8_0_w8a8(&self.stream, w_ptr, x_ptr, out_ptr, m, k) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// T3b — W8A8 Q4_0 DECODE matvec (single row): the lossy, perf-gated int8-
    /// activation MMVQ-style decode path for Q4_0 (symmetric — no min term).
    /// Returns false (caller runs the bit-exact f32 warp matvec) on any miss.
    /// Mirrors [`Self::matvec_q8_0_w8a8`]; `K % 32 == 0`.
    pub fn matvec_q4_0_w8a8(
        &mut self,
        weight_key: usize,
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
    ) -> bool {
        let kind = CudaPackedKind::Q4_0;
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
        // SAFETY: device buffers sized (M,K)/K/M on self.stream; wrapper syncs.
        let res = unsafe { matvec_q4_0_w8a8(&self.stream, w_ptr, x_ptr, out_ptr, m, k) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// T3b — W8A8 Q5_0 DECODE matvec (single row). Symmetric (5th bit from qh).
    /// Mirrors [`Self::matvec_q8_0_w8a8`]; `K % 32 == 0`.
    pub fn matvec_q5_0_w8a8(
        &mut self,
        weight_key: usize,
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
    ) -> bool {
        let kind = CudaPackedKind::Q5_0;
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
        // SAFETY: device buffers sized (M,K)/K/M on self.stream; wrapper syncs.
        let res = unsafe { matvec_q5_0_w8a8(&self.stream, w_ptr, x_ptr, out_ptr, m, k) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// Device-resident W4A8 Q4_K decode matvec — the promoted-expert analogue of
    /// [`Self::matvec_q4k_w4a8`]. Weight already on the device (`w_ptr_dev`): no
    /// `weight_key` / no upload. Same lossy int8-activation kernel, so a promoted
    /// Q4_K expert gets the faster decode path (where its verdict/perf gate
    /// passed) without re-upload. Returns false (caller runs the bit-exact f32
    /// dev-resident matvec) on any miss.
    ///
    /// # Safety
    /// `w_ptr_dev` must be a live device-accessible pointer of at least
    /// `w_nbytes` bytes on this cache's device, valid for the call.
    pub unsafe fn matvec_q4k_w4a8_dev_resident(
        &mut self,
        w_ptr_dev: *const c_void,
        w_nbytes: usize,
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
    ) -> bool {
        let kind = CudaPackedKind::Q4_K;
        if m == 0 || k == 0 || x.len() != k || out.len() != m {
            return false;
        }
        if w_ptr_dev.is_null() {
            return false;
        }
        if k % kind.k_alignment() != 0 || w_nbytes < m * kind.row_bytes(k) {
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
        let w_ptr = w_ptr_dev;
        let x_ptr = self.x_scratch.as_ref().unwrap().ptr as *const f32;
        let out_ptr = self.out_scratch.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: w caller-guaranteed live device buffer; x/out device scratch
        // sized K/M; same kernel as the uploading W4A8 matvec.
        let res = unsafe { matvec_q4_k_w4a8(&self.stream, w_ptr, x_ptr, out_ptr, m, k) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, m * 4) };
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

    /// Blackwell sm_120a MXFP8 (W8A8) block-scaled tensor-core batched GEMM — the
    /// 8-bit sibling of [`gemm_fp4_tc`](Self::gemm_fp4_tc), over the sm_120
    /// block-scaled MMA. This is DISTINCT from the Hopper `wgmma`
    /// [`gemm_fp8_wgmma`](Self::gemm_fp8_wgmma): same MXFP8 weight format
    /// (K/32 × 33B: 32 E4M3 + 1 E8M0 scale) but a different ISA, selected by
    /// `blackwell_tc_available()` on the dispatch side. Same device-buffer
    /// management + `out[n*M + m]` layout as the FP4 path; `K % 32 == 0`. Returns
    /// `false` (caller falls back to the scalar path) on bad shape / budget /
    /// kernel failure, leaving `out` untouched. WRITE-BLIND: compile-only, no
    /// Blackwell to validate here (gated behind the on-device kernel verdict).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mxfp8_tc(
        &mut self,
        weight_key: usize,
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        let row_bytes = (k / 32) * 33;
        if m == 0 || k == 0 || n == 0 || x.len() != n * k || out.len() != n * m {
            return false;
        }
        if k % 32 != 0 || w_bytes.len() < m * row_bytes {
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
        // SAFETY: as gemm_fp4_tc — live device buffers on self.stream, synced.
        let res = unsafe { gemm_mxfp8_tc_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, n, k) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// Blackwell sm_120a MXFP6 (W6A6) block-scaled tensor-core batched GEMM —
    /// as [`gemm_mxfp8_tc`](Self::gemm_mxfp8_tc) but W in MXFP6 (K/32 × 25B: 32
    /// E3M2 + 1 E8M0 scale). WRITE-BLIND.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mxfp6_tc(
        &mut self,
        weight_key: usize,
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        let row_bytes = (k / 32) * 25;
        if m == 0 || k == 0 || n == 0 || x.len() != n * k || out.len() != n * m {
            return false;
        }
        if k % 32 != 0 || w_bytes.len() < m * row_bytes {
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
        // SAFETY: as gemm_fp4_tc — live device buffers on self.stream, synced.
        let res = unsafe { gemm_mxfp6_tc_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, n, k) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// Hopper sm_90a FP8 `wgmma` tensor-core batched GEMM (MXFP8 W8A8) — the
    /// Hopper analogue of [`gemm_fp4_tc`](Self::gemm_fp4_tc). W is MXFP8
    /// (K/32 × 33B: 32 E4M3 + 1 E8M0 scale), X f32 (quantized to E4M3 on the
    /// fly); K % 32 == 0 (the `wgmma` k-dim). Same device-buffer management +
    /// `out[n*M + m]` layout, so it is a drop-in for the scalar batched path
    /// when `hopper_tc_available()`. `tma` selects the cp.async.bulk-staged
    /// variant. Returns `false` (caller falls back to the scalar path) on bad
    /// shape / budget / kernel failure, leaving `out` untouched on failure.
    /// WRITE-BLIND: compile-only, no GH200 to validate.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_wgmma(
        &mut self,
        weight_key: usize,
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        tma: bool,
    ) -> bool {
        // MXFP8 geometry: K/32 × 33B (32 E4M3 elems + 1 E8M0 scale).
        let row_bytes = (k / 32) * 33;
        if m == 0 || k == 0 || n == 0 || x.len() != n * k || out.len() != n * m {
            return false;
        }
        if k % 32 != 0 || w_bytes.len() < m * row_bytes {
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
        // SAFETY: w (M×K MXFP8 blocks), x (N·K f32), out (N·M f32) are live
        // device buffers on `self.stream`; the wrapper synchronizes.
        let res =
            unsafe { gemm_mxfp8_wgmma_f32(&self.stream, w_ptr, x_ptr, out_ptr, m, n, k, tma) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    // ── MoE tiered-expert device-resident TC GEMMs ──────────────────────────
    // Promoted-expert twins of the four tensor-core batched GEMMs above: the
    // weight already lives on the device (`w_ptr_dev`, managed memory), so these
    // skip `weight_key`/`ensure_weight` and the host→device re-upload, feeding
    // the resident pointer to the SAME kernel (identical geometry checks). They
    // close the last batched tradeoff — a promoted NVFP4/MXFP4/MXFP8/MXFP6
    // expert on an already-GPU-placed layer keeps its TC GEMM. Each is
    // verdict-gated at the dispatch (fail-closed: if the kernel is write-blind /
    // unvalidated, its verdict is off and this never runs). `w_ptr_dev` must be
    // a live device buffer of `>= w_nbytes` bytes on this cache's device.

    /// Device-resident [`Self::gemm_fp4_tc`] (NVFP4 / MXFP4, W4A4).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_fp4_tc_dev_resident(
        &mut self,
        kind: CudaFp4TcKind,
        w_ptr_dev: *const c_void,
        w_nbytes: usize,
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        let (kalign, row_bytes) = match kind {
            CudaFp4TcKind::Nvfp4 => (16usize, (k / 16) * 9),
            CudaFp4TcKind::Mxfp4 => (32usize, (k / 32) * 17),
        };
        if m == 0 || k == 0 || n == 0 || x.len() != n * k || out.len() != n * m {
            return false;
        }
        if w_ptr_dev.is_null() || k % 64 != 0 || k % kalign != 0 || w_nbytes < m * row_bytes {
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
        let x_ptr = self.x_scratch.as_ref().unwrap().ptr as *const f32;
        let out_ptr = self.out_scratch.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: w caller-guaranteed live device buffer; x/out device scratch
        // sized N·K / N·M; same kernel as `gemm_fp4_tc`.
        let res =
            unsafe { gemm_fp4_tc_f32(kind, &self.stream, w_ptr_dev, x_ptr, out_ptr, m, n, k) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// Device-resident [`Self::gemm_mxfp8_tc`] (Blackwell MXFP8, W8A8).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_mxfp8_tc_dev_resident(
        &mut self,
        w_ptr_dev: *const c_void,
        w_nbytes: usize,
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        let row_bytes = (k / 32) * 33;
        if m == 0 || k == 0 || n == 0 || x.len() != n * k || out.len() != n * m {
            return false;
        }
        if w_ptr_dev.is_null() || k % 32 != 0 || w_nbytes < m * row_bytes {
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
        let x_ptr = self.x_scratch.as_ref().unwrap().ptr as *const f32;
        let out_ptr = self.out_scratch.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: as gemm_fp4_tc_dev_resident; same kernel as `gemm_mxfp8_tc`.
        let res = unsafe { gemm_mxfp8_tc_f32(&self.stream, w_ptr_dev, x_ptr, out_ptr, m, n, k) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// Device-resident [`Self::gemm_mxfp6_tc`] (Blackwell MXFP6, W6A6).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_mxfp6_tc_dev_resident(
        &mut self,
        w_ptr_dev: *const c_void,
        w_nbytes: usize,
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        let row_bytes = (k / 32) * 25;
        if m == 0 || k == 0 || n == 0 || x.len() != n * k || out.len() != n * m {
            return false;
        }
        if w_ptr_dev.is_null() || k % 32 != 0 || w_nbytes < m * row_bytes {
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
        let x_ptr = self.x_scratch.as_ref().unwrap().ptr as *const f32;
        let out_ptr = self.out_scratch.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: as gemm_fp4_tc_dev_resident; same kernel as `gemm_mxfp6_tc`.
        let res = unsafe { gemm_mxfp6_tc_f32(&self.stream, w_ptr_dev, x_ptr, out_ptr, m, n, k) };
        if res.is_err() || consume_error_count() != 0 {
            return false;
        }
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * m * 4) };
        self.out_scratch.as_ref().unwrap().download(out_bytes)
    }

    /// Device-resident [`Self::gemm_fp8_wgmma`] (Hopper MXFP8 wgmma, W8A8).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_fp8_wgmma_dev_resident(
        &mut self,
        w_ptr_dev: *const c_void,
        w_nbytes: usize,
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        tma: bool,
    ) -> bool {
        let row_bytes = (k / 32) * 33;
        if m == 0 || k == 0 || n == 0 || x.len() != n * k || out.len() != n * m {
            return false;
        }
        if w_ptr_dev.is_null() || k % 32 != 0 || w_nbytes < m * row_bytes {
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
        let x_ptr = self.x_scratch.as_ref().unwrap().ptr as *const f32;
        let out_ptr = self.out_scratch.as_ref().unwrap().ptr as *mut f32;
        // SAFETY: as gemm_fp4_tc_dev_resident; same kernel as `gemm_fp8_wgmma`.
        let res =
            unsafe { gemm_mxfp8_wgmma_f32(&self.stream, w_ptr_dev, x_ptr, out_ptr, m, n, k, tma) };
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
