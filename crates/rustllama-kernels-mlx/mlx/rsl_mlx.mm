// Apple-Metal / MLX kernels for rustllama's macOS backend.
//
// Objective-C++ host shim: it owns the Metal device/queue + MLX stream and
// exposes the `rsl_mlx_*` C ABI that `src/lib.rs` binds to. It is the peer
// of the CUDA crate's `cuda/rsl_cuda.cu` and the SYCL crate's
// `cpp/rsl_kernels.cpp`.
//
// ============================ STATUS: PHASE 0 ============================
// This file is a SKELETON. Every entry point exists (so the crate links on
// a Mac) but the bodies are inert no-ops guarded by `RSL_MLX_HAVE_METAL`
// (undefined in Phase 0). With Metal OFF:
//   * rsl_mlx_device_count() returns 0  → the backend is inert; the engine
//     runs on CPU / SYCL / CUDA exactly as it does on a non-Apple host.
//   * stream_create / malloc return NULL → MlxMatvecCache::new() yields None.
//   * kernels return -1 (they are never reached while device_count()==0).
// build.rs compiles this with plain clang++ (no Metal SDK required yet), so
// a Mac builds it WITHOUT MLX installed. This is the localized starting
// point for Phase 1.
//
// ============================ PHASE 1 TODO ==============================
// Define RSL_MLX_HAVE_METAL (build.rs: `-DRSL_MLX_HAVE_METAL=1`) and fill in:
//   * Device enumeration via mlx-c (`mlx_default_device` / metadata) or
//     `MTLCopyAllDevices()` — return the real device count + name/mem +
//     MTLDevice.registryID; synthesize the 16-byte uuid for the tuner
//     fingerprint (registryID spread over 8 bytes || a hash of the name).
//   * A stream = { id<MTLDevice>, id<MTLCommandQueue> } (and/or an
//     `mlx_stream`). On Apple UNIFIED memory a "device buffer" is a
//     `id<MTLBuffer>` created `StorageModeShared`, whose `.contents` is
//     host-addressable — so malloc_from_host can `newBufferWithBytesNoCopy:`
//     over the GGUF weight bytes (zero-copy) and memcpy_h2d/d2h degenerate
//     to `std::memcpy` (or a no-op when the pointer already aliases host).
//   * The packed matvecs, GQA flash decode/prefill, and on-the-fly dequant
//     as Metal compute kernels in `rsl_mlx.metal` — dispatched here via a
//     `MTLComputePipelineState` per kernel + `MTLComputeCommandEncoder`.
//     Each must be a BYTE-EXACT port of the CPU reference (validated by
//     `doctor --cuda-parity`'s Metal analogue against the CPU kernels),
//     the same discipline the SYCL/CUDA ports follow.
//   * Or, where an mlx-c op already matches (mlx quantized matmul, SDPA),
//     call it directly instead of a hand-written shader.
//
// Keep this file, `rsl_mlx.h`, `rsl_mlx.def`, and the `extern "C"` block in
// `src/lib.rs` in lock-step.

#include "rsl_mlx.h"

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <new>

// Flip on in Phase 1 (build.rs `-DRSL_MLX_HAVE_METAL=1`) once the Metal
// shader library (`rsl_mlx.metal` → `.metallib`) + mlx-c link are wired.
#ifndef RSL_MLX_HAVE_METAL
#define RSL_MLX_HAVE_METAL 0
#endif

#if RSL_MLX_HAVE_METAL
// TODO(phase1): the real Metal / MLX includes.
//   #import <Metal/Metal.h>
//   #import <Foundation/Foundation.h>
//   #include <mlx/c/mlx.h>   // Apple's official mlx-c C API
#endif

// Per-thread kernel-error latch, mirroring the CUDA/SYCL crates. A
// launch/encode failure bumps it; the Rust wrapper drains it via
// rsl_mlx_consume_error_count() after each call and turns a non-zero count
// into a CPU fallback.
static thread_local int g_rsl_mlx_errors = 0;

extern "C" int rsl_mlx_consume_error_count(void) {
    int n = g_rsl_mlx_errors;
    g_rsl_mlx_errors = 0;
    return n;
}

// ---- Device query -----------------------------------------------------

extern "C" int rsl_mlx_device_count(void) {
#if RSL_MLX_HAVE_METAL
    // TODO(phase1): return the number of usable Metal GPUs. On Apple Silicon
    // this is normally 1 (the unified SoC GPU):
    //   NSArray<id<MTLDevice>> *devs = MTLCopyAllDevices(); // or MTLCreateSystemDefaultDevice()
    //   return (int)devs.count;
    return 0;
#else
    return 0; // inert stub: no Metal device
#endif
}

extern "C" int rsl_mlx_device_info(int idx, char *name, int name_cap,
                                   unsigned long long *total_mem,
                                   unsigned long long *registry_id,
                                   unsigned char *uuid /* 16 bytes or null */) {
    (void)idx;
    if (name && name_cap > 0) name[0] = '\0';
    if (total_mem) *total_mem = 0;
    if (registry_id) *registry_id = 0;
    if (uuid) std::memset(uuid, 0, 16);
#if RSL_MLX_HAVE_METAL
    // TODO(phase1): fill from the id<MTLDevice> at `idx`:
    //   strncpy(name, dev.name.UTF8String, name_cap-1);
    //   *total_mem   = dev.recommendedMaxWorkingSetSize; // or hasUnifiedMemory pool
    //   *registry_id = dev.registryID;                   // stable IO-registry id
    //   // synthesize a 16-byte uuid for system_fingerprint() parity:
    //   memcpy(uuid, &registryID, 8); /* + name hash in the high 8 bytes */
    //   return 0;
#endif
    return -1; // inert: no device to describe
}

// ---- Host-pointer reference kernels (parity-pattern establishers) -----

extern "C" int rsl_mlx_rmsnorm_f32(const float *x, const float *w, float *y,
                                   int n_rows, int d, float eps) {
    (void)x; (void)w; (void)y; (void)n_rows; (void)d; (void)eps;
    // TODO(phase1): y = x * rsqrt(mean(x^2)+eps) * w, one threadgroup/row.
    return -1;
}

extern "C" int rsl_mlx_matvec_f32(const float *W, const float *x, float *out,
                                  int m_rows, int k_dim) {
    (void)W; (void)x; (void)out; (void)m_rows; (void)k_dim;
    // TODO(phase1): out[m] = sum_k W[m*k+k]*x[k], one simdgroup/row reduce.
    return -1;
}

// ---- Stream / device-buffer lifecycle ---------------------------------
//
// Phase 0: no real stream object — stream_create returns NULL so the Rust
// MlxStream::create() / MlxMatvecCache::new() yield None and the backend
// stays off. Phase 1 defines `struct rsl_mlx_stream { id<MTLDevice>;
// id<MTLCommandQueue>; /* + mlx_stream */ }` and a shared-storage allocator.

extern "C" rsl_mlx_stream *rsl_mlx_stream_create(int device_index) {
    (void)device_index;
#if RSL_MLX_HAVE_METAL
    // TODO(phase1): allocate + return a rsl_mlx_stream bound to the MTLDevice
    // at `device_index` with a fresh MTLCommandQueue. NULL on failure.
#endif
    return nullptr;
}

extern "C" void rsl_mlx_stream_destroy(rsl_mlx_stream *s) {
    (void)s; // TODO(phase1): release the queue/device (ARC) + free `s`.
}

extern "C" void *rsl_mlx_malloc_from_host(rsl_mlx_stream *s, const void *src,
                                          unsigned long long n_bytes) {
    (void)s; (void)src; (void)n_bytes;
    // TODO(phase1): id<MTLBuffer> buf = [dev newBufferWithLength:n_bytes
    //   options:MTLResourceStorageModeShared]; memcpy(buf.contents, src, n);
    //   return buf.contents (host-addressable on unified memory). For a
    //   read-only weight, prefer newBufferWithBytesNoCopy: over `src` to
    //   avoid the copy entirely.
    return nullptr;
}

extern "C" void *rsl_mlx_malloc_device(rsl_mlx_stream *s,
                                       unsigned long long n_bytes) {
    (void)s; (void)n_bytes;
    // TODO(phase1): as malloc_from_host but uninitialized.
    return nullptr;
}

extern "C" void rsl_mlx_free(rsl_mlx_stream *s, void *dev_ptr) {
    (void)s; (void)dev_ptr;
    // TODO(phase1): release the MTLBuffer that owns `dev_ptr` (tracked in a
    // side map keyed by contents pointer, since we hand Rust raw pointers).
}

extern "C" int rsl_mlx_memcpy_h2d(rsl_mlx_stream *s, void *dst_dev,
                                  const void *src_host,
                                  unsigned long long n_bytes) {
    (void)s;
    // On unified memory this is a plain host memcpy (dst_dev is a shared
    // MTLBuffer's contents pointer). Kept correct even in Phase 0 so the
    // Rust device-buffer helpers behave; unused while device_count()==0.
    if (dst_dev && src_host && n_bytes) std::memcpy(dst_dev, src_host, (size_t)n_bytes);
    return 0;
}

extern "C" int rsl_mlx_memcpy_d2h(rsl_mlx_stream *s, void *dst_host,
                                  const void *src_dev,
                                  unsigned long long n_bytes) {
    (void)s;
    if (dst_host && src_dev && n_bytes) std::memcpy(dst_host, src_dev, (size_t)n_bytes);
    return 0;
}

// ---- Kernels ----------------------------------------------------------
//
// All kernel entry points below are Phase-1 TODO. They return -1 in Phase 0
// (never reached while device_count()==0). Phase 1 replaces each body with a
// Metal compute dispatch (or an mlx-c op) that is a byte-exact port of the
// CPU reference, matched by the parity harness. The `RSL_MLX_STUB_KERNEL`
// macro collapses the boilerplate no-op body for the many signatures.

#define RSL_MLX_STUB_KERNEL { return -1; }

// PTQ1_0 (Bonsai ternary) + Prism Hadamard.
extern "C" int rsl_mlx_matvec_ptq1_0_packed_f32(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K) { (void)s;(void)w;(void)x;(void)out;(void)M;(void)K; RSL_MLX_STUB_KERNEL }
extern "C" int rsl_mlx_matvec_ptq1_0_packed_f32_batched(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K, int N) { (void)s;(void)w;(void)x;(void)out;(void)M;(void)K;(void)N; RSL_MLX_STUB_KERNEL }
extern "C" int rsl_mlx_hadamard_forward(rsl_mlx_stream *s, const float *x,
    const float *signs, float *out, int n_elems, int block) { (void)s;(void)x;(void)signs;(void)out;(void)n_elems;(void)block; RSL_MLX_STUB_KERNEL }

// Packed-quant matvecs (single + batched). TODO(phase1): one Metal kernel
// per quant; the on-device dequant math ports byte-exact from the CPU
// reference (rustllama-kernels-cpu) and the SYCL/CUDA kernels. The
// codebook/grid tables the IQ quants need (IQ1S/IQ2*/IQ3* grids +
// KSIGNS/KMASK) should be staged into `.metal` constants by build.rs, the
// same way the SYCL/CUDA builds emit `iq_grids*.inl`.
#define RSL_MLX_DEFINE_PACKED(NAME)                                           \
    extern "C" int rsl_mlx_##NAME(rsl_mlx_stream *s, const void *w,           \
        const float *x, float *out, int M, int K)                             \
        { (void)s;(void)w;(void)x;(void)out;(void)M;(void)K; RSL_MLX_STUB_KERNEL } \
    extern "C" int rsl_mlx_##NAME##_batched(rsl_mlx_stream *s, const void *w, \
        const float *x, float *out, int M, int K, int N)                      \
        { (void)s;(void)w;(void)x;(void)out;(void)M;(void)K;(void)N; RSL_MLX_STUB_KERNEL }
RSL_MLX_DEFINE_PACKED(matvec_q8_0_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q4_k_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q6_k_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q5_k_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q2_k_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q8_k_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q4_0_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q5_0_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q4_1_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q5_1_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq4_nl_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq4_xs_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq2_xxs_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq2_xs_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq2_s_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq3_xxs_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq3_s_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq1_s_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq1_m_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_nvfp4_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_mxfp4_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_mxfp6_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_mxfp8_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q3_k_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_pq2_0_packed_f32)
#undef RSL_MLX_DEFINE_PACKED

// Forward-pass primitives.
extern "C" int rsl_mlx_add_rmsnorm_f32(rsl_mlx_stream *s, float *hidden,
    const float *branch, const float *w, float *y_norm, int n_rows, int d, float eps)
    { (void)s;(void)hidden;(void)branch;(void)w;(void)y_norm;(void)n_rows;(void)d;(void)eps; RSL_MLX_STUB_KERNEL }
extern "C" int rsl_mlx_rope_f32(rsl_mlx_stream *s, float *qk, int n_heads,
    int head_dim, int pos, const float *inv_freq)
    { (void)s;(void)qk;(void)n_heads;(void)head_dim;(void)pos;(void)inv_freq; RSL_MLX_STUB_KERNEL }
extern "C" int rsl_mlx_silu_mul_f32(rsl_mlx_stream *s, const float *x,
    const float *y, float *out, int n)
    { (void)s;(void)x;(void)y;(void)out;(void)n; RSL_MLX_STUB_KERNEL }
extern "C" int rsl_mlx_embedding_lookup_f32(rsl_mlx_stream *s, const float *table,
    const int *ids, float *out, int n_ids, int d)
    { (void)s;(void)table;(void)ids;(void)out;(void)n_ids;(void)d; RSL_MLX_STUB_KERNEL }

// FlashAttention (F32 K/V). TODO(phase1): GQA online-softmax flash decode
// (one query/head) + causal prefill; port from the CUDA/SYCL flash kernels.
extern "C" int rsl_mlx_flash_attn_decode_f32(rsl_mlx_stream *s, const float *q,
    const float *k, const float *v, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len)
    { (void)s;(void)q;(void)k;(void)v;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len; RSL_MLX_STUB_KERNEL }
extern "C" int rsl_mlx_flash_attn_prefill_f32(rsl_mlx_stream *s, const float *q,
    const float *k, const float *v, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len_base, int n_new)
    { (void)s;(void)q;(void)k;(void)v;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len_base;(void)n_new; RSL_MLX_STUB_KERNEL }

// Quantized-KV FlashAttention. TODO(phase1): dequant K/V on the fly inside
// the flash kernel; byte-exact ports of the CPU reference q4_0/nvfp4/mxfp*/
// turboquant/q8_0 KV kernels.
#define RSL_MLX_DEFINE_FLASH_KV(SUFFIX)                                       \
    extern "C" int rsl_mlx_flash_attn_decode_##SUFFIX(rsl_mlx_stream *s,      \
        const float *q, const void *k_packed, const void *v_packed,          \
        float *out, int n_heads, int n_kv_heads, int head_dim, int max_ctx,  \
        int kv_len)                                                          \
        { (void)s;(void)q;(void)k_packed;(void)v_packed;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len; RSL_MLX_STUB_KERNEL } \
    extern "C" int rsl_mlx_flash_attn_prefill_##SUFFIX(rsl_mlx_stream *s,     \
        const float *q, const void *k_packed, const void *v_packed,          \
        float *out, int n_heads, int n_kv_heads, int head_dim, int max_ctx,  \
        int kv_len_base, int n_new)                                          \
        { (void)s;(void)q;(void)k_packed;(void)v_packed;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len_base;(void)n_new; RSL_MLX_STUB_KERNEL }
RSL_MLX_DEFINE_FLASH_KV(q4_0)
RSL_MLX_DEFINE_FLASH_KV(nvfp4)
RSL_MLX_DEFINE_FLASH_KV(mxfp4)
RSL_MLX_DEFINE_FLASH_KV(mxfp6)
RSL_MLX_DEFINE_FLASH_KV(mxfp8)
#undef RSL_MLX_DEFINE_FLASH_KV

// TurboQuant KV flash carries per-row f32 scales + a `bits` selector, so it
// has its own signature (not the macro above).
extern "C" int rsl_mlx_flash_attn_decode_tq(rsl_mlx_stream *s, const float *q,
    const void *k_packed, const void *v_packed, const float *k_scales,
    const float *v_scales, int bits, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len)
    { (void)s;(void)q;(void)k_packed;(void)v_packed;(void)k_scales;(void)v_scales;(void)bits;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len; RSL_MLX_STUB_KERNEL }
extern "C" int rsl_mlx_flash_attn_prefill_tq(rsl_mlx_stream *s, const float *q,
    const void *k_packed, const void *v_packed, const float *k_scales,
    const float *v_scales, int bits, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len_base, int n_new)
    { (void)s;(void)q;(void)k_packed;(void)v_packed;(void)k_scales;(void)v_scales;(void)bits;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len_base;(void)n_new; RSL_MLX_STUB_KERNEL }

// Q8_0 KV flash: i8 slab + per-row f32 scale (its own signature too).
extern "C" int rsl_mlx_flash_attn_decode_q8_0(rsl_mlx_stream *s, const float *q,
    const void *k_packed, const void *v_packed, const float *k_scales,
    const float *v_scales, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len)
    { (void)s;(void)q;(void)k_packed;(void)v_packed;(void)k_scales;(void)v_scales;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len; RSL_MLX_STUB_KERNEL }
extern "C" int rsl_mlx_flash_attn_prefill_q8_0(rsl_mlx_stream *s, const float *q,
    const void *k_packed, const void *v_packed, const float *k_scales,
    const float *v_scales, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len_base, int n_new)
    { (void)s;(void)q;(void)k_packed;(void)v_packed;(void)k_scales;(void)v_scales;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len_base;(void)n_new; RSL_MLX_STUB_KERNEL }

// Sampling.
extern "C" int rsl_mlx_argmax_f32(rsl_mlx_stream *s, const float *logits,
    int vocab, int *out_idx)
    { (void)s;(void)logits;(void)vocab;(void)out_idx; RSL_MLX_STUB_KERNEL }

#undef RSL_MLX_STUB_KERNEL
