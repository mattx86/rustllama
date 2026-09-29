/* rustllama SYCL kernel ABI.
 *
 * All entry points are extern "C" with simple POD signatures so the FFI from
 * Rust is straightforward. Half precision uses the host-platform half ABI
 * (uint16_t bit pattern); we round-trip through Rust's `half::f16`.
 *
 * Phase 0 declares only the kernels we plan to ship in v1; implementations
 * land in phases 3 and 4. Adding a kernel requires touching three places:
 *   - this header
 *   - cpp/rsl_kernels.cpp
 *   - src/lib.rs (Rust FFI shims)
 */

#ifndef RSL_KERNELS_H
#define RSL_KERNELS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rsl_stream rsl_stream;

/* Init / teardown */
rsl_stream* rsl_stream_create(int device_index);
void        rsl_stream_destroy(rsl_stream* s);
int         rsl_sycl_device_count(void);
/* Returns the numeric `sycl::backend` enum value of the queue's
 * runtime backend. -1 on null input. Common values: 1=opencl,
 * 2=ext_oneapi_level_zero. The L0 import path requires 2. */
int         rsl_stream_backend(rsl_stream* s);

/* SYCL interop accessors — return raw pointers to the queue/device/
 * context the stream owns. Used by rustllama-onednn-sys to construct
 * `dnnl_engine_t` + `dnnl_stream_t` via `dnnl_sycl_interop_*`
 * functions without exposing SYCL types across the FFI boundary.
 *
 * The returned pointers alias fields on `*s` and are valid for as
 * long as `*s` outlives the caller. Callers MUST NOT call free on
 * them. Returns NULL on null input. */
void*       rsl_stream_sycl_queue(rsl_stream* s);
void*       rsl_stream_sycl_device(rsl_stream* s);
void*       rsl_stream_sycl_context(rsl_stream* s);

/* Drains the most recent `zeMemAllocHost` ze_result_t from the L0
 * import path's side-channel and resets it. Returns 0 when nothing
 * has been stored (no failed import since last consume). Useful for
 * surfacing the exact L0 error code (e.g. UNSUPPORTED_FEATURE =
 * 0x78000003) when the public import call returns its category code 4. */
uint32_t    rsl_consume_last_l0_import_code(void);

/* Per-thread error tracking. The kernel TU catches C++ exceptions
 * at every extern "C" boundary so they don't unwind into Rust
 * (which would abort the process). Each caught exception
 * increments a thread-local counter and stores its message; Rust
 * wrappers consume that counter after each FFI call and convert
 * a non-zero count into a `SyclError`, so the engine can fall
 * back to CPU on a kernel failure instead of silently producing
 * garbage output.
 *
 * `rsl_consume_error_count` reads + resets the counter atomically
 * for the calling thread.
 *
 * `rsl_get_last_error_message` writes the most recent caught
 * exception's `what()` into the caller's buffer (NUL-terminated,
 * UTF-8, truncated to fit). It does NOT reset the message —
 * `rsl_consume_error_count == 0` after consumption is the
 * authoritative "no errors since last check" signal.
 */
int  rsl_consume_error_count(void);
void rsl_get_last_error_message(char* buf, int capacity);

/* Per-device descriptive info, used by the autotuner to build a
 * stable device fingerprint that keys cache entries. All string
 * outputs are written as null-terminated UTF-8, truncated to fit
 * the caller's buffer (the buffer must be sized ≥ 1 to hold the
 * terminator). The `*_capacity` arguments are the total buffer
 * length including the terminator slot.
 *
 *   device_index: 0-based SYCL device index, in the same order as
 *                 `rsl_sycl_device_count` reports.
 *   name_out / name_capacity: device name (e.g. "Intel(R) Arc(TM) A770 Graphics").
 *   driver_out / driver_capacity: driver version string.
 *   vendor_id_out: PCI/SPIR vendor ID (e.g. 0x8086 for Intel).
 *   vram_bytes_out: global device memory in bytes.
 *   uuid_out: 16 bytes, or null; zeroed when unknown.
 *   is_integrated_out: 1 = integrated / host-unified-memory GPU (Iris Xe),
 *                      0 = discrete with dedicated VRAM (Arc). Defaults to
 *                      0 on any query error. May be null to skip.
 *
 * Returns 0 on success, -1 on any failure (index out of range,
 * SYCL unavailable, query failed). All output pointers may be
 * null to skip that field. */
int rsl_sycl_device_info(int device_index,
                         char* name_out, int name_capacity,
                         char* driver_out, int driver_capacity,
                         uint32_t* vendor_id_out,
                         uint64_t* vram_bytes_out,
                         uint8_t* uuid_out /* 16 bytes, or null; zeroed when unknown */,
                         uint8_t* is_integrated_out /* 1=integrated, 0=discrete; or null */);

/* USM-shared allocator. Returns memory that is page-mapped on both
 * the CPU and the bound device — on integrated GPUs (shared LPDDR)
 * this is near-zero-cost vs the explicit malloc_device + memcpy
 * pattern; on discrete GPUs the runtime moves pages on demand.
 *
 * Returns NULL on allocation failure or when the stream is NULL.
 * Free with rsl_usm_free. Callers may read/write the returned
 * pointer from CPU code directly (it's plain memory) and pass it
 * into kernels that take USM pointers without an explicit copy. */
void* rsl_usm_alloc_shared(rsl_stream* s, size_t n_bytes);
void  rsl_usm_free(rsl_stream* s, void* ptr);

/* Dedicated-VRAM weight tier: allocate device-local USM (malloc_device)
 * and copy `n_bytes` from the host `src_host` into it (blocking H2D).
 * Unlike rsl_usm_alloc_shared, the returned pointer is NOT host-
 * accessible — on an integrated GPU it lands in the reserved VRAM
 * aperture (Dedicated GPU memory), separate from system RAM. Pass it
 * into the same USM-pointer kernels unchanged. Returns NULL when the
 * device lacks usm_device_allocations, the alloc fails, or the copy
 * throws. Free with rsl_usm_free. */
void* rsl_usm_alloc_device_from_host(rsl_stream* s, const void* src_host, size_t n_bytes);

/* Consume the most-recent USM allocation diagnostic into `dst_buf`.
 * Returns the byte length copied (excluding the NUL); 0 when no
 * diagnostic has been queued since the last consume call. Single-
 * shot so the Rust side logs each failure once. NUL-terminates the
 * buffer when capacity allows. */
int   rsl_consume_last_usm_alloc_diag(char* dst_buf, int capacity);

/* GEMM family. fp16 in, fp16 out. */
void rsl_gemm_f16(rsl_stream* s,
                  const uint16_t* A, const uint16_t* B, uint16_t* C,
                  int M, int N, int K,
                  int lda, int ldb, int ldc);

void rsl_gemm_q4k_f16(rsl_stream* s,
                      const void* A_q4k, const uint16_t* B, uint16_t* C,
                      int M, int N, int K);

void rsl_gemm_q8_0_f16(rsl_stream* s,
                       const void* A_q8_0, const uint16_t* B, uint16_t* C,
                       int M, int N, int K);

/* NVFP4 dequant: unpack `n_blocks` 9-byte NVFP4 blocks (16 elements
 * each) from `w_nvfp4` into f16 output buffer `out`. Each block
 * carries 8 packed E2M1 nibbles + 1 FP8 E4M3 scale byte. The
 * codebook is hardcoded inside the kernel TU; no codebook pointer
 * needs crossing the ABI. */
void rsl_dequant_nvfp4(rsl_stream* s,
                       const void* w_nvfp4,
                       uint16_t* out,
                       int n_blocks);

/* NVFP4 matvec: `out[M] = W[M,K] @ x[K]` with W in NVFP4 packed
 * layout and `x` / `out` in f16. `K` must be a multiple of 16
 * (NVFP4 block size). Memory layout for W matches the CPU
 * `matvec_nvfp4_w_f32_a` family: row-major,
 * `M * (K / 16) * 9` bytes. */
void rsl_matvec_nvfp4_f16(rsl_stream* s,
                          const void* w_nvfp4,
                          const uint16_t* x,
                          uint16_t* out,
                          int M, int K);

/* Elementwise / norm / attention. */
void rsl_rmsnorm(rsl_stream* s,
                 const uint16_t* x, const uint16_t* w, uint16_t* y,
                 int n_rows, int d, float eps);

/* USM-resident variant of rsl_rmsnorm: the x/w/y pointers MUST be
 * USM allocations (e.g. from rsl_usm_alloc_shared) that the bound
 * device can dereference directly. No internal H2D/D2H — the
 * kernel reads/writes the pointers in-place. On integrated GPUs
 * this collapses the per-call copy cost the host-pointer variant
 * pays. Output buffer must be at least n_rows*d uint16_t in size. */
void rsl_rmsnorm_usm(rsl_stream* s,
                     const uint16_t* x_usm, const uint16_t* w_usm,
                     uint16_t* y_usm,
                     int n_rows, int d, float eps);

/* Fused USM-resident RMSNorm + residual add. Computes
 *   y_usm[i] = (x_usm[i] / norm(x_row)) * w_usm[i] + residual_usm[i]
 * in one kernel pass. Caller passes the residual stream (the
 * pre-norm hidden state in a transformer block) and the kernel
 * folds the elementwise add into the RMSNorm write back. Replaces
 * the historical (rmsnorm → kernel barrier → elementwise add) pair
 * at every layer's pre-attention and pre-FFN norm site.
 *
 * All four pointers MUST be USM allocations on the same context.
 * `residual_usm` MAY alias `x_usm` (the kernel reads each value
 * before writing). */
void rsl_rmsnorm_residual_usm(rsl_stream* s,
                              const uint16_t* x_usm,
                              const uint16_t* w_usm,
                              const uint16_t* residual_usm,
                              uint16_t* y_usm,
                              int n_rows, int d, float eps);

/* Fused "add residual + RMSNorm" — the Llama-family pre-norm pattern.
 * Computes
 *   hidden_usm[i] = hidden_usm[i] + branch_usm[i]              (residual)
 *   y_norm_usm[i] = rmsnorm(hidden_usm_row)[i] * w_usm[i]      (pre-norm)
 * in one kernel pass. Replaces the (add_inplace → barrier → rmsnorm)
 * sequence at every post-attention and post-FFN norm site.
 *
 * `hidden_usm` is read AND written; `branch_usm` is read-only;
 * `y_norm_usm` is the next layer's pre-norm input. All four pointers
 * MUST be USM allocations on the same context. */
void rsl_add_rmsnorm_usm(rsl_stream* s,
                         uint16_t* hidden_usm,
                         const uint16_t* branch_usm,
                         const uint16_t* w_usm,
                         uint16_t* y_norm_usm,
                         int n_rows, int d, float eps);

/* USM-resident FlashAttention decode: fuses Q·Kᵀ, softmax, and ·V
 * into one kernel using the online-softmax recurrence. Pointers
 * are USM allocations on the bound device's context.
 *
 * Layout:
 *   q_usm:   [n_heads, head_dim]
 *   k_usm:   [n_kv_heads, max_ctx, head_dim]
 *   v_usm:   [n_kv_heads, max_ctx, head_dim]
 *   out_usm: [n_heads, head_dim]
 *
 * Each work-item handles one head, walking kv_len positions
 * sequentially with the online-softmax state. GQA is supported
 * (n_heads is a multiple of n_kv_heads); each Q head reads from
 * `kv_h = h / (n_heads / n_kv_heads)`. */
void rsl_flash_attn_decode_usm(rsl_stream* s,
                               const uint16_t* q_usm,
                               const uint16_t* k_usm,
                               const uint16_t* v_usm,
                               uint16_t* out_usm,
                               int n_heads, int n_kv_heads,
                               int head_dim, int max_ctx, int kv_len);

/* FA-v2 decode (sub-group cooperation). Same shape/semantics as the
 * v1 entry above, but each attention head is processed by a 16-WI
 * sub-group instead of a single WI; the head_dim is striped across
 * SG lanes. Numerically equivalent (not bit-identical) — Q·K dots
 * are reduced across the SG (parallel partial sums) rather than
 * accumulated serially in one WI.
 *
 * Caller must guarantee `head_dim % 16 == 0` and `head_dim <= 256`;
 * the kernel rejects other shapes by returning early (caller falls
 * back to v1). */
void rsl_flash_attn_decode_v2_usm(rsl_stream* s,
                                  const uint16_t* q_usm,
                                  const uint16_t* k_usm,
                                  const uint16_t* v_usm,
                                  uint16_t* out_usm,
                                  int n_heads, int n_kv_heads,
                                  int head_dim, int max_ctx, int kv_len);

/* FA-v3 decode (SLM K/V tiling + sub-group cooperation). Same shape
 * constraints as v2 (head_dim % 16 == 0, ≤ 256). Reads K/V from
 * SLM-resident tiles loaded cooperatively per outer block of kv
 * positions; reduces USM bandwidth pressure and gives a more
 * predictable memory access pattern than v2's per-WI USM gathers.
 * Numerically equivalent to v2 / v1 / CPU. */
void rsl_flash_attn_decode_v3_usm(rsl_stream* s,
                                  const uint16_t* q_usm,
                                  const uint16_t* k_usm,
                                  const uint16_t* v_usm,
                                  uint16_t* out_usm,
                                  int n_heads, int n_kv_heads,
                                  int head_dim, int max_ctx, int kv_len);

/* USM-resident FlashAttention prefill — F32 variant.
 *
 *   q_usm:       [N, n_heads, head_dim]            row-major
 *   k_cache_usm: [n_kv_heads, max_ctx, head_dim]
 *   v_cache_usm: [n_kv_heads, max_ctx, head_dim]
 *   out_usm:     [N, n_heads, head_dim]            row-major (zeroed by kernel)
 *
 * Caller must already have appended the `n_new` new K/V rows to the
 * cache at positions `[kv_len_base, kv_len_base + n_new)` — the
 * kernel walks `t in [0, kv_len_base + q_pos]` (causal mask, query
 * sees own position). GQA: `n_heads % n_kv_heads == 0`; head `hh`
 * reads from `kv_h = hh / (n_heads / n_kv_heads)`.
 *
 * Parallelization: 2D nd_range `[round_up(n_heads), n_new]` with
 * local `[RSL_LWS, 1]` — one work-item per (head, q_pos) pair, no
 * cross-WI sync (each maintains its own online-softmax state). The
 * inner t-loop's float op order matches the CPU scalar reference so
 * outputs are bit-identical for matching inputs. */
void rsl_flash_attn_prefill_usm(rsl_stream* s,
                                const float* q_usm,
                                const float* k_cache_usm,
                                const float* v_cache_usm,
                                float* out_usm,
                                int n_heads, int n_kv_heads,
                                int head_dim, int max_ctx,
                                int kv_len_base, int n_new);

/* FA-v2 prefill (sub-group cooperation). Same shape/semantics as the
 * v1 entry above, but each (head, q_pos) is processed by a 16-WI
 * sub-group instead of a single WI. Caller must guarantee
 * `head_dim % 16 == 0` and `head_dim <= 256`; the kernel rejects
 * other shapes by returning early. Output is numerically equivalent
 * (not bit-identical) to v1. */
void rsl_flash_attn_prefill_v2_usm(rsl_stream* s,
                                   const float* q_usm,
                                   const float* k_cache_usm,
                                   const float* v_cache_usm,
                                   float* out_usm,
                                   int n_heads, int n_kv_heads,
                                   int head_dim, int max_ctx,
                                   int kv_len_base, int n_new);

/* FA-v3 prefill (SLM K/V tiling + sub-group cooperation). Same shape
 * constraints as v2. Per-q causal mask drives a per-row tile bound
 * so q_pos rows only walk kv positions they're allowed to see. */
void rsl_flash_attn_prefill_v3_usm(rsl_stream* s,
                                   const float* q_usm,
                                   const float* k_cache_usm,
                                   const float* v_cache_usm,
                                   float* out_usm,
                                   int n_heads, int n_kv_heads,
                                   int head_dim, int max_ctx,
                                   int kv_len_base, int n_new);

/* Quantized-KV FlashAttention (F32 Q / F32 out, packed K/V dequantized
 * on the fly). Byte-exact ports of the CPU reference kernels in
 * rustllama-kernels-cpu (q4_0_kv.rs / nvfp4.rs / turboquant.rs): each
 * work-item dequantizes one K (then V) row into a private buffer per kv
 * position, then runs the same online-softmax recurrence as the F32
 * flash entries. Shapes match the F32 entries — q/out F32
 * `[..,n_heads,head_dim]`; k/v packed as `[n_kv_heads, max_ctx,
 * bytes_per_row]`:
 *   Q4_0  : bytes_per_row = (head_dim/32)*18, head_dim % 32 == 0
 *   NVFP4 : bytes_per_row = (head_dim/16)*9,  head_dim % 16 == 0
 *   TQ    : bytes_per_row = ceil(head_dim*bits/8), head_dim a power of
 *           two, bits in {1,2,4,8}; per-row f32 scales in
 *           k_scales_usm/v_scales_usm = `[n_kv_heads*max_ctx]` (indexed
 *           kv_h*max_ctx + t, matching the packed row layout).
 * head_dim is capped at 256; the kernel early-returns on larger /
 * mis-shaped inputs so the caller falls back to CPU. All pointers are
 * USM allocations on the bound device's context. */
void rsl_flash_attn_decode_q4_0_usm(rsl_stream* s,
                                    const float* q_usm,
                                    const void* k_packed_usm,
                                    const void* v_packed_usm,
                                    float* out_usm,
                                    int n_heads, int n_kv_heads,
                                    int head_dim, int max_ctx, int kv_len);
void rsl_flash_attn_prefill_q4_0_usm(rsl_stream* s,
                                     const float* q_usm,
                                     const void* k_packed_usm,
                                     const void* v_packed_usm,
                                     float* out_usm,
                                     int n_heads, int n_kv_heads,
                                     int head_dim, int max_ctx,
                                     int kv_len_base, int n_new);
void rsl_flash_attn_decode_nvfp4_usm(rsl_stream* s,
                                     const float* q_usm,
                                     const void* k_packed_usm,
                                     const void* v_packed_usm,
                                     float* out_usm,
                                     int n_heads, int n_kv_heads,
                                     int head_dim, int max_ctx, int kv_len);
void rsl_flash_attn_prefill_nvfp4_usm(rsl_stream* s,
                                      const float* q_usm,
                                      const void* k_packed_usm,
                                      const void* v_packed_usm,
                                      float* out_usm,
                                      int n_heads, int n_kv_heads,
                                      int head_dim, int max_ctx,
                                      int kv_len_base, int n_new);
/* OCP Microscaling (MX) KV FlashAttention: MXFP4 / MXFP6 / MXFP8. Same
 * shape/semantics as the NVFP4 entries above; the K/V slab is
 * `[n_kv_heads, max_ctx, bytes_per_row]` with 32-element blocks (one
 * trailing E8M0 scale byte each), so `head_dim % 32 == 0` and
 *   MXFP4 : bytes_per_row = (head_dim/32)*17
 *   MXFP6 : bytes_per_row = (head_dim/32)*25
 *   MXFP8 : bytes_per_row = (head_dim/32)*33
 * head_dim is capped at 256; the kernel early-returns on larger /
 * mis-shaped inputs so the caller falls back to CPU. */
void rsl_flash_attn_decode_mxfp4_usm(rsl_stream* s,
                                     const float* q_usm,
                                     const void* k_packed_usm,
                                     const void* v_packed_usm,
                                     float* out_usm,
                                     int n_heads, int n_kv_heads,
                                     int head_dim, int max_ctx, int kv_len);
void rsl_flash_attn_prefill_mxfp4_usm(rsl_stream* s,
                                      const float* q_usm,
                                      const void* k_packed_usm,
                                      const void* v_packed_usm,
                                      float* out_usm,
                                      int n_heads, int n_kv_heads,
                                      int head_dim, int max_ctx,
                                      int kv_len_base, int n_new);
void rsl_flash_attn_decode_mxfp6_usm(rsl_stream* s,
                                     const float* q_usm,
                                     const void* k_packed_usm,
                                     const void* v_packed_usm,
                                     float* out_usm,
                                     int n_heads, int n_kv_heads,
                                     int head_dim, int max_ctx, int kv_len);
void rsl_flash_attn_prefill_mxfp6_usm(rsl_stream* s,
                                      const float* q_usm,
                                      const void* k_packed_usm,
                                      const void* v_packed_usm,
                                      float* out_usm,
                                      int n_heads, int n_kv_heads,
                                      int head_dim, int max_ctx,
                                      int kv_len_base, int n_new);
void rsl_flash_attn_decode_mxfp8_usm(rsl_stream* s,
                                     const float* q_usm,
                                     const void* k_packed_usm,
                                     const void* v_packed_usm,
                                     float* out_usm,
                                     int n_heads, int n_kv_heads,
                                     int head_dim, int max_ctx, int kv_len);
void rsl_flash_attn_prefill_mxfp8_usm(rsl_stream* s,
                                      const float* q_usm,
                                      const void* k_packed_usm,
                                      const void* v_packed_usm,
                                      float* out_usm,
                                      int n_heads, int n_kv_heads,
                                      int head_dim, int max_ctx,
                                      int kv_len_base, int n_new);
void rsl_flash_attn_decode_tq_usm(rsl_stream* s,
                                  const float* q_usm,
                                  const void* k_packed_usm,
                                  const void* v_packed_usm,
                                  const float* k_scales_usm,
                                  const float* v_scales_usm,
                                  int bits,
                                  float* out_usm,
                                  int n_heads, int n_kv_heads,
                                  int head_dim, int max_ctx, int kv_len);
void rsl_flash_attn_prefill_tq_usm(rsl_stream* s,
                                   const float* q_usm,
                                   const void* k_packed_usm,
                                   const void* v_packed_usm,
                                   const float* k_scales_usm,
                                   const float* v_scales_usm,
                                   int bits,
                                   float* out_usm,
                                   int n_heads, int n_kv_heads,
                                   int head_dim, int max_ctx,
                                   int kv_len_base, int n_new);
/* Q8_0-KV FlashAttention: F32 Q / F32 out, i8 K/V slab + per-row f32 scale.
 * Byte-exact port of the CPU gqa_attention_flash_decode_q8_0 / _prefill_q8_0.
 * K/V are PLAIN i8 slabs [n_kv_heads, max_ctx, head_dim] (one byte per
 * element, NOT the 34B/32 GGUF Q8_0 block layout); k_scales/v_scales are
 * per-row absmax f32 [n_kv_heads*max_ctx] (indexed kv_h*max_ctx + t, same
 * shape as the TQ scales). The kernel dots the raw i8 codes and factors the
 * row scale out of the inner loop (no materialized row, no head_dim cap). */
void rsl_flash_attn_decode_q8_0_usm(rsl_stream* s,
                                    const float* q_usm,
                                    const void* k_packed_usm,
                                    const void* v_packed_usm,
                                    const float* k_scales_usm,
                                    const float* v_scales_usm,
                                    float* out_usm,
                                    int n_heads, int n_kv_heads,
                                    int head_dim, int max_ctx, int kv_len);
void rsl_flash_attn_prefill_q8_0_usm(rsl_stream* s,
                                     const float* q_usm,
                                     const void* k_packed_usm,
                                     const void* v_packed_usm,
                                     const float* k_scales_usm,
                                     const float* v_scales_usm,
                                     float* out_usm,
                                     int n_heads, int n_kv_heads,
                                     int head_dim, int max_ctx,
                                     int kv_len_base, int n_new);

/* USM-resident F16 GEMM: `C[M,N] = A[M,K] @ B[K,N]`, row-major.
 * Pointers MUST be USM allocations on the bound device's context.
 * Same algorithm as `rsl_gemm_f16` (naive tiled, conservative
 * 16x16 tile dims) but with no internal H2D / D2H — the kernel
 * reads A and B and writes C in place. Use M=1 for row-vector ×
 * matrix or N=1 for matrix × column-vector (matvec). */
void rsl_gemm_f16_usm(rsl_stream* s,
                      const uint16_t* a_usm,
                      const uint16_t* b_usm,
                      uint16_t* c_usm,
                      int M, int N, int K,
                      int lda, int ldb, int ldc);

/* USM-resident Q8_0 weight × F32 activation matvec.
 *
 *   out[M] = sum_b w_scales[m, b] * sum_d_in_block(w_q[m, b*32 + d] * x[b*32 + d])
 *
 * Weight layout matches the engine's `Dtype::Q8_0Raw` storage:
 *   - `w_q_usm: [M, K]` row-major i8 (K must be a multiple of 32)
 *   - `w_scales_usm: [M, K/32]` row-major f32, one scale per 32-element block
 * Activation and output are plain f32. Use one work-item per
 * output row — each walks K with inline i8 → f32 widen and
 * per-block scale fold-in. */
void rsl_matvec_q8_0_f32_usm(rsl_stream* s,
                             const int8_t* w_q_usm,
                             const float* w_scales_usm,
                             const float* x_usm,
                             float* out_usm,
                             int M, int K);

/* USM-resident Q8_0 weight × F32 activation matvec — packed GGUF
 * layout variant. Consumes the raw on-disk Q8_0 block layout
 * (34 bytes per block: 2-byte f16 scale + 32 i8 weights, repeated
 * `K/32` times per row, `M` rows). This is the byte-for-byte layout
 * the engine already holds in memory after mmap, so no per-load
 * repack is needed — the engine just copies `w_bytes` into a USM
 * allocation once and dispatches matvecs against it.
 *
 *   out[m] = sum_b f16_to_f32(scale_b) * sum_d(qs[b, d] * x[b*32+d])
 *
 * `w_bytes_usm` size = `M * (K/32) * 34` bytes. `K` must be a
 * multiple of 32. One work-item per output row. */
void rsl_matvec_q8_0_packed_f32_usm(rsl_stream* s,
                                    const void* w_bytes_usm,
                                    const float* x_usm,
                                    float* out_usm,
                                    int M, int K, int lws);

/* USM-resident Q4_K_M weight × F32 activation matvec — packed GGUF
 * layout (the v1 default quant for ~all coding-model GGUFs:
 * Qwen2.5-Coder, DeepSeek-Coder, Mistral, Llama-3.x). Consumes the
 * raw on-disk Q4_K_M super-block layout (144 bytes per 256-weight
 * super-block):
 *
 *   bytes [0..2)   d:      f16 super-block scale
 *   bytes [2..4)   dmin:   f16 super-block min
 *   bytes [4..16)  scales[12]: 8 packed 6-bit scales + 8 packed
 *                              6-bit mins encoded across 12 bytes
 *                              (see unpack_q4k_scales in CPU ref)
 *   bytes [16..144) qs[128]: 4-bit packed weights, 32 bytes per
 *                            pair of 32-weight sub-blocks (low/high
 *                            nibble), 4 such pairs per super-block
 *
 * `out[m] = sum over super-blocks of (
 *     sum_{sub in [0..8)} sc_sub * sum_{l in [0..32)} qnibble * x_l
 *     - sum_{sub in [0..8)} mn_sub * sum_{l in [0..32)} x_l )`
 *
 * `w_bytes_usm` size = `M * (K/256) * 144` bytes. `K` must be a
 * multiple of 256. One work-item per output row. */
/* `lws` selects the local work-group size for this dispatch. Pass 0
 * for the hand-picked default (64). Tuned values come from the
 * autotuner cache: `{16, 32, 64, 128, 256}` are the compiled-in
 * candidates; other values fall back to the default. Adding a
 * candidate = add a `case` in `rsl_matvec_q4_k_packed_f32_usm` in
 * cpp/rsl_kernels.cpp AND recompile (adds one SPIR-V variant). */
// PrismML PTQ1_0 (Bonsai ternary, 28 B / 128 weights) packed matvec.
// K must be a multiple of 128.
void rsl_matvec_ptq1_0_packed_f32_usm(rsl_stream* s,
                                      const void* w_bytes_usm,
                                      const float* x_usm,
                                      float* out_usm,
                                      int M, int K,
                                      int lws);

// Batched PTQ1_0 matvec over N contiguous input rows.
void rsl_matvec_ptq1_0_packed_f32_batched_usm(rsl_stream* s,
                                              const void* w_bytes_usm,
                                              const float* x_usm,
                                              float* out_usm,
                                              int M, int K, int N,
                                              int lws);

// Blockwise Prism Hadamard rotation: out = WHT(signs*x)/sqrt(block)
// per block-sized span. block must be a power of two <= 4096 and
// divide n_elems.
void rsl_hadamard_forward_usm(rsl_stream* s,
                              const float* x_usm,
                              const float* signs_usm,
                              float* out_usm,
                              int n_elems, int block);

void rsl_matvec_q4_k_packed_f32_usm(rsl_stream* s,
                                    const void* w_bytes_usm,
                                    const float* x_usm,
                                    float* out_usm,
                                    int M, int K, int lws);

/* USM-resident Q5_K_M weight × F32 activation matvec — packed GGUF
 * layout. Same super-block-of-256 layout as Q4_K_M plus an extra
 * 32-byte `qh` ("q high") section that supplies the 5th bit per
 * weight, giving 176 bytes per super-block:
 *
 *   bytes [0..2)   d:      f16 super-block scale
 *   bytes [2..4)   dmin:   f16 super-block min
 *   bytes [4..16)  scales[12]: 8 packed 6-bit scales + 8 packed
 *                              6-bit mins (same encoding as Q4_K_M)
 *   bytes [16..48) qh[32]: high-bit stream — bit `2*group` of
 *                          qh[l] is the 5th bit of the low-nibble
 *                          weight; bit `2*group+1` is for the
 *                          high-nibble weight
 *   bytes [48..176) qs[128]: 4-bit packed low nibbles
 *
 * Each weight is reconstructed as `qs_nibble | (qh_bit << 4)`, in
 * [0, 31]; dequant is `d * sc_sub * 5bit - dmin * mn_sub`.
 *
 * `w_bytes_usm` size = `M * (K/256) * 176` bytes. `K` must be a
 * multiple of 256. One work-item per output row. */
void rsl_matvec_q5_k_packed_f32_usm(rsl_stream* s,
                                    const void* w_bytes_usm,
                                    const float* x_usm,
                                    float* out_usm,
                                    int M, int K, int lws);

/* USM-resident Q6_K weight × F32 activation matvec — packed GGUF
 * super-block layout. 210 bytes per 256-weight super-block:
 *
 *   bytes [0..128)    ql[128]: 4-bit low nibbles (2 per byte)
 *   bytes [128..192)  qh[64]:  2-bit high bits packed 4 weights
 *                              per byte
 *   bytes [192..208)  scales[16]: per-sub-block i8 scales (8 sub-
 *                              blocks per super-block, 2 halves)
 *   bytes [208..210)  d:        f16 super-block scale
 *
 * Reconstructs each 6-bit weight as
 *   ((ql_nibble | (qh_bits << 4)) - 32) — signed [-32, 31].
 *
 * Mostly used as the LM head's storage in Q4_K_M variant GGUFs
 * (the body of the model is Q4_K but the LM head is Q6_K for
 * output-precision reasons). Without this kernel, those huge LM
 * head matvecs run on CPU, costing ~1/3 of decode time on a 1.5B
 * coding model. */
void rsl_matvec_q6_k_packed_f32_usm(rsl_stream* s,
                                    const void* w_bytes_usm,
                                    const float* x_usm,
                                    float* out_usm,
                                    int M, int K, int lws);

/* F4: USM-resident IQ4_NL weight × F32 activation matvec. 18 bytes
 * per 32-weight block (f16 d, 16-byte nibble qs). Each nibble
 * indexes a 16-entry signed codebook (KVALUES_IQ4XS). Mirrors the
 * CPU scalar reference at `kernels-cpu::matvec_iq4_nl_w_f32_a`. */
void rsl_matvec_iq4_nl_packed_f32_usm(rsl_stream* s,
                                      const void* w_bytes_usm,
                                      const float* x_usm,
                                      float* out_usm,
                                      int M, int K, int lws);

/* GPU offload of IQ1_S codebook search. Inputs:
 *   targets:    [n_chunks * 8] f32 USM (each 8-element chunk independently searched)
 *   delta:      per-batch scalar applied to grid values before search
 *   grid_f32:   [2048 * 8] f32 USM (precomputed IQ1S codebook)
 * Outputs:
 *   out_grid_idx:     [n_chunks] u16 USM (chosen grid index per chunk)
 *   out_signed_score: [n_chunks] f32 USM (target · (grid + delta))
 *   out_norm_sq:      [n_chunks] f32 USM (|grid + delta|²)
 *
 * Topology: one subgroup per chunk; each lane evaluates 128 of the
 * 2048 candidates; subgroup reduction picks the winner. Mirrors the
 * CPU `best_iq1s_grid_for_chunk_scalar` reference; needs GPU hardware
 * validation before defaulting to GPU for production runs. */
void rsl_iq_search_8elt_delta_iq1s(rsl_stream* s,
                                   const float* targets,
                                   float delta,
                                   const float* grid_f32,
                                   uint16_t* out_grid_idx,
                                   float* out_signed_score,
                                   float* out_norm_sq,
                                   int n_chunks);

/* F1 (sampler GPU offload, first piece): USM-resident argmax over
 * a logits buffer. Returns the lowest-index token id with the
 * maximum logit value — bit-identical tie-break to the CPU
 * `argmax_scalar` reference.
 *
 *   logits_usm : [vocab] f32 USM
 *   out_idx_usm: [1] i32 USM (the chosen token id)
 *
 * Topology: one work-group of 16 lanes; each lane sweeps a stride
 * of LWS across the vocab. Subgroup reduction picks the winner.
 * Hardware validation pending. */
void rsl_sampler_argmax_usm(rsl_stream* s,
                            const float* logits_usm,
                            int vocab,
                            int* out_idx_usm);

/* F1 (sampler GPU offload, second piece): fused temperature scale +
 * numerically-stable softmax, in-place. Three passes: max-find with
 * `inv_temp` multiply folded in, exp-and-sum, divide-by-sum. The
 * output is in probability space; bit-identical in algorithm to the
 * CPU `fused_temp_softmax_inplace_scalar` reference (subject to
 * `native::exp` precision differences across vendors).
 *
 *   logits_usm: [vocab] f32 USM (mutated in-place)
 *   vocab:      f32 element count
 *   inv_temp:   reciprocal of caller's sampling temperature
 *
 * Topology: one work-group of 16 lanes sweeping the vocab via
 * stride-LWS. Hardware validation pending. */
void rsl_sampler_temp_softmax_usm(rsl_stream* s,
                                  float* logits_usm,
                                  int vocab,
                                  float inv_temp);

/* F1 (sampler GPU offload, third piece): multinomial draw from a
 * normalized probability distribution. Reads RNG state from
 * `rng_state_usm`, advances it by one SplitMix64 step, writes the
 * updated state back through the same pointer. Linear left-to-right
 * cumsum walk for bit-identical match with the CPU reference under
 * the same seed.
 *
 *   probs_usm    : [vocab] f32 USM (post-softmax, normalized)
 *   rng_state_usm: [1] u64 USM (mutated in-place; carries seed-derived
 *                  state across successive sample calls)
 *   out_idx_usm  : [1] i32 USM (the chosen token id)
 *
 * Hardware validation pending. */
void rsl_sampler_multinomial_usm(rsl_stream* s,
                                 const float* probs_usm,
                                 int vocab,
                                 uint64_t* rng_state_usm,
                                 int* out_idx_usm);

/* F1 (sampler GPU offload, fourth piece): pre-softmax penalty pass.
 * Applies repetition (multiplicative) + frequency + presence (both
 * additive) penalties in-place on `logits_usm`, matching the CPU
 * `apply_all_penalties` semantics. `recent_usm` is the recent-token
 * window; out-of-vocab token ids are skipped.
 *
 *   logits_usm : [vocab] f32 USM (mutated in-place)
 *   recent_usm : [recent_n] u32 USM
 *   repeat:    multiplicative penalty (1.0 = no-op)
 *   frequency: per-occurrence additive penalty
 *   presence:  per-unique-token additive penalty
 *
 * Hardware validation pending. */
void rsl_sampler_penalty_usm(rsl_stream* s,
                             float* logits_usm,
                             int vocab,
                             const uint32_t* recent_usm,
                             int recent_n,
                             float repeat,
                             float frequency,
                             float presence);

/* F1 (sampler GPU offload, fifth piece): top-k mask + renormalize.
 * Mirrors the CPU `apply_top_k`: finds the k-th-largest probability,
 * zeros everything strictly below it, renormalizes the survivors.
 * Tie-break: probabilities exactly equal to the k-th value are kept.
 *
 *   probs_usm: [vocab] f32 USM (mutated in-place)
 *   k:         number of probabilities to keep
 *
 * Constraint: GPU-side k is capped at MAX_TOP_K_GPU = 256. Larger k
 * is uncommon (typical top-k ∈ [1, 100]); the host wrapper falls
 * back to CPU for k > 256. */
void rsl_sampler_top_k_usm(rsl_stream* s,
                           float* probs_usm,
                           int vocab,
                           int k);

/* F1 (sampler GPU offload, sixth piece): top-p (nucleus) mask +
 * renormalize. Caps at the top-MAX_TOP_P_GPU=1024 most-probable
 * tokens; if cum sum at rank 1024 doesn't reach `p`, writes
 * `*needs_fallback_usm = 1` and leaves `probs_usm` UNCHANGED so
 * the host wrapper can run CPU `apply_top_p` instead. On success
 * (`*needs_fallback_usm = 0`), `probs_usm` has been masked +
 * renormalized in-place.
 *
 *   probs_usm:           [vocab] f32 USM (mutated in-place on success)
 *   p:                   nucleus threshold (caller must pre-gate `p ∈ (0, 1)`)
 *   needs_fallback_usm:  [1] i32 USM (0 = success, 1 = CPU fallback)
 *
 * Hardware validation pending. */
void rsl_sampler_top_p_usm(rsl_stream* s,
                           float* probs_usm,
                           int vocab,
                           float p,
                           int* needs_fallback_usm);

/* F4: Batched 8-element grid + sign-mask codebook search. Handles
 * IQ2_XXS (n_grid=256), IQ2_XS (n_grid=512), and IQ2_S (n_grid=1024).
 * Same algorithm as the CPU `search_chunk_8_scalar` reference:
 *   - for each grid entry g ∈ [0, n_grid), compute greedy sign mask
 *     (flip lanes with negative `target·grid`), parity-fix odd
 *     popcount by flipping the smallest-|contrib| lane, then
 *     reverse-lookup `sign_idx = ksigns_rev[mask]` (0xFF = skip)
 *   - track the (idx, sign_idx, signed_score, norm_sq) with max
 *     `score²/norm`
 *
 *   targets:           [n_chunks, 8] f32 USM
 *   grid_f32:          [n_grid, 8] f32 USM
 *   grid_norm_sq_table:[n_grid] f32 USM — precomputed |grid|²
 *   ksigns_rev:        [256] u8 USM — inverse KSIGNS_IQ2XS
 *   out_grid_idx:      [n_chunks] u16 USM
 *   out_sign_idx:      [n_chunks] u8 USM
 *   out_signed_score:  [n_chunks] f32 USM
 *   out_grid_norm_sq:  [n_chunks] f32 USM
 *
 * Topology: one subgroup (LWS=16) per chunk; lanes stride across
 * candidates by LWS. Hardware validation pending. */
void rsl_iq_search_8elt_signed(rsl_stream* s,
                               const float* targets,
                               const float* grid_f32,
                               const float* grid_norm_sq_table,
                               const uint8_t* ksigns_rev,
                               int n_grid,
                               uint16_t* out_grid_idx,
                               uint8_t* out_sign_idx,
                               float* out_signed_score,
                               float* out_grid_norm_sq,
                               int n_chunks);

/* F4: Batched 4-element paired-grid + sign-mask codebook search.
 * Handles IQ3_XXS (n_grid=256) and IQ3_S (n_grid=512). Splits each
 * 8-element chunk into lo (j=0..3) + hi (j=4..7), runs two
 * independent 4-element grid searches, combines via
 * `KMASK_IQ2XS[j]` for lo and `KMASK_IQ2XS[j+4]` for hi, then
 * parity-fixes if odd popcount. Same Cauchy-Schwarz comparison
 * as the IQ2 path.
 *
 *   targets:           [n_chunks, 8] f32 USM
 *   grid_f32:          [n_grid, 4] f32 USM (4-element entries)
 *   grid_norm_sq_table:[n_grid] f32 USM
 *   kmask:             [8] u8 USM — KMASK_IQ2XS
 *   ksigns_rev:        [256] u8 USM — inverse KSIGNS_IQ2XS
 *   out_grid1_idx:     [n_chunks] u16 USM (lo half grid pick)
 *   out_grid2_idx:     [n_chunks] u16 USM (hi half grid pick)
 *   out_sign_idx:      [n_chunks] u8 USM
 *   out_signed_score:  [n_chunks] f32 USM
 *   out_grid_norm_sq:  [n_chunks] f32 USM
 *
 * Hardware validation pending. */
void rsl_iq_search_4elt_paired_signed(rsl_stream* s,
                                       const float* targets,
                                       const float* grid_f32,
                                       const float* grid_norm_sq_table,
                                       const uint8_t* kmask,
                                       const uint8_t* ksigns_rev,
                                       int n_grid,
                                       uint16_t* out_grid1_idx,
                                       uint16_t* out_grid2_idx,
                                       uint8_t* out_sign_idx,
                                       float* out_signed_score,
                                       float* out_grid_norm_sq,
                                       int n_chunks);

/* F4: USM-resident IQ4_XS weight × F32 activation matvec. 136 bytes
 * per 256-weight super-block: f16 d + u16 scales_h + 4 bytes
 * scales_l + 128 bytes of 4-bit nibbles. Eight sub-blocks of 32
 * weights each carry a 6-bit signed scale (low 4 bits in scales_l,
 * high 2 in scales_h, biased -32). Same codebook as IQ4_NL. */
void rsl_matvec_iq4_xs_packed_f32_usm(rsl_stream* s,
                                      const void* w_bytes_usm,
                                      const float* x_usm,
                                      float* out_usm,
                                      int M, int K, int lws);

/* F4 inference: IQ1_S packed matvec. 50 bytes per 256-weight super-
 * block (f16 d + 32-byte qs + 16-byte qh encoding per-sub-block
 * scale + delta sign + 11-bit grid index per chunk). Mirrors
 * `kernels-cpu::matvec_iq1_s_w_f32_a_scalar`. Required for IQ1_S
 * model inference to run on GPU instead of falling back to CPU
 * (the format the user's APEX-nano-quantized model leans on for
 * the expert FFN tensors). */
void rsl_matvec_iq1_s_packed_f32_usm(rsl_stream* s,
                                     const void* w_bytes_usm,
                                     const float* x_usm,
                                     float* out_usm,
                                     int M, int K, int lws);

/* IQ2_XXS single-row packed-USM matvec: 66 bytes per 256-weight
 * super-block (f16 d + 64-byte qs, sub-block scale in aux1's top
 * nibble, 4 × 7-bit sign-table indices in aux1's low 28 bits,
 * 4 × 8-bit grid indices in aux0). Mirrors
 * `kernels-cpu::matvec_iq2_xxs_w_f32_a_scalar` byte-for-byte;
 * the parity gate in `kernels-cpu::tests::iq2_xxs_sycl_arithmetic_
 * matches_cpu_reference` pins both. Required for IQ2_XXS model
 * inference to run on GPU instead of falling back to CPU. */
void rsl_matvec_iq2_xxs_packed_f32_usm(rsl_stream* s,
                                       const void* w_bytes_usm,
                                       const float* x_usm,
                                       float* out_usm,
                                       int M, int K, int lws);

/* IQ1_M single-row packed-USM matvec: 56 bytes per 256-weight
 * super-block (32-byte qs + 16-byte qh + 8-byte scales). f16 d is
 * reassembled from nibbles across all four scale words; each
 * sub-block has dl1/dl2 3-bit scales + per-lane sign-flip via qh
 * bits 0x08/0x80; reuses the IQ1S 2048-entry codebook. Mirrors
 * `kernels-cpu::matvec_iq1_m_w_f32_a_scalar` byte-for-byte. */
void rsl_matvec_iq1_m_packed_f32_usm(rsl_stream* s,
                                     const void* w_bytes_usm,
                                     const float* x_usm,
                                     float* out_usm,
                                     int M, int K, int lws);

/* IQ2_XS single-row packed-USM matvec: 74 bytes per 256-weight
 * super-block (f16 d + 32 × u16 qs words + 8-byte scales). Each
 * qs word packs a 9-bit grid index (into the 512-entry IQ2XS
 * codebook) + 7-bit sign-table index. Each scale byte holds two
 * 4-bit sub-block scales. Mirrors
 * `kernels-cpu::matvec_iq2_xs_w_f32_a_scalar` byte-for-byte. */
void rsl_matvec_iq2_xs_packed_f32_usm(rsl_stream* s,
                                      const void* w_bytes_usm,
                                      const float* x_usm,
                                      float* out_usm,
                                      int M, int K, int lws);

/* IQ2_S single-row packed-USM matvec: 82 bytes per 256-weight
 * super-block (f16 d + 32-byte qs_lo + 32-byte signs + 8-byte qh
 * + 8-byte scales). 10-bit grid index per chunk into the 1024-
 * entry IQ2S codebook; sign mask stored directly per chunk
 * (no KSIGNS lookup). Mirrors
 * `kernels-cpu::matvec_iq2_s_w_f32_a_scalar` byte-for-byte. */
void rsl_matvec_iq2_s_packed_f32_usm(rsl_stream* s,
                                     const void* w_bytes_usm,
                                     const float* x_usm,
                                     float* out_usm,
                                     int M, int K, int lws);

/* IQ3_XXS single-row packed-USM matvec: 98 bytes per 256-weight
 * super-block (f16 d + 64-byte qs_grid + 32-byte qs_sas).
 * `qs_grid` is 64 × u8 indices into the 256-entry IQ3XXS
 * codebook (each entry is u32 = 4 packed u8 grid coords).
 * `qs_sas` packs four 7-bit sign-table indices + 4-bit sub-block
 * scale shift in each u32 word. Each 8-weight chunk uses two
 * grid lookups (low-4 and high-4 weights, signs split via
 * KMASK[0..4] vs KMASK[4..8]). Mirrors
 * `kernels-cpu::matvec_iq3_xxs_w_f32_a_scalar` byte-for-byte. */
void rsl_matvec_iq3_xxs_packed_f32_usm(rsl_stream* s,
                                       const void* w_bytes_usm,
                                       const float* x_usm,
                                       float* out_usm,
                                       int M, int K, int lws);

/* IQ3_S single-row packed-USM matvec: 110 bytes per 256-weight
 * super-block (f16 d + 64-byte qs + 8-byte qh + 32-byte signs +
 * 4-byte scales). 9-bit grid index per chunk (low 8 from qs,
 * high 1 from qh's odd/even bit-pair trick) into the 512-entry
 * IQ3S codebook. Signs stored inline per chunk. Sub-block scales
 * packed as two 4-bit nibbles per scale byte (4 bytes → 8 nibbles).
 * Mirrors `kernels-cpu::matvec_iq3_s_w_f32_a_scalar` byte-for-byte. */
void rsl_matvec_iq3_s_packed_f32_usm(rsl_stream* s,
                                     const void* w_bytes_usm,
                                     const float* x_usm,
                                     float* out_usm,
                                     int M, int K, int lws);

/* F4 follow-up: batched IQ4_NL packed matvec — N input rows in one
 * kernel launch. Same per-cell math as the single-row variant; lifts
 * IQ4_NL prefill off the CPU fallback path. See top of file for
 * batched I/O layout conventions. */
void rsl_matvec_iq4_nl_packed_f32_batched_usm(rsl_stream* s,
                                              const void* w_bytes_usm,
                                              const float* x_usm,
                                              float* out_usm,
                                              int M, int K, int N, int lws);

/* F4 follow-up: batched IQ4_XS packed matvec — N input rows in one
 * kernel launch. Same per-cell math as the single-row variant; lifts
 * IQ4_XS prefill off the CPU fallback path. */
void rsl_matvec_iq4_xs_packed_f32_batched_usm(rsl_stream* s,
                                              const void* w_bytes_usm,
                                              const float* x_usm,
                                              float* out_usm,
                                              int M, int K, int N, int lws);

/* Batched USM packed matvecs — one kernel launch covering N input
 * rows instead of N serial single-row launches. Inputs are contiguous
 * row-major:
 *   x_usm:   [N, K] f32      x[n, k] = x_usm[n*K + k]
 *   out_usm: [N, M] f32      out[n, m] = out_usm[n*M + m]
 *   out[n, m] = sum_k W[m, k] * x[n, k]
 *
 * Used by the engine's prefill path to amortize per-launch overhead
 * across the (typically 32-512) tokens in a prefill batch. For N=1
 * the single-row variants are still preferred since they save the
 * outer N loop's bounds-check overhead in the kernel. */
void rsl_matvec_q8_0_packed_f32_batched_usm(rsl_stream* s,
                                            const void* w_bytes_usm,
                                            const float* x_usm,
                                            float* out_usm,
                                            int M, int K, int N, int lws);

void rsl_matvec_q4_k_packed_f32_batched_usm(rsl_stream* s,
                                            const void* w_bytes_usm,
                                            const float* x_usm,
                                            float* out_usm,
                                            int M, int K, int N, int lws);

void rsl_matvec_q5_k_packed_f32_batched_usm(rsl_stream* s,
                                            const void* w_bytes_usm,
                                            const float* x_usm,
                                            float* out_usm,
                                            int M, int K, int N, int lws);

void rsl_matvec_q6_k_packed_f32_batched_usm(rsl_stream* s,
                                            const void* w_bytes_usm,
                                            const float* x_usm,
                                            float* out_usm,
                                            int M, int K, int N, int lws);

/* ============================================================
 * Level Zero host-memory import
 * ============================================================ */

/* Try to import an existing Win32 file-mapping HANDLE (from
 * `CreateFileMappingW` over a GGUF file) as a device-accessible host
 * memory allocation. On success, *out_dev_ptr is a pointer that USM
 * kernels can read directly — same effect as
 * `sycl::malloc_shared(size, queue)` followed by a memcpy from the
 * mmap, but without the memcpy.
 *
 * Return codes:
 *   0 = success; *out_dev_ptr is set
 *   1 = invalid arguments (null stream, handle, or output)
 *   2 = SYCL queue isn't backed by Level Zero (OpenCL fallback)
 *   3 = L0 loader DLL or zeMemAllocHost symbol not available
 *   4 = zeMemAllocHost call failed at runtime (driver may not support
 *       importing this handle type — caller should fall back to copy)
 *
 * Engine usage: paired with `rsl_release_imported_usm` on tensor
 * eviction / model unload. Imported pointers must be freed before
 * the SYCL queue's context is destroyed. */
int rsl_try_import_win32_handle_as_usm(rsl_stream* s,
                                       void* mapping_handle,
                                       size_t size,
                                       void** out_dev_ptr);

/* Free a pointer obtained from `rsl_try_import_win32_handle_as_usm`.
 * Safe to call with `dev_ptr == NULL` (no-op). Must run on the same
 * SYCL queue's context that produced the import. */
void rsl_release_imported_usm(rsl_stream* s, void* dev_ptr);

/* Diagnostic-only: bare `zeMemAllocHost` (no import descriptor).
 * Same return-code categories as `rsl_try_import_win32_handle_as_usm`.
 * Used by the engine's probe to distinguish "our struct layout is
 * wrong" from "the import descriptor / handle origin is rejected". */
int rsl_try_alloc_host_baseline(rsl_stream* s, size_t size, void** out_dev_ptr);

/* USM-resident half-split RoPE. In-place rotation on
 * `qk_usm: [n_heads, head_dim]`. `inv_freq_usm: [head_dim/2]`
 * is the pre-computed inverse-frequency table; the kernel uses
 * `pos * inv_freq[j]` for the rotation angle. `head_dim` must be
 * even. */
void rsl_rope_usm(rsl_stream* s,
                  uint16_t* qk_usm, int n_heads, int head_dim, int pos,
                  const uint16_t* inv_freq_usm);

/* USM-resident SwiGLU `out[i] = silu(x[i]) * y[i]` where
 * `silu(v) = v / (1 + exp(-v))`. All three buffers must be USM
 * pointers sized `n` elements. */
void rsl_silu_mul_usm(rsl_stream* s,
                      const uint16_t* x_usm, const uint16_t* y_usm,
                      uint16_t* out_usm, int n);

/* USM-resident embedding gather: `out[i, :] = table[ids[i], :]`.
 * `table_usm: [V, d]` and `out_usm: [n_ids, d]` are USM; `ids` may
 * be a host pointer (only `n_ids` ints; we copy them in via
 * `memcpy_h2d` since IDs are picked on the CPU side). Negative ids
 * produce a zero row, matching the host-pointer variant. */
void rsl_embedding_lookup_usm(rsl_stream* s,
                              const uint16_t* table_usm,
                              const int32_t* ids,
                              uint16_t* out_usm,
                              int n_ids, int d);

void rsl_rope(rsl_stream* s,
              uint16_t* qk, int n_heads, int head_dim, int pos,
              const uint16_t* inv_freq);

void rsl_softmax_attn(rsl_stream* s,
                      uint16_t* qk_scores, const uint16_t* mask,
                      int n_heads, int seq, int kv_len, float scale);

void rsl_silu_mul(rsl_stream* s,
                  const uint16_t* x, const uint16_t* y, uint16_t* out, int n);

void rsl_embedding_lookup(rsl_stream* s,
                          const uint16_t* table, const int32_t* ids,
                          uint16_t* out, int n_ids, int d);

/* Sampling (single-token greedy for now; full sampler stays on the CPU). */
void rsl_sample_argmax(rsl_stream* s,
                       const uint16_t* logits, int vocab, int32_t* out_token);

/* ============================================================
 * ABI sync block — declarations for kernels that live in
 * cpp/rsl_kernels.def + the Rust FFI (src/lib.rs) but were
 * historically missing here. Kept together so the header stays
 * a faithful mirror of the .def export table (see the KEEP IN
 * SYNC note at the top of the .def). Signatures below match the
 * `extern "C"` definitions in cpp/rsl_kernels.cpp exactly.
 * ============================================================ */

/* CPU-parity packed matvecs (mirror the CPU scalar references in
 * rustllama-kernels-cpu). `out[m] = W[m, :] · x[:]`, one work-item per
 * output row. Legacy formats use 32-weight blocks; K-quants + PQ2_0 use
 * their native block size. Same `lws` candidate set {16,32,64,128,256}
 * as the other packed matvecs (0 = default). Byte layouts:
 *   Q4_0:  18 B / 32   (f16 d + 16 nibble bytes; weight = d*(nib-8))
 *   Q5_0:  22 B / 32   (f16 d + u32 qh + 16 nibbles; 5-bit signed -16)
 *   Q4_1:  20 B / 32   (f16 d + f16 min + 16 nibbles; d*nib + min)
 *   Q5_1:  24 B / 32   (f16 d + f16 min + u32 qh + 16 nibbles)
 *   Q2_K:  84 B / 256  (16 scale bytes + 64 qs + f16 d + f16 dmin)
 *   Q3_K: 110 B / 256  (32 hmask + 64 qs + 12 scales + f16 d)
 *   Q8_K: 292 B / 256  (f32 d + 256 i8 + 16 i16 bsums; d is F32)
 *   PQ2_0: 34 B / 128  (f16 d + 32 packed 2-bit; weight = d*(code-1))
 *   MXFP4: 17 B / 32   (16 nibble-pair bytes E2M1 + E8M0 scale)
 *   MXFP6: 25 B / 32   (24-byte LE E3M2 6-bit bitstream + E8M0 scale)
 *   MXFP8: 33 B / 32   (32 E4M3 bytes + E8M0 scale)
 * The three MX* formats are OCP Microscaling: weight = scale * decode(code)
 * where scale is a power-of-two E8M0 byte. Mirror dequant_mxfp4/6/8. */
void rsl_matvec_q4_0_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm,
                                    const float* x_usm, float* out_usm,
                                    int M, int K, int lws);
void rsl_matvec_mxfp4_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm,
                                     const float* x_usm, float* out_usm,
                                     int M, int K, int lws);
void rsl_matvec_mxfp6_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm,
                                     const float* x_usm, float* out_usm,
                                     int M, int K, int lws);
void rsl_matvec_mxfp8_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm,
                                     const float* x_usm, float* out_usm,
                                     int M, int K, int lws);
void rsl_matvec_q5_0_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm,
                                    const float* x_usm, float* out_usm,
                                    int M, int K, int lws);
void rsl_matvec_q4_1_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm,
                                    const float* x_usm, float* out_usm,
                                    int M, int K, int lws);
void rsl_matvec_q5_1_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm,
                                    const float* x_usm, float* out_usm,
                                    int M, int K, int lws);
void rsl_matvec_q2_k_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm,
                                    const float* x_usm, float* out_usm,
                                    int M, int K, int lws);
void rsl_matvec_q3_k_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm,
                                    const float* x_usm, float* out_usm,
                                    int M, int K, int lws);
void rsl_matvec_q8_k_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm,
                                    const float* x_usm, float* out_usm,
                                    int M, int K, int lws);
void rsl_matvec_pq2_0_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm,
                                     const float* x_usm, float* out_usm,
                                     int M, int K, int lws);

/* Batched IQ packed matvecs — N contiguous input rows per launch.
 * See the batched I/O layout note earlier in this header. */
void rsl_matvec_iq1_s_packed_f32_batched_usm(rsl_stream* s, const void* w_bytes_usm,
                                             const float* x_usm, float* out_usm,
                                             int M, int K, int N, int lws);
void rsl_matvec_iq2_xxs_packed_f32_batched_usm(rsl_stream* s, const void* w_bytes_usm,
                                               const float* x_usm, float* out_usm,
                                               int M, int K, int N, int lws);
void rsl_matvec_iq1_m_packed_f32_batched_usm(rsl_stream* s, const void* w_bytes_usm,
                                             const float* x_usm, float* out_usm,
                                             int M, int K, int N, int lws);
void rsl_matvec_iq2_xs_packed_f32_batched_usm(rsl_stream* s, const void* w_bytes_usm,
                                              const float* x_usm, float* out_usm,
                                              int M, int K, int N, int lws);
void rsl_matvec_iq2_s_packed_f32_batched_usm(rsl_stream* s, const void* w_bytes_usm,
                                             const float* x_usm, float* out_usm,
                                             int M, int K, int N, int lws);
void rsl_matvec_iq3_xxs_packed_f32_batched_usm(rsl_stream* s, const void* w_bytes_usm,
                                               const float* x_usm, float* out_usm,
                                               int M, int K, int N, int lws);
void rsl_matvec_iq3_s_packed_f32_batched_usm(rsl_stream* s, const void* w_bytes_usm,
                                             const float* x_usm, float* out_usm,
                                             int M, int K, int N, int lws);

/* Fused "add residual + RMSNorm" — F32 variant of rsl_add_rmsnorm_usm.
 *   hidden[i] = hidden[i] + branch[i]; y_norm[i] = rmsnorm(hidden)[i]*w[i] */
void rsl_add_rmsnorm_f32_usm(rsl_stream* s,
                             float* hidden_usm, const float* branch_usm,
                             const float* w_usm, float* y_norm_usm,
                             int n_rows, int d, float eps);

/* IQ1_S codebook search, all-three-deltas variant + imatrix-weighted
 * companion. Each writes the (abs, pos, neg) winner triple per chunk. */
void rsl_iq_search_8elt_delta_iq1s_all3(rsl_stream* s,
                                        const float* targets, float delta,
                                        const float* grid_f32,
                                        uint16_t* out_grid_idx_abs, float* out_signed_score_abs, float* out_norm_sq_abs,
                                        uint16_t* out_grid_idx_pos, float* out_signed_score_pos, float* out_norm_sq_pos,
                                        uint16_t* out_grid_idx_neg, float* out_signed_score_neg, float* out_norm_sq_neg,
                                        int n_chunks);
void rsl_iq_search_8elt_delta_iq1s_all3_w(rsl_stream* s,
                                          const float* targets, const float* weights, float delta,
                                          const float* grid_f32,
                                          uint16_t* out_grid_idx_abs, float* out_signed_score_abs, float* out_norm_sq_abs,
                                          uint16_t* out_grid_idx_pos, float* out_signed_score_pos, float* out_norm_sq_pos,
                                          uint16_t* out_grid_idx_neg, float* out_signed_score_neg, float* out_norm_sq_neg,
                                          int n_chunks);

/* G2: KV-cache Q8_0 quantize-on-store. Strided write into the KV cache
 * layout [n_kv_heads × max_ctx × head_dim] from a contiguous activation
 * buffer [n_new × n_kv_heads × head_dim]. */
void rsl_kv_quantize_q8_0_store_usm(rsl_stream* s,
                                    const float* src_usm,
                                    int8_t* q_dst_usm,
                                    float* scales_dst_usm,
                                    int n_new, int n_kv_heads,
                                    int head_dim, int max_ctx, int kv_len_base);

/* G6: K-quant block encoders (analytical, USM-in/USM-out). `src_usm` is
 * [n_blocks × 256] f32; `dst_usm` receives the packed block bytes. */
void rsl_encode_q6_k_blocks_usm(rsl_stream* s, const float* src_usm, uint8_t* dst_usm, int n_blocks);
void rsl_encode_q3_k_blocks_usm(rsl_stream* s, const float* src_usm, uint8_t* dst_usm, int n_blocks);
void rsl_encode_q4_k_blocks_usm(rsl_stream* s, const float* src_usm, uint8_t* dst_usm, int n_blocks);
void rsl_encode_q5_k_blocks_usm(rsl_stream* s, const float* src_usm, uint8_t* dst_usm, int n_blocks);

/* H4: gate+up FUSED matvec — one launch computes both gate_out[m] and
 * up_out[m] against the shared x_usm row. Per packed-quant dtype. */
void rsl_matvec_q4_k_gate_up_fused_usm(rsl_stream* s, const void* gate_w_bytes_usm, const void* up_w_bytes_usm, const float* x_usm, float* gate_out_usm, float* up_out_usm, int M, int K, int lws);
void rsl_matvec_q8_0_gate_up_fused_usm(rsl_stream* s, const void* gate_w_bytes_usm, const void* up_w_bytes_usm, const float* x_usm, float* gate_out_usm, float* up_out_usm, int M, int K, int lws);
void rsl_matvec_q5_k_gate_up_fused_usm(rsl_stream* s, const void* gate_w_bytes_usm, const void* up_w_bytes_usm, const float* x_usm, float* gate_out_usm, float* up_out_usm, int M, int K, int lws);
void rsl_matvec_q6_k_gate_up_fused_usm(rsl_stream* s, const void* gate_w_bytes_usm, const void* up_w_bytes_usm, const float* x_usm, float* gate_out_usm, float* up_out_usm, int M, int K, int lws);
void rsl_matvec_iq4_nl_gate_up_fused_usm(rsl_stream* s, const void* gate_w_bytes_usm, const void* up_w_bytes_usm, const float* x_usm, float* gate_out_usm, float* up_out_usm, int M, int K, int lws);
void rsl_matvec_iq4_xs_gate_up_fused_usm(rsl_stream* s, const void* gate_w_bytes_usm, const void* up_w_bytes_usm, const float* x_usm, float* gate_out_usm, float* up_out_usm, int M, int K, int lws);
void rsl_matvec_iq1_s_gate_up_fused_usm(rsl_stream* s, const void* gate_w_bytes_usm, const void* up_w_bytes_usm, const float* x_usm, float* gate_out_usm, float* up_out_usm, int M, int K, int lws);
void rsl_matvec_iq2_xxs_gate_up_fused_usm(rsl_stream* s, const void* gate_w_bytes_usm, const void* up_w_bytes_usm, const float* x_usm, float* gate_out_usm, float* up_out_usm, int M, int K, int lws);
void rsl_matvec_iq1_m_gate_up_fused_usm(rsl_stream* s, const void* gate_w_bytes_usm, const void* up_w_bytes_usm, const float* x_usm, float* gate_out_usm, float* up_out_usm, int M, int K, int lws);
void rsl_matvec_iq2_xs_gate_up_fused_usm(rsl_stream* s, const void* gate_w_bytes_usm, const void* up_w_bytes_usm, const float* x_usm, float* gate_out_usm, float* up_out_usm, int M, int K, int lws);
void rsl_matvec_iq2_s_gate_up_fused_usm(rsl_stream* s, const void* gate_w_bytes_usm, const void* up_w_bytes_usm, const float* x_usm, float* gate_out_usm, float* up_out_usm, int M, int K, int lws);
void rsl_matvec_iq3_xxs_gate_up_fused_usm(rsl_stream* s, const void* gate_w_bytes_usm, const void* up_w_bytes_usm, const float* x_usm, float* gate_out_usm, float* up_out_usm, int M, int K, int lws);
void rsl_matvec_iq3_s_gate_up_fused_usm(rsl_stream* s, const void* gate_w_bytes_usm, const void* up_w_bytes_usm, const float* x_usm, float* gate_out_usm, float* up_out_usm, int M, int K, int lws);
void rsl_matvec_ptq1_0_gate_up_fused_usm(rsl_stream* s, const void* gate_w_bytes_usm, const void* up_w_bytes_usm, const float* x_usm, float* gate_out_usm, float* up_out_usm, int M, int K, int lws);

/* H6: fused matvec + residual-add + RMSNorm (post-attn norm site). One
 * workgroup per token; hidden_usm is read+written, y_norm_usm receives
 * the normalized+scaled output. Per packed-quant dtype. */
void rsl_matvec_q4_k_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm, const float* attn_usm, float* hidden_usm, const float* w_norm_usm, float* y_norm_usm, int M, int K, float eps, int lws);
void rsl_matvec_q8_0_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm, const float* attn_usm, float* hidden_usm, const float* w_norm_usm, float* y_norm_usm, int M, int K, float eps, int lws);
void rsl_matvec_q5_k_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm, const float* attn_usm, float* hidden_usm, const float* w_norm_usm, float* y_norm_usm, int M, int K, float eps, int lws);
void rsl_matvec_q6_k_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm, const float* attn_usm, float* hidden_usm, const float* w_norm_usm, float* y_norm_usm, int M, int K, float eps, int lws);
void rsl_matvec_iq4_nl_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm, const float* attn_usm, float* hidden_usm, const float* w_norm_usm, float* y_norm_usm, int M, int K, float eps, int lws);
void rsl_matvec_iq4_xs_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm, const float* attn_usm, float* hidden_usm, const float* w_norm_usm, float* y_norm_usm, int M, int K, float eps, int lws);
void rsl_matvec_iq1_s_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm, const float* attn_usm, float* hidden_usm, const float* w_norm_usm, float* y_norm_usm, int M, int K, float eps, int lws);
void rsl_matvec_iq1_m_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm, const float* attn_usm, float* hidden_usm, const float* w_norm_usm, float* y_norm_usm, int M, int K, float eps, int lws);
void rsl_matvec_iq2_xxs_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm, const float* attn_usm, float* hidden_usm, const float* w_norm_usm, float* y_norm_usm, int M, int K, float eps, int lws);
void rsl_matvec_iq2_xs_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm, const float* attn_usm, float* hidden_usm, const float* w_norm_usm, float* y_norm_usm, int M, int K, float eps, int lws);
void rsl_matvec_iq2_s_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm, const float* attn_usm, float* hidden_usm, const float* w_norm_usm, float* y_norm_usm, int M, int K, float eps, int lws);
void rsl_matvec_iq3_xxs_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm, const float* attn_usm, float* hidden_usm, const float* w_norm_usm, float* y_norm_usm, int M, int K, float eps, int lws);
void rsl_matvec_iq3_s_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm, const float* attn_usm, float* hidden_usm, const float* w_norm_usm, float* y_norm_usm, int M, int K, float eps, int lws);
void rsl_matvec_ptq1_0_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm, const float* attn_usm, float* hidden_usm, const float* w_norm_usm, float* y_norm_usm, int M, int K, float eps, int lws);

/* H8: F16-input mixed-precision matvec. `x_f16` is the activation vector
 * as F16 bits; weights stay packed, accumulator + output stay F32. Gated
 * behind RUSTLLAMA_MIXED_PRECISION_MATVEC at the call site. */
void rsl_matvec_q8_0_f16in_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K, int lws);
void rsl_matvec_q4_k_f16in_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K, int lws);
void rsl_matvec_q5_k_f16in_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K, int lws);
void rsl_matvec_q6_k_f16in_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K, int lws);
void rsl_matvec_iq4_nl_f16in_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K, int lws);
void rsl_matvec_iq4_xs_f16in_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K, int lws);
void rsl_matvec_iq1_s_f16in_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K, int lws);
void rsl_matvec_iq1_m_f16in_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K, int lws);
void rsl_matvec_iq2_xxs_f16in_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K, int lws);
void rsl_matvec_iq2_xs_f16in_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K, int lws);
void rsl_matvec_iq2_s_f16in_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K, int lws);
void rsl_matvec_iq3_xxs_f16in_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K, int lws);
void rsl_matvec_iq3_s_f16in_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K, int lws);
void rsl_matvec_ptq1_0_f16in_packed_f32_usm(rsl_stream* s, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K, int lws);

/* G5: K-quant → F32 dequant kernels (USM-in / USM-out). One work-item
 * per 256-element super-block. `bytes_usm` and `out_usm` are caller-
 * managed USM pointers; the kernel just dispatches and waits.
 * `n_blocks` is the count of 256-element blocks. */
void rsl_dequant_q4_k_to_f32_usm(rsl_stream* s,
                                  const void* bytes_usm,
                                  float* out_usm,
                                  int n_blocks);
void rsl_dequant_q3_k_to_f32_usm(rsl_stream* s,
                                  const void* bytes_usm,
                                  float* out_usm,
                                  int n_blocks);
void rsl_dequant_q5_k_to_f32_usm(rsl_stream* s,
                                  const void* bytes_usm,
                                  float* out_usm,
                                  int n_blocks);
void rsl_dequant_q6_k_to_f32_usm(rsl_stream* s,
                                  const void* bytes_usm,
                                  float* out_usm,
                                  int n_blocks);

#ifdef __cplusplus
}
#endif

#endif /* RSL_KERNELS_H */
