/* C ABI surface for rustllama's native CUDA kernels (see cuda/rsl_cuda.cu).
 * Mirrors the extern-"C" shim style of the SYCL crate's rsl_kernels.h. */
#ifndef RSL_CUDA_H
#define RSL_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

/* Number of visible NVIDIA CUDA devices (0 on failure). */
int rsl_cuda_device_count(void);

/* Fill device `idx`'s name/total-mem/compute-capability + 16-byte UUID
 * (uuid may be null; zeroed when unknown). Returns 0 on success, -1 if
 * the device query fails. */
int rsl_cuda_device_info(int idx, char *name, int name_cap,
                         unsigned long long *total_mem,
                         int *cc_major, int *cc_minor,
                         unsigned char *uuid);

/* y = x * rsqrt(mean(x^2) + eps) * w, per row of length `d`. */
int rsl_cuda_rmsnorm_f32(const float *x, const float *w, float *y,
                         int n_rows, int d, float eps);

/* out[m] = sum_k W[m*k_dim + k] * x[k]  (row-major W, m_rows x k_dim). */
int rsl_cuda_matvec_f32(const float *W, const float *x, float *out,
                        int m_rows, int k_dim);

/* ============================================================
 * Device-resident path (streams + device buffers)
 * ============================================================
 * The CUDA analogue of the SYCL crate's USM/stream model: a stream owns a
 * device + a cudaStream_t; weights live in device memory (uploaded once);
 * the packed-quant matvecs read device pointers directly. Establishes the
 * pattern for the full kernel port (Q4_K/Q6_K/Q8_0/flash-attn follow). */

typedef struct rsl_cuda_stream rsl_cuda_stream;

/* Create a stream bound to device `device_index` (cudaSetDevice +
 * cudaStreamCreate). NULL on failure / no such device. */
rsl_cuda_stream *rsl_cuda_stream_create(int device_index);
void rsl_cuda_stream_destroy(rsl_cuda_stream *s);

/* Allocate `n_bytes` of device memory and copy from host `src` (blocking
 * H2D on the stream). Returns a device pointer, or NULL on failure. Free
 * with rsl_cuda_free. `src` may be NULL to allocate uninitialized. */
void *rsl_cuda_malloc_from_host(rsl_cuda_stream *s, const void *src,
                                unsigned long long n_bytes);
void *rsl_cuda_malloc_device(rsl_cuda_stream *s, unsigned long long n_bytes);
void rsl_cuda_free(rsl_cuda_stream *s, void *dev_ptr);

/* Host<->device copies on the stream (blocking). Return 0 on success. */
int rsl_cuda_memcpy_h2d(rsl_cuda_stream *s, void *dst_dev, const void *src_host,
                        unsigned long long n_bytes);
int rsl_cuda_memcpy_d2h(rsl_cuda_stream *s, void *dst_host, const void *src_dev,
                        unsigned long long n_bytes);

/* PrismML PTQ1_0 (Bonsai ternary) packed matvec. Block layout 28 B / 128
 * weights: { qs:[u8;24], qh:[u8;2], d:f16 }. K must be a multiple of 128.
 * All pointers are DEVICE pointers. `w_bytes` = M*(K/128)*28 bytes.
 * out[m] = sum_blocks d * sum_trits (trit-1)*x.  Returns 0 on success. */
int rsl_cuda_matvec_ptq1_0_packed_f32(rsl_cuda_stream *s, const void *w_bytes_dev,
                                      const float *x_dev, float *out_dev,
                                      int M, int K);

/* Batched PTQ1_0 over N contiguous input rows: x_dev is [N,K], out_dev is
 * [N,M] (out[n,m]). Returns 0 on success. */
int rsl_cuda_matvec_ptq1_0_packed_f32_batched(rsl_cuda_stream *s,
                                              const void *w_bytes_dev,
                                              const float *x_dev, float *out_dev,
                                              int M, int K, int N);

/* Blockwise Prism Hadamard: out = WHT(signs*x)/sqrt(block) per block-sized
 * span. `block` a power of two <= 4096 dividing n_elems. Device pointers.
 * Returns 0 on success. */
int rsl_cuda_hadamard_forward(rsl_cuda_stream *s, const float *x_dev,
                              const float *signs_dev, float *out_dev,
                              int n_elems, int block);

/* K-quant packed matvecs (device pointers; one thread per output row).
 * `w_bytes` is the raw GGUF super-block layout; single-row `x`=[K], `out`=[M];
 * batched `x`=[N,K], `out`=[N,M]. Returns 0 on success.
 *   Q8_0: 34 B / 32 wts, K%32==0.  Q4_K: 144 B / 256, K%256==0.
 *   Q6_K: 210 B / 256, K%256==0. */
int rsl_cuda_matvec_q8_0_packed_f32(rsl_cuda_stream *s, const void *w_bytes,
                                    const float *x, float *out, int M, int K);
int rsl_cuda_matvec_q8_0_packed_f32_batched(rsl_cuda_stream *s, const void *w_bytes,
                                            const float *x, float *out, int M, int K, int N);
int rsl_cuda_matvec_q4_k_packed_f32(rsl_cuda_stream *s, const void *w_bytes,
                                    const float *x, float *out, int M, int K);
int rsl_cuda_matvec_q4_k_packed_f32_batched(rsl_cuda_stream *s, const void *w_bytes,
                                            const float *x, float *out, int M, int K, int N);
int rsl_cuda_matvec_q6_k_packed_f32(rsl_cuda_stream *s, const void *w_bytes,
                                    const float *x, float *out, int M, int K);
int rsl_cuda_matvec_q6_k_packed_f32_batched(rsl_cuda_stream *s, const void *w_bytes,
                                            const float *x, float *out, int M, int K, int N);

/* ============================================================
 * Forward-pass primitives (device-resident, F32). Ports of the SYCL USM
 * kernels; all pointers are DEVICE pointers on the stream's device.
 * Return 0 on success, negative on bad args / launch failure.
 * ============================================================ */

/* hidden[i] += branch[i] (in place); y_norm[i] = rmsnorm(hidden_row)[i]*w[i].
 * Rows of length `d`. Mirrors rsl_add_rmsnorm_usm. */
int rsl_cuda_add_rmsnorm_f32(rsl_cuda_stream *s, float *hidden,
                             const float *branch, const float *w,
                             float *y_norm, int n_rows, int d, float eps);

/* Rotate the (j, j+head_dim/2) pair of each head by pos*inv_freq[j].
 * qk = [n_heads, head_dim]; inv_freq = [head_dim/2]. head_dim even. */
int rsl_cuda_rope_f32(rsl_cuda_stream *s, float *qk, int n_heads, int head_dim,
                      int pos, const float *inv_freq);

/* out[i] = silu(x[i]) * y[i], over n elements. */
int rsl_cuda_silu_mul_f32(rsl_cuda_stream *s, const float *x, const float *y,
                          float *out, int n);

/* out[i,:] = table[ids[i],:] (d elems); ids[i] < 0 -> zero row. `ids` is a
 * DEVICE pointer (the CUDA worker uploads it). */
int rsl_cuda_embedding_lookup_f32(rsl_cuda_stream *s, const float *table,
                                  const int *ids, float *out, int n_ids, int d);

/* FlashAttention decode (single query per head):
 *   q=[n_heads,head_dim], k/v=[n_kv_heads,max_ctx,head_dim],
 *   out=[n_heads,head_dim]. GQA via n_heads%n_kv_heads==0. */
int rsl_cuda_flash_attn_decode_f32(rsl_cuda_stream *s, const float *q,
                                   const float *k, const float *v, float *out,
                                   int n_heads, int n_kv_heads, int head_dim,
                                   int max_ctx, int kv_len);

/* FlashAttention prefill (causal, n_new queries):
 *   q/out=[n_new,n_heads,head_dim], k/v=[n_kv_heads,max_ctx,head_dim].
 *   query q_pos attends absolute [0, kv_len_base+q_pos]. */
int rsl_cuda_flash_attn_prefill_f32(rsl_cuda_stream *s, const float *q,
                                    const float *k, const float *v, float *out,
                                    int n_heads, int n_kv_heads, int head_dim,
                                    int max_ctx, int kv_len_base, int n_new);

/* Greedy argmax over `vocab` F32 logits; writes chosen index to out_idx[0]
 * (lowest index on ties). */
int rsl_cuda_argmax_f32(rsl_cuda_stream *s, const float *logits, int vocab,
                        int *out_idx);

/* Consume + reset this thread's kernel error latch (mirrors the SYCL
 * crate's per-thread error tracking so the Rust wrapper can fall back to
 * CPU on a kernel failure). Returns the count since the last consume. */
int rsl_cuda_consume_error_count(void);

#ifdef __cplusplus
}
#endif

#endif /* RSL_CUDA_H */
