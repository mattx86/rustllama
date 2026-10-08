/* C ABI surface for rustllama's macOS Apple-Metal / MLX kernels
 * (see mlx/rsl_mlx.mm). Mirrors the extern-"C" shim style of the CUDA
 * crate's rsl_cuda.h and the SYCL crate's rsl_kernels.h.
 *
 * DESIGN: Apple Silicon (aarch64-apple-darwin) is the only host with a
 * real Metal GPU + MLX; everywhere else (Windows, Linux, and Intel macOS
 * — which has no MLX Metal GPU path) these symbols come from a no-op stub
 * so the crate links and rsl_mlx_device_count() reports 0 devices → the
 * engine runs on CPU / SYCL / CUDA. See build.rs.
 *
 * KEEP IN SYNC with mlx/rsl_mlx.def, the `extern "C"` block in src/lib.rs,
 * and the entry points in mlx/rsl_mlx.mm. */
#ifndef RSL_MLX_H
#define RSL_MLX_H

#ifdef __cplusplus
extern "C" {
#endif

/* Number of usable Metal GPU devices exposed through MLX (0 on failure /
 * non-Apple-Silicon). On Apple Silicon this is normally 1 (the unified
 * SoC GPU); Mac Pro / eGPU multi-GPU is out of scope for v1. */
int rsl_mlx_device_count(void);

/* Fill device `idx`'s name/total-mem + Metal registryID + 16-byte UUID
 * (uuid/registry_id may be null; zeroed when unknown). Returns 0 on
 * success, -1 if the device query fails.
 *
 * Metal has no CUDA-style 16-byte device UUID; the stable driver-invariant
 * identifier is MTLDevice.registryID (a uint64_t from the IO registry).
 * Phase 1 also synthesizes a 16-byte `uuid` (e.g. registryID || name hash)
 * so rustllama_tuner::system_fingerprint() can dedup this GPU across
 * backends the same way it does SYCL/CUDA UUIDs. */
int rsl_mlx_device_info(int idx, char *name, int name_cap,
                        unsigned long long *total_mem,
                        unsigned long long *registry_id,
                        unsigned char *uuid);

/* y = x * rsqrt(mean(x^2) + eps) * w, per row of length `d`. */
int rsl_mlx_rmsnorm_f32(const float *x, const float *w, float *y,
                        int n_rows, int d, float eps);

/* out[m] = sum_k W[m*k_dim + k] * x[k]  (row-major W, m_rows x k_dim). */
int rsl_mlx_matvec_f32(const float *W, const float *x, float *out,
                       int m_rows, int k_dim);

/* ============================================================
 * Device-resident path (streams + device buffers)
 * ============================================================
 * The MLX analogue of the CUDA crate's stream model: a stream owns a
 * device + a Metal command queue (and/or an MLX stream). Because Apple
 * Silicon is UNIFIED-MEMORY, a "device buffer" is a MTLBuffer with
 * StorageModeShared that is ALSO host-addressable (unlike CUDA device
 * memory) — so on Apple the H2D/D2H copies can degenerate to no-ops /
 * zero-copy wraps in Phase 1. The surface still mirrors CUDA's so the
 * host-orchestrated dispatch (MlxMatvecCache) is identical shape. */

typedef struct rsl_mlx_stream rsl_mlx_stream;

/* Create a stream bound to device `device_index`. NULL on failure. */
rsl_mlx_stream *rsl_mlx_stream_create(int device_index);
void rsl_mlx_stream_destroy(rsl_mlx_stream *s);

/* Allocate `n_bytes` of device memory and copy from host `src`. Returns a
 * device pointer (a MTLBuffer contents pointer on Apple), or NULL on
 * failure. Free with rsl_mlx_free. `src` may be NULL to allocate
 * uninitialized. */
void *rsl_mlx_malloc_from_host(rsl_mlx_stream *s, const void *src,
                               unsigned long long n_bytes);
void *rsl_mlx_malloc_device(rsl_mlx_stream *s, unsigned long long n_bytes);
void rsl_mlx_free(rsl_mlx_stream *s, void *dev_ptr);

/* Host<->device copies on the stream (blocking). Return 0 on success.
 * On unified memory these are memcpy (or a no-op when the pointer already
 * aliases host memory). */
int rsl_mlx_memcpy_h2d(rsl_mlx_stream *s, void *dst_dev, const void *src_host,
                       unsigned long long n_bytes);
int rsl_mlx_memcpy_d2h(rsl_mlx_stream *s, void *dst_host, const void *src_dev,
                       unsigned long long n_bytes);

/* PrismML PTQ1_0 (Bonsai ternary) packed matvec. Block layout 28 B / 128
 * weights: { qs:[u8;24], qh:[u8;2], d:f16 }. K must be a multiple of 128.
 * All pointers are DEVICE pointers. `w_bytes` = M*(K/128)*28 bytes.
 * out[m] = sum_blocks d * sum_trits (trit-1)*x.  Returns 0 on success. */
int rsl_mlx_matvec_ptq1_0_packed_f32(rsl_mlx_stream *s, const void *w_bytes_dev,
                                     const float *x_dev, float *out_dev,
                                     int M, int K);
int rsl_mlx_matvec_ptq1_0_packed_f32_batched(rsl_mlx_stream *s,
                                             const void *w_bytes_dev,
                                             const float *x_dev, float *out_dev,
                                             int M, int K, int N);

/* Blockwise Prism Hadamard: out = WHT(signs*x)/sqrt(block) per block-sized
 * span. `block` a power of two <= 4096 dividing n_elems. Device pointers.
 * Returns 0 on success. */
int rsl_mlx_hadamard_forward(rsl_mlx_stream *s, const float *x_dev,
                             const float *signs_dev, float *out_dev,
                             int n_elems, int block);

/* Packed-quant matvecs (device pointers; byte-exact ports of the CPU
 * reference impls, matched by doctor parity harnesses). Single: `x`=[K],
 * `out`=[M]; batched: `x`=[N,K], `out`=[N,M]. Return 0 on success. K must
 * be a multiple of the format's block width:
 *   Q5_K,Q2_K,Q8_K,Q3_K,IQ4_XS,IQ2_xxs/xs/s,IQ3_xxs/s,IQ1_s/m: 256
 *   Q4_K,Q6_K: 256; Q8_0/Q4_0/Q5_0/Q4_1/Q5_1/IQ4_NL/MXFP4/6/8: 32
 *   NVFP4: 16; PTQ1_0/PQ2_0: 128
 * Block bytes/super-block: Q8_0 34, Q4_K 144, Q6_K 210, Q5_K 176, Q2_K 84,
 *   Q8_K 292, Q4_0 18, Q5_0 22, Q4_1 20, Q5_1 24, IQ4_NL 18, IQ4_XS 136,
 *   IQ2_XXS 66, IQ2_XS 74, IQ2_S 82, IQ3_XXS 98, IQ3_S 110, IQ1_S 50,
 *   IQ1_M 56, NVFP4 9, MXFP4 17, MXFP6 25, MXFP8 33, Q3_K 110, PQ2_0 34. */
#define RSL_MLX_DECL_PACKED(NAME)                                             \
    int rsl_mlx_##NAME(rsl_mlx_stream *s, const void *w_bytes,               \
                       const float *x, float *out, int M, int K);            \
    int rsl_mlx_##NAME##_batched(rsl_mlx_stream *s, const void *w_bytes,     \
                                 const float *x, float *out, int M, int K,   \
                                 int N);
RSL_MLX_DECL_PACKED(matvec_q8_0_packed_f32)
RSL_MLX_DECL_PACKED(matvec_q4_k_packed_f32)
RSL_MLX_DECL_PACKED(matvec_q6_k_packed_f32)
RSL_MLX_DECL_PACKED(matvec_q5_k_packed_f32)
RSL_MLX_DECL_PACKED(matvec_q2_k_packed_f32)
RSL_MLX_DECL_PACKED(matvec_q8_k_packed_f32)
RSL_MLX_DECL_PACKED(matvec_q4_0_packed_f32)
RSL_MLX_DECL_PACKED(matvec_q5_0_packed_f32)
RSL_MLX_DECL_PACKED(matvec_q4_1_packed_f32)
RSL_MLX_DECL_PACKED(matvec_q5_1_packed_f32)
RSL_MLX_DECL_PACKED(matvec_iq4_nl_packed_f32)
RSL_MLX_DECL_PACKED(matvec_iq4_xs_packed_f32)
RSL_MLX_DECL_PACKED(matvec_iq2_xxs_packed_f32)
RSL_MLX_DECL_PACKED(matvec_iq2_xs_packed_f32)
RSL_MLX_DECL_PACKED(matvec_iq2_s_packed_f32)
RSL_MLX_DECL_PACKED(matvec_iq3_xxs_packed_f32)
RSL_MLX_DECL_PACKED(matvec_iq3_s_packed_f32)
RSL_MLX_DECL_PACKED(matvec_iq1_s_packed_f32)
RSL_MLX_DECL_PACKED(matvec_iq1_m_packed_f32)
RSL_MLX_DECL_PACKED(matvec_nvfp4_packed_f32)
RSL_MLX_DECL_PACKED(matvec_mxfp4_packed_f32)
RSL_MLX_DECL_PACKED(matvec_mxfp6_packed_f32)
RSL_MLX_DECL_PACKED(matvec_mxfp8_packed_f32)
RSL_MLX_DECL_PACKED(matvec_q3_k_packed_f32)
RSL_MLX_DECL_PACKED(matvec_pq2_0_packed_f32)
#undef RSL_MLX_DECL_PACKED

/* Fused gate+up matvec (decode): one dispatch computes gate_out + up_out for
 * the FFN. gw/uw = packed gate/up weights [M,K]; gout/uout = [M] device ptrs.
 * Reuses the format's single-matvec dequant (bit-exact). Return 0 on success. */
#define RSL_MLX_DECL_GATE_UP_FUSED(NAME)                                       \
    int rsl_mlx_##NAME##_gate_up_fused(rsl_mlx_stream *s, const void *gw,      \
                                       const void *uw, const float *x,         \
                                       float *gout, float *uout, int M, int K);
RSL_MLX_DECL_GATE_UP_FUSED(matvec_q8_0_packed_f32)
RSL_MLX_DECL_GATE_UP_FUSED(matvec_q4_k_packed_f32)
RSL_MLX_DECL_GATE_UP_FUSED(matvec_q6_k_packed_f32)
RSL_MLX_DECL_GATE_UP_FUSED(matvec_q5_k_packed_f32)
RSL_MLX_DECL_GATE_UP_FUSED(matvec_iq4_nl_packed_f32)
RSL_MLX_DECL_GATE_UP_FUSED(matvec_iq4_xs_packed_f32)
RSL_MLX_DECL_GATE_UP_FUSED(matvec_iq1_s_packed_f32)
RSL_MLX_DECL_GATE_UP_FUSED(matvec_iq1_m_packed_f32)
RSL_MLX_DECL_GATE_UP_FUSED(matvec_iq2_xxs_packed_f32)
RSL_MLX_DECL_GATE_UP_FUSED(matvec_iq2_xs_packed_f32)
RSL_MLX_DECL_GATE_UP_FUSED(matvec_iq2_s_packed_f32)
RSL_MLX_DECL_GATE_UP_FUSED(matvec_iq3_xxs_packed_f32)
RSL_MLX_DECL_GATE_UP_FUSED(matvec_iq3_s_packed_f32)
RSL_MLX_DECL_GATE_UP_FUSED(matvec_ptq1_0_packed_f32)
#undef RSL_MLX_DECL_GATE_UP_FUSED

/* ============================================================
 * Forward-pass primitives (device-resident, F32). Ports of the CUDA/SYCL
 * kernels; all pointers are DEVICE pointers on the stream's device.
 * Return 0 on success, negative on bad args / launch failure.
 * ============================================================ */

/* hidden[i] += branch[i] (in place); y_norm[i] = rmsnorm(hidden_row)[i]*w[i].
 * Rows of length `d`. */
int rsl_mlx_add_rmsnorm_f32(rsl_mlx_stream *s, float *hidden,
                            const float *branch, const float *w,
                            float *y_norm, int n_rows, int d, float eps);

/* Rotate the (j, j+head_dim/2) pair of each head by pos*inv_freq[j].
 * qk = [n_heads, head_dim]; inv_freq = [head_dim/2]. head_dim even. */
int rsl_mlx_rope_f32(rsl_mlx_stream *s, float *qk, int n_heads, int head_dim,
                     int pos, const float *inv_freq);

/* out[i] = silu(x[i]) * y[i], over n elements. */
int rsl_mlx_silu_mul_f32(rsl_mlx_stream *s, const float *x, const float *y,
                         float *out, int n);

/* out[i,:] = table[ids[i],:] (d elems); ids[i] < 0 -> zero row. `ids` is a
 * DEVICE pointer. */
int rsl_mlx_embedding_lookup_f32(rsl_mlx_stream *s, const float *table,
                                 const int *ids, float *out, int n_ids, int d);

/* FlashAttention decode (single query per head):
 *   q=[n_heads,head_dim], k/v=[n_kv_heads,max_ctx,head_dim],
 *   out=[n_heads,head_dim]. GQA via n_heads%n_kv_heads==0. */
int rsl_mlx_flash_attn_decode_f32(rsl_mlx_stream *s, const float *q,
                                  const float *k, const float *v, float *out,
                                  int n_heads, int n_kv_heads, int head_dim,
                                  int max_ctx, int kv_len);

/* FlashAttention prefill (causal, n_new queries):
 *   q/out=[n_new,n_heads,head_dim], k/v=[n_kv_heads,max_ctx,head_dim].
 *   query q_pos attends absolute [0, kv_len_base+q_pos]. */
int rsl_mlx_flash_attn_prefill_f32(rsl_mlx_stream *s, const float *q,
                                   const float *k, const float *v, float *out,
                                   int n_heads, int n_kv_heads, int head_dim,
                                   int max_ctx, int kv_len_base, int n_new);

/* Quantized-KV FlashAttention: F32 Q / F32 out, packed K/V dequantized on
 * the fly (byte-exact ports of the CPU reference kernels). Same shapes as
 * the F32 flash entries; k/v packed as [n_kv_heads, max_ctx, bytes_per_row]:
 *   Q4_0-KV : bytes_per_row = (head_dim/32)*18, head_dim % 32 == 0
 *   NVFP4-KV: bytes_per_row = (head_dim/16)*9,  head_dim % 16 == 0
 *   MXFP4/6/8-KV: (head_dim/32)*{17,25,33}, head_dim % 32 == 0
 *   TQ-KV   : bytes_per_row = ceil(head_dim*bits/8), head_dim a power of
 *             two, bits in {1,2,4,8}; per-row f32 scales in k_scales/
 *             v_scales = [n_kv_heads*max_ctx] (indexed kv_h*max_ctx + t)
 *   Q8_0-KV : PLAIN i8 slab [n_kv_heads, max_ctx, head_dim] + per-row f32
 *             scale (NOT the 34B/32 GGUF block layout)
 * head_dim is capped at 256; the launcher returns -1 on larger / mis-shaped
 * inputs so the caller falls back to the CPU kernel. Return 0 on success. */
int rsl_mlx_flash_attn_decode_q4_0(rsl_mlx_stream *s, const float *q,
                                   const void *k_packed, const void *v_packed,
                                   float *out, int n_heads, int n_kv_heads,
                                   int head_dim, int max_ctx, int kv_len);
int rsl_mlx_flash_attn_prefill_q4_0(rsl_mlx_stream *s, const float *q,
                                    const void *k_packed, const void *v_packed,
                                    float *out, int n_heads, int n_kv_heads,
                                    int head_dim, int max_ctx,
                                    int kv_len_base, int n_new);
int rsl_mlx_flash_attn_decode_nvfp4(rsl_mlx_stream *s, const float *q,
                                    const void *k_packed, const void *v_packed,
                                    float *out, int n_heads, int n_kv_heads,
                                    int head_dim, int max_ctx, int kv_len);
int rsl_mlx_flash_attn_prefill_nvfp4(rsl_mlx_stream *s, const float *q,
                                     const void *k_packed, const void *v_packed,
                                     float *out, int n_heads, int n_kv_heads,
                                     int head_dim, int max_ctx,
                                     int kv_len_base, int n_new);
int rsl_mlx_flash_attn_decode_mxfp4(rsl_mlx_stream *s, const float *q,
                                    const void *k_packed, const void *v_packed,
                                    float *out, int n_heads, int n_kv_heads,
                                    int head_dim, int max_ctx, int kv_len);
int rsl_mlx_flash_attn_prefill_mxfp4(rsl_mlx_stream *s, const float *q,
                                     const void *k_packed, const void *v_packed,
                                     float *out, int n_heads, int n_kv_heads,
                                     int head_dim, int max_ctx,
                                     int kv_len_base, int n_new);
int rsl_mlx_flash_attn_decode_mxfp6(rsl_mlx_stream *s, const float *q,
                                    const void *k_packed, const void *v_packed,
                                    float *out, int n_heads, int n_kv_heads,
                                    int head_dim, int max_ctx, int kv_len);
int rsl_mlx_flash_attn_prefill_mxfp6(rsl_mlx_stream *s, const float *q,
                                     const void *k_packed, const void *v_packed,
                                     float *out, int n_heads, int n_kv_heads,
                                     int head_dim, int max_ctx,
                                     int kv_len_base, int n_new);
int rsl_mlx_flash_attn_decode_mxfp8(rsl_mlx_stream *s, const float *q,
                                    const void *k_packed, const void *v_packed,
                                    float *out, int n_heads, int n_kv_heads,
                                    int head_dim, int max_ctx, int kv_len);
int rsl_mlx_flash_attn_prefill_mxfp8(rsl_mlx_stream *s, const float *q,
                                     const void *k_packed, const void *v_packed,
                                     float *out, int n_heads, int n_kv_heads,
                                     int head_dim, int max_ctx,
                                     int kv_len_base, int n_new);
int rsl_mlx_flash_attn_decode_tq(rsl_mlx_stream *s, const float *q,
                                 const void *k_packed, const void *v_packed,
                                 const float *k_scales, const float *v_scales,
                                 int bits, float *out, int n_heads,
                                 int n_kv_heads, int head_dim, int max_ctx,
                                 int kv_len);
int rsl_mlx_flash_attn_prefill_tq(rsl_mlx_stream *s, const float *q,
                                  const void *k_packed, const void *v_packed,
                                  const float *k_scales, const float *v_scales,
                                  int bits, float *out, int n_heads,
                                  int n_kv_heads, int head_dim, int max_ctx,
                                  int kv_len_base, int n_new);
int rsl_mlx_flash_attn_decode_q8_0(rsl_mlx_stream *s, const float *q,
                                   const void *k_packed, const void *v_packed,
                                   const float *k_scales, const float *v_scales,
                                   float *out, int n_heads, int n_kv_heads,
                                   int head_dim, int max_ctx, int kv_len);
int rsl_mlx_flash_attn_prefill_q8_0(rsl_mlx_stream *s, const float *q,
                                    const void *k_packed, const void *v_packed,
                                    const float *k_scales, const float *v_scales,
                                    float *out, int n_heads, int n_kv_heads,
                                    int head_dim, int max_ctx,
                                    int kv_len_base, int n_new);

/* Greedy argmax over `vocab` F32 logits; writes chosen index to out_idx[0]
 * (lowest index on ties). */
int rsl_mlx_argmax_f32(rsl_mlx_stream *s, const float *logits, int vocab,
                       int *out_idx);

/* Consume + reset this thread's kernel error latch (mirrors the CUDA/SYCL
 * crates' per-thread error tracking so the Rust wrapper can fall back to
 * CPU on a kernel failure). Returns the count since the last consume. */
int rsl_mlx_consume_error_count(void);

#ifdef __cplusplus
}
#endif

#endif /* RSL_MLX_H */
