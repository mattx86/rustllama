// Native CUDA kernels for rustllama's NVIDIA backend.
//
// This establishes the pattern + extern-"C" FFI surface: device query, a
// block-reduction (RMSNorm), and a warp-reduced mat-vec. The performance-
// critical quant-packed matvecs (PTQ1_0, Q4_K, Q6_K, Q8_0) and flash
// attention port from the SYCL kernels
// (crates/rustllama-kernels-sycl/cpp/rsl_kernels.cpp), keeping the same
// bitwise/tolerance parity discipline against the CPU reference.
//
// The reference host wrappers below copy H2D/D2H per call for clarity; a
// production backend keeps weights + KV device-resident on a stream (the
// USM-style path the SYCL layer uses) — that's a follow-on once the kernel
// math is validated on real hardware.

#include "rsl_cuda.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <new>

// Per-thread kernel-error latch. A launch/sync failure increments it; the
// Rust wrapper drains it via rsl_cuda_consume_error_count after each call and
// turns a non-zero count into a CPU fallback (mirrors the SYCL crate).
static thread_local int g_rsl_cuda_errors = 0;

// Check the last launch + a synchronize; on error, log once, bump the latch,
// and return non-zero.
static int rsl_cuda_check(const char *fn) {
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[rsl-cuda] %s failed: %s\n", fn, cudaGetErrorString(e));
        ++g_rsl_cuda_errors;
        return -1;
    }
    return 0;
}

extern "C" int rsl_cuda_consume_error_count(void) {
    int n = g_rsl_cuda_errors;
    g_rsl_cuda_errors = 0;
    return n;
}

extern "C" int rsl_cuda_device_count(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

extern "C" int rsl_cuda_device_info(int idx, char *name, int name_cap,
                                    unsigned long long *total_mem,
                                    int *cc_major, int *cc_minor,
                                    unsigned char *uuid /* 16 bytes or null */) {
    cudaDeviceProp p;
    if (uuid) std::memset(uuid, 0, 16);
    if (cudaGetDeviceProperties(&p, idx) != cudaSuccess) return -1;
    if (name && name_cap > 0) {
        std::strncpy(name, p.name, (size_t)name_cap - 1);
        name[name_cap - 1] = '\0';
    }
    if (total_mem)  *total_mem = (unsigned long long)p.totalGlobalMem;
    if (cc_major)   *cc_major  = p.major;
    if (cc_minor)   *cc_minor  = p.minor;
    // Stable, driver-invariant device UUID (cudaDeviceProp.uuid is a
    // cudaUUID_t = 16 raw bytes).
    if (uuid) std::memcpy(uuid, p.uuid.bytes, 16);
    return 0;
}

// ---- RMSNorm: one block per row, block-reduce sum of squares ----------
__global__ void rmsnorm_kernel(const float *x, const float *w, float *y,
                               int d, float eps) {
    int row = blockIdx.x;
    const float *xr = x + (size_t)row * d;
    float *yr = y + (size_t)row * d;

    __shared__ float sh[256];
    float local = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) local += xr[i] * xr[i];
    sh[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] / (float)d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) yr[i] = xr[i] * inv * w[i];
}

extern "C" int rsl_cuda_rmsnorm_f32(const float *x, const float *w, float *y,
                                    int n_rows, int d, float eps) {
    if (d <= 0 || n_rows <= 0) return -1;
    size_t nx = (size_t)n_rows * d;
    float *dx = nullptr, *dw = nullptr, *dy = nullptr;
    if (cudaMalloc(&dx, nx * sizeof(float)) != cudaSuccess) return -2;
    if (cudaMalloc(&dw, (size_t)d * sizeof(float)) != cudaSuccess) { cudaFree(dx); return -2; }
    if (cudaMalloc(&dy, nx * sizeof(float)) != cudaSuccess) { cudaFree(dx); cudaFree(dw); return -2; }
    cudaMemcpy(dx, x, nx * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w, (size_t)d * sizeof(float), cudaMemcpyHostToDevice);
    int threads = 256;
    rmsnorm_kernel<<<n_rows, threads>>>(dx, dw, dy, d, eps);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) cudaMemcpy(y, dy, nx * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dw); cudaFree(dy);
    return err == cudaSuccess ? 0 : -3;
}

// ---- f32 mat-vec: out[m] = sum_k W[m*k + k] * x[k] --------------------
// One warp (32 lanes) reduces the k-dimension for a row; blockDim.y rows
// per block.
__global__ void matvec_f32_kernel(const float *W, const float *x, float *out,
                                  int m_rows, int k_dim) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= m_rows) return;
    const float *wr = W + (size_t)row * k_dim;
    float acc = 0.f;
    for (int k = threadIdx.x; k < k_dim; k += warpSize) acc += wr[k] * x[k];
    for (int off = warpSize >> 1; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (threadIdx.x == 0) out[row] = acc;
}

extern "C" int rsl_cuda_matvec_f32(const float *W, const float *x, float *out,
                                   int m_rows, int k_dim) {
    if (m_rows <= 0 || k_dim <= 0) return -1;
    size_t wn = (size_t)m_rows * k_dim;
    float *dW = nullptr, *dx = nullptr, *dout = nullptr;
    if (cudaMalloc(&dW, wn * sizeof(float)) != cudaSuccess) return -2;
    if (cudaMalloc(&dx, (size_t)k_dim * sizeof(float)) != cudaSuccess) { cudaFree(dW); return -2; }
    if (cudaMalloc(&dout, (size_t)m_rows * sizeof(float)) != cudaSuccess) { cudaFree(dW); cudaFree(dx); return -2; }
    cudaMemcpy(dW, W, wn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, (size_t)k_dim * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(32, 8);
    dim3 grid((unsigned)((m_rows + block.y - 1) / block.y));
    matvec_f32_kernel<<<grid, block>>>(dW, dx, dout, m_rows, k_dim);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) cudaMemcpy(out, dout, (size_t)m_rows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dW); cudaFree(dx); cudaFree(dout);
    return err == cudaSuccess ? 0 : -3;
}

// ============================================================
// Device-resident path: stream + device buffers + packed kernels
// ============================================================

struct rsl_cuda_stream {
    int device;
    cudaStream_t stream;
};

extern "C" rsl_cuda_stream *rsl_cuda_stream_create(int device_index) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || device_index < 0 || device_index >= n) {
        return nullptr;
    }
    if (cudaSetDevice(device_index) != cudaSuccess) return nullptr;
    rsl_cuda_stream *s = new (std::nothrow) rsl_cuda_stream();
    if (!s) return nullptr;
    s->device = device_index;
    if (cudaStreamCreate(&s->stream) != cudaSuccess) {
        delete s;
        return nullptr;
    }
    return s;
}

extern "C" void rsl_cuda_stream_destroy(rsl_cuda_stream *s) {
    if (!s) return;
    cudaSetDevice(s->device);
    cudaStreamDestroy(s->stream);
    delete s;
}

extern "C" void *rsl_cuda_malloc_from_host(rsl_cuda_stream *s, const void *src,
                                           unsigned long long n_bytes) {
    if (!s || n_bytes == 0) return nullptr;
    cudaSetDevice(s->device);
    void *p = nullptr;
    if (cudaMalloc(&p, (size_t)n_bytes) != cudaSuccess) return nullptr;
    if (src) {
        if (cudaMemcpy(p, src, (size_t)n_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(p);
            return nullptr;
        }
    }
    return p;
}

extern "C" void *rsl_cuda_malloc_device(rsl_cuda_stream *s, unsigned long long n_bytes) {
    return rsl_cuda_malloc_from_host(s, nullptr, n_bytes);
}

extern "C" void rsl_cuda_free(rsl_cuda_stream *s, void *dev_ptr) {
    if (!s || !dev_ptr) return;
    cudaSetDevice(s->device);
    cudaFree(dev_ptr);
}

extern "C" int rsl_cuda_memcpy_h2d(rsl_cuda_stream *s, void *dst_dev,
                                   const void *src_host, unsigned long long n_bytes) {
    if (!s) return -1;
    cudaSetDevice(s->device);
    return cudaMemcpy(dst_dev, src_host, (size_t)n_bytes, cudaMemcpyHostToDevice) == cudaSuccess
               ? 0
               : -1;
}

extern "C" int rsl_cuda_memcpy_d2h(rsl_cuda_stream *s, void *dst_host,
                                   const void *src_dev, unsigned long long n_bytes) {
    if (!s) return -1;
    cudaSetDevice(s->device);
    return cudaMemcpy(dst_host, src_dev, (size_t)n_bytes, cudaMemcpyDeviceToHost) == cudaSuccess
               ? 0
               : -1;
}

__device__ __forceinline__ float rsl_f16_bits_to_f32(unsigned short bits) {
    return __half2float(__ushort_as_half(bits));
}

// ---- PTQ1_0 (Bonsai ternary) packed matvec: one thread per output row --
// Block: 28 B / 128 weights { qs:[24], qh:[2], d:f16 }. Trit extraction is
// the CPU/SYCL 8-bit trick: q' = q*3^n (u8 wrap), trit = ((q'*3)>>8) - 1.
__device__ __forceinline__ float ptq1_0_row_dot(const unsigned char *row,
                                                 const float *x, int blocks_per_row) {
    const unsigned char pow3[5] = {1, 3, 9, 27, 81};
    float acc = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        const unsigned char *blk = row + b * 28;
        const unsigned char *qs = blk;
        const unsigned char *qh = blk + 24;
        unsigned short d_bits =
            (unsigned short)blk[26] | ((unsigned short)blk[27] << 8);
        float d = rsl_f16_bits_to_f32(d_bits);
        const int x_base = b * 128;
        float sum = 0.0f;
        for (int n = 0; n < 5; ++n) {
            unsigned char p3 = pow3[n];
            int e0 = x_base + n * 16;
            for (int mm = 0; mm < 16; ++mm) {
                unsigned char qv = (unsigned char)(qs[mm] * p3);
                int trit = (((int)qv * 3) >> 8) - 1;
                sum += (float)trit * x[e0 + mm];
            }
        }
        for (int n = 0; n < 5; ++n) {
            unsigned char p3 = pow3[n];
            int e0 = x_base + 80 + n * 8;
            for (int mm = 0; mm < 8; ++mm) {
                unsigned char qv = (unsigned char)(qs[16 + mm] * p3);
                int trit = (((int)qv * 3) >> 8) - 1;
                sum += (float)trit * x[e0 + mm];
            }
        }
        for (int n = 0; n < 4; ++n) {
            unsigned char p3 = pow3[n];
            int e0 = x_base + 120 + n * 2;
            for (int hh = 0; hh < 2; ++hh) {
                unsigned char qv = (unsigned char)(qh[hh] * p3);
                int trit = (((int)qv * 3) >> 8) - 1;
                sum += (float)trit * x[e0 + hh];
            }
        }
        acc += d * sum;
    }
    return acc;
}

__global__ void ptq1_0_matvec_kernel(const unsigned char *w_bytes, const float *x,
                                     float *out, int M, int K) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    const int blocks_per_row = K / 128;
    const int bytes_per_row = blocks_per_row * 28;
    out[m] = ptq1_0_row_dot(w_bytes + (size_t)m * bytes_per_row, x, blocks_per_row);
}

__global__ void ptq1_0_matvec_batched_kernel(const unsigned char *w_bytes,
                                             const float *x, float *out,
                                             int M, int K, int N) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (m >= M || n >= N) return;
    const int blocks_per_row = K / 128;
    const int bytes_per_row = blocks_per_row * 28;
    const float *x_row = x + (size_t)n * K;
    out[(size_t)n * M + m] =
        ptq1_0_row_dot(w_bytes + (size_t)m * bytes_per_row, x_row, blocks_per_row);
}

extern "C" int rsl_cuda_matvec_ptq1_0_packed_f32(rsl_cuda_stream *s,
                                                 const void *w_bytes_dev,
                                                 const float *x_dev, float *out_dev,
                                                 int M, int K) {
    if (!s || !w_bytes_dev || !x_dev || !out_dev || M <= 0 || K <= 0 || (K % 128) != 0) {
        return -1;
    }
    cudaSetDevice(s->device);
    int threads = 128;
    int blocks = (M + threads - 1) / threads;
    ptq1_0_matvec_kernel<<<blocks, threads, 0, s->stream>>>(
        (const unsigned char *)w_bytes_dev, x_dev, out_dev, M, K);
    return rsl_cuda_check("rsl_cuda_matvec_ptq1_0_packed_f32");
}

extern "C" int rsl_cuda_matvec_ptq1_0_packed_f32_batched(rsl_cuda_stream *s,
                                                         const void *w_bytes_dev,
                                                         const float *x_dev,
                                                         float *out_dev,
                                                         int M, int K, int N) {
    if (!s || !w_bytes_dev || !x_dev || !out_dev || M <= 0 || K <= 0 || N <= 0 ||
        (K % 128) != 0) {
        return -1;
    }
    cudaSetDevice(s->device);
    int threads = 128;
    dim3 blocks((M + threads - 1) / threads, (unsigned)N);
    ptq1_0_matvec_batched_kernel<<<blocks, threads, 0, s->stream>>>(
        (const unsigned char *)w_bytes_dev, x_dev, out_dev, M, K, N);
    return rsl_cuda_check("rsl_cuda_matvec_ptq1_0_packed_f32_batched");
}

// ---- Blockwise Prism Hadamard: out = WHT(signs*x)/sqrt(block) per span --
// One block per span; butterfly staged in shared memory with a barrier per
// stage (the SYCL kernel's CUDA companion).
__global__ void hadamard_forward_kernel(const float *x, const float *signs,
                                        float *out, int block, float inv_sqrt) {
    extern __shared__ float slm[];
    int blk = blockIdx.x;
    int lws = blockDim.x;
    int tid = threadIdx.x;
    int per_thread = block / lws;
    int base = blk * block;
    for (int i = 0; i < per_thread; ++i) {
        int e = tid * per_thread + i;
        slm[e] = x[base + e] * signs[base + e];
    }
    __syncthreads();
    for (int half = 1; half < block; half <<= 1) {
        for (int i = 0; i < per_thread; ++i) {
            int e = tid * per_thread + i;
            if (e < block / 2) {
                int lo = (e / half) * (half << 1) + (e % half);
                int hi = lo + half;
                float a = slm[lo];
                float bqv = slm[hi];
                slm[lo] = a + bqv;
                slm[hi] = a - bqv;
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < per_thread; ++i) {
        int e = tid * per_thread + i;
        out[base + e] = slm[e] * inv_sqrt;
    }
}

extern "C" int rsl_cuda_hadamard_forward(rsl_cuda_stream *s, const float *x_dev,
                                         const float *signs_dev, float *out_dev,
                                         int n_elems, int block) {
    if (!s || !x_dev || !signs_dev || !out_dev || n_elems <= 0 || block <= 0 ||
        (n_elems % block) != 0 || (block & (block - 1)) != 0 || block > 4096) {
        return -1;
    }
    cudaSetDevice(s->device);
    int n_blocks = n_elems / block;
    int lws = block < 256 ? block : 256;
    float inv_sqrt = 1.0f / sqrtf((float)block);
    hadamard_forward_kernel<<<n_blocks, lws, (size_t)block * sizeof(float), s->stream>>>(
        x_dev, signs_dev, out_dev, block, inv_sqrt);
    return rsl_cuda_check("rsl_cuda_hadamard_forward");
}

// ============================================================
// K-quant packed matvecs (Q8_0 / Q4_K / Q6_K) — one thread per row,
// device-pointer args. Byte-exact ports of the SYCL impls. A macro
// generates the single + batched kernels + launchers from each `row_dot`.
// ============================================================

// Q8_0: 34 B / 32-weight block { f16 scale, 32 i8 }. K % 32 == 0.
__device__ __forceinline__ float q8_0_row_dot(const unsigned char *row,
                                              const float *x, int bpr) {
    float acc = 0.0f;
    for (int b = 0; b < bpr; ++b) {
        const unsigned char *blk = row + b * 34;
        unsigned short sb = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
        float scale = rsl_f16_bits_to_f32(sb);
        int xo = b * 32;
        float dot = 0.0f;
        for (int d = 0; d < 32; ++d) {
            dot += (float)((signed char)blk[2 + d]) * x[xo + d];
        }
        acc += scale * dot;
    }
    return acc;
}

// Q4_K_M: 144 B / 256-weight super-block. K % 256 == 0.
__device__ __forceinline__ float q4_k_row_dot(const unsigned char *row,
                                              const float *x, int bpr) {
    float acc = 0.0f;
    for (int b = 0; b < bpr; ++b) {
        const unsigned char *blk = row + b * 144;
        unsigned short d_bits = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
        unsigned short m_bits = (unsigned short)blk[2] | ((unsigned short)blk[3] << 8);
        float d = rsl_f16_bits_to_f32(d_bits);
        float dmin = rsl_f16_bits_to_f32(m_bits);
        const unsigned char *sb = blk + 4;
        unsigned char sc[8], mn[8];
        for (int j = 0; j < 8; ++j) {
            if (j < 4) {
                sc[j] = sb[j] & 0x3F;
                mn[j] = sb[j + 4] & 0x3F;
            } else {
                sc[j] = (sb[j + 4] & 0x0F) | ((sb[j - 4] >> 6) << 4);
                mn[j] = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4);
            }
        }
        const unsigned char *qs = blk + 16;
        const int x_base = b * 256;
        for (int group = 0; group < 4; ++group) {
            const unsigned char *qc = qs + group * 32;
            float d_lo = d * (float)sc[group * 2];
            float m_lo = dmin * (float)mn[group * 2];
            float d_hi = d * (float)sc[group * 2 + 1];
            float m_hi = dmin * (float)mn[group * 2 + 1];
            int x_lo = x_base + group * 64;
            int x_hi = x_lo + 32;
            for (int l = 0; l < 32; ++l) {
                unsigned char qb = qc[l];
                float q_lo = (float)(qb & 0x0F);
                float q_hi = (float)(qb >> 4);
                acc += (d_lo * q_lo - m_lo) * x[x_lo + l];
                acc += (d_hi * q_hi - m_hi) * x[x_hi + l];
            }
        }
    }
    return acc;
}

// Q6_K: 210 B / 256-weight super-block. K % 256 == 0.
__device__ __forceinline__ float q6_k_row_dot(const unsigned char *row,
                                              const float *x, int bpr) {
    float acc = 0.0f;
    for (int b = 0; b < bpr; ++b) {
        const unsigned char *blk = row + b * 210;
        const unsigned char *ql = blk;
        const unsigned char *qh = blk + 128;
        const signed char *scales = (const signed char *)(blk + 192);
        unsigned short d_bits = (unsigned short)blk[208] | ((unsigned short)blk[209] << 8);
        float d = rsl_f16_bits_to_f32(d_bits);
        int x_base = b * 256;
        for (int n = 0; n < 2; ++n) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16 + n * 8;
                int qh_byte = qh[32 * n + l];
                int q1 = ((int)(ql[64 * n + l] & 0x0F) | (((qh_byte >> 0) & 0x03) << 4)) - 32;
                int q2 = ((int)(ql[64 * n + l + 32] & 0x0F) | (((qh_byte >> 2) & 0x03) << 4)) - 32;
                int q3 = ((int)(ql[64 * n + l] >> 4) | (((qh_byte >> 4) & 0x03) << 4)) - 32;
                int q4 = ((int)(ql[64 * n + l + 32] >> 4) | (((qh_byte >> 6) & 0x03) << 4)) - 32;
                float s0 = (float)scales[is];
                float s1 = (float)scales[is + 2];
                float s2 = (float)scales[is + 4];
                float s3 = (float)scales[is + 6];
                int base = n * 128 + l;
                acc += d * s0 * (float)q1 * x[x_base + base];
                acc += d * s1 * (float)q2 * x[x_base + base + 32];
                acc += d * s2 * (float)q3 * x[x_base + base + 64];
                acc += d * s3 * (float)q4 * x[x_base + base + 96];
            }
        }
    }
    return acc;
}

// Generate `<name>` (single) + `<name>_batched` kernels and their extern-"C"
// launchers from a device `row_dot`. `kmod` = K divisor; `bpb`/`epb` =
// bytes/elements per block.
#define RSL_PACKED_MATVEC(NAME, ROWDOT, KMOD, BPB, EPB)                       \
    __global__ void NAME##_kernel(const unsigned char *w, const float *x,     \
                                  float *out, int M, int K) {                 \
        int m = blockIdx.x * blockDim.x + threadIdx.x;                        \
        if (m >= M) return;                                                   \
        int bpr = K / (EPB);                                                  \
        out[m] = ROWDOT(w + (size_t)m * bpr * (BPB), x, bpr);                 \
    }                                                                         \
    __global__ void NAME##_batched_kernel(const unsigned char *w,            \
                                          const float *x, float *out, int M,  \
                                          int K, int N) {                     \
        int m = blockIdx.x * blockDim.x + threadIdx.x;                        \
        int n = blockIdx.y;                                                   \
        if (m >= M || n >= N) return;                                         \
        int bpr = K / (EPB);                                                  \
        out[(size_t)n * M + m] =                                              \
            ROWDOT(w + (size_t)m * bpr * (BPB), x + (size_t)n * K, bpr);      \
    }                                                                         \
    extern "C" int rsl_cuda_##NAME(rsl_cuda_stream *s, const void *w,         \
                                   const float *x, float *out, int M, int K) {\
        if (!s || !w || !x || !out || M <= 0 || K <= 0 || (K % (KMOD)) != 0)  \
            return -1;                                                        \
        cudaSetDevice(s->device);                                            \
        int t = 128, b = (M + t - 1) / t;                                    \
        NAME##_kernel<<<b, t, 0, s->stream>>>((const unsigned char *)w, x,   \
                                              out, M, K);                     \
        return rsl_cuda_check("rsl_cuda_" #NAME);                            \
    }                                                                         \
    extern "C" int rsl_cuda_##NAME##_batched(rsl_cuda_stream *s,             \
                                             const void *w, const float *x,   \
                                             float *out, int M, int K,        \
                                             int N) {                         \
        if (!s || !w || !x || !out || M <= 0 || K <= 0 || N <= 0 ||          \
            (K % (KMOD)) != 0)                                                \
            return -1;                                                        \
        cudaSetDevice(s->device);                                            \
        int t = 128;                                                          \
        dim3 b((M + t - 1) / t, (unsigned)N);                                \
        NAME##_batched_kernel<<<b, t, 0, s->stream>>>(                        \
            (const unsigned char *)w, x, out, M, K, N);                       \
        return rsl_cuda_check("rsl_cuda_" #NAME "_batched");                 \
    }

RSL_PACKED_MATVEC(matvec_q8_0_packed_f32, q8_0_row_dot, 32, 34, 32)
RSL_PACKED_MATVEC(matvec_q4_k_packed_f32, q4_k_row_dot, 256, 144, 256)
RSL_PACKED_MATVEC(matvec_q6_k_packed_f32, q6_k_row_dot, 256, 210, 256)

// ============================================================
// Forward-pass primitives (device-resident, F32). Byte-for-byte
// ports of the SYCL USM kernels in rsl_kernels.cpp, kept F32
// throughout (the CUDA backend keeps activations + KV cache in F32).
// One extern-"C" launcher each, all on the stream's device.
// ============================================================

// Add-residual + RMSNorm fused (mirrors rsl_add_rmsnorm_usm):
//   hidden[i] += branch[i]                      (residual, in place)
//   y_norm[i]  = rmsnorm(hidden_row)[i] * w[i]  (pre-norm output)
// One block per row; block-reduce the sum of squares.
__global__ void add_rmsnorm_f32_kernel(float *hidden, const float *branch,
                                       const float *w, float *y_norm,
                                       int d, float eps) {
    int row = blockIdx.x;
    float *hr = hidden + (size_t)row * d;
    float *yr = y_norm + (size_t)row * d;
    const float *br = branch + (size_t)row * d;
    __shared__ float sh[256];
    float local = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = hr[i] + br[i];
        hr[i] = v;
        local += v * v;
    }
    sh[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] / (float)d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) yr[i] = hr[i] * inv * w[i];
}

extern "C" int rsl_cuda_add_rmsnorm_f32(rsl_cuda_stream *s, float *hidden,
                                        const float *branch, const float *w,
                                        float *y_norm, int n_rows, int d,
                                        float eps) {
    if (!s || !hidden || !branch || !w || !y_norm || n_rows <= 0 || d <= 0)
        return -1;
    cudaSetDevice(s->device);
    int threads = d < 256 ? d : 256;
    // round threads up to a power of two <= 256 for the reduction tree
    int t = 1;
    while (t < threads) t <<= 1;
    if (t > 256) t = 256;
    add_rmsnorm_f32_kernel<<<n_rows, t, 0, s->stream>>>(hidden, branch, w,
                                                        y_norm, d, eps);
    return rsl_cuda_check("rsl_cuda_add_rmsnorm_f32");
}

// RoPE (mirrors rsl_rope_usm): rotate the (j, j+half) pair of each head
// by angle = pos * inv_freq[j]. qk is [n_heads, head_dim]; inv_freq is
// [head_dim/2] precomputed inverse frequencies. One thread per (head, j).
__global__ void rope_f32_kernel(float *qk, int n_heads, int head_dim, int pos,
                                const float *inv_freq) {
    int half = head_dim / 2;
    int total = n_heads * half;
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= total) return;
    int head = flat / half;
    int j = flat % half;
    float angle = (float)pos * inv_freq[j];
    float c = cosf(angle), si = sinf(angle);
    int base = head * head_dim;
    float x0 = qk[base + j];
    float x1 = qk[base + j + half];
    qk[base + j] = x0 * c - x1 * si;
    qk[base + j + half] = x0 * si + x1 * c;
}

extern "C" int rsl_cuda_rope_f32(rsl_cuda_stream *s, float *qk, int n_heads,
                                 int head_dim, int pos, const float *inv_freq) {
    if (!s || !qk || !inv_freq || n_heads <= 0 || head_dim <= 0 ||
        (head_dim % 2) != 0)
        return -1;
    cudaSetDevice(s->device);
    int total = n_heads * (head_dim / 2);
    int t = 128, b = (total + t - 1) / t;
    rope_f32_kernel<<<b, t, 0, s->stream>>>(qk, n_heads, head_dim, pos, inv_freq);
    return rsl_cuda_check("rsl_cuda_rope_f32");
}

// SwiGLU (mirrors rsl_silu_mul_usm): out[i] = silu(x[i]) * y[i].
__global__ void silu_mul_f32_kernel(const float *x, const float *y, float *out,
                                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xv = x[i];
    float silu = xv / (1.0f + expf(-xv));
    out[i] = silu * y[i];
}

extern "C" int rsl_cuda_silu_mul_f32(rsl_cuda_stream *s, const float *x,
                                     const float *y, float *out, int n) {
    if (!s || !x || !y || !out || n <= 0) return -1;
    cudaSetDevice(s->device);
    int t = 256, b = (n + t - 1) / t;
    silu_mul_f32_kernel<<<b, t, 0, s->stream>>>(x, y, out, n);
    return rsl_cuda_check("rsl_cuda_silu_mul_f32");
}

// Embedding lookup (mirrors rsl_embedding_lookup_usm): out[i] = table[ids[i]]
// (d elements). ids < 0 → zero row. `ids` is a DEVICE pointer here (the CUDA
// worker uploads it) — no host scratch like the SYCL variant. One block per id.
__global__ void embedding_lookup_f32_kernel(const float *table,
                                            const int *ids, float *out,
                                            int n_ids, int d) {
    int i = blockIdx.x;
    if (i >= n_ids) return;
    int row = ids[i];
    float *o = out + (size_t)i * d;
    if (row < 0) {
        for (int j = threadIdx.x; j < d; j += blockDim.x) o[j] = 0.f;
        return;
    }
    const float *tr = table + (size_t)row * d;
    for (int j = threadIdx.x; j < d; j += blockDim.x) o[j] = tr[j];
}

extern "C" int rsl_cuda_embedding_lookup_f32(rsl_cuda_stream *s,
                                             const float *table, const int *ids,
                                             float *out, int n_ids, int d) {
    if (!s || !table || !ids || !out || n_ids <= 0 || d <= 0) return -1;
    cudaSetDevice(s->device);
    int t = d < 256 ? d : 256;
    embedding_lookup_f32_kernel<<<n_ids, t, 0, s->stream>>>(table, ids, out,
                                                            n_ids, d);
    return rsl_cuda_check("rsl_cuda_embedding_lookup_f32");
}

// FlashAttention decode, F32 (mirrors rsl_flash_attn_decode_usm v1):
//   q:   [n_heads, head_dim]
//   k/v: [n_kv_heads, max_ctx, head_dim]
//   out: [n_heads, head_dim]
// One thread per head; serial online softmax over kv_len positions.
__global__ void flash_attn_decode_f32_kernel(const float *q, const float *k,
                                             const float *v, float *out,
                                             int n_heads, int n_gqa,
                                             int head_dim, int max_ctx,
                                             int kv_len, float scale) {
    int hh = blockIdx.x * blockDim.x + threadIdx.x;
    if (hh >= n_heads) return;
    int kv_h = hh / n_gqa;
    int q_base = hh * head_dim;
    int out_base = hh * head_dim;
    for (int i = 0; i < head_dim; ++i) out[out_base + i] = 0.f;
    float m = -INFINITY, l = 0.f;
    for (int t = 0; t < kv_len; ++t) {
        int k_off = (kv_h * max_ctx + t) * head_dim;
        float s_dot = 0.f;
        for (int i = 0; i < head_dim; ++i) s_dot += q[q_base + i] * k[k_off + i];
        s_dot *= scale;
        float m_new = fmaxf(m, s_dot);
        float rescale = isfinite(m) ? expf(m - m_new) : 0.f;
        float p = expf(s_dot - m_new);
        l = l * rescale + p;
        int v_off = (kv_h * max_ctx + t) * head_dim;
        for (int i = 0; i < head_dim; ++i)
            out[out_base + i] = out[out_base + i] * rescale + p * v[v_off + i];
        m = m_new;
    }
    float inv_l = (l > 0.f) ? 1.f / l : 0.f;
    for (int i = 0; i < head_dim; ++i) out[out_base + i] *= inv_l;
}

extern "C" int rsl_cuda_flash_attn_decode_f32(rsl_cuda_stream *s,
                                              const float *q, const float *k,
                                              const float *v, float *out,
                                              int n_heads, int n_kv_heads,
                                              int head_dim, int max_ctx,
                                              int kv_len) {
    if (!s || !q || !k || !v || !out) return -1;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0)
        return -1;
    if ((n_heads % n_kv_heads) != 0) return -1;
    cudaSetDevice(s->device);
    if (kv_len <= 0) {
        cudaMemsetAsync(out, 0, (size_t)n_heads * head_dim * sizeof(float),
                        s->stream);
        return rsl_cuda_check("rsl_cuda_flash_attn_decode_f32");
    }
    int n_gqa = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    int t = 64, b = (n_heads + t - 1) / t;
    flash_attn_decode_f32_kernel<<<b, t, 0, s->stream>>>(
        q, k, v, out, n_heads, n_gqa, head_dim, max_ctx, kv_len, scale);
    return rsl_cuda_check("rsl_cuda_flash_attn_decode_f32");
}

// FlashAttention prefill, F32 (mirrors rsl_flash_attn_prefill_usm):
//   q:   [n_new, n_heads, head_dim]  row-major
//   k/v: [n_kv_heads, max_ctx, head_dim]
//   out: [n_new, n_heads, head_dim]  row-major
// Causal: query at new position q_pos attends absolute [0, kv_len_base+q_pos].
// 2D grid: x = head, y = q_pos. One thread per (head, q_pos).
__global__ void flash_attn_prefill_f32_kernel(const float *q, const float *k,
                                              const float *v, float *out,
                                              int n_heads, int n_gqa,
                                              int head_dim, int max_ctx,
                                              int kv_len_base, int n_new,
                                              float scale) {
    int hh = blockIdx.x * blockDim.x + threadIdx.x;
    int q_pos = blockIdx.y;
    if (hh >= n_heads || q_pos >= n_new) return;
    int kv_h = hh / n_gqa;
    int q_off = (q_pos * n_heads + hh) * head_dim;
    int out_off = q_off;
    int kv_len_for_q = kv_len_base + q_pos + 1;
    for (int i = 0; i < head_dim; ++i) out[out_off + i] = 0.f;
    float m = -INFINITY, l = 0.f;
    for (int t = 0; t < kv_len_for_q; ++t) {
        int kv_off = (kv_h * max_ctx + t) * head_dim;
        float s_dot = 0.f;
        for (int i = 0; i < head_dim; ++i)
            s_dot += q[q_off + i] * k[kv_off + i];
        s_dot *= scale;
        float m_new = fmaxf(m, s_dot);
        float rescale = isfinite(m) ? expf(m - m_new) : 0.f;
        float p = expf(s_dot - m_new);
        l = l * rescale + p;
        for (int i = 0; i < head_dim; ++i)
            out[out_off + i] = out[out_off + i] * rescale + p * v[kv_off + i];
        m = m_new;
    }
    float inv_l = (l > 0.f) ? 1.f / l : 0.f;
    for (int i = 0; i < head_dim; ++i) out[out_off + i] *= inv_l;
}

extern "C" int rsl_cuda_flash_attn_prefill_f32(rsl_cuda_stream *s,
                                               const float *q, const float *k,
                                               const float *v, float *out,
                                               int n_heads, int n_kv_heads,
                                               int head_dim, int max_ctx,
                                               int kv_len_base, int n_new) {
    if (!s || !q || !k || !v || !out) return -1;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0 ||
        n_new <= 0)
        return -1;
    if ((n_heads % n_kv_heads) != 0) return -1;
    if (kv_len_base < 0 || kv_len_base + n_new > max_ctx) return -1;
    cudaSetDevice(s->device);
    int n_gqa = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    int t = 64;
    dim3 b((n_heads + t - 1) / t, (unsigned)n_new);
    flash_attn_prefill_f32_kernel<<<b, t, 0, s->stream>>>(
        q, k, v, out, n_heads, n_gqa, head_dim, max_ctx, kv_len_base, n_new,
        scale);
    return rsl_cuda_check("rsl_cuda_flash_attn_prefill_f32");
}

// Greedy argmax over an F32 logits buffer (mirrors rsl_sampler_argmax_usm):
// writes the index of the max logit to out_idx[0]. Single block; each thread
// strides the vocab, then a block reduction picks the winner (lowest index
// on ties, matching the CPU reference's first-max semantics).
__global__ void argmax_f32_kernel(const float *logits, int vocab,
                                  int *out_idx) {
    __shared__ float sval[256];
    __shared__ int sidx[256];
    int tid = threadIdx.x;
    float best = -INFINITY;
    int besti = -1;
    for (int i = tid; i < vocab; i += blockDim.x) {
        float v = logits[i];
        if (v > best) { best = v; besti = i; }
    }
    sval[tid] = best;
    sidx[tid] = besti;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid + s];
            int oi = sidx[tid + s];
            // Prefer strictly-greater; on equal value prefer the lower index.
            if (ov > sval[tid] || (ov == sval[tid] && oi >= 0 &&
                                   (sidx[tid] < 0 || oi < sidx[tid]))) {
                sval[tid] = ov;
                sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) out_idx[0] = sidx[0];
}

extern "C" int rsl_cuda_argmax_f32(rsl_cuda_stream *s, const float *logits,
                                   int vocab, int *out_idx) {
    if (!s || !logits || !out_idx || vocab <= 0) return -1;
    cudaSetDevice(s->device);
    argmax_f32_kernel<<<1, 256, 0, s->stream>>>(logits, vocab, out_idx);
    return rsl_cuda_check("rsl_cuda_argmax_f32");
}

