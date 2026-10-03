// ============================================================================
// rsl_blackwell.cuh — Blackwell SM12x tensor-core GEMMs (hand-rolled PTX).
// ============================================================================
//
// Phase 1+ of the "all Blackwell perf" effort. These kernels drive NVIDIA's
// 5th-gen tensor cores on *consumer / workstation* Blackwell (SM12x: the
// RTX 50-series and the DGX Spark GB10, which is **sm_121**) via the warp-level
// block-scaled `mma.sync` family. (Datacenter Blackwell — sm_100/B200 — uses a
// different `tcgen05.mma` + Tensor-Memory programming model; SM12x has NO
// tcgen05, no Tensor Memory, no cluster multicast, so we use `mma.sync`, the
// SM80-style warp-collective MMA, extended with Blackwell's block-scaling.)
//
// CLEAN-ROOM: hand-rolled from the PTX ISA + public layout descriptions
// (NVIDIA PTX ISA 9.x block-scaling section; Colfax Research's SM12x NVFP4
// block-scaled GEMM tutorial for the operand + scale-factor TV layout). NO
// CUTLASS is vendored or linked — only its documented layouts are used as the
// correctness spec.
//
// ---------------------------------------------------------------------------
// WRITE-BLIND STATUS (read before trusting the numbers)
// ---------------------------------------------------------------------------
// This file was authored on a host with NO NVIDIA GPU. Its validation ceiling
// here is "nvcc + ptxas accept the instructions for sm_120a". The *numeric*
// correctness of the operand fragment layout and the block-scale selector
// semantics is NOT verified here — it is validated on the user's DGX Spark
// (GB10 / sm_121) via `rustllama doctor --cuda-parity`. Every place where the
// exact HW layout is assumed-from-docs rather than measured is tagged
// `SPARK-VALIDATE`.
//
// ---------------------------------------------------------------------------
// The load-bearing instruction (confirmed to assemble for `.target sm_120a`):
//
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col
//       .f32.e2m1.e2m1.f32.ue4m3
//       {d0..d3}, {a0..a3}, {b0,b1}, {c0..c3}, sfa,{bidA,tidA}, sfb,{bidB,tidB};
//
//   * D/C : 16x8 f32 accumulator, 4 regs/thread (canonical m16n8 layout).
//   * A   : 16x64 FP4 (e2m1), 4 regs/thread = 32 nibbles (the WEIGHT tile).
//   * B   : 64x8 FP4 (e2m1), 2 regs/thread = 16 nibbles (the ACTIVATION tile).
//   * sfa/sfb : one .b32 scale register each + two immediate selectors
//               {byte-id, thread-id}. scale_vec::4X => 4 scale factors span the
//               K=64 atom (NVFP4 micro-block = 16). MXFP4 uses scale_vec::2X +
//               ue8m0 (micro-block = 32 => 2 per atom).
//
// IMPORTANT — this is a W4A4 path: `mma.sync.kind::mxf4nvf4` multiplies FP4 x
// FP4, so the f32 activation tile is **dynamically quantized to NVFP4** before
// the MMA (per-16 E4M3 block scale, same recipe as the weight encoder). That
// adds activation quantization error on top of the weight's — the scalar
// reference path (`nvfp4_row_dot`) keeps f32 activations (W4A16), so the TC
// path is graded against it with a LOOSER tolerance. Callers that need the
// tighter W4A16 numerics keep the scalar path; the TC path trades accuracy for
// the 4-bit tensor-core throughput and is opt-in at the dispatch layer.
// ============================================================================
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Real tensor-core path is compiled only when (a) build.rs targeted an SM12x
// architecture-accelerated arch (`-DRSL_BLACKWELL_TC`, i.e. sm_120a/121a/12xf)
// AND (b) this device pass is Blackwell (`__CUDA_ARCH__ >= 1200`). Otherwise a
// scalar fallback body is compiled so the default Ampere->Hopper build (and any
// plain sm_120) still builds; the host launchers additionally refuse to launch
// off a real SM12x device (see rsl_cuda_blackwell_tc_available).
#if defined(RSL_BLACKWELL_TC) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
#define RSL_BW_DEVICE_TC 1
#endif

namespace rslbw {

// ---------------------------------------------------------------------------
// Scalar element encoders (f32 -> packed low-precision), for on-the-fly
// activation quantization. Decoders (e4m3/e8m0 -> f32) already live in
// rsl_cuda.cu; this header is #included AFTER them so RSL_NVFP4_CODEBOOK and
// rsl_e4m3_to_f32/rsl_e8m0_to_f32 are visible.
// ---------------------------------------------------------------------------

// f32 -> NVFP4 E2M1 nibble (nearest codebook entry). Returns 0..15 (bit 3 =
// sign). Mirrors the CPU encoder's nearest-of-codebook choice.
__device__ __forceinline__ unsigned rsl_f32_to_e2m1(float v) {
    const float sign = v < 0.0f ? 8.0f : 0.0f;  // high nibble bit
    float a = fabsf(v);
    // Positive codebook magnitudes {0,.5,1,1.5,2,3,4,6}; pick nearest.
    const float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    int best = 0;
    float bestd = fabsf(a - lut[0]);
    #pragma unroll
    for (int i = 1; i < 8; ++i) {
        float d = fabsf(a - lut[i]);
        if (d < bestd) { bestd = d; best = i; }
    }
    return (unsigned)best | (unsigned)sign;
}

// f32 -> FP8 E4M3 byte. Saturating, round-to-nearest-even on the 3-bit
// mantissa; subnormals flush to 0. Byte-compatible with rsl_e4m3_to_f32's
// decode. (Port of rustllama-kernels-cpu nvfp4::f32_to_e4m3.)
__device__ __forceinline__ unsigned char rsl_f32_to_e4m3(float x) {
    if (isnan(x)) return 0x7F;
    unsigned char sign = signbit(x) ? 0x80 : 0x00;
    float mag = fabsf(x);
    const float E4M3_MAX = 448.0f;
    if (mag >= E4M3_MAX) return sign | 0x7E;  // saturate (0x7F/FF = NaN)
    if (mag == 0.0f) return sign;
    unsigned bits = __float_as_uint(mag);
    int f32_exp = (int)((bits >> 23) & 0xFF) - 127;
    unsigned f32_mant = bits & 0x7FFFFF;
    int e = f32_exp + 7;  // E4M3 bias 7
    if (e <= 0) return sign;             // subnormal -> flush
    if (e > 15) return sign | 0x7E;      // overflow -> saturate
    unsigned mant = (f32_mant >> 20) & 0x07;
    unsigned dropped = f32_mant & ((1u << 20) - 1);
    unsigned half = 1u << 19;
    if (dropped > half || (dropped == half && (mant & 1))) mant += 1;
    if (mant > 0x07) { if (e + 1 > 15) return sign | 0x7E; e += 1; mant = 0; }
    return sign | (unsigned char)((e << 3) | mant);
}

// f32 (>0) -> OCP E8M0 power-of-two exponent byte: value = 2^(byte-127).
// Picks the smallest 2^x >= x so a block's max element fits the codebook top.
// 0 -> 0 (=> 2^-127). (Approx port of mxfp block-scale picker.)
__device__ __forceinline__ unsigned char rsl_f32_to_e8m0(float x) {
    if (!(x > 0.0f)) return 0;
    int e;
    frexpf(x, &e);                 // x in [0.5,1)*2^e  => 2^(e-1) <= x < 2^e
    int biased = (e - 1) + 127;    // floor(log2 x) + bias
    if (biased < 0) biased = 0;
    if (biased > 254) biased = 254;
    return (unsigned char)biased;
}

// f32 -> OCP E3M2 (MXFP6 element): 6-bit `s eee mm` (1 sign, 3 exp bias 3,
// 2 mantissa). Bit-compatible with rsl_e3m2_to_f32's decode. No Inf/NaN;
// saturates at 28 = (1+3/4)*2^4. Used to quantize activations for the W6A6
// MXFP6 tensor-core path.
__device__ __forceinline__ unsigned char rsl_f32_to_e3m2(float x) {
    unsigned char sign = signbit(x) ? 0x20 : 0x00;   // bit 5
    float mag = fabsf(x);
    if (mag >= 28.0f) return sign | 0x1F;            // exp7 mant3
    if (mag == 0.0f) return sign;
    unsigned bits = __float_as_uint(mag);
    int e = (int)((bits >> 23) & 0xFF) - 127 + 3;    // bias 3
    unsigned fm = bits & 0x7FFFFF;
    if (e <= 0) return sign;                          // subnormal -> flush
    if (e > 7) return sign | 0x1F;
    unsigned mant = (fm >> 21) & 0x3, dropped = fm & ((1u << 21) - 1), half = 1u << 20;
    if (dropped > half || (dropped == half && (mant & 1))) mant += 1;
    if (mant > 3) { if (e + 1 > 7) return sign | 0x1F; e += 1; mant = 0; }
    return sign | (unsigned char)((e << 2) | mant);
}

#ifdef RSL_BW_DEVICE_TC
// ---------------------------------------------------------------------------
// Hand-rolled block-scaled MMA wrappers (sm_120a). These are the clean-room
// primitives the whole Blackwell effort is built on; each matches the exact
// operand grouping the NVVM front-end emitted for the confirmed instruction.
// ---------------------------------------------------------------------------

// NVFP4: FP4xFP4 -> f32, per-16 UE4M3 block scales (scale_vec::4X).
__device__ __forceinline__ void rsl_bw_mma_nvfp4(
    float d[4], const unsigned a[4], const unsigned b[2], const float c[4],
    unsigned sfa, unsigned sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
        ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
        "%14, {0, 0}, %15, {0, 0};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sfa), "r"(sfb));
}

// MXFP4: FP4xFP4 -> f32, per-32 UE8M0 block scales (scale_vec::2X).
__device__ __forceinline__ void rsl_bw_mma_mxfp4(
    float d[4], const unsigned a[4], const unsigned b[2], const float c[4],
    unsigned sfa, unsigned sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64"
        ".row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
        "%14, {0, 0}, %15, {0, 0};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sfa), "r"(sfb));
}

// MXFP8: E4M3xE4M3 -> f32, per-32 UE8M0 block scale (scale_vec::1X, m16n8k32).
// kind::mxf8f6f4 packs each 8-bit element in a byte; A=4 regs(16) B=2 regs(8).
__device__ __forceinline__ void rsl_bw_mma_mxf8(
    float d[4], const unsigned a[4], const unsigned b[2], const float c[4],
    unsigned sfa, unsigned sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32"
        ".row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
        "%14, {0, 0}, %15, {0, 0};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sfa), "r"(sfb));
}

// MXFP6: E3M2xE3M2 -> f32, per-32 UE8M0 block scale (scale_vec::1X, m16n8k32).
// The 6-bit E3M2 code rides the low bits of each 8-bit container.
__device__ __forceinline__ void rsl_bw_mma_mxf6(
    float d[4], const unsigned a[4], const unsigned b[2], const float c[4],
    unsigned sfa, unsigned sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32"
        ".row.col.f32.e3m2.e3m2.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
        "%14, {0, 0}, %15, {0, 0};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sfa), "r"(sfb));
}
#endif  // RSL_BW_DEVICE_TC

// ---------------------------------------------------------------------------
// FP4 fragment layout (m16n8k64, kind::mxf4nvf4), per Colfax SM12x + PTX ISA.
// groupID g = lane>>2 (0..7), quad-lane q = lane&3 (0..3).
//   A (weight 16x64): thread owns 32 e2m1 = 4 regs:
//       a0 = W[row=g    , K = q*8 + 0..7]
//       a1 = W[row=g    , K = 32 + q*8 + 0..7]
//       a2 = W[row=g+8  , K = q*8 + 0..7]
//       a3 = W[row=g+8  , K = 32 + q*8 + 0..7]
//   B (activation 64x8, col-major): thread owns 16 e2m1 = 2 regs, column n=g:
//       b0 = X[n=g, K = q*8 + 0..7]
//       b1 = X[n=g, K = 32 + q*8 + 0..7]
//   D (16x8 f32): d0=(g,2q) d1=(g,2q+1) d2=(g+8,2q) d3=(g+8,2q+1)   [col = token n]
// SPARK-VALIDATE: the A/B element->lane map above and the SFA/SFB scale-register
// packing + {byte-id,thread-id} selectors (left 0 here) are taken from the
// public TV-layout docs, not measured. --cuda-parity on the Spark is the
// arbiter; a mismatch shows as MISCOMPUTE and is a layout-index fix, not a
// toolchain problem.
// ---------------------------------------------------------------------------

// Read one NVFP4 nibble: weight row m (global), global K index gk. Row stride
// is (K/16)*9 bytes; block b=gk/16 holds 8 code bytes + 1 E4M3 scale byte.
__device__ __forceinline__ unsigned rsl_nvfp4_nibble(
    const unsigned char* w, int m, int gk, int K) {
    long long row_base = (long long)m * (K / 16) * 9;
    int b = gk >> 4, wi = gk & 15;
    unsigned char byte = w[row_base + (long long)b * 9 + (wi >> 1)];
    return (wi & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
}
// NVFP4 block E4M3 scale byte for weight row m, K-block kb (= gk/16).
__device__ __forceinline__ unsigned char rsl_nvfp4_scale(
    const unsigned char* w, int m, int kb, int K) {
    long long row_base = (long long)m * (K / 16) * 9;
    return w[row_base + (long long)kb * 9 + 8];
}

// Pack 8 consecutive weight nibbles (row m, global K = gk0..gk0+7) into a u32
// A-register; out-of-range rows contribute 0 (edge tiles).
__device__ __forceinline__ unsigned rsl_pack_a_nvfp4(
    const unsigned char* w, int m, int M, int gk0, int K) {
    if (m >= M) return 0u;
    unsigned r = 0u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) r |= rsl_nvfp4_nibble(w, m, gk0 + i, K) << (4 * i);
    return r;
}

// ---------------------------------------------------------------------------
// NVFP4 W4A4 tensor-core GEMM: out[n*M + m] = sum_k W[m,k]*X[n,k], W NVFP4,
// X f32 (quantized to NVFP4 on the fly). One warp per (16-row x 8-token) tile.
// Requires K % 64 == 0. M, N arbitrary (edges masked).
// ---------------------------------------------------------------------------
__global__ void rsl_bw_gemm_nvfp4_kernel(
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, int M, int N, int K) {
    const int m0 = blockIdx.x * 16;
    const int n0 = blockIdx.y * 8;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;     // 0..7
    const int q = lane & 3;      // 0..3

#ifdef RSL_BW_DEVICE_TC
    // Stage the 8x64 f32 activation tile (per K-step) so each lane can derive
    // its NVFP4 block scales (max over 16) without a warp-wide reduction.
    __shared__ float xs[8][64];
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    const int rowA0 = m0 + g, rowA1 = m0 + g + 8;

    for (int k0 = 0; k0 < K; k0 += 64) {
        // (Re)stage activation tile X[n0:8, k0:k0+64].
        for (int idx = lane; idx < 8 * 64; idx += 32) {
            int row = idx >> 6, col = idx & 63;
            int n = n0 + row;
            xs[row][col] = (n < N) ? x[(long long)n * K + k0 + col] : 0.0f;
        }
        __syncwarp();

        // ---- A fragment (weights, 4 regs) ----
        unsigned a[4];
        a[0] = rsl_pack_a_nvfp4(w, rowA0, M, k0 + q * 8, K);
        a[1] = rsl_pack_a_nvfp4(w, rowA0, M, k0 + 32 + q * 8, K);
        a[2] = rsl_pack_a_nvfp4(w, rowA1, M, k0 + q * 8, K);
        a[3] = rsl_pack_a_nvfp4(w, rowA1, M, k0 + 32 + q * 8, K);
        // SFA: pack row rowA0's 4 K-block E4M3 scales (bytes 0..3). SPARK-VALIDATE.
        unsigned sfa = 0u;
        if (rowA0 < M) {
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                sfa |= (unsigned)rsl_nvfp4_scale(w, rowA0, (k0 >> 4) + j, K) << (8 * j);
        }

        // ---- B fragment (activations, quantized to NVFP4) ----
        // Compute the 4 per-16 block scales for token row g from smem.
        float bscale[4];
        unsigned sfb = 0u;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float mx = 0.f;
            #pragma unroll
            for (int t = 0; t < 16; ++t) mx = fmaxf(mx, fabsf(xs[g][j * 16 + t]));
            float s = mx > 0.f ? mx / 6.0f : 1.0f;
            bscale[j] = s;
            sfb |= (unsigned)rsl_f32_to_e4m3(s) << (8 * j);
        }
        unsigned b[2];
        {
            unsigned r0 = 0u, r1 = 0u;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int kk0 = q * 8 + i, kk1 = 32 + q * 8 + i;
                float inv0 = 1.0f / bscale[kk0 >> 4];
                float inv1 = 1.0f / bscale[kk1 >> 4];
                r0 |= rsl_f32_to_e2m1(xs[g][kk0] * inv0) << (4 * i);
                r1 |= rsl_f32_to_e2m1(xs[g][kk1] * inv1) << (4 * i);
            }
            b[0] = r0; b[1] = r1;
        }

        float d[4];
        rsl_bw_mma_nvfp4(d, a, b, c, sfa, sfb);
        c[0] = d[0]; c[1] = d[1]; c[2] = d[2]; c[3] = d[3];
        __syncwarp();
    }

    // Store the 16x8 tile: d0=(g,2q) d1=(g,2q+1) d2=(g+8,2q) d3=(g+8,2q+1).
    int cols[2] = {2 * q, 2 * q + 1};
    int rows[2] = {m0 + g, m0 + g + 8};
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        int m = rows[rr];
        if (m >= M) continue;
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
            int n = n0 + cols[cc];
            if (n >= N) continue;
            out[(long long)n * M + m] = c[rr * 2 + cc];
        }
    }
#else
    // -------- Scalar fallback (non-Blackwell build / device pass) --------
    // Correct NVFP4(W) x f32(A) GEMM so the kernel is valid everywhere; only
    // ever launched if the host gate mis-fires (it normally refuses). One warp
    // still owns a 16x8 tile; each lane does a couple of scalar dots.
    for (int rr = 0; rr < 2; ++rr) {
        int m = (rr == 0) ? (m0 + g) : (m0 + g + 8);
        if (m >= M) continue;
        for (int cc = 0; cc < 2; ++cc) {
            int n = n0 + 2 * q + cc;
            if (n >= N) continue;
            float acc = 0.f;
            for (int kk = 0; kk < K; ++kk) {
                int kb = kk >> 4;
                float sc = rsl_e4m3_to_f32(rsl_nvfp4_scale(w, m, kb, K));
                unsigned nib = rsl_nvfp4_nibble(w, m, kk, K);
                acc += sc * RSL_NVFP4_CODEBOOK[nib] * x[(long long)n * K + kk];
            }
            out[(long long)n * M + m] = acc;
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// MXFP4 W4A4 tensor-core GEMM (per-32 UE8M0 block scale, scale_vec::2X).
// Weight is MXFP4: 17 bytes / 32 elems (16 code bytes + 1 E8M0 scale). The
// m16n8k64 atom spans 2 MXFP4 blocks. Same tiling as NVFP4.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned rsl_mxfp4_nibble(
    const unsigned char* w, int m, int gk, int K) {
    long long row_base = (long long)m * (K / 32) * 17;
    int b = gk >> 5, wi = gk & 31;
    unsigned char byte = w[row_base + (long long)b * 17 + (wi >> 1)];
    return (wi & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
}
__device__ __forceinline__ unsigned char rsl_mxfp4_scale(
    const unsigned char* w, int m, int kb, int K) {
    long long row_base = (long long)m * (K / 32) * 17;
    return w[row_base + (long long)kb * 17 + 16];
}
__device__ __forceinline__ unsigned rsl_pack_a_mxfp4(
    const unsigned char* w, int m, int M, int gk0, int K) {
    if (m >= M) return 0u;
    unsigned r = 0u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) r |= rsl_mxfp4_nibble(w, m, gk0 + i, K) << (4 * i);
    return r;
}

__global__ void rsl_bw_gemm_mxfp4_kernel(
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, int M, int N, int K) {
    const int m0 = blockIdx.x * 16;
    const int n0 = blockIdx.y * 8;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int q = lane & 3;

#ifdef RSL_BW_DEVICE_TC
    __shared__ float xs[8][64];
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    const int rowA0 = m0 + g, rowA1 = m0 + g + 8;

    for (int k0 = 0; k0 < K; k0 += 64) {
        for (int idx = lane; idx < 8 * 64; idx += 32) {
            int row = idx >> 6, col = idx & 63;
            int n = n0 + row;
            xs[row][col] = (n < N) ? x[(long long)n * K + k0 + col] : 0.0f;
        }
        __syncwarp();

        unsigned a[4];
        a[0] = rsl_pack_a_mxfp4(w, rowA0, M, k0 + q * 8, K);
        a[1] = rsl_pack_a_mxfp4(w, rowA0, M, k0 + 32 + q * 8, K);
        a[2] = rsl_pack_a_mxfp4(w, rowA1, M, k0 + q * 8, K);
        a[3] = rsl_pack_a_mxfp4(w, rowA1, M, k0 + 32 + q * 8, K);
        // SFA: 2 MXFP4 (per-32) block E8M0 scales over K=64 (scale_vec::2X).
        unsigned sfa = 0u;
        if (rowA0 < M) {
            sfa |= (unsigned)rsl_mxfp4_scale(w, rowA0, (k0 >> 5) + 0, K) << 0;
            sfa |= (unsigned)rsl_mxfp4_scale(w, rowA0, (k0 >> 5) + 1, K) << 8;
        }

        // B: quantize activation token g to MXFP4 (2 per-32 blocks over K=64).
        float bscale[2];
        unsigned sfb = 0u;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            float mx = 0.f;
            #pragma unroll
            for (int t = 0; t < 32; ++t) mx = fmaxf(mx, fabsf(xs[g][j * 32 + t]));
            unsigned char e8 = rsl_f32_to_e8m0(mx > 0.f ? mx / 6.0f : 1.0f);
            bscale[j] = rsl_e8m0_to_f32(e8);
            sfb |= (unsigned)e8 << (8 * j);
        }
        unsigned b[2];
        {
            unsigned r0 = 0u, r1 = 0u;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int kk0 = q * 8 + i, kk1 = 32 + q * 8 + i;
                r0 |= rsl_f32_to_e2m1(xs[g][kk0] / bscale[kk0 >> 5]) << (4 * i);
                r1 |= rsl_f32_to_e2m1(xs[g][kk1] / bscale[kk1 >> 5]) << (4 * i);
            }
            b[0] = r0; b[1] = r1;
        }

        float d[4];
        rsl_bw_mma_mxfp4(d, a, b, c, sfa, sfb);
        c[0] = d[0]; c[1] = d[1]; c[2] = d[2]; c[3] = d[3];
        __syncwarp();
    }

    int cols[2] = {2 * q, 2 * q + 1};
    int rows[2] = {m0 + g, m0 + g + 8};
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        int m = rows[rr];
        if (m >= M) continue;
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
            int n = n0 + cols[cc];
            if (n >= N) continue;
            out[(long long)n * M + m] = c[rr * 2 + cc];
        }
    }
#else
    for (int rr = 0; rr < 2; ++rr) {
        int m = (rr == 0) ? (m0 + g) : (m0 + g + 8);
        if (m >= M) continue;
        for (int cc = 0; cc < 2; ++cc) {
            int n = n0 + 2 * q + cc;
            if (n >= N) continue;
            float acc = 0.f;
            for (int kk = 0; kk < K; ++kk) {
                int kb = kk >> 5;
                float sc = rsl_e8m0_to_f32(rsl_mxfp4_scale(w, m, kb, K));
                unsigned nib = rsl_mxfp4_nibble(w, m, kk, K);
                acc += sc * RSL_NVFP4_CODEBOOK[nib] * x[(long long)n * K + kk];
            }
            out[(long long)n * M + m] = acc;
        }
    }
#endif
}

// ===========================================================================
// Phase 2+3: FP8 (MXFP8) + FP6 (MXFP6) block-scaled tensor-core GEMMs.
// kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32 (UE8M0, per-32 block).
// 8-bit containers: A=4 regs(16 elems) B=2 regs(8) D/C=4 f32. K-step 32.
// Fragment layout (m16n8k32, 8-bit), g=lane>>2, q=lane&3:
//   a0=A[g,q*4..+4] a1=A[g+8,q*4..+4] a2=A[g,16+q*4..+4] a3=A[g+8,16+q*4..+4]
//   b0=B[n=g,q*4..+4] b1=B[n=g,16+q*4..+4]  d0=(g,2q) d1=(g,2q+1) d2/3 row g+8
// W8A8 / W6A6 (activations quantized to E4M3/E3M2 on the fly). SPARK-VALIDATE
// (same discipline as the FP4 header banner).
// ===========================================================================

// Store the warp's 16x8 f32 accumulator ACC[4] to out[n*M+m] (col-major in M),
// masking M/N edges. Uses g,q,m0,n0,M,N,out in scope.
#define RSL_BW_STORE_TILE(ACC)                                                 \
    do {                                                                       \
        int _cols[2] = {2 * q, 2 * q + 1};                                     \
        int _rows[2] = {m0 + g, m0 + g + 8};                                   \
        for (int _rr = 0; _rr < 2; ++_rr) {                                    \
            int _m = _rows[_rr];                                               \
            if (_m >= M) continue;                                             \
            for (int _cc = 0; _cc < 2; ++_cc) {                                \
                int _n = n0 + _cols[_cc];                                      \
                if (_n >= N) continue;                                         \
                out[(long long)_n * M + _m] = (ACC)[_rr * 2 + _cc];            \
            }                                                                  \
        }                                                                      \
    } while (0)

// Scalar fallback GEMM body: out[n,m] = sum_k DECODED * x[n,kk], DECODED an
// expression in (w,m,kk,K). Never launched (host gate refuses off-Blackwell);
// present only so the non-TC build compiles. Uses g,q,m0,n0,M,N,K,w,x,out.
#define RSL_BW_FALLBACK(DECODED)                                               \
    do {                                                                       \
        for (int _rr = 0; _rr < 2; ++_rr) {                                    \
            int m = (_rr == 0) ? (m0 + g) : (m0 + g + 8);                      \
            if (m >= M) continue;                                              \
            for (int _cc = 0; _cc < 2; ++_cc) {                                \
                int n = n0 + 2 * q + _cc;                                      \
                if (n >= N) continue;                                          \
                float acc = 0.f;                                              \
                for (int kk = 0; kk < K; ++kk) acc += (DECODED)*x[(long long)n * K + kk]; \
                out[(long long)n * M + m] = acc;                              \
            }                                                                  \
        }                                                                      \
    } while (0)

// MXFP8 readers: 33 B / 32 (32 E4M3 bytes + E8M0 tail).
__device__ __forceinline__ unsigned char rsl_mxfp8_byte(const unsigned char* w, int m, int gk, int K) {
    long long rb = (long long)m * (K / 32) * 33; return w[rb + (long long)(gk >> 5) * 33 + (gk & 31)];
}
__device__ __forceinline__ unsigned char rsl_mxfp8_scale(const unsigned char* w, int m, int kb, int K) {
    long long rb = (long long)m * (K / 32) * 33; return w[rb + (long long)kb * 33 + 32];
}
// MXFP6 readers: 25 B / 32 (LE 6-bit E3M2 stream + E8M0 tail) -> 6-bit code in a byte.
__device__ __forceinline__ unsigned char rsl_mxfp6_code(const unsigned char* w, int m, int gk, int K) {
    long long rb = (long long)m * (K / 32) * 25;
    const unsigned char* c = w + rb + (long long)(gk >> 5) * 25;  // 24 code bytes
    int bp = (gk & 31) * 6, bi = bp >> 3, off = bp & 7;
    unsigned lo = c[bi], hi = (bi + 1 < 24) ? c[bi + 1] : 0u;
    return (unsigned char)(((lo | (hi << 8)) >> off) & 0x3F);
}
__device__ __forceinline__ unsigned char rsl_mxfp6_scale(const unsigned char* w, int m, int kb, int K) {
    long long rb = (long long)m * (K / 32) * 25; return w[rb + (long long)kb * 25 + 24];
}
// Pack 4 consecutive 8-bit containers into an A-reg. fmt: 0=mxfp8, 1=mxfp6.
__device__ __forceinline__ unsigned rsl_pack_a_8bit(const unsigned char* w, int m, int M, int gk0, int K, int fmt) {
    if (m >= M) return 0u;
    unsigned r = 0u;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned char by = (fmt == 0) ? rsl_mxfp8_byte(w, m, gk0 + i, K) : rsl_mxfp6_code(w, m, gk0 + i, K);
        r |= (unsigned)by << (8 * i);
    }
    return r;
}

// MXFP8 W8A8 tensor-core GEMM. K % 32 == 0.
__global__ void rsl_bw_gemm_mxfp8_kernel(const unsigned char* __restrict__ w, const float* __restrict__ x,
                                         float* __restrict__ out, int M, int N, int K) {
    const int m0 = blockIdx.x * 16, n0 = blockIdx.y * 8, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
#ifdef RSL_BW_DEVICE_TC
    __shared__ float xs[8][32];
    float c[4] = {0, 0, 0, 0}; const int rA0 = m0 + g, rA1 = m0 + g + 8;
    for (int k0 = 0; k0 < K; k0 += 32) {
        for (int idx = lane; idx < 256; idx += 32) { int r = idx >> 5, cl = idx & 31, n = n0 + r; xs[r][cl] = (n < N) ? x[(long long)n * K + k0 + cl] : 0.f; }
        __syncwarp();
        unsigned a[4];
        a[0] = rsl_pack_a_8bit(w, rA0, M, k0 + q * 4, K, 0); a[1] = rsl_pack_a_8bit(w, rA1, M, k0 + q * 4, K, 0);
        a[2] = rsl_pack_a_8bit(w, rA0, M, k0 + 16 + q * 4, K, 0); a[3] = rsl_pack_a_8bit(w, rA1, M, k0 + 16 + q * 4, K, 0);
        unsigned sfa = (rA0 < M) ? (unsigned)rsl_mxfp8_scale(w, rA0, k0 >> 5, K) : 0u;
        float mx = 0.f;
        #pragma unroll
        for (int t = 0; t < 32; ++t) mx = fmaxf(mx, fabsf(xs[g][t]));
        unsigned char e8 = rsl_f32_to_e8m0(mx > 0.f ? mx / 448.0f : 1.0f); float bs = rsl_e8m0_to_f32(e8); unsigned sfb = e8;
        unsigned b[2] = {0, 0};
        #pragma unroll
        for (int i = 0; i < 4; ++i) { b[0] |= (unsigned)rsl_f32_to_e4m3(xs[g][q * 4 + i] / bs) << (8 * i); b[1] |= (unsigned)rsl_f32_to_e4m3(xs[g][16 + q * 4 + i] / bs) << (8 * i); }
        float d[4]; rsl_bw_mma_mxf8(d, a, b, c, sfa, sfb); c[0] = d[0]; c[1] = d[1]; c[2] = d[2]; c[3] = d[3];
        __syncwarp();
    }
    RSL_BW_STORE_TILE(c);
#else
    RSL_BW_FALLBACK(rsl_e8m0_to_f32(rsl_mxfp8_scale(w, m, kk >> 5, K)) * rsl_e4m3_to_f32(rsl_mxfp8_byte(w, m, kk, K)));
#endif
}

// MXFP6 W6A6 tensor-core GEMM. K % 32 == 0.
__global__ void rsl_bw_gemm_mxfp6_kernel(const unsigned char* __restrict__ w, const float* __restrict__ x,
                                         float* __restrict__ out, int M, int N, int K) {
    const int m0 = blockIdx.x * 16, n0 = blockIdx.y * 8, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
#ifdef RSL_BW_DEVICE_TC
    __shared__ float xs[8][32];
    float c[4] = {0, 0, 0, 0}; const int rA0 = m0 + g, rA1 = m0 + g + 8;
    for (int k0 = 0; k0 < K; k0 += 32) {
        for (int idx = lane; idx < 256; idx += 32) { int r = idx >> 5, cl = idx & 31, n = n0 + r; xs[r][cl] = (n < N) ? x[(long long)n * K + k0 + cl] : 0.f; }
        __syncwarp();
        unsigned a[4];
        a[0] = rsl_pack_a_8bit(w, rA0, M, k0 + q * 4, K, 1); a[1] = rsl_pack_a_8bit(w, rA1, M, k0 + q * 4, K, 1);
        a[2] = rsl_pack_a_8bit(w, rA0, M, k0 + 16 + q * 4, K, 1); a[3] = rsl_pack_a_8bit(w, rA1, M, k0 + 16 + q * 4, K, 1);
        unsigned sfa = (rA0 < M) ? (unsigned)rsl_mxfp6_scale(w, rA0, k0 >> 5, K) : 0u;
        float mx = 0.f;
        #pragma unroll
        for (int t = 0; t < 32; ++t) mx = fmaxf(mx, fabsf(xs[g][t]));
        unsigned char e8 = rsl_f32_to_e8m0(mx > 0.f ? mx / 28.0f : 1.0f); float bs = rsl_e8m0_to_f32(e8); unsigned sfb = e8;
        unsigned b[2] = {0, 0};
        #pragma unroll
        for (int i = 0; i < 4; ++i) { b[0] |= (unsigned)rsl_f32_to_e3m2(xs[g][q * 4 + i] / bs) << (8 * i); b[1] |= (unsigned)rsl_f32_to_e3m2(xs[g][16 + q * 4 + i] / bs) << (8 * i); }
        float d[4]; rsl_bw_mma_mxf6(d, a, b, c, sfa, sfb); c[0] = d[0]; c[1] = d[1]; c[2] = d[2]; c[3] = d[3];
        __syncwarp();
    }
    RSL_BW_STORE_TILE(c);
#else
    RSL_BW_FALLBACK(rsl_e8m0_to_f32(rsl_mxfp6_scale(w, m, kk >> 5, K)) * rsl_e3m2_to_f32(rsl_mxfp6_code(w, m, kk, K)));
#endif
}

// ===========================================================================
// Phase 4: TMA (Tensor Memory Accelerator) async-copy operand staging.
// Replaces the NVFP4 kernel's per-thread activation-staging loop with a
// cp.async.bulk (non-tensor: one copy per token row = 64 contiguous f32 =
// 256 B) + mbarrier. No CUtensorMap / driver API needed (that would be the
// 2D-strided tensor variant). All mbarrier / cp.async.bulk PTX confirmed to
// assemble for sm_120a. Single-buffered (copy -> wait -> compute); the
// double-buffered overlap that actually hides the copy latency is the perf
// follow-up. SPARK-VALIDATE: the mbarrier phase/transaction-count semantics
// and the bulk-copy alignment are assumed-from-docs. Dispatch-gated behind a
// separate opt-in; parity validates it == the scalar reference.
// ===========================================================================
#ifdef RSL_BW_DEVICE_TC
__device__ __forceinline__ void rsl_bw_mbar_init(unsigned mbar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(mbar), "r"(count) : "memory");
}
__device__ __forceinline__ void rsl_bw_mbar_expect(unsigned mbar, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(mbar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void rsl_bw_bulk_g2s(unsigned dst_smem, const void* src_g, int bytes, unsigned mbar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(dst_smem), "l"(src_g), "r"(bytes), "r"(mbar) : "memory");
}
__device__ __forceinline__ void rsl_bw_mbar_wait(unsigned mbar, int phase) {
    asm volatile(
        "{ .reg .pred p; L_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra L_%=; }\n"
        :: "r"(mbar), "r"(phase) : "memory");
}
#endif

// TMA-staged NVFP4 GEMM — identical math to rsl_bw_gemm_nvfp4_kernel; only the
// 8x64 activation tile staging differs (cp.async.bulk + mbarrier per K-step).
__global__ void rsl_bw_gemm_nvfp4_tma_kernel(
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, int M, int N, int K) {
    const int m0 = blockIdx.x * 16, n0 = blockIdx.y * 8, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
#ifdef RSL_BW_DEVICE_TC
    __shared__ alignas(16) float xs[8][64];
    __shared__ alignas(8) unsigned long long mbar;
    unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);
    if (lane == 0) rsl_bw_mbar_init(mb, 1);
    __syncwarp();
    float c[4] = {0, 0, 0, 0}; const int rA0 = m0 + g, rA1 = m0 + g + 8;
    int phase = 0;
    for (int k0 = 0; k0 < K; k0 += 64) {
        int valid = N - n0; if (valid > 8) valid = 8; if (valid < 0) valid = 0;
        if (lane == 0) {
            rsl_bw_mbar_expect(mb, valid * 64 * (int)sizeof(float));
            for (int r = 0; r < valid; ++r) {
                unsigned dst = (unsigned)__cvta_generic_to_shared(&xs[r][0]);
                rsl_bw_bulk_g2s(dst, &x[(long long)(n0 + r) * K + k0], 64 * (int)sizeof(float), mb);
            }
        }
        // Zero the edge rows not covered by a bulk copy so B quant is defined.
        for (int idx = lane; idx < (8 - valid) * 64; idx += 32) { int r = valid + (idx >> 6); xs[r][idx & 63] = 0.f; }
        rsl_bw_mbar_wait(mb, phase); phase ^= 1;
        __syncwarp();
        unsigned a[4];
        a[0] = rsl_pack_a_nvfp4(w, rA0, M, k0 + q * 8, K); a[1] = rsl_pack_a_nvfp4(w, rA0, M, k0 + 32 + q * 8, K);
        a[2] = rsl_pack_a_nvfp4(w, rA1, M, k0 + q * 8, K); a[3] = rsl_pack_a_nvfp4(w, rA1, M, k0 + 32 + q * 8, K);
        unsigned sfa = 0u;
        if (rA0 < M) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) sfa |= (unsigned)rsl_nvfp4_scale(w, rA0, (k0 >> 4) + j, K) << (8 * j);
        }
        float bscale[4]; unsigned sfb = 0u;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float mx = 0.f;
            #pragma unroll
            for (int t = 0; t < 16; ++t) mx = fmaxf(mx, fabsf(xs[g][j * 16 + t]));
            float s = mx > 0.f ? mx / 6.0f : 1.0f; bscale[j] = s;
            sfb |= (unsigned)rsl_f32_to_e4m3(s) << (8 * j);
        }
        unsigned b[2] = {0, 0};
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int kk0 = q * 8 + i, kk1 = 32 + q * 8 + i;
            b[0] |= rsl_f32_to_e2m1(xs[g][kk0] / bscale[kk0 >> 4]) << (4 * i);
            b[1] |= rsl_f32_to_e2m1(xs[g][kk1] / bscale[kk1 >> 4]) << (4 * i);
        }
        float d[4]; rsl_bw_mma_nvfp4(d, a, b, c, sfa, sfb); c[0] = d[0]; c[1] = d[1]; c[2] = d[2]; c[3] = d[3];
        __syncwarp();
    }
    RSL_BW_STORE_TILE(c);
#else
    RSL_BW_FALLBACK(rsl_e4m3_to_f32(rsl_nvfp4_scale(w, m, kk >> 4, K)) * RSL_NVFP4_CODEBOOK[rsl_nvfp4_nibble(w, m, kk, K)]);
#endif
}

// ===========================================================================
// Phase 5: 2:4 structured-sparse FP8 tensor-core GEMM (OPT-IN, quality caveat).
// mma.sp::ordered_metadata.m16n8k64.kind::f8f6f4 (E4M3). Logical K=64 with 2:4
// sparsity: A compressed to 16x32 (4 regs/thread) + 2-bit metadata; B full
// 64x8 (4 regs/thread). Confirmed to assemble for sm_120a (toolchain proof
// variant 8).
//
// QUALITY CAVEAT: forcing a dense model into a 2:4 pattern = magnitude-pruning
// 50% of each 4-group's weights. That is lossless ONLY for a model TRAINED for
// 2:4; on an arbitrary dense model it SILENTLY degrades output. This path is
// therefore OPT-IN (default OFF) and intended for sparsity-aware models or
// accepted-loss scenarios. The on-the-fly magnitude prune here is a convenience
// for experimentation; production use should prune / fine-tune offline.
//
// SPARK-VALIDATE (MORE than the other phases): the compressed-A fragment layout
// AND the ordered_metadata bit encoding are assumed from the CUTLASS sparse
// spec and are NOT verifiable without hardware. --cuda-parity on the Spark (vs
// the 2:4-pruned dense reference) is the arbiter; expect an index fix here.
// ===========================================================================
#ifdef RSL_BW_DEVICE_TC
// 2:4 sparse E4M3 x E4M3 -> f32 (non-block). A compressed (4 regs), B full
// (4 regs), metadata e (1 reg), sparsity selector 0x0.
__device__ __forceinline__ void rsl_bw_mma_sp_f8(
    float d[4], const unsigned a[4], const unsigned b[4], const float c[4], unsigned e) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4"
        ".f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(e));
}
#endif

// Prune one 4-group of E4M3 bytes to 2:4 (keep the 2 largest |decoded|): writes
// the 2 kept bytes to out[0..2] (in ascending original position) and returns
// the metadata nibble (bits[1:0]=first kept pos, bits[3:2]=second kept pos).
__device__ __forceinline__ unsigned rsl_bw_prune24_group(
    const unsigned char gb[4], unsigned char out[2]) {
    float mag[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) mag[i] = fabsf(rsl_e4m3_to_f32(gb[i]));
    int i0 = 0;
    #pragma unroll
    for (int i = 1; i < 4; ++i) if (mag[i] > mag[i0]) i0 = i;
    int i1 = -1;
    #pragma unroll
    for (int i = 0; i < 4; ++i) if (i != i0 && (i1 < 0 || mag[i] > mag[i1])) i1 = i;
    if (i1 < i0) { int t = i0; i0 = i1; i1 = t; }
    out[0] = gb[i0]; out[1] = gb[i1];
    return (unsigned)(i0 & 3) | ((unsigned)(i1 & 3) << 2);
}

// 2:4-sparse FP8 GEMM. W dense E4M3 (1 B/elem, row-major, scale 1.0); X f32
// (quantized to E4M3 per token-row per K-step, scale applied to the partial,
// so the mma runs with C=0 and we accumulate scale*D). K % 64 == 0.
__global__ void rsl_bw_gemm_fp8_sp24_kernel(
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, int M, int N, int K) {
    const int m0 = blockIdx.x * 16, n0 = blockIdx.y * 8, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
#ifdef RSL_BW_DEVICE_TC
    __shared__ float xs[8][64];
    float c[4] = {0, 0, 0, 0};
    const int rA0 = m0 + g, rA1 = m0 + g + 8;
    for (int k0 = 0; k0 < K; k0 += 64) {
        for (int idx = lane; idx < 8 * 64; idx += 32) { int r = idx >> 6, cl = idx & 63, n = n0 + r; xs[r][cl] = (n < N) ? x[(long long)n * K + k0 + cl] : 0.f; }
        __syncwarp();
        // Compressed A (16x32 = 16 kept E4M3/thread = 4 regs) + 32-bit metadata.
        // Thread (g,q) owns rows {g,g+8} and logical cols [q*16,q*16+16) per row
        // = 4 2:4-groups/row; 4 groups/row x 2 rows = 8 groups -> 16 kept bytes.
        // SPARK-VALIDATE: (thread -> logical col) map + ordered_metadata order.
        unsigned a[4] = {0, 0, 0, 0}, e = 0u;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            int m = (half == 0) ? rA0 : rA1;
            #pragma unroll
            for (int grp = 0; grp < 4; ++grp) {
                int base = q * 16 + grp * 4;
                unsigned char kept[2] = {0, 0}, gb[4];
                unsigned md = 0u;
                if (m < M) {
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) gb[t] = w[(long long)m * K + k0 + base + t];
                    md = rsl_bw_prune24_group(gb, kept);
                }
                int reg = half * 2 + (grp >> 1);
                int shift = (grp & 1) * 16;
                a[reg] |= ((unsigned)kept[0] | ((unsigned)kept[1] << 8)) << shift;
                e |= md << ((half * 4 + grp) * 4);
            }
        }
        // Full B (64x8 = 16 E4M3/thread = 4 regs), cols n=g, logical K
        // [q*16,q*16+16) matching A. Quantized to E4M3, one scale per row/K-step.
        float mx = 0.f;
        #pragma unroll
        for (int t = 0; t < 64; ++t) mx = fmaxf(mx, fabsf(xs[g][t]));
        float bs = mx > 0.f ? mx / 448.0f : 1.0f;
        unsigned b[4] = {0, 0, 0, 0};
        #pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
            unsigned r = 0u;
            #pragma unroll
            for (int t = 0; t < 4; ++t) r |= (unsigned)rsl_f32_to_e4m3(xs[g][q * 16 + chunk * 4 + t] / bs) << (8 * t);
            b[chunk] = r;
        }
        float z[4] = {0, 0, 0, 0}, d[4];
        rsl_bw_mma_sp_f8(d, a, b, z, e);
        c[0] += bs * d[0]; c[1] += bs * d[1]; c[2] += bs * d[2]; c[3] += bs * d[3];
        __syncwarp();
    }
    RSL_BW_STORE_TILE(c);
#else
    // Fallback: 2:4-pruned dense reference (prune each 4-group, dot with x).
    for (int _rr = 0; _rr < 2; ++_rr) {
        int m = (_rr == 0) ? (m0 + g) : (m0 + g + 8);
        if (m >= M) continue;
        for (int _cc = 0; _cc < 2; ++_cc) {
            int n = n0 + 2 * q + _cc;
            if (n >= N) continue;
            float acc = 0.f;
            for (int k0 = 0; k0 < K; k0 += 4) {
                unsigned char gb[4];
                #pragma unroll
                for (int t = 0; t < 4; ++t) gb[t] = w[(long long)m * K + k0 + t];
                unsigned char kp[2];
                unsigned md = rsl_bw_prune24_group(gb, kp);
                int i0 = md & 3, i1 = (md >> 2) & 3;
                acc += rsl_e4m3_to_f32(kp[0]) * x[(long long)n * K + k0 + i0];
                acc += rsl_e4m3_to_f32(kp[1]) * x[(long long)n * K + k0 + i1];
            }
            out[(long long)n * M + m] = acc;
        }
    }
#endif
}

}  // namespace rslbw

// ---------------------------------------------------------------------------
// Host entry points (extern "C"). Gate on a real SM12x Blackwell device +
// compiled TC support; otherwise return -2 ("unavailable") so the Rust caller
// falls back to the scalar packed-matvec path. K % 64 == 0 required.
// ---------------------------------------------------------------------------

// 1 if this build compiled the TC path AND device `dev` is SM12x Blackwell.
extern "C" int rsl_cuda_blackwell_tc_available(int dev) {
#if defined(RSL_BLACKWELL_TC)
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return 0;
    return (p.major == 12) ? 1 : 0;  // SM12x consumer/workstation Blackwell
#else
    (void)dev;
    return 0;
#endif
}

// KMOD = the K divisor the kernel's atom requires (64 for FP4 m16n8k64, 32 for
// the 8-bit m16n8k32 MXFP8/MXFP6 kernels).
#define RSL_BW_GEMM_LAUNCH_K(NAME, KERNEL, KMOD)                                \
    extern "C" int NAME(rsl_cuda_stream* s, const void* w, const float* x,      \
                        float* out, int M, int N, int K) {                      \
        if (!s || !w || !x || !out || M <= 0 || N <= 0 || K <= 0 || (K % (KMOD))) \
            return -1;                                                          \
        if (!rsl_cuda_blackwell_tc_available(s->device)) return -2;            \
        cudaSetDevice(s->device);                                              \
        dim3 grid((M + 15) / 16, (N + 7) / 8);                                 \
        rslbw::KERNEL<<<grid, 32, 0, s->stream>>>(                             \
            (const unsigned char*)w, x, out, M, N, K);                          \
        return rsl_cuda_check("" #NAME);                                       \
    }
#define RSL_BW_GEMM_LAUNCH(NAME, KERNEL) RSL_BW_GEMM_LAUNCH_K(NAME, KERNEL, 64)

RSL_BW_GEMM_LAUNCH(rsl_cuda_gemm_nvfp4_tc_f32, rsl_bw_gemm_nvfp4_kernel)
RSL_BW_GEMM_LAUNCH(rsl_cuda_gemm_mxfp4_tc_f32, rsl_bw_gemm_mxfp4_kernel)
RSL_BW_GEMM_LAUNCH_K(rsl_cuda_gemm_mxfp8_tc_f32, rsl_bw_gemm_mxfp8_kernel, 32)
RSL_BW_GEMM_LAUNCH_K(rsl_cuda_gemm_mxfp6_tc_f32, rsl_bw_gemm_mxfp6_kernel, 32)
// Phase 4: TMA-staged NVFP4 variant (same contract + K%64 as the plain NVFP4).
RSL_BW_GEMM_LAUNCH(rsl_cuda_gemm_nvfp4_tc_tma_f32, rsl_bw_gemm_nvfp4_tma_kernel)
// Phase 5: 2:4 structured-sparse FP8 (dense E4M3 weight, pruned on-device).
RSL_BW_GEMM_LAUNCH(rsl_cuda_gemm_fp8_sp24_f32, rsl_bw_gemm_fp8_sp24_kernel)
