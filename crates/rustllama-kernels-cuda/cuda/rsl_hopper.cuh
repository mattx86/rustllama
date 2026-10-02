// ============================================================================
// rsl_hopper.cuh — NVIDIA Hopper sm_90a tensor-core GEMM (hand-rolled PTX).
// ============================================================================
//
// 4th-gen tensor cores on Grace-Hopper (GH200 / sm_90) via the warpgroup-level
// asynchronous MMA, `wgmma.mma_async`. This is the Hopper analogue of the
// Blackwell SM12x effort in rsl_blackwell.cuh — but Hopper and Blackwell speak
// *different* tensor-core dialects:
//
//   * Blackwell SM12x (sm_120a/121a): `mma.sync` — a WARP-collective MMA whose
//     A/B operands live in REGISTERS, extended with per-block scale factors
//     (NVFP4/MXFP4/MXFP8/MXFP6). See rsl_blackwell.cuh.
//   * Hopper (sm_90a): `wgmma.mma_async` — a WARPGROUP-collective (128-thread)
//     ASYNC MMA whose A/B operands come from SHARED MEMORY via 64-bit matrix
//     descriptors (or A from registers). There is NO per-block scale factor in
//     the instruction: Hopper wgmma does plain FP8 x FP8 -> f32. Block scaling
//     for a microscaling format (MXFP8) is therefore applied AROUND the wgmma,
//     in f32, by this kernel (see the K-loop below) — not by the hardware.
//
// Hopper has NO FP4 tensor cores (FP4/NVFP4 are Blackwell-only), so this file
// implements FP8 only (E4M3). NVFP4/MXFP4/MXFP6 on a GH200 stay on the scalar
// packed-matvec path; only the FP8 (MXFP8, W8A8) prefill GEMM is accelerated.
//
// CLEAN-ROOM: hand-rolled from the PTX ISA 8.x `wgmma.mma_async` section
// (operand order, FP8 shapes, the SS / register-D accumulator fragment) and
// the public shared-memory matrix-descriptor layout (CUTLASS `GmmaDescriptor`
// bit packing; Colfax Research's Hopper GEMM tutorial for the core-matrix TV
// layout). NO CUTLASS is vendored or linked — only its documented layouts are
// used as the correctness spec.
//
// ---------------------------------------------------------------------------
// WRITE-BLIND STATUS (read before trusting the numbers)
// ---------------------------------------------------------------------------
// This file was authored on a host with NO NVIDIA GPU of ANY kind, and the
// only NVIDIA hardware the project has access to is a Blackwell DGX Spark
// (sm_121) — NOT a Hopper. So this path is **compile-only, forever, for us**:
// its validation ceiling is "nvcc + ptxas accept `wgmma.mma_async` FP8 for
// `.target sm_90a`" (confirmed: a minimal `wgmma.m64n16k32.f32.e4m3.e4m3`
// snippet assembles for -gencode=arch=compute_90a,code=sm_90a). NOTHING here
// is runtime-validated: the numeric correctness of the wgmma accumulator
// fragment -> (row,col) map, the shared-memory matrix-descriptor field values
// (LBO/SBO/swizzle), and the MXFP8 block-scale application are all
// assumed-from-docs and tagged `GH200-VALIDATE`. The arbiter is a GH200 owner
// running `rustllama doctor --cuda-parity`; a mismatch there is a layout-index
// fix, NOT a toolchain problem. (The analogue of rsl_blackwell.cuh's
// `SPARK-VALIDATE`, renamed so nothing here is ever mistaken for validated.)
//
// ---------------------------------------------------------------------------
// The load-bearing instruction (confirmed to assemble for `.target sm_90a`):
//
//   wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3
//       {d0..d7}, desc-a, desc-b, scale-d, +1, +1;
//
//   * D   : 64x16 f32 accumulator, 8 regs/thread across the 128-thread
//           warpgroup (canonical wgmma m64nN fragment).
//   * A   : 64x32 FP8 (e4m3) WEIGHT tile, shared-memory descriptor (SS form).
//   * B   : 32x16 FP8 (e4m3) ACTIVATION tile, shared-memory descriptor.
//   * scale-d : a predicate (0 => D = A*B overwrite; 1 => D += A*B). We use 0
//               per K-step and accumulate manually in f32 so the MXFP8 per-32
//               block scales can be folded in (Hopper wgmma has no block scale).
//   * +1,+1 : imm-scale-a / imm-scale-b (operand sign scale; FP8 has no
//             imm-trans operands — TN layout only, A row-major / B col-major).
//
// W8A8: the f32 activation is dynamically quantized to E4M3 (per-token, per-32
// block E8M0 scale) before the MMA, same recipe as the MXFP8 weight encoder.
// That adds activation-quant error on top of the weight's, so the parity probe
// grades the TC path against the scalar W8A16 reference with a LOOSER tolerance
// (exactly as the Blackwell MXFP8 probe does).
// ============================================================================
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Real wgmma path compiles only when (a) build.rs targeted the Hopper
// architecture-accelerated arch (`-DRSL_HOPPER_TC`, i.e. an sm_90a token in
// RUSTLLAMA_CUDA_ARCHS) AND (b) this device pass is Hopper (__CUDA_ARCH__ ==
// 900 — the `a` suffix does NOT change __CUDA_ARCH__, it only unlocks the
// accelerated instructions at ptxas). Otherwise a scalar fallback body compiles
// so the default Ampere->Hopper build (and any plain sm_90) still builds; the
// host launchers additionally refuse to launch off a real Hopper device (see
// rsl_cuda_hopper_tc_available).
#if defined(RSL_HOPPER_TC) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
#define RSL_HOP_DEVICE_TC 1
#endif

namespace rslhop {

#ifdef RSL_HOP_DEVICE_TC
// ---------------------------------------------------------------------------
// Warpgroup MMA primitives (sm_90a). Hand-rolled from the PTX ISA; each matches
// the operand grouping the confirmed minimal snippet assembled with.
// ---------------------------------------------------------------------------

// 64-bit shared-memory matrix descriptor (CUTLASS GmmaDescriptor bit packing).
// Fields are byte offsets encoded >>4 (16-byte granularity):
//   bits 0-13  : start address of the tile in shared memory
//   bits 16-29 : LBO — leading-dimension byte offset (stride to the next core
//                matrix along the operand's contiguous direction)
//   bits 32-45 : SBO — stride-dimension byte offset (the other core-matrix axis)
//   bits 49-51 : matrix base offset (into a swizzle period; 0 here, tiles are
//                allocated at a 128-byte boundary)
//   bits 62-63 : swizzle mode (0 = none)
// GH200-VALIDATE: the LBO/SBO values the kernels pass below describe a
// no-swizzle, K-major core-matrix tiling taken from the public layout docs,
// not measured. Wrong values show up as MISCOMPUTE under --cuda-parity, a
// descriptor-arithmetic fix — the instruction still assembles regardless.
__device__ __forceinline__ uint64_t rsl_hop_make_desc(
    unsigned smem_addr, unsigned lbo, unsigned sbo, unsigned swizzle) {
    uint64_t d = 0;
    d |= ((uint64_t)((smem_addr & 0x3FFFFu) >> 4)) << 0;
    d |= ((uint64_t)((lbo & 0x3FFFFu) >> 4)) << 16;
    d |= ((uint64_t)((sbo & 0x3FFFFu) >> 4)) << 32;
    d |= ((uint64_t)(swizzle & 0x3u)) << 62;
    return d;
}

// Ordering/sync primitives bracketing an async wgmma group. `fence` publishes
// the prior register/shared writes to the async proxy; `commit` closes the
// group; `wait 0` blocks the warpgroup until the committed group retires, so
// the accumulator regs are safe to read.
__device__ __forceinline__ void rsl_hop_wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void rsl_hop_wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void rsl_hop_wgmma_wait0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// m64n16k32 FP8 E4M3xE4M3 -> f32, both operands shared-memory descriptors (SS).
// `scale_d` != 0 accumulates into D; == 0 overwrites. imm-scale-a/b pinned +1.
__device__ __forceinline__ void rsl_hop_wgmma_mxf8_e4m3(
    float d[8], uint64_t desc_a, uint64_t desc_b, int scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, %11, %12;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1));
}

// ---------------------------------------------------------------------------
// TMA (Tensor Memory Accelerator) async-copy helpers. cp.async.bulk + mbarrier
// were INTRODUCED on Hopper (sm_90) and carry over byte-for-byte to Blackwell;
// these mirror the identical wrappers in rsl_blackwell.cuh (which are compiled
// only for __CUDA_ARCH__ >= 1200, hence re-declared here under the Hopper gate
// rather than shared). Single-buffered copy -> wait -> compute; the
// double-buffered overlap that actually hides the latency is a perf follow-up.
// GH200-VALIDATE: the mbarrier phase/transaction-count semantics and bulk-copy
// alignment are assumed-from-docs.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void rsl_hop_mbar_init(unsigned mbar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(mbar), "r"(count) : "memory");
}
__device__ __forceinline__ void rsl_hop_mbar_expect(unsigned mbar, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(mbar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void rsl_hop_bulk_g2s(unsigned dst_smem, const void* src_g, int bytes, unsigned mbar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(dst_smem), "l"(src_g), "r"(bytes), "r"(mbar) : "memory");
}
__device__ __forceinline__ void rsl_hop_mbar_wait(unsigned mbar, int phase) {
    asm volatile(
        "{ .reg .pred p; L_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra L_%=; }\n"
        :: "r"(mbar), "r"(phase) : "memory");
}
#endif  // RSL_HOP_DEVICE_TC

// ---------------------------------------------------------------------------
// wgmma m64n16k32 accumulator fragment -> global (row m, col n) map.
// A warpgroup = 4 warps (threadIdx.x: warp = tid>>5, lane = tid&31). Warp w
// owns the 16 output rows [w*16, w*16+15]. Within a warp the 16x16 sub-tile is
// the m16n8 mma layout tiled over N/8 = 2 column groups; thread `lane` holds 8
// f32 regs r=0..7:
//   group = r>>2 (col block of 8), sub = r&3
//   row_local = (lane>>2) + ((sub>>1) ? 8 : 0)         [sub 0,1 -> +0; 2,3 -> +8]
//   col_local = group*8 + (lane&3)*2 + (sub&1)
//   m = m0 + w*16 + row_local ;  n = n0 + col_local
// GH200-VALIDATE: this is the documented wgmma fragment TV map, not measured.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int rsl_hop_frag_m(int m0, int warp, int lane, int r) {
    return m0 + warp * 16 + (lane >> 2) + (((r & 3) >> 1) ? 8 : 0);
}
__device__ __forceinline__ int rsl_hop_frag_n(int n0, int lane, int r) {
    return n0 + (r >> 2) * 8 + (lane & 3) * 2 + (r & 1);
}

// ===========================================================================
// MXFP8 (W8A8) Hopper wgmma GEMM: out[n*M + m] = sum_k W[m,k] * X[n,k], W in
// MXFP8 (33 B / 32: 32 E4M3 bytes + E8M0 scale byte), X f32 (quantized to E4M3
// on the fly, per-token per-32 block scale). One WARPGROUP (128 threads) per
// (64-row m) x (16-token n) output tile. Requires K % 32 == 0.
//
// MXFP8 block scaling on Hopper: wgmma has no block-scale operand, so the
// K-loop steps 32 (= exactly one MX block = one wgmma k32), issues the wgmma
// with scale_d = 0 (overwrite), then folds the per-row WEIGHT scale and
// per-token ACTIVATION scale into each accumulator register in f32 before
// adding to the running tile. Correct for MX microscaling; the per-register
// scale lookup relies on the fragment map above (GH200-VALIDATE).
// ===========================================================================
__global__ void rsl_hop_gemm_mxfp8_kernel(
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, int M, int N, int K) {
    const int m0 = blockIdx.x * 64;
    const int n0 = blockIdx.y * 16;
    const int tid = threadIdx.x;         // 0..127 (one warpgroup)
    const int warp = tid >> 5;           // 0..3
    const int lane = tid & 31;           // 0..31

#ifdef RSL_HOP_DEVICE_TC
    // A (weight) and B (activation) E4M3 tiles, 128-byte aligned so the
    // descriptor's 16-byte-granular start address + core-matrix strides are
    // well-defined. act scale is one E8M0-derived f32 per token for this block.
    __shared__ alignas(128) unsigned char as[64 * 32];
    __shared__ alignas(128) unsigned char bs[16 * 32];
    __shared__ float ascale[16];
    float c[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int k0 = 0; k0 < K; k0 += 32) {
        const int kb = k0 >> 5;
        // Stage the 64x32 weight tile (raw E4M3 bytes). 2048 B / 128 thr = 16 ea.
        for (int i = tid; i < 64 * 32; i += 128) {
            int r = i >> 5, col = i & 31, m = m0 + r;
            as[i] = (m < M) ? rslbw::rsl_mxfp8_byte(w, m, k0 + col, K) : 0;
        }
        // Per-token activation block scale (max over the 32 lanes / 448 E4M3max).
        if (tid < 16) {
            int n = n0 + tid;
            float mx = 0.f;
            if (n < N)
                for (int t = 0; t < 32; ++t) mx = fmaxf(mx, fabsf(x[(long long)n * K + k0 + t]));
            ascale[tid] = (mx > 0.f) ? mx / 448.0f : 1.0f;
        }
        __syncthreads();
        // Stage the 16x32 activation tile quantized to E4M3 (B, col operand).
        for (int i = tid; i < 16 * 32; i += 128) {
            int r = i >> 5, col = i & 31, n = n0 + r;
            float v = (n < N) ? x[(long long)n * K + k0 + col] : 0.0f;
            bs[i] = rslbw::rsl_f32_to_e4m3(v / ascale[r]);
        }
        __syncthreads();

        unsigned a_sa = (unsigned)__cvta_generic_to_shared(&as[0]);
        unsigned b_sa = (unsigned)__cvta_generic_to_shared(&bs[0]);
        // No-swizzle K-major core-matrix strides (16 B to the next K core col,
        // 256 B to the next 8-row group). GH200-VALIDATE.
        uint64_t desc_a = rsl_hop_make_desc(a_sa, 16, 256, 0);
        uint64_t desc_b = rsl_hop_make_desc(b_sa, 16, 256, 0);

        float draw[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        rsl_hop_wgmma_fence();
        rsl_hop_wgmma_mxf8_e4m3(draw, desc_a, desc_b, /*scale_d=*/0);
        rsl_hop_wgmma_commit();
        rsl_hop_wgmma_wait0();

        // Fold MX block scales (weight per-row, activation per-token) in f32.
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int m = rsl_hop_frag_m(m0, warp, lane, r);
            int n = rsl_hop_frag_n(n0, lane, r);
            float ws = (m < M) ? rsl_e8m0_to_f32(rslbw::rsl_mxfp8_scale(w, m, kb, K)) : 0.0f;
            c[r] += ws * ascale[n - n0] * draw[r];
        }
        __syncthreads();
    }

    // Store the 64x16 tile (col-major in M: out[n*M + m]), masking M/N edges.
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int m = rsl_hop_frag_m(m0, warp, lane, r);
        int n = rsl_hop_frag_n(n0, lane, r);
        if (m < M && n < N) out[(long long)n * M + m] = c[r];
    }
#else
    // -------- Scalar fallback (non-Hopper build / device pass) --------
    // Correct MXFP8(W) x f32(A) GEMM so the kernel is valid everywhere; only
    // ever launched if the host gate mis-fires (it normally refuses). The
    // 128-thread block strides over the 64x16 tile, one scalar dot per output.
    (void)warp; (void)lane;
    for (int idx = tid; idx < 64 * 16; idx += 128) {
        int rl = idx >> 4, cl = idx & 15, m = m0 + rl, n = n0 + cl;
        if (m >= M || n >= N) continue;
        float acc = 0.f;
        for (int kk = 0; kk < K; ++kk) {
            float sc = rsl_e8m0_to_f32(rslbw::rsl_mxfp8_scale(w, m, kk >> 5, K));
            acc += sc * rsl_e4m3_to_f32(rslbw::rsl_mxfp8_byte(w, m, kk, K)) * x[(long long)n * K + kk];
        }
        out[(long long)n * M + m] = acc;
    }
#endif
}

// ===========================================================================
// TMA-staged MXFP8 wgmma GEMM — identical math to rsl_hop_gemm_mxfp8_kernel;
// only the activation staging differs: the 16x32 f32 activation tile is pulled
// in with cp.async.bulk + an mbarrier (one bulk copy per token row = 32
// contiguous f32 = 128 B) instead of a per-thread global-load loop, then
// quantized to E4M3 in shared memory. Reuses the Hopper-born TMA machinery
// (identical to rsl_blackwell.cuh). Single-buffered. GH200-VALIDATE.
// ===========================================================================
__global__ void rsl_hop_gemm_mxfp8_tma_kernel(
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, int M, int N, int K) {
    const int m0 = blockIdx.x * 64;
    const int n0 = blockIdx.y * 16;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

#ifdef RSL_HOP_DEVICE_TC
    __shared__ alignas(128) unsigned char as[64 * 32];
    __shared__ alignas(128) unsigned char bs[16 * 32];
    __shared__ alignas(16) float xs[16][32];    // f32 activation staging (TMA dst)
    __shared__ float ascale[16];
    __shared__ alignas(8) unsigned long long mbar;
    unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);
    if (tid == 0) rsl_hop_mbar_init(mb, 1);
    __syncthreads();
    float c[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int phase = 0;

    for (int k0 = 0; k0 < K; k0 += 32) {
        const int kb = k0 >> 5;
        int valid = N - n0; if (valid > 16) valid = 16; if (valid < 0) valid = 0;
        // Bulk-copy `valid` token rows (32 f32 = 128 B each) into xs via TMA.
        if (tid == 0) {
            rsl_hop_mbar_expect(mb, valid * 32 * (int)sizeof(float));
            for (int r = 0; r < valid; ++r) {
                unsigned dst = (unsigned)__cvta_generic_to_shared(&xs[r][0]);
                rsl_hop_bulk_g2s(dst, &x[(long long)(n0 + r) * K + k0], 32 * (int)sizeof(float), mb);
            }
        }
        // Zero the edge rows not covered by a bulk copy so quant is defined.
        for (int idx = tid; idx < (16 - valid) * 32; idx += 128) {
            int r = valid + (idx >> 5);
            xs[r][idx & 31] = 0.f;
        }
        rsl_hop_mbar_wait(mb, phase); phase ^= 1;
        __syncthreads();

        // Stage weights (direct global->smem, same as the non-TMA kernel).
        for (int i = tid; i < 64 * 32; i += 128) {
            int r = i >> 5, col = i & 31, m = m0 + r;
            as[i] = (m < M) ? rslbw::rsl_mxfp8_byte(w, m, k0 + col, K) : 0;
        }
        // Per-token activation block scale from the TMA-staged f32 tile.
        if (tid < 16) {
            float mx = 0.f;
            for (int t = 0; t < 32; ++t) mx = fmaxf(mx, fabsf(xs[tid][t]));
            ascale[tid] = (mx > 0.f) ? mx / 448.0f : 1.0f;
        }
        __syncthreads();
        for (int i = tid; i < 16 * 32; i += 128) {
            int r = i >> 5, col = i & 31;
            bs[i] = rslbw::rsl_f32_to_e4m3(xs[r][col] / ascale[r]);
        }
        __syncthreads();

        unsigned a_sa = (unsigned)__cvta_generic_to_shared(&as[0]);
        unsigned b_sa = (unsigned)__cvta_generic_to_shared(&bs[0]);
        uint64_t desc_a = rsl_hop_make_desc(a_sa, 16, 256, 0);
        uint64_t desc_b = rsl_hop_make_desc(b_sa, 16, 256, 0);

        float draw[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        rsl_hop_wgmma_fence();
        rsl_hop_wgmma_mxf8_e4m3(draw, desc_a, desc_b, /*scale_d=*/0);
        rsl_hop_wgmma_commit();
        rsl_hop_wgmma_wait0();

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int m = rsl_hop_frag_m(m0, warp, lane, r);
            int n = rsl_hop_frag_n(n0, lane, r);
            float ws = (m < M) ? rsl_e8m0_to_f32(rslbw::rsl_mxfp8_scale(w, m, kb, K)) : 0.0f;
            c[r] += ws * ascale[n - n0] * draw[r];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int m = rsl_hop_frag_m(m0, warp, lane, r);
        int n = rsl_hop_frag_n(n0, lane, r);
        if (m < M && n < N) out[(long long)n * M + m] = c[r];
    }
#else
    (void)warp; (void)lane;
    for (int idx = tid; idx < 64 * 16; idx += 128) {
        int rl = idx >> 4, cl = idx & 15, m = m0 + rl, n = n0 + cl;
        if (m >= M || n >= N) continue;
        float acc = 0.f;
        for (int kk = 0; kk < K; ++kk) {
            float sc = rsl_e8m0_to_f32(rslbw::rsl_mxfp8_scale(w, m, kk >> 5, K));
            acc += sc * rsl_e4m3_to_f32(rslbw::rsl_mxfp8_byte(w, m, kk, K)) * x[(long long)n * K + kk];
        }
        out[(long long)n * M + m] = acc;
    }
#endif
}

}  // namespace rslhop

// ---------------------------------------------------------------------------
// Host entry points (extern "C"). Gate on a real Hopper device (cudaDeviceProp
// major == 9) + compiled TC support; otherwise return -2 ("unavailable") so the
// Rust caller falls back to the scalar packed-matvec path. K % 32 == 0 required.
// ---------------------------------------------------------------------------

// 1 if this build compiled the wgmma path AND device `dev` is Hopper (sm_90x).
extern "C" int rsl_cuda_hopper_tc_available(int dev) {
#if defined(RSL_HOPPER_TC)
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return 0;
    return (p.major == 9) ? 1 : 0;  // Hopper sm_90 / sm_90a (GH200, H100)
#else
    (void)dev;
    return 0;
#endif
}

// One warpgroup (128 threads) per 64x16 output tile; K % 32 == 0.
#define RSL_HOP_GEMM_LAUNCH(NAME, KERNEL)                                       \
    extern "C" int NAME(rsl_cuda_stream* s, const void* w, const float* x,      \
                        float* out, int M, int N, int K) {                      \
        if (!s || !w || !x || !out || M <= 0 || N <= 0 || K <= 0 || (K % 32))   \
            return -1;                                                          \
        if (!rsl_cuda_hopper_tc_available(s->device)) return -2;               \
        cudaSetDevice(s->device);                                              \
        dim3 grid((M + 63) / 64, (N + 15) / 16);                               \
        rslhop::KERNEL<<<grid, 128, 0, s->stream>>>(                           \
            (const unsigned char*)w, x, out, M, N, K);                          \
        return rsl_cuda_check("" #NAME);                                       \
    }

RSL_HOP_GEMM_LAUNCH(rsl_cuda_gemm_mxfp8_wgmma_f32, rsl_hop_gemm_mxfp8_kernel)
RSL_HOP_GEMM_LAUNCH(rsl_cuda_gemm_mxfp8_wgmma_tma_f32, rsl_hop_gemm_mxfp8_tma_kernel)
