// ============================================================================
// rsl_xmx.hpp — Intel Xe-Matrix-Extensions (XMX / DPAS) joint_matrix GEMM.
// ============================================================================
//
// The Intel analogue of the CUDA Blackwell/Hopper tensor-core GEMMs: a
// bf16 x bf16 -> f32 matrix-multiply built on SYCL's portable
// `sycl::ext::oneapi::experimental::matrix::joint_matrix` API, which lowers to
// the DPAS (Dot-Product-Accumulate-Systolic) / XMX units on Xe-HPG (Arc) and
// Xe-HPC (Ponte Vecchio / Data Center GPU Max) hardware.
//
// Role: accelerates the PREFILL / batched matmul of a quantized weight. The
// dispatch layer (accel.rs) dequantizes the packed weight into an f32 tile and
// supplies the f32 activation rows; this kernel narrows both operands to bf16
// and accumulates the product in f32 on the XMX systolic array. Decode-time
// GEMV keeps the existing software-decode SYCL path (tensor cores need the
// matrix shape). Like every GGUF quant, bf16 is a lateral-precision step: the
// dequant->bf16 narrowing is the only added error, graded with a loose
// tolerance in the parity probe.
//
// ---------------------------------------------------------------------------
// WRITE-BLIND STATUS (read before trusting the numbers)
// ---------------------------------------------------------------------------
// Authored on a host whose only GPU is an Intel Iris Xe (Xe-LP) — which has NO
// XMX units — so this path is compile-only here: the ceiling is "icpx accepts
// the joint_matrix GEMM for an XMX-capable target". NOTHING is runtime-
// validated: the tile shape selection, the SLM staging + sub-group geometry,
// and the joint_matrix row/col mapping are taken from the oneAPI
// `sycl_ext_oneapi_matrix` spec + Intel's joint_matrix examples, not measured.
// Everything assumed-from-spec is tagged ARC-VALIDATE. The arbiter is an Arc /
// PVC owner running `doctor --sycl-parity` (the bf16 XMX GEMM vs the CPU f32
// reference); a mismatch there is a layout-index fix, not a build problem.
//
// Gated on RSL_SYCL_XMX (set by build.rs when an XMX-capable target is built)
// AND, at runtime, on DeviceInfo.xmx_capable (device aspect query). Otherwise
// the caller falls back to the existing software-decode SYCL matvec, so the
// Iris Xe / any non-XMX device build + behavior is unchanged.
// ============================================================================
#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

// The matrix extension is experimental; the namespace path is stable across
// oneAPI 2024+ (sycl_ext_oneapi_matrix). Only pulled in under the XMX gate so
// a non-XMX build never depends on the experimental header surface.
#ifdef RSL_SYCL_XMX
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#endif

namespace rslxmx {

// XMX bf16 systolic tile. Intel DPAS exposes M in {1..8}, N = execution-width
// (16 on Xe-HPG/HPC), K = 16 for bf16 (32 bytes / the 2-byte element). We tile
// the output in TM x TN blocks, each a warp/sub-group-cooperative joint_matrix
// MAD looped over K in TK steps. ARC-VALIDATE: these are the documented bf16
// DPAS tile dims; the actual combination a given device supports is queried via
// joint_matrix's `matrix_combinations` at runtime by the capability probe.
static constexpr int RSL_XMX_TM = 8;
static constexpr int RSL_XMX_TN = 16;
static constexpr int RSL_XMX_TK = 16;
static constexpr int RSL_XMX_SG = 16;  // sub-group size (Xe exec width)

#ifdef RSL_SYCL_XMX

namespace syclex = sycl::ext::oneapi::experimental::matrix;

// Narrow an f32 lane to bf16 (round-to-nearest-even via the truncating cast the
// bf16 ctor performs). Shared by the SLM staging loops below.
static inline sycl::ext::oneapi::bfloat16 rsl_f32_to_bf16(float v) {
    return sycl::ext::oneapi::bfloat16(v);
}

// bf16 XMX GEMM: out[n*M + m] = sum_k W[m*K + k] * X[n*K + k], W and X f32 in
// global memory (the dequanted weight + the activation rows), out f32. One
// sub-group (SG lanes) cooperates on a TM x TN output tile; the K loop issues
// joint_matrix_mad over TK-wide bf16 atoms, staging both operands through SLM
// so the loads are the contiguous layout joint_matrix_load expects.
//
// A operand = W tile (TM x TK), row-major (`layout::row_major`).
// B operand = X tile (TK x TN): X is row-major [n,k], and the B atom wants
// K-major, so the tile is transposed into SLM as [k,n] (ARC-VALIDATE: the
// packed_b / col-major staging is the fiddliest bit — the parity probe is the
// arbiter).
inline void rsl_xmx_gemm_bf16_f32(sycl::queue& q, const float* W, const float* X,
                                  float* out, int M, int N, int K) {
    const int tiles_m = (M + RSL_XMX_TM - 1) / RSL_XMX_TM;
    const int tiles_n = (N + RSL_XMX_TN - 1) / RSL_XMX_TN;

    q.submit([&](sycl::handler& h) {
        // SLM: one A tile (TM x TK) + one B tile (TK x TN) in bf16 per work-group.
        sycl::local_accessor<sycl::ext::oneapi::bfloat16, 1> as(
            sycl::range<1>(RSL_XMX_TM * RSL_XMX_TK), h);
        sycl::local_accessor<sycl::ext::oneapi::bfloat16, 1> bs(
            sycl::range<1>(RSL_XMX_TK * RSL_XMX_TN), h);
        // f32 scratch for the accumulator store — joint_matrix_store needs an
        // f32 pointer and we can't reinterpret the bf16 A tile to float*.
        sycl::local_accessor<float, 1> cs(
            sycl::range<1>(RSL_XMX_TM * RSL_XMX_TN), h);

        // One work-group == one sub-group (SG lanes) per (tile_m, tile_n).
        sycl::nd_range<2> ndr(
            sycl::range<2>(static_cast<size_t>(tiles_m), static_cast<size_t>(tiles_n) * RSL_XMX_SG),
            sycl::range<2>(1, RSL_XMX_SG));

        h.parallel_for(ndr, [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(RSL_XMX_SG)]] {
            const int tm = static_cast<int>(it.get_group(0));
            const int tn = static_cast<int>(it.get_group(1));
            const int m0 = tm * RSL_XMX_TM;
            const int n0 = tn * RSL_XMX_TN;
            const int lane = static_cast<int>(it.get_local_id(1));
            auto sg = it.get_sub_group();

            auto A = as.get_pointer();
            auto B = bs.get_pointer();

            syclex::joint_matrix<sycl::sub_group, sycl::ext::oneapi::bfloat16,
                                 syclex::use::a, RSL_XMX_TM, RSL_XMX_TK,
                                 syclex::layout::row_major> a_frag;
            syclex::joint_matrix<sycl::sub_group, sycl::ext::oneapi::bfloat16,
                                 syclex::use::b, RSL_XMX_TK, RSL_XMX_TN,
                                 syclex::layout::row_major> b_frag;
            syclex::joint_matrix<sycl::sub_group, float, syclex::use::accumulator,
                                 RSL_XMX_TM, RSL_XMX_TN> c_frag;
            syclex::joint_matrix_fill(sg, c_frag, 0.0f);

            for (int k0 = 0; k0 < K; k0 += RSL_XMX_TK) {
                // Stage A (TM x TK) from W[m,k] -> bf16 SLM, row-major. SG lanes
                // cover the TM*TK elements; zero-pad the M/K tails.
                for (int i = lane; i < RSL_XMX_TM * RSL_XMX_TK; i += RSL_XMX_SG) {
                    int r = i / RSL_XMX_TK, c = i % RSL_XMX_TK;
                    int m = m0 + r, k = k0 + c;
                    A[i] = (m < M && k < K) ? rsl_f32_to_bf16(W[(long long)m * K + k])
                                            : sycl::ext::oneapi::bfloat16(0.0f);
                }
                // Stage B (TK x TN) from X[n,k] -> bf16 SLM as [k,n] (row-major
                // over the atom's K x N shape). X is [n,k] row-major, so this
                // transposes the tile. ARC-VALIDATE.
                for (int i = lane; i < RSL_XMX_TK * RSL_XMX_TN; i += RSL_XMX_SG) {
                    int kk = i / RSL_XMX_TN, nn = i % RSL_XMX_TN;
                    int k = k0 + kk, n = n0 + nn;
                    B[i] = (n < N && k < K) ? rsl_f32_to_bf16(X[(long long)n * K + k])
                                            : sycl::ext::oneapi::bfloat16(0.0f);
                }
                sycl::group_barrier(sg);

                syclex::joint_matrix_load(sg, a_frag, A, RSL_XMX_TK);
                syclex::joint_matrix_load(sg, b_frag, B, RSL_XMX_TN);
                // oneAPI 2026.0 signature: joint_matrix_mad(group, D, A, B, C)
                // fills D = A*B + C in place (void return). D and C are the SAME
                // accumulator (layout::dynamic) so it accumulates across the K loop.
                syclex::joint_matrix_mad(sg, c_frag, a_frag, b_frag, c_frag);
                sycl::group_barrier(sg);
            }

            // Store the TM x TN accumulator to out[n*M + m] (column-major in M,
            // matching the CUDA TC GEMMs' out layout). joint_matrix_store writes
            // row-major [TM,TN] into a scratch; re-scatter to the strided out.
            // ARC-VALIDATE: the store->(m,n) mapping.
            auto C = cs.get_pointer();
            syclex::joint_matrix_store(sg, c_frag, C, RSL_XMX_TN,
                                       syclex::layout::row_major);
            sycl::group_barrier(sg);
            for (int i = lane; i < RSL_XMX_TM * RSL_XMX_TN; i += RSL_XMX_SG) {
                int r = i / RSL_XMX_TN, c = i % RSL_XMX_TN;
                int m = m0 + r, n = n0 + c;
                if (m < M && n < N) out[(long long)n * M + m] = C[i];
            }
        });
    });
}

#endif  // RSL_SYCL_XMX

// Runtime capability probe: does `dev` expose a bf16 XMX/DPAS joint_matrix
// combination? Queries the device's matrix combinations (the ext_oneapi_matrix
// aspect/arch). Returns false on any non-XMX device (Iris Xe Xe-LP, OpenCL
// fallback) so the dispatch uses the software path. Compiled unconditionally so
// the host side can gate even in a non-XMX build (returns false there).
inline bool rsl_xmx_device_capable(const sycl::device& dev) {
#ifdef RSL_SYCL_XMX
    // Xe-HPG (Arc) / Xe-HPC (PVC) expose the matrix aspect; Xe-LP (Iris Xe)
    // does not. ARC-VALIDATE: aspect name across oneAPI versions.
    try {
        return dev.has(sycl::aspect::ext_intel_matrix);
    } catch (...) {
        return false;
    }
#else
    (void)dev;
    return false;
#endif
}

}  // namespace rslxmx

// ---------------------------------------------------------------------------
// Host entry (extern "C", exported via rsl_kernels.def). Included by
// rsl_kernels.cpp AFTER `struct rsl_stream`, so `s->q` is the SYCL queue.
// Contract mirrors the CUDA TC GEMMs: out[n*M + m] = sum_k W[m,k]*X[n,k], W/X
// f32 (the dequanted weight + activation rows), out f32. Returns -2 when the
// build lacks the XMX path (RSL_SYCL_XMX undefined) so the Rust caller falls
// back to the software-decode SYCL matvec; -1 bad args; -3 SYCL exception.
// WRITE-BLIND: compile-only — runtime-UNVALIDATED (no XMX GPU here).
// ---------------------------------------------------------------------------
extern "C" int rsl_sycl_gemm_bf16_xmx_f32(rsl_stream* s, const float* w,
                                          const float* x, float* out,
                                          int m, int k, int n) {
#ifdef RSL_SYCL_XMX
    if (!s || !w || !x || !out || m <= 0 || k <= 0 || n <= 0) return -1;
    try {
        rslxmx::rsl_xmx_gemm_bf16_f32(s->q, w, x, out, m, k, n);
        s->q.wait();
        return 0;
    } catch (const sycl::exception&) {
        return -3;
    }
#else
    (void)s; (void)w; (void)x; (void)out; (void)m; (void)k; (void)n;
    return -2;
#endif
}

// Runtime capability probe exposed to the Rust host: does the stream's device
// expose the bf16 XMX/DPAS joint_matrix combination? Returns 1 = capable,
// 0 = not (Iris Xe Xe-LP / OpenCL fallback / non-XMX build / null stream).
// Lets the accel dispatch skip the XMX launch (and its f32 weight dequant)
// entirely on a device that would only fault/fall back. Compiled
// unconditionally — in a non-XMX build rsl_xmx_device_capable() returns false.
extern "C" int rsl_sycl_xmx_available(rsl_stream* s) {
    if (!s) return 0;
    try {
        return rslxmx::rsl_xmx_device_capable(s->q.get_device()) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

