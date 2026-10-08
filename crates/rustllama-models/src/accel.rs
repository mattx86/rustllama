//! Per-thread SYCL accelerator hook for the forward pass.
//!
//! The forward pass in [`crate::llama_arch`] calls into pure-Rust
//! CPU kernels by default. When a SYCL stream is installed on the
//! current thread via [`SyclStreamGuard`], specific dispatch points
//! consult [`try_rmsnorm_f32`] (and friends, as they get added) and
//! route the op to the GPU. The kernel call returns `false` on any
//! failure — `Unavailable` (mock build / no GPU), `InvalidShape`,
//! or runtime errors — so the caller transparently falls back to
//! the CPU kernel.
//!
//! Why TLS instead of threading through `forward_one`:
//! - `forward_one` has 7+ call sites in `rustllama-engine::cpu`;
//!   adding a parameter touches every one.
//! - The SYCL `Stream` is `!Send + !Sync` (SYCL keeps thread-local
//!   state in its runtime), so per-thread installation matches the
//!   underlying constraint exactly.
//! - Each generation call runs inside its own `spawn_blocking` task
//!   on a tokio worker thread; the engine creates a stream at the
//!   start, installs it for the duration of the request, and the
//!   guard's `Drop` impl tears it down on the way out.
//!
//! Gating: today only [`try_rmsnorm_f32`] is implemented end-to-end.
//! The other kernels (rope, silu, softmax, gemm) have parity checks
//! in `rustllama-engine::sycl_accel` but the forward-pass dispatch
//! lands one kernel at a time so each can be validated independently
//! on the user's hardware.

use half::f16;
use rustllama_kernels_cuda as ck;
use rustllama_kernels_mlx as mk;
use rustllama_kernels_sycl as sk;
use rustllama_tensor::{as_bytes, Dtype, Tensor};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// ============================================================
// Native CUDA packed-matvec dispatch (NVIDIA backend)
// ============================================================
//
// The CUDA analogue of the SYCL USM matvec ladder below. When a CUDA
// device is present the engine flips the per-thread `CUDA_ACTIVE` flag
// and the packed-matvec dispatch points route the four CUDA-supported
// kinds (PTQ1_0/Q8_0/Q4_K/Q6_K) here FIRST — taking precedence over
// SYCL, matching the user's "gated on CUDA present, with or without a
// SYCL device" choice. Everything is inert when no CUDA device exists:
// [`cuda_cache`] returns `None` (never constructed) and every CUDA
// attempt short-circuits to `false`, so the proven SYCL/CPU path is
// bit-for-bit unchanged on non-NVIDIA hosts (e.g. this Intel box).
//
// Unlike SYCL's host-accessible shared USM, CUDA device memory is not
// host-readable, so weights can't live in `Storage`; the cache keeps
// its own device-resident weight table (uploaded once, keyed by the
// GGUF host pointer). It's process-wide behind a mutex — CUDA stream
// ops serialize, so concurrent requests are correct if slower.

/// Whether the native CUDA packed-matvec backend is active: a usable
/// CUDA device exists (and the env off-switch isn't set). No per-thread
/// opt-in — the cache is process-wide and thread-agnostic, so this is
/// simply "is there a CUDA device to dispatch to". Inert (false, cached)
/// on non-NVIDIA hosts, so no engine wiring is needed and the SYCL/CPU
/// path is untouched there.
#[inline]
pub fn cuda_active() -> bool {
    cuda_cache().is_some()
}

/// Whether a native CUDA compute device is available for dispatch.
/// Public mirror of [`cuda_active`] for operator-facing reporting.
pub fn cuda_available() -> bool {
    cuda_cache().is_some()
}

/// The process-wide device-resident CUDA matvec cache. Lazily created
/// once: `Some` only when an NVIDIA device is visible, the env
/// off-switch (`RUSTLLAMA_CUDA_DISPATCH=0`) isn't set, and a stream +
/// budget could be established; `None` (cached) otherwise. Budget =
/// 85% of device 0's total memory, matching the SYCL VRAM-fit reserve.
fn cuda_cache() -> Option<&'static Mutex<ck::CudaMatvecCache>> {
    static CACHE: OnceLock<Option<Mutex<ck::CudaMatvecCache>>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            // Env off-switch for A/B (mirrors RUSTLLAMA_SYCL_DISPATCH).
            if std::env::var("RUSTLLAMA_CUDA_DISPATCH")
                .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
                .unwrap_or(false)
            {
                return None;
            }
            if ck::device_count() == 0 {
                return None;
            }
            let budget = ck::device_info(0)
                .map(|i| ((i.total_mem_bytes as f64) * 0.85) as usize)
                .unwrap_or(0);
            if budget == 0 {
                return None;
            }
            ck::CudaMatvecCache::new(0, budget).map(Mutex::new)
        })
        .as_ref()
}

/// Map a weight [`Dtype`] to the CUDA cache's packed kind, or `None`
/// for dtypes the CUDA backend doesn't dispatch (non-packed F16/BF16
/// and NVFP4's non-`Raw` markers) — those fall through to the SYCL/CPU
/// ladder. Every packed-quant `*Raw` dtype the loader produces now has
/// a CUDA arm (the kernel surface + `CudaMatvecCache` cover all of
/// them; `k_alignment`/`row_bytes` on `CudaPackedKind` gate each shape).
fn dtype_to_cuda_kind(dtype: Dtype) -> Option<ck::CudaPackedKind> {
    match dtype {
        Dtype::PTQ1_0Raw => Some(ck::CudaPackedKind::Ptq1_0),
        Dtype::Q8_0Raw => Some(ck::CudaPackedKind::Q8_0),
        Dtype::Q4_KRaw => Some(ck::CudaPackedKind::Q4_K),
        Dtype::Q6_KRaw => Some(ck::CudaPackedKind::Q6_K),
        Dtype::Q5_KRaw => Some(ck::CudaPackedKind::Q5_K),
        Dtype::Q2_KRaw => Some(ck::CudaPackedKind::Q2_K),
        Dtype::Q8_KRaw => Some(ck::CudaPackedKind::Q8_K),
        Dtype::Q4_0Raw => Some(ck::CudaPackedKind::Q4_0),
        Dtype::Q5_0Raw => Some(ck::CudaPackedKind::Q5_0),
        Dtype::Q4_1Raw => Some(ck::CudaPackedKind::Q4_1),
        Dtype::Q5_1Raw => Some(ck::CudaPackedKind::Q5_1),
        Dtype::IQ4_NLRaw => Some(ck::CudaPackedKind::Iq4_Nl),
        Dtype::IQ4_XSRaw => Some(ck::CudaPackedKind::Iq4_Xs),
        Dtype::IQ2_XXSRaw => Some(ck::CudaPackedKind::Iq2_Xxs),
        Dtype::IQ2_XSRaw => Some(ck::CudaPackedKind::Iq2_Xs),
        Dtype::IQ2_SRaw => Some(ck::CudaPackedKind::Iq2_S),
        Dtype::IQ3_XXSRaw => Some(ck::CudaPackedKind::Iq3_Xxs),
        Dtype::IQ3_SRaw => Some(ck::CudaPackedKind::Iq3_S),
        Dtype::IQ1_SRaw => Some(ck::CudaPackedKind::Iq1_S),
        Dtype::IQ1_MRaw => Some(ck::CudaPackedKind::Iq1_M),
        Dtype::Nvfp4Raw => Some(ck::CudaPackedKind::Nvfp4),
        Dtype::Q3_KRaw => Some(ck::CudaPackedKind::Q3_K),
        Dtype::PQ2_0Raw => Some(ck::CudaPackedKind::Pq2_0),
        Dtype::Mxfp4Raw => Some(ck::CudaPackedKind::Mxfp4),
        Dtype::Mxfp6Raw => Some(ck::CudaPackedKind::Mxfp6),
        Dtype::Mxfp8Raw => Some(ck::CudaPackedKind::Mxfp8),
        _ => None,
    }
}

/// Single-row packed matvec via the native CUDA backend. Returns
/// `false` on any miss (no device, over budget, kernel failure) so the
/// caller falls through to the SYCL/CPU ladder; leaves `out` untouched
/// on failure.
fn try_matvec_packed_cuda(
    kind: ck::CudaPackedKind,
    weight_key: usize,
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) -> bool {
    let Some(cache) = cuda_cache() else {
        return false;
    };
    let Ok(mut guard) = cache.lock() else {
        return false;
    };
    guard.matvec_packed(kind, weight_key, w_bytes, x, out, m, k)
}

/// Batched (`n`-row) packed matvec via the native CUDA backend.
#[allow(clippy::too_many_arguments)]
fn try_matvec_packed_cuda_batched(
    kind: ck::CudaPackedKind,
    weight_key: usize,
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let Some(cache) = cuda_cache() else {
        return false;
    };
    let Ok(mut guard) = cache.lock() else {
        return false;
    };
    guard.matvec_packed_batched(kind, weight_key, w_bytes, x, out, m, k, n)
}

/// Batched Q4_K matvec via the native CUDA prefill GEMM (verdict `gemm:q4_k_f32`,
/// bit-exact). Weights are the SAME packed Q4_K bytes; the kernel stages a
/// dequantized weight super-block in shared memory and reuses it across the
/// token tile (the weight-reuse fix for the re-streaming batched matvec).
#[allow(clippy::too_many_arguments)]
fn try_gemm_q4k_f32_batched(
    weight_key: usize,
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let Some(cache) = cuda_cache() else {
        return false;
    };
    let Ok(mut guard) = cache.lock() else {
        return false;
    };
    guard.gemm_packed_batched_q4k_f32(weight_key, w_bytes, x, out, m, k, n)
}

/// Batched Q4_K matvec via the native CUDA int8 tensor-core GEMM (verdict
/// `gemm:q4_k_w8a8_tc`, W8A8/lossy). Weights are the SAME packed Q4_K bytes.
#[allow(clippy::too_many_arguments)]
fn try_gemm_q4k_w8a8_tc_batched(
    weight_key: usize,
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let Some(cache) = cuda_cache() else {
        return false;
    };
    let Ok(mut guard) = cache.lock() else {
        return false;
    };
    guard.gemm_packed_batched_q4k_w8a8_tc(weight_key, w_bytes, x, out, m, k, n)
}

// ------------------------------------------------------------
// Blackwell SM12x FP4 tensor-core GEMM dispatch (opt-in)
// ------------------------------------------------------------

/// Map a weight [`Dtype`] to the Blackwell FP4 tensor-core kind, or `None`
/// (the dtype has no FP4 TC path — fall through to the scalar dispatch).
fn dtype_to_fp4_tc_kind(dtype: Dtype) -> Option<ck::CudaFp4TcKind> {
    match dtype {
        Dtype::Nvfp4Raw => Some(ck::CudaFp4TcKind::Nvfp4),
        Dtype::Mxfp4Raw => Some(ck::CudaFp4TcKind::Mxfp4),
        _ => None,
    }
}

// ------------------------------------------------------------
// On-device GPU-kernel validation verdicts (auto-enable, no env gate)
// ------------------------------------------------------------

/// Per-process GPU-kernel validation verdicts, loaded once from the
/// autotuner cache (populated by `tune --validate-kernels`). Maps a
/// parity-probe name (`"gemm:mxfp4_tc"`, `"xmx:gemm"`,
/// `"attn:decode_mxfp8"`, …) to whether that kernel matched its CPU
/// reference ON THIS machine. Mirrors [`load_lws_cache_for_device_once`].
/// Empty when the cache is missing or the validation stage hasn't run, so
/// every [`kernel_verdict`] lookup then fails CLOSED.
fn load_kernel_verdicts_once() -> &'static std::collections::HashMap<String, bool> {
    static CACHE: std::sync::OnceLock<std::collections::HashMap<String, bool>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        let key = rustllama_tuner::system_fingerprint();
        let Some(dir) = rustllama_tuner::default_cache_dir() else {
            return std::collections::HashMap::new();
        };
        match rustllama_tuner::load_cache(&dir, &key) {
            Ok(Some(t)) => t.kernel_verdicts,
            _ => std::collections::HashMap::new(),
        }
    })
}

/// On-device verdict for a specialized GPU kernel: `true` only when the
/// `tune --validate-kernels` stage confirmed it matches the CPU reference
/// on THIS machine. FAIL-CLOSED — a kernel never validated (or whose probe
/// SKIPped because the device lacks the capability) returns `false`, so the
/// specialized path stays off and the base path runs. This replaces the
/// removed `RUSTLLAMA_FP4_TC` / `_FP8_WGMMA` / `_SYCL_XMX` env gates: the
/// verdict IS the automatic on/off decision.
pub(crate) fn kernel_verdict(name: &str) -> bool {
    load_kernel_verdicts_once()
        .get(name)
        .copied()
        .unwrap_or(false)
}

/// Whether the Blackwell FP4 tensor-core GEMM path is enabled. **AUTOMATIC** —
/// no env var: it turns on only when BOTH FP4 TC GEMM kernels (NVFP4 + MXFP4)
/// passed the on-device self-check (`tune --validate-kernels` → `kernel_verdict`)
/// on this machine, a usable NVIDIA GPU is present (`cuda_active()`), and the
/// device is a TC-capable Blackwell (`ck::blackwell_tc_available`). Fail-closed:
/// if either FP4 kernel hasn't been validated or miscomputed, the whole FP4 TC
/// path stays off and the scalar path runs. Cached once (process-wide).
fn fp4_tc_enabled() -> bool {
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| {
        kernel_verdict(rustllama_tuner::VERDICT_GEMM_NVFP4_TC)
            && kernel_verdict(rustllama_tuner::VERDICT_GEMM_MXFP4_TC)
            && cuda_active()
            && ck::blackwell_tc_available(0)
    })
}

/// Batched FP4 tensor-core GEMM via the CUDA cache. `false` on any miss so the
/// caller falls through to the scalar CUDA/SYCL/CPU ladder (untouched `out`).
#[allow(clippy::too_many_arguments)]
fn try_gemm_fp4_tc_batched(
    kind: ck::CudaFp4TcKind,
    weight_key: usize,
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let Some(cache) = cuda_cache() else {
        return false;
    };
    let Ok(mut guard) = cache.lock() else {
        return false;
    };
    guard.gemm_fp4_tc(kind, weight_key, w_bytes, x, out, m, k, n)
}

/// Whether the Blackwell MXFP8 (W8A8) block-scaled tensor-core GEMM path is
/// enabled. **AUTOMATIC** — on only when the MXFP8-TC kernel passed the
/// on-device self-check (`kernel_verdict`), a usable NVIDIA GPU is present, and
/// the device is a TC-capable Blackwell. Fail-closed; cached once. The sibling
/// of [`fp4_tc_enabled`] for the 8-bit block-scaled kernel, which was validated
/// + probed but previously never wired into dispatch.
fn mxfp8_tc_enabled() -> bool {
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| {
        kernel_verdict(rustllama_tuner::VERDICT_GEMM_MXFP8_TC)
            && cuda_active()
            && ck::blackwell_tc_available(0)
    })
}

/// Whether the Blackwell MXFP6 (W6A6) block-scaled tensor-core GEMM path is
/// enabled. As [`mxfp8_tc_enabled`] for the MXFP6 kernel verdict.
fn mxfp6_tc_enabled() -> bool {
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| {
        kernel_verdict(rustllama_tuner::VERDICT_GEMM_MXFP6_TC)
            && cuda_active()
            && ck::blackwell_tc_available(0)
    })
}

/// Batched Blackwell MXFP8 tensor-core GEMM via the CUDA cache. `false` on any
/// miss so the caller falls through to the scalar ladder (untouched `out`).
#[allow(clippy::too_many_arguments)]
fn try_gemm_mxfp8_tc_batched(
    weight_key: usize,
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let Some(cache) = cuda_cache() else {
        return false;
    };
    let Ok(mut guard) = cache.lock() else {
        return false;
    };
    guard.gemm_mxfp8_tc(weight_key, w_bytes, x, out, m, k, n)
}

/// Batched Blackwell MXFP6 tensor-core GEMM via the CUDA cache. See
/// [`try_gemm_mxfp8_tc_batched`].
#[allow(clippy::too_many_arguments)]
fn try_gemm_mxfp6_tc_batched(
    weight_key: usize,
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let Some(cache) = cuda_cache() else {
        return false;
    };
    let Ok(mut guard) = cache.lock() else {
        return false;
    };
    guard.gemm_mxfp6_tc(weight_key, w_bytes, x, out, m, k, n)
}

// ------------------------------------------------------------
// Hopper SM90a FP8 wgmma tensor-core GEMM dispatch (opt-in)
// ------------------------------------------------------------

/// Whether the Hopper FP8 `wgmma` tensor-core GEMM path is enabled.
/// **AUTOMATIC** — no env var: on only when the Hopper FP8 kernel passed the
/// on-device self-check (`kernel_verdict`), a usable NVIDIA GPU is present, and
/// the device is a TC-capable Hopper (`ck::hopper_tc_available`). No parity
/// probe emits this verdict yet (no GH200 is available to validate), so it
/// stays off automatically until one is. Cached once (process-wide).
fn hopper_tc_enabled() -> bool {
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| {
        kernel_verdict(rustllama_tuner::VERDICT_GEMM_FP8_WGMMA)
            && cuda_active()
            && ck::hopper_tc_available(0)
    })
}

/// Whether to use the TMA (`cp.async.bulk`)-staged `wgmma` variant. Defaults to
/// the non-TMA path; the TMA variant would be selected once it has its own
/// validated verdict (there is no on-device TMA-wgmma self-check yet, and no
/// Hopper HW to run it). Cached once.
fn hopper_tc_tma() -> bool {
    false
}

/// Batched Hopper FP8 `wgmma` GEMM via the CUDA cache. `false` on any miss so
/// the caller falls through to the scalar CUDA/SYCL/CPU ladder (untouched
/// `out`). Mirror of [`try_gemm_fp4_tc_batched`].
#[allow(clippy::too_many_arguments)]
fn try_gemm_fp8_wgmma_batched(
    weight_key: usize,
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let Some(cache) = cuda_cache() else {
        return false;
    };
    let Ok(mut guard) = cache.lock() else {
        return false;
    };
    guard.gemm_fp8_wgmma(weight_key, w_bytes, x, out, m, k, n, hopper_tc_tma())
}

// ------------------------------------------------------------
// Intel XMX/DPAS bf16 tensor-core GEMM dispatch (SYCL, opt-in)
// ------------------------------------------------------------

/// Whether the Intel XMX/DPAS bf16 tensor-core GEMM path is enabled.
/// **AUTOMATIC** — no env var: on only when the XMX GEMM kernel passed the
/// on-device self-check (`kernel_verdict("xmx:gemm")`), which the probe records
/// only on an XMX-capable device (Arc Xe-HPG / PVC Xe-HPC), so a passing verdict
/// already implies capability. The per-stream `sk::xmx_available` check still
/// guards the dispatch site. The path dequantizes the weight to f32 and runs a
/// bf16 `joint_matrix` GEMM. Fail-closed; cached once (process-wide).
fn xmx_enabled() -> bool {
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| kernel_verdict(rustllama_tuner::VERDICT_XMX_GEMM))
}

/// **AUTOMATIC** — no env var: the Q4_K prefill GEMM (tiled, shared-mem weight
/// reuse, f32 — BIT-EXACT) is on where its bit-exact parity probe passed on-
/// device (`kernel_verdict("gemm:q4_k_f32")`). Fail-closed: a write-blind kernel
/// bug on an unvalidated arch leaves the verdict unset → the batched matvec runs.
/// This is the weight-reuse fix for the O(M·K·N) re-streaming on the prefill
/// path. Cached once (process-wide).
fn q4k_gemm_f32_enabled() -> bool {
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| {
        kernel_verdict(rustllama_tuner::VERDICT_GEMM_Q4K_F32) && cuda_active()
    })
}

/// **AUTOMATIC** — no env var: the Q4_K int8 tensor-core GEMM (W8A8, LOSSY) is on
/// where its probe recorded a pass (`kernel_verdict("gemm:q4_k_w8a8_tc")`), which
/// requires both correctness (vs a fair W8A8 reference) AND beating the lossless
/// f32 GEMM on-device. So when enabled it is the fastest Q4_K prefill path and is
/// dispatched FIRST; otherwise the bit-exact f32 GEMM runs. Fail-closed; cached.
fn q4k_w8a8_tc_enabled() -> bool {
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| {
        kernel_verdict(rustllama_tuner::VERDICT_GEMM_Q4K_W8A8_TC) && cuda_active()
    })
}

/// Batched XMX/DPAS bf16 GEMM via the USM_ATTN SYCL stream: dequantize W→f32
/// and run the `joint_matrix` GEMM (the kernel zero-pads the M/N/K tails, so no
/// dim alignment is required). `false` on any miss — no USM_ATTN context, the
/// device isn't XMX-capable, the dtype isn't dequantizable to f32, or the
/// kernel reports unavailable/exception — so the caller falls straight through
/// to the proven packed/scalar SYCL/CPU ladder with `out` untouched.
fn try_gemm_xmx_batched(w: &Tensor, x: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) -> bool {
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(ctx) = slot.as_mut() else {
            return false;
        };
        // Skip the f32 weight dequant entirely on a non-XMX device (Iris Xe /
        // OpenCL fallback / non-XMX build) — the probe is cheap, the dequant
        // is not.
        if !sk::xmx_available(&ctx.stream) {
            return false;
        }
        let Some(w_f32) = dtype_dequant_to_f32(w.dtype, as_bytes(w), m * k) else {
            return false; // exotic codebook/ternary/fp4 dtype — no f32 dequant
        };
        matches!(
            sk::gemm_bf16_xmx_f32_host(&ctx.stream, &w_f32, x, out, m, k, n),
            Ok(true)
        )
    })
}

// ============================================================
// Native MLX packed-matvec dispatch (Apple Metal backend)
// ============================================================
//
// The Apple-Metal analogue of the CUDA packed-matvec dispatch above and
// the SYCL USM ladder below — the 4th device tier (beside SYCL, CUDA and
// CPU). When a Metal GPU is present the packed-matvec dispatch points route
// the supported kinds here, mirroring the CUDA precedence (and mutually
// exclusive with it in practice — a host has Metal xor CUDA/SYCL GPUs).
// Everything is INERT off Apple Silicon: `mk::device_count()` returns 0 →
// [`mlx_cache`] is `None` (never constructed) and every MLX attempt short-
// circuits to `false`, so the proven SYCL/CUDA/CPU path is bit-for-bit
// unchanged on Windows/Linux/Intel-mac hosts.
//
// APPLE UNIFIED MEMORY: unlike CUDA's separate device memory, Apple's GPU
// shares one physical pool with the CPU, so the cache's "upload" is a cheap
// StorageModeShared memcpy. The cache keeps CUDA's shape (upload once,
// keyed by the GGUF host pointer) so this dispatch stays backend-agnostic;
// Phase 1 may collapse it to a zero-copy MTLBuffer wrap of the weight bytes.

/// Whether the native MLX (Apple Metal) packed-matvec backend is active:
/// a usable Metal GPU exists (and the env off-switch isn't set). No per-
/// thread opt-in — the cache is process-wide and thread-agnostic. Mirror of
/// [`cuda_active`]; inert (false, cached) off Apple Silicon, so no engine
/// wiring is needed and the SYCL/CUDA/CPU path is untouched there.
#[inline]
pub fn mlx_active() -> bool {
    mlx_cache().is_some()
}

/// Whether a native MLX compute device is available for dispatch.
/// Public mirror of [`mlx_active`] for operator-facing reporting.
pub fn mlx_available() -> bool {
    mlx_cache().is_some()
}

/// The process-wide device-resident MLX matvec cache. Lazily created once:
/// `Some` only when a Metal GPU is visible, the env off-switch
/// (`RUSTLLAMA_MLX_DISPATCH=0`) isn't set, and a stream + budget could be
/// established; `None` (cached) otherwise. Budget = 85% of device 0's total
/// (unified) memory, matching the CUDA/SYCL VRAM-fit reserve. Phase 0:
/// `mk::device_count()` is 0 everywhere (the stub), so this always caches
/// `None` off Apple Silicon.
fn mlx_cache() -> Option<&'static Mutex<mk::MlxMatvecCache>> {
    static CACHE: OnceLock<Option<Mutex<mk::MlxMatvecCache>>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            // Env off-switch for A/B (mirrors RUSTLLAMA_CUDA_DISPATCH).
            if std::env::var("RUSTLLAMA_MLX_DISPATCH")
                .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
                .unwrap_or(false)
            {
                return None;
            }
            if mk::device_count() == 0 {
                return None;
            }
            let budget = mk::device_info(0)
                .map(|i| ((i.total_mem_bytes as f64) * 0.85) as usize)
                .unwrap_or(0);
            if budget == 0 {
                return None;
            }
            mk::MlxMatvecCache::new(0, budget).map(Mutex::new)
        })
        .as_ref()
}

/// Map a weight [`Dtype`] to the MLX cache's packed kind, or `None` for
/// dtypes the MLX backend doesn't dispatch — those fall through to the
/// SYCL/CPU ladder. Byte-for-byte mirror of [`dtype_to_cuda_kind`]: every
/// packed-quant `*Raw` dtype the loader produces has an MLX arm (the kernel
/// surface + `MlxMatvecCache` cover all of them; `k_alignment`/`row_bytes`
/// on `MlxPackedKind` gate each shape).
fn dtype_to_mlx_kind(dtype: Dtype) -> Option<mk::MlxPackedKind> {
    match dtype {
        Dtype::PTQ1_0Raw => Some(mk::MlxPackedKind::Ptq1_0),
        Dtype::Q8_0Raw => Some(mk::MlxPackedKind::Q8_0),
        Dtype::Q4_KRaw => Some(mk::MlxPackedKind::Q4_K),
        Dtype::Q6_KRaw => Some(mk::MlxPackedKind::Q6_K),
        Dtype::Q5_KRaw => Some(mk::MlxPackedKind::Q5_K),
        Dtype::Q2_KRaw => Some(mk::MlxPackedKind::Q2_K),
        Dtype::Q8_KRaw => Some(mk::MlxPackedKind::Q8_K),
        Dtype::Q4_0Raw => Some(mk::MlxPackedKind::Q4_0),
        Dtype::Q5_0Raw => Some(mk::MlxPackedKind::Q5_0),
        Dtype::Q4_1Raw => Some(mk::MlxPackedKind::Q4_1),
        Dtype::Q5_1Raw => Some(mk::MlxPackedKind::Q5_1),
        Dtype::IQ4_NLRaw => Some(mk::MlxPackedKind::Iq4_Nl),
        Dtype::IQ4_XSRaw => Some(mk::MlxPackedKind::Iq4_Xs),
        Dtype::IQ2_XXSRaw => Some(mk::MlxPackedKind::Iq2_Xxs),
        Dtype::IQ2_XSRaw => Some(mk::MlxPackedKind::Iq2_Xs),
        Dtype::IQ2_SRaw => Some(mk::MlxPackedKind::Iq2_S),
        Dtype::IQ3_XXSRaw => Some(mk::MlxPackedKind::Iq3_Xxs),
        Dtype::IQ3_SRaw => Some(mk::MlxPackedKind::Iq3_S),
        Dtype::IQ1_SRaw => Some(mk::MlxPackedKind::Iq1_S),
        Dtype::IQ1_MRaw => Some(mk::MlxPackedKind::Iq1_M),
        Dtype::Nvfp4Raw => Some(mk::MlxPackedKind::Nvfp4),
        Dtype::Q3_KRaw => Some(mk::MlxPackedKind::Q3_K),
        Dtype::PQ2_0Raw => Some(mk::MlxPackedKind::Pq2_0),
        Dtype::Mxfp4Raw => Some(mk::MlxPackedKind::Mxfp4),
        Dtype::Mxfp6Raw => Some(mk::MlxPackedKind::Mxfp6),
        Dtype::Mxfp8Raw => Some(mk::MlxPackedKind::Mxfp8),
        // TODO(mlx-metal): `Dtype::MlxAffineRaw` (Apple MLX affine, kept
        // packed) would route here to a native `mlx_quantized_matmul`
        // (Metal `qmv`/`qmm`) on Apple Silicon — the whole point of the
        // packed-residency format. Until that kernel lands it falls to
        // `None` → the CPU `matvec_mlx_affine_blob_*` path (the must-have,
        // and the only path on this Windows/Intel box where MLX runs on
        // CPU). The SYCL/CUDA ladders likewise have no native MLX-affine
        // kernel; they leave it to CPU (a dequant-to-f16-on-upload GPU
        // shim is a separate follow-up).
        _ => None,
    }
}

/// Single-row packed matvec via the native MLX backend. Returns `false` on
/// any miss (no device, over budget, kernel failure) so the caller falls
/// through to the SYCL/CPU ladder; leaves `out` untouched on failure.
/// Mirror of [`try_matvec_packed_cuda`].
fn try_matvec_packed_mlx(
    kind: mk::MlxPackedKind,
    weight_key: usize,
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) -> bool {
    let Some(cache) = mlx_cache() else {
        return false;
    };
    let Ok(mut guard) = cache.lock() else {
        return false;
    };
    guard.matvec_packed(kind, weight_key, w_bytes, x, out, m, k)
}

/// Batched (`n`-row) packed matvec via the native MLX backend. Mirror of
/// [`try_matvec_packed_cuda_batched`].
#[allow(clippy::too_many_arguments)]
fn try_matvec_packed_mlx_batched(
    kind: mk::MlxPackedKind,
    weight_key: usize,
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let Some(cache) = mlx_cache() else {
        return false;
    };
    let Ok(mut guard) = cache.lock() else {
        return false;
    };
    guard.matvec_packed_batched(kind, weight_key, w_bytes, x, out, m, k, n)
}

// ============================================================
// Phase 4: multi-GPU device-assignment plan + per-device caches
// ============================================================
//
// Everything here is INERT unless a `MultiGpuPlan` is installed on the current
// generate thread — which only happens when the engine's heat planner runs
// (>1 usable GPU or a heat-placement opt-in, AFTER a tune measured per-device
// perf). With no plan installed (the default, and the single-GPU/CPU box) the
// dispatch path is byte-identical to before this section existed: the gate is a
// single `multi_gpu_plan_active()` TLS bool read that returns `false`.

/// Which compute backend a weight is routed to under a multi-GPU plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBackend {
    /// Intel SYCL device (the single active per-thread USM stream).
    Sycl,
    /// NVIDIA CUDA device, addressed by 0-based `device_index`.
    Cuda,
    /// Apple Metal / MLX device, addressed by 0-based `device_index`. The
    /// 4th GPU tier; Apple Silicon is single-GPU + unified-memory in
    /// practice, so the multi-device (`device_index != 0`) cross-GPU
    /// routing the CUDA path has is never exercised for Metal — the
    /// single-device `mlx_active()` dispatch below covers it.
    Metal,
}

/// A single GPU target: backend + 0-based device index within that backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceTarget {
    pub backend: GpuBackend,
    pub device_index: u32,
}

/// Phase 5's output, consumed on the Phase 4 dispatch hot path: a per-tensor
/// device-assignment table. A weight NOT present in the plan (`device_for`
/// returns `None`) lives on the CPU tier; a weight present routes to the named
/// GPU device.
///
/// The plan is AUTHORITATIVE when installed: it SUBSUMES both the flat
/// `n_gpu_layers` layer cutoff and the MoE experts→CPU class split (the engine
/// sets `n_gpu_layers = total` when installing a plan so the cutoff never also
/// forces CPU). It is only installed by the engine's heat planner AFTER a tune
/// has measured per-device perf (see `rustllama_engine::multi_gpu`). With no
/// plan installed (the default, and the single-GPU/CPU common case) the
/// dispatch path is byte-identical to before this type existed.
#[derive(Debug, Clone, Default)]
pub struct MultiGpuPlan {
    /// Exact GGUF tensor name → GPU device. An absent name ⇒ CPU tier.
    by_name: HashMap<String, DeviceTarget>,
    /// The GPU that runs attention (and therefore holds the KV cache).
    /// Informational for the engine; `None` when attention is on CPU.
    attn_device: Option<DeviceTarget>,
}

impl MultiGpuPlan {
    pub fn new(
        by_name: HashMap<String, DeviceTarget>,
        attn_device: Option<DeviceTarget>,
    ) -> Self {
        Self { by_name, attn_device }
    }
    /// The GPU a weight is assigned to, or `None` for the CPU tier.
    /// `_layer_idx` is accepted for future per-layer fallbacks; today the
    /// table is keyed purely by tensor name (exact GGUF names are unique).
    #[inline]
    pub fn device_for(&self, tensor_name: &str, _layer_idx: u32) -> Option<DeviceTarget> {
        self.by_name.get(tensor_name).copied()
    }
    /// The device running attention (KV-cache-resident GPU), if any.
    pub fn attn_device(&self) -> Option<DeviceTarget> {
        self.attn_device
    }
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }
    /// Number of distinct GPU devices this plan routes weights to. `<= 1`
    /// means no cross-GPU routing is needed (single active device handles
    /// every GPU-assigned tensor).
    pub fn distinct_gpu_count(&self) -> usize {
        let mut set: std::collections::HashSet<DeviceTarget> = std::collections::HashSet::new();
        for t in self.by_name.values() {
            set.insert(*t);
        }
        set.len()
    }
}

/// Process-global installed multi-GPU plan. Unlike the per-call TLS knobs
/// (`N_GPU_LAYERS`, `CPU_FORCE_PATTERNS`), a plan is a whole-SYSTEM property
/// installed ONCE at model load by the engine's heat planner, so a single
/// global (visible to every tokio blocking-pool generate thread) is the right
/// shape and avoids re-pushing per call. `RwLock::new(None)` is const, so no
/// lazy init.
static MULTI_GPU_PLAN: std::sync::RwLock<Option<std::sync::Arc<MultiGpuPlan>>> =
    std::sync::RwLock::new(None);

/// Cheap hot-path gate, kept in lockstep with `MULTI_GPU_PLAN`. The dispatch
/// path reads THIS first; a single relaxed-acquire atomic load returning
/// `false` (the default, and the single-GPU/CPU box) means the dispatcher never
/// touches the lock and runs exactly today's code.
static MULTI_GPU_ACTIVE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Install (or clear) the multi-GPU device-assignment plan process-wide.
/// Called by the engine at model load: `Some(plan)` after the heat planner
/// produces one; `None` on every VRAM-fit / tune-measurement load, which also
/// CLEARS any stale plan from a previously-loaded model. Idempotent + safe to
/// call from any thread.
pub fn set_multi_gpu_plan(plan: Option<std::sync::Arc<MultiGpuPlan>>) {
    let active = plan.is_some();
    if let Ok(mut slot) = MULTI_GPU_PLAN.write() {
        *slot = plan;
    }
    MULTI_GPU_ACTIVE.store(active, std::sync::atomic::Ordering::Release);
}

/// Is a multi-GPU plan installed? Hot-path gate — one atomic load. `false`
/// (default) ⇒ the dispatcher short-circuits to the historical path.
#[inline]
pub fn multi_gpu_plan_active() -> bool {
    MULTI_GPU_ACTIVE.load(std::sync::atomic::Ordering::Acquire)
}

/// Snapshot the installed plan (cheap `Arc` clone) for the dispatch branch.
/// Returns `None` (without taking the lock) in the common no-plan case.
#[inline]
fn multi_gpu_plan() -> Option<std::sync::Arc<MultiGpuPlan>> {
    if !multi_gpu_plan_active() {
        return None;
    }
    MULTI_GPU_PLAN.read().ok().and_then(|g| g.clone())
}

/// The default CUDA device index the single-device dispatch path uses (0).
/// The proven `cuda_cache()` above is pinned to it and is left untouched; the
/// per-device map below only ever constructs caches for OTHER indices, so the
/// working single-GPU CUDA path stays byte-identical.
const CUDA_DEFAULT_DEVICE: u32 = 0;

/// Per-device CUDA matvec cache, honoring `RUSTLLAMA_DISABLED_GPUS`. Device 0
/// delegates to the proven process-wide `cuda_cache()` (unchanged). Other
/// device indices get a lazily-built, process-lifetime cache stored in a leaked
/// `&'static Mutex<…>` (one per device, exactly like the `OnceLock` the default
/// path uses). Returns `None` for a missing/disabled device or on alloc
/// failure, so the caller falls through to the existing ladder.
///
/// SCAFFOLD NOTE: this is the Phase 4 "per-device resident weight cache"
/// generalization. It is exercised only when a `MultiGpuPlan` routes a weight
/// to a NON-default CUDA device (i.e. >1 usable NVIDIA GPU + an installed
/// plan), which cannot occur on the single-iGPU validation box — so this
/// multi-device path is UNVALIDATED and gated off by plan-absence in the
/// common case. The single-device CUDA path (`cuda_cache()`) is untouched.
fn cuda_cache_for(device_index: u32) -> Option<&'static Mutex<ck::CudaMatvecCache>> {
    if device_index == CUDA_DEFAULT_DEVICE {
        return cuda_cache();
    }
    if cuda_device_disabled(device_index) {
        return None;
    }
    static MAP: OnceLock<Mutex<HashMap<u32, Option<&'static Mutex<ck::CudaMatvecCache>>>>> =
        OnceLock::new();
    let map = MAP.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = map.lock().ok()?;
    if let Some(slot) = g.get(&device_index) {
        return *slot;
    }
    let built = build_cuda_cache(device_index);
    g.insert(device_index, built);
    built
}

/// Build a leaked, process-lifetime CUDA cache for `device_index`. Mirrors the
/// budget policy of the default `cuda_cache()` (85% of that device's total
/// memory) and the same env off-switch. `None` when the index is out of range,
/// `RUSTLLAMA_CUDA_DISPATCH=0`, or the stream/alloc fails.
fn build_cuda_cache(device_index: u32) -> Option<&'static Mutex<ck::CudaMatvecCache>> {
    if std::env::var("RUSTLLAMA_CUDA_DISPATCH")
        .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
    {
        return None;
    }
    if device_index >= ck::device_count() {
        return None;
    }
    let budget = ck::device_info(device_index)
        .map(|i| ((i.total_mem_bytes as f64) * 0.85) as usize)
        .unwrap_or(0);
    if budget == 0 {
        return None;
    }
    let cache = ck::CudaMatvecCache::new(device_index, budget)?;
    // Leak once (process-lifetime), exactly like the default path's OnceLock —
    // one cache per physical device, reclaimed at process exit.
    Some(&*Box::leak(Box::new(Mutex::new(cache))))
}

/// `RUSTLLAMA_DISABLED_GPUS` parsed to a set of UNIFIED GPU indices. Inlined
/// (rather than depending on `rustllama-runtime`, which `rustllama-models` does
/// not) to mirror the identical parse already in
/// [`first_enabled_sycl_device_index`].
fn disabled_gpus_env() -> std::collections::HashSet<u32> {
    std::env::var("RUSTLLAMA_DISABLED_GPUS")
        .ok()
        .map(|s| {
            s.split(|c: char| c == ',' || c.is_whitespace())
                .filter_map(|t| t.trim().parse::<u32>().ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Count of distinct PHYSICAL SYCL GPUs (deduped by name, as
/// [`first_enabled_sycl_device_index`] does). The unified enumeration lists
/// these first, so a CUDA device's unified index = this count + its CUDA index.
fn distinct_sycl_gpu_count() -> u32 {
    let n = sk::device_count().unwrap_or(0);
    let mut names: std::collections::HashSet<String> = std::collections::HashSet::new();
    for i in 0..n {
        if let Ok(info) = sk::device_info(i) {
            names.insert(info.name);
        }
    }
    names.len() as u32
}

/// Whether a CUDA device's UNIFIED GPU index is in `RUSTLLAMA_DISABLED_GPUS`.
/// Best-effort defense-in-depth (the engine planner is the authority and never
/// routes to a disabled device): on any probe ambiguity it treats the device as
/// ENABLED so a usable GPU is never silently dropped.
fn cuda_device_disabled(cuda_index: u32) -> bool {
    let disabled = disabled_gpus_env();
    if disabled.is_empty() {
        return false;
    }
    let unified = distinct_sycl_gpu_count().saturating_add(cuda_index);
    disabled.contains(&unified)
}

/// Single-row packed matvec on a SPECIFIC CUDA device (Phase 4 cross-GPU
/// routing). Mirrors [`try_matvec_packed_cuda`] but targets `device_index`'s
/// own resident cache. `false` on any miss so the caller falls through.
#[allow(clippy::too_many_arguments)]
fn try_matvec_packed_cuda_dev(
    device_index: u32,
    kind: ck::CudaPackedKind,
    weight_key: usize,
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) -> bool {
    let Some(cache) = cuda_cache_for(device_index) else {
        return false;
    };
    let Ok(mut guard) = cache.lock() else {
        return false;
    };
    guard.matvec_packed(kind, weight_key, w_bytes, x, out, m, k)
}

/// Batched (`n`-row) packed matvec on a SPECIFIC CUDA device.
#[allow(clippy::too_many_arguments)]
fn try_matvec_packed_cuda_dev_batched(
    device_index: u32,
    kind: ck::CudaPackedKind,
    weight_key: usize,
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let Some(cache) = cuda_cache_for(device_index) else {
        return false;
    };
    let Ok(mut guard) = cache.lock() else {
        return false;
    };
    guard.matvec_packed_batched(kind, weight_key, w_bytes, x, out, m, k, n)
}

/// Per-thread reusable scratch buffers for the LLaMA forward pass.
/// Created on first `forward_one` call on a given thread, resized
/// in-place when model dims change (e.g., a different model is
/// loaded in the same engine binary), and reused across every
/// subsequent forward call from the same thread.
///
/// Replaces ~12 `vec![0f32; ...]` allocations per token. At 4K-token
/// prefill that's a 50K-allocation savings per request — small but
/// real on the per-token wall clock.
///
/// All buffers are sized to the model's max dims (`d`, `d_q`,
/// `d_kv`, `d_ff`) and zero-filled on each forward call (the
/// kernels themselves overwrite via `copy_from_slice`-style writes,
/// but the residual-add pattern reads from these buffers first so
/// stale values would corrupt output).
#[derive(Default)]
pub struct ForwardScratch {
    pub hidden: Vec<f32>,
    pub h_norm: Vec<f32>,
    pub q_buf: Vec<f32>,
    pub k_buf: Vec<f32>,
    pub v_buf: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub attn_proj: Vec<f32>,
    pub gate_buf: Vec<f32>,
    pub up_buf: Vec<f32>,
    pub ffn_buf: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub final_norm: Vec<f32>,
    /// MoE-only scratch — sized when `n_experts > 0` was passed to
    /// [`Self::prepare`]. Dense models leave these at length 0 so
    /// the memory cost is just the empty-Vec struct (~24 bytes
    /// each).
    ///
    /// `moe_down`: per-expert FFN's down-projection output buffer
    /// (`[d_model]`). Distinct from `ffn_out` because the MoE path
    /// accumulates the routed-experts' weighted sum into `ffn_out`
    /// while each expert's individual contribution lands here
    /// first.
    pub moe_down: Vec<f32>,
    /// Per-token expert logit scratch (`[n_experts]`). Receives
    /// the `router @ hidden` result before softmax + top-K
    /// selection.
    pub moe_expert_logits: Vec<f32>,
    /// Top-K selection result (`(expert_idx, weight)` per pick).
    /// Reused across MoE forward calls so the Vec's capacity
    /// grows once to `top_k` and stays there.
    pub moe_routed_picks: Vec<(usize, f32)>,
    /// `(head_dim, rope_theta)`-keyed inverse-frequency cache so
    /// we only recompute the table when the model changes. Stored
    /// as `(head_dim, rope_theta_bits)` for cheap exact comparison.
    pub rope_inv_freq: Vec<f32>,
    rope_key: Option<(u32, u32)>,
    /// Per-row WHT/quantize scratch for the TurboQuant KV append
    /// path. Sized to `head_dim`; the TQ append walks `n_kv_heads`
    /// K + V rows per token and would otherwise allocate this Vec
    /// inline per layer per token. Pooling here cuts ~`2 *
    /// n_layers` micro-allocs per generated token on TQ KV configs.
    pub tq_scratch: Vec<f32>,
    /// H2: pooled K/V gather slabs for paged-KV forward. Sized to
    /// `n_kv_heads × max_ctx × head_dim` so the worst-case kv_len
    /// (a fully-grown context) fits without reallocation. Was a
    /// fresh `vec![0f32; n_kv_heads × kv_len × head_dim]` per token
    /// per call ([llama_arch.rs:4764-4765](llama_arch.rs)) — on a
    /// 4K-context 28-layer model that's ~100-200 MB of allocator
    /// pressure per 1000 tokens. Reused across layers + tokens;
    /// callers pass the `kv_len`-sized prefix as a slice into the
    /// pooled storage. Dense models without paged-KV leave these
    /// at length 0 (Vec struct is ~24 bytes).
    pub paged_k_slab: Vec<f32>,
    pub paged_v_slab: Vec<f32>,
}

impl ForwardScratch {
    /// Ensure every buffer is sized to fit the per-forward-call
    /// scratch and that the rope inv-freq table matches the
    /// current model. Cheap on the common case (sizes match,
    /// rope_key matches → just zeros the buffers).
    ///
    /// `n_experts` and `top_k` size the MoE-only buffers; pass
    /// `0` for both on dense models (the buffers will resize
    /// down to 0 but otherwise have no cost).
    pub fn prepare(
        &mut self,
        d: usize,
        d_q: usize,
        d_kv: usize,
        d_ff: usize,
        head_dim: usize,
        rope_theta: f32,
        n_experts: usize,
        top_k: usize,
    ) {
        resize_zero(&mut self.hidden, d);
        resize_zero(&mut self.h_norm, d);
        resize_zero(&mut self.q_buf, d_q);
        resize_zero(&mut self.k_buf, d_kv);
        resize_zero(&mut self.v_buf, d_kv);
        resize_zero(&mut self.attn_out, d_q);
        resize_zero(&mut self.attn_proj, d);
        resize_zero(&mut self.gate_buf, d_ff);
        resize_zero(&mut self.up_buf, d_ff);
        resize_zero(&mut self.ffn_buf, d_ff);
        resize_zero(&mut self.ffn_out, d);
        resize_zero(&mut self.final_norm, d);
        resize_zero(&mut self.tq_scratch, head_dim);
        // MoE scratch — sized only when caller declared MoE
        // model. resize_zero handles shrink-to-zero on dense
        // models cheaply (no realloc, just length update).
        resize_zero(&mut self.moe_down, if n_experts > 0 { d } else { 0 });
        resize_zero(
            &mut self.moe_expert_logits,
            if n_experts > 0 { n_experts } else { 0 },
        );
        // routed_picks holds (usize, f32) tuples; clear & reserve
        // explicitly because resize_zero is f32-only.
        self.moe_routed_picks.clear();
        if top_k > 0 {
            self.moe_routed_picks.reserve(top_k);
        }
        let key = (head_dim as u32, rope_theta.to_bits());
        if self.rope_key != Some(key) {
            self.rope_inv_freq = rope_inv_freq_table(head_dim, rope_theta);
            self.rope_key = Some(key);
        }
    }

    /// H2: ensure the paged-KV gather slabs are sized to hold
    /// `n_kv_heads × kv_len × head_dim` f32s. Idempotent on the
    /// hot path: if the slab is already big enough, zero the prefix
    /// in-place. Otherwise resize-and-zero (grows worst case to
    /// `n_kv_heads × max_ctx × head_dim`).
    ///
    /// Called from the paged-decode/prefill paths instead of per-
    /// token `vec![0f32; slab_len]` allocation. Caller takes the
    /// `[..slab_used]` prefix as `&mut [f32]`.
    pub fn ensure_paged_kv_slab(&mut self, n_kv_heads: usize, kv_len: usize, head_dim: usize) {
        let need = n_kv_heads * kv_len * head_dim;
        resize_zero(&mut self.paged_k_slab, need);
        resize_zero(&mut self.paged_v_slab, need);
    }
}

fn resize_zero(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    } else {
        for x in &mut v[..n] {
            *x = 0.0;
        }
    }
}

thread_local! {
    /// Lazily-initialized per-thread `ForwardScratch`. `RefCell` so
    /// `forward_one` can borrow_mut on entry, fill the buffers, and
    /// release on return. Single-threaded TLS keeps the borrow
    /// trivially uncontended.
    static FORWARD_SCRATCH: RefCell<ForwardScratch> =
        RefCell::new(ForwardScratch::default());
}

/// Run `f` with mutable access to this thread's `ForwardScratch`,
/// pre-sized to the given model dims. The scratch is owned by TLS
/// — it persists across calls so the second forward pass on a
/// thread sees zero scratch allocation.
pub fn with_forward_scratch<R>(
    d: usize,
    d_q: usize,
    d_kv: usize,
    d_ff: usize,
    head_dim: usize,
    rope_theta: f32,
    n_experts: usize,
    top_k: usize,
    f: impl FnOnce(&mut ForwardScratch) -> R,
) -> R {
    FORWARD_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.prepare(d, d_q, d_kv, d_ff, head_dim, rope_theta, n_experts, top_k);
        f(&mut scratch)
    })
}

thread_local! {
    /// Per-thread SYCL-health flag. Starts at `true`; flips to
    /// `false` after [`SYCL_FAIL_BUDGET`] consecutive kernel
    /// failures. Once `false`, every `try_*_f32` / `try_*_usm_f32`
    /// short-circuits and the engine stays on the CPU path for the
    /// remainder of this thread's lifetime.
    ///
    /// Why TLS rather than a global atomic: each generate request
    /// runs on a freshly spawned `spawn_blocking` worker thread,
    /// so resetting per-thread also resets per-request. A worker
    /// that successfully ran SYCL on one request doesn't trip the
    /// breaker for unrelated workers; a worker that hits driver
    /// trouble doesn't poison the whole process.
    static SYCL_HEALTHY: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };

    /// Counter of consecutive SYCL FFI failures observed on this
    /// thread. Reset to `0` whenever a kernel call succeeds; once
    /// it crosses [`SYCL_FAIL_BUDGET`], `SYCL_HEALTHY` flips off
    /// and stays off for this thread.
    static SYCL_FAIL_COUNT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };

    /// The per-thread SYCL stream, set via [`SyclStreamGuard::install`]
    /// and cleared on guard drop. `RefCell` because we need
    /// `borrow_mut` from inside a kernel call (the FFI takes
    /// `&mut SyclStream`). Single-threaded TLS so the borrow is
    /// trivially uncontended.
    static SYCL_STREAM: RefCell<Option<sk::SyclStream>> = const { RefCell::new(None) };

    /// Per-thread "use FlashAttention-decode for F32 KV" flag.
    /// Mirrors `[inference].flash_attention` from config; the engine
    /// sets this at the start of each generate call. Defaults to
    /// `true` so unit tests that don't go through the engine still
    /// pick up the new path (matches the config default).
    ///
    /// Currently only consulted by the F32 KV attention dispatch in
    /// `llama_arch::forward_one`. Quantized KV paths keep their fused
    /// dequant+attention kernels — flash-decode there is a follow-up.
    static FLASH_ATTENTION: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };

    /// Per-thread `[inference].n_gpu_layers` cutoff for hybrid
    /// CPU/GPU placement. Layers with `layer_idx < N_GPU_LAYERS` go
    /// through the SYCL/USM dispatch ladder; layers at or above the
    /// cutoff fall through to CPU kernels even when SYCL is healthy.
    ///
    /// Default `u32::MAX` ("all GPU") so unit tests + non-engine
    /// callers behave like before this knob existed. The engine
    /// sets it at the start of each generate call from
    /// `cfg.inference.n_gpu_layers`.
    ///
    /// Why a layer cutoff vs a per-tensor placement map: this
    /// covers ~90% of the "VRAM-tight 7B on Iris Xe" case with one
    /// scalar. Per-tensor regex overrides
    /// (`placement.overrides`) are the next refinement and slot in
    /// behind a separate TLS map without changing this one.
    static N_GPU_LAYERS: std::cell::Cell<u32> = const { std::cell::Cell::new(u32::MAX) };

    /// Per-thread current layer index, set at the top of each layer
    /// loop iteration in `llama_arch::forward_*`. Read by the
    /// `try_*_usm_f32` dispatch helpers via
    /// [`gpu_active_for_current_layer`] to decide whether this
    /// layer is GPU-resident or CPU-resident.
    ///
    /// Default `0` so a layer 0 in tests that forgot to set this
    /// still gets the GPU path (when SYCL is healthy and
    /// `N_GPU_LAYERS > 0`).
    static CURRENT_LAYER_IDX: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Engine-side setter for `[inference].n_gpu_layers`. Called once
/// per generate / chat call before the forward pass kicks off.
/// `u32::MAX` (or any value ≥ model.n_layers) means "all GPU";
/// `0` means "all CPU."
pub fn set_n_gpu_layers(n: u32) {
    N_GPU_LAYERS.with(|c| c.set(n));
}

/// Read the per-thread `[inference].n_gpu_layers` cutoff.
pub fn n_gpu_layers() -> u32 {
    N_GPU_LAYERS.with(|c| c.get())
}

/// Model-side setter for the layer the forward pass is currently
/// processing. Called at the top of each layer's loop iteration in
/// `llama_arch::forward_*`. The `try_*_usm_f32` helpers consult
/// this via [`gpu_active_for_current_layer`] to short-circuit the
/// GPU path for layers above the placement cutoff.
pub fn set_current_layer_idx(idx: u32) {
    CURRENT_LAYER_IDX.with(|c| c.set(idx));
}

/// Read the per-thread current layer index. Mostly used by the
/// `try_*_usm_f32` helpers + tests.
pub fn current_layer_idx() -> u32 {
    CURRENT_LAYER_IDX.with(|c| c.get())
}

/// Whether the GPU path is enabled for the layer that's currently
/// being processed on this thread. Returns `true` when
/// `current_layer_idx < n_gpu_layers`. The `try_*_usm_f32` dispatch
/// helpers short-circuit and return `false` when this is `false`,
/// causing the engine to fall through to the CPU kernel.
///
/// V1 contract: cutoff applies to ALL kernels in the layer (rmsnorm,
/// rope, silu_mul, all matmuls, attention). Per-tensor overrides
/// ([`tensor_forced_to_cpu`]) consult a separate TLS list and run
/// **before** this scalar cutoff in the matvec dispatch — they let
/// a user pin specific weight tensors (e.g. all `ffn_*`) to CPU
/// even when the layer would otherwise route to GPU.
#[inline]
pub fn gpu_active_for_current_layer() -> bool {
    // A DEVICE_LOST is terminal (the SYCL context is gone). Route every
    // layer to CPU for the rest of the session so the flash-attention USM
    // paths (and anything else gated here) stop hitting the dead device —
    // the packed-matvec dispatch is disabled separately in
    // `packed_kind_disabled`.
    if rustllama_kernels_sycl::device_lost() {
        return false;
    }
    current_layer_idx() < n_gpu_layers()
}

thread_local! {
    /// Per-thread list of `[inference].placement.overrides`
    /// patterns whose `device = "cpu"`. The matvec dispatch
    /// helpers (`try_matvec_tensor_usm_f32` + batched variant)
    /// check tensor names against this list via
    /// [`tensor_forced_to_cpu`]; a match forces the CPU path even
    /// when the layer's `n_gpu_layers` cutoff says GPU.
    ///
    /// V1 matching is **substring**: pattern `"ffn"` matches any
    /// tensor name containing `"ffn"` (e.g. `blk.5.ffn_gate.weight`).
    /// The plan's regex-style examples like `"ffn.*"` work as
    /// substrings of normal tensor names because `.*` chars don't
    /// appear in GGUF names — a user who writes `"ffn"` gets the
    /// same matches with a clearer intent. A future regex upgrade
    /// is a one-line swap of `name.contains(pattern)` for a
    /// `Regex::is_match` call.
    static CPU_FORCE_PATTERNS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Set the per-thread CPU-force pattern list. Engine calls this
/// at the start of each generate / chat call with the patterns
/// pulled from `[inference].placement.overrides` (filtered to
/// entries with `device = "cpu"`). Empty list disables override
/// matching entirely — every dispatch falls through to the
/// scalar `n_gpu_layers` cutoff.
pub fn set_cpu_force_patterns(patterns: Vec<String>) {
    CPU_FORCE_PATTERNS.with(|c| *c.borrow_mut() = patterns);
}

/// Whether `tensor_name` matches any CPU-force pattern in the
/// per-thread list. Called by `try_matvec_tensor_usm_f32` (and
/// the batched variant) BEFORE the `n_gpu_layers` cutoff check.
/// V1 matching is substring; see [`set_cpu_force_patterns`] for
/// the rationale.
pub fn tensor_forced_to_cpu(tensor_name: &str) -> bool {
    // Phase 5 heat plan (installed only post-tune, on >=1 GPU): the plan is the
    // AUTHORITATIVE per-tensor CPU-vs-GPU decision and flows through THIS
    // existing gate rather than a parallel mechanism. When a plan is installed
    // it is the SOLE authority (the engine folds any user CPU-overrides + the
    // MoE experts→CPU split into it at build time), so we early-return here and
    // skip the legacy MoE/pattern rules below. A tensor the plan does not
    // assign to a GPU device is a CPU-tier tensor. Inert (one atomic load,
    // skipped) when no plan is installed → byte-identical to before.
    if multi_gpu_plan_active() {
        if let Some(plan) = multi_gpu_plan() {
            return plan.device_for(tensor_name, current_layer_idx()).is_none();
        }
    }
    // G8: MoE attn-on-GPU / experts-on-CPU split. When
    // `RUSTLLAMA_MOE_EXPERTS_CPU=1`, the per-expert FFN weights
    // (`*ffn_gate_exps*`, `*ffn_up_exps*`, `*ffn_down_exps*`) and the
    // routing tensor (`*ffn_gate_inp*`) force back to CPU even on
    // GPU layers. Rationale: only top-K of N experts fire per token
    // (~3% at K=8, N=256), so pinning the cold majority to CPU saves
    // the iGPU's ~8 GB USM-shared budget for the hot path (attention).
    // Shared experts and dense FFN stay GPU-dispatch.
    if moe_experts_cpu_enabled() && is_moe_expert_weight(tensor_name) {
        return true;
    }
    CPU_FORCE_PATTERNS.with(|c| {
        let patterns = c.borrow();
        patterns.iter().any(|p| !p.is_empty() && tensor_name.contains(p.as_str()))
    })
}

/// G8: routed-experts-on-CPU placement flag. Initialized once from
/// `RUSTLLAMA_MOE_EXPERTS_CPU`, then mutable via
/// [`set_moe_experts_cpu`] so the tuner sweep can A/B placement modes
/// on a live engine and `[inference].moe_placement = "auto"` can
/// resolve from the tuner cache at load.
fn moe_experts_cpu_cell() -> &'static std::sync::atomic::AtomicBool {
    static ENABLED: std::sync::OnceLock<std::sync::atomic::AtomicBool> =
        std::sync::OnceLock::new();
    ENABLED.get_or_init(|| {
        std::sync::atomic::AtomicBool::new(
            std::env::var("RUSTLLAMA_MOE_EXPERTS_CPU")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
        )
    })
}

pub fn moe_experts_cpu_enabled() -> bool {
    moe_experts_cpu_cell().load(std::sync::atomic::Ordering::Relaxed)
}

/// Toggle routed-experts-on-CPU placement at runtime. Takes effect
/// from the next token's dispatch (consulted per matvec via
/// [`tensor_forced_to_cpu`]).
pub fn set_moe_experts_cpu(enabled: bool) {
    moe_experts_cpu_cell().store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// q★-style CPU/GPU co-execution split for the routed-expert loop, in
/// permille of the top-K picks dispatched to the GPU while the rest
/// run on the CPU concurrently (FreeToken's bandwidth-adaptive
/// co-execution, resolved by measurement rather than the analytic
/// `m·BP/BH` formula — the tuner sweeps end-to-end decode throughput,
/// which folds transfer, dispatch overhead, and memory-bus contention
/// into one honest number; on shared-DRAM iGPUs the sweep is expected
/// to return `0`). `0` (default) disables the split path entirely.
fn moe_gpu_split_cell() -> &'static std::sync::atomic::AtomicU32 {
    static SPLIT: std::sync::OnceLock<std::sync::atomic::AtomicU32> = std::sync::OnceLock::new();
    SPLIT.get_or_init(|| {
        std::sync::atomic::AtomicU32::new(
            std::env::var("RUSTLLAMA_MOE_GPU_SPLIT_PERMILLE")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(0)
                .min(1000),
        )
    })
}

pub fn moe_gpu_split_permille() -> u32 {
    moe_gpu_split_cell().load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_moe_gpu_split_permille(permille: u32) {
    moe_gpu_split_cell().store(permille.min(1000), std::sync::atomic::Ordering::Relaxed);
}

/// G7: MoE LRU expert-cache budget (bytes). Backing cell for
/// [`moe_expert_cache_max_bytes`] — initialized once from
/// `RUSTLLAMA_MOE_EXPERT_CACHE_MB`, then mutable via
/// [`set_moe_expert_cache_max_bytes`] so a budget planner can
/// hot-apply a revised cap without a process restart.
fn moe_expert_cache_cap_cell() -> &'static std::sync::atomic::AtomicU64 {
    static CAP: std::sync::OnceLock<std::sync::atomic::AtomicU64> = std::sync::OnceLock::new();
    CAP.get_or_init(|| {
        std::sync::atomic::AtomicU64::new(
            std::env::var("RUSTLLAMA_MOE_EXPERT_CACHE_MB")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0)
                .saturating_mul(1024 * 1024),
        )
    })
}

/// G7: MoE LRU expert-cache budget (MB env var, bytes returned).
/// The forward path pins the observed-hot experts' weight pages in
/// RAM up to this cap — see the expert-pin cache below. mmap +
/// OS page cache approximates LRU implicitly but is unaware of MoE
/// sparsity (~20% of experts handle ~80% of tokens); the explicit
/// cache catches more decode tokens without disk reads, especially
/// for MoE models that don't fit in RAM.
///
/// Returns the configured cap in bytes; `0` (default) means "no
/// explicit cache, rely on OS paging".
pub fn moe_expert_cache_max_bytes() -> u64 {
    moe_expert_cache_cap_cell().load(std::sync::atomic::Ordering::Relaxed)
}

/// Update the expert-cache budget at runtime. Shrinking does not
/// eagerly evict — the next `expert_pin_touch` over-budget check
/// trims the resident set lazily. `0` disables pinning (existing
/// pins remain until [`expert_pin_clear`] or eviction pressure).
pub fn set_moe_expert_cache_max_bytes(bytes: u64) {
    moe_expert_cache_cap_cell().store(bytes, std::sync::atomic::Ordering::Relaxed);
}

// ============================================================
// MoE LRU expert-pin cache
// ============================================================
//
// On a model too big to keep fully resident (e.g. a 14 GB MoE on
// 16 GB RAM with zero-copy file-backed weights), the OS pager evicts
// expert pages by a MoE-blind global LRU and re-reads them from the
// GGUF on demand — ~20% of experts handle ~80% of tokens, but the OS
// doesn't know that, so hot experts get evicted and re-faulted.
//
// This cache `VirtualLock`s the *observed-hot* experts' weight ranges
// into RAM (so they stop re-faulting), evicting the LRU-coldest
// unreferenced expert when over the `RUSTLLAMA_MOE_EXPERT_CACHE_MB`
// budget. Only zero-copy `MmapBorrowed` experts are pinnable: their
// pages are clean + file-backed, so pinning prevents re-reads while
// the cold tail still streams cleanly from disk. (Pinning owned-heap
// experts would just shrink the pageable pool — counterproductive.)
//
// Budget 0 (default) => inactive (no behavior change). The cache is
// process-global behind a Mutex: concurrent-engine forks each drive
// their own worker thread but share one model (and one budget), so a
// per-thread cache would overcommit the budget N× and strand pins on
// idle threads. Lock cost is negligible — one touch/release pair per
// routed expert (~top-K × layers per token) vs. ms-scale matvecs.
// Pins persist until eviction, [`expert_pin_clear`] (called on model
// reload), or process exit (OS releases them).
//
// Alongside residency, the cache keeps per-(layer, expert) access
// counts — the raw material for the "learning cache": the engine
// persists them to a sidecar next to the GGUF and pre-pins the
// hottest experts at next load (Colibrì-style). Counts are keyed by
// (layer, expert) process-wide; with several MoE models loaded at
// once the counts mix — a documented limitation that can only skew
// perf hints, never correctness (residency keys stay address-based).

/// Raw `VirtualLock`/`VirtualUnlock` for the expert-pin cache. The
/// models crate has no `windows-sys` dep, so we declare the two calls
/// inline (matching the kernels-sycl `usm_pin` pattern). Non-Windows
/// is a no-op (returns "not locked").
#[cfg(windows)]
mod expert_mem_lock {
    extern "system" {
        fn VirtualLock(addr: *const core::ffi::c_void, size: usize) -> i32;
        fn VirtualUnlock(addr: *const core::ffi::c_void, size: usize) -> i32;
    }
    pub fn lock(addr: usize, len: usize) -> bool {
        if addr == 0 || len == 0 {
            return false;
        }
        // SAFETY: addr/len describe a live file-backed weight sub-range
        // owned by the model mmap for the cache entry's lifetime.
        unsafe { VirtualLock(addr as *const core::ffi::c_void, len) != 0 }
    }
    pub fn unlock(addr: usize, len: usize) {
        if addr == 0 || len == 0 {
            return;
        }
        // SAFETY: same range previously locked via `lock`.
        unsafe {
            let _ = VirtualUnlock(addr as *const core::ffi::c_void, len);
        }
    }
}
#[cfg(not(windows))]
mod expert_mem_lock {
    pub fn lock(_addr: usize, _len: usize) -> bool {
        false
    }
    pub fn unlock(_addr: usize, _len: usize) {}
}

/// Identifier for one routed expert: `(layer, expert)`. Drives the
/// telemetry / learning-cache side of the expert-pin cache and the
/// prefetch registry. Residency itself stays keyed by the gate
/// weight's address (unique across models; keys never collide).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert: u32,
}

impl ExpertKey {
    pub fn new(layer: u32, expert: u32) -> Self {
        Self { layer, expert }
    }
}

struct ExpertPin {
    /// (addr, len) for gate / up / down weight sub-ranges, all locked.
    ranges: [(usize, usize); 3],
    bytes: usize,
    /// In-flight matvec readers; eviction only when 0.
    refcount: u32,
    last_touched: u64,
}

struct ExpertPinCache {
    /// Keyed on the gate weight's start address (globally unique per
    /// (layer, expert) — distinct mmap offsets).
    resident: std::collections::HashMap<usize, ExpertPin>,
    pinned_bytes: usize,
    tick: u64,
    /// Pin-cache hit/miss counters (resident vs. not on touch).
    hits: u64,
    misses: u64,
    /// Per-(layer, expert) access counts since the last
    /// [`expert_access_take`] — the learning-cache raw material.
    access: std::collections::HashMap<ExpertKey, u64>,
}

impl ExpertPinCache {
    fn new() -> Self {
        Self {
            resident: std::collections::HashMap::new(),
            pinned_bytes: 0,
            tick: 0,
            hits: 0,
            misses: 0,
            access: std::collections::HashMap::new(),
        }
    }
}

/// Process-global pin cache (see the module comment above for why
/// this is not per-thread).
fn expert_pins() -> &'static std::sync::Mutex<ExpertPinCache> {
    static PINS: std::sync::OnceLock<std::sync::Mutex<ExpertPinCache>> =
        std::sync::OnceLock::new();
    PINS.get_or_init(|| std::sync::Mutex::new(ExpertPinCache::new()))
}

/// Evict-then-pin one expert into `c` under `budget` bytes. Assumes
/// the caller holds the cache lock and has verified the expert is not
/// already resident. Returns `false` (cache state unchanged) when the
/// expert can't fit or a `VirtualLock` fails.
fn pin_expert_locked(
    c: &mut ExpertPinCache,
    key: usize,
    ranges: [(usize, usize); 3],
    budget: usize,
    refcount: u32,
    last_touched: u64,
) -> bool {
    let need: usize = ranges.iter().map(|r| r.1).sum();
    if need > budget {
        return false; // single expert larger than the whole budget
    }
    // Evict LRU-cold (refcount 0) entries until `need` fits.
    while c.pinned_bytes + need > budget {
        let victim = c
            .resident
            .iter()
            .filter(|(_, p)| p.refcount == 0)
            .min_by_key(|(_, p)| p.last_touched)
            .map(|(k, _)| *k);
        let Some(vk) = victim else {
            return false; // everything resident is in-flight; just stream
        };
        let v = c.resident.remove(&vk).expect("victim present");
        // Opt-in: preserve the evicted expert in the secondary spill store
        // (optionally lower-bit) BEFORE unlocking, while its pages are still
        // resident. Default-off: `moe_spill_enabled()` is false → no-op, so
        // the eviction path stays byte-identical.
        if moe_spill_enabled() {
            spill_store_on_evict(vk, &v.ranges);
        }
        for (a, l) in v.ranges {
            expert_mem_lock::unlock(a, l);
        }
        c.pinned_bytes -= v.bytes;
    }
    // Pin all three ranges; roll back on partial failure.
    let mut locked = 0usize;
    let mut ok = true;
    for (a, l) in ranges {
        if expert_mem_lock::lock(a, l) {
            locked += 1;
        } else {
            ok = false;
            break;
        }
    }
    if !ok {
        for (a, l) in ranges.iter().take(locked) {
            expert_mem_lock::unlock(*a, *l);
        }
        return false;
    }
    c.pinned_bytes += need;
    c.resident.insert(
        key,
        ExpertPin {
            ranges,
            bytes: need,
            refcount,
            last_touched,
        },
    );
    static FIRST_PIN: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    FIRST_PIN.get_or_init(|| {
        tracing::info!(
            budget_mb = moe_expert_cache_max_bytes() / (1024 * 1024),
            "MoE expert-pin cache active — VirtualLock'ing hot experts in RAM"
        );
    });
    true
}

/// Touch the MoE expert defined by its three weight tensors before its
/// matvecs. When the expert-cache budget is set and the experts are
/// zero-copy file-backed, this pins (VirtualLock) the hot expert's
/// weight pages into RAM — evicting the LRU-coldest unreferenced expert
/// when over budget — so repeat accesses stop re-faulting from disk.
/// `expert_idx` + the TLS current-layer index feed the per-expert
/// access counts behind the learning cache.
///
/// Returns `Some(key)` when the expert is now pinned (pass it to
/// [`expert_pin_release`] after the matvecs to drop the refcount), or
/// `None` when the cache is inactive (budget 0), the experts aren't
/// file-backed, or it couldn't be pinned within budget (it just
/// streams as before — no release needed).
pub fn expert_pin_touch(
    expert_idx: usize,
    gate: &Tensor,
    up: &Tensor,
    down: &Tensor,
) -> Option<usize> {
    let budget = moe_expert_cache_max_bytes() as usize;
    if budget == 0 {
        return None;
    }
    let ekey = ExpertKey::new(current_layer_idx(), expert_idx as u32);
    let (Some(g), Some(u), Some(d)) = (
        gate.storage.mmap_borrowed_ptr_len(),
        up.storage.mmap_borrowed_ptr_len(),
        down.storage.mmap_borrowed_ptr_len(),
    ) else {
        // Not file-backed → nothing to pin. Still count the access so
        // the learning cache observes routing even before zero-copy is
        // enabled (hit/miss stays untouched: nothing was cacheable).
        let mut c = expert_pins().lock().expect("expert-pin lock");
        *c.access.entry(ekey).or_insert(0) += 1;
        return None;
    };
    let key = g.0 as usize;
    let mut c = expert_pins().lock().expect("expert-pin lock");
    c.tick += 1;
    let now = c.tick;
    *c.access.entry(ekey).or_insert(0) += 1;
    if let Some(p) = c.resident.get_mut(&key) {
        p.refcount += 1;
        p.last_touched = now;
        c.hits += 1;
        return Some(key);
    }
    c.misses += 1;
    // Opt-in fault-in hook: note a resident miss the spill store may be able
    // to serve without a full-precision GGUF re-read (the integrator calls
    // `moe_spill_reconstruct(key)` to fetch the bytes). Default-off:
    // `moe_spill_enabled()` is false → no lock taken, no behavior change.
    if moe_spill_enabled() {
        spill_store_note_fault(key);
    }
    let ranges = [
        (g.0 as usize, g.1),
        (u.0 as usize, u.1),
        (d.0 as usize, d.1),
    ];
    if pin_expert_locked(&mut c, key, ranges, budget, 1, now) {
        Some(key)
    } else {
        None
    }
}

/// Drop the in-flight refcount on a pinned expert after its matvecs.
/// No-op for `None` (expert wasn't pinned).
pub fn expert_pin_release(key: Option<usize>) {
    let Some(key) = key else {
        return;
    };
    let mut c = expert_pins().lock().expect("expert-pin lock");
    if let Some(p) = c.resident.get_mut(&key) {
        p.refcount = p.refcount.saturating_sub(1);
    }
}

/// True when the expert whose gate weight is `w_gate_e` is currently
/// resident (pinned / hot) in the MoE pin cache.
///
/// This makes the disk-spill read-redirect residency-aware: a hot
/// expert's weight pages are VirtualLock'd in RAM, so reconstructing it
/// from the (possibly requantized) spill store is pure overhead —
/// reconstruct only on a residency MISS (a cold/evicted expert). Query
/// this BEFORE [`expert_pin_touch`], which pins (→ makes resident) the
/// expert it touches; querying after would always report a hit. A
/// non-file-backed gate weight was never spill-eligible, so it reports
/// `false` (the redirect no-ops on it anyway — no spill record). The
/// key is the gate weight's mmap start address, exactly as the pin
/// cache + spill store key their entries.
pub fn expert_resident_by_gate(w_gate_e: &Tensor) -> bool {
    let Some((addr, _)) = w_gate_e.storage.mmap_borrowed_ptr_len() else {
        return false;
    };
    let key = addr as usize;
    expert_pins()
        .lock()
        .map(|c| c.resident.contains_key(&key))
        .unwrap_or(false)
}

/// Unlock + drop all pinned experts and reset telemetry (model reload
/// / teardown). Safe to call when inactive. The engine calls this (with
/// [`expert_registry_clear`]) before loading a new model so stale pins
/// from the previous model never outlive their mmap.
pub fn expert_pin_clear() {
    let mut c = expert_pins().lock().expect("expert-pin lock");
    for (_, p) in c.resident.drain() {
        for (a, l) in p.ranges {
            expert_mem_lock::unlock(a, l);
        }
    }
    c.pinned_bytes = 0;
    c.tick = 0;
    c.hits = 0;
    c.misses = 0;
    c.access.clear();
    // Opt-in: also drop the disk-spill secondary store on model reload so
    // stale spilled bytes never outlive their source mmap. Default-off no-op.
    if moe_spill_enabled() {
        drop(c);
        spill_store_reset();
    }
}

/// Point-in-time expert-pin cache telemetry, cheap to snapshot.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExpertCacheStats {
    /// Touches that found the expert already pinned.
    pub hits: u64,
    /// Touches of a pinnable expert that wasn't resident.
    pub misses: u64,
    pub pinned_bytes: u64,
    pub budget_bytes: u64,
    pub resident_experts: usize,
    /// Distinct (layer, expert) pairs observed since the last clear /
    /// access-take.
    pub distinct_experts: usize,
}

impl ExpertCacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Snapshot the pin-cache counters (does not reset anything).
pub fn expert_cache_stats() -> ExpertCacheStats {
    let c = expert_pins().lock().expect("expert-pin lock");
    ExpertCacheStats {
        hits: c.hits,
        misses: c.misses,
        pinned_bytes: c.pinned_bytes as u64,
        budget_bytes: moe_expert_cache_max_bytes(),
        resident_experts: c.resident.len(),
        distinct_experts: c.access.len(),
    }
}

/// Take-and-reset the per-(layer, expert) access counts accumulated
/// since the last take. The engine's learning cache merges these into
/// the model's usage sidecar; taking (rather than reading) keeps
/// repeated flushes from double-counting.
pub fn expert_access_take() -> Vec<(ExpertKey, u64)> {
    let mut c = expert_pins().lock().expect("expert-pin lock");
    c.access.drain().collect()
}

// ============================================================
// MoE disk-spill + lower-bit secondary expert store (opt-in)
// ============================================================
//
// Extends the pin cache's eviction path. Today an evicted expert's weight
// pages simply revert to on-demand re-faults from the *original GGUF*
// (native precision). With this feature armed the bytes are instead copied
// — optionally requantized to a lower bit-width — into a process-local
// spill file, and can be reconstructed (dequantized) on the next routing
// hit, shrinking the resident + re-read cost of a large MoE model's cold
// expert tail.
//
// Fully opt-in and default-OFF: with `RUSTLLAMA_MOE_DISK_SPILL` unset every
// entry point below early-returns before touching any lock, file, or
// allocation, so behavior is byte-identical to a build without it.
//
//   RUSTLLAMA_MOE_DISK_SPILL=1       enable the spill store
//   RUSTLLAMA_MOE_STORE_BITS=<4|8>   store the cold tier requantized to this
//                                    bit-width (default 0 = spill native
//                                    bytes verbatim). Sources whose dtype
//                                    can't be dequantized are spilled native
//                                    regardless (see `dtype_dequant_to_f32`).
//
// Lock discipline: the store has its OWN mutex, always acquired *after*
// (never before) the expert-pin mutex, so the eviction seam — which runs
// while the caller holds the pin lock — is deadlock-free. Reconstruction and
// telemetry take only the store lock; the per-part metadata map (read at the
// eviction seam, written at load) is never held across the store lock.

struct MoeSpillConfig {
    disk_spill: bool,
    /// 0 = spill native bytes; 4 or 8 = requantize the cold tier.
    store_bits: u8,
}

fn moe_spill_config() -> &'static MoeSpillConfig {
    static CFG: std::sync::OnceLock<MoeSpillConfig> = std::sync::OnceLock::new();
    CFG.get_or_init(|| {
        let disk_spill = std::env::var("RUSTLLAMA_MOE_DISK_SPILL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let store_bits = match std::env::var("RUSTLLAMA_MOE_STORE_BITS")
            .ok()
            .and_then(|s| s.trim().parse::<u8>().ok())
        {
            Some(4) => 4,
            Some(8) => 8,
            // Unset / unsupported width → keep native (verbatim) spill.
            _ => 0,
        };
        MoeSpillConfig {
            disk_spill,
            store_bits,
        }
    })
}

/// Whether the MoE disk-spill store is enabled (`RUSTLLAMA_MOE_DISK_SPILL`).
/// The single gate every spill entry point checks first; `false` (default)
/// makes the whole feature inert (no files, no locks, no allocations).
pub fn moe_spill_enabled() -> bool {
    moe_spill_config().disk_spill
}

/// Lower-bit width for the cold tier (`RUSTLLAMA_MOE_STORE_BITS`): 0 means
/// spill native bytes verbatim, 4 / 8 requantize on spill.
pub fn moe_spill_store_bits() -> u8 {
    moe_spill_config().store_bits
}

static SPILL_DIR: std::sync::OnceLock<std::sync::RwLock<Option<std::path::PathBuf>>> =
    std::sync::OnceLock::new();

/// Point the spill file at `dir` (the policy layer passes the runtime cache
/// dir). Consulted lazily on the first spill, so call it before the first
/// eviction. When unset, the store falls back to a `rustllama-moe-spill`
/// folder under the OS temp dir. Idempotent.
pub fn set_moe_spill_dir(dir: std::path::PathBuf) {
    let cell = SPILL_DIR.get_or_init(|| std::sync::RwLock::new(None));
    if let Ok(mut w) = cell.write() {
        *w = Some(dir);
    }
}

fn spill_dir() -> std::path::PathBuf {
    if let Some(cell) = SPILL_DIR.get() {
        if let Ok(r) = cell.read() {
            if let Some(d) = r.as_ref() {
                return d.clone();
            }
        }
    }
    std::env::temp_dir().join("rustllama-moe-spill")
}

/// Per-part source metadata for lower-bit requantization, keyed by the gate
/// weight's start address (the exact key the pin cache uses). Filled at load
/// by [`register_layer_experts`] ONLY when lower-bit spill is armed; empty
/// otherwise (native spill needs no dtype).
#[derive(Clone, Copy)]
struct SpillPartMeta {
    dtype: Dtype,
    n_elems: usize,
}

fn spill_part_meta(
) -> &'static std::sync::RwLock<std::collections::HashMap<usize, [SpillPartMeta; 3]>> {
    static M: std::sync::OnceLock<
        std::sync::RwLock<std::collections::HashMap<usize, [SpillPartMeta; 3]>>,
    > = std::sync::OnceLock::new();
    M.get_or_init(|| std::sync::RwLock::new(std::collections::HashMap::new()))
}

#[derive(Clone, Copy)]
enum SpillEncoding {
    /// Verbatim native bytes (exact round-trip).
    Native,
    /// Requantized to `bits` (4 or 8) via the block codec.
    LowBit { bits: u8 },
}

#[derive(Clone, Copy)]
struct SpillPartRec {
    offset: u64,
    stored_len: usize,
    /// Original element count — needed to bound lower-bit decode.
    n_elems: usize,
    /// Original source dtype (carried through for the integrator's context).
    dtype: Dtype,
    encoding: SpillEncoding,
}

struct SpillStore {
    file: Option<std::fs::File>,
    write_off: u64,
    /// gate-addr key → the three (gate, up, down) part records.
    records: std::collections::HashMap<usize, [SpillPartRec; 3]>,
    spilled_bytes: u64,
    evictions_spilled: u64,
    faults_seen: u64,
    reconstructions: u64,
}

impl SpillStore {
    fn new() -> Self {
        Self {
            file: None,
            write_off: 0,
            records: std::collections::HashMap::new(),
            spilled_bytes: 0,
            evictions_spilled: 0,
            faults_seen: 0,
            reconstructions: 0,
        }
    }

    /// Lazily create (truncating) the spill file. Returns `None` on any I/O
    /// error — the caller then simply skips the record, so the expert
    /// re-faults from the GGUF exactly as it does today (never a
    /// correctness problem, only a lost optimization).
    fn ensure_file(&mut self) -> Option<&mut std::fs::File> {
        if self.file.is_none() {
            let dir = spill_dir();
            if std::fs::create_dir_all(&dir).is_err() {
                return None;
            }
            // PID-tagged so concurrent processes never share one store file.
            let path = dir.join(format!("moe-spill-{}.rlspill", std::process::id()));
            let f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)
                .ok()?;
            tracing::debug!(path = %path.display(), "MoE spill store file created");
            self.file = Some(f);
            self.write_off = 0;
        }
        self.file.as_mut()
    }

    fn reset(&mut self) {
        self.records.clear();
        self.spilled_bytes = 0;
        self.write_off = 0;
        if let Some(f) = self.file.as_mut() {
            use std::io::{Seek, SeekFrom};
            let _ = f.set_len(0);
            let _ = f.seek(SeekFrom::Start(0));
        }
    }
}

fn spill_store() -> &'static std::sync::Mutex<SpillStore> {
    static S: std::sync::OnceLock<std::sync::Mutex<SpillStore>> = std::sync::OnceLock::new();
    S.get_or_init(|| std::sync::Mutex::new(SpillStore::new()))
}

/// Block size for the self-contained lower-bit spill codec. 32 matches the
/// classic Q*_0 block so per-block scale overhead stays ~one f32.
const SPILL_BLOCK: usize = 32;

fn spill_quant_one(v: f32, d: f32, qmax: i32) -> i32 {
    if d > 0.0 {
        (v / d).round().clamp(-(qmax as f32), qmax as f32) as i32
    } else {
        0
    }
}

/// Symmetric per-block quantize `values` to `bits` (4 or 8). Layout per
/// block: little-endian f32 scale, then the packed quants (one i8/weight for
/// 8-bit; two signed nibbles/byte, biased by +8, for 4-bit). Round-trips
/// within one quant step — a lossy *secondary* copy, exactly like the
/// on-disk cold tier it stands in for.
fn spill_encode_lowbits(values: &[f32], bits: u8) -> Vec<u8> {
    let qmax = ((1i32 << (bits - 1)) - 1).max(1);
    let mut out = Vec::with_capacity(values.len() + values.len() / 8 + 8);
    for blk in values.chunks(SPILL_BLOCK) {
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = if amax > 0.0 { amax / qmax as f32 } else { 0.0 };
        out.extend_from_slice(&d.to_le_bytes());
        if bits == 8 {
            for &v in blk {
                out.push(spill_quant_one(v, d, qmax) as i8 as u8);
            }
        } else {
            let mut i = 0;
            while i < blk.len() {
                let q0 = spill_quant_one(blk[i], d, qmax);
                let q1 = if i + 1 < blk.len() {
                    spill_quant_one(blk[i + 1], d, qmax)
                } else {
                    0
                };
                let n0 = ((q0 + 8) & 0xF) as u8;
                let n1 = ((q1 + 8) & 0xF) as u8;
                out.push(n0 | (n1 << 4));
                i += 2;
            }
        }
    }
    out
}

/// Inverse of [`spill_encode_lowbits`]. Reconstructs exactly `n_elems`
/// values (block boundaries are recomputed from the remaining count, so the
/// short trailing block matches the encoder). Truncated / short input decodes
/// to the values it can and zero-fills the rest.
fn spill_decode_lowbits(bytes: &[u8], bits: u8, n_elems: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_elems);
    let mut off = 0usize;
    while out.len() < n_elems && off + 4 <= bytes.len() {
        let d = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        off += 4;
        let this = (n_elems - out.len()).min(SPILL_BLOCK);
        if bits == 8 {
            if off + this > bytes.len() {
                break;
            }
            for i in 0..this {
                out.push((bytes[off + i] as i8 as f32) * d);
            }
            off += this;
        } else {
            let nbytes = this.div_ceil(2);
            if off + nbytes > bytes.len() {
                break;
            }
            for i in 0..this {
                let byte = bytes[off + i / 2];
                let nib = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
                out.push(((nib as i32) - 8) as f32 * d);
            }
            off += nbytes;
        }
    }
    out.resize(n_elems, 0.0);
    out
}

/// Dequantize `bytes` of source `dtype` (as stored in the GGUF mmap) to f32.
/// Returns `None` for dtypes we deliberately do NOT requantize (exotic IQ /
/// ternary / fp4 codebook formats, whose extra quant step would compound the
/// error without a size win worth the risk) — the caller then spills those
/// native. Covers the dtypes MoE routed-expert weights realistically use.
fn dtype_dequant_to_f32(dtype: Dtype, bytes: &[u8], n_elems: usize) -> Option<Vec<f32>> {
    use rustllama_gguf::dequant as dq;
    let mut out = vec![0f32; n_elems];
    match dtype {
        Dtype::F32 => {
            for (i, c) in bytes.chunks_exact(4).take(n_elems).enumerate() {
                out[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        Dtype::F16 => dq::dequant_f16(bytes, &mut out),
        Dtype::Bf16 | Dtype::Bf16Raw => dq::dequant_bf16(bytes, &mut out),
        Dtype::Q8_0 | Dtype::Q8_0Raw => dq::dequant_q8_0(bytes, &mut out),
        Dtype::Q4K | Dtype::Q4_KRaw => dq::dequant_q4_k(bytes, &mut out),
        Dtype::Q5K | Dtype::Q5_KRaw => dq::dequant_q5_k(bytes, &mut out),
        Dtype::Q6K | Dtype::Q6_KRaw => dq::dequant_q6_k(bytes, &mut out),
        Dtype::Q4_0Raw => dq::dequant_q4_0(bytes, &mut out),
        Dtype::Q5_0Raw => dq::dequant_q5_0(bytes, &mut out),
        Dtype::Q4_1Raw => dq::dequant_q4_1(bytes, &mut out),
        Dtype::Q5_1Raw => dq::dequant_q5_1(bytes, &mut out),
        Dtype::Q2_KRaw => dq::dequant_q2_k(bytes, &mut out),
        Dtype::Q3_KRaw => dq::dequant_q3_k(bytes, &mut out),
        Dtype::IQ4_XSRaw => dq::dequant_iq4_xs(bytes, &mut out),
        Dtype::IQ4_NLRaw => dq::dequant_iq4_nl(bytes, &mut out),
        // Exotic codebook / ternary / fp4 formats → spill native instead.
        _ => return None,
    }
    Some(out)
}

/// Eviction → spill. Copies (optionally requantizes) the victim expert's
/// three weight sub-ranges into the spill file and records where. Called from
/// [`pin_expert_locked`]'s eviction loop while the pin lock is held AND the
/// victim's pages are still resident, so the raw reads are valid. Best-effort:
/// any I/O failure just skips the record (the expert falls back to a GGUF
/// re-fault). Guarded by [`moe_spill_enabled`] at the call site.
fn spill_store_on_evict(gate_key: usize, ranges: &[(usize, usize); 3]) {
    let want_bits = moe_spill_store_bits();
    // Per-part source metadata is present only when lower-bit is armed and
    // the loader registered this expert; missing → native spill.
    let meta = if want_bits != 0 {
        spill_part_meta()
            .read()
            .ok()
            .and_then(|m| m.get(&gate_key).copied())
    } else {
        None
    };
    let mut store = match spill_store().lock() {
        Ok(s) => s,
        Err(_) => return,
    };
    // Already spilled (re-eviction of a re-pinned expert)? The bytes are
    // identical across evictions, so keep the first record.
    if store.records.contains_key(&gate_key) {
        return;
    }
    let mut recs: [SpillPartRec; 3] = [SpillPartRec {
        offset: 0,
        stored_len: 0,
        n_elems: 0,
        dtype: Dtype::F32,
        encoding: SpillEncoding::Native,
    }; 3];
    let mut ok = true;
    for i in 0..3 {
        let (addr, len) = ranges[i];
        if addr == 0 || len == 0 {
            ok = false;
            break;
        }
        // SAFETY: the victim is resident (VirtualLock'd) at this seam and the
        // GGUF mmap outlives it; addr/len describe a live weight sub-range,
        // the same invariant `expert_mem_lock::unlock` relies on below.
        let src: &[u8] = unsafe { std::slice::from_raw_parts(addr as *const u8, len) };
        let pm = meta.map(|m| m[i]);
        let (payload, enc, dtype, n_elems): (std::borrow::Cow<[u8]>, SpillEncoding, Dtype, usize) =
            match pm {
                Some(pm) if want_bits != 0 => match dtype_dequant_to_f32(pm.dtype, src, pm.n_elems) {
                    Some(f32s) => (
                        std::borrow::Cow::Owned(spill_encode_lowbits(&f32s, want_bits)),
                        SpillEncoding::LowBit { bits: want_bits },
                        pm.dtype,
                        pm.n_elems,
                    ),
                    None => (
                        std::borrow::Cow::Borrowed(src),
                        SpillEncoding::Native,
                        pm.dtype,
                        pm.n_elems,
                    ),
                },
                _ => (
                    std::borrow::Cow::Borrowed(src),
                    SpillEncoding::Native,
                    Dtype::F32,
                    0,
                ),
            };
        let off = store.write_off;
        let stored = payload.len();
        let write_ok = {
            let payload_ref: &[u8] = &payload;
            match store.ensure_file() {
                Some(f) => {
                    use std::io::{Seek, SeekFrom, Write};
                    f.seek(SeekFrom::Start(off)).is_ok() && f.write_all(payload_ref).is_ok()
                }
                None => false,
            }
        };
        if !write_ok {
            ok = false;
            break;
        }
        store.write_off += stored as u64;
        store.spilled_bytes += stored as u64;
        recs[i] = SpillPartRec {
            offset: off,
            stored_len: stored,
            n_elems,
            dtype,
            encoding: enc,
        };
    }
    if ok {
        store.records.insert(gate_key, recs);
        store.evictions_spilled += 1;
    }
}

/// Routing-miss → fault-in telemetry hook. Called from the miss branch of
/// [`expert_pin_touch`] when the store is enabled: records that a resident
/// miss occurred for an expert we may hold a spilled copy of. The actual byte
/// reconstruction is [`moe_spill_reconstruct`], which the MoE forward /
/// integrator calls to obtain the expert's weights without a full-precision
/// GGUF re-read. Kept cheap (one lock + lookup) so the decode miss path isn't
/// burdened.
fn spill_store_note_fault(gate_key: usize) {
    if let Ok(mut store) = spill_store().lock() {
        if store.records.contains_key(&gate_key) {
            store.faults_seen += 1;
        }
    }
}

/// Drop all spilled records + truncate the spill file (model reload). Called
/// from [`expert_pin_clear`] under the same enable gate.
fn spill_store_reset() {
    if let Ok(mut store) = spill_store().lock() {
        store.reset();
    }
    if let Ok(mut m) = spill_part_meta().write() {
        m.clear();
    }
}

/// One reconstructed expert weight part, as handed back on fault-in.
pub enum SpilledPart {
    /// Verbatim native bytes (source `dtype` preserved) — blit straight back.
    Native { bytes: Vec<u8>, dtype: Dtype },
    /// Dequantized f32 values (the source was requantized to a lower
    /// bit-width on spill); `dtype` is the ORIGINAL source dtype.
    Dequant { values: Vec<f32>, dtype: Dtype },
}

/// A spilled expert reconstructed from the secondary store: gate, up, down in
/// that order.
pub struct SpilledExpert {
    pub parts: Vec<SpilledPart>,
}

/// Reconstruct a previously-spilled expert from the secondary store, keyed by
/// its gate weight's start address (`gate.storage.mmap_borrowed_ptr_len().0
/// as usize` — the same key the pin cache uses). Returns `None` when the
/// store is disabled, the expert was never spilled, or a read fails (the
/// caller then falls back to the GGUF as before). Reproduces the native bytes
/// exactly, or the source weights within one lower-bit quant step.
pub fn moe_spill_reconstruct(gate_key: usize) -> Option<SpilledExpert> {
    if !moe_spill_enabled() {
        return None;
    }
    let mut store = spill_store().lock().ok()?;
    let recs = *store.records.get(&gate_key)?;
    let mut parts = Vec::with_capacity(3);
    for rec in recs.iter() {
        let mut buf = vec![0u8; rec.stored_len];
        {
            use std::io::{Read, Seek, SeekFrom};
            let f = store.file.as_mut()?;
            if f.seek(SeekFrom::Start(rec.offset)).is_err() || f.read_exact(&mut buf).is_err() {
                return None;
            }
        }
        let part = match rec.encoding {
            SpillEncoding::Native => SpilledPart::Native {
                bytes: buf,
                dtype: rec.dtype,
            },
            SpillEncoding::LowBit { bits } => SpilledPart::Dequant {
                values: spill_decode_lowbits(&buf, bits, rec.n_elems),
                dtype: rec.dtype,
            },
        };
        parts.push(part);
    }
    store.reconstructions += 1;
    Some(SpilledExpert { parts })
}

/// Point-in-time spill-store telemetry (separate from [`ExpertCacheStats`] so
/// the existing struct's shape is untouched).
#[derive(Debug, Clone, Copy, Default)]
pub struct MoeSpillStats {
    pub enabled: bool,
    pub store_bits: u8,
    pub records: usize,
    pub spilled_bytes: u64,
    pub evictions_spilled: u64,
    pub faults_seen: u64,
    pub reconstructions: u64,
}

/// Snapshot the spill-store counters. Cheap; takes no lock when disabled.
pub fn moe_spill_stats() -> MoeSpillStats {
    if !moe_spill_enabled() {
        return MoeSpillStats {
            store_bits: moe_spill_store_bits(),
            ..Default::default()
        };
    }
    let store = spill_store().lock().expect("moe spill store lock");
    MoeSpillStats {
        enabled: true,
        store_bits: moe_spill_store_bits(),
        records: store.records.len(),
        spilled_bytes: store.spilled_bytes,
        evictions_spilled: store.evictions_spilled,
        faults_seen: store.faults_seen,
        reconstructions: store.reconstructions,
    }
}

// ---------------------------------------------------------------
// Expert range registry + pre-pin + async readahead
// ---------------------------------------------------------------

/// The three locked-range candidates for one expert: gate/up/down
/// `(addr, len)` byte ranges inside the GGUF mmap.
type ExpertRangeTriple = [(usize, usize); 3];
type ExpertRangeMap = std::collections::HashMap<ExpertKey, ExpertRangeTriple>;

/// (layer, expert) → the three locked-range candidates. Filled at
/// model load by [`register_layer_experts`] when the expert views are
/// file-backed; consumed by [`expert_prepin`] (learning-cache
/// pre-pin) and the readahead thread (prefetch by key without
/// touching tensors).
fn expert_ranges() -> &'static std::sync::RwLock<ExpertRangeMap> {
    static RANGES: std::sync::OnceLock<std::sync::RwLock<ExpertRangeMap>> =
        std::sync::OnceLock::new();
    RANGES.get_or_init(|| std::sync::RwLock::new(ExpertRangeMap::new()))
}

/// Whole-layer expert-pool ranges: layer → the three *fused* tensor
/// spans (gate/up/down across every expert). Fed by
/// [`register_layer_experts`] (per-expert slices are consecutive
/// carves of the fused tensor, so the union is one contiguous span
/// per tensor); consumed by [`note_prefill_layer`] — prefill touches
/// (nearly) every expert of a layer, so the layer-ahead readahead
/// streams the whole pool in three coalesced ranges.
fn layer_pool_ranges() -> &'static std::sync::RwLock<std::collections::HashMap<u32, ExpertRangeTriple>> {
    static POOLS: std::sync::OnceLock<
        std::sync::RwLock<std::collections::HashMap<u32, ExpertRangeTriple>>,
    > = std::sync::OnceLock::new();
    POOLS.get_or_init(|| std::sync::RwLock::new(std::collections::HashMap::new()))
}

/// Register one MoE layer's per-expert weight ranges for prefetch /
/// pre-pin. Called from the model loader right after the per-expert
/// views are carved. No-ops per expert unless all three views are
/// zero-copy file-backed (`MmapBorrowed`) — owned-heap experts are
/// always resident and need neither service.
pub fn register_layer_experts(layer: u32, gate: &[Tensor], up: &[Tensor], down: &[Tensor]) {
    // Opt-in lower-bit spill: capture per-part source dtype/shape so the
    // eviction seam can requantize. Only armed when both the spill store and
    // a store-bits width are set — otherwise no map, no allocation, so this
    // function stays byte-identical to today.
    let arm_spill_meta = moe_spill_enabled() && moe_spill_store_bits() != 0;
    let mut spill_metas: Vec<(usize, [SpillPartMeta; 3])> = Vec::new();
    let mut map = expert_ranges().write().expect("expert-range lock");
    let n = gate.len().min(up.len()).min(down.len());
    // Whole-pool span accumulators: (min addr, max end) per tensor.
    let mut pool: Option<[(usize, usize); 3]> = None;
    let mut all_backed = true;
    for e in 0..n {
        let (Some(g), Some(u), Some(d)) = (
            gate[e].storage.mmap_borrowed_ptr_len(),
            up[e].storage.mmap_borrowed_ptr_len(),
            down[e].storage.mmap_borrowed_ptr_len(),
        ) else {
            all_backed = false;
            continue;
        };
        let triple = [(g.0 as usize, g.1), (u.0 as usize, u.1), (d.0 as usize, d.1)];
        map.insert(ExpertKey::new(layer, e as u32), triple);
        if arm_spill_meta {
            spill_metas.push((
                g.0 as usize,
                [
                    SpillPartMeta {
                        dtype: gate[e].dtype,
                        n_elems: gate[e].element_count() as usize,
                    },
                    SpillPartMeta {
                        dtype: up[e].dtype,
                        n_elems: up[e].element_count() as usize,
                    },
                    SpillPartMeta {
                        dtype: down[e].dtype,
                        n_elems: down[e].element_count() as usize,
                    },
                ],
            ));
        }
        pool = Some(match pool {
            None => triple.map(|(a, l)| (a, a + l)),
            Some(mut p) => {
                for i in 0..3 {
                    p[i].0 = p[i].0.min(triple[i].0);
                    p[i].1 = p[i].1.max(triple[i].0 + triple[i].1);
                }
                p
            }
        });
    }
    drop(map);
    if arm_spill_meta && !spill_metas.is_empty() {
        let mut mm = spill_part_meta().write().expect("spill-meta lock");
        for (k, v) in spill_metas {
            mm.insert(k, v);
        }
    }
    if all_backed {
        if let Some(p) = pool {
            layer_pool_ranges()
                .write()
                .expect("layer-pool lock")
                .insert(layer, p.map(|(start, end)| (start, end - start)));
        }
    }
}

/// Number of (layer, expert) pairs currently registered. `0` with the
/// cache enabled means the experts aren't file-backed (zero-copy off)
/// — the engine warns on that combination at load.
pub fn expert_registry_len() -> usize {
    expert_ranges().read().expect("expert-range lock").len()
}

/// Total bytes across every registered expert's three weight ranges —
/// the size of the routed-expert pool the cache could ever hold.
/// Upper-bounds the useful expert-cache budget for the memory-budget
/// planner (pinning more than the pool is pure waste).
pub fn expert_registry_total_bytes() -> u64 {
    expert_ranges()
        .read()
        .expect("expert-range lock")
        .values()
        .map(|r| r.iter().map(|x| x.1 as u64).sum::<u64>())
        .sum()
}

/// Drop all registered ranges (model reload / teardown).
/// Dense per-layer weight spans for layer-ahead readahead: the
/// coalesced file-backed (addr, len) ranges of every weight tensor
/// in one transformer layer. FreeToken's double-buffered prefill
/// streaming, applied to DENSE weights: unlike MoE experts there is
/// no usage pattern to learn (every weight is touched every token),
/// but the ACCESS ORDER is perfectly sequential by layer — so under
/// memory pressure, prefetching layer L+1's spans while L computes
/// converts random-eviction page-cache thrash into overlapped
/// sequential reads. Advisory and lossy, like the expert readahead.
fn dense_layer_ranges(
) -> &'static std::sync::RwLock<std::collections::HashMap<u32, Vec<(usize, usize)>>> {
    static M: std::sync::OnceLock<
        std::sync::RwLock<std::collections::HashMap<u32, Vec<(usize, usize)>>>,
    > = std::sync::OnceLock::new();
    M.get_or_init(|| std::sync::RwLock::new(std::collections::HashMap::new()))
}

/// Register one layer's dense weight tensors for layer-ahead
/// readahead. Non-file-backed tensors (owned copies) are skipped —
/// they can't be evicted, so there is nothing to prefetch. Adjacent
/// spans (gap <= 2 MiB) are coalesced so the prefetch call count
/// stays small.
pub fn register_layer_dense_ranges(layer: u32, tensors: &[&Tensor]) {
    let mut spans: Vec<(usize, usize)> = tensors
        .iter()
        .filter_map(|t| t.storage.mmap_borrowed_ptr_len())
        .map(|(p, l)| (p as usize, l))
        .collect();
    if spans.is_empty() {
        return;
    }
    spans.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
    for (a, l) in spans {
        match merged.last_mut() {
            Some((ma, ml)) if a <= *ma + *ml + (2 << 20) => {
                let end = (a + l).max(*ma + *ml);
                *ml = end - *ma;
            }
            _ => merged.push((a, l)),
        }
    }
    dense_layer_ranges()
        .write()
        .expect("dense-range lock")
        .insert(layer, merged);
}

/// Number of layers with registered dense readahead spans.
pub fn dense_registry_len() -> usize {
    dense_layer_ranges().read().expect("dense-range lock").len()
}

/// `RUSTLLAMA_DENSE_READAHEAD=0` disables the dense layer-ahead
/// prefetch (default ON — advisory hints on resident pages are
/// near-free, and the win appears exactly when RAM pressure starts
/// evicting weight pages mid-sweep).
fn dense_readahead_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("RUSTLLAMA_DENSE_READAHEAD").as_deref() != Ok("0"))
}

pub fn expert_registry_clear() {
    dense_layer_ranges()
        .write()
        .expect("dense-range lock")
        .clear();
    expert_ranges().write().expect("expert-range lock").clear();
    layer_pool_ranges().write().expect("layer-pool lock").clear();
}

/// Pre-pin `ranked` experts (hottest first) up to `max_bytes`,
/// resolving ranges through the registry. Called by the engine at
/// load with the learning-cache ranking, before the first prefill.
/// Pins are inserted refcount-0 (immediately evictable under
/// pressure) with LRU ticks ordered so the hottest expert is the
/// last eviction candidate. Returns `(experts_pinned, bytes_pinned)`.
pub fn expert_prepin(ranked: &[ExpertKey], max_bytes: u64) -> (usize, u64) {
    let budget = moe_expert_cache_max_bytes() as usize;
    let cap = (max_bytes as usize).min(budget);
    if cap == 0 || ranked.is_empty() {
        return (0, 0);
    }
    let ranges_map = expert_ranges().read().expect("expert-range lock");
    // Select the hottest prefix that fits the cap, then pin it in
    // reverse (coldest→hottest) so the hottest carries the highest
    // LRU tick and is evicted last.
    let mut selected: Vec<(usize, ExpertRangeTriple, usize)> = Vec::new();
    let mut planned = 0usize;
    for k in ranked {
        let Some(r) = ranges_map.get(k) else { continue };
        let need: usize = r.iter().map(|x| x.1).sum();
        if planned + need > cap {
            break; // experts are uniform-sized; the first overflow ends the fill
        }
        planned += need;
        selected.push((r[0].0, *r, need));
    }
    drop(ranges_map);
    let mut c = expert_pins().lock().expect("expert-pin lock");
    let mut pinned = 0usize;
    let mut bytes = 0u64;
    for (key, ranges, need) in selected.into_iter().rev() {
        if c.resident.contains_key(&key) {
            continue;
        }
        c.tick += 1;
        let now = c.tick;
        if pin_expert_locked(&mut c, key, ranges, budget, 0, now) {
            pinned += 1;
            bytes += need as u64;
        } else {
            // VirtualLock refusal here means the working-set quota is
            // exhausted — further attempts will fail the same way.
            break;
        }
    }
    (pinned, bytes)
}

/// Advisory `PrefetchVirtualMemory` for the readahead thread. The
/// models crate has no `windows-sys` dep, so the two calls are
/// declared inline (same pattern as `expert_mem_lock`). Non-Windows
/// is a no-op.
#[cfg(windows)]
mod expert_prefetch_sys {
    #[repr(C)]
    struct Win32MemoryRangeEntry {
        virtual_address: *mut core::ffi::c_void,
        number_of_bytes: usize,
    }
    extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
        fn PrefetchVirtualMemory(
            h_process: *mut core::ffi::c_void,
            number_of_entries: usize,
            virtual_addresses: *const Win32MemoryRangeEntry,
            flags: u32,
        ) -> i32;
    }
    /// Best-effort prefetch of the given ranges in one syscall. The
    /// kernel validates the ranges itself, so a stale entry (model
    /// unloaded mid-flight) fails the call harmlessly — nothing is
    /// dereferenced from user code.
    pub fn prefetch(regions: &[(usize, usize)]) {
        if regions.is_empty() {
            return;
        }
        let entries: Vec<Win32MemoryRangeEntry> = regions
            .iter()
            .map(|&(addr, len)| Win32MemoryRangeEntry {
                virtual_address: addr as *mut core::ffi::c_void,
                number_of_bytes: len,
            })
            .collect();
        // SAFETY: entries/count are consistent; the kernel validates
        // the addresses (advisory call, no user-side dereference).
        unsafe {
            let _ = PrefetchVirtualMemory(GetCurrentProcess(), entries.len(), entries.as_ptr(), 0);
        }
    }
}
#[cfg(not(windows))]
mod expert_prefetch_sys {
    pub fn prefetch(_regions: &[(usize, usize)]) {}
}

/// `RUSTLLAMA_MOE_EXPERT_READAHEAD` — default ON (the readahead only
/// runs when the expert cache itself is enabled). `0`/`false` opts out.
fn expert_readahead_enabled() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("RUSTLLAMA_MOE_EXPERT_READAHEAD")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(true)
    })
}

/// Readahead request: either a set of specific experts (decode-time
/// routing hints) or a whole layer's fused expert pool (prefill
/// layer-ahead streaming — prefill touches nearly every expert of a
/// layer, so three coalesced whole-tensor ranges beat hundreds of
/// per-expert ones).
enum PrefetchReq {
    Experts(Vec<ExpertKey>),
    LayerPool(u32),
    DenseLayer(u32),
}

/// Sender to the readahead thread, spawned lazily on first use. The
/// channel is small and lossy on purpose: prefetch is advisory, so
/// when the disk can't keep up we drop requests instead of queueing
/// stale ones behind the decode loop.
fn expert_prefetch_tx() -> &'static std::sync::mpsc::SyncSender<PrefetchReq> {
    static TX: std::sync::OnceLock<std::sync::mpsc::SyncSender<PrefetchReq>> =
        std::sync::OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<PrefetchReq>(64);
        std::thread::Builder::new()
            .name("rl-expert-readahead".into())
            .spawn(move || {
                while let Ok(req) = rx.recv() {
                    match req {
                        PrefetchReq::Experts(keys) => {
                            // Resolve ranges for keys that aren't already
                            // pinned; the OS page cache dedupes repeat
                            // prefetches of resident pages cheaply.
                            let mut regions: Vec<(usize, usize)> =
                                Vec::with_capacity(keys.len() * 3);
                            {
                                let ranges_map =
                                    expert_ranges().read().expect("expert-range lock");
                                let pins = expert_pins().lock().expect("expert-pin lock");
                                for k in keys {
                                    let Some(r) = ranges_map.get(&k) else { continue };
                                    if pins.resident.contains_key(&r[0].0) {
                                        continue; // pinned ⇒ already resident
                                    }
                                    regions.extend_from_slice(r);
                                }
                            }
                            expert_prefetch_sys::prefetch(&regions);
                        }
                        PrefetchReq::DenseLayer(layer) => {
                            let regions: Vec<(usize, usize)> = dense_layer_ranges()
                                .read()
                                .expect("dense-range lock")
                                .get(&layer)
                                .cloned()
                                .unwrap_or_default();
                            if !regions.is_empty() {
                                expert_prefetch_sys::prefetch(&regions);
                            }
                        }
                        PrefetchReq::LayerPool(layer) => {
                            let regions = layer_pool_ranges()
                                .read()
                                .expect("layer-pool lock")
                                .get(&layer)
                                .map(|r| r.to_vec());
                            if let Some(regions) = regions {
                                expert_prefetch_sys::prefetch(&regions);
                            }
                        }
                    }
                }
            })
            .expect("spawn expert readahead thread");
        tx
    })
}

/// Layer-ahead prefill readahead (FreeToken's double-buffered prefill
/// streaming, downgraded correctly for a shared-memory host: it's
/// disk → page cache, not RAM → VRAM). Called by the chunk-prefill
/// forward at the start of each MoE layer with `layer + 1`: the next
/// layer's whole expert pool starts faulting in from disk while this
/// layer computes — and it can start before that layer's routing is
/// known, because a big-enough chunk touches nearly every expert.
/// Advisory and lossy like every other readahead hint.
pub fn note_prefill_layer(layer: u32) {
    // Dense layer-ahead hint (prefill AND decode call sites): fires
    // whenever dense spans are registered, independent of the MoE
    // expert cache. Cheap early-outs keep this ~free per layer.
    if dense_readahead_enabled() && dense_registry_len() != 0 {
        let _ = expert_prefetch_tx().try_send(PrefetchReq::DenseLayer(layer));
    }
    if moe_expert_cache_max_bytes() == 0 || !expert_readahead_enabled() {
        return;
    }
    let _ = expert_prefetch_tx().try_send(PrefetchReq::LayerPool(layer));
}

/// Hint the readahead thread about this token's routed experts.
/// Called from the MoE forward right after routing, before the first
/// expert matvec: prefetches the picked experts for the current layer
/// (covers picks 2..K while pick 1 computes) plus the same expert
/// indices for the next two layers (routing correlates across
/// adjacent layers). Never blocks — a full channel drops the hint.
pub fn note_routed_experts(picks: &[(usize, f32)]) {
    if moe_expert_cache_max_bytes() == 0 || !expert_readahead_enabled() {
        return;
    }
    let layer = current_layer_idx();
    let mut keys: Vec<ExpertKey> = Vec::with_capacity(picks.len() * 3);
    for l in layer..layer.saturating_add(3) {
        for (e, _) in picks {
            keys.push(ExpertKey::new(l, *e as u32));
        }
    }
    let _ = expert_prefetch_tx().try_send(PrefetchReq::Experts(keys));
}

/// G8: recognise MoE per-expert FFN tensor names. GGUF conventions
/// across Mixtral / Qwen-MoE / DeepSeek-MoE / Qwen3.5-MoE pin the
/// routed-expert weights to one of these substring patterns. Shared
/// expert and routing softmax stay on the GPU path.
fn is_moe_expert_weight(tensor_name: &str) -> bool {
    tensor_name.contains("ffn_gate_exps")
        || tensor_name.contains("ffn_up_exps")
        || tensor_name.contains("ffn_down_exps")
        || tensor_name.contains("ffn_gate_inp") // router gate (tiny matvec; CPU is fine)
}

/// Set the per-thread FlashAttention-decode flag. Engine generate
/// paths call this at the start of each request with the current
/// `[inference].flash_attention` config value. The previous value
/// is unused — callers don't need to restore it because the engine
/// always sets it explicitly per call.
pub fn set_flash_attention(enabled: bool) {
    FLASH_ATTENTION.with(|f| f.set(enabled));
}

/// Read the per-thread FlashAttention-decode flag. The attention
/// dispatch in `llama_arch::forward_one` consults this to choose
/// between [`rustllama_kernels_cpu::gqa_attention_one_step`] (the
/// standard 3-pass implementation) and
/// [`rustllama_kernels_cpu::gqa_attention_flash_decode`] (online
/// softmax, no scores scratch).
pub fn flash_attention_enabled() -> bool {
    FLASH_ATTENTION.with(|f| f.get())
}

/// RAII guard: installs a SYCL stream on the current thread on
/// construction and clears it on drop. Returns `None` if the SYCL
/// crate is in mock mode or no device is visible — callers then
/// just continue on the CPU path, no error to handle.
///
/// Holding multiple guards on the same thread is not supported;
/// the most recently installed guard wins, and dropping any guard
/// clears the slot. Generation paths only install one at a time so
/// this is fine in practice.
pub struct SyclStreamGuard {
    _marker: std::marker::PhantomData<*const ()>,
}

impl SyclStreamGuard {
    /// Try to install a SYCL stream for the given device index on
    /// the current thread. Returns `Some(guard)` on success, `None`
    /// if SYCL is unavailable (mock build, no device, etc.). The
    /// caller should treat `None` as "stay on CPU" — not an error.
    pub fn install(device_index: u32) -> Option<Self> {
        let stream = sk::create_stream(device_index).ok()?;
        SYCL_STREAM.with(|s| {
            *s.borrow_mut() = Some(stream);
        });
        Some(Self {
            _marker: std::marker::PhantomData,
        })
    }
}

impl Drop for SyclStreamGuard {
    fn drop(&mut self) {
        SYCL_STREAM.with(|s| {
            *s.borrow_mut() = None;
        });
    }
}

/// Returns `true` if a SYCL stream is currently installed on this
/// thread. Used by `cargo test` to skip the SYCL path in unit tests
/// without faking it out.
pub fn has_sycl_stream() -> bool {
    SYCL_STREAM.with(|s| s.borrow().is_some())
}

/// Try to run RMSNorm on the SYCL device installed on this thread.
/// Returns `true` if the kernel ran (and wrote `y`); `false` if
/// there's no stream installed or the kernel call failed. CPU
/// fallback is the caller's responsibility — pattern:
///
/// ```ignore
/// if !try_rmsnorm_f32(x, w, y, d, eps) {
///     k::rmsnorm_f32_row(x, w, y, eps);
/// }
/// ```
///
/// Conversion: f32 inputs are quantised to f16 bit patterns for the
/// kernel, then dequantised on output. The CPU kernel also operates
/// on f32 with the same f16-equivalent precision; the parity check
/// in [`rustllama_engine::SyclAccelerator::parity_check_rmsnorm`]
/// verifies the two paths agree within ~1e-2 for d=4096.
///
/// `n_rows` is implicit: the function assumes `x.len() == y.len()`
/// and `x.len() % d == 0`, mirroring the CPU kernel's contract.
pub fn try_rmsnorm_f32(x: &[f32], w: &[f32], y: &mut [f32], d: usize, eps: f32) -> bool {
    if !sycl_healthy() {
        return false;
    }
    if d == 0 || x.len() != y.len() || x.len() % d != 0 || w.len() < d {
        return false;
    }
    let n_rows = x.len() / d;
    SYCL_STREAM.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(stream) = slot.as_mut() else {
            return false;
        };
        let x_bits: Vec<u16> = x.iter().map(|v| f16::from_f32(*v).to_bits()).collect();
        let w_bits: Vec<u16> = w[..d].iter().map(|v| f16::from_f32(*v).to_bits()).collect();
        let mut y_bits = vec![0u16; x.len()];
        if sk::rmsnorm(
            stream,
            &x_bits,
            &w_bits,
            &mut y_bits,
            n_rows as u32,
            d as u32,
            eps,
        )
        .is_err()
        {
            note_sycl_failure();
            return false;
        }
        for (i, bits) in y_bits.iter().enumerate() {
            y[i] = f16::from_bits(*bits).to_f32();
        }
        note_sycl_success();
        true
    })
}

/// Try to run half-split RoPE in-place on the SYCL device. Same
/// fall-back semantics: returns `true` only if the GPU wrote the
/// result. `qk` is `[n_heads, head_dim]`, `head_dim` must be even.
///
/// The CPU rope uses `cfg.rope_theta` as the base; the SYCL kernel
/// takes `inv_freq` as a pre-computed table, so the engine computes
/// it once per (head_dim, rope_theta) pair and passes the table in.
pub fn try_rope_f32(
    qk: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: u32,
    inv_freq: &[f32],
) -> bool {
    if !sycl_healthy() {
        return false;
    }
    if head_dim == 0 || head_dim % 2 != 0 {
        return false;
    }
    if qk.len() != n_heads * head_dim {
        return false;
    }
    let half = head_dim / 2;
    if inv_freq.len() < half {
        return false;
    }
    SYCL_STREAM.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(stream) = slot.as_mut() else {
            return false;
        };
        let mut qk_bits: Vec<u16> =
            qk.iter().map(|v| f16::from_f32(*v).to_bits()).collect();
        let freq_bits: Vec<u16> = inv_freq[..half]
            .iter()
            .map(|v| f16::from_f32(*v).to_bits())
            .collect();
        if sk::rope(
            stream,
            &mut qk_bits,
            n_heads as u32,
            head_dim as u32,
            pos,
            &freq_bits,
        )
        .is_err()
        {
            note_sycl_failure();
            return false;
        }
        for (i, bits) in qk_bits.iter().enumerate() {
            qk[i] = f16::from_bits(*bits).to_f32();
        }
        note_sycl_success();
        true
    })
}

/// Compute the RoPE inverse-frequency table for the given head
/// dimension and base. Matches the formula in
/// [`rustllama_kernels_cpu::rope_inplace_neox`] — used by callers
/// of [`try_rope_f32`] to pre-compute the table once per layer.
pub fn rope_inv_freq_table(head_dim: usize, rope_theta: f32) -> Vec<f32> {
    let half = head_dim / 2;
    (0..half)
        .map(|j| {
            let exponent = (2 * j) as f32 / head_dim as f32;
            1.0f32 / rope_theta.powf(exponent)
        })
        .collect()
}

/// Try to run SwiGLU's elementwise `silu(x) * y` on the SYCL device
/// installed on this thread. Same fall-back semantics as
/// [`try_rmsnorm_f32`]: returns `true` only if the kernel actually
/// wrote `out`. `x`, `y`, `out` must all be the same length.
pub fn try_silu_mul_f32(x: &[f32], y: &[f32], out: &mut [f32]) -> bool {
    if !sycl_healthy() {
        return false;
    }
    if x.len() != y.len() || out.len() != x.len() {
        return false;
    }
    SYCL_STREAM.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(stream) = slot.as_mut() else {
            return false;
        };
        let x_bits: Vec<u16> = x.iter().map(|v| f16::from_f32(*v).to_bits()).collect();
        let y_bits: Vec<u16> = y.iter().map(|v| f16::from_f32(*v).to_bits()).collect();
        let mut out_bits = vec![0u16; x.len()];
        if sk::silu_mul(stream, &x_bits, &y_bits, &mut out_bits).is_err() {
            note_sycl_failure();
            return false;
        }
        for (i, bits) in out_bits.iter().enumerate() {
            out[i] = f16::from_bits(*bits).to_f32();
        }
        note_sycl_success();
        true
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spill_lowbits_roundtrip_within_one_quant_step() {
        // 70 elems → 2 full 32-blocks + a 6-elem tail (exercises the short
        // trailing block on both the encode and decode side).
        let vals: Vec<f32> = (0..70).map(|i| (i as f32 - 35.0) * 0.1).collect();
        for &bits in &[8u8, 4u8] {
            let enc = spill_encode_lowbits(&vals, bits);
            let dec = spill_decode_lowbits(&enc, bits, vals.len());
            assert_eq!(dec.len(), vals.len());
            let qmax = ((1i32 << (bits - 1)) - 1) as f32;
            // Per-block symmetric quant error is bounded by one step (amax/qmax).
            let mut start = 0usize;
            while start < vals.len() {
                let end = (start + SPILL_BLOCK).min(vals.len());
                let amax = vals[start..end].iter().fold(0f32, |m, &v| m.max(v.abs()));
                let step = amax / qmax;
                for i in start..end {
                    assert!(
                        (dec[i] - vals[i]).abs() <= step + 1e-6,
                        "bits={bits} i={i}: got {} want {} (step {step})",
                        dec[i],
                        vals[i]
                    );
                }
                start = end;
            }
        }
    }

    #[test]
    fn spill_lowbits_all_zero_block_roundtrips_to_zero() {
        let vals = vec![0.0f32; 40];
        for &bits in &[8u8, 4u8] {
            let dec = spill_decode_lowbits(&spill_encode_lowbits(&vals, bits), bits, vals.len());
            assert_eq!(dec, vals);
        }
    }

    #[test]
    fn spill_dequant_f32_is_identity() {
        let vals = [1.5f32, -2.25, 0.0, 100.0, -0.5];
        let mut bytes = Vec::new();
        for v in &vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = dtype_dequant_to_f32(Dtype::F32, &bytes, vals.len()).unwrap();
        assert_eq!(out, vals.to_vec());
    }

    #[test]
    fn spill_dequant_skips_exotic_dtype() {
        // Exotic codebook formats must fall through to native spill (None).
        assert!(dtype_dequant_to_f32(Dtype::IQ1_SRaw, &[0u8; 50], 256).is_none());
        assert!(dtype_dequant_to_f32(Dtype::PTQ1_0Raw, &[0u8; 28], 128).is_none());
    }

    #[test]
    fn try_rmsnorm_with_no_stream_returns_false() {
        // Without an installed stream the function MUST return false
        // so the caller's CPU fallback fires. Pinning this so a
        // future change that accidentally panics on "no stream"
        // (e.g. unwrap on the TLS slot) gets caught.
        let x = vec![1.0f32; 4096];
        let w = vec![1.0f32; 4096];
        let mut y = vec![0f32; 4096];
        assert!(!try_rmsnorm_f32(&x, &w, &mut y, 4096, 1e-5));
        // `y` should be untouched.
        assert_eq!(y, vec![0f32; 4096]);
    }

    #[test]
    fn flash_attention_flag_round_trips_through_tls() {
        // Engine calls `set_flash_attention(cfg.inference.flash_attention)`
        // at the top of each generate, and `forward_one` reads via
        // `flash_attention_enabled()`. Pin the round-trip so a future
        // change that breaks the TLS doesn't silently route attention
        // through the wrong path.
        set_flash_attention(true);
        assert!(flash_attention_enabled());
        set_flash_attention(false);
        assert!(!flash_attention_enabled());
        // Restore default so subsequent tests see a clean slate.
        set_flash_attention(true);
    }

    #[test]
    fn rope_inv_freq_table_matches_neox_convention() {
        // The SYCL rope kernel expects a pre-computed inv_freq table;
        // the CPU rope reads `rope_theta` directly. Pin that the
        // table builder produces values matching the formula the CPU
        // rope uses internally so the two paths are equivalent.
        let head_dim = 64;
        let rope_theta = 10000.0f32;
        let table = rope_inv_freq_table(head_dim, rope_theta);
        assert_eq!(table.len(), head_dim / 2);
        // First entry (j=0) is 1 / theta^0 = 1.0.
        assert!((table[0] - 1.0).abs() < 1e-6);
        // Last entry (j = half-1 = 31) is 1 / theta^(62/64) ≈ 1.65e-4.
        let expected_last = 1.0f32 / rope_theta.powf(62.0 / 64.0);
        assert!((table[31] - expected_last).abs() < 1e-6);
    }

    #[test]
    fn try_rmsnorm_rejects_bad_shape() {
        let x = vec![1.0f32; 100];
        let w = vec![1.0f32; 32];
        let mut y = vec![0f32; 100];
        // d=32 doesn't divide x.len()=100 → false without touching y.
        assert!(!try_rmsnorm_f32(&x, &w, &mut y, 32, 1e-5));
        // d=0 → false.
        assert!(!try_rmsnorm_f32(&x, &w, &mut y, 0, 1e-5));
        // w shorter than d → false.
        let w_short = vec![1.0f32; 16];
        assert!(!try_rmsnorm_f32(&x, &w_short, &mut y, 32, 1e-5));
    }

    #[test]
    fn packed_weights_preloaded_flag_round_trips() {
        // Per-thread "weights preloaded" gate. Engine sets this once
        // after the first preload; subsequent generations on the same
        // thread skip the (idempotent but log-noisy) re-preload.
        // This test runs on a fresh thread; the flag MUST start false.
        std::thread::spawn(|| {
            assert!(!packed_weights_preloaded(), "fresh thread starts unset");
            mark_packed_weights_preloaded();
            assert!(packed_weights_preloaded(), "set after mark");
        })
        .join()
        .unwrap();
    }

    #[test]
    fn parse_q4k_shape_bucket_round_trips() {
        // The bucket string format is the contract between tuner +
        // engine; pin it here so a typo on either side gets caught.
        assert_eq!(parse_q4k_shape_bucket("M=1536,K=1536"), Some((1536, 1536)));
        assert_eq!(parse_q4k_shape_bucket("M=8960,K=1536"), Some((8960, 1536)));
        assert_eq!(parse_q4k_shape_bucket("M=0,K=0"), Some((0, 0)));
    }

    #[test]
    fn parse_q4k_shape_bucket_rejects_garbage() {
        // Hand-edited cache shouldn't be able to silently load as
        // (0, 0) — every misparse must yield None so the cache loader
        // skips the entry with a warning.
        assert_eq!(parse_q4k_shape_bucket(""), None);
        assert_eq!(parse_q4k_shape_bucket("M=1536"), None); // no comma
        assert_eq!(parse_q4k_shape_bucket("M=abc,K=1536"), None); // non-int M
        assert_eq!(parse_q4k_shape_bucket("X=1536,Y=1536"), None); // wrong prefix
        assert_eq!(parse_q4k_shape_bucket("1536,1536"), None); // missing prefixes
    }

    #[test]
    fn tuned_lws_returns_zero_with_empty_cache() {
        // The OnceLock for the packed-USM LWS cache initializes on
        // first call. In mock mode (no SYCL device fingerprint) the
        // init returns an empty map, so every shape lookup yields 0
        // (the kernel's hand-picked default).
        // Note: this test shares process-global state with other
        // tuned_lws_for callers; if a future test populates the
        // cache, this will start failing — split into a separate
        // test binary if that happens.
        assert_eq!(
            tuned_lws_for(rustllama_tuner::KERNEL_Q4K_PACKED_USM, 1536, 1536),
            0
        );
        assert_eq!(
            tuned_lws_for(rustllama_tuner::KERNEL_Q4K_PACKED_USM, 8960, 1536),
            0
        );
    }

    /// Hybrid-placement TLS round-trip on a fresh thread. The
    /// setter / getter pair must read back the same value, and
    /// `gpu_active_for_current_layer` must compute the comparison
    /// against the current layer index correctly.
    #[test]
    fn n_gpu_layers_tls_round_trips_per_thread() {
        std::thread::spawn(|| {
            // Fresh thread starts with defaults — n_gpu_layers ==
            // u32::MAX, current_layer_idx == 0, so the GPU path
            // is "active" by default.
            assert_eq!(n_gpu_layers(), u32::MAX);
            assert_eq!(current_layer_idx(), 0);
            assert!(
                gpu_active_for_current_layer(),
                "default = all GPU; layer 0 must be active"
            );

            // Engine pushes the config value before each generate.
            set_n_gpu_layers(4);
            assert_eq!(n_gpu_layers(), 4);
            assert!(gpu_active_for_current_layer(), "layer 0 < cutoff 4");

            // Model advances per-layer.
            for i in 0..4u32 {
                set_current_layer_idx(i);
                assert!(
                    gpu_active_for_current_layer(),
                    "layer {i} < cutoff 4 must be GPU"
                );
            }
            for i in 4..8u32 {
                set_current_layer_idx(i);
                assert!(
                    !gpu_active_for_current_layer(),
                    "layer {i} >= cutoff 4 must be CPU"
                );
            }

            // n_gpu_layers = 0 means "all CPU."
            set_n_gpu_layers(0);
            set_current_layer_idx(0);
            assert!(
                !gpu_active_for_current_layer(),
                "n_gpu_layers=0 => layer 0 also CPU"
            );
        })
        .join()
        .unwrap();
    }

    /// Each thread has independent placement TLS. A setter on
    /// thread A doesn't leak to thread B. Regression: a
    /// stray `static` instead of `thread_local!` would surface
    /// here as cross-thread bleed.
    #[test]
    fn n_gpu_layers_tls_is_per_thread_not_shared() {
        // Thread A: cutoff = 2.
        let a = std::thread::spawn(|| {
            set_n_gpu_layers(2);
            // Yield a few times so thread B has a chance to run
            // before we read back; if the TLS were shared, B's
            // write could clobber A's value.
            for _ in 0..10 {
                std::thread::yield_now();
            }
            n_gpu_layers()
        });
        // Thread B: cutoff = 999.
        let b = std::thread::spawn(|| {
            set_n_gpu_layers(999);
            for _ in 0..10 {
                std::thread::yield_now();
            }
            n_gpu_layers()
        });
        assert_eq!(a.join().unwrap(), 2, "thread A's TLS must not see B's write");
        assert_eq!(b.join().unwrap(), 999, "thread B's TLS must not see A's write");
    }

    /// `tensor_forced_to_cpu` is substring-based: `"ffn"` matches
    /// `"blk.5.ffn_gate.weight"` and any other name containing
    /// the substring. Multiple patterns are OR'd. Empty list
    /// matches nothing. Empty-string patterns are skipped (they'd
    /// otherwise match every tensor — almost certainly a config
    /// typo).
    #[test]
    fn tensor_forced_to_cpu_substring_matching() {
        std::thread::spawn(|| {
            // No patterns set yet — nothing matches.
            set_cpu_force_patterns(vec![]);
            assert!(!tensor_forced_to_cpu("blk.5.ffn_gate.weight"));
            assert!(!tensor_forced_to_cpu("token_embd.weight"));

            // Single pattern.
            set_cpu_force_patterns(vec!["ffn".to_string()]);
            assert!(tensor_forced_to_cpu("blk.5.ffn_gate.weight"));
            assert!(tensor_forced_to_cpu("blk.0.ffn_up.weight"));
            assert!(tensor_forced_to_cpu("ffn"));
            assert!(!tensor_forced_to_cpu("blk.5.attn_q.weight"));
            assert!(!tensor_forced_to_cpu("token_embd.weight"));

            // Multiple patterns — any match triggers CPU.
            set_cpu_force_patterns(vec!["ffn".to_string(), "attn_v".to_string()]);
            assert!(tensor_forced_to_cpu("blk.5.ffn_gate.weight"));
            assert!(tensor_forced_to_cpu("blk.10.attn_v.weight"));
            assert!(!tensor_forced_to_cpu("blk.5.attn_q.weight"));

            // Empty-string pattern is skipped (would otherwise
            // match everything, which is almost always a bug).
            set_cpu_force_patterns(vec!["".to_string(), "attn_v".to_string()]);
            assert!(!tensor_forced_to_cpu("blk.5.ffn_gate.weight"));
            assert!(tensor_forced_to_cpu("blk.5.attn_v.weight"));

            // Reset for sibling tests.
            set_cpu_force_patterns(vec![]);
        })
        .join()
        .unwrap();
    }

    /// CPU-force pattern list is per-thread. A setter on one
    /// thread doesn't leak to another — required so concurrent
    /// requests with different configs don't interfere.
    #[test]
    fn cpu_force_patterns_are_per_thread() {
        let a = std::thread::spawn(|| {
            set_cpu_force_patterns(vec!["ffn".to_string()]);
            for _ in 0..10 {
                std::thread::yield_now();
            }
            tensor_forced_to_cpu("blk.0.ffn_gate.weight")
        });
        let b = std::thread::spawn(|| {
            set_cpu_force_patterns(vec!["attn_q".to_string()]);
            for _ in 0..10 {
                std::thread::yield_now();
            }
            // Thread B's TLS has `attn_q`, not `ffn` — the ffn
            // name should NOT match if isolation works.
            tensor_forced_to_cpu("blk.0.ffn_gate.weight")
        });
        assert!(a.join().unwrap(), "thread A pinned ffn → must match");
        assert!(
            !b.join().unwrap(),
            "thread B's pattern is attn_q — ffn must not leak from A"
        );
    }

    /// `try_*_usm_f32` helpers short-circuit when
    /// `gpu_active_for_current_layer()` is false, even when
    /// `usm_attn_enabled()` is true. Mock mode reaches the
    /// gating check before any SYCL call, so this exercises the
    /// gate independently of GPU presence.
    #[test]
    fn try_usm_helpers_short_circuit_when_layer_past_cutoff() {
        std::thread::spawn(|| {
            // Force "GPU off for this layer."
            set_n_gpu_layers(2);
            set_current_layer_idx(5);
            assert!(!gpu_active_for_current_layer());

            // try_silu_mul_usm_f32 takes the simplest input shape
            // — exercise it directly. In mock mode it would
            // normally fall through to the rest of the body and
            // return false at the SYCL boundary; with the
            // placement gate it returns false IMMEDIATELY,
            // skipping the body entirely. Either way the caller
            // sees `false`, which is the contract.
            let g = vec![0.5f32; 16];
            let u = vec![0.3f32; 16];
            let mut out = vec![0f32; 16];
            assert!(
                !try_silu_mul_usm_f32(&g, &u, &mut out),
                "placement-gated silu_mul must return false on a CPU layer"
            );

            // Restore so subsequent tests on this thread aren't
            // affected — TLS persists for the thread's lifetime.
            set_n_gpu_layers(u32::MAX);
            set_current_layer_idx(0);
        })
        .join()
        .unwrap();
    }
}

// ============================================================
// USM-resident attention context
// ============================================================
//
// The accelerator context above (rmsnorm/silu/rope) does per-call
// H2D + compute + D2H over host-pointer kernels. That keeps the
// API ergonomic but pays the round-trip cost every dispatch — fine
// for correctness validation, but the big win on integrated GPUs
// (Iris Xe with shared LPDDR) only materializes when the K/V cache
// stays USM-resident across the entire forward pass.
//
// This module adds that path. A `UsmAttnContext` lazily allocates
// USM K/V caches sized to the model config plus Q/output scratch,
// keeps them alive across the whole `generate()` call via TLS, and
// hands the engine a function-call surface (`try_flash_attn_decode_usm_f32`)
// that:
//   1. Writes the new K/V row at position `pos` directly into the
//      USM cache (zero-copy on integrated GPUs).
//   2. Writes Q into USM scratch.
//   3. Calls `rsl_kernels_sycl::flash_attn_decode_usm`.
//   4. Reads the output back into a host-side `&mut [f32]`.
//
// The output read-back is the only remaining H2D-ish op; on
// integrated GPUs it's also page-mapped so it's effectively a
// memcpy from one part of LPDDR to another.

/// Sized parameters for the USM cache. Used as the key for
/// rebuilding the context when a different model is loaded —
/// any mismatch forces a re-allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UsmAttnConfig {
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    /// Hidden state size — input/output dim of rmsnorm. `0`
    /// signals "rmsnorm USM dispatch disabled" (e.g. when only
    /// the attention path opts in). The context still allocates
    /// the rmsnorm scratch fields below but at zero length.
    d: u32,
    /// FFN intermediate size — input/output dim of silu_mul.
    /// Same `0` = disabled semantics as `d`.
    d_ff: u32,
}

/// Per-thread USM context. Holds the stream + one K cache + one V
/// cache per layer + Q/out scratch sized to the per-call shape.
///
/// All buffers are sized in f16 elements (`u16`). The engine layer
/// converts f32 ↔ f16 on the slice-level when populating Q / reading
/// out, which mirrors the host-pointer kernel API.
struct UsmAttnContext {
    cfg: UsmAttnConfig,
    stream: sk::SyclStream,
    // SAFETY note: the `'static` lifetime on these buffers is a
    // lie — they actually borrow `self.stream` via shared ref.
    // We rebind through a custom Drop impl that frees them before
    // `stream` is dropped, and we never expose these buffers
    // outside the context. The lifetime sleight-of-hand is what
    // lets us store the context in TLS without an outer borrow.
    k_caches: Vec<sk::SyclSharedBuffer<'static, u16>>,
    v_caches: Vec<sk::SyclSharedBuffer<'static, u16>>,
    /// Per-layer count of contiguous valid rows in `k_caches` /
    /// `v_caches`, starting at position 0. The decode entry writes
    /// only the row at `pos`, so attention over `0..=pos` is correct
    /// ONLY when every earlier row was also written into THIS
    /// thread's context for the CURRENT sequence. The context is
    /// thread-local; a generation whose prefill ran elsewhere (a
    /// different worker thread), or that restored a prefix snapshot
    /// into host KV without replaying prefill, would otherwise
    /// attend over zeros or another request's leftover rows —
    /// observed as the "first chat after load answers a phantom
    /// prompt" corruption. Rows validate contiguously; a write at
    /// `pos > valid` is a gap and the dispatcher declines (CPU
    /// attention over host KV is the correct fallback). Dropped to
    /// zero whenever `kv_epoch` falls behind the global
    /// [`USM_KV_EPOCH`] (host KV was reset or restored).
    kv_valid_len: Vec<u32>,
    /// The [`USM_KV_EPOCH`] value this mirror's rows were written
    /// under. `0` at build time so the first dispatch always
    /// re-syncs against the live epoch.
    kv_epoch: u64,
    q_scratch: Option<sk::SyclSharedBuffer<'static, u16>>,
    out_scratch: Option<sk::SyclSharedBuffer<'static, u16>>,
    // Lazy scratch for the non-attention USM kernels. Allocated
    // on first use of the matching `try_*_usm_f32` call, reused
    // across every later call from the same thread. Each
    // `Option` field stores at most one buffer sized to the
    // largest request seen so far — grows monotonically (a
    // larger d_ff request reallocates).
    rmsnorm_x: Option<sk::SyclSharedBuffer<'static, u16>>,
    rmsnorm_w: Option<sk::SyclSharedBuffer<'static, u16>>,
    rmsnorm_y: Option<sk::SyclSharedBuffer<'static, u16>>,
    /// Extra residual-branch USM slot for the fused
    /// `try_add_rmsnorm_usm_f32` path. Shares the same lazy-grow
    /// semantics as `rmsnorm_x`.
    rmsnorm_branch: Option<sk::SyclSharedBuffer<'static, u16>>,
    silu_x: Option<sk::SyclSharedBuffer<'static, u16>>,
    silu_y: Option<sk::SyclSharedBuffer<'static, u16>>,
    silu_out: Option<sk::SyclSharedBuffer<'static, u16>>,
    rope_qk: Option<sk::SyclSharedBuffer<'static, u16>>,
    /// RoPE `inv_freq` table cached + uploaded as USM. Keyed on
    /// `(head_dim, rope_theta_bits)` so it's only rebuilt when the
    /// model changes — same trick as `ForwardScratch::rope_inv_freq`.
    rope_inv_freq: Option<sk::SyclSharedBuffer<'static, u16>>,
    rope_inv_freq_key: Option<(u32, u32)>,
    /// Packed-layout weight USM cache. Keyed on the host weight
    /// bytes' starting address (`as_bytes(w).as_ptr() as usize`) —
    /// stable for the lifetime of the GGUF mmap, so the same model
    /// never re-uploads a weight tensor. Shared across all packed
    /// quant formats (Q8_0Raw, Q4_KRaw, future Q5_K/Q6_K/etc.): the
    /// cache only stores raw bytes, and the kernel called against
    /// the cached pointer depends on the source tensor's `dtype`.
    /// First call uploads, subsequent calls just hand the cached
    /// USM pointer to the kernel. Drops before `self.stream`.
    packed_weight_cache: HashMap<usize, PackedWeightBuf>,
    /// Bytes currently resident in the dedicated-VRAM (`malloc_device`)
    /// tier of `packed_weight_cache`. Tracked separately from
    /// `packed_weight_usm_bytes` because device memory is drawn from the
    /// VRAM aperture, not the shared-LPDDR / host-RAM pool the shared
    /// budget guards. Capped by `device_vram_budget_bytes()`.
    packed_weight_device_bytes: usize,
    /// Running total of bytes uploaded into `packed_weight_cache`,
    /// gated by `RUSTLLAMA_MATVEC_USM_MAX_MB`. On an integrated GPU,
    /// USM-shared memory *is* host RAM, so every weight copied here is
    /// resident twice (file-backed/owned original + this USM copy).
    /// Without a cap the cache grows until it exhausts RAM and the OS
    /// thrashes (file-backed weights + USM copies > physical RAM). When
    /// the cap would be exceeded, [`Self::cache_packed_weight`] declines
    /// the upload and the caller falls through to CPU matvec, which
    /// reads the file-backed bytes in place — single residency.
    packed_weight_usm_bytes: usize,
    /// Adaptive USM gate: resolved on first matvec upload.
    ///   - `None`     — undecided (compute on first cache_packed_weight call).
    ///   - `Some(0)`  — unlimited: the model fits in RAM with comfortable
    ///                  headroom (AvailPhys at first matvec > env_budget +
    ///                  safety), so the env cap would only hurt — bypass it
    ///                  to keep pure-GPU matmul on already-fitting models.
    ///   - `Some(n)`  — enforce the env-var budget `n` bytes (model is near
    ///                  the RAM limit; the cap prevents USM duplication
    ///                  from oversubscribing host memory).
    /// Resolved per-context (per-thread); cleared on engine teardown.
    effective_matvec_usm_budget: Option<usize>,
    /// Per-call activation scratch for Q8_0 packed matvec. Sized to
    /// the largest K seen so far; reused across calls. Same
    /// lazy-grow semantics as `rmsnorm_x`.
    matvec_x_f32: Option<sk::SyclSharedBuffer<'static, f32>>,
    /// `(host_ptr, len)` of the data currently sitting in
    /// `matvec_x_f32`. `None` if the buffer was just (re)allocated
    /// or if the previous call cleared it. Used to skip the
    /// host→USM copy when consecutive matvec calls feed the same
    /// input slab — common pattern in transformer forward passes:
    /// the same `h_norm` row drives Q/K/V projections; the same
    /// `n_row` slab drives gate + up. With this dedup the second
    /// and third calls in such a chain skip the memcpy entirely.
    ///
    /// The pointer is treated as an opaque key — we never
    /// dereference it. Equality is correct as long as the caller
    /// Per-call output scratch for Q8_0 packed matvec. Sized to the
    /// largest M seen so far; reused across calls.
    matvec_out_f32: Option<sk::SyclSharedBuffer<'static, f32>>,
    /// H4: secondary output scratch for the gate+up fused matvec
    /// (`try_matvec_tensor_gate_up_fused_usm_f32`). Holds the `up`
    /// projection while `matvec_out_f32` holds the `gate` projection.
    /// Sized to the largest M (== d_ff) seen so far; reused across
    /// calls.
    matvec_out2_f32: Option<sk::SyclSharedBuffer<'static, f32>>,
    /// H6: norm-weight scratch for the fused matvec+add+rmsnorm kernel
    /// (`try_matvec_out_proj_add_rmsnorm_usm_f32`). Holds the rmsnorm
    /// scale vector (`ffn_norm`, length d_model). Sized to the largest
    /// M seen; reused across calls.
    matvec_wnorm_f32: Option<sk::SyclSharedBuffer<'static, f32>>,
    /// H6: residual-stream scratch for the chained (matvec →
    /// add_rmsnorm_f32) dispatcher. Holds `hidden` (in/out for the
    /// add_rmsnorm pass). Separate from `matvec_out_f32` (which holds
    /// the matvec result == add_rmsnorm's `branch` input) so neither
    /// buffer aliases the other across the two kernel launches.
    matvec_hidden_f32: Option<sk::SyclSharedBuffer<'static, f32>>,
    /// H8: F16-packed activation scratch for the mixed-precision
    /// matvec (`try_matvec_tensor_f16in_usm_f32`). Holds the input
    /// vector converted F32→F16 (length K). Sized to the largest K
    /// seen; reused across calls. Separate from `matvec_x_f32` so the
    /// F32 path's upload-dedup cache key stays valid.
    matvec_x_f16in: Option<sk::SyclSharedBuffer<'static, u16>>,
    /// Per-call activation scratch for F16 GEMM matvec (the
    /// non-quantized weight path through `try_matvec_f16_usm_f32`).
    /// Sized to the largest K seen so far. F16 layout (`u16`)
    /// because `rsl_gemm_f16` wants F16 inputs end-to-end.
    matvec_f16_x: Option<sk::SyclSharedBuffer<'static, u16>>,
    /// Per-call output scratch for F16 GEMM matvec. Sized to the
    /// largest M seen so far. F16 → host converts on read-back.
    matvec_f16_out: Option<sk::SyclSharedBuffer<'static, u16>>,
    /// Per-call USM scratch for the F32 prefill flash-attention
    /// kernel (`try_flash_attn_prefill_usm_f32`). Sized lazily to
    /// the largest request seen, reused across calls. Four
    /// independent slabs: Q (`[n_new, n_heads, head_dim]`), K and V
    /// (`[n_kv_heads, max_ctx, head_dim]`), and output (same shape
    /// as Q). Shared across all transformer layers in a single
    /// forward pass — only one layer's prefill runs at a time, so
    /// per-layer caching is unnecessary at this layer of the stack.
    /// We re-upload K/V slabs on each call as a first-cut design;
    /// a per-layer USM K/V residency optimization is queued for a
    /// later pass (see roadmap: "GPU prefill attention").
    prefill_q_f32: Option<sk::SyclSharedBuffer<'static, f32>>,
    prefill_k_f32: Option<sk::SyclSharedBuffer<'static, f32>>,
    prefill_v_f32: Option<sk::SyclSharedBuffer<'static, f32>>,
    prefill_out_f32: Option<sk::SyclSharedBuffer<'static, f32>>,
    /// Per-call USM scratch for the QUANTIZED-KV prefill flash-attention
    /// kernels (`try_flash_attn_prefill_usm_quant`). Siblings of the F32
    /// prefill slabs above, but holding the format-specific PACKED bytes
    /// (Q4_0 blocks / Q8_0 i8 slab / NVFP4 blocks / TurboQuant levels) for
    /// the whole `[n_kv_heads, max_ctx, bytes_per_row]` slab. Prefill
    /// re-uploads the entire packed slab on every call (no incremental
    /// residency / valid-len gate — the same whole-slab model the F32
    /// prefill helper uses), so there is one shared pair reused across all
    /// layers of a forward pass rather than a per-layer mirror. The `_f32`
    /// Q / output scratch above is reused for the quant path's f32 Q/out.
    /// `*_scales` are allocated only for the formats that carry an
    /// out-of-band scale (Q8_0 per-row absmax, TurboQuant per-row scale),
    /// sized `[n_kv_heads * max_ctx]` and indexed `kv_h * max_ctx + t`.
    prefill_k_packed: Option<sk::SyclSharedBuffer<'static, u8>>,
    prefill_v_packed: Option<sk::SyclSharedBuffer<'static, u8>>,
    prefill_k_scales: Option<sk::SyclSharedBuffer<'static, f32>>,
    prefill_v_scales: Option<sk::SyclSharedBuffer<'static, f32>>,
    // -------- Quantized-KV flash-attention decode mirror --------
    //
    // A SECOND, fully independent per-layer KV mirror used by the
    // quantized-KV decode helpers (`try_flash_attn_decode_usm_q4_0`
    // and friends). Where the f16 `k_caches` / `v_caches` above hold
    // an f16 copy of the KV rows for the F32 path, these hold the
    // format-specific PACKED bytes (Q4_0 blocks / Q8_0 i8 slab /
    // NVFP4 blocks / TurboQuant levels) that the quant kernels read
    // directly — the same bytes the engine's host slab holds, so the
    // GPU output matches the CPU reference bit-for-bit modulo the
    // fp-reduction-order tolerance the F32 path already carries.
    //
    // Design note — WHY a separate mirror + separate validity gate:
    // the two mirrors MUST NOT share staleness state. The F32 path
    // gates on `kv_valid_len` / `kv_epoch`; the quant path gates on
    // `kv_valid_len_q` / `kv_epoch_q`. A model uses exactly one KV
    // dtype per layer for the whole generation, so in practice only
    // one mirror is ever populated — but keeping the gates disjoint
    // means a stray call on the "wrong" path can never validate a row
    // it did not write, and the F32 helper stays byte-for-byte
    // unchanged. Buffers are lazily allocated PER LAYER on first use
    // (unlike the eager f16 caches) so hybrid stacks don't pay a full
    // KV mirror for their SSM / non-attention layers.
    q_k_packed: Vec<Option<sk::SyclSharedBuffer<'static, u8>>>,
    q_v_packed: Vec<Option<sk::SyclSharedBuffer<'static, u8>>>,
    /// Per-row f32 scale mirrors — allocated only for the formats that
    /// carry an out-of-band scale (Q8_0 per-row absmax, TurboQuant
    /// per-row max-abs). `[n_kv_heads * max_ctx]` when present, indexed
    /// `kv_h * max_ctx + t` to match the host slab + the kernel. Left
    /// `None` for Q4_0 / NVFP4 which embed their scale in the packed
    /// block bytes.
    q_k_scales: Vec<Option<sk::SyclSharedBuffer<'static, f32>>>,
    q_v_scales: Vec<Option<sk::SyclSharedBuffer<'static, f32>>>,
    /// `bytes_per_row` the layer's packed mirror was last allocated
    /// for. `0` = unallocated. A change (only possible if a layer's KV
    /// format changed mid-run, which never happens today) forces a
    /// re-alloc + a validity reset for that layer.
    q_bytes_per_row: Vec<u32>,
    /// Contiguous-valid-row tracker for the quant mirror, mirroring
    /// `kv_valid_len` for the f16 path (see its doc). Independent so a
    /// gap on one path can't mask a gap on the other.
    kv_valid_len_q: Vec<u32>,
    /// Epoch the quant mirror's rows were written under. Independent of
    /// the f16 mirror's `kv_epoch`; both track the global
    /// [`USM_KV_EPOCH`] but reset their own valid-lens separately.
    kv_epoch_q: u64,
    /// F32 Q / output scratch for the quant kernels. The quant flash
    /// kernels take an F32 `q` and write an F32 `out` (unlike the f16
    /// `q_scratch` / `out_scratch` the F32 path uses), so these are
    /// separate slabs. Shared across all quant layers — only one layer
    /// computes at a time. Sized `n_heads * head_dim`.
    q_scratch_f32: Option<sk::SyclSharedBuffer<'static, f32>>,
    out_scratch_f32: Option<sk::SyclSharedBuffer<'static, f32>>,
}

impl UsmAttnContext {
    /// Try to build a fresh USM context for the given model
    /// dimensions. Returns `None` if SYCL is unavailable (mock
    /// build, no GPU, alloc failure) — callers fall back to the
    /// host-pointer or CPU path.
    fn try_new(cfg: UsmAttnConfig) -> Option<Self> {
        // Log thread id so a cross-thread mismatch (prepare on thread A,
        // matvec on thread B's USM_ATTN slot which is still None) is
        // immediately visible in the trace.
        let tid = std::thread::current().id();
        let stream = match sk::create_stream(0) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    thread_id = ?tid,
                    error = ?e,
                    "UsmAttnContext::try_new — create_stream(0) failed",
                );
                return None;
            }
        };
        tracing::info!(
            thread_id = ?tid,
            n_layers = cfg.n_layers,
            n_kv_heads = cfg.n_kv_heads,
            max_ctx = cfg.max_ctx,
            head_dim = cfg.head_dim,
            "UsmAttnContext::try_new — beginning K/V cache allocation",
        );
        let mut ctx = UsmAttnContext {
            cfg,
            stream,
            k_caches: Vec::with_capacity(cfg.n_layers as usize),
            v_caches: Vec::with_capacity(cfg.n_layers as usize),
            kv_valid_len: vec![0; cfg.n_layers as usize],
            kv_epoch: 0,
            q_scratch: None,
            out_scratch: None,
            rmsnorm_x: None,
            rmsnorm_w: None,
            rmsnorm_y: None,
            rmsnorm_branch: None,
            silu_x: None,
            silu_y: None,
            silu_out: None,
            rope_qk: None,
            rope_inv_freq: None,
            rope_inv_freq_key: None,
            packed_weight_cache: HashMap::new(),
            packed_weight_device_bytes: 0,
            packed_weight_usm_bytes: 0,
            effective_matvec_usm_budget: None,
            matvec_x_f32: None,
            matvec_out_f32: None,
            matvec_out2_f32: None,
            matvec_wnorm_f32: None,
            matvec_hidden_f32: None,
            matvec_x_f16in: None,
            matvec_f16_x: None,
            matvec_f16_out: None,
            prefill_q_f32: None,
            prefill_k_f32: None,
            prefill_v_f32: None,
            prefill_out_f32: None,
            prefill_k_packed: None,
            prefill_v_packed: None,
            prefill_k_scales: None,
            prefill_v_scales: None,
            // Quant-KV mirror: lazily allocated per layer, so start
            // every layer empty. `n_layers`-sized so `layer_idx` (the
            // same index used for the f16 caches) is always in range.
            q_k_packed: (0..cfg.n_layers).map(|_| None).collect(),
            q_v_packed: (0..cfg.n_layers).map(|_| None).collect(),
            q_k_scales: (0..cfg.n_layers).map(|_| None).collect(),
            q_v_scales: (0..cfg.n_layers).map(|_| None).collect(),
            q_bytes_per_row: vec![0; cfg.n_layers as usize],
            kv_valid_len_q: vec![0; cfg.n_layers as usize],
            kv_epoch_q: 0,
            q_scratch_f32: None,
            out_scratch_f32: None,
        };
        let kv_buf_len =
            (cfg.n_kv_heads as usize) * (cfg.max_ctx as usize) * (cfg.head_dim as usize);
        let q_len = (cfg.n_heads as usize) * (cfg.head_dim as usize);
        // SAFETY: each alloc borrows `&ctx.stream` for `'static`,
        // which is a lie. The Drop impl below frees them before
        // ctx.stream is destroyed. We never lend these buffers
        // outside the context, and the context as a whole is
        // self-consistent at all observable points.
        for i in 0..cfg.n_layers {
            let k_raw = sk::SyclSharedBuffer::<u16>::alloc(&ctx.stream, kv_buf_len);
            let k_buf = match k_raw {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(
                        thread_id = ?tid,
                        layer = i,
                        bytes = kv_buf_len * 2,
                        error = ?e,
                        "UsmAttnContext::try_new — K cache allocation failed",
                    );
                    return None;
                }
            };
            let k = unsafe {
                std::mem::transmute::<
                    sk::SyclSharedBuffer<'_, u16>,
                    sk::SyclSharedBuffer<'static, u16>,
                >(k_buf)
            };
            let v_raw = sk::SyclSharedBuffer::<u16>::alloc(&ctx.stream, kv_buf_len);
            let v_buf = match v_raw {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(
                        thread_id = ?tid,
                        layer = i,
                        bytes = kv_buf_len * 2,
                        error = ?e,
                        "UsmAttnContext::try_new — V cache allocation failed",
                    );
                    return None;
                }
            };
            let v = unsafe {
                std::mem::transmute::<
                    sk::SyclSharedBuffer<'_, u16>,
                    sk::SyclSharedBuffer<'static, u16>,
                >(v_buf)
            };
            ctx.k_caches.push(k);
            ctx.v_caches.push(v);
        }
        ctx.q_scratch = Some(unsafe {
            std::mem::transmute::<
                sk::SyclSharedBuffer<'_, u16>,
                sk::SyclSharedBuffer<'static, u16>,
            >(sk::SyclSharedBuffer::alloc(&ctx.stream, q_len).ok()?)
        });
        ctx.out_scratch = Some(unsafe {
            std::mem::transmute::<
                sk::SyclSharedBuffer<'_, u16>,
                sk::SyclSharedBuffer<'static, u16>,
            >(sk::SyclSharedBuffer::alloc(&ctx.stream, q_len).ok()?)
        });
        // Zero the K/V caches so any unwritten positions don't
        // leak garbage into the attention output (the kernel
        // reads positions 0..kv_len; if kv_len > what we've
        // written it'd be undefined, but we control kv_len from
        // the call site so this is just defense-in-depth).
        for k in &mut ctx.k_caches {
            for slot in k.as_mut_slice() {
                *slot = 0;
            }
        }
        for v in &mut ctx.v_caches {
            for slot in v.as_mut_slice() {
                *slot = 0;
            }
        }
        Some(ctx)
    }

    /// Ensure weight tensor `key` (raw bytes `src`, `expected_bytes`
    /// long) is resident in the packed-USM weight cache. Returns:
    ///   - `true`  — already cached, or freshly uploaded; the caller
    ///     may use `packed_weight_cache[&key]` for the GPU matvec.
    ///   - `false` — declined: the `RUSTLLAMA_MATVEC_USM_MAX_MB` budget
    ///     would be exceeded, or the USM allocation failed. The caller
    ///     must fall through to the CPU matvec, which reads `src` (the
    ///     file-backed / owned bytes) in place — no USM copy, single
    ///     residency. This is the iGPU double-residency guard: on
    ///     shared-memory GPUs the USM copy duplicates host RAM, so we
    ///     cap how much weight data we duplicate before falling back.
    fn cache_packed_weight(
        &mut self,
        key: usize,
        expected_bytes: usize,
        src: &[u8],
        src_is_file_backed: bool,
    ) -> bool {
        if self.packed_weight_cache.contains_key(&key) {
            return true;
        }
        // Dedicated-VRAM (malloc_device) tier, tried first. On the
        // integrated Iris Xe this lands in the reserved VRAM aperture —
        // separate from the shared-LPDDR / system-RAM pool — so hot
        // weights fill dedicated VRAM AND relieve host-RAM pressure
        // (device memory is not double-resident in host RAM the way
        // malloc_shared is). Budget-capped by `device_vram_budget_bytes`;
        // on over-budget or an alloc/copy failure we fall through to the
        // shared tier below. Off by default (budget 0) until Phase 2.
        let dev_budget = device_vram_budget_bytes();
        if dev_budget != 0
            && self
                .packed_weight_device_bytes
                .saturating_add(expected_bytes)
                <= dev_budget
        {
            if let Some(dbuf) =
                sk::SyclDeviceBuffer::alloc_from_host(&self.stream, &src[..expected_bytes])
            {
                // SAFETY: same '_-to-'static pattern as the shared tier —
                // `UsmAttnContext::drop` clears the cache before the
                // stream is destroyed.
                let dbuf = unsafe {
                    std::mem::transmute::<
                        sk::SyclDeviceBuffer<'_>,
                        sk::SyclDeviceBuffer<'static>,
                    >(dbuf)
                };
                self.packed_weight_cache
                    .insert(key, PackedWeightBuf::Device(dbuf));
                self.packed_weight_device_bytes = self
                    .packed_weight_device_bytes
                    .saturating_add(expected_bytes);
                static FIRST_DEVICE_UPLOAD: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                FIRST_DEVICE_UPLOAD.get_or_init(|| {
                    tracing::info!(
                        bytes = expected_bytes,
                        "first packed-matvec DEVICE-VRAM weight upload — dedicated-VRAM tier live"
                    );
                });
                // The weight now lives in device VRAM; drop the redundant
                // file-backed mmap pages to reclaim host RAM (the payoff
                // of the device tier on a discrete GPU).
                if src_is_file_backed && discard_mmap_enabled() {
                    discard_mmap_pages(src.as_ptr(), expected_bytes);
                }
                return true;
            }
            // Device alloc/copy failed — fall through to the shared tier.
        }
        // Dynamic host-RAM safety floor for the shared-USM tier. On this
        // unified-memory iGPU shared USM *is* system LPDDR, so an
        // unbounded weight cache walks free RAM toward OOM. Keep filling
        // USM only while this upload would leave at least the floor free;
        // past that, spill this and subsequent weights to CPU-mmap. One
        // cheap GlobalMemoryStatusEx per *uncached* weight (~µs), and it
        // adapts as KV / activations grow. `floor == 0` disables it.
        let floor = usm_ram_floor_bytes();
        if floor != 0 {
            // With mmap-discard on, uploading a file-backed weight is
            // ~RAM-neutral (its USM copy replaces the discarded mmap
            // pages), so it doesn't count against the floor. Heap-owned
            // weights, or discard disabled, cost their full size.
            let net_cost = if src_is_file_backed && discard_mmap_enabled() {
                0
            } else {
                expected_bytes as u64
            };
            let avail_after = avail_phys_bytes().saturating_sub(net_cost);
            if avail_after < floor {
                log_usm_ram_floor_once(self.packed_weight_usm_bytes, avail_after, floor);
                return false;
            }
        }
        // Resolve the gate budget lazily on first miss, then cache it
        // per-context. The adaptive resolver samples AvailPhys: if the
        // model fits with headroom (legacy small/medium models like
        // qwen2.5-coder), the cap is bypassed → pure GPU matmul, no
        // regression. If RAM is tight (the 14 GB-on-16 GB MoE case),
        // the cap is enforced → CPU fallback prevents thrash.
        let budget = match self.effective_matvec_usm_budget {
            Some(b) => b,
            None => {
                let env = matvec_usm_weight_budget_bytes();
                let eff = resolve_adaptive_matvec_budget(env);
                self.effective_matvec_usm_budget = Some(eff);
                eff
            }
        };
        if budget != 0 && self.packed_weight_usm_bytes.saturating_add(expected_bytes) > budget {
            log_usm_weight_budget_once();
            return false;
        }
        let Ok(buf) = sk::SyclSharedBuffer::<u8>::alloc(&self.stream, expected_bytes) else {
            return false;
        };
        // SAFETY: '_-to-'static transmute pattern — Drop on
        // UsmAttnContext clears the cache before dropping the stream.
        let mut buf = unsafe {
            std::mem::transmute::<
                sk::SyclSharedBuffer<'_, u8>,
                sk::SyclSharedBuffer<'static, u8>,
            >(buf)
        };
        buf.as_mut_slice()[..expected_bytes].copy_from_slice(&src[..expected_bytes]);
        self.packed_weight_cache
            .insert(key, PackedWeightBuf::Shared(buf));
        self.packed_weight_usm_bytes = self
            .packed_weight_usm_bytes
            .saturating_add(expected_bytes);
        static FIRST_USM_UPLOAD: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        FIRST_USM_UPLOAD.get_or_init(|| {
            tracing::info!(
                bytes = expected_bytes,
                "first packed-matvec USM weight upload — GPU dispatch live"
            );
        });
        // The weight now has a shared-USM copy the GPU reads; drop the
        // redundant file-backed mmap pages so the weight is resident once
        // (USM) instead of twice (USM + hot mmap). Re-faults from the
        // GGUF only on a CPU fallback for this weight.
        if src_is_file_backed && discard_mmap_enabled() {
            discard_mmap_pages(src.as_ptr(), expected_bytes);
        }
        true
    }
}

/// Residency tier for a cached packed-quant weight. `Shared` =
/// `malloc_shared` USM (LPDDR, host+device — double-resident in host
/// RAM on an iGPU); `Device` = `malloc_device` USM (dedicated-VRAM
/// aperture, device-only, separate from system RAM). Both hand the
/// matvec kernel the same `*const u8`, so the dispatch path is
/// tier-agnostic.
enum PackedWeightBuf {
    Shared(sk::SyclSharedBuffer<'static, u8>),
    Device(sk::SyclDeviceBuffer<'static>),
}

impl PackedWeightBuf {
    #[inline]
    fn as_ptr(&self) -> *const u8 {
        match self {
            PackedWeightBuf::Shared(b) => b.as_ptr(),
            PackedWeightBuf::Device(b) => b.as_ptr(),
        }
    }
}

/// Budget in bytes for the dedicated-VRAM (`malloc_device`) weight
/// tier. `RUSTLLAMA_DEVICE_VRAM_MB` sets it; default 0 (tier OFF) until
/// Phase 2 sizes it from Sysman-measured device free VRAM. Process-wide
/// (the VRAM aperture is a single device resource shared by all worker
/// threads' contexts).
fn device_vram_budget_bytes() -> usize {
    static CELL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("RUSTLLAMA_DEVICE_VRAM_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0)
            .saturating_mul(1024 * 1024)
    })
}

impl Drop for UsmAttnContext {
    fn drop(&mut self) {
        // Free the buffers BEFORE the stream is destroyed. The
        // `'static` lifetime on the buffer storage is a lie that
        // only holds while `self.stream` is alive, so we must
        // drop them first. Vec::clear runs each Drop impl which
        // calls usm_free(self.stream, ...). After this, the
        // stream is destroyed by SyclStream::Drop in the natural
        // field-order.
        self.k_caches.clear();
        self.v_caches.clear();
        self.q_scratch = None;
        self.out_scratch = None;
        // Non-attention scratch — same Drop-before-stream
        // ordering as the K/V cache.
        self.rmsnorm_x = None;
        self.rmsnorm_w = None;
        self.rmsnorm_y = None;
        self.rmsnorm_branch = None;
        self.silu_x = None;
        self.silu_y = None;
        self.silu_out = None;
        self.rope_qk = None;
        self.rope_inv_freq = None;
        self.rope_inv_freq_key = None;
        // Packed-quant weight cache + matvec scratch: same
        // Drop-before-stream ordering as the K/V cache. clear() runs
        // each buffer's Drop impl, which calls usm_free on `self.stream`.
        self.packed_weight_cache.clear();
        self.packed_weight_device_bytes = 0;
        self.packed_weight_usm_bytes = 0;
        self.effective_matvec_usm_budget = None;
        self.matvec_x_f32 = None;
        self.matvec_out_f32 = None;
        self.matvec_out2_f32 = None;
        self.matvec_wnorm_f32 = None;
        self.matvec_hidden_f32 = None;
        self.matvec_x_f16in = None;
        self.matvec_f16_x = None;
        self.matvec_f16_out = None;
        // F32 prefill flash-attention scratch — same Drop-before-
        // stream ordering as everything else.
        self.prefill_q_f32 = None;
        self.prefill_k_f32 = None;
        self.prefill_v_f32 = None;
        self.prefill_out_f32 = None;
        // Quant-KV prefill packed scratch — same Drop-before-stream order.
        self.prefill_k_packed = None;
        self.prefill_v_packed = None;
        self.prefill_k_scales = None;
        self.prefill_v_scales = None;
        // Quantized-KV decode mirror — `Vec::clear()` runs each
        // buffer's Drop (usm_free on `self.stream`) before the stream
        // is destroyed, matching the K/V cache ordering above.
        self.q_k_packed.clear();
        self.q_v_packed.clear();
        self.q_k_scales.clear();
        self.q_v_scales.clear();
        self.q_scratch_f32 = None;
        self.out_scratch_f32 = None;
        // `self.stream` drops here.
    }
}

/// USM matvec weight-cache budget (`RUSTLLAMA_MATVEC_USM_MAX_MB`, MB).
/// `0` (default) = unlimited (legacy behavior). On an integrated GPU,
/// USM-shared memory is host RAM, so the packed-matvec weight cache
/// duplicates each weight tensor (file-backed/owned original + USM
/// copy). On a model whose weights approach physical RAM this double
/// residency thrashes the pager. Set a cap (e.g. 2048-3072 on a 16 GB
/// box) so only the hottest ~budget of weights get the GPU copy and
/// the rest stay single-resident via CPU matvec.
fn matvec_usm_weight_budget_bytes() -> usize {
    static MAX_MB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let cap = *MAX_MB.get_or_init(|| {
        std::env::var("RUSTLLAMA_MATVEC_USM_MAX_MB")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(0)
    });
    cap.saturating_mul(1024 * 1024) as usize
}

/// Optional minimum free physical RAM (bytes) to preserve while copying
/// packed weights into shared USM. **Off by default (0)** — with
/// mmap-discard on (the default), each file-backed weight's USM copy
/// replaces its discarded mmap pages, so uploading is ~RAM-neutral and
/// there is no double-residency to guard against; a naive floor here
/// instead *kills* GPU dispatch on a RAM-tight box, because `avail_phys`
/// already sits below any useful floor once the multi-GB GGUF mmap is
/// resident (that memory is reclaimable, but `ullAvailPhys` counts it as
/// unavailable). Set `RUSTLLAMA_USM_RAM_FLOOR_MB` to opt into a hard cap
/// (e.g. with mmap-discard disabled, or heap-owned weights); a weight
/// whose *net* RAM cost would drop free RAM below the floor then spills
/// to the CPU matvec path (single-residency).
fn usm_ram_floor_bytes() -> u64 {
    static FLOOR: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *FLOOR.get_or_init(|| {
        std::env::var("RUSTLLAMA_USM_RAM_FLOOR_MB")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(0)
            .saturating_mul(1024 * 1024)
    })
}

/// Whether to discard a weight's file-backed mmap pages after copying it
/// into USM / device memory. This cuts the double-residency: once the
/// GPU reads its USM/device copy, the redundant mmap pages can be
/// released and re-faulted from the GGUF only on the rare CPU fallback.
/// ON by default; `RUSTLLAMA_DISCARD_MMAP_AFTER_USM=0` disables it.
fn discard_mmap_enabled() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| {
        !matches!(
            std::env::var("RUSTLLAMA_DISCARD_MMAP_AFTER_USM")
                .ok()
                .as_deref(),
            Some("0") | Some("false") | Some("FALSE")
        )
    })
}

/// Release the physical pages backing a file-backed mmap range that is
/// now redundant (its bytes have been copied into USM / device memory).
/// Best-effort and safe **only** for read-only file mappings: the caller
/// MUST have verified `Storage::mmap_borrowed_ptr_len().is_some()` — for
/// a file mapping the discarded pages simply re-fault from the GGUF on
/// next access, whereas discarding private/heap memory would re-zero it.
/// Aligns INWARD to whole 4 KiB pages so a page shared with an adjacent
/// tensor is never touched. No-op off Windows/Linux or on an empty range.
#[cfg(windows)]
fn discard_mmap_pages(ptr: *const u8, len: usize) {
    const PAGE: usize = 4096;
    let start = ptr as usize;
    let end = start.saturating_add(len);
    let aligned_start = (start + PAGE - 1) & !(PAGE - 1);
    let aligned_end = end & !(PAGE - 1);
    if aligned_end <= aligned_start {
        return;
    }
    extern "system" {
        fn DiscardVirtualMemory(addr: *mut core::ffi::c_void, size: usize) -> u32;
    }
    // SAFETY: [aligned_start, aligned_end) is a whole-page sub-range of a
    // live read-only file mapping; DiscardVirtualMemory only drops the
    // pages' resident copy (re-faulted from the file on next access).
    unsafe {
        let _ = DiscardVirtualMemory(
            aligned_start as *mut core::ffi::c_void,
            aligned_end - aligned_start,
        );
    }
}
#[cfg(target_os = "linux")]
fn discard_mmap_pages(ptr: *const u8, len: usize) {
    const PAGE: usize = 4096;
    const MADV_DONTNEED: i32 = 4;
    let start = ptr as usize;
    let end = start.saturating_add(len);
    let aligned_start = (start + PAGE - 1) & !(PAGE - 1);
    let aligned_end = end & !(PAGE - 1);
    if aligned_end <= aligned_start {
        return;
    }
    extern "C" {
        fn madvise(addr: *mut core::ffi::c_void, length: usize, advice: i32) -> i32;
    }
    // SAFETY: whole-page sub-range of a live read-only file mapping;
    // MADV_DONTNEED drops resident pages (re-read from the file on next
    // access for a file-backed mapping).
    unsafe {
        let _ = madvise(
            aligned_start as *mut core::ffi::c_void,
            aligned_end - aligned_start,
            MADV_DONTNEED,
        );
    }
}
#[cfg(not(any(windows, target_os = "linux")))]
fn discard_mmap_pages(_ptr: *const u8, _len: usize) {}

/// The raw SYCL device index to install for compute dispatch: the first
/// SYCL GPU (in the stable enumeration order the GUI shows) that is NOT
/// in the disable-list (`RUSTLLAMA_DISABLED_GPUS`). Intel/SYCL devices
/// enumerate first, so their unified index equals their SYCL position;
/// we replicate `gpus_probe`'s dedup-by-name so a unified index maps to
/// the correct raw SYCL index even when one physical GPU exposes several
/// SYCL backends. Returns `None` when every SYCL GPU is disabled (→ the
/// engine stays on the CPU path); `Some(0)` when the list is empty.
pub fn first_enabled_sycl_device_index() -> Option<u32> {
    let disabled: std::collections::HashSet<u32> = std::env::var("RUSTLLAMA_DISABLED_GPUS")
        .ok()
        .map(|s| {
            s.split(|c: char| c == ',' || c.is_whitespace())
                .filter_map(|t| t.trim().parse::<u32>().ok())
                .collect()
        })
        .unwrap_or_default();
    let n = sk::device_count().unwrap_or(0);
    if n == 0 {
        // Preserve the historical "empty list → Some(0)" contract so
        // callers that never had a device still behave as before.
        return if disabled.is_empty() { Some(0) } else { None };
    }
    // Group raw SYCL indices by physical GPU (name), in enumeration
    // order — one physical GPU can expose several backend views (the
    // Iris Xe: index 0 = Level Zero, index 1 = OpenCL). The unified
    // index (what the disable-list refers to) advances once per
    // physical GPU, matching `gpus_probe`'s dedup.
    let mut order: Vec<String> = Vec::new();
    let mut views_by_name: std::collections::HashMap<String, Vec<u32>> =
        std::collections::HashMap::new();
    for i in 0..n {
        let Ok(info) = sk::device_info(i) else { continue };
        views_by_name
            .entry(info.name.clone())
            .or_insert_with(|| {
                order.push(info.name.clone());
                Vec::new()
            })
            .push(i);
    }
    // First non-disabled physical GPU wins; within it, prefer the
    // Level Zero backend view, then OpenCL, then whatever enumerates
    // first (the user's "always Level Zero if available, else OpenCL,
    // collapse accordingly" policy).
    for (unified, name) in order.iter().enumerate() {
        if disabled.contains(&(unified as u32)) {
            continue;
        }
        return Some(preferred_backend_view(&views_by_name[name]));
    }
    None
}

/// Pick the preferred SYCL device index among the backend views of one
/// physical GPU: Level Zero if present, else OpenCL, else the first
/// view. Probes each view's backend via an ephemeral stream (cheap,
/// once at engine setup).
fn preferred_backend_view(views: &[u32]) -> u32 {
    let mut opencl: Option<u32> = None;
    for &i in views {
        match sk::current_backend_name(i) {
            Some("level_zero") => return i,
            Some("opencl") if opencl.is_none() => opencl = Some(i),
            _ => {}
        }
    }
    opencl.or_else(|| views.first().copied()).unwrap_or(0)
}

/// Available physical RAM (bytes) — `GlobalMemoryStatusEx().ullAvailPhys`
/// on Windows, `u64::MAX` on other platforms (signal: "no limit known,
/// assume room"). Used by the adaptive USM gate to decide whether the
/// model fits in RAM with comfortable headroom (skip the cap) or not
/// (enforce it). Cheap — one Win32 call, only on first matvec touch.
#[cfg(windows)]
fn avail_phys_bytes() -> u64 {
    #[repr(C)]
    struct MEMORYSTATUSEX {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }
    extern "system" {
        fn GlobalMemoryStatusEx(lp_buffer: *mut MEMORYSTATUSEX) -> i32;
    }
    let mut s: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    s.dw_length = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: zero-initialized buffer with the correct dwLength as the
    // API requires.
    if unsafe { GlobalMemoryStatusEx(&mut s) } == 0 {
        return u64::MAX; // failed → assume room, don't accidentally enforce gate
    }
    s.ull_avail_phys
}
#[cfg(not(windows))]
fn avail_phys_bytes() -> u64 {
    u64::MAX
}

/// Adaptive gate decision: given the env-var budget, decide whether to
/// actually enforce it on this matvec context. Sample AvailPhys once;
/// if there's comfortable headroom *above* the budget — i.e., even if
/// we duplicated `env_budget` bytes into USM there'd still be ≥ 2 GB
/// free — the model clearly fits and the cap would only hurt (it
/// forces a mixed GPU/CPU matvec path on already-fitting models like
/// qwen2.5-coder, which regresses speed *and* numerics from H6/H8's
/// mixed-precision fusion vs pure GPU). In that case return `0` =
/// unlimited (gate bypassed). Otherwise enforce the env budget.
///
/// `env_budget == 0` (user explicitly off) → return `0` (no gate).
fn resolve_adaptive_matvec_budget(env_budget: usize) -> usize {
    if env_budget == 0 {
        return 0;
    }
    let avail = avail_phys_bytes();
    // 2 GB safety margin above the projected USM duplication.
    let headroom: u64 = 2 * 1024 * 1024 * 1024;
    let projected_residual = avail.saturating_sub(env_budget as u64);
    if projected_residual > headroom {
        log_usm_gate_bypassed_once(avail, env_budget);
        return 0;
    }
    env_budget
}

/// One-time info log: the adaptive gate bypassed the env cap because
/// the model fits with headroom. Surfaces *why* the cap isn't enforced
/// even though the env var is set — diagnoses the qwen2.5-coder-style
/// "small model that previously fit fully on GPU" case.
fn log_usm_gate_bypassed_once(avail: u64, env_budget: usize) {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        tracing::info!(
            avail_phys_mb = avail / (1024 * 1024),
            env_budget_mb = (env_budget as u64) / (1024 * 1024),
            "RUSTLLAMA_MATVEC_USM_MAX_MB set but RAM has headroom — adaptive gate \
             bypassed (model fits, pure GPU matmul stays active)"
        );
    });
}

/// One-time info log the first time the USM weight-cache budget is hit
/// (further matvec weights route to CPU). Surfaces the gate so a slow
/// run isn't a silent mystery.
fn log_usm_weight_budget_once() {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        tracing::info!(
            budget_mb = matvec_usm_weight_budget_bytes() / (1024 * 1024),
            "RUSTLLAMA_MATVEC_USM_MAX_MB reached — further matvec weights run on CPU \
             (reading file-backed bytes in place) to avoid USM double-residency"
        );
    });
}

/// One-time info log the first time the dynamic RAM-safety floor stops
/// USM weight uploads (the automatic unified-pool guard; see
/// [`usm_ram_floor_bytes`]). Reports how much weight is USM-resident so
/// the GPU/CPU split is visible rather than a silent slowdown.
fn log_usm_ram_floor_once(usm_resident_bytes: usize, avail: u64, floor: u64) {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        tracing::info!(
            usm_resident_mb = usm_resident_bytes / (1024 * 1024),
            avail_phys_mb = avail / (1024 * 1024),
            floor_mb = floor / (1024 * 1024),
            "USM weight residency hit the host-RAM safety floor — remaining weights run \
             on CPU (file-backed) to preserve host-RAM headroom (RUSTLLAMA_USM_RAM_FLOOR_MB)"
        );
    });
}

thread_local! {
    /// Per-thread USM attention context. Built lazily on the
    /// first `try_flash_attn_decode_usm_f32` call after the
    /// `RUSTLLAMA_USM_ATTN` env var is set and a matching
    /// config is presented. Rebuilt on config mismatch (e.g. a
    /// different model loaded in the same binary).
    static USM_ATTN: std::cell::RefCell<Option<UsmAttnContext>> =
        const { std::cell::RefCell::new(None) };
    /// Per-thread "weights pre-uploaded" flag. The engine drives
    /// generations on a pool of tokio blocking-pool threads, each
    /// with its own USM context. We want to pre-upload model
    /// weights once per thread (first time that thread sees a
    /// generation), not once per process. A `static OnceLock`
    /// would fire once globally and other threads would silently
    /// skip the preload + pay the lazy-upload cost on their first
    /// generation.
    static USM_WEIGHTS_PRELOADED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Has the calling thread completed the one-time packed-weight
/// pre-upload? Returns `false` until [`mark_packed_weights_preloaded`]
/// is called on this thread. Used by the engine to gate
/// `Model::preload_packed_weights_to_usm` so it runs once per
/// worker thread.
pub fn packed_weights_preloaded() -> bool {
    USM_WEIGHTS_PRELOADED.with(|c| c.get())
}

/// Mark the calling thread as having completed the packed-weight
/// pre-upload. Sets a per-thread flag — does not actually upload
/// anything. The engine calls this after a successful
/// `Model::preload_packed_weights_to_usm`.
pub fn mark_packed_weights_preloaded() {
    USM_WEIGHTS_PRELOADED.with(|c| c.set(true));
}

/// Invalidate the matvec-x dedup cache. Call between forward
/// passes (or any time the engine knows a host buffer that the
/// cache might reference is no longer guaranteed-fresh) so a
/// stale `(ptr, len)` key from the previous generation can't
/// produce a false hit against a re-allocated buffer that happens
/// to land at the same address.
///
/// The cache is also auto-invalidated on USM scratch re-alloc
/// (handled inside the matvec hooks). This function covers the
/// case where the host-side buffer changes without the USM side
/// changing — e.g. a new generation's `hidden` Vec lands at the
/// same heap address as the previous generation's.
/// One-time-per-process driver-acceptance probe for the Level Zero
/// host-pointer import path. Creates a tiny anonymous Win32 file
/// mapping, asks the L0 driver to import it via `zeMemAllocHost`
/// with `ze_external_memory_import_win32_handle_t`, and logs the
/// return code via `tracing::info!`.
///
/// We need this because the L0 spec describes `OPAQUE_WIN32` as
/// "Win32 NT handle" — file-mapping section handles qualify in
/// theory, but the Intel driver implementation may only accept
/// handles produced by other GPU APIs (D3D / Vulkan). The probe
/// tells us BEFORE we refactor `rustllama-gguf` whether the
/// mechanism works on this driver.
///
/// Return codes (matches `rsl_try_import_win32_handle_as_usm`):
///   0 = success → driver accepts the import; Step 1c/1d unblocked
///   2 = SYCL queue isn't L0 → backend selector issue
///   3 = L0 loader / symbol missing
///   4 = `zeMemAllocHost` returned non-success → driver rejects
///       this handle type; engine import path will need a different
///       primitive or to skip the import path entirely
///
/// Idempotent: only runs once per process. Safe to call on
/// non-Windows or with no USM context — both early-return.
pub fn probe_l0_import_once() {
    static PROBED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    PROBED.get_or_init(|| {
        #[cfg(target_os = "windows")]
        {
            // Probe order matters: baseline first to validate our
            // basic L0 call sequence (no import descriptor); then
            // anon + file-backed import attempts. If baseline
            // succeeds but imports fail, the issue is specifically
            // the import descriptor / handle origin — pivot away
            // from L0 import. If baseline ALSO fails, our struct
            // layout is wrong and we should fix it before deciding.
            probe_l0_baseline();
            probe_l0_anon_mapping();
            probe_l0_file_backed_mapping();
        }
        #[cfg(not(target_os = "windows"))]
        {
            tracing::debug!(
                "L0 import probe skipped: non-Windows target (Win32 file mapping path)"
            );
        }
    });
}

/// Baseline: `zeMemAllocHost` with no import descriptor. Verifies
/// our `ze_host_mem_alloc_desc` layout + L0 call sequence work.
#[cfg(target_os = "windows")]
fn probe_l0_baseline() {
    let result = USM_ATTN.with(|cell| -> Option<sk::Result<()>> {
        let slot = cell.borrow();
        let ctx = slot.as_ref()?;
        Some(sk::try_alloc_host_baseline(&ctx.stream, 64 * 1024))
    });
    match result {
        Some(Ok(())) => {
            tracing::info!(
                "L0 import probe (baseline): bare zeMemAllocHost SUCCESS \
                 (no import descriptor). Our struct layout + call sequence \
                 are correct; subsequent import failures are about the \
                 descriptor / handle origin, not our code."
            );
        }
        Some(Err(sk::SyclError::L0ImportUnsupported(code))) => {
            let l0_code = if code == 4 {
                sk::consume_last_l0_import_code()
            } else {
                0
            };
            tracing::warn!(
                code,
                l0_code = format!("0x{l0_code:08x}"),
                "L0 import probe (baseline): bare zeMemAllocHost FAILED. \
                 Our struct layout or call sequence is wrong; fix before \
                 evaluating the import path."
            );
        }
        Some(Err(e)) => {
            tracing::warn!(error = %e, "L0 import probe (baseline): unexpected error");
        }
        None => {
            tracing::debug!("L0 import probe (baseline): skipped (no context)");
        }
    }
}

// Hand-rolled Win32 surface so the engine doesn't take a
// `windows-sys` dependency for two probe calls.
#[cfg(target_os = "windows")]
type Win32Handle = *mut std::ffi::c_void;
#[cfg(target_os = "windows")]
const INVALID_HANDLE_VALUE: Win32Handle = usize::MAX as *mut _;
#[cfg(target_os = "windows")]
const PAGE_READWRITE: u32 = 0x04;
#[cfg(target_os = "windows")]
const GENERIC_READ: u32 = 0x80000000;
#[cfg(target_os = "windows")]
const GENERIC_WRITE: u32 = 0x40000000;
#[cfg(target_os = "windows")]
const CREATE_ALWAYS: u32 = 2;
#[cfg(target_os = "windows")]
const FILE_ATTRIBUTE_TEMPORARY: u32 = 0x100;
#[cfg(target_os = "windows")]
const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x04000000;

#[cfg(target_os = "windows")]
extern "system" {
    fn CreateFileMappingW(
        hFile: Win32Handle,
        lpAttrs: *mut std::ffi::c_void,
        flProtect: u32,
        dwMaximumSizeHigh: u32,
        dwMaximumSizeLow: u32,
        lpName: *const u16,
    ) -> Win32Handle;
    fn CreateFileW(
        lpFileName: *const u16,
        dwDesiredAccess: u32,
        dwShareMode: u32,
        lpSecurityAttributes: *mut std::ffi::c_void,
        dwCreationDisposition: u32,
        dwFlagsAndAttributes: u32,
        hTemplateFile: Win32Handle,
    ) -> Win32Handle;
    fn CloseHandle(h: Win32Handle) -> i32;
    fn GetTempPathW(nBufferLength: u32, lpBuffer: *mut u16) -> u32;
    fn GetTempFileNameW(
        lpPathName: *const u16,
        lpPrefixString: *const u16,
        uUnique: u32,
        lpTempFileName: *mut u16,
    ) -> u32;
}

/// L0 import probe variant: anonymous (page-file backed) mapping.
/// This is the "weakest case" — if the driver accepts arbitrary
/// Win32 NT handles, it must accept this. If it rejects this but
/// accepts file-backed, we'll learn that from the second probe.
#[cfg(target_os = "windows")]
fn probe_l0_anon_mapping() {
    let result = USM_ATTN.with(|cell| -> Option<sk::Result<*mut std::ffi::c_void>> {
        let slot = cell.borrow();
        let ctx = slot.as_ref()?;
        let size: usize = 64 * 1024;
        let mapping = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null_mut(),
                PAGE_READWRITE,
                0,
                size as u32,
                std::ptr::null(),
            )
        };
        if mapping.is_null() {
            return None;
        }
        let r =
            unsafe { sk::try_import_win32_handle_as_usm_raw(&ctx.stream, mapping, size) };
        if let Ok(ptr) = &r {
            unsafe { sk::release_imported_usm_raw(&ctx.stream, *ptr) };
        }
        unsafe { CloseHandle(mapping) };
        Some(r)
    });
    report_l0_probe("anon (page-file backed)", result);
}

/// L0 import probe variant: file-backed mapping over a temp file.
/// This matches the production case where the GGUF is a real file
/// on disk. If the anon probe failed but this succeeds, the driver
/// is being picky about NT handle kind — implies the import path
/// IS viable for real GGUFs.
#[cfg(target_os = "windows")]
fn probe_l0_file_backed_mapping() {
    let result = USM_ATTN.with(|cell| -> Option<sk::Result<*mut std::ffi::c_void>> {
        let slot = cell.borrow();
        let ctx = slot.as_ref()?;
        // Get a temp file path from the OS.
        let mut tmp_dir = [0u16; 260];
        let n = unsafe { GetTempPathW(tmp_dir.len() as u32, tmp_dir.as_mut_ptr()) };
        if n == 0 || n as usize >= tmp_dir.len() {
            tracing::warn!("L0 file-backed probe: GetTempPathW failed");
            return None;
        }
        let prefix: Vec<u16> = "rsl\0".encode_utf16().collect();
        let mut tmp_file = [0u16; 260];
        let n = unsafe {
            GetTempFileNameW(
                tmp_dir.as_ptr(),
                prefix.as_ptr(),
                0,
                tmp_file.as_mut_ptr(),
            )
        };
        if n == 0 {
            tracing::warn!("L0 file-backed probe: GetTempFileNameW failed");
            return None;
        }
        // Open the temp file with DELETE_ON_CLOSE so it auto-cleans
        // when we drop the handle.
        let file = unsafe {
            CreateFileW(
                tmp_file.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                0,
                std::ptr::null_mut(),
                CREATE_ALWAYS,
                FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE,
                std::ptr::null_mut(),
            )
        };
        if file == INVALID_HANDLE_VALUE {
            tracing::warn!("L0 file-backed probe: CreateFileW failed");
            return None;
        }
        let size: usize = 64 * 1024;
        let mapping = unsafe {
            CreateFileMappingW(
                file,
                std::ptr::null_mut(),
                PAGE_READWRITE,
                0,
                size as u32,
                std::ptr::null(),
            )
        };
        if mapping.is_null() {
            unsafe { CloseHandle(file) };
            return None;
        }
        let r =
            unsafe { sk::try_import_win32_handle_as_usm_raw(&ctx.stream, mapping, size) };
        if let Ok(ptr) = &r {
            unsafe { sk::release_imported_usm_raw(&ctx.stream, *ptr) };
        }
        unsafe { CloseHandle(mapping) };
        unsafe { CloseHandle(file) }; // triggers DELETE_ON_CLOSE
        Some(r)
    });
    report_l0_probe("file-backed (temp file)", result);
}

/// Format and log the result of a single L0 import probe variant.
/// Captures the exact `ze_result_t` from the side channel when the
/// probe fails with category code 4 ("driver rejected") so we can
/// distinguish UNSUPPORTED_FEATURE from INVALID_ARGUMENT and friends.
#[cfg(target_os = "windows")]
fn report_l0_probe(variant: &str, result: Option<sk::Result<*mut std::ffi::c_void>>) {
    match result {
        Some(Ok(_)) => {
            tracing::info!(
                variant,
                "L0 import probe: driver ACCEPTS this handle kind \
                 (zeMemAllocHost succeeded). GGUF zero-copy path viable \
                 with this mapping shape."
            );
        }
        Some(Err(sk::SyclError::L0ImportUnsupported(code))) => {
            let kind = match code {
                1 => "invalid args (programmer bug)",
                2 => "SYCL queue not on L0 backend (selector issue)",
                3 => "ze_loader.dll / zeMemAllocHost missing",
                4 => "driver rejected import (zeMemAllocHost returned error)",
                _ => "unknown",
            };
            // For category 4, drain the exact L0 result code so we
            // can decode WHAT the driver objected to.
            let l0_code = if code == 4 {
                sk::consume_last_l0_import_code()
            } else {
                0
            };
            let l0_name = match l0_code {
                0x78000001 => "ERROR_UNINITIALIZED",
                0x78000002 => "ERROR_DEVICE_LOST",
                0x78000003 => "ERROR_UNSUPPORTED_FEATURE",
                0x78000004 => "ERROR_INVALID_ARGUMENT",
                0x70000002 => "ERROR_OUT_OF_HOST_MEMORY",
                0x70000005 => "ERROR_OUT_OF_DEVICE_MEMORY",
                0x70010000..=0x70019fff => "ERROR_INVALID_NULL_*",
                _ if l0_code == 0 => "n/a (no L0 code recorded)",
                _ => "unknown L0 code",
            };
            tracing::warn!(
                variant,
                code,
                kind,
                l0_code = format!("0x{l0_code:08x}"),
                l0_name,
                "L0 import probe: driver does NOT accept this handle kind"
            );
        }
        Some(Err(e)) => {
            tracing::warn!(variant, error = %e, "L0 import probe: unexpected error");
        }
        None => {
            tracing::debug!(variant, "L0 import probe: skipped (no context or setup failure)");
        }
    }
}

/// Look up the autotuner's preferred local-work-group size for a
/// `(kernel, M, K)` triple. Returns `0` when no tuned entry exists
/// — the C++ kernel TU treats `0` as "use the hand-picked default
/// (64)", so a miss falls through cleanly to pre-tuner behavior.
///
/// `kernel_name` is one of the constants in
/// `rustllama_tuner::PACKED_USM_KERNELS`. Using a typed constant
/// avoids both engine-side and tuner-CLI-side typos.
///
/// Cache pipeline:
///   1. [`load_lws_cache_for_device_once`] loads
///      `%LOCALAPPDATA%\rustllama\tuning\<fingerprint>.toml` once
///      per process, on the first matvec dispatch where SYCL is
///      active.
///   2. Cached entries are stored in a process-global `HashMap`
///      keyed on `(kernel_name, M, K)`. Reads are lock-free on the
///      hot path via an `OnceLock<HashMap<…>>`.
///   3. A miss is logged once per `(kernel, shape)` per process so
///      users see "shape X is untuned — running with default LWS"
///      in `gui.log` and can `rustllama tune` to fix it.
#[inline]
fn tuned_lws_for(kernel_name: &'static str, m: usize, k: usize) -> u32 {
    let cache = load_lws_cache_for_device_once();
    match cache.get(&(kernel_name, m, k)) {
        Some(&lws) => lws,
        None => {
            log_untuned_shape_once(kernel_name, m, k);
            0
        }
    }
}

/// Per-process LWS lookup table covering every packed USM matvec
/// kernel listed in `rustllama_tuner::PACKED_USM_KERNELS`. Populated
/// once from the on-disk autotuner cache. Empty when SYCL is mock-
/// mode, the device fingerprint fails, or the cache file is missing
/// — every lookup falls through to `0` (default LWS).
fn load_lws_cache_for_device_once(
) -> &'static std::collections::HashMap<(&'static str, usize, usize), u32> {
    static CACHE: std::sync::OnceLock<
        std::collections::HashMap<(&'static str, usize, usize), u32>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(load_lws_cache_impl)
}

fn load_lws_cache_impl() -> std::collections::HashMap<(&'static str, usize, usize), u32> {
    use std::collections::HashMap;
    // Cache key = whole-system fingerprint (resolves on SYCL/CUDA/CPU hosts).
    // On a non-SYCL host the file simply carries no kernel-LWS entries, so the
    // map ends up empty and every dispatch falls through to the default LWS.
    let key = rustllama_tuner::system_fingerprint();
    let dir = match rustllama_tuner::default_cache_dir() {
        Some(d) => d,
        None => {
            tracing::debug!("tuner: no default cache dir; LWS cache stays empty");
            return HashMap::new();
        }
    };
    let tuning = match rustllama_tuner::load_cache(&dir, &key) {
        Ok(Some(t)) => t,
        Ok(None) => {
            tracing::info!(
                fingerprint = %key,
                cache_path = %rustllama_tuner::cache_path_for(&dir, &key).display(),
                "tuner: no cache file for this system — run `rustllama tune` to populate"
            );
            return HashMap::new();
        }
        Err(e) => {
            tracing::warn!(
                fingerprint = %key,
                error = %e,
                "tuner: failed to read cache file; LWS cache stays empty"
            );
            return HashMap::new();
        }
    };
    let mut map: HashMap<(&'static str, usize, usize), u32> = HashMap::new();
    // Iterate every kernel entry in the cache. Each entry's bucket
    // string is parsed to extract `(M, K)`. Unknown bucket formats
    // are skipped with a one-time warning so a hand-edited cache
    // doesn't silently mis-tune.
    for (kname, bucket, lws) in rustllama_tuner::iter_tuned_lws(&tuning) {
        let interned: Option<&'static str> = rustllama_tuner::PACKED_USM_KERNELS
            .iter()
            .copied()
            .find(|&k| k == kname);
        let Some(kname_static) = interned else {
            tracing::warn!(
                kernel = kname,
                "tuner cache contains entry for unknown kernel; skipping"
            );
            continue;
        };
        match parse_q4k_shape_bucket(&bucket) {
            Some((m, k)) => {
                map.insert((kname_static, m, k), lws);
            }
            None => tracing::warn!(
                kernel = kname,
                bucket,
                "tuner cache entry has malformed shape bucket; skipping"
            ),
        }
    }
    tracing::info!(
        fingerprint = %key,
        entries = map.len(),
        "tuner: packed-USM LWS cache loaded"
    );
    map
}

/// Parse a `"M=<m>,K=<k>"` bucket string (see
/// `rustllama_tuner::q4k_usm_shape_bucket`) back into `(M, K)`.
/// Returns `None` on any deviation from the expected shape so a
/// malformed cache entry can't silently load as `(0, 0)`.
fn parse_q4k_shape_bucket(s: &str) -> Option<(usize, usize)> {
    let (m_part, k_part) = s.split_once(',')?;
    let m = m_part.strip_prefix("M=")?.parse::<usize>().ok()?;
    let k = k_part.strip_prefix("K=")?.parse::<usize>().ok()?;
    Some((m, k))
}

/// Log the first time a given `(kernel, M, K)` triple misses the
/// tuned LWS cache. Pinned per-process per-triple so a chat that
/// hits many distinct shapes doesn't spam `gui.log`. The aggregate
/// "you have N untuned shapes" picture comes from counting the
/// unique log lines.
fn log_untuned_shape_once(kernel: &'static str, m: usize, k: usize) {
    let is_new = {
        let mut g = untuned_shapes_lock().lock().unwrap();
        g.insert((kernel, m, k))
    };
    if is_new {
        tracing::debug!(
            kernel, m, k,
            "{kernel} matvec shape (M={m},K={k}) has no tuned LWS; using kernel default. \
             Run `rustllama tune` to record the winner for this device."
        );
    }
}

/// Per-process registry of "matvec shape (kernel, M, K) was dispatched
/// but the autotuner cache had no entry for it." This is the
/// pull-shaped counterpart to the plan's `TuningRecommended` event —
/// rather than broadcasting on every miss, we accumulate the set and
/// the server exposes a `/v1/tuning/recommendations` endpoint that the
/// GUI Status page polls to render the "N kernels untuned — Tune now"
/// banner.
fn untuned_shapes_lock(
) -> &'static std::sync::Mutex<std::collections::HashSet<(&'static str, usize, usize)>> {
    use std::sync::Mutex;
    static SEEN: std::sync::OnceLock<
        Mutex<std::collections::HashSet<(&'static str, usize, usize)>>,
    > = std::sync::OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}

/// Snapshot the accumulated `(kernel, M, K)` triples that the engine
/// has seen *without* a cached LWS entry. The server exposes this via
/// `/v1/tuning/recommendations`. Returns an owned `Vec` so the lock
/// is held for the duration of the snapshot only.
///
/// Order is insertion-stable across reads (we sort lexicographically
/// here so the wire shape is deterministic and the GUI's "N kernels"
/// count is stable across polls even though the underlying HashSet's
/// iteration order isn't).
pub fn snapshot_untuned_shapes() -> Vec<(&'static str, usize, usize)> {
    let mut out: Vec<(&'static str, usize, usize)> = {
        let g = untuned_shapes_lock().lock().unwrap();
        g.iter().copied().collect()
    };
    out.sort();
    out
}

/// Reset the untuned-shape registry. Called by the server after a
/// successful tune so the GUI's "N kernels untuned" banner clears
/// without needing a full restart. The next dispatch with an actual
/// cache hit won't re-add the entry; a dispatch that still misses
/// will, which is the right surfaced behavior ("we tried, you still
/// have shapes we couldn't tune").
pub fn clear_untuned_shapes() {
    let mut g = untuned_shapes_lock().lock().unwrap();
    g.clear();
}

/// Eagerly upload a packed-quant weight tensor to the USM weight
/// cache. Returns `true` if the upload happened (or was already
/// cached), `false` if the tensor's dtype isn't a USM-supported
/// packed quant, USM isn't enabled / available, the host context
/// isn't ready, or allocation failed.
///
/// Used by [`preload_packed_weights_to_usm`] (and callable
/// standalone) to shift the first-prefill lazy-upload cost to
/// model-load time. On Iris Xe shared memory the upload is
/// effectively a memcpy + page-mapping setup; for a 7B Q4_K_M
/// model this adds ~1-2 s to load but makes the first chat as
/// fast as steady-state chats.
pub fn preload_packed_tensor_to_usm(w: &Tensor) -> bool {
    if !usm_attn_enabled() {
        return false;
    }
    let kind = match w.dtype {
        Dtype::Q8_0Raw => PackedMatvecKind::Q8_0,
        Dtype::Q4_KRaw => PackedMatvecKind::Q4_K,
        Dtype::Q5_KRaw => PackedMatvecKind::Q5_K,
        Dtype::Q6_KRaw => PackedMatvecKind::Q6_K,
        Dtype::IQ4_NLRaw => PackedMatvecKind::IQ4_NL,
        Dtype::IQ4_XSRaw => PackedMatvecKind::IQ4_XS,
        Dtype::IQ1_SRaw => PackedMatvecKind::IQ1_S,
        Dtype::IQ2_XXSRaw => PackedMatvecKind::IQ2_XXS,
        Dtype::IQ1_MRaw => PackedMatvecKind::IQ1_M,
        Dtype::IQ2_XSRaw => PackedMatvecKind::IQ2_XS,
        Dtype::IQ2_SRaw => PackedMatvecKind::IQ2_S,
        Dtype::IQ3_XXSRaw => PackedMatvecKind::IQ3_XXS,
        Dtype::IQ3_SRaw => PackedMatvecKind::IQ3_S,
        Dtype::PTQ1_0Raw => PackedMatvecKind::PTQ1_0,
        Dtype::Q4_0Raw => PackedMatvecKind::Q4_0,
        Dtype::Q5_0Raw => PackedMatvecKind::Q5_0,
        Dtype::Q4_1Raw => PackedMatvecKind::Q4_1,
        Dtype::Q5_1Raw => PackedMatvecKind::Q5_1,
        Dtype::Q2_KRaw => PackedMatvecKind::Q2_K,
        Dtype::Q3_KRaw => PackedMatvecKind::Q3_K,
        Dtype::Q8_KRaw => PackedMatvecKind::Q8_K,
        Dtype::PQ2_0Raw => PackedMatvecKind::PQ2_0,
        Dtype::Mxfp4Raw => PackedMatvecKind::Mxfp4,
        Dtype::Mxfp6Raw => PackedMatvecKind::Mxfp6,
        Dtype::Mxfp8Raw => PackedMatvecKind::Mxfp8,
        _ => return false,
    };
    // Weight tensors are shaped `[M, K]` (out × in). The engine's
    // matvec call passes `(M=shape[0], k=shape[1])`, so we mirror.
    let shape = &w.shape;
    if shape.len() < 2 {
        return false;
    }
    let m = shape[0] as usize;
    let k = shape[1] as usize;
    if k == 0 || m == 0 || k % kind.k_alignment() != 0 {
        return false;
    }
    let w_bytes = as_bytes(w);
    let expected_bytes = m * kind.row_bytes(k);
    if w_bytes.len() < expected_bytes {
        return false;
    }
    let weight_key = w_bytes.as_ptr() as usize;
    // Only file-backed (GGUF mmap) weights are safe to discard after the
    // USM copy — heap-owned/USM storage returns None and is left intact.
    let file_backed = w.storage.mmap_borrowed_ptr_len().is_some();
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(ctx) = slot.as_mut() else {
            return false;
        };
        ctx.cache_packed_weight(weight_key, expected_bytes, w_bytes, file_backed)
    })
}

/// Pre-build the per-thread USM attention context using the
/// model's structural dims. Call this once at the start of every
/// generation request — before any forward pass — so the packed-
/// matvec USM hook can fire during *prefill* instead of having
/// to wait for the first `try_flash_attn_decode_usm_f32` call
/// (which only runs during decode).
///
/// Returns `true` on success or when a matching context already
/// exists; `false` when SYCL is unavailable, USM is opted out via
/// env, or the allocation fails. Callers MUST tolerate `false` —
/// the engine simply continues on the CPU path.
///
/// The `d` / `d_ff` knobs on `UsmAttnConfig` are left at zero
/// here; the rmsnorm / silu / matvec hooks each lazy-grow their
/// own scratch slots on first use, so we don't need to pre-size
/// them.
pub fn prepare_usm_context(
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
) -> bool {
    if !usm_attn_enabled() {
        tracing::info!("prepare_usm_context skipped: usm_attn_enabled() = false");
        return false;
    }
    let cfg = UsmAttnConfig {
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        max_ctx,
        d: 0,
        d_ff: 0,
    };
    let result = USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.as_ref().map_or(false, |ctx| ctx.cfg == cfg) {
            return true;
        }
        *slot = UsmAttnContext::try_new(cfg);
        slot.is_some()
    });
    if result {
        // K/V cache total: 2 (K+V) * n_layers * n_kv_heads * max_ctx
        // * head_dim * 2 bytes (f16). Match the size SyclSharedBuffer
        // allocates inside UsmAttnContext::try_new.
        let kv_total_bytes = 2u64
            * n_layers as u64
            * n_kv_heads as u64
            * max_ctx as u64
            * head_dim as u64
            * 2;
        tracing::info!(
            kv_cache_mib = kv_total_bytes / (1024 * 1024),
            "USM attention context built — K/V caches committed to USM-shared memory"
        );
    } else {
        tracing::warn!("prepare_usm_context failed: USM allocator returned Err — engine will run on CPU/host-pointer SYCL path");
    }
    result
}

/// Global epoch for the host KV caches' identity. Bumped by
/// `KvCache::reset` / `KvCache::restore_prefix` (any wholesale
/// change of what the host KV holds). Every thread-local USM
/// attention mirror records the epoch it was valid under; on the
/// next dispatch with a stale epoch the mirror's `kv_valid_len`
/// drops to zero and rows re-validate contiguously from position 0.
/// This is what makes the thread-local mirror safe when requests
/// hop worker threads or when a prefix restore skips prefill: the
/// mirror can never present another sequence's rows as current.
static USM_KV_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Announce that the host KV caches' content identity changed
/// (reset, or restore from a prefix snapshot). Called by the KV
/// cache itself so every engine path is covered without per-site
/// wiring. Cheap: one relaxed atomic increment.
pub fn usm_attn_kv_epoch_bump() {
    USM_KV_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// After this many consecutive SYCL kernel failures, the per-thread
/// `SYCL_HEALTHY` flag flips to `false` and we stop trying SYCL for
/// the rest of this thread's lifetime. 8 is small enough that we
/// don't waste minutes on a broken driver, large enough to absorb
/// transient hiccups (one or two failed allocations during JIT warm-up).
const SYCL_FAIL_BUDGET: u32 = 8;

/// Per-thread circuit breaker. Returns `true` while SYCL is healthy
/// on this thread; `false` once it's been disabled by repeated
/// failures. Cheap atomic-load on the fast path.
pub fn sycl_healthy() -> bool {
    SYCL_HEALTHY.with(|h| h.get())
}

/// Record a SYCL kernel success on this thread. Resets the
/// consecutive-failure counter. Cheap; the engine hooks call this
/// after every successful kernel dispatch.
pub fn note_sycl_success() {
    SYCL_FAIL_COUNT.with(|c| c.set(0));
}

/// Record a SYCL kernel failure on this thread. Bumps the failure
/// counter; if it crosses [`SYCL_FAIL_BUDGET`] the breaker trips
/// and subsequent `sycl_healthy()` calls return `false`. Logs once
/// at warn level when the trip happens so the user sees a single
/// signal that "SYCL gave up — running on CPU" rather than an
/// undifferentiated stream of swallowed exceptions.
pub fn note_sycl_failure() {
    let tripped = SYCL_FAIL_COUNT.with(|c| {
        let n = c.get() + 1;
        c.set(n);
        n >= SYCL_FAIL_BUDGET
    });
    if tripped && SYCL_HEALTHY.with(|h| h.replace(false)) {
        tracing::warn!(
            failures = SYCL_FAIL_BUDGET,
            "SYCL dispatch produced repeated FFI failures on this thread \
             (e.g. `std::bad_alloc` from the runtime). Disabling SYCL on \
             this worker for the rest of its lifetime; falling back to \
             CPU. Restart the GUI to retry."
        );
    }
}

/// Whether the USM-resident attention / matvec hooks should route
/// to the SYCL kernels. On a real-SYCL build with a visible device,
/// this defaults to **enabled** — callers can opt out with
/// `RUSTLLAMA_USM_ATTN=0` (or `false`) for A/B testing against the
/// host-pointer SYCL path or pure CPU.
///
/// Mock builds and hosts with no SYCL device visible silently
/// return `false` (the underlying `device_count()` query returns
/// `Err(Unavailable)` or `Ok(0)`).
fn usm_attn_enabled() -> bool {
    if !sycl_healthy() {
        return false;
    }
    // Cache BOTH the env read and the device probe. This function is
    // consulted up to three times per matvec dispatch (~1300-2500
    // times per decoded token on a 41-layer MoE); the previous
    // uncached `std::env::var` here took the process-wide env lock
    // and allocated on every call — the single largest constant
    // overhead in the decode loop. A/B lever: set RUSTLLAMA_USM_ATTN
    // before the first forward; cached per process like every other
    // gate in this file.
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        if let Ok(v) = std::env::var("RUSTLLAMA_USM_ATTN") {
            // Explicit setting wins either way — `=1` forces GPU
            // dispatch even on backends the auto-gate refuses.
            return !(v == "0" || v.eq_ignore_ascii_case("false"));
        }
        if !sk::device_count().ok().map(|n| n > 0).unwrap_or(false) {
            return false;
        }
        // Auto mode: GPU dispatch only when the SYCL runtime landed
        // on Level Zero. On the OpenCL fallback (Intel Compute
        // Runtime / ze_loader missing) the kernel suite is NOT
        // numerically trustworthy: on Iris Xe/OpenCL, 2026-09-06,
        // the flash-attention chain and the fused gate_up kernels
        // both produced garbage logits (mixed-script token soup at
        // greedy) while the same model was byte-exact correct on
        // CPU. Under memory pressure the same backend instead threw
        // UR OUT_OF_RESOURCES on every launch — so it was never
        // actually computing in earlier "working" GPU runs. CPU is
        // both correct and, in practice, what this backend silently
        // degraded to anyway.
        let backend = sk::current_backend_name(0);
        let ok = backend == Some("level_zero");
        if !ok {
            tracing::warn!(
                backend = backend.unwrap_or("none"),
                "USM GPU dispatch auto-disabled: SYCL backend is not \
                 Level Zero, and the OpenCL fallback computes wrong \
                 results for parts of the kernel suite on validated \
                 hardware. Running CPU kernels instead. Force GPU with \
                 RUSTLLAMA_USM_ATTN=1, or install the Intel Compute \
                 Runtime / Level Zero loader (ze_loader.dll) for the \
                 validated fast path."
            );
        }
        ok
    })
}

/// Cached flash-attention tier opt-outs (`RUSTLLAMA_FA_V3` /
/// `RUSTLLAMA_FA_V2`, default on). These are consulted on every
/// flash-attention dispatch — the previous per-call `env_on`
/// closures took the env lock twice per attention call. A/B levers:
/// set before the first forward.
fn fa_v3_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("RUSTLLAMA_FA_V3")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off")))
            .unwrap_or(true)
    })
}

fn fa_v2_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("RUSTLLAMA_FA_V2")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off")))
            .unwrap_or(true)
    })
}

/// Try to run F32 KV flash-attention decode on the USM-resident
/// kernel. Returns `true` if the GPU produced an output (written
/// to `out`); `false` for any reason — env var off, SYCL
/// unavailable, alloc failure, shape mismatch. The caller falls
/// back to the existing CPU path on `false`.
///
/// Inputs / outputs are f32 (matching the engine's native dtype);
/// the function performs the f32↔f16 conversion at the USM
/// boundary. Same f16-storage-precision tolerance as the existing
/// host-pointer SYCL kernels.
///
/// `k_row` and `v_row` are the NEW K and V rows for the current
/// position (the engine wrote them to its own scratch but the
/// USM cache is the source of truth on the SYCL path). This
/// function writes them to USM at position `pos` of layer
/// `layer_idx`, then runs the kernel against the cached history
/// at `[0, pos + 1)`.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_usm_f32(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    if !usm_attn_enabled() || !gpu_active_for_current_layer() {
        return false;
    }
    let cfg = UsmAttnConfig {
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        max_ctx,
        // `d` / `d_ff` are zero from this attention-only entry —
        // the rmsnorm / silu dispatchers populate them lazily via
        // their own `with_context` setup when first called. The
        // attention path doesn't need either.
        d: 0,
        d_ff: 0,
    };
    let q_len = (n_heads as usize) * (head_dim as usize);
    let kv_row_len = (n_kv_heads as usize) * (head_dim as usize);
    if q.len() != q_len
        || k_row.len() != kv_row_len
        || v_row.len() != kv_row_len
        || out.len() != q_len
        || layer_idx >= n_layers as usize
        || pos >= max_ctx
    {
        return false;
    }
    let kv_len = (pos + 1) as u32;
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        // Rebuild the context on first use or config change.
        let needs_rebuild = match slot.as_ref() {
            Some(ctx) => ctx.cfg != cfg,
            None => true,
        };
        if needs_rebuild {
            *slot = UsmAttnContext::try_new(cfg);
            if slot.is_none() {
                // Couldn't allocate (mock mode / no device / OOM).
                return false;
            }
        }
        let ctx = slot.as_mut().expect("usm ctx just built");

        // KV-mirror validity gate: attention below reads rows
        // 0..=pos from THIS thread's USM cache, but this call only
        // writes the row at `pos`. If the earlier rows were never
        // written here (prefill ran on another thread or through
        // the host-slab batched path, or a prefix snapshot was
        // restored into host KV without a replay), the kernel
        // would attend over zeros / another request's stale rows —
        // the model answers a phantom prompt. Decline instead; the
        // caller's CPU attention over host KV is always correct.
        // See `kv_valid_len` / `USM_KV_EPOCH`.
        let epoch = USM_KV_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if ctx.kv_epoch != epoch {
            for v in ctx.kv_valid_len.iter_mut() {
                *v = 0;
            }
            ctx.kv_epoch = epoch;
        }
        let valid = ctx.kv_valid_len[layer_idx];
        if pos > valid {
            log_usm_kv_gap_once(layer_idx, pos, valid);
            return false;
        }

        // Write new K/V rows into the USM cache at this position.
        // USM-shared pages — host writes are page-mapped so no
        // memcpy beyond the slice-write itself.
        let k_cache = &mut ctx.k_caches[layer_idx];
        let v_cache = &mut ctx.v_caches[layer_idx];
        let pos_us = pos as usize;
        for h in 0..n_kv_heads as usize {
            let dst_off = (h * max_ctx as usize + pos_us) * head_dim as usize;
            let src_off = h * head_dim as usize;
            for d in 0..head_dim as usize {
                k_cache.as_mut_slice()[dst_off + d] =
                    half::f16::from_f32(k_row[src_off + d]).to_bits();
                v_cache.as_mut_slice()[dst_off + d] =
                    half::f16::from_f32(v_row[src_off + d]).to_bits();
            }
        }
        // Contiguous append extends the valid range; an in-place
        // overwrite (pos < valid, e.g. a new request re-prefilling
        // from position 0 after `usm_attn_kv_invalidate`) keeps it.
        if pos == valid {
            ctx.kv_valid_len[layer_idx] = pos + 1;
        }

        // Write Q into USM scratch.
        let q_scratch = ctx.q_scratch.as_mut().expect("q scratch");
        for (i, dst) in q_scratch.as_mut_slice().iter_mut().enumerate() {
            *dst = half::f16::from_f32(q[i]).to_bits();
        }

        // Run the kernel. Borrows: ctx.stream is shared; the four
        // SyclSharedBuffer values all hold &stream. Safe because
        // the kernel `.wait()`s before returning so no overlap
        // between host slice access and device kernel reads.
        let stream = &ctx.stream;
        let q_buf = ctx.q_scratch.as_ref().expect("q scratch");
        let k_buf = &ctx.k_caches[layer_idx];
        let v_buf = &ctx.v_caches[layer_idx];
        let out_buf_ptr = ctx
            .out_scratch
            .as_mut()
            .expect("out scratch") as *mut sk::SyclSharedBuffer<u16>;
        // SAFETY: `out_buf_ptr` is taken via raw pointer to avoid
        // a borrow-checker conflict with the other shared
        // borrows of ctx; it points to a live field of the ctx
        // which outlives this call.
        let out_buf: &mut sk::SyclSharedBuffer<u16> = unsafe { &mut *out_buf_ptr };
        // H11 (deferred): adaptive flash-attn V3 KV tile sizing.
        // Today the V3 kernel reads `RUSTLLAMA_FLASH_V3_KV_TILE`
        // env var once and uses that tile (16/32/64) for the
        // lifetime. Per-call adaptive sizing (small tile for
        // kv_len < 512, large for kv_len > 4096) would require
        // adding an explicit `tile_size: u32` parameter to
        // `rsl_flash_attn_decode_v3_usm` and three kernel variants
        // (already templated, so just dispatch needs extension).
        // Sized as ~4-6 hours of cross-cutting kernel + FFI work.
        // Tracked as Tier H follow-up.
        //
        // FlashAttention dispatch chain: v3 → v2 → v1.
        //   - v3 (SLM K/V tiling + sub-group cooperation): default for
        //     shape-compatible head_dim. Best memory-access pattern.
        //   - v2 (sub-group cooperation, no SLM): fallback when v3 fails
        //     (driver regression, SLM alloc, etc.).
        //   - v1 (per-WI scalar): universal fallback. Always shape-compat.
        //
        // Env vars (default both on):
        //   RUSTLLAMA_FA_V3=0  → skip v3, start at v2
        //   RUSTLLAMA_FA_V2=0  → skip v2 in the chain too (forces v1)
        //
        // On shape mismatch v3/v2 return `InvalidShape` and we
        // transparently fall through to the next tier regardless of
        // the env vars.
        let want_v3 = fa_v3_enabled();
        let want_v2 = fa_v2_enabled();
        let v3_ok = want_v3
            && sk::flash_attn_decode_v3_usm(
                stream, q_buf, k_buf, v_buf, out_buf,
                n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
            )
            .is_ok();
        let v2_ok = !v3_ok && want_v2
            && sk::flash_attn_decode_v2_usm(
                stream, q_buf, k_buf, v_buf, out_buf,
                n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
            )
            .is_ok();
        if !v3_ok && !v2_ok {
            if sk::flash_attn_decode_usm(
                stream, q_buf, k_buf, v_buf, out_buf,
                n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
            )
            .is_err()
            {
                return false;
            }
        }
        // Read the result back into the caller's f32 buffer.
        for (i, dst) in out.iter_mut().enumerate() {
            *dst = half::f16::from_bits(out_buf.as_slice()[i]).to_f32();
        }
        true
    })
}

// ------------------------------------------------------------------
// Quantized-KV flash-attention decode (USM-resident)
// ------------------------------------------------------------------
//
// These mirror `try_flash_attn_decode_usm_f32` EXACTLY — same env +
// placement gate, same per-thread `UsmAttnContext`, same
// epoch/valid-len staleness gate, same "write the new row into the
// USM mirror at `pos`, run the kernel over `[0, pos+1)`, read `out`
// back" shape — with two differences:
//   1. The mirror holds format-specific PACKED bytes (+ per-row f32
//      scales for Q8_0 / TurboQuant) instead of f16, produced by the
//      SAME CPU quantizers the engine's host slab uses. That keeps
//      the mirror byte-identical to the host slab, so the GPU result
//      equals the CPU flash-decode result to the fp-reduction
//      tolerance the F32 path already tolerates.
//   2. Validity is tracked on `kv_valid_len_q` / `kv_epoch_q`, which
//      are disjoint from the f16 path's `kv_valid_len` / `kv_epoch`.
//      The quant path never writes the f16 mirror and vice-versa, so
//      the two can't corrupt each other's staleness gate.
//
// PREFILL (batched) is wired separately below in
// `try_flash_attn_prefill_usm_quant` / `try_flash_attn_prefill_cuda_quant`
// and their `try_flash_attn_prefill_gpu_*` combinators — same packed
// formats, but the whole-slab re-upload residency model rather than this
// decode path's per-row incremental mirror.

/// Which quantized KV format a decode call targets. Carries the
/// TurboQuant bit-width inline (the only format with a runtime
/// parameter). `Copy` so it threads through the helper cheaply.
#[derive(Clone, Copy)]
enum QuantKv {
    Q4_0,
    Q8_0,
    Nvfp4,
    Tq { bits: u8 },
    Mxfp4,
    Mxfp6,
    Mxfp8,
}

impl QuantKv {
    /// Packed bytes per KV row for this format — must match the
    /// engine host slab's `bytes_per_row` exactly.
    fn bytes_per_row(self, head_dim: usize) -> usize {
        use rustllama_kernels_cpu::{mxfp, nvfp4, q4_0_kv, turboquant};
        match self {
            QuantKv::Q4_0 => (head_dim / q4_0_kv::Q4_0_BLOCK_ELEMS) * q4_0_kv::Q4_0_BLOCK_BYTES,
            // Q8_0 KV is stored as a raw i8 slab (1 byte / element),
            // NOT ggml 34B/32 blocks — matching the CPU Q8_0 kernel.
            QuantKv::Q8_0 => head_dim,
            QuantKv::Nvfp4 => (head_dim / nvfp4::NVFP4_BLOCK_ELEMS) * nvfp4::NVFP4_BLOCK_BYTES,
            QuantKv::Tq { bits } => turboquant::bytes_per_block(head_dim, bits),
            QuantKv::Mxfp4 => {
                (head_dim / mxfp::MXFP4_BLOCK_ELEMS) * mxfp::MXFP4_BLOCK_BYTES
            }
            QuantKv::Mxfp6 => {
                (head_dim / mxfp::MXFP6_BLOCK_ELEMS) * mxfp::MXFP6_BLOCK_BYTES
            }
            QuantKv::Mxfp8 => {
                (head_dim / mxfp::MXFP8_BLOCK_ELEMS) * mxfp::MXFP8_BLOCK_BYTES
            }
        }
    }

    /// Whether the format keeps per-row scales in a side buffer.
    fn has_scales(self) -> bool {
        matches!(self, QuantKv::Q8_0 | QuantKv::Tq { .. })
    }
}

/// Ensure `slot` holds a `u8` USM buffer of EXACTLY `len` bytes on
/// `stream`. Reallocates on a size mismatch (packed mirrors are sized
/// to the full `n_kv_heads * max_ctx * bytes_per_row` slab, so exact
/// sizing — not grow-only — keeps the layout the kernel expects).
/// Returns false on alloc failure so the caller can fall back to CPU.
fn ensure_u8_exact(
    stream: &sk::SyclStream,
    slot: &mut Option<sk::SyclSharedBuffer<'static, u8>>,
    len: usize,
) -> bool {
    if slot.as_ref().map_or(false, |b| b.len() == len) {
        return true;
    }
    // Free the mismatched buffer before allocating its replacement so
    // peak USM doesn't double.
    *slot = None;
    match sk::SyclSharedBuffer::<u8>::alloc(stream, len) {
        Ok(buf) => {
            // SAFETY: same '_-to-'static transmute the other scratch
            // uses — the UsmAttnContext Drop frees these before the
            // stream goes.
            *slot = Some(unsafe {
                std::mem::transmute::<
                    sk::SyclSharedBuffer<'_, u8>,
                    sk::SyclSharedBuffer<'static, u8>,
                >(buf)
            });
            true
        }
        Err(_) => false,
    }
}

/// Shared body for every quantized-KV decode helper. See the module
/// comment above for the design. Returns `true` iff the GPU produced
/// the attention output (written to `out`); `false` for any gate /
/// shape / alloc / kernel failure, on which the caller runs its
/// existing CPU attention over the host slab.
///
/// `k_row` / `v_row` are the NEW (post-RoPE, post-whiten, post-bias)
/// f32 K/V rows for `pos`, length `n_kv_heads * head_dim`. They are
/// quantized here into the per-layer USM mirror with the SAME CPU
/// quantizer the engine used for the host slab, so the two stay
/// byte-identical.
#[allow(clippy::too_many_arguments)]
fn try_flash_attn_decode_usm_quant(
    fmt: QuantKv,
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    if !usm_attn_enabled() || !gpu_active_for_current_layer() {
        return false;
    }
    let cfg = UsmAttnConfig {
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        max_ctx,
        // Attention-only entry (same as the F32 helper) — the
        // rmsnorm / silu dispatchers set `d` / `d_ff` lazily.
        d: 0,
        d_ff: 0,
    };
    let hd = head_dim as usize;
    let n_kv = n_kv_heads as usize;
    let mc = max_ctx as usize;
    let posu = pos as usize;
    let q_len = (n_heads as usize) * hd;
    let kv_row_len = n_kv * hd;
    // Shape gate. Same divisibility rules as the F32 kernel; decline
    // (→ CPU) on any mismatch rather than risk a kernel-side assert.
    if q.len() != q_len
        || k_row.len() != kv_row_len
        || v_row.len() != kv_row_len
        || out.len() != q_len
        || layer_idx >= n_layers as usize
        || pos >= max_ctx
        || n_kv_heads == 0
        || n_heads % n_kv_heads != 0
    {
        return false;
    }
    // Per-format block constraint (matches each CPU quantizer's /
    // kernel's own asserts — we decline instead of panicking).
    let ok_shape = match fmt {
        QuantKv::Q4_0 => hd % 32 == 0,
        QuantKv::Q8_0 => true,
        QuantKv::Nvfp4 => hd % 16 == 0,
        QuantKv::Tq { bits } => matches!(bits, 1 | 2 | 4 | 8) && hd.is_power_of_two(),
        QuantKv::Mxfp4 | QuantKv::Mxfp6 | QuantKv::Mxfp8 => hd % 32 == 0,
    };
    if !ok_shape {
        return false;
    }
    let bytes_per_row = fmt.bytes_per_row(hd);
    let mirror_len = n_kv * mc * bytes_per_row;
    let scales_len = n_kv * mc;
    let kv_len = pos + 1;
    let has_scales = fmt.has_scales();

    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        // Rebuild the context on first use or config change — same as
        // the F32 helper.
        let needs_rebuild = match slot.as_ref() {
            Some(ctx) => ctx.cfg != cfg,
            None => true,
        };
        if needs_rebuild {
            *slot = UsmAttnContext::try_new(cfg);
            if slot.is_none() {
                return false;
            }
        }
        let ctx = slot.as_mut().expect("usm ctx just built");

        // Quant-mirror staleness gate (independent of the f16 path).
        let epoch = USM_KV_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if ctx.kv_epoch_q != epoch {
            for v in ctx.kv_valid_len_q.iter_mut() {
                *v = 0;
            }
            ctx.kv_epoch_q = epoch;
        }
        let valid = ctx.kv_valid_len_q[layer_idx];
        if pos > valid {
            log_usm_kv_gap_once(layer_idx, pos, valid);
            return false;
        }

        // A format change for this layer (never in practice) invalidates
        // the mirror + its valid range.
        if ctx.q_bytes_per_row[layer_idx] as usize != bytes_per_row {
            ctx.q_k_packed[layer_idx] = None;
            ctx.q_v_packed[layer_idx] = None;
            ctx.q_k_scales[layer_idx] = None;
            ctx.q_v_scales[layer_idx] = None;
            ctx.kv_valid_len_q[layer_idx] = 0;
            ctx.q_bytes_per_row[layer_idx] = bytes_per_row as u32;
        }

        // Lazily allocate this layer's mirror (+ scales) and the shared
        // f32 q / out scratch. Any failure → CPU.
        if !ensure_u8_exact(&ctx.stream, &mut ctx.q_k_packed[layer_idx], mirror_len)
            || !ensure_u8_exact(&ctx.stream, &mut ctx.q_v_packed[layer_idx], mirror_len)
            || !ensure_scratch_f32(&ctx.stream, &mut ctx.q_scratch_f32, q_len)
            || !ensure_scratch_f32(&ctx.stream, &mut ctx.out_scratch_f32, q_len)
        {
            return false;
        }
        if has_scales
            && (!ensure_scratch_f32(&ctx.stream, &mut ctx.q_k_scales[layer_idx], scales_len)
                || !ensure_scratch_f32(&ctx.stream, &mut ctx.q_v_scales[layer_idx], scales_len))
        {
            return false;
        }

        // Quantize the new K/V rows into the USM mirror at `pos`, using
        // the SAME quantizers the engine's host slab uses (byte-identical
        // mirror ⇒ GPU output == CPU output within fp-reduction order).
        // Disjoint struct-field borrows (packed vs scales are separate
        // Vecs) let K's packed + scale buffers be written together.
        match fmt {
            QuantKv::Q4_0 => {
                use rustllama_kernels_cpu::q4_0_kv;
                {
                    let kb = ctx.q_k_packed[layer_idx].as_mut().expect("k mirror").as_mut_slice();
                    for h in 0..n_kv {
                        let p = (h * mc + posu) * bytes_per_row;
                        q4_0_kv::quantize_row(&k_row[h * hd..(h + 1) * hd], &mut kb[p..p + bytes_per_row]);
                    }
                }
                {
                    let vb = ctx.q_v_packed[layer_idx].as_mut().expect("v mirror").as_mut_slice();
                    for h in 0..n_kv {
                        let p = (h * mc + posu) * bytes_per_row;
                        q4_0_kv::quantize_row(&v_row[h * hd..(h + 1) * hd], &mut vb[p..p + bytes_per_row]);
                    }
                }
            }
            QuantKv::Q8_0 => {
                {
                    let kb = ctx.q_k_packed[layer_idx].as_mut().expect("k mirror").as_mut_slice();
                    // The mirror stores i8; reinterpret the USM bytes as
                    // i8 for the quantizer (identical bit layout — the
                    // kernel reads them back as i8 internally).
                    let kb_i8: &mut [i8] =
                        unsafe { std::slice::from_raw_parts_mut(kb.as_mut_ptr() as *mut i8, kb.len()) };
                    let ks = ctx.q_k_scales[layer_idx].as_mut().expect("k scales").as_mut_slice();
                    for h in 0..n_kv {
                        let row_idx = h * mc + posu;
                        let dst = row_idx * hd;
                        ks[row_idx] = crate::llama_arch::quantize_row_q8_0(
                            &k_row[h * hd..(h + 1) * hd],
                            &mut kb_i8[dst..dst + hd],
                        );
                    }
                }
                {
                    let vb = ctx.q_v_packed[layer_idx].as_mut().expect("v mirror").as_mut_slice();
                    let vb_i8: &mut [i8] =
                        unsafe { std::slice::from_raw_parts_mut(vb.as_mut_ptr() as *mut i8, vb.len()) };
                    let vs = ctx.q_v_scales[layer_idx].as_mut().expect("v scales").as_mut_slice();
                    for h in 0..n_kv {
                        let row_idx = h * mc + posu;
                        let dst = row_idx * hd;
                        vs[row_idx] = crate::llama_arch::quantize_row_q8_0(
                            &v_row[h * hd..(h + 1) * hd],
                            &mut vb_i8[dst..dst + hd],
                        );
                    }
                }
            }
            QuantKv::Nvfp4 => {
                use rustllama_kernels_cpu::nvfp4;
                let blocks_per_row = hd / nvfp4::NVFP4_BLOCK_ELEMS;
                {
                    let kb = ctx.q_k_packed[layer_idx].as_mut().expect("k mirror").as_mut_slice();
                    for h in 0..n_kv {
                        let p = (h * mc + posu) * bytes_per_row;
                        for b in 0..blocks_per_row {
                            let elem_off = h * hd + b * nvfp4::NVFP4_BLOCK_ELEMS;
                            let blk_dst = p + b * nvfp4::NVFP4_BLOCK_BYTES;
                            nvfp4::quantize_block(
                                &k_row[elem_off..elem_off + nvfp4::NVFP4_BLOCK_ELEMS],
                                &mut kb[blk_dst..blk_dst + nvfp4::NVFP4_BLOCK_BYTES],
                            );
                        }
                    }
                }
                {
                    let vb = ctx.q_v_packed[layer_idx].as_mut().expect("v mirror").as_mut_slice();
                    for h in 0..n_kv {
                        let p = (h * mc + posu) * bytes_per_row;
                        for b in 0..blocks_per_row {
                            let elem_off = h * hd + b * nvfp4::NVFP4_BLOCK_ELEMS;
                            let blk_dst = p + b * nvfp4::NVFP4_BLOCK_BYTES;
                            nvfp4::quantize_block(
                                &v_row[elem_off..elem_off + nvfp4::NVFP4_BLOCK_ELEMS],
                                &mut vb[blk_dst..blk_dst + nvfp4::NVFP4_BLOCK_BYTES],
                            );
                        }
                    }
                }
            }
            QuantKv::Tq { bits } => {
                use rustllama_kernels_cpu::turboquant;
                // quantize_row does an in-place WHT, so copy each row
                // into scratch first (matches the engine host path).
                let mut tq_row = vec![0f32; hd];
                {
                    let kb = ctx.q_k_packed[layer_idx].as_mut().expect("k mirror").as_mut_slice();
                    let ks = ctx.q_k_scales[layer_idx].as_mut().expect("k scales").as_mut_slice();
                    for h in 0..n_kv {
                        let row_idx = h * mc + posu;
                        let p = row_idx * bytes_per_row;
                        tq_row.copy_from_slice(&k_row[h * hd..(h + 1) * hd]);
                        ks[row_idx] = turboquant::quantize_row(&mut tq_row, bits, &mut kb[p..p + bytes_per_row]);
                    }
                }
                {
                    let vb = ctx.q_v_packed[layer_idx].as_mut().expect("v mirror").as_mut_slice();
                    let vs = ctx.q_v_scales[layer_idx].as_mut().expect("v scales").as_mut_slice();
                    for h in 0..n_kv {
                        let row_idx = h * mc + posu;
                        let p = row_idx * bytes_per_row;
                        tq_row.copy_from_slice(&v_row[h * hd..(h + 1) * hd]);
                        vs[row_idx] = turboquant::quantize_row(&mut tq_row, bits, &mut vb[p..p + bytes_per_row]);
                    }
                }
            }
            QuantKv::Mxfp4 | QuantKv::Mxfp6 | QuantKv::Mxfp8 => {
                use rustllama_kernels_cpu::{mxfp, mxfp_kv};
                // Per-format block bytes + block quantizer; the (head,
                // block) row loop is shared. Scale is embedded per block,
                // so there is no side scales buffer (like NVFP4).
                let (blk_bytes, qfn): (usize, fn(&[f32], &mut [u8])) = match fmt {
                    QuantKv::Mxfp4 => (mxfp::MXFP4_BLOCK_BYTES, mxfp_kv::quantize_block_mxfp4),
                    QuantKv::Mxfp6 => (mxfp::MXFP6_BLOCK_BYTES, mxfp_kv::quantize_block_mxfp6),
                    _ => (mxfp::MXFP8_BLOCK_BYTES, mxfp_kv::quantize_block_mxfp8),
                };
                let blocks_per_row = hd / 32;
                {
                    let kb = ctx.q_k_packed[layer_idx].as_mut().expect("k mirror").as_mut_slice();
                    for h in 0..n_kv {
                        let p = (h * mc + posu) * bytes_per_row;
                        for b in 0..blocks_per_row {
                            let eo = h * hd + b * 32;
                            let bd = p + b * blk_bytes;
                            qfn(&k_row[eo..eo + 32], &mut kb[bd..bd + blk_bytes]);
                        }
                    }
                }
                {
                    let vb = ctx.q_v_packed[layer_idx].as_mut().expect("v mirror").as_mut_slice();
                    for h in 0..n_kv {
                        let p = (h * mc + posu) * bytes_per_row;
                        for b in 0..blocks_per_row {
                            let eo = h * hd + b * 32;
                            let bd = p + b * blk_bytes;
                            qfn(&v_row[eo..eo + 32], &mut vb[bd..bd + blk_bytes]);
                        }
                    }
                }
            }
        }

        // A contiguous append extends the valid range; an in-place
        // overwrite (pos < valid) leaves it. Same rule as the F32 path.
        if pos == valid {
            ctx.kv_valid_len_q[layer_idx] = pos + 1;
        }

        // Write Q into the f32 q-scratch.
        {
            let qs = ctx.q_scratch_f32.as_mut().expect("q scratch").as_mut_slice();
            qs[..q_len].copy_from_slice(q);
        }

        // Run the kernel via the raw FFI entries — the same raw-pointer
        // dance the rmsnorm helper uses to avoid simultaneous &mut
        // borrows across struct fields. All pointers reference live USM
        // on `stream`; the kernel `.wait()`s before returning, so the
        // read-back below doesn't race the device.
        let stream_raw: *const sk::SyclStream = &ctx.stream;
        let q_ptr = ctx.q_scratch_f32.as_ref().unwrap().as_ptr();
        let k_ptr = ctx.q_k_packed[layer_idx].as_ref().unwrap().as_ptr();
        let v_ptr = ctx.q_v_packed[layer_idx].as_ref().unwrap().as_ptr();
        let out_ptr = ctx.out_scratch_f32.as_mut().unwrap().as_mut_ptr();
        let (ks_ptr, vs_ptr) = if has_scales {
            (
                ctx.q_k_scales[layer_idx].as_ref().unwrap().as_ptr(),
                ctx.q_v_scales[layer_idx].as_ref().unwrap().as_ptr(),
            )
        } else {
            (std::ptr::null::<f32>(), std::ptr::null::<f32>())
        };
        let ok = unsafe {
            match fmt {
                QuantKv::Q4_0 => sk::flash_attn_decode_q4_0_usm_raw(
                    &*stream_raw, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                ),
                QuantKv::Q8_0 => sk::flash_attn_decode_q8_0_usm_raw(
                    &*stream_raw, q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                ),
                QuantKv::Nvfp4 => sk::flash_attn_decode_nvfp4_usm_raw(
                    &*stream_raw, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                ),
                QuantKv::Tq { bits } => sk::flash_attn_decode_tq_usm_raw(
                    &*stream_raw, q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, bits as u32, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                ),
                QuantKv::Mxfp4 => sk::flash_attn_decode_mxfp4_usm_raw(
                    &*stream_raw, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                ),
                QuantKv::Mxfp6 => sk::flash_attn_decode_mxfp6_usm_raw(
                    &*stream_raw, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                ),
                QuantKv::Mxfp8 => sk::flash_attn_decode_mxfp8_usm_raw(
                    &*stream_raw, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                ),
            }
        }
        .is_ok();
        if !ok {
            return false;
        }

        // Read the f32 result back into the caller's buffer.
        let out_buf = ctx.out_scratch_f32.as_ref().expect("out scratch");
        out.copy_from_slice(&out_buf.as_slice()[..q_len]);
        true
    })
}

/// Quantized-KV (Q4_0) flash-attention decode. See
/// [`try_flash_attn_decode_usm_quant`]. `false` ⇒ caller uses CPU.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_usm_q4_0(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    try_flash_attn_decode_usm_quant(
        QuantKv::Q4_0, q, k_row, v_row, out, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Quantized-KV (Q8_0 i8 slab + per-row absmax) flash-attention decode.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_usm_q8_0(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    try_flash_attn_decode_usm_quant(
        QuantKv::Q8_0, q, k_row, v_row, out, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Quantized-KV (NVFP4) flash-attention decode.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_usm_nvfp4(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    try_flash_attn_decode_usm_quant(
        QuantKv::Nvfp4, q, k_row, v_row, out, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Quantized-KV (TurboQuant, `bits` ∈ {1,2,4,8}) flash-attention decode.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_usm_tq(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    bits: u8,
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    try_flash_attn_decode_usm_quant(
        QuantKv::Tq { bits }, q, k_row, v_row, out, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

// ==================================================================
// CUDA-resident flash-attention decode (F32 + 4 quant KV dtypes)
// ==================================================================
//
// The native-CUDA analogue of the SYCL `try_flash_attn_decode_usm_*`
// helpers above. It mirrors their DESIGN — a per-thread context that
// keeps K/V resident on the accelerator, an epoch/valid-len staleness
// gate, "write the new row, run the kernel over `[0, pos+1)`, read the
// output back" — but with the fundamental CUDA difference the matvec
// layer already lives with (see `CudaMatvecCache`): CUDA device memory
// is NOT host-accessible the way SYCL's shared USM is. The kernel reads
// device pointers, so every input has to be memcpy'd host→device and
// the result device→host.
//
// DEVICE-RESIDENT KV MIRROR (incremental, no host shadow):
//   Each layer keeps a PERSISTENT device-resident K/V mirror sized to
//   the full `[n_kv_heads, max_ctx, bytes_per_row]` slab, allocated
//   lazily on the layer's first decode step. Every step writes ONLY the
//   new row for `pos` into that mirror with an OFFSET H2D copy
//   (`CudaDeviceBuffer::copy_from_host_at`) — `bytes_per_row` at
//   `(h*max_ctx + pos)*bytes_per_row` per kv-head — then launches the
//   kernel over `[0, pos+1)` against the resident mirror. There is NO
//   per-step full-slab re-upload and NO host shadow of the KV history:
//     * F32 stages nothing — the mirror holds raw f32 identical to the
//       caller's `k_row`/`v_row`, so those rows upload straight in.
//     * The quant path quantizes the new row into a tiny reusable host
//       staging buffer (one position's worth, all kv-heads) with the
//       SAME CPU quantizers the engine's host slab uses, then offset-
//       copies that packed row (+ its per-row scale, for Q8_0 /
//       TurboQuant) into the packed mirror.
//   Cost is O(1) H2D per layer per token (a handful of per-kv-head row
//   copies) instead of the old O(context) re-upload, and the redundant
//   second host copy of the KV is gone. This is exactly the SYCL USM
//   decode's per-layer residency model; the only difference is CUDA
//   needs the explicit offset memcpy where SYCL writes the shared page
//   directly. The head-major layout (`h*max_ctx+pos`) means the kv-heads
//   of one new row are `max_ctx*bytes_per_row` apart, so each head is a
//   separate offset copy — still O(1) in the context length.
//
// The gate is fully INDEPENDENT of the SYCL context: its own thread-local
// (`CUDA_ATTN`), its own `kv_valid_len_*` / `kv_epoch_*`. The SYCL path is
// byte-for-byte unchanged; a stray call on one backend can never validate
// a row the other backend wrote.

/// Sized parameters for the CUDA attention context. A mismatch forces a
/// rebuild (different model loaded in the same process/thread). Slimmer
/// than [`UsmAttnConfig`] — the CUDA path only does attention decode, so
/// there are no rmsnorm/silu (`d` / `d_ff`) fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CudaAttnConfig {
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
}

/// Per-thread CUDA attention context. Holds a CUDA stream, PER-LAYER
/// persistent device-resident K/V mirrors (each written one row at a time
/// via an offset H2D copy — no host shadow, no full re-upload), a small
/// reusable host staging buffer for the quant path's new row, shared
/// Q / out device scratch, and disjoint F32 / quant staleness gates.
///
/// FIELD ORDER MATTERS: every `CudaDeviceBuffer<'static>` here — including
/// the ones held inside the per-layer `Vec<Option<..>>` mirrors — actually
/// borrows `self.stream` (the `'static` is the same lifetime lie the SYCL
/// context tells). They are declared BEFORE `stream` so Rust's
/// declaration-order field drop frees them while the stream is still
/// alive — exactly how `CudaMatvecCache` orders its buffers. Dropping a
/// `Vec<Option<CudaDeviceBuffer>>` drops each element (freeing it on the
/// still-live stream) before `stream` itself, so no manual `Drop` is
/// needed as a result.
struct CudaAttnContext {
    cfg: CudaAttnConfig,
    // ---- per-layer persistent F32 K/V device mirrors ----
    /// F32 path: per-layer `[n_kv_heads, max_ctx, head_dim]` f32 device
    /// mirror (`n_kv_heads*max_ctx*head_dim*4` bytes). Allocated lazily on
    /// the layer's first decode step; the new row is written at `pos` with
    /// an offset H2D copy and stays resident across steps.
    dev_k_mirror: Vec<Option<ck::CudaDeviceBuffer<'static>>>,
    dev_v_mirror: Vec<Option<ck::CudaDeviceBuffer<'static>>>,
    // ---- per-layer persistent quant K/V device mirrors (+ scales) ----
    /// Quant path: per-layer packed-byte device mirror
    /// (`n_kv_heads*max_ctx*bytes_per_row` bytes), lazily allocated and
    /// re-allocated on a per-layer format change (see `q_bytes_per_row`).
    dev_qk_mirror: Vec<Option<ck::CudaDeviceBuffer<'static>>>,
    dev_qv_mirror: Vec<Option<ck::CudaDeviceBuffer<'static>>>,
    /// Per-layer per-row f32 scale mirror — allocated only for the formats
    /// that carry out-of-band scales (Q8_0, TurboQuant).
    /// `[n_kv_heads, max_ctx]` f32.
    dev_qk_scales: Vec<Option<ck::CudaDeviceBuffer<'static>>>,
    dev_qv_scales: Vec<Option<ck::CudaDeviceBuffer<'static>>>,
    // ---- shared per-step device scratch (reused across layers) ----
    /// Q input / attention output device scratch, `n_heads*head_dim` f32.
    dev_q: Option<ck::CudaDeviceBuffer<'static>>,
    dev_out: Option<ck::CudaDeviceBuffer<'static>>,
    // ---- reusable host staging for ONE new KV row (quant path only) ----
    /// One position's packed bytes across all kv-heads
    /// (`n_kv_heads*bytes_per_row`), reused for K then V each step: the
    /// new row is quantized here, then offset-copied into the device
    /// mirror. The F32 path needs no staging (it uploads `k_row`/`v_row`
    /// straight into the mirror). Sized lazily; never holds KV history.
    stage_row: Vec<u8>,
    /// One position's per-kv-head f32 scales (`n_kv_heads`), reused for K
    /// then V; only the scale-carrying quant formats touch it.
    stage_scales: Vec<f32>,
    // ---- disjoint staleness gates (F32 vs quant; independent of SYCL) --
    kv_valid_len_f32: Vec<u32>,
    kv_epoch_f32: u64,
    /// Per-layer packed `bytes_per_row` last used, so a format change
    /// (never in practice) re-allocates that layer's quant mirror.
    q_bytes_per_row: Vec<u32>,
    kv_valid_len_q: Vec<u32>,
    kv_epoch_q: u64,
    // ---- prefill device scratch (batched flash-prefill) ----
    // Independent of the decode scratch above: prefill's Q/out are the
    // batched `[n_new, n_heads, head_dim]` shape (not decode's single
    // `[n_heads, head_dim]` row), and prefill uploads the WHOLE K/V slab
    // provided by the caller each call (no per-layer host shadow — the
    // engine already holds the packed/f32 slab, so there is nothing to
    // accumulate). `dev_prefill_k` / `dev_prefill_v` hold either the f32
    // slab (F32 path) or the packed byte slab (quant path); the scales
    // pair is used only by Q8_0 / TurboQuant. Reused across layers — one
    // prefill runs at a time. Declared BEFORE `stream` so field-drop order
    // frees them while the stream is still alive.
    dev_prefill_k: Option<ck::CudaDeviceBuffer<'static>>,
    dev_prefill_v: Option<ck::CudaDeviceBuffer<'static>>,
    dev_prefill_k_scales: Option<ck::CudaDeviceBuffer<'static>>,
    dev_prefill_v_scales: Option<ck::CudaDeviceBuffer<'static>>,
    dev_prefill_q: Option<ck::CudaDeviceBuffer<'static>>,
    dev_prefill_out: Option<ck::CudaDeviceBuffer<'static>>,
    // ---- owning stream (declared LAST so the buffers above free first) --
    stream: ck::CudaStream,
}

impl CudaAttnContext {
    /// Build a fresh CUDA attention context for the given model shape.
    /// `None` when no CUDA device / stream creation fails (callers fall
    /// back to SYCL or CPU). Uses device 0 — the same device the CUDA
    /// matvec cache binds to (`CudaMatvecCache::new(0, ...)`).
    fn try_new(cfg: CudaAttnConfig) -> Option<Self> {
        let stream = ck::CudaStream::create(0)?;
        let n = cfg.n_layers as usize;
        Some(Self {
            cfg,
            dev_k_mirror: (0..n).map(|_| None).collect(),
            dev_v_mirror: (0..n).map(|_| None).collect(),
            dev_qk_mirror: (0..n).map(|_| None).collect(),
            dev_qv_mirror: (0..n).map(|_| None).collect(),
            dev_qk_scales: (0..n).map(|_| None).collect(),
            dev_qv_scales: (0..n).map(|_| None).collect(),
            dev_q: None,
            dev_out: None,
            stage_row: Vec::new(),
            stage_scales: Vec::new(),
            kv_valid_len_f32: vec![0; n],
            kv_epoch_f32: 0,
            q_bytes_per_row: vec![0; n],
            kv_valid_len_q: vec![0; n],
            kv_epoch_q: 0,
            dev_prefill_k: None,
            dev_prefill_v: None,
            dev_prefill_k_scales: None,
            dev_prefill_v_scales: None,
            dev_prefill_q: None,
            dev_prefill_out: None,
            stream,
        })
    }
}

thread_local! {
    /// Per-thread CUDA attention context. Built lazily on the first
    /// `try_flash_attn_decode_cuda_*` call once `cuda_active()` is true
    /// and a matching config is presented; rebuilt on a config change.
    /// Independent of [`USM_ATTN`] — the two backends never share state.
    static CUDA_ATTN: std::cell::RefCell<Option<CudaAttnContext>> =
        const { std::cell::RefCell::new(None) };
}

/// Ensure `buf` is a `len`-element zero-initialized `Vec<u8>`, allocating
/// fault-tolerantly (declines instead of aborting the process on OOM). A
/// size-match short-circuits WITHOUT re-zeroing — fine for the one-row
/// host STAGING buffer, whose contents are overwritten in full each decode
/// step. Returns false → caller falls back to CPU.
fn ensure_cuda_shadow_u8(buf: &mut Vec<u8>, len: usize) -> bool {
    if buf.len() == len {
        return true;
    }
    buf.clear();
    if buf.try_reserve_exact(len).is_err() {
        return false;
    }
    buf.resize(len, 0);
    true
}

/// F32 sibling of [`ensure_cuda_shadow_u8`] (used for the one-row scale
/// staging buffer).
fn ensure_cuda_shadow_f32(buf: &mut Vec<f32>, len: usize) -> bool {
    if buf.len() == len {
        return true;
    }
    buf.clear();
    if buf.try_reserve_exact(len).is_err() {
        return false;
    }
    buf.resize(len, 0.0);
    true
}

/// Ensure `slot` holds a device buffer of at least `need_bytes`. Grows
/// (free-before-alloc) on a size increase, reused otherwise. Returns
/// false on alloc failure so the caller can fall back to CPU. `need_bytes`
/// must be `> 0` (the sole caller only passes attention slabs, which are).
fn ensure_cuda_dev(
    slot: &mut Option<ck::CudaDeviceBuffer<'static>>,
    stream: &ck::CudaStream,
    need_bytes: usize,
) -> bool {
    if slot.as_ref().map_or(false, |b| b.len_bytes() >= need_bytes) {
        return true;
    }
    // Free the too-small buffer before allocating its replacement so peak
    // device memory doesn't double.
    *slot = None;
    match ck::CudaDeviceBuffer::alloc(stream, need_bytes) {
        Some(b) => {
            // SAFETY: the `'_`→`'static` widening is the same lifetime lie
            // the SYCL context tells — the buffer borrows `stream` (a field
            // of the owning context), which is declared last and so
            // outlives every buffer via declaration-order field drop.
            *slot = Some(unsafe {
                std::mem::transmute::<
                    ck::CudaDeviceBuffer<'_>,
                    ck::CudaDeviceBuffer<'static>,
                >(b)
            });
            true
        }
        None => false,
    }
}

/// Quantize the NEW K/V row (`src_row`, `[n_kv, head_dim]` f32) into the
/// one-position host staging buffer, using the SAME CPU quantizers the
/// engine's host slab uses — so the staged bytes (and hence the device
/// mirror) are byte-identical to the host slab and the GPU result matches
/// the CPU reference. `stage_row` is `[n_kv, bytes_per_row]`; `stage_scales`
/// is `[n_kv]` (written only by the scale-carrying formats). Both hold ONE
/// position's worth of data (all kv-heads), packed contiguously per head so
/// the subsequent per-head offset upload can slice them cleanly.
fn cuda_quantize_row_into_stage(
    fmt: QuantKv,
    src_row: &[f32],
    stage_row: &mut [u8],
    stage_scales: &mut [f32],
    n_kv: usize,
    hd: usize,
    bytes_per_row: usize,
) {
    match fmt {
        QuantKv::Q4_0 => {
            use rustllama_kernels_cpu::q4_0_kv;
            for h in 0..n_kv {
                let p = h * bytes_per_row;
                q4_0_kv::quantize_row(
                    &src_row[h * hd..(h + 1) * hd],
                    &mut stage_row[p..p + bytes_per_row],
                );
            }
        }
        QuantKv::Q8_0 => {
            // Q8_0 KV stores a raw i8 slab (bytes_per_row == head_dim);
            // reinterpret the staging bytes as i8 for the quantizer (same
            // bit layout the kernel reads back).
            let row_i8: &mut [i8] = unsafe {
                std::slice::from_raw_parts_mut(stage_row.as_mut_ptr() as *mut i8, stage_row.len())
            };
            for h in 0..n_kv {
                let dst = h * hd;
                stage_scales[h] = crate::llama_arch::quantize_row_q8_0(
                    &src_row[h * hd..(h + 1) * hd],
                    &mut row_i8[dst..dst + hd],
                );
            }
        }
        QuantKv::Nvfp4 => {
            use rustllama_kernels_cpu::nvfp4;
            let blocks_per_row = hd / nvfp4::NVFP4_BLOCK_ELEMS;
            for h in 0..n_kv {
                let p = h * bytes_per_row;
                for b in 0..blocks_per_row {
                    let elem_off = h * hd + b * nvfp4::NVFP4_BLOCK_ELEMS;
                    let blk_dst = p + b * nvfp4::NVFP4_BLOCK_BYTES;
                    nvfp4::quantize_block(
                        &src_row[elem_off..elem_off + nvfp4::NVFP4_BLOCK_ELEMS],
                        &mut stage_row[blk_dst..blk_dst + nvfp4::NVFP4_BLOCK_BYTES],
                    );
                }
            }
        }
        QuantKv::Tq { bits } => {
            use rustllama_kernels_cpu::turboquant;
            // `quantize_row` does an in-place WHT, so copy each row into
            // scratch first (matches the engine host path + the SYCL helper).
            let mut tq_row = vec![0f32; hd];
            for h in 0..n_kv {
                let p = h * bytes_per_row;
                tq_row.copy_from_slice(&src_row[h * hd..(h + 1) * hd]);
                stage_scales[h] = turboquant::quantize_row(
                    &mut tq_row,
                    bits,
                    &mut stage_row[p..p + bytes_per_row],
                );
            }
        }
        QuantKv::Mxfp4 | QuantKv::Mxfp6 | QuantKv::Mxfp8 => {
            use rustllama_kernels_cpu::{mxfp, mxfp_kv};
            let (blk_bytes, qfn): (usize, fn(&[f32], &mut [u8])) = match fmt {
                QuantKv::Mxfp4 => (mxfp::MXFP4_BLOCK_BYTES, mxfp_kv::quantize_block_mxfp4),
                QuantKv::Mxfp6 => (mxfp::MXFP6_BLOCK_BYTES, mxfp_kv::quantize_block_mxfp6),
                _ => (mxfp::MXFP8_BLOCK_BYTES, mxfp_kv::quantize_block_mxfp8),
            };
            let blocks_per_row = hd / 32;
            for h in 0..n_kv {
                let p = h * bytes_per_row;
                for b in 0..blocks_per_row {
                    let eo = h * hd + b * 32;
                    let bd = p + b * blk_bytes;
                    qfn(&src_row[eo..eo + 32], &mut stage_row[bd..bd + blk_bytes]);
                }
            }
        }
    }
}

/// Offset-upload ONE staged position's packed K (or V) row (+ optional
/// per-row scales) into its persistent device mirror at `pos`. The mirror
/// is head-major `[n_kv, max_ctx, bytes_per_row]`, so each kv-head's new
/// row lands at `(h*max_ctx + pos)*bytes_per_row` — a separate bounds-
/// checked `copy_from_host_at` per head (heads are `max_ctx*bytes_per_row`
/// apart, so they can't be coalesced). Scales go to `(h*max_ctx+pos)` in
/// the `[n_kv, max_ctx]` f32 scale mirror. Returns false on any H2D error
/// (→ caller falls back to CPU) — and, crucially, WITHOUT the caller having
/// extended `kv_valid_len`, so a half-written row is never marked valid.
#[allow(clippy::too_many_arguments)]
fn cuda_upload_staged_row(
    mirror: &mut ck::CudaDeviceBuffer<'static>,
    scales_mirror: Option<&mut ck::CudaDeviceBuffer<'static>>,
    stage_row: &[u8],
    stage_scales: &[f32],
    n_kv: usize,
    mc: usize,
    pos: usize,
    bytes_per_row: usize,
    has_scales: bool,
) -> bool {
    for h in 0..n_kv {
        let off = (h * mc + pos) * bytes_per_row;
        if mirror
            .copy_from_host_at(off, &stage_row[h * bytes_per_row..(h + 1) * bytes_per_row])
            .is_err()
        {
            return false;
        }
    }
    if has_scales {
        let sm = match scales_mirror {
            Some(s) => s,
            None => return false,
        };
        let sb: &[u8] = unsafe {
            std::slice::from_raw_parts(stage_scales.as_ptr() as *const u8, n_kv * 4)
        };
        for h in 0..n_kv {
            let off = (h * mc + pos) * 4;
            if sm.copy_from_host_at(off, &sb[h * 4..(h + 1) * 4]).is_err() {
                return false;
            }
        }
    }
    true
}

/// Try to run F32 KV flash-attention decode on the native CUDA backend.
/// Same public signature as [`try_flash_attn_decode_usm_f32`], so the two
/// are interchangeable at the call site. Returns `true` iff the GPU
/// produced the attention output (written to `out`); `false` for any gate
/// / shape / alloc / kernel failure, on which the caller runs SYCL or CPU.
///
/// `k_row` / `v_row` are the NEW K/V rows for `pos` (`n_kv_heads*head_dim`
/// f32). Only that row is written — one offset H2D copy per kv-head — into
/// this layer's PERSISTENT device-resident f32 mirror; the kernel then runs
/// over `[0, pos+1)` against the mirror. No host shadow, no full re-upload.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_cuda_f32(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    if !cuda_active() || !gpu_active_for_current_layer() {
        return false;
    }
    let cfg = CudaAttnConfig { n_layers, n_heads, n_kv_heads, head_dim, max_ctx };
    let hd = head_dim as usize;
    let n_kv = n_kv_heads as usize;
    let mc = max_ctx as usize;
    let posu = pos as usize;
    let q_len = (n_heads as usize) * hd;
    let kv_row_len = n_kv * hd;
    if q.len() != q_len
        || k_row.len() != kv_row_len
        || v_row.len() != kv_row_len
        || out.len() != q_len
        || layer_idx >= n_layers as usize
        || pos >= max_ctx
        || n_kv_heads == 0
        || n_heads % n_kv_heads != 0
    {
        return false;
    }
    let mirror_bytes = n_kv * mc * hd * 4; // full f32 device-mirror bytes
    let row_bytes = hd * 4; // one kv-head's f32 row
    let q_bytes = q_len * 4;
    let kv_len = pos + 1;

    CUDA_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let needs_rebuild = match slot.as_ref() {
            Some(ctx) => ctx.cfg != cfg,
            None => true,
        };
        if needs_rebuild {
            *slot = CudaAttnContext::try_new(cfg);
            if slot.is_none() {
                return false;
            }
        }
        let ctx = slot.as_mut().expect("cuda ctx just built");

        // F32 staleness gate (independent of the quant gate + of SYCL).
        let epoch = USM_KV_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if ctx.kv_epoch_f32 != epoch {
            for v in ctx.kv_valid_len_f32.iter_mut() {
                *v = 0;
            }
            ctx.kv_epoch_f32 = epoch;
        }
        let valid = ctx.kv_valid_len_f32[layer_idx];
        if pos > valid {
            log_usm_kv_gap_once(layer_idx, pos, valid);
            return false;
        }

        // Ensure this layer's persistent F32 device mirror (survives
        // across steps; lazily allocated on the layer's first use) plus the
        // shared Q / out scratch. Each `[layer_idx]` mut-borrows a distinct
        // Vec field, disjoint from the `&ctx.stream` borrow.
        if !ensure_cuda_dev(&mut ctx.dev_k_mirror[layer_idx], &ctx.stream, mirror_bytes)
            || !ensure_cuda_dev(&mut ctx.dev_v_mirror[layer_idx], &ctx.stream, mirror_bytes)
            || !ensure_cuda_dev(&mut ctx.dev_q, &ctx.stream, q_bytes)
            || !ensure_cuda_dev(&mut ctx.dev_out, &ctx.stream, q_bytes)
        {
            return false;
        }

        // Write ONLY the new K/V row into the resident mirror at `pos`, one
        // kv-head at a time (head-major layout ⇒ heads are `max_ctx*head_dim`
        // apart, so they can't be coalesced). The mirror stores raw f32
        // identical to `k_row`/`v_row`, so those rows upload straight in —
        // no host staging, no shadow, no full re-upload.
        {
            let k_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(k_row.as_ptr() as *const u8, kv_row_len * 4)
            };
            let kmir = ctx.dev_k_mirror[layer_idx].as_mut().unwrap();
            for h in 0..n_kv {
                let off = (h * mc + posu) * row_bytes;
                if kmir
                    .copy_from_host_at(off, &k_bytes[h * row_bytes..(h + 1) * row_bytes])
                    .is_err()
                {
                    return false;
                }
            }
        }
        {
            let v_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(v_row.as_ptr() as *const u8, kv_row_len * 4)
            };
            let vmir = ctx.dev_v_mirror[layer_idx].as_mut().unwrap();
            for h in 0..n_kv {
                let off = (h * mc + posu) * row_bytes;
                if vmir
                    .copy_from_host_at(off, &v_bytes[h * row_bytes..(h + 1) * row_bytes])
                    .is_err()
                {
                    return false;
                }
            }
        }
        // Extend the valid range ONLY after the row is fully resident, so a
        // half-written row is never marked valid (a contiguous append at
        // `pos == valid`; an in-place overwrite at `pos < valid` leaves it).
        if pos == valid {
            ctx.kv_valid_len_f32[layer_idx] = pos + 1;
        }

        // Upload Q into the shared scratch.
        {
            let q_src: &[u8] =
                unsafe { std::slice::from_raw_parts(q.as_ptr() as *const u8, q_bytes) };
            if ctx.dev_q.as_mut().unwrap().copy_from_host(q_src).is_err() {
                return false;
            }
        }

        // Gather raw device pointers (disjoint fields) and launch over
        // `[0, pos+1)` against the persistent mirror. No full re-upload.
        let stream = &ctx.stream;
        let q_ptr = ctx.dev_q.as_ref().unwrap().as_ptr() as *const f32;
        let k_ptr = ctx.dev_k_mirror[layer_idx].as_ref().unwrap().as_ptr() as *const f32;
        let v_ptr = ctx.dev_v_mirror[layer_idx].as_ref().unwrap().as_ptr() as *const f32;
        let out_ptr = ctx.dev_out.as_mut().unwrap().as_mut_ptr() as *mut f32;
        // SAFETY: all four are live device pointers on `stream`, sized for
        // the shapes below; the wrapper synchronizes before returning so
        // the read-back doesn't race the kernel.
        let launched = unsafe {
            ck::flash_attn_decode_f32(
                stream, q_ptr, k_ptr, v_ptr, out_ptr,
                n_heads as usize, n_kv, hd, mc, kv_len as usize,
            )
        }
        .is_ok();
        if !launched || ck::consume_error_count() != 0 {
            return false;
        }

        // Read the result back into the caller's f32 buffer.
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, q_bytes) };
        ctx.dev_out.as_ref().unwrap().copy_to_host(out_bytes).is_ok()
    })
}

/// Seed the native-CUDA F32 decode KV mirror with the prefill history for a
/// layer — upload the rows `[valid..pos)` that aren't resident yet from the
/// host KV cache. Without this, [`try_flash_attn_decode_cuda_f32`]'s
/// resident-mirror gap check (`pos > valid`) declines for the WHOLE
/// generation after a prefill: decode starts at `pos = prompt_len` against an
/// empty mirror (`valid == 0`), so attention silently falls back to the CPU
/// every step (the matvec stays on the GPU, so it reads like a slow,
/// context-scaling decode). Call once per layer at the prefill→decode boundary;
/// a no-op once the mirror is current (`valid >= pos`). `k`/`v` are the host
/// KV cache buffers, head-major `[n_kv_heads, max_ctx, head_dim]` (same layout
/// as the mirror), so each head's missing rows upload as one contiguous copy.
#[allow(clippy::too_many_arguments)]
pub fn cuda_decode_seed_kv_f32(
    k: &[f32],
    v: &[f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    if !cuda_active() || !gpu_active_for_current_layer() {
        return false;
    }
    let cfg = CudaAttnConfig { n_layers, n_heads, n_kv_heads, head_dim, max_ctx };
    let hd = head_dim as usize;
    let n_kv = n_kv_heads as usize;
    let mc = max_ctx as usize;
    let posu = pos as usize;
    if n_kv == 0
        || layer_idx >= n_layers as usize
        || posu > mc
        || k.len() < n_kv * mc * hd
        || v.len() < n_kv * mc * hd
    {
        return false;
    }
    let mirror_bytes = n_kv * mc * hd * 4;
    let row_bytes = hd * 4;
    CUDA_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let needs_rebuild = match slot.as_ref() {
            Some(ctx) => ctx.cfg != cfg,
            None => true,
        };
        if needs_rebuild {
            *slot = CudaAttnContext::try_new(cfg);
            if slot.is_none() {
                return false;
            }
        }
        let ctx = slot.as_mut().expect("cuda ctx just built");
        // Same F32 staleness gate as the decode path: a new sequence (epoch
        // bump) invalidates every layer's resident length.
        let epoch = USM_KV_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if ctx.kv_epoch_f32 != epoch {
            for vl in ctx.kv_valid_len_f32.iter_mut() {
                *vl = 0;
            }
            ctx.kv_epoch_f32 = epoch;
        }
        let valid = ctx.kv_valid_len_f32[layer_idx] as usize;
        if valid >= posu {
            return true; // already resident up to here — nothing to seed
        }
        if !ensure_cuda_dev(&mut ctx.dev_k_mirror[layer_idx], &ctx.stream, mirror_bytes)
            || !ensure_cuda_dev(&mut ctx.dev_v_mirror[layer_idx], &ctx.stream, mirror_bytes)
        {
            return false;
        }
        // Upload the missing history rows [valid..pos) per kv-head. Head-major
        // layout ⇒ a head's rows are contiguous, so one copy per head.
        let nrows = posu - valid;
        let len_bytes = nrows * row_bytes;
        for h in 0..n_kv {
            let off_floats = (h * mc + valid) * hd;
            let off_bytes = off_floats * 4;
            let k_src: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    k[off_floats..off_floats + nrows * hd].as_ptr() as *const u8,
                    len_bytes,
                )
            };
            if ctx.dev_k_mirror[layer_idx]
                .as_mut()
                .unwrap()
                .copy_from_host_at(off_bytes, k_src)
                .is_err()
            {
                return false;
            }
            let v_src: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    v[off_floats..off_floats + nrows * hd].as_ptr() as *const u8,
                    len_bytes,
                )
            };
            if ctx.dev_v_mirror[layer_idx]
                .as_mut()
                .unwrap()
                .copy_from_host_at(off_bytes, v_src)
                .is_err()
            {
                return false;
            }
        }
        ctx.kv_valid_len_f32[layer_idx] = pos;
        true
    })
}

/// Seed the native-CUDA QUANTIZED-KV decode mirror with the prefill history
/// for a layer — upload the already-quantized rows `[valid..pos)` straight from
/// the host KV cache. The quant decode path re-quantizes each new row into its
/// device mirror with the SAME CPU quantizer the host slab uses, so the host
/// packed bytes (+ per-row scales, for formats that have them) are
/// BYTE-IDENTICAL to the mirror — a direct copy, no re-quantize. Without this,
/// the quant decode path's gap check (`pos > valid`) declines for the whole
/// generation after a prefill (the same trap the F32 path had), so quant-KV
/// decode attention runs on the CPU. Call once per layer at the prefill→decode
/// boundary; a no-op once resident. `k_packed`/`v_packed` are the host packed
/// slabs (`[n_kv_heads, max_ctx, fmt.bytes_per_row(head_dim)]`);
/// `k_scales`/`v_scales` are the host per-row scale slabs
/// (`[n_kv_heads, max_ctx]` f32) and MUST be `Some` iff `fmt.has_scales()`.
#[allow(clippy::too_many_arguments)]
fn cuda_decode_seed_kv_quant(
    fmt: QuantKv,
    k_packed: &[u8],
    v_packed: &[u8],
    k_scales: Option<&[f32]>,
    v_scales: Option<&[f32]>,
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    if !cuda_active() || !gpu_active_for_current_layer() {
        return false;
    }
    let cfg = CudaAttnConfig { n_layers, n_heads, n_kv_heads, head_dim, max_ctx };
    let hd = head_dim as usize;
    let n_kv = n_kv_heads as usize;
    let mc = max_ctx as usize;
    let posu = pos as usize;
    if n_kv == 0 || layer_idx >= n_layers as usize || posu > mc {
        return false;
    }
    let bytes_per_row = fmt.bytes_per_row(hd);
    let mirror_len = n_kv * mc * bytes_per_row;
    let scales_len = n_kv * mc;
    let has_scales = fmt.has_scales();
    if k_packed.len() < mirror_len || v_packed.len() < mirror_len {
        return false;
    }
    if has_scales {
        match (k_scales, v_scales) {
            (Some(ks), Some(vs)) if ks.len() >= scales_len && vs.len() >= scales_len => {}
            _ => return false,
        }
    }
    CUDA_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let needs_rebuild = match slot.as_ref() {
            Some(ctx) => ctx.cfg != cfg,
            None => true,
        };
        if needs_rebuild {
            *slot = CudaAttnContext::try_new(cfg);
            if slot.is_none() {
                return false;
            }
        }
        let ctx = slot.as_mut().expect("cuda ctx just built");
        let epoch = USM_KV_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if ctx.kv_epoch_q != epoch {
            for vl in ctx.kv_valid_len_q.iter_mut() {
                *vl = 0;
            }
            ctx.kv_epoch_q = epoch;
        }
        // A format change for this layer re-allocates the packed mirror (+
        // scales) and resets its valid range, mirroring the decode path's gate
        // — so the decode call that follows won't realloc and wipe the seeded
        // rows.
        if ctx.q_bytes_per_row[layer_idx] as usize != bytes_per_row {
            ctx.dev_qk_mirror[layer_idx] = None;
            ctx.dev_qv_mirror[layer_idx] = None;
            ctx.dev_qk_scales[layer_idx] = None;
            ctx.dev_qv_scales[layer_idx] = None;
            ctx.kv_valid_len_q[layer_idx] = 0;
            ctx.q_bytes_per_row[layer_idx] = bytes_per_row as u32;
        }
        let valid = ctx.kv_valid_len_q[layer_idx] as usize;
        if valid >= posu {
            return true; // already resident up to here
        }
        if !ensure_cuda_dev(&mut ctx.dev_qk_mirror[layer_idx], &ctx.stream, mirror_len)
            || !ensure_cuda_dev(&mut ctx.dev_qv_mirror[layer_idx], &ctx.stream, mirror_len)
        {
            return false;
        }
        if has_scales
            && (!ensure_cuda_dev(&mut ctx.dev_qk_scales[layer_idx], &ctx.stream, scales_len * 4)
                || !ensure_cuda_dev(&mut ctx.dev_qv_scales[layer_idx], &ctx.stream, scales_len * 4))
        {
            return false;
        }
        // Upload the missing history rows [valid..pos) per kv-head: a contiguous
        // run of packed rows, plus the matching per-row scales when present.
        let nrows = posu - valid;
        for h in 0..n_kv {
            let off_b = (h * mc + valid) * bytes_per_row;
            let len_b = nrows * bytes_per_row;
            if ctx.dev_qk_mirror[layer_idx]
                .as_mut()
                .unwrap()
                .copy_from_host_at(off_b, &k_packed[off_b..off_b + len_b])
                .is_err()
                || ctx.dev_qv_mirror[layer_idx]
                    .as_mut()
                    .unwrap()
                    .copy_from_host_at(off_b, &v_packed[off_b..off_b + len_b])
                    .is_err()
            {
                return false;
            }
            if has_scales {
                let off_s = h * mc + valid; // f32 element index
                let ks = &k_scales.unwrap()[off_s..off_s + nrows];
                let vs = &v_scales.unwrap()[off_s..off_s + nrows];
                let ks_b: &[u8] =
                    unsafe { std::slice::from_raw_parts(ks.as_ptr() as *const u8, nrows * 4) };
                let vs_b: &[u8] =
                    unsafe { std::slice::from_raw_parts(vs.as_ptr() as *const u8, nrows * 4) };
                if ctx.dev_qk_scales[layer_idx]
                    .as_mut()
                    .unwrap()
                    .copy_from_host_at(off_s * 4, ks_b)
                    .is_err()
                    || ctx.dev_qv_scales[layer_idx]
                        .as_mut()
                        .unwrap()
                        .copy_from_host_at(off_s * 4, vs_b)
                        .is_err()
                {
                    return false;
                }
            }
        }
        ctx.kv_valid_len_q[layer_idx] = pos;
        true
    })
}

/// Seed the CUDA Q8_0 (per-row int8 + per-row f32 scale) decode mirror. See
/// [`cuda_decode_seed_kv_quant`]. `k_q`/`v_q` are the host int8 slabs,
/// `k_scales`/`v_scales` the host per-row f32 scales.
#[allow(clippy::too_many_arguments)]
pub fn cuda_decode_seed_kv_q8_0(
    k_q: &[u8],
    v_q: &[u8],
    k_scales: &[f32],
    v_scales: &[f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    cuda_decode_seed_kv_quant(
        QuantKv::Q8_0, k_q, v_q, Some(k_scales), Some(v_scales), layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Seed the CUDA Q4_0 (ggml 18-byte blocks, embedded scale) decode mirror. See
/// [`cuda_decode_seed_kv_quant`]. No separate scales.
#[allow(clippy::too_many_arguments)]
pub fn cuda_decode_seed_kv_q4_0(
    k_q: &[u8],
    v_q: &[u8],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    cuda_decode_seed_kv_quant(
        QuantKv::Q4_0, k_q, v_q, None, None, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Seed the CUDA NVFP4 decode mirror (embedded block scales). See
/// [`cuda_decode_seed_kv_quant`].
#[allow(clippy::too_many_arguments)]
pub fn cuda_decode_seed_kv_nvfp4(
    k_packed: &[u8], v_packed: &[u8], layer_idx: usize, pos: u32,
    n_heads: u32, n_kv_heads: u32, head_dim: u32, max_ctx: u32, n_layers: u32,
) -> bool {
    cuda_decode_seed_kv_quant(
        QuantKv::Nvfp4, k_packed, v_packed, None, None, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Seed the CUDA MXFP4 decode mirror (embedded E8M0 block scales). See
/// [`cuda_decode_seed_kv_quant`].
#[allow(clippy::too_many_arguments)]
pub fn cuda_decode_seed_kv_mxfp4(
    k_packed: &[u8], v_packed: &[u8], layer_idx: usize, pos: u32,
    n_heads: u32, n_kv_heads: u32, head_dim: u32, max_ctx: u32, n_layers: u32,
) -> bool {
    cuda_decode_seed_kv_quant(
        QuantKv::Mxfp4, k_packed, v_packed, None, None, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Seed the CUDA MXFP6 decode mirror (embedded E8M0 block scales). See
/// [`cuda_decode_seed_kv_quant`].
#[allow(clippy::too_many_arguments)]
pub fn cuda_decode_seed_kv_mxfp6(
    k_packed: &[u8], v_packed: &[u8], layer_idx: usize, pos: u32,
    n_heads: u32, n_kv_heads: u32, head_dim: u32, max_ctx: u32, n_layers: u32,
) -> bool {
    cuda_decode_seed_kv_quant(
        QuantKv::Mxfp6, k_packed, v_packed, None, None, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Seed the CUDA MXFP8 decode mirror (embedded E8M0 block scales). See
/// [`cuda_decode_seed_kv_quant`].
#[allow(clippy::too_many_arguments)]
pub fn cuda_decode_seed_kv_mxfp8(
    k_packed: &[u8], v_packed: &[u8], layer_idx: usize, pos: u32,
    n_heads: u32, n_kv_heads: u32, head_dim: u32, max_ctx: u32, n_layers: u32,
) -> bool {
    cuda_decode_seed_kv_quant(
        QuantKv::Mxfp8, k_packed, v_packed, None, None, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Seed the CUDA TurboQuant decode mirror (low-bit packed codes + per-row
/// scales). See [`cuda_decode_seed_kv_quant`].
#[allow(clippy::too_many_arguments)]
pub fn cuda_decode_seed_kv_tq(
    bits: u8,
    k_packed: &[u8], v_packed: &[u8],
    k_scales: &[f32], v_scales: &[f32],
    layer_idx: usize, pos: u32,
    n_heads: u32, n_kv_heads: u32, head_dim: u32, max_ctx: u32, n_layers: u32,
) -> bool {
    cuda_decode_seed_kv_quant(
        QuantKv::Tq { bits }, k_packed, v_packed, Some(k_scales), Some(v_scales),
        layer_idx, pos, n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

// ===========================================================================
// SYCL (USM) + MLX decode-KV seeds — parity twins of the CUDA seeds above.
//
// The USM and MLX quant/F32 decode paths carry the SAME `pos > valid` gap the
// CUDA path had before its seed: after a batched-host prefill, the per-step
// decode call writes only the row at `pos`, so the thread-local device mirror
// holds no history and the dispatcher declines to CPU for the whole
// generation. These seeds fill the missing rows `[valid..pos)` from the host
// KV cache once, at the prefill->decode boundary, so the GPU flash-decode
// engages. The `decode_seed_kv_*` combinators pick the active backend
// (CUDA -> MLX -> SYCL), exactly like `try_flash_attn_decode_gpu_*`. CPU needs
// no seed — its attention reads the host KV slab directly.
// ===========================================================================

/// SYCL (USM) twin of [`cuda_decode_seed_kv_f32`]: fill the f16 USM mirror
/// rows `[valid..pos)` from the host f32 KV slab. The USM F32 mirror stores
/// f16 (`UsmAttnContext::k_caches`), so this CONVERTS per element rather than
/// byte-copying (the one structural difference from the CUDA f32 seed).
#[allow(clippy::too_many_arguments)]
pub fn usm_decode_seed_kv_f32(
    k: &[f32],
    v: &[f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    if !usm_attn_enabled() || !gpu_active_for_current_layer() {
        return false;
    }
    let cfg = UsmAttnConfig {
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        max_ctx,
        d: 0,
        d_ff: 0,
    };
    let hd = head_dim as usize;
    let n_kv = n_kv_heads as usize;
    let mc = max_ctx as usize;
    let posu = pos as usize;
    if n_kv == 0
        || layer_idx >= n_layers as usize
        || posu > mc
        || k.len() < n_kv * mc * hd
        || v.len() < n_kv * mc * hd
    {
        return false;
    }
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let needs_rebuild = match slot.as_ref() {
            Some(ctx) => ctx.cfg != cfg,
            None => true,
        };
        if needs_rebuild {
            *slot = UsmAttnContext::try_new(cfg);
            if slot.is_none() {
                return false;
            }
        }
        let ctx = slot.as_mut().expect("usm ctx just built");
        // Same f16-path staleness gate as the decode entry.
        let epoch = USM_KV_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if ctx.kv_epoch != epoch {
            for vl in ctx.kv_valid_len.iter_mut() {
                *vl = 0;
            }
            ctx.kv_epoch = epoch;
        }
        let valid = ctx.kv_valid_len[layer_idx] as usize;
        if valid >= posu {
            return true; // already resident up to here — nothing to seed
        }
        // Mirror layout == host layout ((h*mc+row)*hd + d), so src and dst
        // indices coincide; convert f32 -> f16-bits the same way the per-pos
        // decode fill does.
        {
            let kc = ctx.k_caches[layer_idx].as_mut_slice();
            for h in 0..n_kv {
                for row in valid..posu {
                    let base = (h * mc + row) * hd;
                    for d in 0..hd {
                        kc[base + d] = half::f16::from_f32(k[base + d]).to_bits();
                    }
                }
            }
        }
        {
            let vc = ctx.v_caches[layer_idx].as_mut_slice();
            for h in 0..n_kv {
                for row in valid..posu {
                    let base = (h * mc + row) * hd;
                    for d in 0..hd {
                        vc[base + d] = half::f16::from_f32(v[base + d]).to_bits();
                    }
                }
            }
        }
        ctx.kv_valid_len[layer_idx] = pos;
        true
    })
}

/// SYCL (USM) twin of [`cuda_decode_seed_kv_quant`]: byte-copy the host packed
/// KV rows `[valid..pos)` (+ per-row scales) into the USM mirror. The USM
/// mirror's packed layout is byte-identical to the host slab (same quantizer,
/// same `(h*mc+pos)*bytes_per_row` layout), so this is a direct slice copy.
#[allow(clippy::too_many_arguments)]
fn usm_decode_seed_kv_quant(
    fmt: QuantKv,
    k_packed: &[u8],
    v_packed: &[u8],
    k_scales: Option<&[f32]>,
    v_scales: Option<&[f32]>,
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    if !usm_attn_enabled() || !gpu_active_for_current_layer() {
        return false;
    }
    let cfg = UsmAttnConfig {
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        max_ctx,
        d: 0,
        d_ff: 0,
    };
    let hd = head_dim as usize;
    let n_kv = n_kv_heads as usize;
    let mc = max_ctx as usize;
    let posu = pos as usize;
    if n_kv == 0 || layer_idx >= n_layers as usize || posu > mc {
        return false;
    }
    let bytes_per_row = fmt.bytes_per_row(hd);
    let mirror_len = n_kv * mc * bytes_per_row;
    let scales_len = n_kv * mc;
    let has_scales = fmt.has_scales();
    if k_packed.len() < mirror_len || v_packed.len() < mirror_len {
        return false;
    }
    if has_scales {
        match (k_scales, v_scales) {
            (Some(ks), Some(vs)) if ks.len() >= scales_len && vs.len() >= scales_len => {}
            _ => return false,
        }
    }
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let needs_rebuild = match slot.as_ref() {
            Some(ctx) => ctx.cfg != cfg,
            None => true,
        };
        if needs_rebuild {
            *slot = UsmAttnContext::try_new(cfg);
            if slot.is_none() {
                return false;
            }
        }
        let ctx = slot.as_mut().expect("usm ctx just built");
        let epoch = USM_KV_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if ctx.kv_epoch_q != epoch {
            for vl in ctx.kv_valid_len_q.iter_mut() {
                *vl = 0;
            }
            ctx.kv_epoch_q = epoch;
        }
        // A format change for this layer re-allocates the packed mirror (+
        // scales) and resets its valid range — mirrors the decode gate so the
        // decode call that follows won't realloc and wipe the seeded rows.
        if ctx.q_bytes_per_row[layer_idx] as usize != bytes_per_row {
            ctx.q_k_packed[layer_idx] = None;
            ctx.q_v_packed[layer_idx] = None;
            ctx.q_k_scales[layer_idx] = None;
            ctx.q_v_scales[layer_idx] = None;
            ctx.kv_valid_len_q[layer_idx] = 0;
            ctx.q_bytes_per_row[layer_idx] = bytes_per_row as u32;
        }
        let valid = ctx.kv_valid_len_q[layer_idx] as usize;
        if valid >= posu {
            return true; // already resident up to here
        }
        if !ensure_u8_exact(&ctx.stream, &mut ctx.q_k_packed[layer_idx], mirror_len)
            || !ensure_u8_exact(&ctx.stream, &mut ctx.q_v_packed[layer_idx], mirror_len)
        {
            return false;
        }
        if has_scales
            && (!ensure_scratch_f32(&ctx.stream, &mut ctx.q_k_scales[layer_idx], scales_len)
                || !ensure_scratch_f32(&ctx.stream, &mut ctx.q_v_scales[layer_idx], scales_len))
        {
            return false;
        }
        let nrows = posu - valid;
        // Head-major layout ⇒ a head's missing rows are contiguous: one slice
        // copy per kv-head, packed bytes then per-row scales.
        {
            let kb = ctx.q_k_packed[layer_idx].as_mut().expect("k mirror").as_mut_slice();
            for h in 0..n_kv {
                let off = (h * mc + valid) * bytes_per_row;
                let len = nrows * bytes_per_row;
                kb[off..off + len].copy_from_slice(&k_packed[off..off + len]);
            }
        }
        {
            let vb = ctx.q_v_packed[layer_idx].as_mut().expect("v mirror").as_mut_slice();
            for h in 0..n_kv {
                let off = (h * mc + valid) * bytes_per_row;
                let len = nrows * bytes_per_row;
                vb[off..off + len].copy_from_slice(&v_packed[off..off + len]);
            }
        }
        if has_scales {
            let ks = k_scales.unwrap();
            let vs = v_scales.unwrap();
            {
                let kscale = ctx.q_k_scales[layer_idx].as_mut().expect("k scales").as_mut_slice();
                for h in 0..n_kv {
                    let off = h * mc + valid;
                    kscale[off..off + nrows].copy_from_slice(&ks[off..off + nrows]);
                }
            }
            {
                let vscale = ctx.q_v_scales[layer_idx].as_mut().expect("v scales").as_mut_slice();
                for h in 0..n_kv {
                    let off = h * mc + valid;
                    vscale[off..off + nrows].copy_from_slice(&vs[off..off + nrows]);
                }
            }
        }
        ctx.kv_valid_len_q[layer_idx] = pos;
        true
    })
}

/// MLX twin of [`cuda_decode_seed_kv_f32`] (f32 device mirror, raw byte copy).
/// Write-blind like the rest of the MLX surface — validated on Apple HW.
#[allow(clippy::too_many_arguments)]
pub fn mlx_decode_seed_kv_f32(
    k: &[f32],
    v: &[f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    if !mlx_active() || !gpu_active_for_current_layer() {
        return false;
    }
    let cfg = MlxAttnConfig { n_layers, n_heads, n_kv_heads, head_dim, max_ctx };
    let hd = head_dim as usize;
    let n_kv = n_kv_heads as usize;
    let mc = max_ctx as usize;
    let posu = pos as usize;
    if n_kv == 0
        || layer_idx >= n_layers as usize
        || posu > mc
        || k.len() < n_kv * mc * hd
        || v.len() < n_kv * mc * hd
    {
        return false;
    }
    let mirror_bytes = n_kv * mc * hd * 4;
    let row_bytes = hd * 4;
    MLX_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let needs_rebuild = match slot.as_ref() {
            Some(ctx) => ctx.cfg != cfg,
            None => true,
        };
        if needs_rebuild {
            *slot = MlxAttnContext::try_new(cfg);
            if slot.is_none() {
                return false;
            }
        }
        let ctx = slot.as_mut().expect("mlx ctx just built");
        let epoch = USM_KV_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if ctx.kv_epoch_f32 != epoch {
            for vl in ctx.kv_valid_len_f32.iter_mut() {
                *vl = 0;
            }
            ctx.kv_epoch_f32 = epoch;
        }
        let valid = ctx.kv_valid_len_f32[layer_idx] as usize;
        if valid >= posu {
            return true;
        }
        if !ensure_mlx_dev(&mut ctx.dev_k_mirror[layer_idx], &ctx.stream, mirror_bytes)
            || !ensure_mlx_dev(&mut ctx.dev_v_mirror[layer_idx], &ctx.stream, mirror_bytes)
        {
            return false;
        }
        let nrows = posu - valid;
        let len_bytes = nrows * row_bytes;
        for h in 0..n_kv {
            let off_floats = (h * mc + valid) * hd;
            let off_bytes = off_floats * 4;
            let k_src: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    k[off_floats..off_floats + nrows * hd].as_ptr() as *const u8,
                    len_bytes,
                )
            };
            if ctx.dev_k_mirror[layer_idx]
                .as_mut()
                .unwrap()
                .copy_from_host_at(off_bytes, k_src)
                .is_err()
            {
                return false;
            }
            let v_src: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    v[off_floats..off_floats + nrows * hd].as_ptr() as *const u8,
                    len_bytes,
                )
            };
            if ctx.dev_v_mirror[layer_idx]
                .as_mut()
                .unwrap()
                .copy_from_host_at(off_bytes, v_src)
                .is_err()
            {
                return false;
            }
        }
        ctx.kv_valid_len_f32[layer_idx] = pos;
        true
    })
}

/// MLX twin of [`cuda_decode_seed_kv_quant`] (byte-copy host packed rows +
/// scales into the device mirror). Write-blind; validated on Apple HW.
#[allow(clippy::too_many_arguments)]
fn mlx_decode_seed_kv_quant(
    fmt: QuantKv,
    k_packed: &[u8],
    v_packed: &[u8],
    k_scales: Option<&[f32]>,
    v_scales: Option<&[f32]>,
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    if !mlx_active() || !gpu_active_for_current_layer() {
        return false;
    }
    let cfg = MlxAttnConfig { n_layers, n_heads, n_kv_heads, head_dim, max_ctx };
    let hd = head_dim as usize;
    let n_kv = n_kv_heads as usize;
    let mc = max_ctx as usize;
    let posu = pos as usize;
    if n_kv == 0 || layer_idx >= n_layers as usize || posu > mc {
        return false;
    }
    let bytes_per_row = fmt.bytes_per_row(hd);
    let mirror_len = n_kv * mc * bytes_per_row;
    let scales_len = n_kv * mc;
    let has_scales = fmt.has_scales();
    if k_packed.len() < mirror_len || v_packed.len() < mirror_len {
        return false;
    }
    if has_scales {
        match (k_scales, v_scales) {
            (Some(ks), Some(vs)) if ks.len() >= scales_len && vs.len() >= scales_len => {}
            _ => return false,
        }
    }
    MLX_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let needs_rebuild = match slot.as_ref() {
            Some(ctx) => ctx.cfg != cfg,
            None => true,
        };
        if needs_rebuild {
            *slot = MlxAttnContext::try_new(cfg);
            if slot.is_none() {
                return false;
            }
        }
        let ctx = slot.as_mut().expect("mlx ctx just built");
        let epoch = USM_KV_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if ctx.kv_epoch_q != epoch {
            for vl in ctx.kv_valid_len_q.iter_mut() {
                *vl = 0;
            }
            ctx.kv_epoch_q = epoch;
        }
        if ctx.q_bytes_per_row[layer_idx] as usize != bytes_per_row {
            ctx.dev_qk_mirror[layer_idx] = None;
            ctx.dev_qv_mirror[layer_idx] = None;
            ctx.dev_qk_scales[layer_idx] = None;
            ctx.dev_qv_scales[layer_idx] = None;
            ctx.kv_valid_len_q[layer_idx] = 0;
            ctx.q_bytes_per_row[layer_idx] = bytes_per_row as u32;
        }
        let valid = ctx.kv_valid_len_q[layer_idx] as usize;
        if valid >= posu {
            return true;
        }
        if !ensure_mlx_dev(&mut ctx.dev_qk_mirror[layer_idx], &ctx.stream, mirror_len)
            || !ensure_mlx_dev(&mut ctx.dev_qv_mirror[layer_idx], &ctx.stream, mirror_len)
        {
            return false;
        }
        if has_scales
            && (!ensure_mlx_dev(&mut ctx.dev_qk_scales[layer_idx], &ctx.stream, scales_len * 4)
                || !ensure_mlx_dev(&mut ctx.dev_qv_scales[layer_idx], &ctx.stream, scales_len * 4))
        {
            return false;
        }
        let nrows = posu - valid;
        for h in 0..n_kv {
            let off_b = (h * mc + valid) * bytes_per_row;
            let len_b = nrows * bytes_per_row;
            if ctx.dev_qk_mirror[layer_idx]
                .as_mut()
                .unwrap()
                .copy_from_host_at(off_b, &k_packed[off_b..off_b + len_b])
                .is_err()
                || ctx.dev_qv_mirror[layer_idx]
                    .as_mut()
                    .unwrap()
                    .copy_from_host_at(off_b, &v_packed[off_b..off_b + len_b])
                    .is_err()
            {
                return false;
            }
            if has_scales {
                let off_s = h * mc + valid;
                let ks = &k_scales.unwrap()[off_s..off_s + nrows];
                let vs = &v_scales.unwrap()[off_s..off_s + nrows];
                let ks_b: &[u8] =
                    unsafe { std::slice::from_raw_parts(ks.as_ptr() as *const u8, nrows * 4) };
                let vs_b: &[u8] =
                    unsafe { std::slice::from_raw_parts(vs.as_ptr() as *const u8, nrows * 4) };
                if ctx.dev_qk_scales[layer_idx]
                    .as_mut()
                    .unwrap()
                    .copy_from_host_at(off_s * 4, ks_b)
                    .is_err()
                    || ctx.dev_qv_scales[layer_idx]
                        .as_mut()
                        .unwrap()
                        .copy_from_host_at(off_s * 4, vs_b)
                        .is_err()
                {
                    return false;
                }
            }
        }
        ctx.kv_valid_len_q[layer_idx] = pos;
        true
    })
}

// ---- Backend-selecting decode-KV seed combinators -------------------------
//
// CUDA -> MLX -> SYCL. Each backend seed self-gates (returns false off its
// backend / on the gpu-inactive layer), so the `||` chain runs exactly one
// real seed and short-circuits. A seed that fails (OOM / H2D error) returns
// false too — the decode then simply falls to CPU, as it did before any seed.

/// F32 decode-KV seed across backends. Call once per layer at the
/// prefill->decode boundary; a no-op once the mirror is resident.
#[allow(clippy::too_many_arguments)]
pub fn decode_seed_kv_f32(
    k: &[f32],
    v: &[f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    cuda_decode_seed_kv_f32(k, v, layer_idx, pos, n_heads, n_kv_heads, head_dim, max_ctx, n_layers)
        || mlx_decode_seed_kv_f32(k, v, layer_idx, pos, n_heads, n_kv_heads, head_dim, max_ctx, n_layers)
        || usm_decode_seed_kv_f32(k, v, layer_idx, pos, n_heads, n_kv_heads, head_dim, max_ctx, n_layers)
}

/// Quantized-KV decode seed across backends (fmt-taking shared body).
#[allow(clippy::too_many_arguments)]
fn decode_seed_kv_quant(
    fmt: QuantKv,
    k_packed: &[u8],
    v_packed: &[u8],
    k_scales: Option<&[f32]>,
    v_scales: Option<&[f32]>,
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    cuda_decode_seed_kv_quant(fmt, k_packed, v_packed, k_scales, v_scales, layer_idx, pos, n_heads, n_kv_heads, head_dim, max_ctx, n_layers)
        || mlx_decode_seed_kv_quant(fmt, k_packed, v_packed, k_scales, v_scales, layer_idx, pos, n_heads, n_kv_heads, head_dim, max_ctx, n_layers)
        || usm_decode_seed_kv_quant(fmt, k_packed, v_packed, k_scales, v_scales, layer_idx, pos, n_heads, n_kv_heads, head_dim, max_ctx, n_layers)
}

/// Q8_0 decode seed across backends. See [`decode_seed_kv_quant`].
#[allow(clippy::too_many_arguments)]
pub fn decode_seed_kv_q8_0(
    k_q: &[u8], v_q: &[u8], k_scales: &[f32], v_scales: &[f32], layer_idx: usize, pos: u32,
    n_heads: u32, n_kv_heads: u32, head_dim: u32, max_ctx: u32, n_layers: u32,
) -> bool {
    decode_seed_kv_quant(
        QuantKv::Q8_0, k_q, v_q, Some(k_scales), Some(v_scales), layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Q4_0 decode seed across backends. See [`decode_seed_kv_quant`].
#[allow(clippy::too_many_arguments)]
pub fn decode_seed_kv_q4_0(
    k_q: &[u8], v_q: &[u8], layer_idx: usize, pos: u32,
    n_heads: u32, n_kv_heads: u32, head_dim: u32, max_ctx: u32, n_layers: u32,
) -> bool {
    decode_seed_kv_quant(
        QuantKv::Q4_0, k_q, v_q, None, None, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// NVFP4 decode seed across backends. See [`decode_seed_kv_quant`].
#[allow(clippy::too_many_arguments)]
pub fn decode_seed_kv_nvfp4(
    k_packed: &[u8], v_packed: &[u8], layer_idx: usize, pos: u32,
    n_heads: u32, n_kv_heads: u32, head_dim: u32, max_ctx: u32, n_layers: u32,
) -> bool {
    decode_seed_kv_quant(
        QuantKv::Nvfp4, k_packed, v_packed, None, None, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// MXFP4 decode seed across backends. See [`decode_seed_kv_quant`].
#[allow(clippy::too_many_arguments)]
pub fn decode_seed_kv_mxfp4(
    k_packed: &[u8], v_packed: &[u8], layer_idx: usize, pos: u32,
    n_heads: u32, n_kv_heads: u32, head_dim: u32, max_ctx: u32, n_layers: u32,
) -> bool {
    decode_seed_kv_quant(
        QuantKv::Mxfp4, k_packed, v_packed, None, None, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// MXFP6 decode seed across backends. See [`decode_seed_kv_quant`].
#[allow(clippy::too_many_arguments)]
pub fn decode_seed_kv_mxfp6(
    k_packed: &[u8], v_packed: &[u8], layer_idx: usize, pos: u32,
    n_heads: u32, n_kv_heads: u32, head_dim: u32, max_ctx: u32, n_layers: u32,
) -> bool {
    decode_seed_kv_quant(
        QuantKv::Mxfp6, k_packed, v_packed, None, None, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// MXFP8 decode seed across backends. See [`decode_seed_kv_quant`].
#[allow(clippy::too_many_arguments)]
pub fn decode_seed_kv_mxfp8(
    k_packed: &[u8], v_packed: &[u8], layer_idx: usize, pos: u32,
    n_heads: u32, n_kv_heads: u32, head_dim: u32, max_ctx: u32, n_layers: u32,
) -> bool {
    decode_seed_kv_quant(
        QuantKv::Mxfp8, k_packed, v_packed, None, None, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// TurboQuant decode seed across backends. See [`decode_seed_kv_quant`].
#[allow(clippy::too_many_arguments)]
pub fn decode_seed_kv_tq(
    bits: u8,
    k_packed: &[u8], v_packed: &[u8], k_scales: &[f32], v_scales: &[f32],
    layer_idx: usize, pos: u32,
    n_heads: u32, n_kv_heads: u32, head_dim: u32, max_ctx: u32, n_layers: u32,
) -> bool {
    decode_seed_kv_quant(
        QuantKv::Tq { bits }, k_packed, v_packed, Some(k_scales), Some(v_scales),
        layer_idx, pos, n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Shared body for every native-CUDA quantized-KV decode helper. Mirrors
/// [`try_flash_attn_decode_usm_quant`] (same shape gate, same CPU
/// quantizers, same epoch/valid-len gate): quantizes ONLY the new row into
/// a tiny reusable host staging buffer and offset-uploads it into this
/// layer's PERSISTENT device-resident packed mirror at `pos` (see the
/// module note) — no host shadow, no full re-upload. Returns `true` iff the
/// GPU produced the output (in `out`); `false` for any gate / shape / alloc
/// / kernel failure, on which the caller runs SYCL or CPU.
#[allow(clippy::too_many_arguments)]
fn try_flash_attn_decode_cuda_quant(
    fmt: QuantKv,
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    if !cuda_active() || !gpu_active_for_current_layer() {
        return false;
    }
    let cfg = CudaAttnConfig { n_layers, n_heads, n_kv_heads, head_dim, max_ctx };
    let hd = head_dim as usize;
    let n_kv = n_kv_heads as usize;
    let mc = max_ctx as usize;
    let posu = pos as usize;
    let q_len = (n_heads as usize) * hd;
    let kv_row_len = n_kv * hd;
    if q.len() != q_len
        || k_row.len() != kv_row_len
        || v_row.len() != kv_row_len
        || out.len() != q_len
        || layer_idx >= n_layers as usize
        || pos >= max_ctx
        || n_kv_heads == 0
        || n_heads % n_kv_heads != 0
    {
        return false;
    }
    // Per-format block constraint (same rules as the SYCL quant helper).
    let ok_shape = match fmt {
        QuantKv::Q4_0 => hd % 32 == 0,
        QuantKv::Q8_0 => true,
        QuantKv::Nvfp4 => hd % 16 == 0,
        QuantKv::Tq { bits } => matches!(bits, 1 | 2 | 4 | 8) && hd.is_power_of_two(),
        QuantKv::Mxfp4 | QuantKv::Mxfp6 | QuantKv::Mxfp8 => hd % 32 == 0,
    };
    if !ok_shape {
        return false;
    }
    let bytes_per_row = fmt.bytes_per_row(hd);
    let mirror_len = n_kv * mc * bytes_per_row; // packed bytes
    let scales_len = n_kv * mc; // f32 elements
    let q_bytes = q_len * 4;
    let kv_len = pos + 1;
    let has_scales = fmt.has_scales();

    CUDA_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let needs_rebuild = match slot.as_ref() {
            Some(ctx) => ctx.cfg != cfg,
            None => true,
        };
        if needs_rebuild {
            *slot = CudaAttnContext::try_new(cfg);
            if slot.is_none() {
                return false;
            }
        }
        let ctx = slot.as_mut().expect("cuda ctx just built");

        // Quant staleness gate (independent of the F32 gate + of SYCL).
        let epoch = USM_KV_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if ctx.kv_epoch_q != epoch {
            for v in ctx.kv_valid_len_q.iter_mut() {
                *v = 0;
            }
            ctx.kv_epoch_q = epoch;
        }
        let valid = ctx.kv_valid_len_q[layer_idx];
        if pos > valid {
            log_usm_kv_gap_once(layer_idx, pos, valid);
            return false;
        }

        // A format change for this layer (never in practice) re-allocates
        // the device mirror (+ scales) and resets its valid range — mirrors
        // the SYCL quant gate.
        if ctx.q_bytes_per_row[layer_idx] as usize != bytes_per_row {
            ctx.dev_qk_mirror[layer_idx] = None;
            ctx.dev_qv_mirror[layer_idx] = None;
            ctx.dev_qk_scales[layer_idx] = None;
            ctx.dev_qv_scales[layer_idx] = None;
            ctx.kv_valid_len_q[layer_idx] = 0;
            ctx.q_bytes_per_row[layer_idx] = bytes_per_row as u32;
        }

        // Ensure this layer's persistent packed device mirror (+ scales),
        // the shared Q / out scratch, and the reusable one-row host staging.
        // Each `[layer_idx]` mut-borrows a distinct Vec field, disjoint from
        // the `&ctx.stream` borrow. Any alloc failure → CPU.
        if !ensure_cuda_dev(&mut ctx.dev_qk_mirror[layer_idx], &ctx.stream, mirror_len)
            || !ensure_cuda_dev(&mut ctx.dev_qv_mirror[layer_idx], &ctx.stream, mirror_len)
            || !ensure_cuda_dev(&mut ctx.dev_q, &ctx.stream, q_bytes)
            || !ensure_cuda_dev(&mut ctx.dev_out, &ctx.stream, q_bytes)
        {
            return false;
        }
        if has_scales
            && (!ensure_cuda_dev(&mut ctx.dev_qk_scales[layer_idx], &ctx.stream, scales_len * 4)
                || !ensure_cuda_dev(&mut ctx.dev_qv_scales[layer_idx], &ctx.stream, scales_len * 4))
        {
            return false;
        }
        // Host staging for ONE position's new row (all kv-heads), reused for
        // K then V. Packed: `n_kv*bytes_per_row` bytes; scales: `n_kv` f32.
        let stage_bytes = n_kv * bytes_per_row;
        if !ensure_cuda_shadow_u8(&mut ctx.stage_row, stage_bytes) {
            return false;
        }
        if has_scales && !ensure_cuda_shadow_f32(&mut ctx.stage_scales, n_kv) {
            return false;
        }

        // Quantize + offset-upload the new K row, then the new V row,
        // reusing the one staging buffer. The SAME CPU quantizers the
        // engine's host slab uses keep each mirror row byte-identical, so
        // the GPU output equals the CPU flash-decode result to the
        // fp-reduction tolerance the F32 path already tolerates. Only ONE
        // row moves host→device per call — no full re-upload, no host
        // shadow. `cuda_upload_staged_row` writes the packed row (+ scales)
        // at `pos` via per-kv-head `copy_from_host_at`, and returns false
        // (→ CPU) on any H2D error BEFORE the valid range is extended, so a
        // half-written row is never marked valid.
        cuda_quantize_row_into_stage(
            fmt, k_row, &mut ctx.stage_row, &mut ctx.stage_scales, n_kv, hd, bytes_per_row,
        );
        if !cuda_upload_staged_row(
            ctx.dev_qk_mirror[layer_idx].as_mut().unwrap(),
            ctx.dev_qk_scales[layer_idx].as_mut(),
            &ctx.stage_row,
            &ctx.stage_scales,
            n_kv, mc, posu, bytes_per_row, has_scales,
        ) {
            return false;
        }
        cuda_quantize_row_into_stage(
            fmt, v_row, &mut ctx.stage_row, &mut ctx.stage_scales, n_kv, hd, bytes_per_row,
        );
        if !cuda_upload_staged_row(
            ctx.dev_qv_mirror[layer_idx].as_mut().unwrap(),
            ctx.dev_qv_scales[layer_idx].as_mut(),
            &ctx.stage_row,
            &ctx.stage_scales,
            n_kv, mc, posu, bytes_per_row, has_scales,
        ) {
            return false;
        }
        // Extend the valid range ONLY after both rows are fully resident
        // (contiguous append at `pos == valid`; in-place overwrite at
        // `pos < valid` leaves it). Same rule as the SYCL quant path.
        if pos == valid {
            ctx.kv_valid_len_q[layer_idx] = pos + 1;
        }

        // Upload Q into the shared scratch.
        {
            let q_src: &[u8] =
                unsafe { std::slice::from_raw_parts(q.as_ptr() as *const u8, q_bytes) };
            if ctx.dev_q.as_mut().unwrap().copy_from_host(q_src).is_err() {
                return false;
            }
        }

        // Gather raw device pointers (disjoint fields) and launch the
        // format-matching kernel against the persistent packed mirror.
        // Packed K/V are passed as `*const c_void` (the kernel's type);
        // scales / q / out as f32 pointers.
        let stream = &ctx.stream;
        let q_ptr = ctx.dev_q.as_ref().unwrap().as_ptr() as *const f32;
        let k_ptr = ctx.dev_qk_mirror[layer_idx].as_ref().unwrap().as_ptr();
        let v_ptr = ctx.dev_qv_mirror[layer_idx].as_ref().unwrap().as_ptr();
        let out_ptr = ctx.dev_out.as_mut().unwrap().as_mut_ptr() as *mut f32;
        let (ks_ptr, vs_ptr) = if has_scales {
            (
                ctx.dev_qk_scales[layer_idx].as_ref().unwrap().as_ptr() as *const f32,
                ctx.dev_qv_scales[layer_idx].as_ref().unwrap().as_ptr() as *const f32,
            )
        } else {
            (std::ptr::null::<f32>(), std::ptr::null::<f32>())
        };
        // SAFETY: every pointer is a live device allocation on `stream`,
        // sized for the shapes below; the wrappers synchronize before
        // returning so the read-back doesn't race the kernel.
        let launched = unsafe {
            match fmt {
                QuantKv::Q4_0 => ck::flash_attn_decode_q4_0(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as usize, n_kv, hd, mc, kv_len as usize,
                ),
                QuantKv::Q8_0 => ck::flash_attn_decode_q8_0(
                    stream, q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, out_ptr,
                    n_heads as usize, n_kv, hd, mc, kv_len as usize,
                ),
                QuantKv::Nvfp4 => ck::flash_attn_decode_nvfp4(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as usize, n_kv, hd, mc, kv_len as usize,
                ),
                QuantKv::Tq { bits } => ck::flash_attn_decode_tq(
                    stream, q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, bits as u32, out_ptr,
                    n_heads as usize, n_kv, hd, mc, kv_len as usize,
                ),
                QuantKv::Mxfp4 => ck::flash_attn_decode_mxfp4(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as usize, n_kv, hd, mc, kv_len as usize,
                ),
                QuantKv::Mxfp6 => ck::flash_attn_decode_mxfp6(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as usize, n_kv, hd, mc, kv_len as usize,
                ),
                QuantKv::Mxfp8 => ck::flash_attn_decode_mxfp8(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as usize, n_kv, hd, mc, kv_len as usize,
                ),
            }
        }
        .is_ok();
        if !launched || ck::consume_error_count() != 0 {
            return false;
        }

        // Read the f32 result back into the caller's buffer.
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, q_bytes) };
        ctx.dev_out.as_ref().unwrap().copy_to_host(out_bytes).is_ok()
    })
}

/// Native-CUDA quantized-KV (Q4_0) flash-attention decode. Same signature
/// as [`try_flash_attn_decode_usm_q4_0`]. `false` ⇒ caller uses SYCL/CPU.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_cuda_q4_0(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    try_flash_attn_decode_cuda_quant(
        QuantKv::Q4_0, q, k_row, v_row, out, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Native-CUDA quantized-KV (Q8_0 i8 slab + per-row absmax) decode.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_cuda_q8_0(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    try_flash_attn_decode_cuda_quant(
        QuantKv::Q8_0, q, k_row, v_row, out, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Native-CUDA quantized-KV (NVFP4) flash-attention decode.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_cuda_nvfp4(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    try_flash_attn_decode_cuda_quant(
        QuantKv::Nvfp4, q, k_row, v_row, out, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Native-CUDA quantized-KV (TurboQuant, `bits` ∈ {1,2,4,8}) decode.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_cuda_tq(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    bits: u8,
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    try_flash_attn_decode_cuda_quant(
        QuantKv::Tq { bits }, q, k_row, v_row, out, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Diagnostic: clear the per-thread CUDA attention context. Sibling of
/// [`clear_usm_attn_context`].
pub fn clear_cuda_attn_context() {
    CUDA_ATTN.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

// ==================================================================
// MLX-resident flash-attention decode + prefill (Apple Metal)
// ==================================================================
//
// The Apple-Metal analogue of the CUDA-resident flash-attention path above.
// It mirrors that design EXACTLY — a per-thread context that keeps K/V
// resident on the accelerator, an epoch/valid-len staleness gate, "write
// the new row, run the kernel over `[0, pos+1)`, read the output back" — and
// keeps CUDA's explicit H2D/D2H shape even though Apple Silicon is UNIFIED
// MEMORY (the CPU and GPU share one physical pool). Keeping the identical
// shape makes this a mechanical mirror; Phase 1 may collapse the offset H2D
// copies into direct writes of the shared MTLBuffer page (like the SYCL USM
// path). The staleness gate is fully INDEPENDENT of both the SYCL and CUDA
// contexts: its own thread-local (`MLX_ATTN`), its own `kv_valid_len_*` /
// `kv_epoch_*`. Everything is INERT off Apple Silicon (`mlx_active()` ==
// false → these are never called), so the SYCL/CUDA/CPU paths are unchanged.
//
// The backend-agnostic HOST helpers are reused from the CUDA block above:
// `cuda_quantize_row_into_stage` (pure CPU quantization into host staging)
// and `ensure_cuda_shadow_u8` / `ensure_cuda_shadow_f32` (plain `Vec`
// growth) touch no device state, so duplicating them under `mlx_` names
// would only add drift. The device-typed helpers (`ensure_mlx_dev`,
// `mlx_upload_staged_row`) mirror their CUDA peers 1:1 over the `mk::` API.

/// Sized parameters for the MLX attention context. Mirror of
/// [`CudaAttnConfig`]; a mismatch forces a rebuild.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MlxAttnConfig {
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
}

/// Per-thread MLX attention context. Byte-for-byte mirror of
/// [`CudaAttnContext`] over the `mk::` device API: a Metal/MLX stream,
/// per-layer persistent device-resident K/V mirrors (written one row at a
/// time via an offset H2D copy — no host shadow, no full re-upload), a small
/// reusable host staging buffer for the quant path's new row, shared Q / out
/// device scratch, and disjoint F32 / quant staleness gates.
///
/// FIELD ORDER MATTERS: every `mk::MlxDeviceBuffer<'static>` here actually
/// borrows `self.stream` (the `'static` is the same lifetime lie the SYCL /
/// CUDA contexts tell); they are declared BEFORE `stream` so declaration-
/// order field drop frees them while the stream is still alive.
struct MlxAttnContext {
    cfg: MlxAttnConfig,
    // ---- per-layer persistent F32 K/V device mirrors ----
    dev_k_mirror: Vec<Option<mk::MlxDeviceBuffer<'static>>>,
    dev_v_mirror: Vec<Option<mk::MlxDeviceBuffer<'static>>>,
    // ---- per-layer persistent quant K/V device mirrors (+ scales) ----
    dev_qk_mirror: Vec<Option<mk::MlxDeviceBuffer<'static>>>,
    dev_qv_mirror: Vec<Option<mk::MlxDeviceBuffer<'static>>>,
    dev_qk_scales: Vec<Option<mk::MlxDeviceBuffer<'static>>>,
    dev_qv_scales: Vec<Option<mk::MlxDeviceBuffer<'static>>>,
    // ---- shared per-step device scratch (reused across layers) ----
    dev_q: Option<mk::MlxDeviceBuffer<'static>>,
    dev_out: Option<mk::MlxDeviceBuffer<'static>>,
    // ---- reusable host staging for ONE new KV row (quant path only) ----
    stage_row: Vec<u8>,
    stage_scales: Vec<f32>,
    // ---- disjoint staleness gates (F32 vs quant; independent of SYCL/CUDA) -
    kv_valid_len_f32: Vec<u32>,
    kv_epoch_f32: u64,
    q_bytes_per_row: Vec<u32>,
    kv_valid_len_q: Vec<u32>,
    kv_epoch_q: u64,
    // ---- prefill device scratch (batched flash-prefill) ----
    dev_prefill_k: Option<mk::MlxDeviceBuffer<'static>>,
    dev_prefill_v: Option<mk::MlxDeviceBuffer<'static>>,
    dev_prefill_k_scales: Option<mk::MlxDeviceBuffer<'static>>,
    dev_prefill_v_scales: Option<mk::MlxDeviceBuffer<'static>>,
    dev_prefill_q: Option<mk::MlxDeviceBuffer<'static>>,
    dev_prefill_out: Option<mk::MlxDeviceBuffer<'static>>,
    // ---- owning stream (declared LAST so the buffers above free first) --
    stream: mk::MlxStream,
}

impl MlxAttnContext {
    /// Build a fresh MLX attention context for the given model shape.
    /// `None` when no Metal device / stream creation fails (callers fall
    /// back to SYCL or CPU). Uses device 0 — the same device the MLX matvec
    /// cache binds to (`MlxMatvecCache::new(0, ...)`).
    fn try_new(cfg: MlxAttnConfig) -> Option<Self> {
        let stream = mk::MlxStream::create(0)?;
        let n = cfg.n_layers as usize;
        Some(Self {
            cfg,
            dev_k_mirror: (0..n).map(|_| None).collect(),
            dev_v_mirror: (0..n).map(|_| None).collect(),
            dev_qk_mirror: (0..n).map(|_| None).collect(),
            dev_qv_mirror: (0..n).map(|_| None).collect(),
            dev_qk_scales: (0..n).map(|_| None).collect(),
            dev_qv_scales: (0..n).map(|_| None).collect(),
            dev_q: None,
            dev_out: None,
            stage_row: Vec::new(),
            stage_scales: Vec::new(),
            kv_valid_len_f32: vec![0; n],
            kv_epoch_f32: 0,
            q_bytes_per_row: vec![0; n],
            kv_valid_len_q: vec![0; n],
            kv_epoch_q: 0,
            dev_prefill_k: None,
            dev_prefill_v: None,
            dev_prefill_k_scales: None,
            dev_prefill_v_scales: None,
            dev_prefill_q: None,
            dev_prefill_out: None,
            stream,
        })
    }
}

thread_local! {
    /// Per-thread MLX attention context. Built lazily on the first
    /// `try_flash_attn_decode_mlx_*` call once `mlx_active()` is true and a
    /// matching config is presented; rebuilt on a config change. Independent
    /// of [`USM_ATTN`] and `CUDA_ATTN` — the three backends never share state.
    static MLX_ATTN: std::cell::RefCell<Option<MlxAttnContext>> =
        const { std::cell::RefCell::new(None) };
}

/// Ensure `slot` holds a device buffer of at least `need_bytes`. Mirror of
/// [`ensure_cuda_dev`] over the `mk::` API (grows free-before-alloc, reused
/// otherwise). `need_bytes` must be `> 0`.
fn ensure_mlx_dev(
    slot: &mut Option<mk::MlxDeviceBuffer<'static>>,
    stream: &mk::MlxStream,
    need_bytes: usize,
) -> bool {
    if slot.as_ref().map_or(false, |b| b.len_bytes() >= need_bytes) {
        return true;
    }
    // Free the too-small buffer before allocating its replacement so peak
    // device memory doesn't double.
    *slot = None;
    match mk::MlxDeviceBuffer::alloc(stream, need_bytes) {
        Some(b) => {
            // SAFETY: the `'_`→`'static` widening is the same lifetime lie
            // the CUDA/SYCL contexts tell — the buffer borrows `stream` (a
            // field of the owning context), which is declared last and so
            // outlives every buffer via declaration-order field drop.
            *slot = Some(unsafe {
                std::mem::transmute::<
                    mk::MlxDeviceBuffer<'_>,
                    mk::MlxDeviceBuffer<'static>,
                >(b)
            });
            true
        }
        None => false,
    }
}

/// Offset-upload ONE staged position's packed K (or V) row (+ optional
/// per-row scales) into its persistent device mirror at `pos`. Mirror of
/// [`cuda_upload_staged_row`] over the `mk::` API — the head-major
/// `[n_kv, max_ctx, bytes_per_row]` layout means each kv-head's new row
/// lands at `(h*max_ctx + pos)*bytes_per_row` (a separate `copy_from_host_at`
/// per head). Returns false on any H2D error BEFORE the caller extends
/// `kv_valid_len`, so a half-written row is never marked valid.
#[allow(clippy::too_many_arguments)]
fn mlx_upload_staged_row(
    mirror: &mut mk::MlxDeviceBuffer<'static>,
    scales_mirror: Option<&mut mk::MlxDeviceBuffer<'static>>,
    stage_row: &[u8],
    stage_scales: &[f32],
    n_kv: usize,
    mc: usize,
    pos: usize,
    bytes_per_row: usize,
    has_scales: bool,
) -> bool {
    for h in 0..n_kv {
        let off = (h * mc + pos) * bytes_per_row;
        if mirror
            .copy_from_host_at(off, &stage_row[h * bytes_per_row..(h + 1) * bytes_per_row])
            .is_err()
        {
            return false;
        }
    }
    if has_scales {
        let sm = match scales_mirror {
            Some(s) => s,
            None => return false,
        };
        let sb: &[u8] = unsafe {
            std::slice::from_raw_parts(stage_scales.as_ptr() as *const u8, n_kv * 4)
        };
        for h in 0..n_kv {
            let off = (h * mc + pos) * 4;
            if sm.copy_from_host_at(off, &sb[h * 4..(h + 1) * 4]).is_err() {
                return false;
            }
        }
    }
    true
}

/// Try to run F32 KV flash-attention decode on the native MLX backend.
/// Mirror of [`try_flash_attn_decode_cuda_f32`]; same public signature as
/// [`try_flash_attn_decode_usm_f32`]. `true` iff the GPU produced `out`.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_mlx_f32(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    if !mlx_active() || !gpu_active_for_current_layer() {
        return false;
    }
    let cfg = MlxAttnConfig { n_layers, n_heads, n_kv_heads, head_dim, max_ctx };
    let hd = head_dim as usize;
    let n_kv = n_kv_heads as usize;
    let mc = max_ctx as usize;
    let posu = pos as usize;
    let q_len = (n_heads as usize) * hd;
    let kv_row_len = n_kv * hd;
    if q.len() != q_len
        || k_row.len() != kv_row_len
        || v_row.len() != kv_row_len
        || out.len() != q_len
        || layer_idx >= n_layers as usize
        || pos >= max_ctx
        || n_kv_heads == 0
        || n_heads % n_kv_heads != 0
    {
        return false;
    }
    let mirror_bytes = n_kv * mc * hd * 4; // full f32 device-mirror bytes
    let row_bytes = hd * 4; // one kv-head's f32 row
    let q_bytes = q_len * 4;
    let kv_len = pos + 1;

    MLX_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let needs_rebuild = match slot.as_ref() {
            Some(ctx) => ctx.cfg != cfg,
            None => true,
        };
        if needs_rebuild {
            *slot = MlxAttnContext::try_new(cfg);
            if slot.is_none() {
                return false;
            }
        }
        let ctx = slot.as_mut().expect("mlx ctx just built");

        // F32 staleness gate (independent of the quant gate + of SYCL/CUDA).
        let epoch = USM_KV_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if ctx.kv_epoch_f32 != epoch {
            for v in ctx.kv_valid_len_f32.iter_mut() {
                *v = 0;
            }
            ctx.kv_epoch_f32 = epoch;
        }
        let valid = ctx.kv_valid_len_f32[layer_idx];
        if pos > valid {
            log_usm_kv_gap_once(layer_idx, pos, valid);
            return false;
        }

        // Ensure this layer's persistent F32 device mirror + shared Q / out
        // scratch (each `[layer_idx]` mut-borrows a distinct Vec field,
        // disjoint from the `&ctx.stream` borrow).
        if !ensure_mlx_dev(&mut ctx.dev_k_mirror[layer_idx], &ctx.stream, mirror_bytes)
            || !ensure_mlx_dev(&mut ctx.dev_v_mirror[layer_idx], &ctx.stream, mirror_bytes)
            || !ensure_mlx_dev(&mut ctx.dev_q, &ctx.stream, q_bytes)
            || !ensure_mlx_dev(&mut ctx.dev_out, &ctx.stream, q_bytes)
        {
            return false;
        }

        // Write ONLY the new K/V row into the resident mirror at `pos`, one
        // kv-head at a time (head-major ⇒ heads are `max_ctx*head_dim` apart).
        {
            let k_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(k_row.as_ptr() as *const u8, kv_row_len * 4)
            };
            let kmir = ctx.dev_k_mirror[layer_idx].as_mut().unwrap();
            for h in 0..n_kv {
                let off = (h * mc + posu) * row_bytes;
                if kmir
                    .copy_from_host_at(off, &k_bytes[h * row_bytes..(h + 1) * row_bytes])
                    .is_err()
                {
                    return false;
                }
            }
        }
        {
            let v_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(v_row.as_ptr() as *const u8, kv_row_len * 4)
            };
            let vmir = ctx.dev_v_mirror[layer_idx].as_mut().unwrap();
            for h in 0..n_kv {
                let off = (h * mc + posu) * row_bytes;
                if vmir
                    .copy_from_host_at(off, &v_bytes[h * row_bytes..(h + 1) * row_bytes])
                    .is_err()
                {
                    return false;
                }
            }
        }
        // Extend the valid range ONLY after the row is fully resident.
        if pos == valid {
            ctx.kv_valid_len_f32[layer_idx] = pos + 1;
        }

        // Upload Q into the shared scratch.
        {
            let q_src: &[u8] =
                unsafe { std::slice::from_raw_parts(q.as_ptr() as *const u8, q_bytes) };
            if ctx.dev_q.as_mut().unwrap().copy_from_host(q_src).is_err() {
                return false;
            }
        }

        // Gather raw device pointers (disjoint fields) and launch over
        // `[0, pos+1)` against the persistent mirror. No full re-upload.
        let stream = &ctx.stream;
        let q_ptr = ctx.dev_q.as_ref().unwrap().as_ptr() as *const f32;
        let k_ptr = ctx.dev_k_mirror[layer_idx].as_ref().unwrap().as_ptr() as *const f32;
        let v_ptr = ctx.dev_v_mirror[layer_idx].as_ref().unwrap().as_ptr() as *const f32;
        let out_ptr = ctx.dev_out.as_mut().unwrap().as_mut_ptr() as *mut f32;
        // SAFETY: all four are live device pointers on `stream`, sized for
        // the shapes below; the wrapper synchronizes before returning so
        // the read-back doesn't race the kernel.
        let launched = unsafe {
            mk::flash_attn_decode_f32(
                stream, q_ptr, k_ptr, v_ptr, out_ptr,
                n_heads as usize, n_kv, hd, mc, kv_len as usize,
            )
        }
        .is_ok();
        if !launched || mk::consume_error_count() != 0 {
            return false;
        }

        // Read the result back into the caller's f32 buffer.
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, q_bytes) };
        ctx.dev_out.as_ref().unwrap().copy_to_host(out_bytes).is_ok()
    })
}

/// Shared body for every native-MLX quantized-KV decode helper. Mirror of
/// [`try_flash_attn_decode_cuda_quant`] over the `mk::` API. `true` iff the
/// GPU produced `out`.
#[allow(clippy::too_many_arguments)]
fn try_flash_attn_decode_mlx_quant(
    fmt: QuantKv,
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    if !mlx_active() || !gpu_active_for_current_layer() {
        return false;
    }
    let cfg = MlxAttnConfig { n_layers, n_heads, n_kv_heads, head_dim, max_ctx };
    let hd = head_dim as usize;
    let n_kv = n_kv_heads as usize;
    let mc = max_ctx as usize;
    let posu = pos as usize;
    let q_len = (n_heads as usize) * hd;
    let kv_row_len = n_kv * hd;
    if q.len() != q_len
        || k_row.len() != kv_row_len
        || v_row.len() != kv_row_len
        || out.len() != q_len
        || layer_idx >= n_layers as usize
        || pos >= max_ctx
        || n_kv_heads == 0
        || n_heads % n_kv_heads != 0
    {
        return false;
    }
    // Per-format block constraint (same rules as the CUDA/SYCL quant helper).
    let ok_shape = match fmt {
        QuantKv::Q4_0 => hd % 32 == 0,
        QuantKv::Q8_0 => true,
        QuantKv::Nvfp4 => hd % 16 == 0,
        QuantKv::Tq { bits } => matches!(bits, 1 | 2 | 4 | 8) && hd.is_power_of_two(),
        QuantKv::Mxfp4 | QuantKv::Mxfp6 | QuantKv::Mxfp8 => hd % 32 == 0,
    };
    if !ok_shape {
        return false;
    }
    let bytes_per_row = fmt.bytes_per_row(hd);
    let mirror_len = n_kv * mc * bytes_per_row; // packed bytes
    let scales_len = n_kv * mc; // f32 elements
    let q_bytes = q_len * 4;
    let kv_len = pos + 1;
    let has_scales = fmt.has_scales();

    MLX_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let needs_rebuild = match slot.as_ref() {
            Some(ctx) => ctx.cfg != cfg,
            None => true,
        };
        if needs_rebuild {
            *slot = MlxAttnContext::try_new(cfg);
            if slot.is_none() {
                return false;
            }
        }
        let ctx = slot.as_mut().expect("mlx ctx just built");

        // Quant staleness gate (independent of the F32 gate + of SYCL/CUDA).
        let epoch = USM_KV_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if ctx.kv_epoch_q != epoch {
            for v in ctx.kv_valid_len_q.iter_mut() {
                *v = 0;
            }
            ctx.kv_epoch_q = epoch;
        }
        let valid = ctx.kv_valid_len_q[layer_idx];
        if pos > valid {
            log_usm_kv_gap_once(layer_idx, pos, valid);
            return false;
        }

        // A format change for this layer re-allocates the device mirror (+
        // scales) and resets its valid range — mirrors the CUDA quant gate.
        if ctx.q_bytes_per_row[layer_idx] as usize != bytes_per_row {
            ctx.dev_qk_mirror[layer_idx] = None;
            ctx.dev_qv_mirror[layer_idx] = None;
            ctx.dev_qk_scales[layer_idx] = None;
            ctx.dev_qv_scales[layer_idx] = None;
            ctx.kv_valid_len_q[layer_idx] = 0;
            ctx.q_bytes_per_row[layer_idx] = bytes_per_row as u32;
        }

        // Ensure this layer's persistent packed device mirror (+ scales),
        // the shared Q / out scratch, and the reusable one-row host staging.
        if !ensure_mlx_dev(&mut ctx.dev_qk_mirror[layer_idx], &ctx.stream, mirror_len)
            || !ensure_mlx_dev(&mut ctx.dev_qv_mirror[layer_idx], &ctx.stream, mirror_len)
            || !ensure_mlx_dev(&mut ctx.dev_q, &ctx.stream, q_bytes)
            || !ensure_mlx_dev(&mut ctx.dev_out, &ctx.stream, q_bytes)
        {
            return false;
        }
        if has_scales
            && (!ensure_mlx_dev(&mut ctx.dev_qk_scales[layer_idx], &ctx.stream, scales_len * 4)
                || !ensure_mlx_dev(&mut ctx.dev_qv_scales[layer_idx], &ctx.stream, scales_len * 4))
        {
            return false;
        }
        // Host staging for ONE position's new row (all kv-heads), reused for
        // K then V. Reuses the backend-agnostic `cuda_`-named host helpers.
        let stage_bytes = n_kv * bytes_per_row;
        if !ensure_cuda_shadow_u8(&mut ctx.stage_row, stage_bytes) {
            return false;
        }
        if has_scales && !ensure_cuda_shadow_f32(&mut ctx.stage_scales, n_kv) {
            return false;
        }

        // Quantize + offset-upload the new K row, then the new V row, reusing
        // the one staging buffer. The SAME CPU quantizers the engine's host
        // slab uses keep each mirror row byte-identical, so the GPU output
        // equals the CPU flash-decode result. Only ONE row moves host→device
        // per call. `mlx_upload_staged_row` returns false (→ CPU) on any H2D
        // error BEFORE the valid range is extended.
        cuda_quantize_row_into_stage(
            fmt, k_row, &mut ctx.stage_row, &mut ctx.stage_scales, n_kv, hd, bytes_per_row,
        );
        if !mlx_upload_staged_row(
            ctx.dev_qk_mirror[layer_idx].as_mut().unwrap(),
            ctx.dev_qk_scales[layer_idx].as_mut(),
            &ctx.stage_row,
            &ctx.stage_scales,
            n_kv, mc, posu, bytes_per_row, has_scales,
        ) {
            return false;
        }
        cuda_quantize_row_into_stage(
            fmt, v_row, &mut ctx.stage_row, &mut ctx.stage_scales, n_kv, hd, bytes_per_row,
        );
        if !mlx_upload_staged_row(
            ctx.dev_qv_mirror[layer_idx].as_mut().unwrap(),
            ctx.dev_qv_scales[layer_idx].as_mut(),
            &ctx.stage_row,
            &ctx.stage_scales,
            n_kv, mc, posu, bytes_per_row, has_scales,
        ) {
            return false;
        }
        // Extend the valid range ONLY after both rows are fully resident.
        if pos == valid {
            ctx.kv_valid_len_q[layer_idx] = pos + 1;
        }

        // Upload Q into the shared scratch.
        {
            let q_src: &[u8] =
                unsafe { std::slice::from_raw_parts(q.as_ptr() as *const u8, q_bytes) };
            if ctx.dev_q.as_mut().unwrap().copy_from_host(q_src).is_err() {
                return false;
            }
        }

        // Gather raw device pointers (disjoint fields) and launch the
        // format-matching kernel against the persistent packed mirror.
        let stream = &ctx.stream;
        let q_ptr = ctx.dev_q.as_ref().unwrap().as_ptr() as *const f32;
        let k_ptr = ctx.dev_qk_mirror[layer_idx].as_ref().unwrap().as_ptr();
        let v_ptr = ctx.dev_qv_mirror[layer_idx].as_ref().unwrap().as_ptr();
        let out_ptr = ctx.dev_out.as_mut().unwrap().as_mut_ptr() as *mut f32;
        let (ks_ptr, vs_ptr) = if has_scales {
            (
                ctx.dev_qk_scales[layer_idx].as_ref().unwrap().as_ptr() as *const f32,
                ctx.dev_qv_scales[layer_idx].as_ref().unwrap().as_ptr() as *const f32,
            )
        } else {
            (std::ptr::null::<f32>(), std::ptr::null::<f32>())
        };
        // SAFETY: every pointer is a live device allocation on `stream`,
        // sized for the shapes below; the wrappers synchronize before
        // returning so the read-back doesn't race the kernel.
        let launched = unsafe {
            match fmt {
                QuantKv::Q4_0 => mk::flash_attn_decode_q4_0(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as usize, n_kv, hd, mc, kv_len as usize,
                ),
                QuantKv::Q8_0 => mk::flash_attn_decode_q8_0(
                    stream, q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, out_ptr,
                    n_heads as usize, n_kv, hd, mc, kv_len as usize,
                ),
                QuantKv::Nvfp4 => mk::flash_attn_decode_nvfp4(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as usize, n_kv, hd, mc, kv_len as usize,
                ),
                QuantKv::Tq { bits } => mk::flash_attn_decode_tq(
                    stream, q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, bits as u32, out_ptr,
                    n_heads as usize, n_kv, hd, mc, kv_len as usize,
                ),
                QuantKv::Mxfp4 => mk::flash_attn_decode_mxfp4(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as usize, n_kv, hd, mc, kv_len as usize,
                ),
                QuantKv::Mxfp6 => mk::flash_attn_decode_mxfp6(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as usize, n_kv, hd, mc, kv_len as usize,
                ),
                QuantKv::Mxfp8 => mk::flash_attn_decode_mxfp8(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as usize, n_kv, hd, mc, kv_len as usize,
                ),
            }
        }
        .is_ok();
        if !launched || mk::consume_error_count() != 0 {
            return false;
        }

        // Read the f32 result back into the caller's buffer.
        let out_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, q_bytes) };
        ctx.dev_out.as_ref().unwrap().copy_to_host(out_bytes).is_ok()
    })
}

/// Native-MLX quantized-KV (Q4_0) flash-attention decode. Same signature as
/// [`try_flash_attn_decode_usm_q4_0`]. `false` ⇒ caller uses SYCL/CPU.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_mlx_q4_0(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    try_flash_attn_decode_mlx_quant(
        QuantKv::Q4_0, q, k_row, v_row, out, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Native-MLX quantized-KV (Q8_0 i8 slab + per-row absmax) decode.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_mlx_q8_0(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    try_flash_attn_decode_mlx_quant(
        QuantKv::Q8_0, q, k_row, v_row, out, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Native-MLX quantized-KV (NVFP4) flash-attention decode.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_mlx_nvfp4(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    try_flash_attn_decode_mlx_quant(
        QuantKv::Nvfp4, q, k_row, v_row, out, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Native-MLX quantized-KV (TurboQuant, `bits` ∈ {1,2,4,8}) decode.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_mlx_tq(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    bits: u8,
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    try_flash_attn_decode_mlx_quant(
        QuantKv::Tq { bits }, q, k_row, v_row, out, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Diagnostic: clear the per-thread MLX attention context. Sibling of
/// [`clear_cuda_attn_context`].
pub fn clear_mlx_attn_context() {
    MLX_ATTN.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

// ------------------------------------------------------------------
// GPU flash-attention decode dispatchers (CUDA → SYCL → CPU)
// ------------------------------------------------------------------
//
// One combinator per KV dtype. Each tries the native CUDA backend first
// when a CUDA device is present (`cuda_active()`), else the SYCL-USM path,
// returning `true` iff a GPU produced the output. The CPU fallback lives
// at the single call site (`if !try_flash_attn_decode_gpu_*(...) { CPU }`),
// so both backends are covered without duplicating it per site. On a
// SYCL-only / CPU-only host the `cuda_active()` short-circuit means these
// behave EXACTLY like calling the SYCL helper directly — no F32/SYCL/CPU
// regression.

/// F32 KV flash-attention decode: CUDA → SYCL → (caller's CPU on `false`).
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_gpu_f32(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    if cuda_active()
        && try_flash_attn_decode_cuda_f32(
            q, k_row, v_row, out, layer_idx, pos,
            n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
        )
    {
        return true;
    }
    if mlx_active()
        && try_flash_attn_decode_mlx_f32(
            q, k_row, v_row, out, layer_idx, pos,
            n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
        )
    {
        return true;
    }
    try_flash_attn_decode_usm_f32(
        q, k_row, v_row, out, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Q4_0 KV flash-attention decode: CUDA → SYCL → (caller's CPU).
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_gpu_q4_0(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    if cuda_active()
        && try_flash_attn_decode_cuda_q4_0(
            q, k_row, v_row, out, layer_idx, pos,
            n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
        )
    {
        return true;
    }
    if mlx_active()
        && try_flash_attn_decode_mlx_q4_0(
            q, k_row, v_row, out, layer_idx, pos,
            n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
        )
    {
        return true;
    }
    try_flash_attn_decode_usm_q4_0(
        q, k_row, v_row, out, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Q8_0 KV flash-attention decode: CUDA → SYCL → (caller's CPU).
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_gpu_q8_0(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    if cuda_active()
        && try_flash_attn_decode_cuda_q8_0(
            q, k_row, v_row, out, layer_idx, pos,
            n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
        )
    {
        return true;
    }
    if mlx_active()
        && try_flash_attn_decode_mlx_q8_0(
            q, k_row, v_row, out, layer_idx, pos,
            n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
        )
    {
        return true;
    }
    try_flash_attn_decode_usm_q8_0(
        q, k_row, v_row, out, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// NVFP4 KV flash-attention decode: CUDA → SYCL → (caller's CPU).
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_gpu_nvfp4(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    if cuda_active()
        && try_flash_attn_decode_cuda_nvfp4(
            q, k_row, v_row, out, layer_idx, pos,
            n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
        )
    {
        return true;
    }
    if mlx_active()
        && try_flash_attn_decode_mlx_nvfp4(
            q, k_row, v_row, out, layer_idx, pos,
            n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
        )
    {
        return true;
    }
    try_flash_attn_decode_usm_nvfp4(
        q, k_row, v_row, out, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// MXFP4/6/8 KV flash-attention decode: CUDA → SYCL → (caller's CPU).
/// These call the generic quant dispatch directly (no per-format
/// intermediate wrappers), same as the prefill combinators below.
macro_rules! mxfp_decode_gpu_combinator {
    ($name:ident, $kv:expr) => {
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            q: &[f32],
            k_row: &[f32],
            v_row: &[f32],
            out: &mut [f32],
            layer_idx: usize,
            pos: u32,
            n_heads: u32,
            n_kv_heads: u32,
            head_dim: u32,
            max_ctx: u32,
            n_layers: u32,
        ) -> bool {
            if cuda_active()
                && try_flash_attn_decode_cuda_quant(
                    $kv, q, k_row, v_row, out, layer_idx, pos,
                    n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
                )
            {
                return true;
            }
            if mlx_active()
                && try_flash_attn_decode_mlx_quant(
                    $kv, q, k_row, v_row, out, layer_idx, pos,
                    n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
                )
            {
                return true;
            }
            try_flash_attn_decode_usm_quant(
                $kv, q, k_row, v_row, out, layer_idx, pos,
                n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
            )
        }
    };
}
mxfp_decode_gpu_combinator!(try_flash_attn_decode_gpu_mxfp4, QuantKv::Mxfp4);
mxfp_decode_gpu_combinator!(try_flash_attn_decode_gpu_mxfp6, QuantKv::Mxfp6);
mxfp_decode_gpu_combinator!(try_flash_attn_decode_gpu_mxfp8, QuantKv::Mxfp8);

/// TurboQuant KV flash-attention decode: CUDA → SYCL → (caller's CPU).
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_decode_gpu_tq(
    q: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    out: &mut [f32],
    bits: u8,
    layer_idx: usize,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    n_layers: u32,
) -> bool {
    if cuda_active()
        && try_flash_attn_decode_cuda_tq(
            q, k_row, v_row, out, bits, layer_idx, pos,
            n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
        )
    {
        return true;
    }
    if mlx_active()
        && try_flash_attn_decode_mlx_tq(
            q, k_row, v_row, out, bits, layer_idx, pos,
            n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
        )
    {
        return true;
    }
    try_flash_attn_decode_usm_tq(
        q, k_row, v_row, out, bits, layer_idx, pos,
        n_heads, n_kv_heads, head_dim, max_ctx, n_layers,
    )
}

/// Diagnostic: clear the per-thread USM attention context.
/// Useful for tests that want to drop the cached resources without
/// having to spin up a new thread.
pub fn clear_usm_attn_context() {
    USM_ATTN.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

// ---------------------------------------------------------------
//
// USM-resident dispatch for rmsnorm / silu_mul / rope. These reuse
// the same stream the USM attention context owns, falling back to
// `false` (CPU fallback) when:
//   - `RUSTLLAMA_USM_ATTN` is not set (no context exists)
//   - the context exists but allocation of a per-op scratch fails
//   - the shape doesn't match the existing scratch
//
// Each function lazily grows its dedicated scratch buffer on the
// existing context's stream; reuse across subsequent calls makes
// the per-call cost a host-side memcpy (f32→f16 / f16→f32) into
// USM-shared pages instead of the alloc+memcpy+memcpy+free
// envelope the host-pointer SYCL kernels pay.

/// Ensure `slot` holds a USM buffer of at least `min_len` u16
/// elements on `stream`. Reallocates if too small. Returns false
/// on alloc failure so callers can short-circuit to CPU.
fn ensure_scratch(
    stream: &sk::SyclStream,
    slot: &mut Option<sk::SyclSharedBuffer<'static, u16>>,
    min_len: usize,
) -> bool {
    let needs_alloc = slot.as_ref().map_or(true, |b| b.len() < min_len);
    if !needs_alloc {
        return true;
    }
    // Free any too-small previous buffer before allocating its
    // replacement so peak USM doesn't double.
    *slot = None;
    let alloc = sk::SyclSharedBuffer::alloc(stream, min_len);
    match alloc {
        Ok(buf) => {
            // SAFETY: same '_-to-'static transmute used by the
            // attention scratch — the Drop impl on UsmAttnContext
            // frees these before the stream goes.
            *slot = Some(unsafe {
                std::mem::transmute::<
                    sk::SyclSharedBuffer<'_, u16>,
                    sk::SyclSharedBuffer<'static, u16>,
                >(buf)
            });
            true
        }
        Err(_) => false,
    }
}

/// Same as [`ensure_scratch`] but for `f32` (the Q8_0 packed
/// matvec uses f32 activations + outputs). Identical reallocate-on-
/// grow / free-before-alloc semantics; isolated as a second
/// function to keep the type-juggling local.
///
/// Returns `Ok(reallocated)` where `reallocated == true` iff the
/// buffer was freed + re-allocated (i.e. its USM address changed).
/// Callers tracking pointer-identity caches (e.g. the matvec_x_f32
/// upload dedup cache) use this to invalidate.
fn ensure_scratch_f32_grow(
    stream: &sk::SyclStream,
    slot: &mut Option<sk::SyclSharedBuffer<'static, f32>>,
    min_len: usize,
) -> Result<bool, ()> {
    let needs_alloc = slot.as_ref().map_or(true, |b| b.len() < min_len);
    if !needs_alloc {
        return Ok(false);
    }
    *slot = None;
    let alloc = sk::SyclSharedBuffer::alloc(stream, min_len);
    match alloc {
        Ok(buf) => {
            // SAFETY: same '_-to-'static transmute pattern.
            *slot = Some(unsafe {
                std::mem::transmute::<
                    sk::SyclSharedBuffer<'_, f32>,
                    sk::SyclSharedBuffer<'static, f32>,
                >(buf)
            });
            Ok(true)
        }
        Err(_) => Err(()),
    }
}

/// Convenience wrapper around [`ensure_scratch_f32_grow`] that
/// returns the original `bool` shape: `true` on success regardless
/// of whether a re-alloc happened. Used by call sites that don't
/// care about pointer-identity caches.
fn ensure_scratch_f32(
    stream: &sk::SyclStream,
    slot: &mut Option<sk::SyclSharedBuffer<'static, f32>>,
    min_len: usize,
) -> bool {
    ensure_scratch_f32_grow(stream, slot, min_len).is_ok()
}

/// Try to run RMSNorm on the USM context's persistent stream.
/// `x` / `y` are length `n_rows * d`; `w` is length `d`. Returns
/// `true` if the GPU produced output (in `y`); `false` to fall
/// back to CPU. Mirrors [`try_rmsnorm_f32`] but bypasses the
/// per-call alloc+memcpy envelope by reusing USM scratch.
pub fn try_rmsnorm_usm_f32(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    d: usize,
    eps: f32,
) -> bool {
    if !usm_attn_enabled() || !gpu_active_for_current_layer() {
        return false;
    }
    if d == 0 || x.len() != y.len() || x.len() % d != 0 || w.len() < d {
        return false;
    }
    let n_rows = x.len() / d;
    let total = n_rows * d;
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(ctx) = slot.as_mut() else {
            return false;
        };
        if !ensure_scratch(&ctx.stream, &mut ctx.rmsnorm_x, total)
            || !ensure_scratch(&ctx.stream, &mut ctx.rmsnorm_w, d)
            || !ensure_scratch(&ctx.stream, &mut ctx.rmsnorm_y, total)
        {
            return false;
        }
        // F3: SIMD F32 → F16-bits packing into USM. Page-mapped on
        // iGPU so the store goes straight to shared memory.
        {
            let x_buf = ctx.rmsnorm_x.as_mut().expect("just allocated");
            rustllama_kernels_cpu::f32_to_f16_bits(x, &mut x_buf.as_mut_slice()[..total]);
        }
        {
            let w_buf = ctx.rmsnorm_w.as_mut().expect("just allocated");
            rustllama_kernels_cpu::f32_to_f16_bits(&w[..d], &mut w_buf.as_mut_slice()[..d]);
        }
        // Run the kernel via the raw FFI entry — the safe wrapper
        // would require simultaneous &mut borrows the borrow
        // checker won't accept across struct fields.
        let stream_raw: *const sk::SyclStream = &ctx.stream;
        let x_ptr = ctx.rmsnorm_x.as_ref().unwrap().as_ptr();
        let w_ptr = ctx.rmsnorm_w.as_ref().unwrap().as_ptr();
        let y_ptr = ctx.rmsnorm_y.as_mut().unwrap().as_mut_ptr();
        // SAFETY: all three buffers are USM allocations on the same
        // stream we pass; sizes validated above; ctx outlives the
        // call (we hold the borrow). The kernel `.wait()`s before
        // returning so host reads below don't race the device.
        let ok = unsafe {
            sk::rmsnorm_usm_raw(
                &*stream_raw,
                x_ptr,
                w_ptr,
                y_ptr,
                n_rows as u32,
                d as u32,
                eps,
            )
        }
        .is_ok();
        if !ok {
            return false;
        }
        // Convert USM f16 → host f32.
        let y_buf = ctx.rmsnorm_y.as_ref().expect("just used");
        for (i, dst) in y.iter_mut().enumerate() {
            *dst = f16::from_bits(y_buf.as_slice()[i]).to_f32();
        }
        true
    })
}

/// Fused "add residual + RMSNorm" USM dispatch. Computes
///   `hidden[i] = hidden[i] + branch[i]`     (residual add, in-place)
///   `y_norm[i] = rmsnorm(hidden_row)[i] * w[i]`
/// in one kernel pass via [`sk::add_rmsnorm_usm_raw`]. Replaces the
/// (CPU `add_inplace_f32` → `try_rmsnorm_usm_f32`) pair at the post-
/// attn and post-FFN norm sites in `llama_arch.rs`. On a 32-layer
/// decode this eliminates ~64 kernel barriers + 64 host-side add
/// loops per generated token.
///
/// Returns `true` if the GPU produced output (in `hidden` AND
/// `y_norm`); `false` to fall back to the unfused path. Same
/// fall-back contract as [`try_rmsnorm_usm_f32`].
pub fn try_add_rmsnorm_usm_f32(
    hidden: &mut [f32],
    branch: &[f32],
    w: &[f32],
    y_norm: &mut [f32],
    d: usize,
    eps: f32,
) -> bool {
    if !usm_attn_enabled() || !gpu_active_for_current_layer() {
        return false;
    }
    if d == 0
        || hidden.len() != branch.len()
        || hidden.len() != y_norm.len()
        || hidden.len() % d != 0
        || w.len() < d
    {
        return false;
    }
    let n_rows = hidden.len() / d;
    let total = n_rows * d;
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(ctx) = slot.as_mut() else {
            return false;
        };
        if !ensure_scratch(&ctx.stream, &mut ctx.rmsnorm_x, total)
            || !ensure_scratch(&ctx.stream, &mut ctx.rmsnorm_branch, total)
            || !ensure_scratch(&ctx.stream, &mut ctx.rmsnorm_w, d)
            || !ensure_scratch(&ctx.stream, &mut ctx.rmsnorm_y, total)
        {
            return false;
        }
        // F3: SIMD F32 → F16-bits packing for the three inputs.
        {
            let h_buf = ctx.rmsnorm_x.as_mut().expect("just allocated");
            rustllama_kernels_cpu::f32_to_f16_bits(hidden, &mut h_buf.as_mut_slice()[..total]);
        }
        {
            let b_buf = ctx.rmsnorm_branch.as_mut().expect("just allocated");
            rustllama_kernels_cpu::f32_to_f16_bits(branch, &mut b_buf.as_mut_slice()[..total]);
        }
        {
            let w_buf = ctx.rmsnorm_w.as_mut().expect("just allocated");
            rustllama_kernels_cpu::f32_to_f16_bits(&w[..d], &mut w_buf.as_mut_slice()[..d]);
        }
        // Dispatch the fused kernel via the raw FFI (the safe wrapper
        // would require overlapping &mut borrows the borrow checker
        // refuses; same pattern as `try_rmsnorm_usm_f32`).
        let stream_raw: *const sk::SyclStream = &ctx.stream;
        let hidden_ptr = ctx.rmsnorm_x.as_mut().unwrap().as_mut_ptr();
        let branch_ptr = ctx.rmsnorm_branch.as_ref().unwrap().as_ptr();
        let w_ptr = ctx.rmsnorm_w.as_ref().unwrap().as_ptr();
        let y_ptr = ctx.rmsnorm_y.as_mut().unwrap().as_mut_ptr();
        // SAFETY: four USM allocations on the same stream we pass,
        // all sized correctly per the checks above. The kernel
        // `.wait()`s before returning.
        let ok = unsafe {
            sk::add_rmsnorm_usm_raw(
                &*stream_raw,
                hidden_ptr,
                branch_ptr,
                w_ptr,
                y_ptr,
                n_rows as u32,
                d as u32,
                eps,
            )
        }
        .is_ok();
        if !ok {
            return false;
        }
        // Copy results back: the residual sum (now in USM `rmsnorm_x`)
        // overwrites `hidden`; the normalized output goes to `y_norm`.
        let h_after = ctx.rmsnorm_x.as_ref().expect("just used");
        for (i, dst) in hidden.iter_mut().enumerate() {
            *dst = f16::from_bits(h_after.as_slice()[i]).to_f32();
        }
        let y_buf = ctx.rmsnorm_y.as_ref().expect("just used");
        for (i, dst) in y_norm.iter_mut().enumerate() {
            *dst = f16::from_bits(y_buf.as_slice()[i]).to_f32();
        }
        true
    })
}

/// USM-resident SwiGLU: `out[i] = silu(x[i]) * y[i]`. All three
/// slices are the same length. Same fall-back semantics as
/// [`try_rmsnorm_usm_f32`].
pub fn try_silu_mul_usm_f32(x: &[f32], y: &[f32], out: &mut [f32]) -> bool {
    if !gpu_active_for_current_layer() {
        return false;
    }
    if !usm_attn_enabled() {
        return false;
    }
    if x.len() != y.len() || out.len() != x.len() || x.is_empty() {
        return false;
    }
    let n = x.len();
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(ctx) = slot.as_mut() else {
            return false;
        };
        if !ensure_scratch(&ctx.stream, &mut ctx.silu_x, n)
            || !ensure_scratch(&ctx.stream, &mut ctx.silu_y, n)
            || !ensure_scratch(&ctx.stream, &mut ctx.silu_out, n)
        {
            return false;
        }
        {
            // F3: SIMD F32 → F16-bits packing (F16C-accelerated).
            // Replaces the scalar `half::f16::from_f32(v).to_bits()`
            // loop that ran twice (one per input) per call.
            let x_buf = ctx.silu_x.as_mut().unwrap();
            rustllama_kernels_cpu::f32_to_f16_bits(x, &mut x_buf.as_mut_slice()[..n]);
        }
        {
            let y_buf = ctx.silu_y.as_mut().unwrap();
            rustllama_kernels_cpu::f32_to_f16_bits(y, &mut y_buf.as_mut_slice()[..n]);
        }
        let stream_raw: *const sk::SyclStream = &ctx.stream;
        let x_ptr = ctx.silu_x.as_ref().unwrap().as_ptr();
        let y_ptr = ctx.silu_y.as_ref().unwrap().as_ptr();
        let out_ptr = ctx.silu_out.as_mut().unwrap().as_mut_ptr();
        let ok =
            // SAFETY: USM allocations on this stream, sized; kernel waits.
            unsafe { sk::silu_mul_usm_raw(&*stream_raw, x_ptr, y_ptr, out_ptr, n as u32) }
                .is_ok();
        if !ok {
            return false;
        }
        let out_buf = ctx.silu_out.as_ref().unwrap();
        for (i, dst) in out.iter_mut().enumerate() {
            *dst = f16::from_bits(out_buf.as_slice()[i]).to_f32();
        }
        true
    })
}

/// USM-resident half-split RoPE applied in-place on `qk`.
/// `qk.len() == n_heads * head_dim`. `head_dim` must be even.
/// `rope_theta` is the base used to build the cached inv-freq
/// table; the table is only recomputed when `(head_dim,
/// rope_theta)` differs from the cached key.
pub fn try_rope_usm_f32(
    qk: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: u32,
    rope_theta: f32,
) -> bool {
    if !usm_attn_enabled() || !gpu_active_for_current_layer() {
        return false;
    }
    if head_dim == 0 || head_dim % 2 != 0 {
        return false;
    }
    let total = n_heads * head_dim;
    if qk.len() != total {
        return false;
    }
    let half = head_dim / 2;
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(ctx) = slot.as_mut() else {
            return false;
        };
        if !ensure_scratch(&ctx.stream, &mut ctx.rope_qk, total) {
            return false;
        }
        // inv_freq table: cached + keyed on (head_dim, rope_theta).
        let key = (head_dim as u32, rope_theta.to_bits());
        let needs_table = ctx.rope_inv_freq_key != Some(key)
            || ctx.rope_inv_freq.as_ref().map_or(true, |b| b.len() < half);
        if needs_table {
            // Drop old before allocating new — keeps peak USM flat.
            ctx.rope_inv_freq = None;
            ctx.rope_inv_freq_key = None;
            let Ok(buf) = sk::SyclSharedBuffer::alloc(&ctx.stream, half) else {
                return false;
            };
            // SAFETY: same '_-to-'static transmute pattern.
            let mut buf = unsafe {
                std::mem::transmute::<
                    sk::SyclSharedBuffer<'_, u16>,
                    sk::SyclSharedBuffer<'static, u16>,
                >(buf)
            };
            for j in 0..half {
                let exponent = (2 * j) as f32 / head_dim as f32;
                let v = 1.0f32 / rope_theta.powf(exponent);
                buf.as_mut_slice()[j] = f16::from_f32(v).to_bits();
            }
            ctx.rope_inv_freq = Some(buf);
            ctx.rope_inv_freq_key = Some(key);
        }
        // Upload qk to USM.
        {
            // F3: SIMD F32 → F16-bits upload (F16C-accelerated).
            let qk_buf = ctx.rope_qk.as_mut().unwrap();
            rustllama_kernels_cpu::f32_to_f16_bits(qk, &mut qk_buf.as_mut_slice()[..total]);
        }
        let stream_raw: *const sk::SyclStream = &ctx.stream;
        let qk_ptr = ctx.rope_qk.as_mut().unwrap().as_mut_ptr();
        let inv_ptr = ctx.rope_inv_freq.as_ref().unwrap().as_ptr();
        let ok =
            // SAFETY: USM allocations on this stream, sized; kernel waits.
            unsafe {
                sk::rope_usm_raw(
                    &*stream_raw,
                    qk_ptr,
                    n_heads as u32,
                    head_dim as u32,
                    pos,
                    inv_ptr,
                )
            }
            .is_ok();
        if !ok {
            return false;
        }
        // Read qk back (in-place semantics).
        let qk_buf = ctx.rope_qk.as_ref().unwrap();
        for (i, dst) in qk.iter_mut().enumerate() {
            *dst = f16::from_bits(qk_buf.as_slice()[i]).to_f32();
        }
        true
    })
}

/// Which packed-layout USM matvec kernel to dispatch against the
/// cached weight bytes. Today only Q8_0Raw and Q4_KRaw are wired;
/// the table grows as more quants get USM matvecs (Q5_K_M, Q6_K
/// are the natural next adds).
#[allow(non_camel_case_types)] // names mirror the GGUF / Dtype naming.
#[derive(Clone, Copy)]
enum PackedMatvecKind {
    /// Q8_0Raw: 34 bytes per 32-weight block, K%32==0.
    Q8_0,
    /// Q4_KRaw: 144 bytes per 256-weight super-block, K%256==0.
    Q4_K,
    /// Q5_KRaw: 176 bytes per 256-weight super-block (Q4_K plus a
    /// 32-byte high-bit `qh` stream), K%256==0.
    Q5_K,
    /// Q6_KRaw: 210 bytes per 256-weight super-block (128 ql + 64
    /// qh + 16 scales + f16 d), K%256==0. Used as the LM head's
    /// storage in Q4_K_M variant GGUFs.
    Q6_K,
    /// F4: IQ4_NLRaw — 18 bytes per 32-weight block, K%32==0.
    /// Simpler-blocked sibling of IQ4_XS; single per-block f16
    /// scale, 16-entry codebook.
    IQ4_NL,
    /// F4: IQ4_XSRaw — 136 bytes per 256-weight super-block,
    /// K%256==0. Eight sub-blocks of 32 weights with 6-bit signed
    /// scales (low 4 bits in `scales_l`, high 2 in `scales_h`).
    /// Widely shipped on Llama 3 / Qwen2 IQ4 variants.
    IQ4_XS,
    /// F4 inference: IQ1_SRaw — 50 bytes per 256-weight super-
    /// block (f16 d + 32-byte qs + 16-byte qh). 11-bit grid index
    /// per chunk into the embedded 2048-entry codebook.
    /// Decompresses to ~1.5 bpw — the format the APEX-nano recipe
    /// uses for expert FFN tensors on big MoE models.
    IQ1_S,
    /// IQ2_XXS — 66 bytes per 256-weight super-block (f16 d +
    /// 64-byte qs). 8-bit grid index into a 256-entry codebook,
    /// 7-bit sign-table index, 4-bit sub-block scale shift.
    /// Decompresses to ~2.1 bpw — one of the most common i-quant
    /// formats for 70B-class models that need to fit in modest
    /// VRAM/RAM budgets.
    IQ2_XXS,
    /// IQ1_M — 56 bytes per 256-weight super-block (32-byte qs +
    /// 16-byte qh + 8-byte scales). Same 2048-entry codebook as
    /// IQ1_S but with per-half sub-block scales (dl1/dl2) +
    /// per-lane delta-sign flip via qh bits. Decompresses to
    /// ~1.75 bpw — a middle ground between IQ1_S (~1.5 bpw) and
    /// the IQ2 family.
    IQ1_M,
    /// IQ2_XS — 74 bytes per 256-weight super-block (f16 d +
    /// 32 × u16 qs + 8-byte scales). 9-bit grid index into the
    /// 512-entry codebook + 7-bit sign-table index per chunk.
    /// Decompresses to ~2.3 bpw — common alongside IQ2_XXS for
    /// 70B-class models.
    IQ2_XS,
    /// IQ2_S — 82 bytes per 256-weight super-block (f16 d +
    /// 32-byte qs_lo + 32-byte signs + 8-byte qh + 8-byte
    /// scales). 10-bit grid index into the 1024-entry codebook;
    /// sign mask stored inline per chunk. Decompresses to ~2.5
    /// bpw — the highest-quality IQ2 variant.
    IQ2_S,
    /// IQ3_XXS — 98 bytes per 256-weight super-block (f16 d +
    /// 64-byte qs_grid + 32-byte qs_sas). 8-bit grid index into
    /// the 256-entry u32 codebook (4 packed u8 grid coords per
    /// entry). Each 8-weight chunk uses two grid lookups (low-4
    /// and high-4 halves). Decompresses to ~3.1 bpw.
    IQ3_XXS,
    /// IQ3_S — 110 bytes per 256-weight super-block (f16 d +
    /// 64-byte qs + 8-byte qh + 32-byte signs + 4-byte scales).
    /// 9-bit grid index into the 512-entry u32 codebook; inline
    /// sign mask; scales packed two-per-byte. Decompresses to
    /// ~3.5 bpw — highest-quality IQ3 variant.
    IQ3_S,
    /// PrismML PTQ1_0 (Bonsai ternary) — 28 bytes per 128-weight
    /// block (24-byte base-3 qs, 2-byte qh, f16 d), K%128==0. The
    /// dtype the Bonsai 2 27B ships every projection/FFN/LM-head
    /// tensor in; decode value = (trit − 1)·d with the 8-bit
    /// multiply-high trit extraction.
    PTQ1_0,
    /// Q4_0Raw: 18 bytes per 32-weight block, K%32==0. Legacy
    /// symmetric 4-bit baseline (pre-K-quant, no min/high-bit).
    Q4_0,
    /// Q5_0Raw: 22 bytes per 32-weight block, K%32==0. Symmetric
    /// 5-bit (Q4_0 plus a 4-byte `qh` high-bit stream).
    Q5_0,
    /// Q4_1Raw: 20 bytes per 32-weight block, K%32==0. Asymmetric
    /// 4-bit (per-block `(d, m)` min/offset).
    Q4_1,
    /// Q5_1Raw: 24 bytes per 32-weight block, K%32==0. Asymmetric
    /// 5-bit sibling of Q4_1 (extra `qh` + a second f16 for `m`).
    Q5_1,
    /// Q2_KRaw: 84 bytes per 256-weight super-block, K%256==0. The
    /// smallest K-quant (2.625 bpw).
    Q2_K,
    /// Q3_KRaw: 110 bytes per 256-weight super-block, K%256==0.
    /// 3-bit K-quant (3.4375 bpw).
    Q3_K,
    /// Q8_KRaw: 292 bytes per 256-weight super-block, K%256==0. The
    /// largest quant (9.125 bpw); usually an activation dtype but we
    /// accept it on the weight side too.
    Q8_K,
    /// PQ2_0Raw (PrismML Bonsai 2-bit): 34 bytes per 128-weight
    /// block, K%128==0. Group-128 sibling of Prism's Q2_0.
    PQ2_0,
    /// MXFP4Raw (OCP Microscaling): 17 bytes per 32-weight block,
    /// K%32==0. Two E2M1 nibbles per byte + one trailing E8M0 scale.
    Mxfp4,
    /// MXFP6Raw (OCP Microscaling, E3M2): 25 bytes per 32-weight
    /// block, K%32==0. LE 6-bit code bitstream + one E8M0 scale.
    Mxfp6,
    /// MXFP8Raw (OCP Microscaling, E4M3): 33 bytes per 32-weight
    /// block, K%32==0. One E4M3 byte per element + one E8M0 scale.
    Mxfp8,
}

/// Number of [`PackedMatvecKind`] variants (sizes the failure-latch
/// array below; `kind as usize` indexes it). New variants MUST be
/// appended at the end of the enum so existing discriminants (= array
/// indices) stay stable.
const PACKED_KIND_COUNT: usize = 25;

/// Consecutive failures before a packed kind stops dispatching.
const PACKED_KIND_DISABLE_AFTER: u32 = 3;

/// While a kind is disabled, re-probe it once every this many skipped
/// dispatch checks. This keeps a *transient* failure (e.g. a USM-alloc
/// blip under momentary memory pressure — say during a big prefill)
/// from disabling a working kernel for the rest of the process: once the
/// pressure eases, the next re-probe succeeds and fully re-enables it. A
/// genuinely-broken kernel (fails every launch) just re-disables on the
/// re-probe, so the amortized failure cost stays ~1 in `RETRY_AFTER`.
const PACKED_KIND_RETRY_AFTER: u32 = 2048;

/// Consecutive-failure counters for the packed USM matvec kernels,
/// indexed by `PackedMatvecKind as usize`. On backends whose driver
/// can't run a packed kernel at all — the OpenCL fallback on Iris Xe
/// throws UR_RESULT_ERROR_OUT_OF_RESOURCES for every launch — each
/// failed dispatch costs a C++ exception, a stderr line, and the CPU
/// fallback work anyway; observed at ~200K failed launches in one
/// short chat session. After [`PACKED_KIND_DISABLE_AFTER`] consecutive
/// failures a kind stops dispatching; a success resets its counter, and
/// [`PACKED_KIND_RETRY_AFTER`] drives a periodic re-probe so a transient
/// failure under memory pressure doesn't permanently disable a working
/// kernel.
static PACKED_KIND_FAILS: [std::sync::atomic::AtomicU32; PACKED_KIND_COUNT] =
    [const { std::sync::atomic::AtomicU32::new(0) }; PACKED_KIND_COUNT];

/// Skipped-dispatch counter per kind while disabled, driving the
/// periodic re-probe (see [`PACKED_KIND_RETRY_AFTER`]).
static PACKED_KIND_SKIPS: [std::sync::atomic::AtomicU32; PACKED_KIND_COUNT] =
    [const { std::sync::atomic::AtomicU32::new(0) }; PACKED_KIND_COUNT];

#[inline]
fn packed_kind_disabled(kind: PackedMatvecKind) -> bool {
    use std::sync::atomic::Ordering;
    // A DEVICE_LOST is terminal — the SYCL context is gone. Disable every
    // packed kind permanently (no re-probe) so we stop hammering the dead
    // device and just use the CPU kernels for the rest of the session.
    if rustllama_kernels_sycl::device_lost() {
        return true;
    }
    if PACKED_KIND_FAILS[kind as usize].load(Ordering::Relaxed) < PACKED_KIND_DISABLE_AFTER {
        return false;
    }
    // Disabled. Count the skip; every RETRY_AFTER skips, drop the fail
    // count just below the threshold so the NEXT dispatch re-probes the
    // kernel. A success there resets to 0 (re-enabled); another failure
    // pushes it back over the threshold (re-disabled) — see
    // `note_packed_kind_result`.
    let skips = PACKED_KIND_SKIPS[kind as usize].fetch_add(1, Ordering::Relaxed) + 1;
    if skips >= PACKED_KIND_RETRY_AFTER {
        PACKED_KIND_SKIPS[kind as usize].store(0, Ordering::Relaxed);
        PACKED_KIND_FAILS[kind as usize]
            .store(PACKED_KIND_DISABLE_AFTER - 1, Ordering::Relaxed);
        return false;
    }
    true
}

/// Record a packed-kernel launch outcome for the failure latch. The
/// launch that trips the latch logs one WARN; afterwards the kind's
/// dispatchers return `false` immediately (CPU fallback) without
/// touching the backend.
fn note_packed_kind_result(kind: PackedMatvecKind, ok: bool) {
    use std::sync::atomic::Ordering;
    let slot = &PACKED_KIND_FAILS[kind as usize];
    if ok {
        slot.store(0, Ordering::Relaxed);
        return;
    }
    let now = slot.fetch_add(1, Ordering::Relaxed) + 1;
    if now == PACKED_KIND_DISABLE_AFTER {
        tracing::warn!(
            kind = kind.name(),
            failures = now,
            "packed GPU matvec kernel disabled for this session after repeated \
             backend failures — this dtype now always uses the CPU kernels"
        );
    }
}

impl PackedMatvecKind {
    /// Kernel-family name for logs.
    fn name(self) -> &'static str {
        match self {
            PackedMatvecKind::Q8_0 => "q8_0",
            PackedMatvecKind::Q4_K => "q4_k",
            PackedMatvecKind::Q5_K => "q5_k",
            PackedMatvecKind::Q6_K => "q6_k",
            PackedMatvecKind::IQ4_NL => "iq4_nl",
            PackedMatvecKind::IQ4_XS => "iq4_xs",
            PackedMatvecKind::IQ1_S => "iq1_s",
            PackedMatvecKind::IQ2_XXS => "iq2_xxs",
            PackedMatvecKind::IQ1_M => "iq1_m",
            PackedMatvecKind::IQ2_XS => "iq2_xs",
            PackedMatvecKind::IQ2_S => "iq2_s",
            PackedMatvecKind::IQ3_XXS => "iq3_xxs",
            PackedMatvecKind::IQ3_S => "iq3_s",
            PackedMatvecKind::PTQ1_0 => "ptq1_0",
            PackedMatvecKind::Q4_0 => "q4_0",
            PackedMatvecKind::Q5_0 => "q5_0",
            PackedMatvecKind::Q4_1 => "q4_1",
            PackedMatvecKind::Q5_1 => "q5_1",
            PackedMatvecKind::Q2_K => "q2_k",
            PackedMatvecKind::Q3_K => "q3_k",
            PackedMatvecKind::Q8_K => "q8_k",
            PackedMatvecKind::PQ2_0 => "pq2_0",
            PackedMatvecKind::Mxfp4 => "mxfp4",
            PackedMatvecKind::Mxfp6 => "mxfp6",
            PackedMatvecKind::Mxfp8 => "mxfp8",
        }
    }

    /// Bytes per row of weights given an inner dim `k`. Used both
    /// for the size of the cache entry and as a sanity check
    /// against the tensor's actual byte length.
    fn row_bytes(self, k: usize) -> usize {
        match self {
            PackedMatvecKind::Q8_0 => (k / 32) * 34,
            PackedMatvecKind::Q4_K => (k / 256) * 144,
            PackedMatvecKind::Q5_K => (k / 256) * 176,
            PackedMatvecKind::Q6_K => (k / 256) * 210,
            PackedMatvecKind::IQ4_NL => (k / 32) * 18,
            PackedMatvecKind::IQ4_XS => (k / 256) * 136,
            PackedMatvecKind::IQ1_S => (k / 256) * 50,
            PackedMatvecKind::IQ2_XXS => (k / 256) * 66,
            PackedMatvecKind::IQ1_M => (k / 256) * 56,
            PackedMatvecKind::IQ2_XS => (k / 256) * 74,
            PackedMatvecKind::IQ2_S => (k / 256) * 82,
            PackedMatvecKind::IQ3_XXS => (k / 256) * 98,
            PackedMatvecKind::IQ3_S => (k / 256) * 110,
            PackedMatvecKind::PTQ1_0 => (k / 128) * 28,
            PackedMatvecKind::Q4_0 => (k / 32) * 18,
            PackedMatvecKind::Q5_0 => (k / 32) * 22,
            PackedMatvecKind::Q4_1 => (k / 32) * 20,
            PackedMatvecKind::Q5_1 => (k / 32) * 24,
            PackedMatvecKind::Q2_K => (k / 256) * 84,
            PackedMatvecKind::Q3_K => (k / 256) * 110,
            PackedMatvecKind::Q8_K => (k / 256) * 292,
            PackedMatvecKind::PQ2_0 => (k / 128) * 34,
            PackedMatvecKind::Mxfp4 => (k / 32) * 17,
            PackedMatvecKind::Mxfp6 => (k / 32) * 25,
            PackedMatvecKind::Mxfp8 => (k / 32) * 33,
        }
    }

    /// K alignment requirement for the format's block size.
    fn k_alignment(self) -> usize {
        match self {
            PackedMatvecKind::Q8_0
            | PackedMatvecKind::IQ4_NL
            | PackedMatvecKind::Q4_0
            | PackedMatvecKind::Q5_0
            | PackedMatvecKind::Q4_1
            | PackedMatvecKind::Q5_1
            | PackedMatvecKind::Mxfp4
            | PackedMatvecKind::Mxfp6
            | PackedMatvecKind::Mxfp8 => 32,
            PackedMatvecKind::Q4_K
            | PackedMatvecKind::Q5_K
            | PackedMatvecKind::Q6_K
            | PackedMatvecKind::IQ4_XS
            | PackedMatvecKind::IQ1_S
            | PackedMatvecKind::IQ2_XXS
            | PackedMatvecKind::IQ1_M
            | PackedMatvecKind::IQ2_XS
            | PackedMatvecKind::IQ2_S
            | PackedMatvecKind::IQ3_XXS
            | PackedMatvecKind::IQ3_S
            | PackedMatvecKind::Q2_K
            | PackedMatvecKind::Q3_K
            | PackedMatvecKind::Q8_K => 256,
            PackedMatvecKind::PTQ1_0 | PackedMatvecKind::PQ2_0 => 128,
        }
    }
}

/// USM-resident weight matvec dispatch with per-dtype routing. The
/// v1-locked quants (Q8_0Raw, Q4_KRaw, Q5_KRaw) are wired to USM
/// kernels; other quant formats return `false` so the caller falls
/// back to the CPU `matvec_tensor`. Adding a new packed quant is a
/// `PackedMatvecKind` variant + a match arm in the kernel-launch
/// block — the call-site fallback chain in `llama_arch.rs` stays
/// unchanged.
///
/// Q4_K_M is the v1 default quant for almost every coding-model GGUF
/// (Qwen2.5-Coder, DeepSeek-Coder, Mistral, Llama-3.x), so when
/// `RUSTLLAMA_USM_ATTN=1` and a Q4_K_M model is loaded, every
/// projection matvec + LM head routes through this hook for the
/// duration of generation.
///
/// The weight buffer is uploaded to USM once per `(host-pointer,
/// byte-len)` pair and cached for the lifetime of the USM context.
/// Subsequent calls against the same weight tensor pay only the
/// small `x` upload + kernel dispatch + small `out` readback.
/// H4 infrastructure: env-var gate for MoE gate+up batched matvec.
/// `RUSTLLAMA_MOE_GATE_UP_FUSED=1` opts into the fused per-expert
/// matvec dispatch (when the per-dtype `matvec_gate_up_packed_*_usm`
/// SYCL kernels are wired). Default off until kernels land. Reading
/// this env var here lets future kernel-implementation work flip a
/// single switch without re-plumbing the engine.
pub fn moe_gate_up_fused_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Default OFF. The default-ON experiment (opt Tranche 3.3;
        // the table covers every shipping expert dtype and the
        // unfused path pays ~320 extra dispatches per decoded token
        // on a 256-expert top-8 model) produced OUTRIGHT GARBAGE
        // logits on the first hardware that actually executed the
        // fused kernels: Iris Xe on the OpenCL fallback backend,
        // Qwen3.6-35B IQ2_XXS experts, 2026-09-06 — greedy decode
        // emitted mixed-script token soup with fused ON and correct
        // text with fused OFF, all else equal. The failure-latch
        // can't catch this class (the kernels return Ok). Do not
        // re-enable by default until the fused kernels have per-
        // dtype GPU-vs-CPU parity tests that run on the target
        // backend. `RUSTLLAMA_MOE_GATE_UP_FUSED=1` opts in for A/B.
        std::env::var("RUSTLLAMA_MOE_GATE_UP_FUSED")
            .ok()
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(false)
    })
}

/// H6 infrastructure: env-var gate for full attn-output-proj +
/// residual + post-attn-norm fusion. `RUSTLLAMA_FUSED_OUT_PROJ_NORM=1`
/// opts into a new per-dtype `matvec_add_rmsnorm_packed_*_usm` kernel
/// (when wired). The two-op `try_add_rmsnorm_usm_f32` fusion (Tier F)
/// is already on by default; this gate covers the additional matvec
/// fold.
pub fn fused_out_proj_norm_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RUSTLLAMA_FUSED_OUT_PROJ_NORM")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// H3 infrastructure: env-var gate for load-time QKV-concat fusion.
/// `RUSTLLAMA_QKV_FUSED=1` opts in to the load-path concatenation of
/// separate `w_q` / `w_k` / `w_v` tensors into a fused
/// `[d_q + 2*d_kv, d_model]` weight that the forward path dispatches
/// via a single matvec call. Default off until the load-path concat
/// + per-block dispatch wiring lands. Hybrid Qwen3.5-MoE DeltaNet
/// already uses fused `attn_qkv` natively and is H3-compliant.
pub fn qkv_fused_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RUSTLLAMA_QKV_FUSED")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// H8 (deferred): F16 input mixed-precision matvec. Today all GPU
/// matvecs read activations as F32 USM. F16 inputs with F32
/// accumulators would cut activation-memory bandwidth ~2× — a
/// material win on Iris Xe's ~8 GB/s memory budget. Implementation
/// requires a new SYCL kernel variant per dtype (Q8_0, Q4_K, Q5_K,
/// Q6_K, IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS — ~12 variants)
/// reading `*const u16` activations + `bits_to_f32()` conversion in
/// the inner loop. F16 has ~6.5e-5 minimum subnormal; near-zero
/// activations risk underflow so a parity bound + env-var-default-off
/// gate is required before promoting. `RUSTLLAMA_MIXED_PRECISION_MATVEC`
/// is reserved as the activation env var; reading it returns `false`
/// until the kernels land.
pub fn mixed_precision_matvec_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RUSTLLAMA_MIXED_PRECISION_MATVEC")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

pub fn try_matvec_tensor_usm_f32(
    w: &Tensor,
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) -> bool {
    // Phase 4 cross-GPU routing (INERT unless a heat plan is installed — one
    // atomic load; `false` on the single-GPU/CPU box → skipped entirely and the
    // path below is byte-identical). Only a NON-default CUDA device is handled
    // here; CPU tensors are already caught by `tensor_forced_to_cpu` (the plan
    // is folded into it above), and default-device-CUDA / active-SYCL targets
    // fall through to the existing blocks. A routed miss also falls through.
    if multi_gpu_plan_active() {
        if let Some(plan) = multi_gpu_plan() {
            if let Some(t) = plan.device_for(&w.name, current_layer_idx()) {
                if t.backend == GpuBackend::Cuda
                    && t.device_index != CUDA_DEFAULT_DEVICE
                    && m != 0
                    && k != 0
                    && x.len() == k
                    && out.len() == m
                    && matvec_above_min_flops(2u64 * m as u64 * k as u64)
                {
                    if let Some(ck_kind) = dtype_to_cuda_kind(w.dtype) {
                        let wb = as_bytes(w);
                        if try_matvec_packed_cuda_dev(
                            t.device_index,
                            ck_kind,
                            wb.as_ptr() as usize,
                            wb,
                            x,
                            out,
                            m,
                            k,
                        ) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    // Native CUDA backend FIRST (before the SYCL-specific gates below —
    // `usm_attn_enabled` is false on a SYCL-less NVIDIA host, so CUDA
    // must not sit behind it). Universal gates only: placement override,
    // the n_gpu_layers layer cutoff, shape, and the tiny-matvec floor.
    // Inert on non-NVIDIA hosts (`cuda_active` == false). A miss falls
    // through to the SYCL/CPU ladder.
    if cuda_active()
        && !tensor_forced_to_cpu(&w.name)
        && current_layer_idx() < n_gpu_layers()
        && m != 0
        && k != 0
        && x.len() == k
        && out.len() == m
        && matvec_above_min_flops(2u64 * m as u64 * k as u64)
    {
        if let Some(ck_kind) = dtype_to_cuda_kind(w.dtype) {
            let wb = as_bytes(w);
            if try_matvec_packed_cuda(ck_kind, wb.as_ptr() as usize, wb, x, out, m, k) {
                return true;
            }
        }
    }
    // Native MLX (Apple Metal) backend — the 4th GPU tier. Same universal
    // gates as the CUDA arm above; inert off Apple Silicon (`mlx_active`
    // == false → skipped entirely, path byte-identical). On a Mac this is
    // chosen in place of CUDA/SYCL (mutually exclusive in practice). A miss
    // falls through to the SYCL/CPU ladder.
    if mlx_active()
        && !tensor_forced_to_cpu(&w.name)
        && current_layer_idx() < n_gpu_layers()
        && m != 0
        && k != 0
        && x.len() == k
        && out.len() == m
        && matvec_above_min_flops(2u64 * m as u64 * k as u64)
    {
        if let Some(mk_kind) = dtype_to_mlx_kind(w.dtype) {
            let wb = as_bytes(w);
            if try_matvec_packed_mlx(mk_kind, wb.as_ptr() as usize, wb, x, out, m, k) {
                return true;
            }
        }
    }
    if !usm_attn_enabled() {
        log_matvec_skip_once("usm_attn_enabled = false");
        return false;
    }
    if tensor_forced_to_cpu(&w.name) {
        // Per-tensor override: user pinned this weight to CPU via
        // `[inference].placement.overrides`. Skip USM, let the
        // caller's CPU matvec fire.
        return false;
    }
    if !gpu_active_for_current_layer() {
        // CPU-resident layer per `[inference].n_gpu_layers` cutoff —
        // skip USM matvec, let the caller fall through to CPU. No
        // log: this is the configured-behavior path, not a failure.
        return false;
    }
    if m == 0 || k == 0 || x.len() != k || out.len() != m {
        log_matvec_skip_once("shape check failed");
        return false;
    }
    // G5: size-based gate. Skip GPU dispatch for tiny matvecs where
    // Level Zero launch + USM marshaling overhead dominates the
    // compute. FLOPs ≈ 2 * M * K (one FMA per output × per weight).
    if !matvec_above_min_flops(2u64 * m as u64 * k as u64) {
        return false;
    }
    let kind = match w.dtype {
        Dtype::Q8_0Raw => PackedMatvecKind::Q8_0,
        Dtype::Q4_KRaw => PackedMatvecKind::Q4_K,
        Dtype::Q5_KRaw => PackedMatvecKind::Q5_K,
        Dtype::Q6_KRaw => PackedMatvecKind::Q6_K,
        Dtype::IQ4_NLRaw => PackedMatvecKind::IQ4_NL,
        Dtype::IQ4_XSRaw => PackedMatvecKind::IQ4_XS,
        Dtype::IQ1_SRaw => PackedMatvecKind::IQ1_S,
        Dtype::IQ2_XXSRaw => PackedMatvecKind::IQ2_XXS,
        Dtype::IQ1_MRaw => PackedMatvecKind::IQ1_M,
        Dtype::IQ2_XSRaw => PackedMatvecKind::IQ2_XS,
        Dtype::IQ2_SRaw => PackedMatvecKind::IQ2_S,
        Dtype::IQ3_XXSRaw => PackedMatvecKind::IQ3_XXS,
        Dtype::IQ3_SRaw => PackedMatvecKind::IQ3_S,
        Dtype::PTQ1_0Raw => PackedMatvecKind::PTQ1_0,
        Dtype::Q4_0Raw => PackedMatvecKind::Q4_0,
        Dtype::Q5_0Raw => PackedMatvecKind::Q5_0,
        Dtype::Q4_1Raw => PackedMatvecKind::Q4_1,
        Dtype::Q5_1Raw => PackedMatvecKind::Q5_1,
        Dtype::Q2_KRaw => PackedMatvecKind::Q2_K,
        Dtype::Q3_KRaw => PackedMatvecKind::Q3_K,
        Dtype::Q8_KRaw => PackedMatvecKind::Q8_K,
        Dtype::PQ2_0Raw => PackedMatvecKind::PQ2_0,
        Dtype::Mxfp4Raw => PackedMatvecKind::Mxfp4,
        Dtype::Mxfp6Raw => PackedMatvecKind::Mxfp6,
        Dtype::Mxfp8Raw => PackedMatvecKind::Mxfp8,
        _ => {
            // Log every unique dtype we see fall through to CPU
            // so we get a complete dtype-distribution map of the
            // model's weights, not just the first one encountered.
            log_unmatched_dtype(w.dtype);
            return false;
        }
    };
    if packed_kind_disabled(kind) {
        // Failure latch tripped for this kind (see PACKED_KIND_FAILS)
        // — skip the backend entirely, caller runs the CPU kernel.
        return false;
    }
    if k % kind.k_alignment() != 0 {
        log_matvec_skip_once("K alignment check failed");
        return false;
    }
    let w_bytes = as_bytes(w);
    let expected_bytes = m * kind.row_bytes(k);
    if w_bytes.len() < expected_bytes {
        log_matvec_skip_once("w_bytes too small");
        return false;
    }
    // Fast path: weight storage is already `Storage::SyclUsm` (the
    // model was loaded with `preload_weights_to_usm`). Skip the
    // host→device upload entirely; the kernel reads the USM pointer
    // directly. `weight_key` falls back to the host-pointer key
    // when storage is `CpuOwned`, preserving the cached-upload
    // behavior for tensors that didn't preload.
    let preloaded_usm: Option<*const u8> = w.storage.sycl_usm_ptr();
    let weight_key = w_bytes.as_ptr() as usize;
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(ctx) = slot.as_mut() else {
            log_matvec_skip_once("USM_ATTN slot is None (prepare_usm_context failed)");
            return false;
        };
        // Ensure activation + output scratch.
        if ensure_scratch_f32_grow(&ctx.stream, &mut ctx.matvec_x_f32, k).is_err() {
            log_matvec_skip_once("ensure_scratch_f32 (x) failed");
            return false;
        }
        if !ensure_scratch_f32(&ctx.stream, &mut ctx.matvec_out_f32, m) {
            log_matvec_skip_once("ensure_scratch_f32 (out) failed");
            return false;
        }
        // Weight: cached USM upload keyed on host pointer. First
        // call uploads + memcpy's the raw GGUF bytes; subsequent
        // calls just hand the cached pointer to the kernel.
        // Preloaded SyclUsm weights skip this branch entirely — the
        // pointer is already on the GPU.
        if preloaded_usm.is_none()
            && !ctx.cache_packed_weight(
                weight_key,
                expected_bytes,
                w_bytes,
                w.storage.mmap_borrowed_ptr_len().is_some(),
            )
        {
            // Over the USM weight-cache budget (or alloc failed) — fall
            // through to CPU matvec, which reads the file-backed bytes
            // in place (no USM copy → no iGPU double-residency).
            return false;
        }
        // Upload the activation UNCONDITIONALLY. A `(ptr, len)`
        // skip-if-same-slab dedup used to live here (and in the
        // fused + batched twins); it saved an ~8 KB host→USM memcpy
        // when Q/K/V read the same `h_norm` row, but the key says
        // nothing about CONTENT — engine scratch lives at stable
        // addresses and is refilled every layer and every request,
        // so any same-pointer-new-content call sequence silently fed
        // the kernel stale activations. This exact class produced
        // "silent gibberish" at the H6 site once before, and the
        // 2026-09-07 parity harness proved every kernel bit-clean in
        // isolation while end-to-end GPU output was garbage — the
        // corruption lived HERE. The unconditional copy costs ~2 µs
        // on shared LPDDR; the bug class is not worth it.
        ctx.matvec_x_f32.as_mut().unwrap().as_mut_slice()[..k].copy_from_slice(x);
        let stream_raw: *const sk::SyclStream = &ctx.stream;
        // Pick the weight pointer: preloaded USM if storage is
        // already SyclUsm; otherwise the cached-upload's pointer.
        let w_ptr = match preloaded_usm {
            Some(p) => p,
            None => ctx.packed_weight_cache[&weight_key].as_ptr(),
        };
        let x_ptr = ctx.matvec_x_f32.as_ref().unwrap().as_ptr();
        let out_ptr = ctx.matvec_out_f32.as_mut().unwrap().as_mut_ptr();
        let ok = match kind {
            // SAFETY: all USM pointers on `stream`; sizes validated
            // above; the kernel `.wait()`s before returning.
            // `lws = 0` selects the kernel TU's hand-picked default
            // (64). The autotuner cache returns the tuned value per
            // `(kernel, M, K)` if present; otherwise 0 → default.
            PackedMatvecKind::Q8_0 => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_Q8_0_PACKED_USM, m, k);
                sk::matvec_q8_0_packed_f32_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::Q4_K => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_Q4K_PACKED_USM, m, k);
                sk::matvec_q4_k_packed_f32_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::Q5_K => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_Q5K_PACKED_USM, m, k);
                sk::matvec_q5_k_packed_f32_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::Q6_K => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_Q6K_PACKED_USM, m, k);
                sk::matvec_q6_k_packed_f32_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ4_NL => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ4_NL_PACKED_USM, m, k);
                sk::matvec_iq4_nl_packed_f32_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ4_XS => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ4_XS_PACKED_USM, m, k);
                sk::matvec_iq4_xs_packed_f32_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ1_S => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ1_S_PACKED_USM, m, k);
                sk::matvec_iq1_s_packed_f32_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ2_XXS => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ2_XXS_PACKED_USM, m, k);
                sk::matvec_iq2_xxs_packed_f32_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ1_M => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ1_M_PACKED_USM, m, k);
                sk::matvec_iq1_m_packed_f32_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ2_XS => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ2_XS_PACKED_USM, m, k);
                sk::matvec_iq2_xs_packed_f32_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ2_S => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ2_S_PACKED_USM, m, k);
                sk::matvec_iq2_s_packed_f32_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ3_XXS => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ3_XXS_PACKED_USM, m, k);
                sk::matvec_iq3_xxs_packed_f32_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ3_S => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ3_S_PACKED_USM, m, k);
                sk::matvec_iq3_s_packed_f32_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::PTQ1_0 => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_PTQ1_0_PACKED_USM, m, k);
                sk::matvec_ptq1_0_packed_f32_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    lws,
                )
            }
            .is_ok(),
            // The 8 CPU-parity packed formats (legacy Q4_0/Q5_0/Q4_1/
            // Q5_1, K-quant Q2_K/Q3_K/Q8_K, PrismML PQ2_0) route through
            // the DIRECT single-row call with `lws = 0` — the
            // kernel-TU-picked default work-group size. These have no
            // autotuner `KERNEL_*` const, so we skip `tuned_lws_for`
            // (no new tuner consts needed); the batched twin below has no
            // arm for them, so PREFILL correctly falls to the CPU batched
            // path while DECODE (this site) uses the GPU.
            PackedMatvecKind::Q4_0 => unsafe {
                sk::matvec_q4_0_packed_f32_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr, m as u32, k as u32, 0,
                )
            }
            .is_ok(),
            PackedMatvecKind::Q5_0 => unsafe {
                sk::matvec_q5_0_packed_f32_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr, m as u32, k as u32, 0,
                )
            }
            .is_ok(),
            PackedMatvecKind::Q4_1 => unsafe {
                sk::matvec_q4_1_packed_f32_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr, m as u32, k as u32, 0,
                )
            }
            .is_ok(),
            PackedMatvecKind::Q5_1 => unsafe {
                sk::matvec_q5_1_packed_f32_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr, m as u32, k as u32, 0,
                )
            }
            .is_ok(),
            PackedMatvecKind::Q2_K => unsafe {
                sk::matvec_q2_k_packed_f32_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr, m as u32, k as u32, 0,
                )
            }
            .is_ok(),
            PackedMatvecKind::Q3_K => unsafe {
                sk::matvec_q3_k_packed_f32_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr, m as u32, k as u32, 0,
                )
            }
            .is_ok(),
            PackedMatvecKind::Q8_K => unsafe {
                sk::matvec_q8_k_packed_f32_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr, m as u32, k as u32, 0,
                )
            }
            .is_ok(),
            PackedMatvecKind::PQ2_0 => unsafe {
                sk::matvec_pq2_0_packed_f32_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr, m as u32, k as u32, 0,
                )
            }
            .is_ok(),
            PackedMatvecKind::Mxfp4 => unsafe {
                sk::matvec_mxfp4_packed_f32_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr, m as u32, k as u32, 0,
                )
            }
            .is_ok(),
            PackedMatvecKind::Mxfp6 => unsafe {
                sk::matvec_mxfp6_packed_f32_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr, m as u32, k as u32, 0,
                )
            }
            .is_ok(),
            PackedMatvecKind::Mxfp8 => unsafe {
                sk::matvec_mxfp8_packed_f32_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr, m as u32, k as u32, 0,
                )
            }
            .is_ok(),
        };
        note_packed_kind_result(kind, ok);
        if !ok {
            log_matvec_skip_once("kernel call returned Err");
            return false;
        }
        // Read out back to host.
        let out_buf = ctx.matvec_out_f32.as_ref().unwrap();
        out.copy_from_slice(&out_buf.as_slice()[..m]);
        true
    })
}

/// H4: Fused gate + up matvec — one USM dispatch computes both
/// `gate_out = gate_w @ x` and `up_out = up_w @ x` in a single
/// kernel that loads each `x[d]` once and accumulates into both
/// dot products. Saves half the activation-cache pressure plus one
/// kernel launch per FFN expert.
///
/// Returns `true` on successful GPU dispatch. Returns `false` when
/// the dtype isn't covered yet (caller falls back to two separate
/// `try_matvec_tensor_usm_f32` calls or the CPU path), the USM
/// context is unavailable, the layer is CPU-resident, or any
/// shape/alignment check fails.
///
/// Currently covers Q4_K and Q8_0; remaining packed dtypes are
/// handled by the dual-matvec fallback path until their fused
/// kernels land. Both weight tensors must share the same dtype and
/// the same `(m, k)` shape.
pub fn try_matvec_tensor_gate_up_fused_usm_f32(
    w_gate: &Tensor,
    w_up: &Tensor,
    x: &[f32],
    gate_out: &mut [f32],
    up_out: &mut [f32],
    m: usize,
    k: usize,
) -> bool {
    if !moe_gate_up_fused_enabled() {
        return false;
    }
    if !usm_attn_enabled() {
        return false;
    }
    if tensor_forced_to_cpu(&w_gate.name) || tensor_forced_to_cpu(&w_up.name) {
        return false;
    }
    if !gpu_active_for_current_layer() {
        return false;
    }
    if m == 0 || k == 0
        || x.len() != k
        || gate_out.len() != m
        || up_out.len() != m
    {
        return false;
    }
    if w_gate.dtype != w_up.dtype {
        return false;
    }
    // FLOPs ≈ 2 ops × 2 * M * K (two FMAs per output × per weight).
    if !matvec_above_min_flops(4u64 * m as u64 * k as u64) {
        return false;
    }
    // Only dtypes with a fused kernel land here. The rest return
    // false (caller dispatches via two `try_matvec_tensor_usm_f32`).
    let kind = match w_gate.dtype {
        Dtype::Q4_KRaw => PackedMatvecKind::Q4_K,
        Dtype::Q8_0Raw => PackedMatvecKind::Q8_0,
        Dtype::Q5_KRaw => PackedMatvecKind::Q5_K,
        Dtype::Q6_KRaw => PackedMatvecKind::Q6_K,
        Dtype::IQ4_NLRaw => PackedMatvecKind::IQ4_NL,
        Dtype::IQ4_XSRaw => PackedMatvecKind::IQ4_XS,
        Dtype::IQ1_SRaw => PackedMatvecKind::IQ1_S,
        Dtype::IQ2_XXSRaw => PackedMatvecKind::IQ2_XXS,
        Dtype::IQ1_MRaw => PackedMatvecKind::IQ1_M,
        Dtype::IQ2_XSRaw => PackedMatvecKind::IQ2_XS,
        Dtype::IQ2_SRaw => PackedMatvecKind::IQ2_S,
        Dtype::IQ3_XXSRaw => PackedMatvecKind::IQ3_XXS,
        Dtype::IQ3_SRaw => PackedMatvecKind::IQ3_S,
        Dtype::PTQ1_0Raw => PackedMatvecKind::PTQ1_0,
        // Mapped for kind-consistency with the single-row gate; there is
        // no fused gate+up kernel for these 8 formats, so they land on
        // the `_ => return false` launch arm below and the caller falls
        // back to two separate `try_matvec_tensor_usm_f32` calls.
        Dtype::Q4_0Raw => PackedMatvecKind::Q4_0,
        Dtype::Q5_0Raw => PackedMatvecKind::Q5_0,
        Dtype::Q4_1Raw => PackedMatvecKind::Q4_1,
        Dtype::Q5_1Raw => PackedMatvecKind::Q5_1,
        Dtype::Q2_KRaw => PackedMatvecKind::Q2_K,
        Dtype::Q3_KRaw => PackedMatvecKind::Q3_K,
        Dtype::Q8_KRaw => PackedMatvecKind::Q8_K,
        Dtype::PQ2_0Raw => PackedMatvecKind::PQ2_0,
        _ => return false,
    };
    if packed_kind_disabled(kind) {
        return false;
    }
    if k % kind.k_alignment() != 0 {
        return false;
    }
    let g_bytes = as_bytes(w_gate);
    let u_bytes = as_bytes(w_up);
    let expected = m * kind.row_bytes(k);
    if g_bytes.len() < expected || u_bytes.len() < expected {
        return false;
    }
    let g_preloaded: Option<*const u8> = w_gate.storage.sycl_usm_ptr();
    let u_preloaded: Option<*const u8> = w_up.storage.sycl_usm_ptr();
    let g_key = g_bytes.as_ptr() as usize;
    let u_key = u_bytes.as_ptr() as usize;
    // File-backed (GGUF mmap) weights are safe to discard after the USM
    // copy; heap/USM storage → None → left intact.
    let g_file_backed = w_gate.storage.mmap_borrowed_ptr_len().is_some();
    let u_file_backed = w_up.storage.mmap_borrowed_ptr_len().is_some();
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(ctx) = slot.as_mut() else {
            return false;
        };
        if ensure_scratch_f32_grow(&ctx.stream, &mut ctx.matvec_x_f32, k).is_err() {
            return false;
        }
        if !ensure_scratch_f32(&ctx.stream, &mut ctx.matvec_out_f32, m) {
            return false;
        }
        if !ensure_scratch_f32(&ctx.stream, &mut ctx.matvec_out2_f32, m) {
            return false;
        }
        // Cache each weight pointer separately. The same gate
        // tensor is read on every token at this layer; same for up.
        for (key, bytes, preloaded, fb) in [
            (g_key, g_bytes, g_preloaded, g_file_backed),
            (u_key, u_bytes, u_preloaded, u_file_backed),
        ] {
            if preloaded.is_some() {
                continue;
            }
            // Over budget on either gate/up weight → CPU matvec for both.
            if !ctx.cache_packed_weight(key, expected, bytes, fb) {
                return false;
            }
        }
        // Unconditional activation upload — the (ptr, len) dedup that
        // lived here fed kernels stale activations; see the comment
        // in `try_matvec_tensor_usm_f32`.
        ctx.matvec_x_f32.as_mut().unwrap().as_mut_slice()[..k].copy_from_slice(x);
        let stream_raw: *const sk::SyclStream = &ctx.stream;
        let g_ptr = match g_preloaded {
            Some(p) => p,
            None => ctx.packed_weight_cache[&g_key].as_ptr(),
        };
        let u_ptr = match u_preloaded {
            Some(p) => p,
            None => ctx.packed_weight_cache[&u_key].as_ptr(),
        };
        let x_ptr = ctx.matvec_x_f32.as_ref().unwrap().as_ptr();
        let gate_ptr = ctx.matvec_out_f32.as_mut().unwrap().as_mut_ptr();
        let up_ptr = ctx.matvec_out2_f32.as_mut().unwrap().as_mut_ptr();
        let ok = match kind {
            PackedMatvecKind::Q4_K => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_Q4K_PACKED_USM, m, k);
                sk::matvec_q4_k_gate_up_fused_usm_raw(
                    &*stream_raw,
                    g_ptr, u_ptr,
                    x_ptr,
                    gate_ptr, up_ptr,
                    m as u32, k as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::Q8_0 => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_Q8_0_PACKED_USM, m, k);
                sk::matvec_q8_0_gate_up_fused_usm_raw(
                    &*stream_raw,
                    g_ptr, u_ptr,
                    x_ptr,
                    gate_ptr, up_ptr,
                    m as u32, k as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::Q5_K => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_Q5K_PACKED_USM, m, k);
                sk::matvec_q5_k_gate_up_fused_usm_raw(
                    &*stream_raw,
                    g_ptr, u_ptr,
                    x_ptr,
                    gate_ptr, up_ptr,
                    m as u32, k as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::Q6_K => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_Q6K_PACKED_USM, m, k);
                sk::matvec_q6_k_gate_up_fused_usm_raw(
                    &*stream_raw,
                    g_ptr, u_ptr,
                    x_ptr,
                    gate_ptr, up_ptr,
                    m as u32, k as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ4_NL => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ4_NL_PACKED_USM, m, k);
                sk::matvec_iq4_nl_gate_up_fused_usm_raw(
                    &*stream_raw,
                    g_ptr, u_ptr,
                    x_ptr,
                    gate_ptr, up_ptr,
                    m as u32, k as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ4_XS => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ4_XS_PACKED_USM, m, k);
                sk::matvec_iq4_xs_gate_up_fused_usm_raw(
                    &*stream_raw,
                    g_ptr, u_ptr,
                    x_ptr,
                    gate_ptr, up_ptr,
                    m as u32, k as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ1_S => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ1_S_PACKED_USM, m, k);
                sk::matvec_iq1_s_gate_up_fused_usm_raw(
                    &*stream_raw,
                    g_ptr, u_ptr,
                    x_ptr,
                    gate_ptr, up_ptr,
                    m as u32, k as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ2_XXS => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ2_XXS_PACKED_USM, m, k);
                sk::matvec_iq2_xxs_gate_up_fused_usm_raw(
                    &*stream_raw,
                    g_ptr, u_ptr,
                    x_ptr,
                    gate_ptr, up_ptr,
                    m as u32, k as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ1_M => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ1_M_PACKED_USM, m, k);
                sk::matvec_iq1_m_gate_up_fused_usm_raw(
                    &*stream_raw,
                    g_ptr, u_ptr,
                    x_ptr,
                    gate_ptr, up_ptr,
                    m as u32, k as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ2_XS => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ2_XS_PACKED_USM, m, k);
                sk::matvec_iq2_xs_gate_up_fused_usm_raw(
                    &*stream_raw,
                    g_ptr, u_ptr,
                    x_ptr,
                    gate_ptr, up_ptr,
                    m as u32, k as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ2_S => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ2_S_PACKED_USM, m, k);
                sk::matvec_iq2_s_gate_up_fused_usm_raw(
                    &*stream_raw,
                    g_ptr, u_ptr,
                    x_ptr,
                    gate_ptr, up_ptr,
                    m as u32, k as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ3_XXS => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ3_XXS_PACKED_USM, m, k);
                sk::matvec_iq3_xxs_gate_up_fused_usm_raw(
                    &*stream_raw,
                    g_ptr, u_ptr,
                    x_ptr,
                    gate_ptr, up_ptr,
                    m as u32, k as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ3_S => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ3_S_PACKED_USM, m, k);
                sk::matvec_iq3_s_gate_up_fused_usm_raw(
                    &*stream_raw,
                    g_ptr, u_ptr,
                    x_ptr,
                    gate_ptr, up_ptr,
                    m as u32, k as u32, lws,
                )
            }
            .is_ok(),
            // PTQ1_0 (Bonsai ternary) fused gate+up — mirrors the Q8_0/
            // Q4_K arms above (same guards, tuner-const LWS source). The
            // remaining 8 packed formats have no fused gate+up kernel and
            // fall to the `_ => return false` arm (caller runs two
            // separate matvecs).
            PackedMatvecKind::PTQ1_0 => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_PTQ1_0_PACKED_USM, m, k);
                sk::matvec_ptq1_0_gate_up_fused_usm_raw(
                    &*stream_raw,
                    g_ptr, u_ptr,
                    x_ptr,
                    gate_ptr, up_ptr,
                    m as u32, k as u32, lws,
                )
            }
            .is_ok(),
            _ => return false,
        };
        note_packed_kind_result(kind, ok);
        if !ok {
            return false;
        }
        let g_buf = ctx.matvec_out_f32.as_ref().unwrap();
        let u_buf = ctx.matvec_out2_f32.as_ref().unwrap();
        gate_out.copy_from_slice(&g_buf.as_slice()[..m]);
        up_out.copy_from_slice(&u_buf.as_slice()[..m]);
        true
    })
}

/// H6: Fused output-projection matvec + residual-add + rmsnorm — one
/// GPU dispatch does `y_norm = rmsnorm(hidden + W·attn) * w_norm` and
/// writes the residual sum back into `hidden`. Replaces the
/// (`matvec_tensor_dispatch(W_o)` → `try_add_rmsnorm_usm_f32`) pair at
/// the post-attention norm site.
///
/// `attn` is the attention output (length `k` == d_q); `W` is the
/// output-projection weight (`[m × k]` packed, m == d_model); `hidden`
/// is the running residual stream (read + written); `w_norm` is the
/// ffn_norm scale; `y_norm` receives the normalized output (next
/// block's gate/up input).
///
/// Gated behind `RUSTLLAMA_FUSED_OUT_PROJ_NORM=1` (default off).
///
/// Implementation (post-bisect rewrite, 2026-05-29): chains two
/// proven kernels instead of a single fused one. The original
/// single-workgroup SLM-staged kernel
/// (`matvec_X_add_rmsnorm_usm`) produced gibberish on
/// qwen2.5-coder-1.5B Q4_K_M — its 1-workgroup topology was the only
/// meaningful divergence from the unfused matvec, and removing it
/// fixed the regression. The dispatcher now does:
///   1. Existing multi-workgroup `matvec_packed_X_f32_usm` (identical
///      kernel as the unfused path → bit-equal matvec result).
///   2. New `add_rmsnorm_f32_usm` (multi-workgroup, F32 precision —
///      skips the F32→F16→F32 round-trip that the legacy
///      `try_add_rmsnorm_usm_f32` requires).
/// One extra kernel launch vs. the legacy single-kernel design, but
/// the matvec topology is now proven-good and F32 throughout.
/// Returns `false` when the gate is off, the dtype has no kernel, the
/// layer is CPU-resident, or any shape check fails — caller falls
/// through to the legacy unfused matvec + F16 add_rmsnorm pair.
#[allow(clippy::too_many_arguments)]
pub fn try_matvec_out_proj_add_rmsnorm_usm_f32(
    w: &Tensor,
    attn: &[f32],
    hidden: &mut [f32],
    w_norm: &[f32],
    y_norm: &mut [f32],
    m: usize,
    k: usize,
    eps: f32,
) -> bool {
    if !fused_out_proj_norm_enabled() {
        return false;
    }
    if !usm_attn_enabled() {
        return false;
    }
    if tensor_forced_to_cpu(&w.name) {
        return false;
    }
    if !gpu_active_for_current_layer() {
        return false;
    }
    if m == 0 || k == 0
        || attn.len() != k
        || hidden.len() != m
        || w_norm.len() < m
        || y_norm.len() != m
    {
        return false;
    }
    // rmsnorm reduces over a single row of length m — the fused
    // kernel uses one workgroup, so very large m would overflow SLM.
    // Cap at 16K (64 KB SLM); above that, fall back to the unfused
    // path (the SLM-staged design doesn't apply).
    if m > 16384 {
        return false;
    }
    let kind = match w.dtype {
        Dtype::Q4_KRaw => PackedMatvecKind::Q4_K,
        Dtype::Q8_0Raw => PackedMatvecKind::Q8_0,
        Dtype::Q5_KRaw => PackedMatvecKind::Q5_K,
        Dtype::Q6_KRaw => PackedMatvecKind::Q6_K,
        Dtype::IQ4_NLRaw => PackedMatvecKind::IQ4_NL,
        Dtype::IQ4_XSRaw => PackedMatvecKind::IQ4_XS,
        Dtype::IQ1_SRaw => PackedMatvecKind::IQ1_S,
        Dtype::IQ1_MRaw => PackedMatvecKind::IQ1_M,
        Dtype::IQ2_XXSRaw => PackedMatvecKind::IQ2_XXS,
        Dtype::IQ2_XSRaw => PackedMatvecKind::IQ2_XS,
        Dtype::IQ2_SRaw => PackedMatvecKind::IQ2_S,
        Dtype::IQ3_XXSRaw => PackedMatvecKind::IQ3_XXS,
        Dtype::IQ3_SRaw => PackedMatvecKind::IQ3_S,
        Dtype::PTQ1_0Raw => PackedMatvecKind::PTQ1_0,
        // Mapped for kind-consistency; there is no fused
        // matvec+add+rmsnorm path for these 8 formats, so stage 1's
        // launch match falls them to `_ => return false` and the caller
        // takes the legacy unfused matvec + F16 add_rmsnorm pair.
        Dtype::Q4_0Raw => PackedMatvecKind::Q4_0,
        Dtype::Q5_0Raw => PackedMatvecKind::Q5_0,
        Dtype::Q4_1Raw => PackedMatvecKind::Q4_1,
        Dtype::Q5_1Raw => PackedMatvecKind::Q5_1,
        Dtype::Q2_KRaw => PackedMatvecKind::Q2_K,
        Dtype::Q3_KRaw => PackedMatvecKind::Q3_K,
        Dtype::Q8_KRaw => PackedMatvecKind::Q8_K,
        Dtype::PQ2_0Raw => PackedMatvecKind::PQ2_0,
        _ => return false,
    };
    if packed_kind_disabled(kind) {
        return false;
    }
    if k % kind.k_alignment() != 0 {
        return false;
    }
    let w_bytes = as_bytes(w);
    let expected = m * kind.row_bytes(k);
    if w_bytes.len() < expected {
        return false;
    }
    let preloaded_usm: Option<*const u8> = w.storage.sycl_usm_ptr();
    let weight_key = w_bytes.as_ptr() as usize;
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(ctx) = slot.as_mut() else {
            return false;
        };
        if ensure_scratch_f32_grow(&ctx.stream, &mut ctx.matvec_x_f32, k).is_err() {
            return false;
        }
        // Scratch layout for the chained (matvec → add_rmsnorm_f32):
        //   matvec_x_f32      ← attn input (K)
        //   matvec_out_f32    ← matvec output dot[m] == add_rmsnorm's `branch` (M)
        //   matvec_hidden_f32 ← residual stream `hidden`, in/out (M)
        //   matvec_wnorm_f32  ← rmsnorm scale w_norm (M)
        //   matvec_out2_f32   ← normalized output y_norm (M)
        if !ensure_scratch_f32(&ctx.stream, &mut ctx.matvec_out_f32, m)
            || !ensure_scratch_f32(&ctx.stream, &mut ctx.matvec_out2_f32, m)
            || !ensure_scratch_f32(&ctx.stream, &mut ctx.matvec_wnorm_f32, m)
            || !ensure_scratch_f32(&ctx.stream, &mut ctx.matvec_hidden_f32, m)
        {
            return false;
        }
        // Weight upload (cached on host-pointer key; preloaded USM
        // weights skip this).
        if preloaded_usm.is_none()
            && !ctx.cache_packed_weight(
                weight_key,
                expected,
                w_bytes,
                w.storage.mmap_borrowed_ptr_len().is_some(),
            )
        {
            return false;
        }
        // Upload attn input. This site always re-uploaded from day
        // one (its comment documented "silent gibberish" from the
        // (ptr, len) dedup); as of 2026-09-07 every path uploads
        // unconditionally — see `try_matvec_tensor_usm_f32`.
        ctx.matvec_x_f32.as_mut().unwrap().as_mut_slice()[..k].copy_from_slice(attn);
        // Upload hidden into its dedicated scratch and w_norm.
        ctx.matvec_hidden_f32.as_mut().unwrap().as_mut_slice()[..m].copy_from_slice(hidden);
        ctx.matvec_wnorm_f32.as_mut().unwrap().as_mut_slice()[..m].copy_from_slice(&w_norm[..m]);
        let stream_raw: *const sk::SyclStream = &ctx.stream;
        let w_ptr = match preloaded_usm {
            Some(p) => p,
            None => ctx.packed_weight_cache[&weight_key].as_ptr(),
        };
        let x_ptr = ctx.matvec_x_f32.as_ref().unwrap().as_ptr();
        let dot_ptr = ctx.matvec_out_f32.as_mut().unwrap().as_mut_ptr();
        let hidden_ptr = ctx.matvec_hidden_f32.as_mut().unwrap().as_mut_ptr();
        let wnorm_ptr = ctx.matvec_wnorm_f32.as_ref().unwrap().as_ptr();
        let y_ptr = ctx.matvec_out2_f32.as_mut().unwrap().as_mut_ptr();
        // Fused H6 path, Stage 1: run the packed matvec, writing `w · x`
        // into the dot buffer. Any kernel miss signals `false` so the
        // caller falls back to the unfused CPU/USM ladder.
        macro_rules! h6_matvec {
            ($f:path, $kc:expr) => {
                unsafe {
                    let lws = tuned_lws_for($kc, m, k);
                    $f(&*stream_raw, w_ptr, x_ptr, dot_ptr, m as u32, k as u32, lws)
                }
                .is_ok()
            };
        }
        let ok_matvec = match kind {
            PackedMatvecKind::Q8_0 => h6_matvec!(sk::matvec_q8_0_packed_f32_usm_raw, rustllama_tuner::KERNEL_Q8_0_PACKED_USM),
            PackedMatvecKind::Q4_K => h6_matvec!(sk::matvec_q4_k_packed_f32_usm_raw, rustllama_tuner::KERNEL_Q4K_PACKED_USM),
            PackedMatvecKind::Q5_K => h6_matvec!(sk::matvec_q5_k_packed_f32_usm_raw, rustllama_tuner::KERNEL_Q5K_PACKED_USM),
            PackedMatvecKind::Q6_K => h6_matvec!(sk::matvec_q6_k_packed_f32_usm_raw, rustllama_tuner::KERNEL_Q6K_PACKED_USM),
            PackedMatvecKind::IQ4_NL => h6_matvec!(sk::matvec_iq4_nl_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ4_NL_PACKED_USM),
            PackedMatvecKind::IQ4_XS => h6_matvec!(sk::matvec_iq4_xs_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ4_XS_PACKED_USM),
            PackedMatvecKind::IQ1_S => h6_matvec!(sk::matvec_iq1_s_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ1_S_PACKED_USM),
            PackedMatvecKind::IQ1_M => h6_matvec!(sk::matvec_iq1_m_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ1_M_PACKED_USM),
            PackedMatvecKind::IQ2_XXS => h6_matvec!(sk::matvec_iq2_xxs_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ2_XXS_PACKED_USM),
            PackedMatvecKind::IQ2_XS => h6_matvec!(sk::matvec_iq2_xs_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ2_XS_PACKED_USM),
            PackedMatvecKind::IQ2_S => h6_matvec!(sk::matvec_iq2_s_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ2_S_PACKED_USM),
            PackedMatvecKind::IQ3_XXS => h6_matvec!(sk::matvec_iq3_xxs_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ3_XXS_PACKED_USM),
            PackedMatvecKind::IQ3_S => h6_matvec!(sk::matvec_iq3_s_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ3_S_PACKED_USM),
            // PTQ1_0 stage-1 matvec — mirrors the Q8_0/Q4_K arms (packed
            // matvec into `dot`, then the SHARED stage-2
            // `add_rmsnorm_f32_usm` below). We deliberately route through
            // `matvec_ptq1_0_packed_f32_usm_raw`, NOT the single-kernel
            // `matvec_ptq1_0_add_rmsnorm_usm`: this function was rewritten
            // (2026-05-29) to chain two proven kernels precisely because
            // the 1-workgroup fused `matvec_X_add_rmsnorm_usm` topology
            // produced gibberish on Q4_K. The packed matvec keeps PTQ1_0
            // on that proven two-stage path.
            PackedMatvecKind::PTQ1_0 => h6_matvec!(sk::matvec_ptq1_0_packed_f32_usm_raw, rustllama_tuner::KERNEL_PTQ1_0_PACKED_USM),
            // The 8 CPU-parity packed formats have no H6 arm — signal
            // miss so the caller takes the unfused ladder.
            _ => return false,
        };
        note_packed_kind_result(kind, ok_matvec);
        if !ok_matvec {
            return false;
        }
        // Stage 2: F32 add+rmsnorm — `hidden += dot`, then
        // `y_norm = rmsnorm(hidden) * w_norm`. Multi-workgroup,
        // one-row-per-work-item topology proven on the unfused path;
        // F32 precision (no F32→F16→F32 round-trip that the legacy
        // `try_add_rmsnorm_usm_f32` requires).
        let ok_norm = unsafe {
            sk::add_rmsnorm_f32_usm_raw(
                &*stream_raw,
                hidden_ptr,
                dot_ptr,
                wnorm_ptr,
                y_ptr,
                1,
                m as u32,
                eps,
            )
        }
        .is_ok();
        let _ = (preloaded_usm, weight_key);
        if !ok_norm {
            return false;
        }
        // Copy back: residual sum → hidden, normalized → y_norm.
        hidden.copy_from_slice(&ctx.matvec_hidden_f32.as_ref().unwrap().as_slice()[..m]);
        y_norm.copy_from_slice(&ctx.matvec_out2_f32.as_ref().unwrap().as_slice()[..m]);
        true
    })
}

/// H8: F16-input mixed-precision packed matvec. Same arithmetic as
/// [`try_matvec_tensor_usm_f32`] but the activation vector is uploaded
/// as F16 (halving its memory traffic on the GPU's per-output-row
/// re-read). Weights stay packed; output stays F32.
///
/// Gated behind `RUSTLLAMA_MIXED_PRECISION_MATVEC=1` (default off):
/// F16 activations drop ~3 mantissa bits, so this stays opt-in until
/// validated within tolerance on hardware (Open Risk #7). Returns
/// `false` when off, the dtype has no F16-input kernel, the layer is
/// CPU-resident, or any shape check fails — caller falls through to
/// the F32 USM path then CPU.
pub fn try_matvec_tensor_f16in_usm_f32(
    w: &Tensor,
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) -> bool {
    if !mixed_precision_matvec_enabled() {
        return false;
    }
    if !usm_attn_enabled() {
        return false;
    }
    if tensor_forced_to_cpu(&w.name) {
        return false;
    }
    if !gpu_active_for_current_layer() {
        return false;
    }
    if m == 0 || k == 0 || x.len() != k || out.len() != m {
        return false;
    }
    if !matvec_above_min_flops(2u64 * m as u64 * k as u64) {
        return false;
    }
    let kind = match w.dtype {
        Dtype::Q8_0Raw => PackedMatvecKind::Q8_0,
        Dtype::Q4_KRaw => PackedMatvecKind::Q4_K,
        Dtype::Q5_KRaw => PackedMatvecKind::Q5_K,
        Dtype::Q6_KRaw => PackedMatvecKind::Q6_K,
        Dtype::IQ4_NLRaw => PackedMatvecKind::IQ4_NL,
        Dtype::IQ4_XSRaw => PackedMatvecKind::IQ4_XS,
        Dtype::IQ1_SRaw => PackedMatvecKind::IQ1_S,
        Dtype::IQ1_MRaw => PackedMatvecKind::IQ1_M,
        Dtype::IQ2_XXSRaw => PackedMatvecKind::IQ2_XXS,
        Dtype::IQ2_XSRaw => PackedMatvecKind::IQ2_XS,
        Dtype::IQ2_SRaw => PackedMatvecKind::IQ2_S,
        Dtype::IQ3_XXSRaw => PackedMatvecKind::IQ3_XXS,
        Dtype::IQ3_SRaw => PackedMatvecKind::IQ3_S,
        Dtype::PTQ1_0Raw => PackedMatvecKind::PTQ1_0,
        _ => return false,
    };
    if packed_kind_disabled(kind) {
        return false;
    }
    if k % kind.k_alignment() != 0 {
        return false;
    }
    let w_bytes = as_bytes(w);
    let expected = m * kind.row_bytes(k);
    if w_bytes.len() < expected {
        return false;
    }
    let preloaded_usm: Option<*const u8> = w.storage.sycl_usm_ptr();
    let weight_key = w_bytes.as_ptr() as usize;
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(ctx) = slot.as_mut() else {
            return false;
        };
        if !ensure_scratch(&ctx.stream, &mut ctx.matvec_x_f16in, k) {
            return false;
        }
        if !ensure_scratch_f32(&ctx.stream, &mut ctx.matvec_out_f32, m) {
            return false;
        }
        if preloaded_usm.is_none()
            && !ctx.cache_packed_weight(
                weight_key,
                expected,
                w_bytes,
                w.storage.mmap_borrowed_ptr_len().is_some(),
            )
        {
            return false;
        }
        // Convert activations F32 → F16 bits into the dedicated u16
        // scratch (this is the bandwidth-saving step — the kernel
        // then reads half the bytes per output-row re-read).
        {
            let xf16 = ctx.matvec_x_f16in.as_mut().unwrap();
            rustllama_kernels_cpu::f32_to_f16_bits(x, &mut xf16.as_mut_slice()[..k]);
        }
        let stream_raw: *const sk::SyclStream = &ctx.stream;
        let w_ptr = match preloaded_usm {
            Some(p) => p,
            None => ctx.packed_weight_cache[&weight_key].as_ptr(),
        };
        let x_ptr = ctx.matvec_x_f16in.as_ref().unwrap().as_ptr();
        let out_ptr = ctx.matvec_out_f32.as_mut().unwrap().as_mut_ptr();
        macro_rules! h8_call {
            ($f:path, $kc:expr) => {
                unsafe {
                    let lws = tuned_lws_for($kc, m, k);
                    $f(&*stream_raw, w_ptr, x_ptr, out_ptr, m as u32, k as u32, lws)
                }
                .is_ok()
            };
        }
        let ok = match kind {
            PackedMatvecKind::Q8_0 => h8_call!(sk::matvec_q8_0_f16in_packed_f32_usm_raw, rustllama_tuner::KERNEL_Q8_0_PACKED_USM),
            PackedMatvecKind::Q4_K => h8_call!(sk::matvec_q4_k_f16in_packed_f32_usm_raw, rustllama_tuner::KERNEL_Q4K_PACKED_USM),
            PackedMatvecKind::Q5_K => h8_call!(sk::matvec_q5_k_f16in_packed_f32_usm_raw, rustllama_tuner::KERNEL_Q5K_PACKED_USM),
            PackedMatvecKind::Q6_K => h8_call!(sk::matvec_q6_k_f16in_packed_f32_usm_raw, rustllama_tuner::KERNEL_Q6K_PACKED_USM),
            PackedMatvecKind::IQ4_NL => h8_call!(sk::matvec_iq4_nl_f16in_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ4_NL_PACKED_USM),
            PackedMatvecKind::IQ4_XS => h8_call!(sk::matvec_iq4_xs_f16in_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ4_XS_PACKED_USM),
            PackedMatvecKind::IQ1_S => h8_call!(sk::matvec_iq1_s_f16in_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ1_S_PACKED_USM),
            PackedMatvecKind::IQ1_M => h8_call!(sk::matvec_iq1_m_f16in_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ1_M_PACKED_USM),
            PackedMatvecKind::IQ2_XXS => h8_call!(sk::matvec_iq2_xxs_f16in_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ2_XXS_PACKED_USM),
            PackedMatvecKind::IQ2_XS => h8_call!(sk::matvec_iq2_xs_f16in_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ2_XS_PACKED_USM),
            PackedMatvecKind::IQ2_S => h8_call!(sk::matvec_iq2_s_f16in_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ2_S_PACKED_USM),
            PackedMatvecKind::IQ3_XXS => h8_call!(sk::matvec_iq3_xxs_f16in_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ3_XXS_PACKED_USM),
            PackedMatvecKind::IQ3_S => h8_call!(sk::matvec_iq3_s_f16in_packed_f32_usm_raw, rustllama_tuner::KERNEL_IQ3_S_PACKED_USM),
            // PTQ1_0 f16-in packed matvec — mirrors the Q8_0/Q4_K arms
            // (same guards, same tuner-const LWS source).
            PackedMatvecKind::PTQ1_0 => h8_call!(sk::matvec_ptq1_0_f16in_packed_f32_usm_raw, rustllama_tuner::KERNEL_PTQ1_0_PACKED_USM),
            // The 8 CPU-parity packed formats have no f16-in H8 kernel —
            // miss to the F32-in path.
            _ => return false,
        };
        note_packed_kind_result(kind, ok);
        if !ok {
            return false;
        }
        out.copy_from_slice(&ctx.matvec_out_f32.as_ref().unwrap().as_slice()[..m]);
        true
    })
}

/// Non-quantized analogue of [`try_matvec_tensor_usm_f32`] for F16
/// weights. The packed-matvec path matches only `Q*Raw` dtypes;
/// this path picks up `Dtype::F16` tensors and dispatches via the
/// generic `rsl_gemm_f16` kernel with `N=1`. Activation + output
/// scratch live in their own `matvec_f16_*` slots (separate from
/// the F32 scratch the quantized path uses) so the two helpers
/// don't fight over the same buffers.
///
/// Returns `false` on:
///   - USM opted out (`usm_attn_enabled() = false`)
///   - per-tensor CPU pinning via `placement.overrides`
///   - the layer's `n_gpu_layers` cutoff puts it on CPU
///   - shape mismatch (caller validates earlier; this is a guard)
///   - `w.dtype != Dtype::F16` (the only dtype this helper handles)
///   - kernel / alloc failure
///
/// On success: kernel-launch overhead dominates for tiny matvecs
/// on integrated GPUs (~50-200 µs on Iris Xe), so this is a real
/// win only for the larger projections (Q/K/V/O at d_model = 4096
/// and the LM head). The caller doesn't need to know that — falls
/// back cleanly to CPU when this returns `false`.
/// GPU embedding-table lookup for F16 tables. Mirrors the existing
/// [`try_matvec_f16_usm_f32`] pattern: caches the F16 table in USM
/// keyed by host pointer, gathers rows into a per-call USM output
/// scratch, then converts the F16 output back to host F32.
///
/// Skip + return `false` (caller falls back to CPU [`k::embed_lookup_tensor`]) when:
///   - USM isn't enabled / no SYCL device
///   - the table isn't F16 (Q-table embeddings still dequant on CPU)
///   - `gpu_active_for_current_layer()` is false (embedding sits at
///     layer 0 conceptually; we honor the `n_gpu_layers` cutoff)
///   - any shape check or alloc fails
///
/// `d` is the embedding dimension (one row per id). Output `out` is
/// laid out `[ids.len(), d]` row-major, matching the CPU path's
/// convention so the caller's downstream code is identical.
/// Upload a host byte slice into a fresh SYCL USM allocation on the
/// engine's thread-local stream, and return it wrapped in a
/// [`rustllama_tensor::Storage::SyclUsm`] variant. Used by the
/// load-time path to pre-resident large/hot weights on the GPU,
/// eliminating the upload-per-call cost in [`try_matvec_f16_usm_f32`]
/// and [`try_embedding_lookup_usm_f32`].
///
/// Returns `None` when:
///   - USM isn't enabled / no SYCL device is visible
///   - The host context isn't ready (engine startup hasn't bound a
///     stream yet on this thread)
///   - The USM allocation itself failed (out of memory, runtime error)
///
/// The closure that frees the allocation captures the stream as a
/// raw `usize` pointer. SAFETY: the caller MUST ensure the stream
/// outlives every [`rustllama_tensor::Storage::SyclUsm`] clone — in
/// practice this is automatic because the engine's `UsmAttnContext`
/// owns both, and the context's `Drop` runs the cache + stream
/// teardown in the right order.
pub fn upload_bytes_to_usm(
    bytes: &[u8],
) -> Option<rustllama_tensor::Storage> {
    if !usm_attn_enabled() || bytes.is_empty() {
        return None;
    }
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(ctx) = slot.as_mut() else {
            return None;
        };
        let n = bytes.len();
        // Allocate USM on the engine's stream.
        let raw = sk::usm_alloc_shared_raw(&ctx.stream, n);
        if raw.is_null() {
            return None;
        }
        // Copy host bytes into the USM allocation. The pointer is
        // shared (CPU + GPU both see it), so this is just a memcpy
        // — no special staging buffer needed.
        // SAFETY: `raw` is a valid USM allocation of `n` bytes per
        // the SYCL runtime contract.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), raw as *mut u8, n);
        }
        // Capture the stream as an opaque `usize` so the closure is
        // `Send + Sync`. The closure runs on the last Arc drop —
        // typically thread-local context teardown, on the same
        // thread that owns the stream.
        let stream_addr = &ctx.stream as *const sk::SyclStream as usize;
        let free_fn: Box<dyn FnOnce(*mut u8) + Send + Sync> = Box::new(move |p| {
            // SAFETY: `stream_addr` points to a `SyclStream` that
            // the engine's `UsmAttnContext` owns and that outlives
            // any `Storage::SyclUsm` allocation it produced. See
            // upload_bytes_to_usm doc comment for the lifetime
            // contract.
            unsafe {
                let stream_ref = &*(stream_addr as *const sk::SyclStream);
                sk::usm_free_raw(stream_ref, p as *mut std::ffi::c_void);
            }
        });
        // SAFETY: `raw` is non-null (checked above) and points to a
        // `n`-byte USM allocation; the closure frees exactly that.
        let storage = unsafe {
            rustllama_tensor::Storage::sycl_usm_from_raw(raw as *mut u8, n, free_fn)
        };
        Some(storage)
    })
}

pub fn try_embedding_lookup_usm_f32(
    table: &Tensor,
    ids: &[i32],
    out: &mut [f32],
    d: usize,
) -> bool {
    use half::f16;

    if !usm_attn_enabled() {
        return false;
    }
    if table.dtype != Dtype::F16 {
        return false;
    }
    if tensor_forced_to_cpu(&table.name) {
        return false;
    }
    if !gpu_active_for_current_layer() {
        return false;
    }
    if ids.is_empty() || d == 0 {
        return false;
    }
    let n_ids = ids.len();
    if out.len() != n_ids * d {
        log_matvec_skip_once("embedding_lookup: out shape mismatch");
        return false;
    }
    let table_bytes = as_bytes(table);
    // Table size in F16 entries = bytes / 2. The kernel takes `d`
    // (row stride) and relies on the caller having sized the table
    // to at least `max(ids)*d` rows — the GGUF token_embd tensor
    // shape `[vocab, d]` always satisfies this.
    if table_bytes.len() < d * 2 {
        log_matvec_skip_once("embedding_lookup: table too small");
        return false;
    }

    // Fast path: the table's storage is already SyclUsm (preloaded
    // via `LlamaModel::preload_weights_to_usm`). Skip the cache
    // lookup + host→device upload entirely — the kernel reads the
    // USM pointer directly. Falls through to the cached-upload path
    // when storage is the default `CpuOwned`.
    if let Some(usm_ptr) = table.storage.sycl_usm_ptr() {
        return USM_ATTN.with(|cell| {
            let mut slot = cell.borrow_mut();
            let Some(ctx) = slot.as_mut() else {
                return false;
            };
            if !ensure_scratch(&ctx.stream, &mut ctx.matvec_f16_out, n_ids * d) {
                log_matvec_skip_once("embedding_lookup: output scratch alloc failed");
                return false;
            }
            let stream_raw: *const sk::SyclStream = &ctx.stream;
            let out_ptr = ctx.matvec_f16_out.as_mut().unwrap().as_mut_ptr();
            // SAFETY: usm_ptr is a USM allocation on this same
            // stream (the engine's preload path bound them together
            // through `upload_bytes_to_usm`).
            let ok = unsafe {
                sk::embedding_lookup_usm_raw(
                    &*stream_raw,
                    usm_ptr as *const u16,
                    ids.as_ptr(),
                    out_ptr,
                    n_ids as u32,
                    d as u32,
                )
            }
            .is_ok();
            if !ok {
                log_matvec_skip_once("embedding_lookup_usm_raw (preloaded) returned Err");
                return false;
            }
            let out_buf = ctx.matvec_f16_out.as_ref().expect("just used");
            let out_slice = &out_buf.as_slice()[..n_ids * d];
            for (i, dst) in out.iter_mut().enumerate() {
                *dst = f16::from_bits(out_slice[i]).to_f32();
            }
            true
        });
    }

    // G4: USM-budget gate. Iris Xe iGPUs share host RAM; a Qwen2.5
    // 152K × 4K F16 token-embedding table is ~600 MB — ~7.5% of the
    // 8 GB iGPU budget before the model proper loads. The gate
    // `RUSTLLAMA_EMBED_GPU_MAX_MB` (default 1024) keeps the cache
    // off-GPU for big-vocab/big-d models that would otherwise eat
    // the budget on a one-time upload. Set `=0` to disable USM
    // upload entirely; fall back to the CPU embedding-lookup path.
    if !embed_cache_fits_budget(table_bytes.len()) {
        log_matvec_skip_once("embedding_lookup: table size exceeds RUSTLLAMA_EMBED_GPU_MAX_MB");
        return false;
    }

    let table_key = table_bytes.as_ptr() as usize;

    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(ctx) = slot.as_mut() else {
            return false;
        };
        // Cache the F16 table in USM. The token-embedding tensor
        // never changes between forward passes within a model load,
        // so the upload happens once on the first call and every
        // subsequent lookup is a pure GPU gather.
        if !ctx.packed_weight_cache.contains_key(&table_key) {
            let Ok(buf) =
                sk::SyclSharedBuffer::<u8>::alloc(&ctx.stream, table_bytes.len())
            else {
                return false;
            };
            // SAFETY: same '_-to-'static transmute pattern as the
            // F16 matvec weight cache. Drop on UsmAttnContext clears
            // the cache before the stream drops.
            let mut buf = unsafe {
                std::mem::transmute::<
                    sk::SyclSharedBuffer<'_, u8>,
                    sk::SyclSharedBuffer<'static, u8>,
                >(buf)
            };
            buf.as_mut_slice().copy_from_slice(table_bytes);
            // Windows: VirtualLock the embedding table into RAM so the
            // OS can't page this fixed-size, every-token-hot USM region
            // to disk. Self-gated by RUSTLLAMA_LOCK_USM (+ _MB budget);
            // no-op when disabled.
            let pinned = buf.pin();
            // Embedding table stays in the SHARED tier: it's VirtualLock-
            // pinned (host-mapped) and gathered every token, so it must
            // remain host-addressable — the device tier is weight-only.
            ctx.packed_weight_cache
                .insert(table_key, PackedWeightBuf::Shared(buf));
            static FIRST_EMB_UPLOAD: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            FIRST_EMB_UPLOAD.get_or_init(|| {
                tracing::info!(
                    bytes = table_bytes.len(),
                    usm_pinned = pinned,
                    "first embedding-table USM upload — GPU lookup live{}",
                    if pinned {
                        " (VirtualLock'd into RAM)"
                    } else {
                        " (not pinned: RUSTLLAMA_LOCK_USM off / over budget)"
                    }
                );
            });
        }
        // Per-call output scratch sized to n_ids*d (F16).
        if !ensure_scratch(&ctx.stream, &mut ctx.matvec_f16_out, n_ids * d) {
            log_matvec_skip_once("embedding_lookup: output scratch alloc failed");
            return false;
        }
        let stream_raw: *const sk::SyclStream = &ctx.stream;
        let table_ptr = ctx.packed_weight_cache[&table_key].as_ptr() as *const u16;
        let out_ptr = ctx.matvec_f16_out.as_mut().unwrap().as_mut_ptr();
        // SAFETY: USM pointers on `stream`; shape validated; ctx
        // outlives the call.
        let ok = unsafe {
            sk::embedding_lookup_usm_raw(
                &*stream_raw,
                table_ptr,
                ids.as_ptr(),
                out_ptr,
                n_ids as u32,
                d as u32,
            )
        }
        .is_ok();
        if !ok {
            log_matvec_skip_once("embedding_lookup_usm_raw returned Err");
            return false;
        }
        // Convert USM F16 → host F32. Length = n_ids * d.
        let out_buf = ctx.matvec_f16_out.as_ref().expect("just used");
        let out_slice = &out_buf.as_slice()[..n_ids * d];
        for (i, dst) in out.iter_mut().enumerate() {
            *dst = f16::from_bits(out_slice[i]).to_f32();
        }
        true
    })
}

/// G5: Size-based matvec dispatch gate. Returns `true` if the matvec
/// has enough FLOPs to justify the GPU launch overhead (Level Zero
/// dispatch + USM marshaling). Reads `RUSTLLAMA_MATVEC_MIN_FLOPS`
/// (default 0 = no gate, always try GPU). The plan calls for a
/// bench-driven default (~64K FLOPs likely); leaving 0 by default so
/// the gate has zero behavioral effect until a user benchmarks their
/// hardware. Sub-threshold matvecs return `false` and fall through
/// to the CPU SIMD path.
fn matvec_above_min_flops(flops: u64) -> bool {
    static MIN_FLOPS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let min = *MIN_FLOPS.get_or_init(|| {
        std::env::var("RUSTLLAMA_MATVEC_MIN_FLOPS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
    });
    if min == 0 {
        return true;
    }
    flops >= min
}

/// G4: Embedding-cache USM-budget gate. Returns `true` if
/// `table_bytes` fits within `RUSTLLAMA_EMBED_GPU_MAX_MB` (default
/// 1024 MB), `false` otherwise. The cap exists because the
/// integrated iGPU's ~8 GB USM-shared budget is the same memory the
/// host CPU also needs — a 1 GB Llama-3 70B embedding table eats
/// 12.5% of the iGPU budget on a one-time upload, leaving less for
/// KV cache, scratch, and matvec workspace. Set `RUSTLLAMA_EMBED_GPU_MAX_MB=0`
/// to force the CPU embedding-lookup path entirely.
fn embed_cache_fits_budget(table_bytes_len: usize) -> bool {
    static MAX_MB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let cap = *MAX_MB.get_or_init(|| {
        std::env::var("RUSTLLAMA_EMBED_GPU_MAX_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(1024)
    });
    if cap == 0 {
        return false;
    }
    let bytes_cap = cap.saturating_mul(1024 * 1024);
    (table_bytes_len as u64) <= bytes_cap
}

pub fn try_matvec_f16_usm_f32(
    w: &Tensor,
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) -> bool {
    use half::f16;

    if !usm_attn_enabled() {
        return false;
    }
    if w.dtype != Dtype::F16 {
        return false;
    }
    if tensor_forced_to_cpu(&w.name) {
        return false;
    }
    if !gpu_active_for_current_layer() {
        return false;
    }
    if m == 0 || k == 0 || x.len() != k || out.len() != m {
        log_matvec_skip_once("F16 matvec shape check failed");
        return false;
    }
    let w_bytes = as_bytes(w);
    let expected_bytes = m * k * 2; // F16 = 2 bytes per element.
    if w_bytes.len() < expected_bytes {
        log_matvec_skip_once("F16 w_bytes too small");
        return false;
    }
    // Fast path: preloaded `Storage::SyclUsm`. Same shape as the
    // packed-matvec fast path — skip the cache + upload entirely
    // when the weight is already on the GPU.
    let preloaded_usm: Option<*const u8> = w.storage.sycl_usm_ptr();
    let weight_key = w_bytes.as_ptr() as usize;
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(ctx) = slot.as_mut() else {
            return false;
        };
        // Activation scratch: F16, sized to K. Reuses the
        // monotonically-growing `matvec_f16_x` slot.
        if !ensure_scratch(&ctx.stream, &mut ctx.matvec_f16_x, k) {
            log_matvec_skip_once("F16 activation scratch alloc failed");
            return false;
        }
        // Output scratch: F16, sized to M.
        if !ensure_scratch(&ctx.stream, &mut ctx.matvec_f16_out, m) {
            log_matvec_skip_once("F16 output scratch alloc failed");
            return false;
        }
        // Weight cache: reuse the packed_weight_cache (it's just
        // u8 bytes — F16 GGUF storage IS consecutive 2-byte values,
        // bit-identical to `half::f16` round-tripped through
        // `to_bits()`). The kernel pointer is cast to `*const u16`
        // at the call site. Preloaded SyclUsm weights skip this
        // branch entirely.
        if preloaded_usm.is_none()
            && !ctx.cache_packed_weight(
                weight_key,
                expected_bytes,
                w_bytes,
                w.storage.mmap_borrowed_ptr_len().is_some(),
            )
        {
            return false;
        }
        // Convert host F32 activation → USM F16. No dedup-by-ptr
        // (unlike the F32 packed path) — F32→F16 conversion is
        // ~K floats of arithmetic which is negligible against the
        // kernel launch.
        {
            let x_buf = ctx.matvec_f16_x.as_mut().expect("just allocated");
            // F3: SIMD F32 → F16-bits packing for the matvec activation
            // upload. Runs on every layer's matvec call (Q/K/V/W_o/up/
            // gate/down/lm_head); the cumulative saving across all layers
            // is significant on Iris Xe.
            let slice = x_buf.as_mut_slice();
            rustllama_kernels_cpu::f32_to_f16_bits(x, &mut slice[..x.len()]);
        }
        let stream_raw: *const sk::SyclStream = &ctx.stream;
        let w_ptr = match preloaded_usm {
            Some(p) => p as *const u16,
            None => ctx.packed_weight_cache[&weight_key].as_ptr() as *const u16,
        };
        let x_ptr = ctx.matvec_f16_x.as_ref().unwrap().as_ptr();
        let out_ptr = ctx.matvec_f16_out.as_mut().unwrap().as_mut_ptr();
        // gemm_f16: C[M,N] = A[M,K] @ B[K,N]. For matvec, N=1,
        // A=weights (lda=K), B=activation (ldb=N=1), C=output
        // (ldc=N=1). Kernel `.wait()`s before returning.
        // SAFETY: all USM pointers on `stream`; sizes validated
        // above; ctx outlives the call.
        let ok = unsafe {
            sk::gemm_f16_usm_raw(
                &*stream_raw,
                w_ptr,
                x_ptr,
                out_ptr,
                m as u32,
                1u32,
                k as u32,
                k as u32, // lda = K (row-major, stride between rows)
                1u32,     // ldb = N = 1
                1u32,     // ldc = N = 1
            )
        }
        .is_ok();
        if !ok {
            log_matvec_skip_once("F16 gemm kernel call returned Err");
            return false;
        }
        // Convert USM F16 output → host F32.
        let out_buf = ctx.matvec_f16_out.as_ref().expect("just used");
        let out_slice = out_buf.as_slice();
        for (i, dst) in out.iter_mut().enumerate() {
            *dst = f16::from_bits(out_slice[i]).to_f32();
        }
        true
    })
}

/// Batched analogue of [`try_matvec_tensor_usm_f32`] — computes
/// `out[n, m] = sum_k W[m, k] * x[n, k]` for `n in 0..N`, `m in
/// 0..M` in a single kernel launch. Inputs are contiguous row-major:
///   - `x`:   `[N, K]` f32  → `x[n*K + k]`
///   - `out`: `[N, M]` f32  → `out[n*M + m]`
///
/// Used by the engine's `forward_prefill_batched_*` paths to replace
/// the per-token loop of `try_matvec_tensor_usm_f32(...)` calls. The
/// win over N single-row launches is amortizing per-launch overhead
/// (≈50-200 µs on Iris Xe) across all N tokens — for a 7B model with
/// ~6 projections × 28 layers × N=64 tokens that's >10000 launches
/// collapsed to ≈160.
///
/// Returns `false` if USM is opted out, if the weight dtype isn't a
/// supported packed-quant format, on shape mismatch, or on any
/// internal alloc / kernel failure — the caller falls back to the
/// per-row loop for this projection (CPU or single-row USM
/// depending on what the per-row path itself chooses).
///
/// Shares `packed_weight_cache` + `matvec_x_f32` + `matvec_out_f32`
/// with the single-row hook: the scratch buffers grow monotonically
/// to `max(N*K, K)` and `max(N*M, M)` so prefill and decode reuse
/// the same allocations without re-alloc churn.
pub fn try_matvec_tensor_batched_usm_f32(
    w: &Tensor,
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    // Phase 4 cross-GPU routing (batched twin — see the single-row hook).
    // INERT unless a heat plan is installed; only a NON-default CUDA device is
    // handled here, and a routed miss falls through.
    if multi_gpu_plan_active() {
        if let Some(plan) = multi_gpu_plan() {
            if let Some(t) = plan.device_for(&w.name, current_layer_idx()) {
                if t.backend == GpuBackend::Cuda
                    && t.device_index != CUDA_DEFAULT_DEVICE
                    && m != 0
                    && k != 0
                    && n != 0
                    && x.len() == n * k
                    && out.len() == n * m
                    && matvec_above_min_flops(2u64 * n as u64 * m as u64 * k as u64)
                {
                    if let Some(ck_kind) = dtype_to_cuda_kind(w.dtype) {
                        let wb = as_bytes(w);
                        if try_matvec_packed_cuda_dev_batched(
                            t.device_index,
                            ck_kind,
                            wb.as_ptr() as usize,
                            wb,
                            x,
                            out,
                            m,
                            k,
                            n,
                        ) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    // Blackwell SM12x FP4 tensor-core GEMM (W4A4) — tried BEFORE the scalar
    // CUDA packed path. OPT-IN and inert unless RUSTLLAMA_FP4_TC=1 on a real
    // SM12x Blackwell TC build (`fp4_tc_enabled()` folds both). Only for the
    // prefill-size batches that amortize the tensor core (n >= 16) with
    // K % 64 == 0; decode-size N and other K stay on the scalar GEMV. A TC
    // miss (returns false) falls straight through to the scalar block below —
    // byte-identical behavior on every non-opted-in / non-Blackwell host.
    if fp4_tc_enabled()
        && !tensor_forced_to_cpu(&w.name)
        && current_layer_idx() < n_gpu_layers()
        && m != 0
        && k != 0
        && n >= 16
        && (k % 64 == 0)
        && x.len() == n * k
        && out.len() == n * m
    {
        if let Some(tc_kind) = dtype_to_fp4_tc_kind(w.dtype) {
            let wb = as_bytes(w);
            if try_gemm_fp4_tc_batched(tc_kind, wb.as_ptr() as usize, wb, x, out, m, k, n) {
                return true;
            }
        }
    }
    // Hopper SM90a FP8 wgmma tensor-core GEMM (W8A8) for MXFP8 weights — the
    // Hopper twin of the Blackwell FP4 TC block above, same prefill-batch gate
    // (n >= 16, K % 32 == 0 for the wgmma k-dim), tried BEFORE the scalar CUDA
    // packed path. OPT-IN + inert unless RUSTLLAMA_FP8_WGMMA=1 on a real sm_90
    // Hopper TC build (`hopper_tc_enabled()` folds both). A miss falls straight
    // through to the scalar block below — byte-identical on every other host.
    if hopper_tc_enabled()
        && !tensor_forced_to_cpu(&w.name)
        && current_layer_idx() < n_gpu_layers()
        && m != 0
        && k != 0
        && n >= 16
        && (k % 32 == 0)
        && x.len() == n * k
        && out.len() == n * m
        && matches!(w.dtype, Dtype::Mxfp8Raw)
    {
        let wb = as_bytes(w);
        if try_gemm_fp8_wgmma_batched(wb.as_ptr() as usize, wb, x, out, m, k, n) {
            return true;
        }
    }
    // Blackwell SM12x MXFP8 / MXFP6 block-scaled tensor-core GEMM (W8A8 / W6A6)
    // — the 8-/6-bit siblings of the FP4 TC block above, over the same sm_120
    // block-scaled MMA. Their kernels (gemm_mxfp8_tc / gemm_mxfp6_tc) passed the
    // SAME on-device `--validate-kernels` probe as the FP4/Q4_K TC paths, so they
    // auto-enable per their verdicts (fail-closed) — closing the gap where an
    // MXFP8/MXFP6 WEIGHT on Blackwell silently stayed on the scalar packed matvec
    // even though its TC kernel was validated. DISTINCT from the Hopper block
    // above (sm_90 `wgmma`, gated off on Blackwell): an MXFP8 weight lands here on
    // Blackwell and there on Hopper. Same prefill-batch gate (n >= 16, K % 32 ==
    // 0 for the 32-elem microscale blocks). A miss falls through to the scalar
    // packed path below — byte-identical on every non-Blackwell host.
    if (mxfp8_tc_enabled() || mxfp6_tc_enabled())
        && !tensor_forced_to_cpu(&w.name)
        && current_layer_idx() < n_gpu_layers()
        && m != 0
        && k != 0
        && n >= 16
        && (k % 32 == 0)
        && x.len() == n * k
        && out.len() == n * m
    {
        let wb = as_bytes(w);
        match w.dtype {
            Dtype::Mxfp8Raw if mxfp8_tc_enabled() => {
                if try_gemm_mxfp8_tc_batched(wb.as_ptr() as usize, wb, x, out, m, k, n) {
                    return true;
                }
            }
            Dtype::Mxfp6Raw if mxfp6_tc_enabled() => {
                if try_gemm_mxfp6_tc_batched(wb.as_ptr() as usize, wb, x, out, m, k, n) {
                    return true;
                }
            }
            _ => {}
        }
    }
    // Intel XMX/DPAS bf16 tensor-core GEMM — the SYCL twin of the CUDA TC gates
    // above. Dequantizes W→f32 and runs a joint_matrix GEMM for prefill-size
    // batches (n >= 16 amortizes the dequant + tensor core); the kernel
    // zero-pads M/N/K tails so no dim alignment is needed. OPT-IN + inert
    // unless RUSTLLAMA_SYCL_XMX=1 on an XMX-capable Arc/PVC device (probed
    // per-stream inside try_gemm_xmx_batched). A miss falls straight through to
    // the scalar SYCL/CPU ladder — byte-identical on Iris Xe / non-opted-in
    // hosts. CUDA-first doctrine preserved: the CUDA gates above win when a
    // CUDA GPU is also present (XMX only engages on a SYCL-active host).
    if xmx_enabled()
        && !tensor_forced_to_cpu(&w.name)
        && current_layer_idx() < n_gpu_layers()
        && m != 0
        && k != 0
        && n >= 16
        && x.len() == n * k
        && out.len() == n * m
    {
        if try_gemm_xmx_batched(w, x, out, m, k, n) {
            return true;
        }
    }
    // Native CUDA backend FIRST (batched twin — see the single-row
    // hook). Universal gates only; inert on non-NVIDIA hosts.
    if cuda_active()
        && !tensor_forced_to_cpu(&w.name)
        && current_layer_idx() < n_gpu_layers()
        && m != 0
        && k != 0
        && n != 0
        && x.len() == n * k
        && out.len() == n * m
        && matvec_above_min_flops(2u64 * n as u64 * m as u64 * k as u64)
    {
        if let Some(ck_kind) = dtype_to_cuda_kind(w.dtype) {
            let wb = as_bytes(w);
            let q4k = matches!(ck_kind, ck::CudaPackedKind::Q4_K);
            // Q4_K int8 tensor-core GEMM (W8A8, lossy) — tried FIRST: its verdict
            // requires beating the lossless f32 GEMM on-device, so when it's on it
            // is the fastest Q4_K prefill path. A miss falls through.
            if q4k
                && q4k_w8a8_tc_enabled()
                && n >= 16
                && k % 256 == 0
                && try_gemm_q4k_w8a8_tc_batched(wb.as_ptr() as usize, wb, x, out, m, k, n)
            {
                return true;
            }
            // Q4_K prefill GEMM (bit-exact, shared-mem weight reuse) — the lossless
            // path: fixes the O(M·K·N) weight re-streaming of the batched matvec.
            if q4k
                && q4k_gemm_f32_enabled()
                && n >= 16
                && k % 256 == 0
                && try_gemm_q4k_f32_batched(wb.as_ptr() as usize, wb, x, out, m, k, n)
            {
                return true;
            }
            if try_matvec_packed_cuda_batched(ck_kind, wb.as_ptr() as usize, wb, x, out, m, k, n) {
                return true;
            }
        }
    }
    // Native MLX (Apple Metal) backend — batched twin of the single-row
    // hook. Universal gates only; inert off Apple Silicon.
    if mlx_active()
        && !tensor_forced_to_cpu(&w.name)
        && current_layer_idx() < n_gpu_layers()
        && m != 0
        && k != 0
        && n != 0
        && x.len() == n * k
        && out.len() == n * m
        && matvec_above_min_flops(2u64 * n as u64 * m as u64 * k as u64)
    {
        if let Some(mk_kind) = dtype_to_mlx_kind(w.dtype) {
            let wb = as_bytes(w);
            if try_matvec_packed_mlx_batched(mk_kind, wb.as_ptr() as usize, wb, x, out, m, k, n) {
                return true;
            }
        }
    }
    if !usm_attn_enabled() {
        log_matvec_batched_skip_once("usm_attn_enabled = false");
        return false;
    }
    if tensor_forced_to_cpu(&w.name) {
        // Per-tensor override (same as the single-row variant).
        return false;
    }
    if !gpu_active_for_current_layer() {
        // Same placement-cutoff path as the single-row variant —
        // CPU-resident layer falls through to the CPU batched
        // matvec.
        return false;
    }
    if m == 0 || k == 0 || n == 0 || x.len() != n * k || out.len() != n * m {
        log_matvec_batched_skip_once("shape check failed");
        return false;
    }
    // G5: size-based gate. Batched FLOPs ≈ 2 * N * M * K.
    if !matvec_above_min_flops(2u64 * n as u64 * m as u64 * k as u64) {
        return false;
    }
    let kind = match w.dtype {
        Dtype::Q8_0Raw => PackedMatvecKind::Q8_0,
        Dtype::Q4_KRaw => PackedMatvecKind::Q4_K,
        Dtype::Q5_KRaw => PackedMatvecKind::Q5_K,
        Dtype::Q6_KRaw => PackedMatvecKind::Q6_K,
        Dtype::IQ4_NLRaw => PackedMatvecKind::IQ4_NL,
        Dtype::IQ4_XSRaw => PackedMatvecKind::IQ4_XS,
        // The IQ / PTQ1_0 batched USM kernels below were live but
        // UNREACHABLE: this gate mapped only the 6 formats above, so
        // batched PREFILL of these 8 formats fell through to the CPU
        // batched matvec while the single-row DECODE gate
        // (`try_matvec_tensor_usm_f32`) already routed them to the GPU.
        // Map them here too so all 14 packed dtypes reach the GPU on
        // prefill, matching the single-row gate exactly.
        Dtype::IQ1_SRaw => PackedMatvecKind::IQ1_S,
        Dtype::IQ2_XXSRaw => PackedMatvecKind::IQ2_XXS,
        Dtype::IQ1_MRaw => PackedMatvecKind::IQ1_M,
        Dtype::IQ2_XSRaw => PackedMatvecKind::IQ2_XS,
        Dtype::IQ2_SRaw => PackedMatvecKind::IQ2_S,
        Dtype::IQ3_XXSRaw => PackedMatvecKind::IQ3_XXS,
        Dtype::IQ3_SRaw => PackedMatvecKind::IQ3_S,
        Dtype::PTQ1_0Raw => PackedMatvecKind::PTQ1_0,
        _ => {
            log_unmatched_dtype(w.dtype);
            return false;
        }
    };
    if packed_kind_disabled(kind) {
        return false;
    }
    if k % kind.k_alignment() != 0 {
        log_matvec_batched_skip_once("K alignment check failed");
        return false;
    }
    let w_bytes = as_bytes(w);
    let expected_bytes = m * kind.row_bytes(k);
    if w_bytes.len() < expected_bytes {
        log_matvec_batched_skip_once("w_bytes too small");
        return false;
    }
    // Same SyclUsm fast-path as the single-row hook. Preloaded
    // weights skip the per-call cache lookup + upload — the kernel
    // reads the USM pointer directly.
    let preloaded_usm: Option<*const u8> = w.storage.sycl_usm_ptr();
    let weight_key = w_bytes.as_ptr() as usize;
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(ctx) = slot.as_mut() else {
            log_matvec_batched_skip_once(
                "USM_ATTN slot is None (prepare_usm_context failed)",
            );
            return false;
        };
        // Activation + output scratch sized to N*K and N*M. Grows
        // monotonically; subsequent smaller calls (single-row decode)
        // reuse without realloc.
        if ensure_scratch_f32_grow(&ctx.stream, &mut ctx.matvec_x_f32, n * k).is_err() {
            log_matvec_batched_skip_once("ensure_scratch_f32 (x) failed");
            return false;
        }
        if !ensure_scratch_f32(&ctx.stream, &mut ctx.matvec_out_f32, n * m) {
            log_matvec_batched_skip_once("ensure_scratch_f32 (out) failed");
            return false;
        }
        // Weight: cached USM upload keyed on host pointer — same
        // cache as the single-row hook, so the second prefill in a
        // session (or any later decode) reuses the same USM weight.
        // Preloaded SyclUsm weights skip this branch entirely.
        if preloaded_usm.is_none()
            && !ctx.cache_packed_weight(
                weight_key,
                expected_bytes,
                w_bytes,
                w.storage.mmap_borrowed_ptr_len().is_some(),
            )
        {
            return false;
        }
        // Unconditional [N, K] activation upload — the (ptr, len)
        // dedup that lived here fed kernels stale activations; see
        // the comment in `try_matvec_tensor_usm_f32`.
        ctx.matvec_x_f32.as_mut().unwrap().as_mut_slice()[..n * k].copy_from_slice(x);
        let stream_raw: *const sk::SyclStream = &ctx.stream;
        let w_ptr = match preloaded_usm {
            Some(p) => p,
            None => ctx.packed_weight_cache[&weight_key].as_ptr(),
        };
        let x_ptr = ctx.matvec_x_f32.as_ref().unwrap().as_ptr();
        let out_ptr = ctx.matvec_out_f32.as_mut().unwrap().as_mut_ptr();
        // Batched reuses the single-row kernel's tuned LWS — the M-dim
        // work distribution per work-group is identical (each work-item
        // still produces one output row), so the LWS sweet spot is
        // (M, K)-driven and doesn't shift with N. Avoids a parallel
        // batched-only cache.
        let ok = match kind {
            // SAFETY: all USM pointers on `stream`; sizes validated
            // above; the kernel `.wait()`s before returning.
            PackedMatvecKind::Q8_0 => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_Q8_0_PACKED_USM, m, k);
                sk::matvec_q8_0_packed_f32_batched_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    n as u32,
                    lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::Q4_K => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_Q4K_PACKED_USM, m, k);
                sk::matvec_q4_k_packed_f32_batched_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    n as u32,
                    lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::Q5_K => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_Q5K_PACKED_USM, m, k);
                sk::matvec_q5_k_packed_f32_batched_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    n as u32,
                    lws,
                )
            }
            .is_ok(),
            // F4 follow-up: batched IQ4_NL / IQ4_XS prefill kernels.
            // Same M/K/N shape as the K-family batched matvecs.
            PackedMatvecKind::IQ4_NL => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ4_NL_PACKED_USM, m, k);
                sk::matvec_iq4_nl_packed_f32_batched_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    n as u32,
                    lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ4_XS => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ4_XS_PACKED_USM, m, k);
                sk::matvec_iq4_xs_packed_f32_batched_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    n as u32,
                    lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::Q6_K => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_Q6K_PACKED_USM, m, k);
                sk::matvec_q6_k_packed_f32_batched_usm_raw(
                    &*stream_raw,
                    w_ptr,
                    x_ptr,
                    out_ptr,
                    m as u32,
                    k as u32,
                    n as u32,
                    lws,
                )
            }
            .is_ok(),
            // G1: IQ-quant + PTQ1_0 batched USM kernels — same
            // arithmetic as the single-row paths above, just iterating
            // over N input activations in one kernel launch. With the
            // dtype→kind gate above now mapping these 8 formats (see the
            // note there), batched PREFILL reaches these GPU arms, so
            // the prior GPU-decode / CPU-prefill asymmetry is closed for
            // every packed format — not just the K-quants.
            PackedMatvecKind::PTQ1_0 => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_PTQ1_0_PACKED_USM, m, k);
                sk::matvec_ptq1_0_packed_f32_batched_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr,
                    m as u32, k as u32, n as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ1_S => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ1_S_PACKED_USM, m, k);
                sk::matvec_iq1_s_packed_f32_batched_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr,
                    m as u32, k as u32, n as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ2_XXS => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ2_XXS_PACKED_USM, m, k);
                sk::matvec_iq2_xxs_packed_f32_batched_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr,
                    m as u32, k as u32, n as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ1_M => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ1_M_PACKED_USM, m, k);
                sk::matvec_iq1_m_packed_f32_batched_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr,
                    m as u32, k as u32, n as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ2_XS => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ2_XS_PACKED_USM, m, k);
                sk::matvec_iq2_xs_packed_f32_batched_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr,
                    m as u32, k as u32, n as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ2_S => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ2_S_PACKED_USM, m, k);
                sk::matvec_iq2_s_packed_f32_batched_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr,
                    m as u32, k as u32, n as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ3_XXS => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ3_XXS_PACKED_USM, m, k);
                sk::matvec_iq3_xxs_packed_f32_batched_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr,
                    m as u32, k as u32, n as u32, lws,
                )
            }
            .is_ok(),
            PackedMatvecKind::IQ3_S => unsafe {
                let lws = tuned_lws_for(rustllama_tuner::KERNEL_IQ3_S_PACKED_USM, m, k);
                sk::matvec_iq3_s_packed_f32_batched_usm_raw(
                    &*stream_raw, w_ptr, x_ptr, out_ptr,
                    m as u32, k as u32, n as u32, lws,
                )
            }
            .is_ok(),
            // The 8 CPU-parity packed formats (Q4_0/Q5_0/Q4_1/Q5_1,
            // Q2_K/Q3_K/Q8_K, PQ2_0) have no batched SYCL kernel and are
            // intentionally NOT mapped in the dtype gate above — this arm
            // is unreachable in practice but keeps the match exhaustive.
            // Their batched PREFILL correctly falls to the CPU batched
            // matvec while single-row DECODE uses the GPU.
            _ => return false,
        };
        note_packed_kind_result(kind, ok);
        if !ok {
            log_matvec_batched_skip_once("kernel call returned Err");
            return false;
        }
        // Read out back to host — full [N, M] slab.
        let out_buf = ctx.matvec_out_f32.as_ref().unwrap();
        out.copy_from_slice(&out_buf.as_slice()[..n * m]);
        true
    })
}

/// Try to run the F32 prefill flash-attention kernel on the bound
/// SYCL device. On success the output is written into `out` and the
/// function returns `true` — the caller skips the CPU prefill call.
/// On any failure (USM disabled, no GPU, scratch alloc failure,
/// shape mismatch, kernel error) returns `false` and the caller
/// runs the CPU prefill path as the fallback. Shapes mirror
/// [`rustllama_kernels_cpu::gqa_attention_flash_prefill`]:
///   - `q`:       `[n_new, n_heads, head_dim]`        f32 row-major
///   - `k_cache`: `[n_kv_heads, max_ctx, head_dim]`   f32
///   - `v_cache`: `[n_kv_heads, max_ctx, head_dim]`   f32
///   - `out`:     `[n_new, n_heads, head_dim]`        f32 row-major
///
/// The caller must have appended the `n_new` new K/V rows to the
/// cache at positions `[kv_len_base, kv_len_base + n_new)` BEFORE
/// invoking — the kernel walks `t in [0, kv_len_base + q_pos]`
/// (causal mask, query sees own position).
///
/// V1 design caveat: this re-uploads the full K/V cache slab on
/// every call. On Iris Xe (shared LPDDR USM) the copy cost is small,
/// but on dGPUs we'll want a per-layer USM K/V residency pool to
/// avoid the PCIe transfer per attention call. Captured in the
/// roadmap; not blocking this first cut.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_prefill_usm_f32(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) -> bool {
    if !usm_attn_enabled() {
        log_prefill_attn_skip_once("usm_attn_enabled = false");
        return false;
    }
    if !gpu_active_for_current_layer() {
        // CPU-resident layer per `[inference].n_gpu_layers` —
        // caller falls back to `k::gqa_attention_flash_prefill`
        // on CPU. No skip-once log: this is the intended path.
        return false;
    }
    if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 || max_ctx == 0 || n_new == 0 {
        log_prefill_attn_skip_once("zero-dim shape");
        return false;
    }
    if n_heads % n_kv_heads != 0 {
        log_prefill_attn_skip_once("n_heads not divisible by n_kv_heads");
        return false;
    }
    if kv_len_base + n_new > max_ctx {
        log_prefill_attn_skip_once("kv_len_base + n_new > max_ctx");
        return false;
    }
    let need_q = n_new * n_heads * head_dim;
    let need_kv = n_kv_heads * max_ctx * head_dim;
    let need_out = need_q;
    if q.len() != need_q
        || k_cache.len() != need_kv
        || v_cache.len() != need_kv
        || out.len() != need_out
    {
        log_prefill_attn_skip_once("input slice length mismatch");
        return false;
    }
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(ctx) = slot.as_mut() else {
            log_prefill_attn_skip_once("USM_ATTN slot is None (prepare_usm_context failed)");
            return false;
        };
        if !ensure_scratch_f32(&ctx.stream, &mut ctx.prefill_q_f32, need_q)
            || !ensure_scratch_f32(&ctx.stream, &mut ctx.prefill_k_f32, need_kv)
            || !ensure_scratch_f32(&ctx.stream, &mut ctx.prefill_v_f32, need_kv)
            || !ensure_scratch_f32(&ctx.stream, &mut ctx.prefill_out_f32, need_out)
        {
            log_prefill_attn_skip_once("prefill scratch alloc failed");
            return false;
        }
        // Re-upload Q / K / V on every call. K/V residency across
        // calls is a follow-up optimization (see fn doc).
        ctx.prefill_q_f32.as_mut().unwrap().as_mut_slice()[..need_q]
            .copy_from_slice(q);
        ctx.prefill_k_f32.as_mut().unwrap().as_mut_slice()[..need_kv]
            .copy_from_slice(k_cache);
        ctx.prefill_v_f32.as_mut().unwrap().as_mut_slice()[..need_kv]
            .copy_from_slice(v_cache);
        let stream_raw: *const sk::SyclStream = &ctx.stream;
        let q_ptr = ctx.prefill_q_f32.as_ref().unwrap().as_ptr();
        let k_ptr = ctx.prefill_k_f32.as_ref().unwrap().as_ptr();
        let v_ptr = ctx.prefill_v_f32.as_ref().unwrap().as_ptr();
        let out_ptr = ctx.prefill_out_f32.as_mut().unwrap().as_mut_ptr();
        // FlashAttention dispatch chain: v3 → v2 → v1 (see the
        // decode-side comment in `try_flash_attn_decode_usm_f32` for
        // env-var contract).
        // SAFETY for all calls below: USM pointers come from
        // `ctx.stream`'s allocator; sizes validated by
        // ensure_scratch_f32 above; the kernel `.wait()`s before
        // returning.
        let want_v3 = fa_v3_enabled();
        let want_v2 = fa_v2_enabled();
        let v3_ok = want_v3
            && unsafe {
                sk::flash_attn_prefill_v3_usm_raw(
                    &*stream_raw, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as u32, n_kv_heads as u32,
                    head_dim as u32, max_ctx as u32,
                    kv_len_base as u32, n_new as u32,
                )
            }
            .is_ok();
        let v2_ok = !v3_ok && want_v2
            && unsafe {
                sk::flash_attn_prefill_v2_usm_raw(
                    &*stream_raw, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as u32, n_kv_heads as u32,
                    head_dim as u32, max_ctx as u32,
                    kv_len_base as u32, n_new as u32,
                )
            }
            .is_ok();
        if !v3_ok && !v2_ok {
            let result = unsafe {
                sk::flash_attn_prefill_usm_raw(
                    &*stream_raw, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as u32, n_kv_heads as u32,
                    head_dim as u32, max_ctx as u32,
                    kv_len_base as u32, n_new as u32,
                )
            };
            if result.is_err() {
                log_prefill_attn_skip_once("kernel call returned Err");
                return false;
            }
        }
        out.copy_from_slice(
            &ctx.prefill_out_f32.as_ref().unwrap().as_slice()[..need_out],
        );
        static FIRST_PREFILL_DISPATCH: std::sync::OnceLock<()> =
            std::sync::OnceLock::new();
        FIRST_PREFILL_DISPATCH.get_or_init(|| {
            tracing::info!(
                n_new,
                kv_len_base,
                n_heads,
                n_kv_heads,
                head_dim,
                "first GPU prefill attention dispatch — kernel live"
            );
        });
        true
    })
}

// ==================================================================
// Quantized-KV FlashAttention PREFILL (SYCL-USM + native CUDA)
// ==================================================================
//
// The batched-prefill analogue of the quantized-KV decode helpers. Where
// decode writes ONE new row into a persistent per-layer mirror and walks
// `[0, pos+1)`, prefill re-uploads the WHOLE packed K/V slab the engine
// already holds (`[n_kv_heads, max_ctx, bytes_per_row]`) and the kernel
// walks each query `q_pos in [0, n_new)` over its causal window
// `[0, kv_len_base + q_pos]`. This mirrors `try_flash_attn_prefill_usm_f32`
// exactly (whole-slab re-upload, no incremental valid-len gate) — so there
// is no staleness epoch here: the caller hands us the full, current slab.
//
// KEY DEVIATION FROM DECODE (intentional, flagged in the task report): the
// decode helper receives the NEW f32 K/V rows and re-quantizes them into
// its own mirror. Prefill CANNOT do that — the raw f32 K/V for earlier
// rows is long gone by the time the sub-chunk attends (only the packed
// cache survives), so re-quantizing would not be byte-exact with the CPU
// reference. Instead prefill takes the ALREADY-PACKED slab (+ optional
// per-row scales) straight from the engine's `KvLayer`, uploads it
// verbatim, and runs the kernel — byte-identical to what the CPU prefill
// kernel reads, so GPU == CPU within the fp-reduction-order tolerance the
// F32 path already carries. The f32 Q / output scratch is shared with the
// F32 prefill helper (only one prefill runs at a time).

/// Shared body for every SYCL-USM quantized-KV prefill helper. `q` / `out`
/// are f32 `[n_new, n_heads, head_dim]`; `k_packed` / `v_packed` are the
/// packed slab `[n_kv_heads, max_ctx, bytes_per_row]`; `k_scales` /
/// `v_scales` are `[n_kv_heads * max_ctx]` per-row f32 scales (empty slices
/// for the formats that embed their scale in the packed bytes). Returns
/// `true` iff the GPU produced the output (in `out`); `false` (→ caller's
/// CPU prefill) for any gate / shape / alloc / kernel failure. Relies on
/// the USM context already being built by `prepare_usm_context` (same as
/// the F32 prefill helper); if it is not, declines to CPU.
#[allow(clippy::too_many_arguments)]
fn try_flash_attn_prefill_usm_quant(
    fmt: QuantKv,
    q: &[f32],
    k_packed: &[u8],
    v_packed: &[u8],
    k_scales: &[f32],
    v_scales: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) -> bool {
    if !usm_attn_enabled() {
        log_prefill_attn_skip_once("usm_attn_enabled = false");
        return false;
    }
    if !gpu_active_for_current_layer() {
        return false;
    }
    if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 || max_ctx == 0 || n_new == 0 {
        log_prefill_attn_skip_once("zero-dim shape");
        return false;
    }
    if n_heads % n_kv_heads != 0 {
        log_prefill_attn_skip_once("n_heads not divisible by n_kv_heads");
        return false;
    }
    if kv_len_base + n_new > max_ctx {
        log_prefill_attn_skip_once("kv_len_base + n_new > max_ctx");
        return false;
    }
    // Per-format block constraint (same rules as the quant decode helper +
    // each kernel's own asserts — decline instead of risking a device-side
    // assert).
    let ok_shape = match fmt {
        QuantKv::Q4_0 => head_dim % 32 == 0,
        QuantKv::Q8_0 => true,
        QuantKv::Nvfp4 => head_dim % 16 == 0,
        QuantKv::Tq { bits } => matches!(bits, 1 | 2 | 4 | 8) && head_dim.is_power_of_two(),
        QuantKv::Mxfp4 | QuantKv::Mxfp6 | QuantKv::Mxfp8 => head_dim % 32 == 0,
    };
    if !ok_shape {
        log_prefill_attn_skip_once("quant prefill block-shape constraint");
        return false;
    }
    let bytes_per_row = fmt.bytes_per_row(head_dim);
    let need_q = n_new * n_heads * head_dim;
    let need_kv = n_kv_heads * max_ctx * bytes_per_row;
    let need_scales = n_kv_heads * max_ctx;
    let need_out = need_q;
    let has_scales = fmt.has_scales();
    if q.len() != need_q
        || k_packed.len() != need_kv
        || v_packed.len() != need_kv
        || out.len() != need_out
    {
        log_prefill_attn_skip_once("quant prefill input slice length mismatch");
        return false;
    }
    if has_scales && (k_scales.len() != need_scales || v_scales.len() != need_scales) {
        log_prefill_attn_skip_once("quant prefill scales length mismatch");
        return false;
    }
    USM_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(ctx) = slot.as_mut() else {
            log_prefill_attn_skip_once("USM_ATTN slot is None (prepare_usm_context failed)");
            return false;
        };
        // f32 Q/out reuse the F32 prefill slabs; packed K/V + scales use
        // the dedicated quant-prefill scratch.
        if !ensure_scratch_f32(&ctx.stream, &mut ctx.prefill_q_f32, need_q)
            || !ensure_scratch_f32(&ctx.stream, &mut ctx.prefill_out_f32, need_out)
            || !ensure_u8_exact(&ctx.stream, &mut ctx.prefill_k_packed, need_kv)
            || !ensure_u8_exact(&ctx.stream, &mut ctx.prefill_v_packed, need_kv)
        {
            log_prefill_attn_skip_once("quant prefill scratch alloc failed");
            return false;
        }
        if has_scales
            && (!ensure_scratch_f32(&ctx.stream, &mut ctx.prefill_k_scales, need_scales)
                || !ensure_scratch_f32(&ctx.stream, &mut ctx.prefill_v_scales, need_scales))
        {
            log_prefill_attn_skip_once("quant prefill scales alloc failed");
            return false;
        }
        // Upload Q + the whole packed K/V slab (+ scales) on every call —
        // the prefill residency model (see the F32 prefill helper's doc).
        ctx.prefill_q_f32.as_mut().unwrap().as_mut_slice()[..need_q].copy_from_slice(q);
        ctx.prefill_k_packed.as_mut().unwrap().as_mut_slice()[..need_kv].copy_from_slice(k_packed);
        ctx.prefill_v_packed.as_mut().unwrap().as_mut_slice()[..need_kv].copy_from_slice(v_packed);
        if has_scales {
            ctx.prefill_k_scales.as_mut().unwrap().as_mut_slice()[..need_scales]
                .copy_from_slice(k_scales);
            ctx.prefill_v_scales.as_mut().unwrap().as_mut_slice()[..need_scales]
                .copy_from_slice(v_scales);
        }
        // Raw-pointer dance to avoid simultaneous &mut borrows across
        // fields — the same pattern the F32 prefill helper uses. SAFETY:
        // all USM pointers come from `ctx.stream`'s allocator, sized by the
        // ensure_* calls above; the kernel `.wait()`s before returning.
        let stream_raw: *const sk::SyclStream = &ctx.stream;
        let q_ptr = ctx.prefill_q_f32.as_ref().unwrap().as_ptr();
        let k_ptr = ctx.prefill_k_packed.as_ref().unwrap().as_ptr();
        let v_ptr = ctx.prefill_v_packed.as_ref().unwrap().as_ptr();
        let out_ptr = ctx.prefill_out_f32.as_mut().unwrap().as_mut_ptr();
        let (ks_ptr, vs_ptr) = if has_scales {
            (
                ctx.prefill_k_scales.as_ref().unwrap().as_ptr(),
                ctx.prefill_v_scales.as_ref().unwrap().as_ptr(),
            )
        } else {
            (std::ptr::null::<f32>(), std::ptr::null::<f32>())
        };
        let ok = unsafe {
            match fmt {
                QuantKv::Q4_0 => sk::flash_attn_prefill_q4_0_usm_raw(
                    &*stream_raw, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as u32, n_kv_heads as u32, head_dim as u32,
                    max_ctx as u32, kv_len_base as u32, n_new as u32,
                ),
                QuantKv::Q8_0 => sk::flash_attn_prefill_q8_0_usm_raw(
                    &*stream_raw, q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, out_ptr,
                    n_heads as u32, n_kv_heads as u32, head_dim as u32,
                    max_ctx as u32, kv_len_base as u32, n_new as u32,
                ),
                QuantKv::Nvfp4 => sk::flash_attn_prefill_nvfp4_usm_raw(
                    &*stream_raw, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as u32, n_kv_heads as u32, head_dim as u32,
                    max_ctx as u32, kv_len_base as u32, n_new as u32,
                ),
                QuantKv::Tq { bits } => sk::flash_attn_prefill_tq_usm_raw(
                    &*stream_raw, q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, bits as u32, out_ptr,
                    n_heads as u32, n_kv_heads as u32, head_dim as u32,
                    max_ctx as u32, kv_len_base as u32, n_new as u32,
                ),
                QuantKv::Mxfp4 => sk::flash_attn_prefill_mxfp4_usm_raw(
                    &*stream_raw, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as u32, n_kv_heads as u32, head_dim as u32,
                    max_ctx as u32, kv_len_base as u32, n_new as u32,
                ),
                QuantKv::Mxfp6 => sk::flash_attn_prefill_mxfp6_usm_raw(
                    &*stream_raw, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as u32, n_kv_heads as u32, head_dim as u32,
                    max_ctx as u32, kv_len_base as u32, n_new as u32,
                ),
                QuantKv::Mxfp8 => sk::flash_attn_prefill_mxfp8_usm_raw(
                    &*stream_raw, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads as u32, n_kv_heads as u32, head_dim as u32,
                    max_ctx as u32, kv_len_base as u32, n_new as u32,
                ),
            }
        }
        .is_ok();
        if !ok {
            log_prefill_attn_skip_once("quant prefill kernel call returned Err");
            return false;
        }
        out.copy_from_slice(&ctx.prefill_out_f32.as_ref().unwrap().as_slice()[..need_out]);
        true
    })
}

/// Try to run F32 KV flash-attention PREFILL on the native CUDA backend.
/// Same public signature as [`try_flash_attn_prefill_usm_f32`]. `k` / `v`
/// are the full f32 slab `[n_kv_heads, max_ctx, head_dim]` the caller
/// already holds; the whole slab + Q are uploaded and the kernel walks each
/// query's causal window. `true` iff the GPU produced `out`; else the
/// combinator falls through to SYCL / CPU.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_prefill_cuda_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) -> bool {
    if !cuda_active() || !gpu_active_for_current_layer() {
        return false;
    }
    if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 || max_ctx == 0 || n_new == 0 {
        return false;
    }
    if n_heads % n_kv_heads != 0 || kv_len_base + n_new > max_ctx {
        return false;
    }
    let need_q = n_new * n_heads * head_dim;
    let need_kv = n_kv_heads * max_ctx * head_dim;
    if q.len() != need_q || k.len() != need_kv || v.len() != need_kv || out.len() != need_q {
        return false;
    }
    let mirror_bytes = need_kv * 4;
    let q_bytes = need_q * 4;
    let out_bytes = need_q * 4;
    CUDA_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        // Prefill runs before any decode, so the context is normally None
        // here. Build a minimal one (n_layers = 1: prefill never indexes
        // the per-layer decode shadows) purely to own a stream + the shared
        // prefill scratch. A later decode with the real per-model config
        // rebuilds harmlessly (prefill's scratch is done with by then).
        if slot.is_none() {
            *slot = CudaAttnContext::try_new(CudaAttnConfig {
                n_layers: 1,
                n_heads: n_heads as u32,
                n_kv_heads: n_kv_heads as u32,
                head_dim: head_dim as u32,
                max_ctx: max_ctx as u32,
            });
            if slot.is_none() {
                return false;
            }
        }
        let ctx = slot.as_mut().expect("cuda ctx just built");
        if !ensure_cuda_dev(&mut ctx.dev_prefill_k, &ctx.stream, mirror_bytes)
            || !ensure_cuda_dev(&mut ctx.dev_prefill_v, &ctx.stream, mirror_bytes)
            || !ensure_cuda_dev(&mut ctx.dev_prefill_q, &ctx.stream, q_bytes)
            || !ensure_cuda_dev(&mut ctx.dev_prefill_out, &ctx.stream, out_bytes)
        {
            return false;
        }
        {
            let s: &[u8] =
                unsafe { std::slice::from_raw_parts(k.as_ptr() as *const u8, mirror_bytes) };
            if ctx.dev_prefill_k.as_mut().unwrap().copy_from_host(s).is_err() {
                return false;
            }
        }
        {
            let s: &[u8] =
                unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, mirror_bytes) };
            if ctx.dev_prefill_v.as_mut().unwrap().copy_from_host(s).is_err() {
                return false;
            }
        }
        {
            let s: &[u8] = unsafe { std::slice::from_raw_parts(q.as_ptr() as *const u8, q_bytes) };
            if ctx.dev_prefill_q.as_mut().unwrap().copy_from_host(s).is_err() {
                return false;
            }
        }
        let stream = &ctx.stream;
        let q_ptr = ctx.dev_prefill_q.as_ref().unwrap().as_ptr() as *const f32;
        let k_ptr = ctx.dev_prefill_k.as_ref().unwrap().as_ptr() as *const f32;
        let v_ptr = ctx.dev_prefill_v.as_ref().unwrap().as_ptr() as *const f32;
        let out_ptr = ctx.dev_prefill_out.as_mut().unwrap().as_mut_ptr() as *mut f32;
        // SAFETY: live device pointers on `stream`, sized above; the wrapper
        // synchronizes before returning so the read-back doesn't race.
        let launched = unsafe {
            ck::flash_attn_prefill_f32(
                stream, q_ptr, k_ptr, v_ptr, out_ptr,
                n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
            )
        }
        .is_ok();
        if !launched || ck::consume_error_count() != 0 {
            return false;
        }
        let out_b: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out_bytes) };
        ctx.dev_prefill_out.as_ref().unwrap().copy_to_host(out_b).is_ok()
    })
}

/// Shared body for every native-CUDA quantized-KV prefill helper. Mirrors
/// [`try_flash_attn_prefill_usm_quant`] (packed slab in, f32 out) but
/// uploads to device scratch instead of USM-shared. `k_packed` / `v_packed`
/// are the packed slab bytes; `k_scales` / `v_scales` the per-row f32
/// scales (empty for scale-embedded formats). `true` iff the GPU produced
/// `out`.
#[allow(clippy::too_many_arguments)]
fn try_flash_attn_prefill_cuda_quant(
    fmt: QuantKv,
    q: &[f32],
    k_packed: &[u8],
    v_packed: &[u8],
    k_scales: &[f32],
    v_scales: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) -> bool {
    if !cuda_active() || !gpu_active_for_current_layer() {
        return false;
    }
    if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 || max_ctx == 0 || n_new == 0 {
        return false;
    }
    if n_heads % n_kv_heads != 0 || kv_len_base + n_new > max_ctx {
        return false;
    }
    let ok_shape = match fmt {
        QuantKv::Q4_0 => head_dim % 32 == 0,
        QuantKv::Q8_0 => true,
        QuantKv::Nvfp4 => head_dim % 16 == 0,
        QuantKv::Tq { bits } => matches!(bits, 1 | 2 | 4 | 8) && head_dim.is_power_of_two(),
        QuantKv::Mxfp4 | QuantKv::Mxfp6 | QuantKv::Mxfp8 => head_dim % 32 == 0,
    };
    if !ok_shape {
        return false;
    }
    let bytes_per_row = fmt.bytes_per_row(head_dim);
    let need_q = n_new * n_heads * head_dim;
    let need_kv = n_kv_heads * max_ctx * bytes_per_row;
    let need_scales = n_kv_heads * max_ctx;
    let has_scales = fmt.has_scales();
    if q.len() != need_q
        || k_packed.len() != need_kv
        || v_packed.len() != need_kv
        || out.len() != need_q
    {
        return false;
    }
    if has_scales && (k_scales.len() != need_scales || v_scales.len() != need_scales) {
        return false;
    }
    let q_bytes = need_q * 4;
    let out_bytes = need_q * 4;
    let scales_bytes = need_scales * 4;
    CUDA_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = CudaAttnContext::try_new(CudaAttnConfig {
                n_layers: 1,
                n_heads: n_heads as u32,
                n_kv_heads: n_kv_heads as u32,
                head_dim: head_dim as u32,
                max_ctx: max_ctx as u32,
            });
            if slot.is_none() {
                return false;
            }
        }
        let ctx = slot.as_mut().expect("cuda ctx just built");
        if !ensure_cuda_dev(&mut ctx.dev_prefill_k, &ctx.stream, need_kv)
            || !ensure_cuda_dev(&mut ctx.dev_prefill_v, &ctx.stream, need_kv)
            || !ensure_cuda_dev(&mut ctx.dev_prefill_q, &ctx.stream, q_bytes)
            || !ensure_cuda_dev(&mut ctx.dev_prefill_out, &ctx.stream, out_bytes)
        {
            return false;
        }
        if has_scales
            && (!ensure_cuda_dev(&mut ctx.dev_prefill_k_scales, &ctx.stream, scales_bytes)
                || !ensure_cuda_dev(&mut ctx.dev_prefill_v_scales, &ctx.stream, scales_bytes))
        {
            return false;
        }
        // Upload the packed slab (+ scales) + Q verbatim. `k_packed` etc.
        // are caller-owned slices (not `ctx` fields), so there is no
        // aliasing with the device buffers they copy into.
        if ctx.dev_prefill_k.as_mut().unwrap().copy_from_host(k_packed).is_err() {
            return false;
        }
        if ctx.dev_prefill_v.as_mut().unwrap().copy_from_host(v_packed).is_err() {
            return false;
        }
        if has_scales {
            {
                let s: &[u8] = unsafe {
                    std::slice::from_raw_parts(k_scales.as_ptr() as *const u8, scales_bytes)
                };
                if ctx.dev_prefill_k_scales.as_mut().unwrap().copy_from_host(s).is_err() {
                    return false;
                }
            }
            {
                let s: &[u8] = unsafe {
                    std::slice::from_raw_parts(v_scales.as_ptr() as *const u8, scales_bytes)
                };
                if ctx.dev_prefill_v_scales.as_mut().unwrap().copy_from_host(s).is_err() {
                    return false;
                }
            }
        }
        {
            let s: &[u8] = unsafe { std::slice::from_raw_parts(q.as_ptr() as *const u8, q_bytes) };
            if ctx.dev_prefill_q.as_mut().unwrap().copy_from_host(s).is_err() {
                return false;
            }
        }
        let stream = &ctx.stream;
        let q_ptr = ctx.dev_prefill_q.as_ref().unwrap().as_ptr() as *const f32;
        let k_ptr = ctx.dev_prefill_k.as_ref().unwrap().as_ptr();
        let v_ptr = ctx.dev_prefill_v.as_ref().unwrap().as_ptr();
        let out_ptr = ctx.dev_prefill_out.as_mut().unwrap().as_mut_ptr() as *mut f32;
        let (ks_ptr, vs_ptr) = if has_scales {
            (
                ctx.dev_prefill_k_scales.as_ref().unwrap().as_ptr() as *const f32,
                ctx.dev_prefill_v_scales.as_ref().unwrap().as_ptr() as *const f32,
            )
        } else {
            (std::ptr::null::<f32>(), std::ptr::null::<f32>())
        };
        // SAFETY: live device pointers on `stream`, sized above; the
        // wrappers synchronize before returning.
        let launched = unsafe {
            match fmt {
                QuantKv::Q4_0 => ck::flash_attn_prefill_q4_0(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
                ),
                QuantKv::Q8_0 => ck::flash_attn_prefill_q8_0(
                    stream, q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
                ),
                QuantKv::Nvfp4 => ck::flash_attn_prefill_nvfp4(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
                ),
                QuantKv::Tq { bits } => ck::flash_attn_prefill_tq(
                    stream, q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, bits as u32, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
                ),
                QuantKv::Mxfp4 => ck::flash_attn_prefill_mxfp4(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
                ),
                QuantKv::Mxfp6 => ck::flash_attn_prefill_mxfp6(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
                ),
                QuantKv::Mxfp8 => ck::flash_attn_prefill_mxfp8(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
                ),
            }
        }
        .is_ok();
        if !launched || ck::consume_error_count() != 0 {
            return false;
        }
        let out_b: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out_bytes) };
        ctx.dev_prefill_out.as_ref().unwrap().copy_to_host(out_b).is_ok()
    })
}

/// F32 KV flash-attention prefill on the native MLX backend. Mirror of
/// [`try_flash_attn_prefill_cuda_f32`] over the `mk::` API. `true` iff the
/// GPU produced `out`.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_prefill_mlx_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) -> bool {
    if !mlx_active() || !gpu_active_for_current_layer() {
        return false;
    }
    if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 || max_ctx == 0 || n_new == 0 {
        return false;
    }
    if n_heads % n_kv_heads != 0 || kv_len_base + n_new > max_ctx {
        return false;
    }
    let need_q = n_new * n_heads * head_dim;
    let need_kv = n_kv_heads * max_ctx * head_dim;
    if q.len() != need_q || k.len() != need_kv || v.len() != need_kv || out.len() != need_q {
        return false;
    }
    let mirror_bytes = need_kv * 4;
    let q_bytes = need_q * 4;
    let out_bytes = need_q * 4;
    MLX_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        // Prefill runs before any decode, so the context is normally None
        // here. Build a minimal one (n_layers = 1: prefill never indexes the
        // per-layer decode shadows) purely to own a stream + prefill scratch.
        if slot.is_none() {
            *slot = MlxAttnContext::try_new(MlxAttnConfig {
                n_layers: 1,
                n_heads: n_heads as u32,
                n_kv_heads: n_kv_heads as u32,
                head_dim: head_dim as u32,
                max_ctx: max_ctx as u32,
            });
            if slot.is_none() {
                return false;
            }
        }
        let ctx = slot.as_mut().expect("mlx ctx just built");
        if !ensure_mlx_dev(&mut ctx.dev_prefill_k, &ctx.stream, mirror_bytes)
            || !ensure_mlx_dev(&mut ctx.dev_prefill_v, &ctx.stream, mirror_bytes)
            || !ensure_mlx_dev(&mut ctx.dev_prefill_q, &ctx.stream, q_bytes)
            || !ensure_mlx_dev(&mut ctx.dev_prefill_out, &ctx.stream, out_bytes)
        {
            return false;
        }
        {
            let s: &[u8] =
                unsafe { std::slice::from_raw_parts(k.as_ptr() as *const u8, mirror_bytes) };
            if ctx.dev_prefill_k.as_mut().unwrap().copy_from_host(s).is_err() {
                return false;
            }
        }
        {
            let s: &[u8] =
                unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, mirror_bytes) };
            if ctx.dev_prefill_v.as_mut().unwrap().copy_from_host(s).is_err() {
                return false;
            }
        }
        {
            let s: &[u8] = unsafe { std::slice::from_raw_parts(q.as_ptr() as *const u8, q_bytes) };
            if ctx.dev_prefill_q.as_mut().unwrap().copy_from_host(s).is_err() {
                return false;
            }
        }
        let stream = &ctx.stream;
        let q_ptr = ctx.dev_prefill_q.as_ref().unwrap().as_ptr() as *const f32;
        let k_ptr = ctx.dev_prefill_k.as_ref().unwrap().as_ptr() as *const f32;
        let v_ptr = ctx.dev_prefill_v.as_ref().unwrap().as_ptr() as *const f32;
        let out_ptr = ctx.dev_prefill_out.as_mut().unwrap().as_mut_ptr() as *mut f32;
        // SAFETY: live device pointers on `stream`, sized above; the wrapper
        // synchronizes before returning so the read-back doesn't race.
        let launched = unsafe {
            mk::flash_attn_prefill_f32(
                stream, q_ptr, k_ptr, v_ptr, out_ptr,
                n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
            )
        }
        .is_ok();
        if !launched || mk::consume_error_count() != 0 {
            return false;
        }
        let out_b: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out_bytes) };
        ctx.dev_prefill_out.as_ref().unwrap().copy_to_host(out_b).is_ok()
    })
}

/// Shared body for every native-MLX quantized-KV prefill helper. Mirror of
/// [`try_flash_attn_prefill_cuda_quant`] over the `mk::` API. `k_packed` /
/// `v_packed` are the packed slab bytes; `k_scales` / `v_scales` the per-row
/// f32 scales (empty for scale-embedded formats). `true` iff the GPU
/// produced `out`.
#[allow(clippy::too_many_arguments)]
fn try_flash_attn_prefill_mlx_quant(
    fmt: QuantKv,
    q: &[f32],
    k_packed: &[u8],
    v_packed: &[u8],
    k_scales: &[f32],
    v_scales: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) -> bool {
    if !mlx_active() || !gpu_active_for_current_layer() {
        return false;
    }
    if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 || max_ctx == 0 || n_new == 0 {
        return false;
    }
    if n_heads % n_kv_heads != 0 || kv_len_base + n_new > max_ctx {
        return false;
    }
    let ok_shape = match fmt {
        QuantKv::Q4_0 => head_dim % 32 == 0,
        QuantKv::Q8_0 => true,
        QuantKv::Nvfp4 => head_dim % 16 == 0,
        QuantKv::Tq { bits } => matches!(bits, 1 | 2 | 4 | 8) && head_dim.is_power_of_two(),
        QuantKv::Mxfp4 | QuantKv::Mxfp6 | QuantKv::Mxfp8 => head_dim % 32 == 0,
    };
    if !ok_shape {
        return false;
    }
    let bytes_per_row = fmt.bytes_per_row(head_dim);
    let need_q = n_new * n_heads * head_dim;
    let need_kv = n_kv_heads * max_ctx * bytes_per_row;
    let need_scales = n_kv_heads * max_ctx;
    let has_scales = fmt.has_scales();
    if q.len() != need_q
        || k_packed.len() != need_kv
        || v_packed.len() != need_kv
        || out.len() != need_q
    {
        return false;
    }
    if has_scales && (k_scales.len() != need_scales || v_scales.len() != need_scales) {
        return false;
    }
    let q_bytes = need_q * 4;
    let out_bytes = need_q * 4;
    let scales_bytes = need_scales * 4;
    MLX_ATTN.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = MlxAttnContext::try_new(MlxAttnConfig {
                n_layers: 1,
                n_heads: n_heads as u32,
                n_kv_heads: n_kv_heads as u32,
                head_dim: head_dim as u32,
                max_ctx: max_ctx as u32,
            });
            if slot.is_none() {
                return false;
            }
        }
        let ctx = slot.as_mut().expect("mlx ctx just built");
        if !ensure_mlx_dev(&mut ctx.dev_prefill_k, &ctx.stream, need_kv)
            || !ensure_mlx_dev(&mut ctx.dev_prefill_v, &ctx.stream, need_kv)
            || !ensure_mlx_dev(&mut ctx.dev_prefill_q, &ctx.stream, q_bytes)
            || !ensure_mlx_dev(&mut ctx.dev_prefill_out, &ctx.stream, out_bytes)
        {
            return false;
        }
        if has_scales
            && (!ensure_mlx_dev(&mut ctx.dev_prefill_k_scales, &ctx.stream, scales_bytes)
                || !ensure_mlx_dev(&mut ctx.dev_prefill_v_scales, &ctx.stream, scales_bytes))
        {
            return false;
        }
        // Upload the packed slab (+ scales) + Q verbatim. `k_packed` etc. are
        // caller-owned slices (not `ctx` fields), so there is no aliasing
        // with the device buffers they copy into.
        if ctx.dev_prefill_k.as_mut().unwrap().copy_from_host(k_packed).is_err() {
            return false;
        }
        if ctx.dev_prefill_v.as_mut().unwrap().copy_from_host(v_packed).is_err() {
            return false;
        }
        if has_scales {
            {
                let s: &[u8] = unsafe {
                    std::slice::from_raw_parts(k_scales.as_ptr() as *const u8, scales_bytes)
                };
                if ctx.dev_prefill_k_scales.as_mut().unwrap().copy_from_host(s).is_err() {
                    return false;
                }
            }
            {
                let s: &[u8] = unsafe {
                    std::slice::from_raw_parts(v_scales.as_ptr() as *const u8, scales_bytes)
                };
                if ctx.dev_prefill_v_scales.as_mut().unwrap().copy_from_host(s).is_err() {
                    return false;
                }
            }
        }
        {
            let s: &[u8] = unsafe { std::slice::from_raw_parts(q.as_ptr() as *const u8, q_bytes) };
            if ctx.dev_prefill_q.as_mut().unwrap().copy_from_host(s).is_err() {
                return false;
            }
        }
        let stream = &ctx.stream;
        let q_ptr = ctx.dev_prefill_q.as_ref().unwrap().as_ptr() as *const f32;
        let k_ptr = ctx.dev_prefill_k.as_ref().unwrap().as_ptr();
        let v_ptr = ctx.dev_prefill_v.as_ref().unwrap().as_ptr();
        let out_ptr = ctx.dev_prefill_out.as_mut().unwrap().as_mut_ptr() as *mut f32;
        let (ks_ptr, vs_ptr) = if has_scales {
            (
                ctx.dev_prefill_k_scales.as_ref().unwrap().as_ptr() as *const f32,
                ctx.dev_prefill_v_scales.as_ref().unwrap().as_ptr() as *const f32,
            )
        } else {
            (std::ptr::null::<f32>(), std::ptr::null::<f32>())
        };
        // SAFETY: live device pointers on `stream`, sized above; the wrappers
        // synchronize before returning.
        let launched = unsafe {
            match fmt {
                QuantKv::Q4_0 => mk::flash_attn_prefill_q4_0(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
                ),
                QuantKv::Q8_0 => mk::flash_attn_prefill_q8_0(
                    stream, q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
                ),
                QuantKv::Nvfp4 => mk::flash_attn_prefill_nvfp4(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
                ),
                QuantKv::Tq { bits } => mk::flash_attn_prefill_tq(
                    stream, q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, bits as u32, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
                ),
                QuantKv::Mxfp4 => mk::flash_attn_prefill_mxfp4(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
                ),
                QuantKv::Mxfp6 => mk::flash_attn_prefill_mxfp6(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
                ),
                QuantKv::Mxfp8 => mk::flash_attn_prefill_mxfp8(
                    stream, q_ptr, k_ptr, v_ptr, out_ptr,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
                ),
            }
        }
        .is_ok();
        if !launched || mk::consume_error_count() != 0 {
            return false;
        }
        let out_b: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out_bytes) };
        ctx.dev_prefill_out.as_ref().unwrap().copy_to_host(out_b).is_ok()
    })
}

// ------------------------------------------------------------------
// GPU flash-attention PREFILL dispatchers (CUDA → SYCL → CPU)
// ------------------------------------------------------------------
//
// One combinator per KV dtype, mirroring the decode dispatchers: try native
// CUDA first when a CUDA device is present, else the SYCL-USM path, else
// (on `false`) the caller's CPU prefill kernel. On a SYCL-only / CPU-only
// host the `cuda_active()` short-circuit makes these behave exactly like
// the SYCL helper — no F32 / SYCL / CPU regression.

/// F32 KV flash-attention prefill: CUDA → SYCL → (caller's CPU on `false`).
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_prefill_gpu_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) -> bool {
    if cuda_active()
        && try_flash_attn_prefill_cuda_f32(
            q, k, v, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
        )
    {
        return true;
    }
    if mlx_active()
        && try_flash_attn_prefill_mlx_f32(
            q, k, v, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
        )
    {
        return true;
    }
    try_flash_attn_prefill_usm_f32(
        q, k, v, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
    )
}

/// Q4_0 KV flash-attention prefill: CUDA → SYCL → (caller's CPU).
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_prefill_gpu_q4_0(
    q: &[f32],
    k_packed: &[u8],
    v_packed: &[u8],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) -> bool {
    if cuda_active()
        && try_flash_attn_prefill_cuda_quant(
            QuantKv::Q4_0, q, k_packed, v_packed, &[], &[], out,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
        )
    {
        return true;
    }
    if mlx_active()
        && try_flash_attn_prefill_mlx_quant(
            QuantKv::Q4_0, q, k_packed, v_packed, &[], &[], out,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
        )
    {
        return true;
    }
    try_flash_attn_prefill_usm_quant(
        QuantKv::Q4_0, q, k_packed, v_packed, &[], &[], out,
        n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
    )
}

/// Q8_0 KV flash-attention prefill: CUDA → SYCL → (caller's CPU). The i8
/// KV slabs are reinterpreted as raw bytes (identical layout — the kernels
/// read them back as i8 internally).
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_prefill_gpu_q8_0(
    q: &[f32],
    k_q: &[i8],
    k_scales: &[f32],
    v_q: &[i8],
    v_scales: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) -> bool {
    let k_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(k_q.as_ptr() as *const u8, k_q.len()) };
    let v_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(v_q.as_ptr() as *const u8, v_q.len()) };
    if cuda_active()
        && try_flash_attn_prefill_cuda_quant(
            QuantKv::Q8_0, q, k_bytes, v_bytes, k_scales, v_scales, out,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
        )
    {
        return true;
    }
    if mlx_active()
        && try_flash_attn_prefill_mlx_quant(
            QuantKv::Q8_0, q, k_bytes, v_bytes, k_scales, v_scales, out,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
        )
    {
        return true;
    }
    try_flash_attn_prefill_usm_quant(
        QuantKv::Q8_0, q, k_bytes, v_bytes, k_scales, v_scales, out,
        n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
    )
}

/// NVFP4 KV flash-attention prefill: CUDA → SYCL → (caller's CPU).
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_prefill_gpu_nvfp4(
    q: &[f32],
    k_packed: &[u8],
    v_packed: &[u8],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) -> bool {
    if cuda_active()
        && try_flash_attn_prefill_cuda_quant(
            QuantKv::Nvfp4, q, k_packed, v_packed, &[], &[], out,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
        )
    {
        return true;
    }
    if mlx_active()
        && try_flash_attn_prefill_mlx_quant(
            QuantKv::Nvfp4, q, k_packed, v_packed, &[], &[], out,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
        )
    {
        return true;
    }
    try_flash_attn_prefill_usm_quant(
        QuantKv::Nvfp4, q, k_packed, v_packed, &[], &[], out,
        n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
    )
}

/// MXFP4/6/8 KV flash-attention prefill: CUDA → SYCL → (caller's CPU).
macro_rules! mxfp_prefill_gpu_combinator {
    ($name:ident, $kv:expr) => {
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            q: &[f32],
            k_packed: &[u8],
            v_packed: &[u8],
            out: &mut [f32],
            n_heads: usize,
            n_kv_heads: usize,
            head_dim: usize,
            max_ctx: usize,
            kv_len_base: usize,
            n_new: usize,
        ) -> bool {
            if cuda_active()
                && try_flash_attn_prefill_cuda_quant(
                    $kv, q, k_packed, v_packed, &[], &[], out,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
                )
            {
                return true;
            }
            if mlx_active()
                && try_flash_attn_prefill_mlx_quant(
                    $kv, q, k_packed, v_packed, &[], &[], out,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
                )
            {
                return true;
            }
            try_flash_attn_prefill_usm_quant(
                $kv, q, k_packed, v_packed, &[], &[], out,
                n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
            )
        }
    };
}
mxfp_prefill_gpu_combinator!(try_flash_attn_prefill_gpu_mxfp4, QuantKv::Mxfp4);
mxfp_prefill_gpu_combinator!(try_flash_attn_prefill_gpu_mxfp6, QuantKv::Mxfp6);
mxfp_prefill_gpu_combinator!(try_flash_attn_prefill_gpu_mxfp8, QuantKv::Mxfp8);

/// TurboQuant KV flash-attention prefill: CUDA → SYCL → (caller's CPU).
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_prefill_gpu_tq(
    q: &[f32],
    k_packed: &[u8],
    k_scales: &[f32],
    v_packed: &[u8],
    v_scales: &[f32],
    bits: u8,
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) -> bool {
    if cuda_active()
        && try_flash_attn_prefill_cuda_quant(
            QuantKv::Tq { bits }, q, k_packed, v_packed, k_scales, v_scales, out,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
        )
    {
        return true;
    }
    if mlx_active()
        && try_flash_attn_prefill_mlx_quant(
            QuantKv::Tq { bits }, q, k_packed, v_packed, k_scales, v_scales, out,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
        )
    {
        return true;
    }
    try_flash_attn_prefill_usm_quant(
        QuantKv::Tq { bits }, q, k_packed, v_packed, k_scales, v_scales, out,
        n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
    )
}

/// The benign SYCL/USM short-circuit reason: the Intel USM subsystem simply is
/// not the active path on this host — no usable Intel GPU, or a CUDA/MLX/CPU
/// host. The call sites pass this exact string; the skip-once loggers match on
/// it to stay quiet (debug-only) rather than emit an alarming "CPU fallback"
/// WARN that reads as a global verdict when it is only the Intel path bowing
/// out of a single call (the CUDA/MLX arm above it does the GPU work).
const USM_REASON_INACTIVE: &str = "usm_attn_enabled = false";

/// Name the active non-SYCL GPU backend, if any. When CUDA or MLX is live the
/// SYCL/USM hooks declining is fully expected — that backend does the GPU work
/// — so the USM skip notices drop to debug level and never read as a global
/// CPU fallback. `None` ⇒ no GPU backend at all (a legitimate CPU-tier host).
fn active_non_sycl_gpu_backend() -> Option<&'static str> {
    if cuda_active() {
        Some("CUDA")
    } else if mlx_active() {
        Some("MLX")
    } else {
        None
    }
}

/// Log the FIRST time the USM attention KV-mirror validity gate
/// declines a GPU dispatch on this process. One line is enough to
/// see the mechanism in `gui.log` (which thread pattern produced a
/// gap); afterwards the declines stay silent — they are the correct,
/// intended CPU fallback, not an error. On a CUDA/MLX host this SYCL
/// attention path is simply unused, so the notice drops to debug.
fn log_usm_kv_gap_once(layer_idx: usize, pos: u32, valid: u32) {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        if let Some(be) = active_non_sycl_gpu_backend() {
            tracing::debug!(
                layer = layer_idx,
                pos,
                valid,
                backend = be,
                "SYCL/USM attention KV-mirror gap — {} is this host's GPU \
                 backend, this SYCL attention path is unused (logged once)",
                be
            );
        } else {
            tracing::info!(
                layer = layer_idx,
                pos,
                valid,
                thread_id = ?std::thread::current().id(),
                "USM attention KV mirror has a row gap on this thread — \
                 declining GPU attention, CPU path takes over (logged once; \
                 this protects against attending over rows another thread or \
                 request wrote)"
            );
        }
    });
}

/// Log the FIRST reason the prefill attention USM hook short-
/// circuited on this process. Separate from the matvec skip-once
/// channels so the user can disambiguate which path fell back when
/// reading `gui.log`.
fn log_prefill_attn_skip_once(reason: &str) {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        if reason == USM_REASON_INACTIVE {
            match active_non_sycl_gpu_backend() {
                Some(be) => tracing::debug!(
                    "SYCL/USM flash-prefill hook inactive — {} is this host's \
                     GPU backend and handles prefill; this SYCL path is unused \
                     (logged once)",
                    be
                ),
                None => tracing::debug!(
                    "SYCL/USM flash-prefill hook inactive — no GPU backend on \
                     this host, using the CPU prefill path (logged once)"
                ),
            }
        } else {
            tracing::warn!(
                reason,
                "SYCL/USM flash-prefill hook short-circuited — CPU prefill for \
                 the affected calls (logged once)"
            );
        }
    });
}

/// Log the FIRST reason the batched matvec USM hook short-circuited
/// on this process. Separate from `log_matvec_skip_once` so the user
/// sees both per-row and batched paths' failure reasons independently.
fn log_matvec_batched_skip_once(reason: &str) {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        if reason == USM_REASON_INACTIVE {
            match active_non_sycl_gpu_backend() {
                Some(be) => tracing::debug!(
                    "SYCL/USM batched-matvec hook inactive — {} is this host's \
                     GPU backend and handles matvec; this SYCL path is unused \
                     (logged once)",
                    be
                ),
                None => tracing::debug!(
                    "SYCL/USM batched-matvec hook inactive — no GPU backend on \
                     this host (logged once)"
                ),
            }
        } else {
            tracing::warn!(
                reason,
                "SYCL/USM batched-matvec hook short-circuited — falling back to \
                 per-row dispatch (logged once)"
            );
        }
    });
}

/// Log the FIRST reason the matvec USM hook short-circuited on this
/// process. Subsequent skip events with the same OR different
/// reasons stay silent — we just want one diagnostic line in
/// `gui.log` per process to confirm what's happening when the user
/// reports "GPU% spiked but Shared GPU memory stayed flat".
fn log_matvec_skip_once(reason: &str) {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        if reason == USM_REASON_INACTIVE {
            match active_non_sycl_gpu_backend() {
                Some(be) => tracing::debug!(
                    "SYCL/USM matvec hook inactive — {} is this host's GPU \
                     matvec backend; only non-offloaded calls (f32 tensors, \
                     CPU-placed layers) use CPU (logged once)",
                    be
                ),
                None => tracing::debug!(
                    "SYCL/USM matvec hook inactive — no GPU matvec backend on \
                     this host, using the CPU path (logged once)"
                ),
            }
        } else {
            tracing::warn!(
                reason,
                "SYCL/USM matvec hook short-circuited — CPU matvec for the \
                 affected calls (logged once)"
            );
        }
    });
}

/// Log each unique `Dtype` that falls through the matvec USM hook's
/// dtype-match to CPU, exactly once per dtype per process. The
/// resulting `gui.log` paints a full picture of the model's weight
/// dtype distribution — useful when a "Q4_K_M" GGUF turns out to
/// contain a mix of `Q4_K` projection weights + `Q6_K` or `F16`
/// for output / specific layers (common variant patterns).
fn log_unmatched_dtype(dt: Dtype) {
    use std::sync::Mutex;
    static SEEN: std::sync::OnceLock<Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    let key = format!("{:?}", dt);
    let is_new = {
        let mut g = seen.lock().unwrap();
        g.insert(key.clone())
    };
    if is_new {
        tracing::warn!(
            dtype = %key,
            "matvec weight with non-packed-USM dtype — CPU dispatch (each \
             unique dtype logged once per process)"
        );
    }
}
