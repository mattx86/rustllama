//! SYCL kernel parity + stability harness (`rustllama doctor --sycl-parity`).
//!
//! Runs every GPU kernel family against its CPU reference on identical
//! inputs and reports a per-kernel verdict. Born from the 2026-09-06
//! debugging session where two kernel classes shipped wrong silently:
//! on the OpenCL fallback backend several kernels computed garbage
//! while returning `Ok` (the engine's failure latch can't catch
//! wrong-but-successful launches), and on Level Zero the IQ2/IQ3
//! packed matvecs crashed the iGPU outright (UR DEVICE_LOST → TDR).
//!
//! **Subprocess isolation**: a kernel that kills the device poisons
//! the whole SYCL context — an in-process loop would die with its
//! patient and never report the rest of the matrix. So the parent
//! spawns `rustllama doctor --sycl-parity-probe <name>` per kernel:
//! the child prints one `PARITY <name> <verdict> ...` line; a crash
//! or hang is the child's problem and becomes a CRASH / HANG verdict
//! in the parent's table.
//!
//! Verdicts:
//!   OK          GPU output matches the CPU reference within tolerance
//!   MISCOMPUTE  kernel returned Ok but the numbers are wrong — the
//!               dangerous class; never enable this kernel here
//!   KERNEL_ERR  kernel returned Err (clean refusal — the engine's
//!               fallback path handles this safely at runtime)
//!   CRASH       child process died (device lost / TDR / abort)
//!   HANG        child exceeded the per-probe timeout and was killed
//!   SKIP        no SYCL device (mock build or no GPU)
//!
//! Run it once per backend: the plain launch PATH lands on OpenCL;
//! prepending `%ONEAPI_ROOT%\2026.0\bin` makes Level Zero visible
//! (see the plan file — the L0 adapter needs umf.dll's deps from
//! that directory).

use std::io::Write as _;
use std::time::{Duration, Instant};

use rustllama_kernels_cpu as k;
use rustllama_kernels_cuda as ck;
use rustllama_kernels_mlx as mk;
use rustllama_kernels_sycl as sk;

// ---------------------------------------------------------------
// Quant block layouts — enough structure to synthesize valid raw
// bytes: arbitrary quant bits are fine (all IQ codebook indices are
// power-of-two sized, so no out-of-bounds lookups), but f16 scale
// slots must hold sane values or NaN/Inf poisons the comparison.
// ---------------------------------------------------------------

struct QuantLayout {
    name: &'static str,
    block_elems: usize,
    block_bytes: usize,
    /// Byte offsets of f16 scale fields within each block, stamped
    /// with small deterministic values. Layouts whose scales are
    /// packed into quant words (IQ1_M) list none and rely on the
    /// finite-reference retry guard below.
    f16_scales: &'static [usize],
}

const LAYOUTS: &[QuantLayout] = &[
    QuantLayout {
        name: "q8_0",
        block_elems: 32,
        block_bytes: 34,
        f16_scales: &[0],
    },
    // Legacy GGML Q4_0: 18 bytes / 32 elems = f16 d + 16 nibble bytes;
    // value = d·(nibble − 8). Single-row + batched SYCL kernels both ship
    // (`matvec_q4_0_packed_f32{,_batched}_usm`); listed here so the matvec +
    // batched-matvec probes exercise them against the CPU reference.
    QuantLayout {
        name: "q4_0",
        block_elems: 32,
        block_bytes: 18,
        f16_scales: &[0],
    },
    QuantLayout {
        name: "q4_k",
        block_elems: 256,
        block_bytes: 144,
        f16_scales: &[0, 2],
    },
    QuantLayout {
        name: "q5_k",
        block_elems: 256,
        block_bytes: 176,
        f16_scales: &[0, 2],
    },
    QuantLayout {
        name: "q6_k",
        block_elems: 256,
        block_bytes: 210,
        f16_scales: &[208],
    },
    QuantLayout {
        name: "iq4_nl",
        block_elems: 32,
        block_bytes: 18,
        f16_scales: &[0],
    },
    QuantLayout {
        name: "iq4_xs",
        block_elems: 256,
        block_bytes: 136,
        f16_scales: &[0],
    },
    QuantLayout {
        name: "iq1_s",
        block_elems: 256,
        block_bytes: 50,
        f16_scales: &[0],
    },
    QuantLayout {
        name: "iq1_m",
        block_elems: 256,
        block_bytes: 56,
        f16_scales: &[],
    },
    QuantLayout {
        name: "iq2_xxs",
        block_elems: 256,
        block_bytes: 66,
        f16_scales: &[0],
    },
    QuantLayout {
        name: "iq2_xs",
        block_elems: 256,
        block_bytes: 74,
        f16_scales: &[0],
    },
    QuantLayout {
        name: "iq2_s",
        block_elems: 256,
        block_bytes: 82,
        f16_scales: &[0],
    },
    QuantLayout {
        name: "iq3_xxs",
        block_elems: 256,
        block_bytes: 98,
        f16_scales: &[0],
    },
    QuantLayout {
        name: "iq3_s",
        block_elems: 256,
        block_bytes: 110,
        f16_scales: &[0],
    },
    // PrismML PTQ1_0 (Bonsai ternary): 24B base-3 qs + 2B qh + f16 d
    // at the tail. Trit extraction is total over arbitrary bytes.
    QuantLayout {
        name: "ptq1_0",
        block_elems: 128,
        block_bytes: 28,
        f16_scales: &[26],
    },
    // OCP Microscaling: 32-elem blocks, trailing E8M0 (1-byte) scale.
    // No f16 scales — `gen_quant_bytes` stamps a sane E8M0 byte (and
    // masks E4M3 NaN element bytes for mxfp8) by name below.
    QuantLayout {
        name: "mxfp4",
        block_elems: 32,
        block_bytes: 17,
        f16_scales: &[],
    },
    QuantLayout {
        name: "mxfp6",
        block_elems: 32,
        block_bytes: 25,
        f16_scales: &[],
    },
    QuantLayout {
        name: "mxfp8",
        block_elems: 32,
        block_bytes: 33,
        f16_scales: &[],
    },
    // CPU-parity legacy + K-quant + PrismML formats: single-row AND batched
    // SYCL matvec kernels ship for each (`matvec_<fmt>_packed_f32{,_batched}_
    // usm`); listed here so both the matvec + matvecb probes exercise them.
    // Q5_0: 22B/32 = f16 d + u32 qh + 16 nibbles.
    QuantLayout { name: "q5_0", block_elems: 32, block_bytes: 22, f16_scales: &[0] },
    // Q4_1: 20B/32 = f16 d + f16 min + 16 nibbles.
    QuantLayout { name: "q4_1", block_elems: 32, block_bytes: 20, f16_scales: &[0, 2] },
    // Q5_1: 24B/32 = f16 d + f16 min + u32 qh + 16 nibbles.
    QuantLayout { name: "q5_1", block_elems: 32, block_bytes: 24, f16_scales: &[0, 2] },
    // Q2_K: 84B/256 = 16 scale bytes + 64 qs + f16 d + f16 dmin (at 80/82).
    QuantLayout { name: "q2_k", block_elems: 256, block_bytes: 84, f16_scales: &[80, 82] },
    // Q3_K: 110B/256 = 32 hmask + 64 qs + 12 scales + f16 d (at 108).
    QuantLayout { name: "q3_k", block_elems: 256, block_bytes: 110, f16_scales: &[108] },
    // Q8_K: 292B/256 = F32 d (at 0) + 256 i8 qs + 16 i16 bsums. The scale is
    // F32, not f16 — `gen_quant_bytes` stamps it by name (see below).
    QuantLayout { name: "q8_k", block_elems: 256, block_bytes: 292, f16_scales: &[] },
    // PQ2_0 (PrismML Bonsai): 34B/128 = f16 d + 32 packed 2-bit codes.
    QuantLayout { name: "pq2_0", block_elems: 128, block_bytes: 34, f16_scales: &[0] },
];

type CpuMatvec = fn(&[u8], &[f32], &mut [f32], usize, usize);
type GpuMatvecRaw =
    unsafe fn(&sk::SyclStream, *const u8, *const f32, *mut f32, u32, u32, u32) -> sk::Result<()>;
type GpuFusedRaw = unsafe fn(
    &sk::SyclStream,
    *const u8,
    *const u8,
    *const f32,
    *mut f32,
    *mut f32,
    u32,
    u32,
    u32,
) -> sk::Result<()>;
// Batched packed matvec: the prefill twin of `GpuMatvecRaw` with an extra `N`
// (input-row count) before the `lws`. `out[n*M + m] = W[m,:]·x[n,:]`.
type GpuMatvecBatchedRaw = unsafe fn(
    &sk::SyclStream,
    *const u8,
    *const f32,
    *mut f32,
    u32,
    u32,
    u32,
    u32,
) -> sk::Result<()>;

fn cpu_matvec_for(name: &str) -> CpuMatvec {
    match name {
        "q8_0" => k::matvec_q8_0_w_f32_a,
        "q4_0" => k::matvec_q4_0_w_f32_a,
        "q4_k" => k::matvec_q4_k_w_f32_a,
        "q5_k" => k::matvec_q5_k_w_f32_a,
        "q6_k" => k::matvec_q6_k_w_f32_a,
        "iq4_nl" => k::matvec_iq4_nl_w_f32_a,
        "iq4_xs" => k::matvec_iq4_xs_w_f32_a,
        "iq1_s" => k::matvec_iq1_s_w_f32_a,
        "iq1_m" => k::matvec_iq1_m_w_f32_a,
        "iq2_xxs" => k::matvec_iq2_xxs_w_f32_a,
        "iq2_xs" => k::matvec_iq2_xs_w_f32_a,
        "iq2_s" => k::matvec_iq2_s_w_f32_a,
        "iq3_xxs" => k::matvec_iq3_xxs_w_f32_a,
        "iq3_s" => k::matvec_iq3_s_w_f32_a,
        "ptq1_0" => k::matvec_ptq1_0_w_f32_a,
        "mxfp4" => k::mxfp::matvec_mxfp4_w_f32_a,
        "mxfp6" => k::mxfp::matvec_mxfp6_w_f32_a,
        "mxfp8" => k::mxfp::matvec_mxfp8_w_f32_a,
        "q5_0" => k::matvec_q5_0_w_f32_a,
        "q4_1" => k::matvec_q4_1_w_f32_a,
        "q5_1" => k::matvec_q5_1_w_f32_a,
        "q2_k" => k::matvec_q2_k_w_f32_a,
        "q3_k" => k::matvec_q3_k_w_f32_a,
        "q8_k" => k::matvec_q8_k_w_f32_a,
        "pq2_0" => k::matvec_pq2_0_w_f32_a,
        _ => unreachable!("unknown dtype {name}"),
    }
}

fn gpu_matvec_for(name: &str) -> GpuMatvecRaw {
    match name {
        "q8_0" => sk::matvec_q8_0_packed_f32_usm_raw,
        "q4_0" => sk::matvec_q4_0_packed_f32_usm_raw,
        "q4_k" => sk::matvec_q4_k_packed_f32_usm_raw,
        "q5_k" => sk::matvec_q5_k_packed_f32_usm_raw,
        "q6_k" => sk::matvec_q6_k_packed_f32_usm_raw,
        "iq4_nl" => sk::matvec_iq4_nl_packed_f32_usm_raw,
        "iq4_xs" => sk::matvec_iq4_xs_packed_f32_usm_raw,
        "iq1_s" => sk::matvec_iq1_s_packed_f32_usm_raw,
        "iq1_m" => sk::matvec_iq1_m_packed_f32_usm_raw,
        "iq2_xxs" => sk::matvec_iq2_xxs_packed_f32_usm_raw,
        "iq2_xs" => sk::matvec_iq2_xs_packed_f32_usm_raw,
        "iq2_s" => sk::matvec_iq2_s_packed_f32_usm_raw,
        "iq3_xxs" => sk::matvec_iq3_xxs_packed_f32_usm_raw,
        "iq3_s" => sk::matvec_iq3_s_packed_f32_usm_raw,
        "ptq1_0" => sk::matvec_ptq1_0_packed_f32_usm_raw,
        "mxfp4" => sk::matvec_mxfp4_packed_f32_usm_raw,
        "mxfp6" => sk::matvec_mxfp6_packed_f32_usm_raw,
        "mxfp8" => sk::matvec_mxfp8_packed_f32_usm_raw,
        "q5_0" => sk::matvec_q5_0_packed_f32_usm_raw,
        "q4_1" => sk::matvec_q4_1_packed_f32_usm_raw,
        "q5_1" => sk::matvec_q5_1_packed_f32_usm_raw,
        "q2_k" => sk::matvec_q2_k_packed_f32_usm_raw,
        "q3_k" => sk::matvec_q3_k_packed_f32_usm_raw,
        "q8_k" => sk::matvec_q8_k_packed_f32_usm_raw,
        "pq2_0" => sk::matvec_pq2_0_packed_f32_usm_raw,
        _ => unreachable!("unknown dtype {name}"),
    }
}

/// F16-activation (mixed-precision) packed matvec: same as [`GpuMatvecRaw`] but
/// the activation vector `x` is `*const u16` (f16 bits), halving its
/// per-output-row re-read bandwidth. `out[m] = W[m,:]·x16`.
type GpuMatvecF16inRaw =
    unsafe fn(&sk::SyclStream, *const u8, *const u16, *mut f32, u32, u32, u32) -> sk::Result<()>;

/// The f16-activation matvec kernel for `name`, or `None` for a format with no
/// f16-in kernel (the mixed-precision path then stays on the f32 matvec). Drives
/// the `matvec_f16in:` probes (P5b: graded vs an f16-activation-roundtrip
/// reference AND perf-gated). Mirrors the dtype coverage of the `h8_call!` table
/// in `accel::try_matvec_tensor_f16in_usm_f32`.
fn gpu_matvec_f16in_for(name: &str) -> Option<GpuMatvecF16inRaw> {
    Some(match name {
        "q8_0" => sk::matvec_q8_0_f16in_packed_f32_usm_raw,
        "q4_k" => sk::matvec_q4_k_f16in_packed_f32_usm_raw,
        "q5_k" => sk::matvec_q5_k_f16in_packed_f32_usm_raw,
        "q6_k" => sk::matvec_q6_k_f16in_packed_f32_usm_raw,
        "iq4_nl" => sk::matvec_iq4_nl_f16in_packed_f32_usm_raw,
        "iq4_xs" => sk::matvec_iq4_xs_f16in_packed_f32_usm_raw,
        "iq1_s" => sk::matvec_iq1_s_f16in_packed_f32_usm_raw,
        "iq1_m" => sk::matvec_iq1_m_f16in_packed_f32_usm_raw,
        "iq2_xxs" => sk::matvec_iq2_xxs_f16in_packed_f32_usm_raw,
        "iq2_xs" => sk::matvec_iq2_xs_f16in_packed_f32_usm_raw,
        "iq2_s" => sk::matvec_iq2_s_f16in_packed_f32_usm_raw,
        "iq3_xxs" => sk::matvec_iq3_xxs_f16in_packed_f32_usm_raw,
        "iq3_s" => sk::matvec_iq3_s_f16in_packed_f32_usm_raw,
        "ptq1_0" => sk::matvec_ptq1_0_f16in_packed_f32_usm_raw,
        _ => return None,
    })
}

/// Per-element f16 round-trip of activations (f32 → f16 → f32). The FAIR
/// reference for an f16-activation kernel (the analogue of [`int8_act_roundtrip`]
/// for the int8 kernels): dequantizing x back to f32 and running the normal
/// f32-activation matvec reproduces what the kernel computes up to accumulation
/// order, so the probe grades KERNEL correctness — not the inherent f16 loss.
fn f16_act_roundtrip(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect()
}

/// The batched (prefill) packed-matvec kernel for `name`, or `None` for a
/// format that has only the single-row kernel (so the batched prefill path
/// falls to CPU). Drives the `matvecb:` probes — the ONLY parity coverage of
/// the batched kernels (the matvec probes above exercise the single-row path).
fn gpu_matvec_batched_for(name: &str) -> Option<GpuMatvecBatchedRaw> {
    Some(match name {
        "q8_0" => sk::matvec_q8_0_packed_f32_batched_usm_raw,
        "q4_0" => sk::matvec_q4_0_packed_f32_batched_usm_raw,
        "q4_k" => sk::matvec_q4_k_packed_f32_batched_usm_raw,
        "q5_k" => sk::matvec_q5_k_packed_f32_batched_usm_raw,
        "q6_k" => sk::matvec_q6_k_packed_f32_batched_usm_raw,
        "iq4_nl" => sk::matvec_iq4_nl_packed_f32_batched_usm_raw,
        "iq4_xs" => sk::matvec_iq4_xs_packed_f32_batched_usm_raw,
        "iq1_s" => sk::matvec_iq1_s_packed_f32_batched_usm_raw,
        "iq1_m" => sk::matvec_iq1_m_packed_f32_batched_usm_raw,
        "iq2_xxs" => sk::matvec_iq2_xxs_packed_f32_batched_usm_raw,
        "iq2_xs" => sk::matvec_iq2_xs_packed_f32_batched_usm_raw,
        "iq2_s" => sk::matvec_iq2_s_packed_f32_batched_usm_raw,
        "iq3_xxs" => sk::matvec_iq3_xxs_packed_f32_batched_usm_raw,
        "iq3_s" => sk::matvec_iq3_s_packed_f32_batched_usm_raw,
        "ptq1_0" => sk::matvec_ptq1_0_packed_f32_batched_usm_raw,
        "q5_0" => sk::matvec_q5_0_packed_f32_batched_usm_raw,
        "q4_1" => sk::matvec_q4_1_packed_f32_batched_usm_raw,
        "q5_1" => sk::matvec_q5_1_packed_f32_batched_usm_raw,
        "q2_k" => sk::matvec_q2_k_packed_f32_batched_usm_raw,
        "q3_k" => sk::matvec_q3_k_packed_f32_batched_usm_raw,
        "q8_k" => sk::matvec_q8_k_packed_f32_batched_usm_raw,
        "pq2_0" => sk::matvec_pq2_0_packed_f32_batched_usm_raw,
        "mxfp4" => sk::matvec_mxfp4_packed_f32_batched_usm_raw,
        "mxfp6" => sk::matvec_mxfp6_packed_f32_batched_usm_raw,
        "mxfp8" => sk::matvec_mxfp8_packed_f32_batched_usm_raw,
        _ => return None,
    })
}

fn gpu_fused_for(name: &str) -> Option<GpuFusedRaw> {
    Some(match name {
        "q8_0" => sk::matvec_q8_0_gate_up_fused_usm_raw,
        "q4_k" => sk::matvec_q4_k_gate_up_fused_usm_raw,
        "q5_k" => sk::matvec_q5_k_gate_up_fused_usm_raw,
        "q6_k" => sk::matvec_q6_k_gate_up_fused_usm_raw,
        "iq4_nl" => sk::matvec_iq4_nl_gate_up_fused_usm_raw,
        "iq4_xs" => sk::matvec_iq4_xs_gate_up_fused_usm_raw,
        "iq1_s" => sk::matvec_iq1_s_gate_up_fused_usm_raw,
        "iq1_m" => sk::matvec_iq1_m_gate_up_fused_usm_raw,
        "iq2_xxs" => sk::matvec_iq2_xxs_gate_up_fused_usm_raw,
        "iq2_xs" => sk::matvec_iq2_xs_gate_up_fused_usm_raw,
        "iq2_s" => sk::matvec_iq2_s_gate_up_fused_usm_raw,
        "iq3_xxs" => sk::matvec_iq3_xxs_gate_up_fused_usm_raw,
        "iq3_s" => sk::matvec_iq3_s_gate_up_fused_usm_raw,
        _ => return None,
    })
}

/// All probe names, in execution order. Attention probes go FIRST:
/// on backends where a later matvec kernel wedges the driver, the
/// attention verdicts are already banked (the parent survives either
/// way, but device resets can leave the driver flaky for a while).
fn probe_names() -> Vec<String> {
    let mut v = vec![
        "attn:decode_v1".to_string(),
        "attn:decode_v2".to_string(),
        "attn:decode_v3".to_string(),
        "attn:prefill_v1".to_string(),
        "attn:prefill_v2".to_string(),
        "attn:prefill_v3".to_string(),
    ];
    // Quantized-KV FlashAttention: the MXFP4/6/8 + NVFP4 flash decode +
    // prefill kernels (F32 Q/out, packed K/V dequantized on the fly) vs
    // the full-precision CPU reference. These are NOT driven by LAYOUTS
    // (that table's block bytes describe the *weight* matvec probes; the
    // KV blocks differ — e.g. NVFP4 is a 16-elem/9-byte KV block, not the
    // matvec layout) so they're listed from QUANT_KV_FORMATS explicitly.
    // Kept adjacent to the f32 attention probes so all attention verdicts
    // bank before the matvec families that can wedge the driver.
    for f in QUANT_KV_FORMATS {
        v.push(format!("attn:decode_{}", f.name));
    }
    for f in QUANT_KV_FORMATS {
        v.push(format!("attn:prefill_{}", f.name));
    }
    // Q8_0 + TQ KV flash (per-row-scale layouts; not QUANT_KV_FORMATS blocks).
    v.push("attn:decode_q8_0".to_string());
    v.push("attn:prefill_q8_0".to_string());
    v.push("attn:decode_tq".to_string());
    v.push("attn:prefill_tq".to_string());
    // Feature 1a: the WHOLE device-resident DECODE attention block (rmsnorm ->
    // q/k/v matvec -> RoPE -> f16-mirror write -> flash -> o matvec -> residual+
    // norm) as one in-order submit-chain, vs the CPU reference end to end. The one
    // orchestration probe that isn't covered by the per-op kernels above.
    v.push("attn:dev_resident_block_sycl".to_string());
    // B1/F2: the F32-precision residual/norm/activation USM fusions vs their
    // CPU references. These keep the residual stream in f32 (no F16 round-trip),
    // which is the correctness fix B1 lands. Small elementwise kernels — safe to
    // bank alongside the attention probes before the matvec families.
    v.push("fused:add_rmsnorm_f32".to_string());
    v.push("fused:rmsnorm_f32".to_string());
    v.push("fused:silu_mul_f32".to_string());
    // P6-fusions: the three forward-path fusions graded vs their two-op CPU
    // references (bit-tolerant f32). `rmsnorm_matvec` / `down_residual` are the
    // new single f32 kernels; `out_proj_norm` is the EXACT chained path the
    // `try_matvec_out_proj_add_rmsnorm_usm_f32` dispatcher runs (packed q4_k
    // matvec → add_rmsnorm_f32). Banked before the matvec families.
    v.push("fused:rmsnorm_matvec".to_string());
    v.push("fused:down_residual".to_string());
    v.push("fused:out_proj_norm".to_string());
    // P17: GPU Gated-DeltaNet chunked-prefill recurrence (the hybrid-model tok/s
    // ceiling) vs the CPU chunked reference. f32 end to end; banked before the
    // matvec families like the other small f32 USM kernels.
    v.push("ssm:delta_net".to_string());
    for l in LAYOUTS {
        v.push(format!("matvec:{}", l.name));
    }
    // Batched (prefill) packed matvecs: the N>1 twin of the matvec probes
    // above. The single-row probes never touch these kernels, so this is
    // their only parity coverage. Only formats with a batched kernel
    // (`gpu_matvec_batched_for`) are listed; the rest fall to CPU on prefill.
    for l in LAYOUTS {
        if gpu_matvec_batched_for(l.name).is_some() {
            v.push(format!("matvecb:{}", l.name));
        }
    }
    for l in LAYOUTS {
        // Only the formats with a fused gate/up kernel (`gpu_fused_for`); the
        // rest (PTQ1_0, Q4_0, the legacy/K-quant/PQ/MXFP formats) have no fused
        // kernel — their FFN dispatch uses two separate single-row matvecs.
        if gpu_fused_for(l.name).is_some() {
            v.push(format!("fused:{}", l.name));
        }
    }
    // C-A: real-MoE-forward coherence probe per fused dtype — the per-expert FFN
    // (several experts, non-512 shapes, SwiGLU, routing sum) graded fused ≡
    // separate ≡ CPU. `accel::moe_gate_up_fused_enabled` ANDs this with the bare
    // `fused:` parity above, so the generic-GEMM pass alone can't re-enable the
    // 2026-09-06 incident path. Adjacent to `fused:` so both bank together.
    for l in LAYOUTS {
        if gpu_fused_for(l.name).is_some() {
            v.push(format!("moe:gate_up_coherent:{}", l.name));
        }
    }
    // P5b: F16-activation (mixed-precision) matvecs — only the formats with an
    // f16-in kernel (`gpu_matvec_f16in_for`). Graded vs an f16-roundtrip ref AND
    // perf-gated; back `accel::mixed_precision_matvec_enabled` per dtype.
    for l in LAYOUTS {
        if gpu_matvec_f16in_for(l.name).is_some() {
            v.push(format!("matvec_f16in:{}", l.name));
        }
    }
    // XMX/DPAS bf16 tensor-core GEMM (Arc Xe-HPG / PVC Xe-HPC). SKIPs unless
    // the device is XMX-capable AND the DLL was built -DRSL_SYCL_XMX; on Iris
    // Xe (Xe-LP) it reports SKIP. Last so it banks after the matvec families.
    v.push("xmx:gemm".to_string());
    // F17 — IQ1_S GPU quant-ENCODER parity. The opt-in `RUSTLLAMA_IQ_GPU=1`
    // quantizer's 8-element delta grid search (`SyclIqEncoder`) vs the CPU
    // reference encoder on identical target blocks. Validates the GPU quantizer,
    // not a forward kernel; SKIPs cleanly when the `encoder` feature is off (slim
    // server build). No dispatch verdict reads it (the quantize path tries the
    // GPU encoder and falls back to CPU on error), so it's excluded from the
    // banked verdicts in `collect_sycl_verdicts`.
    v.push("encoder:iq1_s".to_string());
    v
}

// ---------------------------------------------------------------
// Deterministic data generation
// ---------------------------------------------------------------

struct XorShift(u32);
impl XorShift {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn f32_pm(&mut self, scale: f32) -> f32 {
        ((self.next() >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0 * scale
    }
}

fn gen_quant_bytes(layout: &QuantLayout, rows: usize, k_dim: usize, seed: u32) -> Vec<u8> {
    assert_eq!(k_dim % layout.block_elems, 0);
    let blocks_per_row = k_dim / layout.block_elems;
    let row_bytes = blocks_per_row * layout.block_bytes;
    let mut rng = XorShift(seed | 1);
    let mut bytes: Vec<u8> = (0..rows * row_bytes)
        .map(|_| (rng.next() >> 13) as u8)
        .collect();
    // Stamp sane f16 scales so dequant magnitudes stay small + finite.
    for b in 0..rows * blocks_per_row {
        let off = b * layout.block_bytes;
        for (i, &s) in layout.f16_scales.iter().enumerate() {
            let val = 0.004 + 0.002 * ((b + i) % 5) as f32;
            bytes[off + s..off + s + 2].copy_from_slice(&half::f16::from_f32(val).to_le_bytes());
        }
        if layout.name.starts_with("mxfp") {
            // OCP MX: one E8M0 scale byte at the block tail. Random fill
            // could leave it 0xFF (the E8M0 NaN); stamp a sane exponent
            // near 2^0 so block magnitudes stay O(1) and finite. For
            // mxfp8, also mask the two E4M3 NaN element encodings
            // (0x7F / 0xFF) that random bytes can produce — a NaN weight
            // would poison the cosine comparison for both backends.
            let scale_off = off + layout.block_bytes - 1;
            bytes[scale_off] = (125 + (b % 5)) as u8; // 2^-2 .. 2^2
            if layout.name == "mxfp8" {
                for byte in bytes[off..off + 32].iter_mut() {
                    if *byte == 0x7F || *byte == 0xFF {
                        *byte = 0x38;
                    }
                }
            }
        } else if layout.name == "q8_k" {
            // Q8_K's shared scale is a leading F32 (not f16); random bytes
            // there could be a huge/NaN float and poison the comparison.
            // Stamp a small sane magnitude so d·i8 stays O(1) and finite.
            let val = 0.004 + 0.002 * (b % 5) as f32;
            bytes[off..off + 4].copy_from_slice(&val.to_le_bytes());
        } else if layout.f16_scales.is_empty() {
            // IQ1_M: scales live packed inside the 8-byte scale words
            // (including the split f16 delta). A fixed moderate bit
            // pattern keeps the implied delta finite; the retry guard
            // below catches pathological combinations.
            let sc = layout.block_bytes - 8;
            for byte in bytes[off + sc..off + layout.block_bytes].iter_mut() {
                *byte = 0x32;
            }
        }
    }
    bytes
}

fn gen_x(k_dim: usize, seed: u32) -> Vec<f32> {
    let mut rng = XorShift(seed.wrapping_mul(2654435761) | 1);
    (0..k_dim).map(|_| rng.f32_pm(0.7)).collect()
}

/// Cosine similarity + worst relative error between GPU and CPU.
fn compare(gpu: &[f32], cpu: &[f32]) -> (f64, f64) {
    let mut dot = 0f64;
    let mut ng = 0f64;
    let mut nc = 0f64;
    let mut max_rel = 0f64;
    for (&g, &c) in gpu.iter().zip(cpu.iter()) {
        dot += g as f64 * c as f64;
        ng += (g as f64).powi(2);
        nc += (c as f64).powi(2);
        let rel = ((g - c).abs() as f64) / ((c.abs() as f64) + 1e-3);
        max_rel = max_rel.max(rel);
    }
    let cos = if ng == 0.0 || nc == 0.0 {
        // Both all-zero counts as agreement; one-sided zero doesn't.
        if ng == nc {
            1.0
        } else {
            0.0
        }
    } else {
        dot / (ng.sqrt() * nc.sqrt())
    };
    (cos, max_rel)
}

// --- Probe-result recording (for `tune --validate-kernels`) -----------
//
// Both emit() (SYCL) and cu_emit() (CUDA) funnel every probe's verdict
// line through here. Normally recording is OFF and they just print. The
// `tune --validate-kernels` stage turns it ON around a probe run, then
// drains the recorded `(probe_name, status)` pairs into on-device
// auto-enable verdicts. No behavior change for `doctor --*-parity`.
thread_local! {
    static PROBE_REC: std::cell::RefCell<Option<Vec<(String, String)>>> =
        const { std::cell::RefCell::new(None) };
}

fn probe_rec_push(name: &str, verdict: &str) {
    PROBE_REC.with(|r| {
        if let Some(v) = r.borrow_mut().as_mut() {
            // Canonical verdict key: drop any "(W4A4)"-style presentational
            // suffix on the probe name (e.g. "gemm:mxfp8_tc(W8A8)") so the
            // recorded key matches the bare `rustllama_tuner::VERDICT_*`
            // constants the dispatch gates look up. The printed doctor line
            // keeps the full label.
            let key = name.split('(').next().unwrap_or(name).trim_end();
            v.push((key.to_string(), verdict.to_string()));
        }
    });
}

fn probe_rec_begin() {
    PROBE_REC.with(|r| *r.borrow_mut() = Some(Vec::new()));
}

fn probe_rec_take() -> Vec<(String, String)> {
    PROBE_REC.with(|r| r.borrow_mut().take().unwrap_or_default())
}

/// Map a probe status string to an auto-enable verdict: `Some(true)` =
/// matched the CPU reference (and, for a perf-gated probe, was also faster),
/// `Some(false)` = miscomputed, failed to run, or correct-but-not-faster
/// (`"SLOW"`) — all fail-closed, `None` = SKIP (capability absent — leave
/// unset so the gate's own capability check decides).
fn probe_status_to_verdict(status: &str) -> Option<bool> {
    match status {
        "OK" => Some(true),
        "MISCOMPUTE" | "KERNEL_ERR" | "SLOW" => Some(false),
        _ => None, // "SKIP" and anything unexpected
    }
}

/// Run the CUDA kernel parity probes and return per-probe auto-enable
/// verdicts (prints the same report as `doctor --cuda-parity`). Used by
/// `tune --validate-kernels` to populate the on-device verdict cache.
pub fn collect_cuda_verdicts() -> anyhow::Result<Vec<(String, Option<bool>)>> {
    probe_rec_begin();
    let r = run_cuda_parity();
    let recorded = probe_rec_take();
    r?;
    Ok(recorded
        .into_iter()
        .map(|(name, status)| (name, probe_status_to_verdict(&status)))
        .collect())
}

/// Run the SYCL XMX/DPAS GEMM parity probe and return its verdict. Empty
/// when there is no usable SYCL device (no verdict recorded → XMX stays
/// off). WRITE-BLIND note: in-process here (no XMX HW in play); if a real
/// XMX device ever risks DEVICE_LOST this should move to the subprocess
/// model the other SYCL probes use.
pub fn collect_xmx_verdict() -> Vec<(String, Option<bool>)> {
    let stream = match sk::create_stream(0) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    probe_rec_begin();
    probe_xmx(&stream, "xmx:gemm", Instant::now());
    probe_rec_take()
        .into_iter()
        .map(|(name, status)| (name, probe_status_to_verdict(&status)))
        .collect()
}

/// P5b — bank the `sampler:gpu` verdict: ON only where the GPU sampler chain
/// both MATCHES the CPU sampler AND is FASTER. Runs in-process (like
/// [`collect_xmx_verdict`]); returns empty off SYCL (no verdict → the sampler
/// stays on CPU). Replaces the removed `RUSTLLAMA_SAMPLER_GPU` env gate.
///
/// Correctness: the whole-vocab temp+softmax kernel vs the CPU reference, plus
/// the argmax kernel vs a scalar argmax. Perf: the GPU temp+softmax+argmax (what
/// a greedy sample runs) vs the CPU path over a realistic vocab. The per-kernel
/// host wrappers transfer per call (the engine's `SamplerGpuContext` keeps logits
/// resident across the chain), so this perf read is CONSERVATIVE — it biases
/// toward OFF, never toward wrongly enabling a slow path (fail-closed).
pub fn collect_sampler_gpu_verdict() -> Vec<(String, Option<bool>)> {
    let stream = match sk::create_stream(0) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let key = rustllama_tuner::VERDICT_SAMPLER_GPU.to_string();
    const VOCAB: usize = 32_000;
    let base = gen_x(VOCAB, 0x5A17);
    let inv_temp = 1.0f32 / 0.7;
    // Correctness #1: GPU temp+softmax vs CPU reference.
    let mut cpu = base.clone();
    k::fused_temp_softmax_inplace(&mut cpu, inv_temp);
    let mut gpu = base.clone();
    match sk::sampler_temp_softmax_host(&stream, &mut gpu, inv_temp) {
        Ok(()) => {}
        // No sampler kernel on this device → SKIP (capability absent).
        Err(_) => return vec![(key, None)],
    }
    let (cos, max_rel) = compare(&gpu, &cpu);
    let softmax_ok = cos > 0.999 && max_rel < 0.02;
    // Correctness #2: GPU argmax vs scalar argmax (over the softmaxed probs).
    let cpu_argmax = cpu
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0);
    let argmax_ok = matches!(sk::sampler_argmax_host(&stream, &cpu), Ok(idx) if idx == cpu_argmax);
    if !(softmax_ok && argmax_ok) {
        return vec![(key, Some(false))];
    }
    // Perf: GPU temp+softmax+argmax vs the CPU equivalent over the same vocab.
    const ITERS: usize = 50;
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let mut c = base.clone();
        k::fused_temp_softmax_inplace(&mut c, inv_temp);
        let _ = c
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
    }
    let t_cpu = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    for _ in 0..ITERS {
        let mut g = base.clone();
        if sk::sampler_temp_softmax_host(&stream, &mut g, inv_temp).is_ok() {
            let _ = sk::sampler_argmax_host(&stream, &g);
        }
    }
    let t_gpu = t1.elapsed().as_secs_f64();
    let faster = t_gpu < t_cpu;
    let status = if faster { "OK" } else { "SLOW" };
    emit(
        &key,
        status,
        &format!("cos={cos:.6} gpu={:.2}ms cpu={:.2}ms", t_gpu * 1e3, t_cpu * 1e3),
    );
    vec![(key, Some(faster))]
}

/// Intel NPU compute-offload verdicts for `tune --validate-kernels`:
///   - `npu:matmul_f32` — CORRECTNESS: the NPU MatMul (Phase 5, OpenVINO) matched
///     the CPU reference (parity).
///   - `npu:faster_than_gpu` — PERF: the NPU beat the GPU/CPU at a representative
///     prefill GEMM (`rustllama_engine::npu_perf_probe`). Only measured
///     when the NPU is present AND correct (a slow or broken NPU shouldn't be
///     perf-ranked). The accel dispatch ANDs both, so the NPU auto-enables only
///     where it's correct AND the fastest matrix engine — the measured NPU-vs-GPU
///     decision that replaces the old hardcoded CUDA-first gate.
/// `None` (SKIP) on either when no NPU is present → nothing persisted → the
/// dispatch stays off (fail-closed).
pub fn collect_npu_verdict() -> Vec<(String, Option<bool>)> {
    // Intel (OpenVINO).
    let r = rustllama_npu::npu_matmul_parity();
    let correct = if r.supported { Some(r.pass) } else { None };
    let perf = if matches!(correct, Some(true)) {
        rustllama_engine::npu_perf_probe()
    } else {
        None
    };
    // AMD (ONNX Runtime + VitisAI). Mutually exclusive with Intel in practice;
    // both probed — the absent one SKIPs (None) and persists nothing.
    let amd = rustllama_npu::amd_npu_matmul_parity();
    let amd_correct = if amd.supported { Some(amd.pass) } else { None };
    let amd_perf = if matches!(amd_correct, Some(true)) {
        rustllama_engine::amd_npu_perf_probe()
    } else {
        None
    };
    vec![
        ("npu:matmul_f32".to_string(), correct),
        ("npu:faster_than_gpu".to_string(), perf),
        ("npu:amd_matmul_f32".to_string(), amd_correct),
        ("npu:amd_faster_than_gpu".to_string(), amd_perf),
    ]
}

fn emit(name: &str, verdict: &str, detail: &str) {
    probe_rec_push(name, verdict);
    // Single-line machine-readable protocol; the parent greps for it.
    println!("PARITY {name} {verdict} {detail}");
    let _ = std::io::stdout().flush();
}

// ---------------------------------------------------------------
// Child: run one probe and print its PARITY line.
// ---------------------------------------------------------------

pub fn run_probe(name: &str) -> anyhow::Result<()> {
    let stream = match sk::create_stream(0) {
        Ok(s) => s,
        Err(_) => {
            emit(name, "SKIP", "no-sycl-device");
            return Ok(());
        }
    };
    let started = Instant::now();
    let (kind, rest) = name
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("bad probe name {name}"))?;
    match kind {
        "matvec" => probe_matvec(&stream, name, rest, started),
        "matvecb" => probe_matvec_batched(&stream, name, rest, started),
        "matvec_f16in" => probe_matvec_f16in(&stream, name, rest, started),
        "fused" => probe_fused(&stream, name, rest, started),
        // C-A: `moe:gate_up_coherent:<dtype>` — real-MoE-forward coherence.
        "moe" => match rest.strip_prefix("gate_up_coherent:") {
            Some(dt) => probe_moe_gate_up_coherent(&stream, name, dt, started),
            None => emit(name, "SKIP", "unknown-moe-probe"),
        },
        "attn" => probe_attn(&stream, name, rest, started),
        "xmx" => probe_xmx(&stream, name, started),
        // P17: GPU DeltaNet chunked-prefill recurrence.
        "ssm" => probe_delta_net(&stream, name, rest, started),
        // F17: IQ1_S GPU quant-encoder vs CPU reference.
        "encoder" => probe_iq_encoder(&stream, name, rest, started),
        _ => anyhow::bail!("unknown probe kind {kind}"),
    }
    Ok(())
}

/// Model-realistic shape: the per-expert gate/up matvec of the target
/// MoE (512×2048) — the exact shape that produced DEVICE_LOST on
/// Level Zero for the IQ2/IQ3 families.
const MV_M: usize = 512;
const MV_K: usize = 2048;

/// Deterministic weights whose CPU-reference output is fully finite;
/// bumps the seed on pathological quant-bit combinations.
fn finite_ref(layout: &QuantLayout, cpu: CpuMatvec, x: &[f32]) -> Option<(Vec<u8>, Vec<f32>)> {
    for attempt in 0..8u32 {
        let w = gen_quant_bytes(layout, MV_M, MV_K, 0xC0FFEE ^ (attempt * 7919));
        let mut r = vec![0f32; MV_M];
        cpu(&w, x, &mut r, MV_M, MV_K);
        if r.iter().all(|v| v.is_finite()) {
            return Some((w, r));
        }
    }
    None
}

// ---------------------------------------------------------------
// C-A — real-MoE-forward coherence probe shapes/structure.
//
// The bare `fused:` probe grades ONE 512×2048 GEMM vs the CPU matvec. The
// 2026-09-06 token-soup incident was a FULL-MODEL MoE decode (Qwen3.6-35B,
// IQ2_XXS experts) — several experts per layer, non-512 `[d_ff, d_model]`
// shapes, the SwiGLU nonlinearity, routing-weighted summed. These constants
// reproduce that forward structure on-device so the coherence verdict exercises
// the path the model actually runs, not a generic single GEMM.
// ---------------------------------------------------------------

/// Realistic per-expert gate/up shapes `(d_ff rows = M, d_model cols = K)`,
/// deliberately NON-512 and spanning real MoE experts. `K` is a multiple of 256
/// so every K-quant (256-block) and legacy (32-block) fused format tiles.
const MOE_COH_SHAPES: &[(usize, usize)] = &[(768, 2048), (1536, 1024), (256, 1536)];
/// How many routed experts the coherence probe sums per shape (top-k structure).
const MOE_COH_EXPERTS: usize = 3;
/// Deterministic router weights for the summed experts. The absolute scale is
/// irrelevant to a cosine/relative grade; only that the sum mixes experts.
const MOE_COH_ROUTE_W: [f32; MOE_COH_EXPERTS] = [0.6, 0.3, 0.1];

/// SwiGLU gate activation: `silu(v) = v·sigmoid(v)` — the FFN nonlinearity the
/// fused gate output feeds through before the elementwise `·up`.
fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// Generate a quant weight tensor of shape `(m, k)` whose CPU-reference matvec
/// output over `x` is fully finite, bumping the seed on pathological quant-bit
/// combinations (the `(m, k)`-parametric twin of [`finite_ref`]). Returns the
/// packed bytes + the CPU reference output, or `None` after 8 failed attempts.
fn coh_gen_finite(
    layout: &QuantLayout,
    cpu: CpuMatvec,
    m: usize,
    k: usize,
    x: &[f32],
    seed: u32,
) -> Option<(Vec<u8>, Vec<f32>)> {
    for attempt in 0..8u32 {
        let w = gen_quant_bytes(layout, m, k, seed ^ (attempt.wrapping_mul(2654435761) | 1));
        let mut r = vec![0f32; m];
        cpu(&w, x, &mut r, m, k);
        if r.iter().all(|v| v.is_finite()) {
            return Some((w, r));
        }
    }
    None
}

/// Per-32 symmetric int8 round-trip of activations (scale = absmax/127, round-
/// to-nearest, clamp ±127), matching the CUDA int8 activation quant. The fair
/// reference for int8-activation kernels: because the integer tensor-core dot is
/// exact, dequantizing x back to f32 and running the normal f32-activation Q4_K
/// matvec reproduces the kernel's `d·sc·Σ(q·xq) − dmin·mn·Σxq` identity up to f32
/// accumulation order, so the probe grades kernel correctness — not quant loss.
fn int8_act_roundtrip(x: &[f32]) -> Vec<f32> {
    let mut xr = vec![0f32; x.len()];
    for (bi, blk) in x.chunks(32).enumerate() {
        let amax = blk.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let scale = amax / 127.0;
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        for (j, &v) in blk.iter().enumerate() {
            let q = (v * inv).round().clamp(-127.0, 127.0);
            xr[bi * 32 + j] = q * scale;
        }
    }
    xr
}

/// F17 — IQ1_S SYCL GPU-ENCODER parity probe. Grades the GPU 8-element delta
/// grid search (`SyclIqEncoder::iq_8elt_delta_batched` — the opt-in
/// `RUSTLLAMA_IQ_GPU=1` quantizer behind the CLI `encoder` feature) against the
/// CPU reference encoder (`CpuFallbackEncoder`, the AVX2/scalar
/// `best_iq1s_grid_for_chunk`) on identical target blocks. Both maximize the SAME
/// objective — `dot²/norm` over the 2048-entry IQ1_S grid (`grid point + delta`)
/// — so a correct GPU search reaches each chunk's optimum. Graded on that
/// objective (the quant-optimality criterion, robust to a legitimate lowest-lane
/// tie-break picking an equally-good grid index) with the exact grid-index match
/// rate reported as detail. Validatable HERE on Iris Xe.
///
/// SKIPs when the `encoder` feature is off (slim server build) or no SYCL encoder
/// can be constructed. This proves the GPU quantizer ≡ the CPU quantizer; it is
/// NOT a forward-path kernel and banks no dispatch verdict.
#[cfg(feature = "encoder")]
fn probe_iq_encoder(_stream: &sk::SyclStream, name: &str, rest: &str, started: Instant) {
    use rustllama_gguf::iq_gpu::{CpuFallbackEncoder, Iq8EltDeltaPick, IqGpuEncoder};
    if rest != "iq1_s" {
        emit(name, "SKIP", "unknown-encoder-probe");
        return;
    }
    // The GPU encoder owns its own SYCL queue + the uploaded IQ1_S grid; build it
    // on device 0 (the same device `_stream` enumerated). A construction failure
    // (no Intel SYCL device, USM upload refused) is a clean SKIP.
    let gpu_enc = match sk::iq_encoder::SyclIqEncoder::new(0) {
        Ok(e) => e,
        Err(e) => {
            emit(name, "SKIP", &format!("no-gpu-encoder ({e})"));
            return;
        }
    };
    let cpu_enc = CpuFallbackEncoder;

    const N: usize = 256; // chunks per delta
    // IQ1_M uses ±delta sub-block scaling; include a zero and an IQ1_S-style
    // positive delta so the sweep exercises the sign of the grid offset.
    let deltas = [-0.0625f32, 0.0, 0.125];
    let zero = Iq8EltDeltaPick { grid_idx: 0, signed_score: 0.0, norm_sq: 0.0 };
    let mut obj_gpu: Vec<f32> = Vec::with_capacity(N * deltas.len());
    let mut obj_cpu: Vec<f32> = Vec::with_capacity(N * deltas.len());
    let mut idx_match = 0usize;
    let mut total = 0usize;
    for (di, &delta) in deltas.iter().enumerate() {
        let targets = gen_x(N * 8, 0xA1C0 ^ ((di as u32).wrapping_mul(2654435761) | 1));
        let mut g = vec![zero; N];
        let mut c = vec![zero; N];
        if let Err(e) = gpu_enc.iq_8elt_delta_batched(&targets, delta, &mut g) {
            emit(
                name,
                "KERNEL_ERR",
                &format!("gpu encode failed: {e} ms={}", started.elapsed().as_millis()),
            );
            return;
        }
        if cpu_enc.iq_8elt_delta_batched(&targets, delta, &mut c).is_err() {
            emit(name, "KERNEL_ERR", "cpu reference encode failed");
            return;
        }
        for i in 0..N {
            let og = if g[i].norm_sq > 0.0 {
                g[i].signed_score * g[i].signed_score / g[i].norm_sq
            } else {
                0.0
            };
            let oc = if c[i].norm_sq > 0.0 {
                c[i].signed_score * c[i].signed_score / c[i].norm_sq
            } else {
                0.0
            };
            obj_gpu.push(og);
            obj_cpu.push(oc);
            if g[i].grid_idx == c[i].grid_idx {
                idx_match += 1;
            }
            total += 1;
        }
    }
    let (cos, max_rel) = compare(&obj_gpu, &obj_cpu);
    let ok = cos > 0.9999 && max_rel < 2e-3;
    let ms = started.elapsed().as_millis();
    emit(
        name,
        if ok { "OK" } else { "MISCOMPUTE" },
        &format!("idx_match={idx_match}/{total} cos={cos:.6} max_rel={max_rel:.5} ms={ms}"),
    );
}

/// `encoder` feature off (slim server build): the GPU IQ encoder isn't compiled
/// in, so the probe SKIPs cleanly.
#[cfg(not(feature = "encoder"))]
fn probe_iq_encoder(_stream: &sk::SyclStream, name: &str, _rest: &str, _started: Instant) {
    emit(name, "SKIP", "no-encoder-feature");
}

/// XMX/DPAS bf16 tensor-core GEMM probe. SKIP unless the device is XMX-capable
/// AND the DLL was built with the XMX path (`-DRSL_SYCL_XMX`; otherwise the host
/// wrapper's kernel returns -2 → Ok(false)). bf16 compute vs an f32 CPU
/// reference, so the gate is deliberately loose — it catches a wrong
/// joint_matrix fragment/scale layout (cos→0), not bf16 rounding.
/// `out[n*M + m] = Σ_k W[m,k]·x[n,k]`. WRITE-BLIND (no XMX GPU here); on-Arc
/// numbers settle the real tolerance.
fn probe_xmx(stream: &sk::SyclStream, name: &str, started: Instant) {
    if !sk::xmx_available(stream) {
        emit(name, "SKIP", "device-not-xmx-capable-or-no-xmx-build");
        return;
    }
    let (m, kd, n) = (64usize, 128usize, 16usize); // K % 16 (TK) == 0
    let w = gen_x(m * kd, 71);
    let x = gen_x(n * kd, 73);
    let mut cpu = vec![0f32; n * m];
    for ni in 0..n {
        for mi in 0..m {
            let mut acc = 0f32;
            for ki in 0..kd {
                acc += w[mi * kd + ki] * x[ni * kd + ki];
            }
            cpu[ni * m + mi] = acc;
        }
    }
    let mut gpu = vec![0f32; n * m];
    let res = sk::gemm_bf16_xmx_f32_host(stream, &w, &x, &mut gpu, m, kd, n);
    let ms = started.elapsed().as_millis();
    match res {
        Ok(true) => {
            let (cos, max_rel) = compare(&gpu, &cpu);
            let ok = cos > 0.98 && max_rel < 0.10;
            emit(
                name,
                if ok { "OK" } else { "MISCOMPUTE" },
                &format!("cos={cos:.6} max_rel={max_rel:.4} ms={ms}"),
            );
        }
        Ok(false) => emit(name, "SKIP", &format!("xmx-gemm-unavailable ms={ms}")),
        Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
    }
}

fn probe_matvec(stream: &sk::SyclStream, name: &str, dtype: &str, started: Instant) {
    let layout = LAYOUTS.iter().find(|l| l.name == dtype).expect("layout");
    let x = gen_x(MV_K, 42);
    let Some((w, cpu_out)) = finite_ref(layout, cpu_matvec_for(dtype), &x) else {
        emit(name, "SKIP", "no-finite-reference");
        return;
    };
    let (mut wb, mut xb, mut ob) = match (
        sk::SyclSharedBuffer::<u8>::alloc(stream, w.len()),
        sk::SyclSharedBuffer::<f32>::alloc(stream, MV_K),
        sk::SyclSharedBuffer::<f32>::alloc(stream, MV_M),
    ) {
        (Ok(a), Ok(b), Ok(c)) => (a, b, c),
        _ => {
            emit(name, "KERNEL_ERR", "usm-alloc-failed");
            return;
        }
    };
    wb.as_mut_slice().copy_from_slice(&w);
    xb.as_mut_slice().copy_from_slice(&x);
    ob.as_mut_slice().fill(f32::NAN);
    let gpu = gpu_matvec_for(dtype);
    // SAFETY: all pointers are live USM allocations on `stream` of
    // the exact lengths the kernel derives from (m, k); the kernel
    // waits before returning.
    let res = unsafe {
        gpu(
            stream,
            wb.as_ptr(),
            xb.as_ptr(),
            ob.as_mut_ptr(),
            MV_M as u32,
            MV_K as u32,
            0,
        )
    };
    let ms = started.elapsed().as_millis();
    match res {
        Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
        Ok(()) => {
            let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
            let verdict = if cos > 0.999 && max_rel < 0.02 {
                "OK"
            } else {
                "MISCOMPUTE"
            };
            emit(
                name,
                verdict,
                &format!("cos={cos:.6} max_rel={max_rel:.4} ms={ms}"),
            );
        }
    }
}

/// P5b — F16-activation (mixed-precision) matvec probe: grades the f16-in kernel
/// vs the FAIR f16-activation-roundtrip reference (so it checks kernel
/// correctness, not the inherent f16 loss) AND perf-gates it (the path is LOSSY,
/// so it auto-enables only where it is also FASTER than the f32-activation
/// matvec). Verdict key `matvec_f16in:<dtype>`. A `MISCOMPUTE` here on this Iris
/// Xe is EXPECTED+correct for an affected dtype — it keeps the lossy path OFF.
fn probe_matvec_f16in(stream: &sk::SyclStream, name: &str, dtype: &str, started: Instant) {
    let layout = LAYOUTS.iter().find(|l| l.name == dtype).expect("layout");
    let Some(gpu_f16) = gpu_matvec_f16in_for(dtype) else {
        emit(name, "SKIP", "no-f16in-kernel");
        return;
    };
    let gpu_f32 = gpu_matvec_for(dtype);
    let cpu = cpu_matvec_for(dtype);
    let x = gen_x(MV_K, 42);
    // Pick a weight whose f32 CPU output is finite (same picker as the f32 probe).
    let Some((w, _)) = finite_ref(layout, cpu, &x) else {
        emit(name, "SKIP", "no-finite-reference");
        return;
    };
    // Fair reference: the f32 matvec run on f16-roundtripped activations.
    let xr = f16_act_roundtrip(&x);
    let mut cpu_out = vec![0f32; MV_M];
    cpu(&w, &xr, &mut cpu_out, MV_M, MV_K);
    if !cpu_out.iter().all(|v| v.is_finite()) {
        emit(name, "SKIP", "no-finite-reference");
        return;
    }
    let x16: Vec<u16> = x.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
    let (mut wb, mut xb16, mut xbf32, mut ob) = match (
        sk::SyclSharedBuffer::<u8>::alloc(stream, w.len()),
        sk::SyclSharedBuffer::<u16>::alloc(stream, MV_K),
        sk::SyclSharedBuffer::<f32>::alloc(stream, MV_K),
        sk::SyclSharedBuffer::<f32>::alloc(stream, MV_M),
    ) {
        (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
        _ => {
            emit(name, "KERNEL_ERR", "usm-alloc-failed");
            return;
        }
    };
    wb.as_mut_slice().copy_from_slice(&w);
    xb16.as_mut_slice().copy_from_slice(&x16);
    xbf32.as_mut_slice().copy_from_slice(&x);
    ob.as_mut_slice().fill(f32::NAN);
    // Correctness: run the f16-in kernel once, grade vs the fair reference.
    // SAFETY: live USM allocations on `stream` sized (M,K)/(K)/(M); the kernel
    // derives extents from (m,k) and waits before returning.
    if let Err(e) =
        unsafe { gpu_f16(stream, wb.as_ptr(), xb16.as_ptr(), ob.as_mut_ptr(), MV_M as u32, MV_K as u32, 0) }
    {
        emit(name, "KERNEL_ERR", &format!("{e}"));
        return;
    }
    let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
    if !(cos > 0.999 && max_rel < 0.02) {
        emit(name, "MISCOMPUTE", &format!("cos={cos:.6} max_rel={max_rel:.4}"));
        return;
    }
    // Perf gate: the f16-in path is lossy, so enable it ONLY when it also beats
    // the f32-activation matvec. Warm once, then time ITERS dispatches of each.
    const ITERS: u32 = 40;
    let _ = unsafe { gpu_f16(stream, wb.as_ptr(), xb16.as_ptr(), ob.as_mut_ptr(), MV_M as u32, MV_K as u32, 0) };
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = unsafe { gpu_f16(stream, wb.as_ptr(), xb16.as_ptr(), ob.as_mut_ptr(), MV_M as u32, MV_K as u32, 0) };
    }
    let t_f16 = t0.elapsed().as_secs_f64();
    let _ = unsafe { gpu_f32(stream, wb.as_ptr(), xbf32.as_ptr(), ob.as_mut_ptr(), MV_M as u32, MV_K as u32, 0) };
    let t1 = Instant::now();
    for _ in 0..ITERS {
        let _ = unsafe { gpu_f32(stream, wb.as_ptr(), xbf32.as_ptr(), ob.as_mut_ptr(), MV_M as u32, MV_K as u32, 0) };
    }
    let t_f32 = t1.elapsed().as_secs_f64();
    let ms = started.elapsed().as_millis();
    // C-B: the f16-activation path is LOSSY, so require a CONSERVATIVE ≥10%
    // speedup (`t_f16 < 0.9·t_f32`), not the strict `t_f16 < t_f32` that flipped
    // ON/off run-to-run on ~2% near-ties. A borderline-faster lossy path is not
    // worth the quality hit, and the margin keeps the verdict stable across
    // noisy micro-bench runs. Fail-closed on a tie → SLOW → off.
    const F16_MARGIN: f64 = 0.9;
    if t_f16 < F16_MARGIN * t_f32 {
        emit(
            name,
            "OK",
            &format!("cos={cos:.6} max_rel={max_rel:.4} f16={:.2}ms f32={:.2}ms (≥10% faster) ms={ms}", t_f16 * 1e3, t_f32 * 1e3),
        );
    } else {
        emit(
            name,
            "SLOW",
            &format!("correct but not ≥10% faster: f16={:.2}ms f32={:.2}ms ms={ms}", t_f16 * 1e3, t_f32 * 1e3),
        );
    }
}

/// Prefill micro-batch width for the batched-matvec probe. The accel.rs
/// batched-GEMM dispatch engages at `n >= 16`, so 16 is the realistic smallest
/// batch and keeps buffers modest (16×2048 f32 x, 16×512 f32 out).
const MV_N: usize = 16;

/// Batched (prefill) packed-matvec probe: `out[n*M + m] = W[m,:]·x[n,:]` over N
/// activation rows in one launch, graded against the CPU matvec run per-row.
/// The single-row `probe_matvec` never touches the batched kernels, so this is
/// their only parity check.
fn probe_matvec_batched(stream: &sk::SyclStream, name: &str, dtype: &str, started: Instant) {
    let layout = LAYOUTS.iter().find(|l| l.name == dtype).expect("layout");
    let Some(gpu) = gpu_matvec_batched_for(dtype) else {
        emit(name, "SKIP", "no-batched-kernel");
        return;
    };
    let cpu = cpu_matvec_for(dtype);
    // N distinct activation rows; reuse finite_ref (on row 0) to pick a weight
    // whose CPU output is finite, then run the CPU reference for every row.
    let x = gen_x(MV_N * MV_K, 42);
    let Some((w, _)) = finite_ref(layout, cpu, &x[0..MV_K]) else {
        emit(name, "SKIP", "no-finite-reference");
        return;
    };
    let mut cpu_out = vec![0f32; MV_N * MV_M];
    for n in 0..MV_N {
        let xr = &x[n * MV_K..(n + 1) * MV_K];
        cpu(&w, xr, &mut cpu_out[n * MV_M..(n + 1) * MV_M], MV_M, MV_K);
    }
    if !cpu_out.iter().all(|v| v.is_finite()) {
        emit(name, "SKIP", "no-finite-reference");
        return;
    }
    let (mut wb, mut xb, mut ob) = match (
        sk::SyclSharedBuffer::<u8>::alloc(stream, w.len()),
        sk::SyclSharedBuffer::<f32>::alloc(stream, MV_N * MV_K),
        sk::SyclSharedBuffer::<f32>::alloc(stream, MV_N * MV_M),
    ) {
        (Ok(a), Ok(b), Ok(c)) => (a, b, c),
        _ => {
            emit(name, "KERNEL_ERR", "usm-alloc-failed");
            return;
        }
    };
    wb.as_mut_slice().copy_from_slice(&w);
    xb.as_mut_slice().copy_from_slice(&x);
    ob.as_mut_slice().fill(f32::NAN);
    // SAFETY: three live USM allocations on `stream` sized (M,K)/(N,K)/(N,M);
    // the kernel derives all extents from (m, k, n) and waits before returning.
    let res = unsafe {
        gpu(
            stream,
            wb.as_ptr(),
            xb.as_ptr(),
            ob.as_mut_ptr(),
            MV_M as u32,
            MV_K as u32,
            MV_N as u32,
            0,
        )
    };
    let ms = started.elapsed().as_millis();
    match res {
        Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
        Ok(()) => {
            let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
            let verdict = if cos > 0.999 && max_rel < 0.02 {
                "OK"
            } else {
                "MISCOMPUTE"
            };
            emit(
                name,
                verdict,
                &format!("cos={cos:.6} max_rel={max_rel:.4} ms={ms} n={MV_N}"),
            );
        }
    }
}

/// B1: parity for the fused residual-add + RMSNorm F32 kernel
/// (`rsl_add_rmsnorm_f32_usm`) vs the CPU reference (`add_inplace_f32` then
/// per-row `rmsnorm_f32_row`). F32 end-to-end — the fix for the f16 round-trip
/// that drifted the residual stream on every SYCL decode layer. Grades BOTH the
/// written-back residual (`hidden`) and the normalized output (`y_norm`).
fn probe_add_rmsnorm_f32(stream: &sk::SyclStream, name: &str, started: Instant) {
    const D: usize = 256;
    const N: usize = 3;
    const EPS: f32 = 1e-5;
    let total = N * D;
    let hidden = gen_x(total, 61);
    let branch = gen_x(total, 62);
    let w = gen_x(D, 63);
    let mut cpu_hidden = hidden.clone();
    k::add_inplace_f32(&mut cpu_hidden, &branch);
    let mut cpu_y = vec![0f32; total];
    for r in 0..N {
        k::rmsnorm_f32_row(
            &cpu_hidden[r * D..(r + 1) * D],
            &w,
            &mut cpu_y[r * D..(r + 1) * D],
            EPS,
        );
    }
    let bufs = (
        sk::SyclSharedBuffer::<f32>::alloc(stream, total),
        sk::SyclSharedBuffer::<f32>::alloc(stream, total),
        sk::SyclSharedBuffer::<f32>::alloc(stream, D),
        sk::SyclSharedBuffer::<f32>::alloc(stream, total),
    );
    let (mut hb, mut bb, mut wb, mut yb) = match bufs {
        (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
        _ => {
            emit(name, "KERNEL_ERR", "usm-alloc-failed");
            return;
        }
    };
    hb.as_mut_slice().copy_from_slice(&hidden);
    bb.as_mut_slice().copy_from_slice(&branch);
    wb.as_mut_slice().copy_from_slice(&w);
    yb.as_mut_slice().fill(f32::NAN);
    let res = sk::add_rmsnorm_f32_usm_raw(
        stream,
        hb.as_mut_ptr(),
        bb.as_ptr(),
        wb.as_ptr(),
        yb.as_mut_ptr(),
        N as u32,
        D as u32,
        EPS,
    );
    let ms = started.elapsed().as_millis();
    match res {
        Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
        Ok(()) => {
            let (ch, rh) = compare(hb.as_slice(), &cpu_hidden);
            let (cy, ry) = compare(yb.as_slice(), &cpu_y);
            let (cos, max_rel) = (ch.min(cy), rh.max(ry));
            let verdict = if cos > 0.9999 && max_rel < 0.01 {
                "OK"
            } else {
                "MISCOMPUTE"
            };
            emit(
                name,
                verdict,
                &format!("cos={cos:.6} max_rel={max_rel:.4} ms={ms}"),
            );
        }
    }
}

/// F2: parity for the F32 RMSNorm USM kernel (`rsl_rmsnorm_f32_usm`) vs CPU
/// `rmsnorm_f32_row`. No f16 round-trip of the normalized activation.
fn probe_rmsnorm_f32(stream: &sk::SyclStream, name: &str, started: Instant) {
    const D: usize = 256;
    const N: usize = 3;
    const EPS: f32 = 1e-5;
    let total = N * D;
    let x = gen_x(total, 64);
    let w = gen_x(D, 65);
    let mut cpu_y = vec![0f32; total];
    for r in 0..N {
        k::rmsnorm_f32_row(
            &x[r * D..(r + 1) * D],
            &w,
            &mut cpu_y[r * D..(r + 1) * D],
            EPS,
        );
    }
    let bufs = (
        sk::SyclSharedBuffer::<f32>::alloc(stream, total),
        sk::SyclSharedBuffer::<f32>::alloc(stream, D),
        sk::SyclSharedBuffer::<f32>::alloc(stream, total),
    );
    let (mut xb, mut wb, mut yb) = match bufs {
        (Ok(a), Ok(b), Ok(c)) => (a, b, c),
        _ => {
            emit(name, "KERNEL_ERR", "usm-alloc-failed");
            return;
        }
    };
    xb.as_mut_slice().copy_from_slice(&x);
    wb.as_mut_slice().copy_from_slice(&w);
    yb.as_mut_slice().fill(f32::NAN);
    // SAFETY: three USM allocations on `stream`, sized above; kernel waits.
    let res = unsafe {
        sk::rmsnorm_f32_usm_raw(
            stream,
            xb.as_ptr(),
            wb.as_ptr(),
            yb.as_mut_ptr(),
            N as u32,
            D as u32,
            EPS,
        )
    };
    let ms = started.elapsed().as_millis();
    match res {
        Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
        Ok(()) => {
            let (cos, max_rel) = compare(yb.as_slice(), &cpu_y);
            let verdict = if cos > 0.9999 && max_rel < 0.01 {
                "OK"
            } else {
                "MISCOMPUTE"
            };
            emit(
                name,
                verdict,
                &format!("cos={cos:.6} max_rel={max_rel:.4} ms={ms}"),
            );
        }
    }
}

/// F2: parity for the F32 SwiGLU `silu(x)*y` USM kernel
/// (`rsl_silu_mul_f32_usm`) vs CPU `silu_mul_f32`. No f16 round-trip.
fn probe_silu_mul_f32(stream: &sk::SyclStream, name: &str, started: Instant) {
    const N: usize = 512;
    let x = gen_x(N, 66);
    let y = gen_x(N, 67);
    let mut cpu_out = vec![0f32; N];
    k::silu_mul_f32(&x, &y, &mut cpu_out);
    let bufs = (
        sk::SyclSharedBuffer::<f32>::alloc(stream, N),
        sk::SyclSharedBuffer::<f32>::alloc(stream, N),
        sk::SyclSharedBuffer::<f32>::alloc(stream, N),
    );
    let (mut xb, mut yb, mut ob) = match bufs {
        (Ok(a), Ok(b), Ok(c)) => (a, b, c),
        _ => {
            emit(name, "KERNEL_ERR", "usm-alloc-failed");
            return;
        }
    };
    xb.as_mut_slice().copy_from_slice(&x);
    yb.as_mut_slice().copy_from_slice(&y);
    ob.as_mut_slice().fill(f32::NAN);
    // SAFETY: three USM allocations on `stream`, sized above; kernel waits.
    let res = unsafe {
        sk::silu_mul_f32_usm_raw(stream, xb.as_ptr(), yb.as_ptr(), ob.as_mut_ptr(), N as u32)
    };
    let ms = started.elapsed().as_millis();
    match res {
        Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
        Ok(()) => {
            let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
            let verdict = if cos > 0.9999 && max_rel < 0.01 {
                "OK"
            } else {
                "MISCOMPUTE"
            };
            emit(
                name,
                verdict,
                &format!("cos={cos:.6} max_rel={max_rel:.4} ms={ms}"),
            );
        }
    }
}

/// P17: parity for the GPU Gated-DeltaNet chunked-prefill kernel
/// (`rsl_delta_net_chunk_f32_usm`) vs the CPU chunked reference
/// (`delta_rule_prefill_chunked_with_chunk`) at the same chunk length, batched
/// over all V-heads. Grades BOTH the per-token outputs and the final recurrent
/// state (the recurrence carries state across chunks, so a boundary bug shows in
/// the state even when the early tokens look right). f32 end to end; the chunked
/// scan is tolerance-equal to the per-token recurrence (≈1e-3).
fn probe_delta_net(stream: &sk::SyclStream, name: &str, _rest: &str, started: Instant) {
    const H: usize = 4; // V-heads
    const T: usize = 40; // sequence length (spans several chunks)
    const QK: usize = 16; // head_qk_dim
    const VD: usize = 24; // head_v_dim
    const L0: usize = 8; // chunk length

    // Head-major packed inputs, identical for CPU and GPU. q/k are L2-normed per
    // token (as the real call site does before packing); g ≤ 0; beta ∈ (0.1,0.9).
    let mut q = gen_x(H * T * QK, 71);
    let mut kk = gen_x(H * T * QK, 72);
    let v = gen_x(H * T * VD, 73);
    for row in 0..H * T {
        k::delta_net::l2norm_f32_inplace(&mut q[row * QK..(row + 1) * QK], 1e-6);
        k::delta_net::l2norm_f32_inplace(&mut kk[row * QK..(row + 1) * QK], 1e-6);
    }
    let graw = gen_x(H * T, 74);
    let braw = gen_x(H * T, 75);
    let g: Vec<f32> = graw.iter().map(|x| -0.05 - 0.25 * x.abs()).collect();
    let beta: Vec<f32> = braw.iter().map(|x| 0.1 + 0.8 * x.abs()).collect();

    // CPU reference, per head, state starting at zero.
    let mut cpu_out = vec![0f32; H * T * VD];
    let mut cpu_state = vec![0f32; H * QK * VD];
    for h in 0..H {
        k::delta_net::delta_rule_prefill_chunked_with_chunk(
            &q[h * T * QK..(h + 1) * T * QK],
            &kk[h * T * QK..(h + 1) * T * QK],
            &v[h * T * VD..(h + 1) * T * VD],
            &g[h * T..(h + 1) * T],
            &beta[h * T..(h + 1) * T],
            &mut cpu_state[h * QK * VD..(h + 1) * QK * VD],
            &mut cpu_out[h * T * VD..(h + 1) * T * VD],
            T,
            QK,
            VD,
            L0,
        );
    }

    // GPU: seven USM buffers, run the kernel, read out + state back.
    let bufs = (
        sk::SyclSharedBuffer::<f32>::alloc(stream, H * T * QK),
        sk::SyclSharedBuffer::<f32>::alloc(stream, H * T * QK),
        sk::SyclSharedBuffer::<f32>::alloc(stream, H * T * VD),
        sk::SyclSharedBuffer::<f32>::alloc(stream, H * T),
        sk::SyclSharedBuffer::<f32>::alloc(stream, H * T),
        sk::SyclSharedBuffer::<f32>::alloc(stream, H * QK * VD),
        sk::SyclSharedBuffer::<f32>::alloc(stream, H * T * VD),
    );
    let (mut qb, mut kb, mut vb, mut gb, mut bb, mut sb, mut ob) = match bufs {
        (Ok(a), Ok(b), Ok(c), Ok(d), Ok(e), Ok(f), Ok(gg)) => (a, b, c, d, e, f, gg),
        _ => {
            emit(name, "KERNEL_ERR", "usm-alloc-failed");
            return;
        }
    };
    qb.as_mut_slice().copy_from_slice(&q);
    kb.as_mut_slice().copy_from_slice(&kk);
    vb.as_mut_slice().copy_from_slice(&v);
    gb.as_mut_slice().copy_from_slice(&g);
    bb.as_mut_slice().copy_from_slice(&beta);
    sb.as_mut_slice().fill(0.0); // state starts at zero
    ob.as_mut_slice().fill(f32::NAN);
    // SAFETY: seven USM allocations on `stream`, sized above; kernel waits.
    let res = unsafe {
        sk::delta_net_chunk_f32_usm_raw(
            stream,
            qb.as_ptr(),
            kb.as_ptr(),
            vb.as_ptr(),
            gb.as_ptr(),
            bb.as_ptr(),
            sb.as_mut_ptr(),
            ob.as_mut_ptr(),
            H as u32,
            T as u32,
            QK as u32,
            VD as u32,
            L0 as u32,
        )
    };
    let ms = started.elapsed().as_millis();
    match res {
        Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
        Ok(()) => {
            let (co, ro) = compare(ob.as_slice(), &cpu_out);
            let (cs, rs) = compare(sb.as_slice(), &cpu_state);
            let (cos, max_rel) = (co.min(cs), ro.max(rs));
            let verdict = if cos > 0.999 && max_rel < 0.02 {
                "OK"
            } else {
                "MISCOMPUTE"
            };
            emit(
                name,
                verdict,
                &format!("cos={cos:.6} max_rel={max_rel:.4} ms={ms}"),
            );
        }
    }
}

/// P6-fusions: parity for the fused RMSNorm(input)+matvec f32 kernel
/// (`rsl_rmsnorm_matvec_f32_usm`) vs CPU `rmsnorm_f32_row` then `matvec_f32`,
/// per row. Bit-tolerant (f32, ~1e-4).
fn probe_rmsnorm_matvec(stream: &sk::SyclStream, name: &str, started: Instant) {
    const N: usize = 2; // rows
    const K: usize = 256; // input dim
    const M: usize = 128; // output dim
    const EPS: f32 = 1e-5;
    let x = gen_x(N * K, 81);
    let wnorm = gen_x(K, 82);
    let wmat = gen_x(M * K, 83);
    let mut cpu = vec![0f32; N * M];
    for r in 0..N {
        let mut xn = vec![0f32; K];
        k::rmsnorm_f32_row(&x[r * K..(r + 1) * K], &wnorm, &mut xn, EPS);
        k::matvec_f32(&wmat, &xn, &mut cpu[r * M..(r + 1) * M], M, K);
    }
    let bufs = (
        sk::SyclSharedBuffer::<f32>::alloc(stream, N * K),
        sk::SyclSharedBuffer::<f32>::alloc(stream, K),
        sk::SyclSharedBuffer::<f32>::alloc(stream, M * K),
        sk::SyclSharedBuffer::<f32>::alloc(stream, N * M),
    );
    let (mut xb, mut wnb, mut wmb, mut yb) = match bufs {
        (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
        _ => {
            emit(name, "KERNEL_ERR", "usm-alloc-failed");
            return;
        }
    };
    xb.as_mut_slice().copy_from_slice(&x);
    wnb.as_mut_slice().copy_from_slice(&wnorm);
    wmb.as_mut_slice().copy_from_slice(&wmat);
    yb.as_mut_slice().fill(f32::NAN);
    // SAFETY: four USM allocations on `stream`, sized above; kernel waits.
    let res = unsafe {
        sk::rmsnorm_matvec_f32_usm_raw(
            stream,
            xb.as_ptr(),
            wnb.as_ptr(),
            wmb.as_ptr(),
            yb.as_mut_ptr(),
            N as u32,
            K as u32,
            M as u32,
            EPS,
        )
    };
    let ms = started.elapsed().as_millis();
    match res {
        Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
        Ok(()) => {
            let (cos, max_rel) = compare(yb.as_slice(), &cpu);
            let verdict = if cos > 0.9999 && max_rel < 0.01 {
                "OK"
            } else {
                "MISCOMPUTE"
            };
            emit(name, verdict, &format!("cos={cos:.6} max_rel={max_rel:.4} ms={ms}"));
        }
    }
}

/// P6-fusions: parity for the fused FFN down-proj + residual-add f32 kernel
/// (`rsl_down_residual_f32_usm`) vs CPU `matvec_f32` then `add_inplace_f32`,
/// per row. Bit-tolerant (f32, ~1e-4).
fn probe_down_residual(stream: &sk::SyclStream, name: &str, started: Instant) {
    const N: usize = 2;
    const K: usize = 256;
    const M: usize = 128;
    let ff = gen_x(N * K, 84);
    let wmat = gen_x(M * K, 85);
    let resid = gen_x(N * M, 86);
    let mut cpu = resid.clone();
    for r in 0..N {
        let mut dot = vec![0f32; M];
        k::matvec_f32(&wmat, &ff[r * K..(r + 1) * K], &mut dot, M, K);
        k::add_inplace_f32(&mut cpu[r * M..(r + 1) * M], &dot);
    }
    let bufs = (
        sk::SyclSharedBuffer::<f32>::alloc(stream, N * K),
        sk::SyclSharedBuffer::<f32>::alloc(stream, M * K),
        sk::SyclSharedBuffer::<f32>::alloc(stream, N * M),
        sk::SyclSharedBuffer::<f32>::alloc(stream, N * M),
    );
    let (mut ffb, mut wmb, mut rb, mut yb) = match bufs {
        (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
        _ => {
            emit(name, "KERNEL_ERR", "usm-alloc-failed");
            return;
        }
    };
    ffb.as_mut_slice().copy_from_slice(&ff);
    wmb.as_mut_slice().copy_from_slice(&wmat);
    rb.as_mut_slice().copy_from_slice(&resid);
    yb.as_mut_slice().fill(f32::NAN);
    // SAFETY: four USM allocations on `stream`, sized above; kernel waits.
    let res = unsafe {
        sk::down_residual_f32_usm_raw(
            stream,
            ffb.as_ptr(),
            wmb.as_ptr(),
            rb.as_ptr(),
            yb.as_mut_ptr(),
            N as u32,
            K as u32,
            M as u32,
        )
    };
    let ms = started.elapsed().as_millis();
    match res {
        Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
        Ok(()) => {
            let (cos, max_rel) = compare(yb.as_slice(), &cpu);
            let verdict = if cos > 0.9999 && max_rel < 0.01 {
                "OK"
            } else {
                "MISCOMPUTE"
            };
            emit(name, verdict, &format!("cos={cos:.6} max_rel={max_rel:.4} ms={ms}"));
        }
    }
}

/// P6-fusions: parity for the out-proj + residual + post-attn-norm fusion. Runs
/// the EXACT chained kernels the `try_matvec_out_proj_add_rmsnorm_usm_f32`
/// dispatcher uses — a packed Q4_K matvec (`matvec_q4_k_packed_f32_usm`) into a
/// dot buffer, then `add_rmsnorm_f32_usm` (`hidden += dot`; `y = rmsnorm(hidden)
/// · w_norm`) — vs the CPU reference (Q4_K matvec, `add_inplace_f32`,
/// `rmsnorm_f32_row`). Grades BOTH the normalized output AND the written-back
/// residual. This is the composite verdict the P5b gate was waiting on.
fn probe_out_proj_norm(stream: &sk::SyclStream, name: &str, started: Instant) {
    const M: usize = 256; // d_model (out), %256 for Q4_K
    const K: usize = 256; // attn dim (in), %256 for Q4_K
    const EPS: f32 = 1e-5;
    let Some(layout) = LAYOUTS.iter().find(|l| l.name == "q4_k") else {
        emit(name, "SKIP", "no-q4_k-layout");
        return;
    };
    let attn = gen_x(K, 87);
    let cpu_mv = cpu_matvec_for("q4_k");
    let Some((wbytes, cpu_dot)) = coh_gen_finite(layout, cpu_mv, M, K, &attn, 0x0F6F0001) else {
        emit(name, "SKIP", "no-finite-reference");
        return;
    };
    let hidden = gen_x(M, 88);
    let wnorm = gen_x(M, 89);
    // CPU reference: hidden += dot; y = rmsnorm(hidden) * wnorm.
    let mut cpu_hidden = hidden.clone();
    k::add_inplace_f32(&mut cpu_hidden, &cpu_dot);
    let mut cpu_y = vec![0f32; M];
    k::rmsnorm_f32_row(&cpu_hidden, &wnorm, &mut cpu_y, EPS);
    // GPU chained path.
    let bufs = (
        sk::SyclSharedBuffer::<u8>::alloc(stream, wbytes.len()),
        sk::SyclSharedBuffer::<f32>::alloc(stream, K),
        sk::SyclSharedBuffer::<f32>::alloc(stream, M),
        sk::SyclSharedBuffer::<f32>::alloc(stream, M),
        sk::SyclSharedBuffer::<f32>::alloc(stream, M),
        sk::SyclSharedBuffer::<f32>::alloc(stream, M),
    );
    let (mut wb, mut ab, mut dotb, mut hb, mut wnb, mut yb) = match bufs {
        (Ok(a), Ok(b), Ok(c), Ok(d), Ok(e), Ok(f)) => (a, b, c, d, e, f),
        _ => {
            emit(name, "KERNEL_ERR", "usm-alloc-failed");
            return;
        }
    };
    wb.as_mut_slice().copy_from_slice(&wbytes);
    ab.as_mut_slice().copy_from_slice(&attn);
    hb.as_mut_slice().copy_from_slice(&hidden);
    wnb.as_mut_slice().copy_from_slice(&wnorm);
    dotb.as_mut_slice().fill(f32::NAN);
    yb.as_mut_slice().fill(f32::NAN);
    // Stage 1: packed Q4_K matvec → dot. SAFETY: USM allocations on `stream`.
    let r1 = unsafe {
        sk::matvec_q4_k_packed_f32_usm_raw(
            stream,
            wb.as_ptr(),
            ab.as_ptr(),
            dotb.as_mut_ptr(),
            M as u32,
            K as u32,
            0,
        )
    };
    if let Err(e) = r1 {
        emit(name, "KERNEL_ERR", &format!("matvec {e}"));
        return;
    }
    // Stage 2: F32 add+rmsnorm (1 row of width M). SAFETY: as above.
    let r2 = sk::add_rmsnorm_f32_usm_raw(
        stream,
        hb.as_mut_ptr(),
        dotb.as_ptr(),
        wnb.as_ptr(),
        yb.as_mut_ptr(),
        1,
        M as u32,
        EPS,
    );
    let ms = started.elapsed().as_millis();
    match r2 {
        Err(e) => emit(name, "KERNEL_ERR", &format!("add_rmsnorm {e} ms={ms}")),
        Ok(()) => {
            let (cy, ry) = compare(yb.as_slice(), &cpu_y);
            let (ch, rh) = compare(hb.as_slice(), &cpu_hidden);
            let (cos, max_rel) = (cy.min(ch), ry.max(rh));
            let verdict = if cos > 0.999 && max_rel < 0.02 {
                "OK"
            } else {
                "MISCOMPUTE"
            };
            emit(name, verdict, &format!("cos={cos:.6} max_rel={max_rel:.4} ms={ms}"));
        }
    }
}

fn probe_fused(stream: &sk::SyclStream, name: &str, dtype: &str, started: Instant) {
    // B1/F2: the F32 residual/norm/activation fusions are not dtype-keyed
    // gate/up matvecs — dispatch them before the LAYOUTS lookup (which would
    // panic on these names).
    match dtype {
        "add_rmsnorm_f32" => return probe_add_rmsnorm_f32(stream, name, started),
        "rmsnorm_f32" => return probe_rmsnorm_f32(stream, name, started),
        "silu_mul_f32" => return probe_silu_mul_f32(stream, name, started),
        // P6-fusions.
        "rmsnorm_matvec" => return probe_rmsnorm_matvec(stream, name, started),
        "down_residual" => return probe_down_residual(stream, name, started),
        "out_proj_norm" => return probe_out_proj_norm(stream, name, started),
        _ => {}
    }
    let layout = LAYOUTS.iter().find(|l| l.name == dtype).expect("layout");
    let x = gen_x(MV_K, 43);
    let cpu = cpu_matvec_for(dtype);
    let Some((wg, cpu_gate)) = finite_ref(layout, cpu, &x) else {
        emit(name, "SKIP", "no-finite-reference");
        return;
    };
    // Second, distinct weight tensor for `up`.
    let wu = gen_quant_bytes(layout, MV_M, MV_K, 0xBEEF0007);
    let mut cpu_up = vec![0f32; MV_M];
    cpu(&wu, &x, &mut cpu_up, MV_M, MV_K);
    if !cpu_up.iter().all(|v| v.is_finite()) {
        emit(name, "SKIP", "no-finite-reference");
        return;
    }
    let bufs = (
        sk::SyclSharedBuffer::<u8>::alloc(stream, wg.len()),
        sk::SyclSharedBuffer::<u8>::alloc(stream, wu.len()),
        sk::SyclSharedBuffer::<f32>::alloc(stream, MV_K),
        sk::SyclSharedBuffer::<f32>::alloc(stream, MV_M),
        sk::SyclSharedBuffer::<f32>::alloc(stream, MV_M),
    );
    let (mut gb, mut ub, mut xb, mut gob, mut uob) = match bufs {
        (Ok(a), Ok(b), Ok(c), Ok(d), Ok(e)) => (a, b, c, d, e),
        _ => {
            emit(name, "KERNEL_ERR", "usm-alloc-failed");
            return;
        }
    };
    gb.as_mut_slice().copy_from_slice(&wg);
    ub.as_mut_slice().copy_from_slice(&wu);
    xb.as_mut_slice().copy_from_slice(&x);
    gob.as_mut_slice().fill(f32::NAN);
    uob.as_mut_slice().fill(f32::NAN);
    let Some(gpu) = gpu_fused_for(dtype) else {
        emit(name, "SKIP", "no-fused-kernel");
        return;
    };
    // SAFETY: as in `probe_matvec`; two weight tensors, two outputs.
    let res = unsafe {
        gpu(
            stream,
            gb.as_ptr(),
            ub.as_ptr(),
            xb.as_ptr(),
            gob.as_mut_ptr(),
            uob.as_mut_ptr(),
            MV_M as u32,
            MV_K as u32,
            0,
        )
    };
    let ms = started.elapsed().as_millis();
    match res {
        Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
        Ok(()) => {
            let (cg, rg) = compare(gob.as_slice(), &cpu_gate);
            let (cu, ru) = compare(uob.as_slice(), &cpu_up);
            let (cos, max_rel) = (cg.min(cu), rg.max(ru));
            let verdict = if cos > 0.999 && max_rel < 0.02 {
                "OK"
            } else {
                "MISCOMPUTE"
            };
            emit(
                name,
                verdict,
                &format!("cos={cos:.6} max_rel={max_rel:.4} ms={ms}"),
            );
        }
    }
}

/// C-A — real-MoE-forward coherence probe (verdict `moe:gate_up_coherent:<dtype>`).
/// For each realistic expert shape in [`MOE_COH_SHAPES`], builds
/// [`MOE_COH_EXPERTS`] distinct experts and runs the per-expert FFN the model
/// executes — `silu(gate)·up`, routing-weighted summed — THREE ways: the GPU
/// FUSED gate/up kernel, the GPU SEPARATE two-matvec path, and the CPU
/// reference. It then grades the fused intermediate against BOTH the separate
/// path (does fusing perturb the answer? — the exact thing the enabled vs
/// disabled dispatch differs by) AND the CPU reference (is the fused math right
/// at all?). `OK` only if every shape passes both; `MISCOMPUTE` otherwise —
/// which is the correct fail-closed verdict for a device/shape where fusing
/// diverges, the 2026-09-06 failure mode the bare `fused:` GEMM probe misses.
fn probe_moe_gate_up_coherent(
    stream: &sk::SyclStream,
    name: &str,
    dtype: &str,
    started: Instant,
) {
    let layout = LAYOUTS.iter().find(|l| l.name == dtype).expect("layout");
    let cpu = cpu_matvec_for(dtype);
    let Some(gpu_fused) = gpu_fused_for(dtype) else {
        emit(name, "SKIP", "no-fused-kernel");
        return;
    };
    let gpu_mv = gpu_matvec_for(dtype);

    // Worst case across all shapes: OK iff every shape agrees on both axes.
    let mut worst_cos_cpu = 1.0f64;
    let mut worst_rel_cpu = 0.0f64;
    let mut worst_cos_sep = 1.0f64;
    let mut worst_rel_sep = 0.0f64;

    for (si, &(m, k)) in MOE_COH_SHAPES.iter().enumerate() {
        let x = gen_x(k, 0x5EED_0001 ^ (si as u32).wrapping_mul(7919));
        // Device activation buffer (shared by every expert's 3 dispatches).
        let Ok(mut xb) = sk::SyclSharedBuffer::<f32>::alloc(stream, k) else {
            emit(name, "KERNEL_ERR", "usm-alloc-failed(x)");
            return;
        };
        xb.as_mut_slice().copy_from_slice(&x);

        let mut inter_cpu = vec![0f32; m];
        let mut inter_sep = vec![0f32; m];
        let mut inter_fused = vec![0f32; m];

        for e in 0..MOE_COH_EXPERTS {
            let w_route = MOE_COH_ROUTE_W[e];
            let sg = 0xC0DE_0000u32 ^ ((si * 97 + e * 13) as u32);
            let su = 0xBEEF_0000u32 ^ ((si * 131 + e * 17) as u32);
            let Some((wg, cpu_g)) = coh_gen_finite(layout, cpu, m, k, &x, sg) else {
                emit(name, "SKIP", "no-finite-reference(gate)");
                return;
            };
            let Some((wu, cpu_u)) = coh_gen_finite(layout, cpu, m, k, &x, su) else {
                emit(name, "SKIP", "no-finite-reference(up)");
                return;
            };
            // Device weight + output buffers for this expert.
            let (mut gb, mut ub, mut sep_g_b, mut sep_u_b, mut fus_g_b, mut fus_u_b) = match (
                sk::SyclSharedBuffer::<u8>::alloc(stream, wg.len()),
                sk::SyclSharedBuffer::<u8>::alloc(stream, wu.len()),
                sk::SyclSharedBuffer::<f32>::alloc(stream, m),
                sk::SyclSharedBuffer::<f32>::alloc(stream, m),
                sk::SyclSharedBuffer::<f32>::alloc(stream, m),
                sk::SyclSharedBuffer::<f32>::alloc(stream, m),
            ) {
                (Ok(a), Ok(b), Ok(c), Ok(d), Ok(e_), Ok(f)) => (a, b, c, d, e_, f),
                _ => {
                    emit(name, "KERNEL_ERR", "usm-alloc-failed(expert)");
                    return;
                }
            };
            gb.as_mut_slice().copy_from_slice(&wg);
            ub.as_mut_slice().copy_from_slice(&wu);
            sep_g_b.as_mut_slice().fill(f32::NAN);
            sep_u_b.as_mut_slice().fill(f32::NAN);
            fus_g_b.as_mut_slice().fill(f32::NAN);
            fus_u_b.as_mut_slice().fill(f32::NAN);

            // SEPARATE path: two independent single-row matvecs.
            // SAFETY: live USM buffers sized (M,K)/(K)/(M); kernel waits before return.
            let sep = unsafe {
                gpu_mv(stream, gb.as_ptr(), xb.as_ptr(), sep_g_b.as_mut_ptr(), m as u32, k as u32, 0)
                    .and_then(|()| {
                        gpu_mv(stream, ub.as_ptr(), xb.as_ptr(), sep_u_b.as_mut_ptr(), m as u32, k as u32, 0)
                    })
            };
            if let Err(e_) = sep {
                emit(name, "KERNEL_ERR", &format!("separate-matvec {e_}"));
                return;
            }
            // FUSED path: one dispatch computes both projections.
            // SAFETY: as above; two weights, two outputs.
            let fus = unsafe {
                gpu_fused(
                    stream,
                    gb.as_ptr(),
                    ub.as_ptr(),
                    xb.as_ptr(),
                    fus_g_b.as_mut_ptr(),
                    fus_u_b.as_mut_ptr(),
                    m as u32,
                    k as u32,
                    0,
                )
            };
            if let Err(e_) = fus {
                emit(name, "KERNEL_ERR", &format!("fused {e_}"));
                return;
            }

            // Accumulate the routing-weighted SwiGLU intermediate for each path.
            let (sg_out, su_out) = (sep_g_b.as_slice(), sep_u_b.as_slice());
            let (fg_out, fu_out) = (fus_g_b.as_slice(), fus_u_b.as_slice());
            for j in 0..m {
                inter_cpu[j] += w_route * silu(cpu_g[j]) * cpu_u[j];
                inter_sep[j] += w_route * silu(sg_out[j]) * su_out[j];
                inter_fused[j] += w_route * silu(fg_out[j]) * fu_out[j];
            }
        }

        let (cos_cpu, rel_cpu) = compare(&inter_fused, &inter_cpu);
        let (cos_sep, rel_sep) = compare(&inter_fused, &inter_sep);
        worst_cos_cpu = worst_cos_cpu.min(cos_cpu);
        worst_rel_cpu = worst_rel_cpu.max(rel_cpu);
        worst_cos_sep = worst_cos_sep.min(cos_sep);
        worst_rel_sep = worst_rel_sep.max(rel_sep);
    }

    let ms = started.elapsed().as_millis();
    // fused≡CPU (correctness, SwiGLU-amplified so looser) AND fused≡separate
    // (fusion must not perturb the answer; both GPU, so tighter).
    let cpu_ok = worst_cos_cpu > 0.999 && worst_rel_cpu < 0.05;
    let sep_ok = worst_cos_sep > 0.9995 && worst_rel_sep < 0.02;
    let verdict = if cpu_ok && sep_ok { "OK" } else { "MISCOMPUTE" };
    let backend = sk::current_backend_name(0).unwrap_or("none");
    emit(
        name,
        verdict,
        &format!(
            "backend={backend} cos_cpu={worst_cos_cpu:.6} rel_cpu={worst_rel_cpu:.4} \
             cos_sep={worst_cos_sep:.6} rel_sep={worst_rel_sep:.4} ms={ms}"
        ),
    );
}

fn to_f16_buf<'s>(
    stream: &'s sk::SyclStream,
    data: &[f32],
) -> sk::Result<sk::SyclSharedBuffer<'s, u16>> {
    let mut b = sk::SyclSharedBuffer::<u16>::alloc(stream, data.len())?;
    for (dst, &s) in b.as_mut_slice().iter_mut().zip(data.iter()) {
        *dst = half::f16::from_f32(s).to_bits();
    }
    Ok(b)
}

/// Real target-model attention geometry (Qwen3.6 full-attn layers).
const AT_HEADS: usize = 16;
const AT_KV_HEADS: usize = 2;
const AT_HEAD_DIM: usize = 256;
const AT_MAX_CTX: usize = 512;
const AT_KV_LEN: usize = 333;
const AT_PREFILL_BASE: usize = 64;
const AT_PREFILL_NEW: usize = 32;

// ---------------------------------------------------------------
// Quantized-KV FlashAttention formats (shared by the SYCL + CUDA
// harnesses). Each row carries the per-block geometry needed to pack an
// f32 KV cache into the exact `[n_kv_heads, max_ctx, bytes_per_row]`
// layout the GPU flash kernels (and their CPU references in
// `mxfp_kv.rs` / `nvfp4.rs`) dequantize, plus the grading tolerance.
//
// WHY per-format tolerances (the matvec probes use one fixed gate):
// the matvec probes quantize the weight and feed the SAME bytes to both
// the GPU kernel and the CPU reference, so the quant noise cancels and a
// tight 0.999/0.02 gate isolates the kernel. Here we deliberately grade
// the quant-KV GPU output against the FULL-PRECISION f32 reference (the
// un-quantized K/V), so the gate must absorb the format's KV round-trip
// error — 4-bit MXFP4 loosest, E4M3 MXFP8 tightest, NVFP4 tighter than
// MXFP4 because its finer per-16-elem E4M3 block scale beats MXFP4's
// per-32-elem power-of-two scale. These bounds are conservative starting
// points; on-device runs on the user's HW settle the final numbers.
struct QuantKvFormat {
    name: &'static str,
    /// Elements per quant block: 32 for MXFP*, 16 for NVFP4.
    block_elems: usize,
    /// Bytes per quant block: MXFP4 17, MXFP6 25, MXFP8 33, NVFP4 9.
    block_bytes: usize,
    /// Minimum cosine similarity vs the f32 reference for an OK verdict.
    cos_min: f64,
    /// Maximum worst-element relative error for an OK verdict.
    rel_max: f64,
}

const QUANT_KV_FORMATS: &[QuantKvFormat] = &[
    QuantKvFormat { name: "mxfp4", block_elems: 32, block_bytes: 17, cos_min: 0.930, rel_max: 0.60 },
    QuantKvFormat { name: "mxfp6", block_elems: 32, block_bytes: 25, cos_min: 0.980, rel_max: 0.25 },
    QuantKvFormat { name: "mxfp8", block_elems: 32, block_bytes: 33, cos_min: 0.995, rel_max: 0.10 },
    QuantKvFormat { name: "nvfp4", block_elems: 16, block_bytes: 9, cos_min: 0.970, rel_max: 0.30 },
    // Q4_0 KV (ggml 18B/32 block, embedded f16 scale) — live on all three GPU
    // flash-decode/prefill backends but previously ungraded. 4-bit, so the same
    // tolerance band as mxfp4. Formats with SEPARATE per-row scales (Q8_0 / TQ)
    // don't fit this block struct and are a follow-up (they need a scales
    // buffer threaded through the probe + the kernel call).
    QuantKvFormat { name: "q4_0", block_elems: 32, block_bytes: 18, cos_min: 0.930, rel_max: 0.60 },
];

/// Look up a quant-KV format by the suffix of an `attn:{decode,prefill}_*`
/// probe name (e.g. `"mxfp4"`); `None` for the f32 `v1/v2/v3` variants.
fn quant_kv_format(name: &str) -> Option<&'static QuantKvFormat> {
    QUANT_KV_FORMATS.iter().find(|f| f.name == name)
}

/// Pack an f32 KV cache `[n_kv_heads, max_ctx, head_dim]` (only the first
/// `upto` timesteps populated) into `fmt`'s block layout
/// `[n_kv_heads, max_ctx, bytes_per_row]`, using the SAME per-block CPU
/// quantizers the production KV-cache writer uses — so the bytes are
/// bit-identical to what the engine hands the GPU flash kernels.
fn quantize_kv_cache(
    fmt: &QuantKvFormat,
    cache: &[f32],
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    upto: usize,
) -> Vec<u8> {
    let blocks_per_row = head_dim / fmt.block_elems;
    let bytes_per_row = blocks_per_row * fmt.block_bytes;
    let mut packed = vec![0u8; n_kv_heads * max_ctx * bytes_per_row];
    for h in 0..n_kv_heads {
        for t in 0..upto {
            let src = (h * max_ctx + t) * head_dim;
            let dst = (h * max_ctx + t) * bytes_per_row;
            for b in 0..blocks_per_row {
                let e = &cache[src + b * fmt.block_elems..src + (b + 1) * fmt.block_elems];
                let o = &mut packed[dst + b * fmt.block_bytes..dst + (b + 1) * fmt.block_bytes];
                match fmt.name {
                    "mxfp4" => k::mxfp_kv::quantize_block_mxfp4(e, o),
                    "mxfp6" => k::mxfp_kv::quantize_block_mxfp6(e, o),
                    "mxfp8" => k::mxfp_kv::quantize_block_mxfp8(e, o),
                    "nvfp4" => k::nvfp4::quantize_block(e, o),
                    // One 32-elem Q4_0 block (quantize_row handles exactly a
                    // multiple of 32 → here, a single 18B block).
                    "q4_0" => k::q4_0_kv::quantize_row(e, o),
                    _ => unreachable!("unknown quant-KV format {}", fmt.name),
                }
            }
        }
    }
    packed
}

/// Inverse of [`quantize_kv_cache`]: dequantize a packed KV cache back to the
/// f32 `[n_kv_heads, max_ctx, head_dim]` layout, using the SAME per-block CPU
/// decoders the production KV reader uses. Feeds the SAME-QUANT flash reference
/// (quantize → dequantize round-trip of the original cache) so the parity probe
/// measures the kernel's correctness, not the KV quantization loss.
fn dequant_kv_cache(
    fmt: &QuantKvFormat,
    packed: &[u8],
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    upto: usize,
) -> Vec<f32> {
    let blocks_per_row = head_dim / fmt.block_elems;
    let bytes_per_row = blocks_per_row * fmt.block_bytes;
    let mut out = vec![0f32; n_kv_heads * max_ctx * head_dim];
    for h in 0..n_kv_heads {
        for t in 0..upto {
            let src = (h * max_ctx + t) * bytes_per_row;
            let dst = (h * max_ctx + t) * head_dim;
            let prow = &packed[src..src + bytes_per_row];
            let orow = &mut out[dst..dst + head_dim];
            match fmt.name {
                "mxfp4" => k::mxfp_kv::dequantize_row_mxfp4(prow, orow),
                "mxfp6" => k::mxfp_kv::dequantize_row_mxfp6(prow, orow),
                "mxfp8" => k::mxfp_kv::dequantize_row_mxfp8(prow, orow),
                "nvfp4" => {
                    // nvfp4 exposes a per-BLOCK decoder; loop the row's blocks.
                    for b in 0..blocks_per_row {
                        k::nvfp4::dequantize_block(
                            &prow[b * fmt.block_bytes..(b + 1) * fmt.block_bytes],
                            &mut orow[b * fmt.block_elems..(b + 1) * fmt.block_elems],
                        );
                    }
                }
                // q4_0 decodes a whole row (dequantize_row handles head_dim/32
                // blocks), inverse of the quantize_row used above.
                "q4_0" => k::q4_0_kv::dequantize_row(prow, orow),
                _ => unreachable!("unknown quant-KV format {}", fmt.name),
            }
        }
    }
    out
}

// ---- Per-row-scale KV quantizers (Q8_0) ----
// Q8_0 KV does NOT use `QuantKvFormat`'s block layout: the engine's
// `KvLayer::Q8_0` stores K/V as a PLAIN i8 slab [n_kv_heads, max_ctx, head_dim]
// (one byte/element) + a SEPARATE per-row absmax f32 scale in a parallel buffer
// [n_kv_heads*max_ctx] (indexed `kv_h*max_ctx + t`). The flash kernels take the
// slab + scales as distinct arguments, so this needs its own quantizer/dequant
// + probe path. `code = round(val*127/absmax)` (symmetric), `scale = absmax/127`
// — the probe's same-quant reference dequantizes the IDENTICAL slab+scales, so
// it measures kernel correctness independent of the exact quant formula.

/// Quantize a KV cache to the Q8_0 per-row-scale layout. Returns
/// `(i8 slab [n_kv_heads*max_ctx*head_dim], f32 scales [n_kv_heads*max_ctx])`.
fn quantize_kv_q8_0(
    cache: &[f32],
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    upto: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut slab = vec![0i8; n_kv_heads * max_ctx * head_dim];
    let mut scales = vec![0f32; n_kv_heads * max_ctx];
    for h in 0..n_kv_heads {
        for t in 0..upto {
            let row = (h * max_ctx + t) * head_dim;
            let r = &cache[row..row + head_dim];
            let absmax = r.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let (scale, inv) = if absmax > 0.0 {
                (absmax / 127.0, 127.0 / absmax)
            } else {
                (0.0, 0.0)
            };
            scales[h * max_ctx + t] = scale;
            for i in 0..head_dim {
                slab[row + i] = (r[i] * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (slab, scales)
}

/// Same-quant dequant of a Q8_0 KV slab: `val = code * scale` (per row). The
/// exact inverse of [`quantize_kv_q8_0`], for the same-quant flash reference.
fn dequant_kv_q8_0(
    slab: &[i8],
    scales: &[f32],
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    upto: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; n_kv_heads * max_ctx * head_dim];
    for h in 0..n_kv_heads {
        for t in 0..upto {
            let row = (h * max_ctx + t) * head_dim;
            let scale = scales[h * max_ctx + t];
            for i in 0..head_dim {
                out[row + i] = slab[row + i] as f32 * scale;
            }
        }
    }
    out
}

/// Reinterpret an `&[i8]` slab as the `&[u8]` byte view the device-buffer
/// uploaders expect (i8 and u8 have identical layout; the kernels read the
/// bytes as signed).
fn i8_as_bytes(slab: &[i8]) -> &[u8] {
    // SAFETY: i8 and u8 are both 1-byte, same alignment; reinterpreting the
    // slice for a read-only H2D copy is sound.
    unsafe { std::slice::from_raw_parts(slab.as_ptr() as *const u8, slab.len()) }
}

// ---- Per-row-scale KV quantizers (TQ / TurboQuant) ----
// TQ KV is the engine's KvLayer::Tq: each head_dim row is WHT-rotated + packed
// to `bits`-per-element level codes (`turboquant::bytes_per_block(head_dim,
// bits)` bytes/row) with a SEPARATE per-row f32 scale [n_kv_heads*max_ctx]. The
// flash kernels take the packed slab + scales + a runtime `bits` arg. Same probe
// shape as Q8_0, plus `bits`. Parity is bits-independent (same-quant reference
// uses the identical `dequantize_row`, which reverses the WHT + scale).

/// Quantize a KV cache to the TQ per-row layout at `bits`. Returns
/// `(packed slab [n_kv_heads*max_ctx*bytes_per_block], f32 scales
/// [n_kv_heads*max_ctx])`.
fn quantize_kv_tq(
    cache: &[f32],
    bits: u8,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    upto: usize,
) -> (Vec<u8>, Vec<f32>) {
    let bpr = k::turboquant::bytes_per_block(head_dim, bits);
    let mut slab = vec![0u8; n_kv_heads * max_ctx * bpr];
    let mut scales = vec![0f32; n_kv_heads * max_ctx];
    let mut row = vec![0f32; head_dim];
    for h in 0..n_kv_heads {
        for t in 0..upto {
            let ridx = h * max_ctx + t;
            let src = ridx * head_dim;
            row.copy_from_slice(&cache[src..src + head_dim]);
            let p = ridx * bpr;
            // quantize_row WHT-rotates `row` in place + returns the per-row scale.
            scales[ridx] = k::turboquant::quantize_row(&mut row, bits, &mut slab[p..p + bpr]);
        }
    }
    (slab, scales)
}

/// Same-quant dequant of a TQ KV slab (inverse WHT + per-row scale): the exact
/// inverse of [`quantize_kv_tq`], for the same-quant flash reference.
fn dequant_kv_tq(
    slab: &[u8],
    scales: &[f32],
    bits: u8,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    upto: usize,
) -> Vec<f32> {
    let bpr = k::turboquant::bytes_per_block(head_dim, bits);
    let mut out = vec![0f32; n_kv_heads * max_ctx * head_dim];
    for h in 0..n_kv_heads {
        for t in 0..upto {
            let ridx = h * max_ctx + t;
            let p = ridx * bpr;
            let dst = ridx * head_dim;
            k::turboquant::dequantize_row(
                &slab[p..p + bpr],
                scales[ridx],
                bits,
                &mut out[dst..dst + head_dim],
            );
        }
    }
    out
}

/// Representative TQ bit-width for the parity probes. Parity is bits-independent
/// (same-quant reference), so one value exercises the pack/unpack + kernel path;
/// 2-bit is the mid-range sub-byte case (avoids tq1's sign-only + tq8's
/// byte-aligned edges).
const TQ_PROBE_BITS: u8 = 2;

/// Feature 1a parity probe: the WHOLE device-resident DECODE attention block
/// (`rsl_decode_attn_block_usm`) vs a CPU reference computing the identical
/// sequence on identical inputs. This is the only orchestration probe — the
/// individual kernels (matvec / flash / rmsnorm) are already covered above — so
/// it validates the CHAINING, the f32-activation residency, and the f16 KV-mirror
/// scatter/read. Q4_K weights, a small realistic shape, a short pre-seeded KV
/// history (so the flash spans multiple positions). cos ~1.0 / small max_rel:
/// the chain is f32 end-to-end, the only lossy step is the f16 KV mirror (same
/// rounding the per-op decode path carries), so the gate matches the f32 flash
/// probe's (cos > 0.995, max_rel < 0.10).
///
/// This probe exercises the FFN-INCLUSIVE path: it passes the dense SwiGLU FFN
/// weights (gate/up/down) so the block runs gate/up→silu→down→residual on-device
/// too, and grades the FINAL `hidden` (attention residual + FFN residual) against
/// a CPU reference that computes attention AND the same SwiGLU FFN. (When the FFN
/// runs, `h_norm2` is an unread intermediate, so it is not graded — the device
/// doesn't copy it back.) The attention-only path is covered by production decode
/// + is the fallback; this validates the bigger per-layer win.
fn probe_decode_block(stream: &sk::SyclStream, name: &str, started: Instant) {
    use std::ffi::c_void;
    const HEAD_DIM: usize = 128;
    const N_HEADS: usize = 8;
    const N_KV: usize = 2;
    const D_MODEL: usize = 1024;
    const D_FF: usize = 2048; // SwiGLU FFN inner dim (256-aligned for Q4_K)
    const MAX_CTX: usize = 64;
    const POS: usize = 5; // 5 seeded history rows + this token => kv_len = 6
    const EPS: f32 = 1e-5;
    const THETA: f32 = 10000.0;
    let d_q = N_HEADS * HEAD_DIM;
    let d_kv = N_KV * HEAD_DIM;
    let half = HEAD_DIM / 2;
    let kv_len = POS + 1;

    // Q4_K weights (M x K packed bytes): Wq [d_q x D_MODEL], Wk/Wv [d_kv x
    // D_MODEL], Wo [D_MODEL x d_q]. Same synthesis + CPU reference matvec the
    // matvec probes use, so the per-row dequant matches the device kernel. The
    // dense-FFN weights: Wgate/Wup [D_FF x D_MODEL], Wdown [D_MODEL x D_FF].
    let layout = LAYOUTS.iter().find(|l| l.name == "q4_k").expect("q4_k layout");
    let cpu_mv = cpu_matvec_for("q4_k");
    let wq = gen_quant_bytes(layout, d_q, D_MODEL, 0x1111);
    let wk = gen_quant_bytes(layout, d_kv, D_MODEL, 0x2222);
    let wv = gen_quant_bytes(layout, d_kv, D_MODEL, 0x3333);
    let wo = gen_quant_bytes(layout, D_MODEL, d_q, 0x4444);
    let wgate = gen_quant_bytes(layout, D_FF, D_MODEL, 0x5555);
    let wup = gen_quant_bytes(layout, D_FF, D_MODEL, 0x6666);
    let wdown = gen_quant_bytes(layout, D_MODEL, D_FF, 0x7777);

    // Norms, hidden, inv_freq, and the synthetic KV history.
    let mut rng = XorShift(0xDEC0DE);
    let hidden0: Vec<f32> = (0..D_MODEL).map(|_| rng.f32_pm(0.6)).collect();
    let attn_norm: Vec<f32> = (0..D_MODEL).map(|_| 0.5 + rng.f32_pm(0.2)).collect();
    let ffn_norm: Vec<f32> = (0..D_MODEL).map(|_| 0.5 + rng.f32_pm(0.2)).collect();
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| 1.0f32 / THETA.powf(2.0 * j as f32 / HEAD_DIM as f32))
        .collect();
    // History rows [0, POS) per kv-head (head-major), as f32 then f16-rounded.
    let hist_elems = N_KV * MAX_CTX * HEAD_DIM;
    let mut k_hist_f16 = vec![0u16; hist_elems];
    let mut v_hist_f16 = vec![0u16; hist_elems];
    for h in 0..N_KV {
        for t in 0..POS {
            for d in 0..HEAD_DIM {
                let idx = (h * MAX_CTX + t) * HEAD_DIM + d;
                k_hist_f16[idx] = half::f16::from_f32(rng.f32_pm(0.6)).to_bits();
                v_hist_f16[idx] = half::f16::from_f32(rng.f32_pm(0.6)).to_bits();
            }
        }
    }

    // ---- CPU reference (identical op sequence) ----
    let rope = |qk: &mut [f32], nh: usize| {
        for hh in 0..nh {
            let base = hh * HEAD_DIM;
            for j in 0..half {
                let angle = POS as f32 * inv_freq[j];
                let (s, c) = angle.sin_cos();
                let x0 = qk[base + j];
                let x1 = qk[base + j + half];
                qk[base + j] = x0 * c - x1 * s;
                qk[base + j + half] = x0 * s + x1 * c;
            }
        }
    };
    let mut hnorm = vec![0f32; D_MODEL];
    k::rmsnorm_f32_row(&hidden0, &attn_norm, &mut hnorm, EPS);
    let mut q_ref = vec![0f32; d_q];
    let mut k_ref = vec![0f32; d_kv];
    let mut v_ref = vec![0f32; d_kv];
    cpu_mv(&wq, &hnorm, &mut q_ref, d_q, D_MODEL);
    cpu_mv(&wk, &hnorm, &mut k_ref, d_kv, D_MODEL);
    cpu_mv(&wv, &hnorm, &mut v_ref, d_kv, D_MODEL);
    rope(&mut q_ref, N_HEADS);
    rope(&mut k_ref, N_KV);
    // Build the full f32 KV cache the reference flash reads: the f16-rounded
    // history for [0,POS) + the f16-rounded new (post-RoPE K / plain V) at POS —
    // matching exactly what the device mirror holds.
    let mut kcache = vec![0f32; hist_elems];
    let mut vcache = vec![0f32; hist_elems];
    for h in 0..N_KV {
        for t in 0..POS {
            for d in 0..HEAD_DIM {
                let idx = (h * MAX_CTX + t) * HEAD_DIM + d;
                kcache[idx] = half::f16::from_bits(k_hist_f16[idx]).to_f32();
                vcache[idx] = half::f16::from_bits(v_hist_f16[idx]).to_f32();
            }
        }
        for d in 0..HEAD_DIM {
            let idx = (h * MAX_CTX + POS) * HEAD_DIM + d;
            kcache[idx] = half::f16::from_f32(k_ref[h * HEAD_DIM + d]).to_f32();
            vcache[idx] = half::f16::from_f32(v_ref[h * HEAD_DIM + d]).to_f32();
        }
    }
    // ---- Device: upload weights + allocate the F32 KV mirror ONCE ----
    // Weights / norms / RoPE'd Q/K/V (q_ref…vcache above) don't depend on the
    // SWA window or Gemma2 softcap, so they're computed once; the per-case loop
    // below only recomputes the attention + downstream reference, re-seeds the
    // mirror history, and resets `hidden_dev` for an independent run.
    let alloc = (|| -> sk::Result<_> {
        let mut wqb = sk::SyclSharedBuffer::<u8>::alloc(stream, wq.len())?;
        wqb.as_mut_slice().copy_from_slice(&wq);
        let mut wkb = sk::SyclSharedBuffer::<u8>::alloc(stream, wk.len())?;
        wkb.as_mut_slice().copy_from_slice(&wk);
        let mut wvb = sk::SyclSharedBuffer::<u8>::alloc(stream, wv.len())?;
        wvb.as_mut_slice().copy_from_slice(&wv);
        let mut wob = sk::SyclSharedBuffer::<u8>::alloc(stream, wo.len())?;
        wob.as_mut_slice().copy_from_slice(&wo);
        let mut wgb = sk::SyclSharedBuffer::<u8>::alloc(stream, wgate.len())?;
        wgb.as_mut_slice().copy_from_slice(&wgate);
        let mut wub = sk::SyclSharedBuffer::<u8>::alloc(stream, wup.len())?;
        wub.as_mut_slice().copy_from_slice(&wup);
        let mut wdb = sk::SyclSharedBuffer::<u8>::alloc(stream, wdown.len())?;
        wdb.as_mut_slice().copy_from_slice(&wdown);
        let kmir = sk::SyclSharedBuffer::<u16>::alloc(stream, hist_elems)?;
        let vmir = sk::SyclSharedBuffer::<u16>::alloc(stream, hist_elems)?;
        Ok((wqb, wkb, wvb, wob, wgb, wub, wdb, kmir, vmir))
    })();
    let (wqb, wkb, wvb, wob, wgb, wub, wdb, mut kmir, mut vmir) = match alloc {
        Ok(t) => t,
        Err(_) => {
            emit(name, "KERNEL_ERR", "usm-alloc-failed");
            return;
        }
    };

    // Three cases drive the ONE `attn:dev_resident_block_sycl` verdict:
    //   (0, 0.0)          passthrough — MUST stay bit-identical to the
    //                     pre-feature block (window=0/softcap=0 fold away).
    //   (DEV_WINDOW, 0.0) sliding-window — kv_len is only 6 here, so a small
    //                     window=3 genuinely trims (unlike the flash probes'
    //                     window=16; the block's seeded history is tiny).
    //   (0, 50.0)         Gemma2 attention-logit softcap (the real value).
    // The CPU reference applies the SAME (window, softcap) via the fused
    // `gqa_attention_one_step_capped_windowed`, so a block whose internal
    // flash-decode ignores either uniform fails the combined verdict
    // (fail-closed). A latent-bug guard: before this feature an SWA model reaching
    // the device block attended the FULL history (no window arg) — the window
    // case here would MISCOMPUTE on that pre-fix path.
    const DEV_WINDOW: usize = 3;
    let cases: [(u32, f32); 3] = [(0, 0.0), (DEV_WINDOW as u32, 0.0), (0, 50.0)];
    let mut ok_all = true;
    let mut worst_cos = 1.0f64;
    let mut worst_rel = 0.0f64;
    let mut any_decline = false;
    let mut ffn_ran_all = true;
    let mut kernel_err: Option<String> = None;
    for (window, softcap) in cases {
        // CPU reference attention + o-proj + residual + post-norm + dense FFN for
        // THIS (window, softcap). Only the attention step depends on the uniforms.
        let mut attn_ref = vec![0f32; d_q];
        k::gqa_attention_one_step_capped_windowed(
            &q_ref, &kcache, &vcache, &mut attn_ref,
            N_HEADS, N_KV, HEAD_DIM, MAX_CTX, kv_len, window as usize, softcap,
        );
        let mut o_ref = vec![0f32; D_MODEL];
        cpu_mv(&wo, &attn_ref, &mut o_ref, D_MODEL, d_q);
        let mut hidden_ref = hidden0.clone();
        for i in 0..D_MODEL {
            hidden_ref[i] += o_ref[i];
        }
        let mut hnorm2_ref = vec![0f32; D_MODEL];
        k::rmsnorm_f32_row(&hidden_ref, &ffn_norm, &mut hnorm2_ref, EPS);
        // Dense SwiGLU FFN reference: gate/up = W·hnorm2, act = silu(gate)⊙up,
        // ffn_out = Wdown·act, hidden += ffn_out — the same op sequence the device
        // FFN leg runs. `hidden_ref` becomes the FINAL post-FFN residual we grade.
        let mut gate_ref = vec![0f32; D_FF];
        let mut up_ref = vec![0f32; D_FF];
        cpu_mv(&wgate, &hnorm2_ref, &mut gate_ref, D_FF, D_MODEL);
        cpu_mv(&wup, &hnorm2_ref, &mut up_ref, D_FF, D_MODEL);
        let mut act_ref = vec![0f32; D_FF];
        k::silu_mul_f32(&gate_ref, &up_ref, &mut act_ref);
        let mut ffn_out_ref = vec![0f32; D_MODEL];
        cpu_mv(&wdown, &act_ref, &mut ffn_out_ref, D_MODEL, D_FF);
        for i in 0..D_MODEL {
            hidden_ref[i] += ffn_out_ref[i];
        }
        if !hidden_ref.iter().all(|v| v.is_finite()) {
            emit(name, "SKIP", "cpu-reference-nonfinite");
            return;
        }
        // Re-seed the mirror history [0,POS) + reset hidden for an independent
        // run (the block overwrites only row POS, but reset for clarity/safety).
        kmir.as_mut_slice().copy_from_slice(&k_hist_f16);
        vmir.as_mut_slice().copy_from_slice(&v_hist_f16);
        let mut hidden_dev = hidden0.clone();
        let mut hnorm2_dev = vec![0f32; D_MODEL];
        let mut k_row_dev = vec![0f32; d_kv];
        let mut v_row_dev = vec![0f32; d_kv];
        // SAFETY: all USM buffers (weights, FFN weights, mirrors) live on `stream`;
        // the norm / hidden / inv_freq / out slices are host-sized to the dims; the
        // block syncs before returning. kq/kk/kv/ko + FFN codes = 0 (Q4_K). Passing
        // the dense-FFN tuple exercises the FFN-inclusive path. `window`/`softcap`
        // are this case's SWA / Gemma2 uniforms.
        let res = unsafe {
            sk::decode_attn_block_usm_raw(
                stream,
                wqb.as_ptr() as *const c_void, 0,
                wkb.as_ptr() as *const c_void, 0,
                wvb.as_ptr() as *const c_void, 0,
                wob.as_ptr() as *const c_void, 0,
                Some((
                    wgb.as_ptr() as *const c_void, 0,
                    wub.as_ptr() as *const c_void, 0,
                    wdb.as_ptr() as *const c_void, 0,
                    D_FF as u32,
                )),
                attn_norm.as_ptr(),
                ffn_norm.as_ptr(),
                inv_freq.as_ptr(),
                hidden_dev.as_mut_ptr(),
                hnorm2_dev.as_mut_ptr(),
                k_row_dev.as_mut_ptr(),
                v_row_dev.as_mut_ptr(),
                kmir.as_mut_ptr(),
                vmir.as_mut_ptr(),
                N_HEADS as u32,
                N_KV as u32,
                HEAD_DIM as u32,
                D_MODEL as u32,
                POS as u32,
                MAX_CTX as u32,
                kv_len as u32,
                EPS,
                window,
                softcap,
            )
        };
        match res {
            // `ffn_ran` must be true — the probe always supplies dispatchable Q4_K
            // FFN weights, so the device FFN leg ran. Grade the per-layer residual
            // DELTA (`hidden - hidden0`) so the large shared input residual can't
            // mask a miscompute (cosine of two near-equal big vectors is ~1).
            Ok((true, ffn_ran)) => {
                let gpu_delta: Vec<f32> =
                    hidden_dev.iter().zip(&hidden0).map(|(h, h0)| h - h0).collect();
                let cpu_delta: Vec<f32> =
                    hidden_ref.iter().zip(&hidden0).map(|(h, h0)| h - h0).collect();
                let (cos, max_rel) = compare(&gpu_delta, &cpu_delta);
                if !(cos > 0.995 && max_rel < 0.10 && ffn_ran) {
                    ok_all = false;
                }
                if !ffn_ran {
                    ffn_ran_all = false;
                }
                if cos < worst_cos {
                    worst_cos = cos;
                }
                if max_rel > worst_rel {
                    worst_rel = max_rel;
                }
            }
            Ok((false, _)) => {
                any_decline = true;
            }
            Err(e) => {
                kernel_err = Some(format!("{e}"));
                break;
            }
        }
    }
    let ms = started.elapsed().as_millis();
    if let Some(e) = kernel_err {
        emit(name, "KERNEL_ERR", &format!("{e} ms={ms}"));
    } else if any_decline {
        emit(name, "SKIP", &format!("block-declined ms={ms}"));
    } else {
        let verdict = if ok_all { "OK" } else { "MISCOMPUTE" };
        emit(
            name,
            verdict,
            &format!(
                "cos={worst_cos:.6} max_rel={worst_rel:.4} ffn_ran={ffn_ran_all} \
                 (pass+swa{DEV_WINDOW}+cap50) ms={ms}"
            ),
        );
    }
}

fn probe_attn(stream: &sk::SyclStream, name: &str, variant: &str, started: Instant) {
    // Quant-KV variants (decode_mxfp4 … prefill_nvfp4) run the F32-Q /
    // packed-K/V flash kernels and grade against the full-precision CPU
    // reference; the f32 variants (v1/v2/v3) below stay on the f16 path.
    // Split here so each keeps its own buffer setup (f16 vs f32 + packed).
    // Q8_0 KV first: its per-row-scale layout isn't a `QuantKvFormat`, so it has
    // its own probe (routed before the block-format check, which would miss it).
    if variant == "dev_resident_block_sycl" {
        probe_decode_block(stream, name, started);
        return;
    }
    if variant == "decode_q8_0" {
        probe_attn_q8_0(stream, name, "decode", started);
        return;
    }
    if variant == "prefill_q8_0" {
        probe_attn_q8_0(stream, name, "prefill", started);
        return;
    }
    if variant == "decode_tq" {
        probe_attn_tq(stream, name, "decode", started);
        return;
    }
    if variant == "prefill_tq" {
        probe_attn_tq(stream, name, "prefill", started);
        return;
    }
    if let Some(fmt) = variant.strip_prefix("decode_").and_then(quant_kv_format) {
        probe_attn_quant_kv(stream, name, "decode", fmt, started);
        return;
    }
    if let Some(fmt) = variant.strip_prefix("prefill_").and_then(quant_kv_format) {
        probe_attn_quant_kv(stream, name, "prefill", fmt, started);
        return;
    }
    let kv_elems = AT_KV_HEADS * AT_MAX_CTX * AT_HEAD_DIM;
    let mut rng = XorShift(0xA77E17);
    let fill_kv = |rng: &mut XorShift, upto: usize| -> Vec<f32> {
        let mut v = vec![0f32; kv_elems];
        for h in 0..AT_KV_HEADS {
            for t in 0..upto {
                for d in 0..AT_HEAD_DIM {
                    v[(h * AT_MAX_CTX + t) * AT_HEAD_DIM + d] = rng.f32_pm(0.6);
                }
            }
        }
        v
    };
    if let Some(v) = variant.strip_prefix("decode_") {
        let kcache = fill_kv(&mut rng, AT_KV_LEN);
        let vcache = fill_kv(&mut rng, AT_KV_LEN);
        let q: Vec<f32> = (0..AT_HEADS * AT_HEAD_DIM)
            .map(|_| rng.f32_pm(0.6))
            .collect();
        let alloc = (|| -> sk::Result<_> {
            let qb = to_f16_buf(stream, &q)?;
            let kb = to_f16_buf(stream, &kcache)?;
            let vb = to_f16_buf(stream, &vcache)?;
            let ob = sk::SyclSharedBuffer::<u16>::alloc(stream, AT_HEADS * AT_HEAD_DIM)?;
            Ok((qb, kb, vb, ob))
        })();
        let (qb, kb, vb, mut ob) = match alloc {
            Ok(t) => t,
            Err(_) => {
                emit(name, "KERNEL_ERR", "usm-alloc-failed");
                return;
            }
        };
        // Three cases drive the ONE `attn:decode_{v1,v2,v3}` verdict:
        //   (0, 0.0)   passthrough — must stay bit-identical to the pre-feature
        //              kernel (window=0/softcap=0 fold away).
        //   (16, 0.0)  sliding-window — AT_KV_LEN=333 so window=16 genuinely
        //              trims to the last 16 cached positions.
        //   (0, 50.0)  Gemma2 attention-logit softcap (the real value).
        // The CPU reference applies the SAME (window, softcap) via the fused
        // `gqa_attention_one_step_capped_windowed` ((0,0.0) delegates to the full
        // kernel bit-identically), so a kernel that ignores either uniform fails
        // the combined verdict (fail-closed).
        let cases: [(u32, f32); 3] = [(0, 0.0), (16, 0.0), (0, 50.0)];
        let mut ok_all = true;
        let mut worst_cos = 1.0f64;
        let mut worst_rel = 0.0f64;
        let mut kernel_err: Option<String> = None;
        for (window, softcap) in cases {
            let mut cpu_out = vec![0f32; AT_HEADS * AT_HEAD_DIM];
            k::gqa_attention_one_step_capped_windowed(
                &q, &kcache, &vcache, &mut cpu_out,
                AT_HEADS, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN,
                window as usize, softcap,
            );
            for b in ob.as_mut_slice().iter_mut() {
                *b = 0;
            }
            let res = match v {
                "v1" => sk::flash_attn_decode_usm(
                    stream, &qb, &kb, &vb, &mut ob,
                    AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32, AT_KV_LEN as u32, window, softcap,
                ),
                "v2" => sk::flash_attn_decode_v2_usm(
                    stream, &qb, &kb, &vb, &mut ob,
                    AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32, AT_KV_LEN as u32, window, softcap,
                ),
                "v3" => sk::flash_attn_decode_v3_usm(
                    stream, &qb, &kb, &vb, &mut ob,
                    AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32, AT_KV_LEN as u32, window, softcap,
                    0, // P19: tile_size 0 ⇒ the kernel's built-in heuristic
                ),
                _ => {
                    emit(name, "SKIP", "unknown-variant");
                    return;
                }
            };
            if let Err(e) = res {
                kernel_err = Some(format!("{e}"));
                break;
            }
            let gpu_out: Vec<f32> = ob
                .as_slice()
                .iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect();
            // f16 I/O + online softmax reordering: looser gate than f32 matvec.
            let (cos, max_rel) = compare(&gpu_out, &cpu_out);
            if !(cos > 0.995 && max_rel < 0.10) {
                ok_all = false;
            }
            if cos < worst_cos {
                worst_cos = cos;
            }
            if max_rel > worst_rel {
                worst_rel = max_rel;
            }
        }
        let ms = started.elapsed().as_millis();
        match kernel_err {
            Some(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
            None => {
                let verdict = if ok_all { "OK" } else { "MISCOMPUTE" };
                emit(
                    name,
                    verdict,
                    &format!(
                        "cos={worst_cos:.6} max_rel={worst_rel:.4} \
                         (pass+swa16+cap50) ms={ms}"
                    ),
                );
            }
        }
        return;
    }

    if let Some(v) = variant.strip_prefix("prefill_") {
        let upto = AT_PREFILL_BASE + AT_PREFILL_NEW;
        let kcache = fill_kv(&mut rng, upto);
        let vcache = fill_kv(&mut rng, upto);
        let q: Vec<f32> = (0..AT_PREFILL_NEW * AT_HEADS * AT_HEAD_DIM)
            .map(|_| rng.f32_pm(0.6))
            .collect();
        let alloc = (|| -> sk::Result<_> {
            let mut qb = sk::SyclSharedBuffer::<f32>::alloc(stream, q.len())?;
            qb.as_mut_slice().copy_from_slice(&q);
            let mut kb = sk::SyclSharedBuffer::<f32>::alloc(stream, kcache.len())?;
            kb.as_mut_slice().copy_from_slice(&kcache);
            let mut vb = sk::SyclSharedBuffer::<f32>::alloc(stream, vcache.len())?;
            vb.as_mut_slice().copy_from_slice(&vcache);
            let ob = sk::SyclSharedBuffer::<f32>::alloc(stream, q.len())?;
            Ok((qb, kb, vb, ob))
        })();
        let (qb, kb, vb, mut ob) = match alloc {
            Ok(t) => t,
            Err(_) => {
                emit(name, "KERNEL_ERR", "usm-alloc-failed");
                return;
            }
        };
        // Three cases drive the ONE `attn:prefill_{v1,v2,v3}` verdict:
        //   (0, 0.0)   passthrough — must stay bit-identical to the pre-feature
        //              kernel (window=0/softcap=0 fold away).
        //   (16, 0.0)  sliding-window — queries at abs pos 64..95 each attend
        //              their last 16 positions (window=16 genuinely trims).
        //   (0, 50.0)  Gemma2 attention-logit softcap (the real value).
        // The CPU reference applies the SAME (window, softcap) via the fused
        // `gqa_attention_flash_prefill_capped_windowed`, so a kernel that ignores
        // either uniform fails the combined verdict (fail-closed).
        let cases: [(u32, f32); 3] = [(0, 0.0), (16, 0.0), (0, 50.0)];
        let mut ok_all = true;
        let mut worst_cos = 1.0f64;
        let mut worst_rel = 0.0f64;
        let mut kernel_err: Option<String> = None;
        for (window, softcap) in cases {
            let mut cpu_out = vec![0f32; q.len()];
            k::gqa_attention_flash_prefill_capped_windowed(
                &q, &kcache, &vcache, &mut cpu_out,
                AT_HEADS, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX,
                AT_PREFILL_BASE, AT_PREFILL_NEW, window as usize, softcap,
            );
            ob.as_mut_slice().fill(f32::NAN);
            // SAFETY: USM pointers live on `stream`, lengths match the shape
            // arguments; kernels wait before returning. window/softcap = this case.
            let res = unsafe {
                match v {
                    "v1" => sk::flash_attn_prefill_usm_raw(
                        stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                        AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                        AT_MAX_CTX as u32, AT_PREFILL_BASE as u32, AT_PREFILL_NEW as u32,
                        window, softcap,
                    ),
                    "v2" => sk::flash_attn_prefill_v2_usm_raw(
                        stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                        AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                        AT_MAX_CTX as u32, AT_PREFILL_BASE as u32, AT_PREFILL_NEW as u32,
                        window, softcap,
                    ),
                    "v3" => sk::flash_attn_prefill_v3_usm_raw(
                        stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                        AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                        AT_MAX_CTX as u32, AT_PREFILL_BASE as u32, AT_PREFILL_NEW as u32,
                        window, softcap,
                    ),
                    _ => {
                        emit(name, "SKIP", "unknown-variant");
                        return;
                    }
                }
            };
            if let Err(e) = res {
                kernel_err = Some(format!("{e}"));
                break;
            }
            let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
            if !(cos > 0.999 && max_rel < 0.05) {
                ok_all = false;
            }
            if cos < worst_cos {
                worst_cos = cos;
            }
            if max_rel > worst_rel {
                worst_rel = max_rel;
            }
        }
        let ms = started.elapsed().as_millis();
        match kernel_err {
            Some(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
            None => {
                let verdict = if ok_all { "OK" } else { "MISCOMPUTE" };
                emit(
                    name,
                    verdict,
                    &format!(
                        "cos={worst_cos:.6} max_rel={worst_rel:.4} \
                         (pass+swa16+cap50) ms={ms}"
                    ),
                );
            }
        }
        return;
    }

    emit(name, "SKIP", "unknown-variant");
}

/// Quant-KV FlashAttention parity (SYCL). Mirrors `probe_attn`'s f32 body
/// but for the packed-KV kernels: generate the SAME f32 Q/K/V the f32
/// probe uses, quantize K/V into `fmt`'s block layout, run the GPU
/// quant-KV flash decode/prefill (F32 Q/out, packed U8 K/V), and grade
/// against the FULL-PRECISION CPU reference (`k::gqa_attention_one_step`
/// / `..._flash_prefill`). The per-`fmt` tolerance absorbs the KV
/// round-trip error, since only the GPU side sees the quantized cache.
fn probe_attn_quant_kv(
    stream: &sk::SyclStream,
    name: &str,
    dir: &str,
    fmt: &QuantKvFormat,
    started: Instant,
) {
    // The flash kernels require head_dim to be a whole number of quant
    // blocks. AT_HEAD_DIM=256 satisfies both 32 (MXFP*) and 16 (NVFP4);
    // guard anyway so a future geometry change fails loud, not silent.
    if AT_HEAD_DIM % fmt.block_elems != 0 {
        emit(name, "SKIP", "head-dim-not-block-aligned");
        return;
    }
    let kv_elems = AT_KV_HEADS * AT_MAX_CTX * AT_HEAD_DIM;
    // Identical RNG seed + fill order to `probe_attn` so the quant probe
    // sees byte-for-byte the same f32 Q/K/V as the f32 attention probe.
    let mut rng = XorShift(0xA77E17);
    let fill_kv = |rng: &mut XorShift, upto: usize| -> Vec<f32> {
        let mut v = vec![0f32; kv_elems];
        for h in 0..AT_KV_HEADS {
            for t in 0..upto {
                for d in 0..AT_HEAD_DIM {
                    v[(h * AT_MAX_CTX + t) * AT_HEAD_DIM + d] = rng.f32_pm(0.6);
                }
            }
        }
        v
    };

    if dir == "decode" {
        let kcache = fill_kv(&mut rng, AT_KV_LEN);
        let vcache = fill_kv(&mut rng, AT_KV_LEN);
        let q: Vec<f32> = (0..AT_HEADS * AT_HEAD_DIM)
            .map(|_| rng.f32_pm(0.6))
            .collect();
        // Quantize K/V into the packed block layout the kernel decodes.
        let kp = quantize_kv_cache(fmt, &kcache, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN);
        let vp = quantize_kv_cache(fmt, &vcache, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN);
        // SAME-QUANT reference (Phase 2): dequantize the IDENTICAL packed K/V and
        // run the f32 attention on THAT, so max_rel isolates the kernel's
        // FMA/expf reassociation from the KV QUANTIZATION LOSS. The old
        // full-precision f32-KV reference measured the quant loss instead, which
        // flagged every SYCL quant-KV format as a false MISCOMPUTE (cos > 0.99,
        // large max_rel). Matches the CUDA probe's reference.
        let kc_dq = dequant_kv_cache(fmt, &kp, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN);
        let vc_dq = dequant_kv_cache(fmt, &vp, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN);
        let alloc = (|| -> sk::Result<_> {
            let mut qb = sk::SyclSharedBuffer::<f32>::alloc(stream, q.len())?;
            qb.as_mut_slice().copy_from_slice(&q);
            let mut kb = sk::SyclSharedBuffer::<u8>::alloc(stream, kp.len())?;
            kb.as_mut_slice().copy_from_slice(&kp);
            let mut vb = sk::SyclSharedBuffer::<u8>::alloc(stream, vp.len())?;
            vb.as_mut_slice().copy_from_slice(&vp);
            let ob = sk::SyclSharedBuffer::<f32>::alloc(stream, AT_HEADS * AT_HEAD_DIM)?;
            Ok((qb, kb, vb, ob))
        })();
        let (qb, kb, vb, mut ob) = match alloc {
            Ok(t) => t,
            Err(_) => {
                emit(name, "KERNEL_ERR", "usm-alloc-failed");
                return;
            }
        };
        // F1: fold three cases into the ONE attn:decode_<fmt> verdict (same
        // pattern as the f32 probe): passthrough (0,0.0) must stay bit-identical,
        // a sliding-window case (16,0.0) — AT_KV_LEN trims to the last 16 — and
        // the Gemma2 softcap (0,50.0). The CPU reference runs the SAME (window,
        // softcap) over the IDENTICAL dequantized K/V, so a cap/window-blind
        // kernel fails the combined verdict (fail-closed).
        let cases: [(u32, f32); 3] = [(0, 0.0), (16, 0.0), (0, 50.0)];
        let mut ok_all = true;
        let mut worst_cos = 1.0f64;
        let mut worst_rel = 0.0f64;
        let mut kernel_err: Option<String> = None;
        for (window, softcap) in cases {
            let mut cpu_out = vec![0f32; AT_HEADS * AT_HEAD_DIM];
            k::gqa_attention_one_step_capped_windowed(
                &q, &kc_dq, &vc_dq, &mut cpu_out,
                AT_HEADS, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN,
                window as usize, softcap,
            );
            ob.as_mut_slice().fill(f32::NAN);
            // SAFETY: all four are live USM allocations on `stream`; q/out are
            // F32 [n_heads*head_dim], k/v packed U8 [n_kv_heads*max_ctx*
            // bytes_per_row] (built by quantize_kv_cache); the kernels wait.
            let res = unsafe {
                match fmt.name {
                    "mxfp4" => sk::flash_attn_decode_mxfp4_usm_raw(
                        stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                        AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                        AT_MAX_CTX as u32, AT_KV_LEN as u32, window, softcap,
                    ),
                    "mxfp6" => sk::flash_attn_decode_mxfp6_usm_raw(
                        stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                        AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                        AT_MAX_CTX as u32, AT_KV_LEN as u32, window, softcap,
                    ),
                    "mxfp8" => sk::flash_attn_decode_mxfp8_usm_raw(
                        stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                        AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                        AT_MAX_CTX as u32, AT_KV_LEN as u32, window, softcap,
                    ),
                    "nvfp4" => sk::flash_attn_decode_nvfp4_usm_raw(
                        stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                        AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                        AT_MAX_CTX as u32, AT_KV_LEN as u32, window, softcap,
                    ),
                    "q4_0" => sk::flash_attn_decode_q4_0_usm_raw(
                        stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                        AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                        AT_MAX_CTX as u32, AT_KV_LEN as u32, window, softcap,
                    ),
                    _ => {
                        emit(name, "SKIP", "unknown-format");
                        return;
                    }
                }
            };
            if let Err(e) = res {
                kernel_err = Some(format!("{e}"));
                break;
            }
            let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
            if !(cos > fmt.cos_min && max_rel < fmt.rel_max) {
                ok_all = false;
            }
            if cos < worst_cos {
                worst_cos = cos;
            }
            if max_rel > worst_rel {
                worst_rel = max_rel;
            }
        }
        let ms = started.elapsed().as_millis();
        match kernel_err {
            Some(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
            None => {
                let verdict = if ok_all { "OK" } else { "MISCOMPUTE" };
                emit(
                    name,
                    verdict,
                    &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa16+cap50) ms={ms}"),
                );
            }
        }
        return;
    }

    if dir == "prefill" {
        let upto = AT_PREFILL_BASE + AT_PREFILL_NEW;
        let kcache = fill_kv(&mut rng, upto);
        let vcache = fill_kv(&mut rng, upto);
        let q: Vec<f32> = (0..AT_PREFILL_NEW * AT_HEADS * AT_HEAD_DIM)
            .map(|_| rng.f32_pm(0.6))
            .collect();
        let kp = quantize_kv_cache(fmt, &kcache, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, upto);
        let vp = quantize_kv_cache(fmt, &vcache, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, upto);
        // SAME-QUANT reference (Phase 2) — see the decode arm above: grade the
        // kernel against a dequantized round-trip of the identical packed K/V so
        // max_rel reflects kernel reassociation, not quantization loss.
        let kc_dq = dequant_kv_cache(fmt, &kp, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, upto);
        let vc_dq = dequant_kv_cache(fmt, &vp, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, upto);
        let alloc = (|| -> sk::Result<_> {
            let mut qb = sk::SyclSharedBuffer::<f32>::alloc(stream, q.len())?;
            qb.as_mut_slice().copy_from_slice(&q);
            let mut kb = sk::SyclSharedBuffer::<u8>::alloc(stream, kp.len())?;
            kb.as_mut_slice().copy_from_slice(&kp);
            let mut vb = sk::SyclSharedBuffer::<u8>::alloc(stream, vp.len())?;
            vb.as_mut_slice().copy_from_slice(&vp);
            let ob = sk::SyclSharedBuffer::<f32>::alloc(stream, q.len())?;
            Ok((qb, kb, vb, ob))
        })();
        let (qb, kb, vb, mut ob) = match alloc {
            Ok(t) => t,
            Err(_) => {
                emit(name, "KERNEL_ERR", "usm-alloc-failed");
                return;
            }
        };
        // F1: three cases into the ONE attn:prefill_<fmt> verdict (see the decode
        // arm): passthrough (0,0.0), a sliding window (16,0.0), the Gemma2 softcap
        // (0,50.0); CPU `..._flash_prefill_capped_windowed` applies the SAME over
        // the identical dequantized K/V.
        let cases: [(u32, f32); 3] = [(0, 0.0), (16, 0.0), (0, 50.0)];
        let mut ok_all = true;
        let mut worst_cos = 1.0f64;
        let mut worst_rel = 0.0f64;
        let mut kernel_err: Option<String> = None;
        for (window, softcap) in cases {
            let mut cpu_out = vec![0f32; q.len()];
            k::gqa_attention_flash_prefill_capped_windowed(
                &q, &kc_dq, &vc_dq, &mut cpu_out,
                AT_HEADS, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX,
                AT_PREFILL_BASE, AT_PREFILL_NEW, window as usize, softcap,
            );
            ob.as_mut_slice().fill(f32::NAN);
            // SAFETY: as in the decode arm; q/out are F32 [n_new*n_heads*
            // head_dim], k/v packed U8; kernels wait before returning.
            let res = unsafe {
                match fmt.name {
                    "mxfp4" => sk::flash_attn_prefill_mxfp4_usm_raw(
                        stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                        AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                        AT_MAX_CTX as u32, AT_PREFILL_BASE as u32, AT_PREFILL_NEW as u32,
                        window, softcap,
                    ),
                    "mxfp6" => sk::flash_attn_prefill_mxfp6_usm_raw(
                        stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                        AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                        AT_MAX_CTX as u32, AT_PREFILL_BASE as u32, AT_PREFILL_NEW as u32,
                        window, softcap,
                    ),
                    "mxfp8" => sk::flash_attn_prefill_mxfp8_usm_raw(
                        stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                        AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                        AT_MAX_CTX as u32, AT_PREFILL_BASE as u32, AT_PREFILL_NEW as u32,
                        window, softcap,
                    ),
                    "nvfp4" => sk::flash_attn_prefill_nvfp4_usm_raw(
                        stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                        AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                        AT_MAX_CTX as u32, AT_PREFILL_BASE as u32, AT_PREFILL_NEW as u32,
                        window, softcap,
                    ),
                    "q4_0" => sk::flash_attn_prefill_q4_0_usm_raw(
                        stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                        AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                        AT_MAX_CTX as u32, AT_PREFILL_BASE as u32, AT_PREFILL_NEW as u32,
                        window, softcap,
                    ),
                    _ => {
                        emit(name, "SKIP", "unknown-format");
                        return;
                    }
                }
            };
            if let Err(e) = res {
                kernel_err = Some(format!("{e}"));
                break;
            }
            let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
            if !(cos > fmt.cos_min && max_rel < fmt.rel_max) {
                ok_all = false;
            }
            if cos < worst_cos {
                worst_cos = cos;
            }
            if max_rel > worst_rel {
                worst_rel = max_rel;
            }
        }
        let ms = started.elapsed().as_millis();
        match kernel_err {
            Some(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
            None => {
                let verdict = if ok_all { "OK" } else { "MISCOMPUTE" };
                emit(
                    name,
                    verdict,
                    &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa16+cap50) ms={ms}"),
                );
            }
        }
        return;
    }

    emit(name, "SKIP", "unknown-direction");
}

/// Q8_0 KV FlashAttention probe (per-row-scale layout). Unlike
/// `probe_attn_quant_kv` (block-embedded scales), Q8_0 KV is a plain i8 slab +
/// a SEPARATE per-row f32 scale buffer, so the kernel takes `k_scales`/
/// `v_scales` args. Graded against the SAME-QUANT reference (dequantize the
/// identical slab+scales, run the f32 flash reference) so the metric isolates
/// kernel reassociation from KV quant loss. Iris-Xe-validatable (not write-blind).
fn probe_attn_q8_0(stream: &sk::SyclStream, name: &str, dir: &str, started: Instant) {
    let mut rng = XorShift(0xA77E17);
    let fill_kv = |rng: &mut XorShift, upto: usize| -> Vec<f32> {
        let mut v = vec![0f32; AT_KV_HEADS * AT_MAX_CTX * AT_HEAD_DIM];
        for h in 0..AT_KV_HEADS {
            for t in 0..upto {
                for d in 0..AT_HEAD_DIM {
                    v[(h * AT_MAX_CTX + t) * AT_HEAD_DIM + d] = rng.f32_pm(0.6);
                }
            }
        }
        v
    };
    // Build the 6 USM buffers (q F32, k/v i8 slabs, k/v f32 scales, out F32).
    let run = |q: &[f32],
               kslab: &[i8],
               ksc: &[f32],
               vslab: &[i8],
               vsc: &[f32],
               out_len: usize|
     -> sk::Result<(
        sk::SyclSharedBuffer<f32>,
        sk::SyclSharedBuffer<u8>,
        sk::SyclSharedBuffer<u8>,
        sk::SyclSharedBuffer<f32>,
        sk::SyclSharedBuffer<f32>,
        sk::SyclSharedBuffer<f32>,
    )> {
        let mut qb = sk::SyclSharedBuffer::<f32>::alloc(stream, q.len())?;
        qb.as_mut_slice().copy_from_slice(q);
        let mut kb = sk::SyclSharedBuffer::<u8>::alloc(stream, kslab.len())?;
        kb.as_mut_slice().copy_from_slice(i8_as_bytes(kslab));
        let mut vb = sk::SyclSharedBuffer::<u8>::alloc(stream, vslab.len())?;
        vb.as_mut_slice().copy_from_slice(i8_as_bytes(vslab));
        let mut ksb = sk::SyclSharedBuffer::<f32>::alloc(stream, ksc.len())?;
        ksb.as_mut_slice().copy_from_slice(ksc);
        let mut vsb = sk::SyclSharedBuffer::<f32>::alloc(stream, vsc.len())?;
        vsb.as_mut_slice().copy_from_slice(vsc);
        let mut ob = sk::SyclSharedBuffer::<f32>::alloc(stream, out_len)?;
        ob.as_mut_slice().fill(f32::NAN);
        Ok((qb, kb, vb, ksb, vsb, ob))
    };

    if dir == "decode" {
        let kcache = fill_kv(&mut rng, AT_KV_LEN);
        let vcache = fill_kv(&mut rng, AT_KV_LEN);
        let q: Vec<f32> = (0..AT_HEADS * AT_HEAD_DIM).map(|_| rng.f32_pm(0.6)).collect();
        let (kslab, ksc) = quantize_kv_q8_0(&kcache, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN);
        let (vslab, vsc) = quantize_kv_q8_0(&vcache, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN);
        let kdq = dequant_kv_q8_0(&kslab, &ksc, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN);
        let vdq = dequant_kv_q8_0(&vslab, &vsc, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN);
        let (qb, kb, vb, ksb, vsb, mut ob) =
            match run(&q, &kslab, &ksc, &vslab, &vsc, AT_HEADS * AT_HEAD_DIM) {
                Ok(t) => t,
                Err(_) => {
                    emit(name, "KERNEL_ERR", "usm-alloc-failed");
                    return;
                }
            };
        // F1: passthrough (0,0.0) + sliding-window (16,0.0) + Gemma2 softcap
        // (0,50.0) fold into the ONE attn:decode_q8_0 verdict; CPU reference
        // caps/windows the identical dequantized K/V.
        let cases: [(u32, f32); 3] = [(0, 0.0), (16, 0.0), (0, 50.0)];
        let mut ok_all = true;
        let mut worst_cos = 1.0f64;
        let mut worst_rel = 0.0f64;
        let mut kernel_err: Option<String> = None;
        for (window, softcap) in cases {
            let mut cpu_out = vec![0f32; AT_HEADS * AT_HEAD_DIM];
            k::gqa_attention_one_step_capped_windowed(
                &q, &kdq, &vdq, &mut cpu_out,
                AT_HEADS, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN,
                window as usize, softcap,
            );
            ob.as_mut_slice().fill(f32::NAN);
            // SAFETY: all USM buffers live on `stream`, sized to the geometry;
            // the kernel waits before returning.
            let res = unsafe {
                sk::flash_attn_decode_q8_0_usm_raw(
                    stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ksb.as_ptr(), vsb.as_ptr(),
                    ob.as_mut_ptr(), AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32, AT_KV_LEN as u32, window, softcap,
                )
            };
            if let Err(e) = res {
                kernel_err = Some(format!("{e}"));
                break;
            }
            let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
            if !(cos > 0.999 && max_rel < 0.05) {
                ok_all = false;
            }
            if cos < worst_cos {
                worst_cos = cos;
            }
            if max_rel > worst_rel {
                worst_rel = max_rel;
            }
        }
        let ms = started.elapsed().as_millis();
        match kernel_err {
            Some(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
            None => {
                let verdict = if ok_all { "OK" } else { "MISCOMPUTE" };
                emit(name, verdict, &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa16+cap50) ms={ms}"));
            }
        }
    } else if dir == "prefill" {
        let upto = AT_PREFILL_BASE + AT_PREFILL_NEW;
        let kcache = fill_kv(&mut rng, upto);
        let vcache = fill_kv(&mut rng, upto);
        let q: Vec<f32> = (0..AT_PREFILL_NEW * AT_HEADS * AT_HEAD_DIM)
            .map(|_| rng.f32_pm(0.6))
            .collect();
        let (kslab, ksc) = quantize_kv_q8_0(&kcache, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, upto);
        let (vslab, vsc) = quantize_kv_q8_0(&vcache, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, upto);
        let kdq = dequant_kv_q8_0(&kslab, &ksc, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, upto);
        let vdq = dequant_kv_q8_0(&vslab, &vsc, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, upto);
        let (qb, kb, vb, ksb, vsb, mut ob) =
            match run(&q, &kslab, &ksc, &vslab, &vsc, q.len()) {
                Ok(t) => t,
                Err(_) => {
                    emit(name, "KERNEL_ERR", "usm-alloc-failed");
                    return;
                }
            };
        // F1: passthrough + sliding-window + Gemma2 softcap into the ONE verdict.
        let cases: [(u32, f32); 3] = [(0, 0.0), (16, 0.0), (0, 50.0)];
        let mut ok_all = true;
        let mut worst_cos = 1.0f64;
        let mut worst_rel = 0.0f64;
        let mut kernel_err: Option<String> = None;
        for (window, softcap) in cases {
            let mut cpu_out = vec![0f32; q.len()];
            k::gqa_attention_flash_prefill_capped_windowed(
                &q, &kdq, &vdq, &mut cpu_out,
                AT_HEADS, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_PREFILL_BASE, AT_PREFILL_NEW,
                window as usize, softcap,
            );
            ob.as_mut_slice().fill(f32::NAN);
            // SAFETY: as the decode arm, q/out sized for AT_PREFILL_NEW queries.
            let res = unsafe {
                sk::flash_attn_prefill_q8_0_usm_raw(
                    stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ksb.as_ptr(), vsb.as_ptr(),
                    ob.as_mut_ptr(), AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32, AT_PREFILL_BASE as u32, AT_PREFILL_NEW as u32,
                    window, softcap,
                )
            };
            if let Err(e) = res {
                kernel_err = Some(format!("{e}"));
                break;
            }
            let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
            if !(cos > 0.999 && max_rel < 0.05) {
                ok_all = false;
            }
            if cos < worst_cos {
                worst_cos = cos;
            }
            if max_rel > worst_rel {
                worst_rel = max_rel;
            }
        }
        let ms = started.elapsed().as_millis();
        match kernel_err {
            Some(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
            None => {
                let verdict = if ok_all { "OK" } else { "MISCOMPUTE" };
                emit(name, verdict, &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa16+cap50) ms={ms}"));
            }
        }
    } else {
        emit(name, "SKIP", "unknown-direction");
    }
}

/// TQ (TurboQuant) KV FlashAttention probe. Like `probe_attn_q8_0` but the slab
/// is WHT-packed `bits`-per-element level codes (`turboquant`) and the kernel
/// takes a runtime `bits` arg. Graded same-quant (identical slab+scales
/// dequantized via the inverse WHT) at a representative `TQ_PROBE_BITS`.
fn probe_attn_tq(stream: &sk::SyclStream, name: &str, dir: &str, started: Instant) {
    let bits = TQ_PROBE_BITS;
    let mut rng = XorShift(0xA77E17);
    let fill_kv = |rng: &mut XorShift, upto: usize| -> Vec<f32> {
        let mut v = vec![0f32; AT_KV_HEADS * AT_MAX_CTX * AT_HEAD_DIM];
        for h in 0..AT_KV_HEADS {
            for t in 0..upto {
                for d in 0..AT_HEAD_DIM {
                    v[(h * AT_MAX_CTX + t) * AT_HEAD_DIM + d] = rng.f32_pm(0.6);
                }
            }
        }
        v
    };
    let run = |q: &[f32],
               kslab: &[u8],
               ksc: &[f32],
               vslab: &[u8],
               vsc: &[f32],
               out_len: usize|
     -> sk::Result<(
        sk::SyclSharedBuffer<f32>,
        sk::SyclSharedBuffer<u8>,
        sk::SyclSharedBuffer<u8>,
        sk::SyclSharedBuffer<f32>,
        sk::SyclSharedBuffer<f32>,
        sk::SyclSharedBuffer<f32>,
    )> {
        let mut qb = sk::SyclSharedBuffer::<f32>::alloc(stream, q.len())?;
        qb.as_mut_slice().copy_from_slice(q);
        let mut kb = sk::SyclSharedBuffer::<u8>::alloc(stream, kslab.len())?;
        kb.as_mut_slice().copy_from_slice(kslab);
        let mut vb = sk::SyclSharedBuffer::<u8>::alloc(stream, vslab.len())?;
        vb.as_mut_slice().copy_from_slice(vslab);
        let mut ksb = sk::SyclSharedBuffer::<f32>::alloc(stream, ksc.len())?;
        ksb.as_mut_slice().copy_from_slice(ksc);
        let mut vsb = sk::SyclSharedBuffer::<f32>::alloc(stream, vsc.len())?;
        vsb.as_mut_slice().copy_from_slice(vsc);
        let mut ob = sk::SyclSharedBuffer::<f32>::alloc(stream, out_len)?;
        ob.as_mut_slice().fill(f32::NAN);
        Ok((qb, kb, vb, ksb, vsb, ob))
    };

    if dir == "decode" {
        let kcache = fill_kv(&mut rng, AT_KV_LEN);
        let vcache = fill_kv(&mut rng, AT_KV_LEN);
        let q: Vec<f32> = (0..AT_HEADS * AT_HEAD_DIM).map(|_| rng.f32_pm(0.6)).collect();
        let (kslab, ksc) = quantize_kv_tq(&kcache, bits, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN);
        let (vslab, vsc) = quantize_kv_tq(&vcache, bits, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN);
        let kdq = dequant_kv_tq(&kslab, &ksc, bits, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN);
        let vdq = dequant_kv_tq(&vslab, &vsc, bits, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN);
        let (qb, kb, vb, ksb, vsb, mut ob) =
            match run(&q, &kslab, &ksc, &vslab, &vsc, AT_HEADS * AT_HEAD_DIM) {
                Ok(t) => t,
                Err(_) => {
                    emit(name, "KERNEL_ERR", "usm-alloc-failed");
                    return;
                }
            };
        // F1: passthrough + sliding-window + Gemma2 softcap into the ONE verdict.
        let cases: [(u32, f32); 3] = [(0, 0.0), (16, 0.0), (0, 50.0)];
        let mut ok_all = true;
        let mut worst_cos = 1.0f64;
        let mut worst_rel = 0.0f64;
        let mut kernel_err: Option<String> = None;
        for (window, softcap) in cases {
            let mut cpu_out = vec![0f32; AT_HEADS * AT_HEAD_DIM];
            k::gqa_attention_one_step_capped_windowed(
                &q, &kdq, &vdq, &mut cpu_out,
                AT_HEADS, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN,
                window as usize, softcap,
            );
            ob.as_mut_slice().fill(f32::NAN);
            // SAFETY: all USM buffers live on `stream`, sized to the geometry;
            // the kernel waits before returning.
            let res = unsafe {
                sk::flash_attn_decode_tq_usm_raw(
                    stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ksb.as_ptr(), vsb.as_ptr(),
                    bits as u32, ob.as_mut_ptr(), AT_HEADS as u32, AT_KV_HEADS as u32,
                    AT_HEAD_DIM as u32, AT_MAX_CTX as u32, AT_KV_LEN as u32, window, softcap,
                )
            };
            if let Err(e) = res {
                kernel_err = Some(format!("{e}"));
                break;
            }
            let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
            if !(cos > 0.999 && max_rel < 0.05) {
                ok_all = false;
            }
            if cos < worst_cos {
                worst_cos = cos;
            }
            if max_rel > worst_rel {
                worst_rel = max_rel;
            }
        }
        let ms = started.elapsed().as_millis();
        match kernel_err {
            Some(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
            None => {
                let verdict = if ok_all { "OK" } else { "MISCOMPUTE" };
                emit(name, verdict, &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa16+cap50) ms={ms} bits={bits}"));
            }
        }
    } else if dir == "prefill" {
        let upto = AT_PREFILL_BASE + AT_PREFILL_NEW;
        let kcache = fill_kv(&mut rng, upto);
        let vcache = fill_kv(&mut rng, upto);
        let q: Vec<f32> = (0..AT_PREFILL_NEW * AT_HEADS * AT_HEAD_DIM)
            .map(|_| rng.f32_pm(0.6))
            .collect();
        let (kslab, ksc) = quantize_kv_tq(&kcache, bits, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, upto);
        let (vslab, vsc) = quantize_kv_tq(&vcache, bits, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, upto);
        let kdq = dequant_kv_tq(&kslab, &ksc, bits, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, upto);
        let vdq = dequant_kv_tq(&vslab, &vsc, bits, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, upto);
        let (qb, kb, vb, ksb, vsb, mut ob) =
            match run(&q, &kslab, &ksc, &vslab, &vsc, q.len()) {
                Ok(t) => t,
                Err(_) => {
                    emit(name, "KERNEL_ERR", "usm-alloc-failed");
                    return;
                }
            };
        // F1: passthrough + sliding-window + Gemma2 softcap into the ONE verdict.
        let cases: [(u32, f32); 3] = [(0, 0.0), (16, 0.0), (0, 50.0)];
        let mut ok_all = true;
        let mut worst_cos = 1.0f64;
        let mut worst_rel = 0.0f64;
        let mut kernel_err: Option<String> = None;
        for (window, softcap) in cases {
            let mut cpu_out = vec![0f32; q.len()];
            k::gqa_attention_flash_prefill_capped_windowed(
                &q, &kdq, &vdq, &mut cpu_out,
                AT_HEADS, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_PREFILL_BASE, AT_PREFILL_NEW,
                window as usize, softcap,
            );
            ob.as_mut_slice().fill(f32::NAN);
            // SAFETY: as the decode arm, q/out sized for AT_PREFILL_NEW queries.
            let res = unsafe {
                sk::flash_attn_prefill_tq_usm_raw(
                    stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ksb.as_ptr(), vsb.as_ptr(),
                    bits as u32, ob.as_mut_ptr(), AT_HEADS as u32, AT_KV_HEADS as u32,
                    AT_HEAD_DIM as u32, AT_MAX_CTX as u32, AT_PREFILL_BASE as u32, AT_PREFILL_NEW as u32,
                    window, softcap,
                )
            };
            if let Err(e) = res {
                kernel_err = Some(format!("{e}"));
                break;
            }
            let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
            if !(cos > 0.999 && max_rel < 0.05) {
                ok_all = false;
            }
            if cos < worst_cos {
                worst_cos = cos;
            }
            if max_rel > worst_rel {
                worst_rel = max_rel;
            }
        }
        let ms = started.elapsed().as_millis();
        match kernel_err {
            Some(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
            None => {
                let verdict = if ok_all { "OK" } else { "MISCOMPUTE" };
                emit(name, verdict, &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa16+cap50) ms={ms} bits={bits}"));
            }
        }
    } else {
        emit(name, "SKIP", "unknown-direction");
    }
}

// ---------------------------------------------------------------
// Parent: spawn one child per probe, classify, print the matrix.
// ---------------------------------------------------------------

const PROBE_TIMEOUT: Duration = Duration::from_secs(90);

/// Drive every SYCL parity probe (one isolated subprocess each) and return a row
/// `(name, verdict, detail)` per probe, printing the per-probe matrix as it goes.
/// `verdict` is the probe's status string: `OK` / `MISCOMPUTE` / `KERNEL_ERR` /
/// `SKIP` / `SLOW` (from the child's `PARITY` line) or `CRASH` / `HANG` (the
/// parent's classification of a dead/timed-out child). Shared by [`run_parent`]
/// (prints the summary) and [`collect_sycl_verdicts`] (P5a: banks the verdicts).
fn run_parent_rows() -> anyhow::Result<Vec<(String, String, String)>> {
    let exe = std::env::current_exe()?;
    let names = probe_names();
    println!(
        "SYCL kernel parity harness — {} probes, one subprocess each \
         (a probe that crashes or hangs the device only takes its child down)",
        names.len()
    );
    if let Some(b) = sk::current_backend_name(0) {
        println!("SYCL backend: {b}");
    } else {
        println!("SYCL backend: none visible (mock build or no device) — expect SKIPs");
    }
    println!();
    let mut rows: Vec<(String, String, String)> = Vec::new();
    for name in &names {
        let mut child = std::process::Command::new(&exe)
            .args(["doctor", "--sycl-parity-probe", name])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()?;
        let started = Instant::now();
        let status = loop {
            match child.try_wait()? {
                Some(s) => break Some(s),
                None if started.elapsed() > PROBE_TIMEOUT => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                None => std::thread::sleep(Duration::from_millis(200)),
            }
        };
        let mut out = String::new();
        if let Some(mut so) = child.stdout.take() {
            use std::io::Read as _;
            let _ = so.read_to_string(&mut out);
        }
        let parsed = out
            .lines()
            .rev()
            .find_map(|l| l.strip_prefix(&format!("PARITY {name} ")))
            .map(|rest| {
                let (verdict, detail) = rest.split_once(' ').unwrap_or((rest, ""));
                (verdict.to_string(), detail.to_string())
            });
        let (verdict, detail) = match (status, parsed) {
            (Some(s), Some((v, d))) if s.success() => (v, d),
            (Some(s), _) => (
                "CRASH".to_string(),
                format!("child exit {:?} (device lost / abort)", s.code()),
            ),
            (None, _) => (
                "HANG".to_string(),
                format!("killed after {}s", PROBE_TIMEOUT.as_secs()),
            ),
        };
        println!("  {name:<24} {verdict:<11} {detail}");
        rows.push((name.clone(), verdict, detail));
    }
    Ok(rows)
}

pub fn run_parent() -> anyhow::Result<()> {
    let rows = run_parent_rows()?;
    let mut counts: std::collections::BTreeMap<&'static str, usize> = Default::default();
    for (_, verdict, _) in &rows {
        let key: &'static str = match verdict.as_str() {
            "OK" => "OK",
            "MISCOMPUTE" => "MISCOMPUTE",
            "KERNEL_ERR" => "KERNEL_ERR",
            "SLOW" => "SLOW",
            "CRASH" => "CRASH",
            "HANG" => "HANG",
            _ => "SKIP",
        };
        *counts.entry(key).or_default() += 1;
    }
    println!();
    let summary: Vec<String> = counts.iter().map(|(k, v)| format!("{k}={v}")).collect();
    println!("summary: {}", summary.join("  "));
    let bad: Vec<&(String, String, String)> = rows
        .iter()
        .filter(|(_, v, _)| matches!(v.as_str(), "MISCOMPUTE" | "CRASH" | "HANG"))
        .collect();
    if !bad.is_empty() {
        println!(
            "\n{} kernel(s) are NOT safe on this backend (MISCOMPUTE = wrong \
             numbers returned as Ok; CRASH/HANG = device loss). Keep GPU \
             dispatch gated off for these.",
            bad.len()
        );
    } else if counts.get("OK").copied().unwrap_or(0) > 0 {
        println!("\nall exercised kernels match their CPU references on this backend.");
    }
    Ok(())
}

/// P5a — bank the SYCL parity verdicts into the tuner cache. Runs the full probe
/// matrix ([`run_parent_rows`]) and maps each row's status to an auto-enable
/// verdict: `SKIP` ⇒ `None` (capability absent; leave unset so the gate's own
/// capability check decides), `CRASH` / `HANG` ⇒ `Some(false)` (device loss ⇒
/// fail-closed; `probe_status_to_verdict` leaves those `None`, so they are forced
/// here), everything else through [`probe_status_to_verdict`] (`OK` ⇒
/// `Some(true)`; `MISCOMPUTE` / `KERNEL_ERR` / `SLOW` ⇒ `Some(false)`). This is
/// what finally banks the SYCL `fused:*` / `matvec:*` / `matvecb:*` / `attn:*` /
/// `matvec_f16in:*` verdicts — previously the SYCL harness printed them but
/// nothing persisted them. `xmx:gemm` is intentionally skipped: it is banked by
/// the dedicated in-process [`collect_xmx_verdict`], so excluding it here keeps a
/// single source of truth for that key.
pub fn collect_sycl_verdicts() -> anyhow::Result<Vec<(String, Option<bool>)>> {
    let rows = run_parent_rows()?;
    Ok(rows
        .into_iter()
        // `xmx:gemm` is banked by the dedicated in-process `collect_xmx_verdict`.
        // `encoder:iq1_s` (F17) is a quantize-time GPU-encoder parity diagnostic
        // that no forward dispatch gate reads, so it banks no verdict either.
        .filter(|(name, _, _)| name != "xmx:gemm" && name != "encoder:iq1_s")
        .map(|(name, status, _detail)| {
            let v = match status.as_str() {
                "CRASH" | "HANG" => Some(false),
                other => probe_status_to_verdict(other),
            };
            (name, v)
        })
        .collect())
}

// ===============================================================
// CUDA parity (`rustllama doctor --cuda-parity`)
// ===============================================================
//
// The NVIDIA analogue of the SYCL harness above: run each native CUDA
// kernel against its CPU reference on identical inputs and report a
// per-kernel verdict. Run in-process — the simple element-wise/attention
// kernels here don't carry the SYCL harness's device-loss risk (which
// forced its subprocess-per-probe design), and the target is real
// NVIDIA hardware where a launch failure surfaces cleanly through the
// error latch (`ck::consume_error_count`) rather than a TDR.
//
// Reuses the quant-byte synthesis + CPU matvec references from the SYCL
// harness; the forward-pass kernels get small inline references that
// mirror the ported kernel math (see cuda/rsl_cuda.cu).

/// Upload an f32 slice into a fresh device buffer (bytes reinterpret).
fn cu_upload_f32<'s>(stream: &'s ck::CudaStream, data: &[f32]) -> Option<ck::CudaDeviceBuffer<'s>> {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    ck::CudaDeviceBuffer::from_host(stream, bytes)
}

/// Download `n` f32 from a device buffer.
fn cu_download_f32(buf: &ck::CudaDeviceBuffer<'_>, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    let _ = buf.copy_to_host(bytes);
    out
}

/// Upload an f32 slice to an MLX device buffer (the Metal analogue of
/// [`cu_upload_f32`]). `None` on alloc/copy failure.
fn mk_upload_f32<'s>(stream: &'s mk::MlxStream, data: &[f32]) -> Option<mk::MlxDeviceBuffer<'s>> {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    mk::MlxDeviceBuffer::from_host(stream, bytes)
}

/// Download `n` f32 from an MLX device buffer.
fn mk_download_f32(buf: &mk::MlxDeviceBuffer<'_>, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    let _ = buf.copy_to_host(bytes);
    out
}

fn cu_emit(name: &str, verdict: &str, detail: &str) {
    probe_rec_push(name, verdict);
    println!("  {name:<24} {verdict:<11} {detail}");
}

/// Verdict from a cos / max_rel pair, appending to the running counts.
fn cu_grade(
    name: &str,
    gpu: &[f32],
    cpu: &[f32],
    cos_min: f64,
    rel_max: f64,
    counts: &mut std::collections::BTreeMap<&'static str, usize>,
) {
    let (cos, max_rel) = compare(gpu, cpu);
    let ok = cos > cos_min && max_rel < rel_max;
    let key: &'static str = if ok { "OK" } else { "MISCOMPUTE" };
    *counts.entry(key).or_default() += 1;
    cu_emit(name, key, &format!("cos={cos:.6} max_rel={max_rel:.4}"));
}

/// Blackwell FP4 tensor-core GEMM parity probe. Builds N f32 activation rows,
/// computes the batched CPU reference `out[n*M + m]` via the scalar FP4 matvec
/// (`cpu_row` is the per-row W×x kernel), runs the TC GEMM, and grades with a
/// loose W4A4-vs-W4A16 gate. `w` is the packed FP4 weight (M×K).
/// Which Blackwell TC GEMM a probe exercises.
#[derive(Clone, Copy)]
enum TcGemm {
    Nvfp4,
    Nvfp4Tma,
    Mxfp4,
    Mxfp8,
    Mxfp6,
    /// Hopper sm_90a FP8 `wgmma` GEMM (MXFP8 W8A8) — same weight/ref as
    /// `Mxfp8`, but dispatches the Hopper `wgmma` kernel (non-TMA).
    Fp8Wgmma,
}

#[allow(clippy::too_many_arguments)]
fn cu_tc_gemm_probe(
    stream: &ck::CudaStream,
    name: &str,
    which: TcGemm,
    w: &[u8],
    cpu_row: fn(&[u8], &[f32], &mut [f32], usize, usize),
    m: usize,
    k: usize,
    n: usize,
    counts: &mut std::collections::BTreeMap<&'static str, usize>,
) {
    let x = gen_x(n * k, 123);
    // Batched reference: out[n*M + m] = row_n · W[m] (col-major in M, as the
    // GEMM writes it).
    let mut cpu = vec![0f32; n * m];
    for ni in 0..n {
        let mut row = vec![0f32; m];
        cpu_row(w, &x[ni * k..(ni + 1) * k], &mut row, m, k);
        cpu[ni * m..(ni + 1) * m].copy_from_slice(&row);
    }
    let (Some(wb), Some(xb), Some(mut ob)) = (
        ck::CudaDeviceBuffer::from_host(stream, w),
        cu_upload_f32(stream, &x),
        ck::CudaDeviceBuffer::alloc(stream, n * m * 4),
    ) else {
        cu_emit(name, "KERNEL_ERR", "device-alloc-failed");
        *counts.entry("KERNEL_ERR").or_default() += 1;
        return;
    };
    let wp = wb.as_ptr();
    let xp = xb.as_ptr() as *const f32;
    let op = ob.as_mut_ptr() as *mut f32;
    // SAFETY: wb (M×K packed blocks), xb (N·K f32), ob (N·M f32) are live device
    // buffers on `stream`; the wrappers synchronize before returning.
    let res = unsafe {
        match which {
            TcGemm::Nvfp4 => ck::gemm_fp4_tc_f32(ck::CudaFp4TcKind::Nvfp4, stream, wp, xp, op, m, n, k),
            TcGemm::Nvfp4Tma => ck::gemm_nvfp4_tc_tma_f32(stream, wp, xp, op, m, n, k),
            TcGemm::Mxfp4 => ck::gemm_fp4_tc_f32(ck::CudaFp4TcKind::Mxfp4, stream, wp, xp, op, m, n, k),
            TcGemm::Mxfp8 => ck::gemm_mxfp8_tc_f32(stream, wp, xp, op, m, n, k),
            TcGemm::Mxfp6 => ck::gemm_mxfp6_tc_f32(stream, wp, xp, op, m, n, k),
            TcGemm::Fp8Wgmma => ck::gemm_mxfp8_wgmma_f32(stream, wp, xp, op, m, n, k, false),
        }
    };
    match res {
        Err(e) => {
            cu_emit(name, "KERNEL_ERR", &format!("{e}"));
            *counts.entry("KERNEL_ERR").or_default() += 1;
        }
        Ok(()) => cu_grade(name, &cu_download_f32(&ob, n * m), &cpu, 0.85, 0.60, counts),
    }
}

/// E4M3 byte -> f32 (reuse the CPU kernel's decoder).
fn ck_e4m3(b: u8) -> f32 {
    k::nvfp4::e4m3_to_f32(b)
}

/// 2:4 magnitude-pruned DENSE reference: out[n*M+m] = sum over each 4-group of
/// K of the 2 largest-|.| E4M3 weights dotted with x. Mirrors the sparse
/// kernel's prune + compute (minus the kernel's activation E4M3 round-trip), so
/// the probe isolates the mma.sp fragment/metadata layout. `w` row-major E4M3.
fn sp24_ref(w: &[u8], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * m];
    for ni in 0..n {
        for mi in 0..m {
            let mut acc = 0f32;
            let mut kk = 0;
            while kk < k {
                let mags = [
                    ck_e4m3(w[mi * k + kk]).abs(),
                    ck_e4m3(w[mi * k + kk + 1]).abs(),
                    ck_e4m3(w[mi * k + kk + 2]).abs(),
                    ck_e4m3(w[mi * k + kk + 3]).abs(),
                ];
                let mut i0 = 0;
                for t in 1..4 {
                    if mags[t] > mags[i0] {
                        i0 = t;
                    }
                }
                let mut i1 = usize::MAX;
                for t in 0..4 {
                    if t != i0 && (i1 == usize::MAX || mags[t] > mags[i1]) {
                        i1 = t;
                    }
                }
                acc += ck_e4m3(w[mi * k + kk + i0]) * x[ni * k + kk + i0];
                acc += ck_e4m3(w[mi * k + kk + i1]) * x[ni * k + kk + i1];
                kk += 4;
            }
            out[ni * m + mi] = acc;
        }
    }
    out
}

// ---- CPU references for the forward-pass kernels ----

fn ref_rope(qk: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, inv_freq: &[f32]) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for j in 0..half {
            let angle = pos as f32 * inv_freq[j];
            let (si, c) = angle.sin_cos();
            let x0 = qk[base + j];
            let x1 = qk[base + j + half];
            qk[base + j] = x0 * c - x1 * si;
            qk[base + j + half] = x0 * si + x1 * c;
        }
    }
}

fn ref_silu_mul(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter()
        .zip(y)
        .map(|(&xv, &yv)| (xv / (1.0 + (-xv).exp())) * yv)
        .collect()
}

// Retained as the simple (window=0, softcap=0) flash-decode reference twin of
// `ref_flash_prefill`. The quant-KV CUDA/MLX probes now grade against the
// `_capped_windowed` reference (so they exercise SWA + soft-cap), so this twin
// has no live caller at the moment — kept for symmetry + ad-hoc reference use.
#[allow(dead_code)]
fn ref_flash_decode(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) -> Vec<f32> {
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut out = vec![0f32; n_heads * head_dim];
    for hh in 0..n_heads {
        let kv_h = hh / n_gqa;
        let qb = hh * head_dim;
        let ob = hh * head_dim;
        let (mut m, mut l) = (f32::NEG_INFINITY, 0.0f32);
        for t in 0..kv_len {
            let koff = (kv_h * max_ctx + t) * head_dim;
            let mut sdot = 0.0f32;
            for i in 0..head_dim {
                sdot += q[qb + i] * kc[koff + i];
            }
            sdot *= scale;
            let m_new = m.max(sdot);
            let rescale = if m.is_finite() {
                (m - m_new).exp()
            } else {
                0.0
            };
            let p = (sdot - m_new).exp();
            l = l * rescale + p;
            for i in 0..head_dim {
                out[ob + i] = out[ob + i] * rescale + p * vc[koff + i];
            }
            m = m_new;
        }
        let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
        for i in 0..head_dim {
            out[ob + i] *= inv;
        }
    }
    out
}

fn ref_flash_prefill(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) -> Vec<f32> {
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut out = vec![0f32; n_new * n_heads * head_dim];
    for qp in 0..n_new {
        for hh in 0..n_heads {
            let kv_h = hh / n_gqa;
            let qoff = (qp * n_heads + hh) * head_dim;
            let kv_len_for_q = kv_len_base + qp + 1;
            let (mut m, mut l) = (f32::NEG_INFINITY, 0.0f32);
            for t in 0..kv_len_for_q {
                let koff = (kv_h * max_ctx + t) * head_dim;
                let mut sdot = 0.0f32;
                for i in 0..head_dim {
                    sdot += q[qoff + i] * kc[koff + i];
                }
                sdot *= scale;
                let m_new = m.max(sdot);
                let rescale = if m.is_finite() {
                    (m - m_new).exp()
                } else {
                    0.0
                };
                let p = (sdot - m_new).exp();
                l = l * rescale + p;
                for i in 0..head_dim {
                    out[qoff + i] = out[qoff + i] * rescale + p * vc[koff + i];
                }
                m = m_new;
            }
            let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
            for i in 0..head_dim {
                out[qoff + i] *= inv;
            }
        }
    }
    out
}

/// C-A CUDA coherence helper: dispatch the single-row packed matvec for one of
/// the 14 fused-capable formats by name, so the coherence probe can run gate +
/// up without triplicating the match. SAFETY: `w`/`x`/`o` must be live device
/// buffers sized `(m,k)`/`(k)`/`(m)`; the wrapper synchronizes before returning.
#[allow(clippy::too_many_arguments)]
unsafe fn cuda_coh_matvec(
    dtype: &str,
    s: &ck::CudaStream,
    w: *const std::ffi::c_void,
    x: *const f32,
    o: *mut f32,
    m: usize,
    k: usize,
) -> Result<(), ck::CudaError> {
    match dtype {
        "q8_0" => ck::matvec_q8_0_packed_f32(s, w, x, o, m, k),
        "q4_k" => ck::matvec_q4_k_packed_f32(s, w, x, o, m, k),
        "q6_k" => ck::matvec_q6_k_packed_f32(s, w, x, o, m, k),
        "q5_k" => ck::matvec_q5_k_packed_f32(s, w, x, o, m, k),
        "iq4_nl" => ck::matvec_iq4_nl_packed_f32(s, w, x, o, m, k),
        "iq4_xs" => ck::matvec_iq4_xs_packed_f32(s, w, x, o, m, k),
        "iq1_s" => ck::matvec_iq1_s_packed_f32(s, w, x, o, m, k),
        "iq1_m" => ck::matvec_iq1_m_packed_f32(s, w, x, o, m, k),
        "iq2_xxs" => ck::matvec_iq2_xxs_packed_f32(s, w, x, o, m, k),
        "iq2_xs" => ck::matvec_iq2_xs_packed_f32(s, w, x, o, m, k),
        "iq2_s" => ck::matvec_iq2_s_packed_f32(s, w, x, o, m, k),
        "iq3_xxs" => ck::matvec_iq3_xxs_packed_f32(s, w, x, o, m, k),
        "iq3_s" => ck::matvec_iq3_s_packed_f32(s, w, x, o, m, k),
        "ptq1_0" => ck::matvec_ptq1_0_packed_f32(s, w, x, o, m, k),
        _ => Err(ck::CudaError::Kernel(-999)),
    }
}

/// C-A CUDA coherence helper: dispatch the fused gate+up matvec for one of the
/// 14 fused-capable formats by name. SAFETY as [`cuda_coh_matvec`], with two
/// weights + two outputs.
#[allow(clippy::too_many_arguments)]
unsafe fn cuda_coh_fused(
    dtype: &str,
    s: &ck::CudaStream,
    gw: *const std::ffi::c_void,
    uw: *const std::ffi::c_void,
    x: *const f32,
    go: *mut f32,
    uo: *mut f32,
    m: usize,
    k: usize,
) -> Result<(), ck::CudaError> {
    match dtype {
        "q8_0" => ck::matvec_q8_0_gate_up_fused(s, gw, uw, x, go, uo, m, k),
        "q4_k" => ck::matvec_q4_k_gate_up_fused(s, gw, uw, x, go, uo, m, k),
        "q6_k" => ck::matvec_q6_k_gate_up_fused(s, gw, uw, x, go, uo, m, k),
        "q5_k" => ck::matvec_q5_k_gate_up_fused(s, gw, uw, x, go, uo, m, k),
        "iq4_nl" => ck::matvec_iq4_nl_gate_up_fused(s, gw, uw, x, go, uo, m, k),
        "iq4_xs" => ck::matvec_iq4_xs_gate_up_fused(s, gw, uw, x, go, uo, m, k),
        "iq1_s" => ck::matvec_iq1_s_gate_up_fused(s, gw, uw, x, go, uo, m, k),
        "iq1_m" => ck::matvec_iq1_m_gate_up_fused(s, gw, uw, x, go, uo, m, k),
        "iq2_xxs" => ck::matvec_iq2_xxs_gate_up_fused(s, gw, uw, x, go, uo, m, k),
        "iq2_xs" => ck::matvec_iq2_xs_gate_up_fused(s, gw, uw, x, go, uo, m, k),
        "iq2_s" => ck::matvec_iq2_s_gate_up_fused(s, gw, uw, x, go, uo, m, k),
        "iq3_xxs" => ck::matvec_iq3_xxs_gate_up_fused(s, gw, uw, x, go, uo, m, k),
        "iq3_s" => ck::matvec_iq3_s_gate_up_fused(s, gw, uw, x, go, uo, m, k),
        "ptq1_0" => ck::matvec_ptq1_0_gate_up_fused(s, gw, uw, x, go, uo, m, k),
        _ => Err(ck::CudaError::Kernel(-999)),
    }
}

pub fn run_cuda_parity() -> anyhow::Result<()> {
    println!("CUDA kernel parity harness (native NVIDIA backend, in-process)");
    let n_dev = ck::device_count();
    if n_dev == 0 {
        println!("no CUDA device visible (no NVIDIA driver / GPU, or CPU-only host) — all SKIP");
        println!("\nsummary: SKIP (no CUDA device)");
        return Ok(());
    }
    match ck::device_info(0) {
        Ok(info) => println!(
            "CUDA device 0: {} ({} MB, compute {}.{}), {} device(s) total",
            info.name,
            info.total_mem_bytes / (1024 * 1024),
            info.compute_capability.0,
            info.compute_capability.1,
            n_dev,
        ),
        Err(_) => println!("{n_dev} CUDA device(s) (info query failed)"),
    }
    let Some(stream) = ck::CudaStream::create(0) else {
        println!("failed to create a CUDA stream on device 0 — all SKIP");
        return Ok(());
    };
    println!();
    let mut counts: std::collections::BTreeMap<&'static str, usize> = Default::default();

    // ---- Packed matvecs the CUDA backend implements ----
    // All 25 packed formats SYCL grades — CUDA ships + dispatches the same set,
    // so grade them all here too (was 17, then the first 7 before that). The
    // final 8 (q4_0/q5_0/q4_1/q5_1/q2_k/q3_k/q8_k/pq2_0) were graded on SYCL but
    // not CUDA; now CUDA matches SYCL format-for-format, single-row AND batched.
    // K constraints (ptq1_0 %128, q8_0/q4_0/q5_0/q4_1/q5_1 %32, the K-quants
    // %256) are all satisfied by MV_K=2048.
    for dtype in [
        "ptq1_0", "q8_0", "q4_k", "q6_k", "mxfp4", "mxfp6", "mxfp8", "q5_k",
        "iq4_nl", "iq4_xs", "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s",
        "iq3_xxs", "iq3_s", "q4_0", "q5_0", "q4_1", "q5_1", "q2_k", "q3_k",
        "q8_k", "pq2_0",
    ] {
        let name = format!("matvec:{dtype}");
        let layout = LAYOUTS.iter().find(|l| l.name == dtype).expect("layout");
        let x = gen_x(MV_K, 42);
        let Some((w, cpu_out)) = finite_ref(layout, cpu_matvec_for(dtype), &x) else {
            cu_emit(&name, "SKIP", "no-finite-reference");
            *counts.entry("SKIP").or_default() += 1;
            continue;
        };
        let (Some(wb), Some(xb), Some(mut ob)) = (
            ck::CudaDeviceBuffer::from_host(&stream, &w),
            cu_upload_f32(&stream, &x),
            ck::CudaDeviceBuffer::alloc(&stream, MV_M * 4),
        ) else {
            cu_emit(&name, "KERNEL_ERR", "device-alloc-failed");
            *counts.entry("KERNEL_ERR").or_default() += 1;
            continue;
        };
        // SAFETY: all three are live device buffers on `stream` sized for
        // (M,K); the wrapper synchronizes before returning.
        let res = unsafe {
            let w = wb.as_ptr();
            let x = xb.as_ptr() as *const f32;
            let o = ob.as_mut_ptr() as *mut f32;
            match dtype {
                "ptq1_0" => ck::matvec_ptq1_0_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "q8_0" => ck::matvec_q8_0_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "q4_k" => ck::matvec_q4_k_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "q6_k" => ck::matvec_q6_k_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "mxfp4" => ck::matvec_mxfp4_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "mxfp6" => ck::matvec_mxfp6_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "mxfp8" => ck::matvec_mxfp8_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "q5_k" => ck::matvec_q5_k_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "iq4_nl" => ck::matvec_iq4_nl_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "iq4_xs" => ck::matvec_iq4_xs_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "iq1_s" => ck::matvec_iq1_s_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "iq1_m" => ck::matvec_iq1_m_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "iq2_xxs" => ck::matvec_iq2_xxs_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "iq2_xs" => ck::matvec_iq2_xs_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "iq2_s" => ck::matvec_iq2_s_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "iq3_xxs" => ck::matvec_iq3_xxs_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "iq3_s" => ck::matvec_iq3_s_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "q4_0" => ck::matvec_q4_0_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "q5_0" => ck::matvec_q5_0_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "q4_1" => ck::matvec_q4_1_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "q5_1" => ck::matvec_q5_1_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "q2_k" => ck::matvec_q2_k_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "q3_k" => ck::matvec_q3_k_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "q8_k" => ck::matvec_q8_k_packed_f32(&stream, w, x, o, MV_M, MV_K),
                "pq2_0" => ck::matvec_pq2_0_packed_f32(&stream, w, x, o, MV_M, MV_K),
                _ => unreachable!(),
            }
        };
        match res {
            Err(e) => {
                cu_emit(&name, "KERNEL_ERR", &format!("{e}"));
                *counts.entry("KERNEL_ERR").or_default() += 1;
            }
            Ok(()) => cu_grade(
                &name,
                &cu_download_f32(&ob, MV_M),
                &cpu_out,
                0.999,
                0.02,
                &mut counts,
            ),
        }
    }

    // ---- iq3_xxs per-weight divergence diagnostic (multi-block) ----
    // iq3_xxs miscomputes on-device though its arithmetic matches the CPU/SYCL
    // reference. A single-block (K=256) sweep MATCHES, but the full matvec
    // (K=2048, 8 blocks/row) diverges — so the defect is multi-block (per-block
    // byte / x offset, or weight-value-dependent). Probe M=1, K=2048 with
    // unit-vector inputs x=e_j ⇒ out[0] = weight j; the weight spans 8 blocks,
    // each with d=f16(1.0) and per-block-varied grid/sign bytes. Dump the first
    // divergent weights with their decoded (block / ib32 / l / grid1-or-2 /
    // byte) so the pattern is obvious (e.g. "only block≥1" = an offset bug, or
    // "specific grid index" = a value bug). CPU ref via the matvec probe's path.
    {
        let name = "iq3_xxs:perweight";
        let cpu = cpu_matvec_for("iq3_xxs");
        const NB: usize = 8; // blocks per row
        const K: usize = NB * 256; // 2048
        let mut w = vec![0u8; NB * 98];
        for b in 0..NB {
            let o = b * 98;
            w[o] = 0x00;
            w[o + 1] = 0x3c; // f16 1.0
            for i in 0..64 {
                w[o + 2 + i] = ((b * 31 + i * 7 + 3) & 0xff) as u8;
            }
            for i in 0..32 {
                w[o + 66 + i] = ((b * 17 + i * 13 + 5) & 0xff) as u8;
            }
        }
        match (
            ck::CudaDeviceBuffer::from_host(&stream, &w),
            ck::CudaDeviceBuffer::alloc(&stream, 4),
        ) {
            (Some(wb), Some(mut ob)) => {
                let mut diverged: Vec<(usize, f32, f32)> = Vec::new();
                for j in 0..K {
                    let mut xj = vec![0f32; K];
                    xj[j] = 1.0;
                    let mut cpu_out = [0f32; 1];
                    cpu(&w, &xj, &mut cpu_out, 1, K);
                    let Some(xb) = cu_upload_f32(&stream, &xj) else {
                        continue;
                    };
                    let ok = unsafe {
                        ck::matvec_iq3_xxs_packed_f32(
                            &stream,
                            wb.as_ptr(),
                            xb.as_ptr() as *const f32,
                            ob.as_mut_ptr() as *mut f32,
                            1,
                            K,
                        )
                    };
                    if ok.is_err() {
                        continue;
                    }
                    let gpu = cu_download_f32(&ob, 1)[0];
                    let c = cpu_out[0];
                    let denom = c.abs().max(1e-6);
                    if (gpu - c).abs() / denom > 1e-3 {
                        diverged.push((j, c, gpu));
                    }
                }
                if diverged.is_empty() {
                    cu_emit(name, "OK", &format!("all {K} weights match ({NB} blocks)"));
                    *counts.entry("OK").or_default() += 1;
                } else {
                    // Per-block divergence histogram so "only block≥1" is obvious.
                    let mut per_block = [0usize; NB];
                    for &(j, _, _) in &diverged {
                        per_block[j / 256] += 1;
                    }
                    cu_emit(
                        name,
                        "MISCOMPUTE",
                        &format!(
                            "{} / {K} weights diverge; per-block {:?}",
                            diverged.len(),
                            per_block
                        ),
                    );
                    *counts.entry("MISCOMPUTE").or_default() += 1;
                    for &(j, c, g) in diverged.iter().take(24) {
                        let blk = j / 256;
                        let p = j % 256;
                        let ib32 = p / 32;
                        let pp = p % 32;
                        let l = pp / 8;
                        let pl = pp % 8;
                        let which = if pl < 4 { "g1" } else { "g2" };
                        let byte = pl % 4;
                        println!(
                            "      w[{j:>4}] cpu={c:+.4} gpu={g:+.4}  (blk={blk} ib32={ib32} l={l} {which} byte={byte})"
                        );
                    }
                }
            }
            _ => {
                cu_emit(name, "KERNEL_ERR", "device-alloc-failed");
                *counts.entry("KERNEL_ERR").or_default() += 1;
            }
        }
    }

    // ---- Batched (prefill) packed matvecs ----
    // The CUDA batched kernels (`matvec_<fmt>_packed_f32_batched`, the N-lifted
    // prefill twins) run during prefill for every non-Q4_K format (and Q4_K
    // when the GEMM gate doesn't hold) but were NEVER graded — the matvec loop
    // above only exercises the single-row kernels, and the SYCL `matvecb:` probe
    // is SYCL-only. Grade the same 17 formats batched here: [N,M] GPU output vs
    // the CPU reference run per-row. (Q4_K's prefill GEMM is graded separately as
    // `gemm:q4_k_f32`; this grades the generic batched matvec path.)
    for dtype in [
        "ptq1_0", "q8_0", "q4_k", "q6_k", "mxfp4", "mxfp6", "mxfp8", "q5_k",
        "iq4_nl", "iq4_xs", "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s",
        "iq3_xxs", "iq3_s", "q4_0", "q5_0", "q4_1", "q5_1", "q2_k", "q3_k",
        "q8_k", "pq2_0",
    ] {
        let name = format!("matvecb:{dtype}");
        let layout = LAYOUTS.iter().find(|l| l.name == dtype).expect("layout");
        let cpu = cpu_matvec_for(dtype);
        let x = gen_x(MV_N * MV_K, 42);
        // Pick a weight whose row-0 CPU output is finite, then run the CPU
        // reference for all N rows (mirrors the SYCL batched probe).
        let Some((w, _)) = finite_ref(layout, cpu, &x[0..MV_K]) else {
            cu_emit(&name, "SKIP", "no-finite-reference");
            *counts.entry("SKIP").or_default() += 1;
            continue;
        };
        let mut cpu_out = vec![0f32; MV_N * MV_M];
        for row in 0..MV_N {
            cpu(
                &w,
                &x[row * MV_K..(row + 1) * MV_K],
                &mut cpu_out[row * MV_M..(row + 1) * MV_M],
                MV_M,
                MV_K,
            );
        }
        if !cpu_out.iter().all(|v| v.is_finite()) {
            cu_emit(&name, "SKIP", "no-finite-reference");
            *counts.entry("SKIP").or_default() += 1;
            continue;
        }
        let (Some(wb), Some(xb), Some(mut ob)) = (
            ck::CudaDeviceBuffer::from_host(&stream, &w),
            cu_upload_f32(&stream, &x),
            ck::CudaDeviceBuffer::alloc(&stream, MV_N * MV_M * 4),
        ) else {
            cu_emit(&name, "KERNEL_ERR", "device-alloc-failed");
            *counts.entry("KERNEL_ERR").or_default() += 1;
            continue;
        };
        // SAFETY: w (M,K), x (N,K), o (N,M) are live device buffers on `stream`;
        // each batched wrapper derives extents from (M,K,N) and synchronizes.
        let res = unsafe {
            let w = wb.as_ptr();
            let x = xb.as_ptr() as *const f32;
            let o = ob.as_mut_ptr() as *mut f32;
            match dtype {
                "ptq1_0" => ck::matvec_ptq1_0_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "q8_0" => ck::matvec_q8_0_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "q4_k" => ck::matvec_q4_k_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "q6_k" => ck::matvec_q6_k_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "mxfp4" => ck::matvec_mxfp4_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "mxfp6" => ck::matvec_mxfp6_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "mxfp8" => ck::matvec_mxfp8_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "q5_k" => ck::matvec_q5_k_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "iq4_nl" => ck::matvec_iq4_nl_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "iq4_xs" => ck::matvec_iq4_xs_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "iq1_s" => ck::matvec_iq1_s_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "iq1_m" => ck::matvec_iq1_m_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "iq2_xxs" => ck::matvec_iq2_xxs_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "iq2_xs" => ck::matvec_iq2_xs_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "iq2_s" => ck::matvec_iq2_s_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "iq3_xxs" => ck::matvec_iq3_xxs_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "iq3_s" => ck::matvec_iq3_s_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "q4_0" => ck::matvec_q4_0_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "q5_0" => ck::matvec_q5_0_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "q4_1" => ck::matvec_q4_1_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "q5_1" => ck::matvec_q5_1_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "q2_k" => ck::matvec_q2_k_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "q3_k" => ck::matvec_q3_k_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "q8_k" => ck::matvec_q8_k_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                "pq2_0" => ck::matvec_pq2_0_packed_f32_batched(&stream, w, x, o, MV_M, MV_K, MV_N),
                _ => unreachable!(),
            }
        };
        match res {
            Err(e) => {
                cu_emit(&name, "KERNEL_ERR", &format!("{e}"));
                *counts.entry("KERNEL_ERR").or_default() += 1;
            }
            Ok(()) => cu_grade(
                &name,
                &cu_download_f32(&ob, MV_N * MV_M),
                &cpu_out,
                0.999,
                0.02,
                &mut counts,
            ),
        }
    }

    // ---- Fused gate+up matvec (decode) ----
    // One launch computes gate_out + up_out, reusing each format's row_dot — so
    // bit-exact to the single matvec. Grades the concatenated [gate; up] output
    // against two independent CPU matvecs (distinct gate/up weights), confirming
    // the dual-output write + the fusion wiring. Same 14 formats as SYCL.
    for dtype in [
        "q8_0", "q4_k", "q6_k", "q5_k", "iq4_nl", "iq4_xs", "iq1_s", "iq1_m",
        "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "ptq1_0",
    ] {
        let name = format!("fused:{dtype}");
        let layout = LAYOUTS.iter().find(|l| l.name == dtype).expect("layout");
        let x = gen_x(MV_K, 44);
        let cpu = cpu_matvec_for(dtype);
        let Some((wg, cpu_g)) = finite_ref(layout, cpu, &x) else {
            cu_emit(&name, "SKIP", "no-finite-reference");
            *counts.entry("SKIP").or_default() += 1;
            continue;
        };
        // Second, distinct weight for `up`.
        let wu = gen_quant_bytes(layout, MV_M, MV_K, 0xBEEF0007);
        let mut cpu_u = vec![0f32; MV_M];
        cpu(&wu, &x, &mut cpu_u, MV_M, MV_K);
        if !cpu_u.iter().all(|v| v.is_finite()) {
            cu_emit(&name, "SKIP", "no-finite-reference");
            *counts.entry("SKIP").or_default() += 1;
            continue;
        }
        let (Some(gwb), Some(uwb), Some(xb), Some(mut gob), Some(mut uob)) = (
            ck::CudaDeviceBuffer::from_host(&stream, &wg),
            ck::CudaDeviceBuffer::from_host(&stream, &wu),
            cu_upload_f32(&stream, &x),
            ck::CudaDeviceBuffer::alloc(&stream, MV_M * 4),
            ck::CudaDeviceBuffer::alloc(&stream, MV_M * 4),
        ) else {
            cu_emit(&name, "KERNEL_ERR", "device-alloc-failed");
            *counts.entry("KERNEL_ERR").or_default() += 1;
            continue;
        };
        // SAFETY: five live device buffers on `stream` sized for (M,K)/M; the
        // wrapper synchronizes before returning.
        let res = unsafe {
            let gw = gwb.as_ptr();
            let uw = uwb.as_ptr();
            let xp = xb.as_ptr() as *const f32;
            let go = gob.as_mut_ptr() as *mut f32;
            let uo = uob.as_mut_ptr() as *mut f32;
            match dtype {
                "q8_0" => ck::matvec_q8_0_gate_up_fused(&stream, gw, uw, xp, go, uo, MV_M, MV_K),
                "q4_k" => ck::matvec_q4_k_gate_up_fused(&stream, gw, uw, xp, go, uo, MV_M, MV_K),
                "q6_k" => ck::matvec_q6_k_gate_up_fused(&stream, gw, uw, xp, go, uo, MV_M, MV_K),
                "q5_k" => ck::matvec_q5_k_gate_up_fused(&stream, gw, uw, xp, go, uo, MV_M, MV_K),
                "iq4_nl" => ck::matvec_iq4_nl_gate_up_fused(&stream, gw, uw, xp, go, uo, MV_M, MV_K),
                "iq4_xs" => ck::matvec_iq4_xs_gate_up_fused(&stream, gw, uw, xp, go, uo, MV_M, MV_K),
                "iq1_s" => ck::matvec_iq1_s_gate_up_fused(&stream, gw, uw, xp, go, uo, MV_M, MV_K),
                "iq1_m" => ck::matvec_iq1_m_gate_up_fused(&stream, gw, uw, xp, go, uo, MV_M, MV_K),
                "iq2_xxs" => ck::matvec_iq2_xxs_gate_up_fused(&stream, gw, uw, xp, go, uo, MV_M, MV_K),
                "iq2_xs" => ck::matvec_iq2_xs_gate_up_fused(&stream, gw, uw, xp, go, uo, MV_M, MV_K),
                "iq2_s" => ck::matvec_iq2_s_gate_up_fused(&stream, gw, uw, xp, go, uo, MV_M, MV_K),
                "iq3_xxs" => ck::matvec_iq3_xxs_gate_up_fused(&stream, gw, uw, xp, go, uo, MV_M, MV_K),
                "iq3_s" => ck::matvec_iq3_s_gate_up_fused(&stream, gw, uw, xp, go, uo, MV_M, MV_K),
                "ptq1_0" => ck::matvec_ptq1_0_gate_up_fused(&stream, gw, uw, xp, go, uo, MV_M, MV_K),
                _ => unreachable!(),
            }
        };
        match res {
            Err(e) => {
                cu_emit(&name, "KERNEL_ERR", &format!("{e}"));
                *counts.entry("KERNEL_ERR").or_default() += 1;
            }
            Ok(()) => {
                let mut got = cu_download_f32(&gob, MV_M);
                got.extend_from_slice(&cu_download_f32(&uob, MV_M));
                let mut refv = cpu_g.clone();
                refv.extend_from_slice(&cpu_u);
                cu_grade(&name, &got, &refv, 0.999, 0.02, &mut counts);
            }
        }
    }

    // ---- C-A: real-MoE-forward coherence (verdict moe:gate_up_coherent:<dtype>) ----
    // CUDA twin of the SYCL coherence probe (WRITE-BLIND — settles on the pod):
    // the per-expert FFN (several experts, non-512 shapes, SwiGLU, routing sum)
    // graded FUSED ≡ SEPARATE two-matvec ≡ CPU. `accel::moe_gate_up_fused_enabled`
    // ANDs this with the bare `fused:` parity above, so a generic-GEMM pass alone
    // cannot re-enable the 2026-09-06 incident path.
    for dtype in [
        "q8_0", "q4_k", "q6_k", "q5_k", "iq4_nl", "iq4_xs", "iq1_s", "iq1_m",
        "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "ptq1_0",
    ] {
        let name = format!("moe:gate_up_coherent:{dtype}");
        let layout = LAYOUTS.iter().find(|l| l.name == dtype).expect("layout");
        let cpu = cpu_matvec_for(dtype);
        let mut worst_cos_cpu = 1.0f64;
        let mut worst_rel_cpu = 0.0f64;
        let mut worst_cos_sep = 1.0f64;
        let mut worst_rel_sep = 0.0f64;
        let mut failed: Option<(&'static str, String)> = None;
        'shapes: for (si, &(m, k)) in MOE_COH_SHAPES.iter().enumerate() {
            let x = gen_x(k, 0x5EED_0001 ^ (si as u32).wrapping_mul(7919));
            let Some(xb) = cu_upload_f32(&stream, &x) else {
                failed = Some(("KERNEL_ERR", "device-alloc-failed(x)".into()));
                break 'shapes;
            };
            let mut inter_cpu = vec![0f32; m];
            let mut inter_sep = vec![0f32; m];
            let mut inter_fused = vec![0f32; m];
            for e in 0..MOE_COH_EXPERTS {
                let w_route = MOE_COH_ROUTE_W[e];
                let sg = 0xC0DE_0000u32 ^ ((si * 97 + e * 13) as u32);
                let su = 0xBEEF_0000u32 ^ ((si * 131 + e * 17) as u32);
                let Some((wg, cpu_g)) = coh_gen_finite(layout, cpu, m, k, &x, sg) else {
                    failed = Some(("SKIP", "no-finite-reference(gate)".into()));
                    break 'shapes;
                };
                let Some((wu, cpu_u)) = coh_gen_finite(layout, cpu, m, k, &x, su) else {
                    failed = Some(("SKIP", "no-finite-reference(up)".into()));
                    break 'shapes;
                };
                let (
                    Some(gwb),
                    Some(uwb),
                    Some(mut sg_o),
                    Some(mut su_o),
                    Some(mut fg_o),
                    Some(mut fu_o),
                ) = (
                    ck::CudaDeviceBuffer::from_host(&stream, &wg),
                    ck::CudaDeviceBuffer::from_host(&stream, &wu),
                    ck::CudaDeviceBuffer::alloc(&stream, m * 4),
                    ck::CudaDeviceBuffer::alloc(&stream, m * 4),
                    ck::CudaDeviceBuffer::alloc(&stream, m * 4),
                    ck::CudaDeviceBuffer::alloc(&stream, m * 4),
                )
                else {
                    failed = Some(("KERNEL_ERR", "device-alloc-failed(expert)".into()));
                    break 'shapes;
                };
                // SAFETY: live device buffers on `stream` sized (M,K)/(M); each
                // wrapper synchronizes before returning.
                let res = unsafe {
                    let gw = gwb.as_ptr();
                    let uw = uwb.as_ptr();
                    let xp = xb.as_ptr() as *const f32;
                    let sgo = sg_o.as_mut_ptr() as *mut f32;
                    let suo = su_o.as_mut_ptr() as *mut f32;
                    let fgo = fg_o.as_mut_ptr() as *mut f32;
                    let fuo = fu_o.as_mut_ptr() as *mut f32;
                    cuda_coh_matvec(dtype, &stream, gw, xp, sgo, m, k)
                        .and_then(|()| cuda_coh_matvec(dtype, &stream, uw, xp, suo, m, k))
                        .and_then(|()| cuda_coh_fused(dtype, &stream, gw, uw, xp, fgo, fuo, m, k))
                };
                if let Err(e_) = res {
                    failed = Some(("KERNEL_ERR", format!("{e_}")));
                    break 'shapes;
                }
                let sg_out = cu_download_f32(&sg_o, m);
                let su_out = cu_download_f32(&su_o, m);
                let fg_out = cu_download_f32(&fg_o, m);
                let fu_out = cu_download_f32(&fu_o, m);
                for j in 0..m {
                    inter_cpu[j] += w_route * silu(cpu_g[j]) * cpu_u[j];
                    inter_sep[j] += w_route * silu(sg_out[j]) * su_out[j];
                    inter_fused[j] += w_route * silu(fg_out[j]) * fu_out[j];
                }
            }
            let (cos_cpu, rel_cpu) = compare(&inter_fused, &inter_cpu);
            let (cos_sep, rel_sep) = compare(&inter_fused, &inter_sep);
            worst_cos_cpu = worst_cos_cpu.min(cos_cpu);
            worst_rel_cpu = worst_rel_cpu.max(rel_cpu);
            worst_cos_sep = worst_cos_sep.min(cos_sep);
            worst_rel_sep = worst_rel_sep.max(rel_sep);
        }
        if let Some((v, detail)) = failed {
            cu_emit(&name, v, &detail);
            *counts.entry(v).or_default() += 1;
            continue;
        }
        let cpu_ok = worst_cos_cpu > 0.999 && worst_rel_cpu < 0.05;
        let sep_ok = worst_cos_sep > 0.9995 && worst_rel_sep < 0.02;
        let verdict = if cpu_ok && sep_ok { "OK" } else { "MISCOMPUTE" };
        cu_emit(
            &name,
            verdict,
            &format!(
                "cos_cpu={worst_cos_cpu:.6} rel_cpu={worst_rel_cpu:.4} \
                 cos_sep={worst_cos_sep:.6} rel_sep={worst_rel_sep:.4}"
            ),
        );
        *counts.entry(verdict).or_default() += 1;
    }

    // ---- Q4_K prefill GEMM (verdict: gemm:q4_k_f32, BIT-EXACT + perf-gated) ----
    // Two-part gate (the GEMM is bit-exact, but it's dispatched FIRST, so it must
    // also be the fastest or it would regress): (1) correctness vs the f32 Q4_K
    // matvec reference at the bit-exact tolerance (0.999/0.02); (2) it must beat
    // the bit-exact batched matvec on a prefill shape, else "SLOW" and the
    // dispatch keeps the existing path. Fail-closed.
    #[allow(clippy::never_loop)]
    for _gemm in 0..1usize {
        const GEMM_N: usize = 8;
        let name = "gemm:q4_k_f32";
        let layout = LAYOUTS.iter().find(|l| l.name == "q4_k").expect("layout");
        let cpu = cpu_matvec_for("q4_k");
        let xb_host: Vec<f32> = gen_x(GEMM_N * MV_K, 2024);
        let Some((w, _)) = finite_ref(layout, cpu, &xb_host[..MV_K]) else {
            cu_emit(name, "SKIP", "no-finite-reference");
            *counts.entry("SKIP").or_default() += 1;
            continue;
        };
        let mut cpu_out = vec![0f32; GEMM_N * MV_M];
        for r in 0..GEMM_N {
            let mut row = vec![0f32; MV_M];
            cpu(&w, &xb_host[r * MV_K..(r + 1) * MV_K], &mut row, MV_M, MV_K);
            cpu_out[r * MV_M..(r + 1) * MV_M].copy_from_slice(&row);
        }
        let (Some(wb), Some(xbd), Some(mut ob)) = (
            ck::CudaDeviceBuffer::from_host(&stream, &w),
            cu_upload_f32(&stream, &xb_host),
            ck::CudaDeviceBuffer::alloc(&stream, GEMM_N * MV_M * 4),
        ) else {
            cu_emit(name, "KERNEL_ERR", "device-alloc-failed");
            *counts.entry("KERNEL_ERR").or_default() += 1;
            continue;
        };
        // SAFETY: three live device buffers on `stream` sized for (N,M)/(N,K);
        // the wrapper synchronizes before returning.
        let res = unsafe {
            ck::gemm_q4_k_f32(
                &stream,
                wb.as_ptr(),
                xbd.as_ptr() as *const f32,
                ob.as_mut_ptr() as *mut f32,
                MV_M,
                MV_K,
                GEMM_N,
            )
        };
        if let Err(e) = res {
            cu_emit(name, "KERNEL_ERR", &format!("{e}"));
            *counts.entry("KERNEL_ERR").or_default() += 1;
            continue;
        }
        // (1) Correctness — bit-exact vs the f32 Q4_K matvec reference.
        let (cos, max_rel) = compare(&cu_download_f32(&ob, GEMM_N * MV_M), &cpu_out);
        if !(cos > 0.999 && max_rel < 0.02) {
            cu_emit(name, "MISCOMPUTE", &format!("cos={cos:.6} max_rel={max_rel:.4}"));
            *counts.entry("MISCOMPUTE").or_default() += 1;
            continue;
        }
        // (2) Performance — the GEMM is dispatched FIRST, so enable it only if it
        // beats the bit-exact batched matvec on a prefill shape; otherwise "SLOW"
        // (decided off) and the existing path runs.
        const PM: usize = 2048;
        const PK: usize = 4096;
        const PN: usize = 512; // the real prefill chunk — a GEMM needs the batch
        const WARMUP: usize = 3;
        const ITERS: usize = 20;
        let wp = gen_quant_bytes(layout, PM, PK, 0xF00D);
        let xp: Vec<f32> = gen_x(PN * PK, 91);
        let (Some(wpb), Some(xpb), Some(mut opb)) = (
            ck::CudaDeviceBuffer::from_host(&stream, &wp),
            cu_upload_f32(&stream, &xp),
            ck::CudaDeviceBuffer::alloc(&stream, PN * PM * 4),
        ) else {
            cu_emit(name, "SLOW", "perf-shape device alloc failed");
            *counts.entry("SLOW").or_default() += 1;
            continue;
        };
        let (wptr, xptr, optr) = (
            wpb.as_ptr(),
            xpb.as_ptr() as *const f32,
            opb.as_mut_ptr() as *mut f32,
        );
        // SAFETY: live device buffers sized for (PM,PK,PN); each wrapper syncs
        // per call, so the ratios are fair.
        let (t_gemm, t_bit) = unsafe {
            for _ in 0..WARMUP {
                let _ = ck::gemm_q4_k_f32(&stream, wptr, xptr, optr, PM, PK, PN);
                let _ = ck::matvec_q4_k_packed_f32_batched(&stream, wptr, xptr, optr, PM, PK, PN);
            }
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                let _ = ck::gemm_q4_k_f32(&stream, wptr, xptr, optr, PM, PK, PN);
            }
            let tg = t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            for _ in 0..ITERS {
                let _ = ck::matvec_q4_k_packed_f32_batched(&stream, wptr, xptr, optr, PM, PK, PN);
            }
            let tb = t1.elapsed().as_secs_f64();
            (tg, tb)
        };
        let detail = format!(
            "gemm {:.3} vs matvec {:.3} ms/call ({PM}x{PK}x{PN})",
            t_gemm * 1e3 / ITERS as f64,
            t_bit * 1e3 / ITERS as f64,
        );
        if t_gemm < t_bit * 0.95 {
            cu_emit(name, "OK", &detail);
            *counts.entry("OK").or_default() += 1;
        } else {
            cu_emit(name, "SLOW", &detail);
            *counts.entry("SLOW").or_default() += 1;
        }
    }

    // ---- Q4_K int8 tensor-core GEMM (verdict: gemm:q4_k_w8a8_tc, W8A8 LOSSY) ----
    // Two-part gate: (1) correctness vs the FAIR W8A8 reference (the SAME per-32
    // int8 activation round-trip, then the f32 Q4_K matvec — int8 TC math is
    // exact, so only quant rounding differs); (2) it must beat the LOSSLESS f32
    // GEMM on a prefill shape, else "SLOW" (the f32 GEMM stays the path). It is
    // dispatched FIRST when enabled, so this gate decides lossy-vs-lossless.
    #[allow(clippy::never_loop)]
    for _tc in 0..1usize {
        const TC_N: usize = 8;
        let name = "gemm:q4_k_w8a8_tc";
        let layout = LAYOUTS.iter().find(|l| l.name == "q4_k").expect("layout");
        let cpu = cpu_matvec_for("q4_k");
        let xb_host: Vec<f32> = gen_x(TC_N * MV_K, 4242);
        let Some((w, _)) = finite_ref(layout, cpu, &xb_host[..MV_K]) else {
            cu_emit(name, "SKIP", "no-finite-reference");
            *counts.entry("SKIP").or_default() += 1;
            continue;
        };
        let mut cpu_out = vec![0f32; TC_N * MV_M];
        for r in 0..TC_N {
            let xr = int8_act_roundtrip(&xb_host[r * MV_K..(r + 1) * MV_K]);
            let mut row = vec![0f32; MV_M];
            cpu(&w, &xr, &mut row, MV_M, MV_K);
            cpu_out[r * MV_M..(r + 1) * MV_M].copy_from_slice(&row);
        }
        let (Some(wb), Some(xbd), Some(mut ob)) = (
            ck::CudaDeviceBuffer::from_host(&stream, &w),
            cu_upload_f32(&stream, &xb_host),
            ck::CudaDeviceBuffer::alloc(&stream, TC_N * MV_M * 4),
        ) else {
            cu_emit(name, "KERNEL_ERR", "device-alloc-failed");
            *counts.entry("KERNEL_ERR").or_default() += 1;
            continue;
        };
        // SAFETY: live device buffers on `stream` sized for (N,M)/(N,K).
        let res = unsafe {
            ck::gemm_q4_k_w8a8_tc(
                &stream,
                wb.as_ptr(),
                xbd.as_ptr() as *const f32,
                ob.as_mut_ptr() as *mut f32,
                MV_M,
                MV_K,
                TC_N,
            )
        };
        if let Err(e) = res {
            cu_emit(name, "KERNEL_ERR", &format!("{e}"));
            *counts.entry("KERNEL_ERR").or_default() += 1;
            continue;
        }
        let (cos, max_rel) = compare(&cu_download_f32(&ob, TC_N * MV_M), &cpu_out);
        if !(cos > 0.999 && max_rel < 0.05) {
            cu_emit(name, "MISCOMPUTE", &format!("cos={cos:.6} max_rel={max_rel:.4}"));
            *counts.entry("MISCOMPUTE").or_default() += 1;
            continue;
        }
        const PM: usize = 2048;
        const PK: usize = 4096;
        const PN: usize = 512;
        const WARMUP: usize = 3;
        const ITERS: usize = 20;
        let wp = gen_quant_bytes(layout, PM, PK, 0x8AC8);
        let xp: Vec<f32> = gen_x(PN * PK, 55);
        let (Some(wpb), Some(xpb), Some(mut opb)) = (
            ck::CudaDeviceBuffer::from_host(&stream, &wp),
            cu_upload_f32(&stream, &xp),
            ck::CudaDeviceBuffer::alloc(&stream, PN * PM * 4),
        ) else {
            cu_emit(name, "SLOW", "perf-shape device alloc failed");
            *counts.entry("SLOW").or_default() += 1;
            continue;
        };
        let (wptr, xptr, optr) = (
            wpb.as_ptr(),
            xpb.as_ptr() as *const f32,
            opb.as_mut_ptr() as *mut f32,
        );
        // SAFETY: live device buffers sized for (PM,PK,PN); each wrapper syncs.
        let (t_tc, t_gemm) = unsafe {
            for _ in 0..WARMUP {
                let _ = ck::gemm_q4_k_w8a8_tc(&stream, wptr, xptr, optr, PM, PK, PN);
                let _ = ck::gemm_q4_k_f32(&stream, wptr, xptr, optr, PM, PK, PN);
            }
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                let _ = ck::gemm_q4_k_w8a8_tc(&stream, wptr, xptr, optr, PM, PK, PN);
            }
            let tt = t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            for _ in 0..ITERS {
                let _ = ck::gemm_q4_k_f32(&stream, wptr, xptr, optr, PM, PK, PN);
            }
            let tg = t1.elapsed().as_secs_f64();
            (tt, tg)
        };
        let detail = format!(
            "tc {:.3} vs f32gemm {:.3} ms/call ({PM}x{PK}x{PN})",
            t_tc * 1e3 / ITERS as f64,
            t_gemm * 1e3 / ITERS as f64,
        );
        if t_tc < t_gemm * 0.95 {
            cu_emit(name, "OK", &detail);
            *counts.entry("OK").or_default() += 1;
        } else {
            cu_emit(name, "SLOW", &detail);
            *counts.entry("SLOW").or_default() += 1;
        }
    }

    // ---- Q4_K W4A8 DECODE matvec (verdict: matvec:q4_k_w4a8, LOSSY + perf-gated) ----
    // The decode (single-row) twin of the W8A8 GEMM above. Same two-part gate:
    // (1) correctness vs the FAIR W8A8 reference (int8-activation round-trip →
    // f32 Q4_K matvec — the int8 dot is exact, only activation rounding differs);
    // (2) it must beat the bit-exact f32 warp matvec on a decode shape, else
    // "SLOW". Dispatched FIRST on single-row Q4_K decode when enabled, so this
    // gate decides lossy-vs-lossless — matching how `gemm:q4_k_w8a8_tc` gates
    // prefill.
    #[allow(clippy::never_loop)]
    for _w4 in 0..1usize {
        let name = "matvec:q4_k_w4a8";
        let layout = LAYOUTS.iter().find(|l| l.name == "q4_k").expect("layout");
        let cpu = cpu_matvec_for("q4_k");
        let x = gen_x(MV_K, 4343);
        let Some((w, _)) = finite_ref(layout, cpu, &x) else {
            cu_emit(name, "SKIP", "no-finite-reference");
            *counts.entry("SKIP").or_default() += 1;
            continue;
        };
        // Fair W4A8 reference: int8-activation round-trip, then the f32 Q4_K matvec.
        let xr = int8_act_roundtrip(&x);
        let mut cpu_out = vec![0f32; MV_M];
        cpu(&w, &xr, &mut cpu_out, MV_M, MV_K);
        let (Some(wb), Some(xbd), Some(mut ob)) = (
            ck::CudaDeviceBuffer::from_host(&stream, &w),
            cu_upload_f32(&stream, &x),
            ck::CudaDeviceBuffer::alloc(&stream, MV_M * 4),
        ) else {
            cu_emit(name, "KERNEL_ERR", "device-alloc-failed");
            *counts.entry("KERNEL_ERR").or_default() += 1;
            continue;
        };
        // SAFETY: live device buffers on `stream` sized (M,K)/(K)/(M).
        let res = unsafe {
            ck::matvec_q4_k_w4a8(
                &stream,
                wb.as_ptr(),
                xbd.as_ptr() as *const f32,
                ob.as_mut_ptr() as *mut f32,
                MV_M,
                MV_K,
            )
        };
        if let Err(e) = res {
            cu_emit(name, "KERNEL_ERR", &format!("{e}"));
            *counts.entry("KERNEL_ERR").or_default() += 1;
            continue;
        }
        let (cos, max_rel) = compare(&cu_download_f32(&ob, MV_M), &cpu_out);
        if !(cos > 0.999 && max_rel < 0.05) {
            cu_emit(name, "MISCOMPUTE", &format!("cos={cos:.6} max_rel={max_rel:.4}"));
            *counts.entry("MISCOMPUTE").or_default() += 1;
            continue;
        }
        // Perf: W4A8 vs the bit-exact f32 warp matvec on a decode shape (M×1).
        const PM: usize = 4096;
        const PK: usize = 4096;
        const WARMUP: usize = 3;
        const ITERS: usize = 50;
        let wp = gen_quant_bytes(layout, PM, PK, 0x9AD9);
        let xp: Vec<f32> = gen_x(PK, 57);
        let (Some(wpb), Some(xpb), Some(mut opb)) = (
            ck::CudaDeviceBuffer::from_host(&stream, &wp),
            cu_upload_f32(&stream, &xp),
            ck::CudaDeviceBuffer::alloc(&stream, PM * 4),
        ) else {
            cu_emit(name, "SLOW", "perf-shape device alloc failed");
            *counts.entry("SLOW").or_default() += 1;
            continue;
        };
        let (wptr, xptr, optr) = (
            wpb.as_ptr(),
            xpb.as_ptr() as *const f32,
            opb.as_mut_ptr() as *mut f32,
        );
        // SAFETY: live device buffers sized (PM,PK); each wrapper syncs.
        let (t_w4a8, t_f32) = unsafe {
            for _ in 0..WARMUP {
                let _ = ck::matvec_q4_k_w4a8(&stream, wptr, xptr, optr, PM, PK);
                let _ = ck::matvec_q4_k_packed_f32(&stream, wptr, xptr, optr, PM, PK);
            }
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                let _ = ck::matvec_q4_k_w4a8(&stream, wptr, xptr, optr, PM, PK);
            }
            let t4 = t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            for _ in 0..ITERS {
                let _ = ck::matvec_q4_k_packed_f32(&stream, wptr, xptr, optr, PM, PK);
            }
            let tf = t1.elapsed().as_secs_f64();
            (t4, tf)
        };
        let detail = format!(
            "w4a8 {:.4} vs f32 {:.4} ms/call ({PM}x{PK})",
            t_w4a8 * 1e3 / ITERS as f64,
            t_f32 * 1e3 / ITERS as f64,
        );
        if t_w4a8 < t_f32 * 0.95 {
            cu_emit(name, "OK", &detail);
            *counts.entry("OK").or_default() += 1;
        } else {
            cu_emit(name, "SLOW", &detail);
            *counts.entry("SLOW").or_default() += 1;
        }
    }

    // ---- Blackwell SM12x FP4 tensor-core GEMM (W4A4) ----
    // The new hand-rolled block-scaled `mma.sync` path (cuda/rsl_blackwell.cuh).
    // SKIP unless this is a real SM12x Blackwell device built with the TC path
    // (RUSTLLAMA_CUDA_ARCHS including sm_120a/121a). The TC GEMM quantizes the
    // activation to FP4 on the fly (W4A4), while the CPU reference keeps f32
    // activations (W4A16), so the gate is deliberately LOOSE — it catches gross
    // fragment/scale layout bugs (which drive cosine toward 0), not the
    // activation-quant error. On-device numbers on the Spark settle the real
    // tolerance; until then every assumption tagged SPARK-VALIDATE in the
    // header is what this probe exists to check.
    if !ck::blackwell_tc_available(0) {
        for nm in [
            "gemm:nvfp4_tc(W4A4)", "gemm:nvfp4_tc_tma(W4A4)", "gemm:mxfp4_tc(W4A4)",
            "gemm:mxfp8_tc(W8A8)", "gemm:mxfp6_tc(W6A6)", "gemm:fp8_sp24(2:4)",
        ] {
            cu_emit(nm, "SKIP", "not-sm12x-blackwell-or-no-tc-build");
            *counts.entry("SKIP").or_default() += 1;
        }
    } else {
        let (m, kd, n) = (64usize, 128usize, 16usize); // K % 64 == 0 (and % 32)
        // NVFP4 weight: quantize random f32 → valid per-16 E4M3 blocks.
        let wsrc = gen_x(m * kd, 77);
        let mut wnv = vec![0u8; m * (kd / 16) * 9];
        k::nvfp4::quantize_matrix(m, kd, &wsrc, &mut wnv);
        cu_tc_gemm_probe(
            &stream, "gemm:nvfp4_tc(W4A4)", TcGemm::Nvfp4, &wnv,
            k::nvfp4::matvec_nvfp4_w_f32_a, m, kd, n, &mut counts,
        );
        // TMA-staged NVFP4 (Phase 4): same weight/ref, cp.async.bulk staging.
        cu_tc_gemm_probe(
            &stream, "gemm:nvfp4_tc_tma(W4A4)", TcGemm::Nvfp4Tma, &wnv,
            k::nvfp4::matvec_nvfp4_w_f32_a, m, kd, n, &mut counts,
        );
        // MXFP4/6/8 weights: reuse the harness's valid-byte synth (E8M0 blocks).
        let wmx4 = gen_quant_bytes(LAYOUTS.iter().find(|l| l.name == "mxfp4").unwrap(), m, kd, 0xC0FFEE);
        cu_tc_gemm_probe(&stream, "gemm:mxfp4_tc(W4A4)", TcGemm::Mxfp4, &wmx4, k::mxfp::matvec_mxfp4_w_f32_a, m, kd, n, &mut counts);
        let wmx8 = gen_quant_bytes(LAYOUTS.iter().find(|l| l.name == "mxfp8").unwrap(), m, kd, 0xBEEF11);
        cu_tc_gemm_probe(&stream, "gemm:mxfp8_tc(W8A8)", TcGemm::Mxfp8, &wmx8, k::mxfp::matvec_mxfp8_w_f32_a, m, kd, n, &mut counts);
        let wmx6 = gen_quant_bytes(LAYOUTS.iter().find(|l| l.name == "mxfp6").unwrap(), m, kd, 0xBEEF22);
        cu_tc_gemm_probe(&stream, "gemm:mxfp6_tc(W6A6)", TcGemm::Mxfp6, &wmx6, k::mxfp::matvec_mxfp6_w_f32_a, m, kd, n, &mut counts);

        // 2:4 structured-sparse FP8 (Phase 5). Dense E4M3 weight; pruned
        // on-device. Graded vs the 2:4-pruned dense reference (loose: the GPU
        // path also E4M3-quantizes the activation). This probe is what reveals
        // whether the compressed-A / metadata layout is right on the Spark.
        let wsp: Vec<u8> = gen_x(m * kd, 97).iter().map(|&v| k::nvfp4::f32_to_e4m3(v)).collect();
        let xsp = gen_x(n * kd, 131);
        let cpu_sp = sp24_ref(&wsp, &xsp, m, kd, n);
        if let (Some(wb), Some(xb), Some(mut ob)) = (
            ck::CudaDeviceBuffer::from_host(&stream, &wsp),
            cu_upload_f32(&stream, &xsp),
            ck::CudaDeviceBuffer::alloc(&stream, n * m * 4),
        ) {
            // SAFETY: wb (M·K E4M3 bytes), xb (N·K f32), ob (N·M f32) live on stream.
            let res = unsafe {
                ck::gemm_fp8_sp24_f32(
                    &stream, wb.as_ptr(), xb.as_ptr() as *const f32,
                    ob.as_mut_ptr() as *mut f32, m, n, kd,
                )
            };
            match res {
                Err(e) => {
                    cu_emit("gemm:fp8_sp24(2:4)", "KERNEL_ERR", &format!("{e}"));
                    *counts.entry("KERNEL_ERR").or_default() += 1;
                }
                Ok(()) => cu_grade("gemm:fp8_sp24(2:4)", &cu_download_f32(&ob, n * m), &cpu_sp, 0.85, 0.60, &mut counts),
            }
        }
    }

    // ---- Hopper sm_90a FP8 wgmma GEMM (MXFP8 W8A8) ----
    // SKIP on non-Hopper (including Blackwell sm_12x, which is compute-major
    // 12, not 9). On a real GH200 this validates the wgmma kernel so
    // `hopper_tc_enabled()` auto-enables it — symmetric with the Blackwell FP4
    // path above. Same MXFP8 weight + W8A16 CPU reference as `gemm:mxfp8_tc`;
    // the only difference is the kernel dispatched.
    if ck::hopper_tc_available(0) {
        let (m, kd, n) = (64usize, 128usize, 16usize); // K % 32 == 0 (wgmma k-dim)
        let whop = gen_quant_bytes(
            LAYOUTS.iter().find(|l| l.name == "mxfp8").unwrap(),
            m,
            kd,
            0xBEEF33,
        );
        cu_tc_gemm_probe(
            &stream,
            "gemm:fp8_wgmma",
            TcGemm::Fp8Wgmma,
            &whop,
            k::mxfp::matvec_mxfp8_w_f32_a,
            m,
            kd,
            n,
            &mut counts,
        );
    } else {
        cu_emit("gemm:fp8_wgmma", "SKIP", "not-a-hopper-device");
        *counts.entry("SKIP").or_default() += 1;
    }

    // ---- Dense f32 matvec (host-copy reference wrapper) ----
    {
        let m = 64usize;
        let kd = 128usize;
        let w = gen_x(m * kd, 7);
        let x = gen_x(kd, 11);
        let mut cpu = vec![0f32; m];
        for r in 0..m {
            let mut acc = 0.0f32;
            for c in 0..kd {
                acc += w[r * kd + c] * x[c];
            }
            cpu[r] = acc;
        }
        let mut gpu = vec![0f32; m];
        match ck::matvec_f32(&w, &x, &mut gpu, m, kd) {
            Ok(()) => cu_grade("matvec:f32", &gpu, &cpu, 0.9999, 0.01, &mut counts),
            Err(e) => {
                cu_emit("matvec:f32", "KERNEL_ERR", &format!("{e}"));
                *counts.entry("KERNEL_ERR").or_default() += 1;
            }
        }
    }

    // ---- RMSNorm (host-copy reference wrapper) ----
    {
        let rows = 4usize;
        let d = 256usize;
        let x = gen_x(rows * d, 21);
        let wv = gen_x(d, 22);
        let eps = 1e-5f32;
        let mut cpu = vec![0f32; rows * d];
        for r in 0..rows {
            let mut ss = 0.0f32;
            for i in 0..d {
                ss += x[r * d + i] * x[r * d + i];
            }
            let inv = 1.0 / (ss / d as f32 + eps).sqrt();
            for i in 0..d {
                cpu[r * d + i] = x[r * d + i] * inv * wv[i];
            }
        }
        let mut gpu = vec![0f32; rows * d];
        match ck::rmsnorm_f32(&x, &wv, &mut gpu, rows, d, eps) {
            Ok(()) => cu_grade("rmsnorm:f32", &gpu, &cpu, 0.9999, 0.01, &mut counts),
            Err(e) => {
                cu_emit("rmsnorm:f32", "KERNEL_ERR", &format!("{e}"));
                *counts.entry("KERNEL_ERR").or_default() += 1;
            }
        }
    }

    // ---- P17: Gated-DeltaNet chunked-prefill recurrence (write-blind) ----
    // Same construction as the SYCL `ssm:delta_net` probe: head-major packed
    // inputs (q/k L2-normed per token), CPU chunked reference per head, graded
    // against the CUDA kernel's per-token output AND the final recurrent state
    // (concatenated). Validated on the pod; the dispatch gate reads this verdict.
    {
        const H: usize = 4;
        const T: usize = 40;
        const QK: usize = 16;
        const VD: usize = 24;
        const L0: usize = 8;
        let mut q = gen_x(H * T * QK, 71);
        let mut kk = gen_x(H * T * QK, 72);
        let v = gen_x(H * T * VD, 73);
        for row in 0..H * T {
            k::delta_net::l2norm_f32_inplace(&mut q[row * QK..(row + 1) * QK], 1e-6);
            k::delta_net::l2norm_f32_inplace(&mut kk[row * QK..(row + 1) * QK], 1e-6);
        }
        let g: Vec<f32> = gen_x(H * T, 74).iter().map(|x| -0.05 - 0.25 * x.abs()).collect();
        let beta: Vec<f32> = gen_x(H * T, 75).iter().map(|x| 0.1 + 0.8 * x.abs()).collect();
        let mut cpu_out = vec![0f32; H * T * VD];
        let mut cpu_state = vec![0f32; H * QK * VD];
        for h in 0..H {
            k::delta_net::delta_rule_prefill_chunked_with_chunk(
                &q[h * T * QK..(h + 1) * T * QK],
                &kk[h * T * QK..(h + 1) * T * QK],
                &v[h * T * VD..(h + 1) * T * VD],
                &g[h * T..(h + 1) * T],
                &beta[h * T..(h + 1) * T],
                &mut cpu_state[h * QK * VD..(h + 1) * QK * VD],
                &mut cpu_out[h * T * VD..(h + 1) * T * VD],
                T,
                QK,
                VD,
                L0,
            );
        }
        let mut gpu_out = vec![0f32; H * T * VD];
        let mut gpu_state = vec![0f32; H * QK * VD];
        match ck::delta_net_chunk_f32(
            &q, &kk, &v, &g, &beta, &mut gpu_state, &mut gpu_out, H, T, QK, VD, L0,
        ) {
            Ok(()) => {
                let mut gpu_all = gpu_out.clone();
                gpu_all.extend_from_slice(&gpu_state);
                let mut cpu_all = cpu_out.clone();
                cpu_all.extend_from_slice(&cpu_state);
                cu_grade("ssm:delta_net", &gpu_all, &cpu_all, 0.999, 0.02, &mut counts);
            }
            Err(e) => {
                cu_emit("ssm:delta_net", "KERNEL_ERR", &format!("{e}"));
                *counts.entry("KERNEL_ERR").or_default() += 1;
            }
        }
    }

    // ---- RoPE (device-resident) ----
    {
        let n_heads = 8usize;
        let head_dim = 64usize;
        let pos = 37usize;
        let qk = gen_x(n_heads * head_dim, 31);
        let inv_freq: Vec<f32> = (0..head_dim / 2)
            .map(|j| (10000f32).powf(-2.0 * j as f32 / head_dim as f32))
            .collect();
        let mut cpu = qk.clone();
        ref_rope(&mut cpu, n_heads, head_dim, pos, &inv_freq);
        if let (Some(mut qb), Some(fb)) = (
            cu_upload_f32(&stream, &qk),
            cu_upload_f32(&stream, &inv_freq),
        ) {
            // SAFETY: qb/fb are live device f32 buffers on `stream`.
            let res = unsafe {
                ck::rope_f32(
                    &stream,
                    qb.as_mut_ptr() as *mut f32,
                    n_heads,
                    head_dim,
                    pos,
                    fb.as_ptr() as *const f32,
                )
            };
            match res {
                Ok(()) => cu_grade(
                    "rope:f32",
                    &cu_download_f32(&qb, n_heads * head_dim),
                    &cpu,
                    0.9999,
                    0.01,
                    &mut counts,
                ),
                Err(e) => {
                    cu_emit("rope:f32", "KERNEL_ERR", &format!("{e}"));
                    *counts.entry("KERNEL_ERR").or_default() += 1;
                }
            }
        }
    }

    // ---- SwiGLU (device-resident) ----
    {
        let n = 4096usize;
        let x = gen_x(n, 41);
        let y = gen_x(n, 43);
        let cpu = ref_silu_mul(&x, &y);
        if let (Some(xb), Some(yb), Some(mut ob)) = (
            cu_upload_f32(&stream, &x),
            cu_upload_f32(&stream, &y),
            ck::CudaDeviceBuffer::alloc(&stream, n * 4),
        ) {
            // SAFETY: three live device f32 buffers of length n on `stream`.
            let res = unsafe {
                ck::silu_mul_f32(
                    &stream,
                    xb.as_ptr() as *const f32,
                    yb.as_ptr() as *const f32,
                    ob.as_mut_ptr() as *mut f32,
                    n,
                )
            };
            match res {
                Ok(()) => cu_grade(
                    "silu_mul:f32",
                    &cu_download_f32(&ob, n),
                    &cpu,
                    0.9999,
                    0.01,
                    &mut counts,
                ),
                Err(e) => {
                    cu_emit("silu_mul:f32", "KERNEL_ERR", &format!("{e}"));
                    *counts.entry("KERNEL_ERR").or_default() += 1;
                }
            }
        }
    }

    // ---- Embedding lookup (device-resident, device ids) ----
    {
        let vocab = 100usize;
        let d = 128usize;
        let table = gen_x(vocab * d, 51);
        let ids: Vec<i32> = [5i32, 0, 99, -1, 42].to_vec();
        let n_ids = ids.len();
        let mut cpu = vec![0f32; n_ids * d];
        for (i, &row) in ids.iter().enumerate() {
            if row >= 0 {
                cpu[i * d..(i + 1) * d]
                    .copy_from_slice(&table[row as usize * d..(row as usize + 1) * d]);
            }
        }
        let id_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const u8, n_ids * 4) };
        if let (Some(tb), Some(ib), Some(mut ob)) = (
            cu_upload_f32(&stream, &table),
            ck::CudaDeviceBuffer::from_host(&stream, id_bytes),
            ck::CudaDeviceBuffer::alloc(&stream, n_ids * d * 4),
        ) {
            // SAFETY: table [vocab*d] f32, ids [n_ids] i32, out [n_ids*d] f32,
            // all live device buffers on `stream`.
            let res = unsafe {
                ck::embedding_lookup_f32(
                    &stream,
                    tb.as_ptr() as *const f32,
                    ib.as_ptr() as *const std::os::raw::c_int,
                    ob.as_mut_ptr() as *mut f32,
                    n_ids,
                    d,
                )
            };
            match res {
                Ok(()) => cu_grade(
                    "embedding:f32",
                    &cu_download_f32(&ob, n_ids * d),
                    &cpu,
                    0.99999,
                    0.0001,
                    &mut counts,
                ),
                Err(e) => {
                    cu_emit("embedding:f32", "KERNEL_ERR", &format!("{e}"));
                    *counts.entry("KERNEL_ERR").or_default() += 1;
                }
            }
        }
    }

    // ---- FlashAttention decode + prefill (device-resident, f32) ----
    // Each kernel runs THREE cases folded into its single verdict: passthrough
    // (window=0/softcap=0 — must stay bit-identical to the pre-feature kernel), a
    // sliding-window case, and the Gemma2 attention-logit softcap (50.0 — the real
    // value). The CPU reference applies the SAME (window, softcap) via the fused
    // `gqa_attention_flash_{decode,prefill}_capped_windowed` (which at (0,0.0) is
    // the full attention, matching `ref_flash_*`), so a kernel that ignores either
    // uniform fails the verdict (fail-closed). WRITE-BLIND here (no NVIDIA GPU) —
    // validated on-device later via `--cuda-parity`.
    {
        let (n_heads, n_kv_heads, head_dim, max_ctx, kv_len) =
            (8usize, 2usize, 64usize, 128usize, 40usize);
        let q = gen_x(n_heads * head_dim, 61);
        let kc = gen_x(n_kv_heads * max_ctx * head_dim, 63);
        let vc = gen_x(n_kv_heads * max_ctx * head_dim, 67);
        // (window, softcap): kv_len=40 so window=16 genuinely trims to the last 16.
        let dec_cases: [(u32, f32); 3] = [(0, 0.0), (16, 0.0), (0, 50.0)];
        if let (Some(qb), Some(kb), Some(vb), Some(mut ob)) = (
            cu_upload_f32(&stream, &q),
            cu_upload_f32(&stream, &kc),
            cu_upload_f32(&stream, &vc),
            ck::CudaDeviceBuffer::alloc(&stream, n_heads * head_dim * 4),
        ) {
            let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
            let mut err: Option<String> = None;
            for (window, softcap) in dec_cases {
                let mut cpu = vec![0f32; n_heads * head_dim];
                k::gqa_attention_flash_decode_capped_windowed(
                    &q, &kc, &vc, &mut cpu,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                    window as usize, softcap,
                );
                // SAFETY: q/out [n_heads*head_dim], k/v [n_kv_heads*max_ctx*head_dim] device f32.
                let res = unsafe {
                    ck::flash_attn_decode_f32(
                        &stream, qb.as_ptr() as *const f32, kb.as_ptr() as *const f32,
                        vb.as_ptr() as *const f32, ob.as_mut_ptr() as *mut f32,
                        n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                        window as i32, softcap,
                    )
                };
                if let Err(e) = res {
                    err = Some(format!("{e}"));
                    break;
                }
                let (cos, max_rel) = compare(&cu_download_f32(&ob, n_heads * head_dim), &cpu);
                worst_cos = worst_cos.min(cos);
                worst_rel = worst_rel.max(max_rel);
            }
            match err {
                Some(e) => {
                    cu_emit("attn:decode", "KERNEL_ERR", &e);
                    *counts.entry("KERNEL_ERR").or_default() += 1;
                }
                None => {
                    let key: &'static str =
                        if worst_cos > 0.999 && worst_rel < 0.02 { "OK" } else { "MISCOMPUTE" };
                    *counts.entry(key).or_default() += 1;
                    cu_emit(
                        "attn:decode",
                        key,
                        &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa16+cap50)"),
                    );
                }
            }
        }

        let (kv_base, n_new) = (10usize, 6usize);
        let qp = gen_x(n_new * n_heads * head_dim, 71);
        // Prefill queries reach abs pos 15, so window=16 would not trim — use
        // window=4 to genuinely trim (query 15 attends 12..15).
        let pre_cases: [(u32, f32); 3] = [(0, 0.0), (4, 0.0), (0, 50.0)];
        if let (Some(qb), Some(kb), Some(vb), Some(mut ob)) = (
            cu_upload_f32(&stream, &qp),
            cu_upload_f32(&stream, &kc),
            cu_upload_f32(&stream, &vc),
            ck::CudaDeviceBuffer::alloc(&stream, n_new * n_heads * head_dim * 4),
        ) {
            let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
            let mut err: Option<String> = None;
            for (window, softcap) in pre_cases {
                let mut cpu_p = vec![0f32; n_new * n_heads * head_dim];
                k::gqa_attention_flash_prefill_capped_windowed(
                    &qp, &kc, &vc, &mut cpu_p,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
                    window as usize, softcap,
                );
                // SAFETY: q/out [n_new*n_heads*head_dim], k/v caches device f32.
                let res = unsafe {
                    ck::flash_attn_prefill_f32(
                        &stream, qb.as_ptr() as *const f32, kb.as_ptr() as *const f32,
                        vb.as_ptr() as *const f32, ob.as_mut_ptr() as *mut f32,
                        n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
                        window as i32, softcap,
                    )
                };
                if let Err(e) = res {
                    err = Some(format!("{e}"));
                    break;
                }
                let (cos, max_rel) =
                    compare(&cu_download_f32(&ob, n_new * n_heads * head_dim), &cpu_p);
                worst_cos = worst_cos.min(cos);
                worst_rel = worst_rel.max(max_rel);
            }
            match err {
                Some(e) => {
                    cu_emit("attn:prefill", "KERNEL_ERR", &e);
                    *counts.entry("KERNEL_ERR").or_default() += 1;
                }
                None => {
                    let key: &'static str =
                        if worst_cos > 0.999 && worst_rel < 0.02 { "OK" } else { "MISCOMPUTE" };
                    *counts.entry(key).or_default() += 1;
                    cu_emit(
                        "attn:prefill",
                        key,
                        &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa4+cap50)"),
                    );
                }
            }
        }

        // Timing: the warp-cooperative FA-2 prefill kernel at a REALISTIC prefill
        // shape (the parity tile above is 6 tokens — too small to time past launch
        // overhead). Reports the kernel's on-device cost + its speedup over the
        // scalar CPU reference — the honest available comparison, since the naive
        // one-thread-per-query kernel this FA-2 replaced was removed (no in-tree
        // A/B). Human diagnostic only; emits no verdict and doesn't touch `counts`.
        {
            let (th, tkv, thd, tmax, tnew) = (32usize, 4usize, 128usize, 512usize, 256usize);
            let kv_base = 0usize;
            let q = gen_x(tnew * th * thd, 71);
            let kc = gen_x(tkv * tmax * thd, 73);
            let vc = gen_x(tkv * tmax * thd, 79);
            if let (Some(qb), Some(kb), Some(vb), Some(mut ob)) = (
                cu_upload_f32(&stream, &q),
                cu_upload_f32(&stream, &kc),
                cu_upload_f32(&stream, &vc),
                ck::CudaDeviceBuffer::alloc(&stream, tnew * th * thd * 4),
            ) {
                // SAFETY: q/out [tnew*th*thd], k/v [tkv*tmax*thd] device f32;
                // kv_base + tnew = tnew <= tmax holds.
                let launch = |ob: &mut ck::CudaDeviceBuffer| unsafe {
                    ck::flash_attn_prefill_f32(
                        &stream,
                        qb.as_ptr() as *const f32,
                        kb.as_ptr() as *const f32,
                        vb.as_ptr() as *const f32,
                        ob.as_mut_ptr() as *mut f32,
                        th,
                        tkv,
                        thd,
                        tmax,
                        kv_base,
                        tnew,
                        0,   // timing-only: full causal (no SWA window)
                        0.0, // no softcap
                    )
                };
                if launch(&mut ob).is_ok() {
                    // Warmup done (the launch above); a download forces completion.
                    let _ = cu_download_f32(&ob, tnew * th * thd);
                    const ITERS: u32 = 20;
                    let t0 = Instant::now();
                    for _ in 0..ITERS {
                        let _ = launch(&mut ob);
                    }
                    // One D2H after the N serialized stream launches syncs them all.
                    let _ = cu_download_f32(&ob, tnew * th * thd);
                    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
                    let c0 = Instant::now();
                    let _ = ref_flash_prefill(
                        &q, &kc, &vc, th, tkv, thd, tmax, kv_base, tnew,
                    );
                    let cpu_ms = c0.elapsed().as_secs_f64() * 1000.0;
                    let speedup = if gpu_ms > 0.0 { cpu_ms / gpu_ms } else { 0.0 };
                    println!(
                        "  attn:prefill TIMING [{th}h/{tkv}kv/{thd}d, {tnew} new tokens]: \
                         GPU {gpu_ms:.3} ms/launch vs CPU ref {cpu_ms:.1} ms (~{speedup:.0}x)"
                    );
                }
            }
        }
    }

    // ---- Quantized-KV FlashAttention decode + prefill (MXFP4/6/8, NVFP4) ----
    // Same geometry as the f32 flash section above; K/V are quantized to each
    // format's block layout on the CPU (byte-identical to the engine's KV
    // writer), then the native quant-KV kernels dequantize them on the fly.
    // Graded against a SAME-QUANT reference (the identical packed K/V
    // dequantized on the CPU, run through the f32 flash reference), so the probe
    // measures the KERNEL's correctness, not the KV quantization loss — a
    // correct kernel then passes the per-format tolerance with margin. (Was
    // graded vs the full-precision f32 K/V, which conflated the two and flagged
    // the correct kernels as MISCOMPUTE on their relative-error metric.)
    {
        let (n_heads, n_kv_heads, head_dim, max_ctx, kv_len) =
            (8usize, 2usize, 64usize, 128usize, 40usize);
        let (kv_base, n_new) = (10usize, 6usize);
        let q = gen_x(n_heads * head_dim, 61);
        let kc = gen_x(n_kv_heads * max_ctx * head_dim, 63);
        let vc = gen_x(n_kv_heads * max_ctx * head_dim, 67);
        let qp = gen_x(n_new * n_heads * head_dim, 71);
        // Cover both readers from one packed cache: decode reads kv_len rows,
        // prefill reads kv_base+n_new rows.
        let upto = kv_len.max(kv_base + n_new);
        for fmt in QUANT_KV_FORMATS {
            let dname = format!("attn:decode_{}", fmt.name);
            let pname = format!("attn:prefill_{}", fmt.name);
            if head_dim % fmt.block_elems != 0 {
                cu_emit(&dname, "SKIP", "head-dim-not-block-aligned");
                cu_emit(&pname, "SKIP", "head-dim-not-block-aligned");
                *counts.entry("SKIP").or_default() += 2;
                continue;
            }
            let kp = quantize_kv_cache(fmt, &kc, n_kv_heads, head_dim, max_ctx, upto);
            let vp = quantize_kv_cache(fmt, &vc, n_kv_heads, head_dim, max_ctx, upto);

            // Same-quant reference: dequantize the identical packed K/V and run
            // the f32 flash reference on THAT (per-format — each format
            // round-trips differently). F1: the `_capped_windowed` reference
            // applies the per-case (window, softcap) just like the f32 flash
            // section above, so the probe actually exercises the kernel's SWA +
            // Gemma2 soft-cap (fail-closed: a kernel that ignores either uniform
            // misses the combined verdict). At (0,0.0) it equals `ref_flash_*`.
            let kc_dq = dequant_kv_cache(fmt, &kp, n_kv_heads, head_dim, max_ctx, upto);
            let vc_dq = dequant_kv_cache(fmt, &vp, n_kv_heads, head_dim, max_ctx, upto);

            // decode — fold passthrough + sliding-window (kv_len=40 ⇒ window=16
            // trims) + Gemma2 soft-cap into the one `attn:decode_<fmt>` verdict.
            if let (Some(qb), Some(kb), Some(vb), Some(mut ob)) = (
                cu_upload_f32(&stream, &q),
                ck::CudaDeviceBuffer::from_host(&stream, &kp),
                ck::CudaDeviceBuffer::from_host(&stream, &vp),
                ck::CudaDeviceBuffer::alloc(&stream, n_heads * head_dim * 4),
            ) {
                let dec_cases: [(u32, f32); 3] = [(0, 0.0), (16, 0.0), (0, 50.0)];
                let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
                let mut err: Option<String> = None;
                for (window, softcap) in dec_cases {
                    let mut cpu_dec = vec![0f32; n_heads * head_dim];
                    k::gqa_attention_flash_decode_capped_windowed(
                        &q, &kc_dq, &vc_dq, &mut cpu_dec,
                        n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                        window as usize, softcap,
                    );
                    // SAFETY: q/out are F32 device buffers [n_heads*head_dim];
                    // k/v are packed device buffers [n_kv_heads*max_ctx*
                    // bytes_per_row]; the wrappers synchronize before returning.
                    let res = unsafe {
                        let q = qb.as_ptr() as *const f32;
                        let k = kb.as_ptr();
                        let v = vb.as_ptr();
                        let o = ob.as_mut_ptr() as *mut f32;
                        match fmt.name {
                            "mxfp4" => ck::flash_attn_decode_mxfp4(
                                &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                                window as i32, softcap,
                            ),
                            "mxfp6" => ck::flash_attn_decode_mxfp6(
                                &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                                window as i32, softcap,
                            ),
                            "mxfp8" => ck::flash_attn_decode_mxfp8(
                                &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                                window as i32, softcap,
                            ),
                            "nvfp4" => ck::flash_attn_decode_nvfp4(
                                &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                                window as i32, softcap,
                            ),
                            "q4_0" => ck::flash_attn_decode_q4_0(
                                &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                                window as i32, softcap,
                            ),
                            _ => unreachable!(),
                        }
                    };
                    if let Err(e) = res {
                        err = Some(format!("{e}"));
                        break;
                    }
                    let (cos, max_rel) = compare(&cu_download_f32(&ob, n_heads * head_dim), &cpu_dec);
                    worst_cos = worst_cos.min(cos);
                    worst_rel = worst_rel.max(max_rel);
                }
                match err {
                    Some(e) => {
                        cu_emit(&dname, "KERNEL_ERR", &format!("{e}"));
                        *counts.entry("KERNEL_ERR").or_default() += 1;
                    }
                    None => {
                        let key: &'static str = if worst_cos > fmt.cos_min && worst_rel < fmt.rel_max {
                            "OK"
                        } else {
                            "MISCOMPUTE"
                        };
                        *counts.entry(key).or_default() += 1;
                        cu_emit(
                            &dname,
                            key,
                            &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa16+cap50)"),
                        );
                    }
                }
            }

            // prefill — queries reach abs pos 15, so window=4 genuinely trims
            // (query 15 attends 12..15). Same three-case fold.
            if let (Some(qb), Some(kb), Some(vb), Some(mut ob)) = (
                cu_upload_f32(&stream, &qp),
                ck::CudaDeviceBuffer::from_host(&stream, &kp),
                ck::CudaDeviceBuffer::from_host(&stream, &vp),
                ck::CudaDeviceBuffer::alloc(&stream, n_new * n_heads * head_dim * 4),
            ) {
                let pre_cases: [(u32, f32); 3] = [(0, 0.0), (4, 0.0), (0, 50.0)];
                let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
                let mut err: Option<String> = None;
                for (window, softcap) in pre_cases {
                    let mut cpu_pre = vec![0f32; n_new * n_heads * head_dim];
                    k::gqa_attention_flash_prefill_capped_windowed(
                        &qp, &kc_dq, &vc_dq, &mut cpu_pre,
                        n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
                        window as usize, softcap,
                    );
                    // SAFETY: q/out F32 [n_new*n_heads*head_dim]; k/v packed as above.
                    let res = unsafe {
                        let q = qb.as_ptr() as *const f32;
                        let k = kb.as_ptr();
                        let v = vb.as_ptr();
                        let o = ob.as_mut_ptr() as *mut f32;
                        match fmt.name {
                            "mxfp4" => ck::flash_attn_prefill_mxfp4(
                                &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base,
                                n_new, window as i32, softcap,
                            ),
                            "mxfp6" => ck::flash_attn_prefill_mxfp6(
                                &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base,
                                n_new, window as i32, softcap,
                            ),
                            "mxfp8" => ck::flash_attn_prefill_mxfp8(
                                &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base,
                                n_new, window as i32, softcap,
                            ),
                            "nvfp4" => ck::flash_attn_prefill_nvfp4(
                                &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base,
                                n_new, window as i32, softcap,
                            ),
                            "q4_0" => ck::flash_attn_prefill_q4_0(
                                &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base,
                                n_new, window as i32, softcap,
                            ),
                            _ => unreachable!(),
                        }
                    };
                    if let Err(e) = res {
                        err = Some(format!("{e}"));
                        break;
                    }
                    let (cos, max_rel) =
                        compare(&cu_download_f32(&ob, n_new * n_heads * head_dim), &cpu_pre);
                    worst_cos = worst_cos.min(cos);
                    worst_rel = worst_rel.max(max_rel);
                }
                match err {
                    Some(e) => {
                        cu_emit(&pname, "KERNEL_ERR", &format!("{e}"));
                        *counts.entry("KERNEL_ERR").or_default() += 1;
                    }
                    None => {
                        let key: &'static str = if worst_cos > fmt.cos_min && worst_rel < fmt.rel_max {
                            "OK"
                        } else {
                            "MISCOMPUTE"
                        };
                        *counts.entry(key).or_default() += 1;
                        cu_emit(
                            &pname,
                            key,
                            &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa4+cap50)"),
                        );
                    }
                }
            }
        }
    }

    // ---- Q8_0 KV FlashAttention decode + prefill (per-row-scale layout) ----
    // Q8_0 KV is a plain i8 slab + a SEPARATE per-row f32 scale buffer (not a
    // QuantKvFormat block), so it takes its own quantizer + the kernel's k/v
    // scale args. Same-quant reference (dequant the identical slab+scales, run
    // f32 flash) → bit-close, so a tight 8-bit tolerance.
    {
        let (n_heads, n_kv_heads, head_dim, max_ctx, kv_len) = (8usize, 2usize, 64usize, 128usize, 40usize);
        let (kv_base, n_new) = (10usize, 6usize);
        let q = gen_x(n_heads * head_dim, 61);
        let kc = gen_x(n_kv_heads * max_ctx * head_dim, 63);
        let vc = gen_x(n_kv_heads * max_ctx * head_dim, 67);
        let qp = gen_x(n_new * n_heads * head_dim, 71);
        let upto = kv_len.max(kv_base + n_new);
        let (kslab, ksc) = quantize_kv_q8_0(&kc, n_kv_heads, head_dim, max_ctx, upto);
        let (vslab, vsc) = quantize_kv_q8_0(&vc, n_kv_heads, head_dim, max_ctx, upto);
        let kdq = dequant_kv_q8_0(&kslab, &ksc, n_kv_heads, head_dim, max_ctx, upto);
        let vdq = dequant_kv_q8_0(&vslab, &vsc, n_kv_heads, head_dim, max_ctx, upto);
        // decode — fold the three (window, softcap) cases into one verdict, with
        // the `_capped_windowed` f32 reference on the dequantized slab (at (0,0.0)
        // it equals `ref_flash_decode`). kv_len=40 ⇒ window=16 trims.
        if let (Some(qb), Some(kb), Some(vb), Some(ksb), Some(vsb), Some(mut ob)) = (
            cu_upload_f32(&stream, &q),
            ck::CudaDeviceBuffer::from_host(&stream, i8_as_bytes(&kslab)),
            ck::CudaDeviceBuffer::from_host(&stream, i8_as_bytes(&vslab)),
            cu_upload_f32(&stream, &ksc),
            cu_upload_f32(&stream, &vsc),
            ck::CudaDeviceBuffer::alloc(&stream, n_heads * head_dim * 4),
        ) {
            let dec_cases: [(u32, f32); 3] = [(0, 0.0), (16, 0.0), (0, 50.0)];
            let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
            let mut err: Option<String> = None;
            for (window, softcap) in dec_cases {
                let mut cpu_dec = vec![0f32; n_heads * head_dim];
                k::gqa_attention_flash_decode_capped_windowed(
                    &q, &kdq, &vdq, &mut cpu_dec,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                    window as usize, softcap,
                );
                // SAFETY: q/out F32; k/v i8 slabs; k/v scales F32 — all live device
                // buffers on `stream`; the wrapper synchronizes before returning.
                let res = unsafe {
                    ck::flash_attn_decode_q8_0(
                        &stream,
                        qb.as_ptr() as *const f32,
                        kb.as_ptr(),
                        vb.as_ptr(),
                        ksb.as_ptr() as *const f32,
                        vsb.as_ptr() as *const f32,
                        ob.as_mut_ptr() as *mut f32,
                        n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                        window as i32, softcap,
                    )
                };
                if let Err(e) = res {
                    err = Some(format!("{e}"));
                    break;
                }
                let (cos, max_rel) = compare(&cu_download_f32(&ob, n_heads * head_dim), &cpu_dec);
                worst_cos = worst_cos.min(cos);
                worst_rel = worst_rel.max(max_rel);
            }
            match err {
                Some(e) => {
                    cu_emit("attn:decode_q8_0", "KERNEL_ERR", &format!("{e}"));
                    *counts.entry("KERNEL_ERR").or_default() += 1;
                }
                None => {
                    let key: &'static str =
                        if worst_cos > 0.999 && worst_rel < 0.05 { "OK" } else { "MISCOMPUTE" };
                    *counts.entry(key).or_default() += 1;
                    cu_emit(
                        "attn:decode_q8_0",
                        key,
                        &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa16+cap50)"),
                    );
                }
            }
        }
        // prefill — queries reach abs pos 15, so window=4 genuinely trims.
        if let (Some(qb), Some(kb), Some(vb), Some(ksb), Some(vsb), Some(mut ob)) = (
            cu_upload_f32(&stream, &qp),
            ck::CudaDeviceBuffer::from_host(&stream, i8_as_bytes(&kslab)),
            ck::CudaDeviceBuffer::from_host(&stream, i8_as_bytes(&vslab)),
            cu_upload_f32(&stream, &ksc),
            cu_upload_f32(&stream, &vsc),
            ck::CudaDeviceBuffer::alloc(&stream, n_new * n_heads * head_dim * 4),
        ) {
            let pre_cases: [(u32, f32); 3] = [(0, 0.0), (4, 0.0), (0, 50.0)];
            let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
            let mut err: Option<String> = None;
            for (window, softcap) in pre_cases {
                let mut cpu_pre = vec![0f32; n_new * n_heads * head_dim];
                k::gqa_attention_flash_prefill_capped_windowed(
                    &qp, &kdq, &vdq, &mut cpu_pre,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
                    window as usize, softcap,
                );
                // SAFETY: as the decode arm, with q/out sized for n_new queries.
                let res = unsafe {
                    ck::flash_attn_prefill_q8_0(
                        &stream,
                        qb.as_ptr() as *const f32,
                        kb.as_ptr(),
                        vb.as_ptr(),
                        ksb.as_ptr() as *const f32,
                        vsb.as_ptr() as *const f32,
                        ob.as_mut_ptr() as *mut f32,
                        n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
                        window as i32, softcap,
                    )
                };
                if let Err(e) = res {
                    err = Some(format!("{e}"));
                    break;
                }
                let (cos, max_rel) =
                    compare(&cu_download_f32(&ob, n_new * n_heads * head_dim), &cpu_pre);
                worst_cos = worst_cos.min(cos);
                worst_rel = worst_rel.max(max_rel);
            }
            match err {
                Some(e) => {
                    cu_emit("attn:prefill_q8_0", "KERNEL_ERR", &format!("{e}"));
                    *counts.entry("KERNEL_ERR").or_default() += 1;
                }
                None => {
                    let key: &'static str =
                        if worst_cos > 0.999 && worst_rel < 0.05 { "OK" } else { "MISCOMPUTE" };
                    *counts.entry(key).or_default() += 1;
                    cu_emit(
                        "attn:prefill_q8_0",
                        key,
                        &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa4+cap50)"),
                    );
                }
            }
        }
    }

    // ---- TQ (TurboQuant) KV FlashAttention decode + prefill ----
    // WHT-packed bits-per-element slab + per-row f32 scales + a runtime `bits`
    // arg. Same-quant reference at TQ_PROBE_BITS (parity is bits-independent).
    {
        let (n_heads, n_kv_heads, head_dim, max_ctx, kv_len) = (8usize, 2usize, 64usize, 128usize, 40usize);
        let (kv_base, n_new) = (10usize, 6usize);
        let bits = TQ_PROBE_BITS;
        let q = gen_x(n_heads * head_dim, 61);
        let kc = gen_x(n_kv_heads * max_ctx * head_dim, 63);
        let vc = gen_x(n_kv_heads * max_ctx * head_dim, 67);
        let qp = gen_x(n_new * n_heads * head_dim, 71);
        let upto = kv_len.max(kv_base + n_new);
        let (kslab, ksc) = quantize_kv_tq(&kc, bits, n_kv_heads, head_dim, max_ctx, upto);
        let (vslab, vsc) = quantize_kv_tq(&vc, bits, n_kv_heads, head_dim, max_ctx, upto);
        let kdq = dequant_kv_tq(&kslab, &ksc, bits, n_kv_heads, head_dim, max_ctx, upto);
        let vdq = dequant_kv_tq(&vslab, &vsc, bits, n_kv_heads, head_dim, max_ctx, upto);
        // decode — fold the three (window, softcap) cases into one verdict, with
        // the `_capped_windowed` f32 reference on the dequantized slab (at (0,0.0)
        // it equals `ref_flash_decode`). kv_len=40 ⇒ window=16 trims.
        if let (Some(qb), Some(kb), Some(vb), Some(ksb), Some(vsb), Some(mut ob)) = (
            cu_upload_f32(&stream, &q),
            ck::CudaDeviceBuffer::from_host(&stream, &kslab),
            ck::CudaDeviceBuffer::from_host(&stream, &vslab),
            cu_upload_f32(&stream, &ksc),
            cu_upload_f32(&stream, &vsc),
            ck::CudaDeviceBuffer::alloc(&stream, n_heads * head_dim * 4),
        ) {
            let dec_cases: [(u32, f32); 3] = [(0, 0.0), (16, 0.0), (0, 50.0)];
            let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
            let mut err: Option<String> = None;
            for (window, softcap) in dec_cases {
                let mut cpu_dec = vec![0f32; n_heads * head_dim];
                k::gqa_attention_flash_decode_capped_windowed(
                    &q, &kdq, &vdq, &mut cpu_dec,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                    window as usize, softcap,
                );
                // SAFETY: q/out F32; k/v packed TQ slabs; k/v scales F32 — live
                // device buffers on `stream`.
                let res = unsafe {
                    ck::flash_attn_decode_tq(
                        &stream,
                        qb.as_ptr() as *const f32,
                        kb.as_ptr(),
                        vb.as_ptr(),
                        ksb.as_ptr() as *const f32,
                        vsb.as_ptr() as *const f32,
                        bits as u32,
                        ob.as_mut_ptr() as *mut f32,
                        n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                        window as i32, softcap,
                    )
                };
                if let Err(e) = res {
                    err = Some(format!("{e}"));
                    break;
                }
                let (cos, max_rel) = compare(&cu_download_f32(&ob, n_heads * head_dim), &cpu_dec);
                worst_cos = worst_cos.min(cos);
                worst_rel = worst_rel.max(max_rel);
            }
            match err {
                Some(e) => {
                    cu_emit("attn:decode_tq", "KERNEL_ERR", &format!("{e}"));
                    *counts.entry("KERNEL_ERR").or_default() += 1;
                }
                None => {
                    let key: &'static str =
                        if worst_cos > 0.999 && worst_rel < 0.05 { "OK" } else { "MISCOMPUTE" };
                    *counts.entry(key).or_default() += 1;
                    cu_emit(
                        "attn:decode_tq",
                        key,
                        &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa16+cap50)"),
                    );
                }
            }
        }
        // prefill — queries reach abs pos 15, so window=4 genuinely trims.
        if let (Some(qb), Some(kb), Some(vb), Some(ksb), Some(vsb), Some(mut ob)) = (
            cu_upload_f32(&stream, &qp),
            ck::CudaDeviceBuffer::from_host(&stream, &kslab),
            ck::CudaDeviceBuffer::from_host(&stream, &vslab),
            cu_upload_f32(&stream, &ksc),
            cu_upload_f32(&stream, &vsc),
            ck::CudaDeviceBuffer::alloc(&stream, n_new * n_heads * head_dim * 4),
        ) {
            let pre_cases: [(u32, f32); 3] = [(0, 0.0), (4, 0.0), (0, 50.0)];
            let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
            let mut err: Option<String> = None;
            for (window, softcap) in pre_cases {
                let mut cpu_pre = vec![0f32; n_new * n_heads * head_dim];
                k::gqa_attention_flash_prefill_capped_windowed(
                    &qp, &kdq, &vdq, &mut cpu_pre,
                    n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
                    window as usize, softcap,
                );
                // SAFETY: as the decode arm, q/out sized for n_new queries.
                let res = unsafe {
                    ck::flash_attn_prefill_tq(
                        &stream,
                        qb.as_ptr() as *const f32,
                        kb.as_ptr(),
                        vb.as_ptr(),
                        ksb.as_ptr() as *const f32,
                        vsb.as_ptr() as *const f32,
                        bits as u32,
                        ob.as_mut_ptr() as *mut f32,
                        n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
                        window as i32, softcap,
                    )
                };
                if let Err(e) = res {
                    err = Some(format!("{e}"));
                    break;
                }
                let (cos, max_rel) =
                    compare(&cu_download_f32(&ob, n_new * n_heads * head_dim), &cpu_pre);
                worst_cos = worst_cos.min(cos);
                worst_rel = worst_rel.max(max_rel);
            }
            match err {
                Some(e) => {
                    cu_emit("attn:prefill_tq", "KERNEL_ERR", &format!("{e}"));
                    *counts.entry("KERNEL_ERR").or_default() += 1;
                }
                None => {
                    let key: &'static str =
                        if worst_cos > 0.999 && worst_rel < 0.05 { "OK" } else { "MISCOMPUTE" };
                    *counts.entry(key).or_default() += 1;
                    cu_emit(
                        "attn:prefill_tq",
                        key,
                        &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa4+cap50)"),
                    );
                }
            }
        }
    }

    // ---- Greedy argmax ----
    {
        let vocab = 4096usize;
        let mut logits = gen_x(vocab, 81);
        logits[1234] = 12.5; // unique max
        let cpu_idx = 1234i32;
        if let (Some(lb), Some(mut ob)) = (
            cu_upload_f32(&stream, &logits),
            ck::CudaDeviceBuffer::alloc(&stream, 4),
        ) {
            // SAFETY: logits [vocab] f32, out_idx [1] i32 device buffers.
            let res = unsafe {
                ck::argmax_f32(
                    &stream,
                    lb.as_ptr() as *const f32,
                    vocab,
                    ob.as_mut_ptr() as *mut std::os::raw::c_int,
                )
            };
            match res {
                Ok(()) => {
                    let mut idx = [0i32; 1];
                    let ib: &mut [u8] =
                        unsafe { std::slice::from_raw_parts_mut(idx.as_mut_ptr() as *mut u8, 4) };
                    let _ = ob.copy_to_host(ib);
                    let ok = idx[0] == cpu_idx;
                    let key: &'static str = if ok { "OK" } else { "MISCOMPUTE" };
                    *counts.entry(key).or_default() += 1;
                    cu_emit(
                        "argmax:f32",
                        key,
                        &format!("gpu_idx={} cpu_idx={cpu_idx}", idx[0]),
                    );
                }
                Err(e) => {
                    cu_emit("argmax:f32", "KERNEL_ERR", &format!("{e}"));
                    *counts.entry("KERNEL_ERR").or_default() += 1;
                }
            }
        }
    }

    // ---- Qwen3 per-head q/k RMSNorm (device, #2 activation residency) ----
    // In-place RMSNorm of each head's head_dim slice with a shared weight;
    // graded vs the CPU `rmsnorm_f32_row` per head (parallel reduce → not
    // bit-exact, cos/max_rel gated).
    {
        let name = "attn:qk_head_norm";
        const NH: usize = 4; // heads
        const HD: usize = 128; // head_dim
        let x = gen_x(NH * HD, 0x9E3F);
        let w = gen_x(HD, 0x0000_517A);
        let eps = 1e-6f32;
        let mut cpu = x.clone();
        let mut tmp = vec![0f32; HD];
        for h in 0..NH {
            k::rmsnorm_f32_row(&x[h * HD..(h + 1) * HD], &w, &mut tmp, eps);
            cpu[h * HD..(h + 1) * HD].copy_from_slice(&tmp);
        }
        match (cu_upload_f32(&stream, &x), cu_upload_f32(&stream, &w)) {
            (Some(mut buf), Some(wb)) => {
                // SAFETY: buf (NH*HD f32) + wb (HD f32) are live device buffers
                // on `stream`; qk_head_norm writes `buf` in place + synchronizes.
                let ok = unsafe {
                    ck::qk_head_norm_f32(
                        &stream,
                        buf.as_mut_ptr() as *mut f32,
                        wb.as_ptr() as *const f32,
                        NH,
                        HD,
                        eps,
                    )
                }
                .is_ok()
                    && ck::consume_error_count() == 0;
                if ok {
                    cu_grade(name, &cu_download_f32(&buf, NH * HD), &cpu, 0.999, 0.02, &mut counts);
                } else {
                    cu_emit(name, "KERNEL_ERR", "kernel-failed");
                    *counts.entry("KERNEL_ERR").or_default() += 1;
                }
            }
            _ => {
                cu_emit(name, "KERNEL_ERR", "device-alloc-failed");
                *counts.entry("KERNEL_ERR").or_default() += 1;
            }
        }
    }

    // ---- Grouped routed-expert FFN (MoE tiered-expert decode fast path) ----
    // Builds NP experts of gate/up/down for each warp-cooperative quant, runs the
    // whole top-k FFN on-device in one pass, and grades vs a CPU reference over
    // the IDENTICAL packed weights (per-expert matvec → silu_mul → down matvec →
    // weighted sum). Not bit-exact — the GPU accumulates the weighted sum via
    // atomicAdd (FMA reassociation only) — so cos/max_rel gated. The probe name
    // MUST match `accel::moe_ffn_grouped_verdict_name` so the dispatch gate reads
    // the right verdict.
    for (name, dtype, kind) in [
        ("moe:ffn_grouped_q4k", "q4_k", ck::CudaPackedKind::Q4_K),
        ("moe:ffn_grouped_q8_0", "q8_0", ck::CudaPackedKind::Q8_0),
        ("moe:ffn_grouped_q6_k", "q6_k", ck::CudaPackedKind::Q6_K),
    ] {
        const D: usize = 512; // d_model == d_ff; divisible by both 256 and 32
        const NP: usize = 4; // routed experts
        let layout = LAYOUTS.iter().find(|l| l.name == dtype).expect("layout");
        let cpu_mv = cpu_matvec_for(dtype);
        let x = gen_x(D, 0x00A1_1CE5);
        let weights: [f32; NP] = [0.9, 0.05, 0.03, 0.02];
        let mut wg: Vec<Vec<u8>> = Vec::with_capacity(NP);
        let mut wu: Vec<Vec<u8>> = Vec::with_capacity(NP);
        let mut wd: Vec<Vec<u8>> = Vec::with_capacity(NP);
        for e in 0..NP as u32 {
            wg.push(gen_quant_bytes(layout, D, D, 0x6A7E ^ (e * 101)));
            wu.push(gen_quant_bytes(layout, D, D, 0x0D97 ^ (e * 131)));
            wd.push(gen_quant_bytes(layout, D, D, 0xD074 ^ (e * 151)));
        }
        // CPU reference: Σ_e w_e · (Wdown_e · silu(Wgate_e·x) ⊙ (Wup_e·x)).
        let mut cpu = vec![0f32; D];
        let (mut g, mut u, mut ff, mut dn) =
            (vec![0f32; D], vec![0f32; D], vec![0f32; D], vec![0f32; D]);
        for e in 0..NP {
            cpu_mv(&wg[e], &x, &mut g, D, D);
            cpu_mv(&wu[e], &x, &mut u, D, D);
            k::silu_mul_f32(&g, &u, &mut ff);
            cpu_mv(&wd[e], &ff, &mut dn, D, D);
            for j in 0..D {
                cpu[j] += weights[e] * dn[j];
            }
        }
        // GPU: per-expert device-resident weights + the grouped entry on a fresh
        // cache. (All device-0 streams share the runtime primary context, so
        // these plain-device pointers are valid in the cache's stream — the real
        // path uses managed memory, which is cross-context either way.)
        let mut keep = Vec::with_capacity(3 * NP);
        let mut gp: Vec<*const core::ffi::c_void> = Vec::with_capacity(NP);
        let mut upp: Vec<*const core::ffi::c_void> = Vec::with_capacity(NP);
        let mut dp: Vec<*const core::ffi::c_void> = Vec::with_capacity(NP);
        let mut alloc_ok = true;
        for e in 0..NP {
            match (
                ck::CudaDeviceBuffer::from_host(&stream, &wg[e]),
                ck::CudaDeviceBuffer::from_host(&stream, &wu[e]),
                ck::CudaDeviceBuffer::from_host(&stream, &wd[e]),
            ) {
                (Some(a), Some(b), Some(c)) => {
                    gp.push(a.as_ptr());
                    upp.push(b.as_ptr());
                    dp.push(c.as_ptr());
                    keep.push(a);
                    keep.push(b);
                    keep.push(c);
                }
                _ => {
                    alloc_ok = false;
                    break;
                }
            }
        }
        if !alloc_ok {
            cu_emit(name, "KERNEL_ERR", "device-alloc-failed");
            *counts.entry("KERNEL_ERR").or_default() += 1;
        } else if let Some(mut cache) = ck::CudaMatvecCache::new(0, 1 << 20) {
            let mut out_gpu = vec![0f32; D];
            let wv = weights.to_vec();
            // SAFETY: every ptr is a live device buffer of `kind` held in `keep`
            // on device 0, valid for the call; dims are kind-aligned.
            let ok = unsafe {
                cache.moe_ffn_grouped_dev_resident(
                    kind, &x, &gp, &upp, &dp, &wv, &mut out_gpu, D, D,
                )
            };
            if ok {
                cu_grade(name, &out_gpu, &cpu, 0.999, 0.02, &mut counts);
            } else {
                cu_emit(name, "KERNEL_ERR", "grouped-ffn-returned-false");
                *counts.entry("KERNEL_ERR").or_default() += 1;
            }
        } else {
            cu_emit(name, "KERNEL_ERR", "cache-create-failed");
            *counts.entry("KERNEL_ERR").or_default() += 1;
        }
        drop(keep);
    }

    // ---- Feature 1b: CUDA-graph device-resident attention block ----
    // Drive `CudaMatvecCache::decode_attn_block_f32_graph` (the whole attention
    // block captured into a cudaGraph + replayed) vs a CPU reference computing the
    // IDENTICAL op sequence on identical inputs. Validates the CHAINING + the
    // capture/replay + the async D2D KV-mirror scatter (not just the individual
    // kernels, which are covered above). Q4_K weights, a small realistic shape, a
    // short pre-seeded F32 KV history so the flash spans multiple positions. The
    // CUDA block's mirror is full F32 (no f16 rounding), so the only non-exactness
    // is the device parallel reduce in rmsnorm/flash — gated like the f32 flash
    // probe (cos > 0.995, max_rel < 0.10). Name MUST equal the dispatch verdict
    // `attn:dev_resident_block_cuda_graph` so a later validate-kernels run reads it.
    {
        let name = "attn:dev_resident_block_cuda_graph";
        const HEAD_DIM: usize = 128;
        const N_HEADS: usize = 8;
        const N_KV: usize = 2;
        const D_MODEL: usize = 1024;
        const MAX_CTX: usize = 64;
        const POS: usize = 5; // 5 seeded history rows + this token => kv_len = 6
        const EPS: f32 = 1e-5;
        const THETA: f32 = 10000.0;
        let d_q = N_HEADS * HEAD_DIM;
        let d_kv = N_KV * HEAD_DIM;
        let half = HEAD_DIM / 2;
        let kv_len = POS + 1;

        let layout = LAYOUTS.iter().find(|l| l.name == "q4_k").expect("q4_k layout");
        let cpu_mv = cpu_matvec_for("q4_k");
        let wq = gen_quant_bytes(layout, d_q, D_MODEL, 0x1111);
        let wk = gen_quant_bytes(layout, d_kv, D_MODEL, 0x2222);
        let wv = gen_quant_bytes(layout, d_kv, D_MODEL, 0x3333);
        let wo = gen_quant_bytes(layout, D_MODEL, d_q, 0x4444);

        let mut rng = XorShift(0xC0DEC0DE);
        let hidden0: Vec<f32> = (0..D_MODEL).map(|_| rng.f32_pm(0.6)).collect();
        let attn_norm: Vec<f32> = (0..D_MODEL).map(|_| 0.5 + rng.f32_pm(0.2)).collect();
        let ffn_norm: Vec<f32> = (0..D_MODEL).map(|_| 0.5 + rng.f32_pm(0.2)).collect();
        let inv_freq: Vec<f32> = (0..half)
            .map(|j| 1.0f32 / THETA.powf(2.0 * j as f32 / HEAD_DIM as f32))
            .collect();
        // F32 KV history rows [0, POS) per kv-head (head-major) — NO f16 rounding
        // (the CUDA mirror is f32, unlike the SYCL f16 mirror).
        let hist_elems = N_KV * MAX_CTX * HEAD_DIM;
        let mut kcache = vec![0f32; hist_elems];
        let mut vcache = vec![0f32; hist_elems];
        for h in 0..N_KV {
            for t in 0..POS {
                for dd in 0..HEAD_DIM {
                    let idx = (h * MAX_CTX + t) * HEAD_DIM + dd;
                    kcache[idx] = rng.f32_pm(0.6);
                    vcache[idx] = rng.f32_pm(0.6);
                }
            }
        }

        // ---- CPU reference (identical op sequence) ----
        let rope = |qk: &mut [f32], nh: usize| {
            for hh in 0..nh {
                let base = hh * HEAD_DIM;
                for j in 0..half {
                    let angle = POS as f32 * inv_freq[j];
                    let (s, c) = angle.sin_cos();
                    let x0 = qk[base + j];
                    let x1 = qk[base + j + half];
                    qk[base + j] = x0 * c - x1 * s;
                    qk[base + j + half] = x0 * s + x1 * c;
                }
            }
        };
        let mut hnorm = vec![0f32; D_MODEL];
        k::rmsnorm_f32_row(&hidden0, &attn_norm, &mut hnorm, EPS);
        let mut q_ref = vec![0f32; d_q];
        let mut k_ref = vec![0f32; d_kv];
        let mut v_ref = vec![0f32; d_kv];
        cpu_mv(&wq, &hnorm, &mut q_ref, d_q, D_MODEL);
        cpu_mv(&wk, &hnorm, &mut k_ref, d_kv, D_MODEL);
        cpu_mv(&wv, &hnorm, &mut v_ref, d_kv, D_MODEL);
        rope(&mut q_ref, N_HEADS);
        rope(&mut k_ref, N_KV);
        // Reference KV cache: the f32 history + the new post-RoPE K / plain V at
        // POS, exactly what the device mirror holds after the scatter.
        let mut ref_k = kcache.clone();
        let mut ref_v = vcache.clone();
        for h in 0..N_KV {
            for dd in 0..HEAD_DIM {
                let idx = (h * MAX_CTX + POS) * HEAD_DIM + dd;
                ref_k[idx] = k_ref[h * HEAD_DIM + dd];
                ref_v[idx] = v_ref[h * HEAD_DIM + dd];
            }
        }
        // Three cases drive the ONE `attn:dev_resident_block_cuda_graph` verdict
        // (q_ref/k_ref/v_ref/ref_k/ref_v above are case-independent; only the
        // attention + downstream + device run depend on the uniforms):
        //   (0, 0.0)           passthrough — must stay bit-identical.
        //   (GRAPH_DEV_WINDOW, 0.0) SWA — kv_len=6 here, so a small window=3 trims.
        //   (0, 50.0)          Gemma2 attention-logit softcap (the real value).
        // The CPU reference applies the same via `gqa_attention_one_step_capped_
        // windowed`; `hidden` is graded as a DELTA (so the shared input residual
        // can't mask a window/cap miscompute) + `hnorm2` raw. WRITE-BLIND here (no
        // NVIDIA GPU) — validated on-device later via `--cuda-parity`.
        const GRAPH_DEV_WINDOW: usize = 3;
        let cases: [(u32, f32); 3] = [(0, 0.0), (GRAPH_DEV_WINDOW as u32, 0.0), (0, 50.0)];
        match ck::CudaMatvecCache::new(0, 16 << 20) {
            None => {
                cu_emit(name, "KERNEL_ERR", "cache-create-failed");
                *counts.entry("KERNEL_ERR").or_default() += 1;
            }
            Some(mut cache) => {
                let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
                // None = all ran OK; Some(reason) = a SKIP condition broke the loop.
                let mut skip: Option<&'static str> = None;
                for (window, softcap) in cases {
                    // CPU reference attention + o-proj + residual + post-norm.
                    let mut attn_ref = vec![0f32; d_q];
                    k::gqa_attention_one_step_capped_windowed(
                        &q_ref, &ref_k, &ref_v, &mut attn_ref,
                        N_HEADS, N_KV, HEAD_DIM, MAX_CTX, kv_len, window as usize, softcap,
                    );
                    let mut o_ref = vec![0f32; D_MODEL];
                    cpu_mv(&wo, &attn_ref, &mut o_ref, D_MODEL, d_q);
                    let mut hidden_ref = hidden0.clone();
                    for i in 0..D_MODEL {
                        hidden_ref[i] += o_ref[i];
                    }
                    let mut hnorm2_ref = vec![0f32; D_MODEL];
                    k::rmsnorm_f32_row(&hidden_ref, &ffn_norm, &mut hnorm2_ref, EPS);
                    if !hidden_ref.iter().chain(hnorm2_ref.iter()).all(|v| v.is_finite()) {
                        skip = Some("cpu-reference-nonfinite");
                        break;
                    }
                    let mut hidden_dev = hidden0.clone();
                    let mut hnorm2_dev = vec![0f32; D_MODEL];
                    let mut k_row = vec![0f32; d_kv];
                    let mut v_row = vec![0f32; d_kv];
                    // SAFETY: host buffers are sized to the dims; weight slices live for
                    // the call; the cache owns the stream + mirror. n_layers = 1, layer 0.
                    // Re-seeding the mirror each case resets its valid_len to POS so the
                    // POS-write + flash re-run independently. window/softcap = this case.
                    let ok = unsafe {
                        cache.seed_attn_mirror_f32(
                            0, &kcache, &vcache, POS, N_KV, HEAD_DIM, MAX_CTX, 1,
                        ) && cache.decode_attn_block_f32_graph(
                            (ck::CudaPackedKind::Q4_K, wq.as_ptr() as usize, &wq),
                            (ck::CudaPackedKind::Q4_K, wk.as_ptr() as usize, &wk),
                            (ck::CudaPackedKind::Q4_K, wv.as_ptr() as usize, &wv),
                            (ck::CudaPackedKind::Q4_K, wo.as_ptr() as usize, &wo),
                            &attn_norm,
                            &ffn_norm,
                            None,
                            None,
                            &inv_freq,
                            &mut hidden_dev,
                            &mut hnorm2_dev,
                            &mut k_row,
                            &mut v_row,
                            N_HEADS,
                            N_KV,
                            HEAD_DIM,
                            D_MODEL,
                            POS,
                            MAX_CTX,
                            1,
                            0,
                            EPS,
                            window as i32,
                            softcap,
                            None,
                            true,
                            true,
                        )
                    };
                    if !ok {
                        skip = Some("graph-block-declined");
                        break;
                    }
                    // Grade the residual hidden DELTA + the post-attn norm jointly.
                    let mut gpu: Vec<f32> =
                        hidden_dev.iter().zip(&hidden0).map(|(h, h0)| h - h0).collect();
                    gpu.extend_from_slice(&hnorm2_dev);
                    let mut cpu: Vec<f32> =
                        hidden_ref.iter().zip(&hidden0).map(|(h, h0)| h - h0).collect();
                    cpu.extend_from_slice(&hnorm2_ref);
                    let (cos, max_rel) = compare(&gpu, &cpu);
                    worst_cos = worst_cos.min(cos);
                    worst_rel = worst_rel.max(max_rel);
                }
                match skip {
                    Some(reason) => {
                        cu_emit(name, "SKIP", reason);
                        *counts.entry("SKIP").or_default() += 1;
                    }
                    None => {
                        let key: &'static str =
                            if worst_cos > 0.995 && worst_rel < 0.10 { "OK" } else { "MISCOMPUTE" };
                        *counts.entry(key).or_default() += 1;
                        cu_emit(
                            name,
                            key,
                            &format!(
                                "cos={worst_cos:.6} max_rel={worst_rel:.4} \
                                 (pass+swa{GRAPH_DEV_WINDOW}+cap50)"
                            ),
                        );
                    }
                }
            }
        }
    }

    // ---- P7: WHOLE-LAYER CUDA-graph capture (attention + dense SwiGLU FFN) ----
    // The attention-only graph is validated above; this drives the `dense_ffn =
    // Some(..)` arm of `decode_attn_block_f32_graph` so the ENTIRE decode layer —
    // attention block AND the SwiGLU FFN (gate/up → silu_mul → down → residual) —
    // is captured into ONE cudaGraph and replayed, graded vs a CPU reference that
    // computes the identical attention+FFN chain. This is the only coverage of
    // the FFN-INSIDE-the-graph path. Verdict name = the correctness anchor
    // `layer:dev_resident_cuda_graph`. WRITE-BLIND (no NVIDIA GPU here) —
    // validated on-device via `--cuda-parity`. Looser bar than the attention-only
    // probe (cos>0.99, rel<0.15) because the FFN adds three more device matvecs +
    // a silu_mul reduce on top of the attention chain.
    {
        let name = rustllama_tuner::VERDICT_LAYER_DEV_RESIDENT_CUDA_GRAPH;
        const HEAD_DIM: usize = 128;
        const N_HEADS: usize = 8;
        const N_KV: usize = 2;
        const D_MODEL: usize = 1024;
        const D_FF: usize = 256; // Q4_K `down` has k=D_FF → must be a multiple of 256
        const MAX_CTX: usize = 64;
        const POS: usize = 5;
        const EPS: f32 = 1e-5;
        const THETA: f32 = 10000.0;
        let d_q = N_HEADS * HEAD_DIM;
        let d_kv = N_KV * HEAD_DIM;
        let half = HEAD_DIM / 2;
        let kv_len = POS + 1;

        let layout = LAYOUTS.iter().find(|l| l.name == "q4_k").expect("q4_k layout");
        let cpu_mv = cpu_matvec_for("q4_k");
        let wq = gen_quant_bytes(layout, d_q, D_MODEL, 0x1A1A);
        let wk = gen_quant_bytes(layout, d_kv, D_MODEL, 0x2B2B);
        let wv = gen_quant_bytes(layout, d_kv, D_MODEL, 0x3C3C);
        let wo = gen_quant_bytes(layout, D_MODEL, d_q, 0x4D4D);
        // Dense FFN weights: gate/up [D_FF × D_MODEL], down [D_MODEL × D_FF].
        let wg = gen_quant_bytes(layout, D_FF, D_MODEL, 0x5E5E);
        let wu = gen_quant_bytes(layout, D_FF, D_MODEL, 0x6F6F);
        let wd = gen_quant_bytes(layout, D_MODEL, D_FF, 0x7070);

        let mut rng = XorShift(0x5EED_1B0B);
        let hidden0: Vec<f32> = (0..D_MODEL).map(|_| rng.f32_pm(0.6)).collect();
        let attn_norm: Vec<f32> = (0..D_MODEL).map(|_| 0.5 + rng.f32_pm(0.2)).collect();
        let ffn_norm: Vec<f32> = (0..D_MODEL).map(|_| 0.5 + rng.f32_pm(0.2)).collect();
        let inv_freq: Vec<f32> = (0..half)
            .map(|j| 1.0f32 / THETA.powf(2.0 * j as f32 / HEAD_DIM as f32))
            .collect();
        let hist_elems = N_KV * MAX_CTX * HEAD_DIM;
        let mut kcache = vec![0f32; hist_elems];
        let mut vcache = vec![0f32; hist_elems];
        for h in 0..N_KV {
            for t in 0..POS {
                for dd in 0..HEAD_DIM {
                    let idx = (h * MAX_CTX + t) * HEAD_DIM + dd;
                    kcache[idx] = rng.f32_pm(0.6);
                    vcache[idx] = rng.f32_pm(0.6);
                }
            }
        }

        // ---- CPU reference: attention block + SwiGLU FFN + residual ----
        let rope = |qk: &mut [f32], nh: usize| {
            for hh in 0..nh {
                let base = hh * HEAD_DIM;
                for j in 0..half {
                    let angle = POS as f32 * inv_freq[j];
                    let (s, c) = angle.sin_cos();
                    let x0 = qk[base + j];
                    let x1 = qk[base + j + half];
                    qk[base + j] = x0 * c - x1 * s;
                    qk[base + j + half] = x0 * s + x1 * c;
                }
            }
        };
        let mut hnorm = vec![0f32; D_MODEL];
        k::rmsnorm_f32_row(&hidden0, &attn_norm, &mut hnorm, EPS);
        let mut q_ref = vec![0f32; d_q];
        let mut k_ref = vec![0f32; d_kv];
        let mut v_ref = vec![0f32; d_kv];
        cpu_mv(&wq, &hnorm, &mut q_ref, d_q, D_MODEL);
        cpu_mv(&wk, &hnorm, &mut k_ref, d_kv, D_MODEL);
        cpu_mv(&wv, &hnorm, &mut v_ref, d_kv, D_MODEL);
        rope(&mut q_ref, N_HEADS);
        rope(&mut k_ref, N_KV);
        let mut ref_k = kcache.clone();
        let mut ref_v = vcache.clone();
        for h in 0..N_KV {
            for dd in 0..HEAD_DIM {
                let idx = (h * MAX_CTX + POS) * HEAD_DIM + dd;
                ref_k[idx] = k_ref[h * HEAD_DIM + dd];
                ref_v[idx] = v_ref[h * HEAD_DIM + dd];
            }
        }
        let mut attn_ref = vec![0f32; d_q];
        k::gqa_attention_one_step_capped_windowed(
            &q_ref, &ref_k, &ref_v, &mut attn_ref, N_HEADS, N_KV, HEAD_DIM, MAX_CTX, kv_len, 0, 0.0,
        );
        let mut o_ref = vec![0f32; D_MODEL];
        cpu_mv(&wo, &attn_ref, &mut o_ref, D_MODEL, d_q);
        let mut hidden1 = hidden0.clone();
        for i in 0..D_MODEL {
            hidden1[i] += o_ref[i];
        }
        let mut hnorm2_ref = vec![0f32; D_MODEL];
        k::rmsnorm_f32_row(&hidden1, &ffn_norm, &mut hnorm2_ref, EPS);
        // SwiGLU FFN from the post-attn norm, then residual add.
        let mut gate_ref = vec![0f32; D_FF];
        let mut up_ref = vec![0f32; D_FF];
        cpu_mv(&wg, &hnorm2_ref, &mut gate_ref, D_FF, D_MODEL);
        cpu_mv(&wu, &hnorm2_ref, &mut up_ref, D_FF, D_MODEL);
        let mut act_ref = vec![0f32; D_FF];
        k::silu_mul_f32(&gate_ref, &up_ref, &mut act_ref);
        let mut ffn_out_ref = vec![0f32; D_MODEL];
        cpu_mv(&wd, &act_ref, &mut ffn_out_ref, D_MODEL, D_FF);
        let mut hidden_final = hidden1.clone();
        for i in 0..D_MODEL {
            hidden_final[i] += ffn_out_ref[i];
        }

        match ck::CudaMatvecCache::new(0, 16 << 20) {
            None => {
                cu_emit(name, "KERNEL_ERR", "cache-create-failed");
                *counts.entry("KERNEL_ERR").or_default() += 1;
            }
            Some(mut cache) => {
                let mut hidden_dev = hidden0.clone();
                let mut hnorm2_dev = vec![0f32; D_MODEL];
                let mut k_row = vec![0f32; d_kv];
                let mut v_row = vec![0f32; d_kv];
                // SAFETY: host buffers sized to dims; weight slices live for the
                // call; cache owns the stream + mirror. n_layers=1, layer 0,
                // passthrough window/softcap, dense FFN captured in the graph.
                let ok = unsafe {
                    cache.seed_attn_mirror_f32(0, &kcache, &vcache, POS, N_KV, HEAD_DIM, MAX_CTX, 1)
                        && cache.decode_attn_block_f32_graph(
                            (ck::CudaPackedKind::Q4_K, wq.as_ptr() as usize, &wq),
                            (ck::CudaPackedKind::Q4_K, wk.as_ptr() as usize, &wk),
                            (ck::CudaPackedKind::Q4_K, wv.as_ptr() as usize, &wv),
                            (ck::CudaPackedKind::Q4_K, wo.as_ptr() as usize, &wo),
                            &attn_norm,
                            &ffn_norm,
                            None,
                            None,
                            &inv_freq,
                            &mut hidden_dev,
                            &mut hnorm2_dev,
                            &mut k_row,
                            &mut v_row,
                            N_HEADS,
                            N_KV,
                            HEAD_DIM,
                            D_MODEL,
                            POS,
                            MAX_CTX,
                            1,
                            0,
                            EPS,
                            0,
                            0.0,
                            Some((
                                (ck::CudaPackedKind::Q4_K, wg.as_ptr() as usize, wg.as_slice()),
                                (ck::CudaPackedKind::Q4_K, wu.as_ptr() as usize, wu.as_slice()),
                                (ck::CudaPackedKind::Q4_K, wd.as_ptr() as usize, wd.as_slice()),
                                D_FF,
                            )),
                            true,
                            true,
                        )
                };
                if !ok {
                    cu_emit(name, "SKIP", "whole-layer-graph-declined");
                    *counts.entry("SKIP").or_default() += 1;
                } else {
                    // Grade the residual hidden DELTA (so the shared input can't
                    // mask an FFN miscompute) + the post-attn norm jointly.
                    let mut gpu: Vec<f32> =
                        hidden_dev.iter().zip(&hidden0).map(|(h, h0)| h - h0).collect();
                    gpu.extend_from_slice(&hnorm2_dev);
                    let mut cpu: Vec<f32> =
                        hidden_final.iter().zip(&hidden0).map(|(h, h0)| h - h0).collect();
                    cpu.extend_from_slice(&hnorm2_ref);
                    let (cos, max_rel) = compare(&gpu, &cpu);
                    let key: &'static str =
                        if cos > 0.99 && max_rel < 0.15 { "OK" } else { "MISCOMPUTE" };
                    *counts.entry(key).or_default() += 1;
                    cu_emit(name, key, &format!("cos={cos:.6} max_rel={max_rel:.4} (attn+ffn)"));
                }
            }
        }
    }

    println!();
    let summary: Vec<String> = counts.iter().map(|(k, v)| format!("{k}={v}")).collect();
    println!("summary: {}", summary.join("  "));
    let bad = counts.get("MISCOMPUTE").copied().unwrap_or(0)
        + counts.get("KERNEL_ERR").copied().unwrap_or(0);
    if bad > 0 {
        println!(
            "\n{bad} CUDA kernel(s) did NOT match their CPU reference (MISCOMPUTE = wrong \
             numbers; KERNEL_ERR = launch/alloc failure). Investigate before enabling the \
             CUDA backend for these."
        );
    } else if counts.get("OK").copied().unwrap_or(0) > 0 {
        println!("\nall exercised CUDA kernels match their CPU references.");
    }
    Ok(())
}

// ===============================================================
// CPU parity (`rustllama doctor --cpu-parity`)
// ===============================================================
//
// The CPU kernel layer is the *reference* the SYCL and CUDA harnesses
// above grade their GPU kernels against — so a CPU self-check can't
// compare the CPU against a GPU. What it CAN do is exercise the CPU
// layer's own fast paths against a slow, obviously-correct reference:
// the SIMD (AVX-512 / AVX2 / NEON) and rayon-parallel matvecs against a
// naive scalar loop, on THIS host's actual CPU. Which path a kernel
// takes is chosen at runtime by `is_x86_feature_detected!`, so the SIMD
// code that actually ran is a property of the machine, not the build —
// exactly what a `doctor` self-test should confirm. Runs in-process (no
// device to lose); the per-family PASS/FAIL output mirrors the CUDA
// harness above (reusing `cu_grade` / `cu_emit` / `compare`).
//
// Only the matvec families carry a SIMD/parallel seam worth checking
// (f32, f16, and the PTQ1_0 ternary fastdot/batched paths). The
// scalar-only quant matvecs (Q*/IQ*) ARE the reference here — there is
// no second CPU implementation to compare them against — so they are
// validated indirectly by the SYCL/CUDA harnesses (which grade the GPU
// kernels against them) and by the crate's own unit tests, not here.

/// Naive scalar f32 matvec — the obviously-correct reference the SIMD
/// and rayon-parallel f32 paths are graded against. Deliberately
/// un-optimized (single accumulator, source order) so it shares no
/// code with the kernels under test.
fn ref_matvec_f32_scalar(w: &[f32], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    for i in 0..m {
        let row = &w[i * k..(i + 1) * k];
        let mut acc = 0f32;
        for p in 0..k {
            acc += row[p] * x[p];
        }
        out[i] = acc;
    }
}

/// Which CPU SIMD ISA the matvec kernels will dispatch to on this host,
/// for the harness banner. Informational only — the kernels detect the
/// same features the same way at call time.
fn cpu_simd_level() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            "AVX-512"
        } else if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            "AVX2+FMA"
        } else {
            "scalar (no AVX2/FMA)"
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        "NEON"
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "scalar"
    }
}

/// MoE tiered-expert SYCL PROMOTION self-check (`doctor --sycl-tier-parity`).
/// Drives the real promotion + device-resident matvec via the engine (which
/// owns the accel layer) and prints a single OK / MISCOMPUTE / SKIP line. The
/// kernel is already covered by `--sycl-parity`; this validates the promotion
/// plumbing the tier uniquely uses, and SKIPs cleanly off dedicated-SYCL HW.
pub fn run_sycl_tier_parity() -> anyhow::Result<()> {
    println!("MoE tiered-expert SYCL promotion self-check (dedicated Arc/PVC; in-process)");
    match rustllama_engine::sycl_tier_self_check() {
        rustllama_engine::SyclTierCheck::Skip(why) => {
            println!("tier:sycl_promote  SKIP  {why}");
            println!("\nsummary: SKIP");
        }
        rustllama_engine::SyclTierCheck::Pass { cos, max_rel } => {
            println!("tier:sycl_promote  OK    cos={cos:.6} max_rel={max_rel:.4}");
            println!("\nsummary: OK=1 MISCOMPUTE=0");
        }
        rustllama_engine::SyclTierCheck::Miscompute { cos, max_rel } => {
            println!("tier:sycl_promote  MISCOMPUTE  cos={cos:.6} max_rel={max_rel:.4}");
            println!(
                "\nsummary: OK=0 MISCOMPUTE=1 — promoted-expert matvec disagrees with the CPU reference"
            );
        }
    }
    Ok(())
}

pub fn run_cpu_parity() -> anyhow::Result<()> {
    println!(
        "CPU kernel parity harness (matvec SIMD / rayon-parallel paths vs a naive scalar \
         reference, in-process)"
    );
    println!("CPU SIMD dispatch on this host: {}", cpu_simd_level());
    println!();
    let mut counts: std::collections::BTreeMap<&'static str, usize> = Default::default();

    // ---- Dense f32 matvec: SIMD-serial + rayon-parallel vs naive scalar ----
    // MV_M (512) is above the parallel crossover (256 default), so the
    // auto-dispatcher `matvec_f32` takes its rayon path here.
    {
        let (m, kd) = (MV_M, MV_K);
        let w = gen_x(m * kd, 101);
        let x = gen_x(kd, 103);
        let mut refv = vec![0f32; m];
        ref_matvec_f32_scalar(&w, &x, &mut refv, m, kd);

        let mut simd = vec![0f32; m];
        k::matvec_f32_serial(&w, &x, &mut simd, m, kd);
        cu_grade("matvec:f32 simd-serial", &simd, &refv, 0.9999, 0.01, &mut counts);

        let mut par = vec![0f32; m];
        k::matvec_f32(&w, &x, &mut par, m, kd);
        cu_grade("matvec:f32 parallel", &par, &refv, 0.9999, 0.01, &mut counts);
    }

    // ---- F16 matvec: SIMD-serial + rayon-parallel vs the naive scalar gemm ----
    // `gemm_f16_w_f32_a` (N=1) is the always-scalar reference the serial
    // matvec itself falls back to when no AVX2/F16C is detected.
    {
        let (m, kd) = (MV_M, MV_K);
        let wf = gen_x(m * kd, 111);
        let w: Vec<half::f16> = wf.iter().map(|&v| half::f16::from_f32(v)).collect();
        let x = gen_x(kd, 113);
        let mut refv = vec![0f32; m];
        k::gemm_f16_w_f32_a(&w, &x, &mut refv, m, 1, kd);

        let mut simd = vec![0f32; m];
        k::matvec_f16_w_f32_a_serial(&w, &x, &mut simd, m, kd);
        cu_grade("matvec:f16 simd-serial", &simd, &refv, 0.9999, 0.01, &mut counts);

        let mut par = vec![0f32; m];
        k::matvec_f16_w_f32_a(&w, &x, &mut par, m, kd);
        cu_grade("matvec:f16 parallel", &par, &refv, 0.9999, 0.01, &mut counts);
    }

    // ---- PTQ1_0 ternary: production fastdot + batched vs the reference kernel ----
    // Reuses the SYCL harness's quant-byte synthesis + finite-reference
    // guard. `matvec_ptq1_0_w_f32_a` is the reference (AVX2 → scalar);
    // `_fast` and `_batched` are the reassociated SIMD paths, gated to
    // tolerance (not bitwise) by their own contracts.
    {
        let layout = LAYOUTS
            .iter()
            .find(|l| l.name == "ptq1_0")
            .expect("ptq1_0 layout");
        let x = gen_x(MV_K, 121);
        match finite_ref(layout, cpu_matvec_for("ptq1_0"), &x) {
            Some((w, refv)) => {
                let mut fast = vec![0f32; MV_M];
                k::matvec_ptq1_0_w_f32_a_fast(&w, &x, &mut fast, MV_M, MV_K);
                cu_grade("matvec:ptq1_0 fastdot", &fast, &refv, 0.999, 0.02, &mut counts);

                // Batched decode/replay path, single activation row.
                let mut batched = vec![0f32; MV_M];
                k::matvec_ptq1_0_w_f32_a_batched(&w, &x, &mut batched, MV_M, MV_K, 1);
                cu_grade("matvec:ptq1_0 batched", &batched, &refv, 0.999, 0.02, &mut counts);
            }
            None => {
                cu_emit("matvec:ptq1_0", "SKIP", "no-finite-reference");
                *counts.entry("SKIP").or_default() += 1;
            }
        }
    }

    println!();
    let summary: Vec<String> = counts.iter().map(|(k, v)| format!("{k}={v}")).collect();
    println!("summary: {}", summary.join("  "));
    let bad = counts.get("MISCOMPUTE").copied().unwrap_or(0);
    if bad > 0 {
        println!(
            "\n{bad} CPU matvec path(s) diverged from the scalar reference beyond tolerance \
             (MISCOMPUTE) — a SIMD/parallel kernel is miscomputing on this CPU. Investigate \
             before trusting the CPU layer as the parity reference / inference fallback."
        );
    } else if counts.get("OK").copied().unwrap_or(0) > 0 {
        println!(
            "\nall exercised CPU SIMD / parallel matvec paths match the scalar reference on \
             this host."
        );
    }
    Ok(())
}

// ===============================================================
// Metal (MLX) parity (`rustllama doctor --metal-parity`)
// ===============================================================
//
// The Apple-Metal analogue of the CUDA harness: run each native Metal kernel
// against its CPU reference on identical inputs, in-process. Everything SKIPs
// off Apple Silicon (`mk::device_count()==0` on the inert stub), so this runs
// clean (all SKIP) on Windows/Linux and does the real comparison only on a Mac.
//
// Covers the packed-quant matvecs (the bulk of the write-blind Metal kernels:
// all K-quants the shared LAYOUTS describe, the IQ grids, IQ4, MXFP, PTQ1_0)
// plus the host-pointer dense f32 matvec + rmsnorm. The forward-pass primitives
// (rope/silu/embedding), F32 + quantized-KV flash, and argmax are a follow-on
// (they need device-buffer plumbing + their own references); the matvec probes
// here exercise the highest-risk dequant math first.

/// Map a LAYOUTS dtype name to the MLX packed-matvec kind. `None` for names the
/// shared harness describes but the MLX cache doesn't enumerate (there are none
/// today — kept for symmetry with the CUDA/SYCL mappers).
fn mlx_kind_for(name: &str) -> Option<mk::MlxPackedKind> {
    use mk::MlxPackedKind as P;
    Some(match name {
        "q8_0" => P::Q8_0,
        "q4_k" => P::Q4_K,
        "q5_k" => P::Q5_K,
        "q6_k" => P::Q6_K,
        "iq4_nl" => P::Iq4_Nl,
        "iq4_xs" => P::Iq4_Xs,
        "iq1_s" => P::Iq1_S,
        "iq1_m" => P::Iq1_M,
        "iq2_xxs" => P::Iq2_Xxs,
        "iq2_xs" => P::Iq2_Xs,
        "iq2_s" => P::Iq2_S,
        "iq3_xxs" => P::Iq3_Xxs,
        "iq3_s" => P::Iq3_S,
        "ptq1_0" => P::Ptq1_0,
        "mxfp4" => P::Mxfp4,
        "mxfp6" => P::Mxfp6,
        "mxfp8" => P::Mxfp8,
        _ => return None,
    })
}

pub fn run_metal_parity() -> anyhow::Result<()> {
    println!("Metal (MLX) kernel parity harness (Apple GPU backend, in-process)");
    let n_dev = mk::device_count();
    if n_dev == 0 {
        println!(
            "no Metal device visible (non-Apple-Silicon host, or no Metal GPU) — all SKIP"
        );
        println!("\nsummary: SKIP (no Metal device)");
        return Ok(());
    }
    let budget = match mk::device_info(0) {
        Ok(info) => {
            println!(
                "Metal device 0: {} ({} MB), {} device(s) total",
                info.name,
                info.total_mem_bytes / (1024 * 1024),
                n_dev,
            );
            ((info.total_mem_bytes as f64) * 0.85) as usize
        }
        Err(_) => {
            println!("{n_dev} Metal device(s) (info query failed)");
            0
        }
    };
    let Some(mut cache) = mk::MlxMatvecCache::new(0, budget) else {
        println!("failed to create MlxMatvecCache on device 0 — all SKIP");
        println!("\nsummary: SKIP (no cache)");
        return Ok(());
    };
    println!();
    let mut counts: std::collections::BTreeMap<&'static str, usize> = Default::default();

    // ---- Packed matvecs the MLX backend implements ----
    for layout in LAYOUTS {
        let Some(kind) = mlx_kind_for(layout.name) else {
            continue;
        };
        let name = format!("matvec:{}", layout.name);
        let x = gen_x(MV_K, 42);
        let Some((w, cpu_out)) = finite_ref(layout, cpu_matvec_for(layout.name), &x) else {
            cu_emit(&name, "SKIP", "no-finite-reference");
            *counts.entry("SKIP").or_default() += 1;
            continue;
        };
        let mut out = vec![0f32; MV_M];
        let ok = cache.matvec_packed(kind, w.as_ptr() as usize, &w, &x, &mut out, MV_M, MV_K);
        if ok {
            cu_grade(&name, &out, &cpu_out, 0.999, 0.02, &mut counts);
        } else {
            cu_emit(&name, "KERNEL_ERR", "matvec_packed returned false");
            *counts.entry("KERNEL_ERR").or_default() += 1;
        }
    }

    // ---- Batched (prefill) packed matvecs ----
    // MLX twin of the CUDA/SYCL `matvecb:` probe: the N-lifted prefill kernels
    // (`MlxMatvecCache::matvec_packed_batched`) run during prefill but were
    // ungraded. Grade [N,M] GPU output vs the CPU reference run per-row.
    for layout in LAYOUTS {
        let Some(kind) = mlx_kind_for(layout.name) else {
            continue;
        };
        let name = format!("matvecb:{}", layout.name);
        let cpu = cpu_matvec_for(layout.name);
        let x = gen_x(MV_N * MV_K, 42);
        let Some((w, _)) = finite_ref(layout, cpu, &x[0..MV_K]) else {
            cu_emit(&name, "SKIP", "no-finite-reference");
            *counts.entry("SKIP").or_default() += 1;
            continue;
        };
        let mut cpu_out = vec![0f32; MV_N * MV_M];
        for row in 0..MV_N {
            cpu(
                &w,
                &x[row * MV_K..(row + 1) * MV_K],
                &mut cpu_out[row * MV_M..(row + 1) * MV_M],
                MV_M,
                MV_K,
            );
        }
        if !cpu_out.iter().all(|v| v.is_finite()) {
            cu_emit(&name, "SKIP", "no-finite-reference");
            *counts.entry("SKIP").or_default() += 1;
            continue;
        }
        let mut out = vec![0f32; MV_N * MV_M];
        let ok = cache.matvec_packed_batched(
            kind,
            w.as_ptr() as usize,
            &w,
            &x,
            &mut out,
            MV_M,
            MV_K,
            MV_N,
        );
        if ok {
            cu_grade(&name, &out, &cpu_out, 0.999, 0.02, &mut counts);
        } else {
            cu_emit(&name, "KERNEL_ERR", "matvec_packed_batched returned false");
            *counts.entry("KERNEL_ERR").or_default() += 1;
        }
    }

    // ---- Fused gate+up matvec (decode) ----
    // MLX twin of the CUDA/SYCL fused probe: one dispatch computes gate+up,
    // graded against two independent CPU matvecs. Distinct synthetic cache keys
    // (not host pointers) so no stale-weight aliasing across iterations.
    for (idx, dtype) in [
        "q8_0", "q4_k", "q6_k", "q5_k", "iq4_nl", "iq4_xs", "iq1_s", "iq1_m",
        "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "ptq1_0",
    ]
    .iter()
    .enumerate()
    {
        let Some(kind) = mlx_kind_for(dtype) else {
            continue;
        };
        let name = format!("fused:{dtype}");
        let layout = LAYOUTS.iter().find(|l| l.name == *dtype).expect("layout");
        let x = gen_x(MV_K, 44);
        let cpu = cpu_matvec_for(dtype);
        let Some((wg, cpu_g)) = finite_ref(layout, cpu, &x) else {
            cu_emit(&name, "SKIP", "no-finite-reference");
            *counts.entry("SKIP").or_default() += 1;
            continue;
        };
        let wu = gen_quant_bytes(layout, MV_M, MV_K, 0xBEEF0007);
        let mut cpu_u = vec![0f32; MV_M];
        cpu(&wu, &x, &mut cpu_u, MV_M, MV_K);
        if !cpu_u.iter().all(|v| v.is_finite()) {
            cu_emit(&name, "SKIP", "no-finite-reference");
            *counts.entry("SKIP").or_default() += 1;
            continue;
        }
        let (g_key, u_key) = (0x6a7e_0000 + idx * 2, 0x6a7e_0000 + idx * 2 + 1);
        let mut gout = vec![0f32; MV_M];
        let mut uout = vec![0f32; MV_M];
        let ok = cache.matvec_gate_up_fused(
            kind, g_key, &wg, u_key, &wu, &x, &mut gout, &mut uout, MV_M, MV_K,
        );
        if ok {
            let mut got = gout.clone();
            got.extend_from_slice(&uout);
            let mut refv = cpu_g.clone();
            refv.extend_from_slice(&cpu_u);
            cu_grade(&name, &got, &refv, 0.999, 0.02, &mut counts);
        } else {
            cu_emit(&name, "KERNEL_ERR", "matvec_gate_up_fused returned false");
            *counts.entry("KERNEL_ERR").or_default() += 1;
        }
    }

    // ---- C-A: real-MoE-forward coherence (verdict moe:gate_up_coherent:<dtype>) ----
    // MLX twin of the SYCL/CUDA coherence probe (WRITE-BLIND — settles on a Mac):
    // the per-expert FFN (several experts, non-512 shapes, SwiGLU, routing sum)
    // graded FUSED ≡ SEPARATE two-matvec ≡ CPU. `accel::moe_gate_up_fused_enabled`
    // ANDs this with the bare `fused:` parity above. Distinct per-expert cache
    // keys so no stale-weight aliasing across iterations.
    for (di, dtype) in [
        "q8_0", "q4_k", "q6_k", "q5_k", "iq4_nl", "iq4_xs", "iq1_s", "iq1_m",
        "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "ptq1_0",
    ]
    .iter()
    .enumerate()
    {
        let Some(kind) = mlx_kind_for(dtype) else {
            continue;
        };
        let name = format!("moe:gate_up_coherent:{dtype}");
        let layout = LAYOUTS.iter().find(|l| l.name == *dtype).expect("layout");
        let cpu = cpu_matvec_for(dtype);
        let mut worst_cos_cpu = 1.0f64;
        let mut worst_rel_cpu = 0.0f64;
        let mut worst_cos_sep = 1.0f64;
        let mut worst_rel_sep = 0.0f64;
        let mut failed: Option<(&'static str, String)> = None;
        'shapes: for (si, &(m, k)) in MOE_COH_SHAPES.iter().enumerate() {
            let x = gen_x(k, 0x5EED_0001 ^ (si as u32).wrapping_mul(7919));
            let mut inter_cpu = vec![0f32; m];
            let mut inter_sep = vec![0f32; m];
            let mut inter_fused = vec![0f32; m];
            for e in 0..MOE_COH_EXPERTS {
                let w_route = MOE_COH_ROUTE_W[e];
                let sg = 0xC0DE_0000u32 ^ ((si * 97 + e * 13) as u32);
                let su = 0xBEEF_0000u32 ^ ((si * 131 + e * 17) as u32);
                let Some((wg, cpu_g)) = coh_gen_finite(layout, cpu, m, k, &x, sg) else {
                    failed = Some(("SKIP", "no-finite-reference(gate)".into()));
                    break 'shapes;
                };
                let Some((wu, cpu_u)) = coh_gen_finite(layout, cpu, m, k, &x, su) else {
                    failed = Some(("SKIP", "no-finite-reference(up)".into()));
                    break 'shapes;
                };
                // Unique cache keys per (dtype, shape, expert, gate/up).
                let base = 0x4D0E_0000 + ((di * 64 + si * 8 + e) * 2);
                let (g_key, u_key) = (base, base + 1);
                let mut sep_g = vec![0f32; m];
                let mut sep_u = vec![0f32; m];
                let mut fus_g = vec![0f32; m];
                let mut fus_u = vec![0f32; m];
                let ok_sep = cache.matvec_packed(kind, g_key, &wg, &x, &mut sep_g, m, k)
                    && cache.matvec_packed(kind, u_key, &wu, &x, &mut sep_u, m, k);
                if !ok_sep {
                    failed = Some(("KERNEL_ERR", "separate matvec returned false".into()));
                    break 'shapes;
                }
                let ok_fused = cache.matvec_gate_up_fused(
                    kind, g_key, &wg, u_key, &wu, &x, &mut fus_g, &mut fus_u, m, k,
                );
                if !ok_fused {
                    failed = Some(("KERNEL_ERR", "matvec_gate_up_fused returned false".into()));
                    break 'shapes;
                }
                for j in 0..m {
                    inter_cpu[j] += w_route * silu(cpu_g[j]) * cpu_u[j];
                    inter_sep[j] += w_route * silu(sep_g[j]) * sep_u[j];
                    inter_fused[j] += w_route * silu(fus_g[j]) * fus_u[j];
                }
            }
            let (cos_cpu, rel_cpu) = compare(&inter_fused, &inter_cpu);
            let (cos_sep, rel_sep) = compare(&inter_fused, &inter_sep);
            worst_cos_cpu = worst_cos_cpu.min(cos_cpu);
            worst_rel_cpu = worst_rel_cpu.max(rel_cpu);
            worst_cos_sep = worst_cos_sep.min(cos_sep);
            worst_rel_sep = worst_rel_sep.max(rel_sep);
        }
        if let Some((v, detail)) = failed {
            cu_emit(&name, v, &detail);
            *counts.entry(v).or_default() += 1;
            continue;
        }
        let cpu_ok = worst_cos_cpu > 0.999 && worst_rel_cpu < 0.05;
        let sep_ok = worst_cos_sep > 0.9995 && worst_rel_sep < 0.02;
        let verdict = if cpu_ok && sep_ok { "OK" } else { "MISCOMPUTE" };
        cu_emit(
            &name,
            verdict,
            &format!(
                "cos_cpu={worst_cos_cpu:.6} rel_cpu={worst_rel_cpu:.4} \
                 cos_sep={worst_cos_sep:.6} rel_sep={worst_rel_sep:.4}"
            ),
        );
        *counts.entry(verdict).or_default() += 1;
    }

    // ---- Dense f32 matvec (host-pointer reference path) ----
    {
        let (m, kd) = (MV_M, MV_K);
        let w = gen_x(m * kd, 201);
        let x = gen_x(kd, 203);
        let mut cpu = vec![0f32; m];
        ref_matvec_f32_scalar(&w, &x, &mut cpu, m, kd);
        let mut gpu = vec![0f32; m];
        match mk::matvec_f32(&w, &x, &mut gpu, m, kd) {
            Ok(()) => cu_grade("matvec:f32", &gpu, &cpu, 0.9999, 0.01, &mut counts),
            Err(e) => {
                cu_emit("matvec:f32", "KERNEL_ERR", &format!("{e:?}"));
                *counts.entry("KERNEL_ERR").or_default() += 1;
            }
        }
    }

    // ---- RMSNorm (host-pointer reference path) ----
    {
        let rows = 4usize;
        let d = 256usize;
        let x = gen_x(rows * d, 21);
        let wv = gen_x(d, 22);
        let eps = 1e-5f32;
        let mut cpu = vec![0f32; rows * d];
        for r in 0..rows {
            let mut ss = 0.0f32;
            for i in 0..d {
                ss += x[r * d + i] * x[r * d + i];
            }
            let inv = 1.0 / (ss / d as f32 + eps).sqrt();
            for i in 0..d {
                cpu[r * d + i] = x[r * d + i] * inv * wv[i];
            }
        }
        let mut gpu = vec![0f32; rows * d];
        match mk::rmsnorm_f32(&x, &wv, &mut gpu, rows, d, eps) {
            Ok(()) => cu_grade("rmsnorm:f32", &gpu, &cpu, 0.9999, 0.01, &mut counts),
            Err(e) => {
                cu_emit("rmsnorm:f32", "KERNEL_ERR", &format!("{e:?}"));
                *counts.entry("KERNEL_ERR").or_default() += 1;
            }
        }
    }

    // ---- P17: Gated-DeltaNet chunked-prefill recurrence (write-blind) ----
    // Metal twin of the SYCL/CUDA `ssm:delta_net` probe: head-major packed inputs
    // (q/k L2-normed per token), CPU chunked reference per head, graded against the
    // Metal kernel's per-token output AND the final recurrent state (concatenated).
    {
        const H: usize = 4;
        const T: usize = 40;
        const QK: usize = 16;
        const VD: usize = 24;
        const L0: usize = 8;
        let mut q = gen_x(H * T * QK, 71);
        let mut kk = gen_x(H * T * QK, 72);
        let v = gen_x(H * T * VD, 73);
        for row in 0..H * T {
            k::delta_net::l2norm_f32_inplace(&mut q[row * QK..(row + 1) * QK], 1e-6);
            k::delta_net::l2norm_f32_inplace(&mut kk[row * QK..(row + 1) * QK], 1e-6);
        }
        let g: Vec<f32> = gen_x(H * T, 74).iter().map(|x| -0.05 - 0.25 * x.abs()).collect();
        let beta: Vec<f32> = gen_x(H * T, 75).iter().map(|x| 0.1 + 0.8 * x.abs()).collect();
        let mut cpu_out = vec![0f32; H * T * VD];
        let mut cpu_state = vec![0f32; H * QK * VD];
        for h in 0..H {
            k::delta_net::delta_rule_prefill_chunked_with_chunk(
                &q[h * T * QK..(h + 1) * T * QK],
                &kk[h * T * QK..(h + 1) * T * QK],
                &v[h * T * VD..(h + 1) * T * VD],
                &g[h * T..(h + 1) * T],
                &beta[h * T..(h + 1) * T],
                &mut cpu_state[h * QK * VD..(h + 1) * QK * VD],
                &mut cpu_out[h * T * VD..(h + 1) * T * VD],
                T,
                QK,
                VD,
                L0,
            );
        }
        let mut gpu_out = vec![0f32; H * T * VD];
        let mut gpu_state = vec![0f32; H * QK * VD];
        match mk::delta_net_chunk_f32(
            &q, &kk, &v, &g, &beta, &mut gpu_state, &mut gpu_out, H, T, QK, VD, L0,
        ) {
            Ok(()) => {
                let mut gpu_all = gpu_out.clone();
                gpu_all.extend_from_slice(&gpu_state);
                let mut cpu_all = cpu_out.clone();
                cpu_all.extend_from_slice(&cpu_state);
                cu_grade("ssm:delta_net", &gpu_all, &cpu_all, 0.999, 0.02, &mut counts);
            }
            Err(e) => {
                cu_emit("ssm:delta_net", "KERNEL_ERR", &format!("{e:?}"));
                *counts.entry("KERNEL_ERR").or_default() += 1;
            }
        }
    }

    // ---- Device-resident forward kernels: RoPE / SwiGLU / embedding / flash ----
    // The Metal twins of the CUDA device-resident probes (previously ungraded by
    // ANY harness). Same geometry, CPU references, and tolerances as the CUDA
    // section, so verdicts line up across backends. These need a raw MlxStream
    // (the matvec cache owns its own private one); skip the lot if one can't be
    // created. Emits rope:f32 / silu_mul:f32 / embedding:f32 / attn:decode /
    // attn:prefill. WRITE-BLIND (authored off-Apple) — first real check is here.
    if let Some(fs) = mk::MlxStream::create(0) {
        // RoPE
        {
            let n_heads = 8usize;
            let head_dim = 64usize;
            let pos = 37usize;
            let qk = gen_x(n_heads * head_dim, 31);
            let inv_freq: Vec<f32> = (0..head_dim / 2)
                .map(|j| (10000f32).powf(-2.0 * j as f32 / head_dim as f32))
                .collect();
            let mut cpu = qk.clone();
            ref_rope(&mut cpu, n_heads, head_dim, pos, &inv_freq);
            if let (Some(mut qb), Some(fb)) =
                (mk_upload_f32(&fs, &qk), mk_upload_f32(&fs, &inv_freq))
            {
                // SAFETY: qb/fb are live device f32 buffers on `fs`.
                let res = unsafe {
                    mk::rope_f32(
                        &fs,
                        qb.as_mut_ptr() as *mut f32,
                        n_heads,
                        head_dim,
                        pos,
                        fb.as_ptr() as *const f32,
                    )
                };
                match res {
                    Ok(()) => cu_grade(
                        "rope:f32",
                        &mk_download_f32(&qb, n_heads * head_dim),
                        &cpu,
                        0.9999,
                        0.01,
                        &mut counts,
                    ),
                    Err(e) => {
                        cu_emit("rope:f32", "KERNEL_ERR", &format!("{e:?}"));
                        *counts.entry("KERNEL_ERR").or_default() += 1;
                    }
                }
            }
        }

        // SwiGLU (silu(x) * y)
        {
            let n = 4096usize;
            let x = gen_x(n, 41);
            let y = gen_x(n, 43);
            let cpu = ref_silu_mul(&x, &y);
            if let (Some(xb), Some(yb), Some(mut ob)) = (
                mk_upload_f32(&fs, &x),
                mk_upload_f32(&fs, &y),
                mk::MlxDeviceBuffer::alloc(&fs, n * 4),
            ) {
                // SAFETY: three live device f32 buffers of length n on `fs`.
                let res = unsafe {
                    mk::silu_mul_f32(
                        &fs,
                        xb.as_ptr() as *const f32,
                        yb.as_ptr() as *const f32,
                        ob.as_mut_ptr() as *mut f32,
                        n,
                    )
                };
                match res {
                    Ok(()) => cu_grade(
                        "silu_mul:f32",
                        &mk_download_f32(&ob, n),
                        &cpu,
                        0.9999,
                        0.01,
                        &mut counts,
                    ),
                    Err(e) => {
                        cu_emit("silu_mul:f32", "KERNEL_ERR", &format!("{e:?}"));
                        *counts.entry("KERNEL_ERR").or_default() += 1;
                    }
                }
            }
        }

        // Embedding lookup (device ids; row < 0 ⇒ zeros)
        {
            let vocab = 100usize;
            let d = 128usize;
            let table = gen_x(vocab * d, 51);
            let ids: Vec<i32> = [5i32, 0, 99, -1, 42].to_vec();
            let n_ids = ids.len();
            let mut cpu = vec![0f32; n_ids * d];
            for (i, &row) in ids.iter().enumerate() {
                if row >= 0 {
                    cpu[i * d..(i + 1) * d]
                        .copy_from_slice(&table[row as usize * d..(row as usize + 1) * d]);
                }
            }
            let id_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const u8, n_ids * 4) };
            if let (Some(tb), Some(ib), Some(mut ob)) = (
                mk_upload_f32(&fs, &table),
                mk::MlxDeviceBuffer::from_host(&fs, id_bytes),
                mk::MlxDeviceBuffer::alloc(&fs, n_ids * d * 4),
            ) {
                // SAFETY: table [vocab*d] f32, ids [n_ids] i32, out [n_ids*d]
                // f32, all live device buffers on `fs`.
                let res = unsafe {
                    mk::embedding_lookup_f32(
                        &fs,
                        tb.as_ptr() as *const f32,
                        ib.as_ptr() as *const std::os::raw::c_int,
                        ob.as_mut_ptr() as *mut f32,
                        n_ids,
                        d,
                    )
                };
                match res {
                    Ok(()) => cu_grade(
                        "embedding:f32",
                        &mk_download_f32(&ob, n_ids * d),
                        &cpu,
                        0.99999,
                        0.0001,
                        &mut counts,
                    ),
                    Err(e) => {
                        cu_emit("embedding:f32", "KERNEL_ERR", &format!("{e:?}"));
                        *counts.entry("KERNEL_ERR").or_default() += 1;
                    }
                }
            }
        }

        // FlashAttention decode + prefill (f32). Each kernel runs THREE cases
        // folded into its single verdict: passthrough (window=0/softcap=0 — must
        // stay bit-identical), a sliding-window case, and the Gemma2 softcap (50.0).
        // The CPU reference applies the SAME (window, softcap) via the fused
        // `gqa_attention_flash_{decode,prefill}_capped_windowed`, so a kernel that
        // ignores either uniform fails the verdict (fail-closed). WRITE-BLIND here
        // (Metal is Apple-only) — validated on-device later via `--metal-parity`.
        {
            let (n_heads, n_kv_heads, head_dim, max_ctx, kv_len) =
                (8usize, 2usize, 64usize, 128usize, 40usize);
            let q = gen_x(n_heads * head_dim, 61);
            let kc = gen_x(n_kv_heads * max_ctx * head_dim, 63);
            let vc = gen_x(n_kv_heads * max_ctx * head_dim, 67);
            // kv_len=40 ⇒ window=16 genuinely trims to the last 16 cached positions.
            let dec_cases: [(u32, f32); 3] = [(0, 0.0), (16, 0.0), (0, 50.0)];
            if let (Some(qb), Some(kb), Some(vb), Some(mut ob)) = (
                mk_upload_f32(&fs, &q),
                mk_upload_f32(&fs, &kc),
                mk_upload_f32(&fs, &vc),
                mk::MlxDeviceBuffer::alloc(&fs, n_heads * head_dim * 4),
            ) {
                let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
                let mut err: Option<String> = None;
                for (window, softcap) in dec_cases {
                    let mut cpu = vec![0f32; n_heads * head_dim];
                    k::gqa_attention_flash_decode_capped_windowed(
                        &q, &kc, &vc, &mut cpu,
                        n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                        window as usize, softcap,
                    );
                    // SAFETY: q/out [n_heads*head_dim], k/v [n_kv_heads*max_ctx*
                    // head_dim] device f32 on `fs`.
                    let res = unsafe {
                        mk::flash_attn_decode_f32(
                            &fs, qb.as_ptr() as *const f32, kb.as_ptr() as *const f32,
                            vb.as_ptr() as *const f32, ob.as_mut_ptr() as *mut f32,
                            n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                            window as i32, softcap,
                        )
                    };
                    if let Err(e) = res {
                        err = Some(format!("{e:?}"));
                        break;
                    }
                    let (cos, max_rel) = compare(&mk_download_f32(&ob, n_heads * head_dim), &cpu);
                    worst_cos = worst_cos.min(cos);
                    worst_rel = worst_rel.max(max_rel);
                }
                match err {
                    Some(e) => {
                        cu_emit("attn:decode", "KERNEL_ERR", &e);
                        *counts.entry("KERNEL_ERR").or_default() += 1;
                    }
                    None => {
                        let key: &'static str =
                            if worst_cos > 0.999 && worst_rel < 0.02 { "OK" } else { "MISCOMPUTE" };
                        *counts.entry(key).or_default() += 1;
                        cu_emit(
                            "attn:decode",
                            key,
                            &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa16+cap50)"),
                        );
                    }
                }
            }

            let (kv_base, n_new) = (10usize, 6usize);
            let qp = gen_x(n_new * n_heads * head_dim, 71);
            // Prefill queries reach abs pos 15, so window=4 genuinely trims.
            let pre_cases: [(u32, f32); 3] = [(0, 0.0), (4, 0.0), (0, 50.0)];
            if let (Some(qb), Some(kb), Some(vb), Some(mut ob)) = (
                mk_upload_f32(&fs, &qp),
                mk_upload_f32(&fs, &kc),
                mk_upload_f32(&fs, &vc),
                mk::MlxDeviceBuffer::alloc(&fs, n_new * n_heads * head_dim * 4),
            ) {
                let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
                let mut err: Option<String> = None;
                for (window, softcap) in pre_cases {
                    let mut cpu_p = vec![0f32; n_new * n_heads * head_dim];
                    k::gqa_attention_flash_prefill_capped_windowed(
                        &qp, &kc, &vc, &mut cpu_p,
                        n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
                        window as usize, softcap,
                    );
                    // SAFETY: q/out [n_new*n_heads*head_dim], k/v caches device f32.
                    let res = unsafe {
                        mk::flash_attn_prefill_f32(
                            &fs, qb.as_ptr() as *const f32, kb.as_ptr() as *const f32,
                            vb.as_ptr() as *const f32, ob.as_mut_ptr() as *mut f32,
                            n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
                            window as i32, softcap,
                        )
                    };
                    if let Err(e) = res {
                        err = Some(format!("{e:?}"));
                        break;
                    }
                    let (cos, max_rel) =
                        compare(&mk_download_f32(&ob, n_new * n_heads * head_dim), &cpu_p);
                    worst_cos = worst_cos.min(cos);
                    worst_rel = worst_rel.max(max_rel);
                }
                match err {
                    Some(e) => {
                        cu_emit("attn:prefill", "KERNEL_ERR", &e);
                        *counts.entry("KERNEL_ERR").or_default() += 1;
                    }
                    None => {
                        let key: &'static str =
                            if worst_cos > 0.999 && worst_rel < 0.02 { "OK" } else { "MISCOMPUTE" };
                        *counts.entry(key).or_default() += 1;
                        cu_emit(
                            "attn:prefill",
                            key,
                            &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa4+cap50)"),
                        );
                    }
                }
            }
        }

        // ---- Quantized-KV FlashAttention decode + prefill ----
        // Metal twins of the CUDA quant-KV flash probes, with the IDENTICAL
        // SAME-QUANT reference: quantize K/V to each format's block layout,
        // dequantize the exact packed bytes on the CPU, run the f32 flash
        // reference on THAT — so the probe measures kernel correctness, not KV
        // quantization loss (a full-precision ref would false-flag correct
        // kernels on relative error). Same QUANT_KV_FORMATS set + per-format
        // tolerances as CUDA. Previously ungraded on Metal.
        {
            let (n_heads, n_kv_heads, head_dim, max_ctx, kv_len) =
                (8usize, 2usize, 64usize, 128usize, 40usize);
            let (kv_base, n_new) = (10usize, 6usize);
            let q = gen_x(n_heads * head_dim, 61);
            let kc = gen_x(n_kv_heads * max_ctx * head_dim, 63);
            let vc = gen_x(n_kv_heads * max_ctx * head_dim, 67);
            let qp = gen_x(n_new * n_heads * head_dim, 71);
            // Decode reads kv_len rows; prefill reads kv_base+n_new. Quantize up
            // to the larger so one packed cache serves both readers.
            let upto = kv_len.max(kv_base + n_new);
            for fmt in QUANT_KV_FORMATS {
                let dname = format!("attn:decode_{}", fmt.name);
                let pname = format!("attn:prefill_{}", fmt.name);
                if head_dim % fmt.block_elems != 0 {
                    cu_emit(&dname, "SKIP", "head-dim-not-block-aligned");
                    cu_emit(&pname, "SKIP", "head-dim-not-block-aligned");
                    *counts.entry("SKIP").or_default() += 2;
                    continue;
                }
                let kp = quantize_kv_cache(fmt, &kc, n_kv_heads, head_dim, max_ctx, upto);
                let vp = quantize_kv_cache(fmt, &vc, n_kv_heads, head_dim, max_ctx, upto);
                let kc_dq = dequant_kv_cache(fmt, &kp, n_kv_heads, head_dim, max_ctx, upto);
                let vc_dq = dequant_kv_cache(fmt, &vp, n_kv_heads, head_dim, max_ctx, upto);

                // decode — fold passthrough + sliding-window (kv_len=40 ⇒ window=16
                // trims) + Gemma2 soft-cap into the one `attn:decode_<fmt>` verdict,
                // via the `_capped_windowed` f32 ref on the dequantized cache (at
                // (0,0.0) it equals `ref_flash_decode`).
                if let (Some(qb), Some(kb), Some(vb), Some(mut ob)) = (
                    mk_upload_f32(&fs, &q),
                    mk::MlxDeviceBuffer::from_host(&fs, &kp),
                    mk::MlxDeviceBuffer::from_host(&fs, &vp),
                    mk::MlxDeviceBuffer::alloc(&fs, n_heads * head_dim * 4),
                ) {
                    let dec_cases: [(u32, f32); 3] = [(0, 0.0), (16, 0.0), (0, 50.0)];
                    let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
                    let mut err: Option<String> = None;
                    for (window, softcap) in dec_cases {
                        let mut cpu_dec = vec![0f32; n_heads * head_dim];
                        k::gqa_attention_flash_decode_capped_windowed(
                            &q, &kc_dq, &vc_dq, &mut cpu_dec,
                            n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                            window as usize, softcap,
                        );
                        // SAFETY: q/out F32 [n_heads*head_dim]; k/v packed device
                        // buffers [n_kv_heads*max_ctx*bytes_per_row] on `fs`.
                        let res = unsafe {
                            let q = qb.as_ptr() as *const f32;
                            let k = kb.as_ptr();
                            let v = vb.as_ptr();
                            let o = ob.as_mut_ptr() as *mut f32;
                            match fmt.name {
                                "mxfp4" => mk::flash_attn_decode_mxfp4(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len, window as i32, softcap),
                                "mxfp6" => mk::flash_attn_decode_mxfp6(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len, window as i32, softcap),
                                "mxfp8" => mk::flash_attn_decode_mxfp8(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len, window as i32, softcap),
                                "nvfp4" => mk::flash_attn_decode_nvfp4(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len, window as i32, softcap),
                                "q4_0" => mk::flash_attn_decode_q4_0(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len, window as i32, softcap),
                                _ => unreachable!(),
                            }
                        };
                        if let Err(e) = res {
                            err = Some(format!("{e:?}"));
                            break;
                        }
                        let (cos, max_rel) = compare(&mk_download_f32(&ob, n_heads * head_dim), &cpu_dec);
                        worst_cos = worst_cos.min(cos);
                        worst_rel = worst_rel.max(max_rel);
                    }
                    match err {
                        Some(e) => {
                            cu_emit(&dname, "KERNEL_ERR", &e);
                            *counts.entry("KERNEL_ERR").or_default() += 1;
                        }
                        None => {
                            let key: &'static str = if worst_cos > fmt.cos_min && worst_rel < fmt.rel_max {
                                "OK"
                            } else {
                                "MISCOMPUTE"
                            };
                            *counts.entry(key).or_default() += 1;
                            cu_emit(
                                &dname,
                                key,
                                &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa16+cap50)"),
                            );
                        }
                    }
                }

                // prefill — queries reach abs pos 15, so window=4 genuinely trims.
                if let (Some(qb), Some(kb), Some(vb), Some(mut ob)) = (
                    mk_upload_f32(&fs, &qp),
                    mk::MlxDeviceBuffer::from_host(&fs, &kp),
                    mk::MlxDeviceBuffer::from_host(&fs, &vp),
                    mk::MlxDeviceBuffer::alloc(&fs, n_new * n_heads * head_dim * 4),
                ) {
                    let pre_cases: [(u32, f32); 3] = [(0, 0.0), (4, 0.0), (0, 50.0)];
                    let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
                    let mut err: Option<String> = None;
                    for (window, softcap) in pre_cases {
                        let mut cpu_pre = vec![0f32; n_new * n_heads * head_dim];
                        k::gqa_attention_flash_prefill_capped_windowed(
                            &qp, &kc_dq, &vc_dq, &mut cpu_pre,
                            n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
                            window as usize, softcap,
                        );
                        // SAFETY: q/out F32 [n_new*n_heads*head_dim]; k/v packed as above.
                        let res = unsafe {
                            let q = qb.as_ptr() as *const f32;
                            let k = kb.as_ptr();
                            let v = vb.as_ptr();
                            let o = ob.as_mut_ptr() as *mut f32;
                            match fmt.name {
                                "mxfp4" => mk::flash_attn_prefill_mxfp4(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new, window as i32, softcap),
                                "mxfp6" => mk::flash_attn_prefill_mxfp6(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new, window as i32, softcap),
                                "mxfp8" => mk::flash_attn_prefill_mxfp8(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new, window as i32, softcap),
                                "nvfp4" => mk::flash_attn_prefill_nvfp4(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new, window as i32, softcap),
                                "q4_0" => mk::flash_attn_prefill_q4_0(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new, window as i32, softcap),
                                _ => unreachable!(),
                            }
                        };
                        if let Err(e) = res {
                            err = Some(format!("{e:?}"));
                            break;
                        }
                        let (cos, max_rel) =
                            compare(&mk_download_f32(&ob, n_new * n_heads * head_dim), &cpu_pre);
                        worst_cos = worst_cos.min(cos);
                        worst_rel = worst_rel.max(max_rel);
                    }
                    match err {
                        Some(e) => {
                            cu_emit(&pname, "KERNEL_ERR", &e);
                            *counts.entry("KERNEL_ERR").or_default() += 1;
                        }
                        None => {
                            let key: &'static str = if worst_cos > fmt.cos_min && worst_rel < fmt.rel_max {
                                "OK"
                            } else {
                                "MISCOMPUTE"
                            };
                            *counts.entry(key).or_default() += 1;
                            cu_emit(
                                &pname,
                                key,
                                &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa4+cap50)"),
                            );
                        }
                    }
                }
            }
        }

        // ---- Q8_0 KV FlashAttention decode + prefill (per-row-scale layout) ----
        // Metal twin of the CUDA Q8_0 KV probe: plain i8 slab + separate per-row
        // f32 scales, same-quant reference. Reuses the forward-pass MlxStream.
        {
            let (n_heads, n_kv_heads, head_dim, max_ctx, kv_len) =
                (8usize, 2usize, 64usize, 128usize, 40usize);
            let (kv_base, n_new) = (10usize, 6usize);
            let q = gen_x(n_heads * head_dim, 61);
            let kc = gen_x(n_kv_heads * max_ctx * head_dim, 63);
            let vc = gen_x(n_kv_heads * max_ctx * head_dim, 67);
            let qp = gen_x(n_new * n_heads * head_dim, 71);
            let upto = kv_len.max(kv_base + n_new);
            let (kslab, ksc) = quantize_kv_q8_0(&kc, n_kv_heads, head_dim, max_ctx, upto);
            let (vslab, vsc) = quantize_kv_q8_0(&vc, n_kv_heads, head_dim, max_ctx, upto);
            let kdq = dequant_kv_q8_0(&kslab, &ksc, n_kv_heads, head_dim, max_ctx, upto);
            let vdq = dequant_kv_q8_0(&vslab, &vsc, n_kv_heads, head_dim, max_ctx, upto);
            // decode — fold the three (window, softcap) cases into one verdict, with
            // the `_capped_windowed` f32 ref on the dequantized slab (at (0,0.0) it
            // equals `ref_flash_decode`). kv_len=40 ⇒ window=16 trims.
            if let (Some(qb), Some(kb), Some(vb), Some(ksb), Some(vsb), Some(mut ob)) = (
                mk_upload_f32(&fs, &q),
                mk::MlxDeviceBuffer::from_host(&fs, i8_as_bytes(&kslab)),
                mk::MlxDeviceBuffer::from_host(&fs, i8_as_bytes(&vslab)),
                mk_upload_f32(&fs, &ksc),
                mk_upload_f32(&fs, &vsc),
                mk::MlxDeviceBuffer::alloc(&fs, n_heads * head_dim * 4),
            ) {
                let dec_cases: [(u32, f32); 3] = [(0, 0.0), (16, 0.0), (0, 50.0)];
                let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
                let mut err: Option<String> = None;
                for (window, softcap) in dec_cases {
                    let mut cpu_dec = vec![0f32; n_heads * head_dim];
                    k::gqa_attention_flash_decode_capped_windowed(
                        &q, &kdq, &vdq, &mut cpu_dec,
                        n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                        window as usize, softcap,
                    );
                    // SAFETY: q/out F32; k/v i8 slabs; k/v scales F32 — all live
                    // device buffers on `fs`.
                    let res = unsafe {
                        mk::flash_attn_decode_q8_0(
                            &fs,
                            qb.as_ptr() as *const f32,
                            kb.as_ptr(),
                            vb.as_ptr(),
                            ksb.as_ptr() as *const f32,
                            vsb.as_ptr() as *const f32,
                            ob.as_mut_ptr() as *mut f32,
                            n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                            window as i32, softcap,
                        )
                    };
                    if let Err(e) = res {
                        err = Some(format!("{e:?}"));
                        break;
                    }
                    let (cos, max_rel) = compare(&mk_download_f32(&ob, n_heads * head_dim), &cpu_dec);
                    worst_cos = worst_cos.min(cos);
                    worst_rel = worst_rel.max(max_rel);
                }
                match err {
                    Some(e) => {
                        cu_emit("attn:decode_q8_0", "KERNEL_ERR", &e);
                        *counts.entry("KERNEL_ERR").or_default() += 1;
                    }
                    None => {
                        let key: &'static str =
                            if worst_cos > 0.999 && worst_rel < 0.05 { "OK" } else { "MISCOMPUTE" };
                        *counts.entry(key).or_default() += 1;
                        cu_emit(
                            "attn:decode_q8_0",
                            key,
                            &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa16+cap50)"),
                        );
                    }
                }
            }
            // prefill — queries reach abs pos 15, so window=4 genuinely trims.
            if let (Some(qb), Some(kb), Some(vb), Some(ksb), Some(vsb), Some(mut ob)) = (
                mk_upload_f32(&fs, &qp),
                mk::MlxDeviceBuffer::from_host(&fs, i8_as_bytes(&kslab)),
                mk::MlxDeviceBuffer::from_host(&fs, i8_as_bytes(&vslab)),
                mk_upload_f32(&fs, &ksc),
                mk_upload_f32(&fs, &vsc),
                mk::MlxDeviceBuffer::alloc(&fs, n_new * n_heads * head_dim * 4),
            ) {
                let pre_cases: [(u32, f32); 3] = [(0, 0.0), (4, 0.0), (0, 50.0)];
                let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
                let mut err: Option<String> = None;
                for (window, softcap) in pre_cases {
                    let mut cpu_pre = vec![0f32; n_new * n_heads * head_dim];
                    k::gqa_attention_flash_prefill_capped_windowed(
                        &qp, &kdq, &vdq, &mut cpu_pre,
                        n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
                        window as usize, softcap,
                    );
                    // SAFETY: as the decode arm, with q/out sized for n_new queries.
                    let res = unsafe {
                        mk::flash_attn_prefill_q8_0(
                            &fs,
                            qb.as_ptr() as *const f32,
                            kb.as_ptr(),
                            vb.as_ptr(),
                            ksb.as_ptr() as *const f32,
                            vsb.as_ptr() as *const f32,
                            ob.as_mut_ptr() as *mut f32,
                            n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
                            window as i32, softcap,
                        )
                    };
                    if let Err(e) = res {
                        err = Some(format!("{e:?}"));
                        break;
                    }
                    let (cos, max_rel) =
                        compare(&mk_download_f32(&ob, n_new * n_heads * head_dim), &cpu_pre);
                    worst_cos = worst_cos.min(cos);
                    worst_rel = worst_rel.max(max_rel);
                }
                match err {
                    Some(e) => {
                        cu_emit("attn:prefill_q8_0", "KERNEL_ERR", &e);
                        *counts.entry("KERNEL_ERR").or_default() += 1;
                    }
                    None => {
                        let key: &'static str =
                            if worst_cos > 0.999 && worst_rel < 0.05 { "OK" } else { "MISCOMPUTE" };
                        *counts.entry(key).or_default() += 1;
                        cu_emit(
                            "attn:prefill_q8_0",
                            key,
                            &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa4+cap50)"),
                        );
                    }
                }
            }
        }

        // ---- TQ (TurboQuant) KV FlashAttention decode + prefill ----
        // Metal twin of the CUDA TQ probe: WHT-packed slab + per-row scales +
        // `bits`, same-quant reference at TQ_PROBE_BITS. Reuses the MlxStream.
        {
            let (n_heads, n_kv_heads, head_dim, max_ctx, kv_len) =
                (8usize, 2usize, 64usize, 128usize, 40usize);
            let (kv_base, n_new) = (10usize, 6usize);
            let bits = TQ_PROBE_BITS;
            let q = gen_x(n_heads * head_dim, 61);
            let kc = gen_x(n_kv_heads * max_ctx * head_dim, 63);
            let vc = gen_x(n_kv_heads * max_ctx * head_dim, 67);
            let qp = gen_x(n_new * n_heads * head_dim, 71);
            let upto = kv_len.max(kv_base + n_new);
            let (kslab, ksc) = quantize_kv_tq(&kc, bits, n_kv_heads, head_dim, max_ctx, upto);
            let (vslab, vsc) = quantize_kv_tq(&vc, bits, n_kv_heads, head_dim, max_ctx, upto);
            let kdq = dequant_kv_tq(&kslab, &ksc, bits, n_kv_heads, head_dim, max_ctx, upto);
            let vdq = dequant_kv_tq(&vslab, &vsc, bits, n_kv_heads, head_dim, max_ctx, upto);
            // decode — fold the three (window, softcap) cases into one verdict, with
            // the `_capped_windowed` f32 ref on the dequantized slab (at (0,0.0) it
            // equals `ref_flash_decode`). kv_len=40 ⇒ window=16 trims.
            if let (Some(qb), Some(kb), Some(vb), Some(ksb), Some(vsb), Some(mut ob)) = (
                mk_upload_f32(&fs, &q),
                mk::MlxDeviceBuffer::from_host(&fs, &kslab),
                mk::MlxDeviceBuffer::from_host(&fs, &vslab),
                mk_upload_f32(&fs, &ksc),
                mk_upload_f32(&fs, &vsc),
                mk::MlxDeviceBuffer::alloc(&fs, n_heads * head_dim * 4),
            ) {
                let dec_cases: [(u32, f32); 3] = [(0, 0.0), (16, 0.0), (0, 50.0)];
                let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
                let mut err: Option<String> = None;
                for (window, softcap) in dec_cases {
                    let mut cpu_dec = vec![0f32; n_heads * head_dim];
                    k::gqa_attention_flash_decode_capped_windowed(
                        &q, &kdq, &vdq, &mut cpu_dec,
                        n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                        window as usize, softcap,
                    );
                    // SAFETY: q/out F32; k/v packed TQ slabs; k/v scales F32 — live
                    // device buffers on `fs`.
                    let res = unsafe {
                        mk::flash_attn_decode_tq(
                            &fs,
                            qb.as_ptr() as *const f32,
                            kb.as_ptr(),
                            vb.as_ptr(),
                            ksb.as_ptr() as *const f32,
                            vsb.as_ptr() as *const f32,
                            bits as u32,
                            ob.as_mut_ptr() as *mut f32,
                            n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                            window as i32, softcap,
                        )
                    };
                    if let Err(e) = res {
                        err = Some(format!("{e:?}"));
                        break;
                    }
                    let (cos, max_rel) = compare(&mk_download_f32(&ob, n_heads * head_dim), &cpu_dec);
                    worst_cos = worst_cos.min(cos);
                    worst_rel = worst_rel.max(max_rel);
                }
                match err {
                    Some(e) => {
                        cu_emit("attn:decode_tq", "KERNEL_ERR", &e);
                        *counts.entry("KERNEL_ERR").or_default() += 1;
                    }
                    None => {
                        let key: &'static str =
                            if worst_cos > 0.999 && worst_rel < 0.05 { "OK" } else { "MISCOMPUTE" };
                        *counts.entry(key).or_default() += 1;
                        cu_emit(
                            "attn:decode_tq",
                            key,
                            &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa16+cap50)"),
                        );
                    }
                }
            }
            // prefill — queries reach abs pos 15, so window=4 genuinely trims.
            if let (Some(qb), Some(kb), Some(vb), Some(ksb), Some(vsb), Some(mut ob)) = (
                mk_upload_f32(&fs, &qp),
                mk::MlxDeviceBuffer::from_host(&fs, &kslab),
                mk::MlxDeviceBuffer::from_host(&fs, &vslab),
                mk_upload_f32(&fs, &ksc),
                mk_upload_f32(&fs, &vsc),
                mk::MlxDeviceBuffer::alloc(&fs, n_new * n_heads * head_dim * 4),
            ) {
                let pre_cases: [(u32, f32); 3] = [(0, 0.0), (4, 0.0), (0, 50.0)];
                let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
                let mut err: Option<String> = None;
                for (window, softcap) in pre_cases {
                    let mut cpu_pre = vec![0f32; n_new * n_heads * head_dim];
                    k::gqa_attention_flash_prefill_capped_windowed(
                        &qp, &kdq, &vdq, &mut cpu_pre,
                        n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
                        window as usize, softcap,
                    );
                    // SAFETY: as the decode arm, q/out sized for n_new queries.
                    let res = unsafe {
                        mk::flash_attn_prefill_tq(
                            &fs,
                            qb.as_ptr() as *const f32,
                            kb.as_ptr(),
                            vb.as_ptr(),
                            ksb.as_ptr() as *const f32,
                            vsb.as_ptr() as *const f32,
                            bits as u32,
                            ob.as_mut_ptr() as *mut f32,
                            n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
                            window as i32, softcap,
                        )
                    };
                    if let Err(e) = res {
                        err = Some(format!("{e:?}"));
                        break;
                    }
                    let (cos, max_rel) =
                        compare(&mk_download_f32(&ob, n_new * n_heads * head_dim), &cpu_pre);
                    worst_cos = worst_cos.min(cos);
                    worst_rel = worst_rel.max(max_rel);
                }
                match err {
                    Some(e) => {
                        cu_emit("attn:prefill_tq", "KERNEL_ERR", &e);
                        *counts.entry("KERNEL_ERR").or_default() += 1;
                    }
                    None => {
                        let key: &'static str =
                            if worst_cos > 0.999 && worst_rel < 0.05 { "OK" } else { "MISCOMPUTE" };
                        *counts.entry(key).or_default() += 1;
                        cu_emit(
                            "attn:prefill_tq",
                            key,
                            &format!("cos={worst_cos:.6} max_rel={worst_rel:.4} (pass+swa4+cap50)"),
                        );
                    }
                }
            }
        }

        // ---- Feature 1c: device-resident DECODE attention block ----
        // The orchestration probe (Metal twin of the SYCL `probe_decode_block`):
        // the WHOLE per-layer decode attention chain in ONE command buffer
        // (rsl_mlx_decode_attn_block_f32) vs a CPU reference computing the
        // identical sequence on identical inputs. The individual kernels (matvec /
        // flash / rmsnorm / rope) are already covered above, so this validates the
        // CHAINING, the f32-activation residency, and the F32 KV-mirror
        // scatter/read. The MLX mirror is F32 (unlike the SYCL block's f16
        // mirror), so the chain is bit-exact to the CPU reference up to float
        // reduction order — no f16 rounding anywhere. Q4_K weights, a small
        // realistic shape, a short pre-seeded KV history (so the flash spans
        // multiple positions). Same loose gate as the SYCL decode-block probe
        // (cos > 0.995, max_rel < 0.10) — the f32 mirror only makes passing
        // easier, so this won't false-pass a broken kernel.
        //
        // Like the SYCL probe, this exercises the FFN-INCLUSIVE path: it supplies
        // the dense SwiGLU FFN weights (gate/up/down) so the block runs the whole
        // dense layer on-device, and grades the FINAL `hidden` (attention + FFN
        // residual) vs a CPU reference computing attention AND the same SwiGLU FFN.
        {
            const HEAD_DIM: usize = 128;
            const N_HEADS: usize = 8;
            const N_KV: usize = 2;
            const D_MODEL: usize = 1024;
            const D_FF: usize = 2048; // SwiGLU FFN inner dim (256-aligned for Q4_K)
            const MAX_CTX: usize = 64;
            const POS: usize = 5; // 5 seeded history rows + this token => kv_len = 6
            const EPS: f32 = 1e-5;
            const THETA: f32 = 10000.0;
            let name = "attn:dev_resident_block_mlx";
            let d_q = N_HEADS * HEAD_DIM;
            let d_kv = N_KV * HEAD_DIM;
            let half = HEAD_DIM / 2;
            let kv_len = POS + 1;
            // Q4_K weights (M x K packed bytes): Wq [d_q x D_MODEL], Wk/Wv
            // [d_kv x D_MODEL], Wo [D_MODEL x d_q] — same synthesis + CPU matvec
            // the matvec probes use, so the per-row dequant matches the kernel.
            // The dense-FFN weights: Wgate/Wup [D_FF x D_MODEL], Wdown [D_MODEL x D_FF].
            let layout = LAYOUTS.iter().find(|l| l.name == "q4_k").expect("q4_k layout");
            let cpu_mv = cpu_matvec_for("q4_k");
            let wq = gen_quant_bytes(layout, d_q, D_MODEL, 0x1111);
            let wk = gen_quant_bytes(layout, d_kv, D_MODEL, 0x2222);
            let wv = gen_quant_bytes(layout, d_kv, D_MODEL, 0x3333);
            let wo = gen_quant_bytes(layout, D_MODEL, d_q, 0x4444);
            let wgate = gen_quant_bytes(layout, D_FF, D_MODEL, 0x5555);
            let wup = gen_quant_bytes(layout, D_FF, D_MODEL, 0x6666);
            let wdown = gen_quant_bytes(layout, D_MODEL, D_FF, 0x7777);
            let mut rng = XorShift(0xDEC0DE);
            let hidden0: Vec<f32> = (0..D_MODEL).map(|_| rng.f32_pm(0.6)).collect();
            let attn_norm: Vec<f32> = (0..D_MODEL).map(|_| 0.5 + rng.f32_pm(0.2)).collect();
            let ffn_norm: Vec<f32> = (0..D_MODEL).map(|_| 0.5 + rng.f32_pm(0.2)).collect();
            let inv_freq: Vec<f32> = (0..half)
                .map(|j| 1.0f32 / THETA.powf(2.0 * j as f32 / HEAD_DIM as f32))
                .collect();
            // F32 KV history [0,POS) per kv-head (head-major) — exact, no
            // rounding (the MLX mirror is f32). Rows >= POS stay zero; the block
            // writes row POS itself.
            let mirror_elems = N_KV * MAX_CTX * HEAD_DIM;
            let mut k_hist = vec![0f32; mirror_elems];
            let mut v_hist = vec![0f32; mirror_elems];
            for h in 0..N_KV {
                for t in 0..POS {
                    for d in 0..HEAD_DIM {
                        let idx = (h * MAX_CTX + t) * HEAD_DIM + d;
                        k_hist[idx] = rng.f32_pm(0.6);
                        v_hist[idx] = rng.f32_pm(0.6);
                    }
                }
            }
            // ---- CPU reference (identical op sequence, all f32) ----
            let rope = |qk: &mut [f32], nh: usize| {
                for hh in 0..nh {
                    let base = hh * HEAD_DIM;
                    for j in 0..half {
                        let angle = POS as f32 * inv_freq[j];
                        let (s, c) = angle.sin_cos();
                        let x0 = qk[base + j];
                        let x1 = qk[base + j + half];
                        qk[base + j] = x0 * c - x1 * s;
                        qk[base + j + half] = x0 * s + x1 * c;
                    }
                }
            };
            let mut hnorm = vec![0f32; D_MODEL];
            k::rmsnorm_f32_row(&hidden0, &attn_norm, &mut hnorm, EPS);
            let mut q_ref = vec![0f32; d_q];
            let mut k_ref = vec![0f32; d_kv];
            let mut v_ref = vec![0f32; d_kv];
            cpu_mv(&wq, &hnorm, &mut q_ref, d_q, D_MODEL);
            cpu_mv(&wk, &hnorm, &mut k_ref, d_kv, D_MODEL);
            cpu_mv(&wv, &hnorm, &mut v_ref, d_kv, D_MODEL);
            rope(&mut q_ref, N_HEADS);
            rope(&mut k_ref, N_KV);
            // The full f32 KV cache the reference flash reads: history [0,POS) +
            // the new (post-RoPE K / plain V) at POS — matching exactly what the
            // device mirror holds after the block's KV-write.
            let mut kcache = k_hist.clone();
            let mut vcache = v_hist.clone();
            for h in 0..N_KV {
                for d in 0..HEAD_DIM {
                    let idx = (h * MAX_CTX + POS) * HEAD_DIM + d;
                    kcache[idx] = k_ref[h * HEAD_DIM + d];
                    vcache[idx] = v_ref[h * HEAD_DIM + d];
                }
            }
            // Allocate device weights + the F32 KV mirror ONCE. q_ref/k_ref/v_ref/
            // kcache/vcache above are case-independent; the per-case loop only
            // recomputes the attention + downstream reference and resets
            // `hidden_dev`. The mirror history [0,POS) is uploaded once and the
            // block only overwrites row POS (deterministic from the reset hidden),
            // so no re-seed is needed between cases.
            if let (
                Some(bwq), Some(bwk), Some(bwv), Some(bwo),
                Some(bwg), Some(bwu), Some(bwd), Some(mut kmir), Some(mut vmir),
            ) = (
                mk::MlxDeviceBuffer::from_host(&fs, &wq),
                mk::MlxDeviceBuffer::from_host(&fs, &wk),
                mk::MlxDeviceBuffer::from_host(&fs, &wv),
                mk::MlxDeviceBuffer::from_host(&fs, &wo),
                mk::MlxDeviceBuffer::from_host(&fs, &wgate),
                mk::MlxDeviceBuffer::from_host(&fs, &wup),
                mk::MlxDeviceBuffer::from_host(&fs, &wdown),
                mk_upload_f32(&fs, &k_hist),
                mk_upload_f32(&fs, &v_hist),
            ) {
                // Three cases drive the ONE `attn:dev_resident_block_mlx` verdict:
                //   (0, 0.0)            passthrough — must stay bit-identical.
                //   (MLX_DEV_WINDOW, 0.0) SWA — kv_len=6, so window=3 trims.
                //   (0, 50.0)           Gemma2 attention-logit softcap (real value).
                // The CPU reference applies the same via `gqa_attention_one_step_
                // capped_windowed`; grade the residual DELTA so the shared input
                // residual can't mask a window/cap miscompute. WRITE-BLIND (Metal
                // is Apple-only) — validated on-device later via `--metal-parity`.
                const MLX_DEV_WINDOW: usize = 3;
                let cases: [(u32, f32); 3] = [(0, 0.0), (MLX_DEV_WINDOW as u32, 0.0), (0, 50.0)];
                let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
                // Set on the FIRST non-OK outcome, which breaks the loop.
                let mut early: Option<(&'static str, String)> = None;
                for (window, softcap) in cases {
                    // CPU reference attention + o-proj + residual + post-norm + FFN.
                    let mut attn_ref = vec![0f32; d_q];
                    k::gqa_attention_one_step_capped_windowed(
                        &q_ref, &kcache, &vcache, &mut attn_ref,
                        N_HEADS, N_KV, HEAD_DIM, MAX_CTX, kv_len, window as usize, softcap,
                    );
                    let mut o_ref = vec![0f32; D_MODEL];
                    cpu_mv(&wo, &attn_ref, &mut o_ref, D_MODEL, d_q);
                    let mut hidden_ref = hidden0.clone();
                    for i in 0..D_MODEL {
                        hidden_ref[i] += o_ref[i];
                    }
                    let mut hnorm2_ref = vec![0f32; D_MODEL];
                    k::rmsnorm_f32_row(&hidden_ref, &ffn_norm, &mut hnorm2_ref, EPS);
                    let mut gate_ref = vec![0f32; D_FF];
                    let mut up_ref = vec![0f32; D_FF];
                    cpu_mv(&wgate, &hnorm2_ref, &mut gate_ref, D_FF, D_MODEL);
                    cpu_mv(&wup, &hnorm2_ref, &mut up_ref, D_FF, D_MODEL);
                    let mut act_ref = vec![0f32; D_FF];
                    k::silu_mul_f32(&gate_ref, &up_ref, &mut act_ref);
                    let mut ffn_out_ref = vec![0f32; D_MODEL];
                    cpu_mv(&wdown, &act_ref, &mut ffn_out_ref, D_MODEL, D_FF);
                    for i in 0..D_MODEL {
                        hidden_ref[i] += ffn_out_ref[i];
                    }
                    if !hidden_ref.iter().all(|v| v.is_finite()) {
                        early = Some(("SKIP", "cpu-reference-nonfinite".to_string()));
                        break;
                    }
                    let mut hidden_dev = hidden0.clone();
                    let mut hnorm2_dev = vec![0f32; D_MODEL];
                    let mut k_row_dev = vec![0f32; d_kv];
                    let mut v_row_dev = vec![0f32; d_kv];
                    // SAFETY: weights/FFN weights/mirror are live device buffers on
                    // `fs`; the host slices are sized to the dims; kq/kk/kv/ko + FFN
                    // codes = 0 (Q4_K); the block syncs before returning. The dense-FFN
                    // tuple exercises the FFN path. window/softcap = this case.
                    let res = unsafe {
                        mk::decode_attn_block_f32(
                            &fs,
                            bwq.as_ptr(), 0,
                            bwk.as_ptr(), 0,
                            bwv.as_ptr(), 0,
                            bwo.as_ptr(), 0,
                            Some((bwg.as_ptr(), 0, bwu.as_ptr(), 0, bwd.as_ptr(), 0, D_FF as u32)),
                            attn_norm.as_ptr(),
                            ffn_norm.as_ptr(),
                            inv_freq.as_ptr(),
                            hidden_dev.as_mut_ptr(),
                            hnorm2_dev.as_mut_ptr(),
                            k_row_dev.as_mut_ptr(),
                            v_row_dev.as_mut_ptr(),
                            kmir.as_mut_ptr(),
                            vmir.as_mut_ptr(),
                            N_HEADS as u32, N_KV as u32, HEAD_DIM as u32, D_MODEL as u32,
                            POS as u32, MAX_CTX as u32, kv_len as u32, EPS,
                            window as i32, softcap,
                        )
                    };
                    match res {
                        // `ffn_ran` must be true — dispatchable Q4_K FFN weights are
                        // supplied. Grade the residual DELTA so the shared input
                        // residual can't mask a miscompute.
                        Ok((true, ffn_ran)) => {
                            if !ffn_ran {
                                early = Some(("MISCOMPUTE", "ffn-leg-did-not-run".to_string()));
                                break;
                            }
                            let gpu_delta: Vec<f32> =
                                hidden_dev.iter().zip(&hidden0).map(|(h, h0)| h - h0).collect();
                            let cpu_delta: Vec<f32> =
                                hidden_ref.iter().zip(&hidden0).map(|(h, h0)| h - h0).collect();
                            let (cos, max_rel) = compare(&gpu_delta, &cpu_delta);
                            worst_cos = worst_cos.min(cos);
                            worst_rel = worst_rel.max(max_rel);
                        }
                        Ok((false, _)) => {
                            early = Some(("SKIP", "block-declined".to_string()));
                            break;
                        }
                        Err(e) => {
                            early = Some(("KERNEL_ERR", format!("{e:?}")));
                            break;
                        }
                    }
                }
                match early {
                    Some((verdict, detail)) => {
                        cu_emit(name, verdict, &detail);
                        *counts.entry(verdict).or_default() += 1;
                    }
                    None => {
                        let key: &'static str =
                            if worst_cos > 0.995 && worst_rel < 0.10 { "OK" } else { "MISCOMPUTE" };
                        *counts.entry(key).or_default() += 1;
                        cu_emit(
                            name,
                            key,
                            &format!(
                                "cos={worst_cos:.6} max_rel={worst_rel:.4} \
                                 (pass+swa{MLX_DEV_WINDOW}+cap50)"
                            ),
                        );
                    }
                }
            }
        }
    }

    println!();
    let summary: Vec<String> = counts.iter().map(|(k, v)| format!("{k}={v}")).collect();
    println!("summary: {}", summary.join("  "));
    let bad = counts.get("MISCOMPUTE").copied().unwrap_or(0)
        + counts.get("KERNEL_ERR").copied().unwrap_or(0);
    if bad > 0 {
        println!(
            "\n{bad} Metal kernel(s) diverged from the CPU reference or failed to launch — \
             investigate before trusting the Metal backend (these kernels were authored \
             write-blind on a non-Apple host; this harness is their first real check)."
        );
    } else if counts.get("OK").copied().unwrap_or(0) > 0 {
        println!("\nall exercised Metal kernels match the CPU reference on this device.");
    }
    Ok(())
}

// =====================================================================
// NPU compute-offload parity (`rustllama doctor --npu-parity`)
// =====================================================================

/// Grade the NPU MatMul offloads (Phase 5) against the CPU reference — Intel
/// (OpenVINO) and AMD (ONNX Runtime + VitisAI). Each SKIPs cleanly off its
/// vendor's NPU / runtime. The first real numeric validation of the write-blind
/// offload paths (the dispatch verdicts auto-enable each only on OK).
pub fn run_npu_parity() -> anyhow::Result<()> {
    fn report(label: &str, r: rustllama_npu::NpuParity, verdict_name: &str) -> &'static str {
        if !r.supported {
            println!("  {label:<22} SKIP        no NPU device / runtime not loaded");
            return "SKIP";
        }
        let v = if r.pass { "OK" } else { "MISCOMPUTE" };
        println!("  {label:<22} {v:<11} cos={:.6} max_rel={:.4}", r.cos, r.max_rel);
        let _ = verdict_name;
        if r.pass {
            "OK"
        } else {
            "MISCOMPUTE"
        }
    }

    println!("NPU compute-offload parity (Phase 5)");
    let intel = report(
        "intel matmul:f32",
        rustllama_npu::npu_matmul_parity(),
        "npu:matmul_f32",
    );
    let amd = report(
        "amd   matmul:f32",
        rustllama_npu::amd_npu_matmul_parity(),
        "npu:amd_matmul_f32",
    );

    let any_ok = intel == "OK" || amd == "OK";
    let any_bad = intel == "MISCOMPUTE" || amd == "MISCOMPUTE";
    if any_bad {
        println!(
            "\nsummary: MISCOMPUTE — an NPU MatMul offload disagrees with the CPU \
             reference; that dispatch stays fail-closed (offload off)."
        );
    } else if any_ok {
        println!(
            "\nsummary: OK — the NPU MatMul offload matches the CPU reference; the \
             dispatch verdict can auto-enable it."
        );
    } else {
        println!("\nsummary: SKIP (no NPU offload available on this host)");
    }
    Ok(())
}
