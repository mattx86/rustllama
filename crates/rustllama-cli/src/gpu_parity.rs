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
    // XMX/DPAS bf16 tensor-core GEMM (Arc Xe-HPG / PVC Xe-HPC). SKIPs unless
    // the device is XMX-capable AND the DLL was built -DRSL_SYCL_XMX; on Iris
    // Xe (Xe-LP) it reports SKIP. Last so it banks after the matvec families.
    v.push("xmx:gemm".to_string());
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
        "fused" => probe_fused(&stream, name, rest, started),
        "attn" => probe_attn(&stream, name, rest, started),
        "xmx" => probe_xmx(&stream, name, started),
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

fn probe_fused(stream: &sk::SyclStream, name: &str, dtype: &str, started: Instant) {
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

fn probe_attn(stream: &sk::SyclStream, name: &str, variant: &str, started: Instant) {
    // Quant-KV variants (decode_mxfp4 … prefill_nvfp4) run the F32-Q /
    // packed-K/V flash kernels and grade against the full-precision CPU
    // reference; the f32 variants (v1/v2/v3) below stay on the f16 path.
    // Split here so each keeps its own buffer setup (f16 vs f32 + packed).
    // Q8_0 KV first: its per-row-scale layout isn't a `QuantKvFormat`, so it has
    // its own probe (routed before the block-format check, which would miss it).
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
        let mut cpu_out = vec![0f32; AT_HEADS * AT_HEAD_DIM];
        k::gqa_attention_one_step(
            &q,
            &kcache,
            &vcache,
            &mut cpu_out,
            AT_HEADS,
            AT_KV_HEADS,
            AT_HEAD_DIM,
            AT_MAX_CTX,
            AT_KV_LEN,
        );
        let alloc = (|| -> sk::Result<_> {
            let qb = to_f16_buf(stream, &q)?;
            let kb = to_f16_buf(stream, &kcache)?;
            let vb = to_f16_buf(stream, &vcache)?;
            let ob = sk::SyclSharedBuffer::<u16>::alloc(stream, cpu_out.len())?;
            Ok((qb, kb, vb, ob))
        })();
        let (qb, kb, vb, mut ob) = match alloc {
            Ok(t) => t,
            Err(_) => {
                emit(name, "KERNEL_ERR", "usm-alloc-failed");
                return;
            }
        };
        let res = match v {
            "v1" => sk::flash_attn_decode_usm(
                stream,
                &qb,
                &kb,
                &vb,
                &mut ob,
                AT_HEADS as u32,
                AT_KV_HEADS as u32,
                AT_HEAD_DIM as u32,
                AT_MAX_CTX as u32,
                AT_KV_LEN as u32,
            ),
            "v2" => sk::flash_attn_decode_v2_usm(
                stream,
                &qb,
                &kb,
                &vb,
                &mut ob,
                AT_HEADS as u32,
                AT_KV_HEADS as u32,
                AT_HEAD_DIM as u32,
                AT_MAX_CTX as u32,
                AT_KV_LEN as u32,
            ),
            "v3" => sk::flash_attn_decode_v3_usm(
                stream,
                &qb,
                &kb,
                &vb,
                &mut ob,
                AT_HEADS as u32,
                AT_KV_HEADS as u32,
                AT_HEAD_DIM as u32,
                AT_MAX_CTX as u32,
                AT_KV_LEN as u32,
            ),
            _ => {
                emit(name, "SKIP", "unknown-variant");
                return;
            }
        };
        let ms = started.elapsed().as_millis();
        match res {
            Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
            Ok(()) => {
                let gpu_out: Vec<f32> = ob
                    .as_slice()
                    .iter()
                    .map(|&b| half::f16::from_bits(b).to_f32())
                    .collect();
                // f16 I/O + online softmax reordering: looser gate
                // than the f32 matvec probes.
                let (cos, max_rel) = compare(&gpu_out, &cpu_out);
                let verdict = if cos > 0.995 && max_rel < 0.10 {
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
        return;
    }

    if let Some(v) = variant.strip_prefix("prefill_") {
        let upto = AT_PREFILL_BASE + AT_PREFILL_NEW;
        let kcache = fill_kv(&mut rng, upto);
        let vcache = fill_kv(&mut rng, upto);
        let q: Vec<f32> = (0..AT_PREFILL_NEW * AT_HEADS * AT_HEAD_DIM)
            .map(|_| rng.f32_pm(0.6))
            .collect();
        let mut cpu_out = vec![0f32; q.len()];
        k::gqa_attention_flash_prefill(
            &q,
            &kcache,
            &vcache,
            &mut cpu_out,
            AT_HEADS,
            AT_KV_HEADS,
            AT_HEAD_DIM,
            AT_MAX_CTX,
            AT_PREFILL_BASE,
            AT_PREFILL_NEW,
        );
        let alloc = (|| -> sk::Result<_> {
            let mut qb = sk::SyclSharedBuffer::<f32>::alloc(stream, q.len())?;
            qb.as_mut_slice().copy_from_slice(&q);
            let mut kb = sk::SyclSharedBuffer::<f32>::alloc(stream, kcache.len())?;
            kb.as_mut_slice().copy_from_slice(&kcache);
            let mut vb = sk::SyclSharedBuffer::<f32>::alloc(stream, vcache.len())?;
            vb.as_mut_slice().copy_from_slice(&vcache);
            let mut ob = sk::SyclSharedBuffer::<f32>::alloc(stream, cpu_out.len())?;
            ob.as_mut_slice().fill(f32::NAN);
            Ok((qb, kb, vb, ob))
        })();
        let (qb, kb, vb, mut ob) = match alloc {
            Ok(t) => t,
            Err(_) => {
                emit(name, "KERNEL_ERR", "usm-alloc-failed");
                return;
            }
        };
        // SAFETY: USM pointers live on `stream`, lengths match the
        // shape arguments; kernels wait before returning.
        let res = unsafe {
            match v {
                "v1" => sk::flash_attn_prefill_usm_raw(
                    stream,
                    qb.as_ptr(),
                    kb.as_ptr(),
                    vb.as_ptr(),
                    ob.as_mut_ptr(),
                    AT_HEADS as u32,
                    AT_KV_HEADS as u32,
                    AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32,
                    AT_PREFILL_BASE as u32,
                    AT_PREFILL_NEW as u32,
                ),
                "v2" => sk::flash_attn_prefill_v2_usm_raw(
                    stream,
                    qb.as_ptr(),
                    kb.as_ptr(),
                    vb.as_ptr(),
                    ob.as_mut_ptr(),
                    AT_HEADS as u32,
                    AT_KV_HEADS as u32,
                    AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32,
                    AT_PREFILL_BASE as u32,
                    AT_PREFILL_NEW as u32,
                ),
                "v3" => sk::flash_attn_prefill_v3_usm_raw(
                    stream,
                    qb.as_ptr(),
                    kb.as_ptr(),
                    vb.as_ptr(),
                    ob.as_mut_ptr(),
                    AT_HEADS as u32,
                    AT_KV_HEADS as u32,
                    AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32,
                    AT_PREFILL_BASE as u32,
                    AT_PREFILL_NEW as u32,
                ),
                _ => {
                    emit(name, "SKIP", "unknown-variant");
                    return;
                }
            }
        };
        let ms = started.elapsed().as_millis();
        match res {
            Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
            Ok(()) => {
                let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
                let verdict = if cos > 0.999 && max_rel < 0.05 {
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
        let mut cpu_out = vec![0f32; AT_HEADS * AT_HEAD_DIM];
        k::gqa_attention_one_step(
            &q,
            &kc_dq,
            &vc_dq,
            &mut cpu_out,
            AT_HEADS,
            AT_KV_HEADS,
            AT_HEAD_DIM,
            AT_MAX_CTX,
            AT_KV_LEN,
        );
        let alloc = (|| -> sk::Result<_> {
            let mut qb = sk::SyclSharedBuffer::<f32>::alloc(stream, q.len())?;
            qb.as_mut_slice().copy_from_slice(&q);
            let mut kb = sk::SyclSharedBuffer::<u8>::alloc(stream, kp.len())?;
            kb.as_mut_slice().copy_from_slice(&kp);
            let mut vb = sk::SyclSharedBuffer::<u8>::alloc(stream, vp.len())?;
            vb.as_mut_slice().copy_from_slice(&vp);
            let mut ob = sk::SyclSharedBuffer::<f32>::alloc(stream, cpu_out.len())?;
            ob.as_mut_slice().fill(f32::NAN);
            Ok((qb, kb, vb, ob))
        })();
        let (qb, kb, vb, mut ob) = match alloc {
            Ok(t) => t,
            Err(_) => {
                emit(name, "KERNEL_ERR", "usm-alloc-failed");
                return;
            }
        };
        // SAFETY: all four are live USM allocations on `stream`; q/out are
        // F32 [n_heads*head_dim], k/v packed U8 [n_kv_heads*max_ctx*
        // bytes_per_row] (built by quantize_kv_cache); the kernels wait
        // before returning.
        let res = unsafe {
            match fmt.name {
                "mxfp4" => sk::flash_attn_decode_mxfp4_usm_raw(
                    stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                    AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32, AT_KV_LEN as u32,
                ),
                "mxfp6" => sk::flash_attn_decode_mxfp6_usm_raw(
                    stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                    AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32, AT_KV_LEN as u32,
                ),
                "mxfp8" => sk::flash_attn_decode_mxfp8_usm_raw(
                    stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                    AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32, AT_KV_LEN as u32,
                ),
                "nvfp4" => sk::flash_attn_decode_nvfp4_usm_raw(
                    stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                    AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32, AT_KV_LEN as u32,
                ),
                "q4_0" => sk::flash_attn_decode_q4_0_usm_raw(
                    stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                    AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32, AT_KV_LEN as u32,
                ),
                _ => {
                    emit(name, "SKIP", "unknown-format");
                    return;
                }
            }
        };
        let ms = started.elapsed().as_millis();
        match res {
            Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
            Ok(()) => {
                let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
                let verdict = if cos > fmt.cos_min && max_rel < fmt.rel_max {
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
        let mut cpu_out = vec![0f32; q.len()];
        k::gqa_attention_flash_prefill(
            &q,
            &kc_dq,
            &vc_dq,
            &mut cpu_out,
            AT_HEADS,
            AT_KV_HEADS,
            AT_HEAD_DIM,
            AT_MAX_CTX,
            AT_PREFILL_BASE,
            AT_PREFILL_NEW,
        );
        let alloc = (|| -> sk::Result<_> {
            let mut qb = sk::SyclSharedBuffer::<f32>::alloc(stream, q.len())?;
            qb.as_mut_slice().copy_from_slice(&q);
            let mut kb = sk::SyclSharedBuffer::<u8>::alloc(stream, kp.len())?;
            kb.as_mut_slice().copy_from_slice(&kp);
            let mut vb = sk::SyclSharedBuffer::<u8>::alloc(stream, vp.len())?;
            vb.as_mut_slice().copy_from_slice(&vp);
            let mut ob = sk::SyclSharedBuffer::<f32>::alloc(stream, cpu_out.len())?;
            ob.as_mut_slice().fill(f32::NAN);
            Ok((qb, kb, vb, ob))
        })();
        let (qb, kb, vb, mut ob) = match alloc {
            Ok(t) => t,
            Err(_) => {
                emit(name, "KERNEL_ERR", "usm-alloc-failed");
                return;
            }
        };
        // SAFETY: as in the decode arm; q/out are F32 [n_new*n_heads*
        // head_dim], k/v packed U8; kernels wait before returning.
        let res = unsafe {
            match fmt.name {
                "mxfp4" => sk::flash_attn_prefill_mxfp4_usm_raw(
                    stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                    AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32, AT_PREFILL_BASE as u32, AT_PREFILL_NEW as u32,
                ),
                "mxfp6" => sk::flash_attn_prefill_mxfp6_usm_raw(
                    stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                    AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32, AT_PREFILL_BASE as u32, AT_PREFILL_NEW as u32,
                ),
                "mxfp8" => sk::flash_attn_prefill_mxfp8_usm_raw(
                    stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                    AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32, AT_PREFILL_BASE as u32, AT_PREFILL_NEW as u32,
                ),
                "nvfp4" => sk::flash_attn_prefill_nvfp4_usm_raw(
                    stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                    AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32, AT_PREFILL_BASE as u32, AT_PREFILL_NEW as u32,
                ),
                "q4_0" => sk::flash_attn_prefill_q4_0_usm_raw(
                    stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ob.as_mut_ptr(),
                    AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                    AT_MAX_CTX as u32, AT_PREFILL_BASE as u32, AT_PREFILL_NEW as u32,
                ),
                _ => {
                    emit(name, "SKIP", "unknown-format");
                    return;
                }
            }
        };
        let ms = started.elapsed().as_millis();
        match res {
            Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
            Ok(()) => {
                let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
                let verdict = if cos > fmt.cos_min && max_rel < fmt.rel_max {
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
        let mut cpu_out = vec![0f32; AT_HEADS * AT_HEAD_DIM];
        k::gqa_attention_one_step(
            &q, &kdq, &vdq, &mut cpu_out,
            AT_HEADS, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN,
        );
        let (qb, kb, vb, ksb, vsb, mut ob) =
            match run(&q, &kslab, &ksc, &vslab, &vsc, cpu_out.len()) {
                Ok(t) => t,
                Err(_) => {
                    emit(name, "KERNEL_ERR", "usm-alloc-failed");
                    return;
                }
            };
        // SAFETY: all USM buffers live on `stream`, sized to the geometry; the
        // kernel waits before returning.
        let res = unsafe {
            sk::flash_attn_decode_q8_0_usm_raw(
                stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ksb.as_ptr(), vsb.as_ptr(),
                ob.as_mut_ptr(), AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                AT_MAX_CTX as u32, AT_KV_LEN as u32,
            )
        };
        let ms = started.elapsed().as_millis();
        match res {
            Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
            Ok(()) => {
                let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
                let verdict = if cos > 0.999 && max_rel < 0.05 { "OK" } else { "MISCOMPUTE" };
                emit(name, verdict, &format!("cos={cos:.6} max_rel={max_rel:.4} ms={ms}"));
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
        let mut cpu_out = vec![0f32; q.len()];
        k::gqa_attention_flash_prefill(
            &q, &kdq, &vdq, &mut cpu_out,
            AT_HEADS, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_PREFILL_BASE, AT_PREFILL_NEW,
        );
        let (qb, kb, vb, ksb, vsb, mut ob) =
            match run(&q, &kslab, &ksc, &vslab, &vsc, cpu_out.len()) {
                Ok(t) => t,
                Err(_) => {
                    emit(name, "KERNEL_ERR", "usm-alloc-failed");
                    return;
                }
            };
        // SAFETY: as the decode arm, q/out sized for AT_PREFILL_NEW queries.
        let res = unsafe {
            sk::flash_attn_prefill_q8_0_usm_raw(
                stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ksb.as_ptr(), vsb.as_ptr(),
                ob.as_mut_ptr(), AT_HEADS as u32, AT_KV_HEADS as u32, AT_HEAD_DIM as u32,
                AT_MAX_CTX as u32, AT_PREFILL_BASE as u32, AT_PREFILL_NEW as u32,
            )
        };
        let ms = started.elapsed().as_millis();
        match res {
            Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
            Ok(()) => {
                let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
                let verdict = if cos > 0.999 && max_rel < 0.05 { "OK" } else { "MISCOMPUTE" };
                emit(name, verdict, &format!("cos={cos:.6} max_rel={max_rel:.4} ms={ms}"));
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
        let mut cpu_out = vec![0f32; AT_HEADS * AT_HEAD_DIM];
        k::gqa_attention_one_step(
            &q, &kdq, &vdq, &mut cpu_out,
            AT_HEADS, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN,
        );
        let (qb, kb, vb, ksb, vsb, mut ob) =
            match run(&q, &kslab, &ksc, &vslab, &vsc, cpu_out.len()) {
                Ok(t) => t,
                Err(_) => {
                    emit(name, "KERNEL_ERR", "usm-alloc-failed");
                    return;
                }
            };
        // SAFETY: all USM buffers live on `stream`, sized to the geometry; the
        // kernel waits before returning.
        let res = unsafe {
            sk::flash_attn_decode_tq_usm_raw(
                stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ksb.as_ptr(), vsb.as_ptr(),
                bits as u32, ob.as_mut_ptr(), AT_HEADS as u32, AT_KV_HEADS as u32,
                AT_HEAD_DIM as u32, AT_MAX_CTX as u32, AT_KV_LEN as u32,
            )
        };
        let ms = started.elapsed().as_millis();
        match res {
            Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
            Ok(()) => {
                let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
                let verdict = if cos > 0.999 && max_rel < 0.05 { "OK" } else { "MISCOMPUTE" };
                emit(name, verdict, &format!("cos={cos:.6} max_rel={max_rel:.4} ms={ms} bits={bits}"));
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
        let mut cpu_out = vec![0f32; q.len()];
        k::gqa_attention_flash_prefill(
            &q, &kdq, &vdq, &mut cpu_out,
            AT_HEADS, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_PREFILL_BASE, AT_PREFILL_NEW,
        );
        let (qb, kb, vb, ksb, vsb, mut ob) =
            match run(&q, &kslab, &ksc, &vslab, &vsc, cpu_out.len()) {
                Ok(t) => t,
                Err(_) => {
                    emit(name, "KERNEL_ERR", "usm-alloc-failed");
                    return;
                }
            };
        // SAFETY: as the decode arm, q/out sized for AT_PREFILL_NEW queries.
        let res = unsafe {
            sk::flash_attn_prefill_tq_usm_raw(
                stream, qb.as_ptr(), kb.as_ptr(), vb.as_ptr(), ksb.as_ptr(), vsb.as_ptr(),
                bits as u32, ob.as_mut_ptr(), AT_HEADS as u32, AT_KV_HEADS as u32,
                AT_HEAD_DIM as u32, AT_MAX_CTX as u32, AT_PREFILL_BASE as u32, AT_PREFILL_NEW as u32,
            )
        };
        let ms = started.elapsed().as_millis();
        match res {
            Err(e) => emit(name, "KERNEL_ERR", &format!("{e} ms={ms}")),
            Ok(()) => {
                let (cos, max_rel) = compare(ob.as_slice(), &cpu_out);
                let verdict = if cos > 0.999 && max_rel < 0.05 { "OK" } else { "MISCOMPUTE" };
                emit(name, verdict, &format!("cos={cos:.6} max_rel={max_rel:.4} ms={ms} bits={bits}"));
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

pub fn run_parent() -> anyhow::Result<()> {
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
    let mut counts: std::collections::BTreeMap<&'static str, usize> = Default::default();
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
        let key: &'static str = match verdict.as_str() {
            "OK" => "OK",
            "MISCOMPUTE" => "MISCOMPUTE",
            "KERNEL_ERR" => "KERNEL_ERR",
            "CRASH" => "CRASH",
            "HANG" => "HANG",
            _ => "SKIP",
        };
        *counts.entry(key).or_default() += 1;
        println!("  {name:<20} {verdict:<11} {detail}");
        rows.push((name.clone(), verdict, detail));
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
    {
        let (n_heads, n_kv_heads, head_dim, max_ctx, kv_len) =
            (8usize, 2usize, 64usize, 128usize, 40usize);
        let q = gen_x(n_heads * head_dim, 61);
        let kc = gen_x(n_kv_heads * max_ctx * head_dim, 63);
        let vc = gen_x(n_kv_heads * max_ctx * head_dim, 67);
        let cpu = ref_flash_decode(&q, &kc, &vc, n_heads, n_kv_heads, head_dim, max_ctx, kv_len);
        if let (Some(qb), Some(kb), Some(vb), Some(mut ob)) = (
            cu_upload_f32(&stream, &q),
            cu_upload_f32(&stream, &kc),
            cu_upload_f32(&stream, &vc),
            ck::CudaDeviceBuffer::alloc(&stream, n_heads * head_dim * 4),
        ) {
            // SAFETY: q/out [n_heads*head_dim], k/v [n_kv_heads*max_ctx*head_dim] device f32.
            let res = unsafe {
                ck::flash_attn_decode_f32(
                    &stream,
                    qb.as_ptr() as *const f32,
                    kb.as_ptr() as *const f32,
                    vb.as_ptr() as *const f32,
                    ob.as_mut_ptr() as *mut f32,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_ctx,
                    kv_len,
                )
            };
            match res {
                Ok(()) => cu_grade(
                    "attn:decode",
                    &cu_download_f32(&ob, n_heads * head_dim),
                    &cpu,
                    0.999,
                    0.02,
                    &mut counts,
                ),
                Err(e) => {
                    cu_emit("attn:decode", "KERNEL_ERR", &format!("{e}"));
                    *counts.entry("KERNEL_ERR").or_default() += 1;
                }
            }
        }

        let (kv_base, n_new) = (10usize, 6usize);
        let qp = gen_x(n_new * n_heads * head_dim, 71);
        let cpu_p = ref_flash_prefill(
            &qp, &kc, &vc, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
        );
        if let (Some(qb), Some(kb), Some(vb), Some(mut ob)) = (
            cu_upload_f32(&stream, &qp),
            cu_upload_f32(&stream, &kc),
            cu_upload_f32(&stream, &vc),
            ck::CudaDeviceBuffer::alloc(&stream, n_new * n_heads * head_dim * 4),
        ) {
            // SAFETY: q/out [n_new*n_heads*head_dim], k/v caches device f32.
            let res = unsafe {
                ck::flash_attn_prefill_f32(
                    &stream,
                    qb.as_ptr() as *const f32,
                    kb.as_ptr() as *const f32,
                    vb.as_ptr() as *const f32,
                    ob.as_mut_ptr() as *mut f32,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_ctx,
                    kv_base,
                    n_new,
                )
            };
            match res {
                Ok(()) => cu_grade(
                    "attn:prefill",
                    &cu_download_f32(&ob, n_new * n_heads * head_dim),
                    &cpu_p,
                    0.999,
                    0.02,
                    &mut counts,
                ),
                Err(e) => {
                    cu_emit("attn:prefill", "KERNEL_ERR", &format!("{e}"));
                    *counts.entry("KERNEL_ERR").or_default() += 1;
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
            // round-trips differently).
            let kc_dq = dequant_kv_cache(fmt, &kp, n_kv_heads, head_dim, max_ctx, upto);
            let vc_dq = dequant_kv_cache(fmt, &vp, n_kv_heads, head_dim, max_ctx, upto);
            let cpu_dec = ref_flash_decode(
                &q, &kc_dq, &vc_dq, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
            );
            let cpu_pre = ref_flash_prefill(
                &qp, &kc_dq, &vc_dq, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
            );

            // decode
            if let (Some(qb), Some(kb), Some(vb), Some(mut ob)) = (
                cu_upload_f32(&stream, &q),
                ck::CudaDeviceBuffer::from_host(&stream, &kp),
                ck::CudaDeviceBuffer::from_host(&stream, &vp),
                ck::CudaDeviceBuffer::alloc(&stream, n_heads * head_dim * 4),
            ) {
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
                        ),
                        "mxfp6" => ck::flash_attn_decode_mxfp6(
                            &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                        ),
                        "mxfp8" => ck::flash_attn_decode_mxfp8(
                            &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                        ),
                        "nvfp4" => ck::flash_attn_decode_nvfp4(
                            &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                        ),
                        "q4_0" => ck::flash_attn_decode_q4_0(
                            &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                        ),
                        _ => unreachable!(),
                    }
                };
                match res {
                    Ok(()) => cu_grade(
                        &dname,
                        &cu_download_f32(&ob, n_heads * head_dim),
                        &cpu_dec,
                        fmt.cos_min,
                        fmt.rel_max,
                        &mut counts,
                    ),
                    Err(e) => {
                        cu_emit(&dname, "KERNEL_ERR", &format!("{e}"));
                        *counts.entry("KERNEL_ERR").or_default() += 1;
                    }
                }
            }

            // prefill
            if let (Some(qb), Some(kb), Some(vb), Some(mut ob)) = (
                cu_upload_f32(&stream, &qp),
                ck::CudaDeviceBuffer::from_host(&stream, &kp),
                ck::CudaDeviceBuffer::from_host(&stream, &vp),
                ck::CudaDeviceBuffer::alloc(&stream, n_new * n_heads * head_dim * 4),
            ) {
                // SAFETY: q/out F32 [n_new*n_heads*head_dim]; k/v packed as above.
                let res = unsafe {
                    let q = qb.as_ptr() as *const f32;
                    let k = kb.as_ptr();
                    let v = vb.as_ptr();
                    let o = ob.as_mut_ptr() as *mut f32;
                    match fmt.name {
                        "mxfp4" => ck::flash_attn_prefill_mxfp4(
                            &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base,
                            n_new,
                        ),
                        "mxfp6" => ck::flash_attn_prefill_mxfp6(
                            &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base,
                            n_new,
                        ),
                        "mxfp8" => ck::flash_attn_prefill_mxfp8(
                            &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base,
                            n_new,
                        ),
                        "nvfp4" => ck::flash_attn_prefill_nvfp4(
                            &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base,
                            n_new,
                        ),
                        "q4_0" => ck::flash_attn_prefill_q4_0(
                            &stream, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base,
                            n_new,
                        ),
                        _ => unreachable!(),
                    }
                };
                match res {
                    Ok(()) => cu_grade(
                        &pname,
                        &cu_download_f32(&ob, n_new * n_heads * head_dim),
                        &cpu_pre,
                        fmt.cos_min,
                        fmt.rel_max,
                        &mut counts,
                    ),
                    Err(e) => {
                        cu_emit(&pname, "KERNEL_ERR", &format!("{e}"));
                        *counts.entry("KERNEL_ERR").or_default() += 1;
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
        let cpu_dec = ref_flash_decode(&q, &kdq, &vdq, n_heads, n_kv_heads, head_dim, max_ctx, kv_len);
        let cpu_pre = ref_flash_prefill(&qp, &kdq, &vdq, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new);
        // decode
        if let (Some(qb), Some(kb), Some(vb), Some(ksb), Some(vsb), Some(mut ob)) = (
            cu_upload_f32(&stream, &q),
            ck::CudaDeviceBuffer::from_host(&stream, i8_as_bytes(&kslab)),
            ck::CudaDeviceBuffer::from_host(&stream, i8_as_bytes(&vslab)),
            cu_upload_f32(&stream, &ksc),
            cu_upload_f32(&stream, &vsc),
            ck::CudaDeviceBuffer::alloc(&stream, n_heads * head_dim * 4),
        ) {
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
                )
            };
            match res {
                Ok(()) => cu_grade(
                    "attn:decode_q8_0",
                    &cu_download_f32(&ob, n_heads * head_dim),
                    &cpu_dec,
                    0.999,
                    0.05,
                    &mut counts,
                ),
                Err(e) => {
                    cu_emit("attn:decode_q8_0", "KERNEL_ERR", &format!("{e}"));
                    *counts.entry("KERNEL_ERR").or_default() += 1;
                }
            }
        }
        // prefill
        if let (Some(qb), Some(kb), Some(vb), Some(ksb), Some(vsb), Some(mut ob)) = (
            cu_upload_f32(&stream, &qp),
            ck::CudaDeviceBuffer::from_host(&stream, i8_as_bytes(&kslab)),
            ck::CudaDeviceBuffer::from_host(&stream, i8_as_bytes(&vslab)),
            cu_upload_f32(&stream, &ksc),
            cu_upload_f32(&stream, &vsc),
            ck::CudaDeviceBuffer::alloc(&stream, n_new * n_heads * head_dim * 4),
        ) {
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
                )
            };
            match res {
                Ok(()) => cu_grade(
                    "attn:prefill_q8_0",
                    &cu_download_f32(&ob, n_new * n_heads * head_dim),
                    &cpu_pre,
                    0.999,
                    0.05,
                    &mut counts,
                ),
                Err(e) => {
                    cu_emit("attn:prefill_q8_0", "KERNEL_ERR", &format!("{e}"));
                    *counts.entry("KERNEL_ERR").or_default() += 1;
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
        let cpu_dec = ref_flash_decode(&q, &kdq, &vdq, n_heads, n_kv_heads, head_dim, max_ctx, kv_len);
        let cpu_pre = ref_flash_prefill(&qp, &kdq, &vdq, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new);
        // decode
        if let (Some(qb), Some(kb), Some(vb), Some(ksb), Some(vsb), Some(mut ob)) = (
            cu_upload_f32(&stream, &q),
            ck::CudaDeviceBuffer::from_host(&stream, &kslab),
            ck::CudaDeviceBuffer::from_host(&stream, &vslab),
            cu_upload_f32(&stream, &ksc),
            cu_upload_f32(&stream, &vsc),
            ck::CudaDeviceBuffer::alloc(&stream, n_heads * head_dim * 4),
        ) {
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
                )
            };
            match res {
                Ok(()) => cu_grade(
                    "attn:decode_tq",
                    &cu_download_f32(&ob, n_heads * head_dim),
                    &cpu_dec,
                    0.999,
                    0.05,
                    &mut counts,
                ),
                Err(e) => {
                    cu_emit("attn:decode_tq", "KERNEL_ERR", &format!("{e}"));
                    *counts.entry("KERNEL_ERR").or_default() += 1;
                }
            }
        }
        // prefill
        if let (Some(qb), Some(kb), Some(vb), Some(ksb), Some(vsb), Some(mut ob)) = (
            cu_upload_f32(&stream, &qp),
            ck::CudaDeviceBuffer::from_host(&stream, &kslab),
            ck::CudaDeviceBuffer::from_host(&stream, &vslab),
            cu_upload_f32(&stream, &ksc),
            cu_upload_f32(&stream, &vsc),
            ck::CudaDeviceBuffer::alloc(&stream, n_new * n_heads * head_dim * 4),
        ) {
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
                )
            };
            match res {
                Ok(()) => cu_grade(
                    "attn:prefill_tq",
                    &cu_download_f32(&ob, n_new * n_heads * head_dim),
                    &cpu_pre,
                    0.999,
                    0.05,
                    &mut counts,
                ),
                Err(e) => {
                    cu_emit("attn:prefill_tq", "KERNEL_ERR", &format!("{e}"));
                    *counts.entry("KERNEL_ERR").or_default() += 1;
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

        // FlashAttention decode + prefill (f32)
        {
            let (n_heads, n_kv_heads, head_dim, max_ctx, kv_len) =
                (8usize, 2usize, 64usize, 128usize, 40usize);
            let q = gen_x(n_heads * head_dim, 61);
            let kc = gen_x(n_kv_heads * max_ctx * head_dim, 63);
            let vc = gen_x(n_kv_heads * max_ctx * head_dim, 67);
            let cpu =
                ref_flash_decode(&q, &kc, &vc, n_heads, n_kv_heads, head_dim, max_ctx, kv_len);
            if let (Some(qb), Some(kb), Some(vb), Some(mut ob)) = (
                mk_upload_f32(&fs, &q),
                mk_upload_f32(&fs, &kc),
                mk_upload_f32(&fs, &vc),
                mk::MlxDeviceBuffer::alloc(&fs, n_heads * head_dim * 4),
            ) {
                // SAFETY: q/out [n_heads*head_dim], k/v [n_kv_heads*max_ctx*
                // head_dim] device f32 on `fs`.
                let res = unsafe {
                    mk::flash_attn_decode_f32(
                        &fs,
                        qb.as_ptr() as *const f32,
                        kb.as_ptr() as *const f32,
                        vb.as_ptr() as *const f32,
                        ob.as_mut_ptr() as *mut f32,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_len,
                    )
                };
                match res {
                    Ok(()) => cu_grade(
                        "attn:decode",
                        &mk_download_f32(&ob, n_heads * head_dim),
                        &cpu,
                        0.999,
                        0.02,
                        &mut counts,
                    ),
                    Err(e) => {
                        cu_emit("attn:decode", "KERNEL_ERR", &format!("{e:?}"));
                        *counts.entry("KERNEL_ERR").or_default() += 1;
                    }
                }
            }

            let (kv_base, n_new) = (10usize, 6usize);
            let qp = gen_x(n_new * n_heads * head_dim, 71);
            let cpu_p = ref_flash_prefill(
                &qp, &kc, &vc, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
            );
            if let (Some(qb), Some(kb), Some(vb), Some(mut ob)) = (
                mk_upload_f32(&fs, &qp),
                mk_upload_f32(&fs, &kc),
                mk_upload_f32(&fs, &vc),
                mk::MlxDeviceBuffer::alloc(&fs, n_new * n_heads * head_dim * 4),
            ) {
                // SAFETY: q/out [n_new*n_heads*head_dim], k/v caches device f32.
                let res = unsafe {
                    mk::flash_attn_prefill_f32(
                        &fs,
                        qb.as_ptr() as *const f32,
                        kb.as_ptr() as *const f32,
                        vb.as_ptr() as *const f32,
                        ob.as_mut_ptr() as *mut f32,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_base,
                        n_new,
                    )
                };
                match res {
                    Ok(()) => cu_grade(
                        "attn:prefill",
                        &mk_download_f32(&ob, n_new * n_heads * head_dim),
                        &cpu_p,
                        0.999,
                        0.02,
                        &mut counts,
                    ),
                    Err(e) => {
                        cu_emit("attn:prefill", "KERNEL_ERR", &format!("{e:?}"));
                        *counts.entry("KERNEL_ERR").or_default() += 1;
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
                let cpu_dec = ref_flash_decode(
                    &q, &kc_dq, &vc_dq, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                );
                let cpu_pre = ref_flash_prefill(
                    &qp, &kc_dq, &vc_dq, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
                );

                // decode
                if let (Some(qb), Some(kb), Some(vb), Some(mut ob)) = (
                    mk_upload_f32(&fs, &q),
                    mk::MlxDeviceBuffer::from_host(&fs, &kp),
                    mk::MlxDeviceBuffer::from_host(&fs, &vp),
                    mk::MlxDeviceBuffer::alloc(&fs, n_heads * head_dim * 4),
                ) {
                    // SAFETY: q/out F32 [n_heads*head_dim]; k/v packed device
                    // buffers [n_kv_heads*max_ctx*bytes_per_row] on `fs`.
                    let res = unsafe {
                        let q = qb.as_ptr() as *const f32;
                        let k = kb.as_ptr();
                        let v = vb.as_ptr();
                        let o = ob.as_mut_ptr() as *mut f32;
                        match fmt.name {
                            "mxfp4" => mk::flash_attn_decode_mxfp4(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len),
                            "mxfp6" => mk::flash_attn_decode_mxfp6(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len),
                            "mxfp8" => mk::flash_attn_decode_mxfp8(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len),
                            "nvfp4" => mk::flash_attn_decode_nvfp4(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len),
                            "q4_0" => mk::flash_attn_decode_q4_0(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_len),
                            _ => unreachable!(),
                        }
                    };
                    match res {
                        Ok(()) => cu_grade(
                            &dname,
                            &mk_download_f32(&ob, n_heads * head_dim),
                            &cpu_dec,
                            fmt.cos_min,
                            fmt.rel_max,
                            &mut counts,
                        ),
                        Err(e) => {
                            cu_emit(&dname, "KERNEL_ERR", &format!("{e:?}"));
                            *counts.entry("KERNEL_ERR").or_default() += 1;
                        }
                    }
                }

                // prefill
                if let (Some(qb), Some(kb), Some(vb), Some(mut ob)) = (
                    mk_upload_f32(&fs, &qp),
                    mk::MlxDeviceBuffer::from_host(&fs, &kp),
                    mk::MlxDeviceBuffer::from_host(&fs, &vp),
                    mk::MlxDeviceBuffer::alloc(&fs, n_new * n_heads * head_dim * 4),
                ) {
                    // SAFETY: q/out F32 [n_new*n_heads*head_dim]; k/v packed as above.
                    let res = unsafe {
                        let q = qb.as_ptr() as *const f32;
                        let k = kb.as_ptr();
                        let v = vb.as_ptr();
                        let o = ob.as_mut_ptr() as *mut f32;
                        match fmt.name {
                            "mxfp4" => mk::flash_attn_prefill_mxfp4(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new),
                            "mxfp6" => mk::flash_attn_prefill_mxfp6(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new),
                            "mxfp8" => mk::flash_attn_prefill_mxfp8(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new),
                            "nvfp4" => mk::flash_attn_prefill_nvfp4(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new),
                            "q4_0" => mk::flash_attn_prefill_q4_0(&fs, q, k, v, o, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new),
                            _ => unreachable!(),
                        }
                    };
                    match res {
                        Ok(()) => cu_grade(
                            &pname,
                            &mk_download_f32(&ob, n_new * n_heads * head_dim),
                            &cpu_pre,
                            fmt.cos_min,
                            fmt.rel_max,
                            &mut counts,
                        ),
                        Err(e) => {
                            cu_emit(&pname, "KERNEL_ERR", &format!("{e:?}"));
                            *counts.entry("KERNEL_ERR").or_default() += 1;
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
            let cpu_dec = ref_flash_decode(&q, &kdq, &vdq, n_heads, n_kv_heads, head_dim, max_ctx, kv_len);
            let cpu_pre = ref_flash_prefill(&qp, &kdq, &vdq, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new);
            // decode
            if let (Some(qb), Some(kb), Some(vb), Some(ksb), Some(vsb), Some(mut ob)) = (
                mk_upload_f32(&fs, &q),
                mk::MlxDeviceBuffer::from_host(&fs, i8_as_bytes(&kslab)),
                mk::MlxDeviceBuffer::from_host(&fs, i8_as_bytes(&vslab)),
                mk_upload_f32(&fs, &ksc),
                mk_upload_f32(&fs, &vsc),
                mk::MlxDeviceBuffer::alloc(&fs, n_heads * head_dim * 4),
            ) {
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
                    )
                };
                match res {
                    Ok(()) => cu_grade(
                        "attn:decode_q8_0",
                        &mk_download_f32(&ob, n_heads * head_dim),
                        &cpu_dec,
                        0.999,
                        0.05,
                        &mut counts,
                    ),
                    Err(e) => {
                        cu_emit("attn:decode_q8_0", "KERNEL_ERR", &format!("{e:?}"));
                        *counts.entry("KERNEL_ERR").or_default() += 1;
                    }
                }
            }
            // prefill
            if let (Some(qb), Some(kb), Some(vb), Some(ksb), Some(vsb), Some(mut ob)) = (
                mk_upload_f32(&fs, &qp),
                mk::MlxDeviceBuffer::from_host(&fs, i8_as_bytes(&kslab)),
                mk::MlxDeviceBuffer::from_host(&fs, i8_as_bytes(&vslab)),
                mk_upload_f32(&fs, &ksc),
                mk_upload_f32(&fs, &vsc),
                mk::MlxDeviceBuffer::alloc(&fs, n_new * n_heads * head_dim * 4),
            ) {
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
                    )
                };
                match res {
                    Ok(()) => cu_grade(
                        "attn:prefill_q8_0",
                        &mk_download_f32(&ob, n_new * n_heads * head_dim),
                        &cpu_pre,
                        0.999,
                        0.05,
                        &mut counts,
                    ),
                    Err(e) => {
                        cu_emit("attn:prefill_q8_0", "KERNEL_ERR", &format!("{e:?}"));
                        *counts.entry("KERNEL_ERR").or_default() += 1;
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
            let cpu_dec = ref_flash_decode(&q, &kdq, &vdq, n_heads, n_kv_heads, head_dim, max_ctx, kv_len);
            let cpu_pre = ref_flash_prefill(&qp, &kdq, &vdq, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new);
            // decode
            if let (Some(qb), Some(kb), Some(vb), Some(ksb), Some(vsb), Some(mut ob)) = (
                mk_upload_f32(&fs, &q),
                mk::MlxDeviceBuffer::from_host(&fs, &kslab),
                mk::MlxDeviceBuffer::from_host(&fs, &vslab),
                mk_upload_f32(&fs, &ksc),
                mk_upload_f32(&fs, &vsc),
                mk::MlxDeviceBuffer::alloc(&fs, n_heads * head_dim * 4),
            ) {
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
                    )
                };
                match res {
                    Ok(()) => cu_grade(
                        "attn:decode_tq",
                        &mk_download_f32(&ob, n_heads * head_dim),
                        &cpu_dec,
                        0.999,
                        0.05,
                        &mut counts,
                    ),
                    Err(e) => {
                        cu_emit("attn:decode_tq", "KERNEL_ERR", &format!("{e:?}"));
                        *counts.entry("KERNEL_ERR").or_default() += 1;
                    }
                }
            }
            // prefill
            if let (Some(qb), Some(kb), Some(vb), Some(ksb), Some(vsb), Some(mut ob)) = (
                mk_upload_f32(&fs, &qp),
                mk::MlxDeviceBuffer::from_host(&fs, &kslab),
                mk::MlxDeviceBuffer::from_host(&fs, &vslab),
                mk_upload_f32(&fs, &ksc),
                mk_upload_f32(&fs, &vsc),
                mk::MlxDeviceBuffer::alloc(&fs, n_new * n_heads * head_dim * 4),
            ) {
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
                    )
                };
                match res {
                    Ok(()) => cu_grade(
                        "attn:prefill_tq",
                        &mk_download_f32(&ob, n_new * n_heads * head_dim),
                        &cpu_pre,
                        0.999,
                        0.05,
                        &mut counts,
                    ),
                    Err(e) => {
                        cu_emit("attn:prefill_tq", "KERNEL_ERR", &format!("{e:?}"));
                        *counts.entry("KERNEL_ERR").or_default() += 1;
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
