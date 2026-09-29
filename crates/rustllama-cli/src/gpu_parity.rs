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

fn cpu_matvec_for(name: &str) -> CpuMatvec {
    match name {
        "q8_0" => k::matvec_q8_0_w_f32_a,
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
        _ => unreachable!("unknown dtype {name}"),
    }
}

fn gpu_matvec_for(name: &str) -> GpuMatvecRaw {
    match name {
        "q8_0" => sk::matvec_q8_0_packed_f32_usm_raw,
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
        _ => unreachable!("unknown dtype {name}"),
    }
}

fn gpu_fused_for(name: &str) -> GpuFusedRaw {
    match name {
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
        _ => unreachable!("unknown dtype {name}"),
    }
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
    for l in LAYOUTS {
        v.push(format!("matvec:{}", l.name));
    }
    for l in LAYOUTS {
        // PTQ1_0 and the MXFP* formats have no fused gate/up kernel
        // (their FFN dispatch uses the plain matvec, falling back to two
        // separate single-row matvecs); probe the matvec only.
        if l.name == "ptq1_0" || l.name.starts_with("mxfp") {
            continue;
        }
        v.push(format!("fused:{}", l.name));
    }
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

fn emit(name: &str, verdict: &str, detail: &str) {
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
        "fused" => probe_fused(&stream, name, rest, started),
        "attn" => probe_attn(&stream, name, rest, started),
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
    let gpu = gpu_fused_for(dtype);
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
                    _ => unreachable!("unknown quant-KV format {}", fmt.name),
                }
            }
        }
    }
    packed
}

fn probe_attn(stream: &sk::SyclStream, name: &str, variant: &str, started: Instant) {
    // Quant-KV variants (decode_mxfp4 … prefill_nvfp4) run the F32-Q /
    // packed-K/V flash kernels and grade against the full-precision CPU
    // reference; the f32 variants (v1/v2/v3) below stay on the f16 path.
    // Split here so each keeps its own buffer setup (f16 vs f32 + packed).
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
        // Full-precision reference on the UN-quantized K/V.
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
        // Quantize K/V into the packed block layout the kernel decodes.
        let kp = quantize_kv_cache(fmt, &kcache, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN);
        let vp = quantize_kv_cache(fmt, &vcache, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, AT_KV_LEN);
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
        // Full-precision reference on the UN-quantized K/V.
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
        let kp = quantize_kv_cache(fmt, &kcache, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, upto);
        let vp = quantize_kv_cache(fmt, &vcache, AT_KV_HEADS, AT_HEAD_DIM, AT_MAX_CTX, upto);
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

fn cu_emit(name: &str, verdict: &str, detail: &str) {
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
    // (K constraints: ptq1_0 %128, q8_0 %32, q4_k/q6_k %256 — MV_K=2048 ok.)
    for dtype in ["ptq1_0", "q8_0", "q4_k", "q6_k", "mxfp4", "mxfp6", "mxfp8"] {
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
    // Same geometry as the f32 flash section above; K/V are quantized to
    // each format's block layout on the CPU (byte-identical to the engine's
    // KV writer), then the native quant-KV kernels dequantize them on the
    // fly. Graded against the FULL-PRECISION f32 reference, so the per-format
    // tolerance (from QUANT_KV_FORMATS) must absorb the KV round-trip error.
    {
        let (n_heads, n_kv_heads, head_dim, max_ctx, kv_len) =
            (8usize, 2usize, 64usize, 128usize, 40usize);
        let (kv_base, n_new) = (10usize, 6usize);
        let q = gen_x(n_heads * head_dim, 61);
        let kc = gen_x(n_kv_heads * max_ctx * head_dim, 63);
        let vc = gen_x(n_kv_heads * max_ctx * head_dim, 67);
        let qp = gen_x(n_new * n_heads * head_dim, 71);
        // Full-precision references on the un-quantized K/V (reused by every
        // format — the quantization lives only on the GPU side).
        let cpu_dec =
            ref_flash_decode(&q, &kc, &vc, n_heads, n_kv_heads, head_dim, max_ctx, kv_len);
        let cpu_pre = ref_flash_prefill(
            &qp, &kc, &vc, n_heads, n_kv_heads, head_dim, max_ctx, kv_base, n_new,
        );
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
