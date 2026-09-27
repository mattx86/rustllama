//! Microbench: serial vs rayon-parallel F16 matvec across a grid of
//! M values, with K fixed. Run via:
//!
//! ```
//! cargo run --release -p rustllama-kernels-cpu --example matvec_parallel_crossover
//! ```
//!
//! Output columns: `M | serial_us | parallel_us | speedup`.
//!
//! Use the crossover row to pick the gate inside the engine's
//! dispatcher — below the crossover, the serial path wins because
//! rayon's work-stealing overhead exceeds the matmul work. Above,
//! parallel dominates. For typical decode matvec on a 7B model
//! (`M=4096, K=4096`) the parallel path should clearly win on any
//! ≥4-core host.

use std::time::Instant;

use half::f16;
use rustllama_kernels_cpu::{matvec_f16_w_f32_a, matvec_f16_w_f32_a_parallel};

fn main() {
    // K fixed at 4096 — the typical d_model for a 7B-class model's
    // hidden width. Vary M to probe the crossover.
    const K: usize = 4096;
    const REPEATS: usize = 5;
    const WARMUP: usize = 1;

    let m_values = [
        16usize, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    ];

    // Pre-build the largest workspace once; smaller M values use a
    // prefix of the weights.
    let max_m = *m_values.iter().max().unwrap();
    let weights: Vec<f16> = (0..max_m * K)
        .map(|i| f16::from_f32(((i % 17) as f32 - 8.0) * 0.01))
        .collect();
    let x: Vec<f32> = (0..K).map(|i| ((i % 31) as f32 - 15.0) * 0.05).collect();

    println!("Iris Xe / serial-vs-parallel F16 matvec crossover (K = {K})");
    println!("rayon pool: default (heuristic-sized)");
    println!();
    println!("    M | serial (us) | parallel (us) | speedup");
    println!("------+-------------+---------------+--------");

    for &m in &m_values {
        let w = &weights[..m * K];
        let mut out_serial = vec![0f32; m];
        let mut out_parallel = vec![0f32; m];

        // Warmup. Discard.
        for _ in 0..WARMUP {
            matvec_f16_w_f32_a(w, &x, &mut out_serial, m, K);
            matvec_f16_w_f32_a_parallel(w, &x, &mut out_parallel, m, K);
        }

        let t = Instant::now();
        for _ in 0..REPEATS {
            matvec_f16_w_f32_a(w, &x, &mut out_serial, m, K);
        }
        let serial_us = t.elapsed().as_micros() as f64 / REPEATS as f64;

        let t = Instant::now();
        for _ in 0..REPEATS {
            matvec_f16_w_f32_a_parallel(w, &x, &mut out_parallel, m, K);
        }
        let parallel_us = t.elapsed().as_micros() as f64 / REPEATS as f64;

        let speedup = serial_us / parallel_us.max(1.0);
        // Marker: where parallel first overtakes serial.
        let marker = if speedup >= 1.05 { "  ←" } else { "" };
        println!(
            "{m:>5} | {serial_us:>11.1} | {parallel_us:>13.1} | {speedup:>5.2}x{marker}"
        );

        // Quick correctness check: both paths must produce the same
        // output (within fp tolerance — both use the same SIMD inner
        // loop, just with different work distribution).
        let max_diff = out_serial
            .iter()
            .zip(out_parallel.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "serial vs parallel disagree at M={m}: max_diff = {max_diff}"
        );
    }

    println!();
    println!("crossover = first row marked ←");
    println!(
        "wire the gate at that M into the dispatcher: below → serial, above → parallel"
    );
}
