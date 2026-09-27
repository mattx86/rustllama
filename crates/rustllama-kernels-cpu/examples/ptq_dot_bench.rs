//! Focused microbench for the batched PTQ1_0 ternary dot — the
//! VTune target for microarchitecture analysis (isolates the kernel
//! from model/serve noise) and a standalone throughput probe.
//!
//! Usage: ptq_dot_bench [seconds] [fast|bitwise]
//!   Loops the two Bonsai FFN shapes (gate/up 17408x5120 and down
//!   5120x17408) at a 64-token batch and prints G mul-adds/s.

use rustllama_kernels_cpu::matvec_ptq1_0_w_f32_a_batched;

fn gen_ptq1_0(m: usize, k: usize, seed: u32) -> Vec<u8> {
    let row_bytes = (k / 128) * 28;
    let mut bytes = vec![0u8; m * row_bytes];
    let mut s = seed;
    for blk in bytes.chunks_mut(28) {
        for q in blk[..26].iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *q = (s >> 24) as u8;
        }
        let d = half::f16::from_f32(0.004);
        blk[26..28].copy_from_slice(&d.to_le_bytes());
    }
    bytes
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seconds: f64 = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(20.0);
    let mode = args.get(2).map(|s| s.as_str()).unwrap_or("fast").to_string();
    if mode == "fast" {
        // The public entry reads the env lever.
        std::env::set_var("RUSTLLAMA_TERNARY_FASTDOT", "1");
    }
    let n_rows = 64usize;
    let shapes = [(17408usize, 5120usize), (5120usize, 17408usize)];
    let weights: Vec<Vec<u8>> = shapes
        .iter()
        .enumerate()
        .map(|(i, &(m, k))| gen_ptq1_0(m, k, 77 + i as u32))
        .collect();
    let xs: Vec<Vec<f32>> = shapes
        .iter()
        .map(|&(_, k)| {
            (0..n_rows * k)
                .map(|i| ((i * 37 + 11) % 97) as f32 * 0.013 - 0.6)
                .collect()
        })
        .collect();
    let mut outs: Vec<Vec<f32>> = shapes.iter().map(|&(m, _)| vec![0.0; n_rows * m]).collect();

    // Warmup.
    for (i, &(m, k)) in shapes.iter().enumerate() {
        matvec_ptq1_0_w_f32_a_batched(&weights[i], &xs[i], &mut outs[i], m, k, n_rows);
    }

    let start = std::time::Instant::now();
    let mut iters = 0u64;
    let mut checksum = 0f64;
    while start.elapsed().as_secs_f64() < seconds {
        for (i, &(m, k)) in shapes.iter().enumerate() {
            matvec_ptq1_0_w_f32_a_batched(&weights[i], &xs[i], &mut outs[i], m, k, n_rows);
            checksum += outs[i][0] as f64;
        }
        iters += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let ma_per_iter: f64 = shapes
        .iter()
        .map(|&(m, k)| (m * k * n_rows) as f64)
        .sum();
    let total_ma = ma_per_iter * iters as f64;
    println!(
        "mode={mode} iters={iters} elapsed={elapsed:.1}s throughput={:.1} G mul-adds/s (checksum {checksum:.3})",
        total_ma / elapsed / 1e9
    );
}
