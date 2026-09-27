//! Scan a PTQ1_0 (Bonsai ternary) GGUF and report the zero
//! distribution — the data grounding for "can dense-ternary behave
//! like MoE" (structured sparsity we could skip).
//!
//! Reports, aggregated over all PTQ1_0 tensors and for the largest
//! FFN tensors: overall {-1,0,+1} fractions; block-level structure
//! (fraction of 128-blocks that are ALL-zero → skippable matvec
//! blocks); row-level (fraction of output rows that are all-zero →
//! dead neurons); and column/lane structure (are certain positions
//! within the 128-block zero across most blocks → structured column
//! sparsity). Usage: ptq_zero_scan <gguf-path>

use rustllama_gguf::Gguf;
use rustllama_gguf::GgmlType;

const BLOCK_BYTES: usize = 28;
const QS_BYTES: usize = 24;
const QK: usize = 128;
const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
const STAGES: [usize; 3] = [32, 16, 8];

/// Decode one 28-byte block into 128 trits in {-1,0,1}, element order
/// matching the kernel walk.
fn decode_block(blk: &[u8], out: &mut [i8; QK]) {
    let qs = &blk[..QS_BYTES];
    let qh = &blk[QS_BYTES..QS_BYTES + 2];
    let mut e = 0usize;
    let mut j = 0usize;
    for &c in STAGES.iter() {
        while j + c <= QS_BYTES {
            for n in 0..5 {
                for mm in 0..c {
                    let q = qs[j + mm].wrapping_mul(POW3[n]);
                    out[e] = ((((q as u16) * 3) >> 8) as i32 - 1) as i8;
                    e += 1;
                }
            }
            j += c;
        }
    }
    for n in 0..4 {
        for h in 0..2 {
            let q = qh[h].wrapping_mul(POW3[n]);
            out[e] = ((((q as u16) * 3) >> 8) as i32 - 1) as i8;
            e += 1;
        }
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: ptq_zero_scan <gguf>");
    let g = Gguf::open(&path).expect("open gguf");

    let mut g_neg = 0u64;
    let mut g_zero = 0u64;
    let mut g_pos = 0u64;
    let mut g_blocks = 0u64;
    let mut g_allzero_blocks = 0u64;
    let mut g_rows = 0u64;
    let mut g_allzero_rows = 0u64;
    // Per-block-position zero counts (is column j structurally zero?).
    let mut lane_zero = [0u64; QK];
    let mut n_ptq = 0usize;

    // Track the worst/best FFN tensors for a per-tensor sample.
    let mut samples: Vec<(String, f64, f64, f64)> = Vec::new(); // name, zero%, allzero-block%, allzero-row%

    for t in g.tensors() {
        if t.dtype != GgmlType::PTQ1_0 {
            continue;
        }
        n_ptq += 1;
        let bytes = g.tensor_bytes(&t.name).expect("tensor bytes");
        // row length in elements = last dim's product? dims[0] is the
        // in-features (k), rows = product of the rest. PTQ1_0 rows are
        // k-contiguous; k = dims[0].
        let k = t.dims[0] as usize;
        let rows: usize = t.dims[1..].iter().product::<u64>().max(1) as usize;
        let blocks_per_row = k / QK;
        let row_bytes = blocks_per_row * BLOCK_BYTES;

        let mut t_zero = 0u64;
        let mut t_total = 0u64;
        let mut t_allzero_blocks = 0u64;
        let mut t_allzero_rows = 0u64;
        let mut tb = [0i8; QK];

        for r in 0..rows {
            let mut row_all_zero = true;
            for b in 0..blocks_per_row {
                let off = r * row_bytes + b * BLOCK_BYTES;
                if off + BLOCK_BYTES > bytes.len() {
                    break;
                }
                decode_block(&bytes[off..off + BLOCK_BYTES], &mut tb);
                let mut blk_zero = 0u32;
                for (i, &v) in tb.iter().enumerate() {
                    match v {
                        0 => {
                            blk_zero += 1;
                            lane_zero[i] += 1;
                            t_zero += 1;
                            g_zero += 1;
                        }
                        1 => g_pos += 1,
                        _ => g_neg += 1,
                    }
                }
                t_total += QK as u64;
                g_blocks += 1;
                if blk_zero == QK as u32 {
                    g_allzero_blocks += 1;
                    t_allzero_blocks += 1;
                } else {
                    row_all_zero = false;
                }
            }
            g_rows += 1;
            if row_all_zero && blocks_per_row > 0 {
                g_allzero_rows += 1;
                t_allzero_rows += 1;
            }
        }

        if t_total > 0 && (t.name.contains("ffn") || t.name.contains("exps")) {
            samples.push((
                t.name.clone(),
                t_zero as f64 / t_total as f64 * 100.0,
                t_allzero_blocks as f64 / (rows * blocks_per_row).max(1) as f64 * 100.0,
                t_allzero_rows as f64 / rows.max(1) as f64 * 100.0,
            ));
        }
    }

    let total = (g_neg + g_zero + g_pos).max(1);
    println!("PTQ1_0 tensors scanned: {n_ptq}");
    println!(
        "overall trits: -1 {:.1}%   0 {:.1}%   +1 {:.1}%   (total {} trits)",
        g_neg as f64 / total as f64 * 100.0,
        g_zero as f64 / total as f64 * 100.0,
        g_pos as f64 / total as f64 * 100.0,
        total
    );
    println!(
        "block structure: {} / {} blocks are ALL-ZERO ({:.4}%)  <- skippable matvec blocks",
        g_allzero_blocks,
        g_blocks,
        g_allzero_blocks as f64 / g_blocks.max(1) as f64 * 100.0
    );
    println!(
        "row structure:   {} / {} output rows are ALL-ZERO ({:.4}%)  <- dead neurons",
        g_allzero_rows,
        g_rows,
        g_allzero_rows as f64 / g_rows.max(1) as f64 * 100.0
    );
    // Lane structure: min/max/avg zero-fraction across the 128 block
    // positions. Uniform ≈ scattered zeros (no column sparsity);
    // spiky ≈ some positions structurally zero.
    let per_lane_blocks = g_blocks.max(1);
    let lane_fracs: Vec<f64> = lane_zero
        .iter()
        .map(|&z| z as f64 / per_lane_blocks as f64 * 100.0)
        .collect();
    let lmin = lane_fracs.iter().cloned().fold(f64::INFINITY, f64::min);
    let lmax = lane_fracs.iter().cloned().fold(0.0, f64::max);
    let lavg = lane_fracs.iter().sum::<f64>() / QK as f64;
    println!(
        "lane structure:  per-position zero% across blocks: min {lmin:.1}  avg {lavg:.1}  max {lmax:.1}  (flat=scattered, spiky=column-structured)"
    );
    println!("\nsample FFN tensors (name | zero% | all-zero-block% | dead-row%):");
    samples.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (name, z, bz, rz) in samples.iter().take(6) {
        println!("  {name:<40} {z:5.1}  {bz:7.4}  {rz:6.3}");
    }
}
