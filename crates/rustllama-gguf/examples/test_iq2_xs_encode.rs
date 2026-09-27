//! Reproduce the embed-table corruption: take the original GGUF's
//! Q2_K token_embd row → dequant → encode as IQ2_XS via the CPU
//! encoder → check if output bytes are non-zero.
//!
//! If output is all zeros, the bug is in the CPU encoder (not the
//! GPU encoder, since the user's requant might have used GPU). If
//! output is non-zero, the bug is GPU-encoder-specific.
//!
//! Usage: cargo run --release -p rustllama-gguf --example test_iq2_xs_encode -- /path/to/original.gguf

use rustllama_gguf::{dequant, encode_iq_vec, Gguf};

fn main() {
    let path = std::env::args().nth(1).expect("usage: test_iq2_xs_encode PATH");
    let g = Gguf::open(&path).expect("open gguf");
    let info = g
        .tensor("token_embd.weight")
        .expect("no token_embd.weight")
        .clone();
    let bytes = g
        .tensor_bytes("token_embd.weight")
        .expect("read tensor bytes");
    println!("source: {path}");
    println!("  dtype: {:?}", info.dtype);
    println!("  dims:  {:?}", info.dims);
    let d_model = info.dims[0] as usize;

    // Take a few token rows from the original and re-encode each as
    // IQ2_XS via the CPU encoder. Verify each produces non-zero
    // output bytes. The token IDs cover the embed range.
    for &token_id in &[0u32, 1, 100, 5834, 100_000, 248_319] {
        // Q2_K row layout: 256 elements per super-block, each super-
        // block stored as a known number of bytes. dequant_q2_k
        // handles the byte slicing.
        // For Q2_K: 84 bytes per 256-elem block. Q3_K: 110. etc.
        let row_bytes_per_block = match info.dtype {
            rustllama_gguf::GgmlType::Q2_K => 84,
            rustllama_gguf::GgmlType::Q3_K => 110,
            rustllama_gguf::GgmlType::Q4_K => 144,
            rustllama_gguf::GgmlType::Q5_K => 176,
            rustllama_gguf::GgmlType::Q6_K => 210,
            other => panic!("source dtype {other:?} not handled in this test"),
        };
        let blocks_per_row = d_model / 256;
        let row_bytes = blocks_per_row * row_bytes_per_block;
        let row_start = (token_id as usize) * row_bytes;
        if row_start + row_bytes > bytes.len() {
            println!("  token {token_id}: out of bounds");
            continue;
        }
        let row = &bytes[row_start..row_start + row_bytes];
        let mut row_f32 = vec![0f32; d_model];
        match info.dtype {
            rustllama_gguf::GgmlType::Q2_K => dequant::dequant_q2_k(row, &mut row_f32),
            rustllama_gguf::GgmlType::Q3_K => dequant::dequant_q3_k(row, &mut row_f32),
            rustllama_gguf::GgmlType::Q4_K => dequant::dequant_q4_k(row, &mut row_f32),
            rustllama_gguf::GgmlType::Q5_K => dequant::dequant_q5_k(row, &mut row_f32),
            rustllama_gguf::GgmlType::Q6_K => dequant::dequant_q6_k(row, &mut row_f32),
            _ => unreachable!(),
        }
        let f32_rms = (row_f32.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
            / row_f32.len() as f64)
            .sqrt();
        let f32_nonzero = row_f32.iter().filter(|v| **v != 0.0).count();
        let f32_first4 = &row_f32[..4];

        // Re-encode as IQ2_XS via the CPU encoder (the synchronous
        // standalone variant, no GPU encoder involved).
        const IQ2XS_BLOCK_BYTES: usize = 74;
        let dst_len = blocks_per_row * IQ2XS_BLOCK_BYTES;
        let mut iq2xs_dst = vec![0u8; dst_len];
        encode_iq_vec::encode_iq2_xs(&row_f32, &mut iq2xs_dst);

        let dst_nonzero = iq2xs_dst.iter().filter(|b| **b != 0).count();
        let dst_first8: Vec<String> = iq2xs_dst.iter().take(8).map(|b| format!("{b:02x}")).collect();
        println!(
            "  token {token_id}:"
        );
        println!(
            "    Q2_K → f32: rms={f32_rms:.4e} nonzero={f32_nonzero}/{d_model} first4={f32_first4:?}"
        );
        println!(
            "    f32 → IQ2_XS: nonzero_bytes={dst_nonzero}/{dst_len} first8={}",
            dst_first8.join(" ")
        );
    }
}
