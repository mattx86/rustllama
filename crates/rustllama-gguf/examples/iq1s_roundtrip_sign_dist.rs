//! Empirically check the sign distribution of IQ1_S round-trip.
//! Encode a balanced N(0,σ) random vector, dequant, and report the
//! sign distribution of the dequantized values. If our encoder is
//! correctly handling signs, the dequant should produce roughly
//! 50/50 positive/negative.
use rustllama_gguf::{dequant::dequant_iq1_s, encode_iq_vec::encode_iq1_s};

fn main() {
    const QK_K: usize = 256;
    const BLOCK_IQ1_S_BYTES: usize = 50;
    const N_BLOCKS: usize = 32;

    // Build N(0, σ) source like real model weights.
    let n = N_BLOCKS * QK_K;
    let mut s = 0x12345u64;
    let mut src = vec![0f32; n];
    for v in src.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((s >> 11) as u32 & 0x00FF_FFFF) as f32 / (1u32 << 24) as f32;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = ((s >> 11) as u32 & 0x00FF_FFFF) as f32 / (1u32 << 24) as f32;
        // Box-Muller
        let mag = (-2.0 * u1.ln().max(-37.0)).sqrt();
        *v = mag * (2.0 * std::f32::consts::PI * u2).cos() * 0.02; // σ ≈ 0.02 (typical)
    }
    let src_pos = src.iter().filter(|v| **v > 0.0).count();
    let src_neg = src.iter().filter(|v| **v < 0.0).count();
    let src_rms = (src.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
    println!("source: rms={src_rms:.3e}, +ve={src_pos}/{n} ({:.1}%), -ve={src_neg}/{n} ({:.1}%)",
        100.0 * src_pos as f32 / n as f32,
        100.0 * src_neg as f32 / n as f32);

    let mut enc = vec![0u8; N_BLOCKS * BLOCK_IQ1_S_BYTES];
    encode_iq1_s(&src, &mut enc);

    let mut dec = vec![0f32; n];
    dequant_iq1_s(&enc, &mut dec);
    let dec_pos = dec.iter().filter(|v| **v > 0.0).count();
    let dec_neg = dec.iter().filter(|v| **v < 0.0).count();
    let dec_zero = dec.iter().filter(|v| **v == 0.0).count();
    let dec_rms = (dec.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
    println!("dequant: rms={dec_rms:.3e}, +ve={dec_pos}/{n} ({:.1}%), -ve={dec_neg}/{n} ({:.1}%), =0={dec_zero}",
        100.0 * dec_pos as f32 / n as f32,
        100.0 * dec_neg as f32 / n as f32);

    // Per-block-mean: are blocks biased one way?
    let mut neg_block_count = 0;
    let mut pos_block_count = 0;
    for b in 0..N_BLOCKS {
        let block = &dec[b * QK_K..(b + 1) * QK_K];
        let mean: f32 = block.iter().sum::<f32>() / QK_K as f32;
        if mean < 0.0 { neg_block_count += 1; } else if mean > 0.0 { pos_block_count += 1; }
    }
    println!("per-block mean: {neg_block_count}/{N_BLOCKS} negative-mean, {pos_block_count}/{N_BLOCKS} positive-mean");

    // Reconstruction MSE
    let mse: f64 = src.iter().zip(dec.iter()).map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2)).sum::<f64>() / n as f64;
    println!("MSE = {mse:.3e} (RMSE = {:.3e})", mse.sqrt());
}
