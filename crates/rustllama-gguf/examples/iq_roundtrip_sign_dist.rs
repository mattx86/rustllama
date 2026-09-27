//! Roundtrip sign-distribution test for all IQ formats we use.
//! Encode → dequant a balanced N(0,σ) source; report sign balance.
use rustllama_gguf::{
    dequant::{
        dequant_iq1_m, dequant_iq1_s, dequant_iq2_s, dequant_iq2_xs, dequant_iq2_xxs,
        dequant_iq3_s, dequant_iq3_xxs,
    },
    encode_iq_vec::{
        encode_iq1_m, encode_iq1_s, encode_iq2_s, encode_iq2_xs, encode_iq2_xxs, encode_iq3_s,
        encode_iq3_xxs,
    },
};

fn run<E: Fn(&[f32], &mut [u8]), D: Fn(&[u8], &mut [f32])>(
    name: &str,
    block_bytes: usize,
    encode: E,
    dequant: D,
) {
    const QK_K: usize = 256;
    const N_BLOCKS: usize = 32;
    let n = N_BLOCKS * QK_K;
    let mut s = 0x12345u64;
    let mut src = vec![0f32; n];
    for v in src.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((s >> 11) as u32 & 0x00FF_FFFF) as f32 / (1u32 << 24) as f32;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = ((s >> 11) as u32 & 0x00FF_FFFF) as f32 / (1u32 << 24) as f32;
        let mag = (-2.0 * u1.ln().max(-37.0)).sqrt();
        *v = mag * (2.0 * std::f32::consts::PI * u2).cos() * 0.02;
    }
    let mut enc = vec![0u8; N_BLOCKS * block_bytes];
    encode(&src, &mut enc);
    let mut dec = vec![0f32; n];
    dequant(&enc, &mut dec);
    let pos = dec.iter().filter(|v| **v > 0.0).count();
    let neg = dec.iter().filter(|v| **v < 0.0).count();
    let mse: f64 = src
        .iter()
        .zip(dec.iter())
        .map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2))
        .sum::<f64>()
        / n as f64;
    let mut neg_block = 0;
    let mut pos_block = 0;
    for b in 0..N_BLOCKS {
        let block = &dec[b * QK_K..(b + 1) * QK_K];
        let mean: f32 = block.iter().sum::<f32>() / QK_K as f32;
        if mean < 0.0 {
            neg_block += 1;
        } else if mean > 0.0 {
            pos_block += 1;
        }
    }
    let imbalance = (pos as f32 - neg as f32).abs() / n as f32;
    println!(
        "{:8} +ve={:>5} ({:>4.1}%) -ve={:>5} ({:>4.1}%) imb={:>4.1}% blocks: {}+/{}- RMSE={:.3e}",
        name,
        pos,
        100.0 * pos as f32 / n as f32,
        neg,
        100.0 * neg as f32 / n as f32,
        100.0 * imbalance,
        pos_block,
        neg_block,
        mse.sqrt()
    );
}

fn main() {
    println!("source target: 50% +ve / 50% -ve, blocks ~16+/16-");
    run("IQ1_S", 50, encode_iq1_s, dequant_iq1_s);
    run("IQ1_M", 56, encode_iq1_m, dequant_iq1_m);
    run("IQ2_XXS", 66, encode_iq2_xxs, dequant_iq2_xxs);
    run("IQ2_XS", 74, encode_iq2_xs, dequant_iq2_xs);
    run("IQ2_S", 82, encode_iq2_s, dequant_iq2_s);
    run("IQ3_XXS", 98, encode_iq3_xxs, dequant_iq3_xxs);
    run("IQ3_S", 110, encode_iq3_s, dequant_iq3_s);
}
