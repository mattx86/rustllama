//! Print rms (sqrt(mean(x*x))) of per-layer norm tensors so we can
//! spot if any layer's norm weights are out of band (signature of
//! mis-aligned tensor offsets during load).
//!
//! Usage: cargo run --release -p rustllama-gguf --example dump_norm_rms -- /path/to/model.gguf
use rustllama_gguf::{Gguf, GgmlType};

fn rms_f32_bytes(bytes: &[u8]) -> f64 {
    let n = bytes.len() / 4;
    let mut s = 0f64;
    for i in 0..n {
        let v = f32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]);
        s += (v as f64) * (v as f64);
    }
    (s / (n as f64)).sqrt()
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_norm_rms PATH");
    let g = Gguf::open(&path).expect("open gguf");
    println!("file: {path}");
    println!("{:<40} {:>8} {:>12}", "tensor", "elems", "rms");
    for li in 0..42 {
        for stem in [
            "attn_norm",
            "post_attention_norm",
            "attn_q_norm",
            "attn_k_norm",
            "ssm_norm",
        ] {
            let name = format!("blk.{li}.{stem}.weight");
            if let Some(info) = g.tensor(&name) {
                if info.dtype == GgmlType::F32 {
                    if let Some(bytes) = g.tensor_bytes(&name) {
                        let rms = rms_f32_bytes(bytes);
                        let n_elems: u64 = info.dims.iter().product();
                        println!("{name:<40} {n_elems:>8} {rms:>12.6e}");
                    }
                }
            }
        }
    }
}
