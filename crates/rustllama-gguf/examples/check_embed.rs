//! Diagnostic: open a GGUF and report whether `token_embd.weight` is
//! literally all-zero bytes. Used to confirm whether a misbehaving
//! quantized model's embed table got zeroed somewhere along the
//! load/requant pipeline.
//!
//! Usage: cargo run --release -p rustllama-gguf --example check_embed -- /path/to/model.gguf

use rustllama_gguf::Gguf;

fn main() {
    let path = std::env::args().nth(1).expect("usage: check_embed PATH");
    let g = Gguf::open(&path).expect("open gguf");
    let info = g
        .tensor("token_embd.weight")
        .expect("no token_embd.weight in file")
        .clone();
    let bytes = g
        .tensor_bytes("token_embd.weight")
        .expect("read tensor bytes");
    println!("file: {path}");
    println!("  shape: {:?}", info.dims);
    println!("  dtype: {:?}", info.dtype);
    println!("  bytes: {} ({:.2} MiB)", bytes.len(), bytes.len() as f64 / (1024.0 * 1024.0));
    let nonzero = bytes.iter().filter(|b| **b != 0).count();
    println!("  nonzero bytes: {nonzero} of {} ({:.4}%)", bytes.len(), 100.0 * nonzero as f64 / bytes.len() as f64);
    // Sample first 32 bytes + bytes around several token-row offsets.
    let n = info.dims.iter().product::<u64>() as usize;
    let dim_total = if info.dims.len() == 2 {
        info.dims[1] as usize
    } else {
        n
    };
    let row_bytes = bytes.len() / dim_total.max(1);
    println!("  row_bytes (assuming [vocab, d] row-major): {row_bytes}");
    for off in [0usize, row_bytes, row_bytes * 5834, row_bytes * (dim_total / 2), bytes.len().saturating_sub(16)].iter() {
        if let Some(slice) = bytes.get(*off..*off + 16) {
            let nz = slice.iter().filter(|b| **b != 0).count();
            let hex: Vec<String> = slice.iter().take(16).map(|b| format!("{b:02x}")).collect();
            println!("  off={off:>11} nonzero={nz}/16 {}", hex.join(" "));
        }
    }
}
