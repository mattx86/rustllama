//! Print every tensor name + dtype + dims + byte-size for a GGUF.
//! Used to vet recipe coverage when re-quantizing.
//! Usage: cargo run --release -p rustllama-gguf --example list_tensors -- /path/to/model.gguf
use rustllama_gguf::Gguf;
use std::collections::BTreeMap;

fn main() {
    let path = std::env::args().nth(1).expect("usage: list_tensors PATH");
    let g = Gguf::open(&path).expect("open gguf");
    let mut by_pattern: BTreeMap<String, (u64, u64, String)> = BTreeMap::new();
    let mut total_bytes: u64 = 0;
    let mut total_count: u64 = 0;
    for info in g.tensors() {
        let bytes = info.byte_size as u64;
        total_bytes += bytes;
        total_count += 1;
        let canonical = canonicalize(&info.name);
        let entry = by_pattern.entry(canonical).or_insert((0, 0, format!("{:?}", info.dtype)));
        entry.0 += 1;
        entry.1 += bytes;
        if !entry.2.contains(&format!("{:?}", info.dtype)) {
            entry.2.push_str(&format!("|{:?}", info.dtype));
        }
    }
    println!("source: {path}");
    println!("total: {total_count} tensors, {:.2} MiB", total_bytes as f64 / (1024.0 * 1024.0));
    println!();
    println!("{:<55} {:>6} {:>14}  dtype", "pattern", "count", "MiB");
    for (pattern, (count, bytes, dtype)) in &by_pattern {
        println!(
            "{:<55} {:>6} {:>14.2}  {}",
            pattern,
            count,
            *bytes as f64 / (1024.0 * 1024.0),
            dtype
        );
    }
}

// Collapse numeric layer ids and expert ids so e.g.
// "blk.3.ffn_gate_exps.weight" and "blk.7.ffn_gate_exps.weight" both
// map to "blk.*.ffn_gate_exps.weight".
fn canonicalize(name: &str) -> String {
    name.split('.')
        .map(|seg| if seg.chars().all(|c| c.is_ascii_digit()) { "*".to_string() } else { seg.to_string() })
        .collect::<Vec<_>>()
        .join(".")
}
