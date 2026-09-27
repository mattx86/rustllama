//! Dump which tensors exist for blk.40 (the NextN slot).
use rustllama_gguf::Gguf;

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_blk40 PATH");
    let g = Gguf::open(&path).expect("open gguf");
    println!("file: {path}");
    println!("--- blk.40.* tensors ---");
    for t in g.tensors() {
        if t.name.starts_with("blk.40.") {
            println!("  {}: dims={:?} dtype={:?}", t.name, t.dims, t.dtype);
        }
    }
    println!("--- blk.39.* tensors (for comparison) ---");
    for t in g.tensors() {
        if t.name.starts_with("blk.39.") {
            println!("  {}: dims={:?} dtype={:?}", t.name, t.dims, t.dtype);
        }
    }
}
