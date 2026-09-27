//! Verify shared expert tensor dims match config'd d_ff.
use rustllama_gguf::Gguf;

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_shexp_shapes PATH");
    let g = Gguf::open(&path).expect("open gguf");
    for name in [
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_gate_inp_shexp.weight",
        "blk.0.ffn_gate_shexp.weight",
        "blk.0.ffn_up_shexp.weight",
        "blk.0.ffn_down_shexp.weight",
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
    ] {
        match g.tensor(name) {
            Some(t) => println!("  {name}: dims={:?} dtype={:?}", t.dims, t.dtype),
            None => println!("  {name}: MISSING"),
        }
    }
}
