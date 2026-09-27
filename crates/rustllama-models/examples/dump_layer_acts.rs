//! Per-layer activation dumper for diff against an HF reference forward.
//!
//! Loads a tiny qwen35moe-format GGUF, runs `forward_one_hybrid_to_hidden`
//! on a single token, and writes the hidden-state vector AT EACH LAYER
//! BOUNDARY to a binary .npy-compatible file. The Python diff script
//! then loads HF's npz + our .bin files and computes per-layer cosine
//! + relative-error.
//!
//! Usage:
//!   cargo run --release -p rustllama-models --example dump_layer_acts -- \
//!     <model.gguf> <token_id> <out_dir>
//!
//! Mode: sets RUSTLLAMA_DUMP_ACTS=<out_dir> and runs the forward; the
//! forward writes a file per layer-stage (after embed, post-ssm,
//! post-ssm-ffn, post-attn, post-attn-ffn, final post-block,
//! post-norm, logits).

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use rustllama_gguf::Gguf;
use rustllama_models::llama_arch::{DeltaNetCache, HybridLayer, KvCache, LlamaModel};

fn write_f32_bin(path: &PathBuf, data: &[f32]) {
    let mut f = fs::File::create(path).expect("create dump file");
    let bytes: &[u8] = bytemuck::cast_slice(data);
    f.write_all(bytes).expect("write dump file");
}

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().expect("usage: dump_layer_acts <gguf> <token_id> <out_dir>");
    let token_id: i32 = args
        .next()
        .expect("token_id")
        .parse()
        .expect("token_id u32");
    let out_dir: PathBuf = args
        .next()
        .expect("out_dir")
        .into();
    fs::create_dir_all(&out_dir).expect("create out_dir");

    // Tell the forward to dump per-layer hidden states. The forward
    // checks RUSTLLAMA_DUMP_ACTS_DIR each layer and writes
    // `<dir>/layer_<NN>_<stage>.bin` as raw f32 little-endian.
    env::set_var("RUSTLLAMA_DUMP_ACTS_DIR", out_dir.to_str().unwrap());

    println!("opening {path}");
    let gguf = Gguf::open(&path).expect("open gguf");
    let model = LlamaModel::load_allow_moe(&gguf).expect("load model");
    let cfg = model.cfg.clone();
    println!(
        "config: n_layers={} d_model={} n_heads={} n_kv_heads={} head_dim={} rope_dim={} vocab={} moe={:?}",
        cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rope_dim,
        cfg.vocab_size, cfg.moe
    );

    let mut kv = KvCache::new_with_dtype(
        &cfg,
        1,
        rustllama_models::llama_arch::KvDtype::F32,
    );
    let hybrid_layers: &[HybridLayer] = model
        .weights
        .hybrid_layers
        .as_ref()
        .expect("model is not hybrid — this tool only diagnoses qwen35moe");
    let mut dn = DeltaNetCache::new_for_hybrid(&cfg, hybrid_layers)
        .expect("DeltaNetCache for hybrid");

    let mut logits = vec![0.0f32; cfg.vocab_size];
    println!("running forward at pos=0 for token_id={token_id}...");
    model.forward_one_hybrid(token_id, 0, &mut kv, &mut dn, &mut logits);

    // Also write final logits.
    write_f32_bin(&out_dir.join("logits.bin"), &logits);
    println!("wrote logits.bin ({} floats)", logits.len());
    println!("done. activations in {}", out_dir.display());
}
