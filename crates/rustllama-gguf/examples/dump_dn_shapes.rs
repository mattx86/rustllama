//! Print the per-tensor dims of one DeltaNet block from a GGUF so we
//! can verify whether the loader splits qkvz the way HF expects.
//! Usage: cargo run --release -p rustllama-gguf --example dump_dn_shapes -- /path/to/model.gguf
use rustllama_gguf::Gguf;

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_dn_shapes PATH");
    let g = Gguf::open(&path).expect("open gguf");
    println!("file: {path}");
    for name in [
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_alpha.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.ssm_conv1d.weight",
        "blk.0.ssm_out.weight",
        "blk.0.ssm_a",
        "blk.0.ssm_dt.bias",
        "blk.0.ssm_norm.weight",
        "blk.1.attn_q.weight",
        "blk.1.attn_k.weight",
        "blk.1.attn_v.weight",
        "blk.1.attn_output.weight",
        "blk.1.attn_q_norm.weight",
        "blk.1.attn_k_norm.weight",
    ] {
        match g.tensor(name) {
            Some(t) => println!("  {name}: dims={:?} dtype={:?}", t.dims, t.dtype),
            None => println!("  {name}: MISSING"),
        }
    }
    // Echo a handful of metadata keys that pin down the geometry.
    let interesting = [
        "general.architecture",
        "qwen35moe.block_count",
        "qwen35moe.embedding_length",
        "qwen35moe.attention.head_count",
        "qwen35moe.attention.head_count_kv",
        "qwen35moe.attention.key_length",
        "qwen35moe.attention.value_length",
        "qwen35moe.attention.layer_norm_rms_epsilon",
        "qwen35moe.rope.dimension_count",
        "qwen35moe.rope.freq_base",
        "qwen35moe.rope.scaling.type",
        "qwen35moe.rope.scaling.factor",
        "qwen35moe.expert_count",
        "qwen35moe.expert_used_count",
        "qwen35moe.expert_shared_count",
        "qwen35moe.ssm.inner_size",
        "qwen35moe.ssm.head_count",
        "qwen35moe.ssm.head_dim",
        "qwen35moe.ssm.conv_kernel",
        "qwen35moe.ssm.state_size",
        "qwen35moe.ssm.group_count",
        "qwen35moe.nextn_predict_layers",
    ];
    println!();
    for k in interesting {
        match g.metadata().iter().find(|(kk, _)| kk == k) {
            Some((_, v)) => println!("  {k} = {:?}", v),
            None => println!("  {k}: (not present)"),
        }
    }
}
