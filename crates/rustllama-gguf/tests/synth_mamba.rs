//! F9 — the synthetic PURE Mamba/SSM GGUF fixture: the hybrid SSM tensor
//! emission with ZERO full-attention layers (`full_attention_interval` set one
//! past the last layer). This is the free-crate half of the F9 coverage: it
//! pins the fixture's metadata + tensor layout. The real `forward_one_hybrid`
//! load + run on this fixture lives in `rustllama-models` / `rustllama-engine`
//! (a pure Mamba/SSM model IS the hybrid forward with no attention layers).
//!
//! Run with: `cargo test -p rustllama-gguf --features synth`.
#![cfg(feature = "synth")]

use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthHybrid, SynthLlama};
use rustllama_gguf::{Gguf, MetadataValue};

fn u32_meta(g: &Gguf, k: &str) -> Option<u32> {
    match g.metadata_get(k)? {
        MetadataValue::U32(v) => Some(*v),
        MetadataValue::U64(v) => Some(*v as u32),
        MetadataValue::I32(v) if *v >= 0 => Some(*v as u32),
        _ => None,
    }
}

#[test]
fn synth_pure_ssm_emits_ssm_tensors_on_every_layer_and_no_attention() {
    let n_layers = 4u32;
    let path = std::env::temp_dir().join("rustllama-pure-ssm-meta.gguf");
    // F8/F9-shaped geometry: head_dim | ssm_inner (32/8 = 4 v-heads) and
    // head_dim | d_model (32/8 = 4 qk-heads), so the hybrid forward's shape
    // asserts all hold. `pure_ssm` sets full_attention_interval = n_layers + 1.
    let params = SynthLlama {
        n_layers: n_layers as u32,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        d_model: 32,
        d_ff: 64,
        vocab: 32,
        ctx: 64,
        hybrid: Some(SynthHybrid::pure_ssm(n_layers, 32)),
        ..SynthLlama::default()
    };
    write_synthetic_llama_gguf(&path, &params);
    let g = Gguf::open(&path).expect("open");

    // Hybrid detection keys: full_attention_interval (> n_layers → pure SSM) +
    // ssm.state_size must both be present on the `llama` arch.
    assert_eq!(g.architecture(), Some("llama"));
    assert_eq!(
        u32_meta(&g, "llama.full_attention_interval"),
        Some(n_layers + 1),
        "full_attention_interval must be past the last layer (zero attention)"
    );
    assert!(u32_meta(&g, "llama.ssm.state_size").is_some());
    assert!(u32_meta(&g, "llama.ssm.conv_kernel").is_some());

    // EVERY layer carries the SSM/DeltaNet tensor set and NO full-attention
    // tensors — the loader keys layer kind on `attn_q.weight` presence.
    for i in 0..n_layers {
        let p = format!("blk.{i}");
        assert!(
            g.tensor(&format!("{p}.attn_qkv.weight")).is_some(),
            "layer {i}: fused SSM attn_qkv present"
        );
        assert!(
            g.tensor(&format!("{p}.ssm_conv1d.weight")).is_some(),
            "layer {i}: ssm_conv1d present"
        );
        assert!(
            g.tensor(&format!("{p}.ssm_alpha.weight")).is_some(),
            "layer {i}: ssm_alpha (B-proj) present"
        );
        assert!(
            g.tensor(&format!("{p}.ssm_beta.weight")).is_some(),
            "layer {i}: ssm_beta (C-proj) present"
        );
        assert!(
            g.tensor(&format!("{p}.ssm_out.weight")).is_some(),
            "layer {i}: ssm_out present"
        );
        assert!(
            g.tensor(&format!("{p}.attn_q.weight")).is_none(),
            "layer {i}: NO full-attention attn_q (pure SSM)"
        );
    }

    let _ = std::fs::remove_file(&path);
}
