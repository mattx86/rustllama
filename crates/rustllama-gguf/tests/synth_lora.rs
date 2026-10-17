//! P11: the synthetic LoRA adapter GGUF writer emits the public ggml
//! adapter keys (`adapter.type = "lora"` + `adapter.lora.alpha`) + the
//! per-projection `.lora_a` / `.lora_b` factor pairs in the right shapes.
//! This is the free-crate half of the P11 fixture coverage (the real
//! merge / delta-apply run in `rustllama-models`).
//!
//! Run with: `cargo test -p rustllama-gguf --features synth`.
#![cfg(feature = "synth")]

use rustllama_gguf::synth::{write_synthetic_lora_adapter_gguf, SynthLlama, SynthLoraAdapter};
use rustllama_gguf::{GgmlType, Gguf, MetadataValue};

fn str_meta<'a>(g: &'a Gguf, k: &str) -> Option<&'a str> {
    match g.metadata_get(k)? {
        MetadataValue::String(s) => Some(s.as_str()),
        _ => None,
    }
}

#[test]
fn synth_lora_emits_adapter_metadata_and_factor_pairs() {
    let path = std::env::temp_dir().join("rustllama-lora-fixture.gguf");
    let base = SynthLlama::default();
    let params = SynthLoraAdapter {
        base: base.clone(),
        rank: 8,
        alpha: 16.0,
        seed: 0xABCD_1234,
    };
    write_synthetic_lora_adapter_gguf(&path, &params);

    let g = Gguf::open(&path).expect("open synthetic lora adapter gguf");

    // Adapter identity keys.
    assert_eq!(str_meta(&g, "general.architecture"), Some("llama"));
    assert_eq!(str_meta(&g, "general.type"), Some("adapter"));
    assert_eq!(str_meta(&g, "adapter.type"), Some("lora"));
    assert_eq!(
        g.metadata_get("adapter.lora.alpha")
            .and_then(|v| v.as_f32()),
        Some(16.0)
    );

    // Base geometry → expected factor shapes.
    let d_model = base.d_model as u64;
    let d_q = (base.n_heads * base.head_dim) as u64;
    let d_kv = (base.n_kv_heads * base.head_dim) as u64;
    let d_ff = base.d_ff as u64;
    let rank = params.rank as u64;

    // Spot-check layer 0's attn_q pair: a ne = [in, rank], b ne = [rank, out].
    let a = g
        .tensor("blk.0.attn_q.weight.lora_a")
        .expect("attn_q lora_a present");
    assert_eq!(a.dims, vec![d_model, rank], "lora_a ne must be [in, rank]");
    assert_eq!(a.dtype, GgmlType::F32, "factors are F32");
    let b = g
        .tensor("blk.0.attn_q.weight.lora_b")
        .expect("attn_q lora_b present");
    assert_eq!(b.dims, vec![rank, d_q], "lora_b ne must be [rank, out]");

    // ffn_down has the transposed shape (out = d_model, in = d_ff).
    let fd_a = g
        .tensor("blk.0.ffn_down.weight.lora_a")
        .expect("ffn_down lora_a present");
    assert_eq!(fd_a.dims, vec![d_ff, rank]);
    let fd_b = g
        .tensor("blk.0.ffn_down.weight.lora_b")
        .expect("ffn_down lora_b present");
    assert_eq!(fd_b.dims, vec![rank, d_model]);

    // attn_k / attn_v use the GQA kv width.
    let k_b = g
        .tensor("blk.1.attn_k.weight.lora_b")
        .expect("layer-1 attn_k lora_b present");
    assert_eq!(k_b.dims, vec![rank, d_kv]);

    // Every adapted projection emits BOTH halves — 7 projections × 2 layers.
    let n_lora_a = g
        .tensors()
        .iter()
        .filter(|t| t.name.ends_with(".lora_a"))
        .count();
    let n_lora_b = g
        .tensors()
        .iter()
        .filter(|t| t.name.ends_with(".lora_b"))
        .count();
    assert_eq!(n_lora_a, 7 * base.n_layers as usize);
    assert_eq!(n_lora_b, 7 * base.n_layers as usize);

    let _ = std::fs::remove_file(&path);
}
