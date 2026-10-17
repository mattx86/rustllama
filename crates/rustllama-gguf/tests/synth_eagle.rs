//! P10b: the synthetic EAGLE draft-head GGUF writer emits the clean-room EAGLE
//! adapter keys (`general.type = "adapter"` + `adapter.type = "eagle"`) + the
//! `eagle.*` fusion / block tensors in the shapes matched to the base geometry.
//! Free-crate half of the P10b fixture coverage (the loader + drafter run in
//! `rustllama-models` / `rustllama-engine`).
//!
//! Run with: `cargo test -p rustllama-gguf --features synth`.
#![cfg(feature = "synth")]

use rustllama_gguf::synth::{write_synthetic_eagle_head_gguf, SynthEagleHead, SynthLlama};
use rustllama_gguf::{GgmlType, Gguf, MetadataValue};

fn str_meta<'a>(g: &'a Gguf, k: &str) -> Option<&'a str> {
    match g.metadata_get(k)? {
        MetadataValue::String(s) => Some(s.as_str()),
        _ => None,
    }
}

#[test]
fn synth_eagle_emits_adapter_metadata_and_tensors() {
    let path = std::env::temp_dir().join("rustllama-eagle-fixture.gguf");
    let base = SynthLlama::default();
    write_synthetic_eagle_head_gguf(
        &path,
        &SynthEagleHead {
            base: base.clone(),
            ..SynthEagleHead::default()
        },
    );
    let g = Gguf::open(&path).expect("open synthetic eagle head gguf");

    // Adapter identity keys.
    assert_eq!(str_meta(&g, "general.architecture"), Some("llama"));
    assert_eq!(str_meta(&g, "general.type"), Some("adapter"));
    assert_eq!(str_meta(&g, "adapter.type"), Some("eagle"));

    let d_model = base.d_model as u64;
    let d_q = (base.n_heads * base.head_dim) as u64;
    let d_kv = (base.n_kv_heads * base.head_dim) as u64;
    let d_ff = base.d_ff as u64;

    // Fusion: ggml ne = [in = 2·d, out = d].
    let fc = g.tensor("eagle.fc.weight").expect("fc present");
    assert_eq!(fc.dims, vec![2 * d_model, d_model], "fc ne = [2·d, d]");
    assert_eq!(fc.dtype, GgmlType::F32);
    assert!(g.tensor("eagle.fc.bias").is_some());
    assert!(g.tensor("eagle.enorm.weight").is_some());
    assert!(g.tensor("eagle.hnorm.weight").is_some());

    // Decoder block shapes (ggml ne = [in, out]).
    assert_eq!(
        g.tensor("eagle.attn_q.weight").unwrap().dims,
        vec![d_model, d_q]
    );
    assert_eq!(
        g.tensor("eagle.attn_k.weight").unwrap().dims,
        vec![d_model, d_kv]
    );
    assert_eq!(
        g.tensor("eagle.attn_v.weight").unwrap().dims,
        vec![d_model, d_kv]
    );
    assert_eq!(
        g.tensor("eagle.attn_output.weight").unwrap().dims,
        vec![d_q, d_model]
    );
    assert_eq!(
        g.tensor("eagle.ffn_gate.weight").unwrap().dims,
        vec![d_model, d_ff]
    );
    assert_eq!(
        g.tensor("eagle.ffn_up.weight").unwrap().dims,
        vec![d_model, d_ff]
    );
    assert_eq!(
        g.tensor("eagle.ffn_down.weight").unwrap().dims,
        vec![d_ff, d_model]
    );

    // Tied by default: no own LM head / embedding.
    assert!(g.tensor("eagle.output.weight").is_none());
    assert!(g.tensor("eagle.token_embd.weight").is_none());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn synth_eagle_own_lm_head_emits_output_and_embedding() {
    let path = std::env::temp_dir().join("rustllama-eagle-ownhead-fixture.gguf");
    let base = SynthLlama::default();
    write_synthetic_eagle_head_gguf(
        &path,
        &SynthEagleHead {
            base: base.clone(),
            own_lm_head: true,
            ..SynthEagleHead::default()
        },
    );
    let g = Gguf::open(&path).expect("open eagle head gguf");
    let d_model = base.d_model as u64;
    let vocab = base.vocab as u64;
    assert!(g.tensor("eagle.output_norm.weight").is_some());
    assert_eq!(
        g.tensor("eagle.output.weight").unwrap().dims,
        vec![d_model, vocab]
    );
    assert_eq!(
        g.tensor("eagle.token_embd.weight").unwrap().dims,
        vec![d_model, vocab]
    );
    let _ = std::fs::remove_file(&path);
}
