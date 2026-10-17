//! M1: the synthetic DeepSeek-V2/V3 MLA GGUF writer emits the expected
//! `deepseek2` metadata + the full Multi-head Latent Attention tensor set
//! in the right shapes. This is the free-crate half of the M1 fixture
//! coverage (the real `forward_one_mla` runs in `rustllama-models`).
//!
//! Run with: `cargo test -p rustllama-gguf --features synth`.
#![cfg(feature = "synth")]

use rustllama_gguf::synth::{write_synthetic_deepseek_mla_gguf, SynthMla, SynthMoe};
use rustllama_gguf::{Gguf, MetadataValue};

fn u32_meta(g: &Gguf, k: &str) -> Option<u32> {
    match g.metadata_get(k)? {
        MetadataValue::U32(v) => Some(*v),
        MetadataValue::I32(v) if *v >= 0 => Some(*v as u32),
        _ => None,
    }
}

#[test]
fn synth_mla_emits_deepseek2_metadata_and_tensors() {
    let path = std::env::temp_dir().join("rustllama-mla-meta.gguf");
    let p = SynthMla {
        n_heads: 2,
        d_model: 32,
        q_lora_rank: Some(16),
        kv_lora_rank: 16,
        qk_nope_head_dim: 8,
        qk_rope_head_dim: 8,
        v_head_dim: 8,
        ..SynthMla::default()
    };
    write_synthetic_deepseek_mla_gguf(&path, &p);
    let g = Gguf::open(&path).expect("open");

    assert_eq!(g.architecture(), Some("deepseek2"));
    assert_eq!(u32_meta(&g, "deepseek2.attention.kv_lora_rank"), Some(16));
    assert_eq!(u32_meta(&g, "deepseek2.attention.q_lora_rank"), Some(16));
    assert_eq!(u32_meta(&g, "deepseek2.rope.dimension_count"), Some(8));
    assert_eq!(u32_meta(&g, "deepseek2.attention.key_length"), Some(8));
    assert_eq!(u32_meta(&g, "deepseek2.attention.value_length"), Some(8));

    // Low-rank Q tensors present; direct Q absent.
    assert!(g.tensor("blk.0.attn_q_a.weight").is_some());
    assert!(g.tensor("blk.0.attn_q_a_norm.weight").is_some());
    assert!(g.tensor("blk.0.attn_q_b.weight").is_some());
    assert!(g.tensor("blk.0.attn_q.weight").is_none());

    // KV path tensors + shapes. GGUF dims are `[in, out]`.
    let kv_a = g.tensor("blk.0.attn_kv_a_mqa.weight").expect("kv_a_mqa");
    assert_eq!(
        kv_a.dims,
        vec![32, 16 + 8],
        "attn_kv_a_mqa = [d_model, kvr+rope]"
    );
    let kv_b = g.tensor("blk.0.attn_kv_b.weight").expect("kv_b");
    // out = n_heads*(qk_nope + v) = 2*(8+8) = 32.
    assert_eq!(
        kv_b.dims,
        vec![16, 32],
        "attn_kv_b = [kvr, n_heads*(nope+v)]"
    );
    assert!(g.tensor("blk.0.attn_kv_a_norm.weight").is_some());
    assert!(g.tensor("blk.0.attn_output.weight").is_some());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn synth_mla_direct_q_variant_omits_low_rank_q() {
    let path = std::env::temp_dir().join("rustllama-mla-directq.gguf");
    let p = SynthMla {
        q_lora_rank: None,
        ..SynthMla::default()
    };
    write_synthetic_deepseek_mla_gguf(&path, &p);
    let g = Gguf::open(&path).expect("open");
    assert!(
        g.tensor("blk.0.attn_q.weight").is_some(),
        "direct Q present"
    );
    assert!(g.tensor("blk.0.attn_q_a.weight").is_none(), "no low-rank Q");
    assert!(g.metadata_get("deepseek2.attention.q_lora_rank").is_none());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn synth_mla_moe_variant_emits_expert_tensors_after_dense_prefix() {
    let path = std::env::temp_dir().join("rustllama-mla-moe.gguf");
    let p = SynthMla {
        n_layers: 3,
        n_dense_layers: 1,
        moe: Some(SynthMoe {
            n_experts: 4,
            n_experts_used: 2,
            n_experts_shared: 1,
        }),
        ..SynthMla::default()
    };
    write_synthetic_deepseek_mla_gguf(&path, &p);
    let g = Gguf::open(&path).expect("open");
    assert_eq!(u32_meta(&g, "deepseek2.expert_count"), Some(4));
    // Layer 0 is dense (first_k_dense_replace), layers 1-2 are MoE.
    assert!(
        g.tensor("blk.0.ffn_gate.weight").is_some(),
        "layer 0 dense FFN"
    );
    assert!(g.tensor("blk.0.ffn_gate_exps.weight").is_none());
    assert!(
        g.tensor("blk.1.ffn_gate_exps.weight").is_some(),
        "layer 1 MoE FFN"
    );
    assert!(
        g.tensor("blk.1.ffn_gate_shexp.weight").is_some(),
        "shared expert"
    );
    let _ = std::fs::remove_file(&path);
}
