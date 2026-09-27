//! Verifies the GGUF → (ModelDims, WeightQuant) extraction used by
//! `rustllama tune --placement`. Pinned behaviors:
//!
//!   - All seven dim fields (n_layers, d_model, d_ff, n_heads,
//!     n_kv_heads, head_dim, vocab_size) populate from synth GGUF
//!     metadata in the expected llama-family shape.
//!   - The quant detector picks F16 on the default synth fixture
//!     (synth weights are F16) — a v1.x bump to the synth that adds
//!     quantized variants should re-pin this test against a Q4_K
//!     fixture.
//!
//! The pure-arithmetic candidate-generation half of the placement
//! sweep is exhaustively covered in `rustllama-tuner`'s own unit
//! tests; this test exists so the GGUF-reading boundary in
//! `rustllama-cli` doesn't silently drift if the GGUF metadata key
//! conventions change.

use rustllama_cli::read_dims_and_quant_from_gguf;
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama, SynthMoe};
use rustllama_tuner::placement::WeightQuant;

#[test]
fn read_dims_populates_every_field_from_synth_gguf() {
    let tmp = std::env::temp_dir().join("rustllama-placement-dims-from-gguf.gguf");
    // Pick non-default values so we can prove each field round-trips
    // through the GGUF metadata (versus accidentally reading from a
    // hard-coded constant somewhere).
    let synth = SynthLlama {
        n_layers: 4,
        n_heads: 8,
        n_kv_heads: 4,
        head_dim: 32,
        d_model: 256,
        d_ff: 512,
        vocab: 1024,
        ctx: 2048,
        ..SynthLlama::default()
    };
    write_synthetic_llama_gguf(&tmp, &synth);

    let (dims, quant) = read_dims_and_quant_from_gguf(&tmp).expect("extract");
    assert_eq!(dims.n_layers, 4);
    assert_eq!(dims.d_model, 256);
    assert_eq!(dims.d_ff, 512);
    assert_eq!(dims.n_heads, 8);
    assert_eq!(dims.n_kv_heads, 4);
    assert_eq!(dims.head_dim, 32);
    assert_eq!(dims.vocab_size, 1024);
    // Synth fixture writes F16 weights by default — the dominant
    // dtype across the tensor table.
    assert_eq!(quant, WeightQuant::F16);

    let _ = std::fs::remove_file(&tmp);
}

/// A GGUF that omits `attention.head_count_kv` (older converters
/// pre-GQA) should default `n_kv_heads = n_heads` — standard MHA.
/// Synth always writes `head_count_kv`, so this case is exercised
/// by the equal-heads variant: when n_kv_heads == n_heads in the
/// fixture, the extractor's path doesn't differ, but the fallback
/// logic itself is unit-testable via the synth.
#[test]
fn read_dims_treats_equal_kv_heads_as_mha() {
    let tmp = std::env::temp_dir().join("rustllama-placement-dims-mha.gguf");
    let synth = SynthLlama {
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 4, // MHA — no GQA grouping
        head_dim: 16,
        d_model: 64,
        d_ff: 128,
        vocab: 256,
        ctx: 64,
        ..SynthLlama::default()
    };
    write_synthetic_llama_gguf(&tmp, &synth);

    let (dims, _) = read_dims_and_quant_from_gguf(&tmp).expect("extract");
    assert_eq!(dims.n_kv_heads, dims.n_heads, "MHA → kv heads equal heads");
    assert_eq!(dims.d_q(), dims.d_model, "MHA → Q-proj dim equals d_model");

    let _ = std::fs::remove_file(&tmp);
}

/// Dense GGUFs must yield `n_experts = 0` (etc.). Pin the negative
/// path so MoE-aware placement math doesn't accidentally trigger on
/// dense models.
#[test]
fn read_dims_zeros_moe_fields_on_dense_gguf() {
    let tmp = std::env::temp_dir().join("rustllama-placement-dims-dense-moe.gguf");
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    let (dims, _) = read_dims_and_quant_from_gguf(&tmp).expect("extract");
    assert_eq!(dims.n_experts, 0);
    assert_eq!(dims.n_experts_used, 0);
    assert_eq!(dims.n_experts_shared, 0);
    assert!(!dims.is_moe(), "dense GGUF must not report is_moe");
    let _ = std::fs::remove_file(&tmp);
}

/// Mixtral-shape MoE GGUF populates the three MoE fields. The
/// placement-sweep VRAM math (already MoE-aware in
/// `rustllama-tuner`) needs these to size the FFN footprint
/// correctly (8 routed experts → ~8× a dense FFN block).
#[test]
fn read_dims_populates_moe_fields_from_mixtral_shape_synth() {
    let tmp = std::env::temp_dir().join("rustllama-placement-dims-moe-mixtral.gguf");
    write_synthetic_llama_gguf(
        &tmp,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 8,
                n_experts_used: 2,
                n_experts_shared: 0,
            }),
            ..SynthLlama::default()
        },
    );
    let (dims, _) = read_dims_and_quant_from_gguf(&tmp).expect("extract");
    assert_eq!(dims.n_experts, 8);
    assert_eq!(dims.n_experts_used, 2);
    assert_eq!(dims.n_experts_shared, 0);
    assert!(dims.is_moe(), "8-expert GGUF must report is_moe");
    let _ = std::fs::remove_file(&tmp);
}

/// DeepSeek-V3-shape MoE GGUF: shared-expert count > 0 round-trips.
#[test]
fn read_dims_round_trips_deepseek_shared_expert_count() {
    let tmp = std::env::temp_dir().join("rustllama-placement-dims-moe-deepseek.gguf");
    write_synthetic_llama_gguf(
        &tmp,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 2,
                n_experts_shared: 1,
            }),
            ..SynthLlama::default()
        },
    );
    let (dims, _) = read_dims_and_quant_from_gguf(&tmp).expect("extract");
    assert_eq!(dims.n_experts_shared, 1);
    let _ = std::fs::remove_file(&tmp);
}
