//! Integration tests for `CpuEngine::load_auto` and the explicit
//! `load_safetensors` constructor — A-3 of the AWQ/GPTQ arc.
//!
//! Builds synthetic safetensors directories on disk (config.json +
//! a minimal AWQ model.safetensors) and verifies:
//! - `load_auto` dispatches by extension.
//! - `load_safetensors` finds the sibling config.json.
//! - Missing config.json / tokenizer.json each produce a clear error.

use std::collections::BTreeMap;
use std::path::PathBuf;

use bytemuck::cast_slice;
use half::f16;
use rustllama_engine::CpuEngine;
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

/// Pack 8 4-bit values (lane order 0..8) into one int32 little-endian.
fn pack8(lanes: [u8; 8]) -> i32 {
    let mut acc: u32 = 0;
    for (k, v) in lanes.iter().enumerate() {
        acc |= ((*v as u32) & 0xF) << (k * 4);
    }
    acc as i32
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f16_bytes(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|h| h.to_le_bytes()).collect()
}

/// Build a single-AWQ-linear's qweight/scales/qzeros byte buffers
/// suitable for a [in_features, out_features] linear with one group.
fn awq_linear(in_f: usize, out_f: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let out_packs = out_f / 8;
    let qw: Vec<i32> = (0..in_f * out_packs).map(|_| pack8([1; 8])).collect();
    let sc: Vec<f16> = vec![f16::from_f32(0.5); out_f]; // 1 group × out_f
    let qz: Vec<i32> = (0..out_packs).map(|_| pack8([0; 8])).collect();
    (i32_bytes(&qw), f16_bytes(&sc), i32_bytes(&qz))
}

/// Write a minimal AWQ-shape safetensors directory to a temp dir.
/// Returns the directory path. `with_tokenizer` controls whether a
/// `tokenizer.json` is written (used to exercise the
/// "tokenizer missing" warning path).
fn build_awq_dir(tag: &str, with_tokenizer: bool) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rustllama-auto-load-{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let vocab = 4usize;
    let d_model = 16usize;
    let d_ff = 32usize;
    let n_heads = 2usize;
    let head_dim = 8usize;

    // Per-linear quant trio bytes — kept owned at this scope so the
    // TensorView borrows stay live across serialize.
    let (qw_q, sc_q, qz_q) = awq_linear(d_model, d_model);
    let (qw_k, sc_k, qz_k) = awq_linear(d_model, d_model);
    let (qw_v, sc_v, qz_v) = awq_linear(d_model, d_model);
    let (qw_o, sc_o, qz_o) = awq_linear(d_model, d_model);
    let (qw_g, sc_g, qz_g) = awq_linear(d_model, d_ff);
    let (qw_u, sc_u, qz_u) = awq_linear(d_model, d_ff);
    let (qw_d, sc_d, qz_d) = awq_linear(d_ff, d_model);
    let embd: Vec<u8> = f16_bytes(&vec![f16::from_f32(0.1); vocab * d_model]);
    let attn_norm: Vec<u8> = f16_bytes(&vec![f16::from_f32(1.0); d_model]);
    let ffn_norm: Vec<u8> = f16_bytes(&vec![f16::from_f32(1.0); d_model]);
    let out_norm: Vec<u8> = f16_bytes(&vec![f16::from_f32(1.0); d_model]);
    let lm_head: Vec<u8> = f16_bytes(&vec![f16::from_f32(0.1); vocab * d_model]);

    let out_packs_model = d_model / 8;
    let out_packs_ff = d_ff / 8;
    let mut map: BTreeMap<String, TensorView<'_>> = BTreeMap::new();
    for (proj, qw, sc, qz) in [
        ("q_proj", &qw_q, &sc_q, &qz_q),
        ("k_proj", &qw_k, &sc_k, &qz_k),
        ("v_proj", &qw_v, &sc_v, &qz_v),
        ("o_proj", &qw_o, &sc_o, &qz_o),
    ] {
        map.insert(
            format!("model.layers.0.self_attn.{proj}.qweight"),
            TensorView::new(StDtype::I32, vec![d_model, out_packs_model], qw)
                .unwrap(),
        );
        map.insert(
            format!("model.layers.0.self_attn.{proj}.scales"),
            TensorView::new(StDtype::F16, vec![1, d_model], sc).unwrap(),
        );
        map.insert(
            format!("model.layers.0.self_attn.{proj}.qzeros"),
            TensorView::new(StDtype::I32, vec![1, out_packs_model], qz).unwrap(),
        );
    }
    for (proj, qw, sc, qz) in [
        ("gate_proj", &qw_g, &sc_g, &qz_g),
        ("up_proj", &qw_u, &sc_u, &qz_u),
    ] {
        map.insert(
            format!("model.layers.0.mlp.{proj}.qweight"),
            TensorView::new(StDtype::I32, vec![d_model, out_packs_ff], qw)
                .unwrap(),
        );
        map.insert(
            format!("model.layers.0.mlp.{proj}.scales"),
            TensorView::new(StDtype::F16, vec![1, d_ff], sc).unwrap(),
        );
        map.insert(
            format!("model.layers.0.mlp.{proj}.qzeros"),
            TensorView::new(StDtype::I32, vec![1, out_packs_ff], qz).unwrap(),
        );
    }
    map.insert(
        "model.layers.0.mlp.down_proj.qweight".into(),
        TensorView::new(StDtype::I32, vec![d_ff, out_packs_model], &qw_d)
            .unwrap(),
    );
    map.insert(
        "model.layers.0.mlp.down_proj.scales".into(),
        TensorView::new(StDtype::F16, vec![1, d_model], &sc_d).unwrap(),
    );
    map.insert(
        "model.layers.0.mlp.down_proj.qzeros".into(),
        TensorView::new(StDtype::I32, vec![1, out_packs_model], &qz_d).unwrap(),
    );
    map.insert(
        "model.embed_tokens.weight".into(),
        TensorView::new(StDtype::F16, vec![vocab, d_model], &embd).unwrap(),
    );
    map.insert(
        "model.layers.0.input_layernorm.weight".into(),
        TensorView::new(StDtype::F16, vec![d_model], &attn_norm).unwrap(),
    );
    map.insert(
        "model.layers.0.post_attention_layernorm.weight".into(),
        TensorView::new(StDtype::F16, vec![d_model], &ffn_norm).unwrap(),
    );
    map.insert(
        "model.norm.weight".into(),
        TensorView::new(StDtype::F16, vec![d_model], &out_norm).unwrap(),
    );
    map.insert(
        "lm_head.weight".into(),
        TensorView::new(StDtype::F16, vec![vocab, d_model], &lm_head).unwrap(),
    );
    let blob = safetensors::serialize(&map, &None).unwrap();
    std::fs::write(dir.join("model.safetensors"), &blob).unwrap();

    let config = format!(
        r#"{{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": {d_model},
            "intermediate_size": {d_ff},
            "num_hidden_layers": 1,
            "num_attention_heads": {n_heads},
            "num_key_value_heads": {n_heads},
            "head_dim": {head_dim},
            "vocab_size": {vocab},
            "max_position_embeddings": 64,
            "tie_word_embeddings": false,
            "bos_token_id": 0,
            "eos_token_id": 0
        }}"#
    );
    std::fs::write(dir.join("config.json"), &config).unwrap();

    if with_tokenizer {
        // Minimal HF tokenizers JSON: empty BPE with 4 fixed-id tokens.
        // Small enough to live inline and round-trip through
        // `tokenizers::Tokenizer::from_file`.
        let tok_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1, "c": 2, "d": 3},
                "merges": [],
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false
            }
        }"#;
        std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();
    }
    dir
}

#[test]
fn load_safetensors_loads_minimal_awq_directory() {
    let dir = build_awq_dir("safetensors", true);
    let st_path = dir.join("model.safetensors");
    let engine = CpuEngine::load_safetensors(&st_path, 64).expect("safetensors load");
    // Architecture sanity.
    assert_eq!(engine.llama_config().n_layers, 1);
    assert_eq!(engine.llama_config().d_model, 16);
    assert_eq!(engine.vocab_size(), 4);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_auto_routes_gguf_extension_to_gguf_path() {
    // Synthesize a tiny GGUF so the GGUF branch of load_auto actually
    // runs. Failure would still test the dispatcher in isolation, but
    // a successful load proves both arms compile + share state.
    use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};
    let p = std::env::temp_dir().join("rustllama-auto-route-gguf.gguf");
    write_synthetic_llama_gguf(&p, &SynthLlama::default());
    let engine = CpuEngine::load_auto(&p, 16).expect("gguf load via auto");
    // Synth default has 32-token vocab.
    assert_eq!(engine.vocab_size(), 32);
    let _ = std::fs::remove_file(&p);
}

#[test]
fn load_auto_routes_safetensors_extension_to_safetensors_path() {
    let dir = build_awq_dir("auto-st", true);
    let st_path = dir.join("model.safetensors");
    let engine = CpuEngine::load_auto(&st_path, 64).expect("safetensors via auto");
    assert_eq!(engine.llama_config().n_layers, 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_auto_rejects_unknown_extension() {
    let p = std::env::temp_dir().join("rustllama-auto-bogus.pt");
    std::fs::write(&p, b"not a real checkpoint").unwrap();
    let err = match CpuEngine::load_auto(&p, 16) {
        Ok(_) => panic!("must reject .pt extension"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("unrecognized model file extension"),
        "error should explain the rejection: {msg}"
    );
    let _ = std::fs::remove_file(&p);
}

#[test]
fn load_safetensors_errors_when_config_json_is_missing() {
    let dir = build_awq_dir("no-config", true);
    let _ = std::fs::remove_file(dir.join("config.json"));
    let st_path = dir.join("model.safetensors");
    let err = match CpuEngine::load_safetensors(&st_path, 64) {
        Ok(_) => panic!("must error without config.json"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("config.json"),
        "error should reference the missing file: {msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_safetensors_loads_without_tokenizer_with_warning() {
    // Missing tokenizer.json is a degraded mode, not an error —
    // the model loads and forward_one still works.
    let dir = build_awq_dir("no-tokenizer", false);
    let st_path = dir.join("model.safetensors");
    let engine =
        CpuEngine::load_safetensors(&st_path, 64).expect("load without tokenizer");
    // Architecture sanity confirms the load worked end-to-end even
    // without the tokenizer file present.
    assert_eq!(engine.llama_config().n_layers, 1);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Sanity verification of the test fixture itself: prove the dequant
/// produced 0.5-valued cells in `w_q` so the engine sees real numbers,
/// not garbage. Catches a synth-fixture regression where, e.g., the
/// trio for `q_proj` got mislabeled.
#[test]
fn loaded_safetensors_engine_holds_dequanted_weights() {
    let dir = build_awq_dir("dequant-sanity", false);
    let st_path = dir.join("model.safetensors");
    let engine = CpuEngine::load_safetensors(&st_path, 64).expect("load");
    // Internal field access via the __model_for_test hook from V-6b.
    let model = engine.__model_for_test();
    let blk = &model.weights.blocks[0];
    let bytes: Vec<u8> = match &blk.w_q.storage {
        rustllama_tensor::Storage::CpuOwned(b) => b.as_ref().to_vec(),
        _ => panic!("expected CpuOwned"),
    };
    let f16s: &[f16] = cast_slice(&bytes);
    for v in f16s {
        assert!(
            (v.to_f32() - 0.5).abs() < 1e-3,
            "every w_q cell should dequant to (1 - 0) * 0.5 = 0.5; got {}",
            v.to_f32()
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ===================================================================
// MLX (Apple mlx-lm / mlx-community) load path — Phase B.
//
// These mint a synthetic-but-REAL-HF-named MLX affine model DIRECTORY
// (config.json with a `quantization` block + model.safetensors with
// `<module>.weight`(u32)/`.scales`/`.biases` triples) via the Phase-A
// writer, then load it through the public `load_auto` dispatcher — the
// same entry `serve` uses. This exercises the real
// HF-module → GGUF-slot mapping + the affine dequant + the reused llama
// builder + a live forward pass, WITHOUT any network download.
// ===================================================================

/// Deterministic pseudo-random f32 in `[-1, 1)` — a fixed spread so the
/// synthetic MLX weights are reproducible without an `rand` dep.
fn mlx_pseudo(i: usize) -> f32 {
    let h = (i as u32).wrapping_mul(2654435761) ^ 0x9E37_79B9;
    (h % 2003) as f32 / 2003.0 * 2.0 - 1.0
}

/// Build a one-layer Llama-family MLX affine model directory with the
/// attention/MLP linears AND the token embedding / lm_head quantized
/// (mlx-community's `QuantizedEmbedding`), plus full-precision norms.
/// group_size 32, 4-bit — the common mlx-community shape. Real HF module
/// names so the engine maps `model.layers.0.self_attn.q_proj.weight` →
/// `blk.0.attn_q.weight` exactly as the AWQ path does.
fn build_mlx_dir(tag: &str) -> PathBuf {
    use rustllama_kernels_cpu::mlx_affine::quantize_mlx_affine;
    use rustllama_safetensors::{MlxFullDtype, MlxWriteTensor};

    let dir = std::env::temp_dir().join(format!("rustllama-mlx-load-{tag}"));
    let _ = std::fs::remove_dir_all(&dir);

    let vocab = 32usize;
    let d_model = 64usize;
    let d_ff = 128usize;
    let n_heads = 2usize; // head_dim 32 → d_q = d_kv = 64 = d_model
    let head_dim = 32usize;
    let group = 32usize;
    let bits = 4u32;

    // A row-major [out_features, in_features] affine-quantized linear.
    fn quant(name: &str, out_f: usize, in_f: usize, group: usize, bits: u32) -> MlxWriteTensor {
        let w: Vec<f32> = (0..out_f * in_f).map(mlx_pseudo).collect();
        let (packed, scales, biases) = quantize_mlx_affine(&w, group, bits);
        MlxWriteTensor::Quant {
            name: name.to_string(),
            packed,
            scales,
            biases,
            group_size: group,
            bits,
            shape: vec![out_f as u64, in_f as u64],
        }
    }
    // A full-precision 1-D f16 tensor (norm).
    fn full_f16(name: &str, n: usize) -> MlxWriteTensor {
        let bytes: Vec<u8> = (0..n).flat_map(|_| f16::from_f32(1.0).to_le_bytes()).collect();
        MlxWriteTensor::Full {
            name: name.to_string(),
            dtype: MlxFullDtype::F16,
            shape: vec![n as u64],
            bytes,
        }
    }

    let tensors = vec![
        quant("model.layers.0.self_attn.q_proj", d_model, d_model, group, bits),
        quant("model.layers.0.self_attn.k_proj", d_model, d_model, group, bits),
        quant("model.layers.0.self_attn.v_proj", d_model, d_model, group, bits),
        quant("model.layers.0.self_attn.o_proj", d_model, d_model, group, bits),
        quant("model.layers.0.mlp.gate_proj", d_ff, d_model, group, bits),
        quant("model.layers.0.mlp.up_proj", d_ff, d_model, group, bits),
        quant("model.layers.0.mlp.down_proj", d_model, d_ff, group, bits),
        // QuantizedEmbedding: [vocab, d_model], groups along d_model.
        quant("model.embed_tokens", vocab, d_model, group, bits),
        quant("lm_head", vocab, d_model, group, bits),
        full_f16("model.layers.0.input_layernorm.weight", d_model),
        full_f16("model.layers.0.post_attention_layernorm.weight", d_model),
        full_f16("model.norm.weight", d_model),
    ];

    let mut extra = serde_json::Map::new();
    extra.insert("architectures".into(), serde_json::json!(["LlamaForCausalLM"]));
    extra.insert("hidden_size".into(), serde_json::json!(d_model));
    extra.insert("intermediate_size".into(), serde_json::json!(d_ff));
    extra.insert("num_hidden_layers".into(), serde_json::json!(1));
    extra.insert("num_attention_heads".into(), serde_json::json!(n_heads));
    extra.insert("num_key_value_heads".into(), serde_json::json!(n_heads));
    extra.insert("head_dim".into(), serde_json::json!(head_dim));
    extra.insert("vocab_size".into(), serde_json::json!(vocab));
    extra.insert("max_position_embeddings".into(), serde_json::json!(64));
    extra.insert("tie_word_embeddings".into(), serde_json::json!(false));
    extra.insert("bos_token_id".into(), serde_json::json!(0));
    extra.insert("eos_token_id".into(), serde_json::json!(0));

    rustllama_safetensors::write_mlx_dir(&dir, &tensors, group, bits, extra, None)
        .expect("write mlx dir");
    dir
}

#[test]
fn load_auto_loads_mlx_directory_and_runs_forward() {
    let dir = build_mlx_dir("auto");
    // Route through the public auto-dispatcher exactly as `serve` does.
    let engine = CpuEngine::load_auto(&dir, 64).expect("mlx load via load_auto");
    assert_eq!(engine.llama_config().n_layers, 1);
    assert_eq!(engine.llama_config().d_model, 64);
    assert_eq!(engine.vocab_size(), 32);

    // Forward pass executes end-to-end. Synthetic weights → the output is
    // NOT coherent, but the dequant → build → forward chain must run and
    // advance exactly the requested number of tokens without panicking.
    let sampling = rustllama_engine::SamplingParams::default();
    let prompt: Vec<i32> = vec![0, 1, 2, 3];
    let out = engine
        .generate_token_ids(&prompt, 4, &sampling)
        .expect("mlx forward runs");
    assert_eq!(out.len(), 4, "should emit exactly n_new tokens");
    for &t in &out {
        assert!((t as usize) < engine.vocab_size(), "token {t} out of vocab range");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_mlx_direct_entry_loads_with_dequanted_weights() {
    // Direct `load_mlx` (bypassing the dispatcher) + a weight sanity
    // check: the dequanted q_proj must hold finite, non-zero values.
    let dir = build_mlx_dir("direct");
    let engine = CpuEngine::load_mlx(&dir, 64).expect("load_mlx");
    let model = engine.__model_for_test();
    let blk = &model.weights.blocks[0];
    // q_proj dequants to [d_model, d_model] = [64, 64], stored f16.
    assert_eq!(blk.w_q.shape, vec![64, 64]);
    let bytes: Vec<u8> = match &blk.w_q.storage {
        rustllama_tensor::Storage::CpuOwned(b) => b.as_ref().to_vec(),
        _ => panic!("expected CpuOwned"),
    };
    let f16s: &[f16] = cast_slice(&bytes);
    assert_eq!(f16s.len(), 64 * 64);
    assert!(
        f16s.iter().all(|h| h.to_f32().is_finite()),
        "every dequanted weight must be finite"
    );
    assert!(
        f16s.iter().any(|h| h.to_f32().abs() > 1e-4),
        "dequant must not zero out the whole tensor"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_auto_routes_awq_directory_to_safetensors_fallback() {
    // A directory that is AWQ (not MLX: `.scales`+`.qzeros`, no `.biases`)
    // must NOT be detected as MLX; load_auto falls back to its
    // `model.safetensors`. Pins the is_mlx_dir discriminator.
    let dir = build_awq_dir("auto-dir-awq", true);
    let engine = CpuEngine::load_auto(&dir, 64).expect("awq dir via load_auto");
    assert_eq!(engine.llama_config().n_layers, 1);
    assert_eq!(engine.llama_config().d_model, 16);
    let _ = std::fs::remove_dir_all(&dir);
}
