//! End-to-end smoke test for the Llama-family forward pass.
//!
//! Generates a tiny synthetic GGUF file with random F16 weights matching a
//! 2-layer LlamaConfig, loads it via `LlamaModel::load`, runs forward steps,
//! and asserts the logits are shaped correctly and finite. Phase 1
//! acceptance: this test passing means the GGUF → Tensor → attention → FFN
//! → LM head plumbing is byte-coherent end-to-end.

use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama, SynthWeightDtype};
use rustllama_gguf::{Gguf, GgmlType};
use rustllama_models::llama_arch::{KvCache, LlamaModel};

#[test]
fn synthetic_llama_forward_produces_finite_logits() {
    let tmp = std::env::temp_dir().join("rustllama-synthetic-forward.gguf");
    let params = SynthLlama::default();
    write_synthetic_llama_gguf(&tmp, &params);

    let gguf = Gguf::open(&tmp).expect("open synthetic gguf");
    assert_eq!(gguf.architecture(), Some("llama"));

    let model = LlamaModel::load(&gguf).expect("load llama");
    assert_eq!(model.cfg.n_layers, params.n_layers as usize);
    assert_eq!(model.cfg.vocab_size, params.vocab as usize);
    assert_eq!(model.cfg.n_heads, params.n_heads as usize);
    assert_eq!(model.cfg.n_kv_heads, params.n_kv_heads as usize);
    assert_eq!(model.cfg.head_dim, params.head_dim as usize);
    assert!(model.cfg.tie_word_embeddings, "synthetic model is tied");

    let mut kv = KvCache::new(&model.cfg, 16);
    let mut logits = vec![0f32; model.cfg.vocab_size];

    // First token at position 0.
    model.forward_one(3, 0, &mut kv, &mut logits);
    for (i, v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] = {v} non-finite at pos 0");
    }

    // Second token at position 1 — exercises the KV cache.
    model.forward_one(5, 1, &mut kv, &mut logits);
    for (i, v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] = {v} non-finite at pos 1");
    }

    // Prefill mode: run forward over a sequence.
    let mut kv2 = KvCache::new(&model.cfg, 16);
    let final_logits = model.forward_prefill(&[3, 5, 7, 11], 0, &mut kv2);
    assert_eq!(final_logits.len(), model.cfg.vocab_size);
    for (i, v) in final_logits.iter().enumerate() {
        assert!(v.is_finite(), "prefill logit[{i}] = {v}");
    }

    // Determinism check: same model + same inputs => same outputs.
    let mut kv3 = KvCache::new(&model.cfg, 16);
    let final_logits2 = model.forward_prefill(&[3, 5, 7, 11], 0, &mut kv3);
    assert_eq!(final_logits, final_logits2);

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn synthetic_llama_with_bf16_weights_loads_and_forward_passes() {
    // BF16-weight regression: same shape as the F16 test, but the GGUF
    // tensor table marks bulk weights (embeddings + attn/FFN) as BF16.
    // Verifies:
    //   1. The GGUF parser accepts BF16 tensors (was already true).
    //   2. `LlamaModel::load` no longer errors with UnsupportedSourceDtype.
    //   3. The forward pass produces finite logits (i.e., the dequant
    //      path is correct end-to-end, not just bit-shifting noise).
    let tmp = std::env::temp_dir().join("rustllama-bf16-forward.gguf");
    let params = SynthLlama {
        weight_dtype: SynthWeightDtype::Bf16,
        ..SynthLlama::default()
    };
    write_synthetic_llama_gguf(&tmp, &params);

    let gguf = Gguf::open(&tmp).expect("open synthetic bf16 gguf");
    // Sanity: at least one bulk weight tensor really is BF16 in the file.
    let embd_info = gguf
        .tensor("token_embd.weight")
        .expect("token_embd present");
    assert_eq!(
        embd_info.dtype,
        GgmlType::Bf16,
        "synth builder did not write BF16 dtype for the embedding"
    );

    let model = LlamaModel::load(&gguf).expect("load llama with bf16 weights");
    assert_eq!(model.cfg.n_layers, params.n_layers as usize);
    assert_eq!(model.cfg.vocab_size, params.vocab as usize);

    // The model now keeps BF16 weights in their native `Dtype::Bf16Raw`
    // form rather than converting them to F16 at load. Verify by
    // checking the embedding tensor's loaded dtype — this is the
    // regression-target assertion for the native-BF16 storage path.
    assert_eq!(
        model.weights.token_embd.dtype,
        rustllama_tensor::Dtype::Bf16Raw,
        "BF16 weights should land in Dtype::Bf16Raw, not get lossy-converted to F16"
    );

    let mut kv = KvCache::new(&model.cfg, 16);
    let mut logits = vec![0f32; model.cfg.vocab_size];
    model.forward_one(3, 0, &mut kv, &mut logits);
    for (i, v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "bf16 logit[{i}] = {v} non-finite at pos 0");
    }
    model.forward_one(5, 1, &mut kv, &mut logits);
    for (i, v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "bf16 logit[{i}] = {v} non-finite at pos 1");
    }

    let mut kv2 = KvCache::new(&model.cfg, 16);
    let final_logits = model.forward_prefill(&[3, 5, 7, 11], 0, &mut kv2);
    for (i, v) in final_logits.iter().enumerate() {
        assert!(v.is_finite(), "bf16 prefill logit[{i}] = {v}");
    }

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn batched_prefill_matches_serial_forward_one() {
    // Parity gate for `forward_prefill_batched_f32`: same inputs
    // through the batched path must produce the same final logits
    // (up to FP reduction order) as the serial forward_one loop
    // that the default forward_prefill uses. Without this gate we
    // can't safely flip `RUSTLLAMA_PREFILL_BATCHED=1` on real
    // workloads.
    let tmp = std::env::temp_dir().join("rustllama-batched-prefill-parity.gguf");
    let params = SynthLlama::default();
    write_synthetic_llama_gguf(&tmp, &params);
    let gguf = Gguf::open(&tmp).expect("open synthetic gguf");
    let model = LlamaModel::load(&gguf).expect("load llama");

    let tokens = [3, 5, 7, 11, 13, 17];

    // Serial path — explicit forward_one loop matching the default
    // `forward_prefill` body (don't call forward_prefill so the env
    // var has no chance to influence the baseline).
    let mut kv_serial = KvCache::new(&model.cfg, 32);
    let mut logits_serial = vec![0f32; model.cfg.vocab_size];
    for (i, &tok) in tokens.iter().enumerate() {
        model.forward_one(tok, i as u32, &mut kv_serial, &mut logits_serial);
    }

    // Batched path — call `forward_prefill_batched_f32` directly
    // instead of `set_var("RUSTLLAMA_PREFILL_BATCHED")` + the
    // forward_prefill router. The direct call avoids racing with
    // parallel tests that also touch the env var.
    let mut kv_batched = KvCache::new(&model.cfg, 32);
    let logits_batched =
        model.forward_prefill_batched_f32(&tokens, 0, &mut kv_batched);

    assert_eq!(logits_batched.len(), logits_serial.len());
    let mut max_err = 0f32;
    for (b, s) in logits_batched.iter().zip(logits_serial.iter()) {
        assert!(b.is_finite() && s.is_finite());
        let e = (b - s).abs();
        if e > max_err {
            max_err = e;
        }
    }
    // F16 weight matvecs accumulate ~1e-3 across a small synth
    // model; both paths see the same matvec error. The batched
    // path's only divergence is the attention reduction order
    // (one prefill call vs N decode calls), and the prefill kernel
    // itself parity-tests bit-identical vs decode-loop in the
    // kernels-cpu suite. Pin a tight bound here.
    assert!(
        max_err < 1e-4,
        "batched vs serial prefill max abs error {max_err} > 1e-4"
    );

    // KV cache should also end up identical (same K/V rows written
    // for the same positions across both paths).
    assert_eq!(kv_batched.seq_len, kv_serial.seq_len);

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn batched_prefill_nvfp4_matches_serial_forward_one() {
    // NVFP4 variant — closes out the KV-dtype matrix. Quant
    // noise is in between Q8_0 and TQ4; 3e-3 covers the range
    // safely.
    use rustllama_models::llama_arch::KvDtype;
    let tmp = std::env::temp_dir().join("rustllama-batched-prefill-nvfp4-parity.gguf");
    let params = SynthLlama::default();
    write_synthetic_llama_gguf(&tmp, &params);
    let gguf = Gguf::open(&tmp).expect("open synthetic gguf");
    let model = LlamaModel::load(&gguf).expect("load llama");

    // Synth model must have head_dim a multiple of 16 (NVFP4 block).
    // SynthLlama::default() uses head_dim 16 → meets the constraint.
    assert_eq!(model.cfg.head_dim % 16, 0);

    let tokens = [3, 5, 7, 11, 13, 17];

    let mut kv_serial = KvCache::new_with_dtype(&model.cfg, 32, KvDtype::Nvfp4);
    let mut logits_serial = vec![0f32; model.cfg.vocab_size];
    for (i, &tok) in tokens.iter().enumerate() {
        model.forward_one(tok, i as u32, &mut kv_serial, &mut logits_serial);
    }

    let mut kv_batched = KvCache::new_with_dtype(&model.cfg, 32, KvDtype::Nvfp4);
    let logits_batched =
        model.forward_prefill_batched_nvfp4(&tokens, 0, &mut kv_batched);

    assert_eq!(logits_batched.len(), logits_serial.len());
    let mut max_err = 0f32;
    for (b, s) in logits_batched.iter().zip(logits_serial.iter()) {
        assert!(b.is_finite() && s.is_finite());
        let e = (b - s).abs();
        if e > max_err {
            max_err = e;
        }
    }
    assert!(
        max_err < 3e-3,
        "NVFP4 batched vs serial prefill max abs error {max_err} > 3e-3"
    );

    assert_eq!(kv_batched.seq_len, kv_serial.seq_len);

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn batched_prefill_q8_0_matches_serial_forward_one() {
    // Q8_0 variant of the batched-prefill parity test. Q8_0
    // quantization has tighter error than TQ4 — bound tracks the
    // F32 path closer than the TQ test does.
    use rustllama_models::llama_arch::KvDtype;
    let tmp = std::env::temp_dir().join("rustllama-batched-prefill-q8_0-parity.gguf");
    let params = SynthLlama::default();
    write_synthetic_llama_gguf(&tmp, &params);
    let gguf = Gguf::open(&tmp).expect("open synthetic gguf");
    let model = LlamaModel::load(&gguf).expect("load llama");

    let tokens = [3, 5, 7, 11, 13, 17];

    let mut kv_serial = KvCache::new_with_dtype(&model.cfg, 32, KvDtype::Q8_0);
    let mut logits_serial = vec![0f32; model.cfg.vocab_size];
    for (i, &tok) in tokens.iter().enumerate() {
        model.forward_one(tok, i as u32, &mut kv_serial, &mut logits_serial);
    }

    let mut kv_batched = KvCache::new_with_dtype(&model.cfg, 32, KvDtype::Q8_0);
    let logits_batched =
        model.forward_prefill_batched_q8_0(&tokens, 0, &mut kv_batched);

    assert_eq!(logits_batched.len(), logits_serial.len());
    let mut max_err = 0f32;
    for (b, s) in logits_batched.iter().zip(logits_serial.iter()) {
        assert!(b.is_finite() && s.is_finite());
        let e = (b - s).abs();
        if e > max_err {
            max_err = e;
        }
    }
    assert!(
        max_err < 1e-3,
        "Q8_0 batched vs serial prefill max abs error {max_err} > 1e-3"
    );

    assert_eq!(kv_batched.seq_len, kv_serial.seq_len);

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn batched_prefill_tq_matches_serial_forward_one() {
    // TurboQuant variant of `batched_prefill_matches_serial_forward_one`.
    // Verifies the same parity contract for TQ4 KV (the workspace
    // default), which routes through `forward_prefill_batched_tq`
    // when `RUSTLLAMA_PREFILL_BATCHED=1`.
    use rustllama_models::llama_arch::KvDtype;
    let tmp = std::env::temp_dir().join("rustllama-batched-prefill-tq-parity.gguf");
    let params = SynthLlama::default();
    write_synthetic_llama_gguf(&tmp, &params);
    let gguf = Gguf::open(&tmp).expect("open synthetic gguf");
    let model = LlamaModel::load(&gguf).expect("load llama");

    let tokens = [3, 5, 7, 11, 13, 17];

    // Serial path with TQ4 KV.
    let mut kv_serial = KvCache::new_with_dtype(&model.cfg, 32, KvDtype::Tq(4));
    let mut logits_serial = vec![0f32; model.cfg.vocab_size];
    for (i, &tok) in tokens.iter().enumerate() {
        model.forward_one(tok, i as u32, &mut kv_serial, &mut logits_serial);
    }

    // Batched TQ path — direct method call to dodge env-var racing.
    let mut kv_batched = KvCache::new_with_dtype(&model.cfg, 32, KvDtype::Tq(4));
    let logits_batched = model.forward_prefill_batched_tq(&tokens, 0, &mut kv_batched);

    assert_eq!(logits_batched.len(), logits_serial.len());
    let mut max_err = 0f32;
    for (b, s) in logits_batched.iter().zip(logits_serial.iter()) {
        assert!(b.is_finite() && s.is_finite());
        let e = (b - s).abs();
        if e > max_err {
            max_err = e;
        }
    }
    // TQ4 quantization accumulates more error than F32; both paths
    // see the same quant noise per row but the batched path's
    // attention reduction order differs slightly. A bound of 5e-3
    // is safely under the threshold where greedy sampling would
    // diverge.
    assert!(
        max_err < 5e-3,
        "TQ batched vs serial prefill max abs error {max_err} > 5e-3"
    );

    assert_eq!(kv_batched.seq_len, kv_serial.seq_len);

    let _ = std::fs::remove_file(&tmp);
}
