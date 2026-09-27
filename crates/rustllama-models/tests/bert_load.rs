//! End-to-end load test for the BERT-family loader. v1.1
//! foundation: synth GGUF → BertModel::load → verify all tensor
//! refs resolve + config matches the synth params. Forward pass
//! is the next turn; the stub returns NotImplemented here as a
//! contract check.

use rustllama_gguf::synth::{write_synthetic_bert_gguf, SynthBert};
use rustllama_gguf::Gguf;
use rustllama_models::bert_arch::{BertForwardError, BertModel, PoolingType};

#[test]
fn synth_bert_loads_with_correct_dims_and_tensor_refs() {
    let tmp = std::env::temp_dir().join("rustllama-synth-bert-load.gguf");
    let params = SynthBert {
        n_layers: 3,
        n_heads: 4,
        d_model: 64,
        d_ff: 128,
        vocab: 32,
        ctx: 32,
        n_token_types: 2,
        ..SynthBert::default()
    };
    write_synthetic_bert_gguf(&tmp, &params);

    let gguf = Gguf::open(&tmp).expect("open synth bert gguf");
    assert_eq!(gguf.architecture(), Some("bert"));

    let model = BertModel::load(&gguf).expect("load bert");
    assert_eq!(model.cfg.n_layers, params.n_layers as usize);
    assert_eq!(model.cfg.n_heads, params.n_heads as usize);
    assert_eq!(model.cfg.d_model, params.d_model as usize);
    assert_eq!(model.cfg.d_ff, params.d_ff as usize);
    assert_eq!(model.cfg.head_dim, (params.d_model / params.n_heads) as usize);
    assert_eq!(model.cfg.vocab_size, params.vocab as usize);
    assert_eq!(model.cfg.ctx_train, params.ctx as usize);
    assert_eq!(model.cfg.n_token_types, params.n_token_types as usize);
    assert_eq!(model.cfg.pooling_type, PoolingType::Mean);

    // Per-layer tensor bundle is fully populated.
    assert_eq!(model.layers.len(), params.n_layers as usize);
    for (i, layer) in model.layers.iter().enumerate() {
        assert_eq!(layer.attn_q.name, format!("blk.{i}.attn_q.weight"));
        assert!(layer.attn_q_bias.is_some(), "BERT carries bias on attn_q");
        assert!(layer.attn_k_bias.is_some());
        assert!(layer.attn_v_bias.is_some());
        assert!(layer.attn_output_bias.is_some());
        assert!(layer.ffn_up_bias.is_some());
        assert!(layer.ffn_down_bias.is_some());
        assert_eq!(
            layer.attn_output_norm.name,
            format!("blk.{i}.attn_output_norm.weight")
        );
        assert_eq!(
            layer.layer_output_norm.name,
            format!("blk.{i}.layer_output_norm.weight")
        );
    }

    // Embedding-layer tensors all bound.
    assert!(model.token_types.is_some(), "synth has 2 token types");
    assert_eq!(model.position_embd.shape, vec![64u64, 32u64]);

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn bert_forward_embed_returns_finite_vector_of_d_model_length() {
    // End-to-end forward pass: embedding lookup → pre-LN → N
    // transformer blocks (bidirectional attn + LN + GELU FFN + LN)
    // → mean pooling → length-d_model vector. The synth model's
    // RNG-seeded weights aren't trained, so the output values are
    // arbitrary — the test pins finiteness + length + that pooling
    // actually ran (mean ≈ small magnitude, not all-zeros).
    let tmp = std::env::temp_dir().join("rustllama-synth-bert-forward.gguf");
    let params = SynthBert::default();
    write_synthetic_bert_gguf(&tmp, &params);
    let gguf = Gguf::open(&tmp).expect("open");
    let model = BertModel::load(&gguf).expect("load");
    let tokens = vec![0i32, 1, 2, 3, 4];
    let pooled = model.forward_embed(&tokens).expect("forward succeeds");

    assert_eq!(
        pooled.len(),
        params.d_model as usize,
        "pooled embedding has d_model entries"
    );
    for (i, v) in pooled.iter().enumerate() {
        assert!(
            v.is_finite(),
            "pooled[{i}] = {v} is not finite — forward pass produced NaN/Inf"
        );
    }
    // Mean pooling over 5 LN-normalized tokens should produce
    // values in roughly the [-2, 2] range. Anything wildly outside
    // means LN didn't run or the FFN exploded.
    let max_abs = pooled.iter().map(|v| v.abs()).fold(0f32, f32::max);
    assert!(
        max_abs < 10.0,
        "max |pooled value| = {max_abs} is suspiciously large — LN may not be running"
    );

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn bert_forward_embed_mean_pool_differs_from_cls_pool() {
    // Verifies the pooling branch actually consults
    // `cfg.pooling_type`. Build two identical synth GGUFs but
    // with different pooling metadata; the embeddings should
    // differ (CLS = first-token vector; mean = average over all
    // tokens, which is different even for random weights).
    let tmp_mean = std::env::temp_dir().join("rustllama-synth-bert-mean.gguf");
    let tmp_cls = std::env::temp_dir().join("rustllama-synth-bert-cls.gguf");
    write_synthetic_bert_gguf(&tmp_mean, &SynthBert::default());
    write_synthetic_bert_gguf(&tmp_cls, &SynthBert::default());

    let gguf_mean = Gguf::open(&tmp_mean).unwrap();
    let mut model_mean = BertModel::load(&gguf_mean).unwrap();
    let gguf_cls = Gguf::open(&tmp_cls).unwrap();
    let mut model_cls = BertModel::load(&gguf_cls).unwrap();
    // Tweak pooling type post-load (the synth writer hard-codes
    // "mean" in metadata; flipping the field here lets us test
    // both paths against the same weights).
    model_mean.cfg.pooling_type = PoolingType::Mean;
    model_cls.cfg.pooling_type = PoolingType::Cls;

    let tokens = vec![0i32, 1, 2, 3, 4];
    let m_out = model_mean.forward_embed(&tokens).unwrap();
    let c_out = model_cls.forward_embed(&tokens).unwrap();

    // Different pooling → different vectors (in general; identical
    // weights + tokens + everything except pool selector).
    let diff: f32 = m_out
        .iter()
        .zip(c_out.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 0.0,
        "mean-pool and cls-pool of the same model must differ; got identical outputs"
    );

    let _ = std::fs::remove_file(&tmp_mean);
    let _ = std::fs::remove_file(&tmp_cls);
}

#[test]
fn bert_forward_embed_rejects_invalid_inputs() {
    let tmp = std::env::temp_dir().join("rustllama-synth-bert-invalid.gguf");
    let params = SynthBert::default();
    write_synthetic_bert_gguf(&tmp, &params);
    let gguf = Gguf::open(&tmp).unwrap();
    let model = BertModel::load(&gguf).unwrap();

    // Empty tokens → EmptyInput.
    let r = model.forward_embed(&[]);
    assert!(matches!(r, Err(BertForwardError::EmptyInput)));

    // Sequence too long → SequenceTooLong. Synth default ctx=32.
    let too_long: Vec<i32> = (0..(params.ctx as i32 + 1)).map(|i| i % 16).collect();
    let r = model.forward_embed(&too_long);
    assert!(matches!(r, Err(BertForwardError::SequenceTooLong { .. })));

    // Negative token id → TokenOutOfVocab.
    let r = model.forward_embed(&[-1]);
    assert!(matches!(r, Err(BertForwardError::TokenOutOfVocab { .. })));

    // Token id >= vocab → TokenOutOfVocab.
    let r = model.forward_embed(&[params.vocab as i32]);
    assert!(matches!(r, Err(BertForwardError::TokenOutOfVocab { .. })));

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn bert_forward_embed_deterministic_for_same_input() {
    // Same model + same tokens → identical output across calls.
    // Pins deterministic-execution: no RNG drift, no thread-id-
    // dependent ordering, no per-call state leaks.
    let tmp = std::env::temp_dir().join("rustllama-synth-bert-determ.gguf");
    write_synthetic_bert_gguf(&tmp, &SynthBert::default());
    let gguf = Gguf::open(&tmp).unwrap();
    let model = BertModel::load(&gguf).unwrap();
    let tokens = vec![1i32, 2, 3];
    let a = model.forward_embed(&tokens).unwrap();
    let b = model.forward_embed(&tokens).unwrap();
    assert_eq!(a, b, "forward_embed must be deterministic for same input");
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn bert_load_rejects_llama_gguf_with_arch_mismatch() {
    // Loading a Llama GGUF through BertModel must error cleanly
    // with `UnsupportedArch` — guards against silent mis-parsing
    // if the embeddings config path accidentally points at a chat
    // model's GGUF.
    use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};
    use rustllama_models::bert_arch::{BertConfigError, BertLoadError};

    let tmp = std::env::temp_dir().join("rustllama-llama-as-bert-mismatch.gguf");
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    let gguf = Gguf::open(&tmp).expect("open llama");
    let err = BertModel::load(&gguf).expect_err("must reject Llama");
    match err {
        BertLoadError::Config(BertConfigError::UnsupportedArch(name)) => {
            assert_eq!(name, "llama");
        }
        other => panic!("expected UnsupportedArch, got {other:?}"),
    }
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn bert_load_binds_classifier_head_when_present() {
    let tmp = std::env::temp_dir().join("rustllama-synth-bert-classifier.gguf");
    let params = SynthBert::default().with_classifier_head(1);
    write_synthetic_bert_gguf(&tmp, &params);

    let gguf = Gguf::open(&tmp).unwrap();
    let model = BertModel::load(&gguf).unwrap();
    let head = model
        .classifier_head
        .as_ref()
        .expect("classifier head must be bound when synth includes cls.weight");
    assert_eq!(head.n_labels, 1, "single-label BGE-reranker shape");
    assert!(head.bias.is_some(), "synth emits both cls.weight and cls.bias");
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn bert_load_leaves_classifier_head_none_on_pure_embedding_gguf() {
    let tmp = std::env::temp_dir().join("rustllama-synth-bert-noclassifier.gguf");
    write_synthetic_bert_gguf(&tmp, &SynthBert::default());
    let gguf = Gguf::open(&tmp).unwrap();
    let model = BertModel::load(&gguf).unwrap();
    assert!(
        model.classifier_head.is_none(),
        "pure embedding GGUF must not have a classifier head"
    );
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn bert_forward_classify_returns_n_labels_finite_scalars() {
    let tmp = std::env::temp_dir().join("rustllama-synth-bert-fwd-classify.gguf");
    let params = SynthBert::default().with_classifier_head(1);
    write_synthetic_bert_gguf(&tmp, &params);
    let gguf = Gguf::open(&tmp).unwrap();
    let model = BertModel::load(&gguf).unwrap();
    let tokens = vec![2i32, 5, 6, 3]; // [CLS] the a [SEP]
    let out = model.forward_classify(&tokens).expect("classify");
    assert_eq!(out.len(), 1, "n_labels = 1 → single relevance scalar");
    for (i, v) in out.iter().enumerate() {
        assert!(v.is_finite(), "score[{i}] = {v} is not finite");
    }
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn bert_forward_classify_distinguishes_inputs() {
    // Different inputs → different scores. Smoke-test that the
    // classifier head + forward path actually depend on the input.
    let tmp = std::env::temp_dir().join("rustllama-synth-bert-classify-vary.gguf");
    let params = SynthBert::default().with_classifier_head(1);
    write_synthetic_bert_gguf(&tmp, &params);
    let gguf = Gguf::open(&tmp).unwrap();
    let model = BertModel::load(&gguf).unwrap();
    let a = model.forward_classify(&[2i32, 5, 3]).unwrap();
    let b = model.forward_classify(&[2i32, 6, 7, 3]).unwrap();
    assert_ne!(
        a, b,
        "forward_classify must depend on input — got identical scores {a:?} vs {b:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn bert_forward_classify_errors_when_no_classifier_head() {
    let tmp = std::env::temp_dir().join("rustllama-synth-bert-noclassifier-err.gguf");
    write_synthetic_bert_gguf(&tmp, &SynthBert::default());
    let gguf = Gguf::open(&tmp).unwrap();
    let model = BertModel::load(&gguf).unwrap();
    let err = model
        .forward_classify(&[1, 2, 3])
        .expect_err("must error on pure embedding model");
    assert!(matches!(err, BertForwardError::NoClassifierHead));
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn bert_forward_classify_supports_multi_label_head() {
    let tmp = std::env::temp_dir().join("rustllama-synth-bert-multilabel.gguf");
    let params = SynthBert::default().with_classifier_head(5);
    write_synthetic_bert_gguf(&tmp, &params);
    let gguf = Gguf::open(&tmp).unwrap();
    let model = BertModel::load(&gguf).unwrap();
    assert_eq!(model.classifier_head.as_ref().unwrap().n_labels, 5);
    let out = model.forward_classify(&[2i32, 5, 6, 3]).unwrap();
    assert_eq!(out.len(), 5, "5-label classifier → 5 scores");
    let _ = std::fs::remove_file(&tmp);
}
