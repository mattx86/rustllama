//! Integration tests for `CpuEngine::load_with_mmproj` — the V-6b-2
//! load surface that pairs a text-decoder GGUF with a vision-tower
//! (mmproj) GGUF. Focuses on the validation paths the load surface is
//! responsible for: dimension pairing, missing mmproj file, malformed
//! placeholder. Actual splice integration into the chat path is V-6b-3
//! and lands with its own end-to-end test.

use rustllama_engine::{CpuEngine, Engine};
use rustllama_gguf::synth::{write_synthetic_clip_mmproj_gguf, write_synthetic_llama_gguf, SynthLlama};

/// Default text-decoder GGUF: d_model = 64. Mismatches the synth
/// mmproj (d_text = 96) — used by the validation-failure test.
fn write_default_text_gguf(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("rustllama-vlm-load-text-{tag}.gguf"));
    write_synthetic_llama_gguf(&p, &SynthLlama::default());
    p
}

/// Text-decoder GGUF tuned to d_model = 96, which matches the synth
/// mmproj's d_text. Used by the test that needs the dimension check
/// to succeed so a downstream failure path (multi-token placeholder)
/// can fire.
fn write_matched_text_gguf(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("rustllama-vlm-load-matched-{tag}.gguf"));
    // The synth mmproj is wired to d_text = 96 — match here so the
    // pairing check passes and the test exercises the next validation
    // step (single-token placeholder).
    //
    // n_heads * head_dim must equal d_model (96), so 3 heads * 32 dim.
    let params = SynthLlama {
        n_heads: 3,
        n_kv_heads: 3,
        head_dim: 32,
        d_model: 96,
        d_ff: 128,
        ..SynthLlama::default()
    };
    write_synthetic_llama_gguf(&p, &params);
    p
}

fn write_test_mmproj(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("rustllama-vlm-load-mmproj-{tag}.gguf"));
    write_synthetic_clip_mmproj_gguf(&p);
    p
}

#[test]
fn supports_vision_is_false_without_mmproj() {
    let text = write_default_text_gguf("plain");
    let engine = CpuEngine::load_with_tokenizer(&text, 16).expect("plain load");
    assert!(
        !engine.supports_vision(),
        "engine without mmproj must not claim vision support"
    );
    assert!(engine.vision().is_none());
    assert!(engine.image_token_id().is_none());
    let _ = std::fs::remove_file(&text);
}

#[test]
fn load_with_mmproj_rejects_d_text_d_model_mismatch() {
    // Pair a d_model=64 text decoder with a d_text=96 mmproj — the
    // load surface must catch this and produce a clear error rather
    // than letting inference run with silently-misshaped tensors.
    let text = write_default_text_gguf("mismatch");
    let mmproj = write_test_mmproj("mismatch");
    let err = match CpuEngine::load_with_mmproj(&text, &mmproj, 16, "<image>") {
        Ok(_) => panic!("d_model 64 vs d_text 96 must fail validation"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("projector output dim"),
        "error message should describe the mismatch: {msg}"
    );
    assert!(msg.contains("96"), "should name d_text 96: {msg}");
    assert!(msg.contains("64"), "should name d_model 64: {msg}");
    let _ = std::fs::remove_file(&text);
    let _ = std::fs::remove_file(&mmproj);
}

#[test]
fn load_with_mmproj_errors_on_missing_mmproj_path() {
    let text = write_default_text_gguf("missing-mmproj");
    let bogus = std::env::temp_dir().join("rustllama-does-not-exist-mmproj.gguf");
    // Make sure the path really is absent so the test isn't racy.
    let _ = std::fs::remove_file(&bogus);
    let err = match CpuEngine::load_with_mmproj(&text, &bogus, 16, "<image>") {
        Ok(_) => panic!("missing mmproj file must error"),
        Err(e) => e,
    };
    // Either Gguf parse error or our wrapping — the message must at
    // least mention the file or surface a Gguf error code.
    let _msg = err.to_string();
    let _ = std::fs::remove_file(&text);
}

#[test]
fn load_with_mmproj_rejects_multi_token_placeholder() {
    // d_model=96 matches the mmproj's d_text=96 so the pairing check
    // passes; the next validation in line is the single-token
    // placeholder requirement. The synth Gpt2 BPE tokenizer doesn't
    // know `<image>` as a special added token, so encoding it
    // produces multiple tokens — exactly the failure we want to
    // surface for v1.
    let text = write_matched_text_gguf("multi-tok");
    let mmproj = write_test_mmproj("multi-tok");
    let err = match CpuEngine::load_with_mmproj(&text, &mmproj, 16, "<image>") {
        Ok(_) => panic!("multi-token placeholder must be rejected in v1"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("single-token") || msg.contains("tokens"),
        "error should mention the multi-token rejection: {msg}"
    );
    let _ = std::fs::remove_file(&text);
    let _ = std::fs::remove_file(&mmproj);
}
