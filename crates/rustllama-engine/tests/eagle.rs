//! P10b — EAGLE trained draft-head anchors (engine level).
//!
//! EAGLE is model-blind here: no public EAGLE head exists for a base rustllama
//! supports, so end-to-end COHERENCE can't be validated on this box. Correctness
//! rests on the tree-spec stack's standard chain-tree ≡ linear anchor (exactly
//! how the rest of the tree-spec stack shipped):
//!
//!   * `eagle_chain_matches_plain_greedy` — a CHAIN-shaped EAGLE draft tree
//!     (widths all 1), drafted by a synthetic head and verified + accepted via
//!     the SHARED tree-spec stack, commits EXACTLY the plain greedy continuation
//!     token-for-token. This holds regardless of the head's draft QUALITY: a
//!     wrong draft is rejected and replaced by the target argmax (== plain
//!     greedy), so it proves the EAGLE round plumbing (draft → verify → accept →
//!     commit) is correct independent of the head. Mirrors the n-gram stream's
//!     `ngram_stream_from_ids_matches_plain_greedy`.
//!   * `no_head_is_byte_identical` — without an EAGLE head, and with the classic
//!     decode path even when a head IS attached, generation is byte-identical.
//!
//! The synthetic head is a real EAGLE head STRUCTURE (fusion + block + tied LM
//! head) with random weights — enough to exercise the loader + the
//! autoregressive drafter + the whole round; its outputs are nonsense, which is
//! exactly why the anchor leans on the verify/accept stack, not the draft.

use futures::StreamExt;
use rustllama_engine::{CpuEngine, KvDtype, SamplingParams};
use rustllama_gguf::synth::{
    write_synthetic_eagle_head_gguf, write_synthetic_llama_gguf, SynthEagleHead, SynthLlama,
};

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "rustllama-eagle-{name}-{}.gguf",
        std::process::id()
    ))
}

fn greedy(n: u32) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_p: 0.0,
        top_k: 0,
        repeat_penalty: 1.0,
        max_tokens: n,
        stop: vec![],
        seed: 7,
        ..SamplingParams::default()
    }
}

/// Write a synth base + a matched EAGLE head, returning both paths.
fn write_base_and_head(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let base_path = tmp(&format!("{tag}-base"));
    let head_path = tmp(&format!("{tag}-head"));
    let base = SynthLlama::default();
    write_synthetic_llama_gguf(&base_path, &base);
    write_synthetic_eagle_head_gguf(
        &head_path,
        &SynthEagleHead {
            base,
            ..SynthEagleHead::default()
        },
    );
    (base_path, head_path)
}

/// THE chain ≡ linear-EAGLE anchor: a CHAIN-shaped EAGLE draft (widths all 1),
/// drafted by the trained head and verified/accepted via the shared tree-spec
/// stack, must reproduce the plain greedy continuation token-for-token.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn eagle_chain_matches_plain_greedy() {
    let (base, head) = write_base_and_head("chain");
    let mut cpu = CpuEngine::load_with_options(&base, 64, true, KvDtype::F32).expect("load base");
    cpu.attach_eagle_head(&head).expect("attach eagle head");
    assert!(cpu.eagle_speculative(), "dense model + head ⇒ EAGLE engaged");
    // A chain tree (width 1 at every depth) is the degenerate case the parity
    // anchor pins: it must reduce EXACTLY to linear greedy decode.
    cpu.set_tree_speculative(Some(vec![1, 1, 1]));

    let prompt = [5i32, 9, 7, 5, 9, 7, 5];
    let n = 12u32;
    let head_arc = cpu.eagle_head().expect("eagle head present");
    let mut stream = cpu
        .speculate_eagle_stream_from_ids(prompt.to_vec(), greedy(n), head_arc)
        .expect("eagle stream");
    let mut eagle_ids: Vec<u32> = Vec::new();
    while let Some(item) = stream.next().await {
        eagle_ids.push(item.expect("eagle stream token").id);
    }
    assert!(!eagle_ids.is_empty(), "EAGLE stream produced no tokens");

    let plain = cpu
        .generate_token_ids(&prompt, n, &greedy(n))
        .expect("plain greedy");
    let common = eagle_ids.len().min(plain.len());
    assert_eq!(
        &eagle_ids[..common],
        &plain[..common],
        "chain-shaped EAGLE greedy must equal plain greedy token-for-token"
    );

    let _ = std::fs::remove_file(&base);
    let _ = std::fs::remove_file(&head);
}

/// A branching EAGLE tree (widths > 1) is a well-formed round that still
/// commits at least one token and matches plain greedy on its accepted prefix
/// (greedy acceptance is lossless regardless of the branching shape).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn eagle_branching_tree_is_lossless_greedy() {
    let (base, head) = write_base_and_head("branch");
    let mut cpu = CpuEngine::load_with_options(&base, 64, true, KvDtype::F32).expect("load base");
    cpu.attach_eagle_head(&head).expect("attach eagle head");
    cpu.set_tree_speculative(Some(vec![2, 2, 2])); // caterpillar

    let prompt = [4i32, 11, 4, 11, 4, 11, 4];
    let n = 10u32;
    let head_arc = cpu.eagle_head().expect("eagle head present");
    let mut stream = cpu
        .speculate_eagle_stream_from_ids(prompt.to_vec(), greedy(n), head_arc)
        .expect("eagle stream");
    let mut eagle_ids: Vec<u32> = Vec::new();
    while let Some(item) = stream.next().await {
        eagle_ids.push(item.expect("eagle stream token").id);
    }
    assert!(!eagle_ids.is_empty(), "branching EAGLE produced no tokens");
    let plain = cpu
        .generate_token_ids(&prompt, n, &greedy(n))
        .expect("plain greedy");
    let common = eagle_ids.len().min(plain.len());
    assert_eq!(
        &eagle_ids[..common],
        &plain[..common],
        "branching EAGLE greedy must still match plain greedy on the accepted prefix"
    );
    let _ = std::fs::remove_file(&base);
    let _ = std::fs::remove_file(&head);
}

/// Fail-closed: without an EAGLE head the engine never engages EAGLE, and even
/// WITH a head attached the classic (non-stream) decode path is byte-identical
/// to the un-attached engine — a head present can't perturb classic decode.
#[test]
fn no_head_is_byte_identical() {
    let (base, head) = write_base_and_head("identical");
    let prompt = [5i32, 9, 7, 5, 9, 7, 5];
    let n = 12u32;

    let plain_engine =
        CpuEngine::load_with_options(&base, 64, true, KvDtype::F32).expect("load plain");
    assert!(plain_engine.eagle_head().is_none(), "fresh engine has no head");
    assert!(!plain_engine.eagle_speculative());
    let reference = plain_engine
        .generate_token_ids(&prompt, n, &greedy(n))
        .expect("plain greedy");

    let mut eagle_engine =
        CpuEngine::load_with_options(&base, 64, true, KvDtype::F32).expect("load w/ head");
    eagle_engine.attach_eagle_head(&head).expect("attach");
    assert!(eagle_engine.eagle_speculative(), "head ⇒ EAGLE available");
    // The classic decode path (generate_token_ids) does NOT route through any
    // speculative drafter, so an attached head leaves it byte-identical.
    let with_head = eagle_engine
        .generate_token_ids(&prompt, n, &greedy(n))
        .expect("classic greedy with head attached");
    assert_eq!(
        reference, with_head,
        "classic decode must be byte-identical whether or not an EAGLE head is attached"
    );

    let _ = std::fs::remove_file(&base);
    let _ = std::fs::remove_file(&head);
}

/// A malformed companion (not tagged `adapter.type = "eagle"`) is rejected at
/// attach — the fail-closed contract (a bad head fails the load loudly rather
/// than silently degrading decode).
#[test]
fn non_eagle_companion_is_rejected() {
    let base = tmp("reject-base");
    write_synthetic_llama_gguf(&base, &SynthLlama::default());
    // A plain base GGUF is not an EAGLE head (no `adapter.type = "eagle"`).
    let mut cpu = CpuEngine::load_with_options(&base, 64, true, KvDtype::F32).expect("load");
    let err = cpu.attach_eagle_head(&base);
    assert!(err.is_err(), "a non-eagle GGUF must be rejected at attach");
    assert!(cpu.eagle_head().is_none(), "a failed attach leaves no head");
    let _ = std::fs::remove_file(&base);
}
