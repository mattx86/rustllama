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
    assert!(
        cpu.eagle_speculative(),
        "dense model + head ⇒ EAGLE engaged"
    );
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
    assert!(
        plain_engine.eagle_head().is_none(),
        "fresh engine has no head"
    );
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

/// P10b-tail — the `#[ignore]`d REAL-head coherence hook. Runs ONLY when
/// `RUSTLLAMA_REAL_EAGLE_BASE` (a real base model GGUF) and
/// `RUSTLLAMA_REAL_EAGLE_HEAD` (a matching trained EAGLE draft-head GGUF) are
/// BOTH set. No such head exists for a supported base on-box, so the test is
/// `#[ignore]` (never runs in normal CI) and early-returns cleanly when the env
/// vars are unset — model-blind until a real head exists.
///
/// EAGLE's tree-verify is lossless under greedy acceptance: a wrong draft is
/// rejected and replaced by the target argmax, so the committed tokens MUST
/// equal the base model's plain greedy continuation token-for-token. This hook
/// loads the real base + head, runs a CHAIN-shaped EAGLE stream (widths all 1,
/// the degenerate linear case), and asserts the accepted prefix equals a
/// non-speculative greedy run on the same base — real-head COHERENCE, on HW.
///
/// Run on a box that has a real head with, e.g.:
///   RUSTLLAMA_REAL_EAGLE_BASE=/models/base.gguf \
///   RUSTLLAMA_REAL_EAGLE_HEAD=/models/eagle-head.gguf \
///   cargo test -p rustllama-engine --test eagle -- --ignored --exact \
///     eagle_real_head_matches_base_greedy
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires RUSTLLAMA_REAL_EAGLE_BASE + RUSTLLAMA_REAL_EAGLE_HEAD pointing at a real base + trained EAGLE head (none on-box)"]
async fn eagle_real_head_matches_base_greedy() {
    let (base_path, head_path) = match (
        std::env::var_os("RUSTLLAMA_REAL_EAGLE_BASE"),
        std::env::var_os("RUSTLLAMA_REAL_EAGLE_HEAD"),
    ) {
        (Some(b), Some(h)) => (std::path::PathBuf::from(b), std::path::PathBuf::from(h)),
        _ => {
            eprintln!(
                "skipping eagle_real_head_matches_base_greedy: set RUSTLLAMA_REAL_EAGLE_BASE and \
                 RUSTLLAMA_REAL_EAGLE_HEAD to run the real-head coherence hook"
            );
            return;
        }
    };

    let mut cpu = CpuEngine::load_with_options(&base_path, 2048, true, KvDtype::F32)
        .expect("load real base model");
    cpu.attach_eagle_head(&head_path)
        .expect("attach real EAGLE head (must match the base config)");
    assert!(
        cpu.eagle_speculative(),
        "real base + real head ⇒ EAGLE engaged"
    );
    // Chain tree (width 1 at every depth) is the degenerate linear case: EAGLE
    // must reduce EXACTLY to greedy decode on the base.
    cpu.set_tree_speculative(Some(vec![1, 1, 1, 1]));

    // Tokenize a short prompt with the base's own tokenizer.
    let prompt_ids: Vec<i32> = {
        let tok = cpu.tokenizer().expect("base model carries a tokenizer");
        tok.encode("The quick brown fox", tok.add_bos_token())
            .expect("encode prompt")
            .into_iter()
            .map(|v| v as i32)
            .collect()
    };
    assert!(!prompt_ids.is_empty(), "prompt tokenized to zero tokens");

    let n = 32u32;
    let head_arc = cpu.eagle_head().expect("eagle head present");
    let mut stream = cpu
        .speculate_eagle_stream_from_ids(prompt_ids.clone(), greedy(n), head_arc)
        .expect("eagle stream");
    let mut eagle_ids: Vec<u32> = Vec::new();
    while let Some(item) = stream.next().await {
        eagle_ids.push(item.expect("eagle stream token").id);
    }
    assert!(!eagle_ids.is_empty(), "real-head EAGLE produced no tokens");

    let plain = cpu
        .generate_token_ids(&prompt_ids, n, &greedy(n))
        .expect("plain greedy on the base");
    let common = eagle_ids.len().min(plain.len());
    assert!(common > 0, "no accepted/greedy overlap to compare");
    assert_eq!(
        &eagle_ids[..common],
        &plain[..common],
        "real EAGLE head: the accepted tokens must equal the base's plain greedy \
         continuation token-for-token (the tree-verify is lossless under greedy)"
    );
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
