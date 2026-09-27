//! Engine-level integration tests for the paged KV backend.
//!
//! The model-layer parity tests
//! (`llama_arch::tests::forward_*_paged_*`) prove the
//! `forward_*_paged_f32` functions return bit-identical logits to
//! their contiguous siblings on synthetic Llama weights. These
//! tests prove the wiring all the way through `CpuEngine`:
//!   1. `load_with_options_and_layout("paged", ...)` succeeds on a
//!      real synthetic GGUF.
//!   2. `generate_token_ids` runs end-to-end through the paged
//!      backend (load → state.kv_backend → forward_one dispatch via
//!      `forward_one_via_backend` → sampler).
//!   3. The token sequence matches the contiguous run byte-for-
//!      byte under greedy sampling.
//!
//! Anything subtly wrong in the new wiring — wrong layout passed
//! through `load_inner`, `forward_one_via_backend` returning the
//! wrong dispatch branch, `state.kv_backend.set_seq_len` being a
//! no-op when it shouldn't be — would surface here as a token
//! divergence rather than just a clean compile.

use rustllama_engine::{CpuEngine, SamplingParams};
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};

fn write_gguf(tag: &str) -> std::path::PathBuf {
    let tmp = std::env::temp_dir().join(format!("rustllama-paged-{tag}.gguf"));
    let params = SynthLlama::default();
    write_synthetic_llama_gguf(&tmp, &params);
    tmp
}

fn greedy(n: u32) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_p: 0.0,
        top_k: 0,
        repeat_penalty: 1.0,
        max_tokens: n,
        stop: vec![],
        seed: 0,
        ..SamplingParams::default()
    }
}

/// End-to-end parity: same prompt through `CpuEngine` loaded with
/// `kv_cache_layout = "contiguous"` vs `"paged"` produces the same
/// greedy token sequence. This is the milestone test for 3.6e —
/// proves the whole load → state.kv_backend → forward_one_via_backend
/// chain works for a real loaded model.
#[test]
fn paged_engine_matches_contiguous_greedy_sequence() {
    use rustllama_models::llama_arch::KvDtype;
    let tmp = write_gguf("e2e-parity");

    // Disable prefix cache on both engines so the test compares
    // pure forward-pass behavior, not whether a cache lookup
    // happened to fire.
    let mut contig = CpuEngine::load_with_options_and_layout(
        &tmp, 32, true, KvDtype::F32, "contiguous",
    )
    .expect("contiguous load");
    contig.set_prefix_cache(false);

    let mut paged = CpuEngine::load_with_options_and_layout(
        &tmp, 32, true, KvDtype::F32, "paged",
    )
    .expect("paged load");
    paged.set_prefix_cache(false);

    let prompt: Vec<i32> = (0..8).collect();
    let out_contig = contig
        .generate_token_ids(&prompt, 4, &greedy(4))
        .expect("contiguous generate");
    let out_paged = paged
        .generate_token_ids(&prompt, 4, &greedy(4))
        .expect("paged generate");

    assert_eq!(
        out_contig, out_paged,
        "paged engine diverged from contiguous: paged={out_paged:?} contig={out_contig:?}"
    );

    let _ = std::fs::remove_file(&tmp);
}

/// Two back-to-back generations on the same paged engine must
/// each succeed and produce coherent output. The engine's
/// per-request reset path (`EngineState::reset` → `kv_backend.reset()`)
/// must release the prior request's pages back to the table; a
/// leak would surface here as the second request failing with
/// "page pool short" the moment the previous request's pages
/// weren't reclaimed.
#[test]
fn paged_engine_reuses_pages_across_requests() {
    use rustllama_models::llama_arch::KvDtype;
    let tmp = write_gguf("page-reuse");
    let mut paged = CpuEngine::load_with_options_and_layout(
        &tmp, 32, true, KvDtype::F32, "paged",
    )
    .expect("paged load");
    paged.set_prefix_cache(false);

    let prompt_a: Vec<i32> = (0..8).collect();
    let prompt_b: Vec<i32> = (4..12).collect();

    let out_a = paged
        .generate_token_ids(&prompt_a, 3, &greedy(3))
        .expect("gen a");
    assert_eq!(out_a.len(), 3, "first request must produce 3 tokens");

    let out_b = paged
        .generate_token_ids(&prompt_b, 3, &greedy(3))
        .expect("gen b");
    assert_eq!(
        out_b.len(),
        3,
        "second request must succeed (pages reused from first)"
    );

    let _ = std::fs::remove_file(&tmp);
}

/// Engine load rejects an unknown `kv_cache_layout` with a
/// clear error message. Catches the case where a user typos the
/// config field — they'd see a useful diagnostic instead of a
/// silent fallback to contiguous (which would mask the typo).
#[test]
fn paged_engine_rejects_unknown_layout_at_load() {
    use rustllama_models::llama_arch::KvDtype;
    let tmp = write_gguf("unknown-layout");
    let msg = match CpuEngine::load_with_options_and_layout(
        &tmp, 32, true, KvDtype::F32, "memory-mapped",
    ) {
        Ok(_) => panic!("unknown layout must fail load"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("memory-mapped"),
        "error message must name the bad value: {msg}"
    );
    assert!(
        msg.contains("contiguous") && msg.contains("paged"),
        "error message must list valid options: {msg}"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// A paged `CpuEngine` can be forked — `fork_for_concurrent_use`
/// rebuilds a matching paged backend on the fork (each fork gets
/// its own page pool, no sharing). Both parent and fork must be
/// able to run independent generations against the same shared
/// model weights without corrupting each other's KV state.
///
/// This is the integration check for the 3.6e fork wire-up. The
/// server's `MultiFlightPool` calls `fork_for_concurrent_use` N
/// times at load to build the concurrent-serving pool; if the
/// paged fork path were broken, every `[server].concurrency > 1`
/// load with `kv_cache_layout = "paged"` would panic or produce
/// nonsense. This test catches that without spinning up the full
/// server stack.
#[test]
fn paged_engine_fork_is_independent() {
    use rustllama_models::llama_arch::KvDtype;
    let tmp = write_gguf("paged-fork");
    let mut parent = CpuEngine::load_with_options_and_layout(
        &tmp, 32, true, KvDtype::F32, "paged",
    )
    .expect("paged parent load");
    parent.set_prefix_cache(false);

    let mut fork = parent.fork_for_concurrent_use();
    fork.set_prefix_cache(false);

    // Both run the same prompt → both must produce equal-length
    // token sequences. Bit-equal across parent+fork is the
    // strongest form of "fork is a real engine, not a placeholder."
    let prompt: Vec<i32> = (0..6).collect();
    let out_parent = parent
        .generate_token_ids(&prompt, 3, &greedy(3))
        .expect("parent generate");
    let out_fork = fork
        .generate_token_ids(&prompt, 3, &greedy(3))
        .expect("fork generate");
    assert_eq!(
        out_parent, out_fork,
        "paged parent vs paged fork: same prompt, different outputs (parent={out_parent:?}, fork={out_fork:?})"
    );

    // Now run a DIFFERENT prompt on the fork while the parent's
    // cache state is still around — if pages were shared between
    // parent and fork, the fork's writes would corrupt the
    // parent's state and a second parent generation would diverge.
    let prompt_b: Vec<i32> = (6..12).collect();
    let _ = fork
        .generate_token_ids(&prompt_b, 3, &greedy(3))
        .expect("fork second generate");
    let out_parent_again = parent
        .generate_token_ids(&prompt, 3, &greedy(3))
        .expect("parent second generate");
    assert_eq!(
        out_parent, out_parent_again,
        "parent's output changed after fork ran a different prompt — \
         pages must not be shared between forks (got {out_parent_again:?}, expected {out_parent:?})"
    );

    let _ = std::fs::remove_file(&tmp);
}

/// Engine load accepts `paged` + Q8_0 `kv_dtype`. H9b lifted the
/// historic "paged requires F32" constraint — all four KV dtypes
/// build a paged backend now, so users who requested Q8_0 to save
/// KV memory get exactly that.
#[test]
fn paged_engine_accepts_q8_0_kv_dtype_at_load() {
    use rustllama_models::llama_arch::KvDtype;
    let tmp = write_gguf("paged-q8");
    let engine = CpuEngine::load_with_options_and_layout(
        &tmp, 32, true, KvDtype::Q8_0, "paged",
    )
    .expect("paged + Q8_0 loads (H9b)");
    drop(engine);
    let _ = std::fs::remove_file(&tmp);
}
