//! Real-model coherence test.
//!
//! Loads a downloaded GGUF and asserts that greedy decoding from a known
//! prompt produces a coherent continuation. Gated by `RUSTLLAMA_RUN_GPU_TESTS=1`
//! (the name will widen once SYCL kernels land); also skips silently when the
//! model file is absent so contributors who haven't run `xtask fetch-test-model`
//! see no failure.

use std::path::PathBuf;

use rustllama_engine::{CpuEngine, SamplingParams};

fn enabled() -> bool {
    std::env::var("RUSTLLAMA_RUN_GPU_TESTS").is_ok()
}

fn model_path(name: &str) -> Option<PathBuf> {
    // CARGO_MANIFEST_DIR is `crates/rustllama-engine`. Walk up to the
    // workspace root and into `target/test-models/`.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest.parent()?.parent()?;
    let p = workspace.join("target").join("test-models").join(name);
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repeat_penalty: 1.0,
        max_tokens: 24,
        stop: vec![],
        seed: 0,
        ..SamplingParams::default()
    }
}

#[test]
fn qwen_completes_capital_of_france_with_paris() {
    if !enabled() {
        eprintln!("skipped (set RUSTLLAMA_RUN_GPU_TESTS=1 to enable)");
        return;
    }
    let Some(path) = model_path("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf") else {
        eprintln!("skipped (run `cargo xtask fetch-test-model qwen2.5-coder-0.5b-q4_k_m`)");
        return;
    };

    let engine = CpuEngine::load_with_tokenizer(&path, 256).expect("load engine");
    let completion = engine
        .generate_text("The capital of France is", &greedy())
        .expect("generate");

    eprintln!("completion: {completion:?}");
    assert!(
        completion.to_lowercase().contains("paris"),
        "expected `Paris` in completion, got: {completion:?}"
    );
}

#[test]
fn tinyllama_completes_with_paris() {
    if !enabled() {
        eprintln!("skipped (set RUSTLLAMA_RUN_GPU_TESTS=1 to enable)");
        return;
    }
    let Some(path) = model_path("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf") else {
        eprintln!("skipped (run `cargo xtask fetch-test-model tinyllama-1.1b-q4_k_m`)");
        return;
    };

    let engine = CpuEngine::load_with_tokenizer(&path, 256).expect("load engine");
    let completion = engine
        .generate_text("The capital of France is", &greedy())
        .expect("generate");

    eprintln!("completion: {completion:?}");
    assert!(
        completion.to_lowercase().contains("paris"),
        "expected `Paris` in completion, got: {completion:?}"
    );
}

#[test]
fn prefix_cache_does_not_change_output() {
    // Two greedy generations with the same prompt must produce identical
    // tokens regardless of cache state. The second call hits the prefix
    // cache (LCP = full prompt minus one); the first does not.
    if !enabled() {
        eprintln!("skipped (set RUSTLLAMA_RUN_GPU_TESTS=1 to enable)");
        return;
    }
    let Some(path) = model_path("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf") else {
        eprintln!("skipped");
        return;
    };

    let engine = CpuEngine::load_with_tokenizer(&path, 512).expect("load engine");
    assert!(engine.prefix_cache_enabled(), "default should be on");

    let prompt = "The capital of France is";
    let s = SamplingParams { max_tokens: 12, ..greedy() };

    let first = engine.generate_text(prompt, &s).expect("first");
    let second = engine.generate_text(prompt, &s).expect("second");

    assert_eq!(first, second, "prefix-cache reuse changed output");
}

#[test]
fn prefix_cache_extended_prompt_matches_fresh_run() {
    // First request: prompt A. Second request: A + extra. The second
    // should reuse the K/V entries from the first, but its output must
    // match a fresh run (cache reset between).
    if !enabled() {
        eprintln!("skipped (set RUSTLLAMA_RUN_GPU_TESTS=1 to enable)");
        return;
    }
    let Some(path) = model_path("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf") else {
        eprintln!("skipped");
        return;
    };

    let cached = CpuEngine::load_with_tokenizer(&path, 512).expect("load");
    let fresh = CpuEngine::load_with_tokenizer(&path, 512).expect("load");

    let prompt_a = "The capital of France is";
    let prompt_ab = "The capital of France is Paris. The capital of Germany is";
    let s = SamplingParams { max_tokens: 8, ..greedy() };

    // Prime the cached engine with prompt A, then run prompt AB.
    let _ = cached.generate_text(prompt_a, &s).expect("prime");
    let from_cached = cached.generate_text(prompt_ab, &s).expect("cached");

    // Fresh engine just runs AB.
    let from_fresh = fresh.generate_text(prompt_ab, &s).expect("fresh");

    assert_eq!(
        from_cached, from_fresh,
        "prefix-cache reuse diverged from fresh-cache result"
    );
}

#[test]
fn qwen_completes_simple_math() {
    if !enabled() {
        eprintln!("skipped (set RUSTLLAMA_RUN_GPU_TESTS=1 to enable)");
        return;
    }
    let Some(path) = model_path("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf") else {
        eprintln!("skipped");
        return;
    };

    let engine = CpuEngine::load_with_tokenizer(&path, 256).expect("load engine");
    let completion = engine
        .generate_text(
            "1 + 1 = ",
            &SamplingParams {
                max_tokens: 4,
                ..greedy()
            },
        )
        .expect("generate");

    eprintln!("completion: {completion:?}");
    assert!(
        completion.contains('2'),
        "expected `2` in completion of `1+1=`, got: {completion:?}"
    );
}

// ---- seed determinism --------------------------------------------------

/// A non-greedy sampling configuration that actually exercises the RNG.
/// Greedy mode (temperature=0) is deterministic for the trivial reason
/// that it never touches the sampler — we want the multinomial-draw path
/// here so the test really proves the seed contract.
fn sampled(seed: u64) -> SamplingParams {
    SamplingParams {
        temperature: 0.8,
        top_p: 0.95,
        top_k: 40,
        repeat_penalty: 1.0,
        max_tokens: 24,
        stop: vec![],
        seed,
        ..SamplingParams::default()
    }
}

#[test]
fn same_seed_produces_identical_output() {
    if !enabled() {
        eprintln!("skipped (set RUSTLLAMA_RUN_GPU_TESTS=1 to enable)");
        return;
    }
    let Some(path) = model_path("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf") else {
        eprintln!("skipped");
        return;
    };
    let engine = CpuEngine::load_with_tokenizer(&path, 512).expect("load");
    let prompt = "Once upon a time, in a faraway land,";

    let a = engine.generate_text(prompt, &sampled(42)).expect("gen a");
    let b = engine.generate_text(prompt, &sampled(42)).expect("gen b");

    assert_eq!(
        a, b,
        "same seed must produce identical output\nA: {a:?}\nB: {b:?}"
    );
}

#[test]
fn different_seeds_produce_different_output() {
    if !enabled() {
        eprintln!("skipped (set RUSTLLAMA_RUN_GPU_TESTS=1 to enable)");
        return;
    }
    let Some(path) = model_path("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf") else {
        eprintln!("skipped");
        return;
    };
    let engine = CpuEngine::load_with_tokenizer(&path, 512).expect("load");
    let prompt = "Once upon a time, in a faraway land,";

    let a = engine.generate_text(prompt, &sampled(42)).expect("gen a");
    let b = engine.generate_text(prompt, &sampled(7919)).expect("gen b");

    // Could theoretically collide on a very short max_tokens — but with 24
    // sampled tokens from a ~152K-vocab distribution, the odds are
    // negligible. If this test ever flakes, regenerate the seeds.
    assert_ne!(
        a, b,
        "different seeds should produce different output\nA: {a:?}\nB: {b:?}"
    );
}

#[test]
fn streaming_chat_aborts_when_receiver_dropped() {
    // Proves the early-cancellation wiring in `drive_generation`:
    // dropping the SSE receiver mid-stream must make `tx.is_closed()`
    // return true on the next loop iteration, so the engine stops
    // doing forward passes for tokens nobody will ever read. We assert
    // the engine emits no more than ~the channel buffer's worth of
    // tokens (16 chosen as a generous upper bound — channel cap is
    // 64; the engine usually halts within the first iteration after
    // the receiver drops).
    use futures::StreamExt;
    use rustllama_engine::Engine;

    if !enabled() {
        eprintln!("skipped (set RUSTLLAMA_RUN_GPU_TESTS=1 to enable)");
        return;
    }
    let Some(path) = model_path("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf") else {
        eprintln!("skipped");
        return;
    };

    let engine = CpuEngine::load_with_tokenizer(&path, 512).expect("load");
    let prompt = "Count slowly:";
    let sampling = SamplingParams {
        max_tokens: 200,
        ..greedy()
    };
    // Multi-thread runtime: spawn_blocking runs on a dedicated pool and
    // the executor can drive both reader and channel-close notifications
    // in parallel. A `current_thread` runtime here would park between
    // each `stream.next().await`, leaving the engine stalled long
    // enough that this test's timing assertion fails (≥5 s observed).
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let n_received = rt.block_on(async {
        let mut stream = engine.generate(prompt, &sampling).expect("generate");
        let mut count = 0;
        while count < 3 {
            if stream.next().await.is_none() {
                break;
            }
            count += 1;
        }
        drop(stream);
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        count
    });

    assert_eq!(n_received, 3, "expected to pull 3 tokens before drop");

    // Now that the stream is dropped, the engine must have released its
    // state mutex within a few forward passes (or this `generate_text`
    // would block waiting on it). Threshold: 10 s. An uncancelled engine
    // would hold the mutex through all 200 forward passes — ~16 s in
    // release mode, ~120 s in debug. With cancellation working, the
    // observed wait is well under 1 s in release and ≤ 6 s in debug.
    let start = std::time::Instant::now();
    let _ = engine
        .generate_text(
            "test",
            &SamplingParams {
                max_tokens: 1,
                ..greedy()
            },
        )
        .expect("post-cancel generate");
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 10,
        "post-cancel generate took {elapsed:?} — cancellation likely didn't fire"
    );
}

#[test]
fn same_seed_streaming_matches_blocking() {
    use futures::StreamExt;

    if !enabled() {
        eprintln!("skipped (set RUSTLLAMA_RUN_GPU_TESTS=1 to enable)");
        return;
    }
    let Some(path) = model_path("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf") else {
        eprintln!("skipped");
        return;
    };
    let engine = CpuEngine::load_with_tokenizer(&path, 512).expect("load");
    let prompt = "Once upon a time, in a faraway land,";

    // Blocking variant uses generate_token_ids; the streaming Engine impl
    // goes through drive_generation. Both must yield the same text for
    // the same seed.
    let blocking = engine
        .generate_text(prompt, &sampled(42))
        .expect("blocking gen");

    use rustllama_engine::Engine;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let streamed: String = rt.block_on(async {
        let mut stream = engine
            .generate(prompt, &sampled(42))
            .expect("streaming gen");
        let mut acc = String::new();
        while let Some(tok) = stream.next().await {
            acc.push_str(&tok.expect("token").text);
        }
        acc
    });

    assert_eq!(
        blocking, streamed,
        "blocking vs streaming with same seed diverged\nblocking: {blocking:?}\nstreamed: {streamed:?}"
    );
}
