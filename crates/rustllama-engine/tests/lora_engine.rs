//! P11 Phase 2/3 — engine-side resident multi-LoRA (CpuEngine).
//!
//! Anchors:
//!   - attaching an adapter but NOT selecting it is byte-identical to an engine
//!     with no adapter attached (the invariant);
//!   - selecting a resident adapter per request changes the output;
//!   - an unknown adapter name fails the request (never a silent base serve).

use rustllama_engine::{CpuEngine, SamplingParams};
use rustllama_gguf::synth::{
    write_synthetic_llama_gguf, write_synthetic_lora_adapter_gguf, SynthLlama, SynthLoraAdapter,
};

fn tmp(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "rustllama-lora-engine-{tag}-{}.gguf",
        std::process::id()
    ))
}

/// Greedy (deterministic) sampling, optionally naming a resident adapter.
fn greedy(n: u32, adapter: Option<&str>) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_p: 0.0,
        top_k: 0,
        repeat_penalty: 1.0,
        max_tokens: n,
        stop: vec![],
        seed: 0,
        lora_adapter: adapter.map(|s| s.to_string()),
        ..SamplingParams::default()
    }
}

fn write_adapter(path: &std::path::Path, seed: u64) {
    // A deliberately STRONG adapter (large alpha) so the low-rank delta shifts
    // the greedy argmax — the engine test needs an observable TOKEN change, not
    // just a logit nudge (the bit-level delta math is pinned in the models-crate
    // `lora` tests).
    write_synthetic_lora_adapter_gguf(
        path,
        &SynthLoraAdapter {
            base: SynthLlama::default(),
            rank: 8,
            alpha: 400.0,
            seed,
        },
    );
}

#[test]
fn cpu_engine_resident_lora_per_request_selection() {
    let base = tmp("base");
    write_synthetic_llama_gguf(&base, &SynthLlama::default());
    let ad_a = tmp("a");
    let ad_b = tmp("b");
    write_adapter(&ad_a, 0xA11CE);
    write_adapter(&ad_b, 0xB0B);

    let engine = CpuEngine::load(&base, 32).expect("load engine");
    assert_eq!(engine.lora_adapter_names().len(), 0);
    assert!(
        engine
            .attach_lora_adapter("tune-a", &ad_a)
            .expect("attach a")
            > 0
    );
    assert!(
        engine
            .attach_lora_adapter("tune-b", &ad_b)
            .expect("attach b")
            > 0
    );
    assert!(engine.has_lora_adapter("tune-a"));
    assert!(engine.has_lora_adapter("tune-b"));
    assert!(!engine.has_lora_adapter("ghost"));
    assert_eq!(engine.lora_adapter_names().len(), 2);

    let prompt = [3i32, 5, 7];

    // Base (no selection) is deterministic AND byte-identical to a fresh engine
    // that never had an adapter attached — attaching does not touch the weights.
    let base_out = engine
        .generate_token_ids(&prompt, 6, &greedy(6, None))
        .expect("base gen");
    let engine_plain = CpuEngine::load(&base, 32).expect("load plain");
    let plain_out = engine_plain
        .generate_token_ids(&prompt, 6, &greedy(6, None))
        .expect("plain gen");
    assert_eq!(
        base_out, plain_out,
        "an attached-but-unselected adapter must be byte-identical to no adapter"
    );

    // Selecting the adapter changes the output.
    let a_out = engine
        .generate_token_ids(&prompt, 6, &greedy(6, Some("tune-a")))
        .expect("lora-a gen");
    assert_ne!(
        base_out, a_out,
        "selecting a resident adapter must change the output"
    );
    // A second selection reproduces the same result (deterministic routing).
    let a_out2 = engine
        .generate_token_ids(&prompt, 6, &greedy(6, Some("tune-a")))
        .expect("lora-a gen 2");
    assert_eq!(a_out, a_out2, "per-request selection is deterministic");

    // A second resident adapter, selected by its own name, is also applied
    // (its output differs from the base). We assert each adapter changes the
    // output rather than that A and B differ from EACH OTHER: on this tiny
    // random synth model a strong delta can collapse the greedy argmax of two
    // different adapters onto the same degenerate token run even though both
    // shift the logits away from the base. The distinct-adapters-distinct-logits
    // property is pinned bit-level in the models-crate `lora` tests
    // (`ref_a != ref_b`); here the engine-layer claim is that NAME → that
    // adapter's weights are applied per request.
    let b_out = engine
        .generate_token_ids(&prompt, 6, &greedy(6, Some("tune-b")))
        .expect("lora-b gen");
    assert_ne!(
        base_out, b_out,
        "selecting the second resident adapter must also change the output"
    );
    let b_out2 = engine
        .generate_token_ids(&prompt, 6, &greedy(6, Some("tune-b")))
        .expect("lora-b gen 2");
    assert_eq!(
        b_out, b_out2,
        "per-request selection is deterministic for B too"
    );

    // The base is still byte-identical AFTER adapted requests (the scope is
    // restored each time — no leaked thread-active adapter).
    let base_again = engine
        .generate_token_ids(&prompt, 6, &greedy(6, None))
        .expect("base gen again");
    assert_eq!(
        base_out, base_again,
        "base must stay byte-identical after adapted runs"
    );

    // An unknown adapter name fails the request (never a silent base serve).
    let err = engine.generate_token_ids(&prompt, 6, &greedy(6, Some("ghost")));
    assert!(
        err.is_err(),
        "unknown adapter must fail, not serve the base"
    );

    for p in [&base, &ad_a, &ad_b] {
        let _ = std::fs::remove_file(p);
    }
}
