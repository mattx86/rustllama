//! Phase 8 (kernel-level): parity test for the NextN/MTP composition
//! (`nextn_compose_logits_f32`) vs a pure-numpy reference.
//!
//! Validates: embed lookup → enorm + hnorm → concat → eh_proj →
//! shared_head_norm → LM head. The composition is what predicts
//! token t+2 given the just-predicted token t+1 and the post-block
//! hidden state — the qwen35moe NextN/MTP path.
//!
//! Regenerate fixture: `python scripts/qwen35moe_reference_nextn.py`

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use rustllama_models::llama_arch::{nextn_compose_logits_f32, NextNHead};
use rustllama_tensor::Tensor;

fn fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("nextn_v1.json");
    p
}

fn read_floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

#[test]
fn nextn_compose_logits_matches_python_reference() {
    let path = fixture_path();
    if !path.exists() {
        panic!(
            "fixture missing at {} — regenerate via \
             `python scripts/qwen35moe_reference_nextn.py`",
            path.display()
        );
    }
    let v: serde_json::Value =
        serde_json::from_reader(BufReader::new(File::open(&path).expect("open")))
            .expect("parse fixture");

    let shapes = v.get("shapes").expect("shapes");
    let d_model = shapes["d_model"].as_u64().unwrap() as usize;
    let vocab = shapes["vocab"].as_u64().unwrap() as usize;
    let rms_eps = shapes["rms_eps"].as_f64().unwrap() as f32;
    let next_token_id = shapes["next_token_id"].as_u64().unwrap() as i32;

    let w = v.get("weights").expect("weights");
    let token_embd = Tensor::from_vec_f32(
        "token_embd",
        vec![vocab as u64, d_model as u64],
        read_floats(&w["token_embd"]),
    );
    let head = NextNHead {
        eh_proj: Tensor::from_vec_f32(
            "blk.last.nextn.eh_proj",
            vec![d_model as u64, 2 * d_model as u64],
            read_floats(&w["eh_proj"]),
        ),
        embed_norm: read_floats(&w["embed_norm"]),
        hidden_norm: read_floats(&w["hidden_norm"]),
        shared_head_norm: read_floats(&w["shared_head_norm"]),
    };
    let lm_head = Tensor::from_vec_f32(
        "output",
        vec![vocab as u64, d_model as u64],
        read_floats(&w["lm_head"]),
    );

    let hidden = read_floats(v.get("hidden").expect("hidden"));
    let expected_logits = read_floats(v.get("expected_logits").expect("expected_logits"));

    let mut logits = vec![0f32; vocab];
    nextn_compose_logits_f32(
        &hidden,
        next_token_id,
        &token_embd,
        &head,
        &lm_head,
        rms_eps,
        &mut logits,
    );

    const TOL: f32 = 1e-5;
    let mut max_abs = 0.0f32;
    let mut argmax = 0usize;
    for (i, (g, e)) in logits.iter().zip(expected_logits.iter()).enumerate() {
        let d = (g - e).abs();
        if d > max_abs {
            max_abs = d;
            argmax = i;
        }
    }
    println!(
        "nextn_logits: len={} max_abs={:.2e} (idx {}, got={:.6e} exp={:.6e}) tol={:.2e}",
        logits.len(), max_abs, argmax, logits[argmax], expected_logits[argmax], TOL
    );
    assert!(max_abs < TOL, "max abs error {max_abs} exceeded tolerance {TOL}");
}
