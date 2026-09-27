//! Phase 8 (kernel-level): parity test for the MoE FFN
//! (`moe::moe_ffn_one_into_parts`) vs a pure-numpy reference.
//!
//! Validates both:
//!   - the router top-K + softmax + renormalization
//!   - per-expert SwiGLU FFN (gate / up / down)
//!   - the shared expert WITH a router-gated weight (qwen35moe)
//!     AND with the DeepSeek-V3-style always-on convention.
//!
//! Regenerate fixture: `python scripts/qwen35moe_reference_moe.py`

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use rustllama_models::moe::{expert_view, moe_ffn_one_into_parts};
use rustllama_tensor::Tensor;

fn fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("moe_ffn_v1.json");
    p
}

fn read_floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("expected array of floats")
        .iter()
        .map(|x| x.as_f64().expect("expected number") as f32)
        .collect()
}

fn read_usize(s: &serde_json::Value, key: &str) -> usize {
    s.get(key)
        .unwrap_or_else(|| panic!("shapes.{key} missing"))
        .as_u64()
        .unwrap_or_else(|| panic!("shapes.{key} not u64"))
        as usize
}

#[test]
fn moe_ffn_one_into_parts_matches_python_reference() {
    let path = fixture_path();
    if !path.exists() {
        panic!(
            "fixture missing at {} — regenerate via \
             `python scripts/qwen35moe_reference_moe.py`",
            path.display()
        );
    }
    let v: serde_json::Value =
        serde_json::from_reader(BufReader::new(File::open(&path).expect("open")))
            .expect("parse fixture");

    let shapes = v.get("shapes").expect("shapes");
    let d_model = read_usize(shapes, "d_model");
    let d_ff = read_usize(shapes, "d_ff");
    let n_experts = read_usize(shapes, "n_experts");
    let top_k = read_usize(shapes, "top_k");

    let w = v.get("weights").expect("weights");

    // Router: [n_experts, d_model] F32 tensor.
    let router_flat = read_floats(&w["router"]);
    let router = Tensor::from_vec_f32(
        "router",
        vec![n_experts as u64, d_model as u64],
        router_flat,
    );

    // Per-expert 3D tensors → carve into per-expert views via the
    // existing `expert_view` helper (the production load path uses
    // it too, so this exercises the same slicing logic).
    let gate_3d = Tensor::from_vec_f32(
        "ffn_gate_exps",
        vec![n_experts as u64, d_ff as u64, d_model as u64],
        read_floats(&w["gate_per_expert_flat"]),
    );
    let up_3d = Tensor::from_vec_f32(
        "ffn_up_exps",
        vec![n_experts as u64, d_ff as u64, d_model as u64],
        read_floats(&w["up_per_expert_flat"]),
    );
    let down_3d = Tensor::from_vec_f32(
        "ffn_down_exps",
        vec![n_experts as u64, d_model as u64, d_ff as u64],
        read_floats(&w["down_per_expert_flat"]),
    );
    let gate_per_expert: Vec<Tensor> = (0..n_experts)
        .map(|e| expert_view(&gate_3d, e, d_ff, d_model))
        .collect();
    let up_per_expert: Vec<Tensor> = (0..n_experts)
        .map(|e| expert_view(&up_3d, e, d_ff, d_model))
        .collect();
    let down_per_expert: Vec<Tensor> = (0..n_experts)
        .map(|e| expert_view(&down_3d, e, d_model, d_ff))
        .collect();

    // Shared expert tensors.
    let w_gate_shared = Tensor::from_vec_f32(
        "ffn_gate_shexp",
        vec![d_ff as u64, d_model as u64],
        read_floats(&w["w_gate_shared"]),
    );
    let w_up_shared = Tensor::from_vec_f32(
        "ffn_up_shexp",
        vec![d_ff as u64, d_model as u64],
        read_floats(&w["w_up_shared"]),
    );
    let w_down_shared = Tensor::from_vec_f32(
        "ffn_down_shexp",
        vec![d_model as u64, d_ff as u64],
        read_floats(&w["w_down_shared"]),
    );
    let shared_router_tensor = Tensor::from_vec_f32(
        "ffn_gate_inp_shexp",
        vec![1, d_model as u64],
        read_floats(&w["shared_router"]),
    );

    let hidden = read_floats(v.get("hidden").expect("hidden"));
    let expected_gated = read_floats(v.get("expected_out_gated").expect("expected_out_gated"));
    let expected_always_on =
        read_floats(v.get("expected_out_always_on").expect("expected_out_always_on"));

    // Scratch buffers (caller-managed in the production path).
    let mut gate_buf = vec![0f32; d_ff];
    let mut up_buf = vec![0f32; d_ff];
    let mut ff_buf = vec![0f32; d_ff];
    let mut down_buf = vec![0f32; d_model];
    let mut expert_logits = vec![0f32; n_experts];
    let mut picks: Vec<(usize, f32)> = Vec::with_capacity(top_k);

    // 1. With router-gated shared expert (qwen35moe convention).
    let mut out_gated = vec![0f32; d_model];
    moe_ffn_one_into_parts(
        &hidden, &router,
        &gate_per_expert, &up_per_expert, &down_per_expert,
        Some(&w_gate_shared), Some(&w_up_shared), Some(&w_down_shared),
        Some(&shared_router_tensor),
        d_model, d_ff, n_experts, top_k,
        &mut out_gated,
        &mut gate_buf, &mut up_buf, &mut ff_buf, &mut down_buf,
        &mut expert_logits, &mut picks,
    );

    // 2. With always-on shared expert (DeepSeek-V3 convention).
    let mut out_always_on = vec![0f32; d_model];
    moe_ffn_one_into_parts(
        &hidden, &router,
        &gate_per_expert, &up_per_expert, &down_per_expert,
        Some(&w_gate_shared), Some(&w_up_shared), Some(&w_down_shared),
        None, // shared_router absent → always-on
        d_model, d_ff, n_experts, top_k,
        &mut out_always_on,
        &mut gate_buf, &mut up_buf, &mut ff_buf, &mut down_buf,
        &mut expert_logits, &mut picks,
    );

    // Tolerance 1e-5: per-expert SwiGLU FFNs sum to O(0.1) outputs;
    // softmax + multiple matvecs accumulate ~1-2 ULP of rounding noise
    // per element. 1e-5 is tight enough to catch math bugs, loose
    // enough not to flag genuine FP rounding.
    const TOL: f32 = 1e-5;

    let cmp = |got: &[f32], exp: &[f32], label: &str| {
        assert_eq!(got.len(), exp.len(), "{label}: length mismatch");
        let mut max_abs = 0.0f32;
        let mut argmax = 0usize;
        for (i, (g, e)) in got.iter().zip(exp.iter()).enumerate() {
            let d = (g - e).abs();
            if d > max_abs {
                max_abs = d;
                argmax = i;
            }
        }
        let pass = max_abs < TOL;
        println!(
            "{label}: len={} max_abs={:.2e} (idx {}, got={:.6e} exp={:.6e}) tol={:.2e} pass={}",
            got.len(), max_abs, argmax, got[argmax], exp[argmax], TOL, pass
        );
        assert!(pass, "{label}: max abs error {max_abs} exceeded tolerance {TOL}");
    };

    cmp(&out_gated, &expected_gated, "out_gated (qwen35moe shared-router)");
    cmp(&out_always_on, &expected_always_on, "out_always_on (DeepSeek-V3 always-on shared)");
}
