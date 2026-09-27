//! Phase 8 (kernel-level): parity test for the Gated DeltaNet layer
//! forward vs a pure-numpy Python reference.
//!
//! The Python reference (`scripts/qwen35moe_reference_deltanet.py`)
//! implements the layer math step-by-step from the fla-org / paper
//! equations and dumps a JSON fixture with:
//!   - tiny but real-shape weights
//!   - two input tokens (so we exercise state carry across calls)
//!   - the expected outputs from each token
//!
//! This test loads the fixture and runs the same forward through
//! the Rust kernel-CPU module's
//! `delta_net_layer_forward_f32`. If the math implementations agree,
//! outputs match within FP rounding tolerance.
//!
//! Regenerate the fixture (e.g. after a math change) via:
//!   python scripts/qwen35moe_reference_deltanet.py

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use rustllama_kernels_cpu::delta_net::delta_net_layer_forward_f32;

fn fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("deltanet_layer_v1.json");
    p
}

fn read_floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("expected array of floats")
        .iter()
        .map(|x| x.as_f64().expect("expected number") as f32)
        .collect()
}

/// Pull a numeric shape field out of the fixture's `shapes` object.
fn read_shape(s: &serde_json::Value, key: &str) -> usize {
    s.get(key)
        .unwrap_or_else(|| panic!("shapes.{key} missing"))
        .as_u64()
        .unwrap_or_else(|| panic!("shapes.{key} not u64"))
        as usize
}

#[test]
fn deltanet_layer_forward_matches_python_reference() {
    let path = fixture_path();
    if !path.exists() {
        // Fixture is checked in. If it goes missing in a contributor's
        // local checkout (or CI), point them at the regen command
        // rather than just failing with a confusing IO error.
        panic!(
            "fixture missing at {} — regenerate via \
             `python scripts/qwen35moe_reference_deltanet.py`",
            path.display()
        );
    }
    let f = File::open(&path).expect("open fixture");
    let v: serde_json::Value =
        serde_json::from_reader(BufReader::new(f)).expect("parse fixture");

    let shapes = v.get("shapes").expect("shapes");
    let d_model = read_shape(shapes, "d_model");
    let ssm_inner = read_shape(shapes, "ssm_inner");
    let n_qk_heads = read_shape(shapes, "n_qk_heads");
    let n_v_heads = read_shape(shapes, "n_v_heads");
    let head_qk_dim = read_shape(shapes, "head_qk_dim");
    let head_v_dim = read_shape(shapes, "head_v_dim");
    let conv_kernel = read_shape(shapes, "conv_kernel");
    let rms_eps = shapes
        .get("rms_eps")
        .and_then(|x| x.as_f64())
        .unwrap_or(1e-5) as f32;

    let w = v.get("weights").expect("weights");
    let w_qkv = read_floats(&w["W_qkv"]);
    let w_gate = read_floats(&w["W_gate"]);
    let w_alpha = read_floats(&w["W_alpha"]);
    let w_beta = read_floats(&w["W_beta"]);
    let w_conv1d = read_floats(&w["W_conv1d"]);
    let a_log = read_floats(&w["A_log"]);
    let dt_bias = read_floats(&w["dt_bias"]);
    let ssm_norm = read_floats(&w["ssm_norm"]);
    let w_out = read_floats(&w["W_out"]);

    let hidden_a = read_floats(v.get("hidden_a").expect("hidden_a"));
    let hidden_b = read_floats(v.get("hidden_b").expect("hidden_b"));
    let expected_a = read_floats(v.get("expected_out_a").expect("expected_out_a"));
    let expected_b = read_floats(v.get("expected_out_b").expect("expected_out_b"));

    let qkv_dim = 2 * ssm_inner;
    let mut conv_state = vec![0.0f32; (conv_kernel - 1) * qkv_dim];
    let mut rec_state = vec![0.0f32; n_v_heads * head_qk_dim * head_v_dim];

    let mut out_a = vec![0.0f32; d_model];
    delta_net_layer_forward_f32(
        &hidden_a, &w_qkv, &w_gate, &w_conv1d, &w_alpha, &w_beta,
        &a_log, &dt_bias, &ssm_norm, &w_out,
        &mut conv_state, &mut rec_state, &mut out_a,
        d_model, ssm_inner, n_qk_heads, n_v_heads,
        head_qk_dim, head_v_dim, conv_kernel, rms_eps,
    );

    let mut out_b = vec![0.0f32; d_model];
    delta_net_layer_forward_f32(
        &hidden_b, &w_qkv, &w_gate, &w_conv1d, &w_alpha, &w_beta,
        &a_log, &dt_bias, &ssm_norm, &w_out,
        &mut conv_state, &mut rec_state, &mut out_b,
        d_model, ssm_inner, n_qk_heads, n_v_heads,
        head_qk_dim, head_v_dim, conv_kernel, rms_eps,
    );

    // Tolerance: 5e-4 absolute. The reference uses double-precision
    // accumulation in places where the Rust kernel uses f32, so a
    // few ULP of drift is expected. The output magnitude is ~1e-2
    // so this is ~5% relative — tight enough to catch real bugs,
    // loose enough not to flag genuine FP rounding noise.
    const TOL: f32 = 5e-4;

    let cmp = |got: &[f32], exp: &[f32], label: &str| {
        assert_eq!(got.len(), exp.len(), "{label}: length mismatch");
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f32;
        let mut argmax = 0usize;
        for (i, (g, e)) in got.iter().zip(exp.iter()).enumerate() {
            let d = (g - e).abs();
            sum_abs += d;
            if d > max_abs {
                max_abs = d;
                argmax = i;
            }
        }
        let mean_abs = sum_abs / got.len() as f32;
        let pass = max_abs < TOL;
        // Print summary either way — useful for tightening or
        // diagnosing.
        println!(
            "{label}: len={} max_abs={:.2e} (idx {}, got={:.6e} exp={:.6e}) mean_abs={:.2e} tol={:.2e} pass={}",
            got.len(), max_abs, argmax, got[argmax], exp[argmax], mean_abs, TOL, pass
        );
        assert!(pass, "{label}: max abs error {max_abs} exceeded tolerance {TOL}");
    };

    cmp(&out_a, &expected_a, "out_a (first token)");
    cmp(&out_b, &expected_b, "out_b (second token, post-state-carry)");
}
