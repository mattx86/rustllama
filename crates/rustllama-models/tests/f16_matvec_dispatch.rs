//! Focused tests for `try_matvec_f16_usm_f32`. Two layers of
//! coverage:
//!
//!   1. **Mock-mode shape contract** (this file): in default test
//!      builds `usm_attn_enabled()` is `false`, so the helper
//!      returns `false` for every input. We pin the no-op +
//!      no-panic behavior so a future change to the early-return
//!      ordering doesn't accidentally regress the CPU fallback.
//!
//!   2. **End-to-end CPU parity** (separate file `forward.rs`):
//!      drives a synth F16 model through the whole forward pass.
//!      My dispatch change inserts the F16-USM tier between
//!      quantized and CPU; on mock-mode builds the new tier is a
//!      no-op so the existing forward test catches any regression
//!      in the CPU fallback path.
//!
//! SYCL-mode parity (comparing CPU vs GPU outputs within F16
//! tolerance) isn't covered here — that requires a SYCL build +
//! device + would race other USM tests for the per-thread context.
//! The kernel itself has parity tests in rustllama-kernels-sycl.

use rustllama_models::accel::try_matvec_f16_usm_f32;
use rustllama_tensor::{Dtype, Tensor};

fn f16_weight(name: &str, m: u64, k: u64) -> Tensor {
    let mut t = Tensor::zeros_cpu(Dtype::F16, vec![m, k]);
    t.name = name.to_string();
    t
}

fn bf16_weight(name: &str, m: u64, k: u64) -> Tensor {
    let mut t = Tensor::zeros_cpu(Dtype::Bf16, vec![m, k]);
    t.name = name.to_string();
    t
}

#[test]
fn try_matvec_f16_usm_returns_false_in_mock_mode() {
    // Default test build = no SYCL context → usm_attn_enabled is
    // false → helper short-circuits to false without touching
    // anything else. Pinning the no-op so the CPU fallback path
    // stays load-bearing.
    let w = f16_weight("test.f16.weight", 16, 32);
    let x = vec![0.0f32; 32];
    let mut out = vec![0.0f32; 16];
    let used_gpu = try_matvec_f16_usm_f32(&w, &x, &mut out, 16, 32);
    assert!(
        !used_gpu,
        "mock-mode build → no USM context → helper must return false"
    );
    // The output is untouched on the false path — callers fall
    // back to CPU and overwrite it themselves.
}

#[test]
fn try_matvec_f16_usm_rejects_non_f16_dtype() {
    // The helper is F16-only; quantized dtypes go through
    // try_matvec_tensor_usm_f32 instead. Non-F16 / non-quant dtypes
    // like Bf16 still hit the helper via matvec_tensor_dispatch but
    // must cleanly return false so the CPU path takes over.
    let w = bf16_weight("test.bf16.weight", 16, 32);
    let x = vec![0.0f32; 32];
    let mut out = vec![0.0f32; 16];
    let used_gpu = try_matvec_f16_usm_f32(&w, &x, &mut out, 16, 32);
    assert!(
        !used_gpu,
        "non-F16 dtype → helper rejects without erroring"
    );
}

#[test]
fn try_matvec_f16_usm_rejects_shape_mismatch_without_panic() {
    // Mock-mode short-circuit means we don't actually reach the
    // shape check, but pinning that calling with wrong sizes
    // never panics protects against a future change that lifts
    // the usm_attn_enabled guard.
    let w = f16_weight("test.f16.shape", 16, 32);
    let wrong_len_x = vec![0.0f32; 31]; // expected 32
    let mut wrong_len_out = vec![0.0f32; 17]; // expected 16
    let used = try_matvec_f16_usm_f32(&w, &wrong_len_x, &mut wrong_len_out, 16, 32);
    assert!(!used, "wrong shapes → false, never panic");

    // Zero dims also short-circuit cleanly.
    let mut out = vec![0.0f32; 0];
    let used = try_matvec_f16_usm_f32(&w, &[], &mut out, 0, 0);
    assert!(!used, "zero dims → false");
}
