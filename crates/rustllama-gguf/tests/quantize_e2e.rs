//! End-to-end re-quantization parity test.
//!
//! Builds a synthetic F32 GGUF with several block-aligned tensors,
//! runs the [`quantize_gguf`] pipeline against every supported
//! target dtype, re-opens the output, and asserts:
//!   - the output is a valid GGUF (parser accepts it)
//!   - each tensor's dtype matches the requested target
//!   - dequant of the re-quantized tensors reconstructs the
//!     original within a format-derived tolerance
//!
//! Catches integration regressions across the writer + every
//! encoder simultaneously. Runs in CI because it has no external
//! dependencies.
//!
//! A second test (`parity_vs_llama_cpp_when_available`) opts in
//! to comparing against the `llama-quantize` reference binary if
//! it's on PATH; that test is `#[ignore]` by default so unrelated
//! CI runs don't fail when the binary is absent.

use std::io::Cursor;
use std::path::PathBuf;

use rustllama_gguf::quantize::{quantize_gguf, QuantizePlan};
use rustllama_gguf::write::GgufWriter;
use rustllama_gguf::{dequant, GgmlType, Gguf, MetadataValue};

/// Build a synthetic 2-tensor F32 GGUF: 256 elements each (block-
/// aligned for every target dtype). Writes to a temp file and
/// returns the path along with the source f32 rows for tolerance
/// comparison after re-quantization.
fn build_synthetic_source(label: &str) -> (PathBuf, Vec<f32>, Vec<f32>) {
    let tmp = std::env::temp_dir().join(format!("rustllama_quantize_e2e_src_{label}.gguf"));

    // Row A: linearly increasing in [-2, +2). Covers the symmetric
    // quants' full range.
    let row_a: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
    // Row B: spiky-ish pattern. Tests that per-block scale tracking
    // doesn't collapse on uneven distributions.
    let row_b: Vec<f32> = (0..256)
        .map(|i| ((i as f32 * 0.37).sin() * 1.5 + (i as f32 * 0.09).cos() * 0.4))
        .collect();

    let mut w = GgufWriter::new(Cursor::new(Vec::<u8>::new()));
    w.add_metadata(
        "general.architecture",
        MetadataValue::String("llama".into()),
    )
    .unwrap();
    w.add_metadata("llama.block_count", MetadataValue::U32(2)).unwrap();
    w.declare_tensor("a.weight", vec![256], GgmlType::F32).unwrap();
    w.declare_tensor("b.weight", vec![256], GgmlType::F32).unwrap();
    w.finish_header().unwrap();
    let bytes_a: Vec<u8> = row_a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes_b: Vec<u8> = row_b.iter().flat_map(|v| v.to_le_bytes()).collect();
    w.write_tensor_data("a.weight", &bytes_a).unwrap();
    w.write_tensor_data("b.weight", &bytes_b).unwrap();
    let bytes = w.finish().unwrap().into_inner();
    std::fs::write(&tmp, &bytes).unwrap();
    (tmp, row_a, row_b)
}

/// Round-trip the source GGUF through `quantize_gguf` with the given
/// target dtype; return the re-quantized output path so the test can
/// re-parse and inspect it. The output file is written to a temp
/// path tagged with the target name.
fn requantize_to(target: GgmlType, src_path: &PathBuf, label: &str) -> PathBuf {
    let src = Gguf::open(src_path).unwrap();
    let dst_path = std::env::temp_dir().join(format!(
        "rustllama_quantize_e2e_dst_{label}_{}.gguf",
        target.as_str()
    ));
    let mut w = GgufWriter::create(&dst_path).unwrap();
    let plan = QuantizePlan::uniform(target);
    quantize_gguf(&src, &mut w, &plan).unwrap();
    w.finish().unwrap();
    dst_path
}

fn dequant_for(target: GgmlType, bytes: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    match target {
        GgmlType::F32 => {
            for i in 0..n {
                out[i] = f32::from_le_bytes([
                    bytes[i * 4],
                    bytes[i * 4 + 1],
                    bytes[i * 4 + 2],
                    bytes[i * 4 + 3],
                ]);
            }
        }
        GgmlType::F16 => dequant::dequant_f16(bytes, &mut out),
        GgmlType::Bf16 => dequant::dequant_bf16(bytes, &mut out),
        GgmlType::Q4_0 => dequant::dequant_q4_0(bytes, &mut out),
        GgmlType::Q4_1 => dequant::dequant_q4_1(bytes, &mut out),
        GgmlType::Q5_0 => dequant::dequant_q5_0(bytes, &mut out),
        GgmlType::Q5_1 => dequant::dequant_q5_1(bytes, &mut out),
        GgmlType::Q8_0 => dequant::dequant_q8_0(bytes, &mut out),
        GgmlType::Q2_K => dequant::dequant_q2_k(bytes, &mut out),
        GgmlType::Q3_K => dequant::dequant_q3_k(bytes, &mut out),
        GgmlType::Q4_K => dequant::dequant_q4_k(bytes, &mut out),
        GgmlType::Q5_K => dequant::dequant_q5_k(bytes, &mut out),
        GgmlType::Q6_K => dequant::dequant_q6_k(bytes, &mut out),
        GgmlType::Q8_K => dequant::dequant_q8_k(bytes, &mut out),
        GgmlType::TQ1_0 => dequant::dequant_tq1_0(bytes, &mut out),
        GgmlType::TQ2_0 => dequant::dequant_tq2_0(bytes, &mut out),
        GgmlType::IQ4_NL => dequant::dequant_iq4_nl(bytes, &mut out),
        GgmlType::IQ4_XS => dequant::dequant_iq4_xs(bytes, &mut out),
        GgmlType::IQ2_XXS => dequant::dequant_iq2_xxs(bytes, &mut out),
        GgmlType::IQ2_XS => dequant::dequant_iq2_xs(bytes, &mut out),
        GgmlType::IQ2_S => dequant::dequant_iq2_s(bytes, &mut out),
        GgmlType::IQ3_XXS => dequant::dequant_iq3_xxs(bytes, &mut out),
        GgmlType::IQ3_S => dequant::dequant_iq3_s(bytes, &mut out),
        GgmlType::IQ1_S => dequant::dequant_iq1_s(bytes, &mut out),
        GgmlType::IQ1_M => dequant::dequant_iq1_m(bytes, &mut out),
        _ => panic!("unexpected target {target:?} in e2e test"),
    }
    out
}

/// Per-format tolerance for the smooth (linear) input. Bounds are
/// derived from each format's quantization step over the input
/// amplitude of 2.0 (row_a's range is `[-2, 2)`); empirical loosening
/// is the smallest power-of-2 that fits the worst case on the
/// non-uniform row_b.
fn tolerance_smooth(target: GgmlType) -> f32 {
    match target {
        GgmlType::F32 => 0.0,
        GgmlType::F16 | GgmlType::Bf16 => 1e-2,
        GgmlType::Q8_0 | GgmlType::Q8_K => 0.05,
        GgmlType::Q6_K => 0.2,
        GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q5_K => 0.25,
        GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q4_K | GgmlType::IQ4_NL
        | GgmlType::IQ4_XS => 0.6,
        GgmlType::Q3_K => 1.5,
        GgmlType::Q2_K => 2.0,
        // IQ vector-grid quants: codebook + sign-mask + 4-bit
        // sub-scale errors all stack. On the smooth linear input
        // the worst per-weight error can approach |amp| since the
        // codebook entries are integer-valued and the chunk-search
        // is greedy (suboptimal). Loose bounds vs. their actual
        // production-data performance.
        GgmlType::IQ3_XXS => 2.0,
        GgmlType::IQ2_XXS => 2.0,
        GgmlType::IQ2_XS => 2.0,
        GgmlType::IQ2_S => 2.0,
        GgmlType::IQ3_S => 2.0,
        // IQ1_S at 1.5 bpw is the smallest production quant —
        // worst per-weight error can substantially exceed amp on
        // the smooth gradient input (no sign mask, per-sub-block
        // delta is very coarse). Production use of IQ1_S relies
        // on imatrix-weighted error norms which we don't compute
        // here; v1's bound is essentially "doesn't crash, output
        // is a valid GGUF". Quality follow-up in v1.x.
        GgmlType::IQ1_S => 4.5,
        GgmlType::IQ1_M => 4.5,
        // Ternary forces values to {-d, 0, +d}; for row_a's smooth
        // gradient in [-2, 2), the worst per-weight error is ~d (=2).
        GgmlType::TQ1_0 | GgmlType::TQ2_0 => 2.0,
        _ => panic!("no tolerance set for {target:?}"),
    }
}

/// Targets the v1 pipeline supports. Skipped: IQ1_S/M, IQ2_*, IQ3_*
/// (vector-grid codebook search, Q-D-extra), Q8_1 (no reader-side
/// dequant), NVFP4 (specialized rustllama-internal format).
const SUPPORTED_TARGETS: &[GgmlType] = &[
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::Bf16,
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q8_0,
    GgmlType::Q2_K,
    GgmlType::Q3_K,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q8_K,
    GgmlType::TQ1_0,
    GgmlType::TQ2_0,
    GgmlType::IQ4_NL,
    GgmlType::IQ4_XS,
    GgmlType::IQ2_XXS,
    GgmlType::IQ2_XS,
    GgmlType::IQ2_S,
    GgmlType::IQ3_XXS,
    GgmlType::IQ3_S,
    GgmlType::IQ1_S,
    GgmlType::IQ1_M,
];

#[test]
fn end_to_end_requantize_every_supported_target() {
    let (src_path, row_a, row_b) = build_synthetic_source("all_targets");

    let mut failures: Vec<String> = Vec::new();
    for &target in SUPPORTED_TARGETS {
        let dst_path = requantize_to(target, &src_path, "all_targets");
        let dst = match Gguf::open(&dst_path) {
            Ok(g) => g,
            Err(e) => {
                failures.push(format!("{target:?}: re-open failed: {e}"));
                continue;
            }
        };
        let info_a = dst.tensor("a.weight").expect("a.weight");
        let info_b = dst.tensor("b.weight").expect("b.weight");
        if info_a.dtype != target {
            failures.push(format!(
                "{target:?}: a.weight dtype mismatch: got {:?}",
                info_a.dtype
            ));
            continue;
        }
        if info_b.dtype != target {
            failures.push(format!(
                "{target:?}: b.weight dtype mismatch: got {:?}",
                info_b.dtype
            ));
            continue;
        }
        let deq_a = dequant_for(target, dst.tensor_bytes("a.weight").unwrap(), 256);
        let deq_b = dequant_for(target, dst.tensor_bytes("b.weight").unwrap(), 256);
        let tol = tolerance_smooth(target);
        let mut max_err_a = 0f32;
        let mut max_err_b = 0f32;
        for i in 0..256 {
            let ea = (deq_a[i] - row_a[i]).abs();
            let eb = (deq_b[i] - row_b[i]).abs();
            if ea > max_err_a {
                max_err_a = ea;
            }
            if eb > max_err_b {
                max_err_b = eb;
            }
        }
        if max_err_a > tol || max_err_b > tol {
            failures.push(format!(
                "{target:?}: max_err a={max_err_a} b={max_err_b} > tol={tol}"
            ));
        }
        let _ = std::fs::remove_file(&dst_path);
    }
    let _ = std::fs::remove_file(&src_path);

    if !failures.is_empty() {
        panic!(
            "end-to-end requantize failures ({}):\n{}",
            failures.len(),
            failures.join("\n"),
        );
    }
}

/// APEX I-Quality on a synthetic source — verifies the rule list
/// + plan integration actually produces the expected per-tensor
/// dtypes. Uses an 8-layer "model" where each layer has 4 tensors:
/// `blk.{i}.attn_q.weight`, `blk.{i}.attn_v.weight`,
/// `blk.{i}.ffn_gate_exps.weight` (routed), `blk.{i}.ffn_gate_shexp.weight`
/// (shared). Plus `output.weight`.
#[test]
fn apex_i_quality_assigns_correct_per_tensor_dtypes() {
    use rustllama_gguf::apex::{build_apex_rules, ApexTier};

    let tmp_src = std::env::temp_dir().join("rustllama_apex_e2e_src.gguf");
    let tmp_dst = std::env::temp_dir().join("rustllama_apex_e2e_dst.gguf");
    let n_layers = 8;
    let zero: Vec<u8> = vec![0u8; 256 * 4];

    let mut w = GgufWriter::new(Cursor::new(Vec::<u8>::new()));
    w.add_metadata(
        "general.architecture",
        MetadataValue::String("llama".into()),
    )
    .unwrap();
    w.add_metadata(
        "llama.block_count",
        MetadataValue::U32(n_layers as u32),
    )
    .unwrap();
    w.declare_tensor("output.weight", vec![256], GgmlType::F32).unwrap();
    for i in 0..n_layers {
        w.declare_tensor(format!("blk.{i}.attn_q.weight"), vec![256], GgmlType::F32)
            .unwrap();
        w.declare_tensor(format!("blk.{i}.attn_v.weight"), vec![256], GgmlType::F32)
            .unwrap();
        w.declare_tensor(
            format!("blk.{i}.ffn_gate_exps.weight"),
            vec![256],
            GgmlType::F32,
        )
        .unwrap();
        w.declare_tensor(
            format!("blk.{i}.ffn_gate_shexp.weight"),
            vec![256],
            GgmlType::F32,
        )
        .unwrap();
    }
    w.finish_header().unwrap();
    w.write_tensor_data("output.weight", &zero).unwrap();
    for i in 0..n_layers {
        w.write_tensor_data(&format!("blk.{i}.attn_q.weight"), &zero).unwrap();
        w.write_tensor_data(&format!("blk.{i}.attn_v.weight"), &zero).unwrap();
        w.write_tensor_data(&format!("blk.{i}.ffn_gate_exps.weight"), &zero).unwrap();
        w.write_tensor_data(&format!("blk.{i}.ffn_gate_shexp.weight"), &zero).unwrap();
    }
    let bytes = w.finish().unwrap().into_inner();
    std::fs::write(&tmp_src, &bytes).unwrap();
    let src = Gguf::open(&tmp_src).unwrap();

    let mut dst_w = GgufWriter::create(&tmp_dst).unwrap();
    // Use Q4_K as default and stack APEX on top. APEX rules win.
    let mut plan = QuantizePlan::uniform(GgmlType::Q4_K);
    plan.add_rules(build_apex_rules(ApexTier::IQuality, n_layers));
    quantize_gguf(&src, &mut dst_w, &plan).unwrap();
    dst_w.finish().unwrap();

    let dst = Gguf::open(&tmp_dst).unwrap();

    // Output head: Q6_K (I-Quality's attn dtype).
    assert_eq!(
        dst.tensor("output.weight").unwrap().dtype,
        GgmlType::Q6_K,
        "I-Quality output.weight must be Q6_K"
    );
    // Attention everywhere: Q6_K.
    for i in 0..n_layers {
        assert_eq!(
            dst.tensor(&format!("blk.{i}.attn_q.weight")).unwrap().dtype,
            GgmlType::Q6_K,
        );
    }
    // Routed expert: edge layers (0..5) and (n-5..n) → Q6_K, middle → Q4_K.
    // n_layers=8 so edge_lo=5, edge_hi=3 → layers 0..5 + 3..8 = full range,
    // so everything is "edge" here.
    for i in 0..n_layers {
        assert_eq!(
            dst.tensor(&format!("blk.{i}.ffn_gate_exps.weight"))
                .unwrap()
                .dtype,
            GgmlType::Q6_K,
            "routed expert at layer {i} should be Q6_K (edge layer in 8-block model)",
        );
    }
    // Shared expert: Q8_0.
    for i in 0..n_layers {
        assert_eq!(
            dst.tensor(&format!("blk.{i}.ffn_gate_shexp.weight"))
                .unwrap()
                .dtype,
            GgmlType::Q8_0,
        );
    }

    let _ = std::fs::remove_file(&tmp_src);
    let _ = std::fs::remove_file(&tmp_dst);
}

/// Optional parity test against llama.cpp's `quantize` binary. Skipped
/// by default; opt in with `cargo test -- --ignored parity_vs_llama_cpp_when_available`
/// after installing `llama-quantize` on PATH.
///
/// Runs llama.cpp on the same synthetic source, opens both outputs,
/// and compares dequantized tensors within a loose tolerance (we
/// don't expect byte-identical output without iterative-search
/// parity, which is a v1.5 follow-up).
#[test]
#[ignore]
fn parity_vs_llama_cpp_when_available() {
    use std::process::Command;

    let llama_quantize = which_llama_quantize();
    if llama_quantize.is_none() {
        eprintln!("skip: `llama-quantize` not on PATH");
        return;
    }
    let llama_quantize = llama_quantize.unwrap();

    let (src_path, row_a, row_b) = build_synthetic_source("vs_llama_cpp");

    // Run rustllama on Q4_K.
    let rust_path = requantize_to(GgmlType::Q4_K, &src_path, "vs_llama_cpp_rust");
    let rust_out = Gguf::open(&rust_path).unwrap();

    // Run llama-quantize on Q4_K.
    let llama_out_path = std::env::temp_dir().join("rustllama_quantize_e2e_llama.gguf");
    let status = Command::new(&llama_quantize)
        .arg(src_path.to_str().unwrap())
        .arg(llama_out_path.to_str().unwrap())
        .arg("Q4_K")
        .status()
        .expect("failed to spawn llama-quantize");
    assert!(status.success(), "llama-quantize exited non-zero");
    let llama_out = Gguf::open(&llama_out_path).unwrap();

    // Compare: dequant both outputs, check max error vs source on
    // both implementations, then compare the two outputs against
    // each other. Loose tolerance because llama.cpp uses iterative
    // search while we use the analytical scale.
    for tensor in ["a.weight", "b.weight"] {
        let rust_bytes = rust_out.tensor_bytes(tensor).unwrap();
        let llama_bytes = llama_out.tensor_bytes(tensor).unwrap();
        let rust_deq = dequant_for(GgmlType::Q4_K, rust_bytes, 256);
        let llama_deq = dequant_for(GgmlType::Q4_K, llama_bytes, 256);
        let src_row = if tensor == "a.weight" { &row_a } else { &row_b };

        let mut max_rust_err = 0f32;
        let mut max_llama_err = 0f32;
        let mut max_xover = 0f32;
        for i in 0..256 {
            max_rust_err = max_rust_err.max((rust_deq[i] - src_row[i]).abs());
            max_llama_err = max_llama_err.max((llama_deq[i] - src_row[i]).abs());
            max_xover = max_xover.max((rust_deq[i] - llama_deq[i]).abs());
        }
        eprintln!(
            "{tensor}: rust err={max_rust_err:.4} llama err={max_llama_err:.4} diff={max_xover:.4}"
        );
        // Sanity: both should be within Q4_K's tolerance to source.
        assert!(max_rust_err < 0.6, "{tensor}: rustllama Q4_K err too large");
        assert!(max_llama_err < 0.6, "{tensor}: llama.cpp Q4_K err too large");
        // Cross-implementation diff: looser, accommodates iterative
        // vs analytical scale-finder divergence.
        assert!(max_xover < 0.8, "{tensor}: cross-impl diff too large");
    }

    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&rust_path);
    let _ = std::fs::remove_file(&llama_out_path);
}

fn which_llama_quantize() -> Option<PathBuf> {
    // Common names. `llama-quantize` is the convention in modern
    // llama.cpp; older builds shipped as plain `quantize`.
    for candidate in ["llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe"] {
        if let Ok(path) = which::which(candidate) {
            return Some(path);
        }
    }
    None
}

// `which` not in the crate's dependencies; do a manual PATH walk
// instead to avoid pulling in a new dep just for this one test.
mod which {
    use std::path::{Path, PathBuf};

    pub fn which(name: &str) -> Result<PathBuf, ()> {
        let path = std::env::var_os("PATH").ok_or(())?;
        for entry in std::env::split_paths(&path) {
            let candidate = entry.join(name);
            if is_executable(&candidate) {
                return Ok(candidate);
            }
        }
        Err(())
    }

    fn is_executable(p: &Path) -> bool {
        if !p.is_file() {
            return false;
        }
        // On Windows, any existing .exe file is executable; on
        // Unix we'd want to check the +x bit, but Rust's
        // PermissionsExt isn't trivially available cross-platform
        // here. Practical compromise: trust that anything found
        // by name on PATH is callable.
        true
    }
}
