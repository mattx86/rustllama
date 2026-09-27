// Throwaway diagnostic: dequantize each tensor in a GGUF and report
// any NaN/Inf, plus min/max. Distinguishes "encoder produced invalid
// (non-finite) blocks" from "finite but too-lossy" on a re-quanted
// model. Usage:
//   cargo run -p rustllama-gguf --example scan_finite --release -- <model.gguf>
//
// Handles only the dtypes present in test_v8 (F32 / IQ1_S / IQ2_XXS);
// anything else is skipped.
use rustllama_gguf::dequant;
use rustllama_gguf::{Gguf, GgmlType};

fn main() {
    let path = std::env::args().nth(1).expect("usage: scan_finite <model.gguf>");
    let gguf = Gguf::open(&path).expect("open gguf");
    let infos: Vec<_> = gguf
        .tensors()
        .iter()
        .map(|t| (t.name.clone(), t.dtype, t.element_count() as usize))
        .collect();

    let mut bad = 0usize;
    let mut scanned = 0usize;
    let mut g_min = f32::INFINITY;
    let mut g_max = f32::NEG_INFINITY;
    // Per-dtype finite-range accumulators so we can see if a whole
    // class (e.g. IQ1_S experts) sits at an insane magnitude.
    let mut iq1s_min = f32::INFINITY;
    let mut iq1s_max = f32::NEG_INFINITY;

    for (name, dtype, n) in &infos {
        let Some(bytes) = gguf.tensor_bytes(name) else { continue };
        let mut out = vec![0f32; *n];
        let handled = match dtype {
            GgmlType::F32 => {
                for (i, c) in bytes[..*n * 4].chunks_exact(4).enumerate() {
                    out[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
                true
            }
            GgmlType::IQ1_S => { dequant::dequant_iq1_s(bytes, &mut out); true }
            GgmlType::IQ2_XXS => { dequant::dequant_iq2_xxs(bytes, &mut out); true }
            _ => false,
        };
        if !handled { continue; }
        scanned += 1;
        let mut nan = 0usize;
        let mut inf = 0usize;
        let mut tmin = f32::INFINITY;
        let mut tmax = f32::NEG_INFINITY;
        for &v in &out {
            if v.is_nan() { nan += 1; }
            else if v.is_infinite() { inf += 1; }
            else { if v < tmin { tmin = v; } if v > tmax { tmax = v; } }
        }
        if nan > 0 || inf > 0 {
            bad += 1;
            println!("  BAD  {name:48} {dtype:?}  nan={nan} inf={inf} (n={n})");
        } else {
            if tmin < g_min { g_min = tmin; }
            if tmax > g_max { g_max = tmax; }
            if matches!(dtype, GgmlType::IQ1_S) {
                if tmin < iq1s_min { iq1s_min = tmin; }
                if tmax > iq1s_max { iq1s_max = tmax; }
            }
            if std::env::var("VERBOSE").is_ok() {
                println!("  ok   {name:48} {dtype:?}  min={tmin:+.4e} max={tmax:+.4e}");
            }
        }
    }

    println!("\n=== scan_finite summary ===");
    println!("scanned tensors : {scanned}");
    println!("non-finite      : {bad}");
    println!("finite range    : min={g_min:+.4e}  max={g_max:+.4e}");
    if iq1s_min.is_finite() {
        println!("IQ1_S range     : min={iq1s_min:+.4e}  max={iq1s_max:+.4e}");
    }
    if bad == 0 {
        println!("RESULT: all scanned weights are FINITE (no NaN/Inf) -- not an invalid-block encoder bug.");
    } else {
        println!("RESULT: {bad} tensor(s) contain NaN/Inf -- encoder produced invalid blocks.");
    }
}
