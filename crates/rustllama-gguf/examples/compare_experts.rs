// Round-trip reconstruction check: dequant the same expert tensors
// from three GGUFs and measure how well IQ1_S and IQ2_XXS reconstruct
// a higher-precision reference quant. Distinguishes "encoder bug"
// (near-zero correlation) from "precision cliff" (high-but-correlated
// error).
//
//   cargo run -p rustllama-gguf --example compare_experts --release -- \
//       <reference.gguf> <iq1s.gguf> <iq2xxs.gguf>
use rustllama_gguf::dequant;
use rustllama_gguf::{Gguf, GgmlType};

fn dq(g: &Gguf, name: &str) -> Option<Vec<f32>> {
    let info = g.tensor(name)?;
    let n = info.element_count() as usize;
    let bytes = g.tensor_bytes(name)?;
    let mut out = vec![0f32; n];
    match info.dtype {
        GgmlType::Q3_K => dequant::dequant_q3_k(bytes, &mut out),
        GgmlType::IQ1_S => dequant::dequant_iq1_s(bytes, &mut out),
        GgmlType::IQ2_XXS => dequant::dequant_iq2_xxs(bytes, &mut out),
        other => {
            eprintln!("  ({name}: unexpected dtype {other:?}, skipping)");
            return None;
        }
    }
    Some(out)
}

fn compare(label: &str, a: &[f32], r: &[f32]) {
    // a = candidate, r = reference. cosine + rms error + magnitude ratio.
    let n = a.len().min(r.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nr = 0f64;
    let mut se = 0f64;
    for i in 0..n {
        let (x, y) = (a[i] as f64, r[i] as f64);
        dot += x * y;
        na += x * x;
        nr += y * y;
        se += (x - y) * (x - y);
    }
    let cos = if na > 0.0 && nr > 0.0 { dot / (na.sqrt() * nr.sqrt()) } else { 0.0 };
    let rms_err = (se / n as f64).sqrt();
    let rms_ref = (nr / n as f64).sqrt();
    let rms_cand = (na / n as f64).sqrt();
    println!(
        "    {label:8} cos={cos:+.4}  rms_err={rms_err:.4e}  |cand|={rms_cand:.4e}  |ref|={rms_ref:.4e}  err/ref={:.2}",
        rms_err / rms_ref.max(1e-12)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (rp, p1, p2) = (&args[0], &args[1], &args[2]);
    let gref = Gguf::open(rp).expect("open ref");
    let g1 = Gguf::open(p1).expect("open iq1s");
    let g2 = Gguf::open(p2).expect("open iq2xxs");

    // A few representative routed-expert tensors across the depth.
    let names = [
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_down_exps.weight",
        "blk.20.ffn_gate_exps.weight",
        "blk.20.ffn_up_exps.weight",
        "blk.40.ffn_down_exps.weight",
    ];
    println!("cos near +1 => faithful reconstruction; cos near 0 => garbage (encoder bug)\n");
    for name in names {
        let Some(rref) = dq(&gref, name) else { println!("{name}: ref missing"); continue };
        println!("{name}  (ref dtype {:?}, n={})", gref.tensor(name).unwrap().dtype, rref.len());
        if let Some(a1) = dq(&g1, name) {
            compare("IQ1_S", &a1, &rref);
        }
        if let Some(a2) = dq(&g2, name) {
            compare("IQ2_XXS", &a2, &rref);
        }
        println!();
    }
}
