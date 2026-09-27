//! Importance-matrix activation collector.
//!
//! When active, the matvec dispatch hooks call [`record`] with each
//! weight tensor's name and the input activation row `x` (length =
//! input-column count `n_in`). We accumulate, per tensor, the sum of
//! squared activations per input column plus a sample count; the mean
//! (`sumsq / samples`) is the per-column importance the quantizer
//! weights its error by.
//!
//! Gated by a single relaxed atomic so the hook is ~free during
//! normal inference (one load + branch). Only the `imatrix`
//! calibration command flips it on.
//!
//! MoE note: per-expert matvecs run against `expert_view` tensors
//! named `"{parent}.e{idx}"`. We strip the `.e<N>` suffix so every
//! expert of a tensor accumulates into one parent-keyed vector —
//! matching the consumption side, which broadcasts a single
//! `n_in`-length vector across all rows *and* experts.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

static COLLECTING: AtomicBool = AtomicBool::new(false);

struct Accum {
    sumsq: Vec<f64>,
    samples: u64,
}

static DATA: OnceLock<Mutex<HashMap<String, Accum>>> = OnceLock::new();

fn data() -> &'static Mutex<HashMap<String, Accum>> {
    DATA.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Strip a trailing `.e<digits>` expert-view suffix, returning the
/// parent GGUF tensor name. `blk.0.ffn_gate_exps.weight.e7` →
/// `blk.0.ffn_gate_exps.weight`. Non-expert names pass through.
fn parent_name(name: &str) -> &str {
    if let Some(pos) = name.rfind(".e") {
        let suffix = &name[pos + 2..];
        if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) {
            return &name[..pos];
        }
    }
    name
}

/// True while a calibration run is collecting. Checked at the top of
/// the matvec dispatch hooks.
#[inline]
pub fn is_collecting() -> bool {
    COLLECTING.load(Ordering::Relaxed)
}

/// Begin a fresh collection run (clears any prior accumulators).
pub fn begin() {
    data().lock().unwrap().clear();
    COLLECTING.store(true, Ordering::Relaxed);
}

/// Record one activation sample for `name`'s input columns:
/// `sumsq[i] += x[i]²`, `samples += 1`. Cheap shape guard skips a
/// sample whose width disagrees with the established accumulator
/// (shouldn't happen for a fixed model, defensive only).
pub fn record(name: &str, x: &[f32]) {
    let key = parent_name(name);
    let mut map = data().lock().unwrap();
    let acc = map
        .entry(key.to_string())
        .or_insert_with(|| Accum { sumsq: vec![0.0; x.len()], samples: 0 });
    if acc.sumsq.len() != x.len() {
        return;
    }
    for (s, &v) in acc.sumsq.iter_mut().zip(x.iter()) {
        *s += (v as f64) * (v as f64);
    }
    acc.samples += 1;
}

/// Stop collecting and return per-tensor mean-squared-activation
/// importance vectors (`sumsq / samples`), keyed by parent GGUF
/// tensor name.
pub fn finish() -> HashMap<String, Vec<f32>> {
    COLLECTING.store(false, Ordering::Relaxed);
    let map = data().lock().unwrap();
    map.iter()
        .filter(|(_, acc)| acc.samples > 0)
        .map(|(name, acc)| {
            let inv = 1.0 / acc.samples as f64;
            let imp: Vec<f32> = acc.sumsq.iter().map(|&s| (s * inv) as f32).collect();
            (name.clone(), imp)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parent_name_strips_expert_suffix() {
        assert_eq!(parent_name("blk.0.ffn_gate_exps.weight.e7"), "blk.0.ffn_gate_exps.weight");
        assert_eq!(parent_name("blk.12.ffn_down_exps.weight.e255"), "blk.12.ffn_down_exps.weight");
        // Non-expert names untouched.
        assert_eq!(parent_name("blk.0.attn_q.weight"), "blk.0.attn_q.weight");
        // `.e` not followed by digits is not an expert suffix.
        assert_eq!(parent_name("blk.0.some.example"), "blk.0.some.example");
    }

    #[test]
    fn record_and_finish_accumulate_mean_sq() {
        begin();
        // Two samples for one tensor; importance = mean of squares.
        record("t.weight", &[1.0, 2.0, 0.0]);
        record("t.weight", &[3.0, 0.0, 4.0]);
        // Expert views fold into the parent.
        record("moe.weight.e0", &[2.0, 2.0]);
        record("moe.weight.e1", &[0.0, 4.0]);
        let out = finish();
        let t = out.get("t.weight").unwrap();
        // col0: (1+9)/2 = 5; col1: (4+0)/2 = 2; col2: (0+16)/2 = 8.
        assert!((t[0] - 5.0).abs() < 1e-4);
        assert!((t[1] - 2.0).abs() < 1e-4);
        assert!((t[2] - 8.0).abs() < 1e-4);
        let m = out.get("moe.weight").unwrap();
        // col0: (4+0)/2 = 2; col1: (4+16)/2 = 10.
        assert!((m[0] - 2.0).abs() < 1e-4);
        assert!((m[1] - 10.0).abs() < 1e-4);
        assert!(!is_collecting());
    }
}
