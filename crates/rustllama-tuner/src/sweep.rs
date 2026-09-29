//! Generic kernel-parameter sweep harness.
//!
//! Used by both the engine (when auto-tuning at first load — phase 5.1)
//! and the `rustllama tune` CLI to find the winning value for a tunable
//! kernel parameter (e.g. local work-group size) across a candidate set
//! at a fixed problem shape.
//!
//! Closure-based so the same harness works for any kernel: the caller
//! supplies a `FnMut(u32) -> Result<(), String>` that runs ONE kernel
//! launch for the given parameter value. The harness handles warmup,
//! repeated timed runs, median selection, and outlier rejection.
//!
//! Scope (Stage B of phase 5):
//!   - One tunable axis per sweep (LWS today; can be extended to a
//!     tuple later via a different harness or repeated single-axis
//!     sweeps).
//!   - Median-of-N timing with a 2-deviation outlier filter. No
//!     model-aware early-stop yet — that lands when sweeps move to
//!     end-to-end tok/s instead of per-kernel µs.

use std::time::{Duration, Instant};

/// Single candidate's measurement summary.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateResult {
    pub value: u32,
    /// Median elapsed time across timed runs (after dropping outliers).
    pub median: Duration,
    /// Per-run elapsed times in submission order (warmups not included).
    pub samples: Vec<Duration>,
    /// `false` if the closure returned `Err` on any run — we still
    /// record any successful samples but mark the candidate failed so
    /// the picker skips it.
    pub ok: bool,
    /// Last error message from the closure, if any.
    pub error: Option<String>,
}

/// Result of a single-axis sweep.
#[derive(Debug, Clone)]
pub struct SweepResult {
    /// All candidates' results in input order.
    pub results: Vec<CandidateResult>,
    /// The candidate value with the lowest median time among the
    /// `ok` candidates. `None` when EVERY candidate failed (caller
    /// should fall back to a hand-picked default).
    pub winner: Option<u32>,
}

impl SweepResult {
    /// `(value, median)` pair for the winner, or `None` if no
    /// candidate succeeded.
    pub fn winner_pair(&self) -> Option<(u32, Duration)> {
        let v = self.winner?;
        self.results
            .iter()
            .find(|c| c.value == v)
            .map(|c| (c.value, c.median))
    }
}

/// Configuration knobs for a single sweep. Defaults via [`Default`].
#[derive(Debug, Clone, Copy)]
pub struct SweepConfig {
    /// Calls discarded before timing starts. 1 is enough for the
    /// quick-by-default sweep: the first launch warms the SYCL
    /// runtime's JIT cache + the device's instruction cache, which is
    /// what the timed runs need discarded. `--thorough` bumps this back
    /// up (see `cmd_tune`) for tighter cold-effect rejection.
    pub warmup_runs: usize,
    /// Timed launches per candidate. 3 gives a plain median (below the
    /// 5-sample trim threshold) that already ranks LWS candidates
    /// reliably — the winner is chosen by relative order, which is
    /// stable at 3 samples on a quiet host. `--thorough` widens this
    /// for tighter confidence intervals when absolute µs matter.
    pub timed_runs: usize,
    /// Skip remaining timed runs for this candidate when partial
    /// median already exceeds `early_stop_ratio × current_best`.
    /// `0.0` disables. Recommended: `2.0` — drop a candidate as soon
    /// as it's 2× slower than the running best.
    pub early_stop_ratio: f64,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            // Quick by default: 1 warmup + 3 timed runs is enough to
            // rank LWS candidates. `--thorough` restores the wider
            // (exhaustive) sweep in `cmd_tune`.
            warmup_runs: 1,
            timed_runs: 3,
            early_stop_ratio: 2.0,
        }
    }
}

/// Run a single-axis parameter sweep. For each candidate value, the
/// `run` closure is invoked `warmup_runs + timed_runs` times; only
/// the timed runs are recorded. After dropping the fastest and
/// slowest sample (when at least 5 samples exist), the candidate's
/// median is its score. The winner is the candidate with the lowest
/// median across all successful candidates.
///
/// `run` returns `Err(msg)` to flag a failed launch — the candidate's
/// `ok` is set to false and it's skipped in winner selection. A
/// failing candidate doesn't abort the whole sweep.
///
/// The closure is responsible for ensuring the kernel actually
/// completes before returning — for SYCL that's `q.submit().wait()`
/// (which our existing kernels already do).
pub fn sweep<F>(candidates: &[u32], cfg: SweepConfig, mut run: F) -> SweepResult
where
    F: FnMut(u32) -> std::result::Result<(), String>,
{
    let mut results: Vec<CandidateResult> = Vec::with_capacity(candidates.len());
    let mut best_median: Option<Duration> = None;

    for &value in candidates {
        // Warmup: discard timing, but still surface errors — if the
        // closure can't even warm up, mark the candidate failed and
        // move on.
        let mut warmup_err: Option<String> = None;
        for _ in 0..cfg.warmup_runs {
            if let Err(e) = run(value) {
                warmup_err = Some(e);
                break;
            }
        }
        if let Some(err) = warmup_err {
            results.push(CandidateResult {
                value,
                median: Duration::ZERO,
                samples: Vec::new(),
                ok: false,
                error: Some(err),
            });
            continue;
        }

        // Timed runs with early-stop.
        let mut samples: Vec<Duration> = Vec::with_capacity(cfg.timed_runs);
        let mut run_err: Option<String> = None;
        for i in 0..cfg.timed_runs {
            let t0 = Instant::now();
            let r = run(value);
            let elapsed = t0.elapsed();
            if let Err(e) = r {
                run_err = Some(e);
                break;
            }
            samples.push(elapsed);
            // Early stop: if this candidate is already 2× slower than
            // the running best (using mean so far as a cheap proxy),
            // skip the rest. Only kicks in after we have ≥2 samples
            // to avoid one cold sample tanking a good candidate.
            if cfg.early_stop_ratio > 0.0 && samples.len() >= 2 {
                if let Some(best) = best_median {
                    let mean_so_far: Duration = samples.iter().sum::<Duration>() / (i + 1) as u32;
                    if mean_so_far.as_secs_f64() > cfg.early_stop_ratio * best.as_secs_f64() {
                        break;
                    }
                }
            }
        }

        let ok = run_err.is_none() && !samples.is_empty();
        let median = if ok {
            compute_trimmed_median(&samples)
        } else {
            Duration::ZERO
        };
        if ok {
            if best_median.is_none_or(|b| median < b) {
                best_median = Some(median);
            }
        }
        results.push(CandidateResult {
            value,
            median,
            samples,
            ok,
            error: run_err,
        });
    }

    let winner = results
        .iter()
        .filter(|c| c.ok)
        .min_by_key(|c| c.median)
        .map(|c| c.value);

    SweepResult { results, winner }
}

/// Median of `samples` after dropping the fastest and slowest when
/// `samples.len() >= 5`. Below that threshold we just take the plain
/// median (sort + middle element) since outlier rejection on a tiny
/// window would discard real signal.
fn compute_trimmed_median(samples: &[Duration]) -> Duration {
    let mut sorted: Vec<Duration> = samples.to_vec();
    sorted.sort();
    let slice = if sorted.len() >= 5 {
        &sorted[1..sorted.len() - 1]
    } else {
        &sorted[..]
    };
    slice[slice.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic deterministic timing: each call returns immediately
    /// but we control the apparent elapsed time via a counter in the
    /// closure's captured state. Validates the sweep math without
    /// depending on real GPU.
    #[test]
    fn sweep_picks_lowest_median() {
        // Candidate 64 is "fastest"; candidate 32 is slowest.
        let mut call_count = 0u32;
        // Latencies in fake nanoseconds, one entry per call (across
        // all candidates + warmups). The harness will call each
        // candidate (warmup_runs + timed_runs) times — with defaults
        // that's 2+7 = 9 per candidate, 27 calls total for 3 candidates.
        // We script latencies so each candidate's median lands at the
        // expected value.
        let latencies_per_candidate: [&[u64]; 3] = [
            &[300, 300, 200, 200, 200, 200, 200, 200, 200], // 16: median ≈ 200
            &[500, 500, 100, 100, 100, 100, 100, 100, 100], // 32: median ≈ 100  ← winner
            &[400, 400, 150, 150, 150, 150, 150, 150, 150], // 64: median ≈ 150
        ];
        let mut flat: Vec<u64> = Vec::new();
        for arr in &latencies_per_candidate {
            flat.extend_from_slice(arr);
        }

        let cfg = SweepConfig {
            warmup_runs: 2,
            timed_runs: 7,
            early_stop_ratio: 0.0, // disable early stop for deterministic test
        };
        let result = sweep(&[16, 32, 64], cfg, |_lws| {
            // Sleep for the scripted latency. Use a busy-wait so the
            // delay is precise — std::thread::sleep has a >1ms floor
            // on Windows.
            let target = Duration::from_micros(flat[call_count as usize]);
            call_count += 1;
            let start = Instant::now();
            while start.elapsed() < target {
                std::hint::spin_loop();
            }
            Ok(())
        });

        assert_eq!(result.winner, Some(32), "candidate 32 has lowest median");
        let r32 = result.results.iter().find(|c| c.value == 32).unwrap();
        assert!(
            r32.ok && r32.samples.len() == 7,
            "candidate 32 ran all 7 timed runs"
        );
    }

    #[test]
    fn sweep_skips_failed_candidates() {
        // Candidate 32 errors on first warmup; its samples is empty
        // and it's excluded from winner selection.
        let mut call_count = 0u32;
        let cfg = SweepConfig {
            warmup_runs: 1,
            timed_runs: 3,
            early_stop_ratio: 0.0,
        };
        let result = sweep(&[16, 32, 64], cfg, |lws| {
            call_count += 1;
            if lws == 32 {
                Err("simulated kernel failure".to_string())
            } else {
                std::thread::sleep(Duration::from_micros(50));
                Ok(())
            }
        });
        assert!(result.winner.is_some());
        assert_ne!(result.winner, Some(32), "failed candidate cannot win");
        let r32 = result.results.iter().find(|c| c.value == 32).unwrap();
        assert!(!r32.ok);
        assert_eq!(
            r32.error.as_deref(),
            Some("simulated kernel failure"),
            "error message preserved"
        );
    }

    #[test]
    fn sweep_all_failed_yields_no_winner() {
        let cfg = SweepConfig::default();
        let result = sweep(&[16, 32, 64], cfg, |_lws| Err("always fails".to_string()));
        assert_eq!(result.winner, None);
        assert!(result.winner_pair().is_none());
        assert!(result.results.iter().all(|c| !c.ok));
    }

    #[test]
    fn trimmed_median_drops_fastest_and_slowest_for_n_ge_5() {
        // 5 samples: drop fastest (10) + slowest (100), median of
        // remaining 3 [20, 30, 40] is 30.
        let samples = vec![
            Duration::from_micros(10),
            Duration::from_micros(20),
            Duration::from_micros(30),
            Duration::from_micros(40),
            Duration::from_micros(100),
        ];
        let m = compute_trimmed_median(&samples);
        assert_eq!(m, Duration::from_micros(30));
    }

    #[test]
    fn trimmed_median_uses_plain_median_for_small_n() {
        // 3 samples: no trimming, plain median = middle = 20.
        let samples = vec![
            Duration::from_micros(10),
            Duration::from_micros(20),
            Duration::from_micros(30),
        ];
        let m = compute_trimmed_median(&samples);
        assert_eq!(m, Duration::from_micros(20));
    }
}
