//! Per-model auto-tuning orchestration for the server.
//!
//! The full 10-stage sweep already exists as the CLI `tune --all`
//! orchestrator (`rustllama-cli::cmd_tune_all`), which loads the model,
//! measures kernel-LWS / per-device-perf / kv_dtype-coherence / flash-attn
//! / kv-layout / placement (CPU-vs-GPU dispatch) / batch-size / threads /
//! decision-calibration / MTP-self-spec / chunked-SSM-prefill, and persists
//! each winner to the per-device tuner cache. Rather than re-implement that
//! orchestration here, the server runs its own binary as a subprocess
//! (`current_exe() tune --all --model <path> ...`) and parses the
//! `Stage <n>/10:` lines from its stdout into a process-global
//! [`TuneProgress`] that the GUI polls via `GET /v1/tune/progress`. The
//! sweep prints one non-integer sub-stage header (`Stage 1b/10`, the
//! per-device-perf refinement between stages 1 and 2); the parser treats
//! it as a half-step so the progress bar advances rather than stalling.
//!
//! Why a subprocess and not an in-process call: the sweep reloads the
//! model several times (once per kv_dtype candidate, etc.), and doing it
//! in a child process keeps that churn — and any allocator high-water —
//! out of the long-lived server address space. The child writes to the
//! shared tuner-cache TOML; when it exits, the caller's normal load path
//! reads the fresh winners. When the weight pages are file-backed mmaps,
//! the child and a still-loaded parent share them in the OS page cache
//! rather than doubling physical RAM.

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::AppState;

/// Live progress of the current (or most recent) auto-tune run. A single
/// tune runs at a time; `active` gates a second concurrent request.
#[derive(Clone, serde::Serialize, Default)]
pub struct TuneProgress {
    /// True while a sweep is in flight.
    pub active: bool,
    /// The model being tuned (file stem).
    pub model_id: String,
    /// 1-based index of the current stage (0 before the first stage).
    pub stage_idx: u32,
    /// Total stages in the sweep (10 for `--all`).
    pub stage_total: u32,
    /// Human-readable current stage name (e.g. "placement (measured)").
    pub stage_name: String,
    /// Overall completion 0-100, derived from stage index.
    pub pct: f32,
    /// The most recent meaningful output line from the sweep.
    pub line: String,
    /// Recent output lines (capped), oldest first — drives the log view.
    pub log: Vec<String>,
    /// True once the sweep finished (success or failure).
    pub done: bool,
    /// Set to the failure reason when the sweep errored; `None` on success.
    pub error: Option<String>,
    /// Wall-clock start (unix seconds) of the current/last run.
    pub started_unix: u64,
}

const LOG_CAP: usize = 60;

fn progress() -> &'static Mutex<TuneProgress> {
    static P: OnceLock<Mutex<TuneProgress>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(TuneProgress::default()))
}

/// Snapshot the current progress for the GET endpoint.
pub fn snapshot() -> TuneProgress {
    progress().lock().map(|g| g.clone()).unwrap_or_default()
}

/// True if a tune is currently running (single-flight guard).
pub fn is_active() -> bool {
    progress().lock().map(|g| g.active).unwrap_or(false)
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn begin(model_id: &str, stage_total: u32) {
    if let Ok(mut g) = progress().lock() {
        *g = TuneProgress {
            active: true,
            model_id: model_id.to_string(),
            stage_idx: 0,
            stage_total,
            stage_name: "starting…".to_string(),
            pct: 0.0,
            line: String::new(),
            log: Vec::new(),
            done: false,
            error: None,
            started_unix: now_unix(),
        };
    }
}

/// Strip ANSI/VT escape sequences (CSI `ESC [ … final`) so the progress
/// window shows clean text — the tune subprocess's tracing output is
/// colorized and would otherwise leak `\x1b[32m`-style codes into the log.
/// Char-based so multi-byte UTF-8 survives.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            if chars.peek() == Some(&'[') {
                chars.next();
                while let Some(&d) = chars.peek() {
                    chars.next();
                    if ('@'..='~').contains(&d) {
                        break;
                    }
                }
            }
            // lone ESC (or non-CSI) — dropped
        } else {
            out.push(c);
        }
    }
    out
}

fn push_line(raw: &str) {
    let line = strip_ansi(raw.trim_end());
    let line = line.trim_end().to_string();
    if line.is_empty() {
        return;
    }
    if let Ok(mut g) = progress().lock() {
        // Parse "Stage <label>/T: name" headers to advance the bar. The
        // CLI `tune --all` sweep numbers its headers `1, 1b, 2 .. 10` over
        // a denominator of `10`, so the numerator is NOT always an integer:
        // `1b` is the per-device-perf sub-stage wedged between 1 and 2. We
        // split the leading digits (the displayed stage number) off any
        // trailing letter, and treat a sub-stage as a half-step for the
        // percentage so the bar advances past stage 1 instead of stalling.
        if let Some(rest) = line.strip_prefix("Stage ") {
            if let Some((frac, name)) = rest.split_once(':') {
                if let Some((label, total)) = frac.trim().split_once('/') {
                    let label = label.trim();
                    let digits: String =
                        label.chars().take_while(|c| c.is_ascii_digit()).collect();
                    let is_substage = !digits.is_empty() && digits.len() != label.len();
                    if let (Ok(n), Ok(t)) =
                        (digits.parse::<u32>(), total.trim().parse::<u32>())
                    {
                        g.stage_idx = n;
                        g.stage_total = t.max(1);
                        g.stage_name = name.trim().to_string();
                        // Percentage tracks completed stages; the current
                        // stage counts as in-progress (n-1 finished), and a
                        // sub-stage (`1b`) sits a half-step past its parent.
                        let completed =
                            (n.saturating_sub(1)) as f32 + if is_substage { 0.5 } else { 0.0 };
                        g.pct = (completed / g.stage_total as f32) * 100.0;
                    }
                }
            }
        }
        g.line = line.clone();
        g.log.push(line);
        if g.log.len() > LOG_CAP {
            let overflow = g.log.len() - LOG_CAP;
            g.log.drain(0..overflow);
        }
    }
}

fn finish(err: Option<String>) {
    if let Ok(mut g) = progress().lock() {
        g.active = false;
        g.done = true;
        g.pct = if err.is_none() { 100.0 } else { g.pct };
        if err.is_some() {
            g.error = err;
        }
    }
}

/// What a tune run covers. `Full` = the 10-stage `tune --all` sweep for the
/// selected model; `Placement` = just the CPU-vs-GPU dispatch measurement
/// (`tune --placement`), the quick "tune CPU & GPUs" action.
#[derive(Clone, Copy, PartialEq)]
pub enum TuneScope {
    Full,
    Placement,
}

impl TuneScope {
    fn from_str(s: Option<&str>) -> Self {
        match s {
            Some("placement") | Some("dispatch") | Some("cpu_gpu") => TuneScope::Placement,
            _ => TuneScope::Full,
        }
    }
}

/// Run a tune against `model_path` as a subprocess, streaming its
/// stdout/stderr into the global [`TuneProgress`]. Blocks until the child
/// exits. For `Full`, `force` re-runs every stage (else cached stages are
/// skipped); `Placement` always re-measures. Returns `Err` with a short
/// reason on spawn/exit failure (also latched into progress).
pub fn run_autotune_blocking(
    model_path: &Path,
    force: bool,
    scope: TuneScope,
    console: bool,
) -> Result<(), String> {
    let model_id = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let stage_total = if scope == TuneScope::Placement { 1 } else { 10 };
    begin(&model_id, stage_total);

    let exe = std::env::current_exe()
        .map_err(|e| format!("cannot locate current executable: {e}"))?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("tune");
    match scope {
        TuneScope::Placement => {
            // Single-stage run — `tune --placement` prints no "Stage N/10"
            // header, so seed the stage label for the progress window.
            cmd.arg("--placement");
            if let Ok(mut g) = progress().lock() {
                g.stage_idx = 1;
                g.stage_name = "CPU / GPU dispatch (placement)".to_string();
            }
        }
        TuneScope::Full => {
            // Quick by default: the mandatory first-load autotune runs the
            // full stage coverage with the cheap (quick) measurement sizes
            // so a fresh model is tuned in ~15-30 min, not hours. A user who
            // wants the exhaustive sweep runs `rustllama tune --all --thorough`
            // by hand.
            cmd.arg("--all");
            if force {
                cmd.arg("--force");
            } else {
                // First-load path: skip stages that already have a winner.
                cmd.arg("--skip-cached");
            }
        }
    }
    cmd.arg("--model").arg(model_path);
    // Ask the child's tracing layer not to colorize — belt-and-suspenders
    // alongside strip_ansi(), so the progress log stays clean text.
    cmd.env("NO_COLOR", "1");
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    push_line(&format!(
        "launching {} for {model_id} (this can take several minutes)",
        match scope {
            TuneScope::Placement => "CPU/GPU dispatch tune",
            TuneScope::Full => "full sweep",
        }
    ));

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            let msg = format!("failed to launch tune subprocess: {e}");
            finish(Some(msg.clone()));
            return Err(msg);
        }
    };

    // Drain stderr on a helper thread so a chatty child can't deadlock on
    // a full stderr pipe while we read stdout.
    let stderr = child.stderr.take();
    let stderr_handle = stderr.map(|err| {
        std::thread::spawn(move || {
            let reader = BufReader::new(err);
            for line in reader.lines().map_while(Result::ok) {
                // Surface warnings/errors from the sweep, but not the
                // verbose tracing INFO noise.
                if line.contains("WARN") || line.contains("ERROR") || line.contains("error") {
                    if console {
                        eprintln!("{line}");
                    }
                    push_line(&line);
                }
            }
        })
    });

    if let Some(out) = child.stdout.take() {
        let reader = BufReader::new(out);
        for line in reader.lines().map_while(Result::ok) {
            // Echo the child's own progress output to our console (CLI
            // `serve`), so a first-load sweep reads exactly like running
            // `rustllama tune --all` by hand. The GUI passes `console:
            // false` and renders progress from `/v1/tune/progress`.
            if console {
                println!("{line}");
            }
            push_line(&line);
        }
    }

    let status = child.wait();
    if let Some(h) = stderr_handle {
        let _ = h.join();
    }

    match status {
        Ok(s) if s.success() => {
            push_line("=== tune --all complete ===");
            finish(None);
            Ok(())
        }
        Ok(s) => {
            let msg = format!("tune subprocess exited with {s}");
            finish(Some(msg.clone()));
            Err(msg)
        }
        Err(e) => {
            let msg = format!("failed to wait on tune subprocess: {e}");
            finish(Some(msg.clone()));
            Err(msg)
        }
    }
}

/// Shared MANDATORY first-load auto-tune for BOTH the HTTP `/v1/models/load`
/// handler (GUI) and the CLI `serve` startup load. Runs the full 10-stage
/// sweep (blocking) whenever no tune is already in flight and this model has
/// no tuner-cache entry for this device — so a fresh model is always tuned
/// once, transparently, before it is served, and `serve` is as turnkey as
/// the desktop app. There is no opt-out: a model that is already cached is an
/// instant no-op, so this is safe to call unconditionally on every load.
///
/// `console` echoes sweep progress to stdout (CLI `serve`); the GUI passes
/// `false` and renders progress by polling `/v1/tune/progress`. Non-fatal
/// by contract: any failure is logged and swallowed so the caller still
/// proceeds to load the model with config + coherence-guardrail defaults.
/// Blocking — callers on an async runtime wrap this in `spawn_blocking`.
pub fn maybe_first_load_autotune(model_path: &Path, console: bool) {
    let key = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown-model")
        .to_string();
    if is_active() {
        return;
    }
    if is_untuned(&key) {
        tracing::info!(
            model = %key,
            "first-load auto-tune: no tuner-cache entry for this device — running the \
             full sweep before load"
        );
        if console {
            println!(
                "rustllama: first load of `{key}` on this device — auto-tuning before \
                 serving (one-time, usually a few minutes)."
            );
        }
        if let Err(e) = run_autotune_blocking(model_path, false, TuneScope::Full, console) {
            tracing::warn!(error = %e, "first-load auto-tune failed; loading with defaults");
            if console {
                eprintln!(
                    "rustllama: auto-tune did not complete ({e}); loading with default settings."
                );
            }
        }
    } else if kernel_validation_stale() {
        // The model is already tuned, so the full sweep is skipped — but the
        // GPU-kernel verdicts are missing or were stamped by a DIFFERENT
        // build (whose kernels may differ). Re-validate just the kernels so
        // the specialized accel paths auto-(re)enable WITHOUT a full re-tune
        // or the user clearing any cache. Fast + hardware-only.
        tracing::info!(
            "GPU-kernel verdicts are absent or from a different build — re-validating \
             kernels so the accelerated paths auto-enable"
        );
        if console {
            println!("rustllama: validating GPU kernels on this device (one-time per build).");
        }
        run_validate_kernels_blocking(console);
    }
}

/// Whether the on-device GPU-kernel verdicts need (re)validating for the
/// CURRENT build: `true` when the tuner cache's `rustllama_version` differs
/// from this build (a kernel change may have flipped a verdict) or there is
/// no cache yet. Independent of whether any model is tuned. Conservative: an
/// unreadable cache returns `false` so we never spawn a probe on a bad cache.
fn kernel_validation_stale() -> bool {
    let Some(dir) = rustllama_tuner::default_cache_dir() else {
        return false;
    };
    let key = rustllama_tuner::system_fingerprint();
    match rustllama_tuner::load_cache(&dir, &key) {
        Ok(Some(t)) => t.rustllama_version != env!("CARGO_PKG_VERSION"),
        Ok(None) => true,
        Err(_) => false,
    }
}

/// Run just the `tune --validate-kernels` stage as a subprocess (fast,
/// hardware-only, model-independent): it runs the GPU kernel parity probes,
/// persists a pass/fail verdict per kernel, and stamps the build version so
/// this doesn't re-run until the next build. Non-fatal — failures are logged.
fn run_validate_kernels_blocking(console: bool) {
    begin("validate-kernels", 1);
    if let Ok(mut g) = progress().lock() {
        g.stage_idx = 1;
        g.stage_name = "validate GPU kernels".to_string();
    }
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => {
            finish(Some(format!("cannot locate current executable: {e}")));
            return;
        }
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("tune").arg("--validate-kernels").env("NO_COLOR", "1");
    if !console {
        cmd.stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
    }
    match cmd.status() {
        Ok(s) if s.success() => finish(None),
        Ok(s) => finish(Some(format!("validate-kernels exited with {s}"))),
        Err(e) => finish(Some(format!("failed to launch validate-kernels: {e}"))),
    }
}

/// Whether `model_key` has never been auto-tuned on this device — i.e. the
/// per-device tuner cache has no per-model placement entry for it. Used to
/// gate first-load auto-tune. Conservative: any probe failure (no SYCL
/// device, unreadable cache) returns `false` so we never block a load on a
/// tune we can't key.
pub fn is_untuned(model_key: &str) -> bool {
    let Some(dir) = rustllama_tuner::default_cache_dir() else {
        return false;
    };
    let key = rustllama_tuner::system_fingerprint();
    match rustllama_tuner::load_cache(&dir, &key) {
        Ok(Some(t)) => !t.placement.contains_key(model_key),
        Ok(None) => true, // no cache file yet → never tuned
        Err(_) => false,
    }
}

// ---- HTTP handlers ---------------------------------------------------------

/// `GET /v1/tune/progress` — the GUI polls this to drive the progress
/// window during a first-load or manual re-tune.
pub async fn progress_handler() -> Response {
    Json(snapshot()).into_response()
}

#[derive(serde::Deserialize)]
pub struct RetuneRequest {
    /// Model to re-tune: a `.gguf` path or a cache file-stem name.
    #[serde(alias = "model_id")]
    model: String,
    /// Re-run every stage (invalidate the cache). Defaults to true for an
    /// explicit re-tune; pass false to only fill in missing stages.
    #[serde(default = "default_true")]
    force: bool,
    /// `"all"` (default) = full per-model sweep; `"placement"` = just the
    /// CPU-vs-GPU dispatch measurement (the "tune CPU & GPUs" action).
    #[serde(default)]
    scope: Option<String>,
}

fn default_true() -> bool {
    true
}

#[derive(serde::Serialize)]
struct RetuneResponse {
    tuned: String,
    ok: bool,
}

/// `POST /v1/tune/model` — force a re-tune of an already-known model. Runs
/// the sweep synchronously (the GUI polls `/v1/tune/progress` meanwhile),
/// then the caller reloads the model to apply the fresh winners. Rejects a
/// second concurrent tune with 409.
pub async fn retune_handler(
    State(_state): State<AppState>,
    Json(req): Json<RetuneRequest>,
) -> Response {
    if is_active() {
        return (
            StatusCode::CONFLICT,
            "a tune is already running; wait for it to finish",
        )
            .into_response();
    }
    let Some(path) = resolve_model_path(&req.model) else {
        return (
            StatusCode::NOT_FOUND,
            format!("model not found: {}", req.model),
        )
            .into_response();
    };
    let force = req.force;
    let scope = TuneScope::from_str(req.scope.as_deref());
    // Run the blocking sweep off the async worker pool.
    let result =
        tokio::task::spawn_blocking(move || run_autotune_blocking(&path, force, scope, false))
            .await
            .unwrap_or_else(|e| Err(format!("tune task panicked: {e}")));
    match result {
        Ok(()) => Json(RetuneResponse {
            tuned: req.model,
            ok: true,
        })
        .into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}

/// Resolve a model path or cache file-stem name to an absolute `.gguf`
/// path, mirroring the `/v1/models/load` `name` resolution.
fn resolve_model_path(model: &str) -> Option<std::path::PathBuf> {
    let p = std::path::PathBuf::from(model);
    if p.is_file() {
        return Some(p);
    }
    let cache = rustllama_hub::default_cache_dir()?;
    // Models dropped directly in the models dir sit at depth 1;
    // `list_cached` only walks the hub layout (`owner__repo/file.gguf` at
    // depth 2), so check the root-level `<stem>.gguf` explicitly first —
    // otherwise a locally-placed GGUF (not hub-pulled) can't be re-tuned
    // by its id.
    let direct = cache.join(format!("{model}.gguf"));
    if direct.is_file() {
        return Some(direct);
    }
    let paths = rustllama_hub::list_cached(&cache).ok()?;
    paths.into_iter().find(|p| {
        p.file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s == model)
            .unwrap_or(false)
    })
}
