//! Runtime paths, server-record lockfile, and process discovery.
//!
//! Path resolution lives here so every other crate (config / hub / tuner
//! / cli / server) goes through one switchable source of truth.
//!
//! **Default = binary-relative.** All state roots in the directory that holds
//! the running `rustllama` exe (`.`): `config.toml` sits beside it, and
//! `models/`, `tuning/`, `sessions/`, `runtime/`, `logs/` are subdirectories.
//! So an unzipped/downloaded rustllama is self-contained — copy the folder,
//! and its config + models + caches travel with it. `--config` and config
//! keys still override individual paths.
//!
//! **Opt out to the OS standard dirs** (`%APPDATA%`/`%LOCALAPPDATA%` on
//! Windows, XDG on Linux) by setting `RUSTLLAMA_SYSTEM_DIRS` in the env or
//! dropping a `system.flag` next to the exe. (The default is also the OS dirs
//! as a fallback when the exe directory can't be resolved.)

use std::path::PathBuf;
use std::sync::OnceLock;

use fs2::FileExt;
use serde::{Deserialize, Serialize};

pub mod crash;
pub mod gpu_detect;
pub mod gpu_paths;
pub mod sandbox;

pub use gpu_paths::ensure_gpu_dll_search_paths;

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("cannot resolve runtime directory (no %LOCALAPPDATA% / home)")]
    NoRuntimeDir,
}

pub type Result<T> = std::result::Result<T, RuntimeError>;

/// One-shot snapshot of where each kind of state lives on disk. Computed
/// once at process startup so we don't re-resolve `BaseDirs` (which can
/// be slow on Windows) and so portable-mode detection is consistent
/// across the run.
#[derive(Debug, Clone)]
pub struct Paths {
    /// `config.toml` directory.
    pub config_dir: PathBuf,
    /// Model GGUF cache.
    pub cache_dir: PathBuf,
    /// REPL sessions.
    pub sessions_dir: PathBuf,
    /// Autotuner cache.
    pub tuning_dir: PathBuf,
    /// Server record + lockfile.
    pub runtime_dir: PathBuf,
    /// True when these were derived from an exe-relative root rather
    /// than the OS dirs. Surfaced on `/healthz` and `doctor`.
    pub portable: bool,
    /// Directory that crash logs land in (also panic-hook target).
    pub crash_log_dir: PathBuf,
}

static PATHS: OnceLock<Paths> = OnceLock::new();

/// Compute and cache the path map for this process. Subsequent calls
/// return the same `Paths` (no re-resolution). Detection logic:
///   1. `RUSTLLAMA_SYSTEM_DIRS` env var set, or a `system.flag` next to the
///      exe → OS standard dirs.
///   2. Otherwise (the default) → binary-relative: `config.toml` beside the
///      exe, `models/` + `tuning/` + `sessions/` + `runtime/` + `logs/` as
///      subdirectories.
///   3. Fallback → OS standard dirs when the exe directory can't be resolved.
pub fn paths() -> &'static Paths {
    PATHS.get_or_init(resolve_paths)
}

/// Reset the cached paths. Test-only — production code calls `paths()`
/// once and never invalidates.
#[doc(hidden)]
pub fn _reset_paths_for_test() {
    // Not literally resetting `OnceLock` (no public API for that), but
    // calls go through this for documentation. Tests that need a fresh
    // resolution should use `resolve_paths()` directly.
}

/// Visible for the binary entry point: detect portable mode + build
/// the path map, but DON'T install it. Use [`set_paths`] to publish.
pub fn resolve_paths() -> Paths {
    // Default: binary-relative. Root all state next to the exe so rustllama is
    // self-contained — `config.toml` beside the binary, `models/` + `tuning/` +
    // `sessions/` + `runtime/` + `logs/` as subdirectories. Opt out to the OS
    // dirs with RUSTLLAMA_SYSTEM_DIRS / a `system.flag`. Falls back to the OS
    // dirs when the exe directory can't be resolved (unusual).
    if !use_system_dirs() {
        if let Some(root) = exe_dir() {
            return Paths {
                config_dir: root.clone(),
                cache_dir: root.join("models"),
                sessions_dir: root.join("sessions"),
                tuning_dir: root.join("tuning"),
                runtime_dir: root.join("runtime"),
                crash_log_dir: root.join("logs"),
                portable: true,
            };
        }
    }
    {
        // OS standard dirs (opt-in via RUSTLLAMA_SYSTEM_DIRS, or the fallback
        // when there's no resolvable exe directory).
        let base = directories::BaseDirs::new();
        let config_root = base
            .as_ref()
            .map(|b| b.config_dir().join("rustllama"))
            .unwrap_or_else(|| PathBuf::from("rustllama"));
        let data_root = base
            .as_ref()
            .map(|b| b.data_local_dir().join("rustllama"))
            .unwrap_or_else(|| PathBuf::from("rustllama"));
        Paths {
            config_dir: config_root.clone(),
            cache_dir: data_root.join("models"),
            sessions_dir: config_root.join("sessions"),
            tuning_dir: data_root.join("tuning"),
            runtime_dir: data_root.join("runtime"),
            crash_log_dir: data_root.join("logs"),
            portable: false,
        }
    }
}

/// Install a custom path map. Called by the binary if it wants to
/// override detection (e.g., a test, or a one-off invocation that
/// passes `--portable`). Idempotent: only the first call sticks.
pub fn set_paths(p: Paths) -> std::result::Result<(), Paths> {
    PATHS.set(p)
}

/// Opt out of the binary-relative default and use the OS standard dirs.
/// Triggered by `RUSTLLAMA_SYSTEM_DIRS` in the env or a `system.flag` next to
/// the exe. (`RUSTLLAMA_PORTABLE` / `portable.flag` are still accepted but are
/// now no-ops — binary-relative is the default.)
fn use_system_dirs() -> bool {
    if std::env::var_os("RUSTLLAMA_SYSTEM_DIRS").is_some() {
        return true;
    }
    if let Some(dir) = exe_dir() {
        if dir.join("system.flag").exists() {
            return true;
        }
    }
    false
}

fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerRecord {
    pub pid: u32,
    pub port: u16,
    pub bind_addr: String,
    pub started_at: String,
    pub model_id: Option<String>,
    pub version: String,
    /// `"serve"` for a foreground `rustllama serve`, `"chat-ephemeral"` for
    /// a server spun up by `rustllama chat`, `"gui-embedded"` for one
    /// embedded in the GUI process.
    pub owner: String,
}

pub fn runtime_dir() -> Result<PathBuf> {
    Ok(paths().runtime_dir.clone())
}

pub fn record_path() -> Result<PathBuf> {
    Ok(runtime_dir()?.join("server.json"))
}

pub fn lock_path() -> Result<PathBuf> {
    Ok(runtime_dir()?.join("server.lock"))
}

/// Single-instance server lock. Holds an OS advisory *exclusive* lock on
/// `runtime/server.lock` for as long as it's alive; dropping it releases the
/// lock and removes the file. `rustllama serve` (and the GUI-embedded server)
/// should acquire this at startup and keep the guard for the process lifetime,
/// so a second `serve` against the same runtime dir fails fast with a clear
/// "already running" instead of silently clobbering `server.json`. The OS TCP
/// port bind is the ultimate backstop — two servers can't bind the same port —
/// but this is an earlier, port-independent signal keyed to the runtime dir.
#[derive(Debug)]
pub struct ServerLock {
    file: std::fs::File,
    path: PathBuf,
}

impl Drop for ServerLock {
    fn drop(&mut self) {
        // Advisory lock releases on handle close too, but be explicit, then
        // remove the (now-unlocked) marker file.
        let _ = FileExt::unlock(&self.file);
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Try to acquire the exclusive single-instance [`ServerLock`]. Creates the
/// runtime dir + `server.lock` and takes a NON-blocking advisory exclusive
/// lock on it:
///   - `Ok(Some(guard))` — acquired; hold the guard for the server's lifetime.
///   - `Ok(None)`        — another live process already holds it (a server is
///                         already running against this runtime dir).
///   - `Err(_)`          — filesystem error creating/opening the lock file.
pub fn acquire_server_lock() -> Result<Option<ServerLock>> {
    let dir = runtime_dir()?;
    std::fs::create_dir_all(&dir)?;
    let path = lock_path()?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&path)?;
    match file.try_lock_exclusive() {
        Ok(()) => Ok(Some(ServerLock { file, path })),
        // Contention (another process holds it) is the "already running"
        // signal, not a hard error. fs2 surfaces it with a platform-specific
        // error whose kind matches `lock_contended_error()`.
        Err(e) if e.kind() == fs2::lock_contended_error().kind() => Ok(None),
        Err(e) => Err(RuntimeError::Io(e)),
    }
}

/// Read the live-server record from disk if it exists and the recorded
/// process is still the same live server. Stale records are removed.
pub fn read_alive_record() -> Result<Option<ServerRecord>> {
    let p = record_path()?;
    if !p.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&p)?;
    let rec: ServerRecord = serde_json::from_str(&raw)?;
    if is_record_alive(&rec) {
        Ok(Some(rec))
    } else {
        // Best-effort cleanup of the stale discovery record ONLY. We do NOT
        // remove `server.lock` here: its lifetime is owned by the live
        // `ServerLock` guard (a freshly-started server may already hold it),
        // and deleting a held advisory-lock file would undermine the
        // single-instance guarantee.
        let _ = std::fs::remove_file(&p);
        Ok(None)
    }
}

pub fn write_record(rec: &ServerRecord) -> Result<()> {
    let dir = runtime_dir()?;
    std::fs::create_dir_all(&dir)?;
    let p = record_path()?;
    std::fs::write(&p, serde_json::to_vec_pretty(rec)?)?;
    Ok(())
}

pub fn remove_record() -> Result<()> {
    let p = record_path()?;
    if p.exists() {
        std::fs::remove_file(p)?;
    }
    Ok(())
}

pub fn is_alive(pid: u32) -> bool {
    let mut s = sysinfo::System::new();
    s.refresh_processes(sysinfo::ProcessesToUpdate::All);
    s.process(sysinfo::Pid::from_u32(pid)).is_some()
}

/// Max allowed gap (seconds) between a live process's OS start time and the
/// `started_at` stamp in its [`ServerRecord`] before we treat the PID as
/// having been *reused* by an unrelated process. The stamp is written within a
/// short startup window of process launch, so a healthy server's gap is a few
/// seconds; a recycled PID's is not.
const PID_REUSE_TOLERANCE_SECS: u64 = 120;

/// Parse the `epoch-<unix-secs>` stamp `serve` writes into
/// [`ServerRecord::started_at`]. Returns `None` for any other / legacy format
/// so callers fall back to a PID-only liveness check rather than mis-reading a
/// valid record as stale.
fn parse_started_at_epoch(started_at: &str) -> Option<u64> {
    started_at
        .trim()
        .strip_prefix("epoch-")?
        .trim()
        .parse::<u64>()
        .ok()
}

/// Liveness check for a specific [`ServerRecord`] that defends against PID
/// reuse. Bare [`is_alive`] only asks "does *some* process with this PID
/// exist?" — after a server dies its PID can be recycled by an unrelated
/// process, which [`is_alive`] would misreport as a running server. This
/// additionally cross-checks the live process's start time against the
/// record's `started_at` (both UNIX seconds); a mismatch beyond
/// [`PID_REUSE_TOLERANCE_SECS`] means the PID was reused ⇒ not our server.
/// When `started_at` can't be parsed we fall back to the PID-only check, so a
/// legacy / foreign stamp never causes a false "dead".
pub fn is_record_alive(rec: &ServerRecord) -> bool {
    let mut s = sysinfo::System::new();
    s.refresh_processes(sysinfo::ProcessesToUpdate::All);
    let Some(proc_) = s.process(sysinfo::Pid::from_u32(rec.pid)) else {
        return false;
    };
    match parse_started_at_epoch(&rec.started_at) {
        Some(recorded) => proc_.start_time().abs_diff(recorded) <= PID_REUSE_TOLERANCE_SECS,
        None => true,
    }
}

/// Snapshot of the host's RAM situation in bytes. Refreshes memory
/// info via sysinfo on every call — typical cost is a few syscalls,
/// well under a millisecond on Windows / Linux. Used by the server's
/// load endpoint to refuse model loads that would push the host past
/// its RAM budget, and surfaced on the GUI Status page.
///
/// `available_bytes` / `total_bytes` track **physical RAM**. The
/// `commit_*` fields track the OS's commit budget — physical RAM
/// plus swap / page-file space the kernel can reserve for
/// allocations. On Windows that's `GlobalMemoryStatusEx` reporting
/// `ullAvailPageFile` / `ullTotalPageFile`. On Linux it's
/// `MemAvailable + SwapFree` / `MemTotal + SwapTotal` from
/// `/proc/meminfo`. The model-load pre-flight checks commit, not
/// physical-available — a host with a generous page file can load
/// a model that won't fit in physical RAM (the OS pages cold pages
/// out; GPU-resident weights stay hot in USM).
pub struct MemoryInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub used_bytes: u64,
    pub commit_total_bytes: u64,
    pub commit_available_bytes: u64,
}

pub fn memory_info() -> MemoryInfo {
    let mut s = sysinfo::System::new();
    s.refresh_memory();
    let total = s.total_memory();
    let avail = s.available_memory();
    let (commit_total, commit_available) = commit_budget(total, avail, &s);
    MemoryInfo {
        total_bytes: total,
        available_bytes: avail,
        used_bytes: total.saturating_sub(avail),
        commit_total_bytes: commit_total,
        commit_available_bytes: commit_available,
    }
}

/// System-wide CPU utilization (%) plus the CPU brand string, for the
/// GUI status bar (CPU-over-RAM paired row). sysinfo computes CPU% as a
/// delta between two refreshes, so a `System` is persisted behind a
/// `OnceLock<Mutex<..>>` and refreshed once per call; the metrics
/// endpoint polls ~every 2 s, well above sysinfo's ~200 ms minimum CPU
/// refresh interval, so each poll yields an accurate figure. The very
/// first call after startup reports ~0 % (no prior sample yet).
pub struct CpuInfo {
    pub utilization_pct: f32,
    pub brand: String,
}

pub fn cpu_info() -> CpuInfo {
    use std::sync::{Mutex, OnceLock};
    static SYS: OnceLock<Mutex<sysinfo::System>> = OnceLock::new();
    let sys = SYS.get_or_init(|| Mutex::new(sysinfo::System::new()));
    let mut s = sys.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    // Refreshes usage (delta since last call) + populates the CPU list
    // so `brand()` is available.
    s.refresh_cpu_all();
    let utilization_pct = s.global_cpu_usage();
    let brand = s
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .filter(|b| !b.is_empty())
        .unwrap_or_else(|| "CPU".to_string());
    CpuInfo {
        utilization_pct,
        brand,
    }
}

/// Aggregate GPU engine utilization % (0..=100) via the Windows PDH
/// "GPU Engine" performance counters — the compute + 3D engine
/// instances summed. Works on the integrated Iris Xe (whose Level-Zero
/// Sysman exposes no engine-activity groups) and covers NVIDIA/AMD on
/// Windows too. For a single-GPU host this is that GPU's utilization;
/// multi-GPU per-adapter attribution (via the counter's LUID) is a
/// follow-up. Blocks ~120 ms for the two-sample rate. `None` on any PDH
/// failure. Non-Windows always returns `None`.
#[cfg(windows)]
pub fn gpu_busy_pct() -> Option<f32> {
    use windows_sys::Win32::System::Performance::{
        PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData,
        PdhGetFormattedCounterArrayW, PdhOpenQueryW, PDH_FMT_COUNTERVALUE_ITEM_W,
        PDH_FMT_DOUBLE,
    };
    const PDH_MORE_DATA: u32 = 0x800007D2;
    const PDH_CSTATUS_VALID_DATA: u32 = 0;
    let path: Vec<u16> = "\\GPU Engine(*)\\Utilization Percentage\0"
        .encode_utf16()
        .collect();
    // SAFETY: standard PDH open → add → collect×2 → array → close
    // sequence; every fallible step is checked and closes the query on
    // failure. The formatted-array buffer is sized by the first call.
    unsafe {
        let mut query: isize = 0;
        if PdhOpenQueryW(std::ptr::null(), 0, &mut query) != 0 {
            return None;
        }
        let mut counter: isize = 0;
        if PdhAddEnglishCounterW(query, path.as_ptr(), 0, &mut counter) != 0 {
            PdhCloseQuery(query);
            return None;
        }
        // Utilization is a rate — it needs two samples spaced in time.
        if PdhCollectQueryData(query) != 0 {
            PdhCloseQuery(query);
            return None;
        }
        std::thread::sleep(std::time::Duration::from_millis(120));
        if PdhCollectQueryData(query) != 0 {
            PdhCloseQuery(query);
            return None;
        }
        // First array call queries the required buffer size.
        let mut buf_size: u32 = 0;
        let mut item_count: u32 = 0;
        let rc = PdhGetFormattedCounterArrayW(
            counter,
            PDH_FMT_DOUBLE,
            &mut buf_size,
            &mut item_count,
            std::ptr::null_mut(),
        );
        if rc != PDH_MORE_DATA || buf_size == 0 {
            PdhCloseQuery(query);
            return None;
        }
        let mut buf = vec![0u8; buf_size as usize];
        let rc = PdhGetFormattedCounterArrayW(
            counter,
            PDH_FMT_DOUBLE,
            &mut buf_size,
            &mut item_count,
            buf.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W,
        );
        if rc != 0 {
            PdhCloseQuery(query);
            return None;
        }
        let items = std::slice::from_raw_parts(
            buf.as_ptr() as *const PDH_FMT_COUNTERVALUE_ITEM_W,
            item_count as usize,
        );
        let mut sum = 0.0f64;
        for it in items {
            if it.FmtValue.CStatus != PDH_CSTATUS_VALID_DATA {
                continue;
            }
            let name = wide_ptr_to_string(it.szName);
            // Compute + 3D engines approximate "GPU busy with
            // compute/render work"; skip copy/video/etc.
            if name.contains("engtype_3D") || name.contains("engtype_Compute") {
                sum += it.FmtValue.Anonymous.doubleValue;
            }
        }
        PdhCloseQuery(query);
        Some(sum.clamp(0.0, 100.0) as f32)
    }
}

#[cfg(windows)]
unsafe fn wide_ptr_to_string(p: *const u16) -> String {
    if p.is_null() {
        return String::new();
    }
    let mut len = 0usize;
    while *p.add(len) != 0 {
        len += 1;
    }
    String::from_utf16_lossy(std::slice::from_raw_parts(p, len))
}

#[cfg(not(windows))]
pub fn gpu_busy_pct() -> Option<f32> {
    None
}

/// Windows PDH fallback for **GPU memory in use** (bytes) — the "GPU
/// Adapter Memory" perf counters, summing `Dedicated Usage` + `Shared
/// Usage` across adapter instances. On the integrated Iris Xe (whose
/// Level-Zero Sysman exposes no memory module, so `zesMemoryGetState`
/// yields nothing) Dedicated ≈ 0 and Shared carries the real usage; on a
/// dGPU Dedicated is the VRAM figure. Together they match Task Manager's
/// GPU memory. These are instantaneous gauges, so a single collect
/// suffices (unlike the rate-based utilization counter). For a single-GPU
/// host this is that GPU's memory; multi-GPU per-adapter attribution (via
/// the counter's LUID) is a follow-up. `None` on any PDH failure.
/// Non-Windows always returns `None`.
#[cfg(windows)]
pub fn gpu_mem_used_bytes() -> Option<u64> {
    use windows_sys::Win32::System::Performance::{
        PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData,
        PdhGetFormattedCounterArrayW, PdhOpenQueryW, PDH_FMT_COUNTERVALUE_ITEM_W,
        PDH_FMT_LARGE,
    };
    const PDH_MORE_DATA: u32 = 0x800007D2;
    const PDH_CSTATUS_VALID_DATA: u32 = 0;
    let paths = [
        "\\GPU Adapter Memory(*)\\Dedicated Usage\0",
        "\\GPU Adapter Memory(*)\\Shared Usage\0",
    ];
    // SAFETY: standard PDH open → add → collect → array → close sequence;
    // every fallible step is checked and the query is closed on any exit.
    // The formatted-array buffer is sized by the first (PDH_MORE_DATA) call.
    unsafe {
        let mut query: isize = 0;
        if PdhOpenQueryW(std::ptr::null(), 0, &mut query) != 0 {
            return None;
        }
        let mut counters = [0isize; 2];
        for (i, p) in paths.iter().enumerate() {
            let w: Vec<u16> = p.encode_utf16().collect();
            if PdhAddEnglishCounterW(query, w.as_ptr(), 0, &mut counters[i]) != 0 {
                PdhCloseQuery(query);
                return None;
            }
        }
        // Gauge counters — one sample is enough (no rate to derive).
        if PdhCollectQueryData(query) != 0 {
            PdhCloseQuery(query);
            return None;
        }
        let mut total: u64 = 0;
        let mut any = false;
        for &counter in &counters {
            let mut buf_size: u32 = 0;
            let mut item_count: u32 = 0;
            let rc = PdhGetFormattedCounterArrayW(
                counter,
                PDH_FMT_LARGE,
                &mut buf_size,
                &mut item_count,
                std::ptr::null_mut(),
            );
            if rc != PDH_MORE_DATA || buf_size == 0 {
                continue;
            }
            let mut buf = vec![0u8; buf_size as usize];
            let rc = PdhGetFormattedCounterArrayW(
                counter,
                PDH_FMT_LARGE,
                &mut buf_size,
                &mut item_count,
                buf.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W,
            );
            if rc != 0 {
                continue;
            }
            let items = std::slice::from_raw_parts(
                buf.as_ptr() as *const PDH_FMT_COUNTERVALUE_ITEM_W,
                item_count as usize,
            );
            for it in items {
                if it.FmtValue.CStatus != PDH_CSTATUS_VALID_DATA {
                    continue;
                }
                let v = it.FmtValue.Anonymous.largeValue;
                if v > 0 {
                    total = total.saturating_add(v as u64);
                    any = true;
                }
            }
        }
        PdhCloseQuery(query);
        if any {
            Some(total)
        } else {
            None
        }
    }
}

#[cfg(not(windows))]
pub fn gpu_mem_used_bytes() -> Option<u64> {
    None
}

/// Disk I/O throughput for **this process** as `(read_bytes/s, write_bytes/s)`
/// — surfaced as the "Model I/O" status-bar meter (the server process's file
/// reads are dominated by the GGUF weights during load / cold-page faults;
/// writes are KV/conversation-DB). Delta-based against the previous call, so
/// the rate is averaged over the metrics poll interval (no blocking sample).
/// `None` off Windows/Linux.
#[cfg(windows)]
pub fn process_io_bytes_per_sec() -> Option<(u64, u64)> {
    #[repr(C)]
    struct IoCounters {
        read_ops: u64,
        write_ops: u64,
        other_ops: u64,
        read_bytes: u64,
        write_bytes: u64,
        other_bytes: u64,
    }
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn GetProcessIoCounters(handle: isize, counters: *mut IoCounters) -> i32;
    }
    let mut c: IoCounters = unsafe { std::mem::zeroed() };
    // SAFETY: pseudo-handle from GetCurrentProcess; `c` is live stack storage.
    if unsafe { GetProcessIoCounters(GetCurrentProcess(), &mut c) } == 0 {
        return None;
    }
    static PREV: std::sync::OnceLock<std::sync::Mutex<Option<(u64, u64, std::time::Instant)>>> =
        std::sync::OnceLock::new();
    let cell = PREV.get_or_init(|| std::sync::Mutex::new(None));
    let now = std::time::Instant::now();
    let mut g = cell.lock().ok()?;
    let out = match *g {
        Some((pr, pw, pt)) => {
            let dt = now.duration_since(pt).as_secs_f64();
            if dt > 0.05 {
                (
                    ((c.read_bytes.saturating_sub(pr)) as f64 / dt) as u64,
                    ((c.write_bytes.saturating_sub(pw)) as f64 / dt) as u64,
                )
            } else {
                (0, 0)
            }
        }
        None => (0, 0),
    };
    *g = Some((c.read_bytes, c.write_bytes, now));
    Some(out)
}

/// System paging I/O as `(read_bytes/s, write_bytes/s)` — the "Page/Swap I/O"
/// meter. Read = `\Memory\Pages Input/sec` (hard-fault reads from disk),
/// write = `\Memory\Pages Output/sec` (writes to the pagefile), each × the
/// 4 KiB page size. Delta-based against the raw PDH counters (no blocking
/// sample). `None` off Windows/Linux or on any PDH failure.
#[cfg(windows)]
pub fn paging_io_bytes_per_sec() -> Option<(u64, u64)> {
    use windows_sys::Win32::System::Performance::{
        PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData, PdhGetRawCounterValue,
        PdhOpenQueryW, PDH_RAW_COUNTER,
    };
    const PAGE: u64 = 4096;
    let paths = ["\\Memory\\Pages Input/sec\0", "\\Memory\\Pages Output/sec\0"];
    // SAFETY: standard PDH open → add → collect → raw-read → close; every
    // fallible step is checked and the query is closed before returning.
    let (in_cum, out_cum) = unsafe {
        let mut query: isize = 0;
        if PdhOpenQueryW(std::ptr::null(), 0, &mut query) != 0 {
            return None;
        }
        let mut counters = [0isize; 2];
        for (i, p) in paths.iter().enumerate() {
            let w: Vec<u16> = p.encode_utf16().collect();
            if PdhAddEnglishCounterW(query, w.as_ptr(), 0, &mut counters[i]) != 0 {
                PdhCloseQuery(query);
                return None;
            }
        }
        if PdhCollectQueryData(query) != 0 {
            PdhCloseQuery(query);
            return None;
        }
        let read_raw = |c: isize| -> i64 {
            let mut raw: PDH_RAW_COUNTER = std::mem::zeroed();
            let mut ctype: u32 = 0;
            if PdhGetRawCounterValue(c, &mut ctype, &mut raw) == 0 {
                raw.FirstValue.max(0)
            } else {
                0
            }
        };
        let vals = (read_raw(counters[0]) as u64, read_raw(counters[1]) as u64);
        PdhCloseQuery(query);
        vals
    };
    static PREV: std::sync::OnceLock<std::sync::Mutex<Option<(u64, u64, std::time::Instant)>>> =
        std::sync::OnceLock::new();
    let cell = PREV.get_or_init(|| std::sync::Mutex::new(None));
    let now = std::time::Instant::now();
    let mut g = cell.lock().ok()?;
    let out = match *g {
        Some((pin, pout, pt)) => {
            let dt = now.duration_since(pt).as_secs_f64();
            if dt > 0.05 {
                (
                    ((in_cum.saturating_sub(pin)) as f64 / dt * PAGE as f64) as u64,
                    ((out_cum.saturating_sub(pout)) as f64 / dt * PAGE as f64) as u64,
                )
            } else {
                (0, 0)
            }
        }
        None => (0, 0),
    };
    *g = Some((in_cum, out_cum, now));
    Some(out)
}

/// Linux: this process's disk I/O from `/proc/self/io` (`read_bytes` /
/// `write_bytes`), delta-based.
#[cfg(target_os = "linux")]
pub fn process_io_bytes_per_sec() -> Option<(u64, u64)> {
    let txt = std::fs::read_to_string("/proc/self/io").ok()?;
    let field = |k: &str| -> u64 {
        txt.lines()
            .find_map(|l| l.strip_prefix(k).and_then(|v| v.trim().parse::<u64>().ok()))
            .unwrap_or(0)
    };
    let (r, w) = (field("read_bytes:"), field("write_bytes:"));
    Some(io_delta("proc_self_io", r, w))
}

/// Linux: swap I/O from `/proc/vmstat` (`pswpin` / `pswpout` pages),
/// delta-based × page size.
#[cfg(target_os = "linux")]
pub fn paging_io_bytes_per_sec() -> Option<(u64, u64)> {
    let txt = std::fs::read_to_string("/proc/vmstat").ok()?;
    let field = |k: &str| -> u64 {
        txt.lines()
            .find_map(|l| {
                let mut it = l.split_whitespace();
                (it.next() == Some(k)).then(|| it.next().and_then(|v| v.parse::<u64>().ok()))
                    .flatten()
            })
            .unwrap_or(0)
    };
    let page = 4096u64;
    let (r, w) = (field("pswpin") * page, field("pswpout") * page);
    Some(io_delta("proc_vmstat_swap", r, w))
}

/// Shared Linux delta helper: turns cumulative `(read, write)` byte counters
/// keyed by `slot` into per-second rates against the previous call.
#[cfg(target_os = "linux")]
fn io_delta(slot: &'static str, read_cum: u64, write_cum: u64) -> (u64, u64) {
    use std::collections::HashMap;
    static PREV: std::sync::OnceLock<
        std::sync::Mutex<HashMap<&'static str, (u64, u64, std::time::Instant)>>,
    > = std::sync::OnceLock::new();
    let cell = PREV.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let now = std::time::Instant::now();
    let mut g = match cell.lock() {
        Ok(g) => g,
        Err(_) => return (0, 0),
    };
    let out = match g.get(slot) {
        Some(&(pr, pw, pt)) => {
            let dt = now.duration_since(pt).as_secs_f64();
            if dt > 0.05 {
                (
                    ((read_cum.saturating_sub(pr)) as f64 / dt) as u64,
                    ((write_cum.saturating_sub(pw)) as f64 / dt) as u64,
                )
            } else {
                (0, 0)
            }
        }
        None => (0, 0),
    };
    g.insert(slot, (read_cum, write_cum, now));
    out
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn process_io_bytes_per_sec() -> Option<(u64, u64)> {
    None
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn paging_io_bytes_per_sec() -> Option<(u64, u64)> {
    None
}

/// GPU indices (in the stable RustLlama enumeration order) the user has
/// asked to ignore, parsed from `RUSTLLAMA_DISABLED_GPUS` (comma- or
/// whitespace-separated, e.g. "0,2"). The CLI promotes
/// `[inference].disabled_gpus` into this env at startup, so it is the
/// single source of truth for both the metrics GPU probe and the
/// compute-dispatch device selection. Empty when unset/blank.
pub fn disabled_gpu_indices() -> std::collections::HashSet<u32> {
    std::env::var("RUSTLLAMA_DISABLED_GPUS")
        .ok()
        .map(|s| {
            s.split(|c: char| c == ',' || c.is_whitespace())
                .filter_map(|t| t.trim().parse::<u32>().ok())
                .collect()
        })
        .unwrap_or_default()
}

/// LOGICAL-PROCESSOR indices the user has asked to exclude from the CPU
/// compute pool, parsed from `RUSTLLAMA_DISABLED_CPUS` (comma- or
/// whitespace-separated, e.g. "8,9,10,11"). The exact CPU-tier analog of
/// [`disabled_gpu_indices`]: the CLI promotes `[inference].disabled_cpus`
/// into this env at startup, so it is the single source of truth for both
/// the thread-pool affinity pinning (workers are placed only on enabled
/// procs) and the tuner's CPU-config-aware fingerprint. Empty when
/// unset/blank. Indices are P/E-core-aware — see [`cpu_topology`] for the
/// per-processor P-core vs E-core tags they map to.
pub fn disabled_cpu_indices() -> std::collections::HashSet<u32> {
    std::env::var("RUSTLLAMA_DISABLED_CPUS")
        .ok()
        .map(|s| {
            s.split(|c: char| c == ',' || c.is_whitespace())
                .filter_map(|t| t.trim().parse::<u32>().ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Whether the CPU compute tier is enabled (weights may be placed on the
/// CPU/RAM tier). Reads `RUSTLLAMA_CPU_ENABLED` — `"0"`/`"false"` disables,
/// anything else (or unset) enables. The CLI promotes
/// `[inference].cpu_enabled` / `serve --no-cpu` into this env. When `false`
/// the placement planner refuses a model that doesn't fit GPU-only rather
/// than spilling to the CPU. Mirrors the disable-list env-promotion pattern.
pub fn cpu_tier_enabled() -> bool {
    match std::env::var("RUSTLLAMA_CPU_ENABLED") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false"))
        }
        Err(_) => true,
    }
}

/// Whether the GPU compute tier is enabled (weights may be placed on a GPU
/// tier). Reads `RUSTLLAMA_GPU_ENABLED` — `"0"`/`"false"` disables, anything
/// else (or unset) enables. The CLI promotes `[inference].gpu_enabled` /
/// `serve --no-gpu` into this env. When `false` the placement planner offers
/// no GPU tier and every layer runs on the CPU/RAM tier — the symmetric
/// mirror of [`cpu_tier_enabled`].
pub fn gpu_tier_enabled() -> bool {
    match std::env::var("RUSTLLAMA_GPU_ENABLED") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false"))
        }
        Err(_) => true,
    }
}

/// Whether VRAM-only weight residency was requested. Reads
/// `RUSTLLAMA_VRAM_ONLY` — `"1"`/`"true"` enables. The CLI promotes
/// `[inference].vram_only` / `serve --vram-only` into this env. A no-op on
/// unified-memory GPUs (no separate dedicated VRAM); see the config field.
pub fn vram_only_requested() -> bool {
    match std::env::var("RUSTLLAMA_VRAM_ONLY") {
        Ok(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true")
        }
        Err(_) => false,
    }
}

/// Performance/Efficiency classification of a logical processor. On hybrid
/// Intel parts (e.g. the i9-13900H) the OS reports an efficiency class per
/// physical core; we map the highest class to [`CoreClass::Performance`]
/// (P-core) and the lowest to [`CoreClass::Efficiency`] (E-core). On
/// non-hybrid / non-x86 hosts every processor is [`CoreClass::Unknown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoreClass {
    Performance,
    Efficiency,
    Unknown,
}

/// One logical processor (OS scheduling unit) in the CPU topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalProcessor {
    /// 0-based logical-processor index in OS enumeration order. This is the
    /// index `disabled_cpus` / `RUSTLLAMA_DISABLED_CPUS` refers to and the
    /// index the thread-pool affinity mask pins against.
    pub index: u32,
    /// P-core vs E-core (vs Unknown on non-hybrid / non-x86).
    pub core_class: CoreClass,
}

/// Stable CPU package identity, folded into the tuner fingerprint so a
/// different CPU (or the same CPU with a different enabled-core set)
/// re-tunes. `family`/`model`/`stepping` come from CPUID leaf 1 on x86;
/// `brand` from the CPUID brand-string leaves (or the OS on non-x86).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuPackage {
    pub brand: String,
    pub family: u32,
    pub model: u32,
    pub stepping: u32,
}

/// Enumerated CPU topology: every logical processor tagged P/E/Unknown,
/// plus the package identity. See [`cpu_topology`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuTopology {
    pub logical: Vec<LogicalProcessor>,
    pub package: CpuPackage,
}

/// Enumerate the host CPU topology: each logical processor tagged as a
/// P-core / E-core / Unknown, plus the package identity (brand + CPUID
/// family/model/stepping).
///
/// Detection:
/// - **Windows** — `GetLogicalProcessorInformationEx(RelationProcessorCore)`
///   gives each physical core's `EfficiencyClass` and the group-affinity
///   mask of its logical processors; higher class = P-core on Intel hybrid.
/// - **Linux** — enumerates `/sys/devices/system/cpu/cpu*/` and reads
///   `cpu_capacity` (higher = P-core, ARM big.LITTLE + some Intel hybrid);
///   absent ⇒ Unknown. Brand from `/proc/cpuinfo`.
/// - **Other / non-hybrid / non-x86** — every processor is `Unknown`, still
///   enumerated as `0..available_parallelism`.
///
/// This never fails: on any probe error it falls back to enumerating
/// `0..available_parallelism` as `Unknown`.
pub fn cpu_topology() -> CpuTopology {
    let package = cpu_package_identity();
    let logical = enumerate_logical_processors();
    let logical = if logical.is_empty() {
        // Failure fallback: at least enumerate the count we know about.
        let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        (0..n as u32)
            .map(|index| LogicalProcessor { index, core_class: CoreClass::Unknown })
            .collect()
    } else {
        logical
    };
    CpuTopology { logical, package }
}

/// CPUID-based package identity on x86; OS brand + zeroed IDs elsewhere.
#[cfg(target_arch = "x86_64")]
fn cpu_package_identity() -> CpuPackage {
    // SAFETY: CPUID is always available on x86_64; leaves 1 and the
    // 0x80000002..=4 brand leaves are architectural.
    unsafe {
        use std::arch::x86_64::__cpuid;
        let r = __cpuid(1);
        let eax = r.eax;
        let stepping = eax & 0xF;
        let base_model = (eax >> 4) & 0xF;
        let base_family = (eax >> 8) & 0xF;
        let ext_model = (eax >> 16) & 0xF;
        let ext_family = (eax >> 20) & 0xFF;
        let family = if base_family == 0xF { base_family + ext_family } else { base_family };
        let model = if base_family == 0x6 || base_family == 0xF {
            (ext_model << 4) | base_model
        } else {
            base_model
        };
        // Brand string from the extended leaves, when supported.
        let max_ext = __cpuid(0x8000_0000).eax;
        let mut brand = String::new();
        if max_ext >= 0x8000_0004 {
            let mut bytes = Vec::with_capacity(48);
            for leaf in [0x8000_0002u32, 0x8000_0003, 0x8000_0004] {
                let l = __cpuid(leaf);
                for reg in [l.eax, l.ebx, l.ecx, l.edx] {
                    bytes.extend_from_slice(&reg.to_le_bytes());
                }
            }
            // NUL-terminated ASCII, space-padded.
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            brand = String::from_utf8_lossy(&bytes[..end]).trim().to_string();
        }
        if brand.is_empty() {
            brand = "CPU".to_string();
        }
        CpuPackage { brand, family, model, stepping }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn cpu_package_identity() -> CpuPackage {
    // Non-x86: no CPUID. Take the brand from the OS probe, leave the
    // numeric IDs zeroed (the tuner still keys off brand + enabled cores).
    CpuPackage {
        brand: cpu_info().brand,
        family: 0,
        model: 0,
        stepping: 0,
    }
}

/// Windows logical-processor enumeration via
/// `GetLogicalProcessorInformationEx(RelationProcessorCore)`.
#[cfg(windows)]
fn enumerate_logical_processors() -> Vec<LogicalProcessor> {
    // Raw FFI (consistent with this module's other Win32 externs, e.g.
    // `GetProcessIoCounters`) — the RelationProcessorCore records are
    // variable-length (a flexible `GroupMask` array) and carry a union, so
    // hand-rolled `#[repr(C)]` structs + manual record-walking are cleaner
    // and steadier than the windows-sys typedefs.
    const RELATION_PROCESSOR_CORE: u32 = 0;
    const ERROR_INSUFFICIENT_BUFFER: u32 = 122;

    #[repr(C)]
    struct GroupAffinity {
        mask: usize,
        group: u16,
        reserved: [u16; 3],
    }
    #[repr(C)]
    struct ProcessorRelationship {
        flags: u8,
        efficiency_class: u8,
        reserved: [u8; 20],
        group_count: u16,
        // ANYSIZE_ARRAY: `group_count` entries follow.
        group_mask: [GroupAffinity; 1],
    }
    #[repr(C)]
    struct InfoHeader {
        relationship: u32,
        size: u32,
    }

    extern "system" {
        fn GetLogicalProcessorInformationEx(
            relationship: u32,
            buffer: *mut u8,
            returned_length: *mut u32,
        ) -> i32;
        fn GetLastError() -> u32;
    }

    // SAFETY: standard two-call size-then-fill pattern. The first call with
    // a null buffer returns FALSE + ERROR_INSUFFICIENT_BUFFER and sets the
    // required length; the second fills a correctly-sized buffer. We only
    // read fields within each record's own `size` span.
    unsafe {
        let mut len: u32 = 0;
        let rc = GetLogicalProcessorInformationEx(
            RELATION_PROCESSOR_CORE,
            std::ptr::null_mut(),
            &mut len,
        );
        if rc != 0 || GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0 {
            return Vec::new();
        }
        let mut buf = vec![0u8; len as usize];
        let rc = GetLogicalProcessorInformationEx(
            RELATION_PROCESSOR_CORE,
            buf.as_mut_ptr(),
            &mut len,
        );
        if rc == 0 {
            return Vec::new();
        }

        // Walk the variable-length records, collecting (index, efficiency).
        let mut cores: Vec<(u32, u8)> = Vec::new();
        let mut off = 0usize;
        let total = len as usize;
        while off + std::mem::size_of::<InfoHeader>() <= total {
            let hdr = &*(buf.as_ptr().add(off) as *const InfoHeader);
            let rec_size = hdr.size as usize;
            if rec_size == 0 || off + rec_size > total {
                break;
            }
            if hdr.relationship == RELATION_PROCESSOR_CORE {
                let pr = &*(buf.as_ptr().add(off + std::mem::size_of::<InfoHeader>())
                    as *const ProcessorRelationship);
                let eff = pr.efficiency_class;
                let gc = pr.group_count as usize;
                let masks = std::slice::from_raw_parts(pr.group_mask.as_ptr(), gc.max(1));
                for ga in masks.iter().take(gc) {
                    let base = (ga.group as u32) * 64;
                    let mut m = ga.mask;
                    let mut bit = 0u32;
                    while m != 0 {
                        if m & 1 != 0 {
                            cores.push((base + bit, eff));
                        }
                        m >>= 1;
                        bit += 1;
                    }
                }
            }
            off += rec_size;
        }
        classify(cores)
    }
}

/// Linux logical-processor enumeration via sysfs.
#[cfg(target_os = "linux")]
fn enumerate_logical_processors() -> Vec<LogicalProcessor> {
    let base = std::path::Path::new("/sys/devices/system/cpu");
    let Ok(rd) = std::fs::read_dir(base) else {
        return Vec::new();
    };
    // First pass: collect each CPU's raw capacity + core_type hint. We can't
    // bucket until the peak capacity is known (see below), so defer the
    // P/E classification.
    let mut raw: Vec<(u32, Option<u32>, Option<u8>)> = Vec::new();
    for entry in rd.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // Match "cpuN" (not "cpufreq", "cpuidle", ...).
        let Some(rest) = name.strip_prefix("cpu") else { continue };
        let Ok(index) = rest.parse::<u32>() else { continue };
        // `cpu_capacity` (higher = more capable = P-core). Absent on plain
        // symmetric x86; falls through to Unknown via the single-class rule.
        let cap = std::fs::read_to_string(entry.path().join("cpu_capacity"))
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok());
        // Some Intel hybrid kernels expose `topology/core_type`
        // ("intel_core"=P > "intel_atom"=E); reduce it to the same rank the
        // shared classifier buckets.
        let ct_rank = std::fs::read_to_string(entry.path().join("topology/core_type"))
            .ok()
            .map(|s| {
                let ct = s.trim();
                if ct.contains("core") {
                    2u8
                } else if ct.contains("atom") {
                    1
                } else {
                    0
                }
            });
        raw.push((index, cap, ct_rank));
    }

    // Peak reported capacity across all cores. ARM DynamIQ reports
    // `cpu_capacity` on a scale whose max is ~1024 (e.g. big=1024,
    // LITTLE=~512), so the OLD absolute `.min(255) as u8` clamp collapsed BOTH
    // clusters onto 255 — a single distinct rank ⇒ every core misclassified as
    // Unknown. Bucket by capacity RELATIVE to the peak instead, so big.LITTLE
    // (and 3-tier DynamIQ) split correctly at any scale: a core within
    // `PERF_CAPACITY_PERCENT`% of the peak is Performance-class, the rest are
    // Efficiency-class. Cores with no `cpu_capacity` fall back to the
    // core_type hint (or Unknown). A homogeneous set collapses to one rank and
    // the shared classifier tags every core Unknown, as before.
    const PERF_CAPACITY_PERCENT: u64 = 60;
    let max_cap = raw.iter().filter_map(|(_, c, _)| *c).max().unwrap_or(0);
    let cores: Vec<(u32, u8)> = raw
        .into_iter()
        .map(|(index, cap, ct_rank)| {
            let rank: u8 = match cap {
                Some(c) if max_cap > 0 => {
                    if u64::from(c) * 100 >= u64::from(max_cap) * PERF_CAPACITY_PERCENT {
                        2
                    } else {
                        1
                    }
                }
                _ => ct_rank.unwrap_or(0),
            };
            (index, rank)
        })
        .collect();
    classify(cores)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn enumerate_logical_processors() -> Vec<LogicalProcessor> {
    Vec::new()
}

/// Turn `(index, efficiency_rank)` pairs into classified logical
/// processors. When fewer than two distinct ranks are present the host is
/// non-hybrid (or the OS didn't report classes) → every processor is
/// `Unknown`. Otherwise the max rank is `Performance`, the min is
/// `Efficiency`, anything between is `Unknown`.
#[cfg(any(windows, target_os = "linux"))]
fn classify(mut cores: Vec<(u32, u8)>) -> Vec<LogicalProcessor> {
    cores.sort_by_key(|(i, _)| *i);
    cores.dedup_by_key(|(i, _)| *i);
    let mut ranks: Vec<u8> = cores.iter().map(|(_, e)| *e).collect();
    ranks.sort_unstable();
    ranks.dedup();
    let hybrid = ranks.len() >= 2;
    let (min_r, max_r) = (ranks.first().copied().unwrap_or(0), ranks.last().copied().unwrap_or(0));
    cores
        .into_iter()
        .map(|(index, eff)| {
            let core_class = if !hybrid {
                CoreClass::Unknown
            } else if eff == max_r {
                CoreClass::Performance
            } else if eff == min_r {
                CoreClass::Efficiency
            } else {
                CoreClass::Unknown
            };
            LogicalProcessor { index, core_class }
        })
        .collect()
}

/// Resolve the OS commit budget (physical RAM + page-file / swap).
/// Falls back to `(phys_total, phys_avail)` if the platform-specific
/// probe fails — that yields the old "physical only" behavior, so
/// the pre-flight check stays at least as permissive as before.
#[cfg(windows)]
fn commit_budget(_phys_total: u64, _phys_avail: u64, _s: &sysinfo::System) -> (u64, u64) {
    use windows_sys::Win32::System::SystemInformation::{
        GlobalMemoryStatusEx, MEMORYSTATUSEX,
    };
    let mut ms: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    ms.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: `ms` is a zeroed struct with the documented length
    // field set; `GlobalMemoryStatusEx` is a read-only kernel call.
    let ok = unsafe { GlobalMemoryStatusEx(&mut ms) } != 0;
    if ok {
        (ms.ullTotalPageFile, ms.ullAvailPageFile)
    } else {
        (_phys_total, _phys_avail)
    }
}

#[cfg(not(windows))]
fn commit_budget(phys_total: u64, phys_avail: u64, s: &sysinfo::System) -> (u64, u64) {
    let swap_total = s.total_swap();
    let swap_free = s.free_swap();
    (
        phys_total.saturating_add(swap_total),
        phys_avail.saturating_add(swap_free),
    )
}

/// Resident memory of the current process in bytes. On Windows this
/// is the working-set size (what Task Manager's "Memory (active
/// private working set)" column reports); on Linux it's the
/// "Resident Set Size" from `/proc/self/status`. Useful for sizing
/// model loads against RAM budgets and for answering "did my model
/// actually free its memory after unload?".
///
/// Returns `None` on hosts where `sysinfo` can't refresh the
/// current process (rare — typically requires sandbox restrictions
/// blocking the process-info probe).
pub fn process_rss_bytes() -> Option<u64> {
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};
    let pid = Pid::from_u32(std::process::id());
    let mut s = System::new();
    s.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        ProcessRefreshKind::new().with_memory(),
    );
    s.process(pid).map(|p| p.memory())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes tests that mutate process-global env vars (the path-mode
    /// detection reads `RUSTLLAMA_SYSTEM_DIRS`). cargo runs unit tests in one
    /// process across many threads, so without this two env-mutating tests can
    /// interleave and read each other's state.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn record_path_resolves() {
        let p = record_path();
        assert!(p.is_ok());
    }

    #[test]
    fn memory_info_populates_commit_fields() {
        let m = memory_info();
        // Every host has at least *some* commit budget — total > 0.
        assert!(m.total_bytes > 0);
        assert!(m.commit_total_bytes > 0);
        // Commit total is always >= physical total (RAM + page-file /
        // swap >= RAM). Don't assert strict > because a host with no
        // swap configured is legal and the two will be equal.
        assert!(m.commit_total_bytes >= m.total_bytes,
            "commit_total ({}) should be >= phys_total ({})",
            m.commit_total_bytes, m.total_bytes);
        assert!(m.commit_available_bytes >= m.available_bytes,
            "commit_avail ({}) should be >= phys_avail ({})",
            m.commit_available_bytes, m.available_bytes);
        // Available never exceeds total.
        assert!(m.available_bytes <= m.total_bytes);
        assert!(m.commit_available_bytes <= m.commit_total_bytes);
    }

    #[test]
    fn current_pid_is_alive() {
        let me = std::process::id();
        assert!(is_alive(me));
    }

    /// CURRENT default = binary-relative. With no `RUSTLLAMA_SYSTEM_DIRS` and
    /// no `system.flag`, all state roots beside the exe: `config.toml` *is* the
    /// exe dir, and `models/` `sessions/` `tuning/` `runtime/` `logs/` are
    /// subdirectories. We exercise `resolve_paths()` directly (not the cached
    /// `paths()`) so an earlier test's OnceLock init doesn't leak in.
    #[test]
    fn default_routes_paths_under_exe_dir() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("RUSTLLAMA_SYSTEM_DIRS");
        let p = resolve_paths();
        // In a normal test run the exe dir resolves; if it somehow doesn't,
        // `resolve_paths` falls back to OS dirs and there is nothing to assert.
        if let Some(exe_dir) = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        {
            assert!(p.portable, "expected portable=true by default (binary-relative)");
            assert_eq!(p.config_dir, exe_dir, "config.toml sits beside the exe");
            assert_eq!(p.cache_dir, exe_dir.join("models"));
            assert_eq!(p.sessions_dir, exe_dir.join("sessions"));
            assert_eq!(p.tuning_dir, exe_dir.join("tuning"));
            assert_eq!(p.runtime_dir, exe_dir.join("runtime"));
            assert_eq!(p.crash_log_dir, exe_dir.join("logs"));
        }
    }

    /// Opt out to the OS standard dirs via `RUSTLLAMA_SYSTEM_DIRS`.
    #[test]
    fn system_dirs_env_routes_paths_under_os_dirs() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("RUSTLLAMA_SYSTEM_DIRS", "1");
        let p = resolve_paths();
        std::env::remove_var("RUSTLLAMA_SYSTEM_DIRS");
        assert!(!p.portable, "expected portable=false with RUSTLLAMA_SYSTEM_DIRS set");
        // System paths carry "rustllama" somewhere in the chain.
        assert!(p.config_dir.to_string_lossy().contains("rustllama"));
        assert!(p.cache_dir.to_string_lossy().contains("rustllama"));
    }

    #[test]
    fn default_layout_subdirs_are_distinct() {
        // The six roots must be distinct so a `rm -rf` of one doesn't nuke
        // another. A regression where (e.g.) sessions and tuning collide would
        // silently corrupt state.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("RUSTLLAMA_SYSTEM_DIRS");
        let p = resolve_paths();
        let dirs = [
            &p.config_dir,
            &p.cache_dir,
            &p.sessions_dir,
            &p.tuning_dir,
            &p.runtime_dir,
            &p.crash_log_dir,
        ];
        for i in 0..dirs.len() {
            for j in (i + 1)..dirs.len() {
                assert_ne!(dirs[i], dirs[j], "dirs[{i}] collides with dirs[{j}]");
            }
        }
    }

    #[test]
    fn server_lock_is_exclusive() {
        // Acquire the single-instance lock; a second attempt while it's held
        // must report contention (Ok(None)), and it must be re-acquirable once
        // the guard drops.
        let first = acquire_server_lock().expect("acquire server lock");
        assert!(first.is_some(), "first acquisition should succeed");
        let second = acquire_server_lock().expect("second attempt should not error");
        assert!(second.is_none(), "second acquisition should observe the held lock");
        drop(first);
        let third = acquire_server_lock().expect("re-acquire after release");
        assert!(third.is_some(), "lock should be free again after the guard drops");
    }

    fn test_record(pid: u32, started_at: String) -> ServerRecord {
        ServerRecord {
            pid,
            port: 0,
            bind_addr: String::new(),
            started_at,
            model_id: None,
            version: String::new(),
            owner: "test".into(),
        }
    }

    #[test]
    fn is_record_alive_matches_current_process() {
        // Stamp the record with THIS process's real start time so the
        // start-time cross-check lines up regardless of how long the suite
        // runs (using "now" would false-fail on a slow suite once uptime
        // exceeds the tolerance).
        let mut s = sysinfo::System::new();
        s.refresh_processes(sysinfo::ProcessesToUpdate::All);
        let start = s
            .process(sysinfo::Pid::from_u32(std::process::id()))
            .map(|p| p.start_time())
            .expect("current process visible to sysinfo");
        let rec = test_record(std::process::id(), format!("epoch-{start}"));
        assert!(is_record_alive(&rec));
    }

    #[test]
    fn is_record_alive_rejects_pid_reuse_by_start_time() {
        // Same live PID, but a start-time stamp far in the past ⇒ the PID must
        // be judged reused (record stale).
        let rec = test_record(std::process::id(), "epoch-1000000000".to_string());
        assert!(
            !is_record_alive(&rec),
            "a start-time far from the live process's should read as not-alive"
        );
    }

    #[test]
    fn is_record_alive_unparseable_stamp_falls_back_to_pid() {
        // A legacy / foreign stamp can't be parsed → PID-only liveness, never a
        // false "dead" for a live PID.
        let rec = test_record(std::process::id(), "2026-09-28T12:00:00Z".to_string());
        assert!(is_record_alive(&rec));
    }
}
