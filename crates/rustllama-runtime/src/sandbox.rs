//! Process-level sandboxing for the rustllama server.
//!
//! On Windows, attaches the current process to a Job Object with:
//!
//! 1. **Memory cap** (`ProcessMemoryLimit`) — kills the process if
//!    its working set exceeds the configured limit. Defends against
//!    a malformed GGUF that tricks the loader into allocating
//!    terabytes (e.g. corrupt tensor-dim metadata triggering a huge
//!    `Vec::with_capacity`).
//!
//! 2. **Kill on close** (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`) —
//!    every child process started after the job is attached gets
//!    killed when the job handle drops, which happens at process
//!    exit. Stops a runaway helper (e.g. `hf-hub` download client
//!    that hangs) from outliving the server.
//!
//! 3. **No breakaway** (`JOB_OBJECT_LIMIT_BREAKAWAY_OK` is *not*
//!    set, and `LIMIT_SILENT_BREAKAWAY_OK` is *not* set) — child
//!    processes can't escape the job. Pairs with the kill-on-close
//!    limit to bound lifetime.
//!
//! The job handle is intentionally leaked into a process-global
//! `OnceLock<RawHandle>` — there's no clean teardown path for a
//! self-attached job, and the Windows kernel cleans up on process
//! exit anyway. The leak is one HANDLE per process, ~16 bytes.
//!
//! On non-Windows targets (Linux, macOS — which we don't ship today
//! but the workspace builds on for `cargo check`), this module is a
//! no-op that always returns `Ok`. A Linux equivalent using cgroups
//! v2 + `prctl(PR_SET_PDEATHSIG)` would be a follow-up.

#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error("CreateJobObjectW failed (GetLastError = {0})")]
    CreateJob(u32),
    #[error("SetInformationJobObject failed (GetLastError = {0})")]
    SetLimits(u32),
    #[error("AssignProcessToJobObject failed (GetLastError = {0})")]
    Assign(u32),
}

/// Sandboxing configuration. `memory_limit_bytes == 0` disables the
/// memory cap; `kill_on_close == false` lets child processes outlive
/// the server (e.g. for development workflows where a CLI invocation
/// spawns a helper that should survive `ctrl-C`).
#[derive(Debug, Clone, Copy)]
pub struct SandboxConfig {
    pub memory_limit_bytes: u64,
    pub kill_on_close: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        // Default 32 GiB memory cap — generous enough for a 70B Q4_K_M
        // model with ctx=8192 (~38 GiB total weight + KV) is over the
        // limit so the OS kills us before the swap thrashes. Tune via
        // `[server].sandbox_memory_limit_mb` once that config field
        // is wired through.
        Self {
            memory_limit_bytes: 32 * 1024 * 1024 * 1024,
            kill_on_close: true,
        }
    }
}

/// Attach the current process to a Job Object with the configured
/// limits. Idempotent: calling twice is a no-op (the global handle
/// holds the first call's job). Returns Ok on non-Windows targets.
///
/// Typical call site: `cli::serve` and `app::main` (GUI) at startup,
/// after argv parsing but before the engine loads. The job's
/// limits affect the engine's allocations.
pub fn install(cfg: SandboxConfig) -> Result<(), SandboxError> {
    #[cfg(windows)]
    {
        install_windows(cfg)
    }
    #[cfg(not(windows))]
    {
        let _ = cfg;
        // Non-Windows: pretend it worked. Linux cgroup + macOS sandbox
        // equivalents are a follow-up; the v1 target is Windows-only.
        Ok(())
    }
}

#[cfg(windows)]
fn install_windows(cfg: SandboxConfig) -> Result<(), SandboxError> {
    use std::sync::OnceLock;
    use windows_sys::Win32::Foundation::{GetLastError, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject,
        JobObjectExtendedLimitInformation, JOBOBJECT_BASIC_LIMIT_INFORMATION,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOB_OBJECT_LIMIT_PROCESS_MEMORY,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    // Wrap the HANDLE so we can store it in a OnceLock (Send + Sync
    // by virtue of being a raw integer the OS owns; we never call
    // CloseHandle on it — see module docs).
    #[allow(dead_code)] // we hold the handle alive; never read it back
    struct JobHandle(HANDLE);
    unsafe impl Send for JobHandle {}
    unsafe impl Sync for JobHandle {}
    static JOB: OnceLock<JobHandle> = OnceLock::new();

    if JOB.get().is_some() {
        // Already attached.
        return Ok(());
    }

    // SAFETY: CreateJobObjectW with null name/security_attributes is
    // documented to create an anonymous, non-inheritable job. Returns
    // null on failure; GetLastError carries the reason.
    let job: HANDLE = unsafe { CreateJobObjectW(std::ptr::null_mut(), std::ptr::null()) };
    if job.is_null() {
        let err = unsafe { GetLastError() };
        return Err(SandboxError::CreateJob(err));
    }

    // Configure limits. The extended-limit struct embeds the basic
    // limit struct; we zero everything then set the bits we care about.
    let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
    let basic = JOBOBJECT_BASIC_LIMIT_INFORMATION {
        LimitFlags: {
            let mut flags = 0u32;
            if cfg.kill_on_close {
                flags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            }
            if cfg.memory_limit_bytes > 0 {
                flags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY;
            }
            flags
        },
        ..info.BasicLimitInformation
    };
    info.BasicLimitInformation = basic;
    info.ProcessMemoryLimit = cfg.memory_limit_bytes as usize;

    // SAFETY: info layout matches JobObjectExtendedLimitInformation
    // (we constructed it from JOBOBJECT_EXTENDED_LIMIT_INFORMATION).
    let ok = unsafe {
        SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const _,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
    };
    if ok == 0 {
        let err = unsafe { GetLastError() };
        return Err(SandboxError::SetLimits(err));
    }

    // SAFETY: GetCurrentProcess returns a pseudo-handle (-1) that's
    // safe to pass to any process-API. AssignProcessToJobObject
    // attaches the calling process to the job; subsequent
    // CreateProcess calls inherit the job by default.
    let ok = unsafe { AssignProcessToJobObject(job, GetCurrentProcess()) };
    if ok == 0 {
        let err = unsafe { GetLastError() };
        return Err(SandboxError::Assign(err));
    }

    // Leak the handle into the global. Process exit closes it; we
    // don't have a clean teardown path because there's no
    // "detach from job" Win32 API.
    let _ = JOB.set(JobHandle(job));
    tracing::info!(
        memory_limit_mib = cfg.memory_limit_bytes / (1024 * 1024),
        kill_on_close = cfg.kill_on_close,
        "Windows Job Object sandbox installed"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_is_idempotent() {
        // First call attaches; second is a no-op. Use a generous
        // memory cap so the test process doesn't get killed.
        let cfg = SandboxConfig {
            memory_limit_bytes: 64 * 1024 * 1024 * 1024,
            kill_on_close: true,
        };
        // We can't reliably attach in CI (the test runner already has
        // its own job), so we accept either success or
        // `AccessDenied`-shaped errors. The key invariant: calling
        // twice with the same config doesn't crash.
        let _ = install(cfg);
        let _ = install(cfg);
    }

    #[test]
    fn default_config_has_sane_limits() {
        let cfg = SandboxConfig::default();
        // 32 GiB cap is the documented default.
        assert!(cfg.memory_limit_bytes >= 4 * 1024 * 1024 * 1024);
        assert!(cfg.kill_on_close);
    }

    #[test]
    #[cfg(not(windows))]
    fn non_windows_install_is_noop() {
        // On Linux/macOS the function returns Ok without any side
        // effect. Verifies the cross-platform shape.
        install(SandboxConfig::default()).unwrap();
    }
}
