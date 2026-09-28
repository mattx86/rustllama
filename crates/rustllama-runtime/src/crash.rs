//! Panic hook + crash log. Best-effort: a panic mid-shutdown will at
//! least leave a file behind that an operator can grep for in a post-
//! mortem. The hook captures:
//!
//!   - panic message
//!   - panic location (file:line)
//!   - backtrace (when `RUST_BACKTRACE` is set; standard Rust behavior)
//!   - PID, version, timestamp, command line
//!
//! Writes to `<paths.crash_log_dir>/crash-<unix-secs>-<pid>.log`. One
//! file per panic so a panicking process doesn't overwrite the
//! evidence from an earlier one. After writing, the original (default
//! or RUST_LOG-configured) panic hook still runs so stderr behavior is
//! preserved.

use std::cell::Cell;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

thread_local! {
    /// Set while this thread is executing the crash-log panic hook. Lets the
    /// hook detect (and refuse) re-entry on the same thread — see the hook.
    static IN_CRASH_HOOK: Cell<bool> = const { Cell::new(false) };
}

/// Install the panic hook. Idempotent — only the first call sticks.
/// Pass the rustllama version + a tag (e.g. "serve", "chat", "lsp") so
/// the crash log identifies which subcommand was running.
pub fn install_panic_hook(version: &str, tag: &str) {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    if INSTALLED.set(()).is_err() {
        return;
    }
    let version = version.to_string();
    let tag = tag.to_string();
    let prior = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Re-entrancy guard. `write_crash_log` touches the paths OnceLock; if
        // THIS panic fired while that OnceLock was still initializing on this
        // thread (i.e. the first `paths()` call is on the stack and
        // `resolve_paths` panicked), re-entering it would panic *inside* the
        // hook — which the panic runtime turns into an immediate abort with no
        // crash log at all. If we find ourselves re-entered on the same
        // thread, skip the paths-dependent work and just leave a stderr
        // breadcrumb before deferring to the prior hook. (`write_crash_log`
        // itself also avoids re-initializing the OnceLock; this flag is the
        // belt to that suspenders.)
        if IN_CRASH_HOOK.with(|f| f.replace(true)) {
            let _ = writeln!(
                std::io::stderr(),
                "rustllama: panic while handling a panic; crash log skipped"
            );
            prior(info);
            return;
        }
        let _ = write_crash_log(&version, &tag, info);
        IN_CRASH_HOOK.with(|f| f.set(false));
        prior(info);
    }));
}

fn write_crash_log(
    version: &str,
    tag: &str,
    info: &std::panic::PanicHookInfo<'_>,
) -> std::io::Result<()> {
    // Per-process write lock so concurrent panics on different threads
    // don't interleave bytes into the same file.
    static WRITE_LOCK: Mutex<()> = Mutex::new(());
    let _g = WRITE_LOCK.lock().ok();

    // Resolve the crash-log dir WITHOUT initializing the paths OnceLock. If a
    // panic fired inside the very first `paths()` call, `get_or_init` is still
    // on this thread's stack; calling `paths()` again here would reenter the
    // OnceLock and panic — a panic *inside* the panic hook, which the runtime
    // turns into an abort with no log. Use the already-resolved value when
    // present; otherwise fall back to the process CWD so we still leave a file.
    let crash_log_dir = crate::PATHS
        .get()
        .map(|p| p.crash_log_dir.clone())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    std::fs::create_dir_all(&crash_log_dir)?;

    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let pid = std::process::id();
    let path = crash_log_dir.join(format!("crash-{secs}-{pid}.log"));

    let mut file = std::fs::File::create(&path)?;
    writeln!(file, "rustllama crash log")?;
    writeln!(file, "version:    {version}")?;
    writeln!(file, "subcommand: {tag}")?;
    writeln!(file, "pid:        {pid}")?;
    writeln!(file, "timestamp:  epoch-{secs}")?;
    writeln!(file, "argv:       {:?}", std::env::args().collect::<Vec<_>>())?;
    writeln!(file, "host_os:    {}", std::env::consts::OS)?;
    writeln!(file, "host_arch:  {}", std::env::consts::ARCH)?;
    writeln!(file)?;

    // Panic payload — try the common &'static str / String shapes first.
    let payload = info.payload();
    let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "(non-string panic payload)".to_string()
    };
    writeln!(file, "panic:      {msg}")?;
    if let Some(loc) = info.location() {
        writeln!(file, "location:   {}:{}", loc.file(), loc.line())?;
    }

    // Backtrace — honors RUST_BACKTRACE / RUST_LIB_BACKTRACE.
    let bt = std::backtrace::Backtrace::capture();
    writeln!(file)?;
    writeln!(file, "backtrace:")?;
    writeln!(file, "{bt}")?;

    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn write_crash_log_creates_a_file() {
        // We don't trigger an actual panic in the test (that'd kill the
        // test runner). Instead, use catch_unwind to capture a panic
        // payload, then construct a synthetic PanicHookInfo via the
        // panic hook by re-raising. The easier route: just call
        // `paths()` to make sure the dir resolves, then directly write
        // a synthetic crash log via the helper machinery. Since
        // PanicHookInfo can't be constructed publicly, this test only
        // confirms the crash_log_dir resolves and is writable.
        let p = crate::paths();
        std::fs::create_dir_all(&p.crash_log_dir).expect("mkdir crash dir");
        // Touch a marker file to prove the dir is writable.
        let marker = p.crash_log_dir.join(".rustllama-crash-log-test");
        std::fs::write(&marker, b"ok").expect("write marker");
        assert!(marker.exists());
        let _ = std::fs::remove_file(&marker);
    }
}
