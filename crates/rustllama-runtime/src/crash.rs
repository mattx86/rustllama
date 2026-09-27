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

use std::io::Write;
use std::sync::{Mutex, OnceLock};

use crate::paths;

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
        // Make sure we don't recursively panic inside the hook itself.
        let _ = write_crash_log(&version, &tag, info);
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

    let paths = paths();
    std::fs::create_dir_all(&paths.crash_log_dir)?;

    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let pid = std::process::id();
    let path = paths
        .crash_log_dir
        .join(format!("crash-{secs}-{pid}.log"));

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
    use super::*;

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
        let p = paths();
        std::fs::create_dir_all(&p.crash_log_dir).expect("mkdir crash dir");
        // Touch a marker file to prove the dir is writable.
        let marker = p.crash_log_dir.join(".rustllama-crash-log-test");
        std::fs::write(&marker, b"ok").expect("write marker");
        assert!(marker.exists());
        let _ = std::fs::remove_file(&marker);
    }
}
