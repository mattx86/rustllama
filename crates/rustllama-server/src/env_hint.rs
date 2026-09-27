//! Host-environment awareness for tool / function-calling.
//!
//! When a chat request carries `tools`, the server injects a short,
//! clearly-labeled `[Host environment]` system block describing the
//! SERVER's host — OS family + arch, plus which shells / interpreters
//! are on `PATH` — so the model emits shell/command tool calls in the
//! right dialect (PowerShell on Windows, bash/POSIX on Linux/macOS).
//!
//! The probe runs exactly once (cached in a `OnceLock`) and NEVER spawns
//! a process: shell/tool detection is a pure `PATH` scan that stats
//! candidate executable paths.
//!
//! CAVEAT: this describes the SERVER's host. For remote deployments
//! where tool calls actually execute on a DIFFERENT machine, the
//! operator should disable it via `[server].tool_environment_hint =
//! false` so the hint doesn't mislead the model about where its shell
//! commands will run.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// Server-wide toggle mirroring `[server].tool_environment_hint`.
/// Defaults to `true` so embedded servers and unit/integration tests
/// (which never call the setter) match the config default. `cli::serve`
/// sets it once at startup from the resolved config.
static ENABLED: AtomicBool = AtomicBool::new(true);

/// Mirror `[server].tool_environment_hint` into the process-global
/// toggle. Called once at server startup.
pub fn set_tool_environment_hint_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// Whether the host-environment hint should be injected into tools
/// requests.
pub fn tool_environment_hint_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// The cached `[Host environment]` block. Computed exactly once, then
/// handed out as a `&'static str` on every request.
pub fn host_environment_hint() -> &'static str {
    static HINT: OnceLock<String> = OnceLock::new();
    HINT.get_or_init(build_host_environment_hint).as_str()
}

/// User-facing OS name from `std::env::consts::OS`.
fn os_label() -> &'static str {
    match std::env::consts::OS {
        "windows" => "Windows",
        "linux" => "Linux",
        "macos" => "macOS",
        other => other,
    }
}

fn build_host_environment_hint() -> String {
    let os = os_label();
    let arch = std::env::consts::ARCH;
    let has = path_has_executable;

    let mut lines: Vec<String> = Vec::new();
    lines.push("[Host environment]".to_string());
    lines.push(format!("OS: {os} ({arch})"));

    if cfg!(windows) {
        let pwsh = has("pwsh");
        let powershell = has("powershell");
        let cmd = has("cmd");
        let mut parts: Vec<String> = Vec::new();
        if pwsh || powershell {
            let mut present: Vec<&str> = Vec::new();
            if pwsh {
                present.push("pwsh");
            }
            if powershell {
                present.push("powershell.exe");
            }
            parts.push(format!("PowerShell preferred ({} present)", present.join("/")));
        } else {
            parts.push("PowerShell preferred".to_string());
        }
        if cmd {
            parts.push("cmd.exe also available".to_string());
        }
        lines.push(format!("Shell: {}.", parts.join("; ")));
    } else {
        let bash = has("bash");
        let sh = has("sh");
        let zsh = has("zsh");
        let mut present: Vec<&str> = Vec::new();
        if bash {
            present.push("bash");
        }
        if sh {
            present.push("sh");
        }
        if zsh {
            present.push("zsh");
        }
        let preferred = if bash {
            "bash"
        } else if zsh {
            "zsh"
        } else {
            "sh"
        };
        let detail = if present.is_empty() {
            "POSIX shell".to_string()
        } else {
            format!("{preferred} preferred ({} present)", present.join("/"))
        };
        lines.push(format!("Shell: {detail}."));
    }

    // Common developer tools an agent is likely to reach for.
    let mut tools: Vec<&str> = Vec::new();
    if has("git") {
        tools.push("git");
    }
    if has("python") || has("python3") {
        tools.push("python");
    }
    if has("node") {
        tools.push("node");
    }
    if !tools.is_empty() {
        lines.push(format!("Detected tools: {}.", tools.join(", ")));
    }

    if cfg!(windows) {
        lines.push(
            "When a tool runs shell commands on this host, prefer PowerShell syntax.".to_string(),
        );
    } else {
        lines.push(
            "When a tool runs shell commands on this host, prefer POSIX/bash syntax.".to_string(),
        );
    }

    lines.join("\n")
}

/// Scan `PATH` for an executable named `name`, probing common Windows
/// extension variants (`.exe` / `.cmd` / `.bat`) as well. No process is
/// spawned — we only stat candidate paths.
fn path_has_executable(name: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    let candidates: Vec<String> = if cfg!(windows) {
        vec![
            name.to_string(),
            format!("{name}.exe"),
            format!("{name}.cmd"),
            format!("{name}.bat"),
        ]
    } else {
        vec![name.to_string()]
    };
    for dir in std::env::split_paths(&path) {
        for cand in &candidates {
            if dir.join(cand).is_file() {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hint_starts_with_label_and_names_the_host_os() {
        let hint = host_environment_hint();
        assert!(
            hint.starts_with("[Host environment]"),
            "must be clearly labeled: {hint}"
        );
        if cfg!(windows) {
            assert!(hint.contains("Windows"), "hint: {hint}");
            assert!(hint.contains("PowerShell"), "hint: {hint}");
        } else if cfg!(target_os = "macos") {
            assert!(hint.contains("macOS"), "hint: {hint}");
            assert!(hint.contains("POSIX/bash"), "hint: {hint}");
        } else {
            assert!(hint.contains("Linux"), "hint: {hint}");
            assert!(hint.contains("POSIX/bash"), "hint: {hint}");
        }
        // The arch should be present in the OS line.
        assert!(hint.contains(std::env::consts::ARCH), "hint: {hint}");
    }

    #[test]
    fn enabled_flag_roundtrips_and_restores_default() {
        assert!(tool_environment_hint_enabled(), "default is on");
        set_tool_environment_hint_enabled(false);
        assert!(!tool_environment_hint_enabled());
        // Restore so we don't perturb other tests sharing the process.
        set_tool_environment_hint_enabled(true);
        assert!(tool_environment_hint_enabled());
    }
}
