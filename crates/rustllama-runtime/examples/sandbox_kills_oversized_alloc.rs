//! Functional check for the Windows Job Object sandbox.
//!
//! Run:
//! ```
//! cargo run --release -p rustllama-runtime --example sandbox_kills_oversized_alloc
//! ```
//!
//! What it does:
//!
//! 1. Installs the sandbox with a deliberately-low 256 MiB cap.
//! 2. Allocates a 512 MiB `Vec<u8>` in 64 MiB chunks, touching each
//!    page to force the kernel to commit RAM (Vec::with_capacity
//!    alone reserves address space but doesn't necessarily commit).
//! 3. Expects to be killed by the OS partway through (process exits
//!    via the Job Object's memory-limit enforcement).
//!
//! If the example prints "all 512 MiB committed without termination",
//! the sandbox is NOT engaging. Check:
//!   - Are you on Windows? (Linux + macOS sandbox return Ok without
//!     installing limits in v1.)
//!   - Is the process already inside a parent Job Object that
//!     refuses nesting? (Common in CI / IDE-spawned shells.)
//!   - Does the GUI / cmd.exe parent have `JOB_OBJECT_LIMIT_BREAKAWAY_OK`
//!     unset on its own job? (We can't escape a parent that pinned us.)

use rustllama_runtime::sandbox::{install, SandboxConfig};

fn main() {
    eprintln!("sandbox functional check: installing 256 MiB cap…");
    let cfg = SandboxConfig {
        memory_limit_bytes: 256 * 1024 * 1024,
        kill_on_close: true,
    };
    match install(cfg) {
        Ok(()) => eprintln!("  sandbox installed (or already attached to a parent)"),
        Err(e) => {
            eprintln!("  install failed: {e}");
            eprintln!("  exiting; sandbox isn't engaged so the alloc test is meaningless");
            return;
        }
    }
    eprintln!();
    eprintln!("allocating 512 MiB in 64 MiB chunks…");
    eprintln!("(expected: OS kills the process around 256 MiB)");
    eprintln!();
    let mut bufs: Vec<Vec<u8>> = Vec::new();
    for i in 1..=8 {
        let mut buf = vec![0u8; 64 * 1024 * 1024];
        // Touch every page so the kernel commits RAM. Without this,
        // Vec only reserves virtual address space and the job limit
        // never fires.
        for (j, b) in buf.iter_mut().enumerate() {
            if j % 4096 == 0 {
                *b = (j as u8).wrapping_add(i as u8);
            }
        }
        bufs.push(buf);
        eprintln!("  committed chunk {i}/8 ({:>3} MiB total)", i * 64);
    }
    eprintln!();
    eprintln!("all 512 MiB committed without termination");
    eprintln!("sandbox is NOT enforcing the memory cap — see troubleshooting above");
}
