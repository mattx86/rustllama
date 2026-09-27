//! Startup autodetection of GPU-runtime library search paths, so the
//! SYCL/CUDA-enabled binary runs from a bare shell without the
//! `scripts\run-sycl.bat` launcher prepending PATH.
//!
//! # Why this works (Windows)
//!
//! `rsl_kernels.dll` (the SYCL kernel DLL) is **delay-loaded** by the
//! binary (see `app/src-tauri/build.rs`: `/DELAYLOAD:rsl_kernels.dll`).
//! Delay-load defers binding the DLL — and thus its transitive oneAPI
//! dependencies (`sycl*.dll`, `ur_loader.dll`, `ur_adapter_level_zero*
//! .dll`, `libhwloc-15.dll`, …) — until the first `rsl_*` FFI call,
//! which happens well after `main()`. So if we prepend the oneAPI
//! runtime directories to the process `PATH` at the very top of
//! `main()`, the Windows loader's legacy search order picks them up
//! when the delay-loaded DLL finally binds. This is exactly what
//! run-sycl.bat did, performed in-process instead.
//!
//! Without delay-load this would be too late: an import-table-linked
//! DLL binds at process init, before any Rust code runs.
//!
//! # Layout mirrored from run-sycl.bat
//!   - `<ONEAPI_ROOT>\compiler\latest\bin`  — sycl*.dll, ur_loader.dll,
//!     ur_adapter_*.dll, UMF.dll, libmmd.dll
//!   - `<ONEAPI_ROOT>\<version>\bin`         — libhwloc-15.dll (the
//!     UR Level-Zero adapter's transitive dep; without it L0 fails to
//!     load and SYCL silently falls back to OpenCL)
//!   - `<CUDA_PATH>\bin`                      — cudart / nvcc runtime
//!     (NVIDIA hosts; harmless no-op where absent)
//!
//! Non-Windows is a no-op: the ELF loader reads `LD_LIBRARY_PATH` only
//! at exec time, so an in-process prepend has no effect — Linux GPU
//! path resolution is handled via rpath / dlopen (Milestone 2).

use std::sync::OnceLock;

/// Idempotently prepend the autodetected GPU-runtime directories to the
/// process `PATH` (Windows). Safe to call from every entry point; the
/// work runs once. No-op on non-Windows and when no toolkit is found.
pub fn ensure_gpu_dll_search_paths() {
    static DONE: OnceLock<()> = OnceLock::new();
    DONE.get_or_init(|| {
        #[cfg(windows)]
        prepend_windows_gpu_dirs();
    });
}

#[cfg(windows)]
fn prepend_windows_gpu_dirs() {
    use std::path::PathBuf;

    let debug = std::env::var_os("RUSTLLAMA_DLL_PATH_DEBUG").is_some();
    let mut add: Vec<PathBuf> = Vec::new();

    // ---- Intel oneAPI (SYCL / Level Zero) ----
    // ONEAPI_ROOT (explicit) → ONEAPI_ROOT_DEFAULT (from .cargo config
    // / packaging) → the standard install location.
    let oneapi_root = std::env::var_os("ONEAPI_ROOT")
        .or_else(|| std::env::var_os("ONEAPI_ROOT_DEFAULT"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files (x86)\Intel\oneAPI"));
    if oneapi_root.is_dir() {
        // 1. compiler\latest\bin (the SYCL runtime DLLs).
        let comp = oneapi_root.join("compiler").join("latest").join("bin");
        if comp.is_dir() {
            add.push(comp);
        }
        // 2. The shared-runtime <version>\bin that ships libhwloc-15.dll.
        //    Scan the top-level version dirs (2026.0, 2025.2, …) for the
        //    one that actually contains it; take the highest match.
        if let Ok(entries) = std::fs::read_dir(&oneapi_root) {
            let mut candidates: Vec<PathBuf> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.is_dir()
                        && p.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with("20"))
                        && p.join("bin").join("libhwloc-15.dll").is_file()
                })
                .collect();
            candidates.sort();
            if let Some(best) = candidates.pop() {
                add.push(best.join("bin"));
            }
        }
    }

    // ---- NVIDIA CUDA (Milestone 2 backend; harmless to resolve now) ----
    if let Some(cuda) = std::env::var_os("CUDA_PATH").map(PathBuf::from) {
        let bin = cuda.join("bin");
        if bin.is_dir() {
            add.push(bin);
        }
    }

    if add.is_empty() {
        if debug {
            eprintln!("[gpu_paths] no oneAPI/CUDA runtime dirs found; leaving PATH unchanged");
        }
        return;
    }

    // Prepend (search-first), de-duped against what's already on PATH.
    let existing = std::env::var_os("PATH").unwrap_or_default();
    let existing_lower: Vec<String> = std::env::split_paths(&existing)
        .map(|p| p.to_string_lossy().to_lowercase())
        .collect();
    let mut new_paths: Vec<PathBuf> = Vec::new();
    for d in &add {
        let dl = d.to_string_lossy().to_lowercase();
        if !existing_lower.contains(&dl) {
            new_paths.push(d.clone());
        }
    }
    new_paths.extend(std::env::split_paths(&existing));
    match std::env::join_paths(&new_paths) {
        Ok(joined) => {
            std::env::set_var("PATH", &joined);
            if debug {
                eprintln!("[gpu_paths] prepended to PATH: {add:?}");
            }
        }
        Err(e) => {
            if debug {
                eprintln!("[gpu_paths] join_paths failed: {e}; PATH unchanged");
            }
        }
    }
}
