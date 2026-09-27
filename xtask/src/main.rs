//! rustllama xtask runner.
//!
//! Invoked as `cargo xtask <subcommand>`. Phase 0 implements `doctor`;
//! `fetch-test-model` lands in phase 1 once we have a CPU inference path
//! that needs a real GGUF to validate against.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "xtask", about = "rustllama dev tasks")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Probe build environment: rustc, oneAPI/icx/MKL/DNNL, SYCL devices, Node toolchain.
    Doctor,
    /// Download a small GGUF to target/test-models for integration tests.
    FetchTestModel {
        /// Which test model to fetch.
        #[arg(default_value = "tinyllama-1.1b-q4_k_m")]
        which: String,
    },
    /// Build the workspace with the right env vars resolved for SYCL.
    /// All compute backends (SYCL + CUDA + CPU) are always compiled in.
    Build {
        #[arg(long)]
        release: bool,
    },
    /// Stage Intel oneAPI redistributable DLLs into a directory
    /// ready to bundle into the MSI installer.
    ///
    /// Walks `%ONEAPI_ROOT%\compiler\latest\` for the
    /// manifest of runtime DLLs documented in
    /// [docs/oneapi-redist.md], copies them to `--out` (default:
    /// `target/redist-staging/`), and prints a markdown table of
    /// `(name, source path, size)` suitable for pasting into the
    /// redist doc. Errors on any missing DLL (rather than
    /// silently shipping an incomplete bundle).
    ///
    /// Run on a host that has oneAPI Base Toolkit 2025.0+ installed.
    StageRedist {
        /// Output directory for staged DLLs. Defaults to
        /// `target/redist-staging/`.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Skip the error on missing DLLs — useful for surveying
        /// what the local oneAPI install actually ships before
        /// committing to a manifest.
        #[arg(long)]
        allow_missing: bool,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    match args.cmd {
        Cmd::Doctor => doctor(),
        Cmd::FetchTestModel { which } => fetch_test_model(&which),
        Cmd::Build { release } => build_ws(release),
        Cmd::StageRedist { out, allow_missing } => stage_redist(out, allow_missing),
    }
}

fn doctor() -> Result<()> {
    println!("== rustllama doctor ==\n");

    let rustc =
        capture(Command::new("rustc").arg("--version")).unwrap_or_else(|e| format!("ERROR: {e}"));
    let cargo =
        capture(Command::new("cargo").arg("--version")).unwrap_or_else(|e| format!("ERROR: {e}"));
    println!("toolchain:");
    println!("  rustc        = {}", rustc.trim());
    println!("  cargo        = {}", cargo.trim());

    println!();
    println!("oneAPI / SYCL:");
    let oneapi_root = std::env::var("ONEAPI_ROOT")
        .ok()
        .filter(|p| PathBuf::from(p).exists())
        .or_else(|| {
            let p = "C:\\Program Files (x86)\\Intel\\oneAPI";
            PathBuf::from(p).exists().then(|| p.to_string())
        });
    match oneapi_root.as_deref() {
        Some(p) => {
            println!("  ONEAPI_ROOT  = {p}");
            check_subdir(p, "compiler\\latest");
            let icx = capture(Command::new("icx").arg("--version"))
                .map(|s| s.lines().next().unwrap_or("").to_string())
                .unwrap_or_else(|_| "NOT FOUND on PATH".into());
            println!("  icx          = {icx}");
            let sycl_ls = capture(&mut Command::new("sycl-ls"))
                .unwrap_or_else(|_| "NOT FOUND on PATH (run setvars.bat?)".into());
            println!("  sycl-ls      =");
            for line in sycl_ls.lines() {
                println!("                {line}");
            }
        }
        None => {
            println!("  ONEAPI_ROOT  = NOT FOUND");
            println!(
                "  -> install Intel oneAPI Base Toolkit 2025.0+ (https://www.intel.com/oneapi)"
            );
            println!("  -> phase 3+ (SYCL kernels) cannot build without it");
        }
    }

    println!();
    println!("Node toolchain (phase 7+):");
    let node =
        capture(Command::new("node").arg("--version")).unwrap_or_else(|_| "NOT FOUND".into());
    let pnpm =
        capture(Command::new("pnpm").arg("--version")).unwrap_or_else(|_| "NOT FOUND".into());
    println!("  node         = {}", node.trim());
    println!("  pnpm         = {}", pnpm.trim());

    println!();
    println!("paths:");
    if let Some(p) = std::env::var_os("LOCALAPPDATA") {
        println!("  LOCALAPPDATA = {}", p.to_string_lossy());
    }
    if let Some(p) = std::env::var_os("APPDATA") {
        println!("  APPDATA      = {}", p.to_string_lossy());
    }

    println!();
    println!("engine features:");
    // Static catalog — what this build supports. Worth printing
    // even when no model is loaded so users can verify "does
    // rustllama load my IQ3_S Mixtral?" before spending the
    // download bandwidth on a hub pull.
    println!("  architectures: llama-family (Llama 3, Qwen2.5-Coder, Mistral,");
    println!("                 Phi-3, DeepSeek-Coder-V2), Qwen3-MoE, Mixtral,");
    println!("                 DeepSeek-V3 (MoE w/ shared experts), BERT");
    println!("                 (embeddings + reranker classifier head)");
    println!("  weight quants: F32, F16, BF16,");
    println!("                 Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,");
    println!("                 Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K,");
    println!("                 IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,");
    println!("                 IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, NVFP4");
    println!("  KV dtypes:     F32, Q8_0, TurboQuant (1/2/4/8 bit), NVFP4");
    println!("  serving paths: single-flight CpuEngine + paged-batch");
    println!("                 continuous batching (PagedBatchEngine)");
    println!("  observability: prefix-cache hit-rate / cumulative stats /");
    println!("                 audit-log w/ credential masking /");
    println!("                 OpenAI system_fingerprint");

    Ok(())
}

fn capture(cmd: &mut Command) -> Result<String> {
    let out = cmd
        .output()
        .with_context(|| format!("failed to spawn {:?}", cmd.get_program()))?;
    if !out.status.success() {
        anyhow::bail!(
            "exit {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn check_subdir(root: &str, sub: &str) {
    let p = PathBuf::from(root).join(sub);
    if p.exists() {
        println!("  {sub:<12} = found ({})", p.display());
    } else {
        println!("  {sub:<12} = MISSING ({})", p.display());
    }
}

/// Catalog of test models keyed by short name.
fn test_model_catalog(which: &str) -> Result<(&'static str, &'static str)> {
    Ok(match which {
        "tinyllama-1.1b-q4_k_m" | "tinyllama" => (
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        ),
        "qwen2.5-coder-0.5b-q4_k_m" | "qwen0.5b" => (
            "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
            "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
        ),
        "qwen2.5-coder-1.5b-q4_k_m" | "qwen1.5b" => (
            "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
            "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        ),
        other => anyhow::bail!(
            "unknown test model `{other}`. \
             Known: tinyllama-1.1b-q4_k_m, qwen2.5-coder-0.5b-q4_k_m, qwen2.5-coder-1.5b-q4_k_m"
        ),
    })
}

fn fetch_test_model(which: &str) -> Result<()> {
    let (repo, filename) = test_model_catalog(which)?;
    let target_dir = PathBuf::from("target").join("test-models");
    std::fs::create_dir_all(&target_dir).context("create target/test-models")?;
    let dst = target_dir.join(filename);
    if dst.exists() {
        println!("already cached: {}", dst.display());
        return Ok(());
    }

    let hub_ref = rustllama_hub::HubRef::parse(&format!("{repo}:{filename}"))
        .map_err(|e| anyhow::anyhow!("internal: hub ref: {e}"))?;
    println!("fetching {repo}/{filename} -> {}", dst.display());

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("tokio runtime")?;
    let dst_clone = dst.clone();
    rt.block_on(async move {
        let pb = indicatif::ProgressBar::new_spinner();
        pb.enable_steady_tick(std::time::Duration::from_millis(120));
        let cached = rustllama_hub::download(&hub_ref, &target_dir, Some(&pb))
            .await
            .map_err(|e| anyhow::anyhow!("download failed: {e}"))?;
        pb.finish_and_clear();
        if cached != dst_clone {
            // download() puts files in <cache>/owner__repo/<filename>, but for
            // xtask we want them flat under target/test-models/.
            std::fs::copy(&cached, &dst_clone)
                .with_context(|| format!("copy {} -> {}", cached.display(), dst_clone.display()))?;
        }
        anyhow::Ok(())
    })?;

    println!("saved: {}", dst.display());
    Ok(())
}

fn build_ws(release: bool) -> Result<()> {
    let mut cmd = Command::new("cargo");
    cmd.arg("build").arg("--workspace");
    if release {
        cmd.arg("--release");
    }
    // SYCL + CUDA + CPU kernels are always compiled in (real-only), so
    // there are no GPU features to pass — but `cargo build --workspace`
    // now always requires Intel oneAPI (icx/icpx) and the CUDA Toolkit.
    let status = cmd.status().context("cargo build")?;
    if !status.success() {
        anyhow::bail!("cargo build failed");
    }
    Ok(())
}

// ============================================================
// stage-redist
// ============================================================
//
// Manifest of the runtime DLLs the rustllama MSI must ship
// alongside `rustllama.exe`. Sourced from
// [docs/oneapi-redist.md] — keep the two in sync when the
// pinned oneAPI version bumps.
//
// Each entry maps a **glob pattern** (single `*` wildcard) →
// the oneAPI sub-tree where it lives. Patterns absorb the
// version-digit drift between oneAPI releases (2025: `sycl7.dll`,
// 2026: `sycl9.dll`; 2025: `mkl_core.2.dll`, 2026:
// `mkl_core.3.dll`). The staging walker enumerates the relevant
// `<oneapi_root>/{group}/latest/bin/` directory and picks the
// release-variant match (excludes the `d`-suffix debug builds
// — `mkl_sycl_blas.6.dll` not `mkl_sycl_blasd.6.dll`).
/// Pinned to **oneAPI 2026**. The 2025 install ships
/// `sycl7.dll` + `pi_level_zero.dll` + `pi_opencl.dll` instead;
/// swap those entries when downgrading. The `xtask stage-redist`
/// run on each target oneAPI version surfaces which entries are
/// missing so the manifest can be refined per release.
///
/// Entries are exact filenames where the names are stable across
/// minor versions (Intel C++ runtime, dnnl) and glob patterns
/// (`*` wildcard) where Intel embeds the ABI version in the
/// filename (oneMKL family). Patterns are bounded — e.g.
/// `mkl_core.*.dll` matches `mkl_core.2.dll` and `mkl_core.3.dll`
/// but NOT `mkl_core_extras.6.dll`. When multiple files match a
/// pattern (release + debug variant in same dir), alphabetical
/// sort picks the release (see `find_in_dir`).
const REDIST_MANIFEST: &[(&str, RedistGroup)] = &[
    // SYCL runtime — 2026 ships sycl9.dll (1.7 MB); 2025 ships
    // sycl7.dll. The exact filename matches just one to keep
    // sycl-jit.dll (190 MB, build-time only) from getting
    // bundled — JIT isn't required at runtime for AOT-compiled
    // kernels.
    ("sycl9.dll", RedistGroup::Compiler),
    // Unified Runtime adapters — 2026 replaced `pi_level_zero.dll`
    // + `pi_opencl.dll` with these. v2 ships alongside v1 in
    // 2026.x; the glob picks the highest-versioned via sort
    // order (`v2` > no-suffix).
    ("ur_adapter_level_zero*.dll", RedistGroup::Compiler),
    ("ur_adapter_opencl.dll", RedistGroup::Compiler),
    // OpenCL runtime — `OpenCL.dll` is the loader; `intelocl64.dll`
    // is Intel's CPU OpenCL implementation. Required for SYCL
    // workloads that fall back to the OpenCL device when Level
    // Zero is unavailable.
    ("OpenCL.dll", RedistGroup::Compiler),
    ("intelocl64.dll", RedistGroup::Compiler),
    // Intel C++ runtime — unversioned filenames, stable across
    // releases. `libmmd` is Intel Math library; `svml_dispmd` is
    // the vectorized math dispatcher; `libiomp5md` is Intel OMP.
    ("libmmd.dll", RedistGroup::Compiler),
    ("svml_dispmd.dll", RedistGroup::Compiler),
    ("libiomp5md.dll", RedistGroup::Compiler),
    // NOTE: oneMKL (mkl_*.dll, ~172 MB) and oneDNN (dnnl.dll, ~77 MB)
    // used to be bundled here, but they were ONLY oneDNN's GEMM backend.
    // oneDNN is gone (our SYCL kernels are self-contained), so they are
    // no longer redistributed — this trims the installer by ~249 MB.
];

#[derive(Debug, Clone, Copy)]
enum RedistGroup {
    Compiler,
}

impl RedistGroup {
    fn subdir(self) -> &'static str {
        match self {
            RedistGroup::Compiler => "compiler",
        }
    }
}

fn stage_redist(out: Option<PathBuf>, allow_missing: bool) -> Result<()> {
    let oneapi_root = std::env::var("ONEAPI_ROOT")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .or_else(|| {
            let p = PathBuf::from("C:\\Program Files (x86)\\Intel\\oneAPI");
            p.exists().then_some(p)
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "ONEAPI_ROOT not set and the default install path doesn't exist. \
                 Install Intel oneAPI Base Toolkit 2025.0+ (https://www.intel.com/oneapi) \
                 or set ONEAPI_ROOT manually."
            )
        })?;
    println!("== rustllama stage-redist ==\n");
    println!("ONEAPI_ROOT = {}\n", oneapi_root.display());

    let out_dir = out.unwrap_or_else(|| PathBuf::from("target").join("redist-staging"));
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("create out dir {}", out_dir.display()))?;
    println!("staging into: {}\n", out_dir.display());

    let mut found = 0usize;
    let mut missing: Vec<&str> = Vec::new();
    let mut manifest_rows: Vec<(String, PathBuf, u64)> = Vec::new();

    for (pattern, group) in REDIST_MANIFEST {
        let src = match find_redist_dll(&oneapi_root, pattern, *group) {
            Some(p) => p,
            None => {
                missing.push(pattern);
                println!("  MISSING  {pattern}");
                continue;
            }
        };
        // Stage under the source filename (the resolved pattern's
        // actual match, not the glob string).
        let actual_name = src
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| anyhow::anyhow!("non-UTF8 dll filename: {}", src.display()))?;
        let dst = out_dir.join(actual_name);
        let size = std::fs::metadata(&src)
            .with_context(|| format!("stat {}", src.display()))?
            .len();
        std::fs::copy(&src, &dst)
            .with_context(|| format!("copy {} -> {}", src.display(), dst.display()))?;
        println!(
            "  ok       {pattern:<32} -> {actual_name} ({size} bytes)",
        );
        manifest_rows.push((actual_name.to_string(), src, size));
        found += 1;
    }

    println!("\nstaged: {found}/{}", REDIST_MANIFEST.len());
    if !missing.is_empty() {
        println!("missing: {}", missing.join(", "));
        if !allow_missing {
            anyhow::bail!(
                "{} expected DLL pattern(s) not found under {}. Re-run with --allow-missing \
                 to survey what IS present without erroring; or install oneAPI \
                 components that ship the missing DLLs.",
                missing.len(),
                oneapi_root.display(),
            );
        }
    }

    println!("\n== manifest (paste into docs/oneapi-redist.md) ==\n");
    println!("| DLL | Source | Size (bytes) |");
    println!("|-----|--------|--------------|");
    for (name, src, size) in &manifest_rows {
        println!("| `{name}` | `{}` | {size} |", src.display());
    }

    Ok(())
}

/// Walk the oneAPI install for a DLL matching `pattern` under
/// the expected group's `latest/bin/` subdir, with fallbacks for
/// the awkward cases where Intel ships a DLL in a non-`bin/`
/// location or without the `latest` junction.
///
/// `pattern` is a glob with at most one `*` wildcard (matches any
/// run of chars). When multiple DLLs in the same directory match
/// (e.g. `mkl_sycl_blas.6.dll` + `mkl_sycl_blasd.6.dll`), the
/// release-variant rule strips the `d`-suffix debug builds: any
/// filename whose stem ends in `d` is skipped unless it's the
/// only match.
fn find_redist_dll(
    oneapi_root: &std::path::Path,
    pattern: &str,
    group: RedistGroup,
) -> Option<PathBuf> {
    let group_root = oneapi_root.join(group.subdir()).join("latest");
    let search_dirs = [
        group_root.join("bin"),
        group_root.join("opt").join("compiler").join("bin"),
        group_root.join("lib"),
    ];
    for dir in &search_dirs {
        if let Some(p) = find_in_dir(dir, pattern) {
            return Some(p);
        }
    }
    // Versioned-subdir fallback: walk `<group>/<version>/bin/`
    // when the `latest` junction is missing.
    let group_root = oneapi_root.join(group.subdir());
    if let Ok(entries) = std::fs::read_dir(&group_root) {
        for entry in entries.flatten() {
            if let Some(p) = find_in_dir(&entry.path().join("bin"), pattern) {
                return Some(p);
            }
        }
    }
    None
}

/// Glob-match `pattern` against filenames in `dir`. Returns the
/// release-variant match (debug `d`-suffix filenames lose ties).
///
/// Heuristic: in oneAPI's bin directories, release filenames
/// alphabetically sort before their debug siblings — e.g.
/// `sycl9.dll < sycl9d.dll` and
/// `mkl_sycl_blas.6.dll < mkl_sycl_blasd.6.dll` (`.` is ASCII 46,
/// `d` is ASCII 100). So picking the first sorted match yields
/// the release build, with no per-filename allow-list needed.
fn find_in_dir(dir: &std::path::Path, pattern: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut matches: Vec<PathBuf> = Vec::new();
    for e in entries.flatten() {
        let name = e.file_name();
        let name_str = name.to_string_lossy();
        if glob_match(pattern, &name_str) {
            matches.push(e.path());
        }
    }
    if matches.is_empty() {
        return None;
    }
    matches.sort();
    Some(matches.into_iter().next().unwrap())
}

/// Minimal `*`-glob matcher: `pattern` may contain any number of
/// `*` wildcards (each matches an arbitrary, possibly empty,
/// substring). All non-`*` parts must appear in `name` in
/// order. Returns true on full match (anchored at both ends).
fn glob_match(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == name;
    }
    let mut cursor = 0usize;
    // First part must be a prefix (no leading `*`).
    if !parts[0].is_empty() {
        if !name.starts_with(parts[0]) {
            return false;
        }
        cursor = parts[0].len();
    }
    // Middle parts in order.
    for &p in &parts[1..parts.len() - 1] {
        if p.is_empty() {
            continue;
        }
        match name[cursor..].find(p) {
            Some(idx) => cursor += idx + p.len(),
            None => return false,
        }
    }
    // Last part must be a suffix (no trailing `*`).
    let last = parts[parts.len() - 1];
    if last.is_empty() {
        return true;
    }
    name.len() >= cursor + last.len() && name.ends_with(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Manifest is non-empty and contains expected anchor
    /// pattern strings. Catches a partial deletion that would
    /// silently ship a broken bundle.
    #[test]
    fn manifest_includes_required_runtime_anchors() {
        let names: Vec<&str> = REDIST_MANIFEST.iter().map(|(n, _)| *n).collect();
        // Anchors that MUST be present (oneAPI 2026 pin).
        for must_have in &["sycl9.dll", "libmmd.dll"] {
            assert!(
                names.contains(must_have),
                "manifest must include {must_have} (rustllama.exe won't run without it)",
            );
        }
        // No duplicates — a pattern listed twice would stage
        // matching DLLs twice and bloat the installer.
        let mut sorted = names.clone();
        sorted.sort();
        let original_len = sorted.len();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            original_len,
            "manifest has duplicate entries"
        );
    }

    /// `find_redist_dll` locates files under `<group>/latest/bin/`
    /// for both exact-name and glob-pattern entries. Uses a fake
    /// `oneAPI` directory in `temp` so the test works on any
    /// host regardless of whether the real oneAPI is installed.
    #[test]
    fn find_redist_dll_locates_files_via_exact_and_glob_patterns() {
        let tmp = std::env::temp_dir().join("rustllama-stage-redist-test");
        let _ = std::fs::remove_dir_all(&tmp);
        let bin = tmp.join("compiler").join("latest").join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(bin.join("sycl9.dll"), b"fake").unwrap();
        std::fs::write(bin.join("sycl9d.dll"), b"debug").unwrap(); // debug variant
        std::fs::write(bin.join("libmmd.dll"), b"fake").unwrap();

        // Exact match.
        let found = find_redist_dll(&tmp, "libmmd.dll", RedistGroup::Compiler);
        assert_eq!(found.as_deref().unwrap().file_name().unwrap(), "libmmd.dll");

        // Glob match — picks release over debug variant.
        let found = find_redist_dll(&tmp, "sycl*.dll", RedistGroup::Compiler);
        assert_eq!(
            found.as_deref().unwrap().file_name().unwrap(),
            "sycl9.dll",
            "must pick release (sycl9.dll) over debug (sycl9d.dll)"
        );

        // Missing file → None.
        let missing = find_redist_dll(&tmp, "no_such.dll", RedistGroup::Compiler);
        assert!(missing.is_none());

        // Walk-fallback: file in a versioned subdir without the
        // `latest` junction.
        let alt = tmp.join("dnnl").join("2025.0.0").join("bin");
        std::fs::create_dir_all(&alt).unwrap();
        std::fs::write(alt.join("dnnl.dll"), b"fake").unwrap();
        let found = find_redist_dll(&tmp, "dnnl.dll", RedistGroup::Dnnl);
        assert_eq!(
            found.as_deref().unwrap().file_name().unwrap(),
            "dnnl.dll",
            "walk fallback should find versioned-subdir bin/<name>"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `glob_match`: exact match when no wildcard; `*` matches
    /// any (possibly empty) substring at the wildcard position;
    /// multiple wildcards work; anchored at both ends.
    #[test]
    fn glob_match_handles_exact_and_wildcard() {
        // Exact (no `*`).
        assert!(glob_match("libmmd.dll", "libmmd.dll"));
        assert!(!glob_match("libmmd.dll", "libmmdx.dll"));
        assert!(!glob_match("libmmd.dll", "libmmd.dlls"));

        // Single wildcard.
        assert!(glob_match("sycl*.dll", "sycl7.dll"));
        assert!(glob_match("sycl*.dll", "sycl9.dll"));
        assert!(glob_match("sycl*.dll", "sycl.dll")); // empty middle
        assert!(!glob_match("sycl*.dll", "sycl9.dl"));
        assert!(!glob_match("sycl*.dll", "Xsycl9.dll"));

        // Wildcard in the middle.
        assert!(glob_match("mkl_core.*.dll", "mkl_core.2.dll"));
        assert!(glob_match("mkl_core.*.dll", "mkl_core.10.dll"));
        assert!(!glob_match("mkl_core.*.dll", "mkl_cored.2.dll"));

        // Multi-wildcard.
        assert!(glob_match("mkl_*_blas.*.dll", "mkl_sycl_blas.6.dll"));
        assert!(!glob_match("mkl_*_blas.*.dll", "mkl_sycl_lapack.6.dll"));

        // No anchor required when pattern starts/ends with `*`.
        assert!(glob_match("*.dll", "anything.dll"));
        assert!(glob_match("mkl*", "mkl_sycl_blas.6.dll"));
        assert!(glob_match("*", ""));
        assert!(glob_match("*", "anything"));
    }

    /// `find_in_dir` picks the release sibling when both
    /// release and debug variants of a glob match exist in the
    /// same directory. Alphabetical sort puts `.` (46) before `d`
    /// (100), so `mkl_sycl_blas.6.dll` < `mkl_sycl_blasd.6.dll`.
    #[test]
    fn find_in_dir_prefers_release_over_debug_via_alphabetical_sort() {
        let tmp = std::env::temp_dir().join("rustllama-find-in-dir-test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        // Three matches for `mkl_sycl_blas*.dll`: one release,
        // one debug, one similarly-named lapack (won't match the
        // narrower pattern below).
        std::fs::write(tmp.join("mkl_sycl_blas.6.dll"), b"release").unwrap();
        std::fs::write(tmp.join("mkl_sycl_blasd.6.dll"), b"debug").unwrap();
        std::fs::write(tmp.join("mkl_sycl_lapack.6.dll"), b"unrelated").unwrap();

        let found = find_in_dir(&tmp, "mkl_sycl_blas.*.dll");
        assert_eq!(
            found.unwrap().file_name().unwrap(),
            "mkl_sycl_blas.6.dll",
            "must pick release variant via sort order"
        );

        // Wider glob `mkl_sycl_*.dll` matches all three; release
        // `blas` still wins the sort.
        let found = find_in_dir(&tmp, "mkl_sycl_*.dll");
        assert_eq!(found.unwrap().file_name().unwrap(), "mkl_sycl_blas.6.dll");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
