//! Build script for the Apple-Metal / MLX kernel translation unit.
//!
//! This crate is REAL-ONLY and always compiled (like the SYCL and CUDA
//! kernel crates) — there are no `mlx`/`mock` cargo features. What it
//! builds is decided purely by the TARGET:
//!
//!   * `aarch64-apple-darwin` (Apple Silicon) → the REAL path: compile the
//!     Objective-C++ host shim `mlx/rsl_mlx.mm` (+ the Metal shaders in
//!     `mlx/rsl_mlx.metal`) into `librsl_mlx.dylib`, link the Metal /
//!     Foundation frameworks and Apple's `libmlx`. Only Apple Silicon has a
//!     Metal GPU + MLX, so this is the one target that gets a live backend.
//!
//!   * Everything else — Windows, Linux, AND Intel macOS (`x86_64-apple-
//!     darwin`, which has NO MLX Metal GPU path) → an INERT no-op stub: a
//!     tiny generated C translation unit that defines every `rsl_mlx_*`
//!     symbol as `long long name(){return 0;}`, compiled into a STATIC
//!     archive that links straight into the binary. `rsl_mlx_device_count()`
//!     then returns 0 at runtime → the engine sees no MLX device and runs
//!     on CPU / SYCL / CUDA. This is the same discipline the SYCL crate
//!     uses to stub oneAPI on non-x86 (`build_sycl_stub`); here we stub
//!     Metal/MLX on every non-Apple-Silicon host.
//!
//! The stub gate is therefore `CARGO_CFG_TARGET_OS != "macos" ||
//! CARGO_CFG_TARGET_ARCH != "aarch64"`.
//!
//! WHY a static archive (not a shared lib) for the stub: an inert stub only
//! needs its symbols to RESOLVE at link time, so the simplest, most portable
//! option is to let the `cc` crate compile the no-op TU into a static lib —
//! it handles MSVC / clang-cl / clang / gcc detection and emits the
//! `rustc-link-lib=static=` directive itself, avoiding the fragile per-OS
//! DLL/import-lib (`.def`) and `.so`/rpath/copy machinery. Nothing ships at
//! runtime because the stub is linked in. (The REAL macOS path still builds a
//! swappable `.dylib` so the Phase-1 Metal kernels can be bundled + relinked
//! via `@rpath`.)

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_path = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR set by cargo"));

    println!("cargo:rerun-if-changed=mlx/rsl_mlx.def");
    println!("cargo:rerun-if-changed=mlx/rsl_mlx.h");
    println!("cargo:rerun-if-changed=mlx/rsl_mlx.mm");
    println!("cargo:rerun-if-changed=mlx/rsl_mlx.metal");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MLX_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MLX_INCLUDE_DIR");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    // Apple Silicon (aarch64-apple-darwin) is the ONLY target with a real
    // Metal GPU + MLX. Everything else stubs.
    if target_os == "macos" && target_arch == "aarch64" {
        build_mlx_real(&out_path);
    } else {
        build_mlx_stub(&out_path);
    }
}

/// REAL path (Apple Silicon). Phase-0 SKELETON.
///
/// In Phase 0 this compiles ONLY the Objective-C++ host shim
/// `mlx/rsl_mlx.mm`, whose entry points are inert no-ops (they return 0 /
/// -1 and `rsl_mlx_device_count()` returns 0). That is enough to (a) give
/// the crate real `rsl_mlx_*` symbols so it LINKS on a Mac, and (b) keep the
/// backend inert until the Metal kernels + mlx-c enumeration land. The
/// `.mm` file is the localized place Phase 1 fills in.
///
/// Phase-1 TODO (all marked in `mlx/rsl_mlx.mm` / `.metal` too):
///   1. Compile the Metal shader library:
///        xcrun -sdk macosx metal   -c mlx/rsl_mlx.metal -o $OUT/rsl_mlx.air
///        xcrun -sdk macosx metallib   $OUT/rsl_mlx.air   -o $OUT/rsl_mlx.metallib
///      and either embed it (bin2c / `-sectcreate`) or load it at runtime
///      via `newLibraryWithURL:`. Then `#define RSL_MLX_HAVE_METAL 1` for
///      the `.mm` compile so the real kernel bodies replace the no-ops.
///   2. Link Apple's `libmlx` (the mlx-c C API) — set MLX_LIB_DIR /
///      MLX_INCLUDE_DIR (see below); this build already wires the flags.
///   3. Real device enumeration through mlx-c / `MTLCopyAllDevices()`.
fn build_mlx_real(out_path: &std::path::Path) {
    // clang++ ships with the Xcode Command Line Tools on every Mac; a
    // missing one should fail loudly (matches the SYCL/CUDA "toolchain
    // required" posture).
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "clang++".to_string());
    if !command_ok(&cxx) {
        panic!(
            "rustllama-kernels-mlx: `{cxx}` not found. Install the Xcode \
             Command Line Tools (`xcode-select --install`) so clang++ is on \
             PATH, then rebuild."
        );
    }

    // Phase 1a: compile the Metal shader to a `.metallib` and bin2c-embed it
    // into a generated header the `.mm` #includes. Embedding (vs a sidecar
    // `.metallib` + newLibraryWithURL:) keeps librsl_mlx.dylib self-contained
    // and relocatable — the release bundles copy the dylib beside the binary
    // with no runtime path lookup. This must run BEFORE the `.mm` compile: the
    // compile adds `-I$OUT` so the shim can `#include "rsl_mlx_metallib.h"`.
    build_metallib_header(out_path);

    // Optional MLX SDK locations, read like the CUDA build reads CUDA_PATH.
    // In Phase 0 the `.mm` shim does not yet #include the MLX headers or
    // call libmlx, so these are only used when present (Phase 1 makes them
    // required once the real kernels land).
    let mlx_include = std::env::var("MLX_INCLUDE_DIR").ok();
    let mlx_lib = std::env::var("MLX_LIB_DIR").ok();

    let dylib = out_path.join("librsl_mlx.dylib");
    let mut cmd = Command::new(&cxx);
    cmd.args([
        "-std=c++17",
        "-fobjc-arc", // ARC for the ObjC++ Metal objects the shim holds
        "-O3",
        "-fPIC",
        "-dynamiclib",
        "-fvisibility=default", // export the extern "C" rsl_mlx_* symbols
        "-Imlx",
    ]);
    // Phase 1a: turn on the real Metal bodies and put the generated
    // `rsl_mlx_metallib.h` (emitted by build_metallib_header above) on the
    // include path so the `.mm` can #include the embedded shader library.
    cmd.arg("-DRSL_MLX_HAVE_METAL=1");
    cmd.arg(format!("-I{}", out_path.display()));
    if let Some(inc) = &mlx_include {
        cmd.arg(format!("-I{inc}"));
    }
    cmd.arg("mlx/rsl_mlx.mm")
        // The dylib's install name is @rpath-relative so the consuming
        // binary's `@loader_path` rpath (below) resolves it beside itself,
        // matching the Linux `$ORIGIN` scheme the release bundles use.
        .arg("-install_name")
        .arg("@rpath/librsl_mlx.dylib")
        .arg("-o")
        .arg(&dylib);
    // Metal + Foundation exist on every Mac; linking them now (even though
    // the Phase-0 shim doesn't call them yet) keeps the link line
    // representative and harmless. Phase 1 may also need
    // MetalPerformanceShaders / QuartzCore.
    cmd.args(["-framework", "Metal", "-framework", "Foundation"]);
    // Apple's mlx-c library, when its lib dir is known. Skipped in Phase 0
    // if unset so the crate still builds on a Mac WITHOUT MLX installed
    // (the shim has no MLX references yet).
    if let Some(libdir) = &mlx_lib {
        cmd.arg(format!("-L{libdir}"));
        cmd.arg("-lmlx");
    }

    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to invoke {cxx} for mlx/rsl_mlx.mm: {e}"));
    if !status.success() {
        panic!("{cxx} failed to build librsl_mlx.dylib");
    }

    println!("cargo:rustc-link-search=native={}", out_path.display());
    println!("cargo:rustc-link-lib=dylib=rsl_mlx");
    // Frameworks the produced binary needs at link time.
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    if let Some(libdir) = &mlx_lib {
        println!("cargo:rustc-link-search=native={libdir}");
        println!("cargo:rustc-link-lib=dylib=mlx");
    } else {
        // Phase 1a is PURE Metal (device + buffers + f32 matvec via the Metal
        // frameworks only), so libmlx is not needed and this is not a warning
        // worth failing over. mlx-c lands with the Phase-1b/1c kernels that opt
        // to call mlx ops directly; set MLX_LIB_DIR / MLX_INCLUDE_DIR then.
        println!(
            "cargo:warning=rustllama-kernels-mlx: MLX_LIB_DIR unset — building \
             the Phase-1a pure-Metal backend WITHOUT libmlx (expected; mlx-c is \
             only needed for later kernel phases that call mlx ops directly)."
        );
    }
    // Resolve `librsl_mlx.dylib` beside the executable at runtime (Apple's
    // analogue of Linux `$ORIGIN`).
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");

    copy_shared_lib_next_to_binaries(out_path, "librsl_mlx.dylib");
}

/// Compile `mlx/rsl_mlx.metal` → `$OUT/rsl_mlx.air` → `$OUT/rsl_mlx.metallib`
/// (via `xcrun metal` / `xcrun metallib`), then bin2c the `.metallib` bytes
/// into `$OUT/rsl_mlx_metallib.h` as a `static const unsigned char
/// rsl_mlx_metallib[]` + `rsl_mlx_metallib_len`, which the `.mm` embeds and
/// loads with `[dev newLibraryWithData:]`.
///
/// NOTE: the Metal shader compiler (`xcrun -sdk macosx metal`) ships with the
/// FULL Xcode / the "Metal Toolchain" component — NOT the Command Line Tools
/// alone. A CLT-only Mac builds the `.mm` but fails here; run_xcrun panics with
/// that guidance.
fn build_metallib_header(out_path: &std::path::Path) {
    let air = out_path.join("rsl_mlx.air");
    let metallib = out_path.join("rsl_mlx.metallib");

    // .metal → .air (AIR object), then .air → .metallib (loadable library).
    run_xcrun(&[
        "-sdk",
        "macosx",
        "metal",
        "-c",
        "mlx/rsl_mlx.metal",
        "-o",
        air.to_str().expect("OUT_DIR path is valid UTF-8"),
    ]);
    run_xcrun(&[
        "-sdk",
        "macosx",
        "metallib",
        air.to_str().expect("OUT_DIR path is valid UTF-8"),
        "-o",
        metallib.to_str().expect("OUT_DIR path is valid UTF-8"),
    ]);

    // bin2c: embed the metallib bytes so the dylib is self-contained.
    let bytes = std::fs::read(&metallib).unwrap_or_else(|e| {
        panic!(
            "rustllama-kernels-mlx build.rs: read {}: {e}",
            metallib.display()
        )
    });
    let mut h = String::with_capacity(bytes.len() * 6 + 512);
    h.push_str("/* AUTO-GENERATED by build.rs — rsl_mlx.metal compiled to a\n");
    h.push_str(" * .metallib and embedded here so librsl_mlx.dylib is\n");
    h.push_str(" * self-contained (no runtime .metallib path lookup). The .mm\n");
    h.push_str(" * loads it via dispatch_data_create + [dev newLibraryWithData:]. */\n");
    h.push_str("static const unsigned char rsl_mlx_metallib[] = {\n");
    for (i, b) in bytes.iter().enumerate() {
        if i % 16 == 0 {
            h.push_str("    ");
        }
        h.push_str(&format!("0x{b:02x},"));
        if i % 16 == 15 {
            h.push('\n');
        }
    }
    h.push_str("\n};\n");
    h.push_str(&format!(
        "static const unsigned long rsl_mlx_metallib_len = {};\n",
        bytes.len()
    ));
    let hdr = out_path.join("rsl_mlx_metallib.h");
    std::fs::write(&hdr, h).unwrap_or_else(|e| {
        panic!(
            "rustllama-kernels-mlx build.rs: write {}: {e}",
            hdr.display()
        )
    });
}

/// Run `xcrun <args>`, panicking with actionable guidance on failure. The
/// Metal compiler is part of Xcode proper, so a missing/failed invocation
/// usually means only the Command Line Tools are installed.
fn run_xcrun(args: &[&str]) {
    let status = Command::new("xcrun").args(args).status().unwrap_or_else(|e| {
        panic!(
            "rustllama-kernels-mlx: failed to invoke `xcrun {}`: {e}. The Metal \
             shader compiler ships with full Xcode (not the Command Line Tools \
             alone) — install Xcode and run `sudo xcode-select -s \
             /Applications/Xcode.app`.",
            args.join(" ")
        )
    });
    if !status.success() {
        panic!(
            "rustllama-kernels-mlx: `xcrun {}` failed. Ensure full Xcode + the \
             Metal toolchain are installed (`xcode-select -s \
             /Applications/Xcode.app`); the CLT-only install lacks the Metal \
             compiler.",
            args.join(" ")
        );
    }
}

/// Non-Apple-Silicon targets (Windows, Linux, Intel macOS): build a no-op C
/// stub that provides every `rsl_mlx_*` FFI symbol the Rust crate
/// references, so it links without any Metal/MLX toolchain. At runtime
/// `rsl_mlx_device_count()` returns 0 → no MLX device → CPU / SYCL / CUDA
/// path. The symbol list is read from `mlx/rsl_mlx.def` (the authoritative
/// export set). Mirrors `build_sycl_stub` in rustllama-kernels-sycl.
///
/// Each symbol is defined as `long long name() { return 0; }`: empty parens
/// accept any caller ABI, and a 0 integer/pointer return is correct for the
/// only functions actually invoked before device_count reports zero devices
/// (the device/stream probes — integer + pointer returns). The kernel entry
/// points are never called when there is no device; they only need to
/// resolve at link time.
fn build_mlx_stub(out_path: &std::path::Path) {
    let names = parse_def_exports("mlx/rsl_mlx.def");
    assert!(
        !names.is_empty(),
        "rustllama-kernels-mlx build.rs: no rsl_mlx_* exports parsed from rsl_mlx.def"
    );

    let mut c = String::with_capacity(16 * 1024);
    c.push_str("/* AUTO-GENERATED by build.rs — no-op stub for non-Apple-Silicon\n");
    c.push_str(" * targets. Apple Metal / MLX only exists on aarch64-apple-darwin;\n");
    c.push_str(" * on Windows / Linux / Intel-macOS these no-op symbols satisfy the\n");
    c.push_str(" * FFI so the crate links. rsl_mlx_device_count() returns 0 → no MLX\n");
    c.push_str(" * device → the engine runs on CPU / SYCL / CUDA. */\n");
    for n in &names {
        c.push_str("long long ");
        c.push_str(n);
        c.push_str("(){return 0;}\n");
    }
    let stub_c = out_path.join("rsl_mlx_stub.c");
    std::fs::write(&stub_c, c).expect("write rsl_mlx_stub.c");

    // Compile the no-op symbols into a STATIC archive via the `cc` crate: for
    // an inert stub the symbols only need to resolve at link time, and `cc`
    // handles MSVC / clang-cl / clang / gcc detection + the link directives on
    // every host. No DLL/import-lib/.so/rpath/copy is needed — nothing ships
    // at runtime. (See the module doc for why static beats shared here.)
    cc::Build::new()
        .file(&stub_c)
        .opt_level(2)
        .warnings(false)
        .compile("rsl_mlx_stub");
}

/// Parse the `EXPORTS` section of a `.def` file, returning every `rsl_*`
/// symbol name in order. Comments (`;`) and the `LIBRARY`/`EXPORTS`
/// directives are skipped. Mirrors the SYCL stub's parser.
fn parse_def_exports(def_path: &str) -> Vec<String> {
    let def = std::fs::read_to_string(def_path)
        .unwrap_or_else(|e| panic!("rustllama-kernels-mlx build.rs: read {def_path}: {e}"));
    let mut names: Vec<String> = Vec::new();
    let mut in_exports = false;
    for line in def.lines() {
        let t = line.trim();
        if t.eq_ignore_ascii_case("EXPORTS") {
            in_exports = true;
            continue;
        }
        if !in_exports || t.is_empty() || t.starts_with(';') {
            continue;
        }
        if let Some(name) = t.split_whitespace().next() {
            if name.starts_with("rsl_") {
                names.push(name.to_string());
            }
        }
    }
    names
}

/// True if `<cmd> --version` runs and succeeds (i.e. the tool is on PATH).
fn command_ok(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Copy the freshly built shared library into `target/{profile}/`,
/// `.../examples/` and `.../deps/` so every binary this workspace produces
/// can resolve it at runtime (Windows' DLL search + Unix's `$ORIGIN` /
/// `@loader_path` rpath all start from the executable's directory).
///
/// OUT_DIR ≈ `<workspace>/target/{profile}/build/<crate>-<hash>/out`, so
/// the profile directory is OUT_DIR's 3rd ancestor (`out` → `<hash>` →
/// `build` → `{profile}`).
fn copy_shared_lib_next_to_binaries(out_path: &std::path::Path, name: &str) {
    let src = out_path.join(name);
    if let Some(profile_dir) = out_path.ancestors().nth(3) {
        for sub in &["", "examples", "deps"] {
            let dest_dir = profile_dir.join(sub);
            if !dest_dir.exists() {
                let _ = std::fs::create_dir_all(&dest_dir);
            }
            if dest_dir.exists() {
                let dest = dest_dir.join(name);
                if let Err(e) = std::fs::copy(&src, &dest) {
                    println!(
                        "cargo:warning=could not copy {name} to {}: {e}",
                        dest.display()
                    );
                }
            }
        }
    }
}
