//! Build script for the SYCL kernel translation unit.
//!
//! In `mock` mode (the default), this is a no-op so the crate compiles
//! without oneAPI installed.
//!
//! In `sycl` mode, we invoke `icx` (the Intel oneAPI compiler) directly
//! to build `cpp/rsl_kernels.cpp` into a `.dll`. The DLL approach is
//! REQUIRED rather than a static `.lib` because Rust's linker
//! (`link.exe`) doesn't understand SYCL's device-code embedding
//! scheme:
//!
//!   - SYCL emits per-kernel device-image descriptors + static
//!     initializers ("registers") in special object-file sections.
//!   - When `link.exe` links a static `.lib` into a Rust exe, those
//!     sections either get dropped or aren't finalized (no
//!     SYCL-aware link step runs over them), and at runtime every
//!     `q.submit` throws `std::bad_alloc` because the SYCL runtime
//!     can't find any device images registered under our kernel
//!     typeids.
//!   - `/WHOLEARCHIVE:` preserves the sections, but doesn't run the
//!     SYCL link finalizer — kernels get dispatched but the
//!     completion-event registry isn't wired, so `q.submit().wait()`
//!     hangs indefinitely.
//!
//! Building as a `.dll` linked by `icx` itself solves both: icx is
//! the SYCL-aware linker, it processes device images correctly, and
//! the resulting `.dll` is a normal Windows DLL that Rust can
//! dynamically link to.
//!
//! Build flow:
//!   1. `icx -fsycl -fsycl-targets=spir64 -shared cpp/rsl_kernels.cpp
//!       -o $OUT_DIR/rsl_kernels.dll`
//!   2. Tell cargo to link against the `.dll`'s import `.lib`.
//!   3. Copy the `.dll` into `target/{profile}/` so the produced exe
//!      can resolve it at runtime (Windows DLL search starts from
//!      the exe's directory).

use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Real-only crate: on x86_64 the SYCL backend is ALWAYS compiled from
    // the real C++ TU (requires Intel oneAPI `icx`/`icpx` on PATH).
    //
    // Intel oneAPI / SYCL is **x86_64-only** and **never existed on macOS**
    // (no icx/icpx for Apple, any arch). So on any non-x86 target (e.g. an
    // NVIDIA Grace / DGX Spark ARM box) AND on every Mac we instead build a
    // tiny no-op C stub that satisfies the `rsl_*` FFI symbols so the crate
    // LINKS, with `rsl_sycl_device_count()` returning 0 at runtime → the
    // engine sees no SYCL device and runs on CPU + whatever native GPU
    // backend the host has (CUDA on Linux/aarch64, MLX/Metal on Apple
    // Silicon). The Rust `imp` module is pure FFI (extern decls + safe
    // wrappers) and compiles unchanged on any architecture. Runtime
    // selection of SYCL vs CUDA vs MLX vs CPU happens in the engine's
    // startup device detection.
    let out_path = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR set by cargo"));

    println!("cargo:rerun-if-changed=cpp/rsl_kernels.def");
    println!("cargo:rerun-if-changed=build.rs");

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    // Stub the SYCL backend wherever Intel oneAPI cannot exist:
    //   * non-x86_64 (aarch64, …): oneAPI/SYCL is an x86_64-only toolchain.
    //   * macOS on ANY arch: Intel never shipped icx/icpx for macOS, and an
    //     Intel Mac (x86_64) therefore has no SYCL compiler either. Real
    //     compute on a Mac is CPU + the MLX/Metal backend (Apple Silicon);
    //     SYCL is always inert there. Gating macOS here keeps the real
    //     icx/icpx path exclusively on x86_64 Windows/Linux — the platforms
    //     the parent build-verifies — so those builds are byte-for-byte
    //     unchanged (the macOS branch is unreachable unless TARGET_OS==macos).
    if target_arch != "x86_64" || target_os == "macos" {
        build_sycl_stub(&out_path, &target_os, &target_arch);
        return;
    }

    // ---- x86_64: real SYCL C++ backend ----
    // Generate `iq_grids.inl` — C++ constexpr arrays for the IQ
    // codebook tables (IQ1S 2048×u64 grid + IQ2XXS 256×u64 grid +
    // sign-table helpers). The Rust source of truth lives in the
    // gguf crate (`iq1_grid::IQ1S_GRID`, `dequant::IQ2XXS_GRID`,
    // `dequant::KSIGNS_IQ2XS`, `dequant::KMASK_IQ2XS`); we extract
    // the numeric literals from those files (text munging — no
    // build dep on the gguf crate's Rust types) and emit them as
    // C++ arrays the kernel TU `#include`s. Regenerated whenever
    // either source file changes. Platform-independent.
    generate_iq_grids_inl(&out_path.join("iq_grids.inl"));
    println!("cargo:rerun-if-changed=../rustllama-gguf/src/iq1_grid.rs");
    println!("cargo:rerun-if-changed=../rustllama-gguf/src/dequant.rs");

    // Compile the kernel TU into a shared library using the SYCL-aware
    // compiler as its own linker (see module docs — the DLL/.so route is
    // required so SYCL's device-image registration survives). Windows
    // uses `icx` (MSVC-style driver → `.dll` + import lib); Linux uses
    // `icpx` (GCC-style driver → `.so`). (`target_os` computed above; this
    // point is only reached on x86_64 non-macOS, i.e. Windows or Linux.)
    if target_os == "windows" {
        build_windows(&out_path);
    } else {
        build_unix(&out_path);
    }

    println!("cargo:rerun-if-changed=cpp/rsl_kernels.cpp");
    println!("cargo:rerun-if-changed=cpp/rsl_xmx.hpp");
    println!("cargo:rerun-if-changed=include/rsl_kernels.h");
    println!("cargo:rerun-if-env-changed=ONEAPI_ROOT");
    println!("cargo:rerun-if-env-changed=CMPLR_ROOT");
}

/// Stub targets (non-x86_64, OR macOS on any arch): build a no-op C stub
/// that provides every `rsl_*` FFI symbol the Rust crate references, so it
/// links without Intel oneAPI. oneAPI/SYCL is x86_64-only AND has never
/// existed on macOS (no icx/icpx for Apple), so both the aarch64 hosts
/// (NVIDIA Grace / DGX Spark) and every Mac (Apple Silicon or Intel) take
/// this path. At runtime `rsl_sycl_device_count()` returns 0 → no SYCL
/// device → the engine runs on CPU + (per host) the native CUDA or MLX/Metal
/// backend. The symbol list is read from `cpp/rsl_kernels.def` (the
/// authoritative export set).
///
/// Each symbol is defined as `long long name() { return 0; }`: empty
/// parens accept any caller ABI, and a 0 integer/pointer return is
/// correct for the only functions actually invoked before device_count
/// reports zero devices (the device/stream probes — integer + pointer
/// returns). The kernel entry points are never called when there is no
/// device; they only need to resolve at link time.
///
/// The stub is emitted as a shared library named exactly like the real one
/// (`librsl_kernels.{so,dylib}`) so the crate's `dylib=rsl_kernels` link
/// directive + the copy-beside-binary + release-bundling machinery all apply
/// unchanged. The Mach-O vs ELF differences (see below) are the only reason
/// this branches on `target_os`.
fn build_sycl_stub(out_path: &std::path::Path, target_os: &str, target_arch: &str) {
    let def = std::fs::read_to_string("cpp/rsl_kernels.def")
        .expect("rustllama-kernels-sycl build.rs: read cpp/rsl_kernels.def");
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
    assert!(
        !names.is_empty(),
        "rustllama-kernels-sycl build.rs: no rsl_* exports parsed from rsl_kernels.def"
    );

    let mut c = String::with_capacity(16 * 1024);
    c.push_str("/* AUTO-GENERATED by build.rs — no-SYCL stub for targets with no\n");
    c.push_str(" * Intel oneAPI: every non-x86_64 arch (oneAPI is x86_64-only, e.g.\n");
    c.push_str(" * an NVIDIA Grace / DGX Spark aarch64 box) AND macOS on any arch\n");
    c.push_str(" * (Intel never shipped icx/icpx for Apple). These no-op symbols\n");
    c.push_str(" * satisfy the FFI so the crate links; rsl_sycl_device_count()\n");
    c.push_str(" * returns 0 → no SYCL device → the engine runs on CPU + the native\n");
    c.push_str(" * CUDA (Linux/aarch64) or MLX/Metal (Apple Silicon) backend. */\n");
    for n in &names {
        c.push_str("long long ");
        c.push_str(n);
        c.push_str("(){return 0;}\n");
    }
    let stub_c = out_path.join("rsl_kernels_stub.c");
    std::fs::write(&stub_c, c).expect("write rsl_kernels_stub.c");

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());

    if target_os == "macos" {
        // ---- Mach-O (macOS): emit a `.dylib`, not a `.so`. ----
        // Apple's linker (ld64) resolves a `dylib=rsl_kernels` directive to
        // `librsl_kernels.dylib` (or `.tbd`) and does NOT search for `.so`,
        // so the ELF path below would fail to link. Mach-O differences vs
        // ELF that this branch handles:
        //   * `-dynamiclib` (not `-shared`) to produce a Mach-O dylib.
        //   * `-install_name @rpath/librsl_kernels.dylib` so the dependent
        //     binary records an @rpath-relative reference; the binary's
        //     `@loader_path` rpath (added below) then resolves it beside
        //     itself — the Mach-O analogue of ELF `$ORIGIN`. release-macos.sh
        //     later rewrites this to `@loader_path/../lib` when it relocates
        //     the dylib into `lib/`.
        //   * `-arch arm64|x86_64` so a cross-arch build (`cargo build
        //     --target {aarch64,x86_64}-apple-darwin`) compiles the stub for
        //     the requested Mac arch, matching what rustc emits.
        let clang_arch = macos_clang_arch(target_arch);
        let dylib_path = out_path.join("librsl_kernels.dylib");
        let status = Command::new(&cc)
            .args(["-O2", "-arch", clang_arch, "-dynamiclib"])
            .args(["-install_name", "@rpath/librsl_kernels.dylib"])
            .arg(&stub_c)
            .arg("-o")
            .arg(&dylib_path)
            .status()
            .unwrap_or_else(|e| {
                panic!("rustllama-kernels-sycl build.rs: failed to invoke C compiler {cc:?} for the macOS SYCL stub: {e}")
            });
        if !status.success() {
            panic!("rustllama-kernels-sycl build.rs: {cc} failed to build the no-op SYCL stub .dylib");
        }
        println!("cargo:rustc-link-search=native={}", out_path.display());
        println!("cargo:rustc-link-lib=dylib=rsl_kernels");
        // Resolve `librsl_kernels.dylib` beside the executable at runtime
        // (Apple's analogue of Linux `$ORIGIN`).
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
        copy_shared_lib_next_to_binaries(out_path, "librsl_kernels.dylib");
        return;
    }

    // ---- ELF (non-x86_64 Linux, e.g. aarch64): emit a `.so`. ----
    // Named exactly like the real one so the same `dylib=rsl_kernels` link
    // directive + `$ORIGIN` rpath + copy-beside-binary machinery apply
    // unchanged.
    let so_path = out_path.join("librsl_kernels.so");
    let status = Command::new(&cc)
        .args(["-O2", "-fPIC", "-shared"])
        .arg(&stub_c)
        .arg("-o")
        .arg(&so_path)
        .status()
        .unwrap_or_else(|e| {
            panic!("rustllama-kernels-sycl build.rs: failed to invoke C compiler {cc:?} for the SYCL stub: {e}")
        });
    if !status.success() {
        panic!("rustllama-kernels-sycl build.rs: {cc} failed to build the no-op SYCL stub .so");
    }

    println!("cargo:rustc-link-search=native={}", out_path.display());
    println!("cargo:rustc-link-lib=dylib=rsl_kernels");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    copy_shared_lib_next_to_binaries(out_path, "librsl_kernels.so");
}

/// Map a Rust `CARGO_CFG_TARGET_ARCH` to the clang `-arch` name used on
/// macOS: Rust says `aarch64`, Apple's clang says `arm64`; `x86_64` is
/// spelled the same. Only meaningful on a macOS (Mach-O) target.
fn macos_clang_arch(target_arch: &str) -> &'static str {
    match target_arch {
        "aarch64" => "arm64",
        _ => "x86_64",
    }
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
/// `.../examples/` and `.../deps/` so every binary this workspace
/// produces can resolve it at runtime (Windows' DLL search + Linux's
/// `$ORIGIN` rpath both start from the executable's directory).
///
/// OUT_DIR ≈ `<workspace>/target/{profile}/build/<crate>-<hash>/out`,
/// so the profile directory is OUT_DIR's 3rd ancestor (`out` → `<hash>`
/// → `build` → `{profile}`).
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

/// Windows: build `rsl_kernels.dll` with `icx` (MSVC-style driver) and
/// link Rust against its import `.lib`. The `.def` file supplies the
/// exports (link.exe won't export our `extern "C"` symbols otherwise).
fn build_windows(out_path: &std::path::Path) {
    let oneapi_root = std::env::var("ONEAPI_ROOT").unwrap_or_else(|_| {
        std::env::var("ONEAPI_ROOT_DEFAULT")
            .unwrap_or_else(|_| "C:\\Program Files (x86)\\Intel\\oneAPI".into())
    });
    if !std::path::Path::new(&oneapi_root).exists() {
        panic!(
            "rustllama-kernels-sycl: ONEAPI_ROOT not found at {oneapi_root:?}. \
             Install Intel oneAPI Base Toolkit 2025.0 or set ONEAPI_ROOT."
        );
    }
    if !command_ok("icx") {
        panic!(
            "rustllama-kernels-sycl: `icx` not on PATH. Source the oneAPI \
             environment first (e.g. via `scripts\\build-env.bat`) and \
             rebuild. ONEAPI_ROOT was resolved to {oneapi_root:?}."
        );
    }

    let dll_path = out_path.join("rsl_kernels.dll");
    let imp_lib = out_path.join("rsl_kernels.lib");
    let pdb_path = out_path.join("rsl_kernels.pdb");

    // Compile + link to .dll in a single icx invocation. By acting as
    // both compiler and linker, icx runs its `clang-linker-wrapper`
    // device-code-link pass and embeds the SYCL kernels correctly into
    // the DLL's data sections + registers them via static initializers
    // that fire on DLL load. Use relative source paths — Windows
    // `\\?\`-prefixed paths from `fs::canonicalize` confuse icx/clang's
    // `#include` resolver; cargo runs the build script with cwd = the
    // crate root, so these resolve correctly.
    let status = Command::new("icx")
        .args([
            "-fsycl",
            "-fsycl-targets=spir64",
            "-O3",
            "/EHsc",
            "-MD",
            "-DRUSTLLAMA_HAS_SYCL",
            "-LD", // produce a .dll (MSVC-style: -LD = build dynamic library)
            "-Iinclude",
            "cpp/rsl_kernels.cpp",
        ])
        .args(xmx_defs())
        .arg(format!("-I{}", out_path.display()))
        .arg(format!("-Fe{}", dll_path.display()))
        .arg(format!("-Fo{}", out_path.join("rsl_kernels.obj").display()))
        // Linker options (icx forwards `-link` args to link.exe). The
        // `.def` file lists every extern "C" function we expose to Rust;
        // without it the DLL has no exports and Rust's link step fails
        // with LNK2019 on every kernel name.
        .arg("-link")
        .arg("/DEF:cpp/rsl_kernels.def")
        .arg(format!("/IMPLIB:{}", imp_lib.display()))
        .arg(format!("/PDB:{}", pdb_path.display()))
        .status()
        .expect("failed to invoke icx as compiler+linker");
    if !status.success() {
        panic!("icx failed to build rsl_kernels.dll");
    }

    println!("cargo:rustc-link-search=native={}", out_path.display());
    println!("cargo:rustc-link-lib=dylib=rsl_kernels");

    // Ensure the oneAPI compiler lib dir is on the search path so the
    // import lib's SYCL-runtime references resolve.
    let compiler_lib = format!("{oneapi_root}\\compiler\\latest\\lib");
    if std::path::Path::new(&compiler_lib).exists() {
        println!("cargo:rustc-link-search=native={compiler_lib}");
    } else if let Ok(entries) = std::fs::read_dir(format!("{oneapi_root}\\compiler")) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let lib = p.join("lib");
                if lib.exists() {
                    println!("cargo:rustc-link-search=native={}", lib.display());
                }
            }
        }
    }

    copy_shared_lib_next_to_binaries(out_path, "rsl_kernels.dll");
}

/// Always emit `-DRSL_SYCL_XMX` so the Intel XMX/DPAS `joint_matrix` bf16 GEMM
/// path in `cpp/rsl_xmx.hpp` is compiled into EVERY SYCL build and is available
/// for the runtime on-device self-check (`tune --validate-kernels`) to validate
/// — there is no build-time or runtime env gate. Safe on non-XMX dev GPUs (Iris
/// Xe, Xe-LP): the kernels target `spir64` (JIT SPIR-V, see `build_unix`/
/// `build_windows`), so the `joint_matrix` code lowers to generic SPIR-V and is
/// only JIT-compiled + launched when the device is XMX-capable AND the verdict
/// enables it; on Xe-LP it is never launched. The extern "C" entry's symbol set
/// / `.def` is unchanged (the `sk::xmx_available` probe still guards dispatch).
fn xmx_defs() -> Vec<&'static str> {
    vec!["-DRSL_SYCL_XMX"]
}

/// Linux: build `librsl_kernels.so` with `icpx` (GCC-style DPC++ driver;
/// falls back to `icx`) acting as its own SYCL-aware linker. `extern "C"`
/// symbols are exported by default under default visibility, so no `.def`
/// / version script is needed. An `$ORIGIN` rpath lets the produced
/// binary find the `.so` beside itself, matching Windows' DLL search.
fn build_unix(out_path: &std::path::Path) {
    let oneapi_root =
        std::env::var("ONEAPI_ROOT").unwrap_or_else(|_| "/opt/intel/oneapi".to_string());
    if !std::path::Path::new(&oneapi_root).exists() {
        panic!(
            "rustllama-kernels-sycl: ONEAPI_ROOT not found at {oneapi_root:?}. \
             Install Intel oneAPI Base Toolkit or set ONEAPI_ROOT (Linux \
             default /opt/intel/oneapi); source setvars.sh so icpx is on PATH."
        );
    }
    // Prefer the C++ driver `icpx`; fall back to `icx`.
    let cxx = if command_ok("icpx") {
        "icpx"
    } else if command_ok("icx") {
        "icx"
    } else {
        panic!(
            "rustllama-kernels-sycl: neither `icpx` nor `icx` on PATH. Source \
             the oneAPI environment (`. {oneapi_root}/setvars.sh`) and rebuild."
        );
    };

    let so_path = out_path.join("librsl_kernels.so");
    let status = Command::new(cxx)
        .args([
            "-fsycl",
            "-fsycl-targets=spir64",
            "-O3",
            "-fPIC",
            "-shared",
            "-fvisibility=default",
            "-DRUSTLLAMA_HAS_SYCL",
            "-Iinclude",
            "cpp/rsl_kernels.cpp",
        ])
        .args(xmx_defs())
        .arg(format!("-I{}", out_path.display()))
        .arg("-o")
        .arg(&so_path)
        .status()
        .unwrap_or_else(|e| panic!("failed to invoke {cxx} as compiler+linker: {e}"));
    if !status.success() {
        panic!("{cxx} failed to build librsl_kernels.so");
    }

    println!("cargo:rustc-link-search=native={}", out_path.display());
    println!("cargo:rustc-link-lib=dylib=rsl_kernels");

    // oneAPI runtime lib dir(s) (libsycl.so etc.) so the linker can
    // resolve our `.so`'s transitive SYCL-runtime references.
    for cand in [
        format!("{oneapi_root}/compiler/latest/lib"),
        format!("{oneapi_root}/compiler/latest/linux/lib"),
    ] {
        if std::path::Path::new(&cand).exists() {
            println!("cargo:rustc-link-search=native={cand}");
        }
    }
    // Resolve `librsl_kernels.so` beside the executable at runtime.
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");

    copy_shared_lib_next_to_binaries(out_path, "librsl_kernels.so");
}

/// Extract the IQ codebook tables from the gguf crate's Rust
/// sources and write them out as C++ `constexpr` arrays the SYCL
/// kernel TU can `#include`. Done via text munging (no Rust-level
/// dep on the gguf crate at build time — that would couple the two
/// crates' build orders messily).
///
/// Currently emits:
///   * `IQ1S_GRID_SYCL[2048]` (from `iq1_grid::IQ1S_GRID`)
///   * `IQ2XXS_GRID_SYCL[256]` (from `dequant::IQ2XXS_GRID`)
///   * `KSIGNS_IQ2XS_SYCL[128]` (from `dequant::KSIGNS_IQ2XS`)
///   * `KMASK_IQ2XS_SYCL[8]` (from `dequant::KMASK_IQ2XS`)
fn generate_iq_grids_inl(out_path: &std::path::Path) {
    let iq1_src = std::path::Path::new("../rustllama-gguf/src/iq1_grid.rs");
    let iq1_text = std::fs::read_to_string(iq1_src).unwrap_or_else(|e| {
        panic!("rustllama-kernels-sycl build.rs: failed to read {iq1_src:?}: {e}")
    });
    let dequant_src = std::path::Path::new("../rustllama-gguf/src/dequant.rs");
    let dequant_text = std::fs::read_to_string(dequant_src).unwrap_or_else(|e| {
        panic!("rustllama-kernels-sycl build.rs: failed to read {dequant_src:?}: {e}")
    });

    // IQ1S_GRID: `0x<hex>u64` literals, 2048 entries.
    let iq1s_grid = extract_suffixed_hex_u64s(&iq1_text);
    assert_eq!(
        iq1s_grid.len(),
        2048,
        "iq1_grid.rs: expected 2048 IQ1S_GRID entries, found {}",
        iq1s_grid.len()
    );

    // IQ2XXS_GRID: `0x<hex>,` literals in a `pub const IQ2XXS_GRID: [u64; 256] = [...]` block.
    let iq2xxs_grid = parse_const_array_hex_u64(&dequant_text, "IQ2XXS_GRID", 256);
    // IQ2XS_GRID: `pub const IQ2XS_GRID: [u64; 512] = [...]` block.
    let iq2xs_grid = parse_const_array_hex_u64(&dequant_text, "IQ2XS_GRID", 512);
    // IQ2S_GRID: `pub const IQ2S_GRID: [u64; 1024] = [...]` block.
    let iq2s_grid = parse_const_array_hex_u64(&dequant_text, "IQ2S_GRID", 1024);
    // IQ3XXS_GRID: `pub const IQ3XXS_GRID: [u32; 256] = [...]` block.
    let iq3xxs_grid = parse_const_array_hex_u32(&dequant_text, "IQ3XXS_GRID", 256);
    // IQ3S_GRID: `pub const IQ3S_GRID: [u32; 512] = [...]` block.
    let iq3s_grid = parse_const_array_hex_u32(&dequant_text, "IQ3S_GRID", 512);
    // KSIGNS_IQ2XS: decimal `u8` literals, 128 entries.
    let ksigns_iq2xs = parse_const_array_dec_u8(&dequant_text, "KSIGNS_IQ2XS", 128);
    // KMASK_IQ2XS: decimal `u8` literals, 8 entries.
    let kmask_iq2xs = parse_const_array_dec_u8(&dequant_text, "KMASK_IQ2XS", 8);

    let mut out = String::with_capacity(128 * 1024);
    out.push_str("// AUTO-GENERATED by rustllama-kernels-sycl/build.rs\n");
    out.push_str("// Do not edit by hand. Sources:\n");
    out.push_str("//   rustllama-gguf::iq1_grid::IQ1S_GRID\n");
    out.push_str("//   rustllama-gguf::dequant::{IQ2XXS_GRID, KSIGNS_IQ2XS, KMASK_IQ2XS}\n\n");
    out.push_str("#pragma once\n");
    out.push_str("#include <cstdint>\n\n");
    out.push_str("namespace rsl {\n");

    emit_u64_array(&mut out, "IQ1S_GRID_SYCL", 2048, &iq1s_grid);
    emit_u64_array(&mut out, "IQ2XXS_GRID_SYCL", 256, &iq2xxs_grid);
    emit_u64_array(&mut out, "IQ2XS_GRID_SYCL", 512, &iq2xs_grid);
    emit_u64_array(&mut out, "IQ2S_GRID_SYCL", 1024, &iq2s_grid);
    emit_u32_array(&mut out, "IQ3XXS_GRID_SYCL", 256, &iq3xxs_grid);
    emit_u32_array(&mut out, "IQ3S_GRID_SYCL", 512, &iq3s_grid);
    emit_u8_array(&mut out, "KSIGNS_IQ2XS_SYCL", 128, &ksigns_iq2xs);
    emit_u8_array(&mut out, "KMASK_IQ2XS_SYCL", 8, &kmask_iq2xs);

    out.push_str("} // namespace rsl\n");
    std::fs::write(out_path, out).expect("write iq_grids.inl");
}

/// Scan a Rust source file for every `0x<hex>u64` literal and
/// return the parsed u64 values in encounter order. Used to read
/// the IQ1S_GRID block which has its entries written with the
/// explicit `u64` suffix on each literal.
fn extract_suffixed_hex_u64s(src: &str) -> Vec<u64> {
    let mut out = Vec::new();
    for line in src.lines() {
        let line = line.split("//").next().unwrap_or(line);
        let mut rest = line;
        while let Some(start) = rest.find("0x") {
            rest = &rest[start + 2..];
            let end = rest
                .find(|c: char| !c.is_ascii_hexdigit())
                .unwrap_or(rest.len());
            if end == 0 {
                continue;
            }
            let hex = &rest[..end];
            let after = &rest[end..];
            if after.starts_with("u64") {
                out.push(u64::from_str_radix(hex, 16).expect("hex u64"));
            }
            rest = &rest[end..];
        }
    }
    out
}

/// Find a `pub const NAME: ... = [ ... ];` block and parse all
/// `0x<hex>` literals inside the array body as u64. Suffix is
/// optional (handles both `0xABCu64,` and `0xABC,` styles).
fn parse_const_array_hex_u64(src: &str, name: &str, expected: usize) -> Vec<u64> {
    let body = extract_const_array_body(src, name)
        .unwrap_or_else(|| panic!("could not locate `pub const {name}` array body"));
    let mut out = Vec::new();
    let mut rest = body.as_str();
    while let Some(start) = rest.find("0x") {
        rest = &rest[start + 2..];
        let end = rest
            .find(|c: char| !c.is_ascii_hexdigit())
            .unwrap_or(rest.len());
        if end == 0 {
            continue;
        }
        let hex = &rest[..end];
        out.push(u64::from_str_radix(hex, 16).expect("hex u64"));
        rest = &rest[end..];
    }
    assert_eq!(
        out.len(),
        expected,
        "{name}: expected {expected} entries, found {}",
        out.len()
    );
    out
}

/// Same shape as `parse_const_array_hex_u64`, but for `[u32; N]`
/// arrays (IQ3 codebooks store 4 packed u8 grid coords per entry,
/// vs IQ2's 8).
fn parse_const_array_hex_u32(src: &str, name: &str, expected: usize) -> Vec<u32> {
    let body = extract_const_array_body(src, name)
        .unwrap_or_else(|| panic!("could not locate `pub const {name}` array body"));
    let mut out = Vec::new();
    let mut rest = body.as_str();
    while let Some(start) = rest.find("0x") {
        rest = &rest[start + 2..];
        let end = rest
            .find(|c: char| !c.is_ascii_hexdigit())
            .unwrap_or(rest.len());
        if end == 0 {
            continue;
        }
        let hex = &rest[..end];
        out.push(u32::from_str_radix(hex, 16).expect("hex u32"));
        rest = &rest[end..];
    }
    assert_eq!(
        out.len(),
        expected,
        "{name}: expected {expected} entries, found {}",
        out.len()
    );
    out
}

/// Find a `pub const NAME: ... = [ ... ];` block and parse all
/// decimal integer literals inside the array body as u8. Comments
/// stripped per line. Used for the KSIGNS_IQ2XS / KMASK_IQ2XS
/// sign-mask tables.
fn parse_const_array_dec_u8(src: &str, name: &str, expected: usize) -> Vec<u8> {
    let body = extract_const_array_body(src, name)
        .unwrap_or_else(|| panic!("could not locate `pub const {name}` array body"));
    let mut cleaned = String::with_capacity(body.len());
    for line in body.lines() {
        let line = line.split("//").next().unwrap_or(line);
        cleaned.push_str(line);
        cleaned.push('\n');
    }
    let out: Vec<u8> = cleaned
        .split(|c: char| !c.is_ascii_digit())
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<u32>().expect("dec u8") as u8)
        .collect();
    assert_eq!(
        out.len(),
        expected,
        "{name}: expected {expected} entries, found {}",
        out.len()
    );
    out
}

/// Locate `pub const NAME ... = [ ... ];` and return the array
/// body (text between the `= [` and the matching `]`). Bracket
/// counting tracks nested `[...]` inside the body (none in our
/// current sources, but cheap insurance).
fn extract_const_array_body(src: &str, name: &str) -> Option<String> {
    let needle = format!("pub const {name}");
    let start = src.find(&needle)?;
    let after = &src[start..];
    let eq_bracket = after.find("= [")?;
    let body_start = start + eq_bracket + 3;
    let rest = &src[body_start..];
    let mut depth = 1;
    for (i, c) in rest.char_indices() {
        if c == '[' {
            depth += 1;
        } else if c == ']' {
            depth -= 1;
            if depth == 0 {
                return Some(rest[..i].to_string());
            }
        }
    }
    None
}

fn emit_u64_array(out: &mut String, name: &str, expected: usize, values: &[u64]) {
    assert_eq!(values.len(), expected);
    out.push_str(&format!(
        "constexpr std::uint64_t {name}[{expected}] = {{\n"
    ));
    for (i, v) in values.iter().enumerate() {
        if i % 4 == 0 {
            out.push_str("    ");
        }
        out.push_str(&format!("0x{:016x}ULL,", v));
        if i % 4 == 3 {
            out.push('\n');
        } else {
            out.push(' ');
        }
    }
    if values.len() % 4 != 0 {
        out.push('\n');
    }
    out.push_str("};\n\n");
}

fn emit_u32_array(out: &mut String, name: &str, expected: usize, values: &[u32]) {
    assert_eq!(values.len(), expected);
    out.push_str(&format!(
        "constexpr std::uint32_t {name}[{expected}] = {{\n"
    ));
    for (i, v) in values.iter().enumerate() {
        if i % 8 == 0 {
            out.push_str("    ");
        }
        out.push_str(&format!("0x{:08x}u,", v));
        if i % 8 == 7 {
            out.push('\n');
        } else {
            out.push(' ');
        }
    }
    if values.len() % 8 != 0 {
        out.push('\n');
    }
    out.push_str("};\n\n");
}

fn emit_u8_array(out: &mut String, name: &str, expected: usize, values: &[u8]) {
    assert_eq!(values.len(), expected);
    out.push_str(&format!(
        "constexpr std::uint8_t {name}[{expected}] = {{\n"
    ));
    for (i, v) in values.iter().enumerate() {
        if i % 16 == 0 {
            out.push_str("    ");
        }
        out.push_str(&format!("{},", v));
        if i % 16 == 15 {
            out.push('\n');
        } else {
            out.push(' ');
        }
    }
    if values.len() % 16 != 0 {
        out.push('\n');
    }
    out.push_str("};\n\n");
}
