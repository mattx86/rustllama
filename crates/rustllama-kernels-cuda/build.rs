//! Build script for the CUDA kernel translation unit.
//!
//! This crate always compiles the real kernels (no mock) — it's an
//! optional `cuda`-backend dependency, so it's only ever built where the
//! CUDA toolkit is present. `nvcc` compiles `cuda/rsl_cuda.cu` into a
//! STATIC library linked into the binary, together with the static CUDA
//! runtime (`cudart_static`). The device code (PTX/cubin fatbin) is
//! embedded in the object, so the kernels are "compiled in"; the only
//! remaining runtime dependency is the NVIDIA driver (`libcuda`), which is
//! present on any NVIDIA host. `--whole-archive` preserves nvcc's fatbin
//! registration initializers, which a plain static link would drop (the
//! same class of problem the SYCL crate solves with a DLL).
//!
//! Requires the CUDA Toolkit (`nvcc`) on PATH, or `CUDACXX`/`CUDA_PATH`
//! set. Target GPU architectures come from `RUSTLLAMA_CUDA_ARCHS`
//! (e.g. "80;86;89;90"), defaulting to Ampere→Hopper.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = manifest.join("cuda").join("rsl_cuda.cu");
    let inc = manifest.join("include");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-changed={}", inc.join("rsl_cuda.h").display());
    println!("cargo:rerun-if-env-changed=RUSTLLAMA_CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDACXX");

    // macOS: there is NO CUDA on Apple. NVIDIA dropped macOS CUDA support
    // around 2019 (last toolkit 10.2 / macOS 10.13) and never shipped it for
    // Apple Silicon, so `nvcc` does not exist on any Mac. Build an inert
    // no-op STATIC stub instead of invoking nvcc: every `rsl_cuda_*` FFI
    // symbol resolves as `long long name(){return 0;}`, so the crate links
    // and rsl_cuda_device_count() returns 0 → the engine sees no NVIDIA
    // device and runs on CPU + (Apple Silicon) the MLX/Metal backend. Gated
    // strictly on TARGET_OS==macos so the real nvcc path on Windows/Linux is
    // byte-for-byte unchanged (this branch is unreachable off macOS). Mirrors
    // the SYCL/MLX stub discipline. Returns early — no nvcc, no iq_grids
    // staging, no cudart/whole-archive/stdc++ link directives.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        build_cuda_stub(&manifest, &out);
        return;
    }

    // Generate `iq_grids_cuda.inl` — `__device__ const` arrays for the IQ
    // codebook tables (grids + sign helpers) the packed-IQ matvec kernels
    // read on-device. The Rust source of truth lives in the gguf crate
    // (`iq1_grid::IQ1S_GRID`, `dequant::{IQ2XXS_GRID, IQ2XS_GRID, IQ2S_GRID,
    // IQ3XXS_GRID, IQ3S_GRID, KSIGNS_IQ2XS, KMASK_IQ2XS}`); we extract the
    // numeric literals by text munging (no build dep on the gguf crate's
    // Rust types) and emit them as CUDA `__device__ const` arrays the kernel
    // TU `#include`s. This mirrors the SYCL crate's `iq_grids.inl` staging
    // (kernels-sycl/build.rs) — same source, same values, CUDA storage
    // qualifier. Regenerated whenever either source file changes.
    generate_iq_grids_cuda_inl(&out.join("iq_grids_cuda.inl"));
    println!("cargo:rerun-if-changed=../rustllama-gguf/src/iq1_grid.rs");
    println!("cargo:rerun-if-changed=../rustllama-gguf/src/dequant.rs");

    let windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    let nvcc = find_nvcc(windows);

    // gencode flags for the requested SM architectures.
    // Ampere → Hopper by default (Turing sm_75 also fine on CUDA 12/13).
    // Add Blackwell (100;120) via RUSTLLAMA_CUDA_ARCHS on CUDA 12.8+/13.
    let archs = std::env::var("RUSTLLAMA_CUDA_ARCHS")
        .unwrap_or_else(|_| "80;86;89;90".to_string());
    let arch_list: Vec<&str> = archs.split(';').filter(|s| !s.is_empty()).collect();
    let mut cmd = Command::new(&nvcc);
    cmd.arg("-c").arg(&src);
    cmd.arg("-O3").arg("--std=c++17");
    cmd.arg(format!("-I{}", inc.display()));
    // OUT_DIR holds the generated `iq_grids_cuda.inl` the kernel TU includes.
    cmd.arg(format!("-I{}", out.display()));
    for a in &arch_list {
        cmd.arg(format!("-gencode=arch=compute_{a},code=sm_{a}"));
    }
    // Also embed PTX for the HIGHEST requested arch. `code=sm_X` bakes in
    // SASS only, which the driver will NOT run on a newer GPU — so a build
    // for sm_80..90 fails on Blackwell (e.g. the GB10 in an NVIDIA DGX
    // Spark) with "no kernel image is available". Adding `code=compute_X`
    // embeds forward-compatible PTX the driver JIT-compiles for any newer
    // architecture (a one-time JIT cost at first load). Set
    // RUSTLLAMA_CUDA_ARCHS to the target's native SM for best perf.
    if let Some(max) = arch_list
        .iter()
        .max_by_key(|a| a.parse::<u32>().unwrap_or(0))
    {
        cmd.arg(format!("-gencode=arch=compute_{max},code=compute_{max}"));
    }
    // PIC on Linux; static MSVC CRT-agnostic on Windows (host code is tiny).
    if !windows {
        cmd.arg("--compiler-options").arg("-fPIC");
    }
    let obj = out.join(if windows { "rsl_cuda.obj" } else { "rsl_cuda.o" });
    cmd.arg("-o").arg(&obj);

    let status = cmd.status().unwrap_or_else(|e| {
        panic!(
            "rustllama-kernels-cuda: failed to run nvcc ({nvcc:?}): {e}. \
             Install the CUDA Toolkit or set CUDACXX to the nvcc path."
        )
    });
    assert!(status.success(), "nvcc failed to compile {}", src.display());

    // Archive the object into a static lib cargo can link.
    let ar = if windows { "lib" } else { "ar" };
    let lib = out.join(if windows { "rsl_cuda.lib" } else { "librsl_cuda.a" });
    let ar_status = if windows {
        Command::new(ar)
            .arg(format!("/OUT:{}", lib.display()))
            .arg(&obj)
            .status()
    } else {
        Command::new(ar).arg("crs").arg(&lib).arg(&obj).status()
    };
    let ok = ar_status.map(|s| s.success()).unwrap_or(false);
    assert!(ok, "failed to archive rsl_cuda object into {}", lib.display());

    println!("cargo:rustc-link-search=native={}", out.display());
    // whole-archive so nvcc's fatbin registration ctors survive the link.
    if windows {
        println!("cargo:rustc-link-arg=/WHOLEARCHIVE:rsl_cuda");
        println!("cargo:rustc-link-lib=static=rsl_cuda");
    } else {
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!("cargo:rustc-link-lib=static=rsl_cuda");
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
    }

    // Static CUDA runtime + its deps, so only the NVIDIA driver is dynamic.
    if let Some(dir) = cuda_lib_dir() {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    if windows {
        println!("cargo:rustc-link-lib=static=cudart_static");
    } else {
        println!("cargo:rustc-link-lib=static=cudart_static");
        // cudart_static's dependencies on glibc/pthread/rt/dl.
        for l in ["culibos", "pthread", "rt", "dl"] {
            println!("cargo:rustc-link-lib=dylib={l}");
        }
        // nvcc's HOST object references the C++ runtime: `operator
        // new/delete` (the stream ctor uses `new (std::nothrow)`),
        // exception personality (`__gxx_personality_v0`) and static-init
        // guards (`__cxa_guard_*`). Rust links via the C driver (`cc`),
        // which doesn't pull in libstdc++, so link it explicitly.
        // Without this a HEADLESS Linux binary fails at link with
        // undefined C++ symbols — a GUI build only linked because
        // GTK/webkit dragged libstdc++ in transitively. Same on aarch64.
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

/// macOS: build an inert no-op STATIC archive that resolves every CUDA FFI
/// symbol the Rust crate references, so the crate links without `nvcc`
/// (which does not exist on macOS — NVIDIA dropped Mac CUDA ~2019 and never
/// targeted Apple Silicon). `rsl_cuda_device_count()` returns 0 at runtime →
/// no NVIDIA device → the engine runs on CPU + the MLX/Metal backend. This
/// mirrors the SYCL crate's `build_sycl_stub` and the MLX crate's
/// `build_mlx_stub`.
///
/// The authoritative symbol set is the `extern "C"` block in this crate's
/// `src/lib.rs`. The crate has no `.def`, and the C header lists the packed
/// matvecs through an `RSL_CUDA_DECL_PACKED(...)` macro, so the header can't
/// be scraped literally — the Rust extern declarations are the exact,
/// fully-expanded set of names that must resolve at link. Each is emitted as
/// `long long name(){return 0;}`: empty parens accept any caller ABI, and a 0
/// return is correct for the probes actually called before device_count
/// reports zero (integer/pointer returns); the kernel entry points are never
/// invoked when there is no device and only need to resolve at link time.
///
/// Built as a STATIC archive — compiled with `cc` (clang on macOS) and
/// archived with `ar crs`, the same tools the real nvcc path uses to archive
/// `rsl_cuda.o` — so nothing ships at runtime; the stub links straight into
/// the binary. Unlike the real path it needs NO `--whole-archive` (there are
/// no fatbin registration ctors to preserve), NO `cudart_static`, and NO
/// `libstdc++` (it is plain C).
fn build_cuda_stub(manifest: &std::path::Path, out: &std::path::Path) {
    let lib_rs = manifest.join("src").join("lib.rs");
    println!("cargo:rerun-if-changed={}", lib_rs.display());
    let names = parse_extern_rsl_cuda_fns(&lib_rs);
    assert!(
        !names.is_empty(),
        "rustllama-kernels-cuda build.rs: no `fn rsl_cuda_*` externs parsed from src/lib.rs"
    );

    let mut c = String::with_capacity(16 * 1024);
    c.push_str("/* AUTO-GENERATED by build.rs — no-CUDA stub for macOS.\n");
    c.push_str(" * NVIDIA CUDA does not exist on Apple (nvcc dropped macOS ~2019 and\n");
    c.push_str(" * never targeted Apple Silicon). These no-op symbols satisfy the FFI\n");
    c.push_str(" * so the crate links; rsl_cuda_device_count() returns 0 → no NVIDIA\n");
    c.push_str(" * device → the engine runs on CPU + the MLX/Metal backend. */\n");
    for n in &names {
        c.push_str("long long ");
        c.push_str(n);
        c.push_str("(){return 0;}\n");
    }
    let stub_c = out.join("rsl_cuda_stub.c");
    std::fs::write(&stub_c, &c).expect("write rsl_cuda_stub.c");

    // Compile the no-op TU, then archive it. macOS `cc` is clang and `ar`
    // supports `crs`, matching the non-Windows archiving of the real path.
    // `-arch arm64|x86_64` lets a cross-arch build (`cargo build --target
    // {aarch64,x86_64}-apple-darwin`) compile the stub for the requested Mac
    // arch, matching what rustc emits (Rust says `aarch64`, clang says
    // `arm64`; `x86_64` is spelled the same).
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let clang_arch = if target_arch == "aarch64" { "arm64" } else { "x86_64" };
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let ar = std::env::var("AR").unwrap_or_else(|_| "ar".to_string());

    let obj = out.join("rsl_cuda_stub.o");
    let status = Command::new(&cc)
        .args(["-O2", "-arch", clang_arch, "-c"])
        .arg(&stub_c)
        .arg("-o")
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| {
            panic!("rustllama-kernels-cuda build.rs: failed to invoke C compiler {cc:?} for the macOS CUDA stub: {e}")
        });
    assert!(
        status.success(),
        "rustllama-kernels-cuda build.rs: {cc} failed to compile the no-op CUDA stub"
    );

    let lib = out.join("librsl_cuda_stub.a");
    // `ar crs` APPENDS to an existing archive, so remove a stale one first to
    // keep incremental rebuilds clean.
    let _ = std::fs::remove_file(&lib);
    let ar_status = Command::new(&ar)
        .arg("crs")
        .arg(&lib)
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| {
            panic!("rustllama-kernels-cuda build.rs: failed to invoke {ar:?} to archive the CUDA stub: {e}")
        });
    assert!(
        ar_status.success(),
        "rustllama-kernels-cuda build.rs: {ar} failed to archive the CUDA stub into {}",
        lib.display()
    );

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=rsl_cuda_stub");
}

/// Parse this crate's `extern "C"` FFI block and return every `rsl_cuda_*`
/// function name it declares, in source order. Each declaration line reads
/// `    fn rsl_cuda_NAME(...`; the safe wrappers below the block are named
/// `device_count`/`device_info`/… (no `rsl_` prefix) and *call*, not declare,
/// the externs, so a leading `fn rsl_cuda_` token is an unambiguous filter
/// (a `pub fn` wrapper would trim-start with `pub`, not `fn`). Used only by
/// the macOS stub.
fn parse_extern_rsl_cuda_fns(lib_rs: &std::path::Path) -> Vec<String> {
    let text = std::fs::read_to_string(lib_rs)
        .unwrap_or_else(|e| panic!("rustllama-kernels-cuda build.rs: read {lib_rs:?}: {e}"));
    let mut names = Vec::new();
    for line in text.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("fn rsl_cuda_") {
            // `rest` = "NAME(...) -> ...;" → take up to the first char that
            // can't be part of a Rust identifier.
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            if end > 0 {
                names.push(format!("rsl_cuda_{}", &rest[..end]));
            }
        }
    }
    names
}

/// Resolve the `nvcc` compiler. `CUDACXX` wins; otherwise prefer the
/// `bin/nvcc` under `CUDA_PATH`/`CUDA_HOME`/`/usr/local/cuda` (robust when
/// the toolkit's `bin` isn't on the build subprocess's PATH yet — e.g. a
/// fresh `winget install Nvidia.CUDA` before a shell restart); finally
/// fall back to a bare `nvcc` on PATH.
fn find_nvcc(windows: bool) -> String {
    if let Ok(cc) = std::env::var("CUDACXX") {
        return cc;
    }
    let exe = if windows { "nvcc.exe" } else { "nvcc" };
    let roots = std::env::var("CUDA_PATH")
        .into_iter()
        .chain(std::env::var("CUDA_HOME"))
        .map(PathBuf::from)
        .chain(std::iter::once(PathBuf::from("/usr/local/cuda")));
    for root in roots {
        let p = root.join("bin").join(exe);
        if p.exists() {
            return p.to_string_lossy().into_owned();
        }
    }
    "nvcc".to_string()
}

/// Best-effort CUDA `lib`/`lib64` directory from `CUDA_PATH`/`CUDA_HOME`
/// or the conventional install location.
fn cuda_lib_dir() -> Option<PathBuf> {
    let root = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            let d = PathBuf::from("/usr/local/cuda");
            d.exists().then_some(d)
        })?;
    for sub in ["lib64", "lib/x64", "lib"] {
        let p = root.join(sub);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Extract the IQ codebook tables from the gguf crate's Rust sources and
/// write them out as CUDA `__device__ const` arrays the kernel TU can
/// `#include`. Done via text munging (no Rust-level dep on the gguf crate
/// at build time). This is the CUDA analogue of the SYCL crate's
/// `generate_iq_grids_inl`; the numeric values are identical — only the
/// storage qualifier (`__device__ const` vs SYCL `constexpr`) and the
/// `_CUDA` name suffix differ. The kernels read these directly from device
/// global memory (read-only cached), like ggml-cuda's `iq*_grid` tables.
///
/// Emits (in `namespace rsl`):
///   * `IQ1S_GRID_CUDA[2048]`  (u64, from `iq1_grid::IQ1S_GRID`)
///   * `IQ2XXS_GRID_CUDA[256]` (u64, from `dequant::IQ2XXS_GRID`)
///   * `IQ2XS_GRID_CUDA[512]`  (u64, from `dequant::IQ2XS_GRID`)
///   * `IQ2S_GRID_CUDA[1024]`  (u64, from `dequant::IQ2S_GRID`)
///   * `IQ3XXS_GRID_CUDA[256]` (u32, from `dequant::IQ3XXS_GRID`)
///   * `IQ3S_GRID_CUDA[512]`   (u32, from `dequant::IQ3S_GRID`)
///   * `KSIGNS_IQ2XS_CUDA[128]`(u8,  from `dequant::KSIGNS_IQ2XS`)
///   * `KMASK_IQ2XS_CUDA[8]`   (u8,  from `dequant::KMASK_IQ2XS`)
fn generate_iq_grids_cuda_inl(out_path: &std::path::Path) {
    let iq1_src = std::path::Path::new("../rustllama-gguf/src/iq1_grid.rs");
    let iq1_text = std::fs::read_to_string(iq1_src)
        .unwrap_or_else(|e| panic!("rustllama-kernels-cuda build.rs: failed to read {iq1_src:?}: {e}"));
    let dequant_src = std::path::Path::new("../rustllama-gguf/src/dequant.rs");
    let dequant_text = std::fs::read_to_string(dequant_src)
        .unwrap_or_else(|e| panic!("rustllama-kernels-cuda build.rs: failed to read {dequant_src:?}: {e}"));

    // IQ1S_GRID: `0x<hex>u64` literals, 2048 entries.
    let iq1s_grid = extract_suffixed_hex_u64s(&iq1_text);
    assert_eq!(
        iq1s_grid.len(),
        2048,
        "iq1_grid.rs: expected 2048 IQ1S_GRID entries, found {}",
        iq1s_grid.len()
    );

    let iq2xxs_grid = parse_const_array_hex_u64(&dequant_text, "IQ2XXS_GRID", 256);
    let iq2xs_grid = parse_const_array_hex_u64(&dequant_text, "IQ2XS_GRID", 512);
    let iq2s_grid = parse_const_array_hex_u64(&dequant_text, "IQ2S_GRID", 1024);
    let iq3xxs_grid = parse_const_array_hex_u32(&dequant_text, "IQ3XXS_GRID", 256);
    let iq3s_grid = parse_const_array_hex_u32(&dequant_text, "IQ3S_GRID", 512);
    let ksigns_iq2xs = parse_const_array_dec_u8(&dequant_text, "KSIGNS_IQ2XS", 128);
    let kmask_iq2xs = parse_const_array_dec_u8(&dequant_text, "KMASK_IQ2XS", 8);

    let mut out = String::with_capacity(128 * 1024);
    out.push_str("// AUTO-GENERATED by rustllama-kernels-cuda/build.rs\n");
    out.push_str("// Do not edit by hand. Sources:\n");
    out.push_str("//   rustllama-gguf::iq1_grid::IQ1S_GRID\n");
    out.push_str("//   rustllama-gguf::dequant::{IQ2XXS_GRID, IQ2XS_GRID, IQ2S_GRID,\n");
    out.push_str("//     IQ3XXS_GRID, IQ3S_GRID, KSIGNS_IQ2XS, KMASK_IQ2XS}\n");
    out.push_str("// Read on-device by the packed-IQ matvec kernels in rsl_cuda.cu.\n\n");
    out.push_str("#pragma once\n");
    out.push_str("#include <cstdint>\n\n");
    out.push_str("namespace rsl {\n");

    emit_u64_array(&mut out, "IQ1S_GRID_CUDA", 2048, &iq1s_grid);
    emit_u64_array(&mut out, "IQ2XXS_GRID_CUDA", 256, &iq2xxs_grid);
    emit_u64_array(&mut out, "IQ2XS_GRID_CUDA", 512, &iq2xs_grid);
    emit_u64_array(&mut out, "IQ2S_GRID_CUDA", 1024, &iq2s_grid);
    emit_u32_array(&mut out, "IQ3XXS_GRID_CUDA", 256, &iq3xxs_grid);
    emit_u32_array(&mut out, "IQ3S_GRID_CUDA", 512, &iq3s_grid);
    emit_u8_array(&mut out, "KSIGNS_IQ2XS_CUDA", 128, &ksigns_iq2xs);
    emit_u8_array(&mut out, "KMASK_IQ2XS_CUDA", 8, &kmask_iq2xs);

    out.push_str("} // namespace rsl\n");
    std::fs::write(out_path, out).expect("write iq_grids_cuda.inl");
}

/// Scan a Rust source file for every `0x<hex>u64` literal and return the
/// parsed values in encounter order (IQ1S_GRID uses the explicit `u64`
/// suffix per literal). Comments stripped per line.
fn extract_suffixed_hex_u64s(src: &str) -> Vec<u64> {
    let mut out = Vec::new();
    for line in src.lines() {
        let line = line.split("//").next().unwrap_or(line);
        let mut rest = line;
        while let Some(start) = rest.find("0x") {
            rest = &rest[start + 2..];
            let end = rest.find(|c: char| !c.is_ascii_hexdigit()).unwrap_or(rest.len());
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

/// Find `pub const NAME: ... = [ ... ];` and parse all `0x<hex>` literals
/// inside the array body as u64 (suffix optional).
fn parse_const_array_hex_u64(src: &str, name: &str, expected: usize) -> Vec<u64> {
    let body = extract_const_array_body(src, name)
        .unwrap_or_else(|| panic!("could not locate `pub const {name}` array body"));
    let mut out = Vec::new();
    let mut rest = body.as_str();
    while let Some(start) = rest.find("0x") {
        rest = &rest[start + 2..];
        let end = rest.find(|c: char| !c.is_ascii_hexdigit()).unwrap_or(rest.len());
        if end == 0 {
            continue;
        }
        out.push(u64::from_str_radix(&rest[..end], 16).expect("hex u64"));
        rest = &rest[end..];
    }
    assert_eq!(out.len(), expected, "{name}: expected {expected} entries, found {}", out.len());
    out
}

/// Same as `parse_const_array_hex_u64` but for `[u32; N]` arrays.
fn parse_const_array_hex_u32(src: &str, name: &str, expected: usize) -> Vec<u32> {
    let body = extract_const_array_body(src, name)
        .unwrap_or_else(|| panic!("could not locate `pub const {name}` array body"));
    let mut out = Vec::new();
    let mut rest = body.as_str();
    while let Some(start) = rest.find("0x") {
        rest = &rest[start + 2..];
        let end = rest.find(|c: char| !c.is_ascii_hexdigit()).unwrap_or(rest.len());
        if end == 0 {
            continue;
        }
        out.push(u32::from_str_radix(&rest[..end], 16).expect("hex u32"));
        rest = &rest[end..];
    }
    assert_eq!(out.len(), expected, "{name}: expected {expected} entries, found {}", out.len());
    out
}

/// Find `pub const NAME: ... = [ ... ];` and parse all decimal integer
/// literals inside the array body as u8 (comments stripped per line).
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
    assert_eq!(out.len(), expected, "{name}: expected {expected} entries, found {}", out.len());
    out
}

/// Locate `pub const NAME ... = [ ... ];` and return the array body (text
/// between `= [` and the matching `]`), bracket-counting nested `[...]`.
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
    out.push_str(&format!("__device__ const std::uint64_t {name}[{expected}] = {{\n"));
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
    out.push_str(&format!("__device__ const std::uint32_t {name}[{expected}] = {{\n"));
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
    out.push_str(&format!("__device__ const std::uint8_t {name}[{expected}] = {{\n"));
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
