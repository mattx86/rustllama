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
