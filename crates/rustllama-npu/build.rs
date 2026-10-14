// Build script for `rustllama-npu`.
//
// Does real work ONLY under the `npu-openvino` feature: it links the OpenVINO C
// API import library (`openvino_c`) from an installed OpenVINO SDK so the Intel
// NPU enumeration FFI in `src/openvino.rs` resolves at link time. The SDK root
// is located via `OPENVINO_DIR` or `INTEL_OPENVINO_DIR` (the latter is what the
// SDK's `setupvars` script exports).
//
// With NO `npu-*` feature the script is a no-op, so the DEFAULT build needs no
// SDK and stays toolchain-free (CI-compilable). The vendor backend is opt-in,
// mirroring how the CUDA/SYCL kernel crates are real-only but inert off their
// hardware — here the vendor runtime is pulled in only when explicitly enabled.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "npu-openvino")]
    link_openvino();
}

#[cfg(feature = "npu-openvino")]
fn link_openvino() {
    use std::path::{Path, PathBuf};

    println!("cargo:rerun-if-env-changed=OPENVINO_DIR");
    println!("cargo:rerun-if-env-changed=INTEL_OPENVINO_DIR");

    let root = std::env::var("OPENVINO_DIR")
        .or_else(|_| std::env::var("INTEL_OPENVINO_DIR"))
        .unwrap_or_else(|_| {
            panic!(
                "the `npu-openvino` feature requires the OpenVINO SDK: set \
                 OPENVINO_DIR (or INTEL_OPENVINO_DIR) to the toolkit root (the \
                 directory that contains `runtime/`, i.e. the folder holding \
                 setupvars.bat / setupvars.sh)."
            )
        });
    let root = PathBuf::from(root);

    // Accept either the toolkit root (…/openvino_toolkit_*) or the `runtime`
    // dir itself. The C API lib + headers live under `runtime/`.
    let runtime = if root.join("runtime").is_dir() {
        root.join("runtime")
    } else {
        root.clone()
    };

    // Candidate per-OS lib dirs, most-specific first. Windows ships
    // Release/Debug split import libs under lib/intel64/<cfg>; Linux/macOS put
    // the shared object directly under lib/intel64 (or lib/). We add every
    // candidate that exists to the search path — the Debug import lib is
    // `openvino_cd`, so `openvino_c` only ever resolves to the Release lib.
    let mut found = false;
    for cand in [
        runtime.join("lib").join("intel64").join("Release"),
        runtime.join("lib").join("intel64"),
        runtime.join("lib"),
    ] {
        if cand.is_dir() {
            println!("cargo:rustc-link-search=native={}", cand.display());
            found = true;
        }
    }
    if !found {
        panic!(
            "OpenVINO SDK at {} has no runtime lib directory (looked under \
             runtime/lib/intel64/Release, runtime/lib/intel64, runtime/lib)",
            runtime.display()
        );
    }

    // The import/shared library is `openvino_c` on all platforms.
    println!("cargo:rustc-link-lib=dylib=openvino_c");

    // Help note for discoverability of the headers (not needed for linking,
    // but surfaces the include path for anyone extending the bindings).
    let _ = Path::new(&runtime).join("include");
}
