//! Tauri's frontend codegen is run only when the `gui` feature is enabled,
//! so phase-0 builds (no Node toolchain installed) succeed.

fn main() {
    // The SYCL kernels are ALWAYS linked (the kernels-sycl crate is
    // real-only), so this per-platform link setup is unconditional.
    // `rustc-link-arg` from THIS (the binary) crate reaches the final
    // link; the same arg from the kernels-sycl dep would NOT.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "windows" {
        // Delay-load the SYCL kernel DLL so its transitive oneAPI
        // dependencies (sycl*.dll, ur_loader, libhwloc-15, …) are NOT
        // bound at process init. That lets `main()` prepend the
        // autodetected oneAPI dirs to PATH before the first `rsl_*` call
        // (see rustllama_runtime::ensure_gpu_dll_search_paths), which is
        // what makes a bare-shell launch work without run-sycl.bat.
        // delayimp.lib provides the delay-load thunk resolver; found via
        // the MSVC LIB path (build-env.bat sources vcvarsall).
        println!("cargo:rustc-link-arg=delayimp.lib");
        println!("cargo:rustc-link-arg=/DELAYLOAD:rsl_kernels.dll");
    } else {
        // On Linux the SYCL kernels live in librsl_kernels.so, which the
        // kernels-sycl build script copies next to the produced binary.
        // Add an `$ORIGIN` rpath so the dynamic loader finds it beside the
        // executable (ship the .so alongside the binary).
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    }
    if std::env::var("CARGO_FEATURE_GUI").is_ok() {
        #[cfg(feature = "gui")]
        tauri_build::build();
    }
}
