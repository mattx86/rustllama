//! Cross-vendor GPU inventory (runtime detection).
//!
//! Discovers the system's GPUs with **no build-time GPU SDK dependency**:
//! NVIDIA GPUs are found by `dlopen`-ing the CUDA driver library
//! (`nvcuda.dll` on Windows, `libcuda.so.1` on Linux — shipped with the
//! GPU *driver*, not the CUDA toolkit) and calling the stable CUDA Driver
//! API. Intel GPUs are enumerated separately by the SYCL / Level-Zero
//! layer (`rustllama_kernels_sycl::device_count`/`device_info`), which the
//! caller folds together with this to present the full multi-vendor set.
//!
//! Detection is fail-soft: on a host without an NVIDIA driver,
//! [`detect_nvidia`] returns `None` (the library isn't present or `cuInit`
//! fails) — never an error, never a build requirement. This is the
//! foundation for multi-vendor GPU auto-selection and Intel + NVIDIA
//! mix-and-match; the actual CUDA compute backend
//! (`rustllama-kernels-cuda`) plugs in on top once available.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_uint, c_void};

/// One NVIDIA GPU as reported by the CUDA Driver API.
#[derive(Debug, Clone)]
pub struct NvidiaGpu {
    pub index: u32,
    pub name: String,
    pub total_mem_bytes: u64,
    /// Compute capability, e.g. `(8, 9)` for Ada Lovelace.
    pub compute_capability: (i32, i32),
}

/// Outcome of probing the NVIDIA CUDA driver.
#[derive(Debug, Clone)]
pub struct NvidiaInfo {
    /// Driver-reported CUDA version integer, e.g. `12040` = CUDA 12.4.
    pub driver_version: i32,
    pub gpus: Vec<NvidiaGpu>,
}

impl NvidiaInfo {
    /// `"12.4"`-style rendering of [`Self::driver_version`].
    pub fn driver_version_str(&self) -> String {
        let v = self.driver_version;
        format!("{}.{}", v / 1000, (v % 1000) / 10)
    }
}

const CUDA_SUCCESS: c_int = 0;
// Stable CUdevice attribute ids from the CUDA Driver API.
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

/// Load the NVIDIA CUDA *driver* library. `None` on a host without an
/// NVIDIA driver installed.
fn load_cuda_driver() -> Option<libloading::Library> {
    let candidates: &[&str] = if cfg!(windows) {
        &["nvcuda.dll"]
    } else {
        &["libcuda.so.1", "libcuda.so"]
    };
    for name in candidates {
        // SAFETY: loading a well-known system driver library by name; we
        // only invoke documented CUDA Driver API entry points from it.
        if let Ok(lib) = unsafe { libloading::Library::new(name) } {
            return Some(lib);
        }
    }
    None
}

/// Enumerate NVIDIA GPUs via the CUDA Driver API. Returns `None` when no
/// NVIDIA driver is present or `cuInit` fails — the common case on a
/// non-NVIDIA host, and never a hard error.
pub fn detect_nvidia() -> Option<NvidiaInfo> {
    type CuInit = unsafe extern "C" fn(c_uint) -> c_int;
    type CuGetVer = unsafe extern "C" fn(*mut c_int) -> c_int;
    type CuGetCount = unsafe extern "C" fn(*mut c_int) -> c_int;
    type CuDevGet = unsafe extern "C" fn(*mut c_int, c_int) -> c_int;
    type CuDevName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
    type CuDevMem = unsafe extern "C" fn(*mut usize, c_int) -> c_int;
    type CuDevAttr = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;

    let lib = load_cuda_driver()?;
    // SAFETY: symbols are resolved from the real CUDA driver and called
    // with the documented CUDA Driver API ABI; all out-pointers reference
    // live stack storage.
    unsafe {
        let cu_init: libloading::Symbol<CuInit> = lib.get(b"cuInit\0").ok()?;
        if cu_init(0) != CUDA_SUCCESS {
            return None;
        }
        let cu_ver: libloading::Symbol<CuGetVer> = lib.get(b"cuDriverGetVersion\0").ok()?;
        let mut ver: c_int = 0;
        cu_ver(&mut ver);

        let cu_count: libloading::Symbol<CuGetCount> = lib.get(b"cuDeviceGetCount\0").ok()?;
        let mut count: c_int = 0;
        if cu_count(&mut count) != CUDA_SUCCESS || count <= 0 {
            return Some(NvidiaInfo {
                driver_version: ver,
                gpus: Vec::new(),
            });
        }

        let cu_get: libloading::Symbol<CuDevGet> = lib.get(b"cuDeviceGet\0").ok()?;
        let cu_name: libloading::Symbol<CuDevName> = lib.get(b"cuDeviceGetName\0").ok()?;
        // `_v2` is the ABI-versioned 64-bit symbol; fall back to the base.
        let cu_mem: libloading::Symbol<CuDevMem> = lib
            .get(b"cuDeviceTotalMem_v2\0")
            .or_else(|_| lib.get(b"cuDeviceTotalMem\0"))
            .ok()?;
        let cu_attr: libloading::Symbol<CuDevAttr> = lib.get(b"cuDeviceGetAttribute\0").ok()?;

        let mut gpus = Vec::with_capacity(count as usize);
        for i in 0..count {
            let mut dev: c_int = 0;
            if cu_get(&mut dev, i) != CUDA_SUCCESS {
                continue;
            }
            let mut namebuf = [0 as c_char; 256];
            let name = if cu_name(namebuf.as_mut_ptr(), 256, dev) == CUDA_SUCCESS {
                CStr::from_ptr(namebuf.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            } else {
                "NVIDIA GPU".to_string()
            };
            let mut mem: usize = 0;
            cu_mem(&mut mem, dev);
            let mut major: c_int = 0;
            let mut minor: c_int = 0;
            cu_attr(&mut major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
            cu_attr(&mut minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
            gpus.push(NvidiaGpu {
                index: i as u32,
                name,
                total_mem_bytes: mem as u64,
                compute_capability: (major, minor),
            });
        }
        Some(NvidiaInfo {
            driver_version: ver,
            gpus,
        })
    }
}

/// Whether an NVIDIA CUDA driver is present at all (cheap: just tries to
/// load the driver library, without `cuInit`).
pub fn nvidia_driver_present() -> bool {
    load_cuda_driver().is_some()
}

/// One Apple Metal GPU. Mirror of [`NvidiaGpu`], with `registry_id`
/// (Metal's stable `MTLDevice.registryID`) in place of the CUDA compute-
/// capability pair — Metal has no compute-capability version.
#[derive(Debug, Clone)]
pub struct AppleGpu {
    pub index: u32,
    pub name: String,
    /// Total (unified) memory in bytes. On Apple Silicon the GPU shares one
    /// physical pool with the CPU, so this is the shared-RAM size, not a
    /// separate VRAM aperture.
    pub total_mem_bytes: u64,
    /// Metal's `MTLDevice.registryID` — the stable, driver-invariant device
    /// identifier (Metal has no CUDA-style 16-byte UUID). Zeroed in the
    /// Phase-0 placeholder.
    pub registry_id: u64,
}

/// Outcome of probing for Apple Metal GPUs. Mirrors [`NvidiaInfo`]'s shape;
/// Metal exposes no global "driver version" integer, so this is just the GPU
/// list.
#[derive(Debug, Clone)]
pub struct AppleInfo {
    pub gpus: Vec<AppleGpu>,
}

/// Detect Apple Metal GPUs — the macOS analogue of [`detect_nvidia`], kept
/// TOOLCHAIN-FREE (no link against Metal or `rustllama-kernels-mlx`; cfg /
/// dlopen only, exactly like the CUDA-driver probe). Fail-soft: `None` when
/// no Metal GPU can be enumerated, never a build requirement.
///
/// PHASE 0: inert everywhere. On non-macOS hosts there is no Metal GPU, so
/// this is a compile-time `None`. The real macOS enumeration (dlopen
/// `Metal.framework` → `MTLCreateSystemDefaultDevice` for `name` /
/// `recommendedMaxWorkingSetSize` / `registryID`, or `sysctl hw.memsize`
/// for the unified-memory pool) lands in Phase 1 alongside the real MLX
/// kernels; until then it returns `None` so the whole Apple tier stays inert
/// (matching `rustllama_kernels_mlx::device_count() == 0`).
#[cfg(target_os = "macos")]
pub fn detect_apple_gpu() -> Option<AppleInfo> {
    // TODO(Phase 1): dlopen Metal.framework and enumerate the real Metal
    // device(s) here (registryID + unified-memory size). Kept TOOLCHAIN-FREE
    // — no Objective-C link, no kernel-crate dep.
    None
}

/// Non-macOS hosts have no Metal GPU — always `None` (see the macOS variant).
#[cfg(not(target_os = "macos"))]
pub fn detect_apple_gpu() -> Option<AppleInfo> {
    None
}

/// Read the currently-**free** VRAM (bytes) of the NVIDIA GPU at the given
/// CUDA-driver device index (`NvidiaGpu::index`), via the dlopen'd CUDA
/// driver — no CUDA toolkit required. Returns `None` on any failure or on a
/// non-NVIDIA host, never an error.
///
/// This is the clean, stable per-GPU entry point the Phase-5 placement
/// planner calls to size VRAM-resident weights against real headroom (the
/// enumeration probe [`detect_nvidia`] reports *total* VRAM only). When
/// probing every GPU at once, prefer [`nvidia_free_vram_all`], which loads
/// the driver + calls `cuInit` a single time.
///
/// It is deliberately cheap and side-effect-free: it retains the device's
/// **primary** context (a shared, reference-counted context — it does *not*
/// create a new one), makes it current only long enough to call
/// `cuMemGetInfo`, restores the thread's previous current context, and
/// releases the primary-context reference. It never destroys a context it
/// did not create.
pub fn nvidia_free_vram_bytes(device_index: u32) -> Option<u64> {
    type CuInit = unsafe extern "C" fn(c_uint) -> c_int;
    let lib = load_cuda_driver()?;
    // SAFETY: symbols resolved from the real CUDA driver and called with the
    // documented CUDA Driver API ABI; out-pointers reference live storage.
    unsafe {
        let cu_init: libloading::Symbol<CuInit> = lib.get(b"cuInit\0").ok()?;
        if cu_init(0) != CUDA_SUCCESS {
            return None;
        }
        nvidia_free_vram_inner(&lib, device_index)
    }
}

/// Free VRAM (bytes) for every NVIDIA GPU, as `(driver_index, free_bytes)`
/// pairs in CUDA-driver enumeration order. Loads the driver and calls
/// `cuInit` once, then probes each device — cheaper than calling
/// [`nvidia_free_vram_bytes`] per GPU. Empty on any failure / non-NVIDIA
/// host. GPUs whose individual probe fails are simply omitted.
pub fn nvidia_free_vram_all() -> Vec<(u32, u64)> {
    type CuInit = unsafe extern "C" fn(c_uint) -> c_int;
    type CuGetCount = unsafe extern "C" fn(*mut c_int) -> c_int;
    let Some(lib) = load_cuda_driver() else {
        return Vec::new();
    };
    // SAFETY: as in `nvidia_free_vram_bytes`.
    unsafe {
        let Ok(cu_init) = lib.get::<CuInit>(b"cuInit\0") else {
            return Vec::new();
        };
        if cu_init(0) != CUDA_SUCCESS {
            return Vec::new();
        }
        let Ok(cu_count) = lib.get::<CuGetCount>(b"cuDeviceGetCount\0") else {
            return Vec::new();
        };
        let mut count: c_int = 0;
        if cu_count(&mut count) != CUDA_SUCCESS || count <= 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count as u32 {
            if let Some(free) = nvidia_free_vram_inner(&lib, i) {
                out.push((i, free));
            }
        }
        out
    }
}

/// Shared free-VRAM probe body. Assumes `cuInit` has already succeeded on
/// `lib`. Retains the device primary context, makes it current, reads free
/// VRAM via `cuMemGetInfo`, restores the previously-current context, and
/// releases the primary-context reference. Returns free bytes on success.
///
/// SAFETY: `lib` must be the real CUDA driver library with `cuInit` already
/// called; all out-pointers reference live stack storage.
unsafe fn nvidia_free_vram_inner(lib: &libloading::Library, device_index: u32) -> Option<u64> {
    type CuDevGet = unsafe extern "C" fn(*mut c_int, c_int) -> c_int;
    type CuPrimaryCtxRetain = unsafe extern "C" fn(*mut *mut c_void, c_int) -> c_int;
    type CuCtxGetCurrent = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
    type CuCtxSetCurrent = unsafe extern "C" fn(*mut c_void) -> c_int;
    type CuMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> c_int;
    type CuPrimaryCtxRelease = unsafe extern "C" fn(c_int) -> c_int;

    let cu_get: libloading::Symbol<CuDevGet> = lib.get(b"cuDeviceGet\0").ok()?;
    let cu_retain: libloading::Symbol<CuPrimaryCtxRetain> =
        lib.get(b"cuDevicePrimaryCtxRetain\0").ok()?;
    let cu_set: libloading::Symbol<CuCtxSetCurrent> = lib.get(b"cuCtxSetCurrent\0").ok()?;
    // `_v2` is the ABI-versioned 64-bit symbol; fall back to the base.
    let cu_mem: libloading::Symbol<CuMemGetInfo> = lib
        .get(b"cuMemGetInfo_v2\0")
        .or_else(|_| lib.get(b"cuMemGetInfo\0"))
        .ok()?;
    let cu_release: libloading::Symbol<CuPrimaryCtxRelease> = lib
        .get(b"cuDevicePrimaryCtxRelease_v2\0")
        .or_else(|_| lib.get(b"cuDevicePrimaryCtxRelease\0"))
        .ok()?;
    // Best-effort: save/restore the thread's current context so this probe
    // leaves no CUDA thread state behind. Missing on very old drivers → skip.
    let cu_getcur: Option<libloading::Symbol<CuCtxGetCurrent>> =
        lib.get(b"cuCtxGetCurrent\0").ok();

    let mut dev: c_int = 0;
    if cu_get(&mut dev, device_index as c_int) != CUDA_SUCCESS {
        return None;
    }
    // Remember the current context (if any) to restore afterwards.
    let mut prev_ctx: *mut c_void = std::ptr::null_mut();
    if let Some(getcur) = cu_getcur {
        // Ignore the result: a null `prev_ctx` just means "none current".
        getcur(&mut prev_ctx);
    }
    let mut ctx: *mut c_void = std::ptr::null_mut();
    if cu_retain(&mut ctx, dev) != CUDA_SUCCESS || ctx.is_null() {
        return None;
    }
    let free = {
        let mut free_b: usize = 0;
        let mut total_b: usize = 0;
        if cu_set(ctx) == CUDA_SUCCESS && cu_mem(&mut free_b, &mut total_b) == CUDA_SUCCESS {
            Some(free_b as u64)
        } else {
            None
        }
    };
    // Restore the thread's previous current context (best-effort), then
    // balance the retain — decrements the primary context's refcount without
    // destroying a context we didn't create.
    cu_set(prev_ctx);
    cu_release(dev);
    free
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_nvidia_is_fail_soft() {
        // On CI / non-NVIDIA hosts this must be `None` (or `Some` with a
        // sane version on an NVIDIA host) — never a panic.
        match detect_nvidia() {
            None => {}
            Some(info) => {
                assert!(info.driver_version >= 0);
                for g in &info.gpus {
                    assert!(!g.name.is_empty());
                }
            }
        }
    }
}
