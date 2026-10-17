//! AMD NPU (XDNA / Ryzen AI) enumeration via the XRT runtime.
//!
//! WRITE-BLIND: there is no Ryzen AI / XRT SDK or AMD NPU on the dev box, so the
//! XRT *call path* here is authored against the published XRT C API and is NOT
//! runtime-verified; it is validated on Ryzen AI hardware. To keep an unverified
//! guess from ever breaking anything, XRT is loaded at RUNTIME via `libloading`
//! (exactly like `rustllama-l0-sys` dlopen's `ze_loader`) and the whole path
//! FAILS SOFT: a missing library, a missing symbol, or a probe error all yield
//! an empty NPU list — identical to a box with no AMD NPU. This also keeps the
//! build free of any XRT link dependency, so the `npu-ryzenai` feature itself
//! compiles here (only `libloading` is pulled); only the device call is blind.
//!
//! Enumeration primitive: `xclProbe()` from `xrt_coreutil` returns the XRT
//! device count. On a Ryzen AI laptop the sole XRT device is the NPU; on a host
//! that also carries Alveo/FPGA XRT cards this would overcount — narrowing to
//! XDNA devices needs `xclGetDeviceInfo2` / device-class filtering, deferred to
//! the on-hardware validation pass. The modern `xrt::system::enumerate_devices()`
//! is a name-mangled C++ symbol unsuitable for a C FFI, so the stable C entry
//! `xclProbe` is used instead.

use std::os::raw::c_uint;

use crate::{NpuInfo, NpuVendor};

/// `unsigned xclProbe(void)` — number of XRT devices present.
type XclProbeFn = unsafe extern "C" fn() -> c_uint;

/// Candidate XRT runtime library names; the OS loader searches PATH / ld paths
/// (the Ryzen AI installer puts `xrt_coreutil` on PATH).
#[cfg(windows)]
const XRT_LIBS: &[&str] = &["xrt_coreutil.dll"];
#[cfg(not(windows))]
const XRT_LIBS: &[&str] = &["libxrt_coreutil.so", "libxrt_coreutil.so.2"];

/// Enumerate AMD NPUs via XRT. Empty (never panics) on any failure — the
/// fail-soft contract. `index` is left 0 here; the public [`crate::enumerate`]
/// re-indexes the merged vendor set.
pub(crate) fn enumerate() -> Vec<NpuInfo> {
    match probe_count() {
        Some(n) if n > 0 => (0..n)
            .map(|i| NpuInfo {
                vendor: NpuVendor::Amd,
                name: "AMD XDNA NPU".to_string(),
                index: i,
                // NPUs share host DRAM; XRT exposes no simple total-memory query
                // for the NPU here, so leave it unknown.
                memory_bytes: 0,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Load XRT and call `xclProbe`. None on any failure (fail-soft).
fn probe_count() -> Option<u32> {
    let cands: Vec<std::path::PathBuf> = XRT_LIBS
        .iter()
        .flat_map(|n| crate::bundled_lib_candidates(n))
        .collect();
    for cand in cands {
        // SAFETY: Library::new runs the library's initializer (DllMain / _init).
        // XRT is first-party AMD code; the risk is bounded. Any error → try the
        // next candidate, then give up.
        let lib = match unsafe { libloading::Library::new(&cand) } {
            Ok(l) => l,
            Err(e) => {
                tracing::debug!("npu-ryzenai: {} not loadable: {e}", cand.display());
                continue;
            }
        };
        // SAFETY: `xclProbe` is the documented `unsigned xclProbe(void)` export.
        let probe: libloading::Symbol<XclProbeFn> = match unsafe { lib.get(b"xclProbe\0") } {
            Ok(s) => s,
            Err(e) => {
                tracing::debug!("npu-ryzenai: {} has no xclProbe: {e}", cand.display());
                continue;
            }
        };
        // SAFETY: signature matches the export; `xclProbe` needs no prior init.
        let n = unsafe { probe() };
        tracing::info!(
            "npu-ryzenai: XRT ({}) reports {n} device(s)",
            cand.display()
        );
        return Some(n as u32);
    }
    None
}
