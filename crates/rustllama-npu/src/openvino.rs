//! Intel NPU enumeration via the OpenVINO C API, loaded at RUNTIME.
//!
//! The OpenVINO runtime (`openvino_c` + `openvino` + the NPU plugin + TBB) is
//! BUNDLED next to the binary in the release artifact (Apache-2.0,
//! redistributable), so Intel NPU support works with no user install. On a box
//! without those libs — or without an NPU — every call FAILS SOFT to an empty
//! list. Runtime-loading via `libloading` (the rustllama-l0-sys `ze_loader`
//! idiom) keeps the build free of any OpenVINO link/SDK dependency and turns a
//! missing runtime into a no-op rather than a load-time crash (link-time
//! binding would make the whole binary refuse to start without the DLL).
//!
//! Only the read-only enumeration surface is bound (create a Core, list
//! devices, read FULL_DEVICE_NAME); the offload/inference FFI lands with the
//! op-offload path. ABI: the OpenVINO C API is `__cdecl` on Windows and the
//! default SysV C ABI elsewhere — both are Rust's `extern "C"`. `ov_status_e`
//! is a C `enum` (OK == 0, errors negative) → a C `int`.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;

use crate::{NpuInfo, NpuVendor};

/// `ov_status_e` — C enum, `OK == 0`, errors negative. Repr is a C int.
type OvStatus = c_int;
const OV_OK: OvStatus = 0;

/// Opaque `ov_core` handle.
#[repr(C)]
struct OvCore {
    _private: [u8; 0],
}

/// `ov_available_devices_t { char** devices; size_t size; }`.
#[repr(C)]
struct OvAvailableDevices {
    devices: *mut *mut c_char,
    size: usize,
}

// Function-pointer types for the `openvino_c` exports we call. Signatures
// mirror `openvino/c/ov_core.h` + `ov_common.h` from the 2026.4 SDK exactly.
type OvCoreCreateFn = unsafe extern "C" fn(core: *mut *mut OvCore) -> OvStatus;
type OvCoreFreeFn = unsafe extern "C" fn(core: *mut OvCore);
type OvCoreGetAvailableDevicesFn =
    unsafe extern "C" fn(core: *const OvCore, devices: *mut OvAvailableDevices) -> OvStatus;
type OvAvailableDevicesFreeFn = unsafe extern "C" fn(devices: *mut OvAvailableDevices);
type OvCoreGetPropertyFn = unsafe extern "C" fn(
    core: *const OvCore,
    device_name: *const c_char,
    property_key: *const c_char,
    property_value: *mut *mut c_char,
) -> OvStatus;
type OvFreeFn = unsafe extern "C" fn(content: *const c_char);

/// `openvino_c` library file name per OS. The release bundles this (and the
/// bare→SONAME symlink on Linux) next to the binary; [`crate::bundled_lib_candidates`]
/// resolves the exe-relative path.
#[cfg(windows)]
const OV_LIB: &str = "openvino_c.dll";
#[cfg(target_os = "macos")]
const OV_LIB: &str = "libopenvino_c.dylib";
#[cfg(all(unix, not(target_os = "macos")))]
const OV_LIB: &str = "libopenvino_c.so";

/// Enumerate Intel NPUs OpenVINO can see. Returns empty (never panics) on any
/// failure — missing runtime, missing symbol, or FFI error. `index` is left 0
/// here; the public [`crate::enumerate`] re-indexes the merged vendor set.
pub(crate) fn enumerate() -> Vec<NpuInfo> {
    match unsafe { enumerate_inner() } {
        Ok(npus) => npus,
        Err(e) => {
            tracing::debug!("npu-openvino: {e}");
            Vec::new()
        }
    }
}

unsafe fn enumerate_inner() -> Result<Vec<NpuInfo>, String> {
    // Load the bundled OpenVINO C runtime (absent on non-release / non-NPU hosts
    // → fail soft). Keep `lib` alive for the whole call; the fn pointers copied
    // out of it stay valid only while it is loaded.
    let lib = load_runtime()?;
    let ov_core_create = *get::<OvCoreCreateFn>(&lib, b"ov_core_create\0")?;
    let ov_core_free = *get::<OvCoreFreeFn>(&lib, b"ov_core_free\0")?;
    let ov_get_devices =
        *get::<OvCoreGetAvailableDevicesFn>(&lib, b"ov_core_get_available_devices\0")?;
    let ov_devices_free = *get::<OvAvailableDevicesFreeFn>(&lib, b"ov_available_devices_free\0")?;
    let ov_get_property = *get::<OvCoreGetPropertyFn>(&lib, b"ov_core_get_property\0")?;
    let ov_free = *get::<OvFreeFn>(&lib, b"ov_free\0")?;

    let mut core: *mut OvCore = ptr::null_mut();
    let st = ov_core_create(&mut core);
    if st != OV_OK || core.is_null() {
        return Err(format!("ov_core_create failed (status {st})"));
    }

    let mut out = Vec::new();
    let mut devs = OvAvailableDevices {
        devices: ptr::null_mut(),
        size: 0,
    };
    let st = ov_get_devices(core, &mut devs);
    if st != OV_OK {
        ov_core_free(core);
        return Err(format!(
            "ov_core_get_available_devices failed (status {st})"
        ));
    }
    if !devs.devices.is_null() {
        let names = std::slice::from_raw_parts(devs.devices, devs.size);
        for &name_ptr in names {
            if name_ptr.is_null() {
                continue;
            }
            // Device names look like "CPU", "GPU", "GPU.0", "NPU", "NPU.0".
            // Only the Intel NPU plugin reports an "NPU" device.
            let dev = CStr::from_ptr(name_ptr).to_string_lossy();
            if dev != "NPU" && !dev.starts_with("NPU.") {
                continue;
            }
            // FULL_DEVICE_NAME (e.g. "Intel(R) AI Boost"); fall back to the key.
            // The property-key globals are `const char*` whose VALUE is this
            // literal, so passing the string avoids linking the extern statics.
            let mut full = dev.to_string();
            if let Ok(key) = CString::new("FULL_DEVICE_NAME") {
                let mut val: *mut c_char = ptr::null_mut();
                let pst = ov_get_property(core, name_ptr, key.as_ptr(), &mut val);
                if pst == OV_OK && !val.is_null() {
                    full = CStr::from_ptr(val).to_string_lossy().into_owned();
                    ov_free(val);
                }
            }
            out.push(NpuInfo {
                vendor: NpuVendor::Intel,
                name: full,
                index: 0, // re-indexed by the public enumerate()
                // NPUs share host DRAM; OpenVINO exposes no standard total-
                // memory property for the NPU plugin, so leave it unknown.
                memory_bytes: 0,
            });
        }
        ov_devices_free(&mut devs);
    }

    ov_core_free(core);
    Ok(out)
}

/// Load the OpenVINO C runtime from the bundled exe-relative locations, then the
/// bare name; first that loads wins. Err if none load. Shared by enumeration
/// (this module) and compute offload (`offload.rs`).
pub(crate) unsafe fn load_runtime() -> Result<libloading::Library, String> {
    let mut last = String::new();
    for cand in crate::bundled_lib_candidates(OV_LIB) {
        // SAFETY: Library::new runs the library's initializer (DllMain / _init).
        // OpenVINO is first-party Intel code; the risk is bounded.
        match libloading::Library::new(&cand) {
            Ok(l) => return Ok(l),
            Err(e) => last = format!("{}: {e}", cand.display()),
        }
    }
    Err(format!("OpenVINO runtime not loadable ({last})"))
}

/// Look up one symbol, mapping the error to a message.
unsafe fn get<'a, T>(
    lib: &'a libloading::Library,
    name: &[u8],
) -> Result<libloading::Symbol<'a, T>, String> {
    lib.get(name)
        .map_err(|e| format!("missing symbol {}: {e}", String::from_utf8_lossy(name)))
}
