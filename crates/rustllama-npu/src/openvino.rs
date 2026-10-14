//! Intel NPU enumeration via the OpenVINO C API (`openvino_c`).
//!
//! Compiled only under the `npu-openvino` feature; `build.rs` links the
//! `openvino_c` import library from an installed OpenVINO SDK (located via
//! `OPENVINO_DIR` / `INTEL_OPENVINO_DIR`). The default rustllama build carries
//! no `npu-*` feature, so it neither links nor loads OpenVINO.
//!
//! Only the read-only *enumeration* surface is bound here: create a Core, list
//! available devices, keep the ones named `NPU`/`NPU.n`, and read each one's
//! `FULL_DEVICE_NAME`. The offload/inference FFI (compile model, infer request)
//! lands with the op-offload path; it is intentionally not bound yet so this
//! module stays small and auditable.
//!
//! ABI note: the OpenVINO C API is `__cdecl` on Windows and the default SysV
//! C ABI elsewhere — both are exactly Rust's `extern "C"`, so no calling-
//! convention annotation is needed. `ov_status_e` is a C `enum` (OK == 0,
//! errors negative), which lowers to a C `int`.

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

// SAFETY: all signatures mirror `openvino/c/ov_core.h` + `ov_common.h` exactly
// (verified against the 2026.4 SDK headers). `build.rs` provides the link.
extern "C" {
    fn ov_core_create(core: *mut *mut OvCore) -> OvStatus;
    fn ov_core_free(core: *mut OvCore);
    fn ov_core_get_available_devices(
        core: *const OvCore,
        devices: *mut OvAvailableDevices,
    ) -> OvStatus;
    fn ov_available_devices_free(devices: *mut OvAvailableDevices);
    fn ov_core_get_property(
        core: *const OvCore,
        device_name: *const c_char,
        property_key: *const c_char,
        property_value: *mut *mut c_char,
    ) -> OvStatus;
    fn ov_free(content: *const c_char);
}

/// Enumerate Intel NPUs OpenVINO can see. Returns empty (never panics) on any
/// FFI failure — a machine without the NPU plugin/driver simply reports no NPU
/// device, which is the inert path. `index` is left 0 here; the public
/// [`crate::enumerate`] re-indexes the merged vendor set.
pub(crate) fn enumerate() -> Vec<NpuInfo> {
    match unsafe { enumerate_inner() } {
        Ok(npus) => npus,
        Err(e) => {
            tracing::debug!("npu-openvino: enumeration failed: {e}");
            Vec::new()
        }
    }
}

unsafe fn enumerate_inner() -> Result<Vec<NpuInfo>, String> {
    let mut core: *mut OvCore = ptr::null_mut();
    let st = ov_core_create(&mut core);
    if st != OV_OK || core.is_null() {
        return Err(format!("ov_core_create failed (status {st})"));
    }
    // Collect, then free the Core on every path.
    let out = collect_npus(core);
    ov_core_free(core);
    out
}

unsafe fn collect_npus(core: *mut OvCore) -> Result<Vec<NpuInfo>, String> {
    let mut devs = OvAvailableDevices { devices: ptr::null_mut(), size: 0 };
    let st = ov_core_get_available_devices(core, &mut devs);
    if st != OV_OK {
        return Err(format!("ov_core_get_available_devices failed (status {st})"));
    }

    let mut out = Vec::new();
    if !devs.devices.is_null() {
        let names = std::slice::from_raw_parts(devs.devices, devs.size);
        for &name_ptr in names {
            if name_ptr.is_null() {
                continue;
            }
            // Device names look like "CPU", "GPU", "GPU.0", "NPU", "NPU.0".
            // Only the Intel NPU plugin reports an "NPU" device.
            let dev = CStr::from_ptr(name_ptr).to_string_lossy();
            if dev == "NPU" || dev.starts_with("NPU.") {
                let full =
                    device_full_name(core, name_ptr).unwrap_or_else(|| dev.to_string());
                out.push(NpuInfo {
                    vendor: NpuVendor::Intel,
                    name: full,
                    index: 0, // re-indexed by the public enumerate()
                    // NPUs share host DRAM; OpenVINO exposes no standard total-
                    // memory property for the NPU plugin, so leave it unknown.
                    memory_bytes: 0,
                });
            }
        }
    }

    ov_available_devices_free(&mut devs);
    Ok(out)
}

/// Read a device's `FULL_DEVICE_NAME` (e.g. "Intel(R) AI Boost"). Returns None
/// on any failure; the caller falls back to the plugin device key ("NPU").
unsafe fn device_full_name(core: *const OvCore, device_name: *const c_char) -> Option<String> {
    // The property-key globals (`ov_property_key_device_full_name`) are exported
    // as `const char*` symbols whose VALUE is this literal, so passing the string
    // directly avoids linking the extern statics.
    let key = CString::new("FULL_DEVICE_NAME").ok()?;
    let mut val: *mut c_char = ptr::null_mut();
    let st = ov_core_get_property(core, device_name, key.as_ptr(), &mut val);
    if st != OV_OK || val.is_null() {
        return None;
    }
    let name = CStr::from_ptr(val).to_string_lossy().into_owned();
    ov_free(val);
    Some(name)
}
