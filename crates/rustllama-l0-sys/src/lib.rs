//! Dynamic-loading FFI for Intel Level Zero (`ze_loader.dll` on
//! Windows, `libze_loader.so` on Linux). No build-time header
//! dependency: the only thing we need at compile time is correct
//! Rust definitions for the L0 ABI types and function signatures,
//! hand-rolled from the upstream
//! [`level-zero/include/ze_api.h`](https://github.com/oneapi-src/level-zero/blob/master/include/ze_api.h).
//!
//! Why dynamic loading instead of `bindgen + link`?
//!
//!   - The oneAPI compiler distribution doesn't bundle Level Zero
//!     headers (only the SYCL/OpenCL ones). The driver supplies
//!     `ze_loader.dll` in `C:\Windows\System32` but no `.lib` import
//!     library. Static linking would require vendoring upstream
//!     headers and shimming our own `.lib` — both add maintenance
//!     burden for a primitive we expect to use at a handful of call
//!     sites.
//!   - We want graceful degradation on hosts without an Intel GPU
//!     (CI, the typical developer's macOS / non-Intel Windows
//!     laptop). Static linking would force us to stub the API behind
//!     cfg flags; dynamic loading just returns `Unavailable` and the
//!     engine falls back to its existing `malloc_shared + memcpy` path.
//!
//! Scope (today, Step 1a of the L0 import work):
//!
//!   - Load the loader DLL, resolve a small set of entry points.
//!   - Initialize the runtime + enumerate drivers and devices.
//!   - Provide the hand-rolled types + function pointer signatures
//!     that future steps will use for memory import.
//!
//! NOT in scope today: the actual import flow
//! (`zeMemAllocHost` with `ze_external_memory_import_win32_handle_t`
//! chained on the alloc descriptor). That lands in Step 1b alongside
//! the engine-side wiring.

use std::sync::OnceLock;

// ============================================================
// L0 ABI types
// ============================================================
//
// All types here mirror the layout in `level_zero/ze_api.h` v1.13.
// Comments cite the relevant header section so future maintainers
// can cross-check.

/// Result codes returned by every L0 entry point. The upstream
/// `ze_result_t` is a `uint32_t` enum; we don't enumerate every
/// variant here because we treat any non-`SUCCESS` as an opaque
/// error and log the numeric code. The constants in this module
/// cover the values we actually branch on.
pub type ZeResult = u32;

/// `ze_result_t::ZE_RESULT_SUCCESS` — call completed without error.
pub const ZE_RESULT_SUCCESS: ZeResult = 0;
/// `ze_result_t::ZE_RESULT_ERROR_UNINITIALIZED` — `zeInit` not yet
/// called (or it failed) on this process.
pub const ZE_RESULT_ERROR_UNINITIALIZED: ZeResult = 0x78000001;
/// `ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE` — driver
/// recognized the call but doesn't support the requested feature.
/// We treat this as a "fall back to copy path" signal.
pub const ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: ZeResult = 0x78000003;

/// `ze_init_flags_t::ZE_INIT_FLAG_GPU_ONLY` — only enumerate GPU
/// devices, skip VPU/NPU. Matches what SYCL does internally.
pub const ZE_INIT_FLAG_GPU_ONLY: u32 = 1;

/// Opaque handle to an L0 driver instance. Comes back from
/// `zeDriverGet`. `repr(transparent)` over a non-null pointer is
/// the standard idiom for L0 handles — they're opaque to us and
/// only passed back into other L0 calls.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ZeDriverHandle(pub *mut std::ffi::c_void);

/// Opaque handle to an L0 device on a driver. Comes back from
/// `zeDeviceGet`. Today we only need this for diagnostics — Step 1b
/// will use it for `zeMemAllocHost` once the import descriptor is
/// wired up.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ZeDeviceHandle(pub *mut std::ffi::c_void);

// SAFETY: L0 handles are designed to be passed between threads;
// the driver owns the underlying state and synchronizes itself.
// The pointer is opaque — we never deref it from Rust.
unsafe impl Send for ZeDriverHandle {}
unsafe impl Sync for ZeDriverHandle {}
unsafe impl Send for ZeDeviceHandle {}
unsafe impl Sync for ZeDeviceHandle {}

// ============================================================
// Function pointer types
// ============================================================
//
// Stored as raw `unsafe extern "C" fn` so the loader can hand them
// out cheaply. Each signature mirrors the upstream header exactly.
//
// Calling convention: L0 v1.13 defines `ZE_APICALL` as `__cdecl` on
// Windows, which matches Rust's `extern "C"` on x86_64 Windows. (For
// 32-bit Windows we'd need `extern "stdcall"`; we don't support that
// target.)

/// `ze_result_t zeInit(ze_init_flags_t flags)`.
pub type ZeInitFn = unsafe extern "C" fn(flags: u32) -> ZeResult;

/// `ze_result_t zeDriverGet(uint32_t* pCount, ze_driver_handle_t* phDrivers)`.
pub type ZeDriverGetFn =
    unsafe extern "C" fn(p_count: *mut u32, p_drivers: *mut ZeDriverHandle) -> ZeResult;

/// `ze_result_t zeDeviceGet(ze_driver_handle_t hDriver, uint32_t* pCount, ze_device_handle_t* phDevices)`.
pub type ZeDeviceGetFn = unsafe extern "C" fn(
    driver: ZeDriverHandle,
    p_count: *mut u32,
    p_devices: *mut ZeDeviceHandle,
) -> ZeResult;

// ============================================================
// Sysman ABI types
// ============================================================
//
// Sysman is a sibling API to L0 core, exposed via the same loader
// DLL (`ze_loader.dll` on Windows). Function names use the `zes`
// prefix instead of `ze`. Handle types are opaque pointers like L0
// core; the Intel implementation uses the same underlying handle
// for `ze_device_handle_t` and `zes_device_handle_t` so they can
// be cast directly. The Sysman spec doesn't guarantee this in
// general — for our purposes (Intel GPU only) it's safe.
//
// Types mirror `level_zero/zes_api.h` v1.13.

/// `zes_init_flag_t::ZES_INIT_FLAG_PLACEHOLDER` — reserved, must be 0
/// in current spec. The Sysman init call still requires a flags arg.
pub const ZES_INIT_FLAG_PLACEHOLDER: u32 = 0;

/// `zes_device_handle_t` — opaque Sysman device handle. Cast from
/// `ze_device_handle_t` on Intel.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ZesDeviceHandle(pub *mut std::ffi::c_void);

unsafe impl Send for ZesDeviceHandle {}
unsafe impl Sync for ZesDeviceHandle {}

/// `zes_mem_handle_t` — handle to one memory module on a device.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ZesMemHandle(pub *mut std::ffi::c_void);
unsafe impl Send for ZesMemHandle {}
unsafe impl Sync for ZesMemHandle {}

/// `zes_temp_handle_t` — handle to one temperature sensor.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ZesTempHandle(pub *mut std::ffi::c_void);
unsafe impl Send for ZesTempHandle {}
unsafe impl Sync for ZesTempHandle {}

/// `zes_pwr_handle_t` — handle to one power domain.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ZesPwrHandle(pub *mut std::ffi::c_void);
unsafe impl Send for ZesPwrHandle {}
unsafe impl Sync for ZesPwrHandle {}

/// `zes_freq_handle_t` — handle to one frequency domain (GPU clock,
/// memory clock, media clock, etc.). On Intel GPUs the GPU clock is
/// the headline number; memory clock is exposed on Arc parts.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ZesFreqHandle(pub *mut std::ffi::c_void);
unsafe impl Send for ZesFreqHandle {}
unsafe impl Sync for ZesFreqHandle {}

/// `zes_mem_state_t` — memory module state. `size` = total bytes;
/// `free` = currently-free bytes. The version-tagged header is the
/// L0 convention; we set `stype=ZES_STRUCTURE_TYPE_MEM_STATE` (0x1e)
/// and `pNext=NULL` before each call.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct ZesMemState {
    pub stype: u32,
    pub p_next: *mut std::ffi::c_void,
    pub health: u32,
    pub free: u64,
    pub size: u64,
}
unsafe impl Send for ZesMemState {}
unsafe impl Sync for ZesMemState {}

/// `stype` value for `ZesMemState`.
pub const ZES_STRUCTURE_TYPE_MEM_STATE: u32 = 0x1e;

/// `zes_power_energy_counter_t` — cumulative energy + timestamp.
/// The recommended way to derive instantaneous power is two reads
/// separated by a small interval: `power = (e2 - e1) / (t2 - t1)`.
/// Single-shot read returns energy in µJ and timestamp in µs.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct ZesPowerEnergyCounter {
    pub energy_uj: u64,
    pub timestamp_us: u64,
}

/// `zes_temp_sensors_t` — sensor type enum (subset). `GLOBAL` is
/// the package-wide average temperature; `GPU` is the slice-level
/// reading. We pick the first sensor reported regardless of type.
pub const ZES_TEMP_SENSORS_GLOBAL: u32 = 0;
pub const ZES_TEMP_SENSORS_GPU: u32 = 1;

/// `ze_result_t zesInit(zes_init_flags_t flags)`.
pub type ZesInitFn = unsafe extern "C" fn(flags: u32) -> ZeResult;

/// `ze_result_t zesDeviceEnumMemoryModules(zes_device_handle_t, uint32_t*, zes_mem_handle_t*)`.
pub type ZesDeviceEnumMemoryModulesFn = unsafe extern "C" fn(
    device: ZesDeviceHandle,
    p_count: *mut u32,
    p_handles: *mut ZesMemHandle,
) -> ZeResult;

/// `ze_result_t zesMemoryGetState(zes_mem_handle_t, zes_mem_state_t*)`.
pub type ZesMemoryGetStateFn =
    unsafe extern "C" fn(mem: ZesMemHandle, p_state: *mut ZesMemState) -> ZeResult;

/// `ze_result_t zesDeviceEnumTemperatureSensors(...)`.
pub type ZesDeviceEnumTemperatureSensorsFn = unsafe extern "C" fn(
    device: ZesDeviceHandle,
    p_count: *mut u32,
    p_handles: *mut ZesTempHandle,
) -> ZeResult;

/// `ze_result_t zesTemperatureGetState(zes_temp_handle_t, double*)`.
pub type ZesTemperatureGetStateFn =
    unsafe extern "C" fn(temp: ZesTempHandle, p_temperature: *mut f64) -> ZeResult;

/// `ze_result_t zesDeviceEnumPowerDomains(...)`.
pub type ZesDeviceEnumPowerDomainsFn = unsafe extern "C" fn(
    device: ZesDeviceHandle,
    p_count: *mut u32,
    p_handles: *mut ZesPwrHandle,
) -> ZeResult;

/// `ze_result_t zesPowerGetEnergyCounter(zes_pwr_handle_t, zes_power_energy_counter_t*)`.
pub type ZesPowerGetEnergyCounterFn = unsafe extern "C" fn(
    pwr: ZesPwrHandle,
    p_counter: *mut ZesPowerEnergyCounter,
) -> ZeResult;

/// `zes_freq_state_t` — current frequency-domain state. The Sysman
/// spec defines this as a versioned struct (`stype` + `pNext`); we
/// set `stype = ZES_STRUCTURE_TYPE_FREQ_STATE` and leave `pNext`
/// null. The fields we read: `actual` (MHz, current operating
/// frequency); `request`, `tdp`, `efficient`, and `throttle_reasons`
/// are kept on the struct for layout fidelity but unused at the
/// rustllama probe level today.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct ZesFreqState {
    pub stype: u32,
    pub p_next: *mut std::ffi::c_void,
    pub current_voltage: f64,
    pub request: f64,
    pub tdp: f64,
    pub efficient: f64,
    pub actual: f64,
    pub throttle_reasons: u32,
}
unsafe impl Send for ZesFreqState {}
unsafe impl Sync for ZesFreqState {}

/// `stype` value for `ZesFreqState`.
pub const ZES_STRUCTURE_TYPE_FREQ_STATE: u32 = 0x1f;

/// `ze_result_t zesDeviceEnumFrequencyDomains(...)`.
pub type ZesDeviceEnumFrequencyDomainsFn = unsafe extern "C" fn(
    device: ZesDeviceHandle,
    p_count: *mut u32,
    p_handles: *mut ZesFreqHandle,
) -> ZeResult;

/// `ze_result_t zesFrequencyGetState(zes_freq_handle_t, zes_freq_state_t*)`.
pub type ZesFrequencyGetStateFn = unsafe extern "C" fn(
    freq: ZesFreqHandle,
    p_state: *mut ZesFreqState,
) -> ZeResult;

/// `zes_engine_handle_t` — handle to one engine group (compute,
/// render/3D, copy, media, or an "all engines" aggregate).
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ZesEngineHandle(pub *mut std::ffi::c_void);
unsafe impl Send for ZesEngineHandle {}
unsafe impl Sync for ZesEngineHandle {}

/// `zes_engine_stats_t` — cumulative engine activity. `active_time` is
/// the µs the engine group was busy; `timestamp` is the device clock
/// (µs) at sample time. Utilization over an interval is
/// `(active2 - active1) / (timestamp2 - timestamp1)`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct ZesEngineStats {
    pub active_time: u64,
    pub timestamp: u64,
}

/// `ze_result_t zesDeviceEnumEngineGroups(zes_device_handle_t, uint32_t*, zes_engine_handle_t*)`.
pub type ZesDeviceEnumEngineGroupsFn = unsafe extern "C" fn(
    device: ZesDeviceHandle,
    p_count: *mut u32,
    p_handles: *mut ZesEngineHandle,
) -> ZeResult;

/// `ze_result_t zesEngineGetActivity(zes_engine_handle_t, zes_engine_stats_t*)`.
pub type ZesEngineGetActivityFn = unsafe extern "C" fn(
    engine: ZesEngineHandle,
    p_stats: *mut ZesEngineStats,
) -> ZeResult;

// ============================================================
// LevelZero — dynamic loader handle
// ============================================================

/// Errors surfaced by the loader. All but `Unavailable` are
/// programmer / driver bugs; `Unavailable` is the expected path on
/// hosts without an Intel GPU and the engine's caller should treat
/// it as "no L0, use the existing SYCL allocator path".
#[derive(Debug, thiserror::Error)]
pub enum L0Error {
    /// `ze_loader.dll` (Windows) or `libze_loader.so` (Linux) is not
    /// installed on this host, or the OS rejected the load. Common
    /// causes: non-Intel GPU, missing Intel graphics driver,
    /// running under a sandboxed runtime that blocks the loader.
    #[error("Level Zero loader unavailable: {0}")]
    Unavailable(String),
    /// The loader DLL is present but a function we depend on is
    /// missing. Means the user's driver predates the L0 version
    /// we target; we treat it as fatal so the issue gets
    /// surfaced (rather than silently using the wrong codepath).
    #[error("Level Zero loader missing required symbol `{name}`")]
    MissingSymbol { name: &'static str },
    /// An L0 entry point returned a non-success code. Numeric form
    /// avoids decoding every variant of `ze_result_t`; callers can
    /// match on known constants (`ZE_RESULT_ERROR_*`).
    #[error("Level Zero call {call} returned code 0x{code:08x}")]
    Call { call: &'static str, code: ZeResult },
}

/// The dynamically-loaded Level Zero entry points. Process-singleton
/// — there's no benefit to multiple instances since the underlying
/// `ze_loader` library keeps process-wide state in `zeInit`.
///
/// Construction goes through [`LevelZero::load`], which is what the
/// engine actually calls. It dlopens the loader DLL, resolves the
/// function pointers, and runs `zeInit` once. After that, all
/// `LevelZero` operations are zero-cost FFI dispatches.
pub struct LevelZero {
    // Library handle is stored to keep the dlopen alive. Function
    // pointers below are valid only while `_lib` exists, so the
    // struct field order also doubles as drop order (Rust drops
    // fields in declaration order; we want the fn pointers'
    // backing memory to stay alive until `_lib` drops).
    _lib: libloading::Library,
    pub ze_init: ZeInitFn,
    pub ze_driver_get: ZeDriverGetFn,
    pub ze_device_get: ZeDeviceGetFn,
}

impl LevelZero {
    /// Dynamically load the Level Zero loader DLL and resolve the
    /// entry points we use. On success, also calls `zeInit` once
    /// with `ZE_INIT_FLAG_GPU_ONLY` so subsequent calls don't
    /// `ERROR_UNINITIALIZED`. Returns `Unavailable` on any of the
    /// expected "no GPU here" paths and a `Call` error on
    /// unexpected `zeInit` failures.
    ///
    /// Idempotent at the process level: the loader caches the
    /// `Library` handle in a `OnceLock`, so calling this from
    /// multiple threads only loads once.
    pub fn load() -> Result<&'static Self, L0Error> {
        static INSTANCE: OnceLock<Result<LevelZero, L0Error>> = OnceLock::new();
        let slot = INSTANCE.get_or_init(Self::load_impl);
        match slot {
            // SAFETY: `OnceLock<Result<...>>` keeps the `Ok` value
            // pinned for the lifetime of the process, so a `&'static
            // LevelZero` reference is legitimate.
            Ok(l0) => Ok(l0),
            // Errors aren't `Clone`able (thiserror with `String`
            // works, but custom variants need explicit clone). Recompute
            // a fresh equivalent error to return — the load_impl
            // doesn't actually re-run because we're still inside the
            // OnceLock.
            Err(L0Error::Unavailable(msg)) => Err(L0Error::Unavailable(msg.clone())),
            Err(L0Error::MissingSymbol { name }) => Err(L0Error::MissingSymbol { name }),
            Err(L0Error::Call { call, code }) => Err(L0Error::Call { call, code: *code }),
        }
    }

    fn load_impl() -> Result<Self, L0Error> {
        let lib_name = if cfg!(target_os = "windows") {
            "ze_loader.dll"
        } else if cfg!(target_os = "linux") {
            "libze_loader.so.1"
        } else {
            return Err(L0Error::Unavailable(format!(
                "no Level Zero loader name for target_os = {}",
                std::env::consts::OS
            )));
        };
        // SAFETY: libloading::Library::new is unsafe because the
        // loaded code can do arbitrary things in its initializer
        // (DllMain on Windows, _init on Linux). For ze_loader this
        // is well-tested first-party Intel code; the risk is
        // bounded.
        let lib = unsafe { libloading::Library::new(lib_name) }
            .map_err(|e| L0Error::Unavailable(format!("{lib_name}: {e}")))?;
        let ze_init = unsafe { load_sym::<ZeInitFn>(&lib, b"zeInit\0", "zeInit")? };
        let ze_driver_get =
            unsafe { load_sym::<ZeDriverGetFn>(&lib, b"zeDriverGet\0", "zeDriverGet")? };
        let ze_device_get =
            unsafe { load_sym::<ZeDeviceGetFn>(&lib, b"zeDeviceGet\0", "zeDeviceGet")? };
        // Call zeInit once. Subsequent calls are no-ops on the
        // Intel implementation but the spec requires at least one
        // before any other entry point.
        // SAFETY: ze_init signature matches what the loader exports.
        let init_code = unsafe { ze_init(ZE_INIT_FLAG_GPU_ONLY) };
        if init_code != ZE_RESULT_SUCCESS {
            return Err(L0Error::Call {
                call: "zeInit",
                code: init_code,
            });
        }
        tracing::info!(lib_name, "Level Zero loader initialized");
        Ok(LevelZero {
            _lib: lib,
            ze_init,
            ze_driver_get,
            ze_device_get,
        })
    }

    /// Enumerate L0 drivers visible to this process. Returns an
    /// empty Vec when no Intel GPU is present (zeInit succeeded but
    /// there are no GPU drivers); returns `Call` only on a true
    /// driver error.
    pub fn drivers(&self) -> Result<Vec<ZeDriverHandle>, L0Error> {
        let mut count: u32 = 0;
        // First call gets the count.
        // SAFETY: ze_driver_get signature matches the loader's
        // export; passing null for the handles array on a count-
        // query call is the documented L0 contract.
        let r = unsafe { (self.ze_driver_get)(&mut count, std::ptr::null_mut()) };
        if r != ZE_RESULT_SUCCESS {
            return Err(L0Error::Call {
                call: "zeDriverGet (count)",
                code: r,
            });
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut drivers = vec![ZeDriverHandle(std::ptr::null_mut()); count as usize];
        let r = unsafe { (self.ze_driver_get)(&mut count, drivers.as_mut_ptr()) };
        if r != ZE_RESULT_SUCCESS {
            return Err(L0Error::Call {
                call: "zeDriverGet (fetch)",
                code: r,
            });
        }
        Ok(drivers)
    }

    /// Enumerate GPU devices on a given driver. Same two-call
    /// idiom as `drivers()`.
    pub fn devices(&self, driver: ZeDriverHandle) -> Result<Vec<ZeDeviceHandle>, L0Error> {
        let mut count: u32 = 0;
        let r = unsafe { (self.ze_device_get)(driver, &mut count, std::ptr::null_mut()) };
        if r != ZE_RESULT_SUCCESS {
            return Err(L0Error::Call {
                call: "zeDeviceGet (count)",
                code: r,
            });
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut devices = vec![ZeDeviceHandle(std::ptr::null_mut()); count as usize];
        let r = unsafe { (self.ze_device_get)(driver, &mut count, devices.as_mut_ptr()) };
        if r != ZE_RESULT_SUCCESS {
            return Err(L0Error::Call {
                call: "zeDeviceGet (fetch)",
                code: r,
            });
        }
        Ok(devices)
    }
}

// ============================================================
// Sysman loader
// ============================================================

/// A single device's VRAM / thermal / power snapshot, aggregated
/// across all reportable memory modules / temperature sensors /
/// power domains. Returned by [`Sysman::probe`].
///
/// Fields are `Option` so partial probes succeed — older drivers
/// may expose memory but not energy counters, etc. None means
/// "not available on this driver / device", not "error".
#[derive(Debug, Default, Clone, Copy)]
pub struct SysmanReading {
    /// Total VRAM across all memory modules on this device (bytes).
    pub vram_total_bytes: Option<u64>,
    /// Currently-free VRAM across all modules (bytes).
    pub vram_free_bytes: Option<u64>,
    /// Maximum reported temperature across all sensors (°C). `None`
    /// when no temperature sensor is exposed (typical on Iris Xe
    /// integrated graphics — Sysman temp probes are usually only
    /// surfaced on discrete Arc parts).
    pub max_temp_c: Option<f64>,
    /// Cumulative energy counter (µJ) summed over all power domains
    /// plus the timestamp (µs) when sampled. Caller derives
    /// instantaneous power as `(e2 - e1) / (t2 - t1) µW` between two
    /// reads. `None` when no power domain is exposed.
    pub energy_counter: Option<ZesPowerEnergyCounter>,
    /// Current GPU clock frequency in MHz, taken from the first
    /// frequency domain the device exposes (typically the main GPU
    /// clock on Iris Xe / Arc). `None` when the driver doesn't
    /// expose any frequency domains.
    pub gpu_freq_mhz: Option<f64>,
    /// Cumulative engine activity summed across the device's engine
    /// groups plus a device-clock timestamp. Caller derives GPU
    /// utilization from two reads: `(active2 - active1) / (ts2 - ts1)`.
    /// `None` when the driver exposes no engine groups (common on
    /// integrated parts) or the symbols are absent.
    pub engine_stats: Option<ZesEngineStats>,
}

/// Dynamically-loaded Sysman entry points. Process-singleton like
/// [`LevelZero`]; constructed lazily on first probe.
pub struct Sysman {
    _lib: libloading::Library,
    pub zes_init: ZesInitFn,
    pub zes_device_enum_memory_modules: ZesDeviceEnumMemoryModulesFn,
    pub zes_memory_get_state: ZesMemoryGetStateFn,
    /// Optional — not all drivers export this; we treat absence as
    /// "temperature probe disabled" rather than failure to load.
    pub zes_device_enum_temperature_sensors: Option<ZesDeviceEnumTemperatureSensorsFn>,
    pub zes_temperature_get_state: Option<ZesTemperatureGetStateFn>,
    pub zes_device_enum_power_domains: Option<ZesDeviceEnumPowerDomainsFn>,
    pub zes_power_get_energy_counter: Option<ZesPowerGetEnergyCounterFn>,
    /// Optional — same caveat as the temp/power probes; older drivers
    /// don't expose the symbols.
    pub zes_device_enum_frequency_domains: Option<ZesDeviceEnumFrequencyDomainsFn>,
    pub zes_frequency_get_state: Option<ZesFrequencyGetStateFn>,
    /// Optional — engine-activity probe for GPU-utilization %. Absent
    /// on drivers / integrated parts that don't expose engine groups.
    pub zes_device_enum_engine_groups: Option<ZesDeviceEnumEngineGroupsFn>,
    pub zes_engine_get_activity: Option<ZesEngineGetActivityFn>,
}

impl Sysman {
    /// Load the Sysman entry points from the same `ze_loader` DLL
    /// the L0 core loader uses. Calls `zesInit` once. Returns
    /// [`L0Error::Unavailable`] when the loader is absent (no
    /// Intel GPU on this host) or [`L0Error::MissingSymbol`] for
    /// the *required* memory probe entry points — those are
    /// considered fatal because the headline VRAM probe relies on
    /// them. Temperature + power entry points are optional and
    /// `None` when the driver doesn't export them.
    pub fn load() -> Result<&'static Self, L0Error> {
        static INSTANCE: OnceLock<Result<Sysman, L0Error>> = OnceLock::new();
        let slot = INSTANCE.get_or_init(Self::load_impl);
        match slot {
            Ok(s) => Ok(s),
            Err(L0Error::Unavailable(msg)) => Err(L0Error::Unavailable(msg.clone())),
            Err(L0Error::MissingSymbol { name }) => Err(L0Error::MissingSymbol { name }),
            Err(L0Error::Call { call, code }) => Err(L0Error::Call { call, code: *code }),
        }
    }

    fn load_impl() -> Result<Self, L0Error> {
        let lib_name = if cfg!(target_os = "windows") {
            "ze_loader.dll"
        } else if cfg!(target_os = "linux") {
            "libze_loader.so.1"
        } else {
            return Err(L0Error::Unavailable(format!(
                "no Sysman loader for target_os = {}",
                std::env::consts::OS
            )));
        };
        // Pre-zesInit workaround: Intel's L0 loader requires
        // `ZES_ENABLE_SYSMAN=1` to be set in the process environment
        // before any init call, otherwise `zesInit` returns
        // `UNINITIALIZED`. This is documented in Intel's Sysman
        // troubleshooting notes — the env var enables the Sysman
        // execution mode in the driver. Setting it inside the
        // process is fine; the loader reads it lazily on init.
        //
        // SAFETY: std::env::set_var is unsafe in edition 2024 because
        // it can race with concurrent getenv. Sysman::load is gated
        // on a OnceLock so we only ever run this once per process,
        // and we run it before any spawned thread would be reading
        // the env (the loader itself is what reads it). For older
        // editions this is a regular safe call.
        std::env::set_var("ZES_ENABLE_SYSMAN", "1");
        // SAFETY: same reasoning as LevelZero::load_impl — opening
        // first-party Intel code.
        let lib = unsafe { libloading::Library::new(lib_name) }
            .map_err(|e| L0Error::Unavailable(format!("{lib_name}: {e}")))?;
        let zes_init = unsafe { load_sym::<ZesInitFn>(&lib, b"zesInit\0", "zesInit")? };
        let zes_device_enum_memory_modules = unsafe {
            load_sym::<ZesDeviceEnumMemoryModulesFn>(
                &lib,
                b"zesDeviceEnumMemoryModules\0",
                "zesDeviceEnumMemoryModules",
            )?
        };
        let zes_memory_get_state = unsafe {
            load_sym::<ZesMemoryGetStateFn>(
                &lib,
                b"zesMemoryGetState\0",
                "zesMemoryGetState",
            )?
        };
        // Optional entry points — try to load, log on miss but don't fail.
        let zes_device_enum_temperature_sensors = unsafe {
            load_sym::<ZesDeviceEnumTemperatureSensorsFn>(
                &lib,
                b"zesDeviceEnumTemperatureSensors\0",
                "zesDeviceEnumTemperatureSensors",
            )
            .ok()
        };
        let zes_temperature_get_state = unsafe {
            load_sym::<ZesTemperatureGetStateFn>(
                &lib,
                b"zesTemperatureGetState\0",
                "zesTemperatureGetState",
            )
            .ok()
        };
        let zes_device_enum_power_domains = unsafe {
            load_sym::<ZesDeviceEnumPowerDomainsFn>(
                &lib,
                b"zesDeviceEnumPowerDomains\0",
                "zesDeviceEnumPowerDomains",
            )
            .ok()
        };
        let zes_power_get_energy_counter = unsafe {
            load_sym::<ZesPowerGetEnergyCounterFn>(
                &lib,
                b"zesPowerGetEnergyCounter\0",
                "zesPowerGetEnergyCounter",
            )
            .ok()
        };
        let zes_device_enum_frequency_domains = unsafe {
            load_sym::<ZesDeviceEnumFrequencyDomainsFn>(
                &lib,
                b"zesDeviceEnumFrequencyDomains\0",
                "zesDeviceEnumFrequencyDomains",
            )
            .ok()
        };
        let zes_frequency_get_state = unsafe {
            load_sym::<ZesFrequencyGetStateFn>(
                &lib,
                b"zesFrequencyGetState\0",
                "zesFrequencyGetState",
            )
            .ok()
        };
        let zes_device_enum_engine_groups = unsafe {
            load_sym::<ZesDeviceEnumEngineGroupsFn>(
                &lib,
                b"zesDeviceEnumEngineGroups\0",
                "zesDeviceEnumEngineGroups",
            )
            .ok()
        };
        let zes_engine_get_activity = unsafe {
            load_sym::<ZesEngineGetActivityFn>(
                &lib,
                b"zesEngineGetActivity\0",
                "zesEngineGetActivity",
            )
            .ok()
        };
        // Sysman init. Spec says flags should be 0.
        // SAFETY: zes_init signature matches the loader export.
        let init_code = unsafe { zes_init(ZES_INIT_FLAG_PLACEHOLDER) };
        // ZE_RESULT_ERROR_UNSUPPORTED_FEATURE is what Iris Xe returns
        // on some 2026 drivers — Sysman isn't available, but the
        // loader symbols resolve. Treat as Unavailable rather than
        // hard error.
        if init_code != ZE_RESULT_SUCCESS {
            if init_code == ZE_RESULT_ERROR_UNSUPPORTED_FEATURE {
                return Err(L0Error::Unavailable(format!(
                    "zesInit returned UNSUPPORTED_FEATURE (driver doesn't expose Sysman)"
                )));
            }
            return Err(L0Error::Call {
                call: "zesInit",
                code: init_code,
            });
        }
        tracing::info!(lib_name, "Sysman loader initialized");
        Ok(Sysman {
            _lib: lib,
            zes_init,
            zes_device_enum_memory_modules,
            zes_memory_get_state,
            zes_device_enum_temperature_sensors,
            zes_temperature_get_state,
            zes_device_enum_power_domains,
            zes_power_get_energy_counter,
            zes_device_enum_frequency_domains,
            zes_frequency_get_state,
            zes_device_enum_engine_groups,
            zes_engine_get_activity,
        })
    }

    /// Probe one device. Aggregates across all memory modules / temp
    /// sensors / power domains the driver exposes. Reads that
    /// individually fail are silently skipped (treated as "this
    /// sensor unavailable" rather than fatal).
    ///
    /// `device` is the [`ZeDeviceHandle`] from [`LevelZero::devices`]
    /// — Intel uses unified handles for L0 core and Sysman so the
    /// cast is safe.
    pub fn probe(&self, device: ZeDeviceHandle) -> SysmanReading {
        let sys_device = ZesDeviceHandle(device.0);
        let mut reading = SysmanReading::default();

        // ---- Memory ----
        let mut mem_count: u32 = 0;
        // SAFETY: enum-style two-call idiom; first call queries count
        // with null handles array.
        let r = unsafe {
            (self.zes_device_enum_memory_modules)(
                sys_device,
                &mut mem_count,
                std::ptr::null_mut(),
            )
        };
        if r == ZE_RESULT_SUCCESS && mem_count > 0 {
            let mut handles =
                vec![ZesMemHandle(std::ptr::null_mut()); mem_count as usize];
            let r = unsafe {
                (self.zes_device_enum_memory_modules)(
                    sys_device,
                    &mut mem_count,
                    handles.as_mut_ptr(),
                )
            };
            if r == ZE_RESULT_SUCCESS {
                let mut total: u64 = 0;
                let mut free: u64 = 0;
                let mut any = false;
                for h in &handles {
                    let mut state = ZesMemState {
                        stype: ZES_STRUCTURE_TYPE_MEM_STATE,
                        ..Default::default()
                    };
                    let r = unsafe {
                        (self.zes_memory_get_state)(*h, &mut state as *mut _)
                    };
                    if r == ZE_RESULT_SUCCESS {
                        total = total.saturating_add(state.size);
                        free = free.saturating_add(state.free);
                        any = true;
                    }
                }
                if any {
                    reading.vram_total_bytes = Some(total);
                    reading.vram_free_bytes = Some(free);
                }
            }
        }

        // ---- Temperature (optional) ----
        if let (Some(enum_temp), Some(get_temp)) = (
            self.zes_device_enum_temperature_sensors,
            self.zes_temperature_get_state,
        ) {
            let mut temp_count: u32 = 0;
            let r = unsafe { enum_temp(sys_device, &mut temp_count, std::ptr::null_mut()) };
            if r == ZE_RESULT_SUCCESS && temp_count > 0 {
                let mut handles =
                    vec![ZesTempHandle(std::ptr::null_mut()); temp_count as usize];
                let r = unsafe { enum_temp(sys_device, &mut temp_count, handles.as_mut_ptr()) };
                if r == ZE_RESULT_SUCCESS {
                    let mut max_c: Option<f64> = None;
                    for h in &handles {
                        let mut t: f64 = 0.0;
                        let r = unsafe { get_temp(*h, &mut t as *mut f64) };
                        if r == ZE_RESULT_SUCCESS && t.is_finite() {
                            max_c = Some(max_c.map_or(t, |m| m.max(t)));
                        }
                    }
                    reading.max_temp_c = max_c;
                }
            }
        }

        // ---- Power (optional, cumulative energy counter) ----
        if let (Some(enum_pwr), Some(get_energy)) = (
            self.zes_device_enum_power_domains,
            self.zes_power_get_energy_counter,
        ) {
            let mut pwr_count: u32 = 0;
            let r = unsafe { enum_pwr(sys_device, &mut pwr_count, std::ptr::null_mut()) };
            if r == ZE_RESULT_SUCCESS && pwr_count > 0 {
                let mut handles =
                    vec![ZesPwrHandle(std::ptr::null_mut()); pwr_count as usize];
                let r = unsafe { enum_pwr(sys_device, &mut pwr_count, handles.as_mut_ptr()) };
                if r == ZE_RESULT_SUCCESS && !handles.is_empty() {
                    // Sum across power domains (package + cores +
                    // memory, etc.). Single-domain devices report
                    // one number.
                    let mut total_e: u64 = 0;
                    let mut max_ts: u64 = 0;
                    let mut any = false;
                    for h in &handles {
                        let mut counter = ZesPowerEnergyCounter::default();
                        let r =
                            unsafe { get_energy(*h, &mut counter as *mut _) };
                        if r == ZE_RESULT_SUCCESS {
                            total_e = total_e.saturating_add(counter.energy_uj);
                            max_ts = max_ts.max(counter.timestamp_us);
                            any = true;
                        }
                    }
                    if any {
                        reading.energy_counter = Some(ZesPowerEnergyCounter {
                            energy_uj: total_e,
                            timestamp_us: max_ts,
                        });
                    }
                }
            }
        }

        // ---- Frequency (optional, GPU clock) ----
        if let (Some(enum_freq), Some(get_freq)) = (
            self.zes_device_enum_frequency_domains,
            self.zes_frequency_get_state,
        ) {
            let mut freq_count: u32 = 0;
            let r = unsafe { enum_freq(sys_device, &mut freq_count, std::ptr::null_mut()) };
            if r == ZE_RESULT_SUCCESS && freq_count > 0 {
                let mut handles =
                    vec![ZesFreqHandle(std::ptr::null_mut()); freq_count as usize];
                let r = unsafe { enum_freq(sys_device, &mut freq_count, handles.as_mut_ptr()) };
                if r == ZE_RESULT_SUCCESS && !handles.is_empty() {
                    // Pick the first frequency domain. On Intel iGPU
                    // the first domain is the GPU clock; on discrete
                    // Arc, additional domains expose memory + media
                    // clocks. The GPU clock is the headline number
                    // for "is the GPU busy" UI.
                    let mut state = ZesFreqState {
                        stype: ZES_STRUCTURE_TYPE_FREQ_STATE,
                        ..Default::default()
                    };
                    let r =
                        unsafe { get_freq(handles[0], &mut state as *mut _) };
                    if r == ZE_RESULT_SUCCESS && state.actual.is_finite() && state.actual >= 0.0 {
                        reading.gpu_freq_mhz = Some(state.actual);
                    }
                }
            }
        }

        // ---- Engine activity (optional, GPU utilization) ----
        if let (Some(enum_eng), Some(get_act)) = (
            self.zes_device_enum_engine_groups,
            self.zes_engine_get_activity,
        ) {
            let mut eng_count: u32 = 0;
            let r = unsafe { enum_eng(sys_device, &mut eng_count, std::ptr::null_mut()) };
            if r == ZE_RESULT_SUCCESS && eng_count > 0 {
                let mut handles =
                    vec![ZesEngineHandle(std::ptr::null_mut()); eng_count as usize];
                let r = unsafe { enum_eng(sys_device, &mut eng_count, handles.as_mut_ptr()) };
                if r == ZE_RESULT_SUCCESS {
                    // Sum busy time across all engine groups; take the
                    // largest timestamp as the sample instant. The caller
                    // clamps the derived Δactive/Δts ratio to 100 %, so
                    // summing (which can over-count when engines run
                    // concurrently) still yields a sane "GPU busy" bar.
                    let mut active_sum: u64 = 0;
                    let mut ts: u64 = 0;
                    let mut any = false;
                    for h in &handles {
                        let mut stats = ZesEngineStats::default();
                        let r = unsafe { get_act(*h, &mut stats as *mut _) };
                        if r == ZE_RESULT_SUCCESS {
                            active_sum = active_sum.saturating_add(stats.active_time);
                            ts = ts.max(stats.timestamp);
                            any = true;
                        }
                    }
                    if any && ts > 0 {
                        reading.engine_stats = Some(ZesEngineStats {
                            active_time: active_sum,
                            timestamp: ts,
                        });
                    }
                }
            }
        }

        reading
    }
}

/// Look up a single symbol in the loaded library, returning the
/// raw function pointer (not the lifetime-bound `Symbol`). The
/// lifetime is detached so the function pointer can outlive the
/// stack frame; safety is preserved by keeping the `Library` alive
/// inside `LevelZero::_lib`.
///
/// # Safety
///
/// `T` must exactly match the C ABI signature exported by the
/// library under `name`. Cargo can't check this — getting it wrong
/// is UB at the call site.
unsafe fn load_sym<T: Copy>(
    lib: &libloading::Library,
    raw_name: &[u8],
    debug_name: &'static str,
) -> Result<T, L0Error> {
    let sym: libloading::Symbol<T> = lib
        .get(raw_name)
        .map_err(|_| L0Error::MissingSymbol { name: debug_name })?;
    // *sym derefs the Symbol to the underlying T (raw fn ptr).
    Ok(*sym)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_either_succeeds_or_returns_unavailable() {
        // Pin behavior: on any host the call returns either Ok
        // (real driver present) or Err(Unavailable) (no driver),
        // never a panic or an unrelated error. This is the
        // contract the engine relies on for fallback to the SYCL
        // path.
        match LevelZero::load() {
            Ok(_) => {
                // Real driver present (CI-with-Intel-GPU runner).
            }
            Err(L0Error::Unavailable(_)) => {
                // Expected on the default CI runner. Test passes.
            }
            Err(e @ L0Error::Call { .. }) => {
                // zeInit failed with a non-success code — this is
                // a real surface failure worth reporting.
                panic!("zeInit failed unexpectedly: {e}");
            }
            Err(e @ L0Error::MissingSymbol { .. }) => {
                panic!("loader is missing a symbol we depend on: {e}");
            }
        }
    }

    /// Real-hardware probe: run with `cargo test --features l0 -- --ignored l0_real_hardware`.
    /// On a host with an Intel GPU + recent driver, expect:
    ///   - `LevelZero::load()` returns Ok
    ///   - `drivers()` returns at least 1 entry
    ///   - `devices()` on the first driver returns at least 1 device
    #[test]
    #[ignore = "requires Intel GPU + ze_loader.dll; run with --ignored"]
    fn l0_real_hardware() {
        let l0 = LevelZero::load().expect("load");
        let drivers = l0.drivers().expect("drivers()");
        assert!(!drivers.is_empty(), "no L0 drivers visible");
        let devs = l0.devices(drivers[0]).expect("devices()");
        assert!(!devs.is_empty(), "driver has no devices");
        eprintln!(
            "L0 probe: {} driver(s), {} device(s) on driver[0]",
            drivers.len(),
            devs.len()
        );
    }

    /// Sysman loader on hosts without an Intel GPU returns
    /// `Unavailable` cleanly (no panics, no missing-symbol). Pin
    /// the contract the engine relies on when running mock-mode
    /// metrics + reporting "no GPU detected".
    #[test]
    fn sysman_load_either_succeeds_or_returns_unavailable() {
        match Sysman::load() {
            Ok(_) => {
                // Real Intel GPU + Sysman-capable driver present.
            }
            Err(L0Error::Unavailable(_)) => {
                // Default CI / non-Intel host.
            }
            Err(L0Error::MissingSymbol { name }) => {
                panic!("Sysman loader is missing required symbol: {name}");
            }
            Err(L0Error::Call { call, code }) => {
                panic!("Sysman init failed unexpectedly at {call} (code 0x{code:08x})");
            }
        }
    }

    /// Real-hardware Sysman probe. Run with
    /// `cargo test -- --ignored sysman_real_hardware` on an Intel GPU
    /// host. On Iris Xe + recent driver this returns at minimum the
    /// VRAM total + free; temperature and power are best-effort.
    ///
    /// Ordering note: Sysman is loaded FIRST so `zesInit` runs before
    /// `zeInit`. Intel's L0 loader locks into "core mode" if `zeInit`
    /// runs first, which makes `zesInit` return `UNINITIALIZED`.
    /// Setting `ZES_ENABLE_SYSMAN=1` in the environment before
    /// either call is the legacy-mode workaround; the modern
    /// init-order workaround is what we do here.
    #[test]
    #[ignore = "requires Intel GPU + ze_loader.dll Sysman; run with --ignored"]
    fn sysman_real_hardware() {
        let sysman = match Sysman::load() {
            Ok(s) => s,
            Err(L0Error::Unavailable(msg)) => {
                eprintln!("Sysman unavailable on this host: {msg}");
                return;
            }
            Err(e) => panic!("Sysman load failed: {e}"),
        };
        // After Sysman init, do the L0 core init for device enum.
        let l0 = LevelZero::load().expect("L0 load");
        let drivers = l0.drivers().expect("L0 drivers");
        assert!(!drivers.is_empty());
        let devs = l0.devices(drivers[0]).expect("L0 devices");
        assert!(!devs.is_empty());
        let reading = sysman.probe(devs[0]);
        eprintln!("Sysman probe: {reading:?}");
        // VRAM probe is the headline ask — it's the one that should
        // work on every Intel GPU since Iris Xe in 2025+.
        match reading.vram_total_bytes {
            Some(total) => assert!(total > 0, "VRAM total reported zero — driver bug?"),
            None => eprintln!("VRAM probe not exposed on this driver"),
        }
    }
}
