//! AMD Ryzen AI / XDNA NPU COMPUTE offload via ONNX Runtime + the VitisAI EP —
//! the AMD analog of the Intel OpenVINO offload ([`crate::offload`]).
//!
//! WRITE-BLIND: there is no Ryzen AI SDK or AMD NPU on the dev box, so this is
//! authored against the published ONNX Runtime C API and is NOT runtime-verified;
//! it is validated on Ryzen AI hardware via `doctor --npu-parity` + the tuner
//! verdict (same MO as the OpenVINO path). Everything FAILS SOFT (returns `None`)
//! off AMD hardware or on any FFI error, so nothing here can break a non-AMD run.
//!
//! Strategy (whole-graph, like Intel): XDNA is reached only through the vendor
//! runtime, so to offload one op we hand ONNX Runtime a single-`Gemm` ONNX model
//! ([`crate::onnx_ir::gemm_model`]), append the **VitisAI** (Ryzen AI) execution
//! provider so ORT partitions that op onto the NPU, build a session (cached per
//! shape — session build runs the EP's graph compiler and is expensive), and run
//! it. Correctness is parity-gated vs the CPU reference; *when* to route is the
//! measured tier ranking, same as the Intel path.
//!
//! **OrtApi access:** the `OrtApi` is a large struct of ~300 function pointers in
//! a fixed, append-only order. Rather than lay all of them out, we treat
//! `*const OrtApi` as an array of function pointers and index it by each
//! function's ORDINAL (its 0-based position in the struct). The ordinals below
//! are taken from `onnxruntime_c_api.h` and are stable across ORT versions
//! (the struct only ever grows at the end).

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CString};
use std::os::raw::c_int;
use std::ptr;
use std::sync::Mutex;

// Opaque ORT handle types.
#[repr(C)]
struct OrtEnv {
    _p: [u8; 0],
}
#[repr(C)]
struct OrtSessionOptions {
    _p: [u8; 0],
}
#[repr(C)]
struct OrtSession {
    _p: [u8; 0],
}
#[repr(C)]
struct OrtValue {
    _p: [u8; 0],
}
#[repr(C)]
struct OrtMemoryInfo {
    _p: [u8; 0],
}
#[repr(C)]
struct OrtStatus {
    _p: [u8; 0],
}
#[repr(C)]
struct OrtRunOptions {
    _p: [u8; 0],
}

/// `struct OrtApiBase { const OrtApi* (*GetApi)(uint32_t); const char*
/// (*GetVersionString)(void); }`.
#[repr(C)]
struct OrtApiBase {
    get_api: unsafe extern "C" fn(u32) -> *const *const c_void,
    get_version_string: unsafe extern "C" fn() -> *const c_char,
}

// ---- OrtApi function-pointer ordinals (0-based index in `struct OrtApi`,
// ---- from onnxruntime_c_api.h; append-only ⇒ stable across versions) ----
const ORD_GET_ERROR_CODE: usize = 1;
const ORD_CREATE_ENV: usize = 3;
const ORD_CREATE_SESSION_FROM_ARRAY: usize = 8;
const ORD_RUN: usize = 9;
const ORD_CREATE_SESSION_OPTIONS: usize = 10;
const ORD_SET_GRAPH_OPT_LEVEL: usize = 23;
const ORD_CREATE_TENSOR_WITH_DATA: usize = 49;
const ORD_GET_TENSOR_MUTABLE_DATA: usize = 51;
const ORD_CREATE_CPU_MEMORY_INFO: usize = 69;
const ORD_RELEASE_ENV: usize = 92;
const ORD_RELEASE_STATUS: usize = 93;
const ORD_RELEASE_MEMORY_INFO: usize = 94;
const ORD_RELEASE_SESSION: usize = 95;
const ORD_RELEASE_VALUE: usize = 96;
const ORD_RELEASE_SESSION_OPTIONS: usize = 100;

// ORT enum values (from onnxruntime_c_api.h).
const ORT_LOGGING_LEVEL_WARNING: c_int = 2;
const ORT_ENABLE_ALL: c_int = 99; // GraphOptimizationLevel
const ORT_ARENA_ALLOCATOR: c_int = 1; // OrtAllocatorType
const ORT_MEM_TYPE_DEFAULT: c_int = 0; // OrtMemType
const ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: c_int = 1;
/// The ORT API version to request from `GetApi`. All functions we call are in
/// the base struct (ordinals ≤ 100), present since early versions, so a low
/// value loads on any modern ORT (GetApi returns null only if the lib is OLDER
/// than requested).
const ORT_API_VERSION_REQ: u32 = 11;

// ---- FFI fn-pointer types ----
type FnCreateEnv = unsafe extern "C" fn(c_int, *const c_char, *mut *mut OrtEnv) -> *mut OrtStatus;
type FnCreateSessionOptions = unsafe extern "C" fn(*mut *mut OrtSessionOptions) -> *mut OrtStatus;
type FnSetGraphOpt = unsafe extern "C" fn(*mut OrtSessionOptions, c_int) -> *mut OrtStatus;
type FnCreateSessionFromArray = unsafe extern "C" fn(
    *const OrtEnv,
    *const c_void,
    usize,
    *const OrtSessionOptions,
    *mut *mut OrtSession,
) -> *mut OrtStatus;
type FnCreateCpuMemoryInfo =
    unsafe extern "C" fn(c_int, c_int, *mut *mut OrtMemoryInfo) -> *mut OrtStatus;
type FnCreateTensorWithData = unsafe extern "C" fn(
    *const OrtMemoryInfo,
    *mut c_void,
    usize,
    *const i64,
    usize,
    c_int,
    *mut *mut OrtValue,
) -> *mut OrtStatus;
type FnGetTensorMutableData =
    unsafe extern "C" fn(*mut OrtValue, *mut *mut c_void) -> *mut OrtStatus;
type FnRun = unsafe extern "C" fn(
    *mut OrtSession,
    *const OrtRunOptions,
    *const *const c_char,
    *const *const OrtValue,
    usize,
    *const *const c_char,
    usize,
    *mut *mut OrtValue,
) -> *mut OrtStatus;
type FnGetErrorCode = unsafe extern "C" fn(*const OrtStatus) -> c_int;
type FnReleaseEnv = unsafe extern "C" fn(*mut OrtEnv);
type FnReleaseSessionOptions = unsafe extern "C" fn(*mut OrtSessionOptions);
type FnReleaseSession = unsafe extern "C" fn(*mut OrtSession);
type FnReleaseValue = unsafe extern "C" fn(*mut OrtValue);
type FnReleaseMemoryInfo = unsafe extern "C" fn(*mut OrtMemoryInfo);
type FnReleaseStatus = unsafe extern "C" fn(*mut OrtStatus);
/// Dedicated VitisAI (Ryzen AI) EP append — a flat exported symbol (NOT in the
/// OrtApi struct), present when the VitisAI EP is built into the runtime.
type FnAppendVitisAI = unsafe extern "C" fn(
    *mut OrtSessionOptions,
    *const *const c_char,
    *const *const c_char,
    usize,
) -> *mut OrtStatus;

/// Resolved ORT entry points (copied out of the OrtApi fn-pointer array + the
/// dedicated VitisAI symbol).
struct Api {
    create_env: FnCreateEnv,
    create_session_options: FnCreateSessionOptions,
    set_graph_opt: FnSetGraphOpt,
    create_session_from_array: FnCreateSessionFromArray,
    create_cpu_memory_info: FnCreateCpuMemoryInfo,
    create_tensor_with_data: FnCreateTensorWithData,
    get_tensor_mutable_data: FnGetTensorMutableData,
    run: FnRun,
    get_error_code: FnGetErrorCode,
    release_env: FnReleaseEnv,
    release_session_options: FnReleaseSessionOptions,
    release_session: FnReleaseSession,
    release_value: FnReleaseValue,
    release_memory_info: FnReleaseMemoryInfo,
    release_status: FnReleaseStatus,
    append_vitisai: Option<FnAppendVitisAI>,
}

/// Session cache keyed by `(bucket_m, a_cols, b_rows, b_cols, transpose_b)`.
type SessionCache = Mutex<HashMap<(usize, usize, usize, usize, bool), *mut OrtSession>>;

/// A loaded ONNX Runtime with an Env + a CPU memory-info, able to build + run
/// single-op NPU graphs on the AMD NPU. `None` from [`AmdNpuExecutor::new`] when
/// ORT / the VitisAI EP / an AMD NPU is absent.
pub struct AmdNpuExecutor {
    _lib: libloading::Library,
    api: Api,
    env: *mut OrtEnv,
    mem_info: *mut OrtMemoryInfo,
    sessions: SessionCache,
}

// SAFETY: ORT Env/Session are thread-safe for concurrent Run per the C++ contract
// the C API wraps; the session cache is Mutex-guarded. OrtValues are per-call.
unsafe impl Send for AmdNpuExecutor {}
unsafe impl Sync for AmdNpuExecutor {}

/// Candidate ONNX Runtime library names (the Ryzen AI installer puts these on
/// PATH / the loader search path).
#[cfg(windows)]
const ORT_LIBS: &[&str] = &["onnxruntime.dll"];
#[cfg(not(windows))]
const ORT_LIBS: &[&str] = &["libonnxruntime.so", "libonnxruntime.so.1"];

impl AmdNpuExecutor {
    /// Load ONNX Runtime, resolve the API, create an Env + the VitisAI session
    /// prototype. `None` if the runtime is absent or any FFI call fails.
    pub fn new() -> Option<AmdNpuExecutor> {
        unsafe { Self::new_inner() }
    }

    unsafe fn new_inner() -> Option<AmdNpuExecutor> {
        // dlopen ORT (exe-relative candidates then bare name, like the other
        // bundled runtimes).
        let cands: Vec<std::path::PathBuf> = ORT_LIBS
            .iter()
            .flat_map(|n| crate::bundled_lib_candidates(n))
            .collect();
        let lib = cands
            .into_iter()
            .find_map(|c| libloading::Library::new(&c).ok())?;

        // OrtGetApiBase() -> *const OrtApiBase -> GetApi(version) -> OrtApi array.
        let get_api_base: libloading::Symbol<unsafe extern "C" fn() -> *const OrtApiBase> =
            lib.get(b"OrtGetApiBase\0").ok()?;
        let base = get_api_base();
        if base.is_null() {
            return None;
        }
        let arr = ((*base).get_api)(ORT_API_VERSION_REQ);
        if arr.is_null() {
            return None; // lib older than the requested API version.
        }

        // Resolve each function by its confirmed ordinal.
        let api = Api {
            create_env: resolve(arr, ORD_CREATE_ENV),
            create_session_options: resolve(arr, ORD_CREATE_SESSION_OPTIONS),
            set_graph_opt: resolve(arr, ORD_SET_GRAPH_OPT_LEVEL),
            create_session_from_array: resolve(arr, ORD_CREATE_SESSION_FROM_ARRAY),
            create_cpu_memory_info: resolve(arr, ORD_CREATE_CPU_MEMORY_INFO),
            create_tensor_with_data: resolve(arr, ORD_CREATE_TENSOR_WITH_DATA),
            get_tensor_mutable_data: resolve(arr, ORD_GET_TENSOR_MUTABLE_DATA),
            run: resolve(arr, ORD_RUN),
            get_error_code: resolve(arr, ORD_GET_ERROR_CODE),
            release_env: resolve(arr, ORD_RELEASE_ENV),
            release_session_options: resolve(arr, ORD_RELEASE_SESSION_OPTIONS),
            release_session: resolve(arr, ORD_RELEASE_SESSION),
            release_value: resolve(arr, ORD_RELEASE_VALUE),
            release_memory_info: resolve(arr, ORD_RELEASE_MEMORY_INFO),
            release_status: resolve(arr, ORD_RELEASE_STATUS),
            // Dedicated VitisAI EP symbol — absent ⇒ no NPU offload (CPU EP),
            // which is why `has_vitisai()` gates executor creation below.
            append_vitisai: lib
                .get::<FnAppendVitisAI>(b"OrtSessionOptionsAppendExecutionProvider_VitisAI\0")
                .ok()
                .map(|s| *s),
        };

        // Without the VitisAI EP there is no AMD-NPU offload (ORT would just run
        // on CPU), so treat its absence as "no NPU" → inert.
        api.append_vitisai?;

        let logid = CString::new("rustllama").ok()?;
        let mut env: *mut OrtEnv = ptr::null_mut();
        if !(api.create_env)(ORT_LOGGING_LEVEL_WARNING, logid.as_ptr(), &mut env).is_null()
            || env.is_null()
        {
            return None;
        }
        let mut mem_info: *mut OrtMemoryInfo = ptr::null_mut();
        if !(api.create_cpu_memory_info)(ORT_ARENA_ALLOCATOR, ORT_MEM_TYPE_DEFAULT, &mut mem_info)
            .is_null()
            || mem_info.is_null()
        {
            (api.release_env)(env);
            return None;
        }

        Some(AmdNpuExecutor {
            _lib: lib,
            api,
            env,
            mem_info,
            sessions: Mutex::new(HashMap::new()),
        })
    }

    /// `out[n,m] = X[n,k] · W[m,k]ᵀ`, all f32, on the AMD NPU. `None` on any
    /// failure (caller falls back). Mirrors the OpenVINO `matmul_xwt_f32`.
    pub fn matmul_xwt_f32(
        &self,
        x: &[f32],
        n: usize,
        k: usize,
        w: &[f32],
        m: usize,
    ) -> Option<Vec<f32>> {
        if x.len() != n * k || w.len() != m * k || m == 0 || k == 0 || n == 0 {
            return None;
        }
        unsafe { self.exec_gemm(x, n, k, w, m, k, true) }
    }

    /// Row-major `C[m,n] = A[m,k] · B[k,n]`, all f32 (parity-probe entry).
    pub fn matmul_f32(
        &self,
        a: &[f32],
        m: usize,
        k: usize,
        b: &[f32],
        n: usize,
    ) -> Option<Vec<f32>> {
        if a.len() != m * k || b.len() != k * n || m == 0 || k == 0 || n == 0 {
            return None;
        }
        unsafe { self.exec_gemm(a, m, k, b, k, n, false) }
    }

    /// Build (cached) + run a Gemm of `A[a_rows,a_cols]` by `B[b_rows,b_cols]`
    /// with optional `transpose_b`. M-bucketed (see [`crate::offload`]): the
    /// batch dim `a_rows` is padded up to a power-of-two bucket so distinct
    /// prefill lengths reuse one compiled session; the output is sliced back.
    #[allow(clippy::too_many_arguments)]
    unsafe fn exec_gemm(
        &self,
        a: &[f32],
        a_rows: usize,
        a_cols: usize,
        b: &[f32],
        b_rows: usize,
        b_cols: usize,
        transpose_b: bool,
    ) -> Option<Vec<f32>> {
        let out_cols = if transpose_b { b_rows } else { b_cols };
        let m_bucket = bucket_m(a_rows);
        let session = self.session_for(m_bucket, a_cols, b_rows, b_cols, transpose_b)?;

        // Pad A up to the bucket row count (zero rows → zero output rows we drop).
        let a_padded: std::borrow::Cow<[f32]> = if m_bucket == a_rows {
            std::borrow::Cow::Borrowed(a)
        } else {
            let mut v = vec![0f32; m_bucket * a_cols];
            v[..a_rows * a_cols].copy_from_slice(a);
            std::borrow::Cow::Owned(v)
        };

        // Input OrtValues wrapping our host buffers (CreateTensorWithData does not
        // copy — the buffers must outlive the Run, which they do).
        let a_shape = [m_bucket as i64, a_cols as i64];
        let b_shape = [b_rows as i64, b_cols as i64];
        let ta = self.make_tensor(&a_shape, &a_padded)?;
        let tb = match self.make_tensor(&b_shape, b) {
            Some(t) => t,
            None => {
                (self.api.release_value)(ta);
                return None;
            }
        };

        let in_names = [c"A".as_ptr(), c"B".as_ptr()];
        let out_names = [c"Y".as_ptr()];
        let inputs = [ta as *const OrtValue, tb as *const OrtValue];
        let mut out_val: *mut OrtValue = ptr::null_mut();

        let status = (self.api.run)(
            session,
            ptr::null(),
            in_names.as_ptr(),
            inputs.as_ptr(),
            2,
            out_names.as_ptr(),
            1,
            &mut out_val,
        );
        let ok = self.status_ok(status);

        let result = if ok && !out_val.is_null() {
            self.read_tensor(out_val, m_bucket * out_cols)
        } else {
            None
        };
        if !out_val.is_null() {
            (self.api.release_value)(out_val);
        }
        (self.api.release_value)(ta);
        (self.api.release_value)(tb);

        // Slice the padded output back to the real `a_rows`.
        result.map(|full| {
            if m_bucket == a_rows {
                full
            } else {
                full[..a_rows * out_cols].to_vec()
            }
        })
    }

    /// Get (or build + cache) the NPU session for this shape.
    unsafe fn session_for(
        &self,
        a_rows: usize,
        a_cols: usize,
        b_rows: usize,
        b_cols: usize,
        transpose_b: bool,
    ) -> Option<*mut OrtSession> {
        let key = (a_rows, a_cols, b_rows, b_cols, transpose_b);
        if let Ok(cache) = self.sessions.lock() {
            if let Some(&s) = cache.get(&key) {
                return Some(s);
            }
        }
        let session = self.build_session(a_rows, a_cols, b_rows, b_cols, transpose_b)?;
        if let Ok(mut cache) = self.sessions.lock() {
            if let Some(&existing) = cache.get(&key) {
                (self.api.release_session)(session);
                return Some(existing);
            }
            cache.insert(key, session);
        }
        Some(session)
    }

    /// Synthesize the ONNX Gemm model, append the VitisAI EP, build the session.
    unsafe fn build_session(
        &self,
        a_rows: usize,
        a_cols: usize,
        b_rows: usize,
        b_cols: usize,
        transpose_b: bool,
    ) -> Option<*mut OrtSession> {
        let model = crate::onnx_ir::gemm_model(a_rows, a_cols, b_rows, b_cols, transpose_b);

        let mut opts: *mut OrtSessionOptions = ptr::null_mut();
        if !self.status_ok((self.api.create_session_options)(&mut opts)) || opts.is_null() {
            return None;
        }
        // Best-effort graph-opt level + VitisAI EP (resolved in `new`).
        let _ = (self.api.set_graph_opt)(opts, ORT_ENABLE_ALL);
        if let Some(append) = self.api.append_vitisai {
            // No provider options (defaults); the Ryzen AI installer supplies the
            // xclbin/overlay via its own config.
            let _ = append(opts, ptr::null(), ptr::null(), 0);
        }

        let mut session: *mut OrtSession = ptr::null_mut();
        let st = (self.api.create_session_from_array)(
            self.env,
            model.as_ptr() as *const c_void,
            model.len(),
            opts,
            &mut session,
        );
        let ok = self.status_ok(st);
        (self.api.release_session_options)(opts);
        if ok && !session.is_null() {
            Some(session)
        } else {
            None
        }
    }

    /// Wrap a host f32 buffer in an OrtValue tensor of `shape`.
    unsafe fn make_tensor(&self, shape: &[i64], data: &[f32]) -> Option<*mut OrtValue> {
        let mut t: *mut OrtValue = ptr::null_mut();
        let st = (self.api.create_tensor_with_data)(
            self.mem_info,
            data.as_ptr() as *mut c_void,
            std::mem::size_of_val(data),
            shape.as_ptr(),
            shape.len(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut t,
        );
        if self.status_ok(st) && !t.is_null() {
            Some(t)
        } else {
            None
        }
    }

    /// Copy `count` f32 out of an output tensor.
    unsafe fn read_tensor(&self, t: *mut OrtValue, count: usize) -> Option<Vec<f32>> {
        let mut dptr: *mut c_void = ptr::null_mut();
        if !self.status_ok((self.api.get_tensor_mutable_data)(t, &mut dptr)) || dptr.is_null() {
            return None;
        }
        let mut v = vec![0f32; count];
        ptr::copy_nonoverlapping(dptr as *const f32, v.as_mut_ptr(), count);
        Some(v)
    }

    /// `true` if an OrtStatus is success (null); a non-null status is an error —
    /// free it and return false. (We don't surface the message; the caller just
    /// falls back to CPU/GPU.)
    unsafe fn status_ok(&self, status: *mut OrtStatus) -> bool {
        if status.is_null() {
            return true;
        }
        let _ = (self.api.get_error_code)(status); // touch for completeness
        (self.api.release_status)(status);
        false
    }
}

impl Drop for AmdNpuExecutor {
    fn drop(&mut self) {
        unsafe {
            if let Ok(mut cache) = self.sessions.lock() {
                for (_, s) in cache.drain() {
                    (self.api.release_session)(s);
                }
            }
            if !self.mem_info.is_null() {
                (self.api.release_memory_info)(self.mem_info);
            }
            if !self.env.is_null() {
                (self.api.release_env)(self.env);
            }
        }
    }
}

/// Copy a function pointer out of the OrtApi array at `ordinal`.
///
/// SAFETY: `arr` is the `*const OrtApi` ORT handed us, which is a C struct of
/// same-size function pointers; `ordinal` is a valid in-bounds index from
/// `onnxruntime_c_api.h`; `T` is the matching `extern "C"` fn-pointer type.
unsafe fn resolve<T: Copy>(arr: *const *const c_void, ordinal: usize) -> T {
    std::mem::transmute_copy::<*const c_void, T>(&*arr.add(ordinal))
}

/// Round the batch dim `M` up to a power-of-two bucket (min 16) so distinct
/// prefill lengths reuse one compiled session. Mirrors the OpenVINO path.
fn bucket_m(m: usize) -> usize {
    m.next_power_of_two().max(16)
}

use crate::NpuParity;

/// Validate the AMD NPU f32 matmul against a naive CPU reference on a small
/// random problem. `supported == false` when no AMD NPU / ORT + VitisAI EP is
/// present (a SKIP). Thresholds mirror the GPU/Intel parity harness.
pub(crate) fn matmul_parity() -> NpuParity {
    let skip = NpuParity {
        supported: false,
        cos: 0.0,
        max_rel: 0.0,
        pass: false,
    };
    let Some(exec) = AmdNpuExecutor::new() else {
        return skip;
    };
    let (m, k, n) = (3usize, 5usize, 4usize);
    let a: Vec<f32> = (0..m * k)
        .map(|i| (((i * 2654435761) >> 8) as f32 % 17.0) - 8.0)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| (((i * 40503) >> 4) as f32 % 13.0) - 6.0)
        .collect();

    let Some(got) = exec.matmul_f32(&a, m, k, &b, n) else {
        // NPU present but the offload FAILED (bad model / session / infer) →
        // surface as a correctness failure so it gets investigated on hardware.
        return NpuParity {
            supported: true,
            cos: 0.0,
            max_rel: f64::INFINITY,
            pass: false,
        };
    };
    let mut want = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f32;
            for j in 0..k {
                acc += a[r * k + j] * b[j * n + c];
            }
            want[r * n + c] = acc;
        }
    }
    let (mut dot, mut na, mut nb, mut max_rel) = (0f64, 0f64, 0f64, 0f64);
    for i in 0..want.len() {
        let (x, y) = (got[i] as f64, want[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        let denom = y.abs().max(1e-6);
        max_rel = max_rel.max((x - y).abs() / denom);
    }
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    };
    let pass = cos >= 0.9999 && max_rel <= 1e-2;
    NpuParity {
        supported: true,
        cos,
        max_rel,
        pass,
    }
}
