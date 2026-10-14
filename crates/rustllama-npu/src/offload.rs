//! Intel NPU COMPUTE offload via OpenVINO — the Phase-5 op-offload primitive.
//!
//! WRITE-BLIND: authored against the OpenVINO 2026.4 C API but NOT run on an NPU
//! here (no NPU hardware on the dev box). It compiles + is parity-gated; the
//! numeric path is validated on NPU hardware via `doctor --npu-parity` and the
//! `tune --validate-kernels` verdict, exactly like the Blackwell/Hopper/XMX
//! tensor-core kernels. Every entry point FAILS SOFT (returns `None`) off NPU
//! hardware or on any FFI error, so nothing here can break a non-NPU run.
//!
//! Strategy: NPUs (Intel XDNA) are whole-graph accelerators reached only through
//! the vendor runtime — we cannot emit NPU ISA ourselves — so to offload a
//! single op we hand OpenVINO a MINIMAL in-memory IR graph for that op (here a
//! `MatMul`), compile it to the `"NPU"` device (cached per shape — compilation
//! runs the ~96 MB NPU graph compiler and is expensive), and run it through an
//! infer request. Correctness is gated by a parity self-check vs the CPU
//! reference (fail-closed); *when* to route to the NPU is gated by the measured
//! tier ranking (a shape the NPU is slow at simply is not routed there), so this
//! module only has to be CORRECT, not universally fast.
//!
//! Granularity note: a fresh (M,K,N) requires a compile, so the win is on the
//! fixed-shape, compute-heavy PREFILL GEMMs (the weight shapes are fixed; M is
//! the prompt length). Per-token decode matvec is latency-bound by the host
//! round-trip and stays on CPU/GPU. M is bucketed to a power of two
//! ([`bucket_m`]) so distinct prefill lengths reuse one compiled graph.

use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::Mutex;

/// `ov_status_e` — C enum, `OK == 0`.
type OvStatus = c_int;
const OV_OK: OvStatus = 0;
/// `ov_element_type_e::F32` (DYNAMIC=0, BOOLEAN, BF16, F16, F32=4, …).
const OV_F32: c_int = 4;

#[repr(C)]
struct OvCore {
    _p: [u8; 0],
}
#[repr(C)]
struct OvModel {
    _p: [u8; 0],
}
#[repr(C)]
struct OvCompiledModel {
    _p: [u8; 0],
}
#[repr(C)]
struct OvInferRequest {
    _p: [u8; 0],
}
#[repr(C)]
struct OvTensor {
    _p: [u8; 0],
}

/// `ov_shape_t { int64_t rank; int64_t* dims; }`.
#[repr(C)]
struct OvShape {
    rank: i64,
    dims: *mut i64,
}

/// Compiled-graph cache keyed by `(a_rows, a_cols, b_rows, b_cols, transpose_b)`.
type MatmulCache = Mutex<HashMap<(usize, usize, usize, usize, bool), *mut OvCompiledModel>>;

/// `ov_available_devices_t { char** devices; size_t size; }`.
#[repr(C)]
struct OvAvailableDevices {
    devices: *mut *mut c_char,
    size: usize,
}

// FFI fn-pointer types (mirror the 2026.4 C API headers exactly).
type FnCoreCreate = unsafe extern "C" fn(*mut *mut OvCore) -> OvStatus;
type FnCoreFree = unsafe extern "C" fn(*mut OvCore);
type FnGetDevices = unsafe extern "C" fn(*const OvCore, *mut OvAvailableDevices) -> OvStatus;
type FnDevicesFree = unsafe extern "C" fn(*mut OvAvailableDevices);
type FnReadFromMemory = unsafe extern "C" fn(
    *const OvCore,
    *const c_char,
    usize,
    *const OvTensor,
    *mut *mut OvModel,
) -> OvStatus;
type FnCompileProps = unsafe extern "C" fn(
    *const OvCore,
    *const OvModel,
    *const c_char,
    usize,
    *const c_void,
    *mut *mut OvCompiledModel,
) -> OvStatus;
type FnModelFree = unsafe extern "C" fn(*mut OvModel);
type FnCreateRequest =
    unsafe extern "C" fn(*const OvCompiledModel, *mut *mut OvInferRequest) -> OvStatus;
type FnCompiledFree = unsafe extern "C" fn(*mut OvCompiledModel);
type FnShapeCreate = unsafe extern "C" fn(i64, *const i64, *mut OvShape) -> OvStatus;
type FnShapeFree = unsafe extern "C" fn(*mut OvShape) -> OvStatus;
type FnTensorCreate = unsafe extern "C" fn(c_int, OvShape, *mut *mut OvTensor) -> OvStatus;
type FnTensorData = unsafe extern "C" fn(*const OvTensor, *mut *mut c_void) -> OvStatus;
type FnTensorFree = unsafe extern "C" fn(*mut OvTensor);
type FnSetInput =
    unsafe extern "C" fn(*mut OvInferRequest, usize, *const OvTensor) -> OvStatus;
type FnGetOutput =
    unsafe extern "C" fn(*const OvInferRequest, usize, *mut *mut OvTensor) -> OvStatus;
type FnInfer = unsafe extern "C" fn(*mut OvInferRequest) -> OvStatus;
type FnRequestFree = unsafe extern "C" fn(*mut OvInferRequest);

/// Resolved OpenVINO entry points (copied out of the held `Library`).
struct Api {
    core_free: FnCoreFree,
    get_devices: FnGetDevices,
    devices_free: FnDevicesFree,
    read_from_memory: FnReadFromMemory,
    compile_props: FnCompileProps,
    model_free: FnModelFree,
    create_request: FnCreateRequest,
    compiled_free: FnCompiledFree,
    shape_create: FnShapeCreate,
    shape_free: FnShapeFree,
    tensor_create: FnTensorCreate,
    tensor_data: FnTensorData,
    tensor_free: FnTensorFree,
    set_input: FnSetInput,
    get_output: FnGetOutput,
    infer: FnInfer,
    request_free: FnRequestFree,
}

/// A loaded OpenVINO runtime with a live Core, able to compile + run single-op
/// graphs on the Intel NPU. `None` from [`NpuExecutor::new`] off NPU hardware.
pub struct NpuExecutor {
    _lib: libloading::Library, // keep the runtime loaded (the fn ptrs borrow it)
    api: Api,
    core: *mut OvCore,
    // Compiled MatMul graphs (see MatmulCache) — compilation is expensive, so
    // amortize across the fixed prefill shapes.
    matmul_cache: MatmulCache,
}

// SAFETY: the OpenVINO Core + compiled models are thread-safe for compile/infer
// per the C++ contract the C API wraps; the per-shape cache is additionally
// guarded by a Mutex. Infer requests are created per call (never shared).
unsafe impl Send for NpuExecutor {}
unsafe impl Sync for NpuExecutor {}

impl NpuExecutor {
    /// Load OpenVINO, create a Core, and verify an `"NPU"` device exists. `None`
    /// if the runtime is absent, no NPU is present, or any FFI call fails.
    pub fn new() -> Option<NpuExecutor> {
        unsafe { Self::new_inner() }
    }

    unsafe fn new_inner() -> Option<NpuExecutor> {
        let lib = crate::openvino::load_runtime().ok()?;
        let core_create = *sym::<FnCoreCreate>(&lib, b"ov_core_create\0")?;
        let api = Api {
            core_free: *sym(&lib, b"ov_core_free\0")?,
            get_devices: *sym(&lib, b"ov_core_get_available_devices\0")?,
            devices_free: *sym(&lib, b"ov_available_devices_free\0")?,
            read_from_memory: *sym(&lib, b"ov_core_read_model_from_memory_buffer\0")?,
            compile_props: *sym(&lib, b"ov_core_compile_model_props\0")?,
            model_free: *sym(&lib, b"ov_model_free\0")?,
            create_request: *sym(&lib, b"ov_compiled_model_create_infer_request\0")?,
            compiled_free: *sym(&lib, b"ov_compiled_model_free\0")?,
            shape_create: *sym(&lib, b"ov_shape_create\0")?,
            shape_free: *sym(&lib, b"ov_shape_free\0")?,
            tensor_create: *sym(&lib, b"ov_tensor_create\0")?,
            tensor_data: *sym(&lib, b"ov_tensor_data\0")?,
            tensor_free: *sym(&lib, b"ov_tensor_free\0")?,
            set_input: *sym(&lib, b"ov_infer_request_set_input_tensor_by_index\0")?,
            get_output: *sym(&lib, b"ov_infer_request_get_output_tensor_by_index\0")?,
            infer: *sym(&lib, b"ov_infer_request_infer\0")?,
            request_free: *sym(&lib, b"ov_infer_request_free\0")?,
        };
        let mut core: *mut OvCore = ptr::null_mut();
        if core_create(&mut core) != OV_OK || core.is_null() {
            return None;
        }
        let exec = NpuExecutor { _lib: lib, api, core, matmul_cache: Mutex::new(HashMap::new()) };
        if !exec.has_npu() {
            return None; // Core created but no NPU device → inert.
        }
        Some(exec)
    }

    /// True if OpenVINO reports an `"NPU"` device.
    unsafe fn has_npu(&self) -> bool {
        let mut devs = OvAvailableDevices { devices: ptr::null_mut(), size: 0 };
        if (self.api.get_devices)(self.core, &mut devs) != OV_OK || devs.devices.is_null() {
            return false;
        }
        let mut found = false;
        let names = std::slice::from_raw_parts(devs.devices, devs.size);
        for &p in names {
            if p.is_null() {
                continue;
            }
            let d = std::ffi::CStr::from_ptr(p).to_string_lossy();
            if d == "NPU" || d.starts_with("NPU.") {
                found = true;
                break;
            }
        }
        (self.api.devices_free)(&mut devs);
        found
    }

    /// Row-major `C[m,n] = A[m,k] · B[k,n]`, all f32, computed on the NPU.
    /// `None` on any failure (caller falls back to CPU/GPU). Thread-safe.
    pub fn matmul_f32(&self, a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Option<Vec<f32>> {
        if a.len() != m * k || b.len() != k * n || m == 0 || k == 0 || n == 0 {
            return None;
        }
        unsafe { self.exec_matmul(a, m, k, b, k, n, false) }
    }

    /// `out[n,m] = X[n,k] · W[m,k]ᵀ` (OpenVINO `transpose_b=true`), the layout
    /// `rustllama-models`' batched matvec dispatch uses (weight stored `m×k`,
    /// activations `n×k`, output `n×m`). `None` on any failure. Thread-safe.
    pub fn matmul_xwt_f32(&self, x: &[f32], n: usize, k: usize, w: &[f32], m: usize) -> Option<Vec<f32>> {
        if x.len() != n * k || w.len() != m * k || m == 0 || k == 0 || n == 0 {
            return None;
        }
        unsafe { self.exec_matmul(x, n, k, w, m, k, true) }
    }

    /// Compile (cached) + run a MatMul of `A[a_rows,a_cols]` by `B[b_rows,b_cols]`
    /// with optional `transpose_b`. Output is `a_rows × (transpose_b ? b_rows :
    /// b_cols)`.
    ///
    /// **M-bucketing:** the batch dim `a_rows` (the prompt length on a prefill
    /// GEMM) is rounded up to a power-of-two bucket ([`bucket_m`]) so distinct
    /// prefill lengths share ONE compiled graph — compilation runs the ~96 MB
    /// NPU graph compiler and dwarfs the GEMM, so amortizing it across lengths is
    /// the dominant win. The activation is zero-padded to the bucket and the
    /// output sliced back, so the result is identical to an exact-`a_rows`
    /// compile (the padded rows produce padded output rows we drop).
    #[allow(clippy::too_many_arguments)]
    unsafe fn exec_matmul(
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
        let compiled = self.compiled_matmul(m_bucket, a_cols, b_rows, b_cols, transpose_b)?;

        // Pad the activation up to the bucket's row count (zero rows → zero
        // output rows, which we slice off). Borrow when already bucket-aligned.
        let a_padded: std::borrow::Cow<[f32]> = if m_bucket == a_rows {
            std::borrow::Cow::Borrowed(a)
        } else {
            let mut v = vec![0f32; m_bucket * a_cols];
            v[..a_rows * a_cols].copy_from_slice(a);
            std::borrow::Cow::Owned(v)
        };

        let mut req: *mut OvInferRequest = ptr::null_mut();
        if (self.api.create_request)(compiled, &mut req) != OV_OK || req.is_null() {
            return None;
        }
        // Guard so the request is always freed.
        let out = (|| {
            let ta = self.make_input(&[m_bucket, a_cols], &a_padded)?;
            let tb = self.make_input(&[b_rows, b_cols], b)?;
            let ok = (self.api.set_input)(req, 0, ta) == OV_OK
                && (self.api.set_input)(req, 1, tb) == OV_OK
                && (self.api.infer)(req) == OV_OK;
            let result = if ok { self.read_output(req, m_bucket * out_cols) } else { None };
            (self.api.tensor_free)(ta);
            (self.api.tensor_free)(tb);
            result
        })();
        (self.api.request_free)(req);
        // Slice the padded output back to the real `a_rows`.
        out.map(|full| {
            if m_bucket == a_rows {
                full
            } else {
                full[..a_rows * out_cols].to_vec()
            }
        })
    }

    /// Get (or compile + cache) the NPU MatMul graph for this shape.
    unsafe fn compiled_matmul(
        &self,
        a_rows: usize,
        a_cols: usize,
        b_rows: usize,
        b_cols: usize,
        transpose_b: bool,
    ) -> Option<*mut OvCompiledModel> {
        let key = (a_rows, a_cols, b_rows, b_cols, transpose_b);
        if let Ok(cache) = self.matmul_cache.lock() {
            if let Some(&c) = cache.get(&key) {
                return Some(c);
            }
        }
        let compiled = self.compile_matmul(a_rows, a_cols, b_rows, b_cols, transpose_b)?;
        if let Ok(mut cache) = self.matmul_cache.lock() {
            // Re-check: another thread may have compiled the same shape.
            if let Some(&existing) = cache.get(&key) {
                (self.api.compiled_free)(compiled);
                return Some(existing);
            }
            cache.insert(key, compiled);
        }
        Some(compiled)
    }

    /// Read the IR, compile it to the NPU. Frees the source model.
    unsafe fn compile_matmul(
        &self,
        a_rows: usize,
        a_cols: usize,
        b_rows: usize,
        b_cols: usize,
        transpose_b: bool,
    ) -> Option<*mut OvCompiledModel> {
        let xml = matmul_ir(a_rows, a_cols, b_rows, b_cols, transpose_b);
        let xml_c = CString::new(xml).ok()?;
        let bytes = xml_c.as_bytes(); // without the NUL

        let mut model: *mut OvModel = ptr::null_mut();
        if (self.api.read_from_memory)(
            self.core,
            xml_c.as_ptr(),
            bytes.len(),
            ptr::null(), // no weights (both operands are graph inputs)
            &mut model,
        ) != OV_OK
            || model.is_null()
        {
            return None;
        }

        let dev = CString::new("NPU").ok()?;
        let mut compiled: *mut OvCompiledModel = ptr::null_mut();
        let st = (self.api.compile_props)(
            self.core,
            model,
            dev.as_ptr(),
            0,
            ptr::null(),
            &mut compiled,
        );
        (self.api.model_free)(model);
        if st != OV_OK || compiled.is_null() {
            return None;
        }
        Some(compiled)
    }

    /// Allocate an input tensor of `dims` and copy `data` into it.
    unsafe fn make_input(&self, dims: &[usize], data: &[f32]) -> Option<*mut OvTensor> {
        let dims64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let mut shape = OvShape { rank: 0, dims: ptr::null_mut() };
        if (self.api.shape_create)(dims64.len() as i64, dims64.as_ptr(), &mut shape) != OV_OK {
            return None;
        }
        let mut t: *mut OvTensor = ptr::null_mut();
        let st = (self.api.tensor_create)(OV_F32, OvShape { rank: shape.rank, dims: shape.dims }, &mut t);
        // tensor_create copies the shape contents; free our shape buffer.
        (self.api.shape_free)(&mut shape);
        if st != OV_OK || t.is_null() {
            return None;
        }
        let mut dptr: *mut c_void = ptr::null_mut();
        if (self.api.tensor_data)(t, &mut dptr) != OV_OK || dptr.is_null() {
            (self.api.tensor_free)(t);
            return None;
        }
        ptr::copy_nonoverlapping(data.as_ptr(), dptr as *mut f32, data.len());
        Some(t)
    }

    /// Read `count` f32 from output tensor 0 of `req`.
    unsafe fn read_output(&self, req: *mut OvInferRequest, count: usize) -> Option<Vec<f32>> {
        let mut t: *mut OvTensor = ptr::null_mut();
        if (self.api.get_output)(req, 0, &mut t) != OV_OK || t.is_null() {
            return None;
        }
        let mut dptr: *mut c_void = ptr::null_mut();
        let res = if (self.api.tensor_data)(t, &mut dptr) == OV_OK && !dptr.is_null() {
            let mut v = vec![0f32; count];
            ptr::copy_nonoverlapping(dptr as *const f32, v.as_mut_ptr(), count);
            Some(v)
        } else {
            None
        };
        (self.api.tensor_free)(t);
        res
    }
}

impl Drop for NpuExecutor {
    fn drop(&mut self) {
        unsafe {
            if let Ok(mut cache) = self.matmul_cache.lock() {
                for (_, c) in cache.drain() {
                    (self.api.compiled_free)(c);
                }
            }
            if !self.core.is_null() {
                (self.api.core_free)(self.core);
            }
        }
    }
}

/// Round the batch dim `M` (a prefill GEMM's prompt length) up to a
/// compile-amortizing bucket: the next power of two, min 16. A compile runs the
/// ~96 MB NPU graph compiler, so collapsing the many distinct prefill lengths
/// onto ~log2(max_ctx) buckets turns per-length compiles into a handful. The
/// padding waste is at most ~2× the GEMM — negligible beside a fresh compile,
/// and only on the compute-amortized prefill path.
fn bucket_m(m: usize) -> usize {
    m.next_power_of_two().max(16)
}

/// Minimal OpenVINO IR (v11) for `C = MatMul(A[ar,ac], B[br,bc])` with optional
/// `transpose_b`, both f32 graph inputs. Output is `ar × (transpose_b ? br :
/// bc)`. Static shapes (the NPU compiler wants them); no weights blob needed.
fn matmul_ir(ar: usize, ac: usize, br: usize, bc: usize, transpose_b: bool) -> String {
    let tb = if transpose_b { "true" } else { "false" };
    let oc = if transpose_b { br } else { bc };
    format!(
        r#"<?xml version="1.0"?>
<net name="rsl_matmul" version="11">
  <layers>
    <layer id="0" name="A" type="Parameter" version="opset1">
      <data shape="{ar},{ac}" element_type="f32"/>
      <output><port id="0" precision="FP32" names="A"><dim>{ar}</dim><dim>{ac}</dim></port></output>
    </layer>
    <layer id="1" name="B" type="Parameter" version="opset1">
      <data shape="{br},{bc}" element_type="f32"/>
      <output><port id="0" precision="FP32" names="B"><dim>{br}</dim><dim>{bc}</dim></port></output>
    </layer>
    <layer id="2" name="MM" type="MatMul" version="opset1">
      <data transpose_a="false" transpose_b="{tb}"/>
      <input>
        <port id="0" precision="FP32"><dim>{ar}</dim><dim>{ac}</dim></port>
        <port id="1" precision="FP32"><dim>{br}</dim><dim>{bc}</dim></port>
      </input>
      <output><port id="2" precision="FP32" names="C"><dim>{ar}</dim><dim>{oc}</dim></port></output>
    </layer>
    <layer id="3" name="C/sink" type="Result" version="opset1">
      <input><port id="0" precision="FP32"><dim>{ar}</dim><dim>{oc}</dim></port></input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
    <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
  </edges>
</net>
"#
    )
}

/// Look up one symbol; `None` (not panic) if absent.
unsafe fn sym<'a, T>(lib: &'a libloading::Library, name: &[u8]) -> Option<libloading::Symbol<'a, T>> {
    lib.get(name).ok()
}

use crate::NpuParity;

/// Validate the NPU f32 matmul against a naive CPU reference on a small random
/// problem. `supported == false` when no NPU/offload is present (a SKIP, not a
/// failure). Thresholds mirror the GPU parity harness (cos ≥ 0.9999, max_rel
/// small) — the NPU does f32 so only FMA/reassociation differences are expected.
pub(crate) fn matmul_parity() -> NpuParity {
    let skip = NpuParity { supported: false, cos: 0.0, max_rel: 0.0, pass: false };
    let Some(exec) = NpuExecutor::new() else {
        return skip;
    };
    // Small, deterministic pseudo-random inputs (no rng dep).
    let (m, k, n) = (3usize, 5usize, 4usize);
    let a: Vec<f32> = (0..m * k).map(|i| (((i * 2654435761) >> 8) as f32 % 17.0) - 8.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (((i * 40503) >> 4) as f32 % 13.0) - 6.0).collect();

    let Some(got) = exec.matmul_f32(&a, m, k, &b, n) else {
        // An NPU is present but the offload FAILED (bad IR / compile / infer).
        // Surface it as a correctness failure, not a clean SKIP, so the
        // write-blind path gets investigated on hardware rather than silently
        // looking "unsupported".
        return NpuParity { supported: true, cos: 0.0, max_rel: f64::INFINITY, pass: false };
    };
    // CPU reference.
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
    let cos = if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 0.0 };
    let pass = cos >= 0.9999 && max_rel <= 1e-2;
    NpuParity { supported: true, cos, max_rel, pass }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_m_rounds_to_pow2_min16() {
        assert_eq!(bucket_m(1), 16);
        assert_eq!(bucket_m(16), 16);
        assert_eq!(bucket_m(17), 32);
        assert_eq!(bucket_m(31), 32);
        assert_eq!(bucket_m(64), 64);
        assert_eq!(bucket_m(129), 256);
        assert_eq!(bucket_m(2048), 2048);
        // A bucket never shrinks the batch (no lost rows).
        for m in [1usize, 15, 100, 777, 4096] {
            assert!(bucket_m(m) >= m);
        }
    }

    #[test]
    fn matmul_ir_output_cols_follow_transpose_b() {
        // transpose_b=false → out cols = b_cols; true → out cols = b_rows.
        assert!(matmul_ir(16, 8, 8, 32, false).contains("<dim>16</dim><dim>32</dim>"));
        assert!(matmul_ir(16, 8, 32, 8, true).contains("<dim>16</dim><dim>32</dim>"));
    }
}
