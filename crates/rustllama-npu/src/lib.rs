//! Backend-agnostic NPU (Neural Processing Unit) device model.
//!
//! NPUs are reached only through a vendor runtime — Intel via OpenVINO, AMD via
//! the Ryzen AI / XRT stack — so the actual enumeration + offload FFI lives in
//! feature-gated vendor backends. Both are ON by default and RUNTIME-LOAD their
//! vendor runtime (`openvino.rs` dlopen's `openvino_c`; `ryzenai.rs` dlopen's
//! `xrt_coreutil`): the release artifact BUNDLES those Apache-2.0 runtimes next
//! to the binary, so NPU support works with no user install, and a box without
//! the runtime (or without an NPU) enumerates nothing — fail-soft, never a
//! load-time crash. This crate is the vendor-neutral surface the rest of the
//! runtime uses, so NPUs get the same device management as GPUs:
//! **enumeration**, **disable-by-index** (mirroring `RUSTLLAMA_DISABLED_GPUS`),
//! and a **fingerprint** slug folded into the tuner's system fingerprint.
//!
//! With `--no-default-features` (or no NPU hardware) [`enumerate`] returns empty
//! and the whole surface is inert — identical behavior to a build without NPUs.
//!
//! Device-index space: NPUs occupy a 0-based range of their own; the unified
//! cross-backend index space (GPUs offset by SYCL count, then CUDA, then NPUs)
//! is assembled by the placement layer, not here.

use serde::{Deserialize, Serialize};

// Intel NPU enumeration/offload FFI over the OpenVINO C API. Behind the feature
// so the default build neither links nor loads OpenVINO.
#[cfg(feature = "npu-openvino")]
mod openvino;

// AMD NPU enumeration over the XRT runtime (runtime-loaded; see ryzenai.rs).
#[cfg(feature = "npu-ryzenai")]
mod ryzenai;

// AMD NPU COMPUTE offload (Phase 5, whole-graph) — a single-op ONNX model built
// here, run on the Ryzen AI / XDNA NPU via ONNX Runtime + the VitisAI EP.
#[cfg(feature = "npu-ryzenai")]
mod onnx_ir;

// Intel NPU COMPUTE offload (Phase 5) — op-level MatMul via OpenVINO. Write-blind
// (validated on NPU HW via `doctor --npu-parity` + the tuner verdict).
#[cfg(feature = "npu-openvino")]
mod offload;

#[cfg(feature = "npu-openvino")]
pub use offload::NpuExecutor;

// AMD NPU COMPUTE offload (Phase 5, whole-graph) — a single-op ONNX Gemm run on
// the Ryzen AI / XDNA NPU via ONNX Runtime + the VitisAI EP. Write-blind
// (validated on Ryzen AI HW via `doctor --npu-parity` + the tuner verdict).
#[cfg(feature = "npu-ryzenai")]
mod offload_amd;

#[cfg(feature = "npu-ryzenai")]
pub use offload_amd::AmdNpuExecutor;

/// NPU vendor (which runtime drives it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NpuVendor {
    Intel,
    Amd,
    Other,
}

/// One enumerated NPU device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NpuInfo {
    pub vendor: NpuVendor,
    /// Human-readable device name (e.g. "Intel(R) AI Boost").
    pub name: String,
    /// 0-based index within the NPU device space.
    pub index: u32,
    /// Device memory in bytes; 0 when unknown (NPUs often share host memory).
    pub memory_bytes: u64,
}

/// Enumerate all NPUs across the enabled vendor backends, re-indexed 0-based
/// across the merged set. Empty without any `npu-*` feature or NPU hardware.
///
/// Probed once per process and cached: the vendor dlopen + runtime init is
/// non-trivial, and `enumerate` feeds hot paths (the system fingerprint, the
/// placement tier build), so repeating it would be wasteful. The device set
/// does not change within a process run.
pub fn enumerate() -> Vec<NpuInfo> {
    static CACHE: std::sync::OnceLock<Vec<NpuInfo>> = std::sync::OnceLock::new();
    CACHE.get_or_init(enumerate_uncached).clone()
}

/// The actual probe (see [`enumerate`]). Not self-referential — the vendor
/// backends never call back into [`enumerate`] — so it is safe to run inside the
/// `OnceLock` initializer.
fn enumerate_uncached() -> Vec<NpuInfo> {
    #[allow(unused_mut)]
    let mut out: Vec<NpuInfo> = Vec::new();
    #[cfg(feature = "npu-openvino")]
    out.extend(openvino_enumerate());
    #[cfg(feature = "npu-ryzenai")]
    out.extend(ryzenai_enumerate());
    for (i, info) in out.iter_mut().enumerate() {
        info.index = i as u32;
    }
    out
}

// Intel NPU enumeration via the OpenVINO C API, runtime-loaded (`src/openvino.rs`).
#[cfg(feature = "npu-openvino")]
fn openvino_enumerate() -> Vec<NpuInfo> {
    openvino::enumerate()
}

// AMD NPU enumeration via the XRT runtime FFI (`src/ryzenai.rs`). The XRT call
// path is write-blind (validated on Ryzen AI HW) but compiles here and fails
// soft to an empty list off AMD NPU hardware.
#[cfg(feature = "npu-ryzenai")]
fn ryzenai_enumerate() -> Vec<NpuInfo> {
    ryzenai::enumerate()
}

/// Candidate filesystem locations for a BUNDLED vendor runtime library, tried in
/// order by the vendor loaders: next to the binary (the Windows release stages
/// DLLs flat there), in a `lib/` subdir next to the binary (the Linux release's
/// bundled-`.so` dir), then the bare name (a system install / PATH /
/// LD_LIBRARY_PATH). Trying an ABSOLUTE exe-relative path is what makes the
/// bundled case work on Linux: `dlopen` of a bare name does NOT search the
/// executable's directory or its `DT_RUNPATH`, only `DT_RPATH` / LD_LIBRARY_PATH
/// / the default dirs — so a `.so` sitting in `./lib` would otherwise be missed.
#[cfg(any(feature = "npu-openvino", feature = "npu-ryzenai"))]
fn bundled_lib_candidates(name: &str) -> Vec<std::path::PathBuf> {
    let mut v = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.join(name)); // flat next to the binary (Windows)
            v.push(dir.join("lib").join(name)); // ./lib next to the binary (Linux)
        }
    }
    v.push(std::path::PathBuf::from(name)); // system / PATH / LD_LIBRARY_PATH
    v
}

/// Result of the NPU compute-offload parity self-check vs the CPU reference
/// (Phase 5). Feature-agnostic so the `doctor` + tuner-verdict wiring can depend
/// on it in any build; `supported == false` is a SKIP (no NPU / backend off).
#[derive(Debug, Clone, Copy)]
pub struct NpuParity {
    /// An NPU + the offload path were available (else the check is a SKIP).
    pub supported: bool,
    /// Cosine similarity vs the CPU reference (1.0 == identical direction).
    pub cos: f64,
    /// Max relative elementwise error vs the CPU reference.
    pub max_rel: f64,
    /// Passed the correctness thresholds.
    pub pass: bool,
}

impl NpuParity {
    /// The SKIP result — no NPU / offload backend present.
    pub const fn skipped() -> NpuParity {
        NpuParity { supported: false, cos: 0.0, max_rel: 0.0, pass: false }
    }
}

/// Whether Intel NPU COMPUTE offload (OpenVINO MatMul) is available on this host
/// — the backend is compiled in AND an NPU + its runtime are present. Always
/// `false` without the `npu-openvino` feature or off NPU hardware.
pub fn npu_offload_available() -> bool {
    #[cfg(feature = "npu-openvino")]
    {
        return offload::NpuExecutor::new().is_some();
    }
    #[allow(unreachable_code)]
    false
}

/// Run the NPU matmul parity self-check (Phase 5). SKIP ([`NpuParity::skipped`])
/// without the backend or an NPU; otherwise the measured cos / max_rel / pass
/// vs the CPU reference — read by `doctor --npu-parity` and the tuner verdict
/// (`npu:matmul_f32`) that fail-closed-gates the dispatch.
pub fn npu_matmul_parity() -> NpuParity {
    #[cfg(feature = "npu-openvino")]
    {
        return offload::matmul_parity();
    }
    #[allow(unreachable_code)]
    NpuParity::skipped()
}

/// Whether AMD Ryzen AI NPU COMPUTE offload (ONNX Runtime + VitisAI EP) is
/// available — the backend is compiled in AND ORT + the VitisAI EP + an AMD NPU
/// are present. `false` without `npu-ryzenai` or off Ryzen AI hardware.
pub fn amd_npu_offload_available() -> bool {
    #[cfg(feature = "npu-ryzenai")]
    {
        return offload_amd::AmdNpuExecutor::new().is_some();
    }
    #[allow(unreachable_code)]
    false
}

/// Run the AMD NPU matmul parity self-check (Phase 5). SKIP without the backend
/// or an AMD NPU; otherwise the measured cos / max_rel / pass vs the CPU
/// reference — read by `doctor --npu-parity` and the tuner verdict
/// (`npu:amd_matmul_f32`) that fail-closed-gates the AMD dispatch.
pub fn amd_npu_matmul_parity() -> NpuParity {
    #[cfg(feature = "npu-ryzenai")]
    {
        return offload_amd::matmul_parity();
    }
    #[allow(unreachable_code)]
    NpuParity::skipped()
}

/// Parse a comma-separated index list (e.g. "0,2"), ignoring blanks/garbage —
/// matches the `RUSTLLAMA_DISABLED_GPUS` parsing.
fn parse_indices(s: &str) -> Vec<u32> {
    s.split(',')
        .filter_map(|t| t.trim().parse::<u32>().ok())
        .collect()
}

/// NPU indices disabled via `RUSTLLAMA_DISABLED_NPUS` (comma-separated), the NPU
/// analog of `RUSTLLAMA_DISABLED_GPUS`.
pub fn disabled_npu_indices_env() -> Vec<u32> {
    std::env::var("RUSTLLAMA_DISABLED_NPUS")
        .ok()
        .map(|s| parse_indices(&s))
        .unwrap_or_default()
}

/// Whether the NPU tier is enabled at all. `RUSTLLAMA_NPU_ENABLED=0` (promoted
/// from `[inference].npu_enabled = false`) sidelines ALL NPUs. Default true.
pub fn npu_enabled_env() -> bool {
    std::env::var("RUSTLLAMA_NPU_ENABLED")
        .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
        .unwrap_or(true)
}

/// Usable NPUs: enumerated minus the disabled set (and empty when the tier is
/// disabled via `RUSTLLAMA_NPU_ENABLED=0`). `extra_disabled` carries the
/// config's `[inference].disabled_npus` (merged with the env list).
pub fn usable_npus(extra_disabled: &[u32]) -> Vec<NpuInfo> {
    if !npu_enabled_env() {
        return Vec::new();
    }
    let mut disabled = disabled_npu_indices_env();
    disabled.extend_from_slice(extra_disabled);
    enumerate()
        .into_iter()
        .filter(|n| !disabled.contains(&n.index))
        .collect()
}

/// A stable fingerprint slug of the USABLE NPU set (vendor + name + memory,
/// sorted), for inclusion in `rustllama_tuner::system_fingerprint()` so a tuned
/// cache is keyed by the NPU set exactly as it is by the GPU set. `"npu:none"`
/// when there are no usable NPUs (the common case today).
pub fn npu_fingerprint(extra_disabled: &[u32]) -> String {
    let npus = usable_npus(extra_disabled);
    if npus.is_empty() {
        return "npu:none".to_string();
    }
    let mut parts: Vec<String> = npus
        .iter()
        .map(|n| format!("{:?}:{}:{}", n.vendor, n.name, n.memory_bytes))
        .collect();
    parts.sort();
    format!("npu:{}", parts.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_indices_ignores_garbage() {
        assert_eq!(parse_indices("0,2, 3 ,x,"), vec![0, 2, 3]);
        assert_eq!(parse_indices(""), Vec::<u32>::new());
    }

    // Phase-5 offload must FAIL SOFT off NPU hardware: no panic, a SKIP result,
    // and the invariants pass⇒supported and available⇒parity-supported hold.
    #[cfg(feature = "npu-openvino")]
    #[test]
    fn offload_parity_is_fail_soft() {
        let p = npu_matmul_parity();
        assert!(!p.pass || p.supported, "pass can't be true without support");
        assert!(!npu_offload_available() || p.supported);
    }

    // A `--no-default-features` (deviceless) build is guaranteed to enumerate
    // nothing; a build with a vendor backend may enumerate NPUs on NPU hardware,
    // so the "empty" invariant is scoped to the no-backend build.
    #[cfg(not(any(feature = "npu-openvino", feature = "npu-ryzenai")))]
    #[test]
    fn no_backend_enumerates_empty() {
        // No npu-* feature → no devices, inert fingerprint.
        assert!(enumerate().is_empty());
        assert_eq!(npu_fingerprint(&[]), "npu:none");
        assert!(usable_npus(&[]).is_empty());
    }

    // With a vendor backend compiled in, enumeration must FAIL SOFT on a host
    // lacking the bundled runtime / an NPU — i.e. not panic and respect the
    // disable gate. (On actual NPU hardware with the runtime present this would
    // return devices; the assertions hold either way.)
    #[cfg(any(feature = "npu-openvino", feature = "npu-ryzenai"))]
    #[test]
    fn backend_enumeration_is_fail_soft() {
        let all = enumerate();
        // Disabling every enumerated index yields no usable NPUs.
        let all_idx: Vec<u32> = all.iter().map(|n| n.index).collect();
        assert!(usable_npus(&all_idx).is_empty());
        // Fingerprint is deterministic and non-empty.
        assert!(!npu_fingerprint(&[]).is_empty());
    }

    #[test]
    fn fingerprint_is_stable_and_sorted() {
        let a = NpuInfo { vendor: NpuVendor::Intel, name: "B".into(), index: 0, memory_bytes: 0 };
        let b = NpuInfo { vendor: NpuVendor::Amd, name: "A".into(), index: 1, memory_bytes: 0 };
        // Build the slug the way npu_fingerprint does, order-independently.
        let mut p: Vec<String> = [&a, &b]
            .iter()
            .map(|n| format!("{:?}:{}:{}", n.vendor, n.name, n.memory_bytes))
            .collect();
        p.sort();
        assert_eq!(p[0], "Amd:A:0");
        assert_eq!(p[1], "Intel:B:0");
    }
}
