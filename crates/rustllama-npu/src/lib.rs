//! Backend-agnostic NPU (Neural Processing Unit) device model.
//!
//! NPUs are reached only through a vendor runtime — Intel via OpenVINO, AMD via
//! the Ryzen AI / XRT stack — so the actual enumeration + offload FFI lives in
//! feature-gated vendor backends. Intel (`npu-openvino`) is REAL: `openvino.rs`
//! enumerates NPUs through the OpenVINO C API (linked from the SDK by
//! `build.rs`). AMD (`npu-ryzenai`) is a stub until a Ryzen AI SDK is available
//! to compile against. This crate is the vendor-neutral surface the rest of the
//! runtime uses, so NPUs get the same device management as GPUs:
//! **enumeration**, **disable-by-index** (mirroring `RUSTLLAMA_DISABLED_GPUS`),
//! and a **fingerprint** slug folded into the tuner's system fingerprint.
//!
//! With no `npu-*` feature (or no NPU hardware) [`enumerate`] returns empty and
//! the whole surface is inert — identical behavior to a build without NPUs.
//!
//! Device-index space: NPUs occupy a 0-based range of their own; the unified
//! cross-backend index space (GPUs offset by SYCL count, then CUDA, then NPUs)
//! is assembled by the placement layer, not here.

use serde::{Deserialize, Serialize};

// Intel NPU enumeration/offload FFI over the OpenVINO C API. Behind the feature
// so the default build neither links nor loads OpenVINO.
#[cfg(feature = "npu-openvino")]
mod openvino;

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
pub fn enumerate() -> Vec<NpuInfo> {
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

// Intel NPU enumeration via the real OpenVINO C API FFI (`src/openvino.rs`).
#[cfg(feature = "npu-openvino")]
fn openvino_enumerate() -> Vec<NpuInfo> {
    openvino::enumerate()
}

// AMD NPU enumeration stub — the Ryzen AI / XRT FFI lands in a later phase
// (write-blind: no Ryzen AI SDK on this box). Present now so the device model +
// feature gate are complete.
#[cfg(feature = "npu-ryzenai")]
fn ryzenai_enumerate() -> Vec<NpuInfo> {
    tracing::debug!("npu-ryzenai: enumeration not yet implemented");
    Vec::new()
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

    // Only the default (no vendor backend) build is guaranteed deviceless; a
    // build with a real backend (e.g. npu-openvino) may enumerate NPUs on NPU
    // hardware, so the "empty" invariant is scoped to the stub-free build.
    #[cfg(not(any(feature = "npu-openvino", feature = "npu-ryzenai")))]
    #[test]
    fn no_backend_enumerates_empty() {
        // No npu-* feature in a plain build → no devices, inert fingerprint.
        assert!(enumerate().is_empty());
        assert_eq!(npu_fingerprint(&[]), "npu:none");
        assert!(usable_npus(&[]).is_empty());
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
