//! Tensor and device abstractions.
//!
//! Phase 1 ships a real CPU storage backend (`Storage::CpuOwned`) holding
//! contiguous bytes with typed `bytemuck` views. SYCL / USM storage lands in
//! phase 3.

use std::sync::Arc;

use bytemuck::Pod;
use half::f16;

use rustllama_gguf::dequant;
use rustllama_gguf::{GgmlType, Gguf, GgufError, TensorInfo};

/// Logical compute device.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    /// SYCL device by zero-based index. Resolved against `sycl-ls` ordering.
    Sycl(u32),
}

impl Device {
    /// Parse from the config-file format: `"cpu"` or `"sycl:N"`.
    /// Returns `Err(s)` for anything else.
    ///
    /// Note: explicit `std::result::Result` since this module has its
    /// own `Result<T>` alias bound to `TensorError`.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }
        if let Some(rest) = s.strip_prefix("sycl:").or_else(|| s.strip_prefix("SYCL:")) {
            let idx: u32 = rest.parse().map_err(|_| {
                format!("device `{s}`: invalid SYCL index after `sycl:`")
            })?;
            return Ok(Self::Sycl(idx));
        }
        Err(format!(
            "device `{s}`: expected `cpu` or `sycl:N` (N=0..device_count-1)"
        ))
    }

    /// Whether this device targets a SYCL backend.
    pub fn is_sycl(&self) -> bool {
        matches!(self, Self::Sycl(_))
    }

    /// SYCL device index, if any.
    pub fn sycl_index(&self) -> Option<u32> {
        if let Self::Sycl(i) = self {
            Some(*i)
        } else {
            None
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Sycl(i) => write!(f, "sycl:{i}"),
        }
    }
}

/// Element type of a tensor. The quantized variants mirror `GgmlType` so a
/// GGUF tensor maps 1:1 to a [`Tensor`].
///
/// Variant naming follows the GGUF format-name convention
/// (`Q4_K`, `Q5_0`, `Q6_K`, etc.) rather than strict Rust CamelCase so the
/// mapping to the underlying GGUF type is unambiguous at every call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
    /// (Conceptual) Q8_0 — used as a logical type marker only.
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    /// Q8_0 with raw GGUF block bytes preserved in storage (no dequant).
    /// Matvec dispatches to a fused dequant+FMA kernel that reads the raw
    /// bytes directly. ~½ the memory of `F16` after load, with only a
    /// modest compute overhead.
    Q8_0Raw,
    /// Q4_0 with raw GGUF block bytes preserved in storage (no dequant).
    /// 18 bytes per 32 weights vs 64 bytes for F16 — ~72% memory reduction.
    /// 4-bit signed values: 4 bits per weight packed two-per-byte in
    /// `qs`, then subtract 8 for signed range [-8, 7]. The legacy
    /// no-min, no-high-bit baseline format that pre-dates K-quants.
    Q4_0Raw,
    /// Q5_0 with raw GGUF block bytes preserved in storage (no dequant).
    /// 22 bytes per 32 weights vs 64 bytes for F16 — ~65% memory reduction.
    /// 5-bit signed values: 4 low bits in `qs`, 1 high bit in `qh`, then
    /// subtract 16 for signed range [-16, 15].
    Q5_0Raw,
    /// Q4_1 with raw GGUF block bytes preserved (no dequant). 20 bytes
    /// per 32 weights vs 64 bytes for F16 — ~69% memory reduction.
    /// Asymmetric 4-bit: per-block `(d, m)` pair gives `value = d * q + m`
    /// with `q` an unsigned nibble in `[0, 15]`. Same per-block bandwidth
    /// shape as Q4_0 plus one extra f16 for the min/offset.
    Q4_1Raw,
    /// Q5_1 with raw GGUF block bytes preserved (no dequant). 24 bytes
    /// per 32 weights vs 64 bytes for F16 — ~63% memory reduction.
    /// Asymmetric 5-bit: per-block `(d, m)` pair gives `value = d * q + m`
    /// with `q` an unsigned 5-bit value `(low4 | bit5 << 4)` in `[0, 31]`.
    /// Same `qh`/`qs` arrangement as Q5_0 plus one extra f16 for `m`.
    Q5_1Raw,
    /// BF16 ("brain float") with raw GGUF block bytes preserved (no
    /// dequant). 2 bytes per element — same memory cost as F16, but
    /// without the precision loss for values outside f16's range
    /// (≈ |x| > 65504 or |x| < 6.1e-5). On the matvec hot path each
    /// u16 is widened to f32 by a zero-fill left-shift (literally
    /// `(u16 << 16) as f32_bits`); that's a single instruction on
    /// SSE/AVX2/AVX-512 so the compute overhead vs F16 is negligible.
    /// Preferred over `F16` for BF16-source GGUFs so the model's
    /// numeric range stays intact end-to-end.
    Bf16Raw,
    /// Q8_K with raw GGUF block bytes preserved (no dequant). 292
    /// bytes per 256 weights — ~9.125 bpw, the largest of the
    /// quantized formats. Layout: f32 `d` + 256 `i8` quants + 32
    /// bytes of precomputed sub-block sums (used for K-quant matmul
    /// acceleration; ignored on the Q8_K × F32 matvec path). Per-
    /// weight value is `d * q` where `q ∈ [-128, 127]`. Typically
    /// shipped as an intermediate activation dtype for K-quant
    /// matmuls rather than as weight storage; we accept it on the
    /// weight side anyway for completeness.
    Q8_KRaw,
    /// Q2_K with raw GGUF block bytes preserved (no dequant). 84 bytes
    /// per 256 weights vs 512 bytes for F16 — ~84% memory reduction at
    /// 2.625 bpw, the smallest of the K-quant family. Layout: 16 scale
    /// bytes (`(scale, min)` 4-bit pairs per sub-block), 64 q-bytes
    /// (2-bit values packed 4-per-byte across 4 shift positions), f16
    /// `d` + f16 `dmin`. Per-sub-block: `dl = d * (scale & 0xF)`,
    /// `ml = dmin * (scale >> 4)`; per-weight: `value = dl * q - ml`,
    /// `q ∈ [0, 3]`. Same asymmetric-min recipe as Q4_K/Q5_K with the
    /// nibble width dropped to 2 bits.
    Q2_KRaw,
    /// Q3_K with raw GGUF block bytes preserved (no dequant). 110 bytes
    /// per 256 weights vs 512 bytes for F16 — ~78% memory reduction at
    /// 3.4375 bpw. K-quant super-block: 32-byte `hmask` (high bit per
    /// weight) + 64-byte `qs` (low 2 bits per weight, packed across 4
    /// shift positions) + 12-byte packed signed 6-bit sub-scales + f16
    /// `d`. Each 16-weight sub-block uses `dl = d * (scale - 32)`; per-
    /// weight value = `dl * (low2 - (hi_bit_set ? 0 : 4))`, giving the
    /// 8-value range `-4..3` that's the "3-bit signed" reconstruction.
    Q3_KRaw,
    /// Q4_K with raw GGUF block bytes preserved in storage (no dequant).
    /// 144 bytes per 256 weights vs 512 bytes for F16 — ~72% memory
    /// reduction. K-quant super-block: f16 d + f16 dmin + packed (sc, min)
    /// pairs for 8 sub-blocks + 128 bytes of 4-bit unsigned quants. Dequant
    /// formula: `value = d * sc[k] * q4 - dmin * mn[k]`.
    Q4_KRaw,
    /// Q5_K with raw GGUF block bytes preserved in storage (no dequant).
    /// 176 bytes per 256 weights vs 512 bytes for F16 — ~66% memory
    /// reduction. Same layout as Q4_K plus a 32-byte `qh` array providing
    /// the 5th bit per weight. Dequant: `value = d * sc[k] * (q5_lo | q5_hi << 4) - dmin * mn[k]`.
    Q5_KRaw,
    /// Q6_K with raw GGUF block bytes preserved in storage (no dequant).
    /// 210 bytes per 256 weights vs 512 bytes for F16 — ~59% memory
    /// reduction. K-quant super-block: 128B `ql` (low 4 bits per weight) +
    /// 64B `qh` (high 2 bits, packed 4-per-byte) + 16B signed-i8 scales +
    /// 2B f16 super-block scale. Dequant: `value = d * sc[is] * (ql_nibble | qh_2bits<<4) - 32`.
    Q6_KRaw,
    /// IQ4_XS with raw GGUF block bytes preserved in storage (no dequant).
    /// 136 bytes per 256 weights vs 512 bytes for F16 — ~73% memory
    /// reduction. Each 4-bit weight is an index into a 16-entry
    /// signed-int8 codebook (`KVALUES_IQ4NL`) biased toward zero.
    /// Eight 6-bit signed sub-scales packed across `scales_l` (4 bytes,
    /// low nibbles) and `scales_h` (u16, high 2 bits per sub-block).
    IQ4_XSRaw,
    /// IQ4_NL with raw GGUF block bytes preserved in storage (no dequant).
    /// 18 bytes per 32 weights vs 64 bytes for F16 — ~72% memory
    /// reduction. Same codebook as `IQ4_XSRaw` but with a single
    /// per-block scale (no sub-block hierarchy). Simpler dequant +
    /// matvec than IQ4_XS — the 32-element block size matches Q4_0
    /// / Q5_0 / Q8_0 so we get the same `k % 32 == 0` constraint.
    IQ4_NLRaw,
    /// IQ3_S with raw GGUF block bytes preserved in storage (no
    /// dequant). 110 bytes per 256 weights vs 512 bytes for F16 —
    /// ~78% memory reduction. Most compact quant in the supported
    /// set. Each weight is a 9-bit codebook index + 1 sign bit;
    /// per-sub-block scale is `1 + 2*x` (odd, 1..=31). Scalar-only
    /// dequant/matvec for v1; the codebook indirection makes SIMD
    /// vectorization a follow-up.
    IQ3_SRaw,
    /// IQ3_XXS with raw GGUF block bytes preserved in storage (no
    /// dequant). 98 bytes per 256 weights vs 512 bytes for F16 —
    /// ~81% memory reduction at 3.0625 bpw. Each weight uses an
    /// 8-bit index into a 256-entry × 4 i8 codebook
    /// ([`rustllama_gguf::dequant::IQ3XXS_GRID`]) with per-element
    /// sign from a 7-bit sign-table index. Per-sub-block scale:
    /// `db = d * (0.5 + scale) * 0.5`. Scalar-only matvec for v1;
    /// AVX-512 / AVX2 vectorization can slot in alongside the
    /// existing IQ-quant fast paths.
    IQ3_XXSRaw,
    /// IQ2_XXS with raw GGUF block bytes preserved (no dequant). 66
    /// bytes per 256 weights vs 512 for F16 — ~87% memory reduction
    /// at 2.0625 bpw. Each weight comes from a 256-entry × 8-byte
    /// codebook (`IQ2XXS_GRID`) plus a 1-bit sign from `KSIGNS_IQ2XS`;
    /// a 4-bit sub-block scale lives in the top nibble of `aux32[1]`.
    /// Scalar-only matvec for v1 — the codebook indirection makes
    /// SIMD vectorization a follow-up.
    IQ2_XXSRaw,
    /// IQ2_XS with raw GGUF block bytes preserved (no dequant). 74
    /// bytes per 256 weights vs 512 for F16 — ~86% memory reduction
    /// at 2.3125 bpw. Each weight uses a 512-entry × 8-byte codebook
    /// (`IQ2XS_GRID`) with the grid index and sign index packed into
    /// one u16; two 4-bit sub-scales per scales byte cover 32 weights.
    /// Scalar-only matvec for v1.
    IQ2_XSRaw,
    /// IQ1_S with raw GGUF block bytes preserved (no dequant). 50
    /// bytes per 256 weights — ~90% memory reduction at 1.5625 bpw,
    /// the smallest quant in common circulation. Each 8-weight group
    /// is an 11-bit index into a 2048-entry codebook plus a 1-bit
    /// per-sub-block sign-delta flip. Scalar matvec only for v1; the
    /// codebook is large enough that SIMD-gather-based decoding is a
    /// follow-up.
    IQ1_SRaw,
    /// IQ1_M with raw GGUF block bytes preserved (no dequant). 56
    /// bytes per 256 weights — ~89% memory reduction at 1.75 bpw.
    /// Same `IQ1S_GRID` codebook as IQ1_S; the extra ~0.2 bpw buys a
    /// finer per-16-weight scale + delta layout. Scalar matvec for
    /// v1 (same SIMD-deferred reasoning as IQ1_S).
    IQ1_MRaw,
    /// IQ2_S with raw GGUF block bytes preserved (no dequant). 82
    /// bytes per 256 weights vs 512 for F16 — ~84% memory reduction
    /// at 2.5625 bpw. Each weight uses a 1024-entry × 8-byte codebook
    /// (`IQ2S_GRID`); the 10-bit grid index is split as `qs[l] | qh<<8`,
    /// per-element sign indices live in the second half of `qs`, and
    /// two 4-bit sub-scales per `scales` byte mirror the IQ2_XS layout.
    /// Scalar-only matvec for v1.
    IQ2_SRaw,
    /// NVFP4 with raw GGUF block bytes preserved (no dequant). 9 bytes
    /// per 16 weights = 4.5 bpw. Each weight is a 4-bit index into the
    /// `NVFP4_CODEBOOK` (E2M1 codebook: 0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0,
    /// ±4.0, ±6.0). Per-block scale is one FP8 E4M3 byte appended to
    /// the 8 packed code bytes. Runs at FP16-GEMM throughput on
    /// non-Blackwell hardware (software dequant in the matvec hot
    /// loop) — you get FP4's non-uniform accuracy without the
    /// hardware throughput bump.
    Nvfp4Raw,
    /// PQ2_0 (PrismML Bonsai) with raw GGUF block bytes preserved.
    /// 34 bytes per 128 weights (2.125 bpw): f16 scale + 32 bytes of
    /// little-endian 2-bit codes, `d * (code - 1)` with the `11`→+2d
    /// fourth level honored. Group-128 sibling of Prism's Q2_0.
    PQ2_0Raw,
    /// PTQ1_0 (PrismML Bonsai) with raw GGUF block bytes preserved.
    /// 28 bytes per 128 weights (1.75 bpw): 24 base-3 ceiling-packed
    /// qs bytes (5 trits each, staged {32,16,8} → 16+8 for 24 bytes)
    /// + 2 qh bytes (4 trits each) + f16 scale. Decode via the
    /// wrapping-multiply digit extraction. The densest weight format
    /// rustllama executes; a 27B model fits in ~5.9 GB.
    PTQ1_0Raw,
    /// MXFP4 (OCP Microscaling) with raw GGUF block bytes preserved.
    /// 17 bytes per 32 weights (4.25 bpw): 16 bytes of E2M1 4-bit
    /// float codes (two per byte, low nibble first) + 1 trailing E8M0
    /// 8-bit power-of-two block scale. Reconstruction is
    /// `e8m0_scale * e2m1_decode(code)` where E2M1 = {0, ±0.5, ±1,
    /// ±1.5, ±2, ±3, ±4, ±6}. Shares NVFP4's E2M1 element codebook but
    /// with a power-of-two (E8M0) block scale instead of NVFP4's E4M3.
    Mxfp4Raw,
    /// MXFP6 (OCP Microscaling, E3M2 variant) with raw GGUF block
    /// bytes preserved. 25 bytes per 32 weights (6.25 bpw): 24 bytes
    /// of 6-bit E3M2 float codes (1 sign / 3 exp / 2 mantissa, bit-
    /// packed little-endian) + 1 trailing E8M0 block scale.
    /// Reconstruction `e8m0_scale * e3m2_decode(code)`.
    Mxfp6Raw,
    /// MXFP8 (OCP Microscaling, E4M3 variant) with raw GGUF block
    /// bytes preserved. 33 bytes per 32 weights (8.25 bpw): 32 bytes
    /// of 8-bit E4M3 float elements + 1 trailing E8M0 block scale.
    /// Reconstruction `e8m0_scale * e4m3_decode(byte)` (shares NVFP4's
    /// E4M3 element decoder).
    Mxfp8Raw,
    /// FP8 (E4M3, per-TENSOR scale) with raw GGUF element bytes
    /// preserved. 1 byte per weight (8 bpw): each byte is an E4M3
    /// float element; the single f32 scale for the whole tensor lives
    /// in GGUF metadata (NOT in the weight bytes), so this format has
    /// no in-block scale and the matvec kernels take the scale as a
    /// separate argument. Reconstruction `tensor_scale *
    /// e4m3_decode(byte)`.
    Fp8Raw,
}

impl Dtype {
    pub fn from_ggml(t: GgmlType) -> Option<Self> {
        Some(match t {
            GgmlType::F32 => Self::F32,
            GgmlType::F16 => Self::F16,
            GgmlType::Bf16 => Self::Bf16,
            GgmlType::Q8_0 => Self::Q8_0,
            GgmlType::Q4_K => Self::Q4K,
            GgmlType::Q5_K => Self::Q5K,
            GgmlType::Q6_K => Self::Q6K,
            _ => return None,
        })
    }

    pub fn is_quantized(self) -> bool {
        matches!(self, Self::Q8_0 | Self::Q4K | Self::Q5K | Self::Q6K)
    }

    pub fn element_size_bytes(self) -> Option<usize> {
        Some(match self {
            Self::F32 => 4,
            // `Bf16Raw` shares F16's 2B/elem layout but keeps the
            // BF16 bit pattern intact for the matvec kernel.
            Self::F16 | Self::Bf16 | Self::Bf16Raw => 2,
            // Quantized formats are block-structured; element size is
            // fractional and meaningless on its own.
            _ => return None,
        })
    }

    pub fn is_raw_quant(self) -> bool {
        matches!(
            self,
            Self::Q8_0Raw
                | Self::Q4_0Raw
                | Self::Q5_0Raw
                | Self::Q4_1Raw
                | Self::Q5_1Raw
                | Self::Q8_KRaw
                | Self::Q2_KRaw
                | Self::Q3_KRaw
                | Self::Q4_KRaw
                | Self::Q5_KRaw
                | Self::Q6_KRaw
                | Self::IQ4_XSRaw
                | Self::IQ4_NLRaw
                | Self::IQ3_SRaw
                | Self::IQ3_XXSRaw
                | Self::IQ2_XXSRaw
                | Self::IQ2_XSRaw
                | Self::IQ2_SRaw
                | Self::IQ1_SRaw
                | Self::IQ1_MRaw
                | Self::Nvfp4Raw
                | Self::PQ2_0Raw
                | Self::PTQ1_0Raw
                | Self::Mxfp4Raw
                | Self::Mxfp6Raw
                | Self::Mxfp8Raw
                | Self::Fp8Raw
        )
    }

    /// Storage byte count for `n_elements` of this dtype.
    ///
    /// Mirrors `GgmlType::byte_size` for the raw quant variants
    /// (their on-disk block geometry is identical) plus the
    /// non-quantized F32 / F16 / BF16 paths. Used by MoE
    /// per-expert slicing to compute byte offsets without
    /// hard-coding `elements * element_size` (which is wrong for
    /// block-quantized formats).
    ///
    /// Panics if `n_elements` isn't a multiple of the dtype's
    /// block size — quant tensors with non-aligned shapes don't
    /// exist in practice (all real GGUFs use shapes that are
    /// multiples of 32 / 256), and surfacing a clear panic at the
    /// slice site beats silent garbage bytes.
    pub fn byte_size(self, n_elements: u64) -> u64 {
        fn blocks(n: u64, block: u64) -> u64 {
            assert!(
                n % block == 0,
                "element count {n} not divisible by block size {block} \
                 — block-quantized tensors must have aligned shapes"
            );
            n / block
        }
        match self {
            Self::F32 => n_elements * 4,
            Self::F16 | Self::Bf16 | Self::Bf16Raw => n_elements * 2,
            // Logical (non-raw) Q* variants: same on-disk byte
            // size as their Raw counterparts. Conceptually they
            // shouldn't be present as backing storage but accept
            // them so callers don't have to branch on Raw-vs-not.
            Self::Q8_0 | Self::Q8_0Raw => blocks(n_elements, 32) * 34,
            Self::Q4_0Raw => blocks(n_elements, 32) * 18,
            Self::Q5_0Raw => blocks(n_elements, 32) * 22,
            Self::Q4_1Raw => blocks(n_elements, 32) * 20,
            Self::Q5_1Raw => blocks(n_elements, 32) * 24,
            Self::Q2_KRaw => blocks(n_elements, 256) * 84,
            Self::Q8_KRaw => blocks(n_elements, 256) * 292,
            Self::Q3_KRaw => blocks(n_elements, 256) * 110,
            Self::Q4K | Self::Q4_KRaw => blocks(n_elements, 256) * 144,
            Self::Q5K | Self::Q5_KRaw => blocks(n_elements, 256) * 176,
            Self::Q6K | Self::Q6_KRaw => blocks(n_elements, 256) * 210,
            Self::IQ4_XSRaw => blocks(n_elements, 256) * 136,
            Self::IQ4_NLRaw => blocks(n_elements, 32) * 18,
            Self::IQ3_SRaw => blocks(n_elements, 256) * 110,
            Self::IQ3_XXSRaw => blocks(n_elements, 256) * 98,
            Self::IQ2_XXSRaw => blocks(n_elements, 256) * 66,
            Self::IQ2_XSRaw => blocks(n_elements, 256) * 74,
            Self::IQ2_SRaw => blocks(n_elements, 256) * 82,
            Self::IQ1_SRaw => blocks(n_elements, 256) * 50,
            Self::IQ1_MRaw => blocks(n_elements, 256) * 56,
            // NVFP4: 16 elements per block, 9 bytes (8 packed
            // nibbles + 1 fp8 scale byte). Matches rustllama's
            // extension layout.
            Self::Nvfp4Raw => blocks(n_elements, 16) * 9,
            // PrismML ternary group-128 formats.
            Self::PQ2_0Raw => blocks(n_elements, 128) * 34,
            Self::PTQ1_0Raw => blocks(n_elements, 128) * 28,
            // OCP Microscaling: 32 elems/block + 1 E8M0 scale byte.
            // MXFP4 = 16 nibble bytes + 1 = 17; MXFP6(E3M2) = 24 + 1 =
            // 25; MXFP8(E4M3) = 32 + 1 = 33.
            Self::Mxfp4Raw => blocks(n_elements, 32) * 17,
            Self::Mxfp6Raw => blocks(n_elements, 32) * 25,
            Self::Mxfp8Raw => blocks(n_elements, 32) * 33,
            // FP8 (E4M3, per-tensor scale): 1 byte per element, the
            // scale lives in GGUF metadata (not the weight bytes).
            Self::Fp8Raw => n_elements,
        }
    }
}

/// Tensor shape and strides in elements.
pub type Shape = Vec<u64>;
pub type Strides = Vec<i64>;

#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("gguf: {0}")]
    Gguf(#[from] GgufError),
    #[error("unsupported source dtype for tensor `{name}`: {ggml_type:?}")]
    UnsupportedSourceDtype { name: String, ggml_type: GgmlType },
    #[error("missing tensor in GGUF: {0}")]
    MissingTensor(String),
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Shape, got: Shape },
    #[error("dtype mismatch: expected {expected:?}, got {got:?}")]
    DtypeMismatch { expected: Dtype, got: Dtype },
    #[error("device mismatch: expected {expected}, got {got}")]
    DeviceMismatch { expected: Device, got: Device },
}

pub type Result<T> = std::result::Result<T, TensorError>;

/// Backing storage for a tensor's bytes. Phase 1 only implements the owned
/// CPU variant; later phases add `MmapBorrowed` (zero-copy reads from the
/// GGUF mmap) and SYCL USM allocations.
#[derive(Clone)]
pub enum Storage {
    /// Heap-allocated, owned bytes on the CPU. `Arc` so cheap clones share
    /// the same buffer (e.g. embedding weights also used for `lm_head` via
    /// tied embeddings).
    CpuOwned(Arc<[u8]>),
    /// Zero-copy sub-slice of an existing `CpuOwned` Arc. Used for
    /// MoE per-expert views that carve a `[d_ff, d_model]` matrix
    /// out of the packed `[n_experts, d_ff, d_model]` GGUF tensor
    /// without cloning bytes. `offset` + `len` describe the
    /// window into `backing.as_ref()`; the Arc keeps the parent
    /// buffer alive for as long as any slice references it.
    CpuOwnedSlice {
        backing: Arc<[u8]>,
        offset: usize,
        len: usize,
    },
    /// Zero-copy borrow of bytes from a memory-mapped file (the GGUF
    /// mmap). `backing` is a type-erased `Arc<dyn AsRef<[u8]>>` (the
    /// engine passes an `Arc<memmap2::Mmap>`) so this crate stays free
    /// of a mmap dependency; `offset`/`len` window into it. These pages
    /// are clean + file-backed: under memory pressure Windows discards
    /// them and re-reads from the file rather than writing to the
    /// pagefile. Produced by `Tensor::from_gguf` for raw quant tensors
    /// when zero-copy weights are enabled.
    MmapBorrowed {
        backing: Arc<dyn AsRef<[u8]> + Send + Sync>,
        offset: usize,
        len: usize,
    },
    /// SYCL Unified Shared Memory allocation — readable from both CPU
    /// and GPU. The pointer + free function come from
    /// `rustllama-kernels-sycl` via [`SyclUsmAllocation::new_with_free`];
    /// `rustllama-tensor` doesn't depend on the kernel crate directly,
    /// so the free callback is a `Box<dyn FnOnce(*mut u8)>` closure
    /// the kernel crate constructs at upload time.
    ///
    /// `as_bytes()` returns a slice from the same allocation; on
    /// integrated graphics (Iris Xe) this is a zero-cost CPU read of
    /// shared memory; on discrete Arc it triggers a device→host page
    /// migration on first read. GPU kernels in `accel.rs` consume the
    /// allocation by raw pointer (via `cast_slice` or direct ptr
    /// extraction) so the migration only happens when the CPU
    /// genuinely needs to inspect the bytes (debug paths, tests).
    SyclUsm(Arc<SyclUsmAllocation>),
}

/// Refcount-owned SYCL USM allocation. The drop closure returns the
/// pointer to the SYCL runtime (caller's `usm_free_raw` on the right
/// stream). `Arc<SyclUsmAllocation>` makes [`Storage::SyclUsm`] cheap
/// to clone — e.g. tied-embedding tables that double as `lm_head`
/// weights share one allocation across both call sites.
pub struct SyclUsmAllocation {
    ptr: std::ptr::NonNull<u8>,
    len: usize,
    /// Boxed free callback. Wrapped in `Option` so [`Drop::drop`] can
    /// `take()` it (FnOnce can't be called from a `&mut`-only
    /// position otherwise). The callback runs on the last Arc clone
    /// drop, freeing the SYCL allocation on its owning stream.
    free_fn: Option<Box<dyn FnOnce(*mut u8) + Send + Sync>>,
}

// SAFETY: USM allocations are valid across threads — the underlying
// SYCL runtime guarantees pointer validity until the explicit free
// call, and the `Arc` wrapping serializes the drop. The CPU mirror
// view returned by `as_bytes()` is read-only from Rust's perspective,
// so the standard aliasing rules cover it.
unsafe impl Send for SyclUsmAllocation {}
unsafe impl Sync for SyclUsmAllocation {}

impl SyclUsmAllocation {
    /// Wrap a raw USM pointer + a free callback. The callback receives
    /// `ptr` once when the last Arc clone drops. Callers (the SYCL
    /// kernel crate) construct this via [`Storage::sycl_usm_from_raw`]
    /// instead of directly to keep the FFI surface narrow.
    ///
    /// SAFETY: `ptr` must be a USM allocation of `len` bytes, the
    /// closure must free exactly that pointer on exactly the right
    /// stream, and the closure must remain valid for `'static` (which
    /// is automatic since `Box<dyn FnOnce + Send + Sync + 'static>`).
    pub unsafe fn new_with_free(
        ptr: *mut u8,
        len: usize,
        free_fn: Box<dyn FnOnce(*mut u8) + Send + Sync>,
    ) -> Self {
        Self {
            ptr: std::ptr::NonNull::new(ptr).expect("non-null USM pointer"),
            len,
            free_fn: Some(free_fn),
        }
    }

    /// Raw pointer to the USM allocation. GPU kernels read through
    /// this; CPU readers should prefer [`Self::as_bytes`].
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// CPU-side view of the USM bytes. Cheap on integrated GPUs
    /// (Iris Xe / iGPU shared LPDDR); on discrete Arc this triggers
    /// a device→host page migration the first time it's read.
    /// SAFETY: USM allocations remain valid for the lifetime of the
    /// `Arc<SyclUsmAllocation>` (the drop callback is the only path
    /// to free), so the returned slice's lifetime is sound.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for SyclUsmAllocation {
    fn drop(&mut self) {
        if let Some(free_fn) = self.free_fn.take() {
            free_fn(self.ptr.as_ptr());
        }
    }
}

impl std::fmt::Debug for SyclUsmAllocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyclUsmAllocation")
            .field("ptr", &self.ptr.as_ptr())
            .field("len", &self.len)
            .finish()
    }
}

impl Storage {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::CpuOwned(b) => b,
            Self::CpuOwnedSlice {
                backing,
                offset,
                len,
            } => &backing[*offset..*offset + *len],
            Self::MmapBorrowed {
                backing,
                offset,
                len,
            } => {
                let full: &[u8] = (**backing).as_ref();
                &full[*offset..*offset + *len]
            }
            Self::SyclUsm(a) => a.as_bytes(),
        }
    }

    pub fn len_bytes(&self) -> usize {
        match self {
            Self::CpuOwned(b) => b.len(),
            Self::CpuOwnedSlice { len, .. } => *len,
            Self::MmapBorrowed { len, .. } => *len,
            Self::SyclUsm(a) => a.len(),
        }
    }

    /// CPU-backed base pointer + length for the *whole backing buffer*,
    /// used by the Windows page-lock path ([`pagelock`] in the engine
    /// crate). For `CpuOwnedSlice` this returns the parent Arc's full
    /// range so dedup-by-pointer locks the parent once (and never the
    /// per-expert views separately). `MmapBorrowed` returns the windowed
    /// range (offset already applied) since file-backed pages are locked
    /// per borrowed tensor. `SyclUsm` returns `None` (handled separately
    /// by USM pinning at alloc time).
    pub fn cpu_backing_ptr_len(&self) -> Option<(*const u8, usize)> {
        match self {
            Self::CpuOwned(b) => Some((b.as_ptr(), b.len())),
            Self::CpuOwnedSlice { backing, .. } => Some((backing.as_ptr(), backing.len())),
            Self::MmapBorrowed {
                backing,
                offset,
                len,
            } => {
                let full: &[u8] = (**backing).as_ref();
                // SAFETY: offset+len is within the mapped region (validated
                // at construction in Tensor::from_gguf); pointer arithmetic
                // stays in-bounds of the mmap.
                Some((unsafe { full.as_ptr().add(*offset) }, *len))
            }
            Self::SyclUsm(_) => None,
        }
    }

    /// Windowed `(ptr, len)` of this tensor's bytes **only when it is a
    /// zero-copy file-backed (`MmapBorrowed`) tensor** — the exact
    /// sub-range `as_bytes()` covers (offset already applied), not the
    /// parent backing. Returns `None` for owned/USM storage. Used by
    /// the MoE expert-pin cache, which only pins file-backed (clean,
    /// evictable) pages — `VirtualLock`ing owned-heap private pages
    /// would just shrink the pageable pool, the opposite of the goal.
    pub fn mmap_borrowed_ptr_len(&self) -> Option<(*const u8, usize)> {
        match self {
            Self::MmapBorrowed {
                backing,
                offset,
                len,
            } => {
                let full: &[u8] = (**backing).as_ref();
                // SAFETY: offset+len within the mapped region (validated
                // at construction in Tensor::from_gguf_borrowed).
                Some((unsafe { full.as_ptr().add(*offset) }, *len))
            }
            _ => None,
        }
    }

    pub fn cast_slice<T: Pod>(&self) -> &[T] {
        bytemuck::cast_slice(self.as_bytes())
    }

    /// True if this storage already lives in GPU-accessible USM.
    /// Engine kernel call sites consult this to decide whether to
    /// upload-per-call (CpuOwned path) or use the pre-uploaded
    /// pointer directly (SyclUsm path).
    pub fn is_sycl_usm(&self) -> bool {
        matches!(self, Self::SyclUsm(_))
    }

    /// Raw USM pointer when this storage is a SYCL allocation.
    /// Returns `None` for CPU-backed storage. Engine code that
    /// would otherwise `as_bytes()` + upload should branch here
    /// first to skip the host→device copy.
    pub fn sycl_usm_ptr(&self) -> Option<*const u8> {
        match self {
            Self::SyclUsm(a) => Some(a.as_ptr()),
            _ => None,
        }
    }

    /// Wrap an externally-allocated USM pointer + free callback as
    /// a Storage variant. SAFETY: see [`SyclUsmAllocation::new_with_free`].
    pub unsafe fn sycl_usm_from_raw(
        ptr: *mut u8,
        len: usize,
        free_fn: Box<dyn FnOnce(*mut u8) + Send + Sync>,
    ) -> Self {
        Self::SyclUsm(Arc::new(SyclUsmAllocation::new_with_free(ptr, len, free_fn)))
    }

    /// Construct a sub-slice view sharing the backing storage of
    /// `self`. Inherits the underlying Arc — cheap to clone, no
    /// byte copy. Panics if `offset + len` exceeds the source.
    ///
    /// SyclUsm storage is intentionally NOT sliceable here — a USM
    /// sub-allocation needs its own free path (or to remain bound
    /// to the parent allocation's lifetime in a way Storage's enum
    /// shape can't express today). MoE expert slicing therefore
    /// keeps weights in CpuOwned + uploads per-call until a follow-
    /// up turn extends SyclUsm with a `SyclUsmView` arm.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        let backing = match self {
            Self::CpuOwned(b) => Arc::clone(b),
            Self::CpuOwnedSlice {
                backing,
                offset: base_off,
                len: base_len,
            } => {
                assert!(
                    offset + len <= *base_len,
                    "Storage::slice: window {}..{} overflows CpuOwnedSlice len {}",
                    offset,
                    offset + len,
                    base_len
                );
                // Compose offsets so the new view still references
                // the original Arc directly (no chains of slices).
                return Self::CpuOwnedSlice {
                    backing: Arc::clone(backing),
                    offset: base_off + offset,
                    len,
                };
            }
            Self::MmapBorrowed {
                backing,
                offset: base_off,
                len: base_len,
            } => {
                assert!(
                    offset + len <= *base_len,
                    "Storage::slice: window {}..{} overflows MmapBorrowed len {}",
                    offset,
                    offset + len,
                    base_len
                );
                return Self::MmapBorrowed {
                    backing: Arc::clone(backing),
                    offset: base_off + offset,
                    len,
                };
            }
            Self::SyclUsm(_) => panic!(
                "Storage::slice: SyclUsm slicing isn't supported yet — \
                 stick to CpuOwned for tensors that need sub-views (MoE \
                 expert weights, KV cache pages, etc.). The SyclUsm \
                 variant is for whole-tensor uploads only."
            ),
        };
        assert!(
            offset + len <= backing.len(),
            "Storage::slice: window {}..{} overflows CpuOwned len {}",
            offset,
            offset + len,
            backing.len()
        );
        Self::CpuOwnedSlice {
            backing,
            offset,
            len,
        }
    }
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CpuOwned(b) => f.debug_struct("CpuOwned").field("len", &b.len()).finish(),
            Self::CpuOwnedSlice { offset, len, .. } => f
                .debug_struct("CpuOwnedSlice")
                .field("offset", offset)
                .field("len", len)
                .finish(),
            Self::MmapBorrowed { offset, len, .. } => f
                .debug_struct("MmapBorrowed")
                .field("offset", offset)
                .field("len", len)
                .finish(),
            Self::SyclUsm(a) => f
                .debug_struct("SyclUsm")
                .field("ptr", &a.as_ptr())
                .field("len", &a.len())
                .finish(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub device: Device,
    pub dtype: Dtype,
    pub shape: Shape,
    pub strides: Strides,
    pub storage: Storage,
    /// Human-readable name; carried through for diagnostics.
    pub name: String,
}

impl Tensor {
    /// Element count according to the shape.
    pub fn element_count(&self) -> u64 {
        self.shape.iter().copied().product()
    }

    /// Allocate a zero-initialized owned CPU tensor of unquantized dtype.
    pub fn zeros_cpu(dtype: Dtype, shape: Shape) -> Self {
        let n = shape.iter().copied().product::<u64>() as usize;
        let bytes = match dtype {
            Dtype::F32 => vec![0u8; n * 4],
            Dtype::F16 | Dtype::Bf16 => vec![0u8; n * 2],
            other => panic!("zeros_cpu does not support quantized dtype {other:?}"),
        };
        Self {
            device: Device::Cpu,
            dtype,
            strides: contiguous_strides(&shape),
            shape,
            storage: Storage::CpuOwned(bytes.into()),
            name: String::new(),
        }
    }

    /// Wrap an owned `Vec<f32>` as a tensor.
    pub fn from_vec_f32(name: impl Into<String>, shape: Shape, data: Vec<f32>) -> Self {
        let expected: u64 = shape.iter().copied().product();
        assert_eq!(data.len() as u64, expected, "shape vs data length");
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        Self {
            device: Device::Cpu,
            dtype: Dtype::F32,
            strides: contiguous_strides(&shape),
            shape,
            storage: Storage::CpuOwned(bytes.into()),
            name: name.into(),
        }
    }

    /// Load a tensor from a GGUF file by name, dequantizing into the
    /// requested in-memory dtype. v1 supports F16 and F32 as targets.
    pub fn from_gguf(gguf: &Gguf, name: &str, target: Dtype) -> Result<Self> {
        let info = gguf
            .tensor(name)
            .ok_or_else(|| TensorError::MissingTensor(name.to_string()))?;
        let bytes = gguf
            .tensor_bytes(name)
            .ok_or_else(|| TensorError::MissingTensor(name.to_string()))?;
        load_tensor_bytes(info, bytes, target, name)
    }

    /// Zero-copy load: when `target` is a raw passthrough of the source
    /// GGUF dtype (e.g. `IQ1_SRaw` over an `IQ1_S` tensor), borrow the
    /// bytes directly from the mmap (`Storage::MmapBorrowed`) instead of
    /// `to_vec`-ing them — the pages stay clean + file-backed (never
    /// pagefiled). `backing` is the GGUF mmap handle ([`Gguf::mmap_backing`]);
    /// the returned tensor holds an `Arc` clone so the mapping outlives it.
    /// For non-raw targets (dequant→F16, repack) falls back to the owned
    /// [`Self::from_gguf`] path — transformed bytes can't be borrowed.
    pub fn from_gguf_borrowed(
        gguf: &Gguf,
        name: &str,
        target: Dtype,
        backing: Arc<dyn AsRef<[u8]> + Send + Sync>,
    ) -> Result<Self> {
        let info = gguf
            .tensor(name)
            .ok_or_else(|| TensorError::MissingTensor(name.to_string()))?;
        match raw_passthrough_source(target) {
            Some(src_ty) if src_ty == info.dtype => {
                let offset = info.abs_offset as usize;
                let len = info.byte_size as usize;
                // Bounds were validated at Gguf::open (TensorOutOfBounds).
                debug_assert!(
                    offset + len <= (*backing).as_ref().len(),
                    "from_gguf_borrowed: {name} window out of mmap bounds"
                );
                Ok(Tensor {
                    device: Device::Cpu,
                    dtype: target,
                    strides: contiguous_strides(&info.dims),
                    shape: info.dims.clone(),
                    storage: Storage::MmapBorrowed { backing, offset, len },
                    name: name.to_string(),
                })
            }
            // Not a raw passthrough (or source mismatch) → owned path.
            _ => Self::from_gguf(gguf, name, target),
        }
    }
}

/// For a raw passthrough target dtype, the exact GGUF source dtype it
/// borrows. Returns `None` for transformed targets (F32/F16/etc.) that
/// must be re-encoded into owned memory. Mirrors the `*Raw` arms of
/// [`load_tensor_bytes`] (which assert the same source pairing).
pub fn raw_passthrough_source(target: Dtype) -> Option<GgmlType> {
    Some(match target {
        Dtype::Q8_0Raw => GgmlType::Q8_0,
        Dtype::Q4_0Raw => GgmlType::Q4_0,
        Dtype::Q5_0Raw => GgmlType::Q5_0,
        Dtype::Q4_1Raw => GgmlType::Q4_1,
        Dtype::Q5_1Raw => GgmlType::Q5_1,
        Dtype::Q2_KRaw => GgmlType::Q2_K,
        Dtype::Q3_KRaw => GgmlType::Q3_K,
        Dtype::Q4_KRaw => GgmlType::Q4_K,
        Dtype::Q5_KRaw => GgmlType::Q5_K,
        Dtype::Q6_KRaw => GgmlType::Q6_K,
        Dtype::Q8_KRaw => GgmlType::Q8_K,
        Dtype::Bf16Raw => GgmlType::Bf16,
        Dtype::IQ4_XSRaw => GgmlType::IQ4_XS,
        Dtype::IQ4_NLRaw => GgmlType::IQ4_NL,
        Dtype::IQ3_SRaw => GgmlType::IQ3_S,
        Dtype::IQ3_XXSRaw => GgmlType::IQ3_XXS,
        Dtype::IQ2_XXSRaw => GgmlType::IQ2_XXS,
        Dtype::IQ2_XSRaw => GgmlType::IQ2_XS,
        Dtype::IQ2_SRaw => GgmlType::IQ2_S,
        Dtype::IQ1_SRaw => GgmlType::IQ1_S,
        Dtype::IQ1_MRaw => GgmlType::IQ1_M,
        Dtype::Nvfp4Raw => GgmlType::Nvfp4,
        Dtype::PQ2_0Raw => GgmlType::PQ2_0,
        Dtype::PTQ1_0Raw => GgmlType::PTQ1_0,
        Dtype::Mxfp4Raw => GgmlType::Mxfp4,
        Dtype::Mxfp6Raw => GgmlType::Mxfp6,
        Dtype::Mxfp8Raw => GgmlType::Mxfp8,
        Dtype::Fp8Raw => GgmlType::Fp8,
        _ => return None,
    })
}

fn load_tensor_bytes(info: &TensorInfo, src: &[u8], target: Dtype, name: &str) -> Result<Tensor> {
    let n_elements = info.element_count() as usize;
    let shape = info.dims.clone();

    // Stage 1: dequantize / convert to F32.
    let f32_buf: Vec<f32> = match info.dtype {
        GgmlType::F32 => bytemuck::cast_slice::<u8, f32>(src)[..n_elements].to_vec(),
        GgmlType::F16 => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_f16(src, &mut out);
            out
        }
        GgmlType::Bf16 => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_bf16(src, &mut out);
            out
        }
        GgmlType::Q8_0 => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_q8_0(src, &mut out);
            out
        }
        GgmlType::Q2_K => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_q2_k(src, &mut out);
            out
        }
        GgmlType::Q8_K => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_q8_k(src, &mut out);
            out
        }
        GgmlType::Q3_K => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_q3_k(src, &mut out);
            out
        }
        GgmlType::Q4_K => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_q4_k(src, &mut out);
            out
        }
        GgmlType::Q5_K => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_q5_k(src, &mut out);
            out
        }
        GgmlType::Q4_0 => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_q4_0(src, &mut out);
            out
        }
        GgmlType::Q5_0 => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_q5_0(src, &mut out);
            out
        }
        GgmlType::Q4_1 => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_q4_1(src, &mut out);
            out
        }
        GgmlType::Q5_1 => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_q5_1(src, &mut out);
            out
        }
        GgmlType::Q6_K => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_q6_k(src, &mut out);
            out
        }
        GgmlType::IQ4_XS => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_iq4_xs(src, &mut out);
            out
        }
        GgmlType::IQ4_NL => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_iq4_nl(src, &mut out);
            out
        }
        GgmlType::IQ3_S => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_iq3_s(src, &mut out);
            out
        }
        GgmlType::IQ2_XXS => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_iq2_xxs(src, &mut out);
            out
        }
        GgmlType::IQ2_XS => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_iq2_xs(src, &mut out);
            out
        }
        GgmlType::IQ2_S => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_iq2_s(src, &mut out);
            out
        }
        GgmlType::IQ3_XXS => {
            // Decode-on-load path. A raw-storage `IQ3_XXSRaw` variant
            // + specialized scalar matvec is the follow-up that
            // brings memory back down from f32 (4 bytes/weight) to
            // the native 3.0625 bpw. Same staging shape as IQ3_S
            // before its raw variant landed.
            let mut out = vec![0f32; n_elements];
            dequant::dequant_iq3_xxs(src, &mut out);
            out
        }
        GgmlType::IQ1_S => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_iq1_s(src, &mut out);
            out
        }
        GgmlType::IQ1_M => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_iq1_m(src, &mut out);
            out
        }
        GgmlType::Nvfp4 => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_nvfp4(src, &mut out);
            out
        }
        GgmlType::TQ2_0 => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_tq2_0(src, &mut out);
            out
        }
        GgmlType::TQ1_0 => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_tq1_0(src, &mut out);
            out
        }
        GgmlType::PQ2_0 => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_pq2_0(src, &mut out);
            out
        }
        GgmlType::PTQ1_0 => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_ptq1_0(src, &mut out);
            out
        }
        GgmlType::Mxfp4 => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_mxfp4(src, &mut out);
            out
        }
        GgmlType::Mxfp6 => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_mxfp6(src, &mut out);
            out
        }
        GgmlType::Mxfp8 => {
            let mut out = vec![0f32; n_elements];
            dequant::dequant_mxfp8(src, &mut out);
            out
        }
        GgmlType::Fp8 => {
            // FP8 (E4M3) carries its scale per-TENSOR in GGUF metadata,
            // not in the element bytes. The raw-weight path applies that
            // scale in the matvec kernel; this dequant-to-f32 fallback
            // (hit only when FP8 is force-converted to F32/F16 at load,
            // e.g. a sub-L3 tensor) decodes the raw E4M3 elements with an
            // implicit scale of 1.0. `dequant_fp8` takes the scale so the
            // metadata-driven path can pass it once TensorInfo carries it.
            let mut out = vec![0f32; n_elements];
            dequant::dequant_fp8(src, 1.0, &mut out);
            out
        }
        other => {
            return Err(TensorError::UnsupportedSourceDtype {
                name: name.to_string(),
                ggml_type: other,
            })
        }
    };

    // Stage 2: re-encode into the requested target dtype.
    let bytes: Vec<u8> = match target {
        Dtype::F32 => bytemuck::cast_slice(&f32_buf).to_vec(),
        Dtype::F16 => {
            let mut out = Vec::with_capacity(n_elements * 2);
            for v in &f32_buf {
                let h = f16::from_f32(*v);
                let b = h.to_le_bytes();
                out.push(b[0]);
                out.push(b[1]);
            }
            out
        }
        Dtype::Q8_0Raw => {
            // Bypass — caller wants the raw GGUF bytes. Only valid when the
            // source is already Q8_0.
            assert_eq!(
                info.dtype,
                GgmlType::Q8_0,
                "Dtype::Q8_0Raw requires source dtype Q8_0, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::Q4_0Raw => {
            assert_eq!(
                info.dtype,
                GgmlType::Q4_0,
                "Dtype::Q4_0Raw requires source dtype Q4_0, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::Q5_0Raw => {
            assert_eq!(
                info.dtype,
                GgmlType::Q5_0,
                "Dtype::Q5_0Raw requires source dtype Q5_0, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::Q4_1Raw => {
            assert_eq!(
                info.dtype,
                GgmlType::Q4_1,
                "Dtype::Q4_1Raw requires source dtype Q4_1, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::Q5_1Raw => {
            assert_eq!(
                info.dtype,
                GgmlType::Q5_1,
                "Dtype::Q5_1Raw requires source dtype Q5_1, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::Q2_KRaw => {
            assert_eq!(
                info.dtype,
                GgmlType::Q2_K,
                "Dtype::Q2_KRaw requires source dtype Q2_K, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::Q8_KRaw => {
            assert_eq!(
                info.dtype,
                GgmlType::Q8_K,
                "Dtype::Q8_KRaw requires source dtype Q8_K, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::Bf16Raw => {
            assert_eq!(
                info.dtype,
                GgmlType::Bf16,
                "Dtype::Bf16Raw requires source dtype Bf16, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::Q3_KRaw => {
            assert_eq!(
                info.dtype,
                GgmlType::Q3_K,
                "Dtype::Q3_KRaw requires source dtype Q3_K, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::Q4_KRaw => {
            assert_eq!(
                info.dtype,
                GgmlType::Q4_K,
                "Dtype::Q4_KRaw requires source dtype Q4_K, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::Q5_KRaw => {
            assert_eq!(
                info.dtype,
                GgmlType::Q5_K,
                "Dtype::Q5_KRaw requires source dtype Q5_K, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::Q6_KRaw => {
            assert_eq!(
                info.dtype,
                GgmlType::Q6_K,
                "Dtype::Q6_KRaw requires source dtype Q6_K, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::IQ4_XSRaw => {
            assert_eq!(
                info.dtype,
                GgmlType::IQ4_XS,
                "Dtype::IQ4_XSRaw requires source dtype IQ4_XS, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::IQ4_NLRaw => {
            assert_eq!(
                info.dtype,
                GgmlType::IQ4_NL,
                "Dtype::IQ4_NLRaw requires source dtype IQ4_NL, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::IQ3_SRaw => {
            assert_eq!(
                info.dtype,
                GgmlType::IQ3_S,
                "Dtype::IQ3_SRaw requires source dtype IQ3_S, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::IQ3_XXSRaw => {
            assert_eq!(
                info.dtype,
                GgmlType::IQ3_XXS,
                "Dtype::IQ3_XXSRaw requires source dtype IQ3_XXS, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::IQ2_XXSRaw => {
            assert_eq!(
                info.dtype,
                GgmlType::IQ2_XXS,
                "Dtype::IQ2_XXSRaw requires source dtype IQ2_XXS, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::IQ2_XSRaw => {
            assert_eq!(
                info.dtype,
                GgmlType::IQ2_XS,
                "Dtype::IQ2_XSRaw requires source dtype IQ2_XS, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::IQ2_SRaw => {
            assert_eq!(
                info.dtype,
                GgmlType::IQ2_S,
                "Dtype::IQ2_SRaw requires source dtype IQ2_S, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::IQ1_SRaw => {
            assert_eq!(
                info.dtype,
                GgmlType::IQ1_S,
                "Dtype::IQ1_SRaw requires source dtype IQ1_S, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::IQ1_MRaw => {
            assert_eq!(
                info.dtype,
                GgmlType::IQ1_M,
                "Dtype::IQ1_MRaw requires source dtype IQ1_M, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::Nvfp4Raw => {
            assert_eq!(
                info.dtype,
                GgmlType::Nvfp4,
                "Dtype::Nvfp4Raw requires source dtype Nvfp4, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::PQ2_0Raw => {
            assert_eq!(
                info.dtype,
                GgmlType::PQ2_0,
                "Dtype::PQ2_0Raw requires source dtype PQ2_0, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::PTQ1_0Raw => {
            assert_eq!(
                info.dtype,
                GgmlType::PTQ1_0,
                "Dtype::PTQ1_0Raw requires source dtype PTQ1_0, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::Mxfp4Raw => {
            assert_eq!(
                info.dtype,
                GgmlType::Mxfp4,
                "Dtype::Mxfp4Raw requires source dtype Mxfp4, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::Mxfp6Raw => {
            assert_eq!(
                info.dtype,
                GgmlType::Mxfp6,
                "Dtype::Mxfp6Raw requires source dtype Mxfp6, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::Mxfp8Raw => {
            assert_eq!(
                info.dtype,
                GgmlType::Mxfp8,
                "Dtype::Mxfp8Raw requires source dtype Mxfp8, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        Dtype::Fp8Raw => {
            assert_eq!(
                info.dtype,
                GgmlType::Fp8,
                "Dtype::Fp8Raw requires source dtype Fp8, got {:?}",
                info.dtype
            );
            src.to_vec()
        }
        other => panic!("unsupported target dtype {other:?}; phase 1 supports F32/F16 only"),
    };

    Ok(Tensor {
        device: Device::Cpu,
        dtype: target,
        strides: contiguous_strides(&shape),
        shape: shape.clone(),
        storage: Storage::CpuOwned(bytes.into()),
        name: name.to_string(),
    })
}

pub fn contiguous_strides(shape: &[u64]) -> Strides {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0i64; shape.len()];
    out[shape.len() - 1] = 1;
    for i in (0..shape.len() - 1).rev() {
        out[i] = out[i + 1] * shape[i + 1] as i64;
    }
    out
}

/// Safe typed view of a tensor's storage. Panics on dtype mismatch — callers
/// inspect `tensor.dtype` first.
pub fn as_slice_f32(t: &Tensor) -> &[f32] {
    assert_eq!(
        t.dtype,
        Dtype::F32,
        "expected F32 tensor, got {:?}",
        t.dtype
    );
    let n = t.element_count() as usize;
    &t.storage.cast_slice::<f32>()[..n]
}

/// Borrow the raw bytes of a quantized tensor (Q8_0Raw etc.). Caller is
/// responsible for matching the byte layout to the dtype.
pub fn as_bytes(t: &Tensor) -> &[u8] {
    t.storage.as_bytes()
}

pub fn as_slice_f16(t: &Tensor) -> &[f16] {
    assert_eq!(
        t.dtype,
        Dtype::F16,
        "expected F16 tensor, got {:?}",
        t.dtype
    );
    let n = t.element_count() as usize;
    // half::f16 with the `bytemuck` feature implements Pod, so this cast is safe.
    let raw: &[f16] = bytemuck::cast_slice::<u8, f16>(t.storage.as_bytes());
    &raw[..n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sycl_usm_storage_round_trip() {
        // Simulate a USM allocation with an owned heap buffer + a
        // drop closure that records the free call. The Arc'd
        // SyclUsmAllocation should hand back the same bytes via
        // both as_bytes() and the raw pointer, and the free closure
        // should fire exactly once on the last drop.
        let bytes: Box<[u8]> = vec![1, 2, 3, 4, 5, 6, 7, 8].into_boxed_slice();
        let len = bytes.len();
        let ptr = Box::into_raw(bytes) as *mut u8;
        let freed = std::sync::Arc::new(std::sync::Mutex::new(false));
        let freed_capture = std::sync::Arc::clone(&freed);
        let storage = unsafe {
            Storage::sycl_usm_from_raw(
                ptr,
                len,
                Box::new(move |p| {
                    *freed_capture.lock().unwrap() = true;
                    let _ = unsafe {
                        Box::from_raw(std::slice::from_raw_parts_mut(p, len) as *mut [u8])
                    };
                }),
            )
        };
        assert!(storage.is_sycl_usm());
        assert_eq!(storage.len_bytes(), len);
        assert_eq!(storage.as_bytes(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(storage.sycl_usm_ptr().unwrap(), ptr as *const u8);
        // Drop the storage; verify free fires.
        drop(storage);
        assert!(*freed.lock().unwrap(), "free closure should fire on drop");
    }

    #[test]
    fn sycl_usm_clone_shares_allocation() {
        // Arc-backed clones should share the same underlying USM
        // allocation; the free closure fires only when the last
        // clone drops.
        let bytes: Box<[u8]> = vec![42; 16].into_boxed_slice();
        let len = bytes.len();
        let ptr = Box::into_raw(bytes) as *mut u8;
        let free_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let free_count_capture = std::sync::Arc::clone(&free_count);
        let storage = unsafe {
            Storage::sycl_usm_from_raw(
                ptr,
                len,
                Box::new(move |p| {
                    free_count_capture
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let _ = unsafe {
                        Box::from_raw(std::slice::from_raw_parts_mut(p, len) as *mut [u8])
                    };
                }),
            )
        };
        let clone1 = storage.clone();
        let clone2 = storage.clone();
        // All three views see the same bytes.
        assert_eq!(storage.as_bytes()[0], 42);
        assert_eq!(clone1.as_bytes()[8], 42);
        assert_eq!(clone2.sycl_usm_ptr().unwrap(), ptr as *const u8);
        drop(storage);
        drop(clone1);
        assert_eq!(
            free_count.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "free should not fire while clones remain"
        );
        drop(clone2);
        assert_eq!(
            free_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "free should fire exactly once on last drop"
        );
    }

    #[test]
    fn mmap_borrowed_window_and_slice() {
        // Back a MmapBorrowed with an Arc<Vec<u8>> (Vec<u8>: AsRef<[u8]>),
        // standing in for the GGUF mmap. Validate the windowed views +
        // the page-lock accessor + offset-composing slice.
        let full: std::sync::Arc<dyn AsRef<[u8]> + Send + Sync> =
            std::sync::Arc::new((0u8..32).collect::<Vec<u8>>());
        let s = Storage::MmapBorrowed {
            backing: full.clone(),
            offset: 8,
            len: 16,
        };
        // as_bytes windows [8,24).
        assert_eq!(s.len_bytes(), 16);
        assert_eq!(s.as_bytes()[0], 8);
        assert_eq!(s.as_bytes()[15], 23);

        // cpu_backing_ptr_len points at the windowed start (offset applied).
        let base: &[u8] = (*full).as_ref();
        let (ptr, len) = s.cpu_backing_ptr_len().expect("mmap-backed ptr/len");
        assert_eq!(len, 16);
        assert_eq!(ptr, unsafe { base.as_ptr().add(8) });

        // slice composes offsets against the same backing Arc.
        let sub = s.slice(4, 4);
        assert_eq!(sub.as_bytes(), &[12, 13, 14, 15]);
        match &sub {
            Storage::MmapBorrowed { offset, len, .. } => {
                assert_eq!((*offset, *len), (12, 4));
            }
            _ => panic!("slice of MmapBorrowed should stay MmapBorrowed"),
        }
    }

    #[test]
    #[should_panic(expected = "SyclUsm slicing isn't supported")]
    fn sycl_usm_slice_panics() {
        let bytes: Box<[u8]> = vec![0u8; 32].into_boxed_slice();
        let len = bytes.len();
        let ptr = Box::into_raw(bytes) as *mut u8;
        let storage = unsafe {
            Storage::sycl_usm_from_raw(
                ptr,
                len,
                Box::new(move |p| {
                    let _ = unsafe {
                        Box::from_raw(std::slice::from_raw_parts_mut(p, len) as *mut [u8])
                    };
                }),
            )
        };
        let _ = storage.slice(0, 4);
    }

    #[test]
    fn strides_match_row_major() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(contiguous_strides(&[5]), vec![1]);
        assert!(contiguous_strides(&[]).is_empty());
    }

    #[test]
    fn zeros_f32_layout() {
        let t = Tensor::zeros_cpu(Dtype::F32, vec![2, 3]);
        assert_eq!(t.element_count(), 6);
        assert_eq!(t.storage.len_bytes(), 24);
        for v in as_slice_f32(&t) {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn from_vec_f32_round_trip() {
        let t = Tensor::from_vec_f32("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(as_slice_f32(&t), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn f16_view_works() {
        let mut bytes = Vec::with_capacity(4);
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            let b = f16::from_f32(v).to_le_bytes();
            bytes.push(b[0]);
            bytes.push(b[1]);
        }
        let t = Tensor {
            device: Device::Cpu,
            dtype: Dtype::F16,
            shape: vec![4],
            strides: vec![1],
            storage: Storage::CpuOwned(bytes.into()),
            name: "x".into(),
        };
        let view = as_slice_f16(&t);
        assert_eq!(view.len(), 4);
        assert_eq!(view[0].to_f32(), 1.0);
        assert_eq!(view[3].to_f32(), 4.0);
    }

    // ---- Storage::CpuOwnedSlice -------------------------------

    #[test]
    fn storage_slice_returns_correct_byte_window() {
        let bytes: Vec<u8> = (0..32).collect();
        let owned = Storage::CpuOwned(bytes.into());
        let mid = owned.slice(8, 16);
        assert_eq!(mid.len_bytes(), 16);
        assert_eq!(mid.as_bytes(), (8..24).collect::<Vec<u8>>().as_slice());
    }

    #[test]
    fn storage_slice_of_slice_composes_offsets() {
        // Slice([8..24]).slice([4..12]) → bytes [12..20] of the
        // original. Verify the inner-slice path composes offsets
        // rather than chaining wrapper structs.
        let bytes: Vec<u8> = (0..32).collect();
        let owned = Storage::CpuOwned(bytes.into());
        let mid = owned.slice(8, 16);
        let inner = mid.slice(4, 8);
        assert_eq!(inner.len_bytes(), 8);
        assert_eq!(inner.as_bytes(), (12..20).collect::<Vec<u8>>().as_slice());
    }

    #[test]
    fn storage_slice_shares_backing_arc() {
        // Carving a sub-slice must not allocate fresh bytes — pin
        // it via Arc strong-count. Before the slice exists, count
        // is 1 (the CpuOwned variant). After slicing, count is 2
        // (parent + slice).
        let bytes: Arc<[u8]> = vec![0u8; 64].into();
        let count_before = Arc::strong_count(&bytes);
        assert_eq!(count_before, 1);
        let owned = Storage::CpuOwned(Arc::clone(&bytes));
        let count_after_owned = Arc::strong_count(&bytes);
        assert_eq!(count_after_owned, 2, "CpuOwned clones the Arc");
        let _slice = owned.slice(0, 32);
        let count_after_slice = Arc::strong_count(&bytes);
        assert_eq!(
            count_after_slice, 3,
            "slice must share the underlying Arc, not allocate"
        );
    }

    #[test]
    #[should_panic(expected = "Storage::slice")]
    fn storage_slice_panics_on_overflow() {
        let bytes: Vec<u8> = vec![0; 16];
        let owned = Storage::CpuOwned(bytes.into());
        let _ = owned.slice(8, 20);
    }

    // ---- Dtype::byte_size --------------------------------------

    #[test]
    fn dtype_byte_size_flat_dtypes_match_element_size() {
        assert_eq!(Dtype::F32.byte_size(10), 40);
        assert_eq!(Dtype::F16.byte_size(10), 20);
        assert_eq!(Dtype::Bf16.byte_size(10), 20);
        assert_eq!(Dtype::Bf16Raw.byte_size(10), 20);
    }

    #[test]
    fn dtype_byte_size_block_quants_match_block_geometry() {
        // Each K-quant super-block is 256 elements; per-block byte
        // count matches the on-disk GGUF layout. These numbers are
        // pinned by GgmlType::byte_size in rustllama-gguf — keep
        // them in sync.
        assert_eq!(Dtype::Q4_KRaw.byte_size(256), 144);
        assert_eq!(Dtype::Q5_KRaw.byte_size(256), 176);
        assert_eq!(Dtype::Q6_KRaw.byte_size(256), 210);
        assert_eq!(Dtype::Q2_KRaw.byte_size(256), 84);
        assert_eq!(Dtype::Q3_KRaw.byte_size(256), 110);
        assert_eq!(Dtype::Q8_KRaw.byte_size(256), 292);
        assert_eq!(Dtype::IQ4_XSRaw.byte_size(256), 136);
        assert_eq!(Dtype::IQ3_SRaw.byte_size(256), 110);
        assert_eq!(Dtype::IQ3_XXSRaw.byte_size(256), 98);

        // 32-element block quants.
        assert_eq!(Dtype::Q8_0Raw.byte_size(32), 34);
        assert_eq!(Dtype::Q4_0Raw.byte_size(32), 18);
        assert_eq!(Dtype::Q5_0Raw.byte_size(32), 22);
        assert_eq!(Dtype::IQ4_NLRaw.byte_size(32), 18);

        // Multi-block calls scale linearly.
        assert_eq!(Dtype::Q4_KRaw.byte_size(512), 288);
        assert_eq!(Dtype::Q4_KRaw.byte_size(1024), 576);
    }

    #[test]
    fn dtype_byte_size_logical_quant_variants_match_raw() {
        // The non-Raw "logical" variants exist for type marking
        // (Dtype::Q4K) — their byte count must match the Raw
        // counterpart so callers can use byte_size uniformly.
        assert_eq!(Dtype::Q4K.byte_size(256), Dtype::Q4_KRaw.byte_size(256));
        assert_eq!(Dtype::Q5K.byte_size(256), Dtype::Q5_KRaw.byte_size(256));
        assert_eq!(Dtype::Q6K.byte_size(256), Dtype::Q6_KRaw.byte_size(256));
        assert_eq!(Dtype::Q8_0.byte_size(32), Dtype::Q8_0Raw.byte_size(32));
    }

    #[test]
    fn dtype_byte_size_nvfp4_matches_extension_layout() {
        // NVFP4: 16 elements per block, 9 bytes (8 packed nibbles
        // + 1 fp8 scale byte). Mirrors rustllama-kernels-cpu's
        // NVFP4_BLOCK_BYTES constant.
        assert_eq!(Dtype::Nvfp4Raw.byte_size(16), 9);
        assert_eq!(Dtype::Nvfp4Raw.byte_size(64), 36);
    }

    #[test]
    #[should_panic(expected = "not divisible")]
    fn dtype_byte_size_panics_on_misaligned_block_quant() {
        // 256-element block but caller passes 257 — surfaces as a
        // clear panic rather than silent truncation.
        let _ = Dtype::Q4_KRaw.byte_size(257);
    }
}
