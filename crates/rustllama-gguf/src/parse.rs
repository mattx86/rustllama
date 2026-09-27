//! Byte-level parsing of GGUF v3.

use crate::{GgufError, Result, TensorInfo, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC};

/// GGML tensor dtype ids that v1 of rustllama either consumes or at least
/// recognizes well enough to parse the tensor info table without bailing.
///
/// Compute support is narrower than parse support — see
/// [`GgmlType::is_supported_for_compute_v1`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    /// Legacy 4-bit symmetric: 32-weight block of `{ d: f16, qs: [u8; 16] }`.
    /// `value = d * (low2 - 8)`. Fully supported for compute.
    Q4_0 = 2,
    /// Legacy 4-bit asymmetric: 32-weight block of `{ d: f16, m: f16, qs: [u8; 16] }`.
    /// `value = d * q + m` with `q` an unsigned nibble in `[0, 15]`. Like Q4_0
    /// but with an explicit per-block min/offset, so the quantization range
    /// need not be centered at zero. Fully supported for compute.
    Q4_1 = 3,
    /// Legacy 5-bit symmetric: 32-weight block with extra `qh` u32 of
    /// 5th bits. Fully supported for compute.
    Q5_0 = 6,
    /// Legacy 5-bit asymmetric: 32-weight block of `{ d: f16, m: f16, qh: [u8; 4], qs: [u8; 16] }`.
    /// `value = d * q + m` with `q` an unsigned 5-bit value
    /// `(low4 | bit5 << 4)` in `[0, 31]`. Fully supported for compute.
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    /// Q2_K: 84 bytes per 256 weights. Smallest K-quant at 2.625 bpw.
    /// Layout: `{ scales: [u8; 16], qs: [u8; 64], d: f16, dmin: f16 }`.
    /// Each scales byte packs `(low: 4-bit scale, high: 4-bit min)`;
    /// per-weight value is `d*scale*q - dmin*min` where `q ∈ [0, 3]`.
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    /// Q8_K: 292 bytes per 256 weights. ~9.125 bpw.
    /// Layout: `{ d: f32, qs: [i8; 256], bsums: [i16; 16] }`. Usually
    /// used as an *intermediate* activation dtype for K-quant
    /// matmuls (e.g., Q4_K × Q8_K), but we support it as a weight
    /// storage too for completeness.
    Q8_K = 15,
    /// IQ2_XXS: super-block of 256 weights, 66 bytes. 2.0625 bpw.
    /// Layout: `{ d: f16, qs: [u16; 32] }`. Each ib32 sub-block of 32
    /// weights consumes 4 consecutive u16 from `qs` viewed as two u32
    /// halves: `aux32[0]` holds 4 codebook indices into `IQ2XXS_GRID`
    /// (256 entries × 8 packed-i8 grid points per entry); `aux32[1]`
    /// packs four 7-bit sign-indices (into `KSIGNS_IQ2XS`) plus a 4-bit
    /// sub-block scale in the top nibble. Sub-block scale formula:
    /// `db = d * (0.5 + (aux32[1] >> 28)) * 0.25`.
    IQ2_XXS = 16,
    /// IQ2_XS: super-block of 256 weights, 74 bytes. 2.3125 bpw.
    /// Layout: `{ d: f16, qs: [u16; 32], scales: [u8; 8] }`. Each u16
    /// in `qs` carries: low 9 bits = grid index into `IQ2XS_GRID`
    /// (512 entries × 8 packed-i8 grid points), high 7 bits = sign
    /// index into `KSIGNS_IQ2XS`. Each scales byte holds two 4-bit
    /// sub-block scales for one ib32 sub-block of 32 weights: the low
    /// nibble covers the first 16 weights, the high nibble the next
    /// 16. Sub-block scale formula: `db = d * (0.5 + nibble) * 0.25`.
    IQ2_XS = 17,
    /// IQ3_XXS: super-block of 256 weights, 98 bytes. 3.0625 bpw.
    /// Layout: `{ d: f16, qs: [u8; 96] }` where the 96-byte `qs` field
    /// splits as: `qs[0..64]` are grid indices (256 indices × 8 bits
    /// each, indexing the 256-entry `IQ3XXS_GRID` codebook of 4 i8
    /// coords each) and `qs[64..96]` are 8 little-endian u32 words
    /// (one per 32-weight sub-block) packing the sub-block scale in
    /// bits 28..32 and four 7-bit sign-table indices in bits 0..28
    /// (indexing [`KSIGNS_IQ2XS`]). Sub-block scale formula:
    /// `db = d * (0.5 + scale) * 0.5`. **Note:** ggml's enum maps
    /// `id=18` to `IQ3_XXS`, not `IQ2_S` — earlier rustllama versions
    /// had them swapped, silently decoding real IQ3_XXS files as
    /// broken IQ2_S blocks.
    IQ3_XXS = 18,
    /// IQ1_S: super-block of 256 weights, 50 bytes. 1.5625 bpw — the
    /// smallest quant in common circulation. Layout:
    /// `{ d: f16, qs: [u8; 32], qh: [u16; 8] }`. Each ib32 sub-block
    /// of 32 weights uses 1 byte of `qs` + 16 bits of `qh[ib32]`:
    /// 4 grid lookups × 8-weight grid entries = 32 weights. Index
    /// is 11 bits (8 low from `qs`, 3 high from the bottom of `qh`).
    /// Sub-block scale `dl = d * (2*((qh >> 12) & 7) + 1)`; the
    /// high bit of `qh` flips a per-sub-block delta around the grid
    /// values, giving the "near-1-bit" effective representation.
    /// Codebook: `IQ1S_GRID` (2048 entries × 8 packed-i8 weights).
    IQ1_S = 19,
    /// IQ4_NL: 32 weights per 18-byte block (no super-block).
    /// Layout: `{ d: f16, qs: [u8; 16] }` — same 4-bit indices into the
    /// `KVALUES_IQ4NL` non-linear codebook as `IQ4_XS`, but with a
    /// single per-block scale and no sub-block hierarchy. Used by
    /// some llama.cpp quant configurations as a lighter alternative
    /// to `IQ4_XS`.
    IQ4_NL = 20,
    /// IQ3_S: super-block of 256 weights, 110 bytes.
    /// Layout: `{ d: f16, qs: [u8; 64], qh: [u8; 8], signs: [u8; 32], scales: [u8; 4] }`
    /// where `qs[2l]|qh<<8` is a 9-bit index into a 512-entry packed-i8
    /// codebook (`IQ3S_GRID`); each entry holds 4 grid points. Per-element
    /// sign bit from `signs`. Per-sub-block scale: `1 + 2*x` (range 1..31)
    /// where `x` is a 4-bit value packed two-per-byte in `scales`. The
    /// 3-bit codebook variant — popular for very tight quantizations.
    IQ3_S = 21,
    /// IQ2_S: super-block of 256 weights, 82 bytes. 2.5625 bpw.
    /// Layout: `{ d: f16, qs: [u8; 64], qh: [u8; 8], scales: [u8; 8] }`
    /// where the 64-byte `qs` field is split into two contiguous halves:
    /// `qs[0..32]` are the low-8-bit grid indices and `qs[32..64]` are
    /// per-element sign-index bytes (one per 4-weight lookup). `qh[ib32]`
    /// supplies the two high bits of the 10-bit grid index per `l` in
    /// `0..4` (bits `(2*l, 2*l+1)` of `qh[ib32]` become bits 8-9 of the
    /// index). Each scales byte holds two 4-bit sub-scales mirroring
    /// IQ2_XS: low nibble → weights 0..15, high nibble → 16..31 of the
    /// ib32 sub-block. Sub-block scale formula:
    /// `db = d * (0.5 + nibble) * 0.25`. Codebook: `IQ2S_GRID` (1024 entries).
    IQ2_S = 22,
    /// IQ4_XS: super-block of 256 weights, 136 bytes.
    /// Layout: `{ d: f16, scales_h: u16, scales_l: [u8; 4], qs: [u8; 128] }`
    /// where `qs` is 4-bit indices into a 16-entry signed-int8 lookup table.
    /// Used by the popular `iq4_xs` and `iq4_nl` quant variants.
    IQ4_XS = 23,
    /// IQ1_M: super-block of 256 weights, 56 bytes. 1.75 bpw. Same
    /// `IQ1S_GRID` codebook as IQ1_S, with a more elaborate
    /// per-sub-block scale layout: no standalone `d` field, instead
    /// a packed `scales` array reconstructs both the f16 super-block
    /// scale (from the high nibbles of 4 scale bytes) AND 16
    /// 3-bit sub-block scales (covering 16 weights each at two
    /// granularities `dl1`/`dl2`). Per-16-weight delta is selected by
    /// individual bits of `qh` — finer control than IQ1_S's per-32
    /// delta. Layout:
    /// `{ qs: [u8; 32], qh: [u8; 16], scales: [u8; 8] }`.
    IQ1_M = 29,
    /// TQ1_0: ternary 1.6875 bpw. Super-block of 256 weights, 54 bytes.
    /// Layout: `{ qs: [u8; 48], qh: [u8; 4], d: f16 }`. Each qs byte
    /// packs 5 ternary digits via base-3 fixed-point fractional
    /// encoding (each byte ∈ `[0, 242]`); each qh byte packs 4 ternary
    /// digits the same way. Per-weight reconstruction: weight =
    /// `d * (digit - 1)` with `digit ∈ {0, 1, 2}` → values in
    /// `{-d, 0, +d}`. `GGML_TYPE_TQ1_0 = 34` upstream;
    /// `LLAMA_FTYPE_MOSTLY_TQ1_0 = 36` is the separate ftype enum
    /// used by `general.file_type` metadata.
    TQ1_0 = 34,
    /// TQ2_0: ternary 2 bpw. Super-block of 256 weights, 66 bytes.
    /// Layout: `{ qs: [u8; 64], d: f16 }`. Each qs byte packs 4
    /// ternary digits at bit positions 0, 2, 4, 6 (each digit `& 3`
    /// ∈ `{0, 1, 2}`; the 3-value is unused). Per-weight
    /// reconstruction: weight = `d * (digit - 1)`. Faster to decode
    /// than TQ1_0 (plain bit-shift vs base-3 multiply trick) at the
    /// cost of 0.3 bpw. `GGML_TYPE_TQ2_0 = 35` upstream;
    /// `LLAMA_FTYPE_MOSTLY_TQ2_0 = 37` is the separate ftype enum.
    TQ2_0 = 35,
    /// bfloat16 ("brain float"): 1 sign + 8 exponent + 7 mantissa bits.
    /// Same exponent range as f32 with ~3 decimal digits of precision.
    /// Bit-for-bit the upper 16 bits of an IEEE 754 f32, so conversion
    /// to f32 is a zero-fill left-shift (lossless). The Llama-family
    /// loader routes BF16 weights through [`Dtype::Bf16Raw`] (native
    /// BF16 storage + a SIMD matvec that widens to f32 on the fly via
    /// `vpslld` by 16) so the original bit pattern is preserved through
    /// the entire compute path — no lossy BF16 → F16 round-trip.
    Bf16 = 30,
    /// PQ2_0 (PrismML Bonsai): ternary weights in 2-bit slots at group
    /// 128. Block = `{ d: f16, qs: [u8; 32] }` = 34 bytes per 128
    /// weights (2.125 bpw). Codes unpack little-endian 2-bit
    /// (`(qs[j/4] >> ((j%4)*2)) & 3`) and reconstruct as
    /// `d * (code - 1)`: `00`→−d, `01`→0, `10`→+d, and — unlike pure
    /// ternary — `11`→+2d (the fourth level exists in the codec even
    /// though Prism's ternary encoder never emits it; decode must
    /// honor it). Same 2-bit codec as their group-64 Q2_0, distinct
    /// ggml id so both coexist. `GGML_TYPE_PQ2_0 = 142` in the
    /// PrismML llama.cpp fork.
    PQ2_0 = 142,
    /// PTQ1_0 (PrismML Bonsai): ternary base-3 dense trits at group
    /// 128. Block = `{ qs: [u8; 24], qh: [u8; 2], d: f16 }` = 28
    /// bytes per 128 weights (1.75 bpw). Same base-3 fixed-point
    /// packing as upstream TQ1_0 (5 trits/byte in qs, 4 trits/byte
    /// in qh, extraction `((byte * pow3[n]) * 3) >> 8`), but the qs
    /// staging is generalized to chunk sizes {32, 16, 8} so 24 bytes
    /// pack cleanly (TQ1_0's fixed 32+16 staging can't). Weight =
    /// `d * (trit - 1)`. `GGML_TYPE_PTQ1_0 = 143` in the PrismML
    /// llama.cpp fork.
    PTQ1_0 = 143,
    /// NVFP4 (rustllama extension): E2M1 4-bit floating-point with
    /// shared FP8 E4M3 scale per 16-element block. 9 bytes per block
    /// (8 packed nibbles + 1 scale byte) = 4.5 bpw. ID `1024` is
    /// deliberately well above llama.cpp's expanding enum (which is
    /// growing past 30) so a future upstream NVFP4 type won't collide
    /// with our extension. Pinned to this value forever — bumping it
    /// would break every existing rustllama-quantized GGUF on disk.
    Nvfp4 = 1024,
}

impl GgmlType {
    pub fn from_id(id: u32) -> Result<Self> {
        Ok(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            29 => Self::IQ1_M,
            30 => Self::Bf16,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            142 => Self::PQ2_0,
            143 => Self::PTQ1_0,
            1024 => Self::Nvfp4,
            other => return Err(GgufError::UnknownGgmlType(other)),
        })
    }

    /// Short canonical name for this dtype, matching the upstream
    /// GGML naming convention (e.g. `"F16"`, `"Q4_K"`, `"IQ3_S"`).
    /// Used by inspector / metrics surfaces so the disk-format label
    /// is stable across CLI, server JSON, and the GUI.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ2_XS => "IQ2_XS",
            Self::IQ2_S => "IQ2_S",
            Self::IQ1_S => "IQ1_S",
            Self::IQ1_M => "IQ1_M",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ3_S => "IQ3_S",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ4_XS => "IQ4_XS",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::PQ2_0 => "PQ2_0",
            Self::PTQ1_0 => "PTQ1_0",
            Self::Nvfp4 => "NVFP4",
        }
    }

    /// Whether the v1 compute path knows how to execute on this dtype.
    /// Parsing a model containing other dtypes still succeeds; load-time will
    /// reject only the unsupported tensors.
    pub fn is_supported_for_compute_v1(self) -> bool {
        matches!(
            self,
            Self::F32
                | Self::F16
                | Self::Bf16
                | Self::Q4_0
                | Self::Q4_1
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q8_0
                | Self::Q8_K
                | Self::Q2_K
                | Self::Q3_K
                | Self::Q4_K
                | Self::Q5_K
                | Self::Q6_K
                | Self::IQ2_XXS
                | Self::IQ2_XS
                | Self::IQ2_S
                | Self::IQ1_S
                | Self::IQ1_M
                | Self::IQ3_XXS
                | Self::IQ3_S
                | Self::IQ4_NL
                | Self::IQ4_XS
                | Self::TQ1_0
                | Self::TQ2_0
                | Self::PQ2_0
                | Self::PTQ1_0
                | Self::Nvfp4
        )
    }

    /// Size in bytes of `n_elements` of this dtype, accounting for block
    /// structure on quantized formats.
    pub fn byte_size(self, n_elements: u64) -> u64 {
        match self {
            Self::F32 => n_elements * 4,
            Self::F16 | Self::Bf16 => n_elements * 2,
            Self::Q8_0 => blocks(n_elements, 32) * 34,
            Self::Q8_1 => blocks(n_elements, 32) * 36,
            Self::Q4_0 => blocks(n_elements, 32) * 18,
            Self::Q4_1 => blocks(n_elements, 32) * 20,
            Self::Q5_0 => blocks(n_elements, 32) * 22,
            Self::Q5_1 => blocks(n_elements, 32) * 24,
            Self::Q2_K => blocks(n_elements, 256) * 84,
            Self::Q3_K => blocks(n_elements, 256) * 110,
            Self::Q4_K => blocks(n_elements, 256) * 144,
            Self::Q5_K => blocks(n_elements, 256) * 176,
            Self::Q6_K => blocks(n_elements, 256) * 210,
            Self::Q8_K => blocks(n_elements, 256) * 292,
            Self::IQ2_XXS => blocks(n_elements, 256) * 66,
            Self::IQ2_XS => blocks(n_elements, 256) * 74,
            Self::IQ2_S => blocks(n_elements, 256) * 82,
            Self::IQ3_XXS => blocks(n_elements, 256) * 98,
            Self::IQ1_S => blocks(n_elements, 256) * 50,
            Self::IQ4_NL => blocks(n_elements, 32) * 18,
            Self::IQ4_XS => blocks(n_elements, 256) * 136,
            Self::IQ3_S => blocks(n_elements, 256) * 110,
            Self::IQ1_M => blocks(n_elements, 256) * 56,
            Self::TQ1_0 => blocks(n_elements, 256) * 54,
            Self::TQ2_0 => blocks(n_elements, 256) * 66,
            Self::PQ2_0 => blocks(n_elements, 128) * 34,
            Self::PTQ1_0 => blocks(n_elements, 128) * 28,
            Self::Nvfp4 => blocks(n_elements, 16) * 9,
        }
    }
}

fn blocks(n_elements: u64, block_size: u64) -> u64 {
    n_elements.div_ceil(block_size)
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) if *v >= 0 => Some(*v as u32),
            Self::U64(v) if *v <= u32::MAX as u64 => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
}

pub(crate) struct Parsed {
    pub version: u32,
    pub alignment: u64,
    pub metadata: Vec<(String, MetadataValue)>,
    pub tensors: Vec<TensorInfo>,
    pub data_start: u64,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn need(&self, n: usize) -> Result<()> {
        if self.pos + n > self.buf.len() {
            Err(GgufError::UnexpectedEof {
                offset: self.pos,
                need: n,
            })
        } else {
            Ok(())
        }
    }
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.need(n)?;
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }
    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }
    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }
    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }
    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }
    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }
    fn read_u16(&mut self) -> Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }
    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }
    fn read_string(&mut self) -> Result<String> {
        let start = self.pos;
        let len = self.read_u64()? as usize;
        let raw = self.read_bytes(len)?;
        std::str::from_utf8(raw)
            .map(str::to_owned)
            .map_err(|source| GgufError::BadUtf8 {
                offset: start,
                source,
            })
    }
}

const META_U8: u32 = 0;
const META_I8: u32 = 1;
const META_U16: u32 = 2;
const META_I16: u32 = 3;
const META_U32: u32 = 4;
const META_I32: u32 = 5;
const META_F32: u32 = 6;
const META_BOOL: u32 = 7;
const META_STRING: u32 = 8;
const META_ARRAY: u32 = 9;
const META_U64: u32 = 10;
const META_I64: u32 = 11;
const META_F64: u32 = 12;

fn read_value(cur: &mut Cursor<'_>, type_id: u32) -> Result<MetadataValue> {
    Ok(match type_id {
        META_U8 => MetadataValue::U8(cur.read_u8()?),
        META_I8 => MetadataValue::I8(cur.read_i8()?),
        META_U16 => MetadataValue::U16(cur.read_u16()?),
        META_I16 => MetadataValue::I16(cur.read_i16()?),
        META_U32 => MetadataValue::U32(cur.read_u32()?),
        META_I32 => MetadataValue::I32(cur.read_i32()?),
        META_F32 => MetadataValue::F32(cur.read_f32()?),
        META_BOOL => MetadataValue::Bool(cur.read_bool()?),
        META_STRING => MetadataValue::String(cur.read_string()?),
        META_U64 => MetadataValue::U64(cur.read_u64()?),
        META_I64 => MetadataValue::I64(cur.read_i64()?),
        META_F64 => MetadataValue::F64(cur.read_f64()?),
        META_ARRAY => {
            let inner_type = cur.read_u32()?;
            let n = cur.read_u64()? as usize;
            let mut values = Vec::with_capacity(n.min(1 << 16));
            for _ in 0..n {
                values.push(read_value(cur, inner_type)?);
            }
            MetadataValue::Array(values)
        }
        other => return Err(GgufError::UnknownMetadataType(other)),
    })
}

pub(crate) fn parse(buf: &[u8]) -> Result<Parsed> {
    let mut cur = Cursor::new(buf);

    let magic_raw = cur.read_bytes(4)?;
    let mut magic = [0u8; 4];
    magic.copy_from_slice(magic_raw);
    if magic != GGUF_MAGIC {
        return Err(GgufError::BadMagic(magic));
    }

    let version = cur.read_u32()?;
    if version != 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let n_tensors = cur.read_u64()?;
    let n_kv = cur.read_u64()?;

    let mut metadata = Vec::with_capacity(n_kv as usize);
    for _ in 0..n_kv {
        let key = cur.read_string()?;
        let type_id = cur.read_u32()?;
        let value = read_value(&mut cur, type_id)?;
        metadata.push((key, value));
    }

    let alignment = metadata
        .iter()
        .find(|(k, _)| k == "general.alignment")
        .and_then(|(_, v)| v.as_u32().map(|x| x as u64))
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

    let mut tensors: Vec<TensorInfo> = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = cur.read_string()?;
        let n_dims = cur.read_u32()?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(cur.read_u64()?);
        }
        let type_id = cur.read_u32()?;
        let dtype = GgmlType::from_id(type_id)?;
        let rel_offset = cur.read_u64()?;
        let n_elements: u64 = dims.iter().copied().product();
        let byte_size = dtype.byte_size(n_elements);
        tensors.push(TensorInfo {
            name,
            dims,
            dtype,
            rel_offset,
            abs_offset: 0, // patched below once we know data_start
            byte_size,
        });
    }

    let header_end = cur.pos as u64;
    let pad = (alignment - (header_end % alignment)) % alignment;
    let data_start = header_end + pad;

    for t in &mut tensors {
        t.abs_offset = data_start + t.rel_offset;
    }

    Ok(Parsed {
        version,
        alignment,
        metadata,
        tensors,
        data_start,
    })
}

#[cfg(test)]
mod ggml_type_id_tests {
    use super::*;

    /// Regression: `id=18` is the canonical ggml mapping for IQ3_XXS,
    /// not IQ2_S. Earlier rustllama versions had this swapped and
    /// silently decoded real IQ3_XXS GGUFs as broken IQ2_S blocks
    /// (98-byte vs 82-byte block reader). Pin both ids so a future
    /// renumbering attempt would have to update the test too.
    #[test]
    fn id_18_maps_to_iq3_xxs_not_iq2_s() {
        assert_eq!(GgmlType::from_id(18).unwrap(), GgmlType::IQ3_XXS);
    }

    #[test]
    fn id_22_maps_to_iq2_s() {
        assert_eq!(GgmlType::from_id(22).unwrap(), GgmlType::IQ2_S);
    }

    #[test]
    fn iq3_xxs_canonical_name_and_block_size() {
        assert_eq!(GgmlType::IQ3_XXS.as_str(), "IQ3_XXS");
        // 98 bytes per 256-element super-block.
        assert_eq!(GgmlType::IQ3_XXS.byte_size(256), 98);
        assert_eq!(GgmlType::IQ3_XXS.byte_size(512), 196);
    }

    #[test]
    fn iq3_xxs_is_compute_supported() {
        assert!(GgmlType::IQ3_XXS.is_supported_for_compute_v1());
    }
}
