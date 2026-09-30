//! Apple **MLX affine** quantized-weight representation.
//!
//! MLX (`ml-explore/mlx`) is Apple's array framework; `mlx-lm` ships LLM
//! checkpoints in `.safetensors` whose linear/embedding weights are
//! quantized with `mlx.core.quantize(..., mode="affine")` — the default
//! and by far the most common MLX quant. Unlike the GGUF block formats
//! (one contiguous buffer with the scale baked into each block), an MLX
//! affine weight is **three sibling tensors**:
//!
//! - `<module>.weight`  — packed `bits`-bit codes, dtype **uint32**
//! - `<module>.scales`  — per-group scale, dtype **f16 / bf16**
//! - `<module>.biases`  — per-group bias,  dtype **f16 / bf16**
//!
//! so it doesn't fit rustllama's single-`Storage` [`crate::Tensor`]
//! (which carries one byte buffer + one [`crate::Dtype`]). This struct is
//! the dedicated carrier the safetensors loader fills and the CPU
//! dequant/matvec reference (in `rustllama-kernels-cpu`) consumes.
//!
//! # The affine format (sources)
//!
//! Confirmed against the MLX source + docs:
//!
//! - Dequant is a **plain affine** (asymmetric), NOT symmetric:
//!   `w[i] = scales[g] * q[i] + biases[g]`, where `g = i / group_size`
//!   and `q[i]` is the unpacked `bits`-bit code. The bias is `min` of the
//!   group and is **added** (not subtracted). MLX CPU backend inner loop:
//!   `result += xi * (scale * wl[p] + bias)`.
//!   (`mlx/backend/cpu/quantized.cpp`; `mlx.core.quantize` /
//!   `mlx.core.dequantize` docs.)
//! - `group_size` ∈ {32, 64, 128} (default 64); `bits` ∈ {2,3,4,5,6,8}
//!   (default 4). `group_size` consecutive elements **along the input
//!   dimension** share one `(scale, bias)`.
//! - **Packing** is a pure contiguous **little-endian bitstream**:
//!   element `i` occupies bits `[i*bits, (i+1)*bits)` of the packed data
//!   read LSB-first. For power-of-two `bits` (2/4/8) this is the obvious
//!   `pack_factor = 32/bits` elements per uint32 word, low element in the
//!   low bits. For 3/5/6 bits an element straddles byte **and** uint32
//!   boundaries — MLX's `extract_bits<T,3>` proves it, e.g. element 2 is
//!   `(w[0] & 0xc0) >> 6 | (w[1] & 0x1) << 2` (2 bits from byte 0, 1 bit
//!   from byte 1). Because `group_size` is a multiple of 32 and a group
//!   of 32 elements is exactly `bits` uint32 words, there is **no padding
//!   within a group or a row** — the whole weight is one stream. The
//!   unpack logic lives in `rustllama-kernels-cpu::mlx_affine`.
//!
//! # What is / isn't quantized
//!
//! In an mlx-lm checkpoint, `nn.Linear` (and often the token-embedding /
//! `lm_head` via `QuantizedEmbedding`) carry the `.scales`/`.biases`
//! siblings; RMSNorm/LayerNorm weights and biases stay full precision.
//! The loader classifies purely on the presence of those siblings, so it
//! doesn't need to hard-code which module names are quantized.

use std::sync::Arc;

/// A single MLX affine-quantized weight: packed codes + per-group scale
/// and bias + the geometry needed to decode them.
///
/// `shape` is the **logical, dequantized** shape in row-major
/// `[out_features, in_features]` (PyTorch / GGUF `nn.Linear.weight`
/// convention), NOT the on-disk packed shape. Quantization groups run
/// along the last (input) dimension.
#[derive(Clone)]
pub struct MlxAffineQuant {
    /// Packed `bits`-bit codes as a contiguous little-endian bitstream —
    /// exactly the raw bytes of the uint32 `weight` tensor (safetensors
    /// stores uint32 little-endian, so its byte view already *is* the
    /// bitstream the unpacker reads). `Arc` so the engine can share it
    /// cheaply (e.g. a tied embedding that doubles as `lm_head`).
    pub packed: Arc<[u8]>,
    /// Per-group scales, decoded to f32 from the file's f16/bf16. Length
    /// `out_features * (in_features / group_size)`, row-major to match
    /// `shape`.
    pub scales: Vec<f32>,
    /// Per-group biases (the group `min`), decoded to f32. Same length +
    /// layout as `scales`.
    pub biases: Vec<f32>,
    /// Elements per quantization group (32 / 64 / 128).
    pub group_size: usize,
    /// Bits per quantized element (2 / 3 / 4 / 5 / 6 / 8).
    pub bits: u32,
    /// Logical dequantized shape `[out_features, in_features]`.
    pub shape: Vec<u64>,
    /// Human-readable name (the `.weight` tensor name), for diagnostics.
    pub name: String,
}

/// Validation failures for [`MlxAffineQuant::validate`] — mismatched
/// buffer lengths, an unsupported geometry, or a shape that can't hold a
/// whole number of groups. Surfaced at load time so a malformed MLX file
/// fails loudly instead of decoding garbage.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MlxAffineError {
    #[error(
        "mlx affine `{name}`: unsupported bits {bits} (want one of 2,3,4,5,6,8)"
    )]
    UnsupportedBits { name: String, bits: u32 },
    #[error(
        "mlx affine `{name}`: unsupported group_size {group_size} \
         (want one of 32,64,128)"
    )]
    UnsupportedGroupSize { name: String, group_size: usize },
    #[error(
        "mlx affine `{name}`: shape {shape:?} is not 2-D [out_features, in_features]"
    )]
    NotMatrix { name: String, shape: Vec<u64> },
    #[error(
        "mlx affine `{name}`: in_features {in_features} not a multiple of \
         group_size {group_size}"
    )]
    GroupMisaligned {
        name: String,
        in_features: u64,
        group_size: usize,
    },
    #[error(
        "mlx affine `{name}`: packed bytes {got} != expected {expected} \
         ({n_elements} elems × {bits} bits / 8)"
    )]
    PackedLen {
        name: String,
        got: usize,
        expected: usize,
        n_elements: u64,
        bits: u32,
    },
    #[error(
        "mlx affine `{name}`: {which} length {got} != expected {expected} \
         ({n_groups} groups)"
    )]
    GroupBufLen {
        name: String,
        which: &'static str,
        got: usize,
        expected: usize,
        n_groups: usize,
    },
}

impl MlxAffineQuant {
    /// `out_features` (rows) — first logical dim.
    pub fn out_features(&self) -> u64 {
        self.shape.first().copied().unwrap_or(0)
    }

    /// `in_features` (cols) — last logical dim; the quantized axis.
    pub fn in_features(&self) -> u64 {
        self.shape.last().copied().unwrap_or(0)
    }

    /// Total logical element count `out_features * in_features`.
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().copied().product()
    }

    /// Number of `(scale, bias)` groups = `n_elements / group_size`.
    pub fn n_groups(&self) -> u64 {
        if self.group_size == 0 {
            return 0;
        }
        self.n_elements() / self.group_size as u64
    }

    /// Expected packed byte count for this geometry. Exact (never a
    /// ceiling): `in_features` is a multiple of `group_size` which is a
    /// multiple of 32, so `n_elements * bits` is always a multiple of 8.
    pub fn expected_packed_bytes(&self) -> usize {
        (self.n_elements() * self.bits as u64 / 8) as usize
    }

    /// Check the geometry + buffer lengths are self-consistent. Cheap;
    /// the loader calls it before handing the weight downstream.
    pub fn validate(&self) -> Result<(), MlxAffineError> {
        if !matches!(self.bits, 2 | 3 | 4 | 5 | 6 | 8) {
            return Err(MlxAffineError::UnsupportedBits {
                name: self.name.clone(),
                bits: self.bits,
            });
        }
        if !matches!(self.group_size, 32 | 64 | 128) {
            return Err(MlxAffineError::UnsupportedGroupSize {
                name: self.name.clone(),
                group_size: self.group_size,
            });
        }
        if self.shape.len() != 2 {
            return Err(MlxAffineError::NotMatrix {
                name: self.name.clone(),
                shape: self.shape.clone(),
            });
        }
        let in_f = self.in_features();
        if in_f % self.group_size as u64 != 0 {
            return Err(MlxAffineError::GroupMisaligned {
                name: self.name.clone(),
                in_features: in_f,
                group_size: self.group_size,
            });
        }
        let expected_packed = self.expected_packed_bytes();
        if self.packed.len() != expected_packed {
            return Err(MlxAffineError::PackedLen {
                name: self.name.clone(),
                got: self.packed.len(),
                expected: expected_packed,
                n_elements: self.n_elements(),
                bits: self.bits,
            });
        }
        let n_groups = self.n_groups() as usize;
        if self.scales.len() != n_groups {
            return Err(MlxAffineError::GroupBufLen {
                name: self.name.clone(),
                which: "scales",
                got: self.scales.len(),
                expected: n_groups,
                n_groups,
            });
        }
        if self.biases.len() != n_groups {
            return Err(MlxAffineError::GroupBufLen {
                name: self.name.clone(),
                which: "biases",
                got: self.biases.len(),
                expected: n_groups,
                n_groups,
            });
        }
        Ok(())
    }
}

impl std::fmt::Debug for MlxAffineQuant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxAffineQuant")
            .field("name", &self.name)
            .field("shape", &self.shape)
            .field("group_size", &self.group_size)
            .field("bits", &self.bits)
            .field("packed_bytes", &self.packed.len())
            .field("n_groups", &self.n_groups())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(bits: u32, group_size: usize, shape: [u64; 2]) -> MlxAffineQuant {
        let n = (shape[0] * shape[1]) as usize;
        let packed = vec![0u8; n * bits as usize / 8];
        let n_groups = n / group_size;
        MlxAffineQuant {
            packed: packed.into(),
            scales: vec![1.0; n_groups],
            biases: vec![0.0; n_groups],
            group_size,
            bits,
            shape: shape.to_vec(),
            name: "w".into(),
        }
    }

    #[test]
    fn geometry_helpers() {
        let q = mk(4, 64, [8, 128]);
        assert_eq!(q.out_features(), 8);
        assert_eq!(q.in_features(), 128);
        assert_eq!(q.n_elements(), 1024);
        assert_eq!(q.n_groups(), 16); // 1024 / 64
        assert_eq!(q.expected_packed_bytes(), 1024 * 4 / 8); // 512
    }

    #[test]
    fn valid_geometry_passes() {
        for &(bits, gs) in &[(2u32, 32usize), (3, 64), (4, 64), (5, 32), (6, 128), (8, 32)] {
            let q = mk(bits, gs, [4, (gs * 3) as u64]);
            assert_eq!(q.validate(), Ok(()), "bits={bits} gs={gs}");
        }
    }

    #[test]
    fn rejects_bad_geometry() {
        let mut q = mk(4, 64, [8, 128]);
        q.bits = 7;
        assert!(matches!(q.validate(), Err(MlxAffineError::UnsupportedBits { .. })));

        let mut q = mk(4, 64, [8, 128]);
        q.group_size = 48;
        assert!(matches!(
            q.validate(),
            Err(MlxAffineError::UnsupportedGroupSize { .. })
        ));

        let mut q = mk(4, 64, [8, 128]);
        q.scales.push(9.0); // wrong length
        assert!(matches!(q.validate(), Err(MlxAffineError::GroupBufLen { which: "scales", .. })));

        let mut q = mk(4, 64, [8, 128]);
        q.packed = vec![0u8; 3].into(); // wrong packed length
        assert!(matches!(q.validate(), Err(MlxAffineError::PackedLen { .. })));
    }
}
