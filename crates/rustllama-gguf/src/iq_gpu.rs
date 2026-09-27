//! Backend abstraction for the IQ-codebook search step.
//!
//! The encode pipeline's hot loop is per-chunk codebook search:
//! for each 8- (or 4-) weight chunk of input, find the
//! `(grid_idx, sign_idx, signed_score, |g|²)` quadruple from the
//! format-specific codebook that maximizes projection energy.
//! That step is embarrassingly parallel — perfect for GPU offload.
//!
//! This module defines the [`IqGpuEncoder`] trait so the pipeline
//! can dispatch to either:
//!   - the built-in CPU SIMD fallback in this crate (always
//!     available; provides the production default)
//!   - a GPU implementation provided by `rustllama-kernels-sycl`
//!     (opt-in via the SYCL feature; multi-tensor-batched dispatch
//!     for ~30-60× speedup when functional)
//!
//! The pipeline calls into `IqGpuEncoder` via three method shapes
//! that cover all 7 IQ vector-grid formats:
//!   - `iq_8elt_delta_batched` — IQ1_S, IQ1_M (no sign mask;
//!     per-sub-block delta sign)
//!   - `iq_8elt_signed_batched` — IQ2_XXS, IQ2_XS, IQ2_S
//!     (8-element grid + 128 sign-mask codebook)
//!   - `iq_4elt_paired_signed_batched` — IQ3_XXS, IQ3_S
//!     (two 4-element grid lookups per chunk + shared 8-bit
//!     sign mask)
//!
//! Each method takes a **batch** of chunks (flat row-major
//! `[n_chunks × elements_per_chunk]`) to amortize GPU dispatch
//! overhead. The CPU fallback just calls the existing per-chunk
//! search in a loop.

use crate::encode_iq_vec;

/// 8-element grid family (IQ1_S, IQ1_M). Per-chunk pick: best
/// `grid_idx` + the resulting signed `target · (grid + delta)`
/// and `|grid + delta|²`. Caller picks the optimal delta sign
/// per sub-block by aggregating these across chunks.
#[derive(Debug, Clone, Copy)]
pub struct Iq8EltDeltaPick {
    pub grid_idx: u16,
    pub signed_score: f32,
    pub norm_sq: f32,
}

/// 8-element grid family (IQ2_*). Per-chunk pick: best
/// `(grid_idx, sign_idx)` + the resulting signed score and norm.
#[derive(Debug, Clone, Copy)]
pub struct Iq8EltSignedPick {
    pub grid_idx: u16,
    pub sign_idx: u8,
    pub signed_score: f32,
    pub grid_norm_sq: f32,
}

/// 4-element paired grid family (IQ3_*). Per-chunk pick: best
/// `(grid1_idx, grid2_idx, sign_idx)` + combined signed score and
/// summed norm.
#[derive(Debug, Clone, Copy)]
pub struct Iq4EltPairedPick {
    pub grid1_idx: u16,
    pub grid2_idx: u16,
    pub sign_idx: u8,
    pub signed_score: f32,
    pub grid_norm_sq: f32,
}

/// Format selector for the 8-element-grid batched search. Tells
/// the backend which codebook to use (the grid table is
/// implementation-private; the format enum selects it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Iq8EltGridFormat {
    /// 2048-entry IQ1_S grid (also used by IQ1_M).
    Iq1s,
    /// 256-entry IQ2_XXS grid.
    Iq2Xxs,
    /// 512-entry IQ2_XS grid.
    Iq2Xs,
    /// 1024-entry IQ2_S grid.
    Iq2S,
}

/// Format selector for the 4-element paired-grid batched search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Iq4EltGridFormat {
    /// 256-entry IQ3_XXS grid.
    Iq3Xxs,
    /// 512-entry IQ3_S grid.
    Iq3S,
}

/// IQ codebook-search backend. Implementations may dispatch to
/// CPU SIMD (provided here) or GPU SYCL kernels (provided by
/// `rustllama-kernels-sycl` when the feature is enabled).
///
/// All three methods take a flat row-major batch of chunks and
/// fill a caller-allocated output buffer. Returning `Err` on any
/// batch lets the pipeline fall back to the CPU path.
///
/// Implementations are **not** required to be `Send + Sync` — the
/// SYCL backend owns a thread-bound `sycl::queue` and the
/// quantization driver loop is single-threaded when GPU is active
/// (rayon parallelism is on by tensor on the CPU path; the GPU
/// path uses one stream and lets the device parallelize internally).
pub trait IqGpuEncoder {
    /// Find the best IQ1-style pick for each 8-element chunk.
    /// `targets` is `[n_chunks * 8]` flat f32; `delta` is the
    /// per-batch scalar delta (the IQ1 family allows two delta
    /// signs per sub-block; caller batches per delta-sign).
    /// `out` is `[n_chunks]`.
    fn iq_8elt_delta_batched(
        &self,
        targets: &[f32],
        delta: f32,
        out: &mut [Iq8EltDeltaPick],
    ) -> Result<(), IqGpuError>;

    /// #3: GPU variant returning all THREE picks (max-|score|,
    /// max-positive-score, max-negative-score) per chunk in one
    /// dispatch. Default returns `Unavailable` — backends override
    /// when supported. Caller falls back to CPU AVX2 if not available.
    fn iq_8elt_delta_batched_all3(
        &self,
        _targets: &[f32],
        _delta: f32,
        _out_abs: &mut [Iq8EltDeltaPick],
        _out_pos: &mut [Iq8EltDeltaPick],
        _out_neg: &mut [Iq8EltDeltaPick],
    ) -> Result<(), IqGpuError> {
        Err(IqGpuError::Unavailable)
    }

    /// Imatrix-weighted variant of `iq_8elt_delta_batched_all3`.
    /// `weights` is `[n_chunks * 8]` flat f32 — the per-element
    /// importance for each chunk's 8 targets. The grid search
    /// minimizes the importance-weighted reconstruction error;
    /// returned `signed_score` is the weighted dot (Σ w·t·g) and
    /// `norm_sq` is the weighted grid norm (Σ w·g²). Default returns
    /// `Unavailable`; `SyclIqEncoder` overrides. Caller falls back to
    /// CPU AVX2-weighted search when not available.
    fn iq_8elt_delta_batched_all3_w(
        &self,
        _targets: &[f32],
        _weights: &[f32],
        _delta: f32,
        _out_abs: &mut [Iq8EltDeltaPick],
        _out_pos: &mut [Iq8EltDeltaPick],
        _out_neg: &mut [Iq8EltDeltaPick],
    ) -> Result<(), IqGpuError> {
        Err(IqGpuError::Unavailable)
    }

    /// Find the best IQ2-style pick for each 8-element chunk
    /// (with sign-mask selection). `format` selects the codebook.
    fn iq_8elt_signed_batched(
        &self,
        targets: &[f32],
        format: Iq8EltGridFormat,
        out: &mut [Iq8EltSignedPick],
    ) -> Result<(), IqGpuError>;

    /// Find the best IQ3-style pick for each 8-element chunk
    /// (two 4-element grid lookups + 8-bit sign mask).
    /// `format` selects the codebook.
    fn iq_4elt_paired_signed_batched(
        &self,
        targets: &[f32],
        format: Iq4EltGridFormat,
        out: &mut [Iq4EltPairedPick],
    ) -> Result<(), IqGpuError>;

    /// G5: optional GPU dequant for K-quant sources (Q3_K, Q4_K,
    /// Q5_K, Q6_K). Returns `Unavailable` by default; `SyclIqEncoder`
    /// overrides to dispatch the matching SYCL dequant kernel.
    ///
    /// The pipeline calls this from its producer thread when
    /// `RUSTLLAMA_GPU_DEQUANT=1` is set; on `Unavailable` the pipeline
    /// falls back to the CPU `dequant_to_f32` path. `src_dtype` must
    /// be one of the four K-quant variants; other dtypes return
    /// `Unavailable`.
    fn try_dequant_kquant_to_f32(
        &self,
        _src_dtype: crate::GgmlType,
        _src_bytes: &[u8],
        _out_f32: &mut [f32],
    ) -> Result<(), IqGpuError> {
        Err(IqGpuError::Unavailable)
    }

    /// G6: optional GPU encoder for Q6_K. Takes a contiguous batch of
    /// f32 super-blocks and writes encoded bytes (`n_blocks × 210`).
    /// Default returns `Unavailable`; `SyclIqEncoder` overrides to
    /// dispatch the SYCL kernel. Falls back to CPU `encode_q6_k`
    /// when unavailable. Q3_K/Q4_K/Q5_K analogs are pending (iterative
    /// algorithms in Q4/Q5 require careful SYCL port).
    fn try_encode_q6_k_blocks(
        &self,
        _src_f32: &[f32],
        _dst_bytes: &mut [u8],
    ) -> Result<(), IqGpuError> {
        Err(IqGpuError::Unavailable)
    }

    /// G6: optional GPU encoder for Q3_K. Same shape as Q6_K above
    /// but per-block output is 110 bytes (vs Q6_K's 210). Analytical
    /// per-block algorithm — bit-exact parity vs CPU `encode_q3_k`.
    fn try_encode_q3_k_blocks(
        &self,
        _src_f32: &[f32],
        _dst_bytes: &mut [u8],
    ) -> Result<(), IqGpuError> {
        Err(IqGpuError::Unavailable)
    }

    /// G6: optional GPU encoder for Q4_K. Per-block output is 144
    /// bytes. Uses 20-step iterative `make_qkx2_quants_asym<15>`;
    /// bit-exact parity vs CPU `encode_q4_k` modulo FMA precision
    /// in the LS refit (CPU AVX-512 may diverge slightly on adversarial
    /// inputs — same documented behavior as the G1 IQ1_M kernel).
    fn try_encode_q4_k_blocks(
        &self,
        _src_f32: &[f32],
        _dst_bytes: &mut [u8],
    ) -> Result<(), IqGpuError> {
        Err(IqGpuError::Unavailable)
    }

    /// G6: optional GPU encoder for Q5_K. Per-block output is 176
    /// bytes. Same iterative algorithm as Q4_K (`make_qkx2_quants_asym<31>`)
    /// + extra 32-byte `qh` array for the 5th bit.
    fn try_encode_q5_k_blocks(
        &self,
        _src_f32: &[f32],
        _dst_bytes: &mut [u8],
    ) -> Result<(), IqGpuError> {
        Err(IqGpuError::Unavailable)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IqGpuError {
    #[error("GPU backend not available (no Intel SYCL device, or feature disabled)")]
    Unavailable,
    #[error("GPU kernel failed: {0}")]
    KernelFailed(String),
    #[error("output buffer size {got} doesn't match expected {expected} for {n_chunks} chunks")]
    BadOutputSize {
        got: usize,
        expected: usize,
        n_chunks: usize,
    },
    #[error("input size {got} doesn't match expected {expected} for {n_chunks} chunks")]
    BadInputSize {
        got: usize,
        expected: usize,
        n_chunks: usize,
    },
}

/// CPU SIMD fallback — calls the existing per-chunk search
/// helpers in [`encode_iq_vec`]. Always available; the pipeline's
/// default backend when no GPU is requested or detected.
///
/// Performance: this is just a thin batching wrapper. The hot path
/// (per-chunk grid search) is the AVX2 / SSE4.1 code shipped in
/// `encode_iq_vec`. No additional speedup vs the existing direct
/// calls; the trait wrapper exists to provide a uniform API
/// the GPU backend can substitute into.
pub struct CpuFallbackEncoder;

impl IqGpuEncoder for CpuFallbackEncoder {
    fn iq_8elt_delta_batched(
        &self,
        targets: &[f32],
        delta: f32,
        out: &mut [Iq8EltDeltaPick],
    ) -> Result<(), IqGpuError> {
        let n = out.len();
        if targets.len() != n * 8 {
            return Err(IqGpuError::BadInputSize {
                got: targets.len(),
                expected: n * 8,
                n_chunks: n,
            });
        }
        for (i, slot) in out.iter_mut().enumerate() {
            let chunk = &targets[i * 8..(i + 1) * 8];
            let pick = encode_iq_vec::best_iq1s_grid_for_chunk_pub(chunk, delta);
            *slot = Iq8EltDeltaPick {
                grid_idx: pick.0,
                signed_score: pick.1,
                norm_sq: pick.2,
            };
        }
        Ok(())
    }

    fn iq_8elt_delta_batched_all3(
        &self,
        targets: &[f32],
        delta: f32,
        out_abs: &mut [Iq8EltDeltaPick],
        out_pos: &mut [Iq8EltDeltaPick],
        out_neg: &mut [Iq8EltDeltaPick],
    ) -> Result<(), IqGpuError> {
        let n = out_abs.len();
        if targets.len() != n * 8 || out_pos.len() != n || out_neg.len() != n {
            return Err(IqGpuError::BadInputSize {
                got: targets.len(), expected: n * 8, n_chunks: n,
            });
        }
        for i in 0..n {
            let chunk = &targets[i * 8..(i + 1) * 8];
            let (a, p, ng) = encode_iq_vec::best_iq1s_grid_all3(chunk, delta);
            out_abs[i] = Iq8EltDeltaPick { grid_idx: a.grid_idx, signed_score: a.signed_score, norm_sq: a.norm_sq };
            out_pos[i] = Iq8EltDeltaPick { grid_idx: p.grid_idx, signed_score: p.signed_score, norm_sq: p.norm_sq };
            out_neg[i] = Iq8EltDeltaPick { grid_idx: ng.grid_idx, signed_score: ng.signed_score, norm_sq: ng.norm_sq };
        }
        Ok(())
    }

    fn iq_8elt_signed_batched(
        &self,
        targets: &[f32],
        format: Iq8EltGridFormat,
        out: &mut [Iq8EltSignedPick],
    ) -> Result<(), IqGpuError> {
        let n = out.len();
        if targets.len() != n * 8 {
            return Err(IqGpuError::BadInputSize {
                got: targets.len(),
                expected: n * 8,
                n_chunks: n,
            });
        }
        for (i, slot) in out.iter_mut().enumerate() {
            let chunk = &targets[i * 8..(i + 1) * 8];
            let pick = encode_iq_vec::search_chunk_8_for_format(chunk, format);
            *slot = pick;
        }
        Ok(())
    }

    fn iq_4elt_paired_signed_batched(
        &self,
        targets: &[f32],
        format: Iq4EltGridFormat,
        out: &mut [Iq4EltPairedPick],
    ) -> Result<(), IqGpuError> {
        let n = out.len();
        if targets.len() != n * 8 {
            return Err(IqGpuError::BadInputSize {
                got: targets.len(),
                expected: n * 8,
                n_chunks: n,
            });
        }
        for (i, slot) in out.iter_mut().enumerate() {
            let chunk = &targets[i * 8..(i + 1) * 8];
            let pick = encode_iq_vec::search_chunk_iq3_for_format(chunk, format);
            *slot = pick;
        }
        Ok(())
    }
}
